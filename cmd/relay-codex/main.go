// Command relay-codex is the OpenAI-compatible HTTP front-end for the
// `codex` CLI. Unlike relay-claude, this binary does NOT translate codex
// events into Claude shape — it emits OpenAI SSE directly from codex's
// native JSONL events, and exploits codex's first-class features:
//
//   * Thread-based resume: when the client supplies a stable session_id,
//     follow-up turns send only the latest user message via
//     `codex exec resume <thread_id>` instead of re-shipping full history.
//     Big token savings on long conversations.
//   * Native multimodal attachments via `-i FILE`.
//   * Reasoning effort via `-c model_reasoning_effort=`.
//   * Multi-stage UX: command_execution surfaces as tool_calls, reasoning
//     items as thinking deltas — visible in the UI even though codex doesn't
//     stream individual tokens.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"

	"clawrelay-api/pkg/openai"
	"clawrelay-api/pkg/sessions"
)

var version = "1.1.1"

var defaultModel = "codex/gpt-5.5"

// availableModels lists what /v1/models advertises. Codex itself accepts any
// model the underlying OpenAI account has access to; this list is for
// discovery only.
var availableModels = []openai.ModelInfo{
	{ID: "codex/gpt-5.5", Object: "model", Created: 1700000000, OwnedBy: "openai"},
	{ID: "codex/gpt-5.4", Object: "model", Created: 1700000000, OwnedBy: "openai"},
	{ID: "codex/gpt-5.3-codex", Object: "model", Created: 1700000000, OwnedBy: "openai"},
}

var allowedOrigins = []string{
	"http://10.0.100.148:5173",
	"https://goofish-stat.52ritao.cn",
}

var (
	sessionStore *sessions.Store
	threads      *threadMap
	stats        = openai.NewStats()
)

func chatCompletionsHandler(w http.ResponseWriter, r *http.Request) {
	openai.SetCORSHeaders(w, r, allowedOrigins)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != http.MethodPost {
		openai.WriteError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only POST is accepted")
		return
	}

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		openai.WriteError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("Failed to read body: %v", err))
		return
	}
	logBody := openai.SanitizeEnvVarsInLog(string(bodyBytes))
	if len(logBody) <= 4096 {
		log.Printf("Raw request body (%d bytes): %s", len(bodyBytes), logBody)
	} else {
		log.Printf("Raw request body (%d bytes): %s...[truncated]", len(bodyBytes), logBody[:4096])
	}

	var req openai.ChatCompletionRequest
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		openai.WriteError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("Invalid JSON: %v", err))
		return
	}
	if len(req.Messages) == 0 {
		openai.WriteError(w, http.StatusBadRequest, "invalid_request_error", "messages array is required and must not be empty")
		return
	}

	model := req.Model
	if model == "" {
		model = defaultModel
	}

	// Tools are ignored for codex — its tool harness isn't compatible with
	// OpenAI's tool definitions schema. Log loudly so callers debugging
	// missing tool-call behavior aren't confused.
	if len(req.Tools) > 0 {
		log.Printf("WARNING: %d tool definitions provided but codex backend ignores them; "+
			"codex uses its own native tool harness (shell, file edits, web_search, etc.)", len(req.Tools))
	}

	// Per-session attachment + thread-binding directory.
	var sessionDir string
	if req.SessionID != "" {
		sessionDir = sessionStore.AbsDir() + "/" + req.SessionID + "/files"
	}

	// Look up an existing codex thread for this client session — this is the
	// big optimization: when bound, we send only the new user message instead
	// of replaying the entire conversation.
	threadID := threads.Get(req.SessionID)
	if threadID != "" {
		log.Printf("Resuming codex thread_id=%s for session_id=%s", threadID, req.SessionID)
	}

	input := buildCodexInput(&req, model, threadID, sessionDir)

	chatID := openai.GenerateChatID()
	created := time.Now().Unix()

	log.Printf("ChatCompletion request: model=%s stream=%v messages=%d resume=%v images=%d",
		model, req.Stream, len(req.Messages), input.IsResume, len(input.ImagePath))
	log.Printf("codex args: %v (stdin %d bytes)", input.Args, len(input.Stdin))

	includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage

	sessionStore.LogRequest(req.SessionID, &req)

	if req.Stream {
		handleStreamResponse(w, r, input, chatID, created, model, includeUsage, req.WorkingDir, req.EnvVars, req.SessionID)
	} else {
		handleNonStreamResponse(w, r, input, chatID, created, model, req.WorkingDir, req.EnvVars, req.SessionID)
	}
}

func modelsHandler(w http.ResponseWriter, r *http.Request) {
	openai.SetCORSHeaders(w, r, allowedOrigins)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	resp := openai.ModelListResponse{Object: "list", Data: availableModels}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	cmd := exec.Command("codex", "--version")
	if err := cmd.Run(); err != nil {
		http.Error(w, fmt.Sprintf("Codex CLI not available: %v", err), http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "healthy",
		"backend": "codex",
		"version": version,
	})
}

func statsHandler(w http.ResponseWriter, r *http.Request) {
	openai.SetCORSHeaders(w, r, allowedOrigins)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats.Snapshot())
}

func main() {
	port := flag.String("port", "50010", "port to listen on")
	proxy := flag.String("proxy", "", "HTTP/HTTPS proxy URL")
	model := flag.String("model", "", "default model name (e.g. gpt-5.5, gpt-5.4, gpt-5.3-codex)")
	sessionsDir := flag.String("sessions-dir", "sessions", "directory for session log files + attachments + thread bindings")
	logFilePath := flag.String("log-file", "relay-codex.log", "log file path (use - for stdout only)")
	showVersion := flag.Bool("version", false, "show version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Println(version)
		os.Exit(0)
	}

	if *model != "" {
		defaultModel = *model
		// Codex accepts model names without prefix; normalize for display.
		if !strings.Contains(*model, "/") {
			defaultModel = "codex/" + *model
		}
		log.Printf("Default model set to: %s", defaultModel)
	}

	if *proxy != "" {
		os.Setenv("HTTP_PROXY", *proxy)
		os.Setenv("HTTPS_PROXY", *proxy)
		os.Setenv("http_proxy", *proxy)
		os.Setenv("https_proxy", *proxy)
	}

	if *logFilePath != "-" {
		logFile, err := os.OpenFile(*logFilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
		if err != nil {
			log.Fatalf("Failed to open log file: %v", err)
		}
		defer logFile.Close()
		log.SetOutput(io.MultiWriter(os.Stdout, logFile))
	}
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds | log.Lshortfile)

	sessionStore = sessions.New(*sessionsDir)
	sessionStore.StartCleanup(72*time.Hour, 1*time.Hour)
	threads = newThreadMap(sessionStore.AbsDir())

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", chatCompletionsHandler)
	mux.HandleFunc("/v1/models", modelsHandler)
	mux.HandleFunc("/v1/stats", statsHandler)
	mux.HandleFunc("/health", healthHandler)
	mux.HandleFunc("/sessions", sessionStore.ListHandler())
	mux.HandleFunc("/session/", sessionStore.PageHandler())

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "*")
		w.Header().Set("Access-Control-Allow-Headers", "*")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		log.Printf("HTTP %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)
		mux.ServeHTTP(w, r)
	})

	addr := ":" + *port
	if *proxy != "" {
		log.Printf("Using proxy: %s", *proxy)
	}
	log.Printf("Starting relay-codex (Codex CLI → OpenAI) on %s", addr)
	log.Printf("Endpoints:")
	log.Printf("  POST /v1/chat/completions")
	log.Printf("  GET  /v1/models")
	log.Printf("  GET  /v1/stats")
	log.Printf("  GET  /health")
	log.Printf("  GET  /sessions")
	log.Printf("  GET  /session/{id}")

	if err := http.ListenAndServe(addr, handler); err != nil {
		log.Fatal(err)
	}
}
