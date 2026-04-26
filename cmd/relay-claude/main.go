// Command relay-claude is the OpenAI-compatible HTTP front-end that drives
// the `claude` CLI. It accepts /v1/chat/completions requests, spawns
// `claude --output-format stream-json` per request, and translates Claude's
// stream events back into OpenAI SSE.
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
	"path/filepath"
	"strings"
	"time"

	"clawrelay-api/pkg/openai"
	"clawrelay-api/pkg/sessions"
)

var version = "1.1.1"

var defaultModel = "vllm/claude-sonnet-4-6"

// availableModels is what /v1/models returns. Extend as needed; the relay
// passes any unknown model through to claude (so adding here is purely
// cosmetic / discovery for clients).
var availableModels = []openai.ModelInfo{
	{ID: "vllm/claude-sonnet-4-6", Object: "model", Created: 1700000000, OwnedBy: "anthropic"},
	{ID: "vllm/claude-opus-4-6", Object: "model", Created: 1700000000, OwnedBy: "anthropic"},
	{ID: "vllm/claude-haiku-4-5-20251001", Object: "model", Created: 1700000000, OwnedBy: "anthropic"},
	{ID: "minimax/MiniMax-M2.7", Object: "model", Created: 1700000000, OwnedBy: "minimax"},
	{ID: "minimax/MiniMax-M2.5", Object: "model", Created: 1700000000, OwnedBy: "minimax"},
	{ID: "kimi/kimi-k2.5", Object: "model", Created: 1700000000, OwnedBy: "moonshot"},
	{ID: "zhipu/glm-5.1", Object: "model", Created: 1700000000, OwnedBy: "zhipu"},
}

var modelAliases = map[string]string{
	"gpt-4":         "opus",
	"gpt-4o":        "sonnet",
	"gpt-4-turbo":   "sonnet",
	"gpt-3.5-turbo": "haiku",
	"gpt-4o-mini":   "haiku",
}

var allowedOrigins = []string{
	"http://10.0.100.148:5173",
	"https://goofish-stat.52ritao.cn",
}

// Process-wide singletons. Both are created in main() and read from handlers.
var (
	sessionStore *sessions.Store
	stats        = openai.NewStats()
)

func resolveModel(model string) string {
	if idx := strings.LastIndex(model, "/"); idx >= 0 {
		model = model[idx+1:]
	}
	if alias, ok := modelAliases[model]; ok {
		return alias
	}
	return model
}

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

	model := resolveModel(req.Model)
	if model == "" {
		model = defaultModel
	}

	var sessionDir string
	if req.SessionID != "" {
		sessionDir = filepath.Join(sessionStore.AbsDir(), req.SessionID, "files")
	}
	prompt, systemPrompt, tempFiles := buildPromptFromMessages(req.Messages, sessionDir)
	if len(tempFiles) > 0 {
		defer func() {
			for _, f := range tempFiles {
				os.Remove(f)
			}
		}()
	}

	hasTools := len(req.Tools) > 0
	if hasTools {
		systemPrompt += buildToolPrompt(req.Tools)
		log.Printf("Injected %d tool definitions into system prompt", len(req.Tools))
	}

	chatID := openai.GenerateChatID()
	created := time.Now().Unix()

	log.Printf("ChatCompletion request: model=%s stream=%v messages=%d tools=%d",
		model, req.Stream, len(req.Messages), len(req.Tools))

	args, stdinData := buildClaudeArgs(&req, model, prompt, systemPrompt)

	log.Printf("Claude args: %v (stdin length: %d bytes)", args, len(stdinData))

	includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage

	workingDir := req.WorkingDir
	envVars := req.EnvVars
	sessionID := req.SessionID
	sessionStore.LogRequest(sessionID, &req)

	if req.Stream {
		if hasTools {
			handleBufferedStreamResponse(w, r, args, stdinData, chatID, created, model, includeUsage, workingDir, envVars, sessionID)
		} else {
			handleStreamResponse(w, r, args, stdinData, chatID, created, model, includeUsage, workingDir, envVars, sessionID)
		}
	} else {
		handleNonStreamResponse(w, r, args, stdinData, chatID, created, model, hasTools, workingDir, envVars, sessionID)
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
	cmd := exec.Command("claude", "--version")
	if err := cmd.Run(); err != nil {
		http.Error(w, fmt.Sprintf("Claude CLI not available: %v", err), http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "healthy",
		"backend": "claude",
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
	port := flag.String("port", "50009", "port to listen on")
	proxy := flag.String("proxy", "", "HTTP/HTTPS proxy URL (e.g. http://127.0.0.1:7890)")
	model := flag.String("model", "", "default model name (e.g. claude-sonnet-4-6)")
	sessionsDir := flag.String("sessions-dir", "sessions", "directory for session log files + attachments")
	logFilePath := flag.String("log-file", "relay-claude.log", "log file path (use - for stdout only)")
	showVersion := flag.Bool("version", false, "show version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Println(version)
		os.Exit(0)
	}

	if *model != "" {
		defaultModel = *model
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
	log.Printf("Starting relay-claude (Claude Code → OpenAI) on %s", addr)
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
