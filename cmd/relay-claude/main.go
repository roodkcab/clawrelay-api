// Command relay-claude is the OpenAI-compatible HTTP front-end that drives
// the `claude` CLI. It accepts /v1/chat/completions requests, spawns
// `claude --output-format stream-json` per request, and translates Claude's
// stream events back into OpenAI SSE.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"clawrelay-api/pkg/openai"
	"clawrelay-api/pkg/sessions"
)

var version = "2.2.0"

// buildCommit is stamped at build time via:
//
//	go build -ldflags "-X main.buildCommit=$(git rev-parse --short HEAD)"
//
// so /health can tell exactly which source built a running binary (the
// 1.1.6-not-in-repo incident made version numbers alone untrustworthy).
var buildCommit = "unknown"

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

// Process-wide singletons. Created in main() and read from handlers.
var (
	sessionStore *sessions.Store
	stats        = openai.NewStats()

	// relayMode is "v1" (per-request `claude -p`; historically called "legacy"
	// and still accepted under that name as an alias), "channel" (persistent
	// stream-json process per session_id), or "channelv3" (interactive,
	// subscription-billed claude driven via Claude Code "channels"). Only the
	// matching manager is non-nil.
	relayMode  = "v1"
	channelMgr *chanManager
	v3Mgr      *v3Manager
)

// isChannelEligible reports whether a request matches the shape the channel
// mechanism serves: streaming and tool-less. A present session_id reuses a
// persistent process; an absent one runs ephemerally (fresh process per
// request). Non-stream or tool-bearing requests fall through to V1.
func isChannelEligible(req *openai.ChatCompletionRequest) bool {
	return req.Stream && len(req.Tools) == 0
}

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

	// Cap the body before ReadAll: without a limit a single oversized (or
	// malicious) request would be buffered whole into memory. 64 MiB leaves
	// ample headroom for base64-embedded attachments.
	r.Body = http.MaxBytesReader(w, r.Body, 64<<20)
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			openai.WriteError(w, http.StatusRequestEntityTooLarge, "invalid_request_error",
				fmt.Sprintf("Request body exceeds the %d byte limit", maxErr.Limit))
			return
		}
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

	includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage
	hasTools := len(req.Tools) > 0

	// Channel mode: serve every streaming, tool-less request through the
	// stream-json mechanism (never the V1 `claude -p` path). Requests with a
	// session_id reuse a persistent process keyed by session_id; requests
	// without one (e.g. stateless cron tasks) get a fresh, throwaway process
	// that runs once and is killed. Non-stream or tool-bearing shapes still fall
	// through to the V1 path (the channel mechanism only models streaming,
	// tool-less turns; wuji_tools never sends those shapes anyway).
	if relayMode == "channelv3" && isChannelEligible(&req) {
		log.Printf("ChatCompletion request [channelv3]: model=%s session=%s messages=%d", model, req.SessionID, len(req.Messages))
		sessionStore.LogRequest(req.SessionID, &req)
		handleChannelV3Response(w, r, &req, model, includeUsage)
		return
	}

	if relayMode == "channel" && isChannelEligible(&req) {
		if req.SessionID != "" {
			log.Printf("ChatCompletion request [channel]: model=%s session=%s messages=%d", model, req.SessionID, len(req.Messages))
			sessionStore.LogRequest(req.SessionID, &req)
			handleChannelStreamResponse(w, r, &req, model, includeUsage)
		} else {
			log.Printf("ChatCompletion request [channel/ephemeral]: model=%s (no session) messages=%d", model, len(req.Messages))
			handleChannelEphemeralStreamResponse(w, r, &req, model, includeUsage)
		}
		return
	}

	// Falling through to V1 in channel mode with a session_id (a non-stream
	// or tool-bearing request): a live channel worker may be holding that
	// session's jsonl open. Drop it first so the V1 `claude --resume` below
	// can't write the same session concurrently. drop 会等 worker 当前 turn
	// 结束才杀（绝不杀断正在流式输出的轮）；等不到就拒绝本次 fall-through。
	if relayMode == "channel" && channelMgr != nil && req.SessionID != "" {
		if err := channelMgr.drop(r.Context(), req.SessionID); err != nil {
			log.Printf("channel drop failed session=%s: %v", req.SessionID, err)
			openai.WriteError(w, http.StatusServiceUnavailable, "server_error", "session busy in channel mode; retry later")
			return
		}
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
		"commit":  buildCommit,
		"mode":    relayMode,
	})
}

// channelsHandler exposes the live channel-worker registry for debugging and
// for verifying process reuse (each session_id should show a single worker
// across multiple turns). Empty in V1 mode.
func channelsHandler(w http.ResponseWriter, r *http.Request) {
	openai.SetCORSHeaders(w, r, allowedOrigins)
	w.Header().Set("Content-Type", "application/json")
	resp := map[string]any{"mode": relayMode}
	switch {
	case channelMgr != nil:
		resp["workers"] = channelMgr.snapshot()
	case v3Mgr != nil:
		resp["workers"] = v3Mgr.snapshot()
	default:
		resp["workers"] = []any{}
	}
	json.NewEncoder(w).Encode(resp)
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
	mode := flag.String("mode", "v1", "v1=per-request `claude -p` (alias: legacy); channel=persistent stream-json process; channelv3=interactive claude via channels (subscription billing)")
	idleTTL := flag.Duration("idle-ttl", 30*time.Minute, "channel/channelv3 mode: kill a session's process after this much idle time")
	maxChannels := flag.Int("max-channels", 50, "channel/channelv3 mode: max concurrent processes (oldest evicted past this)")
	v3BridgeDir := flag.String("v3-bridge-dir", "/data/relay-v3/bridge", "channelv3 mode: dir with bridge.ts + node_modules")
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

	if v := os.Getenv("RELAY_FIRST_LINE_TIMEOUT_SECS"); v != "" {
		if secs, err := strconv.Atoi(v); err == nil && secs > 0 {
			firstLineTimeout = time.Duration(secs) * time.Second
			log.Printf("first-line watchdog timeout set to %s (RELAY_FIRST_LINE_TIMEOUT_SECS)", firstLineTimeout)
		} else {
			log.Printf("ignoring invalid RELAY_FIRST_LINE_TIMEOUT_SECS=%q", v)
		}
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

	relayMode = *mode
	// "v1" is the per-request `claude -p` path, historically named "legacy".
	// Accept --mode=legacy as a backward-compatible alias (existing deployments
	// and topology still pass it) but normalize to the canonical "v1" so /health
	// and /channels report a single consistent name.
	if relayMode == "legacy" {
		relayMode = "v1"
	}
	if relayMode == "channel" {
		channelMgr = newChanManager(chanManagerConfig{
			IdleTTL:     *idleTTL,
			MaxChannels: *maxChannels,
		})
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		channelMgr.StartReaper(ctx)
		log.Printf("Channel mode ENABLED: idle-ttl=%s max-channels=%d", *idleTTL, *maxChannels)

		// Graceful shutdown: kill every persistent worker process group so no
		// orphaned `claude` child is left blocked on stdin after the relay
		// exits (V1 per-request children self-terminate, channel ones do
		// not). SIGTERM is what the restart SOP's `kill` sends.
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		go func() {
			s := <-sigCh
			log.Printf("Received %s: killing %d channel worker(s) before exit", s, len(channelMgr.snapshot()))
			channelMgr.Stop()
			os.Exit(0)
		}()
	} else if relayMode == "channelv3" {
		v3Mgr = newV3Manager(v3Config{
			BridgeDir:   *v3BridgeDir,
			IdleTTL:     *idleTTL,
			MaxSessions: *maxChannels,
		})
		if err := v3Mgr.startControlServer(); err != nil {
			log.Fatalf("v3 control server: %v", err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		v3Mgr.StartReaper(ctx)
		log.Printf("Channel V3 mode ENABLED (interactive/subscription): bridge-dir=%s idle-ttl=%s max-sessions=%d ctrl=%s",
			*v3BridgeDir, *idleTTL, *maxChannels, v3Mgr.ctrlURL)
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		go func() {
			s := <-sigCh
			log.Printf("Received %s: killing %d interactive claude session(s) before exit", s, len(v3Mgr.snapshot()))
			v3Mgr.Stop()
			os.Exit(0)
		}()
	} else {
		log.Printf("V1 mode (per-request claude -p)")
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", chatCompletionsHandler)
	mux.HandleFunc("/v1/models", modelsHandler)
	mux.HandleFunc("/v1/stats", statsHandler)
	mux.HandleFunc("/health", healthHandler)
	mux.HandleFunc("/channels", channelsHandler)
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
