package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"clawrelay-api/pkg/attachments"
	"clawrelay-api/pkg/openai"
)

// channelInterruptBackstop bounds how long an interrupted turn (stop /
// AskUserQuestion) may keep draining in the background before the worker is
// force-killed. claude's interrupt for in-progress text only takes effect at
// the next message boundary, so a stopped turn finishes naturally within
// seconds-to-tens-of-seconds; this generous window lets that happen and keeps
// the hot process alive for the next request. Only a genuinely stuck turn hits
// the backstop kill — and even then its session is on disk, so the next request
// --resumes it. Overridable in tests.
var channelInterruptBackstop = 90 * time.Second

// buildChannelArgs assembles the claude flags for a persistent stream-json
// process. The trailing session flag (--session-id / --resume) is appended by
// the manager. Unlike legacy buildClaudeArgs the prompt is NOT passed here —
// each turn's user message is written to stdin as a stream-json envelope.
func buildChannelArgs(req *openai.ChatCompletionRequest, model, systemPrompt string) []string {
	var args []string
	args = append(args, "--print")
	args = append(args, "--input-format", "stream-json")
	args = append(args, "--output-format", "stream-json")
	args = append(args, "--include-partial-messages")
	args = append(args, "--verbose")

	if systemPrompt != "" {
		args = append(args, "--append-system-prompt", systemPrompt)
	}
	if req.SystemPromptFile != "" {
		args = append(args, "--append-system-prompt-file", req.SystemPromptFile)
	}
	args = append(args, "--model", model)

	permMode := "bypassPermissions"
	if req.PermissionMode != "" {
		permMode = req.PermissionMode
	}
	args = append(args, "--permission-mode", permMode)

	if req.AllowedTools != "" {
		args = append(args, "--allowedTools", req.AllowedTools)
	}
	for _, dir := range req.AddDirs {
		if dir != "" {
			args = append(args, "--add-dir", dir)
		}
	}

	maxTurns := 20
	if req.MaxTurns != nil {
		maxTurns = *req.MaxTurns
	}
	args = append(args, "--max-turns", openai.FmtInt(maxTurns))

	if req.Effort != "" {
		args = append(args, "--effort", req.Effort)
	}
	if req.Settings != "" {
		args = append(args, "--settings", req.Settings)
	}
	return args
}

// extractSystemPrompt returns the (last) system message content, matching
// buildPromptFromMessages' behavior.
func extractSystemPrompt(messages []openai.ChatMessage) string {
	var sp string
	for i := range messages {
		if messages[i].Role == "system" {
			sp = messages[i].ContentString()
		}
	}
	return sp
}

// lastUserTurnContent extracts the final user message as a single string,
// inlining any attachments as `[Image: /path]` / `[File: /path]` markers
// exactly as buildPromptFromMessages does for the legacy path. Channel mode
// only feeds this newest turn — prior turns already live in the persistent
// process's context. When sessionDir is empty, attachment temp files are
// returned for the caller to clean up.
func lastUserTurnContent(messages []openai.ChatMessage, sessionDir string) (content string, tempFiles []string, ok bool) {
	idx := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			idx = i
			break
		}
	}
	if idx < 0 {
		return "", nil, false
	}
	msg := messages[idx]
	text := msg.ContentString()

	files := attachments.ExtractAndSave(msg.Content, sessionDir, "claude-img", "claude-file")
	for _, a := range files {
		if sessionDir == "" {
			tempFiles = append(tempFiles, a.Path)
		}
	}
	if len(files) > 0 {
		var refs []string
		for _, a := range files {
			if a.IsImage {
				refs = append(refs, fmt.Sprintf("[Image: %s]", a.Path))
			} else {
				refs = append(refs, fmt.Sprintf("[File: %s]", a.Path))
			}
		}
		if text != "" {
			text += "\n"
		}
		text += strings.Join(refs, "\n")
	}
	return text, tempFiles, true
}

// handleChannelStreamResponse serves one /v1/chat/completions turn through the
// persistent channel worker for req.SessionID. The SSE byte stream is produced
// by the shared sseTranslator, identical to the legacy path. The process is
// kept alive across turns; stop and AskUserQuestion interrupt (not kill) it.
func handleChannelStreamResponse(w http.ResponseWriter, r *http.Request, req *openai.ChatCompletionRequest, model string, includeUsage bool) {
	sessionID := req.SessionID
	systemPrompt := extractSystemPrompt(req.Messages)

	var sessionDir string
	if sessionID != "" {
		sessionDir = filepath.Join(sessionStore.AbsDir(), sessionID, "files")
	}
	content, tempFiles, ok := lastUserTurnContent(req.Messages, sessionDir)
	if len(tempFiles) > 0 {
		defer func() {
			for _, f := range tempFiles {
				os.Remove(f)
			}
		}()
	}
	if !ok {
		openai.WriteError(w, http.StatusBadRequest, "invalid_request_error", "no user message to feed channel worker")
		return
	}

	chatID := openai.GenerateChatID()
	created := time.Now().Unix()

	p := spawnParams{
		args:         buildChannelArgs(req, model, systemPrompt),
		workdir:      req.WorkingDir,
		envVars:      req.EnvVars,
		systemPrompt: systemPrompt,
		model:        model,
	}

	worker, err := channelMgr.acquire(sessionID, p)
	if err != nil {
		log.Printf("[channel] acquire failed session=%s: %v", sessionID, err)
		openai.WriteError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	lines, err := worker.beginTurn(content)
	if err != nil {
		// Worker died between acquire and beginTurn (e.g. reaped). One retry
		// with a fresh acquire (which will spawn/resume).
		log.Printf("[channel] beginTurn failed session=%s: %v; retrying acquire", sessionID, err)
		if worker, err = channelMgr.acquire(sessionID, p); err == nil {
			lines, err = worker.beginTurn(content)
		}
		if err != nil {
			openai.WriteError(w, http.StatusInternalServerError, "server_error", err.Error())
			return
		}
	}
	log.Printf("[channel] turn start session=%s claude_sid=%s flag=%s model=%s chat=%s",
		sessionID, worker.SessionID(), worker.usedFlag, model, chatID)

	// Exactly one of these takes ownership of releasing the worker (endTurn):
	// the foreground path on normal completion, or a background drainer when we
	// interrupt (stop / AskUserQuestion). turnReleased guards against both.
	turnReleased := false
	releaseInBackground := func(reason string) {
		if turnReleased {
			return
		}
		turnReleased = true
		log.Printf("[channel] interrupt session=%s reason=%s (keeping process alive)", sessionID, reason)
		_ = worker.interrupt()
		// Hand the turn to a background drainer so the HTTP handler returns now
		// (the client is gone / has its [DONE]); the process is kept alive for
		// the next request on this session. A backstop kills only a genuinely
		// stuck turn — claude's interrupt for in-progress text lands at the next
		// message boundary, so normal turns finish well within the window and
		// their hot process is preserved.
		timer := time.AfterFunc(channelInterruptBackstop, func() {
			log.Printf("[channel] interrupt backstop fired session=%s; killing worker", sessionID)
			worker.kill()
		})
		go func() {
			for range lines { //nolint:revive // drain until result/EOF closes it
			}
			timer.Stop()
			worker.endTurn()
		}()
	}
	defer func() {
		if !turnReleased {
			worker.endTurn()
		}
	}()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	flusher, isFlusher := w.(http.Flusher)
	if !isFlusher {
		log.Printf("[channel] streaming not supported; interrupting + releasing")
		releaseInBackground("not_flushable")
		return
	}

	fmt.Fprintf(w, ": ping\n\n")
	flusher.Flush()

	heartbeat := time.NewTicker(30 * time.Second)
	defer heartbeat.Stop()

	t := newSSETranslator(chatID, created, model, sessionID)
	ctxCh := r.Context().Done()

	for {
		select {
		case <-ctxCh:
			// Upstream disconnected = stop. Interrupt (not kill) and release in
			// the background; the process and its context survive.
			releaseInBackground("client_disconnect")
			return
		case <-heartbeat.C:
			fmt.Fprintf(w, ": keepalive\n\n")
			flusher.Flush()
		case line, lok := <-lines:
			if !lok {
				// Turn's result closed the channel: normal completion.
				t.flushAggLog()
				sessionStore.LogDone(sessionID, t.StreamUsage())
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				log.Printf("[channel] turn end session=%s (completed)", sessionID)
				return
			}
			if line == "" {
				continue
			}
			if t.feed(w, flusher, line, includeUsage) == outcomeAskUserDone {
				// §4.5: tool_call + finish + [DONE] already emitted. Interrupt
				// the hung turn but keep the process alive — the user's answer
				// arrives as a new request on the same session_id.
				releaseInBackground("ask_user_question")
				return
			}
		}
	}
}
