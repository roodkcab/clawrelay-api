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
	// No --print: the relay captures stdout via a pipe (not a TTY), which alone
	// puts claude into non-interactive mode (per `claude --help`: "via -p, or
	// when stdout is not a TTY"). stream-json input/output therefore works
	// without the explicit flag. Verified: multi-turn + interrupt behave
	// identically to the --print variant.
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

// handleChannelEphemeralStreamResponse serves a request that has NO session_id
// while in channel mode. Rather than degrade to the legacy `claude -p` path, it
// runs the request through the same stream-json channel mechanism on a fresh,
// throwaway process: mint a brand-new session_id, spawn one `--input-format
// stream-json` process, feed the full conversation, stream the turn, then kill
// the process. No reuse, no pooling — each independent (e.g. cron) request gets
// its own isolated run, so contexts never cross-contaminate.
//
// The conversation is flattened exactly as the legacy path would build it (no
// prior process context exists), so what claude receives is identical to
// legacy; only the delivery channel differs (stdin stream-json vs `-p`).
func handleChannelEphemeralStreamResponse(w http.ResponseWriter, r *http.Request, req *openai.ChatCompletionRequest, model string, includeUsage bool) {
	// No session → temp attachments, full-conversation flatten (== legacy, so
	// what claude receives is identical; only the delivery channel differs). No
	// empty-prompt rejection here: the legacy path accepts it, so we match it.
	prompt, systemPrompt, tempFiles := buildPromptFromMessages(req.Messages, "")
	if len(tempFiles) > 0 {
		defer func() {
			for _, f := range tempFiles {
				os.Remove(f)
			}
		}()
	}

	chatID := openai.GenerateChatID()
	created := time.Now().Unix()
	sid := newUUID()

	args := append(buildChannelArgs(req, model, systemPrompt), "--session-id", sid)
	worker, err := spawnChanWorker(sid, args, req.WorkingDir, req.EnvVars, "--session-id", func() {})
	if err != nil {
		log.Printf("[channel] ephemeral spawn failed: %v", err)
		openai.WriteError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	// Track so SIGTERM shutdown reaps it and /channels can see it.
	channelMgr.trackInflight(worker)

	lines, err := worker.beginTurn(prompt)
	if err != nil {
		channelMgr.untrackInflight(worker)
		worker.kill()
		openai.WriteError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	// The throwaway process is always killed when this turn ends (completion,
	// stop, or AskUserQuestion). kill() abandons the active turn (unblocking the
	// stdout reader), and the background drain lets all worker goroutines exit.
	finished := false
	finish := func() {
		if finished {
			return
		}
		finished = true
		channelMgr.untrackInflight(worker)
		worker.kill()
		go func() {
			for {
				select {
				case _, ok := <-lines:
					if !ok {
						return
					}
				case <-worker.deadCh:
					return // process reaped; lines may never close
				}
			}
		}()
	}
	defer finish()

	log.Printf("[channel] ephemeral turn start session=%s model=%s chat=%s", sid, model, chatID)

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	flusher, isFlusher := w.(http.Flusher)
	if !isFlusher {
		log.Printf("[channel] ephemeral: streaming not supported; killing")
		return
	}

	fmt.Fprintf(w, ": ping\n\n")
	flusher.Flush()

	heartbeat := time.NewTicker(30 * time.Second)
	defer heartbeat.Stop()

	t := newSSETranslator(chatID, created, model, "", identityMeter{}) // no session → log no-ops
	for {
		select {
		case <-r.Context().Done():
			// Upstream disconnected = stop. Ephemeral run: just kill it.
			log.Printf("[channel] ephemeral stop session=%s (kill)", sid)
			return
		case <-worker.deadCh:
			// Process died mid-turn; `lines` may never close. Finalize and exit
			// rather than block forever (finish kills + drains).
			t.flushAggLog()
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			log.Printf("[channel] ephemeral turn end session=%s (process died)", sid)
			return
		case <-heartbeat.C:
			fmt.Fprintf(w, ": keepalive\n\n")
			flusher.Flush()
		case line, lok := <-lines:
			if !lok {
				t.flushAggLog()
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				log.Printf("[channel] ephemeral turn end session=%s (completed)", sid)
				return
			}
			if line == "" {
				continue
			}
			if t.feed(w, flusher, line, includeUsage) == outcomeAskUserDone {
				// No session to continue on; the card was emitted, kill the run.
				log.Printf("[channel] ephemeral turn end session=%s (ask_user)", sid)
				return
			}
		}
	}
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
			for {
				select {
				case ln, ok := <-lines:
					if !ok {
						timer.Stop()
						worker.endTurn()
						return
					}
					advanceMeterFromLine(worker.meter, ln)
				case <-worker.deadCh:
					// Process died; lines may never close. Stop waiting so the
					// worker is released rather than leaking this goroutine.
					timer.Stop()
					worker.endTurn()
					return
				}
			}
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

	t := newSSETranslator(chatID, created, model, sessionID, worker.meter)
	ctxCh := r.Context().Done()

	for {
		select {
		case <-ctxCh:
			// Upstream disconnected = stop. Interrupt (not kill) and release in
			// the background; the process and its context survive.
			releaseInBackground("client_disconnect")
			return
		case <-worker.deadCh:
			// Process died mid-turn; `lines` may never close. Finalize and let
			// the deferred endTurn release the worker.
			t.flushAggLog()
			sessionStore.LogDone(sessionID, t.StreamUsage())
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			log.Printf("[channel] turn end session=%s (process died)", sessionID)
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
