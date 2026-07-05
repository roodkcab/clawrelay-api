package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os/exec"
	"strings"
	"time"

	"clawrelay-api/pkg/openai"
	"clawrelay-api/pkg/proc"
)

// handleStreamResponse pipes codex JSONL events through to the client as an
// OpenAI-shaped SSE stream. Each codex item becomes one or more OpenAI deltas:
//
//   thread.started          → captured into threadMap, no client output
//   turn.started            → (silent)
//   item.started cmd_exec   → tool_calls delta naming "shell" with command
//   item.completed agent    → text content delta carrying the full message
//   item.completed cmd_exec → tool_calls delta closing the shell call
//   item.completed reasoning→ thinking delta with the reasoning text
//   turn.completed          → finish_reason chunk + usage chunk
//   error / turn.failed     → server_error response and stream end
//
// rebuildFresh drops the stale thread binding and rebuilds the input as a
// fresh full-history session; it is invoked when a resume dies with zero
// stdout output (see the first-line probe below).
func handleStreamResponse(w http.ResponseWriter, r *http.Request, input codexInput, chatID string, created int64, model string, includeUsage bool, workingDir string, envVars map[string]string, sessionID string, rebuildFresh func() codexInput) {
	cmd, lines, waitErr, err := launchCodex(input, workingDir, envVars)
	if err != nil {
		openai.WriteError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	// ---- First-line probe (CODEX-1) ----------------------------------------
	// SSE headers are NOT written yet, so we can still answer with a real HTTP
	// error. Known failure mode (verified empirically): resuming a thread whose
	// rollout file is gone (~/.codex/sessions cleaned, container rebuilt) makes
	// codex exit(1) with COMPLETELY EMPTY stdout — the "no rollout found for
	// thread id ..." text goes to stderr only, so no "error" event ever arrives
	// and the old code streamed a fake empty success. We detect that as "lines
	// closed before the first line", forget the dead binding and retry once as
	// a fresh session with the full history replayed.
	var firstLine string
	haveFirst := false
probe:
	for attempt := 0; ; attempt++ {
		select {
		case <-r.Context().Done():
			proc.KillGroup(cmd)    // client gone before first byte: nothing to harvest
			proc.DrainLines(lines) // unblock producer so it reaches cmd.Wait()
			return
		case l, ok := <-lines:
			if ok {
				firstLine, haveFirst = l, true
				break probe
			}
			// Zero stdout output: codex died before emitting anything.
			werr := waitErr()
			if r.Context().Err() != nil {
				// 断连竞态：零输出可能是连接取消连带杀进程所致，不是死线程
				// 签名 —— 不 Forget、不重试。
				log.Printf("[CODEX] no output + client already gone; not retrying session_id=%s", sessionID)
				return
			}
			if input.IsResume && rebuildFresh != nil && attempt == 0 {
				log.Printf("[CODEX] resume produced no output (exit err: %v) — forgetting stale thread for session_id=%s, retrying as fresh session", werr, sessionID)
				input = rebuildFresh()
				log.Printf("codex retry args: %v (stdin %d bytes)", input.Args, len(input.Stdin))
				cmd, lines, waitErr, err = launchCodex(input, workingDir, envVars)
				if err != nil {
					openai.WriteError(w, http.StatusInternalServerError, "server_error", err.Error())
					return
				}
				continue
			}
			msg := "codex produced no output"
			if werr != nil {
				msg += ": " + werr.Error()
			}
			openai.WriteError(w, http.StatusInternalServerError, "server_error", msg)
			return
		}
	}

	// Disconnect handling is inline in the processLines loop below (single
	// goroutine watching ctx), so there is no watcher/loop race on ctx.

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	flusher, ok := w.(http.Flusher)
	if !ok {
		log.Printf("Streaming not supported by ResponseWriter")
		proc.KillGroup(cmd)
		proc.DrainLines(lines)
		return
	}

	fmt.Fprintf(w, ": ping\n\n")
	flusher.Flush()

	// Codex turns can take >30s on shell commands; without periodic keepalive
	// the client's idle-read timeout will fire.
	heartbeat := time.NewTicker(30 * time.Second)
	defer heartbeat.Stop()

	emit := func(chunk openai.ChatCompletionResponse) {
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	textDelta := func(text string) {
		emit(openai.ChatCompletionResponse{
			ID: chatID, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []openai.ChatCompletionChoice{{
				Index: 0,
				Delta: openai.NewChatMessage("assistant", text),
			}},
		})
	}

	thinkingDelta := func(text string) {
		emit(openai.ChatCompletionResponse{
			ID: chatID, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []openai.ChatCompletionChoice{{
				Index: 0,
				Delta: &openai.ChatMessage{Role: "assistant", Thinking: text},
			}},
		})
	}

	toolCallDelta := func(id, name, args string) {
		emit(openai.ChatCompletionResponse{
			ID: chatID, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []openai.ChatCompletionChoice{{
				Index: 0,
				Delta: &openai.ChatMessage{
					Role: "assistant",
					ToolCalls: []openai.ToolCall{{
						ID:   id,
						Type: "function",
						Function: openai.ToolCallFunction{Name: name, Arguments: args},
					}},
				},
			}},
		})
	}

	finishChunk := func(reason string) {
		emit(openai.ChatCompletionResponse{
			ID: chatID, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []openai.ChatCompletionChoice{{
				Index: 0, Delta: openai.NewChatMessage("", ""), FinishReason: &reason,
			}},
		})
	}

	usageChunk := func(u *openai.UsageInfo) {
		emit(openai.ChatCompletionResponse{
			ID: chatID, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []openai.ChatCompletionChoice{},
			Usage:   u,
		})
	}

	var streamUsage *openai.UsageInfo
	var threadIDSeen string
	turnFailed := false
	emittedAnyContent := false
	sawEvent := false // any successfully parsed JSONL event

	handleLine := func(raw string) {
		line := strings.TrimSpace(raw)
		if line == "" || !strings.HasPrefix(line, "{") {
			// codex sometimes writes non-JSON lines (banner, error logs) to
			// stdout. Skip silently rather than confusing the client.
			if line != "" {
				log.Printf("[CODEX NON-JSON] %s", line)
			}
			return
		}

		log.Printf("[CODEX RAW] %s", openai.Truncate(line, 500))

		var ev codexEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			log.Printf("[CODEX PARSE ERR] %v: %s", err, line)
			return
		}
		sawEvent = true

		switch ev.Type {
		case "thread.started":
			threadIDSeen = ev.ThreadID
			log.Printf("[CODEX] thread_id=%s session_id=%s", ev.ThreadID, sessionID)
			rebindThread(sessionID, ev.ThreadID)

		case "turn.started":
			// no-op

		case "item.started":
			if ev.Item == nil {
				return
			}
			if ev.Item.Type == "command_execution" {
				// Surface to client as a tool_call so the UI can render
				// "Codex is running: <command>" indicators.
				args, _ := json.Marshal(map[string]string{"command": ev.Item.Command})
				toolCallDelta(ev.Item.ID, "shell", string(args))
			}

		case "item.completed":
			if ev.Item == nil {
				return
			}
			switch ev.Item.Type {
			case "agent_message":
				if ev.Item.Text != "" {
					emittedAnyContent = true
					textDelta(ev.Item.Text)
					sessionStore.LogDelta(sessionID, ev.Item.Text)
				}
			case "reasoning":
				if ev.Item.Text != "" {
					thinkingDelta(ev.Item.Text)
				}
			case "command_execution":
				// Log tool result for /sessions viewer.
				exit := ""
				if ev.Item.ExitCode != nil {
					exit = openai.FmtInt(*ev.Item.ExitCode)
				}
				sessionStore.LogToolUse(sessionID, "shell", ev.Item.ID,
					fmt.Sprintf(`{"command":%q,"exit_code":%s,"output":%q}`,
						ev.Item.Command, defaultStr(exit, "null"),
						openai.Truncate(ev.Item.AggregatedOutput, 4096)))
			}

		case "turn.completed":
			if ev.Usage != nil {
				input, output, cacheRead := turnUsage(sessionID, ev.Usage)
				stats.Record(model, input, output, 0, cacheRead, 0)
				log.Printf("Token usage: model=%s input=%d output=%d cache_read=%d (codex, per-turn)",
					model, input, output, cacheRead)
				streamUsage = openai.BuildUsageInfo(input, output, cacheRead, 0)
			}

		case "error", "turn.failed":
			turnFailed = true
			errMsg := ev.Message
			if errMsg == "" && len(ev.Error) > 0 {
				errMsg = string(ev.Error)
			}
			log.Printf("[CODEX ERROR] %s", errMsg)
			// CODEX-3: some codex versions attach usage to the failure event.
			// Record it when present so failed turns still get metered; most
			// failures carry no usage and that's fine (best-effort). The
			// figures are thread-cumulative like any other, so they go
			// through the same per-turn diff (PR #27).
			if ev.Usage != nil {
				inp, out, cr := turnUsage(sessionID, ev.Usage)
				stats.Record(model, inp, out, 0, cr, 0)
				streamUsage = openai.BuildUsageInfo(inp, out, cr, 0)
			}
			// If a resume failed because the thread is gone, drop the stale
			// binding so the next turn starts fresh. Verified real-world text
			// is "no rollout found for thread id ..." — it contains neither
			// "session" nor a stable code, hence the three-noun match. The
			// usage baseline is tied to that thread's cumulative counter, so
			// drop it too.
			if sessionID != "" && input.IsResume && isStaleThreadErr(errMsg) {
				threads.Forget(sessionID)
				meter.Forget(sessionID)
			}
			textDelta("\n\n[codex error] " + errMsg)
			emittedAnyContent = true
		}
	}

	// The probed first line must flow through the same pipeline as the rest.
	if haveFirst {
		handleLine(firstLine)
	}

processLines:
	for {
		select {
		case <-r.Context().Done():
			// CODEX-2: client disconnected mid-turn. SIGINT + background
			// harvest instead of an immediate SIGKILL, so a late
			// turn.completed usage / thread.started binding isn't lost.
			abortCodexAndHarvest(cmd, lines, model, sessionID)
			return
		case <-heartbeat.C:
			fmt.Fprintf(w, ": keepalive\n\n")
			flusher.Flush()
		case line, ok := <-lines:
			if !ok {
				break processLines
			}
			handleLine(line)
		}
	}

	// CODEX-5 fallback: headers are already written so we can't switch to an
	// HTTP error anymore. If the whole stream produced no parseable event and
	// the process exited abnormally (e.g. crashed after the first non-JSON
	// banner line), surface it as visible text instead of a fake empty success.
	if !sawEvent {
		if werr := waitErr(); werr != nil {
			textDelta("⚠️ [codex] 进程异常退出: " + werr.Error())
			emittedAnyContent = true
		}
	}

	// Defensive: if no content was emitted (rare — empty turn or upstream
	// failure with no error event) ensure the stream still terminates cleanly.
	if !emittedAnyContent {
		textDelta("")
	}

	reason := "stop"
	if turnFailed {
		reason = "stop"
	}
	finishChunk(reason)

	if includeUsage && streamUsage != nil {
		usageChunk(streamUsage)
	}

	sessionStore.LogDone(sessionID, streamUsage)

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()

	_ = threadIDSeen // captured for clarity in logs above
}

// handleNonStreamResponse runs codex to completion and returns one OpenAI
// chat.completion JSON. Like the stream path, a resume that dies with zero
// stdout output (dead thread rollout) is retried once as a fresh session via
// rebuildFresh (CODEX-1).
func handleNonStreamResponse(w http.ResponseWriter, r *http.Request, input codexInput, chatID string, created int64, model string, workingDir string, envVars map[string]string, sessionID string, rebuildFresh func() codexInput) {
	cmd, lines, waitErr, err := launchCodex(input, workingDir, envVars)
	if err != nil {
		openai.WriteError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	// Note: WatchDisconnect captures the ORIGINAL lines channel; after a retry
	// relaunch the handler itself keeps ranging over the new channel until the
	// killed process closes it, so the producer never wedges. getCmd is a
	// closure so the watcher always signals the current child.
	defer proc.WatchDisconnect(r.Context(), func() *exec.Cmd { return cmd }, lines)()

	var (
		fullText     strings.Builder
		usage        *openai.UsageInfo
		errorMsg     string
		threadIDSeen string
	)

	for attempt := 0; ; attempt++ {
		sawEvent := false // any successfully parsed JSONL event this run

		for line := range lines {
			line = strings.TrimSpace(line)
			if line == "" || !strings.HasPrefix(line, "{") {
				if line != "" {
					log.Printf("[CODEX NON-JSON] %s", line)
				}
				continue
			}
			log.Printf("[CODEX RAW] %s", openai.Truncate(line, 500))

			var ev codexEvent
			if err := json.Unmarshal([]byte(line), &ev); err != nil {
				log.Printf("[CODEX PARSE ERR] %v: %s", err, line)
				continue
			}
			sawEvent = true

			switch ev.Type {
			case "thread.started":
				threadIDSeen = ev.ThreadID
				rebindThread(sessionID, ev.ThreadID)
			case "item.completed":
				if ev.Item == nil {
					continue
				}
				if ev.Item.Type == "agent_message" && ev.Item.Text != "" {
					fullText.WriteString(ev.Item.Text)
				}
			case "turn.completed":
				if ev.Usage != nil {
					inp, out, cr := turnUsage(sessionID, ev.Usage)
					stats.Record(model, inp, out, 0, cr, 0)
					usage = openai.BuildUsageInfo(inp, out, cr, 0)
				}
			case "error", "turn.failed":
				errorMsg = ev.Message
				if errorMsg == "" && len(ev.Error) > 0 {
					errorMsg = string(ev.Error)
				}
				// CODEX-3: failure events may still carry usage — meter them.
				if ev.Usage != nil {
					inp, out, cr := turnUsage(sessionID, ev.Usage)
					stats.Record(model, inp, out, 0, cr, 0)
					usage = openai.BuildUsageInfo(inp, out, cr, 0)
				}
				if sessionID != "" && input.IsResume && isStaleThreadErr(errorMsg) {
					threads.Forget(sessionID)
					meter.Forget(sessionID)
				}
			}
		}

		if sawEvent {
			break
		}

		// Zero parseable output: same dead-thread signature as the stream
		// path (codex exit=1, stdout empty, error only on stderr). BUT a client
		// disconnect produces the SAME signature via WatchDisconnect's
		// KillGroup — that must NOT forget a valid binding nor relaunch an
		// unwatched orphan run (CODEX review C1).
		werr := waitErr()
		if r.Context().Err() != nil {
			log.Printf("[CODEX] no output because client disconnected (non-stream); not retrying session_id=%s", sessionID)
			return
		}
		if input.IsResume && rebuildFresh != nil && attempt == 0 {
			log.Printf("[CODEX] resume produced no output (exit err: %v) — forgetting stale thread for session_id=%s, retrying as fresh session (non-stream)", werr, sessionID)
			input = rebuildFresh()
			log.Printf("codex retry args: %v (stdin %d bytes)", input.Args, len(input.Stdin))
			// 旧进程已被 cmd.Wait 回收：重建窗口（附件解码+fork，可达几十 ms）内
			// 若断连，WatchDisconnect 不能对可能已复用的旧 PID 发信号。
			cmd = nil
			cmd2, lines2, waitErr2, lerr := launchCodex(input, workingDir, envVars)
			if lerr != nil {
				openai.WriteError(w, http.StatusInternalServerError, "server_error", lerr.Error())
				return
			}
			cmd, lines, waitErr = cmd2, lines2, waitErr2
			continue
		}
		msg := "codex produced no output"
		if werr != nil {
			msg += ": " + werr.Error()
		}
		openai.WriteError(w, http.StatusInternalServerError, "server_error", msg)
		return
	}

	finishReason := "stop"
	body := fullText.String()
	if errorMsg != "" {
		if body == "" {
			openai.WriteError(w, http.StatusInternalServerError, "server_error", "codex: "+errorMsg)
			return
		}
		body += "\n\n[codex error] " + errorMsg
	}
	if body == "" {
		openai.WriteError(w, http.StatusInternalServerError, "server_error", "Empty response from codex")
		return
	}

	resp := openai.ChatCompletionResponse{
		ID: chatID, Object: "chat.completion", Created: created, Model: model,
		Choices: []openai.ChatCompletionChoice{{
			Index:        0,
			Message:      openai.NewChatMessage("assistant", body),
			FinishReason: &finishReason,
		}},
		Usage: usage,
	}

	sessionStore.LogDelta(sessionID, body)
	sessionStore.LogDone(sessionID, usage)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)

	_ = threadIDSeen
}

// isStaleThreadErr reports whether a codex error message indicates the
// resumed thread no longer exists. Verified real-world text is
// "no rollout found for thread id ..." — it contains NEITHER "session" nor a
// stable error code, so we match the three nouns codex uses across versions
// ("session" kept for older/other builds).
func isStaleThreadErr(errMsg string) bool {
	m := strings.ToLower(errMsg)
	// 主签名（实证）："no rollout found for thread id ..."
	if strings.Contains(m, "no rollout found") {
		return true
	}
	// 其余必须是「对象 + 不存在」的组合才算：单凭名词命中就 Forget 的话，
	// Rust panic 文本（"thread '...' panicked"）、限流提示（"usage limit
	// reached for this session"）这类瞬态错误会误删有效绑定，把可自愈故障
	// 放大成不可逆的整会话上下文丢失。
	gone := strings.Contains(m, "not found") ||
		strings.Contains(m, "no longer exists") ||
		strings.Contains(m, "does not exist") ||
		strings.Contains(m, "missing") ||
		strings.Contains(m, "expired")
	return gone && (strings.Contains(m, "thread") ||
		strings.Contains(m, "session") ||
		strings.Contains(m, "rollout"))
}

// abortCodexAndHarvest handles a client disconnect mid-stream: SIGINT the
// codex process group (giving the CLI a chance to flush terminal events),
// then spend up to 8s in the background parsing whatever it still writes so
// a turn.completed usage payload — and a late thread.started binding — are
// not lost, before SIGKILLing whatever is left and draining the channel.
//
// NOTE: whether codex actually emits turn.completed (with usage) after a
// SIGINT is UNVERIFIED (no codex credentials on the dev box); claude
// demonstrably does. Strictly best-effort: if nothing harvestable arrives
// within the window we only lose what was already lost, and the KillGroup
// fallback guarantees no process leak either way.
func abortCodexAndHarvest(cmd *exec.Cmd, lines <-chan string, model, sessionID string) {
	if cmd != nil && cmd.Process != nil {
		log.Printf("[CODEX ABORT] client disconnected, SIGINT process group pid=%d (session_id=%s)", cmd.Process.Pid, sessionID)
	}
	proc.InterruptGroup(cmd)
	go func() {
		deadline := time.NewTimer(8 * time.Second)
		defer deadline.Stop()
		for {
			select {
			case <-deadline.C:
				log.Printf("[CODEX ABORT] harvest window expired for session_id=%s, killing process group", sessionID)
				proc.KillGroup(cmd)
				proc.DrainLines(lines)
				return
			case line, ok := <-lines:
				if !ok {
					// Producer closed the channel → child already reaped by
					// cmd.Wait(). Do NOT KillGroup here: the PID may have been
					// reused and we'd signal an unrelated process group.
					return
				}
				line = strings.TrimSpace(line)
				if line == "" || !strings.HasPrefix(line, "{") {
					continue
				}
				var ev codexEvent
				if json.Unmarshal([]byte(line), &ev) != nil {
					continue
				}
				switch ev.Type {
				case "thread.started":
					// Bind even during abort so the NEXT turn can resume this
					// thread instead of replaying the whole history (rebind
					// drops a foreign baseline if the thread rotated).
					rebindThread(sessionID, ev.ThreadID)
				case "turn.completed":
					if ev.Usage != nil {
						// Same per-turn diff as the foreground paths: the
						// harvested figures are thread-cumulative, and going
						// through turnUsage also advances the baseline so the
						// NEXT turn doesn't re-count this one.
						inp, out, cr := turnUsage(sessionID, ev.Usage)
						stats.Record(model, inp, out, 0, cr, 0)
						log.Printf("Token usage: model=%s input=%d output=%d cache_read=%d (codex, harvested after client abort)",
							model, inp, out, cr)
					}
				}
			}
		}
	}()
}

// rebindThread records the codex thread_id for a session. If codex rotated to a
// DIFFERENT thread (a resume that silently started a fresh thread instead of
// erroring), the old usage baseline is foreign — the new thread's cumulative
// counter restarts from zero — so drop it before rebinding, otherwise the next
// turn would diff against a stale baseline and mis-bill. No-op when meter is nil
// (handler invoked without main()'s init, e.g. tests).
//
// The Get-then-Set is not atomic, but turns for one session are effectively
// serial (a conversation is request/response, and codex can't resume one thread
// concurrently), so the rotation-during-concurrent-turn window is not a concern
// in practice.
func rebindThread(sessionID, threadID string) {
	if sessionID == "" || threadID == "" {
		return
	}
	if meter != nil {
		if prev := threads.Get(sessionID); prev != "" && prev != threadID {
			meter.Forget(sessionID)
		}
	}
	threads.Set(sessionID, threadID)
}

// turnUsage resolves one turn's reportable (input, output, cache_read) from
// codex's reported usage. codex reports thread-cumulative totals on resume, so
// the meter diffs against the session's baseline to recover this turn's figure;
// without that diff a resumed thread's usage climbs monotonically and every turn
// re-counts all earlier turns.
//
// Assumes exactly one turn.completed per request (codex's `exec` contract): each
// call advances the persisted baseline, so multiple turn.completed in one request
// would each bill their own delta correctly but only the last would reach the
// client's usage chunk.
func turnUsage(sessionID string, u *codexUsage) (input, output, cacheRead int) {
	if meter == nil { // defensive: handler invoked without main()'s init (tests)
		return mapUsage(u)
	}
	return meter.perTurn(sessionID, *u)
}

// mapUsage converts codex's usage shape to the (input, output, cache_read)
// triple used by openai.BuildUsageInfo. codex.input_tokens is *total* prompt
// (including cached); we subtract cached_input_tokens to get the uncached
// portion that goes into PromptTokens-without-cache.
func mapUsage(u *codexUsage) (input, output, cacheRead int) {
	output = u.OutputTokens
	cacheRead = u.CachedInputTokens
	input = u.InputTokens - u.CachedInputTokens
	if input < 0 {
		input = u.InputTokens
		cacheRead = 0
	}
	return
}

func defaultStr(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}
