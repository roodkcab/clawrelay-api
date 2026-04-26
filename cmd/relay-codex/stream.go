package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"clawrelay-api/pkg/openai"
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
func handleStreamResponse(w http.ResponseWriter, r *http.Request, input codexInput, chatID string, created int64, model string, includeUsage bool, workingDir string, envVars map[string]string, sessionID string) {
	cmd, lines, err := launchCodex(input, workingDir, envVars)
	if err != nil {
		openai.WriteError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	// Kill subprocess if client drops the SSE connection.
	go func() {
		<-r.Context().Done()
		if cmd != nil && cmd.Process != nil {
			log.Printf("Client disconnected, killing codex process pid=%d", cmd.Process.Pid)
			cmd.Process.Kill()
		}
	}()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	flusher, ok := w.(http.Flusher)
	if !ok {
		log.Printf("Streaming not supported by ResponseWriter")
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

processLines:
	for {
		var line string
		var ok bool
		select {
		case <-r.Context().Done():
			return
		case <-heartbeat.C:
			fmt.Fprintf(w, ": keepalive\n\n")
			flusher.Flush()
			continue
		case line, ok = <-lines:
			if !ok {
				break processLines
			}
		}
		line = strings.TrimSpace(line)
		if line == "" || !strings.HasPrefix(line, "{") {
			// codex sometimes writes non-JSON lines (banner, error logs) to
			// stdout. Skip silently rather than confusing the client.
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

		switch ev.Type {
		case "thread.started":
			threadIDSeen = ev.ThreadID
			log.Printf("[CODEX] thread_id=%s session_id=%s", ev.ThreadID, sessionID)
			if sessionID != "" {
				threads.Set(sessionID, ev.ThreadID)
			}

		case "turn.started":
			// no-op

		case "item.started":
			if ev.Item == nil {
				continue
			}
			if ev.Item.Type == "command_execution" {
				// Surface to client as a tool_call so the UI can render
				// "Codex is running: <command>" indicators.
				args, _ := json.Marshal(map[string]string{"command": ev.Item.Command})
				toolCallDelta(ev.Item.ID, "shell", string(args))
			}

		case "item.completed":
			if ev.Item == nil {
				continue
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
				input, output, cacheRead := mapUsage(ev.Usage)
				stats.Record(model, input, output, 0, cacheRead, 0)
				log.Printf("Token usage: model=%s input=%d output=%d cache_read=%d (codex)",
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
			// If a resume failed because the thread is gone, drop the
			// stale binding so the next turn starts fresh.
			if sessionID != "" && input.IsResume && strings.Contains(errMsg, "session") {
				threads.Forget(sessionID)
			}
			textDelta("\n\n[codex error] " + errMsg)
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
// chat.completion JSON.
func handleNonStreamResponse(w http.ResponseWriter, r *http.Request, input codexInput, chatID string, created int64, model string, workingDir string, envVars map[string]string, sessionID string) {
	cmd, lines, err := launchCodex(input, workingDir, envVars)
	if err != nil {
		openai.WriteError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	go func() {
		<-r.Context().Done()
		if cmd != nil && cmd.Process != nil {
			cmd.Process.Kill()
		}
	}()

	var (
		fullText  strings.Builder
		usage     *openai.UsageInfo
		errorMsg  string
		threadIDSeen string
	)

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

		switch ev.Type {
		case "thread.started":
			threadIDSeen = ev.ThreadID
			if sessionID != "" {
				threads.Set(sessionID, ev.ThreadID)
			}
		case "item.completed":
			if ev.Item == nil {
				continue
			}
			if ev.Item.Type == "agent_message" && ev.Item.Text != "" {
				fullText.WriteString(ev.Item.Text)
			}
		case "turn.completed":
			if ev.Usage != nil {
				inp, out, cr := mapUsage(ev.Usage)
				stats.Record(model, inp, out, 0, cr, 0)
				usage = openai.BuildUsageInfo(inp, out, cr, 0)
			}
		case "error", "turn.failed":
			errorMsg = ev.Message
			if errorMsg == "" && len(ev.Error) > 0 {
				errorMsg = string(ev.Error)
			}
			if sessionID != "" && input.IsResume && strings.Contains(errorMsg, "session") {
				threads.Forget(sessionID)
			}
		}
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
