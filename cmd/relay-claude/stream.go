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

// heartbeatInterval paces the SSE comment heartbeats (`: keepalive`) that keep
// idle-timeout proxies and clients from cutting a quiet stream. Package var so
// tests can shorten it.
var heartbeatInterval = 30 * time.Second

// abortClaudeAndHarvestUsage 中断 V1 进程并在后台限时收割该轮 usage：
// SIGINT → 最多 10s 内解析残余行中的 result（modelUsage 有真实值）→
// stats.RecordTurn 记账（+日志）→ KillGroup 兜底 + 排空。
//
// 实证依据（claude 2.1.199）：被 SIGINT 后 claude 仍会 emit result 事件
// （subtype=error_during_execution），bare usage 全 0，但 modelUsage 各条目带
// 真实 token 和 costUSD，顶层 total_cost_usd 也有值 —— 直接 KillGroup 会把这
// 一轮的消耗整个丢掉，这正是 V1 中断轮从不记账的根因。
func abortClaudeAndHarvestUsage(cmd *exec.Cmd, lines <-chan string, model, sessionID string) {
	proc.InterruptGroup(cmd)
	go func() {
		deadline := time.After(10 * time.Second)
	harvest:
		for {
			select {
			case line, ok := <-lines:
				if !ok {
					break harvest // producer closed: process exited without a result
				}
				if line == "" {
					continue
				}
				var event claudeEvent
				if err := json.Unmarshal([]byte(line), &event); err != nil {
					continue
				}
				if event.Type != "result" {
					continue
				}
				if pm := perModelCounts(&event, model); pm != nil {
					stats.RecordTurn(model, pm)
					log.Printf("interrupted-turn usage recorded: model=%s session=%s per_model=%+v total_cost=$%.4f",
						model, sessionID, pm, event.TotalCostUSD)
				}
				break harvest
			case <-deadline:
				log.Printf("interrupted-turn usage harvest timed out (10s): model=%s session=%s", model, sessionID)
				break harvest
			}
		}
		// Backstop kill (idempotent if SIGINT already ended it) and drain the
		// remaining lines so the producer goroutine reaches cmd.Wait().
		proc.KillGroup(cmd)
		for range lines { //nolint:revive // intentional drain
		}
	}()
}

// sseHandshake sends the SSE response headers and the first `: ping` BEFORE
// claude is ready. ready waits for the CLI's first stdout line (the resume
// sniff path even runs a whole first process), so a slow-starting or hung CLI
// used to leave the client with zero bytes — not even response headers — until
// its read timeout cut the connection (wuji-tools: sock_read=120s →
// "AI 连接出现错误"). Same approach as the V2 channel handler.
//
// Returns ok=false after cleaning up when streaming is unsupported. Once this
// returns ok=true the status code is out: every later failure must close the
// stream with an in-stream error chunk, not an HTTP error.
func sseHandshake(w http.ResponseWriter, ready <-chan error, handle *procHandle, lines <-chan string) (http.Flusher, *time.Ticker, bool) {
	flusher, isFlusher := w.(http.Flusher)
	if !isFlusher {
		// 部署/中间件问题：没有 flusher 做不了 SSE。头还没发，用标准 500，
		// 后台等进程起来后杀掉排空。
		log.Printf("Streaming not supported")
		openai.WriteError(w, http.StatusInternalServerError, "server_error", "streaming unsupported by this connection")
		go func() {
			if err := <-ready; err == nil {
				if cmd := handle.abort(); cmd != nil {
					proc.KillGroup(cmd)
				}
			}
			proc.DrainLines(lines)
		}()
		return nil, nil, false
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, ": ping\n\n")
	flusher.Flush()

	return flusher, time.NewTicker(heartbeatInterval), true
}

// waitClaudeReady blocks until startClaudeStream's ready fires, keeping the
// (already-started) SSE stream alive with comment heartbeats and honoring
// client disconnects. Startup failures — fork error, zero-output crash,
// first-line watchdog kill, failed resume retry — are emitted as a visible
// in-stream error chunk + [DONE] (the status code is long gone), so the
// upstream client renders the reason instead of an empty reply or a timeout.
// Returns false when the handler must stop (error emitted or client gone).
func waitClaudeReady(w http.ResponseWriter, r *http.Request, flusher http.Flusher, heartbeatTicker *time.Ticker, ready <-chan error, handle *procHandle, lines <-chan string, chatID string, created int64, model, sessionID string) bool {
	for {
		select {
		case err := <-ready:
			if err != nil {
				log.Printf("claude startup failed: chat=%s session=%s: %v", chatID, sessionID, err)
				channelEmitErrClose(w, flusher, chatID, created, model, err.Error())
				proc.DrainLines(lines)
				return false
			}
			return true
		case <-heartbeatTicker.C:
			fmt.Fprintf(w, ": keepalive\n\n")
			flusher.Flush()
		case <-r.Context().Done():
			// Client gone while claude was still starting: same treatment as a
			// mid-stream disconnect.
			if cmd := handle.abort(); cmd != nil {
				abortClaudeAndHarvestUsage(cmd, lines, model, sessionID)
			} else {
				proc.DrainLines(lines)
			}
			return false
		}
	}
}

// handleStreamResponse streams Claude output without tool-call detection.
// Used when no tools are defined in the request (fast path). The stream-json →
// OpenAI SSE translation is delegated to sseTranslator (shared with the
// channel-mode handler); this function owns only the per-request process
// lifecycle, heartbeats and disconnect handling.
func handleStreamResponse(w http.ResponseWriter, r *http.Request, args []string, prompt, chatID string, created int64, model string, includeUsage bool, workingDir string, envVars map[string]string, sessionID string) {
	lines, ready, handle := startClaudeStream(args, prompt, workingDir, envVars)

	flusher, heartbeatTicker, ok := sseHandshake(w, ready, handle, lines)
	if !ok {
		return
	}
	defer heartbeatTicker.Stop()

	// Disconnect handling is inline in the loops below (single goroutine
	// watching ctx), so there is no watcher/loop race on ctx.

	if !waitClaudeReady(w, r, flusher, heartbeatTicker, ready, handle, lines, chatID, created, model, sessionID) {
		return
	}

	t := newSSETranslator(chatID, created, model, sessionID, identityMeter{}, "")

processLines:
	for {
		var line string
		var ok bool
		select {
		case <-r.Context().Done():
			// disconnect (= upstream stop): SIGINT + harvest the aborted turn's
			// usage before the backstop kill (A3).
			if cmd := handle.abort(); cmd != nil {
				abortClaudeAndHarvestUsage(cmd, lines, model, sessionID)
			} else {
				proc.DrainLines(lines)
			}
			return
		case <-heartbeatTicker.C:
			fmt.Fprintf(w, ": keepalive\n\n")
			flusher.Flush()
			continue
		case line, ok = <-lines:
			if !ok {
				break processLines
			}
		}
		if line == "" {
			continue
		}

		if t.feed(w, flusher, line, includeUsage) == outcomeAskUserDone {
			if cmd := handle.abort(); cmd != nil && cmd.Process != nil {
				log.Printf("[ASK_USER_QUESTION] interrupting Claude process group pid=%d (usage harvest, then kill)", cmd.Process.Pid)
				abortClaudeAndHarvestUsage(cmd, lines, model, sessionID)
			} else {
				proc.DrainLines(lines)
			}
			return
		}
	}

	t.flushAggLog()
	// Process exited without a result event (crash / scanner truncation): give
	// the client a terminal finish chunk before [DONE] (A2).
	t.EmitFinishIfNoResult(w, flusher)
	sessionStore.LogDone(sessionID, t.StreamUsage())

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// handleBufferedStreamResponse streams text deltas in real-time while also
// collecting full output text to detect tool calls at the end. Used when
// tools are defined in the request.
func handleBufferedStreamResponse(w http.ResponseWriter, r *http.Request, args []string, prompt, chatID string, created int64, model string, includeUsage bool, workingDir string, envVars map[string]string, sessionID string) {
	lines, ready, handle := startClaudeStream(args, prompt, workingDir, envVars)

	flusher, heartbeatTicker, ok := sseHandshake(w, ready, handle, lines)
	if !ok {
		return
	}
	defer heartbeatTicker.Stop()

	// Disconnect handling is inline in the loops below (single goroutine
	// watching ctx), so there is no watcher/loop race on ctx.

	if !waitClaudeReady(w, r, flusher, heartbeatTicker, ready, handle, lines, chatID, created, model, sessionID) {
		return
	}

	var fullText strings.Builder
	var filter toolCallFilter
	var streamDeltaSent bool
	var finalUsage *openai.UsageInfo

	nativeTCs := map[int]*nativeToolCall{}
	var nativeTCOrder []int
	var askUserIdx int = -1

processLines:
	for {
		var line string
		var ok bool
		select {
		case <-r.Context().Done():
			// disconnect (= upstream stop): SIGINT + harvest the aborted turn's
			// usage before the backstop kill (A3).
			if cmd := handle.abort(); cmd != nil {
				abortClaudeAndHarvestUsage(cmd, lines, model, sessionID)
			} else {
				proc.DrainLines(lines)
			}
			return
		case <-heartbeatTicker.C:
			fmt.Fprintf(w, ": keepalive\n\n")
			flusher.Flush()
			continue
		case line, ok = <-lines:
			if !ok {
				break processLines
			}
		}
		if line == "" {
			continue
		}

		log.Printf("[BUFFERED STREAM RAW] %s", line)

		var event claudeEvent
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			log.Printf("Failed to parse claude event: %v", err)
			continue
		}

		if event.Type == "stream_event" && event.Event != nil {
			var streamEvt streamAPIEvent
			if err := json.Unmarshal(event.Event, &streamEvt); err != nil {
				continue
			}

			switch streamEvt.Type {
			case "content_block_start":
				if streamEvt.ContentBlock != nil {
					var block streamContentBlock
					if err := json.Unmarshal(streamEvt.ContentBlock, &block); err == nil && block.Type == "tool_use" {
						tc := &nativeToolCall{ID: block.ID, Name: block.Name}
						nativeTCs[streamEvt.Index] = tc
						nativeTCOrder = append(nativeTCOrder, streamEvt.Index)
						log.Printf("[NATIVE TOOL_USE START] index=%d name=%s id=%s", streamEvt.Index, block.Name, block.ID)
						if block.Name == "AskUserQuestion" {
							askUserIdx = streamEvt.Index
							log.Printf("[ASK_USER_QUESTION] detected at index=%d", streamEvt.Index)
						}
					}
				}

			case "content_block_stop":
				if tc, ok := nativeTCs[streamEvt.Index]; ok {
					sessionStore.LogToolUse(sessionID, tc.Name, tc.ID, tc.Args.String())
				}
				if askUserIdx >= 0 && streamEvt.Index == askUserIdx {
					tc := nativeTCs[askUserIdx]
					log.Printf("[ASK_USER_QUESTION] complete, emitting tool_call and closing stream")

					toolCall := openai.ToolCall{
						ID:   tc.ID,
						Type: "function",
						Function: openai.ToolCallFunction{
							Name:      "AskUserQuestion",
							Arguments: tc.Args.String(),
						},
					}
					chunk := openai.ChatCompletionResponse{
						ID:      chatID,
						Object:  "chat.completion.chunk",
						Created: created,
						Model:   model,
						Choices: []openai.ChatCompletionChoice{{
							Index: 0,
							Delta: &openai.ChatMessage{Role: "assistant", ToolCalls: []openai.ToolCall{toolCall}},
						}},
					}
					data, _ := json.Marshal(chunk)
					fmt.Fprintf(w, "data: %s\n\n", data)
					flusher.Flush()

					finishReason := "stop"
					finishChunk := openai.ChatCompletionResponse{
						ID:      chatID,
						Object:  "chat.completion.chunk",
						Created: created,
						Model:   model,
						Choices: []openai.ChatCompletionChoice{
							{Index: 0, Delta: openai.NewChatMessage("", ""), FinishReason: &finishReason},
						},
					}
					data, _ = json.Marshal(finishChunk)
					fmt.Fprintf(w, "data: %s\n\n", data)
					flusher.Flush()

					fmt.Fprintf(w, "data: [DONE]\n\n")
					flusher.Flush()

					if cmd := handle.abort(); cmd != nil && cmd.Process != nil {
						log.Printf("[ASK_USER_QUESTION] interrupting Claude process group pid=%d (usage harvest, then kill)", cmd.Process.Pid)
						abortClaudeAndHarvestUsage(cmd, lines, model, sessionID)
					} else {
						proc.DrainLines(lines)
					}
					return
				}

			case "content_block_delta":
				if streamEvt.Delta != nil {
					var delta streamTextDelta
					if err := json.Unmarshal(streamEvt.Delta, &delta); err != nil {
						continue
					}
					switch delta.Type {
					case "text_delta":
						if delta.Text != "" {
							streamDeltaSent = true
							fullText.WriteString(delta.Text)
							sessionStore.LogDelta(sessionID, delta.Text)
							safeText := filter.Feed(delta.Text)
							if safeText != "" {
								log.Printf("[BUFFERED STREAM DELTA] len=%d content=%q", len(safeText), openai.Truncate(safeText, 200))
								chunk := openai.ChatCompletionResponse{
									ID:      chatID,
									Object:  "chat.completion.chunk",
									Created: created,
									Model:   model,
									Choices: []openai.ChatCompletionChoice{{
										Index: 0,
										Delta: openai.NewChatMessage("assistant", safeText),
									}},
								}
								data, _ := json.Marshal(chunk)
								fmt.Fprintf(w, "data: %s\n\n", data)
								flusher.Flush()
							}
						}
					case "input_json_delta":
						if tc, ok := nativeTCs[streamEvt.Index]; ok {
							tc.Args.WriteString(delta.PartialJSON)
						}
					}
				}
			}
			continue
		}

		if !streamDeltaSent && event.Type == "assistant" {
			text := extractTextFromEvent(&event)
			if text != "" {
				fullText.WriteString(text)
				safeText := filter.Feed(text)
				if safeText != "" {
					log.Printf("[BUFFERED STREAM FALLBACK] len=%d content=%q", len(safeText), openai.Truncate(safeText, 200))
					chunk := openai.ChatCompletionResponse{
						ID:      chatID,
						Object:  "chat.completion.chunk",
						Created: created,
						Model:   model,
						Choices: []openai.ChatCompletionChoice{{
							Index: 0,
							Delta: openai.NewChatMessage("assistant", safeText),
						}},
					}
					data, _ := json.Marshal(chunk)
					fmt.Fprintf(w, "data: %s\n\n", data)
					flusher.Flush()
				}
			}
		}

		if event.Type == "result" {
			// Mirror translate.go's error transparency: an error result whose
			// explanation only lives in Result (nothing streamed yet) must be
			// surfaced as a content delta, or the turn ends cleanly but empty —
			// this path is hit by resume-sniff-forwarded error results too.
			if (event.IsError || strings.HasPrefix(event.Subtype, "error")) && !streamDeltaSent && event.Result != "" {
				log.Printf("result reports error (buffered path): subtype=%s is_error=%v chat=%s session=%s",
					event.Subtype, event.IsError, chatID, sessionID)
				chunk := openai.ChatCompletionResponse{
					ID: chatID, Object: "chat.completion.chunk", Created: created, Model: model,
					Choices: []openai.ChatCompletionChoice{{
						Index: 0,
						Delta: openai.NewChatMessage("assistant", "⚠️ "+event.Result),
					}},
					XRelayError: true,
				}
				data, _ := json.Marshal(chunk)
				fmt.Fprintf(w, "data: %s\n\n", data)
				flusher.Flush()
			}
			if event.Result != "" {
				fullText.Reset()
				fullText.WriteString(event.Result)
			}
			if eu := effectiveUsage(&event); eu != nil {
				finalUsage = openai.BuildUsageInfo(eu.InputTokens, eu.OutputTokens, eu.CacheReadInputTokens, eu.CacheCreationInputTokens)
				// V1 每请求进程：按实际消费模型归属（A4）；下游 usage chunk 仍用聚合值。
				stats.RecordTurn(model, perModelCounts(&event, model))
				log.Printf("Token usage: model=%s input=%d output=%d cache_read=%d cache_create=%d cost=$%.4f (per-model attribution, models=%d)",
					model, eu.InputTokens, eu.OutputTokens,
					eu.CacheReadInputTokens, eu.CacheCreationInputTokens, event.TotalCostUSD,
					len(event.ModelUsage))
			}
		}
	}

	if remaining, _ := filter.Finish(); remaining != "" && !strings.HasPrefix(remaining, "<tool_call") {
		chunk := openai.ChatCompletionResponse{
			ID:      chatID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: []openai.ChatCompletionChoice{{
				Index: 0,
				Delta: openai.NewChatMessage("assistant", remaining),
			}},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	var toolCalls []openai.ToolCall
	for _, idx := range nativeTCOrder {
		tc := nativeTCs[idx]
		toolCalls = append(toolCalls, openai.ToolCall{
			ID:   tc.ID,
			Type: "function",
			Function: openai.ToolCallFunction{
				Name:      tc.Name,
				Arguments: tc.Args.String(),
			},
		})
	}
	if len(toolCalls) > 0 {
		log.Printf("Detected %d native tool_use calls", len(toolCalls))
	} else {
		collected := fullText.String()
		_, toolCalls = parseToolCalls(collected)
		if len(toolCalls) > 0 {
			log.Printf("Detected %d XML tool calls (fallback)", len(toolCalls))
		}
	}

	if len(toolCalls) > 0 {
		for _, tc := range toolCalls {
			chunk := openai.ChatCompletionResponse{
				ID:      chatID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []openai.ChatCompletionChoice{{
					Index: 0,
					Delta: &openai.ChatMessage{Role: "assistant", ToolCalls: []openai.ToolCall{tc}},
				}},
			}
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}

		finishReason := "tool_calls"
		finishChunk := openai.ChatCompletionResponse{
			ID:      chatID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: []openai.ChatCompletionChoice{
				{Index: 0, Delta: openai.NewChatMessage("", ""), FinishReason: &finishReason},
			},
		}
		data, _ := json.Marshal(finishChunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	} else {
		finishReason := "stop"
		finishChunk := openai.ChatCompletionResponse{
			ID:      chatID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: []openai.ChatCompletionChoice{
				{Index: 0, Delta: openai.NewChatMessage("", ""), FinishReason: &finishReason},
			},
		}
		data, _ := json.Marshal(finishChunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	if includeUsage && finalUsage != nil {
		usageChunk := openai.ChatCompletionResponse{
			ID:      chatID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: []openai.ChatCompletionChoice{},
			Usage:   finalUsage,
		}
		data, _ := json.Marshal(usageChunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	sessionStore.LogDone(sessionID, finalUsage)

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// handleNonStreamResponse runs Claude to completion and returns one JSON
// response (as opposed to SSE chunks).
func handleNonStreamResponse(w http.ResponseWriter, r *http.Request, args []string, prompt, chatID string, created int64, model string, hasTools bool, workingDir string, envVars map[string]string, sessionID string) {
	events, lastText, result, usage, rawUsage, costUSD, err := runClaude(args, prompt, workingDir, envVars)
	if err != nil {
		openai.WriteError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	fullText := lastText
	if result != "" {
		fullText = result
	}

	if rawUsage != nil {
		// V1 每请求进程：从收集到的事件里找 result，按实际消费模型归属（A4）。
		var resultEv *claudeEvent
		for i := range events {
			if events[i].Type == "result" {
				resultEv = &events[i]
			}
		}
		stats.RecordTurn(model, perModelCounts(resultEv, model))
		log.Printf("Token usage: model=%s input=%d output=%d cache_read=%d cache_create=%d cost=$%.4f",
			model, rawUsage.InputTokens, rawUsage.OutputTokens,
			rawUsage.CacheReadInputTokens, rawUsage.CacheCreationInputTokens, costUSD)
	}

	var (
		finishReason string
		msg          *openai.ChatMessage
	)

	askUserCalls := extractAskUserQuestionCalls(events)
	if len(askUserCalls) > 0 {
		finishReason = "stop"
		msg = openai.NewChatMessage("assistant", fullText)
		msg.ToolCalls = askUserCalls
		log.Printf("Non-stream: detected AskUserQuestion, returning as tool_call with finish_reason=stop")
	} else if hasTools {
		toolCalls := extractNativeToolCalls(events)
		if len(toolCalls) > 0 {
			finishReason = "tool_calls"
			msg = openai.NewChatMessage("assistant", fullText)
			msg.ToolCalls = toolCalls
			log.Printf("Non-stream: detected %d native tool_use calls", len(toolCalls))
		} else {
			cleanText, xmlCalls := parseToolCalls(fullText)
			if len(xmlCalls) > 0 {
				finishReason = "tool_calls"
				msg = openai.NewChatMessage("assistant", cleanText)
				msg.ToolCalls = xmlCalls
				log.Printf("Non-stream: detected %d XML tool calls (fallback)", len(xmlCalls))
			} else {
				finishReason = "stop"
				msg = openai.NewChatMessage("assistant", fullText)
			}
		}
	} else {
		finishReason = "stop"
		msg = openai.NewChatMessage("assistant", fullText)
	}

	if msg == nil || (msg.ContentString() == "" && len(msg.ToolCalls) == 0) {
		openai.WriteError(w, http.StatusInternalServerError, "server_error", "Empty response from Claude")
		return
	}

	resp := openai.ChatCompletionResponse{
		ID:      chatID,
		Object:  "chat.completion",
		Created: created,
		Model:   model,
		Choices: []openai.ChatCompletionChoice{{
			Index:        0,
			Message:      msg,
			FinishReason: &finishReason,
		}},
		Usage: usage,
	}

	sessionStore.LogDelta(sessionID, fullText)
	sessionStore.LogDone(sessionID, usage)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
