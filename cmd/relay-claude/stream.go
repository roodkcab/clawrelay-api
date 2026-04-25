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

// handleStreamResponse streams Claude output without tool-call detection.
// Used when no tools are defined in the request (fast path).
func handleStreamResponse(w http.ResponseWriter, r *http.Request, args []string, prompt, chatID string, created int64, model string, includeUsage bool, workingDir string, envVars map[string]string, sessionID string) {
	lines, ready, cmdPtr := startClaudeStream(args, prompt, workingDir, envVars)
	if err := <-ready; err != nil {
		openai.WriteError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	go func() {
		<-r.Context().Done()
		if cmd := *cmdPtr; cmd != nil && cmd.Process != nil {
			log.Printf("Client disconnected, killing Claude process pid=%d", cmd.Process.Pid)
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
		log.Printf("Streaming not supported")
		return
	}

	fmt.Fprintf(w, ": ping\n\n")
	flusher.Flush()

	heartbeatTicker := time.NewTicker(30 * time.Second)
	defer heartbeatTicker.Stop()

	var streamDeltaSent bool
	var streamUsage *openai.UsageInfo
	seenToolNames := map[string]bool{}

	var askUserIdx int = -1
	var askUserID string
	var askUserArgs strings.Builder

	var aggType string
	var aggBuf strings.Builder
	var aggCount int

	flushAggLog := func() {
		if aggCount == 0 {
			return
		}
		preview := aggBuf.String()
		if len(preview) > 500 {
			preview = preview[:500] + "..."
		}
		log.Printf("[STREAM %s] chunks=%d len=%d content=%q", strings.ToUpper(aggType), aggCount, aggBuf.Len(), preview)
		aggType = ""
		aggBuf.Reset()
		aggCount = 0
	}

	aggAppend := func(evtType string, text string) {
		if evtType != aggType {
			flushAggLog()
			aggType = evtType
		}
		aggBuf.WriteString(text)
		aggCount++
	}

	type toolBlock struct {
		Name string
		ID   string
		Args strings.Builder
	}
	toolBlocks := map[int]*toolBlock{}

processLines:
	for {
		var line string
		var ok bool
		select {
		case <-r.Context().Done():
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
			if streamEvt.Type == "content_block_start" && streamEvt.ContentBlock != nil {
				var block streamContentBlock
				if err := json.Unmarshal(streamEvt.ContentBlock, &block); err == nil && block.Type == "tool_use" && block.Name != "" {
					flushAggLog()
					log.Printf("[STREAM TOOL_USE] name=%s id=%s", block.Name, block.ID)
					toolBlocks[streamEvt.Index] = &toolBlock{Name: block.Name, ID: block.ID}
					if block.Name == "AskUserQuestion" {
						askUserIdx = streamEvt.Index
						askUserID = block.ID
						log.Printf("[ASK_USER_QUESTION] detected at index=%d", streamEvt.Index)
						continue
					}
					seenToolNames[block.Name] = true
					tc := openai.ToolCall{
						ID:   block.ID,
						Type: "function",
						Function: openai.ToolCallFunction{Name: block.Name, Arguments: ""},
					}
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
			}
			if streamEvt.Type == "content_block_delta" && streamEvt.Delta != nil {
				var delta streamTextDelta
				if err := json.Unmarshal(streamEvt.Delta, &delta); err != nil {
					continue
				}
				if delta.Type == "text_delta" && delta.Text != "" {
					streamDeltaSent = true
					aggAppend("text_delta", delta.Text)
					sessionStore.LogDelta(sessionID, delta.Text)
					chunk := openai.ChatCompletionResponse{
						ID:      chatID,
						Object:  "chat.completion.chunk",
						Created: created,
						Model:   model,
						Choices: []openai.ChatCompletionChoice{{
							Index: 0,
							Delta: openai.NewChatMessage("assistant", delta.Text),
						}},
					}
					data, _ := json.Marshal(chunk)
					fmt.Fprintf(w, "data: %s\n\n", data)
					flusher.Flush()
				}
				if delta.Type == "thinking_delta" && delta.Thinking != "" {
					aggAppend("thinking_delta", delta.Thinking)
					chunk := openai.ChatCompletionResponse{
						ID:      chatID,
						Object:  "chat.completion.chunk",
						Created: created,
						Model:   model,
						Choices: []openai.ChatCompletionChoice{{
							Index: 0,
							Delta: &openai.ChatMessage{Role: "assistant", Thinking: delta.Thinking},
						}},
					}
					data, _ := json.Marshal(chunk)
					fmt.Fprintf(w, "data: %s\n\n", data)
					flusher.Flush()
				}
				if delta.Type == "input_json_delta" {
					if tb, ok := toolBlocks[streamEvt.Index]; ok {
						tb.Args.WriteString(delta.PartialJSON)
					}
					if askUserIdx >= 0 && streamEvt.Index == askUserIdx {
						askUserArgs.WriteString(delta.PartialJSON)
					}
				}
			}
			if streamEvt.Type == "content_block_stop" {
				if tb, ok := toolBlocks[streamEvt.Index]; ok {
					sessionStore.LogToolUse(sessionID, tb.Name, tb.ID, tb.Args.String())
					delete(toolBlocks, streamEvt.Index)
				}
			}
			if streamEvt.Type == "content_block_stop" && askUserIdx >= 0 && streamEvt.Index == askUserIdx {
				flushAggLog()
				log.Printf("[ASK_USER_QUESTION] complete, emitting tool_call and closing stream")

				toolCall := openai.ToolCall{
					ID:   askUserID,
					Type: "function",
					Function: openai.ToolCallFunction{
						Name:      "AskUserQuestion",
						Arguments: askUserArgs.String(),
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

				if cmd := *cmdPtr; cmd != nil && cmd.Process != nil {
					log.Printf("[ASK_USER_QUESTION] killing Claude process pid=%d", cmd.Process.Pid)
					cmd.Process.Kill()
				}
				go func() {
					for range lines {
					}
				}()
				return
			}
			continue
		}

		flushAggLog()

		// Fallback: pull tool_use names from a non-streamed assistant event.
		if event.Type == "assistant" && event.Message != nil {
			var msg claudeMessage
			if err := json.Unmarshal(event.Message, &msg); err == nil {
				for _, c := range msg.Content {
					if c.Type == "tool_use" && c.Name != "" && !seenToolNames[c.Name] {
						seenToolNames[c.Name] = true
						log.Printf("[ASSISTANT TOOL_USE FALLBACK] name=%s", c.Name)
						tc := openai.ToolCall{
							ID:   c.Name,
							Type: "function",
							Function: openai.ToolCallFunction{Name: c.Name, Arguments: ""},
						}
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
				}
			}
		}

		// Fallback: pull text from a non-streamed assistant event.
		if !streamDeltaSent {
			text := extractTextFromEvent(&event)
			if text != "" {
				log.Printf("[STREAM FALLBACK DELTA] len=%d content=%q", len(text), openai.Truncate(text, 200))
				chunk := openai.ChatCompletionResponse{
					ID:      chatID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   model,
					Choices: []openai.ChatCompletionChoice{{
						Index: 0,
						Delta: openai.NewChatMessage("assistant", text),
					}},
				}
				data, _ := json.Marshal(chunk)
				fmt.Fprintf(w, "data: %s\n\n", data)
				flusher.Flush()
			}
		}

		if event.Type == "result" {
			if eu := effectiveUsage(&event); eu != nil {
				stats.Record(model, eu.InputTokens, eu.OutputTokens,
					eu.CacheCreationInputTokens, eu.CacheReadInputTokens, event.TotalCostUSD)
				log.Printf("Token usage: model=%s input=%d output=%d cache_read=%d cache_create=%d cost=$%.4f (subagents_included=%v)",
					model, eu.InputTokens, eu.OutputTokens,
					eu.CacheReadInputTokens, eu.CacheCreationInputTokens, event.TotalCostUSD,
					len(event.ModelUsage) > 0)
				streamUsage = openai.BuildUsageInfo(eu.InputTokens, eu.OutputTokens, eu.CacheReadInputTokens, eu.CacheCreationInputTokens)
			}

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

			if includeUsage && event.Usage != nil {
				usageChunk := openai.ChatCompletionResponse{
					ID:      chatID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   model,
					Choices: []openai.ChatCompletionChoice{},
					Usage:   streamUsage,
				}
				data, _ = json.Marshal(usageChunk)
				fmt.Fprintf(w, "data: %s\n\n", data)
				flusher.Flush()
			}
		}
	}

	flushAggLog()
	sessionStore.LogDone(sessionID, streamUsage)

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// handleBufferedStreamResponse streams text deltas in real-time while also
// collecting full output text to detect tool calls at the end. Used when
// tools are defined in the request.
func handleBufferedStreamResponse(w http.ResponseWriter, r *http.Request, args []string, prompt, chatID string, created int64, model string, includeUsage bool, workingDir string, envVars map[string]string, sessionID string) {
	lines, ready, cmdPtr := startClaudeStream(args, prompt, workingDir, envVars)
	if err := <-ready; err != nil {
		openai.WriteError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	go func() {
		<-r.Context().Done()
		if cmd := *cmdPtr; cmd != nil && cmd.Process != nil {
			log.Printf("Client disconnected, killing Claude process pid=%d", cmd.Process.Pid)
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
		log.Printf("Streaming not supported")
		return
	}

	fmt.Fprintf(w, ": ping\n\n")
	flusher.Flush()

	heartbeatTicker := time.NewTicker(30 * time.Second)
	defer heartbeatTicker.Stop()

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

					if cmd := *cmdPtr; cmd != nil && cmd.Process != nil {
						log.Printf("[ASK_USER_QUESTION] killing Claude process pid=%d", cmd.Process.Pid)
						cmd.Process.Kill()
					}
					go func() {
						for range lines {
						}
					}()
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
			if event.Result != "" {
				fullText.Reset()
				fullText.WriteString(event.Result)
			}
			if eu := effectiveUsage(&event); eu != nil {
				finalUsage = openai.BuildUsageInfo(eu.InputTokens, eu.OutputTokens, eu.CacheReadInputTokens, eu.CacheCreationInputTokens)
				stats.Record(model, eu.InputTokens, eu.OutputTokens,
					eu.CacheCreationInputTokens, eu.CacheReadInputTokens, event.TotalCostUSD)
				log.Printf("Token usage: model=%s input=%d output=%d cache_read=%d cache_create=%d cost=$%.4f (subagents_included=%v)",
					model, eu.InputTokens, eu.OutputTokens,
					eu.CacheReadInputTokens, eu.CacheCreationInputTokens, event.TotalCostUSD,
					len(event.ModelUsage) > 0)
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
		stats.Record(model, rawUsage.InputTokens, rawUsage.OutputTokens,
			rawUsage.CacheCreationInputTokens, rawUsage.CacheReadInputTokens, costUSD)
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
