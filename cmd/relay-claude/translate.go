package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"

	"clawrelay-api/pkg/openai"
)

// toolBlock accumulates one streaming tool_use block's arguments while it
// streams in (fast-path translation, no client-side tool filtering).
type toolBlock struct {
	Name string
	ID   string
	Args strings.Builder
}

// lineOutcome tells the driving loop how to proceed after a translated line.
type lineOutcome int

const (
	// outcomeContinue: nothing special; keep reading the stream.
	outcomeContinue lineOutcome = iota
	// outcomeAskUserDone: an AskUserQuestion tool call has completed. The
	// translator has already emitted the tool_call chunk, the finish chunk and
	// `data: [DONE]`. The caller must terminate the current generation —
	// KillGroup for the legacy per-request process, or an interrupt for the
	// channel-mode persistent process (which is kept alive for the answer).
	outcomeAskUserDone
)

// sseTranslator converts a `claude --output-format stream-json` event stream
// into OpenAI-compatible SSE chunks. It holds the per-response state that
// handleStreamResponse used to keep as loop-local variables, so the identical
// translation can be reused by both the legacy per-request path and the
// channel (persistent stream-json process) path.
//
// The emitted byte stream is identical to the original inline loop; the only
// behavioral change is that terminal control (kill vs interrupt) is delegated
// to the caller via the returned lineOutcome.
type sseTranslator struct {
	chatID    string
	created   int64
	model     string
	sessionID string
	meter     usageMeter
	// meterModel is the model that stats accounting is attributed to. For a
	// persistent channel worker this is the worker's spawn-time boundModel: a
	// hot model switch changes the request `model` immediately, but the running
	// process keeps consuming under the old model until respawn, so billing to
	// the request model would misattribute. Empty means "same as model".
	meterModel string

	streamDeltaSent bool
	streamUsage     *openai.UsageInfo
	sawResult       bool // a result event reached feed (finish chunk emitted)
	seenToolNames   map[string]bool

	askUserIdx  int
	askUserID   string
	askUserArgs strings.Builder

	toolBlocks map[int]*toolBlock

	aggType  string
	aggBuf   strings.Builder
	aggCount int
}

// newSSETranslator: meterModel is the stats-attribution model (see the field
// comment); pass "" for V1/ephemeral paths, where it defaults to model.
func newSSETranslator(chatID string, created int64, model, sessionID string, meter usageMeter, meterModel string) *sseTranslator {
	if meter == nil {
		meter = identityMeter{}
	}
	if meterModel == "" {
		meterModel = model
	}
	return &sseTranslator{
		chatID:        chatID,
		created:       created,
		model:         model,
		sessionID:     sessionID,
		meter:         meter,
		meterModel:    meterModel,
		seenToolNames: map[string]bool{},
		askUserIdx:    -1,
		toolBlocks:    map[int]*toolBlock{},
	}
}

// StreamUsage returns the usage captured from the turn's result event, if any.
func (t *sseTranslator) StreamUsage() *openai.UsageInfo { return t.streamUsage }

// EmitFinishIfNoResult emits a synthetic finish_reason="stop" chunk when the
// stream is ending without ever having seen a result event (process crashed
// mid-turn, stdout truncated by a scanner error, worker died) — otherwise the
// downstream OpenAI client gets [DONE] with no terminal chunk and may treat
// the response as aborted. No-op after a normal turn: the result branch of
// feed already emitted the finish chunk (sawResult=true).
func (t *sseTranslator) EmitFinishIfNoResult(w http.ResponseWriter, flusher http.Flusher) {
	if t.sawResult {
		return
	}
	log.Printf("stream ended without result event: chat=%s session=%s model=%s; emitting synthetic finish chunk",
		t.chatID, t.sessionID, t.model)
	finishReason := "stop"
	t.emit(w, flusher, openai.ChatCompletionResponse{
		ID: t.chatID, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
		Choices: []openai.ChatCompletionChoice{
			{Index: 0, Delta: openai.NewChatMessage("", ""), FinishReason: &finishReason},
		},
	})
}

func (t *sseTranslator) flushAggLog() {
	if t.aggCount == 0 {
		return
	}
	preview := t.aggBuf.String()
	if len(preview) > 500 {
		preview = preview[:500] + "..."
	}
	log.Printf("[STREAM %s] chunks=%d len=%d content=%q", strings.ToUpper(t.aggType), t.aggCount, t.aggBuf.Len(), preview)
	t.aggType = ""
	t.aggBuf.Reset()
	t.aggCount = 0
}

func (t *sseTranslator) aggAppend(evtType, text string) {
	if evtType != t.aggType {
		t.flushAggLog()
		t.aggType = evtType
	}
	t.aggBuf.WriteString(text)
	t.aggCount++
}

// emit marshals one chunk and writes it as an SSE `data:` frame.
func (t *sseTranslator) emit(w http.ResponseWriter, flusher http.Flusher, chunk openai.ChatCompletionResponse) {
	data, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

// feed translates a single stream-json line into SSE chunks. The caller owns
// the read loop (heartbeats, disconnect handling, termination). includeUsage
// mirrors the request's stream_options.include_usage.
func (t *sseTranslator) feed(w http.ResponseWriter, flusher http.Flusher, line string, includeUsage bool) lineOutcome {
	var event claudeEvent
	if err := json.Unmarshal([]byte(line), &event); err != nil {
		log.Printf("Failed to parse claude event: %v", err)
		return outcomeContinue
	}

	if event.Type == "stream_event" && event.Event != nil {
		var streamEvt streamAPIEvent
		if err := json.Unmarshal(event.Event, &streamEvt); err != nil {
			return outcomeContinue
		}
		if streamEvt.Type == "content_block_start" && streamEvt.ContentBlock != nil {
			var block streamContentBlock
			if err := json.Unmarshal(streamEvt.ContentBlock, &block); err == nil && block.Type == "tool_use" && block.Name != "" {
				t.flushAggLog()
				log.Printf("[STREAM TOOL_USE] name=%s id=%s", block.Name, block.ID)
				t.toolBlocks[streamEvt.Index] = &toolBlock{Name: block.Name, ID: block.ID}
				if block.Name == "AskUserQuestion" {
					t.askUserIdx = streamEvt.Index
					t.askUserID = block.ID
					log.Printf("[ASK_USER_QUESTION] detected at index=%d", streamEvt.Index)
					return outcomeContinue
				}
				t.seenToolNames[block.Name] = true
				tc := openai.ToolCall{
					ID:       block.ID,
					Type:     "function",
					Function: openai.ToolCallFunction{Name: block.Name, Arguments: ""},
				}
				t.emit(w, flusher, openai.ChatCompletionResponse{
					ID: t.chatID, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
					Choices: []openai.ChatCompletionChoice{{
						Index: 0,
						Delta: &openai.ChatMessage{Role: "assistant", ToolCalls: []openai.ToolCall{tc}},
					}},
				})
			}
		}
		if streamEvt.Type == "content_block_delta" && streamEvt.Delta != nil {
			var delta streamTextDelta
			if err := json.Unmarshal(streamEvt.Delta, &delta); err != nil {
				return outcomeContinue
			}
			if delta.Type == "text_delta" && delta.Text != "" {
				t.streamDeltaSent = true
				t.aggAppend("text_delta", delta.Text)
				sessionStore.LogDelta(t.sessionID, delta.Text)
				t.emit(w, flusher, openai.ChatCompletionResponse{
					ID: t.chatID, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
					Choices: []openai.ChatCompletionChoice{{
						Index: 0,
						Delta: openai.NewChatMessage("assistant", delta.Text),
					}},
				})
			}
			if delta.Type == "thinking_delta" && delta.Thinking != "" {
				t.aggAppend("thinking_delta", delta.Thinking)
				t.emit(w, flusher, openai.ChatCompletionResponse{
					ID: t.chatID, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
					Choices: []openai.ChatCompletionChoice{{
						Index: 0,
						Delta: &openai.ChatMessage{Role: "assistant", Thinking: delta.Thinking},
					}},
				})
			}
			if delta.Type == "input_json_delta" {
				if tb, ok := t.toolBlocks[streamEvt.Index]; ok {
					tb.Args.WriteString(delta.PartialJSON)
				}
				if t.askUserIdx >= 0 && streamEvt.Index == t.askUserIdx {
					t.askUserArgs.WriteString(delta.PartialJSON)
				}
			}
		}
		if streamEvt.Type == "content_block_stop" {
			if tb, ok := t.toolBlocks[streamEvt.Index]; ok {
				sessionStore.LogToolUse(t.sessionID, tb.Name, tb.ID, tb.Args.String())
				delete(t.toolBlocks, streamEvt.Index)
			}
		}
		if streamEvt.Type == "content_block_stop" && t.askUserIdx >= 0 && streamEvt.Index == t.askUserIdx {
			t.flushAggLog()
			log.Printf("[ASK_USER_QUESTION] complete, emitting tool_call and closing stream")

			toolCall := openai.ToolCall{
				ID:   t.askUserID,
				Type: "function",
				Function: openai.ToolCallFunction{
					Name:      "AskUserQuestion",
					Arguments: t.askUserArgs.String(),
				},
			}
			t.emit(w, flusher, openai.ChatCompletionResponse{
				ID: t.chatID, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
				Choices: []openai.ChatCompletionChoice{{
					Index: 0,
					Delta: &openai.ChatMessage{Role: "assistant", ToolCalls: []openai.ToolCall{toolCall}},
				}},
			})

			finishReason := "stop"
			t.emit(w, flusher, openai.ChatCompletionResponse{
				ID: t.chatID, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
				Choices: []openai.ChatCompletionChoice{
					{Index: 0, Delta: openai.NewChatMessage("", ""), FinishReason: &finishReason},
				},
			})

			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return outcomeAskUserDone
		}
		return outcomeContinue
	}

	t.flushAggLog()

	// Fallback: pull tool_use names from a non-streamed assistant event.
	if event.Type == "assistant" && event.Message != nil {
		var msg claudeMessage
		if err := json.Unmarshal(event.Message, &msg); err == nil {
			for _, c := range msg.Content {
				if c.Type == "tool_use" && c.Name != "" && !t.seenToolNames[c.Name] {
					t.seenToolNames[c.Name] = true
					log.Printf("[ASSISTANT TOOL_USE FALLBACK] name=%s", c.Name)
					tc := openai.ToolCall{
						ID:       c.Name,
						Type:     "function",
						Function: openai.ToolCallFunction{Name: c.Name, Arguments: ""},
					}
					t.emit(w, flusher, openai.ChatCompletionResponse{
						ID: t.chatID, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
						Choices: []openai.ChatCompletionChoice{{
							Index: 0,
							Delta: &openai.ChatMessage{Role: "assistant", ToolCalls: []openai.ToolCall{tc}},
						}},
					})
				}
			}
		}
	}

	// Fallback: pull text from a non-streamed assistant event.
	if !t.streamDeltaSent {
		text := extractTextFromEvent(&event)
		if text != "" {
			log.Printf("[STREAM FALLBACK DELTA] len=%d content=%q", len(text), openai.Truncate(text, 200))
			t.emit(w, flusher, openai.ChatCompletionResponse{
				ID: t.chatID, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
				Choices: []openai.ChatCompletionChoice{{
					Index: 0,
					Delta: openai.NewChatMessage("assistant", text),
				}},
			})
		}
	}

	if event.Type == "result" {
		t.sawResult = true

		// Error results (is_error / subtype=error_*) used to be
		// indistinguishable from success downstream: extractTextFromEvent only
		// reads assistant events, so the explanation claude puts in Result was
		// silently dropped and the client saw an empty, cleanly-finished turn.
		// Surface it as a content delta when nothing has been streamed yet.
		// finish_reason stays "stop" — upstream (wuji) only accepts the
		// standard OpenAI values.
		if event.IsError || strings.HasPrefix(event.Subtype, "error") {
			log.Printf("result reports error: subtype=%s is_error=%v chat=%s session=%s model=%s",
				event.Subtype, event.IsError, t.chatID, t.sessionID, t.model)
			if !t.streamDeltaSent && event.Result != "" {
				t.emit(w, flusher, openai.ChatCompletionResponse{
					ID: t.chatID, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
					Choices: []openai.ChatCompletionChoice{{
						Index: 0,
						Delta: openai.NewChatMessage("assistant", "⚠️ "+event.Result),
					}},
				})
			}
		}

		if eu := effectiveUsage(&event); eu != nil {
			if _, isIdentity := t.meter.(identityMeter); isIdentity {
				// V1 / ephemeral (one process per turn): the result figures ARE
				// the per-turn values. Bill per actual consuming model via
				// modelUsage (sub-agents may run on a different model); the
				// downstream usage chunk keeps the aggregated view.
				stats.RecordTurn(t.model, perModelCounts(&event, t.model))
				log.Printf("Token usage: model=%s input=%d output=%d cache_read=%d cache_create=%d cost=$%.4f (per-model attribution, models=%d)",
					t.model, eu.InputTokens, eu.OutputTokens,
					eu.CacheReadInputTokens, eu.CacheCreationInputTokens, event.TotalCostUSD,
					len(event.ModelUsage))
				t.streamUsage = openai.BuildUsageInfo(eu.InputTokens, eu.OutputTokens, eu.CacheReadInputTokens, eu.CacheCreationInputTokens)
			} else {
				// Persistent channel process: figures are process-cumulative;
				// the meter diffs them into this turn's delta. Attribution uses
				// meterModel (the worker's spawn-time model), not the request
				// model, so a hot model switch can't misattribute (A5).
				cur := usageSnapshot{
					input:          eu.InputTokens,
					output:         eu.OutputTokens,
					cacheCreation:  eu.CacheCreationInputTokens,
					cacheRead:      eu.CacheReadInputTokens,
					costUSD:        event.TotalCostUSD,
					fromModelUsage: len(event.ModelUsage) > 0,
				}
				d := t.meter.perTurn(cur)
				stats.Record(t.meterModel, d.input, d.output, d.cacheCreation, d.cacheRead, d.costUSD)
				log.Printf("Token usage: model=%s input=%d output=%d cache_read=%d cache_create=%d cost=$%.4f raw_input=%d raw_cache_read=%d (subagents=%v)",
					t.meterModel, d.input, d.output, d.cacheRead, d.cacheCreation, d.costUSD,
					cur.input, cur.cacheRead, cur.fromModelUsage)
				t.streamUsage = openai.BuildUsageInfo(d.input, d.output, d.cacheRead, d.cacheCreation)
			}
		}

		finishReason := "stop"
		t.emit(w, flusher, openai.ChatCompletionResponse{
			ID: t.chatID, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
			Choices: []openai.ChatCompletionChoice{
				{Index: 0, Delta: openai.NewChatMessage("", ""), FinishReason: &finishReason},
			},
		})

		if includeUsage && t.streamUsage != nil {
			t.emit(w, flusher, openai.ChatCompletionResponse{
				ID: t.chatID, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
				Choices: []openai.ChatCompletionChoice{},
				Usage:   t.streamUsage,
			})
		}
	}

	return outcomeContinue
}
