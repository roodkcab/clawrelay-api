package main

import (
	"encoding/json"
	"strings"
)

// claudeEvent is one parsed line of `claude --output-format stream-json`.
// The CLI emits a stream of these events: init/system, assistant (with content
// blocks), stream_event (Anthropic Messages-API style sub-events) and result.
type claudeEvent struct {
	Type    string          `json:"type"`
	Subtype string          `json:"subtype,omitempty"`
	Message json.RawMessage `json:"message,omitempty"`
	Event   json.RawMessage `json:"event,omitempty"` // for stream_event wrapper
	Result  string          `json:"result,omitempty"`
	Usage   *claudeUsage    `json:"usage,omitempty"`
	// ModelUsage rolls up sub-agent (Task tool) tokens; when present, prefer
	// it over Usage which only counts the main agent.
	ModelUsage   map[string]claudeModelUsage `json:"modelUsage,omitempty"`
	TotalCostUSD float64                     `json:"total_cost_usd,omitempty"`
}

type claudeModelUsage struct {
	InputTokens              int `json:"inputTokens"`
	OutputTokens             int `json:"outputTokens"`
	CacheCreationInputTokens int `json:"cacheCreationInputTokens"`
	CacheReadInputTokens     int `json:"cacheReadInputTokens"`
}

// streamAPIEvent mirrors the inner event of a stream_event wrapper, matching
// Anthropic's Messages API streaming events.
type streamAPIEvent struct {
	Type         string          `json:"type"`
	Index        int             `json:"index,omitempty"`
	Delta        json.RawMessage `json:"delta,omitempty"`
	ContentBlock json.RawMessage `json:"content_block,omitempty"`
}

type streamContentBlock struct {
	Type string `json:"type"` // "text" or "tool_use"
	ID   string `json:"id,omitempty"`
	Name string `json:"name,omitempty"`
}

type streamTextDelta struct {
	Type        string `json:"type"` // text_delta, input_json_delta, thinking_delta
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
}

// nativeToolCall accumulates a tool_use block while it streams.
type nativeToolCall struct {
	ID   string
	Name string
	Args strings.Builder
}

type claudeMessage struct {
	Content []claudeContent `json:"content"`
}

type claudeContent struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

type claudeUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

// effectiveUsage returns the token usage to report for a result event. When
// modelUsage is present (Claude Code >= 2.x), it is the source of truth as
// it sums main + sub-agents (Task tool). Older CLI versions omit modelUsage,
// in which case usage is the main-agent figure.
func effectiveUsage(ev *claudeEvent) *claudeUsage {
	if ev == nil {
		return nil
	}
	if len(ev.ModelUsage) == 0 {
		return ev.Usage
	}
	var out claudeUsage
	for _, mu := range ev.ModelUsage {
		out.InputTokens += mu.InputTokens
		out.OutputTokens += mu.OutputTokens
		out.CacheCreationInputTokens += mu.CacheCreationInputTokens
		out.CacheReadInputTokens += mu.CacheReadInputTokens
	}
	return &out
}
