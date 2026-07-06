package main

import (
	"encoding/json"
	"strings"

	"clawrelay-api/pkg/openai"
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
	// CostUSD is per-model cost. Present even on SIGINT-aborted turns (claude
	// 2.1.199: result subtype=error_during_execution has bare usage all-zero but
	// modelUsage entries carry real tokens AND costUSD).
	CostUSD float64 `json:"costUSD"`
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

// perModelCounts 把 result 的 modelUsage 转成 stats.RecordTurn 的输入；
// modelUsage 为空时回退单条 {fallbackModel: bare usage + totalCost}。
// 用于 V1（每请求进程）与 ephemeral：这些进程只活一轮，result 里的数字天然是
// per-turn 值，直接按实际消费的模型归属（子代理/Task 工具可能用别的模型）。
func perModelCounts(ev *claudeEvent, fallbackModel string) map[string]openai.TokenCounts {
	if ev == nil {
		return nil
	}
	if len(ev.ModelUsage) > 0 {
		out := make(map[string]openai.TokenCounts, len(ev.ModelUsage))
		var costSum float64
		for model, mu := range ev.ModelUsage {
			out[model] = openai.TokenCounts{
				Input:         mu.InputTokens,
				Output:        mu.OutputTokens,
				CacheCreation: mu.CacheCreationInputTokens,
				CacheRead:     mu.CacheReadInputTokens,
				CostUSD:       mu.CostUSD,
			}
			costSum += mu.CostUSD
		}
		// 兜底：某些 CLI 版本的 modelUsage 条目不带 costUSD——不能让成本静默
		// 归零，把顶层 total_cost_usd 归到 fallbackModel 名下。
		if costSum == 0 && ev.TotalCostUSD > 0 {
			fc := out[fallbackModel]
			fc.CostUSD = ev.TotalCostUSD
			out[fallbackModel] = fc
		}
		return out
	}
	if ev.Usage == nil {
		return nil
	}
	return map[string]openai.TokenCounts{
		fallbackModel: {
			Input:         ev.Usage.InputTokens,
			Output:        ev.Usage.OutputTokens,
			CacheCreation: ev.Usage.CacheCreationInputTokens,
			CacheRead:     ev.Usage.CacheReadInputTokens,
			CostUSD:       ev.TotalCostUSD,
		},
	}
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
