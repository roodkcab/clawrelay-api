// Package openai contains OpenAI Chat Completions–compatible request/response
// types and helpers shared by every relay binary in this repo. It deliberately
// avoids any backend-specific concepts (no Claude/Codex types here) so each
// relay can build its own translation layer on top.
package openai

import (
	"encoding/json"
	"strings"
)

// ChatCompletionRequest mirrors the OpenAI v1 chat/completions body, plus a
// handful of relay-specific extensions that backends may interpret natively
// or ignore (codex ignores Claude-only fields, and vice versa).
type ChatCompletionRequest struct {
	Model         string          `json:"model"`
	Messages      []ChatMessage   `json:"messages"`
	Stream        bool            `json:"stream,omitempty"`
	StreamOptions *StreamOptions  `json:"stream_options,omitempty"`
	Temperature   *float64        `json:"temperature,omitempty"`
	MaxTokens     *int            `json:"max_tokens,omitempty"`
	TopP          *float64        `json:"top_p,omitempty"`
	Stop          json.RawMessage `json:"stop,omitempty"`
	Tools         []Tool          `json:"tools,omitempty"`
	ToolChoice    json.RawMessage `json:"tool_choice,omitempty"`

	// Relay extensions
	WorkingDir       string            `json:"working_dir,omitempty"`
	EnvVars          map[string]string `json:"env_vars,omitempty"`
	MaxTurns         *int              `json:"max_turns,omitempty"`
	SessionID        string            `json:"session_id,omitempty"`
	Effort           string            `json:"effort,omitempty"`            // high/medium/low
	SystemPromptFile string            `json:"system_prompt_file,omitempty"`
	PermissionMode   string            `json:"permission_mode,omitempty"`
	AllowedTools     string            `json:"allowed_tools,omitempty"` // comma-separated
	AddDirs          []string          `json:"add_dirs,omitempty"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type ChatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Name       string          `json:"name,omitempty"`
	Thinking   string          `json:"thinking,omitempty"`
}

type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// NewChatMessage constructs a ChatMessage with a string content payload.
// The content is JSON-encoded so multipart variants can also fit in the same
// json.RawMessage slot.
func NewChatMessage(role, content string) *ChatMessage {
	raw, _ := json.Marshal(content)
	return &ChatMessage{Role: role, Content: raw}
}

// ContentString returns the textual portion of a message, transparently
// handling either a bare string content or an OpenAI multipart array
// (`[{"type":"text","text":"..."}]`).
func (m *ChatMessage) ContentString() string {
	if len(m.Content) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(m.Content, &parts); err == nil {
		var texts []string
		for _, p := range parts {
			if p.Type == "text" && p.Text != "" {
				texts = append(texts, p.Text)
			}
		}
		return strings.Join(texts, "")
	}
	return string(m.Content)
}

type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   *UsageInfo             `json:"usage,omitempty"`
}

type ChatCompletionChoice struct {
	Index        int          `json:"index"`
	Message      *ChatMessage `json:"message,omitempty"`
	Delta        *ChatMessage `json:"delta,omitempty"`
	FinishReason *string      `json:"finish_reason"`
}

type PromptTokensDetails struct {
	CachedTokens        int `json:"cached_tokens"`
	CacheCreationTokens int `json:"cache_creation_tokens,omitempty"`
}

// UsageInfo is the OpenAI-shaped usage block. prompt_tokens here represents
// total input tokens *including* cached/reused tokens; PromptTokensDetails
// breaks out how many were served from cache.
type UsageInfo struct {
	PromptTokens        int                  `json:"prompt_tokens"`
	CompletionTokens    int                  `json:"completion_tokens"`
	TotalTokens         int                  `json:"total_tokens"`
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`
}

type ModelListResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}
