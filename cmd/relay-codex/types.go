package main

import "encoding/json"

// codexEvent is one parsed line of `codex exec --json`. Codex's event schema
// is item-oriented: every assistant message, command execution, file edit,
// and reasoning step arrives as an `item.started` / `item.completed` pair.
type codexEvent struct {
	Type     string          `json:"type"`
	ThreadID string          `json:"thread_id,omitempty"`
	Item     *codexItem      `json:"item,omitempty"`
	Usage    *codexUsage     `json:"usage,omitempty"`
	Message  string          `json:"message,omitempty"`
	Error    json.RawMessage `json:"error,omitempty"`
}

// codexItem covers every concrete `item` payload codex emits. Different item
// types populate different fields; absent fields default to zero values.
type codexItem struct {
	ID                string `json:"id"`
	Type              string `json:"type"` // agent_message, command_execution, reasoning, file_change, ...
	Text              string `json:"text,omitempty"`
	Command           string `json:"command,omitempty"`
	AggregatedOutput  string `json:"aggregated_output,omitempty"`
	ExitCode          *int   `json:"exit_code,omitempty"`
	Status            string `json:"status,omitempty"`

	// Reasoning items carry their thought trace in `text` plus, in some CLI
	// versions, a `summary` / `parts` field. We capture raw to be lenient.
	Raw json.RawMessage `json:"-"`
}

// codexUsage is what `turn.completed.usage` reports. input_tokens here is the
// *total* prompt size (including cached); cached_input_tokens is the cached
// subset. We map cached_input_tokens → cache_read for the OpenAI shape.
type codexUsage struct {
	InputTokens       int `json:"input_tokens"`
	CachedInputTokens int `json:"cached_input_tokens"`
	OutputTokens      int `json:"output_tokens"`
}
