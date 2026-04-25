package main

import (
	"encoding/json"
	"fmt"
	"log"
	"regexp"
	"strings"

	"clawrelay-api/pkg/attachments"
	"clawrelay-api/pkg/openai"
)

// buildToolPrompt formats OpenAI-style tool definitions as a prompt addendum.
// Claude is asked to emit calls via native tool_use; this prompt is a fallback
// that the relay also recognizes via parseToolCalls (XML form).
func buildToolPrompt(tools []openai.Tool) string {
	if len(tools) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("\n\n# Available Tools\n\n")
	sb.WriteString("You have the following tools available. Call them when needed.\n\n")
	for _, tool := range tools {
		sb.WriteString(fmt.Sprintf("## %s\n", tool.Function.Name))
		if tool.Function.Description != "" {
			sb.WriteString(fmt.Sprintf("%s\n", tool.Function.Description))
		}
		if tool.Function.Parameters != nil {
			sb.WriteString(fmt.Sprintf("Parameters: %s\n", string(tool.Function.Parameters)))
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

var toolCallRe = regexp.MustCompile(`(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>`)

// parseToolCalls extracts XML-shaped <tool_call> blocks from a text body.
// Returns the cleaned text with tool blocks removed and the parsed calls.
func parseToolCalls(text string) (cleanText string, toolCalls []openai.ToolCall) {
	matches := toolCallRe.FindAllStringSubmatchIndex(text, -1)
	if len(matches) == 0 {
		return text, nil
	}

	var lastEnd int
	for _, match := range matches {
		cleanText += text[lastEnd:match[0]]
		lastEnd = match[1]

		callJSON := text[match[2]:match[3]]
		var call struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(callJSON), &call); err != nil {
			log.Printf("Failed to parse tool_call JSON: %v, raw: %s", err, callJSON)
			cleanText += text[match[0]:match[1]]
			continue
		}
		toolCalls = append(toolCalls, openai.ToolCall{
			ID:   openai.GenerateToolCallID(),
			Type: "function",
			Function: openai.ToolCallFunction{
				Name:      call.Name,
				Arguments: string(call.Arguments),
			},
		})
	}
	cleanText += text[lastEnd:]
	cleanText = strings.TrimSpace(cleanText)
	return
}

// extractNativeToolCalls scans assistant events for tool_use content blocks
// and converts them into OpenAI-format tool calls. AskUserQuestion is filtered
// out because it has special handling that closes the stream.
func extractNativeToolCalls(events []claudeEvent) []openai.ToolCall {
	var toolCalls []openai.ToolCall
	for _, event := range events {
		if event.Type != "assistant" || event.Message == nil {
			continue
		}
		var msg claudeMessage
		if err := json.Unmarshal(event.Message, &msg); err != nil {
			continue
		}
		for _, c := range msg.Content {
			if c.Type == "tool_use" && c.Name != "" && c.Name != "AskUserQuestion" {
				args := "{}"
				if c.Input != nil {
					args = string(c.Input)
				}
				toolCalls = append(toolCalls, openai.ToolCall{
					ID:   openai.GenerateToolCallID(),
					Type: "function",
					Function: openai.ToolCallFunction{
						Name:      c.Name,
						Arguments: args,
					},
				})
			}
		}
	}
	return toolCalls
}

// extractAskUserQuestionCalls pulls only the AskUserQuestion tool_use blocks.
func extractAskUserQuestionCalls(events []claudeEvent) []openai.ToolCall {
	var calls []openai.ToolCall
	for _, event := range events {
		if event.Type != "assistant" || event.Message == nil {
			continue
		}
		var msg claudeMessage
		if err := json.Unmarshal(event.Message, &msg); err != nil {
			continue
		}
		for _, c := range msg.Content {
			if c.Type == "tool_use" && c.Name == "AskUserQuestion" {
				args := "{}"
				if c.Input != nil {
					args = string(c.Input)
				}
				calls = append(calls, openai.ToolCall{
					ID:   openai.GenerateToolCallID(),
					Type: "function",
					Function: openai.ToolCallFunction{
						Name:      "AskUserQuestion",
						Arguments: args,
					},
				})
			}
		}
	}
	return calls
}

// extractTextFromEvent pulls the text portion of an assistant event,
// suppressing tool_use blocks (those are surfaced separately).
func extractTextFromEvent(event *claudeEvent) string {
	if event.Type != "assistant" || event.Message == nil {
		return ""
	}
	var msg claudeMessage
	if err := json.Unmarshal(event.Message, &msg); err != nil {
		return ""
	}
	var texts []string
	for _, c := range msg.Content {
		switch c.Type {
		case "text":
			if c.Text != "" {
				texts = append(texts, c.Text)
			}
		case "tool_use":
			continue
		}
	}
	return strings.Join(texts, "")
}

// toolCallFilter buffers streaming text so <tool_call>…</tool_call> XML blocks
// are not emitted to the client mid-stream. Safe text passes through;
// in-flight tool blocks are held until closed.
type toolCallFilter struct {
	pending    string
	toolBlocks []string
}

func (f *toolCallFilter) Feed(text string) string {
	f.pending += text
	return f.drain()
}

func (f *toolCallFilter) drain() string {
	var safe strings.Builder
	for len(f.pending) > 0 {
		idx := strings.Index(f.pending, "<")
		if idx < 0 {
			safe.WriteString(f.pending)
			f.pending = ""
			break
		}
		if idx > 0 {
			safe.WriteString(f.pending[:idx])
			f.pending = f.pending[idx:]
		}
		const openTag = "<tool_call>"
		if len(f.pending) < len(openTag) {
			if strings.HasPrefix(openTag, f.pending) {
				break
			}
			safe.WriteByte('<')
			f.pending = f.pending[1:]
			continue
		}
		if !strings.HasPrefix(f.pending, openTag) {
			safe.WriteByte('<')
			f.pending = f.pending[1:]
			continue
		}
		const closeTag = "</tool_call>"
		endIdx := strings.Index(f.pending, closeTag)
		if endIdx < 0 {
			break
		}
		block := f.pending[:endIdx+len(closeTag)]
		f.toolBlocks = append(f.toolBlocks, block)
		f.pending = f.pending[endIdx+len(closeTag):]
	}
	return safe.String()
}

func (f *toolCallFilter) Finish() (remaining string, blocks []string) {
	return f.pending, f.toolBlocks
}

// buildPromptFromMessages flattens an OpenAI message array into a single
// prompt string, separately returning the system prompt. Attachments are
// dumped to disk and inlined as `[Image: /path]` / `[File: /path]` markers.
// When sessionDir is non-empty, attachments are content-hashed there for
// cross-turn dedup; otherwise tempFiles is populated for cleanup.
func buildPromptFromMessages(messages []openai.ChatMessage, sessionDir string) (prompt, systemPrompt string, tempFiles []string) {
	var parts []string
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			systemPrompt = msg.ContentString()
		case "user":
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
			parts = append(parts, fmt.Sprintf("Human: %s", text))
		case "assistant":
			text := msg.ContentString()
			if len(msg.ToolCalls) > 0 {
				var callBlocks []string
				for _, tc := range msg.ToolCalls {
					callBlocks = append(callBlocks, fmt.Sprintf(
						"<tool_call>\n{\"name\": \"%s\", \"arguments\": %s}\n</tool_call>",
						tc.Function.Name, tc.Function.Arguments))
				}
				if text != "" {
					text += "\n\n"
				}
				text += strings.Join(callBlocks, "\n")
			}
			parts = append(parts, fmt.Sprintf("Assistant: %s", text))
		case "tool":
			name := msg.Name
			if name == "" {
				name = "tool"
			}
			parts = append(parts, fmt.Sprintf("Tool result for %s (call_id: %s):\n%s",
				name, msg.ToolCallID, msg.ContentString()))
		default:
			parts = append(parts, fmt.Sprintf("%s: %s", msg.Role, msg.ContentString()))
		}
	}
	prompt = strings.Join(parts, "\n\n")
	return
}
