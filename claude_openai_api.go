package main

import (
	"bufio"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"time"
)

// ---- OpenAI-compatible request/response types ----

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
	WorkingDir    string            `json:"working_dir,omitempty"`
	EnvVars       map[string]string `json:"env_vars,omitempty"`
	MaxTurns      *int              `json:"max_turns,omitempty"`
	SessionID     string            `json:"session_id,omitempty"`
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

// NewChatMessage creates a ChatMessage with string content for responses.
func NewChatMessage(role, content string) *ChatMessage {
	raw, _ := json.Marshal(content)
	return &ChatMessage{Role: role, Content: raw}
}

// ContentString extracts text from content, which can be a string or
// an array of content parts (e.g. [{"type":"text","text":"..."}]).
func (m *ChatMessage) ContentString() string {
	if len(m.Content) == 0 {
		return ""
	}
	// Try string first
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return s
	}
	// Try array of content parts
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
	CachedTokens int `json:"cached_tokens"`
}

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

// ---- Claude stream-json event types ----

type ClaudeEvent struct {
	Type    string          `json:"type"`
	Subtype string          `json:"subtype,omitempty"`
	Message json.RawMessage `json:"message,omitempty"`
	Event   json.RawMessage `json:"event,omitempty"` // for stream_event wrapper
	Result  string          `json:"result,omitempty"`
	Usage   *ClaudeUsage    `json:"usage,omitempty"`
}

// StreamAPIEvent represents the inner event of a stream_event wrapper,
// mirroring Anthropic's Messages API streaming events.
type StreamAPIEvent struct {
	Type         string          `json:"type"` // message_start, content_block_start, content_block_delta, etc.
	Index        int             `json:"index,omitempty"`
	Delta        json.RawMessage `json:"delta,omitempty"`
	ContentBlock json.RawMessage `json:"content_block,omitempty"`
}

// StreamContentBlock represents the content_block field in content_block_start events.
type StreamContentBlock struct {
	Type string `json:"type"` // "text" or "tool_use"
	ID   string `json:"id,omitempty"`
	Name string `json:"name,omitempty"`
}

type StreamTextDelta struct {
	Type        string `json:"type"` // text_delta, input_json_delta, thinking_delta
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
}

// nativeToolCall tracks a tool_use content block from Claude CLI's stream output.
type nativeToolCall struct {
	ID   string
	Name string
	Args strings.Builder
}

type ClaudeMessage struct {
	Content []ClaudeContent `json:"content"`
}

type ClaudeContent struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

type ClaudeUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

// ---- Token statistics ----

type ModelTokenStats struct {
	Requests int64 `json:"requests"`
	Input    int64 `json:"input_tokens"`
	Output   int64 `json:"output_tokens"`
	Total    int64 `json:"total_tokens"`
}

type TokenStatsSnapshot struct {
	TotalRequests int64                       `json:"total_requests"`
	InputTokens   int64                       `json:"input_tokens"`
	OutputTokens  int64                       `json:"output_tokens"`
	TotalTokens   int64                       `json:"total_tokens"`
	PerModel      map[string]*ModelTokenStats `json:"per_model"`
	StartTime     string                      `json:"start_time"`
	Uptime        string                      `json:"uptime"`
}

type tokenStats struct {
	mu        sync.Mutex
	requests  int64
	input     int64
	output    int64
	perModel  map[string]*ModelTokenStats
	startTime time.Time
}

var globalStats = &tokenStats{
	perModel:  make(map[string]*ModelTokenStats),
	startTime: time.Now(),
}

func (s *tokenStats) Record(model string, input, output int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests++
	s.input += int64(input)
	s.output += int64(output)

	ms, ok := s.perModel[model]
	if !ok {
		ms = &ModelTokenStats{}
		s.perModel[model] = ms
	}
	ms.Requests++
	ms.Input += int64(input)
	ms.Output += int64(output)
	ms.Total += int64(input + output)
}

func (s *tokenStats) Snapshot() TokenStatsSnapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	pm := make(map[string]*ModelTokenStats, len(s.perModel))
	for k, v := range s.perModel {
		cp := *v
		pm[k] = &cp
	}
	return TokenStatsSnapshot{
		TotalRequests: s.requests,
		InputTokens:   s.input,
		OutputTokens:  s.output,
		TotalTokens:   s.input + s.output,
		PerModel:      pm,
		StartTime:     s.startTime.Format(time.RFC3339),
		Uptime:        time.Since(s.startTime).Truncate(time.Second).String(),
	}
}

// buildUsageInfo converts Claude's usage into OpenAI-compatible UsageInfo.
// prompt_tokens = input_tokens + cache_read + cache_creation (total input)
// prompt_tokens_details.cached_tokens = cache_read_input_tokens
func buildUsageInfo(cu *ClaudeUsage) *UsageInfo {
	if cu == nil {
		return nil
	}
	promptTokens := cu.InputTokens + cu.CacheReadInputTokens + cu.CacheCreationInputTokens
	u := &UsageInfo{
		PromptTokens:     promptTokens,
		CompletionTokens: cu.OutputTokens,
		TotalTokens:      promptTokens + cu.OutputTokens,
	}
	if cu.CacheReadInputTokens > 0 {
		u.PromptTokensDetails = &PromptTokensDetails{
			CachedTokens: cu.CacheReadInputTokens,
		}
	}
	return u
}

// ---- Model mapping ----

var modelAliases = map[string]string{
	"gpt-4":         "opus",
	"gpt-4o":        "sonnet",
	"gpt-4-turbo":   "sonnet",
	"gpt-3.5-turbo": "haiku",
	"gpt-4o-mini":   "haiku",
}

const defaultModel = "vllm/claude-sonnet-4-6"

var availableModels = []ModelInfo{
	{ID: "vllm/claude-sonnet-4-6", Object: "model", Created: 1700000000, OwnedBy: "anthropic"},
	{ID: "vllm/claude-sonnet-4-20250514", Object: "model", Created: 1700000000, OwnedBy: "anthropic"},
	{ID: "vllm/sonnet", Object: "model", Created: 1700000000, OwnedBy: "anthropic"},
	{ID: "vllm/claude-opus-4-6", Object: "model", Created: 1700000000, OwnedBy: "anthropic"},
	{ID: "vllm/claude-opus-4-20250514", Object: "model", Created: 1700000000, OwnedBy: "anthropic"},
	{ID: "vllm/opus", Object: "model", Created: 1700000000, OwnedBy: "anthropic"},
	{ID: "vllm/claude-haiku-4-5-20251001", Object: "model", Created: 1700000000, OwnedBy: "anthropic"},
	{ID: "vllm/haiku", Object: "model", Created: 1700000000, OwnedBy: "anthropic"},
}

// CORS origins (same as claude_stream_api.go)
var oaiAllowedOrigins = []string{
	"http://10.0.100.148:5173",
	"https://goofish-stat.52ritao.cn",
}

func resolveModel(model string) string {
	if idx := strings.LastIndex(model, "/"); idx >= 0 {
		model = model[idx+1:]
	}
	if alias, ok := modelAliases[model]; ok {
		return alias
	}
	return model
}

func generateChatID() string {
	b := make([]byte, 12)
	rand.Read(b)
	return "chatcmpl-" + hex.EncodeToString(b)
}

func generateToolCallID() string {
	b := make([]byte, 12)
	rand.Read(b)
	return "call_" + hex.EncodeToString(b)
}

func setOAICORSHeaders(w http.ResponseWriter, r *http.Request) {
	origin := r.Header.Get("Origin")
	for _, allowed := range oaiAllowedOrigins {
		if origin == allowed {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			break
		}
	}
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
	w.Header().Set("Access-Control-Max-Age", "3600")
}

// ---- Tool support ----

// buildToolPrompt formats tool definitions for injection into the system prompt.
// Claude will use native tool_use to call these; we intercept from the stream.
func buildToolPrompt(tools []Tool) string {
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


// parseToolCalls extracts <tool_call> blocks from text.
// Returns the remaining clean text and parsed tool calls.
func parseToolCalls(text string) (cleanText string, toolCalls []ToolCall) {
	matches := toolCallRe.FindAllStringSubmatchIndex(text, -1)
	if len(matches) == 0 {
		return text, nil
	}

	var lastEnd int
	for _, match := range matches {
		// match[0]:match[1] is the full match, match[2]:match[3] is the JSON capture group
		cleanText += text[lastEnd:match[0]]
		lastEnd = match[1]

		callJSON := text[match[2]:match[3]]
		var call struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(callJSON), &call); err != nil {
			log.Printf("Failed to parse tool_call JSON: %v, raw: %s", err, callJSON)
			// Put it back as text if parsing fails
			cleanText += text[match[0]:match[1]]
			continue
		}
		toolCalls = append(toolCalls, ToolCall{
			ID:   generateToolCallID(),
			Type: "function",
			Function: ToolCallFunction{
				Name:      call.Name,
				Arguments: string(call.Arguments),
			},
		})
	}
	cleanText += text[lastEnd:]
	cleanText = strings.TrimSpace(cleanText)
	return
}

// extractNativeToolCalls scans Claude CLI events for tool_use content blocks
// (from assistant messages) and converts them to OpenAI-format ToolCalls.
func extractNativeToolCalls(events []ClaudeEvent) []ToolCall {
	var toolCalls []ToolCall
	for _, event := range events {
		if event.Type != "assistant" || event.Message == nil {
			continue
		}
		var msg ClaudeMessage
		if err := json.Unmarshal(event.Message, &msg); err != nil {
			continue
		}
		for _, c := range msg.Content {
			if c.Type == "tool_use" && c.Name != "" {
				args := "{}"
				if c.Input != nil {
					args = string(c.Input)
				}
				toolCalls = append(toolCalls, ToolCall{
					ID:   generateToolCallID(),
					Type: "function",
					Function: ToolCallFunction{
						Name:      c.Name,
						Arguments: args,
					},
				})
			}
		}
	}
	return toolCalls
}

// toolCallFilter intercepts streaming text to prevent <tool_call>...</tool_call>
// blocks from being sent as regular content. Safe text is returned immediately;
// potential tool call blocks are held back until confirmed or the stream ends.
type toolCallFilter struct {
	pending    string
	toolBlocks []string
}

// Feed adds new text and returns any text that is safe to send as content.
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
		// f.pending starts with '<'
		const openTag = "<tool_call>"
		if len(f.pending) < len(openTag) {
			if strings.HasPrefix(openTag, f.pending) {
				break // might still become <tool_call>, hold
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
		// Found <tool_call>, look for closing tag
		const closeTag = "</tool_call>"
		endIdx := strings.Index(f.pending, closeTag)
		if endIdx < 0 {
			break // no end tag yet, keep buffering
		}
		block := f.pending[:endIdx+len(closeTag)]
		f.toolBlocks = append(f.toolBlocks, block)
		f.pending = f.pending[endIdx+len(closeTag):]
	}
	return safe.String()
}

// Finish flushes remaining pending text and returns collected tool call blocks.
func (f *toolCallFilter) Finish() (remaining string, blocks []string) {
	return f.pending, f.toolBlocks
}

// ---- Message conversion ----

// extractAndSaveImages detects image_url content parts in a message, saves each
// base64-encoded image to a temp file, and returns the file paths.
func extractAndSaveImages(content json.RawMessage) []string {
	if len(content) == 0 {
		return nil
	}
	var parts []struct {
		Type     string `json:"type"`
		ImageURL *struct {
			URL string `json:"url"`
		} `json:"image_url,omitempty"`
	}
	if err := json.Unmarshal(content, &parts); err != nil {
		return nil // not an array, no images
	}
	var paths []string
	for _, p := range parts {
		if p.Type != "image_url" || p.ImageURL == nil {
			continue
		}
		url := p.ImageURL.URL
		if !strings.HasPrefix(url, "data:") {
			continue
		}
		comma := strings.Index(url, ",")
		if comma < 0 {
			continue
		}
		header := url[5:comma] // e.g. "image/png;base64"
		b64Data := url[comma+1:]

		ext := ".png"
		if strings.Contains(header, "jpeg") || strings.Contains(header, "jpg") {
			ext = ".jpg"
		} else if strings.Contains(header, "gif") {
			ext = ".gif"
		} else if strings.Contains(header, "webp") {
			ext = ".webp"
		}

		imgBytes, err := base64.StdEncoding.DecodeString(b64Data)
		if err != nil {
			log.Printf("Failed to decode image base64: %v", err)
			continue
		}

		var randBytes [8]byte
		if _, err := rand.Read(randBytes[:]); err != nil {
			continue
		}
		tmpPath := fmt.Sprintf("/tmp/claude-img-%s%s", hex.EncodeToString(randBytes[:]), ext)
		if err := os.WriteFile(tmpPath, imgBytes, 0600); err != nil {
			log.Printf("Failed to write temp image %s: %v", tmpPath, err)
			continue
		}
		log.Printf("Saved attached image to: %s", tmpPath)
		paths = append(paths, tmpPath)
	}
	return paths
}

// buildPromptFromMessages converts OpenAI-style messages into a single prompt string
// and extracts the system prompt separately.
func buildPromptFromMessages(messages []ChatMessage) (prompt string, systemPrompt string, tempFiles []string) {
	var parts []string
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			systemPrompt = msg.ContentString()
		case "user":
			text := msg.ContentString()
			imgPaths := extractAndSaveImages(msg.Content)
			tempFiles = append(tempFiles, imgPaths...)
			if len(imgPaths) > 0 {
				var refs []string
				for _, p := range imgPaths {
					refs = append(refs, fmt.Sprintf("[Image: %s]", p))
				}
				if text != "" {
					text += "\n"
				}
				text += strings.Join(refs, "\n")
			}
			parts = append(parts, fmt.Sprintf("Human: %s", text))
		case "assistant":
			text := msg.ContentString()
			// Reconstruct tool calls as <tool_call> blocks so Claude sees the history
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
			// Tool result message — include with call_id context
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

// extractTextFromEvent parses a Claude stream-json event and returns any text content.
func extractTextFromEvent(event *ClaudeEvent) string {
	if event.Type != "assistant" || event.Message == nil {
		return ""
	}
	var msg ClaudeMessage
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
			// Claude CLI's own built-in tool calls — suppress from output.
			// These are internal to Claude Code (Bash, Read, etc.) and should
			// not leak to the caller as text content.
			continue
		}
	}
	return strings.Join(texts, "")
}

// truncate returns at most maxLen characters of s, appending "..." if truncated.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// ---- Handlers ----

func chatCompletionsHandler(w http.ResponseWriter, r *http.Request) {
	setOAICORSHeaders(w, r)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != http.MethodPost {
		writeOAIError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only POST is accepted")
		return
	}

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		writeOAIError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("Failed to read body: %v", err))
		return
	}
	log.Printf("Raw request body (%d bytes): %s", len(bodyBytes), string(bodyBytes))

	var req ChatCompletionRequest
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		writeOAIError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("Invalid JSON: %v", err))
		return
	}
	if len(req.Messages) == 0 {
		writeOAIError(w, http.StatusBadRequest, "invalid_request_error", "messages array is required and must not be empty")
		return
	}

	model := resolveModel(req.Model)
	if model == "" {
		model = defaultModel
	}
	prompt, systemPrompt, tempFiles := buildPromptFromMessages(req.Messages)
	if len(tempFiles) > 0 {
		defer func() {
			for _, f := range tempFiles {
				os.Remove(f)
			}
		}()
	}

	// Inject tool definitions into system prompt
	hasTools := len(req.Tools) > 0
	if hasTools {
		systemPrompt += buildToolPrompt(req.Tools)
		log.Printf("Injected %d tool definitions into system prompt", len(req.Tools))
	}

	chatID := generateChatID()
	created := time.Now().Unix()

	log.Printf("ChatCompletion request: model=%s stream=%v messages=%d tools=%d",
		model, req.Stream, len(req.Messages), len(req.Tools))

	// Build claude CLI args
	args := []string{}
	if systemPrompt != "" {
		args = append(args, "--system-prompt", systemPrompt)
	}
	args = append(args, "--model", model)
	args = append(args, "--verbose")
	args = append(args, "--output-format", "stream-json")
	args = append(args, "--include-partial-messages")
	args = append(args, "--permission-mode", "bypassPermissions")

	maxTurns := 200
	if req.MaxTurns != nil {
		maxTurns = *req.MaxTurns
	}
	args = append(args, "--max-turns", fmt.Sprintf("%d", maxTurns))

	if req.SessionID != "" {
		args = append(args, "--resume", req.SessionID)
	}

	log.Printf("Claude args: %v (prompt length: %d bytes via stdin)", args, len(prompt))

	includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage

	workingDir := req.WorkingDir
	envVars := req.EnvVars

	if req.Stream {
		if hasTools {
			// Buffer output to properly detect and format tool calls
			handleBufferedStreamResponse(w, r, args, prompt, chatID, created, model, includeUsage, workingDir, envVars)
		} else {
			handleStreamResponse(w, r, args, prompt, chatID, created, model, includeUsage, workingDir, envVars)
		}
	} else {
		handleNonStreamResponse(w, r, args, prompt, chatID, created, model, hasTools, workingDir, envVars)
	}
}

// cleanEnv returns the current environment with CLAUDECODE removed,
// plus any extra KEY=VALUE pairs from the request.
func cleanEnv(extra map[string]string) []string {
	var env []string
	for _, e := range os.Environ() {
		if !strings.HasPrefix(e, "CLAUDECODE=") {
			env = append(env, e)
		}
	}
	for k, v := range extra {
		env = append(env, k+"="+v)
	}
	return env
}

// hasArg returns true if flag appears in args.
func hasArg(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

// replaceArg replaces the first occurrence of old with new in a copy of args.
func replaceArg(args []string, old, new string) []string {
	result := make([]string, len(args))
	copy(result, args)
	for i, a := range result {
		if a == old {
			result[i] = new
			break
		}
	}
	return result
}

// launchClaude starts a claude process and returns:
//   - lines: a channel of stdout lines (closed on EOF after cmd.Wait)
//   - sessErrCh: receives true after stderr is fully consumed if "No conversation found" was seen
func launchClaude(args []string, prompt string, workingDir string, envVars map[string]string) (<-chan string, <-chan bool, error) {
	cmd := exec.Command("claude", args...)
	cmd.Env = cleanEnv(envVars)
	if workingDir != "" {
		cmd.Dir = workingDir
	}
	cmd.Stdin = strings.NewReader(prompt)

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create stdout pipe: %v", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create stderr pipe: %v", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, nil, fmt.Errorf("failed to start claude: %v", err)
	}

	sessErrCh := make(chan bool, 1)
	var stderrDone sync.WaitGroup
	stderrDone.Add(1)
	go func() {
		defer stderrDone.Done()
		s := bufio.NewScanner(stderrPipe)
		found := false
		for s.Scan() {
			line := s.Text()
			log.Printf("Claude stderr: %s", line)
			if strings.Contains(line, "No conversation found with session ID") {
				found = true
			}
		}
		sessErrCh <- found
	}()

	lines := make(chan string, 128)
	go func() {
		defer close(lines)
		s := bufio.NewScanner(stdoutPipe)
		s.Buffer(make([]byte, 1024*1024), 1024*1024)
		for s.Scan() {
			lines <- s.Text()
		}
		stderrDone.Wait()
		if err := cmd.Wait(); err != nil {
			log.Printf("Claude command error: %v", err)
		}
	}()

	return lines, sessErrCh, nil
}

// startClaudeStream starts claude with session-retry support.
// If --resume fails (no output, or only init/system/result events with no real
// content), it retries with --session-id.
// Sends nil on ready when it is safe to write HTTP response headers; sends an
// error if the process could not be started. After ready, read lines from the
// returned channel until it is closed.
func startClaudeStream(args []string, prompt string, workingDir string, envVars map[string]string) (<-chan string, <-chan error) {
	lines := make(chan string, 128)
	ready := make(chan error, 1)

	go func() {
		defer close(lines)

		innerLines, sessErrCh, err := launchClaude(args, prompt, workingDir, envVars)
		if err != nil {
			ready <- err
			return
		}

		if hasArg(args, "--resume") {
			// Buffer early events to detect failed --resume before committing.
			// A failed resume typically emits only init/system/result events
			// (with 0 tokens) then exits. A successful one emits assistant or
			// stream_event events with real content.
			var buffered []string
			gotContent := false
			shouldRetry := false

			for line := range innerLines {
				buffered = append(buffered, line)
				var evt struct {
					Type string `json:"type"`
				}
				if json.Unmarshal([]byte(line), &evt) != nil {
					continue
				}
				if evt.Type == "init" || evt.Type == "system" {
					continue
				}
				if evt.Type == "result" {
					// result as first content event = failed session
					shouldRetry = true
					for range innerLines {
					} // drain
					break
				}
				// Got real content (assistant, stream_event, etc.)
				gotContent = true
				break
			}

			if !gotContent && !shouldRetry {
				// Channel closed with only init/system events or no events
				shouldRetry = true
			}

			if shouldRetry {
				<-sessErrCh
				log.Printf("Resume produced no content, retrying with --session-id")
				retryArgs := replaceArg(args, "--resume", "--session-id")
				innerLines, sessErrCh, err = launchClaude(retryArgs, prompt, workingDir, envVars)
				if err != nil {
					ready <- err
					return
				}
				go func() { <-sessErrCh }()

				firstLine, ok := <-innerLines
				ready <- nil
				if ok {
					lines <- firstLine
					for line := range innerLines {
						lines <- line
					}
				}
				return
			}

			// No retry — forward buffered lines and remaining
			go func() { <-sessErrCh }()
			ready <- nil
			for _, line := range buffered {
				lines <- line
			}
			for line := range innerLines {
				lines <- line
			}
			return
		}

		// No --resume: original behavior
		firstLine, ok := <-innerLines
		if !ok {
			go func() { <-sessErrCh }()
			ready <- nil
			return
		}
		go func() { <-sessErrCh }()
		ready <- nil
		lines <- firstLine
		for line := range innerLines {
			lines <- line
		}
	}()

	return lines, ready
}

// runClaude starts a claude process and collects its output events.
func runClaude(args []string, prompt string, workingDir string, envVars map[string]string) (events []ClaudeEvent, lastText string, result string, usage *UsageInfo, err error) {
	lines, sessErrCh, err := launchClaude(args, prompt, workingDir, envVars)
	if err != nil {
		return
	}

	for line := range lines {
		if line == "" {
			continue
		}
		log.Printf("[CLAUDE RAW] %s", line)
		var event ClaudeEvent
		if jsonErr := json.Unmarshal([]byte(line), &event); jsonErr != nil {
			log.Printf("Failed to parse claude event: %v", jsonErr)
			continue
		}
		events = append(events, event)

		text := extractTextFromEvent(&event)
		if text != "" {
			lastText = text
		}
		if event.Type == "result" {
			if event.Result != "" {
				result = event.Result
			}
			if event.Usage != nil {
				usage = buildUsageInfo(event.Usage)
			}
		}
	}

	// If --resume produced no real content, retry with --session-id.
	// A failed resume may still emit init/system/result events with 0 tokens.
	<-sessErrCh
	if hasArg(args, "--resume") && lastText == "" && result == "" {
		log.Printf("Resume produced no content, retrying with --session-id")
		retryArgs := replaceArg(args, "--resume", "--session-id")
		return runClaude(retryArgs, prompt, workingDir, envVars)
	}
	return
}

// handleStreamResponse streams text output without tool call detection (fast path).
func handleStreamResponse(w http.ResponseWriter, r *http.Request, args []string, prompt string, chatID string, created int64, model string, includeUsage bool, workingDir string, envVars map[string]string) {
	lines, ready := startClaudeStream(args, prompt, workingDir, envVars)
	if err := <-ready; err != nil {
		writeOAIError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

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

	var streamDeltaSent bool // true if we've sent any stream_event deltas
	seenToolNames := map[string]bool{} // deduplicate tool_call chunks across both detection paths

	// Aggregated logging: accumulate chunks of the same event type and flush on type change
	var aggType string      // current aggregated event type (e.g. "text_delta", "thinking_delta")
	var aggBuf strings.Builder // accumulated text content
	var aggCount int         // number of chunks aggregated

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

	for line := range lines {
		if line == "" {
			continue
		}

		var event ClaudeEvent
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			log.Printf("Failed to parse claude event: %v", err)
			continue
		}

		// Handle stream_event with content_block_delta for real-time token streaming
		if event.Type == "stream_event" && event.Event != nil {
			var streamEvt StreamAPIEvent
			if err := json.Unmarshal(event.Event, &streamEvt); err != nil {
				continue
			}
			// Emit tool_calls delta immediately when Claude starts a tool_use block
			if streamEvt.Type == "content_block_start" && streamEvt.ContentBlock != nil {
				var block StreamContentBlock
				if err := json.Unmarshal(streamEvt.ContentBlock, &block); err == nil && block.Type == "tool_use" && block.Name != "" {
					flushAggLog()
					log.Printf("[STREAM TOOL_USE] name=%s id=%s", block.Name, block.ID)
					seenToolNames[block.Name] = true
					tc := ToolCall{
						ID:   block.ID,
						Type: "function",
						Function: ToolCallFunction{Name: block.Name, Arguments: ""},
					}
					chunk := ChatCompletionResponse{
						ID:      chatID,
						Object:  "chat.completion.chunk",
						Created: created,
						Model:   model,
						Choices: []ChatCompletionChoice{
							{
								Index: 0,
								Delta: &ChatMessage{
									Role:      "assistant",
									ToolCalls: []ToolCall{tc},
								},
							},
						},
					}
					data, _ := json.Marshal(chunk)
					fmt.Fprintf(w, "data: %s\n\n", data)
					flusher.Flush()
				}
			}
			if streamEvt.Type == "content_block_delta" && streamEvt.Delta != nil {
				var delta StreamTextDelta
				if err := json.Unmarshal(streamEvt.Delta, &delta); err != nil {
					continue
				}
				if delta.Type == "text_delta" && delta.Text != "" {
					streamDeltaSent = true
					aggAppend("text_delta", delta.Text)

					chunk := ChatCompletionResponse{
						ID:      chatID,
						Object:  "chat.completion.chunk",
						Created: created,
						Model:   model,
						Choices: []ChatCompletionChoice{
							{
								Index:        0,
								Delta:        NewChatMessage("assistant", delta.Text),
								FinishReason: nil,
							},
						},
					}
					data, _ := json.Marshal(chunk)
					fmt.Fprintf(w, "data: %s\n\n", data)
					flusher.Flush()
				}
				if delta.Type == "thinking_delta" && delta.Thinking != "" {
					aggAppend("thinking_delta", delta.Thinking)
					chunk := ChatCompletionResponse{
						ID:      chatID,
						Object:  "chat.completion.chunk",
						Created: created,
						Model:   model,
						Choices: []ChatCompletionChoice{
							{
								Index:        0,
								Delta:        &ChatMessage{Role: "assistant", Thinking: delta.Thinking},
								FinishReason: nil,
							},
						},
					}
					data, _ := json.Marshal(chunk)
					fmt.Fprintf(w, "data: %s\n\n", data)
					flusher.Flush()
				}
			}
			continue
		}

		// Non-delta event: flush aggregated log before processing
		flushAggLog()

		// Fallback: extract tool_use and text from assistant summary events.
		// Handles the case where content_block_start stream events were absent
		// (e.g., claude CLI emits a complete assistant message instead of streaming).
		if event.Type == "assistant" && event.Message != nil {
			var msg ClaudeMessage
			if err := json.Unmarshal(event.Message, &msg); err == nil {
				for _, c := range msg.Content {
					if c.Type == "tool_use" && c.Name != "" && !seenToolNames[c.Name] {
						seenToolNames[c.Name] = true
						log.Printf("[ASSISTANT TOOL_USE FALLBACK] name=%s", c.Name)
						tc := ToolCall{
							ID:   c.Name, // no ID in ClaudeContent; use name as stable key
							Type: "function",
							Function: ToolCallFunction{Name: c.Name, Arguments: ""},
						}
						chunk := ChatCompletionResponse{
							ID:      chatID,
							Object:  "chat.completion.chunk",
							Created: created,
							Model:   model,
							Choices: []ChatCompletionChoice{
								{
									Index: 0,
									Delta: &ChatMessage{
										Role:      "assistant",
										ToolCalls: []ToolCall{tc},
									},
								},
							},
						}
						data, _ := json.Marshal(chunk)
						fmt.Fprintf(w, "data: %s\n\n", data)
						flusher.Flush()
					}
				}
			}
		}

		// Fallback: extract text from assistant events (when stream_event deltas are absent)
		if !streamDeltaSent {
			text := extractTextFromEvent(&event)
			if text != "" {
				log.Printf("[STREAM FALLBACK DELTA] len=%d content=%q", len(text), truncate(text, 200))
				chunk := ChatCompletionResponse{
					ID:      chatID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   model,
					Choices: []ChatCompletionChoice{
						{
							Index:        0,
							Delta:        NewChatMessage("assistant", text),
							FinishReason: nil,
						},
					},
				}
				data, _ := json.Marshal(chunk)
				fmt.Fprintf(w, "data: %s\n\n", data)
				flusher.Flush()
			}
		}

		if event.Type == "result" {
			if event.Usage != nil {
				globalStats.Record(model, event.Usage.InputTokens, event.Usage.OutputTokens)
				log.Printf("Token usage: model=%s input=%d output=%d cache_read=%d cache_create=%d",
					model, event.Usage.InputTokens, event.Usage.OutputTokens,
					event.Usage.CacheReadInputTokens, event.Usage.CacheCreationInputTokens)
			}

			// finish_reason chunk (no usage here, vLLM style)
			finishReason := "stop"
			finishChunk := ChatCompletionResponse{
				ID:      chatID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []ChatCompletionChoice{
					{
						Index:        0,
						Delta:        NewChatMessage("", ""),
						FinishReason: &finishReason,
					},
				},
			}
			data, _ := json.Marshal(finishChunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()

			// Separate usage chunk with empty choices (vLLM/OpenAI style)
			if includeUsage && event.Usage != nil {
				usageChunk := ChatCompletionResponse{
					ID:      chatID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   model,
					Choices: []ChatCompletionChoice{},
					Usage:   buildUsageInfo(event.Usage),
				}
				data, _ = json.Marshal(usageChunk)
				fmt.Fprintf(w, "data: %s\n\n", data)
				flusher.Flush()
			}
		}
	}

	flushAggLog() // flush any remaining aggregated log

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// handleBufferedStreamResponse streams text deltas in real-time while also
// collecting full text to detect tool calls at the end.
// Used when tools are defined in the request.
func handleBufferedStreamResponse(w http.ResponseWriter, r *http.Request, args []string, prompt string, chatID string, created int64, model string, includeUsage bool, workingDir string, envVars map[string]string) {
	lines, ready := startClaudeStream(args, prompt, workingDir, envVars)
	if err := <-ready; err != nil {
		writeOAIError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

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

	// Track text and native tool_use blocks from Claude CLI's stream.
	var fullText strings.Builder
	var filter toolCallFilter
	var streamDeltaSent bool
	var finalUsage *UsageInfo

	// Native tool_use tracking: Claude calls tools via native tool_use (not XML).
	// We intercept content_block_start (tool_use) + content_block_delta (input_json_delta).
	nativeTCs := map[int]*nativeToolCall{}
	var nativeTCOrder []int

	for line := range lines {
		if line == "" {
			continue
		}

		log.Printf("[BUFFERED STREAM RAW] %s", line)

		var event ClaudeEvent
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			log.Printf("Failed to parse claude event: %v", err)
			continue
		}

		// Parse stream_event for text deltas and native tool_use blocks
		if event.Type == "stream_event" && event.Event != nil {
			var streamEvt StreamAPIEvent
			if err := json.Unmarshal(event.Event, &streamEvt); err != nil {
				continue
			}

			switch streamEvt.Type {
			case "content_block_start":
				// Detect tool_use content blocks
				if streamEvt.ContentBlock != nil {
					var block StreamContentBlock
					if err := json.Unmarshal(streamEvt.ContentBlock, &block); err == nil && block.Type == "tool_use" {
						tc := &nativeToolCall{ID: block.ID, Name: block.Name}
						nativeTCs[streamEvt.Index] = tc
						nativeTCOrder = append(nativeTCOrder, streamEvt.Index)
						log.Printf("[NATIVE TOOL_USE START] index=%d name=%s id=%s", streamEvt.Index, block.Name, block.ID)
					}
				}

			case "content_block_delta":
				if streamEvt.Delta != nil {
					var delta StreamTextDelta
					if err := json.Unmarshal(streamEvt.Delta, &delta); err != nil {
						continue
					}
					switch delta.Type {
					case "text_delta":
						if delta.Text != "" {
							streamDeltaSent = true
							fullText.WriteString(delta.Text)
							safeText := filter.Feed(delta.Text)
							if safeText != "" {
								log.Printf("[BUFFERED STREAM DELTA] len=%d content=%q", len(safeText), truncate(safeText, 200))
								chunk := ChatCompletionResponse{
									ID:      chatID,
									Object:  "chat.completion.chunk",
									Created: created,
									Model:   model,
									Choices: []ChatCompletionChoice{
										{
											Index:        0,
											Delta:        NewChatMessage("assistant", safeText),
											FinishReason: nil,
										},
									},
								}
								data, _ := json.Marshal(chunk)
								fmt.Fprintf(w, "data: %s\n\n", data)
								flusher.Flush()
							}
						}
					case "input_json_delta":
						// Accumulate tool call arguments
						if tc, ok := nativeTCs[streamEvt.Index]; ok {
							tc.Args.WriteString(delta.PartialJSON)
						}
					}
				}
			}
			continue
		}

		// Fallback: extract text from assistant events (when stream_event deltas are absent)
		if !streamDeltaSent && event.Type == "assistant" {
			text := extractTextFromEvent(&event)
			if text != "" {
				fullText.WriteString(text)
				safeText := filter.Feed(text)
				if safeText != "" {
					log.Printf("[BUFFERED STREAM FALLBACK] len=%d content=%q", len(safeText), truncate(safeText, 200))
					chunk := ChatCompletionResponse{
						ID:      chatID,
						Object:  "chat.completion.chunk",
						Created: created,
						Model:   model,
						Choices: []ChatCompletionChoice{
							{
								Index:        0,
								Delta:        NewChatMessage("assistant", safeText),
								FinishReason: nil,
							},
						},
					}
					data, _ := json.Marshal(chunk)
					fmt.Fprintf(w, "data: %s\n\n", data)
					flusher.Flush()
				}
			}
		}

		// Collect usage from the final result event
		if event.Type == "result" {
			if event.Result != "" {
				fullText.Reset()
				fullText.WriteString(event.Result)
			}
			if event.Usage != nil {
				finalUsage = buildUsageInfo(event.Usage)
				globalStats.Record(model, event.Usage.InputTokens, event.Usage.OutputTokens)
				log.Printf("Token usage: model=%s input=%d output=%d cache_read=%d cache_create=%d",
					model, event.Usage.InputTokens, event.Usage.OutputTokens,
					event.Usage.CacheReadInputTokens, event.Usage.CacheCreationInputTokens)
			}
		}
	}

	// Flush any remaining buffered text from the filter
	if remaining, _ := filter.Finish(); remaining != "" && !strings.HasPrefix(remaining, "<tool_call") {
		chunk := ChatCompletionResponse{
			ID:      chatID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: []ChatCompletionChoice{
				{
					Index:        0,
					Delta:        NewChatMessage("assistant", remaining),
					FinishReason: nil,
				},
			},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	// Build tool calls: prefer native tool_use, fall back to XML parsing
	var toolCalls []ToolCall
	for _, idx := range nativeTCOrder {
		tc := nativeTCs[idx]
		toolCalls = append(toolCalls, ToolCall{
			ID:   tc.ID,
			Type: "function",
			Function: ToolCallFunction{
				Name:      tc.Name,
				Arguments: tc.Args.String(),
			},
		})
	}
	if len(toolCalls) > 0 {
		log.Printf("Detected %d native tool_use calls", len(toolCalls))
	} else {
		// Fallback: check for <tool_call> XML in text
		collected := fullText.String()
		_, toolCalls = parseToolCalls(collected)
		if len(toolCalls) > 0 {
			log.Printf("Detected %d XML tool calls (fallback)", len(toolCalls))
		}
	}

	if len(toolCalls) > 0 {
		// Send tool calls in OpenAI format
		for _, tc := range toolCalls {
			chunk := ChatCompletionResponse{
				ID:      chatID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []ChatCompletionChoice{
					{
						Index: 0,
						Delta: &ChatMessage{
							Role:      "assistant",
							ToolCalls: []ToolCall{tc},
						},
					},
				},
			}
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}

		finishReason := "tool_calls"
		finishChunk := ChatCompletionResponse{
			ID:      chatID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: []ChatCompletionChoice{
				{Index: 0, Delta: NewChatMessage("", ""), FinishReason: &finishReason},
			},
		}
		data, _ := json.Marshal(finishChunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	} else {
		finishReason := "stop"
		finishChunk := ChatCompletionResponse{
			ID:      chatID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: []ChatCompletionChoice{
				{Index: 0, Delta: NewChatMessage("", ""), FinishReason: &finishReason},
			},
		}
		data, _ := json.Marshal(finishChunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	// Separate usage chunk with empty choices (vLLM/OpenAI style)
	if includeUsage && finalUsage != nil {
		usageChunk := ChatCompletionResponse{
			ID:      chatID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: []ChatCompletionChoice{},
			Usage:   finalUsage,
		}
		data, _ := json.Marshal(usageChunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func handleNonStreamResponse(w http.ResponseWriter, r *http.Request, args []string, prompt string, chatID string, created int64, model string, hasTools bool, workingDir string, envVars map[string]string) {
	events, lastText, result, usage, err := runClaude(args, prompt, workingDir, envVars)
	if err != nil {
		writeOAIError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	fullText := lastText
	if result != "" {
		fullText = result
	}

	// Record token stats
	if usage != nil {
		globalStats.Record(model, usage.PromptTokens, usage.CompletionTokens)
		cached := 0
		if usage.PromptTokensDetails != nil {
			cached = usage.PromptTokensDetails.CachedTokens
		}
		log.Printf("Token usage: model=%s prompt=%d output=%d total=%d cached=%d",
			model, usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens, cached)
	}

	var (
		finishReason string
		msg          *ChatMessage
	)

	if hasTools {
		// Extract native tool_use from events
		var toolCalls []ToolCall
		toolCalls = extractNativeToolCalls(events)
		if len(toolCalls) > 0 {
			finishReason = "tool_calls"
			msg = NewChatMessage("assistant", fullText)
			msg.ToolCalls = toolCalls
			log.Printf("Non-stream: detected %d native tool_use calls", len(toolCalls))
		} else {
			// Fallback: parse <tool_call> XML from text
			cleanText, xmlCalls := parseToolCalls(fullText)
			if len(xmlCalls) > 0 {
				finishReason = "tool_calls"
				msg = NewChatMessage("assistant", cleanText)
				msg.ToolCalls = xmlCalls
				log.Printf("Non-stream: detected %d XML tool calls (fallback)", len(xmlCalls))
			} else {
				finishReason = "stop"
				msg = NewChatMessage("assistant", fullText)
			}
		}
	} else {
		finishReason = "stop"
		msg = NewChatMessage("assistant", fullText)
	}

	if msg == nil || (msg.ContentString() == "" && len(msg.ToolCalls) == 0) {
		writeOAIError(w, http.StatusInternalServerError, "server_error", "Empty response from Claude")
		return
	}

	resp := ChatCompletionResponse{
		ID:      chatID,
		Object:  "chat.completion",
		Created: created,
		Model:   model,
		Choices: []ChatCompletionChoice{
			{
				Index:        0,
				Message:      msg,
				FinishReason: &finishReason,
			},
		},
		Usage: usage,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func modelsHandler(w http.ResponseWriter, r *http.Request) {
	setOAICORSHeaders(w, r)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	resp := ModelListResponse{
		Object: "list",
		Data:   availableModels,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func oaiHealthHandler(w http.ResponseWriter, r *http.Request) {
	cmd := exec.Command("claude", "--version")
	if err := cmd.Run(); err != nil {
		http.Error(w, fmt.Sprintf("Claude CLI not available: %v", err), http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "healthy",
		"claude": "available",
	})
}

func statsHandler(w http.ResponseWriter, r *http.Request) {
	setOAICORSHeaders(w, r)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(globalStats.Snapshot())
}

func writeOAIError(w http.ResponseWriter, statusCode int, errType string, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    errType,
			"code":    statusCode,
		},
	})
}

func main() {
	logFile, err := os.OpenFile("claude_openai_api.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		log.Fatalf("Failed to open log file: %v", err)
	}
	defer logFile.Close()

	multiWriter := io.MultiWriter(os.Stdout, logFile)
	log.SetOutput(multiWriter)
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds | log.Lshortfile)

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", chatCompletionsHandler)
	mux.HandleFunc("/v1/models", modelsHandler)
	mux.HandleFunc("/v1/stats", statsHandler)
	mux.HandleFunc("/health", oaiHealthHandler)

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.Printf("HTTP %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)
		mux.ServeHTTP(w, r)
	})

	port := ":50009"
	log.Printf("Starting Claude OpenAI-compatible API server on %s", port)
	log.Printf("Endpoints:")
	log.Printf("  POST /v1/chat/completions  (OpenAI-compatible chat completions, with tool calling)")
	log.Printf("  GET  /v1/models            (Model list)")
	log.Printf("  GET  /v1/stats             (Token usage statistics)")
	log.Printf("  GET  /health               (Health check)")

	if err := http.ListenAndServe(port, handler); err != nil {
		log.Fatal(err)
	}
}
