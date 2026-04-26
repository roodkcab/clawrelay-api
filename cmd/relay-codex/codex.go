package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"sync"

	"clawrelay-api/pkg/attachments"
	"clawrelay-api/pkg/openai"
)

// cleanEnv mirrors the Claude relay's helper but strips CODEX_* job-control
// vars so a relay running inside another codex session doesn't confuse the
// child.
func cleanEnv(extra map[string]string) []string {
	var env []string
	for _, e := range os.Environ() {
		// Filter codex-internal vars that would otherwise inherit and
		// potentially conflict with the child's own session bookkeeping.
		if strings.HasPrefix(e, "CODEX_RUN_ID=") || strings.HasPrefix(e, "CODEX_SESSION_ID=") {
			continue
		}
		env = append(env, e)
	}
	for k, v := range extra {
		env = append(env, k+"="+v)
	}
	return env
}

// codexInput is the resolved set of arguments to feed `codex exec`, post
// session/thread reconciliation. It captures whether this turn is a fresh
// session or a resume, and whether multipart history needs flattening.
type codexInput struct {
	Args      []string  // CLI args after `codex`
	Stdin     string    // body to pipe on stdin
	IsResume  bool      // true = `codex exec resume <thread_id>` pattern
	ImagePath []string  // images to attach via -i (collected from latest user message)
}

// buildCodexInput converts the high-level OpenAI request into a fully-resolved
// codex CLI invocation. Key optimization: when we already have a thread_id
// for this session, we send only the *latest* user message instead of the
// full history — codex retains conversational state server-side.
func buildCodexInput(req *openai.ChatCompletionRequest, model string, threadID, sessionDir string) codexInput {
	out := codexInput{}

	// `exec` is the non-interactive entrypoint. Resume routes through the
	// `resume` subcommand which takes an explicit thread id.
	out.Args = append(out.Args, "exec")
	if threadID != "" {
		out.Args = append(out.Args, "resume", threadID)
	}

	// JSONL output is the only protocol we know how to parse. --skip-git-repo-check
	// keeps codex from refusing to run when the working dir isn't a repo.
	out.Args = append(out.Args, "--json", "--skip-git-repo-check")

	if model != "" {
		bare := stripPrefix(model)
		out.Args = append(out.Args, "--model", bare)
	}

	// Sandbox mapping. `codex exec resume` only accepts the convenience
	// flags (--full-auto, --dangerously-bypass-approvals-and-sandbox); -s
	// and --add-dir are rejected. The original session's sandbox setting is
	// preserved by codex itself across resumes, so on resume we only honor
	// the convenience flags and silently drop fine-grained overrides.
	resuming := threadID != ""
	switch req.PermissionMode {
	case "":
		// Default to bypass — matches relay-claude's default permission_mode.
		out.Args = append(out.Args, "--dangerously-bypass-approvals-and-sandbox")
	case "bypassPermissions", "bypass":
		out.Args = append(out.Args, "--dangerously-bypass-approvals-and-sandbox")
	case "full-auto", "full_auto", "auto":
		out.Args = append(out.Args, "--full-auto")
	case "read-only", "readonly":
		if !resuming {
			out.Args = append(out.Args, "-s", "read-only")
		}
	case "workspace-write", "workspace_write":
		if !resuming {
			out.Args = append(out.Args, "-s", "workspace-write")
		}
	default:
		if !resuming {
			out.Args = append(out.Args, "-s", req.PermissionMode)
		}
	}

	if !resuming {
		for _, dir := range req.AddDirs {
			if dir != "" {
				out.Args = append(out.Args, "--add-dir", dir)
			}
		}
	}

	// Reasoning effort via codex's TOML override mechanism. The exact key
	// depends on the codex version; this path matches gpt-5.x configs.
	if req.Effort != "" {
		out.Args = append(out.Args, "-c", "model_reasoning_effort="+req.Effort)
	}

	// On a fresh session, set instructions via -c so codex treats them as
	// system-level (rather than appending them visibly to the user prompt).
	// When resuming, the original session's instructions are already retained
	// server-side, so re-sending them would just stuff context unnecessarily.
	if !out.isResume() {
		// Extract first system message; we'll set it as instructions.
		for _, msg := range req.Messages {
			if msg.Role == "system" {
				if sp := msg.ContentString(); sp != "" {
					// codex parses -c values as TOML, so we must stringify.
					out.Args = append(out.Args, "-c", "instructions="+tomlString(sp))
				}
				break
			}
		}
	}

	// Compose the prompt body and gather image attachments from the user
	// messages. On resume we only care about the *latest* user turn; on a
	// fresh start we need the whole history flattened so codex sees it.
	if threadID != "" {
		out.IsResume = true
		out.Stdin, out.ImagePath = lastUserMessage(req.Messages, sessionDir)
	} else {
		out.Stdin, out.ImagePath = flattenForFreshSession(req.Messages, sessionDir)
	}

	// Image attachments via codex's native `-i FILE` mechanism. This is
	// strictly better than embedding `[Image: /path]` in prompt text because
	// codex routes the file through its actual multimodal pipeline.
	for _, p := range out.ImagePath {
		out.Args = append(out.Args, "-i", p)
	}

	// `-` makes codex read prompt from stdin (cleaner than putting it as a
	// CLI arg — quoting issues, length limits).
	out.Args = append(out.Args, "-")

	return out
}

func (c codexInput) isResume() bool { return c.IsResume }

// stripPrefix turns `codex/gpt-5.4` into `gpt-5.4`.
func stripPrefix(model string) string {
	if idx := strings.LastIndex(model, "/"); idx >= 0 {
		return model[idx+1:]
	}
	return model
}

// tomlString quotes a value for safe inclusion in a `-c key=value` override.
// Codex parses the value as TOML, so a bare string with newlines or quotes
// would mis-parse. We wrap in TOML's basic-string form with escaping.
func tomlString(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '\\':
			b.WriteString(`\\`)
		case '"':
			b.WriteString(`\"`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if r < 0x20 {
				b.WriteString(fmt.Sprintf(`\u%04X`, r))
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
	return b.String()
}

// lastUserMessage extracts the most recent user message's text and image
// attachments. Used when we're resuming an existing thread — codex remembers
// the rest, we just hand it the new user turn.
func lastUserMessage(messages []openai.ChatMessage, sessionDir string) (text string, images []string) {
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		if msg.Role != "user" {
			continue
		}
		text = msg.ContentString()
		files := attachments.ExtractAndSave(msg.Content, sessionDir, "codex-img", "codex-file")
		for _, f := range files {
			if f.IsImage {
				images = append(images, f.Path)
			} else {
				// Non-image files: append as a path reference in the text
				// since codex doesn't have a generic file-attach flag.
				if text != "" {
					text += "\n"
				}
				text += fmt.Sprintf("[File: %s]", f.Path)
			}
		}
		return
	}
	return
}

// flattenForFreshSession converts the OpenAI message array into a single
// prompt for the first turn of a brand-new codex session. We omit the system
// message (already passed via -c instructions) and use a clean dialogue
// format that matches GPT's native expectation better than the Claude-style
// "Human:/Assistant:" markers.
func flattenForFreshSession(messages []openai.ChatMessage, sessionDir string) (prompt string, images []string) {
	var parts []string
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			// Already passed via -c instructions; skip.
			continue
		case "user":
			text := msg.ContentString()
			files := attachments.ExtractAndSave(msg.Content, sessionDir, "codex-img", "codex-file")
			for _, f := range files {
				if f.IsImage {
					images = append(images, f.Path)
				} else {
					if text != "" {
						text += "\n"
					}
					text += fmt.Sprintf("[File: %s]", f.Path)
				}
			}
			parts = append(parts, "User: "+text)
		case "assistant":
			text := msg.ContentString()
			parts = append(parts, "Assistant: "+text)
		case "tool":
			name := msg.Name
			if name == "" {
				name = "tool"
			}
			parts = append(parts, fmt.Sprintf("Tool result for %s (call_id: %s):\n%s", name, msg.ToolCallID, msg.ContentString()))
		default:
			parts = append(parts, fmt.Sprintf("%s: %s", msg.Role, msg.ContentString()))
		}
	}
	prompt = strings.Join(parts, "\n\n")
	return
}

// launchCodex starts a `codex` subprocess and returns line channels. lines
// carries stdout JSONL events; the channel closes after Wait. envExtra is
// merged into the inherited environment minus codex bookkeeping vars.
func launchCodex(input codexInput, workingDir string, envExtra map[string]string) (*exec.Cmd, <-chan string, error) {
	cmd := exec.Command("codex", input.Args...)
	cmd.Env = cleanEnv(envExtra)
	if workingDir != "" {
		cmd.Dir = workingDir
	}
	cmd.Stdin = strings.NewReader(input.Stdin)

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create stdout pipe: %v", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create stderr pipe: %v", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, nil, fmt.Errorf("failed to start codex: %v", err)
	}

	var stderrDone sync.WaitGroup
	stderrDone.Add(1)
	go func() {
		defer stderrDone.Done()
		s := bufio.NewScanner(stderrPipe)
		for s.Scan() {
			log.Printf("codex stderr: %s", s.Text())
		}
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
			log.Printf("codex command error: %v", err)
		}
	}()

	return cmd, lines, nil
}
