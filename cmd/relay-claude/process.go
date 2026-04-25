package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"sync"

	"clawrelay-api/pkg/openai"
)

// cleanEnv returns the current environment with CLAUDECODE removed (the
// claude CLI uses this to detect being run inside another Claude session;
// stripping it lets us run the CLI as a normal subprocess), plus any extra
// KEY=VALUE pairs from the request.
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

func hasArg(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

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

// buildClaudeArgs assembles `claude` CLI flags from the OpenAI-shaped request.
// stdinData is the prompt body that should be piped on stdin.
func buildClaudeArgs(req *openai.ChatCompletionRequest, model, prompt, systemPrompt string) (args []string, stdinData string) {
	if systemPrompt != "" {
		args = append(args, "--append-system-prompt", systemPrompt)
	}
	if req.SystemPromptFile != "" {
		args = append(args, "--append-system-prompt-file", req.SystemPromptFile)
	}
	args = append(args, "--model", model)
	args = append(args, "--verbose")
	args = append(args, "--output-format", "stream-json")
	args = append(args, "--include-partial-messages")

	permMode := "bypassPermissions"
	if req.PermissionMode != "" {
		permMode = req.PermissionMode
	}
	args = append(args, "--permission-mode", permMode)

	if req.AllowedTools != "" {
		args = append(args, "--allowedTools", req.AllowedTools)
	}
	for _, dir := range req.AddDirs {
		if dir != "" {
			args = append(args, "--add-dir", dir)
		}
	}

	maxTurns := 200
	if req.MaxTurns != nil {
		maxTurns = *req.MaxTurns
	}
	args = append(args, "--max-turns", openai.FmtInt(maxTurns))

	if req.Effort != "" {
		args = append(args, "--effort", req.Effort)
	}
	if req.SessionID != "" {
		args = append(args, "--resume", req.SessionID)
	}
	return args, prompt
}

// launchClaude starts a `claude` subprocess and returns its stdout-line
// channel along with a stderr-completion signal. The channel closes after
// cmd.Wait. sessErrCh receives true if the "No conversation found" stderr
// marker was seen — used to drive the resume-retry path.
func launchClaude(args []string, prompt, workingDir string, envVars map[string]string) (*exec.Cmd, <-chan string, <-chan bool, error) {
	cmd := exec.Command("claude", args...)
	cmd.Env = cleanEnv(envVars)
	if workingDir != "" {
		cmd.Dir = workingDir
	}
	cmd.Stdin = strings.NewReader(prompt)

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to create stdout pipe: %v", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to create stderr pipe: %v", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, nil, nil, fmt.Errorf("failed to start claude: %v", err)
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

	return cmd, lines, sessErrCh, nil
}

// startClaudeStream wraps launchClaude with auto-retry on resume failure.
// If --resume produced no real content (only init/system/result events), it
// retries the whole command with --session-id instead.
//
// ready signals when it is safe to write HTTP response headers. cmdPtr lets
// the caller kill the process after ready fires.
func startClaudeStream(args []string, prompt, workingDir string, envVars map[string]string) (<-chan string, <-chan error, **exec.Cmd) {
	lines := make(chan string, 128)
	ready := make(chan error, 1)
	var cmdHolder *exec.Cmd
	cmdPtr := &cmdHolder

	go func() {
		defer close(lines)

		cmd, innerLines, sessErrCh, err := launchClaude(args, prompt, workingDir, envVars)
		if err != nil {
			ready <- err
			return
		}
		cmdHolder = cmd

		if hasArg(args, "--resume") {
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
					shouldRetry = true
					for range innerLines {
					}
					break
				}
				gotContent = true
				break
			}

			if !gotContent && !shouldRetry {
				shouldRetry = true
			}

			if shouldRetry {
				<-sessErrCh
				log.Printf("Resume produced no content, retrying with --session-id")
				retryArgs := replaceArg(args, "--resume", "--session-id")
				cmd, innerLines, sessErrCh, err = launchClaude(retryArgs, prompt, workingDir, envVars)
				if err != nil {
					ready <- err
					return
				}
				cmdHolder = cmd
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

	return lines, ready, cmdPtr
}

// runClaude is the non-streaming counterpart: collects all events, the final
// result text, and usage. Auto-retries with --session-id if --resume yields
// no content.
func runClaude(args []string, prompt, workingDir string, envVars map[string]string) (events []claudeEvent, lastText, result string, usage *openai.UsageInfo, rawUsage *claudeUsage, costUSD float64, err error) {
	_, lines, sessErrCh, err := launchClaude(args, prompt, workingDir, envVars)
	if err != nil {
		return
	}

	for line := range lines {
		if line == "" {
			continue
		}
		log.Printf("[CLAUDE RAW] %s", line)
		var event claudeEvent
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
			if eu := effectiveUsage(&event); eu != nil {
				usage = openai.BuildUsageInfo(eu.InputTokens, eu.OutputTokens, eu.CacheReadInputTokens, eu.CacheCreationInputTokens)
				rawUsage = eu
				costUSD = event.TotalCostUSD
			}
		}
	}

	<-sessErrCh
	if hasArg(args, "--resume") && lastText == "" && result == "" {
		log.Printf("Resume produced no content, retrying with --session-id")
		retryArgs := replaceArg(args, "--resume", "--session-id")
		return runClaude(retryArgs, prompt, workingDir, envVars)
	}
	return
}
