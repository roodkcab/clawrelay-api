package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"clawrelay-api/pkg/openai"
	"clawrelay-api/pkg/proc"
)

// firstLineTimeout bounds how long a freshly-started claude process may stay
// completely silent on stdout. A CLI that hangs before its first stream-json
// line (bad binary on PATH, wedged auto-updater, dead proxy) used to stall the
// request until the client's own read timeout killed the connection; the
// watchdog kills it instead so the failure is fast and visible. Must stay
// below the upstream client's idle timeout (wuji-tools: sock_read=120s).
// Overridable via RELAY_FIRST_LINE_TIMEOUT_SECS; package var so tests can
// shorten it.
var firstLineTimeout = 90 * time.Second

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

	maxTurns := 20
	if req.MaxTurns != nil {
		maxTurns = *req.MaxTurns
	}
	args = append(args, "--max-turns", openai.FmtInt(maxTurns))

	if req.Effort != "" {
		args = append(args, "--effort", req.Effort)
	}
	if req.Settings != "" {
		args = append(args, "--settings", req.Settings)
	}
	if req.SessionID != "" {
		args = append(args, "--resume", req.SessionID)
	}
	return args, prompt
}

// launchClaude starts a `claude` subprocess and returns its stdout-line
// channel along with a stderr-completion signal. The channel closes after
// cmd.Wait. sessErrCh receives true if the "No conversation found" stderr
// marker was seen — used to drive the resume-retry path. watchdogTimeout is
// passed by value (callers snapshot firstLineTimeout synchronously) so the
// watchdog goroutine never reads the package var concurrently.
func launchClaude(args []string, prompt, workingDir string, envVars map[string]string, watchdogTimeout time.Duration) (*exec.Cmd, <-chan string, <-chan bool, error) {
	cmd := exec.Command("claude", args...)
	// Own process group so KillGroup can reap the whole tree (node wrapper +
	// native claude binary) instead of orphaning the native child.
	proc.SetNewProcessGroup(cmd)
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
		// Same oversize-line guard as stdout: a single >64KB stderr line (node
		// dumping a response body) would end the scan early, losing the
		// "No conversation found" marker that gates the resume retry, and
		// leaving the pipe undrained — a child blocked writing stderr never
		// exits, so stdout never EOFs and the sniff stalls.
		s.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
		found := false
		for s.Scan() {
			line := s.Text()
			log.Printf("Claude stderr: %s", line)
			if strings.Contains(line, "No conversation found with session ID") {
				found = true
			}
		}
		if err := s.Err(); err != nil {
			log.Printf("Claude stderr scanner error: %v", err)
			io.Copy(io.Discard, stderrPipe) //nolint:errcheck // backstop drain
		}
		sessErrCh <- found
	}()

	// First-line watchdog: a CLI that never prints a single stdout line (bad
	// binary hijacking PATH, wedged auto-updater, dead proxy — the claude04
	// incident) used to hang the request until the upstream client's own read
	// timeout killed the connection. Kill it ourselves so the failure is fast
	// and carries a diagnosable error. Disarmed by the first scanned line, or
	// by process exit (zero-output crash needs no kill).
	firstLineSeen := make(chan struct{})
	var disarmOnce sync.Once
	disarm := func() { disarmOnce.Do(func() { close(firstLineSeen) }) }
	go func() {
		timer := time.NewTimer(watchdogTimeout)
		defer timer.Stop()
		select {
		case <-firstLineSeen:
		case <-timer.C:
			log.Printf("first-line watchdog: claude silent on stdout for %s, killing process group (args=%q)", watchdogTimeout, args)
			proc.KillGroup(cmd)
		}
	}()

	lines := make(chan string, 128)
	go func() {
		defer close(lines)
		s := bufio.NewScanner(stdoutPipe)
		// 8MB cap, same as channel.go's drainStdout: a single stream-json line
		// (e.g. a result event with large tool output) can exceed 1MB, and
		// bufio.ErrTooLong would silently truncate the stream mid-turn.
		s.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
		for s.Scan() {
			disarm()
			lines <- s.Text()
		}
		// stdout EOF: the process has closed stdout, the watchdog's job is
		// over. Disarm BEFORE cmd.Wait() reaps the child — a timer firing in
		// the reap window would KillGroup a possibly-recycled PID (the same
		// hazard proc.WatchDisconnect defends against).
		disarm()
		if err := s.Err(); err != nil {
			// bufio.ErrTooLong (line > 8MB) or a read error: the stream ends
			// early and the caller may never see a result event (see
			// EmitFinishIfNoResult for the downstream mitigation).
			log.Printf("Claude stdout scanner error (stream truncated): %v", err)
		}
		stderrDone.Wait()
		if err := cmd.Wait(); err != nil {
			log.Printf("Claude command error: %v", err)
		}
	}()

	return cmd, lines, sessErrCh, nil
}

// lastOutputNote summarizes the trailing non-empty stdout line (node stack
// traces, wrapper banners printed to stdout) for zero-event error messages —
// otherwise the real failure text would be silently discarded and the error
// would claim "no output" while output existed.
func lastOutputNote(buffered []string) string {
	for i := len(buffered) - 1; i >= 0; i-- {
		line := strings.TrimSpace(buffered[i])
		if line == "" {
			continue
		}
		return fmt.Sprintf("; last output: %s", openai.Truncate(line, 200))
	}
	return ""
}

// procHandle publishes the currently-running claude process to the HTTP
// handler and coordinates cancellation with the producer goroutine, which may
// still be launching (or, on the resume path, re-launching) processes when the
// client disconnects. All access goes through the mutex: an unsynchronized
// **exec.Cmd would be both a data race (the handler's pre-ready disconnect
// read races the producer's assignment) and a TOCTOU — a --session-id retry
// launched just after the handler aborted would become an unkillable orphan
// running the user's full prompt to completion.
type procHandle struct {
	mu      sync.Mutex
	cmd     *exec.Cmd
	aborted bool
}

// publish records a freshly-launched process. Returns false when the handler
// has already aborted — the caller must kill the newcomer and stop instead of
// proceeding (this closes the launch-after-abort orphan window).
func (h *procHandle) publish(cmd *exec.Cmd) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.aborted {
		return false
	}
	h.cmd = cmd
	return true
}

// abort marks the request as cancelled and returns the process to kill. A nil
// return means the producer is between launches; its next publish will return
// false and it kills the newcomer itself.
func (h *procHandle) abort() *exec.Cmd {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.aborted = true
	return h.cmd
}

// startClaudeStream wraps launchClaude with auto-retry on resume failure.
// If --resume produced no real content because the session is actually
// missing (stderr marker), it retries the whole command with --session-id.
//
// ready signals when claude produced its first usable output (or definitively
// failed); the handler sends SSE headers before that and uses the returned
// procHandle to kill the current process on disconnect.
func startClaudeStream(args []string, prompt, workingDir string, envVars map[string]string) (<-chan string, <-chan error, *procHandle) {
	lines := make(chan string, 128)
	ready := make(chan error, 1)
	handle := &procHandle{}

	// Snapshot synchronously: the goroutine below (and the watchdog goroutines
	// it spawns) must not read the package var concurrently with anyone
	// mutating it (tests).
	watchdogTimeout := firstLineTimeout

	go func() {
		defer close(lines)

		cmd, innerLines, sessErrCh, err := launchClaude(args, prompt, workingDir, envVars, watchdogTimeout)
		if err != nil {
			ready <- err
			return
		}
		if !handle.publish(cmd) {
			proc.KillGroup(cmd)
			go func() { <-sessErrCh }()
			proc.DrainLines(innerLines)
			ready <- fmt.Errorf("request cancelled during claude startup")
			return
		}

		if hasArg(args, "--resume") {
			// Sniff the resume run. Everything is buffered — never discarded:
			// an early result event carries the error text (API error / rate
			// limit / auth failure) and the turn's usage, and dropping it used
			// to bypass the translator's error transparency and answer with a
			// delayed empty 200 stream ("未生成文本回复" downstream).
			var buffered []string
			gotContent := false
			sawResult := false

			// Sniff deadline: the first-line watchdog disarms on the very
			// first stdout line, but a resume run prints its init event
			// locally before any API call — a CLI that hangs after that (dead
			// proxy, wedged retry loop) would stall the sniff forever, and the
			// pre-ready keepalives now hide the stall from the client's own
			// idle timeout. Bound the whole sniff with the same budget.
			sniffTimer := time.NewTimer(watchdogTimeout)
		sniff:
			for {
				var line string
				var lok bool
				select {
				case line, lok = <-innerLines:
				case <-sniffTimer.C:
					log.Printf("resume sniff watchdog: no content/result within %s, killing process group (args=%q)", watchdogTimeout, args)
					proc.KillGroup(cmd)
					continue // the kill closes stdout shortly; drain to EOF
				}
				if !lok {
					break sniff
				}
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
					// Result before any content: the run is over. Buffer the
					// remainder to EOF so nothing is lost.
					sawResult = true
					for line := range innerLines {
						buffered = append(buffered, line)
					}
					break sniff
				}
				gotContent = true
				break sniff
			}
			sniffTimer.Stop()

			if gotContent {
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

			// No content and innerLines closed → the process has exited, so
			// the stderr verdict is available. Only an actually-missing session
			// justifies the --session-id retry: retrying against an existing
			// session re-runs the turn at best and produces a second empty
			// stream at worst.
			sessionMissing := <-sessErrCh
			if !sessionMissing {
				if sawResult {
					log.Printf("Resume run ended with a result and no content (session exists); forwarding %d buffered event(s) instead of retrying", len(buffered))
					ready <- nil
					for _, line := range buffered {
						lines <- line
					}
					return
				}
				ready <- fmt.Errorf("claude exited without a usable event%s (session exists; crash or first-line watchdog kill, check relay logs)", lastOutputNote(buffered))
				return
			}

			log.Printf("Resume session not found, retrying with --session-id")
			retryArgs := replaceArg(args, "--resume", "--session-id")
			cmd, innerLines, sessErrCh, err = launchClaude(retryArgs, prompt, workingDir, envVars, watchdogTimeout)
			if err != nil {
				ready <- err
				return
			}
			if !handle.publish(cmd) {
				proc.KillGroup(cmd)
				go func() { <-sessErrCh }()
				proc.DrainLines(innerLines)
				ready <- fmt.Errorf("request cancelled during claude resume retry")
				return
			}
			go func() { <-sessErrCh }()

			firstLine, ok := <-innerLines
			if !ok {
				// The retry also died silently. Surfacing the failure beats the
				// old behavior (ready<-nil + empty 200 stream).
				ready <- fmt.Errorf("claude --session-id retry exited without producing any output")
				return
			}
			ready <- nil
			lines <- firstLine
			for line := range innerLines {
				lines <- line
			}
			return
		}

		firstLine, ok := <-innerLines
		if !ok {
			go func() { <-sessErrCh }()
			ready <- fmt.Errorf("claude exited without producing any output (crash or first-line watchdog kill, check relay logs)")
			return
		}
		go func() { <-sessErrCh }()
		ready <- nil
		lines <- firstLine
		for line := range innerLines {
			lines <- line
		}
	}()

	return lines, ready, handle
}

// runClaude is the non-streaming counterpart: collects all events, the final
// result text, and usage. Auto-retries with --session-id if --resume yields
// no content.
func runClaude(args []string, prompt, workingDir string, envVars map[string]string) (events []claudeEvent, lastText, result string, usage *openai.UsageInfo, rawUsage *claudeUsage, costUSD float64, err error) {
	_, lines, sessErrCh, err := launchClaude(args, prompt, workingDir, envVars, firstLineTimeout)
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

	// Retry with --session-id only when the session is actually missing (stderr
	// marker) — a blind retry on any empty output re-runs the turn against an
	// existing session and hides the original failure.
	sessionMissing := <-sessErrCh
	if hasArg(args, "--resume") && sessionMissing && lastText == "" && result == "" {
		log.Printf("Resume session not found, retrying with --session-id")
		retryArgs := replaceArg(args, "--resume", "--session-id")
		return runClaude(retryArgs, prompt, workingDir, envVars)
	}
	return
}
