package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// writeFakeClaude installs an executable `claude` stub into a fresh temp dir
// and prepends that dir to PATH, so launchClaude's exec.Command("claude", ...)
// resolves to the stub. The script sees CLI args as "$@"/"$*" and can append
// to $ARGSLOG (passed via envVars) to record invocations.
func writeFakeClaude(t *testing.T, script string) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "claude")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
}

func newArgsLog(t *testing.T) (logPath string, env map[string]string, count func() int) {
	t.Helper()
	logPath = filepath.Join(t.TempDir(), "args.log")
	env = map[string]string{"ARGSLOG": logPath}
	count = func() int {
		data, err := os.ReadFile(logPath)
		if err != nil {
			return 0
		}
		return len(strings.Split(strings.TrimSpace(string(data)), "\n"))
	}
	return
}

func waitReadyErr(t *testing.T, ready <-chan error) error {
	t.Helper()
	select {
	case err := <-ready:
		return err
	case <-time.After(10 * time.Second):
		t.Fatal("ready never fired within 10s")
		return nil
	}
}

func collectAllLines(t *testing.T, lines <-chan string) []string {
	t.Helper()
	var out []string
	deadline := time.After(10 * time.Second)
	for {
		select {
		case l, ok := <-lines:
			if !ok {
				return out
			}
			out = append(out, l)
		case <-deadline:
			t.Fatalf("timed out draining lines; got so far: %v", out)
		}
	}
}

const errorResultLine = `{"type":"result","subtype":"error_during_execution","is_error":true,"result":"API Error: overloaded"}`

// A --resume run whose first (and only) non-init event is an error result and
// whose stderr does NOT contain the "No conversation found" marker means the
// session exists and claude genuinely failed (API error / rate limit / auth).
// The old sniff discarded that entire run (result text AND usage) and blindly
// re-ran with --session-id, which typically produced an empty 200 stream. The
// error result must instead be forwarded downstream, with no retry.
func TestResumeForwardsErrorResultWithoutRetry(t *testing.T) {
	writeFakeClaude(t, `echo "$*" >> "$ARGSLOG"
echo '{"type":"system","subtype":"init"}'
echo '`+errorResultLine+`'
`)
	_, env, count := newArgsLog(t)

	lines, ready, _ := startClaudeStream([]string{"--resume", "sess-x"}, "hi", "", env)
	if err := waitReadyErr(t, ready); err != nil {
		t.Fatalf("ready returned error, want nil: %v", err)
	}
	got := collectAllLines(t, lines)

	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, `"result":"API Error: overloaded"`) {
		t.Errorf("error result was not forwarded downstream; got lines: %v", got)
	}
	if n := count(); n != 1 {
		t.Errorf("claude invoked %d times, want 1 (no blind --session-id retry)", n)
	}
}

// The legitimate retry case: stderr carries the "No conversation found with
// session ID" marker, so the session file really is missing and re-running
// with --session-id (recreate the session) is correct.
func TestResumeRetriesWhenSessionMissing(t *testing.T) {
	writeFakeClaude(t, `echo "$*" >> "$ARGSLOG"
case "$*" in
*--session-id*)
  echo '{"type":"system","subtype":"init"}'
  echo '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}}'
  echo '{"type":"result","subtype":"success","result":"hello"}'
  ;;
*--resume*)
  echo "No conversation found with session ID: sess-x" >&2
  exit 1
  ;;
esac
`)
	_, env, count := newArgsLog(t)

	lines, ready, _ := startClaudeStream([]string{"--resume", "sess-x"}, "hi", "", env)
	if err := waitReadyErr(t, ready); err != nil {
		t.Fatalf("ready returned error, want nil: %v", err)
	}
	got := collectAllLines(t, lines)

	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, `"text":"hello"`) {
		t.Errorf("retry output was not forwarded; got lines: %v", got)
	}
	if n := count(); n != 2 {
		t.Errorf("claude invoked %d times, want 2 (resume + --session-id retry)", n)
	}
}

// Zero-output exit on --resume with the session present (silent crash, or the
// first-line watchdog killed a hung CLI): retrying with --session-id used to
// produce a delayed empty 200 stream — the "AI 已完成处理，但未生成文本回复"
// signature. It must surface as an error instead, with no retry.
func TestResumeZeroOutputSessionExistsErrors(t *testing.T) {
	writeFakeClaude(t, `echo "$*" >> "$ARGSLOG"
exit 0
`)
	_, env, count := newArgsLog(t)

	lines, ready, _ := startClaudeStream([]string{"--resume", "sess-x"}, "hi", "", env)
	if err := waitReadyErr(t, ready); err == nil {
		t.Error("ready returned nil, want error for zero-output resume run")
	}
	collectAllLines(t, lines)
	if n := count(); n != 1 {
		t.Errorf("claude invoked %d times, want 1 (no retry when session exists)", n)
	}
}

// Session really missing, but the --session-id retry ALSO exits with zero
// output: must be an error, never an empty 200 stream.
func TestResumeRetryZeroOutputErrors(t *testing.T) {
	writeFakeClaude(t, `echo "$*" >> "$ARGSLOG"
case "$*" in
*--session-id*)
  exit 0
  ;;
*--resume*)
  echo "No conversation found with session ID: sess-x" >&2
  exit 1
  ;;
esac
`)
	_, env, count := newArgsLog(t)

	lines, ready, _ := startClaudeStream([]string{"--resume", "sess-x"}, "hi", "", env)
	if err := waitReadyErr(t, ready); err == nil {
		t.Error("ready returned nil, want error for zero-output retry run")
	}
	collectAllLines(t, lines)
	if n := count(); n != 2 {
		t.Errorf("claude invoked %d times, want 2", n)
	}
}

// Non-resume path: a zero-output exit used to fire ready<-nil and close the
// stream, i.e. an empty 200. Must be an error.
func TestNonResumeZeroOutputErrors(t *testing.T) {
	writeFakeClaude(t, `exit 0`)

	lines, ready, _ := startClaudeStream([]string{"--model", "m"}, "hi", "", nil)
	if err := waitReadyErr(t, ready); err == nil {
		t.Error("ready returned nil, want error for zero-output run")
	}
	collectAllLines(t, lines)
}

// A claude that hangs forever without a single stdout line (claude04-style
// bad-binary hang) must be killed by the first-line watchdog instead of
// stalling the request until the client's own timeout.
func TestFirstLineWatchdogKillsSilentHang(t *testing.T) {
	writeFakeClaude(t, `sleep 30`)
	old := firstLineTimeout
	firstLineTimeout = 500 * time.Millisecond
	defer func() { firstLineTimeout = old }()

	start := time.Now()
	lines, ready, _ := startClaudeStream([]string{"--model", "m"}, "hi", "", nil)
	if err := waitReadyErr(t, ready); err == nil {
		t.Error("ready returned nil, want watchdog-kill error")
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("watchdog took %s, want ~500ms", elapsed)
	}
	collectAllLines(t, lines)
}

// A resume run that prints its init event (disarming the first-line watchdog)
// and then hangs — dead proxy, wedged API retry loop — must be killed by the
// sniff deadline instead of stalling forever behind pre-ready keepalives.
func TestResumeSniffWatchdogKillsPostInitHang(t *testing.T) {
	writeFakeClaude(t, `echo '{"type":"system","subtype":"init"}'
sleep 30`)
	old := firstLineTimeout
	firstLineTimeout = 500 * time.Millisecond
	defer func() { firstLineTimeout = old }()

	start := time.Now()
	lines, ready, _ := startClaudeStream([]string{"--resume", "sess-x"}, "hi", "", nil)
	if err := waitReadyErr(t, ready); err == nil {
		t.Error("ready returned nil, want sniff-watchdog error")
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("sniff watchdog took %s, want ~500ms", elapsed)
	}
	collectAllLines(t, lines)
}

// A stderr line larger than bufio.Scanner's default 64KB buffer must not end
// the stderr scan early — the "No conversation found" marker printed after it
// still has to gate the resume retry.
func TestStderrOversizedLineStillDetectsMissingSession(t *testing.T) {
	writeFakeClaude(t, `echo "$*" >> "$ARGSLOG"
case "$*" in
*--session-id*)
  echo '{"type":"system","subtype":"init"}'
  echo '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"recovered"}}}'
  echo '{"type":"result","subtype":"success","result":"recovered"}'
  ;;
*--resume*)
  head -c 200000 /dev/zero | tr '\0' 'x' >&2
  echo >&2
  echo "No conversation found with session ID: sess-x" >&2
  exit 1
  ;;
esac
`)
	_, env, count := newArgsLog(t)

	lines, ready, _ := startClaudeStream([]string{"--resume", "sess-x"}, "hi", "", env)
	if err := waitReadyErr(t, ready); err != nil {
		t.Fatalf("ready returned error, want nil (marker after oversized line must still be seen): %v", err)
	}
	got := collectAllLines(t, lines)
	if !strings.Contains(strings.Join(got, "\n"), `"text":"recovered"`) {
		t.Errorf("retry output missing; got: %v", got)
	}
	if n := count(); n != 2 {
		t.Errorf("claude invoked %d times, want 2", n)
	}
}

// Zero-event exits whose stdout held non-JSON text (node errors printed to
// stdout) must carry that text in the error instead of claiming "no output".
func TestZeroEventErrorMentionsLastOutput(t *testing.T) {
	writeFakeClaude(t, `echo "Error: Cannot find module 'foo'"
exit 1`)

	lines, ready, _ := startClaudeStream([]string{"--resume", "sess-x"}, "hi", "", nil)
	err := waitReadyErr(t, ready)
	if err == nil {
		t.Fatal("ready returned nil, want error")
	}
	if !strings.Contains(err.Error(), "Cannot find module") {
		t.Errorf("error %q does not mention the CLI's last output", err)
	}
	collectAllLines(t, lines)
}

// Non-stream counterpart: runClaude's resume retry must be gated on the
// session-missing stderr marker too, not fired blindly on empty output.
func TestRunClaudeNoRetryWhenSessionExists(t *testing.T) {
	writeFakeClaude(t, `echo "$*" >> "$ARGSLOG"
exit 0
`)
	_, env, count := newArgsLog(t)

	_, _, _, _, _, _, err := runClaude([]string{"--resume", "sess-x"}, "hi", "", env)
	if err != nil {
		t.Fatalf("runClaude error: %v", err)
	}
	if n := count(); n != 1 {
		t.Errorf("claude invoked %d times, want 1 (no blind retry)", n)
	}
}
