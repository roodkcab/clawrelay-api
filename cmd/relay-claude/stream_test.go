package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"clawrelay-api/pkg/sessions"
)

// waitProcessGone polls until the PID recorded in pidFile is dead (or was
// never written). Fails the test if the process is still alive after 5s.
func waitProcessGone(t *testing.T, pidFile string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		data, err := os.ReadFile(pidFile)
		if err != nil {
			// not (yet / ever) written — treat as gone after a grace period
			time.Sleep(100 * time.Millisecond)
			continue
		}
		pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
		if err != nil || syscall.Kill(pid, 0) != nil {
			return // dead
		}
		time.Sleep(100 * time.Millisecond)
	}
	data, _ := os.ReadFile(pidFile)
	if len(data) == 0 {
		return
	}
	pid, _ := strconv.Atoi(strings.TrimSpace(string(data)))
	if pid > 0 && syscall.Kill(pid, 0) == nil {
		syscall.Kill(-pid, syscall.SIGKILL)
		t.Fatalf("claude process %d still alive 5s after disconnect", pid)
	}
}

// SSE headers and the first ping must go out as soon as the request is
// accepted — NOT after claude's first stdout line. A slow or hung CLI used to
// leave the client with zero bytes (not even response headers), which trips
// wuji-tools' aiohttp sock_read=120s and surfaces as "AI 连接出现错误".
func TestStreamHeadersSentBeforeClaudeFirstOutput(t *testing.T) {
	writeFakeClaude(t, `sleep 1
echo '{"type":"system","subtype":"init"}'
echo '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi-there"}}}'
echo '{"type":"result","subtype":"success","result":"hi-there"}'
`)
	sessionStore = sessions.New(t.TempDir())
	oldHB := heartbeatInterval
	heartbeatInterval = 100 * time.Millisecond
	defer func() { heartbeatInterval = oldHB }()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handleStreamResponse(w, r, []string{"--model", "m"}, "hi", "chat-1", 1, "m", false, "", nil, "")
	}))
	defer srv.Close()

	start := time.Now()
	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	headerLatency := time.Since(start)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}
	// The stub sleeps 1s before its first line; headers must beat that.
	if headerLatency > 700*time.Millisecond {
		t.Errorf("headers took %s, want them before claude's first output (~1s)", headerLatency)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	s := string(body)
	if !strings.Contains(s, ": ping") {
		t.Errorf("body missing initial ': ping' comment:\n%s", s)
	}
	if !strings.Contains(s, ": keepalive") {
		t.Errorf("body missing pre-ready ': keepalive' heartbeat (claude silent for 1s, ticker at 100ms):\n%s", s)
	}
	if !strings.Contains(s, "hi-there") {
		t.Errorf("body missing streamed text:\n%s", s)
	}
	if !strings.Contains(s, "data: [DONE]") {
		t.Errorf("body missing [DONE]:\n%s", s)
	}
}

// Once headers are out, a startup failure (fork error, zero-output crash,
// watchdog kill) can no longer become an HTTP 500 — it must close the stream
// with a visible in-stream error chunk, mirroring the V2 channel handler.
func TestStreamReadyErrorEmitsInStreamErrorChunk(t *testing.T) {
	writeFakeClaude(t, `exit 0`)
	sessionStore = sessions.New(t.TempDir())

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handleStreamResponse(w, r, []string{"--model", "m"}, "hi", "chat-2", 1, "m", false, "", nil, "")
	}))
	defer srv.Close()

	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (headers already sent)", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	s := string(body)
	if !strings.Contains(s, "⚠️") {
		t.Errorf("body missing visible error chunk:\n%s", s)
	}
	if !strings.Contains(s, `"x_relay_error":true`) {
		t.Errorf("error chunk missing x_relay_error marker:\n%s", s)
	}
	if !strings.Contains(s, `"finish_reason":"stop"`) {
		t.Errorf("body missing terminal finish chunk:\n%s", s)
	}
	if !strings.Contains(s, "data: [DONE]") {
		t.Errorf("body missing [DONE]:\n%s", s)
	}
}

// Disconnecting while claude is still silent (pre-ready) must kill the
// process — the pre-ready ctx.Done path reads the current cmd through
// procHandle, which also makes this test race-clean under -race.
func TestPreReadyDisconnectKillsClaude(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "pid")
	writeFakeClaude(t, `echo $$ > "$PIDFILE"
sleep 30`)
	sessionStore = sessions.New(t.TempDir())

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handleStreamResponse(w, r, []string{"--model", "m"}, "hi", "chat-4", 1, "m", false, "",
			map[string]string{"PIDFILE": pidFile}, "")
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	// Wait until the stub has started, then drop the connection mid-startup.
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(pidFile); err == nil {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	cancel()
	resp.Body.Close()

	waitProcessGone(t, pidFile)
}

// Disconnecting during the resume sniff must NOT let the --session-id retry
// become an unkillable orphan: publish() after abort() kills the newcomer.
func TestPreReadyDisconnectDoesNotOrphanRetry(t *testing.T) {
	retryPid := filepath.Join(t.TempDir(), "retry-pid")
	writeFakeClaude(t, `case "$*" in
*--session-id*)
  echo $$ > "$RETRYPID"
  sleep 30
  ;;
*--resume*)
  sleep 0.5
  echo "No conversation found with session ID: sess-x" >&2
  exit 1
  ;;
esac
`)
	sessionStore = sessions.New(t.TempDir())

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handleStreamResponse(w, r, []string{"--resume", "sess-x"}, "hi", "chat-5", 1, "m", false, "",
			map[string]string{"RETRYPID": retryPid}, "")
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	// Cancel while the first (--resume) run is still sleeping, i.e. before the
	// producer decides to launch the retry.
	time.Sleep(150 * time.Millisecond)
	cancel()
	resp.Body.Close()

	// Either the retry was never launched, or it was killed right after
	// publish() returned false.
	waitProcessGone(t, retryPid)
}

// A forwarded error result must be visible on the buffered (tools) path too —
// it has its own result branch that bypasses translate.go.
func TestBufferedStreamErrorResultVisible(t *testing.T) {
	writeFakeClaude(t, `echo '{"type":"system","subtype":"init"}'
echo '`+errorResultLine+`'
`)
	sessionStore = sessions.New(t.TempDir())

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handleBufferedStreamResponse(w, r, []string{"--model", "m"}, "hi", "chat-6", 1, "m", false, "", nil, "")
	}))
	defer srv.Close()

	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	s := string(body)
	if !strings.Contains(s, "API Error: overloaded") {
		t.Errorf("error result text not surfaced on buffered path:\n%s", s)
	}
	if !strings.Contains(s, `"x_relay_error":true`) {
		t.Errorf("error chunk missing x_relay_error marker:\n%s", s)
	}
}

// Same contract for the buffered (tools) variant.
func TestBufferedStreamReadyErrorEmitsInStreamErrorChunk(t *testing.T) {
	writeFakeClaude(t, `exit 0`)
	sessionStore = sessions.New(t.TempDir())

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handleBufferedStreamResponse(w, r, []string{"--model", "m"}, "hi", "chat-3", 1, "m", false, "", nil, "")
	}))
	defer srv.Close()

	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (headers already sent)", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	s := string(body)
	if !strings.Contains(s, "⚠️") {
		t.Errorf("body missing visible error chunk:\n%s", s)
	}
	if !strings.Contains(s, "data: [DONE]") {
		t.Errorf("body missing [DONE]:\n%s", s)
	}
}
