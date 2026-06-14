package main

import (
	"bytes"
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"clawrelay-api/pkg/openai"
	"clawrelay-api/pkg/sessions"
)

// bufWriteCloser is an in-memory io.WriteCloser for capturing stdin writes.
type bufWriteCloser struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *bufWriteCloser) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}
func (b *bufWriteCloser) Close() error { return nil }
func (b *bufWriteCloser) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// ---- turn-boundary demux ----

func TestDrainStdoutTurnBoundary(t *testing.T) {
	w := &chanWorker{ready: make(chan struct{})}
	lines := make(chan string, 64)
	w.curLines = lines

	input := strings.Join([]string{
		`{"type":"system","subtype":"init","session_id":"sid-123"}`,
		`{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}`,
		`{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}}`,
		`{"type":"result","subtype":"success","session_id":"sid-123"}`,
		`{"type":"system","subtype":"stray_after_result"}`, // must be dropped (no active turn)
	}, "\n") + "\n"

	go w.drainStdout(strings.NewReader(input))

	var got []string
	for l := range lines { // closes after the result line
		got = append(got, l)
	}

	if len(got) != 4 {
		t.Fatalf("expected 4 forwarded lines (through result), got %d: %v", len(got), got)
	}
	if !strings.Contains(got[len(got)-1], `"type":"result"`) {
		t.Fatalf("last forwarded line should be the result event, got %q", got[len(got)-1])
	}
	if w.SessionID() != "sid-123" {
		t.Fatalf("session_id not captured, got %q", w.SessionID())
	}
	select {
	case <-w.ready:
	default:
		t.Fatal("ready channel should be closed after session_id capture")
	}
}

func TestDrainStdoutDropsBetweenTurns(t *testing.T) {
	w := &chanWorker{ready: make(chan struct{})}
	// curLines nil: no active turn. Lines must be dropped, not block/panic.
	input := `{"type":"system","session_id":"sid-x"}` + "\n" + `{"type":"result"}` + "\n"

	done := make(chan struct{})
	go func() { w.drainStdout(strings.NewReader(input)); close(done) }()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("drainStdout blocked with no active turn")
	}
	if w.SessionID() != "sid-x" {
		t.Fatalf("session_id should still be captured between turns, got %q", w.SessionID())
	}
}

func TestDrainStdoutClosesTurnOnProcessExit(t *testing.T) {
	// If the process dies mid-turn (no result), drainStdout must still close
	// the active turn channel so the handler unblocks.
	w := &chanWorker{ready: make(chan struct{})}
	lines := make(chan string, 8)
	w.curLines = lines
	input := `{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}}` + "\n"

	go w.drainStdout(strings.NewReader(input)) // EOF after one line, no result

	var got []string
	for l := range lines { // must close via the defer on EOF
		got = append(got, l)
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 forwarded line before EOF, got %d", len(got))
	}
}

// ---- stdin writes ----

func TestBeginTurnWritesUserEnvelope(t *testing.T) {
	buf := &bufWriteCloser{}
	w := &chanWorker{stdin: buf}

	lines, err := w.beginTurn("hello 世界")
	if err != nil {
		t.Fatalf("beginTurn: %v", err)
	}
	defer w.endTurn()
	_ = lines

	var env struct {
		Type    string `json:"type"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(buf.String())), &env); err != nil {
		t.Fatalf("written envelope not valid JSON: %v (%q)", err, buf.String())
	}
	if env.Type != "user" || env.Message.Role != "user" || env.Message.Content != "hello 世界" {
		t.Fatalf("unexpected envelope: %+v", env)
	}
	if !strings.HasSuffix(buf.String(), "\n") {
		t.Fatal("envelope must be newline-terminated (NDJSON)")
	}
}

func TestInterruptWritesControlMessage(t *testing.T) {
	buf := &bufWriteCloser{}
	w := &chanWorker{stdin: buf}
	if err := w.interrupt(); err != nil {
		t.Fatalf("interrupt: %v", err)
	}
	if strings.TrimSpace(buf.String()) != `{"type":"interrupt"}` {
		t.Fatalf("unexpected interrupt payload: %q", buf.String())
	}
}

func TestBeginTurnFailsWhenDead(t *testing.T) {
	w := &chanWorker{stdin: &bufWriteCloser{}}
	w.dead.Store(true)
	if _, err := w.beginTurn("x"); err == nil {
		t.Fatal("beginTurn should fail on a dead worker")
	}
}

// ---- manager: reuse / reaper / evict / die ----

func newWorkerAt(key string, last time.Time) *chanWorker {
	w := &chanWorker{key: key, ready: make(chan struct{}), deadCh: make(chan struct{})}
	w.lastUsed.Store(last.UnixNano())
	return w
}

func TestManagerReusesLiveWorker(t *testing.T) {
	m := newChanManager(chanManagerConfig{MaxChannels: 10})
	w := newWorkerAt("s1", time.Now())
	m.workers["s1"] = w

	got, err := m.acquire("s1", spawnParams{})
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if got != w {
		t.Fatal("acquire should reuse the existing live worker (no spawn)")
	}
}

func TestManagerReaperKillsIdle(t *testing.T) {
	m := newChanManager(chanManagerConfig{IdleTTL: time.Minute})
	fresh := newWorkerAt("fresh", time.Now())
	old := newWorkerAt("old", time.Now().Add(-2*time.Minute))
	m.workers["fresh"] = fresh
	m.workers["old"] = old

	m.reapOnce()

	if _, ok := m.workers["old"]; ok {
		t.Fatal("idle worker should be reaped")
	}
	if !old.Dead() {
		t.Fatal("reaped worker should be marked dead")
	}
	if _, ok := m.workers["fresh"]; !ok {
		t.Fatal("fresh worker should survive reaping")
	}
}

func TestManagerEvictsOldestAtCapacity(t *testing.T) {
	m := newChanManager(chanManagerConfig{MaxChannels: 2})
	a := newWorkerAt("a", time.Now().Add(-3*time.Minute))
	b := newWorkerAt("b", time.Now())
	m.workers["a"] = a
	m.workers["b"] = b

	m.mu.Lock()
	m.evictOldestLocked()
	m.mu.Unlock()

	if _, ok := m.workers["a"]; ok {
		t.Fatal("oldest worker should be evicted")
	}
	if _, ok := m.workers["b"]; !ok {
		t.Fatal("newer worker should remain")
	}
}

func TestManagerOnWorkerDieRemoves(t *testing.T) {
	m := newChanManager(chanManagerConfig{})
	w := newWorkerAt("d", time.Now())
	w.dead.Store(true)
	m.workers["d"] = w

	m.onWorkerDie("d")
	if _, ok := m.workers["d"]; ok {
		t.Fatal("dead worker should be removed by onWorkerDie")
	}
}

// ---- cwd encoding / session existence ----

func TestEncodeClaudeCwd(t *testing.T) {
	cases := map[string]string{
		"/data/skills":            "-data-skills",
		"/home/claude01/work_dir": "-home-claude01-work-dir",
		"/srv/bot_a/skills_v2":    "-srv-bot-a-skills-v2",
	}
	for in, want := range cases {
		if got := encodeClaudeCwd(in); got != want {
			t.Errorf("encodeClaudeCwd(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestClaudeSessionExists(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	workdir := "/data/skills" // non-existent path → encoded as -data-skills
	sid := "abc-123"
	dir := filepath.Join(home, ".claude", "projects", encodeClaudeCwd(workdir))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, sid+".jsonl"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}

	if !claudeSessionExists(workdir, sid) {
		t.Fatal("expected session to exist")
	}
	if claudeSessionExists(workdir, "no-such-session") {
		t.Fatal("missing session should not be reported as existing")
	}
	if claudeSessionExists("", sid) {
		t.Fatal("empty workdir should be false")
	}
}

// ---- channel eligibility (session_id-empty degrade) ----

func TestIsChannelEligible(t *testing.T) {
	mk := func(stream bool, sid string, tools int) *openai.ChatCompletionRequest {
		r := &openai.ChatCompletionRequest{Stream: stream, SessionID: sid}
		for i := 0; i < tools; i++ {
			r.Tools = append(r.Tools, openai.Tool{Type: "function"})
		}
		return r
	}
	cases := []struct {
		name string
		req  *openai.ChatCompletionRequest
		want bool
	}{
		{"stream+session+notools", mk(true, "s1", 0), true},
		{"no session degrades", mk(true, "", 0), false},
		{"non-stream degrades", mk(false, "s1", 0), false},
		{"tools degrade", mk(true, "s1", 2), false},
	}
	for _, c := range cases {
		if got := isChannelEligible(c.req); got != c.want {
			t.Errorf("%s: isChannelEligible = %v, want %v", c.name, got, c.want)
		}
	}
}

// ---- translator parity (shared by legacy + channel) ----

func TestSSETranslatorTextAndResult(t *testing.T) {
	sessionStore = sessions.New(t.TempDir())
	rec := httptest.NewRecorder()
	tr := newSSETranslator("chatcmpl-x", 1700000000, "haiku", "") // empty sessionID → log no-ops

	textLine := `{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}}`
	resultLine := `{"type":"result","subtype":"success","usage":{"input_tokens":10,"output_tokens":5},"total_cost_usd":0.001}`

	if got := tr.feed(rec, rec, textLine, true); got != outcomeContinue {
		t.Fatalf("text line outcome = %v, want continue", got)
	}
	if got := tr.feed(rec, rec, resultLine, true); got != outcomeContinue {
		t.Fatalf("result line outcome = %v, want continue", got)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"content":"Hello"`) {
		t.Errorf("missing text delta chunk in body:\n%s", body)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Errorf("missing finish chunk in body:\n%s", body)
	}
	if !strings.Contains(body, `"prompt_tokens"`) {
		t.Errorf("missing usage chunk in body:\n%s", body)
	}
	if tr.StreamUsage() == nil {
		t.Error("StreamUsage should be populated after result")
	}
}

func TestSSETranslatorAskUserQuestion(t *testing.T) {
	sessionStore = sessions.New(t.TempDir())
	rec := httptest.NewRecorder()
	tr := newSSETranslator("chatcmpl-y", 1700000000, "haiku", "")

	start := `{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"AskUserQuestion"}}}`
	delta := `{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"questions\":[]}"}}}`
	stop := `{"type":"stream_event","event":{"type":"content_block_stop","index":0}}`

	tr.feed(rec, rec, start, true)
	tr.feed(rec, rec, delta, true)
	got := tr.feed(rec, rec, stop, true)

	if got != outcomeAskUserDone {
		t.Fatalf("AskUserQuestion stop outcome = %v, want outcomeAskUserDone", got)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"AskUserQuestion"`) {
		t.Errorf("missing AskUserQuestion tool_call in body:\n%s", body)
	}
	if !strings.Contains(body, `data: [DONE]`) {
		t.Errorf("AskUserQuestion should terminate with [DONE]:\n%s", body)
	}
}
