package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
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
	a := &activeTurn{lines: make(chan string, 64), quit: make(chan struct{})}
	w.cur = a

	input := strings.Join([]string{
		`{"type":"system","subtype":"init","session_id":"sid-123"}`,
		`{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}`,
		`{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}}`,
		`{"type":"result","subtype":"success","session_id":"sid-123"}`,
		`{"type":"system","subtype":"stray_after_result"}`, // must be dropped (no active turn)
	}, "\n") + "\n"

	go w.drainStdout(strings.NewReader(input))

	var got []string
	for l := range a.lines { // closes after the result line
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
	a := &activeTurn{lines: make(chan string, 8), quit: make(chan struct{})}
	w.cur = a
	input := `{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}}` + "\n"

	go w.drainStdout(strings.NewReader(input)) // EOF after one line, no result

	var got []string
	for l := range a.lines { // must close via the defer on EOF
		got = append(got, l)
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 forwarded line before EOF, got %d", len(got))
	}
}

// blockingReader emits `head` then blocks on a release channel, then EOFs. It
// keeps drainStdout's scanner alive so a parked send can be observed.
type blockingReader struct {
	head    []byte
	off     int
	release chan struct{}
	done    bool
}

func (b *blockingReader) Read(p []byte) (int, error) {
	if b.off < len(b.head) {
		n := copy(p, b.head[b.off:])
		b.off += n
		return n, nil
	}
	if !b.done {
		<-b.release
		b.done = true
	}
	return 0, io.EOF
}

// TestAbandonWhileParkedNoPanic is the regression guard for the critical
// send-on-closed race: drainStdout parks on a full-buffer send, then a
// concurrent abandon (as the cmd.Wait waiter / kill would do) must unblock it
// and close `lines` WITHOUT a "send on closed channel" panic.
func TestAbandonWhileParkedNoPanic(t *testing.T) {
	for iter := 0; iter < 50; iter++ {
		w := &chanWorker{ready: make(chan struct{})}
		a := &activeTurn{lines: make(chan string, 1), quit: make(chan struct{})}
		w.cur = a

		// 20 non-result lines into a 1-slot buffer with no consumer → drainStdout
		// parks on the send almost immediately.
		var sb strings.Builder
		for i := 0; i < 20; i++ {
			sb.WriteString(`{"type":"stream_event"}` + "\n")
		}
		br := &blockingReader{head: []byte(sb.String()), release: make(chan struct{})}

		done := make(chan struct{})
		go func() { w.drainStdout(br); close(done) }()

		time.Sleep(2 * time.Millisecond) // let it park on the send
		w.abandonActiveTurn(nil)         // waiter/kill path: must not panic
		close(br.release)                // allow EOF

		for range a.lines { // drain; must be closed, never panic
		}
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatal("drainStdout did not exit after abandon")
		}
	}
}

func TestReaperSkipsInTurnWorker(t *testing.T) {
	m := newChanManager(chanManagerConfig{IdleTTL: time.Minute})
	busy := newWorkerAt("busy", time.Now().Add(-2*time.Minute)) // stale lastUsed
	busy.inTurn.Store(true)                                     // but actively streaming
	idle := newWorkerAt("idle", time.Now().Add(-2*time.Minute))
	m.workers["busy"] = busy
	m.workers["idle"] = idle

	m.reapOnce()

	if _, ok := m.workers["busy"]; !ok {
		t.Fatal("a worker mid-turn must NOT be reaped even with stale lastUsed")
	}
	if _, ok := m.workers["idle"]; ok {
		t.Fatal("an idle worker should still be reaped")
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

// Regression for the round-2 #1 fix: the write can succeed on a child that just
// exited; if deadCh is already closed, beginTurn must fail fast (no live
// drainStdout to ever close the returned lines) and release turnMu.
func TestBeginTurnFailsIfDiedAfterWrite(t *testing.T) {
	w := &chanWorker{stdin: &bufWriteCloser{}, deadCh: make(chan struct{})}
	close(w.deadCh) // process exited, but dead flag not yet set → initial check passes
	if _, err := w.beginTurn("hi"); err == nil {
		t.Fatal("beginTurn must fail when the process died right after the write")
	}
	if !w.turnMu.TryLock() {
		t.Fatal("turnMu must be released after a fast-fail")
	}
	w.turnMu.Unlock()
}

func TestStopKillsInflightWorkers(t *testing.T) {
	m := newChanManager(chanManagerConfig{})
	persistent := newWorkerAt("p", time.Now())
	transient := newWorkerAt("i", time.Now())
	m.workers["p"] = persistent
	m.inflight[transient] = struct{}{}

	m.Stop()

	if !persistent.Dead() || !transient.Dead() {
		t.Fatal("Stop must kill both persistent and inflight workers")
	}
	if len(m.workers) != 0 || len(m.inflight) != 0 {
		t.Fatal("Stop must clear both registries")
	}
}

func TestSnapshotIncludesInflight(t *testing.T) {
	m := newChanManager(chanManagerConfig{})
	m.workers["p"] = newWorkerAt("p", time.Now())
	m.inflight[newWorkerAt("i", time.Now())] = struct{}{}

	snap := m.snapshot()
	if len(snap) != 2 {
		t.Fatalf("snapshot should report workers + inflight, got %d", len(snap))
	}
	var transientSeen bool
	for _, row := range snap {
		if row["transient"] == true {
			transientSeen = true
		}
	}
	if !transientSeen {
		t.Fatal("inflight worker must be marked transient in snapshot")
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

	got, err := m.acquire(context.Background(), "s1", spawnParams{})
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
		{"stream+nosession (ephemeral)", mk(true, "", 0), true}, // no session → ephemeral channel, not legacy -p
		{"non-stream degrades", mk(false, "s1", 0), false},
		{"non-stream nosession degrades", mk(false, "", 0), false},
		{"tools degrade", mk(true, "s1", 2), false},
	}
	for _, c := range cases {
		if got := isChannelEligible(c.req); got != c.want {
			t.Errorf("%s: isChannelEligible = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestNewUUID(t *testing.T) {
	re := regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)
	seen := map[string]bool{}
	for i := 0; i < 1000; i++ {
		u := newUUID()
		if !re.MatchString(u) {
			t.Fatalf("newUUID() = %q is not a valid RFC-4122 v4 UUID", u)
		}
		if seen[u] {
			t.Fatalf("newUUID() produced a duplicate: %q", u)
		}
		seen[u] = true
	}
}

// ---- translator parity (shared by legacy + channel) ----

func TestSSETranslatorTextAndResult(t *testing.T) {
	sessionStore = sessions.New(t.TempDir())
	rec := httptest.NewRecorder()
	tr := newSSETranslator("chatcmpl-x", 1700000000, "haiku", "", identityMeter{}, "") // empty sessionID → log no-ops

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
	tr := newSSETranslator("chatcmpl-y", 1700000000, "haiku", "", identityMeter{}, "")

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

// 同一 cumulativeMeter 跨两轮 feed：第二轮 emit 的是 delta，不是累计。
// 用 modelUsage shape（进程级累计口径）驱动；bare usage 是 per-turn 值，
// 不走差分（实证依据见 usage_meter.go 注释）。
func TestSSETranslatorChannelDiffsCumulativeUsage(t *testing.T) {
	sessionStore = sessions.New(t.TempDir())
	meter := &cumulativeMeter{}

	rec1 := httptest.NewRecorder()
	tr1 := newSSETranslator("c1", 1700000000, "haiku", "", meter, "")
	tr1.feed(rec1, rec1, `{"type":"result","modelUsage":{"claude-haiku":{"inputTokens":100,"outputTokens":50,"cacheReadInputTokens":1000}}}`, true)

	rec2 := httptest.NewRecorder()
	tr2 := newSSETranslator("c2", 1700000000, "haiku", "", meter, "")
	tr2.feed(rec2, rec2, `{"type":"result","modelUsage":{"claude-haiku":{"inputTokens":106,"outputTokens":90,"cacheReadInputTokens":4500}}}`, true)

	u := tr2.StreamUsage()
	if u == nil {
		t.Fatal("turn 2 StreamUsage nil")
	}
	if u.CompletionTokens != 40 {
		t.Fatalf("turn2 completion = %d, want 40 (per-turn delta, not cumulative 90)", u.CompletionTokens)
	}
	// promptTokens = input(6) + cacheRead(3500) = 3506
	if u.PromptTokens != 3506 {
		t.Fatalf("turn2 prompt = %d, want 3506 (delta 6 input + 3500 cacheRead)", u.PromptTokens)
	}
}

// gating fix: modelUsage 在场、顶层 usage 缺失时，仍 emit usage chunk。
func TestSSETranslatorEmitsUsageFromModelUsage(t *testing.T) {
	sessionStore = sessions.New(t.TempDir())
	rec := httptest.NewRecorder()
	tr := newSSETranslator("c", 1700000000, "haiku", "", identityMeter{}, "")
	line := `{"type":"result","modelUsage":{"claude-haiku":{"inputTokens":10,"outputTokens":5,"cacheReadInputTokens":20}}}`
	tr.feed(rec, rec, line, true)
	if !strings.Contains(rec.Body.String(), `"prompt_tokens"`) {
		t.Errorf("usage chunk missing when only modelUsage present (gating bug):\n%s", rec.Body.String())
	}
}

// A2: 进程无 result 就退出时，流必须补一个 finish chunk；正常轮 no-op。
func TestEmitFinishIfNoResult(t *testing.T) {
	sessionStore = sessions.New(t.TempDir())

	// 没见过 result → 合成 finish chunk。
	rec := httptest.NewRecorder()
	tr := newSSETranslator("c-nofin", 1700000000, "haiku", "", identityMeter{}, "")
	tr.feed(rec, rec, `{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}}`, true)
	tr.EmitFinishIfNoResult(rec, rec)
	if !strings.Contains(rec.Body.String(), `"finish_reason":"stop"`) {
		t.Errorf("missing synthetic finish chunk when stream ended without result:\n%s", rec.Body.String())
	}

	// 正常轮（result 已 emit finish）→ no-op，不能出现第二个 finish chunk。
	rec2 := httptest.NewRecorder()
	tr2 := newSSETranslator("c-fin", 1700000000, "haiku", "", identityMeter{}, "")
	tr2.feed(rec2, rec2, `{"type":"result","usage":{"input_tokens":1,"output_tokens":1}}`, false)
	before := rec2.Body.Len()
	tr2.EmitFinishIfNoResult(rec2, rec2)
	if rec2.Body.Len() != before {
		t.Errorf("EmitFinishIfNoResult must be a no-op after a result event:\n%s", rec2.Body.String())
	}
}

// A5: cumulativeMeter 路径的 stats 归属跟 meterModel（worker 的 spawn-time
// boundModel），而不是请求 model —— 热切模型后不再错归属。
func TestSSETranslatorAttributesToMeterModel(t *testing.T) {
	sessionStore = sessions.New(t.TempDir())
	const boundModel = "test-meter-model-bound"
	const reqModel = "test-meter-model-request"

	rowTotal := func(model string) int64 {
		row, ok := stats.Snapshot().PerModel[model]
		if !ok {
			return 0
		}
		return row.Total
	}
	boundBefore, reqBefore := rowTotal(boundModel), rowTotal(reqModel)

	rec := httptest.NewRecorder()
	tr := newSSETranslator("c-mm", 1700000000, reqModel, "", &cumulativeMeter{}, boundModel)
	tr.feed(rec, rec, `{"type":"result","modelUsage":{"claude-x":{"inputTokens":11,"outputTokens":7}}}`, true)

	if got := rowTotal(boundModel) - boundBefore; got != 18 {
		t.Fatalf("boundModel stats delta = %d, want 18", got)
	}
	if got := rowTotal(reqModel) - reqBefore; got != 0 {
		t.Fatalf("request model must not accrue tokens after hot switch, delta = %d", got)
	}
}

// A8: 同一 session 的并发 acquire 单飞——第二个请求等待在飞 spawn 完成后复用其
// worker，而不是并发再起一个进程写同一份 session jsonl。
func TestAcquireWaitsForInflightSpawn(t *testing.T) {
	m := newChanManager(chanManagerConfig{})
	spawnCh := make(chan struct{})
	m.spawning["s1"] = spawnCh // 模拟已有请求正在 spawn

	type result struct {
		w   *chanWorker
		err error
	}
	got := make(chan result, 1)
	go func() {
		w, err := m.acquire(context.Background(), "s1", spawnParams{})
		got <- result{w, err}
	}()

	// 等待方必须挂起在 spawning channel 上，而不是自己再 spawn 一个。
	select {
	case <-got:
		t.Fatal("acquire returned before the in-flight spawn finished")
	case <-time.After(50 * time.Millisecond):
	}

	// spawner 完成：晋升 worker + 释放单飞。
	w := newWorkerAt("s1", time.Now())
	m.mu.Lock()
	m.workers["s1"] = w
	delete(m.spawning, "s1")
	m.mu.Unlock()
	close(spawnCh)

	select {
	case r := <-got:
		if r.err != nil {
			t.Fatalf("acquire after spawn finished: %v", r.err)
		}
		if r.w != w {
			t.Fatal("waiter should reuse the worker promoted by the spawner")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("waiter did not wake after the in-flight spawn finished")
	}
}
