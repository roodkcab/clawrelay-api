package main

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"clawrelay-api/pkg/openai"
)

// nil usage → NO usage chunk: downstream must keep storing NULL ("not
// metered"), never a fake 0 ("free request").
func TestV3EmitCloseNilUsageOmitsChunk(t *testing.T) {
	rec := httptest.NewRecorder()
	v3EmitClose(rec, rec, "c", 1700000000, "haiku", "hello", true, nil) // includeUsage=true
	body := rec.Body.String()
	if strings.Contains(body, `"prompt_tokens"`) {
		t.Errorf("nil usage must not emit a usage chunk (would log as 0, not NULL):\n%s", body)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Errorf("missing finish chunk:\n%s", body)
	}
	if !strings.Contains(body, `data: [DONE]`) {
		t.Errorf("missing [DONE]:\n%s", body)
	}
}

// Harvested usage + includeUsage → a usage chunk between finish and [DONE].
func TestV3EmitCloseEmitsHarvestedUsage(t *testing.T) {
	rec := httptest.NewRecorder()
	u := openai.BuildUsageInfo(10, 5, 3, 2) // prompt=10+3+2=15, completion=5
	v3EmitClose(rec, rec, "c", 1700000000, "haiku", "hello", true, u)
	body := rec.Body.String()
	if !strings.Contains(body, `"prompt_tokens":15`) || !strings.Contains(body, `"completion_tokens":5`) {
		t.Errorf("missing usage chunk fields:\n%s", body)
	}
	if !strings.Contains(body, `"cached_tokens":3`) || !strings.Contains(body, `"cache_creation_tokens":2`) {
		t.Errorf("missing prompt_tokens_details:\n%s", body)
	}
	finIdx := strings.Index(body, `"finish_reason":"stop"`)
	useIdx := strings.Index(body, `"prompt_tokens"`)
	doneIdx := strings.Index(body, "data: [DONE]")
	if !(finIdx >= 0 && useIdx > finIdx && doneIdx > useIdx) {
		t.Errorf("usage chunk must sit between finish chunk and [DONE] (fin=%d use=%d done=%d):\n%s",
			finIdx, useIdx, doneIdx, body)
	}
}

// includeUsage=false suppresses the chunk even when usage was harvested.
func TestV3EmitCloseRespectsIncludeUsage(t *testing.T) {
	rec := httptest.NewRecorder()
	v3EmitClose(rec, rec, "c", 1700000000, "haiku", "hello", false, openai.BuildUsageInfo(1, 1, 0, 0))
	if body := rec.Body.String(); strings.Contains(body, `"prompt_tokens"`) {
		t.Errorf("includeUsage=false must not emit usage chunk:\n%s", body)
	}
}

// ---- turn 串行化 / inbox 生命周期 / .mcp.json 防护 ----

func newTestV3Manager() *v3Manager {
	return newV3Manager(v3Config{BridgeDir: "/nonexistent"})
}

// enqueue 对不存在的 inbox 必须立刻报错，而不是凭空重建条目或永久阻塞——
// inbox 只在 acquire 促升时创建、teardown 时删除。
func TestV3EnqueueNoInboxErrors(t *testing.T) {
	m := newTestV3Manager()
	err := m.enqueue(context.Background(), "ghost", v3Msg{ReqID: "r1", Content: "hi"})
	if err == nil {
		t.Fatal("enqueue into a non-existent inbox must error")
	}
	if _, ok := m.inbox["ghost"]; ok {
		t.Fatal("enqueue must not re-create a torn-down inbox")
	}
}

// inbox 卡满时 enqueue 必须能被 ctx 解救，不能把 handler goroutine 永远吊住。
func TestV3EnqueueCtxEscape(t *testing.T) {
	m := newTestV3Manager()
	q := make(chan v3Msg, 1)
	q <- v3Msg{} // 填满
	m.inbox["sid"] = q
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- m.enqueue(ctx, "sid", v3Msg{ReqID: "r2"}) }()
	cancel()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("cancelled enqueue must return an error")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("enqueue did not honor ctx cancellation")
	}
}

// drainInbox 只清空既有队列，不为未知 sid 重建条目（重建即泄漏）。
func TestV3DrainInboxNoCreate(t *testing.T) {
	m := newTestV3Manager()
	m.drainInbox("ghost")
	if _, ok := m.inbox["ghost"]; ok {
		t.Fatal("drainInbox must not create an inbox entry")
	}
	q := make(chan v3Msg, 8)
	q <- v3Msg{ReqID: "stale"}
	m.inbox["sid"] = q
	m.drainInbox("sid")
	if len(q) != 0 {
		t.Fatalf("drainInbox left %d stale messages", len(q))
	}
}

// handleNext 对未知/已拆除的 sid 返回 404，绝不重建 inbox——否则每个 ephemeral
// 会话被拆后 bridge 的最后一次 poll 都会在 map 里留一个永久条目。
func TestV3HandleNextUnknownSession(t *testing.T) {
	m := newTestV3Manager()
	req := httptest.NewRequest("GET", "/v3/next?session=ghost", nil)
	rec := httptest.NewRecorder()
	m.handleNext(rec, req)
	if rec.Code != 404 {
		t.Fatalf("unknown session must 404, got %d", rec.Code)
	}
	if _, ok := m.inbox["ghost"]; ok {
		t.Fatal("handleNext must not create an inbox entry")
	}
}

// 各 teardown 路径都要把 inbox 一并删掉，防 map 无界增长 + 旧消息重放进新会话。
func TestV3TeardownDeletesInbox(t *testing.T) {
	m := newTestV3Manager()
	mk := func(sid string) *v3Session {
		s := &v3Session{sid: sid, deadCh: make(chan struct{}), turnSlot: make(chan struct{}, 1)}
		s.turnSlot <- struct{}{}
		s.reaped.Store(true) // 无真实进程：kill() 不得走 KillGroup
		s.markUsed()
		m.sessions[sid] = s
		m.inbox[sid] = make(chan v3Msg, 8)
		return s
	}
	mk("a")
	m.killSession("a")
	if _, ok := m.inbox["a"]; ok {
		t.Fatal("killSession must delete inbox")
	}
	s := mk("b")
	s.dead.Store(true)
	m.onSessionDie("b")
	if _, ok := m.inbox["b"]; ok {
		t.Fatal("onSessionDie must delete inbox for the dead mapped session")
	}
	mk("c")
	m.sessions["c"].lastUsed.Store(time.Now().Add(-2 * m.cfg.IdleTTL).UnixNano())
	m.reapOnce()
	if _, ok := m.inbox["c"]; ok {
		t.Fatal("reapOnce must delete inbox")
	}
	mk("d")
	m.Stop()
	if len(m.inbox) != 0 || len(m.sessions) != 0 {
		t.Fatalf("Stop must clear sessions+inbox, got %d/%d", len(m.sessions), len(m.inbox))
	}
}

// onSessionDie 迟到（冷启动重试已促升新会话）时，不得误删新会话与新 inbox。
func TestV3LateOnSessionDieKeepsNewSession(t *testing.T) {
	m := newTestV3Manager()
	fresh := &v3Session{sid: "s", deadCh: make(chan struct{}), turnSlot: make(chan struct{}, 1)}
	fresh.turnSlot <- struct{}{}
	m.sessions["s"] = fresh
	m.inbox["s"] = make(chan v3Msg, 8)
	m.onSessionDie("s") // 死的是旧会话，但 map 里已是活的新会话
	if m.sessions["s"] != fresh {
		t.Fatal("late onSessionDie must keep the promoted live session")
	}
	if _, ok := m.inbox["s"]; !ok {
		t.Fatal("late onSessionDie must keep the new inbox")
	}
}

// ensureBridgeInMCPJSON：既有配置必须被合并保留；非法 JSON 必须拒绝而非清空；
// 写入必须原子（无 .relaytmp 残留）。
func TestEnsureBridgeInMCPJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".mcp.json")

	if err := os.WriteFile(path, []byte(`{"mcpServers":{"mine":{"command":"foo"}}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := ensureBridgeInMCPJSON(path, "/b/bridge.ts"); err != nil {
		t.Fatalf("merge failed: %v", err)
	}
	data, _ := os.ReadFile(path)
	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("result not valid JSON: %v", err)
	}
	servers := cfg["mcpServers"].(map[string]any)
	if _, ok := servers["mine"]; !ok {
		t.Fatal("existing user server was dropped by the merge")
	}
	if _, ok := servers["relaybridge"]; !ok {
		t.Fatal("relaybridge not merged in")
	}
	if _, err := os.Stat(path + ".relaytmp"); !os.IsNotExist(err) {
		t.Fatal("temp file left behind (rename not atomic?)")
	}

	if err := os.WriteFile(path, []byte(`{broken`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := ensureBridgeInMCPJSON(path, "/b/bridge.ts"); err == nil {
		t.Fatal("invalid existing JSON must be refused, not clobbered")
	}
	if data, _ := os.ReadFile(path); string(data) != `{broken` {
		t.Fatal("invalid existing file must be left untouched")
	}
}

// 单飞：同 sid 并发 acquire 在 launch 失败时也只允许串行的 launch 尝试，且全部
// 干净返回错误、不 panic、不留 launching 残留。清空 PATH 让 exec 找不到 claude
// ——绝不能在测试里真的拉起交互式 claude（这台机器上有真的二进制）。
func TestV3AcquireSingleflightFailurePath(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	m := newTestV3Manager()
	p := v3SpawnParams{model: "haiku", workingDir: t.TempDir()}
	var wg sync.WaitGroup
	errs := make([]error, 6)
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, errs[i] = m.acquire(context.Background(), "same-sid", p)
		}(i)
	}
	wg.Wait()
	for i, err := range errs {
		if err == nil {
			t.Fatalf("acquire %d unexpectedly succeeded (claude binary should be absent in tests)", i)
		}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.launching) != 0 {
		t.Fatalf("launching map must be empty after failures, got %d", len(m.launching))
	}
	if len(m.sessions) != 0 {
		t.Fatalf("no session should be registered after failed launches, got %d", len(m.sessions))
	}
}

// acquire 遇到 map 里的死会话时,即使随后的 relaunch 失败,也必须把死会话的 inbox
// 一并清掉——否则 onSessionDie 因 map 里已无此会话而不兜底删 inbox,条目泄漏。
func TestV3AcquireDropsDeadSessionInbox(t *testing.T) {
	t.Setenv("PATH", t.TempDir()) // 让 launch 失败,走 relaunch-fails 路径
	m := newTestV3Manager()
	dead := &v3Session{sid: "s", deadCh: make(chan struct{}), turnSlot: make(chan struct{}, 1)}
	dead.dead.Store(true)
	dead.reaped.Store(true)
	m.sessions["s"] = dead
	m.inbox["s"] = make(chan v3Msg, 8)

	if _, err := m.acquire(context.Background(), "s", v3SpawnParams{workingDir: t.TempDir()}); err == nil {
		t.Fatal("acquire should fail when claude binary is absent")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.inbox["s"]; ok {
		t.Fatal("acquire must delete a dead session's inbox even when relaunch fails")
	}
	if _, ok := m.sessions["s"]; ok {
		t.Fatal("dead session must be removed from the map")
	}
}

// ctx 取消要能把排队等单飞的 acquire 解救出来。
func TestV3AcquireCtxCancelWhileWaiting(t *testing.T) {
	m := newTestV3Manager()
	m.launching["sid"] = make(chan struct{}) // 模拟一个永不结束的 launch
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := m.acquire(ctx, "sid", v3SpawnParams{})
		done <- err
	}()
	cancel()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("cancelled acquire must return an error")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("acquire did not honor ctx cancellation while waiting on single-flight")
	}
}
