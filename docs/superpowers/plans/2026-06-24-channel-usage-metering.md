# Channel 模式 Token Usage 统计修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修正 relay-claude channel(V2) 模式 token usage 被累计重复计数、channelv3(V3) 模式 usage 假记 0 两个问题。

**Architecture:** 引入 `usageMeter` 接口把「raw usage 快照 → 单轮 usage」的策略按 mode 收敛:V1/ephemeral 用 identity(透传)、V2 用 cumulative-diff(状态挂在常驻 `chanWorker` 上)、V3 不 emit usage(下游写 NULL)。差分发生在 `sseTranslator.feed()` 的 result 处理处,作用于 emit SSE / stats.Record / LogDone。

**Tech Stack:** Go (package `main` in `cmd/relay-claude/`)，标准库 `testing` + `net/http/httptest`。

设计依据: `docs/superpowers/specs/2026-06-24-channel-usage-metering-design.md`。

## Global Constraints

- 仅改 `cmd/relay-claude/`（+ 复用 `pkg/openai`），**不改** wuji_tools、数据库 schema、OpenAI API 签名。
- `stats.Record` 参数顺序: `(model, input, output, cacheCreation, cacheRead, costUSD)`。
- `openai.BuildUsageInfo` 参数顺序: `(input, output, cacheRead, cacheCreation)`，全 0 时返回 `nil`。
- 每个 task 跑 `go build ./...` + `go vet ./...` 必须通过；测试用 `go test ./cmd/relay-claude/ -run <Name> -v`，从 repo 根目录执行。
- 不变量(spec §3.4): 只要进程累计单调，`SUM(所有 delta) == 进程最后累计 == 真实总量`；中断/无-result turn 的用量被下一成功 turn 吸收（接受归属偏移）。

---

### Task 1: usageMeter 接口与三种 meter

**Files:**
- Create: `cmd/relay-claude/usage_meter.go`
- Test: `cmd/relay-claude/usage_meter_test.go`

**Interfaces:**
- Produces: `usageSnapshot{input,output,cacheCreation,cacheRead int; costUSD float64; fromModelUsage bool}`；`usageMeter` 接口含 `perTurn(usageSnapshot) usageSnapshot`；`identityMeter{}`；`cumulativeMeter{}`（带内部 baseline）。

- [ ] **Step 1: 写失败测试**

创建 `cmd/relay-claude/usage_meter_test.go`:

```go
package main

import "testing"

func TestIdentityMeterPassesThrough(t *testing.T) {
	cur := usageSnapshot{input: 7, output: 3, cacheRead: 9, cacheCreation: 1, costUSD: 0.5}
	if got := (identityMeter{}).perTurn(cur); got != cur {
		t.Fatalf("identityMeter changed snapshot: %+v != %+v", got, cur)
	}
}

func TestCumulativeMeterFirstTurnEqualsCur(t *testing.T) {
	m := &cumulativeMeter{}
	d := m.perTurn(usageSnapshot{input: 100, output: 50, cacheRead: 1000, cacheCreation: 200, costUSD: 0.10})
	if d.input != 100 || d.output != 50 || d.cacheRead != 1000 || d.cacheCreation != 200 {
		t.Fatalf("first-turn delta = %+v, want raw values", d)
	}
}

func TestCumulativeMeterDiffsMonotonicGrowth(t *testing.T) {
	m := &cumulativeMeter{}
	m.perTurn(usageSnapshot{input: 100, output: 50, cacheRead: 1000, cacheCreation: 200, costUSD: 0.10})
	d := m.perTurn(usageSnapshot{input: 106, output: 90, cacheRead: 4500, cacheCreation: 200, costUSD: 0.18})
	if d.input != 6 || d.output != 40 || d.cacheRead != 3500 || d.cacheCreation != 0 {
		t.Fatalf("second-turn delta = %+v, want increments {6,40,3500,0}", d)
	}
	if d.costUSD < 0.0799 || d.costUSD > 0.0801 {
		t.Fatalf("cost delta = %f, want ~0.08", d.costUSD)
	}
}

func TestCumulativeMeterResetsOnInputRegression(t *testing.T) {
	m := &cumulativeMeter{}
	m.perTurn(usageSnapshot{input: 80000, output: 200000, cacheRead: 47000000})
	d := m.perTurn(usageSnapshot{input: 1300, output: 10000, cacheRead: 4700000})
	if d.input != 1300 || d.output != 10000 || d.cacheRead != 4700000 {
		t.Fatalf("post-compact delta = %+v, want the cur snapshot (no negatives)", d)
	}
}

func TestCumulativeMeterResetsOnSourceShapeChange(t *testing.T) {
	m := &cumulativeMeter{}
	m.perTurn(usageSnapshot{input: 100, output: 50, fromModelUsage: false})
	d := m.perTurn(usageSnapshot{input: 5000, output: 9000, fromModelUsage: true})
	if d.input != 5000 || d.output != 9000 {
		t.Fatalf("shape-change delta = %+v, want the cur snapshot", d)
	}
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `go test ./cmd/relay-claude/ -run 'TestIdentityMeter|TestCumulativeMeter' -v`
Expected: 编译失败 `undefined: usageSnapshot` / `undefined: cumulativeMeter`。

- [ ] **Step 3: 写实现**

创建 `cmd/relay-claude/usage_meter.go`:

```go
package main

import "sync"

// usageSnapshot is one turn's token + cost figures plus a shape tag. All fields
// are comparable, so snapshots can be compared with ==.
type usageSnapshot struct {
	input          int
	output         int
	cacheCreation  int
	cacheRead      int
	costUSD        float64
	fromModelUsage bool // derived from the modelUsage aggregate vs the bare usage field
}

// usageMeter converts a raw usage snapshot (which may be process-cumulative)
// into the per-turn usage to report. Implementations may carry cross-turn state.
type usageMeter interface {
	perTurn(cur usageSnapshot) usageSnapshot
}

// identityMeter passes the snapshot through unchanged. Used by V1 (fresh process
// per request) and channel-ephemeral (single-turn throwaway process), where the
// raw usage is already the per-turn figure.
type identityMeter struct{}

func (identityMeter) perTurn(cur usageSnapshot) usageSnapshot { return cur }

// cumulativeMeter diffs a process-cumulative usage stream into per-turn deltas.
// Owned by a persistent chanWorker; its lifetime is the claude process's, so a
// respawned worker starts from a zero baseline, matching the new process's own
// zero-based cumulative counter.
type cumulativeMeter struct {
	mu   sync.Mutex
	last usageSnapshot
}

func (m *cumulativeMeter) perTurn(cur usageSnapshot) usageSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()

	var d usageSnapshot
	d.fromModelUsage = cur.fromModelUsage

	// Cost is monotonic (a turn never refunds), so always diff and clamp at 0.
	d.costUSD = cur.costUSD - m.last.costUSD
	if d.costUSD < 0 {
		d.costUSD = 0
	}

	// Reset detection. A source-shape change (modelUsage <-> bare usage) makes
	// the cumulative figures non-comparable; an input regression means claude
	// reset its in-process counter (auto-compact / clear). Either way `cur` is
	// itself the per-turn figure for this turn.
	if cur.fromModelUsage != m.last.fromModelUsage || cur.input < m.last.input {
		d.input, d.output = cur.input, cur.output
		d.cacheCreation, d.cacheRead = cur.cacheCreation, cur.cacheRead
	} else {
		d.input = cur.input - m.last.input
		d.output = cur.output - m.last.output
		d.cacheCreation = cur.cacheCreation - m.last.cacheCreation
		d.cacheRead = cur.cacheRead - m.last.cacheRead
	}

	m.last = cur
	return d
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `go test ./cmd/relay-claude/ -run 'TestIdentityMeter|TestCumulativeMeter' -v`
Expected: 5 个测试全 PASS。

- [ ] **Step 5: 提交**

```bash
go vet ./...
git add cmd/relay-claude/usage_meter.go cmd/relay-claude/usage_meter_test.go
git commit -m "feat(relay-claude): usageMeter 接口 + identity/cumulative-diff meter"
```

---

### Task 2: chanWorker 持有 cumulativeMeter

**Files:**
- Modify: `cmd/relay-claude/channel.go`（struct `chanWorker` 50-80；`spawnChanWorker` 138-147 的 struct 字面量）

**Interfaces:**
- Consumes: `cumulativeMeter`（Task 1）
- Produces: `chanWorker.meter *cumulativeMeter`（每个常驻进程一个，spawn 时初始化）

- [ ] **Step 1: 加字段**

在 `chanWorker` struct 末尾（`channel.go` 当前第 79 行 `cur *activeTurn` 之后、闭合 `}` 之前）加:

```go
	// meter diffs this persistent process's cumulative result.usage into
	// per-turn deltas. Its lifetime == the process's, so a respawned worker
	// (new struct) starts from a zero baseline. See usage_meter.go.
	meter *cumulativeMeter
```

- [ ] **Step 2: spawn 时初始化**

在 `spawnChanWorker` 的 `w := &chanWorker{...}` 字面量里（`channel.go` 138-147），在 `deadCh: make(chan struct{}),` 之后加一行:

```go
		meter:     &cumulativeMeter{},
```

- [ ] **Step 3: 构建验证（字段暂未被引用，Go 允许）**

Run: `go build ./... && go vet ./...`
Expected: 无报错（结构体字段可暂时未被读取）。

- [ ] **Step 4: 提交**

```bash
git add cmd/relay-claude/channel.go
git commit -m "feat(relay-claude): chanWorker 持有 per-process cumulativeMeter"
```

---

### Task 3: 把 meter 接入 sseTranslator + 修 gating bug

**Files:**
- Modify: `cmd/relay-claude/translate.go`（struct 44-63；`newSSETranslator` 65-75；`feed` result 处理 271-296）
- Modify: `cmd/relay-claude/stream.go:52`
- Modify: `cmd/relay-claude/channel_handler.go:227,389`
- Modify: `cmd/relay-claude/channel_test.go:451,480`
- Test: `cmd/relay-claude/channel_test.go`（新增 2 个用例）

**Interfaces:**
- Consumes: `usageMeter`/`identityMeter`/`usageSnapshot`（Task 1）、`chanWorker.meter`（Task 2）
- Produces: `newSSETranslator(chatID string, created int64, model, sessionID string, meter usageMeter) *sseTranslator`（新签名，末尾加 meter）

- [ ] **Step 1: 写失败测试**

在 `cmd/relay-claude/channel_test.go` 末尾追加:

```go
// 同一 cumulativeMeter 跨两轮 feed：第二轮 emit 的是 delta，不是累计。
func TestSSETranslatorChannelDiffsCumulativeUsage(t *testing.T) {
	sessionStore = sessions.New(t.TempDir())
	meter := &cumulativeMeter{}

	rec1 := httptest.NewRecorder()
	tr1 := newSSETranslator("c1", 1700000000, "haiku", "", meter)
	tr1.feed(rec1, rec1, `{"type":"result","usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":1000}}`, true)

	rec2 := httptest.NewRecorder()
	tr2 := newSSETranslator("c2", 1700000000, "haiku", "", meter)
	tr2.feed(rec2, rec2, `{"type":"result","usage":{"input_tokens":106,"output_tokens":90,"cache_read_input_tokens":4500}}`, true)

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
	tr := newSSETranslator("c", 1700000000, "haiku", "", identityMeter{})
	line := `{"type":"result","modelUsage":{"claude-haiku":{"inputTokens":10,"outputTokens":5,"cacheReadInputTokens":20}}}`
	tr.feed(rec, rec, line, true)
	if !strings.Contains(rec.Body.String(), `"prompt_tokens"`) {
		t.Errorf("usage chunk missing when only modelUsage present (gating bug):\n%s", rec.Body.String())
	}
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `go test ./cmd/relay-claude/ -run 'TestSSETranslatorChannelDiffs|TestSSETranslatorEmitsUsageFromModelUsage' -v`
Expected: 编译失败 —— `newSSETranslator` 实参数量不符（旧签名 4 参）。

- [ ] **Step 3: 改 sseTranslator struct + 构造函数**

`translate.go`：在 struct 字段 `sessionID string`（第 48 行）之后加 `meter usageMeter`:

```go
	chatID    string
	created   int64
	model     string
	sessionID string
	meter     usageMeter
```

改 `newSSETranslator`（65-75）签名与 body:

```go
func newSSETranslator(chatID string, created int64, model, sessionID string, meter usageMeter) *sseTranslator {
	if meter == nil {
		meter = identityMeter{}
	}
	return &sseTranslator{
		chatID:        chatID,
		created:       created,
		model:         model,
		sessionID:     sessionID,
		meter:         meter,
		seenToolNames: map[string]bool{},
		askUserIdx:    -1,
		toolBlocks:    map[int]*toolBlock{},
	}
}
```

- [ ] **Step 4: 改 feed() 的 result 处理（差分 + gating 修复）**

把 `translate.go` 271-296 整段替换为:

```go
	if event.Type == "result" {
		if eu := effectiveUsage(&event); eu != nil {
			cur := usageSnapshot{
				input:          eu.InputTokens,
				output:         eu.OutputTokens,
				cacheCreation:  eu.CacheCreationInputTokens,
				cacheRead:      eu.CacheReadInputTokens,
				costUSD:        event.TotalCostUSD,
				fromModelUsage: len(event.ModelUsage) > 0,
			}
			d := t.meter.perTurn(cur)
			stats.Record(t.model, d.input, d.output, d.cacheCreation, d.cacheRead, d.costUSD)
			log.Printf("Token usage: model=%s input=%d output=%d cache_read=%d cache_create=%d cost=$%.4f raw_input=%d raw_cache_read=%d (subagents=%v)",
				t.model, d.input, d.output, d.cacheRead, d.cacheCreation, d.costUSD,
				cur.input, cur.cacheRead, cur.fromModelUsage)
			t.streamUsage = openai.BuildUsageInfo(d.input, d.output, d.cacheRead, d.cacheCreation)
		}

		finishReason := "stop"
		t.emit(w, flusher, openai.ChatCompletionResponse{
			ID: t.chatID, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
			Choices: []openai.ChatCompletionChoice{
				{Index: 0, Delta: openai.NewChatMessage("", ""), FinishReason: &finishReason},
			},
		})

		if includeUsage && t.streamUsage != nil {
			t.emit(w, flusher, openai.ChatCompletionResponse{
				ID: t.chatID, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
				Choices: []openai.ChatCompletionChoice{},
				Usage:   t.streamUsage,
			})
		}
	}
```

（与原版差异: 引入 `cur`/`d`，Record/BuildUsageInfo 改用 `d.*`，log 加 `raw_*` 供 spec §8 交叉验证，emit 的 gate 从 `event.Usage != nil` 改为 `t.streamUsage != nil`。）

- [ ] **Step 5: 更新 4 处现有调用点的签名**

`stream.go:52`（V1，新进程单轮 → identity）:
```go
	t := newSSETranslator(chatID, created, model, sessionID, identityMeter{})
```

`channel_handler.go:227`（ephemeral，单轮 throwaway 进程 → identity）:
```go
	t := newSSETranslator(chatID, created, model, "", identityMeter{}) // no session → log no-ops
```

`channel_handler.go:389`（persistent V2 → 用该 worker 的 cumulativeMeter）:
```go
	t := newSSETranslator(chatID, created, model, sessionID, worker.meter)
```

`channel_test.go:451`:
```go
	tr := newSSETranslator("chatcmpl-x", 1700000000, "haiku", "", identityMeter{}) // empty sessionID → log no-ops
```

`channel_test.go:480`:
```go
	tr := newSSETranslator("chatcmpl-y", 1700000000, "haiku", "", identityMeter{})
```

- [ ] **Step 6: 运行测试确认通过**

Run: `go test ./cmd/relay-claude/ -run 'TestSSETranslator' -v`
Expected: 全 PASS（含旧 `TestSSETranslatorTextAndResult`/`TestSSETranslatorAskUserQuestion` 仍绿 + 两个新用例）。

- [ ] **Step 7: 全量构建 + 测试**

Run: `go build ./... && go vet ./... && go test ./cmd/relay-claude/ -v`
Expected: 构建通过，包内测试全 PASS。

- [ ] **Step 8: 提交**

```bash
git add cmd/relay-claude/translate.go cmd/relay-claude/stream.go cmd/relay-claude/channel_handler.go cmd/relay-claude/channel_test.go
git commit -m "fix(relay-claude): channel 模式 usage 跨轮差分 + 修 usage chunk gating"
```

---

### Task 4: 后台 drain 推进 baseline（spec §3.5）

中断（client disconnect / AskUserQuestion）时后台 drainer 丢弃 result，使下一前台 turn 的 delta 含被丢轮的累计。让 drainer 解析 result 推进 baseline，减轻归属偏移（不影响总量正确性，属优化项）。

**Files:**
- Modify: `cmd/relay-claude/usage_meter.go`（新增 `advanceMeterFromLine` 帮助函数）
- Modify: `cmd/relay-claude/channel_handler.go`（`releaseInBackground` 的 goroutine 345-362）
- Test: `cmd/relay-claude/usage_meter_test.go`

**Interfaces:**
- Produces: `advanceMeterFromLine(meter usageMeter, line string)` —— 解析一行 stream-json，若是带 usage 的 result 则推进 cumulative baseline（丢弃 delta）；非 cumulativeMeter 时 no-op。

- [ ] **Step 1: 写失败测试**

在 `usage_meter_test.go` 末尾追加:

```go
func TestAdvanceMeterFromLineAbsorbsDroppedTurn(t *testing.T) {
	m := &cumulativeMeter{}
	// 一个被中断、未走 feed 的 turn：累计推进到 input=500。
	advanceMeterFromLine(m, `{"type":"result","usage":{"input_tokens":500,"output_tokens":300,"cache_read_input_tokens":9000}}`)
	// 下一前台 turn 的累计是 input=506，delta 应只含本轮增量，不含被丢轮的 500。
	d := m.perTurn(usageSnapshot{input: 506, output: 340, cacheRead: 9500})
	if d.input != 6 || d.output != 40 || d.cacheRead != 500 {
		t.Fatalf("delta after background-drain advance = %+v, want {6,40,500}", d)
	}
}

func TestAdvanceMeterFromLineIgnoresNonResult(t *testing.T) {
	m := &cumulativeMeter{}
	advanceMeterFromLine(m, `{"type":"stream_event"}`)
	advanceMeterFromLine(m, `not json`)
	d := m.perTurn(usageSnapshot{input: 100, output: 50})
	if d.input != 100 {
		t.Fatalf("baseline moved on non-result line: delta=%+v", d)
	}
}
```

- [ ] **Step 2: 运行确认失败**

Run: `go test ./cmd/relay-claude/ -run 'TestAdvanceMeterFromLine' -v`
Expected: 编译失败 `undefined: advanceMeterFromLine`。

- [ ] **Step 3: 实现帮助函数**

在 `usage_meter.go` 末尾追加（需要 `encoding/json`，加到 import）:

```go
// advanceMeterFromLine parses a stream-json line and, if it is a result with
// usage, advances a cumulativeMeter's baseline (discarding the per-turn delta).
// Used by the background drainer of an interrupted turn so its cumulative usage
// is absorbed into the baseline instead of bleeding into the next foreground
// turn. No-op for non-cumulative meters.
func advanceMeterFromLine(meter usageMeter, line string) {
	cm, ok := meter.(*cumulativeMeter)
	if !ok {
		return
	}
	var event claudeEvent
	if err := json.Unmarshal([]byte(line), &event); err != nil || event.Type != "result" {
		return
	}
	eu := effectiveUsage(&event)
	if eu == nil {
		return
	}
	cm.perTurn(usageSnapshot{
		input:          eu.InputTokens,
		output:         eu.OutputTokens,
		cacheCreation:  eu.CacheCreationInputTokens,
		cacheRead:      eu.CacheReadInputTokens,
		costUSD:        event.TotalCostUSD,
		fromModelUsage: len(event.ModelUsage) > 0,
	})
}
```

把 `usage_meter.go` 的 import 从 `import "sync"` 改为:

```go
import (
	"encoding/json"
	"sync"
)
```

- [ ] **Step 4: 接入 releaseInBackground**

`channel_handler.go` 的 `releaseInBackground` goroutine（345-362），把读取 `lines` 的 `case` 改为在收到行时推进 baseline:

```go
			case ln, ok := <-lines:
				if !ok {
					timer.Stop()
					worker.endTurn()
					return
				}
				advanceMeterFromLine(worker.meter, ln)
```

（原代码 `case _, ok := <-lines:` 丢弃行；改为 `ln, ok` 并对每行调用 `advanceMeterFromLine`。）

- [ ] **Step 5: 运行测试确认通过**

Run: `go test ./cmd/relay-claude/ -run 'TestAdvanceMeterFromLine' -v`
Expected: 2 个测试 PASS。

- [ ] **Step 6: 全量构建 + 测试 + 提交**

```bash
go build ./... && go vet ./... && go test ./cmd/relay-claude/ -v
git add cmd/relay-claude/usage_meter.go cmd/relay-claude/usage_meter_test.go cmd/relay-claude/channel_handler.go
git commit -m "fix(relay-claude): 中断后台 drain 推进 usage baseline 减轻归属偏移"
```

---

### Task 5: V3 不再 emit 假 0 usage

**Files:**
- Modify: `cmd/relay-claude/channelv3.go`（`v3EmitClose` 的 `includeUsage` 块）
- Test: `cmd/relay-claude/channelv3_test.go`（若不存在则创建）

**Interfaces:**
- Consumes: 无新依赖。
- Produces: `v3EmitClose` 行为变更——不再 emit usage chunk；下游因此写 NULL（未计量）而非 0。

- [ ] **Step 1: 写失败测试**

若 `cmd/relay-claude/channelv3_test.go` 不存在则创建；否则在末尾追加:

```go
package main

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestV3EmitCloseOmitsUsage(t *testing.T) {
	rec := httptest.NewRecorder()
	v3EmitClose(rec, rec, "c", 1700000000, "haiku", "hello", true) // includeUsage=true
	body := rec.Body.String()
	if strings.Contains(body, `"prompt_tokens"`) {
		t.Errorf("V3 must not emit usage (would log as 0, not NULL):\n%s", body)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Errorf("missing finish chunk:\n%s", body)
	}
	if !strings.Contains(body, `data: [DONE]`) {
		t.Errorf("missing [DONE]:\n%s", body)
	}
}
```

（若文件已存在且已有 `package main` 头与这些 import，仅追加函数体，勿重复包声明/import。）

- [ ] **Step 2: 运行确认失败**

Run: `go test ./cmd/relay-claude/ -run 'TestV3EmitCloseOmitsUsage' -v`
Expected: FAIL —— body 含 `"prompt_tokens"`（当前发空 `Usage{}`）。

- [ ] **Step 3: 删除 emit 空 usage 的分支**

在 `channelv3.go` 的 `v3EmitClose` 里，删除整段:

```go
	if includeUsage {
		// Channels don't surface token counts; report zeros so the shape holds.
		usage := openai.ChatCompletionResponse{
			ID: chatID, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []openai.ChatCompletionChoice{}, Usage: &openai.UsageInfo{},
		}
		data, _ = json.Marshal(usage)
		fmt.Fprintf(w, "data: %s\n\n", data)
	}
```

替换为一行注释（保留 `includeUsage` 参数不动，调用方签名不变）:

```go
	// V3 has no token source (interactive PTY, no stream-json). Emit NO usage
	// chunk so downstream stores NULL ("not metered"), not 0 ("free request").
	_ = includeUsage
```

- [ ] **Step 4: 运行测试确认通过**

Run: `go test ./cmd/relay-claude/ -run 'TestV3EmitCloseOmitsUsage' -v`
Expected: PASS。

- [ ] **Step 5: 全量构建 + 测试 + 提交**

```bash
go build ./... && go vet ./... && go test ./cmd/relay-claude/ -v
git add cmd/relay-claude/channelv3.go cmd/relay-claude/channelv3_test.go
git commit -m "fix(relay-claude): V3 不再 emit 假 0 usage，下游写 NULL 标注未计量"
```

---

### Task 6: 部署与线上交叉验证（spec §7、§8）

非编码任务，按 `docs/ops/clawrelay-*.md` SOP 执行。

- [ ] **Step 1: 构建二进制**

Run: `go build ./... && go test ./cmd/relay-claude/ -v`
Expected: 全绿。按现有发布流程产出 relay-claude 二进制。

- [ ] **Step 2: 部署到 channel(claude02/50009) 与 channelv3(claude01/50008) 实例**（按 ops SOP，重启前置检查勿跳过）。

- [ ] **Step 3: 验证 V2（claude02/50009）**

部署后取若干同 session 多轮记录，确认 `robot_chat_logs` 内同 session 的 `input_tokens` 固定时 `cache_read_tokens` 不再单调累计、逐轮值落回单轮规模。比对 relay 日志的 `raw_input=` vs `input=`(delta) 验证差分生效（spec §8）。

- [ ] **Step 4: 验证 V3（claude01/50008）**

确认新记录的 token 四字段写入 `NULL` 而非 0。

---

## Self-Review

**1. Spec coverage**

- §2 usageMeter 接口 → Task 1 ✅
- §3 baseline 落 chanWorker → Task 2 ✅；差分在 emit 前 → Task 3 Step 4 ✅
- §3.1 累计回退 → Task 1 `TestCumulativeMeterResetsOnInputRegression` ✅
- §3.2 source-shape 一致性 → Task 1 `TestCumulativeMeterResetsOnSourceShapeChange` ✅
- §3.3 cost diff → Task 1 cost 分支 + `TestCumulativeMeterDiffsMonotonicGrowth` 断言 ✅
- §3.4 中断归属偏移（不变量） → 设计选择，Task 4 在此前提下减轻偏移 ✅
- §3.5 后台 drain → Task 4 ✅
- §3.6 gating 修复 → Task 3 Step 4 + `TestSSETranslatorEmitsUsageFromModelUsage` ✅
- §3.7 进程重启归零 → Task 2 spawn 新 struct 新 meter（注释说明）+ Task 1 首轮=cur 语义 ✅
- §4 V3 nil meter → Task 5 ✅
- §6 测试清单 → Task 1/3/4/5 各覆盖 ✅
- §7 部署验证 → Task 6 ✅
- §8 DEBUG 日志 raw vs delta → Task 3 Step 4 log 行 ✅

**2. Placeholder scan:** 无 TBD/TODO；每个 code step 含完整代码与确切命令/预期。

**3. Type consistency:** `usageSnapshot`/`usageMeter`/`perTurn`/`cumulativeMeter`/`identityMeter`/`advanceMeterFromLine` 跨 Task 命名一致；`newSSETranslator` 新签名（5 参）在 Task 3 统一更新全部 6 处调用；`stats.Record`/`BuildUsageInfo` 参数顺序与 Global Constraints 一致。
