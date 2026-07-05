package main

import (
	"encoding/json"
	"log"
	"sync"
)

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

// cumulativeMeter converts a persistent process's result-usage stream into
// per-turn deltas. Owned by a chanWorker; its lifetime is the claude process's,
// so a respawned worker (incl. --resume) starts from a zero baseline, matching
// the new process's own zero-based cumulative counters.
//
// 实证口径（claude 2.1.199，写死进设计里）：
//   - result 的 modelUsage 聚合与顶层 total_cost_usd 是【进程级单调累计】；
//   - result 的 bare usage 字段是【per-turn 值】（不是累计）；
//   - 两种 shape 会在同一进程内交替出现（例如无工具的简短轮只给 bare usage）。
//
// 因此 token 基线（modelBase）只跟踪 fromModelUsage=true 的流；bare 轮直接取原
// 值、不动基线 —— 旧实现把 shape 翻转当 reset、把翻回来的累计值整段当全量，
// 一轮就能虚报几万 cache token，这是本次重写要杀死的 bug。
type cumulativeMeter struct {
	mu           sync.Mutex
	lastCost     float64
	hasModelBase bool
	modelBase    usageSnapshot // 仅 fromModelUsage=true 流的基线
}

func clampNonNeg(v int) int {
	if v < 0 {
		return 0
	}
	return v
}

func (m *cumulativeMeter) perTurn(cur usageSnapshot) usageSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()

	var d usageSnapshot
	d.fromModelUsage = cur.fromModelUsage

	// Cost 永远全局差分并 clamp≥0：bare result 的 total_cost_usd 也是进程级
	// 累计口径（与 modelUsage 同源），shape 无关。
	d.costUSD = cur.costUSD - m.lastCost
	if d.costUSD < 0 {
		d.costUSD = 0
	}
	m.lastCost = cur.costUSD

	if !cur.fromModelUsage {
		// bare usage 视为 per-turn 值（基于 2.1.199 实证），直接采用，不动
		// modelBase。代价：bare 间奏轮的 token 会被下一次 modelUsage 差分重复
		// 覆盖（modelUsage 累计包含了这轮），量级可忽略；换来 shape 翻转时不再
		// 把整段累计当全量记账。
		d.input, d.output = cur.input, cur.output
		d.cacheCreation, d.cacheRead = cur.cacheCreation, cur.cacheRead
		return d
	}

	// modelUsage 流：无基线（本进程首个 modelUsage result），或 input 回退
	// （进程内 reset：auto-compact / clear）→ cur 本身就是本轮全量。
	if !m.hasModelBase || cur.input < m.modelBase.input {
		d.input, d.output = cur.input, cur.output
		d.cacheCreation, d.cacheRead = cur.cacheCreation, cur.cacheRead
	} else {
		// 逐字段差分且每字段 clamp≥0：个别字段（如 cacheCreation）可能因
		// compact 单独回退，不能让负数污染全局累加器。
		d.input = clampNonNeg(cur.input - m.modelBase.input)
		d.output = clampNonNeg(cur.output - m.modelBase.output)
		d.cacheCreation = clampNonNeg(cur.cacheCreation - m.modelBase.cacheCreation)
		d.cacheRead = clampNonNeg(cur.cacheRead - m.modelBase.cacheRead)
	}
	m.modelBase = cur
	m.hasModelBase = true
	return d
}

// recordInterruptedTurnUsage parses a stream-json line from an interrupted
// turn's background drainer and, if it is a result with usage, advances the
// cumulativeMeter's baseline AND records the turn's delta into stats — an
// interrupted turn produced no foreground accounting at all, so this is its
// only chance to be billed (SIGINT/stdin-interrupt 后 claude 仍会 emit result，
// 见 proc.InterruptGroup 注释). No-op for non-result lines and non-cumulative
// meters (identityMeter callers harvest via perModelCounts directly).
func recordInterruptedTurnUsage(meter usageMeter, line, model string) {
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
	d := cm.perTurn(usageSnapshot{
		input:          eu.InputTokens,
		output:         eu.OutputTokens,
		cacheCreation:  eu.CacheCreationInputTokens,
		cacheRead:      eu.CacheReadInputTokens,
		costUSD:        event.TotalCostUSD,
		fromModelUsage: len(event.ModelUsage) > 0,
	})
	if d.input > 0 || d.output > 0 || d.cacheCreation > 0 || d.cacheRead > 0 || d.costUSD > 0 {
		stats.Record(model, d.input, d.output, d.cacheCreation, d.cacheRead, d.costUSD)
		log.Printf("[channel] interrupted-turn usage recorded: model=%s input=%d output=%d cache_read=%d cache_create=%d cost=$%.4f",
			model, d.input, d.output, d.cacheRead, d.cacheCreation, d.costUSD)
	}
}
