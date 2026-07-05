package main

import (
	"math"
	"testing"
)

func TestIdentityMeterPassesThrough(t *testing.T) {
	cur := usageSnapshot{input: 7, output: 3, cacheRead: 9, cacheCreation: 1, costUSD: 0.5}
	if got := (identityMeter{}).perTurn(cur); got != cur {
		t.Fatalf("identityMeter changed snapshot: %+v != %+v", got, cur)
	}
}

func TestCumulativeMeterFirstModelUsageTurnEqualsCur(t *testing.T) {
	m := &cumulativeMeter{}
	d := m.perTurn(usageSnapshot{input: 100, output: 50, cacheRead: 1000, cacheCreation: 200, costUSD: 0.10, fromModelUsage: true})
	if d.input != 100 || d.output != 50 || d.cacheRead != 1000 || d.cacheCreation != 200 {
		t.Fatalf("first-turn delta = %+v, want raw values", d)
	}
	if math.Abs(d.costUSD-0.10) > 1e-9 {
		t.Fatalf("first-turn cost = %f, want 0.10", d.costUSD)
	}
}

func TestCumulativeMeterDiffsMonotonicGrowth(t *testing.T) {
	m := &cumulativeMeter{}
	m.perTurn(usageSnapshot{input: 100, output: 50, cacheRead: 1000, cacheCreation: 200, costUSD: 0.10, fromModelUsage: true})
	d := m.perTurn(usageSnapshot{input: 106, output: 90, cacheRead: 4500, cacheCreation: 200, costUSD: 0.18, fromModelUsage: true})
	if d.input != 6 || d.output != 40 || d.cacheRead != 3500 || d.cacheCreation != 0 {
		t.Fatalf("second-turn delta = %+v, want increments {6,40,3500,0}", d)
	}
	if math.Abs(d.costUSD-0.08) > 1e-9 {
		t.Fatalf("cost delta = %f, want ~0.08", d.costUSD)
	}
}

func TestCumulativeMeterResetsOnInputRegression(t *testing.T) {
	// 进程内 reset（auto-compact / clear）：modelUsage 累计 input 回退 → cur 即全量。
	m := &cumulativeMeter{}
	m.perTurn(usageSnapshot{input: 80000, output: 200000, cacheRead: 47000000, fromModelUsage: true})
	d := m.perTurn(usageSnapshot{input: 1300, output: 10000, cacheRead: 4700000, fromModelUsage: true})
	if d.input != 1300 || d.output != 10000 || d.cacheRead != 4700000 {
		t.Fatalf("post-compact delta = %+v, want the cur snapshot (no negatives)", d)
	}
}

// 本次修复的核心回归用例：shape 翻转 bare→modelUsage 不再把累计当全量。
// 实证（claude 2.1.199）：bare usage 是 per-turn 值，modelUsage 是进程累计；
// 旧实现把 shape 变化当 reset，T3 会整段 {1100,1100,66573,8100} 被记成全量。
func TestCumulativeMeterShapeFlipDoesNotDoubleCount(t *testing.T) {
	m := &cumulativeMeter{}

	// T1: modelUsage 累计 {542,516,17157,7563} → 无基线，记全量。
	d1 := m.perTurn(usageSnapshot{input: 542, output: 516, cacheRead: 17157, cacheCreation: 7563, costUSD: 0.10, fromModelUsage: true})
	if d1.input != 542 || d1.output != 516 || d1.cacheRead != 17157 || d1.cacheCreation != 7563 {
		t.Fatalf("T1 delta = %+v, want full modelUsage values", d1)
	}

	// T2: bare {10,38,0,0}（简短间奏轮）→ per-turn 原值，不动 modelUsage 基线。
	d2 := m.perTurn(usageSnapshot{input: 10, output: 38, costUSD: 0.12, fromModelUsage: false})
	if d2.input != 10 || d2.output != 38 || d2.cacheRead != 0 || d2.cacheCreation != 0 {
		t.Fatalf("T2 delta = %+v, want the bare per-turn values {10,38,0,0}", d2)
	}

	// T3: modelUsage 累计 {1100,1100,66573,8100} → 只记 T3-T1 差分，
	// 绝不能把 T3 整段当全量（旧 bug 一轮虚报 ~5 万 cache token）。
	d3 := m.perTurn(usageSnapshot{input: 1100, output: 1100, cacheRead: 66573, cacheCreation: 8100, costUSD: 0.30, fromModelUsage: true})
	if d3.input != 558 || d3.output != 584 || d3.cacheRead != 49416 || d3.cacheCreation != 537 {
		t.Fatalf("T3 delta = %+v, want T3-T1 diff {558,584,49416,537}", d3)
	}

	// cost 全程全局差分：0.10 → 0.02 → 0.18。
	if math.Abs(d1.costUSD-0.10) > 1e-9 || math.Abs(d2.costUSD-0.02) > 1e-9 || math.Abs(d3.costUSD-0.18) > 1e-9 {
		t.Fatalf("cost deltas = {%f,%f,%f}, want {0.10,0.02,0.18}", d1.costUSD, d2.costUSD, d3.costUSD)
	}
}

func TestCumulativeMeterClampsPerFieldNegatives(t *testing.T) {
	// input 未回退（不触发 reset），但个别字段回退 → 逐字段 clamp≥0。
	m := &cumulativeMeter{}
	m.perTurn(usageSnapshot{input: 100, output: 50, cacheRead: 1000, cacheCreation: 200, fromModelUsage: true})
	d := m.perTurn(usageSnapshot{input: 150, output: 60, cacheRead: 900, cacheCreation: 150, fromModelUsage: true})
	if d.input != 50 || d.output != 10 || d.cacheRead != 0 || d.cacheCreation != 0 {
		t.Fatalf("delta = %+v, want per-field clamped {50,10,0,0}", d)
	}
}

func TestCumulativeMeterCostClampsNegative(t *testing.T) {
	m := &cumulativeMeter{}
	m.perTurn(usageSnapshot{costUSD: 0.50, fromModelUsage: true})
	d := m.perTurn(usageSnapshot{costUSD: 0.10, fromModelUsage: true})
	if d.costUSD != 0 {
		t.Fatalf("regressed cost delta = %f, want clamp to 0", d.costUSD)
	}
	// 基线已推进到 0.10：下一轮 0.15 → 差分 0.05。
	d = m.perTurn(usageSnapshot{costUSD: 0.15, fromModelUsage: true})
	if math.Abs(d.costUSD-0.05) > 1e-9 {
		t.Fatalf("post-clamp cost delta = %f, want 0.05", d.costUSD)
	}
}

// ---- recordInterruptedTurnUsage (A7) ----

func statsModelRow(model string) (requests, total int64, cost float64) {
	snap := stats.Snapshot()
	row, ok := snap.PerModel[model]
	if !ok {
		return 0, 0, 0
	}
	return row.Requests, row.Total, row.CostUSD
}

func TestRecordInterruptedTurnUsageAdvancesBaselineAndRecords(t *testing.T) {
	const model = "test-interrupted-a7"
	m := &cumulativeMeter{}
	// 先有一轮前台记账：modelUsage 累计推进到 input=500。
	m.perTurn(usageSnapshot{input: 500, output: 300, cacheRead: 9000, costUSD: 0.10, fromModelUsage: true})

	_, totalBefore, costBefore := statsModelRow(model)
	// 被中断的一轮：SIGINT/stdin-interrupt 后残余 result（modelUsage 有真实值）。
	recordInterruptedTurnUsage(m, `{"type":"result","subtype":"error_during_execution","modelUsage":{"claude-x":{"inputTokens":520,"outputTokens":340,"cacheReadInputTokens":9500}},"total_cost_usd":0.14}`, model)
	_, totalAfter, costAfter := statsModelRow(model)

	// 差分 {20,40,500} = 560 tokens、$0.04 被记入 stats。
	if totalAfter-totalBefore != 560 {
		t.Fatalf("stats total delta = %d, want 560", totalAfter-totalBefore)
	}
	if math.Abs((costAfter-costBefore)-0.04) > 1e-9 {
		t.Fatalf("stats cost delta = %f, want 0.04", costAfter-costBefore)
	}

	// 基线同步推进：下一前台轮只记本轮增量。
	d := m.perTurn(usageSnapshot{input: 526, output: 380, cacheRead: 10000, costUSD: 0.15, fromModelUsage: true})
	if d.input != 6 || d.output != 40 || d.cacheRead != 500 {
		t.Fatalf("delta after interrupted-turn record = %+v, want {6,40,500}", d)
	}
}

func TestRecordInterruptedTurnUsageIgnoresNonResult(t *testing.T) {
	const model = "test-interrupted-nonresult"
	m := &cumulativeMeter{}
	reqBefore, totalBefore, _ := statsModelRow(model)
	recordInterruptedTurnUsage(m, `{"type":"stream_event"}`, model)
	recordInterruptedTurnUsage(m, `not json`, model)
	reqAfter, totalAfter, _ := statsModelRow(model)
	if reqAfter != reqBefore || totalAfter != totalBefore {
		t.Fatal("non-result lines must not touch stats")
	}
	d := m.perTurn(usageSnapshot{input: 100, output: 50, fromModelUsage: true})
	if d.input != 100 {
		t.Fatalf("baseline moved on non-result line: delta=%+v", d)
	}
}

func TestRecordInterruptedTurnUsageZeroDeltaSkipsStats(t *testing.T) {
	const model = "test-interrupted-zerodelta"
	m := &cumulativeMeter{}
	// 基线推进到与残余 result 相同的累计值 → 差分全 0，不应记账。
	m.perTurn(usageSnapshot{input: 500, output: 300, costUSD: 0.10, fromModelUsage: true})
	reqBefore, totalBefore, _ := statsModelRow(model)
	recordInterruptedTurnUsage(m, `{"type":"result","modelUsage":{"claude-x":{"inputTokens":500,"outputTokens":300}},"total_cost_usd":0.10}`, model)
	reqAfter, totalAfter, _ := statsModelRow(model)
	if reqAfter != reqBefore || totalAfter != totalBefore {
		t.Fatal("all-zero delta must not be recorded into stats")
	}
}

func TestRecordInterruptedTurnUsageNoopForIdentityMeter(t *testing.T) {
	const model = "test-interrupted-identity"
	reqBefore, totalBefore, _ := statsModelRow(model)
	recordInterruptedTurnUsage(identityMeter{}, `{"type":"result","usage":{"input_tokens":10,"output_tokens":5}}`, model)
	reqAfter, totalAfter, _ := statsModelRow(model)
	if reqAfter != reqBefore || totalAfter != totalBefore {
		t.Fatal("identityMeter must be a no-op (ephemeral harvests via perModelCounts)")
	}
}

// ---- perModelCounts (A4) ----

func TestPerModelCountsPrefersModelUsage(t *testing.T) {
	ev := &claudeEvent{
		Type: "result",
		Usage: &claudeUsage{ // bare 全 0（SIGINT 场景），必须不被采用
			InputTokens: 0, OutputTokens: 0,
		},
		ModelUsage: map[string]claudeModelUsage{
			"claude-opus-4":  {InputTokens: 100, OutputTokens: 50, CacheReadInputTokens: 1000, CacheCreationInputTokens: 20, CostUSD: 0.30},
			"claude-haiku-4": {InputTokens: 7, OutputTokens: 3, CostUSD: 0.01},
		},
		TotalCostUSD: 0.31,
	}
	pm := perModelCounts(ev, "requested-model")
	if len(pm) != 2 {
		t.Fatalf("perModelCounts entries = %d, want 2", len(pm))
	}
	opus := pm["claude-opus-4"]
	if opus.Input != 100 || opus.Output != 50 || opus.CacheRead != 1000 || opus.CacheCreation != 20 || math.Abs(opus.CostUSD-0.30) > 1e-9 {
		t.Fatalf("opus counts = %+v", opus)
	}
	if _, ok := pm["requested-model"]; ok {
		t.Fatal("fallback model must not appear when modelUsage is present")
	}
}

func TestPerModelCountsFallsBackToBareUsage(t *testing.T) {
	ev := &claudeEvent{
		Type:         "result",
		Usage:        &claudeUsage{InputTokens: 10, OutputTokens: 38, CacheReadInputTokens: 5},
		TotalCostUSD: 0.02,
	}
	pm := perModelCounts(ev, "fallback-model")
	if len(pm) != 1 {
		t.Fatalf("perModelCounts entries = %d, want 1", len(pm))
	}
	c := pm["fallback-model"]
	if c.Input != 10 || c.Output != 38 || c.CacheRead != 5 || math.Abs(c.CostUSD-0.02) > 1e-9 {
		t.Fatalf("fallback counts = %+v", c)
	}
}

func TestPerModelCountsNilOnNoUsage(t *testing.T) {
	if pm := perModelCounts(&claudeEvent{Type: "result"}, "m"); pm != nil {
		t.Fatalf("no usage should yield nil, got %+v", pm)
	}
	if pm := perModelCounts(nil, "m"); pm != nil {
		t.Fatal("nil event should yield nil")
	}
}
