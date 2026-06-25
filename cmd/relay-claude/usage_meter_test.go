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
