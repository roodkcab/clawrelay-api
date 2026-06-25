package main

import (
	"encoding/json"
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
