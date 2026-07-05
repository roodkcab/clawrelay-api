package main

import (
	"os"
	"path/filepath"
	"testing"
)

// codex reports thread-cumulative usage on every turn; the meter must return
// per-turn reportable figures (uncached input, output, cache_read), not totals.
func TestUsageMeterDiffsCumulative(t *testing.T) {
	m := newUsageMeter(t.TempDir())

	// turn1 cumulative: total=100, cached=40, out=10 → uncached input=60.
	in, out, cr := m.perTurn("s1", codexUsage{InputTokens: 100, CachedInputTokens: 40, OutputTokens: 10})
	if in != 60 || out != 10 || cr != 40 {
		t.Fatalf("turn1 = in:%d out:%d cr:%d, want 60/10/40 (first turn, zero baseline)", in, out, cr)
	}

	// turn2 cumulative: total=260, cached=90, out=35 → uncached=170; per-turn
	// delta vs turn1's split: input 170-60=110, cache_read 90-40=50, out 35-10=25.
	in, out, cr = m.perTurn("s1", codexUsage{InputTokens: 260, CachedInputTokens: 90, OutputTokens: 35})
	if in != 110 || out != 25 || cr != 50 {
		t.Fatalf("turn2 = in:%d out:%d cr:%d, want 110/25/50 (per-turn delta)", in, out, cr)
	}
}

// An empty session id means no thread reuse — the snapshot is already per-turn,
// just split into (uncached input, output, cache_read).
func TestUsageMeterPassThroughNoSession(t *testing.T) {
	m := newUsageMeter(t.TempDir())
	in, out, cr := m.perTurn("", codexUsage{InputTokens: 500, CachedInputTokens: 200, OutputTokens: 50})
	if in != 300 || out != 50 || cr != 200 {
		t.Fatalf("no-session = in:%d out:%d cr:%d, want 300/50/200 (split, not diffed)", in, out, cr)
	}
}

// A regression (smaller snapshot than the baseline) within a thread is anomalous
// — every field must floor to zero, NOT emit the whole cur as one turn — and the
// baseline must hold its high-water mark so a later real turn still diffs right.
func TestUsageMeterFloorsRegression(t *testing.T) {
	m := newUsageMeter(t.TempDir())
	m.perTurn("s1", codexUsage{InputTokens: 1000, CachedInputTokens: 400, OutputTokens: 100}) // baseline derived {600,400,100}

	in, out, cr := m.perTurn("s1", codexUsage{InputTokens: 120, CachedInputTokens: 30, OutputTokens: 12})
	if in != 0 || out != 0 || cr != 0 {
		t.Fatalf("regression = in:%d out:%d cr:%d, want all-zero (floored, not raw cur)", in, out, cr)
	}

	// Baseline must still be the high-water {600,400,100}, so the next real turn
	// diffs against it. cur derived {660,440,110} → delta 60/10/40.
	in, out, cr = m.perTurn("s1", codexUsage{InputTokens: 1100, CachedInputTokens: 440, OutputTokens: 110})
	if in != 60 || out != 10 || cr != 40 {
		t.Fatalf("after regression, next = in:%d out:%d cr:%d, want 60/10/40", in, out, cr)
	}
}

// Two concurrent same-session turns completing OUT OF ORDER (larger cumulative
// lands first) must still bill the correct total — the earlier snapshot floors
// to zero, the high-water baseline keeps the sum exact regardless of order.
func TestUsageMeterConcurrentOutOfOrder(t *testing.T) {
	m := newUsageMeter(t.TempDir())
	m.perTurn("s1", codexUsage{InputTokens: 1000, CachedInputTokens: 400, OutputTokens: 100}) // baseline {600,400,100}

	in1, out1, cr1 := m.perTurn("s1", codexUsage{InputTokens: 1400, CachedInputTokens: 520, OutputTokens: 140}) // larger first
	in2, out2, cr2 := m.perTurn("s1", codexUsage{InputTokens: 1200, CachedInputTokens: 460, OutputTokens: 120}) // smaller second

	// Totals must equal high-water {880,520,140} minus baseline {600,400,100}.
	if in1+in2 != 280 || out1+out2 != 40 || cr1+cr2 != 120 {
		t.Fatalf("out-of-order totals = in:%d out:%d cr:%d, want 280/40/120", in1+in2, out1+out2, cr1+cr2)
	}
	if in2 != 0 || out2 != 0 || cr2 != 0 {
		t.Fatalf("the smaller out-of-order snapshot = in:%d out:%d cr:%d, want all-zero (floored)", in2, out2, cr2)
	}
}

// Deriving the split BEFORE diffing (not diffing raw total/cached then
// recombining) keeps attribution correct when the cached subset and the total
// move by different amounts: here total climbs while the cached cumulative dips,
// and the whole increment must land in input — none of it lost.
func TestUsageMeterDerivesBeforeDiff(t *testing.T) {
	m := newUsageMeter(t.TempDir())
	m.perTurn("s1", codexUsage{InputTokens: 100, CachedInputTokens: 80, OutputTokens: 10}) // baseline derived {uncached:20, cr:80, out:10}

	// cur derived {uncached:190, cr:60, out:25}. input = 190-20 = 170; cache_read
	// floors (60<80) to 0; output 25-10 = 15.
	in, out, cr := m.perTurn("s1", codexUsage{InputTokens: 250, CachedInputTokens: 60, OutputTokens: 25})
	if in != 170 || out != 15 || cr != 0 {
		t.Fatalf("derive-before-diff = in:%d out:%d cr:%d, want 170/15/0 (no input lost to the cached dip)", in, out, cr)
	}
}

// A single derived field dipping while others climb must floor only that field.
func TestUsageMeterSingleFieldRegression(t *testing.T) {
	m := newUsageMeter(t.TempDir())
	m.perTurn("s1", codexUsage{InputTokens: 1000, CachedInputTokens: 900, OutputTokens: 100}) // derived {uncached:100, cr:900, out:100}

	// cur derived {uncached:100, cr:910, out:90}: input delta 0, cache_read +10,
	// output floors (90<100) to 0.
	in, out, cr := m.perTurn("s1", codexUsage{InputTokens: 1010, CachedInputTokens: 910, OutputTokens: 90})
	if in != 0 || out != 0 || cr != 10 {
		t.Fatalf("single-field regression = in:%d out:%d cr:%d, want 0/0/10 (only output floored)", in, out, cr)
	}
}

// The codex thread (and its cumulative counter) outlives the relay process, so
// the baseline must survive a restart — a fresh meter reads it back from disk
// and keeps diffing instead of re-counting the whole thread.
func TestUsageMeterPersistsAcrossRestart(t *testing.T) {
	dir := t.TempDir()
	m1 := newUsageMeter(dir)
	m1.perTurn("s1", codexUsage{InputTokens: 300, CachedInputTokens: 100, OutputTokens: 30}) // baseline derived {200,100,30}

	m2 := newUsageMeter(dir) // simulate relay restart: in-memory map is empty
	in, out, cr := m2.perTurn("s1", codexUsage{InputTokens: 420, CachedInputTokens: 130, OutputTokens: 44})
	if in != 90 || out != 14 || cr != 30 {
		t.Fatalf("after restart = in:%d out:%d cr:%d, want 90/14/30 (baseline loaded from disk)", in, out, cr)
	}
}

// Deploying this build onto an already-live thread leaves no baseline file, so
// the first turn bills the whole accumulated cumulative once. This is an accepted
// one-off cost (we can't recover the true baseline without a prior turn); pin it
// so the behavior is intentional and visible, and confirm it self-corrects after.
func TestUsageMeterEmptyBaselineInflatesOnce(t *testing.T) {
	m := newUsageMeter(t.TempDir())

	// derived uncached input = 5_000_000 - 2_000_000 = 3_000_000.
	in, _, cr := m.perTurn("live", codexUsage{InputTokens: 5_000_000, CachedInputTokens: 2_000_000, OutputTokens: 100_000})
	if in != 3_000_000 || cr != 2_000_000 {
		t.Fatalf("first-turn-after-deploy = in:%d cr:%d, want 3_000_000/2_000_000 (accepted one-off)", in, cr)
	}
	in, _, _ = m.perTurn("live", codexUsage{InputTokens: 5_050_000, CachedInputTokens: 2_010_000, OutputTokens: 101_000})
	if in != 40_000 { // derived 3_040_000 - 3_000_000
		t.Fatalf("second turn input = %d, want 40_000 (self-corrected)", in)
	}
}

// A corrupt/truncated baseline file (should be unreachable given atomic writes,
// but possible via FS damage/tampering) must not panic and must not be read as a
// partial number — it falls back to empty and recovers on the next turn.
func TestUsageMeterCorruptBaseline(t *testing.T) {
	dir := t.TempDir()
	sessDir := filepath.Join(dir, "s1")
	if err := os.MkdirAll(sessDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sessDir, "codex_usage.json"), []byte("{\"uncached_input_tokens\":12"), 0600); err != nil {
		t.Fatal(err)
	}

	m := newUsageMeter(dir)
	in, _, _ := m.perTurn("s1", codexUsage{InputTokens: 300, CachedInputTokens: 100, OutputTokens: 30})
	if in != 200 { // treated as empty baseline → derived uncached 300-100
		t.Fatalf("corrupt baseline input = %d, want 200 (treated as empty, not partial)", in)
	}
	// The atomic store should have replaced the corrupt file with a valid one.
	in, _, _ = m.perTurn("s1", codexUsage{InputTokens: 360, CachedInputTokens: 130, OutputTokens: 44})
	if in != 30 { // derived uncached 230 - baseline 200
		t.Fatalf("post-recovery input = %d, want 30", in)
	}
}

// Forget drops the baseline so the replacement thread starts from zero.
func TestUsageMeterForget(t *testing.T) {
	dir := t.TempDir()
	m := newUsageMeter(dir)
	m.perTurn("s1", codexUsage{InputTokens: 500, CachedInputTokens: 200, OutputTokens: 50})

	m.Forget("s1")

	fresh := newUsageMeter(dir) // no in-memory cache; must not find a stale baseline
	in, out, cr := fresh.perTurn("s1", codexUsage{InputTokens: 90, CachedInputTokens: 20, OutputTokens: 9})
	if in != 70 || out != 9 || cr != 20 { // derived split of the raw snapshot, zero baseline
		t.Fatalf("after Forget = in:%d out:%d cr:%d, want 70/9/20 (baseline cleared)", in, out, cr)
	}
}

// When codex rotates a session to a different thread_id (resume silently started
// a new thread), rebindThread must drop the usage baseline — the new thread's
// counter restarts at zero, so the old baseline is foreign. A rebind to the SAME
// thread must NOT drop it.
func TestRebindThreadForgetsOnRotation(t *testing.T) {
	dir := t.TempDir()
	oldThreads, oldMeter := threads, meter
	threads = newThreadMap(dir)
	meter = newUsageMeter(dir)
	defer func() { threads, meter = oldThreads, oldMeter }()

	threads.Set("s1", "T1")
	meter.perTurn("s1", codexUsage{InputTokens: 1000, CachedInputTokens: 400, OutputTokens: 100})

	// Rotation: bind to a different thread → baseline must be forgotten.
	rebindThread("s1", "T2")
	if threads.Get("s1") != "T2" {
		t.Fatalf("thread not rebound: got %q want T2", threads.Get("s1"))
	}
	in, out, cr := meter.perTurn("s1", codexUsage{InputTokens: 120, CachedInputTokens: 30, OutputTokens: 12})
	if in != 90 || out != 12 || cr != 30 { // baseline forgotten → derived split of raw
		t.Fatalf("after rotation = in:%d out:%d cr:%d, want 90/12/30 (baseline forgotten)", in, out, cr)
	}

	// Rebind to the SAME thread → baseline preserved, next turn diffs normally.
	rebindThread("s1", "T2")
	in, out, cr = meter.perTurn("s1", codexUsage{InputTokens: 200, CachedInputTokens: 50, OutputTokens: 20})
	if in != 60 || out != 8 || cr != 20 { // derived {150,50,20} - baseline {90,30,12}
		t.Fatalf("after same-thread rebind = in:%d out:%d cr:%d, want 60/8/20 (baseline preserved)", in, out, cr)
	}
}

// turnUsage must not panic when the global meter is nil (a handler/test path that
// didn't run main()); it falls back to mapping the raw usage.
func TestTurnUsageNilMeterPassThrough(t *testing.T) {
	oldMeter := meter
	meter = nil
	defer func() { meter = oldMeter }()

	in, out, cr := turnUsage("s1", &codexUsage{InputTokens: 1000, CachedInputTokens: 400, OutputTokens: 100})
	if in != 600 || out != 100 || cr != 400 { // raw mapUsage: 1000-400 / 100 / 400
		t.Fatalf("nil-meter turnUsage = in:%d out:%d cr:%d, want 600/100/400 (raw mapUsage)", in, out, cr)
	}
}
