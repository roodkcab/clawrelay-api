package main

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"sync"
)

// usageMeter converts codex's per-thread *cumulative* usage into per-turn
// reportable usage.
//
// Why this exists: codex reports usage on every `turn.completed`, but the
// figures are running totals for the whole thread, not the single turn. Because
// we resume the same thread across requests (see threadMap), a long session's
// raw input/output/cached counts climb monotonically — feeding them straight
// into the chat log double-counts every earlier turn. In production (2026-06) a
// 78-turn session inflated input ~40x (single turn reached 5.5M, far past any
// model context window) before this was caught.
//
// Unlike relay-claude's in-process cumulativeMeter, the codex cumulative counter
// lives in the *thread*, which outlives the relay process — threadMap persists
// thread_id to disk and resumes it after restarts. So the baseline has to be
// persisted per session too: otherwise a relay restart would diff against a zero
// baseline and re-bill the whole thread once. The baseline file sits next to
// codex_thread.txt under the session dir, so session cleanup reaps both
// together — thread gone ⇒ baseline gone ⇒ next thread starts from zero on both
// sides.
//
// Known accepted cost: the FIRST turn after this build is first deployed onto an
// already-live thread has no baseline file yet, so it diffs against zero and
// bills that thread's accumulated total as one turn (a one-off inflation per
// active session, bounded to the backlog at deploy time). There is no way to
// recover the true baseline without a prior turn, so this is accepted rather
// than guarded; see TestUsageMeterEmptyBaselineInflatesOnce which pins it.

// meterBaseline is the per-session high-water mark of the *reportable* cumulative
// usage — already split into uncached-input / cache-read / output via mapUsage.
// Diffing the derived split (rather than codex's raw total+cached, then
// recombining) is what keeps the input/cache_read attribution correct when the
// cached subset and the total move by different amounts across turns.
type meterBaseline struct {
	Uncached  int `json:"uncached_input_tokens"`
	CacheRead int `json:"cache_read_tokens"`
	Output    int `json:"output_tokens"`
}

type usageMeter struct {
	mu      sync.Mutex
	memory  map[string]meterBaseline // session_id → cumulative high-water baseline
	rootDir string
}

func newUsageMeter(rootDir string) *usageMeter {
	return &usageMeter{
		memory:  make(map[string]meterBaseline),
		rootDir: rootDir,
	}
}

// deriveBaseline maps codex's raw cumulative usage to the reportable cumulative
// split, reusing mapUsage so the total→(uncached,cacheRead) rule lives in one
// place (and handles the malformed cached>total case the same way).
func deriveBaseline(u codexUsage) meterBaseline {
	input, output, cacheRead := mapUsage(&u)
	return meterBaseline{Uncached: input, CacheRead: cacheRead, Output: output}
}

// perTurn returns this turn's reportable (input, output, cacheRead) given codex's
// cumulative snapshot `cur`. With no session id there is no thread reuse, so the
// snapshot is already per-turn and is returned as-is (just split).
//
// Otherwise it diffs the derived cumulative split against the session's stored
// baseline, flooring each field at zero, and advances the baseline to the
// per-field high-water mark. Flooring + high-water (rather than a "cur < last ⇒
// emit the whole cur as one turn" reset) keeps totals correct under the two ways
// last and cur can disagree WITHOUT a genuine counter reset:
//
//   - Out-of-order completion of two concurrent same-session turns: the earlier
//     (smaller) snapshot floors to 0 instead of re-billing its full cumulative,
//     and the high-water baseline stops it dragging the baseline backwards, so
//     the running sum stays exact regardless of completion order.
//   - One field dipping while others climb: we never bill a climbing field's
//     entire cumulative just because a different field regressed.
//
// A genuine counter reset (a different/new thread) is handled upstream: thread
// rotation calls Forget, zeroing the baseline so cur passes through as the turn.
func (m *usageMeter) perTurn(sessionID string, cur codexUsage) (input, output, cacheRead int) {
	b := deriveBaseline(cur)
	if sessionID == "" {
		return b.Uncached, b.Output, b.CacheRead
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	last, ok := m.memory[sessionID]
	if !ok {
		last = m.load(sessionID) // read-through from disk on first turn after a restart
	}

	input = max(b.Uncached-last.Uncached, 0)
	output = max(b.Output-last.Output, 0)
	cacheRead = max(b.CacheRead-last.CacheRead, 0)

	// Advance the baseline to the per-field high-water mark so an out-of-order
	// smaller snapshot can't pull it back and double-bill the next turn.
	hi := meterBaseline{
		Uncached:  max(b.Uncached, last.Uncached),
		CacheRead: max(b.CacheRead, last.CacheRead),
		Output:    max(b.Output, last.Output),
	}
	m.memory[sessionID] = hi
	m.store(sessionID, hi)
	return input, output, cacheRead
}

// Forget drops a session's baseline. Called alongside threadMap.Forget when the
// codex thread is gone or rotated — the replacement thread restarts its
// cumulative counter from zero, so the baseline must too.
func (m *usageMeter) Forget(sessionID string) {
	if sessionID == "" {
		return
	}
	m.mu.Lock()
	delete(m.memory, sessionID)
	m.mu.Unlock()
	_ = os.Remove(m.path(sessionID))
}

func (m *usageMeter) load(sessionID string) meterBaseline {
	var b meterBaseline
	data, err := os.ReadFile(m.path(sessionID))
	if err != nil {
		return b // no baseline yet (new session, or reaped by cleanup) — zero is correct
	}
	if err := json.Unmarshal(data, &b); err != nil {
		// A corrupt baseline must NOT silently read as zero — that would re-bill
		// the thread's whole cumulative as one turn (the very inflation this
		// meter prevents). store() writes atomically so this should be
		// unreachable barring external tampering/FS damage; surface it loudly so
		// the one-off over-count is at least detectable.
		log.Printf("[usageMeter] corrupt baseline %s: %v — treating as empty (turn may over-count once)", m.path(sessionID), err)
		return meterBaseline{}
	}
	return b
}

func (m *usageMeter) store(sessionID string, b meterBaseline) {
	dir := filepath.Join(m.rootDir, sessionID)
	if err := os.MkdirAll(dir, 0755); err != nil {
		log.Printf("[usageMeter] mkdir %s: %v — baseline not persisted, may re-bill after restart", dir, err)
		return
	}
	data, err := json.Marshal(b)
	if err != nil {
		log.Printf("[usageMeter] marshal baseline for %s: %v", sessionID, err)
		return
	}
	// Atomic write: a crash mid-write would otherwise leave a truncated file that
	// load() can't parse. tmp+rename within the same dir is atomic on POSIX.
	tmp := m.path(sessionID) + ".tmp"
	if err := os.WriteFile(tmp, data, 0600); err != nil {
		log.Printf("[usageMeter] write %s: %v — baseline not persisted, may re-bill after restart", tmp, err)
		return
	}
	if err := os.Rename(tmp, m.path(sessionID)); err != nil {
		log.Printf("[usageMeter] rename %s: %v — baseline not persisted", tmp, err)
		_ = os.Remove(tmp)
	}
}

func (m *usageMeter) path(sessionID string) string {
	return filepath.Join(m.rootDir, sessionID, "codex_usage.json")
}
