package main

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"clawrelay-api/pkg/openai"
)

// v3AsstLine builds one synthetic transcript assistant row (compact JSON +
// trailing newline), mirroring the real shape: top-level type/requestId/
// isSidechain, usage under message.usage. Content blocks of one message repeat
// this row verbatim with the same requestId.
func v3AsstLine(reqID, model string, in, cc, cr, out int, sidechain bool) string {
	return fmt.Sprintf(`{"parentUuid":null,"isSidechain":%v,"sessionId":"s-1","timestamp":"2026-07-05T00:00:00Z","type":"assistant","requestId":%q,"message":{"role":"assistant","model":%q,"usage":{"input_tokens":%d,"cache_creation_input_tokens":%d,"cache_read_input_tokens":%d,"output_tokens":%d,"service_tier":"standard"}}}`+"\n",
		sidechain, reqID, model, in, cc, cr, out)
}

func v3WriteFile(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "transcript.jsonl")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func v3Append(t *testing.T, path, content string) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := f.WriteString(content); err != nil {
		t.Fatal(err)
	}
}

// One assistant message split over 3 rows (same requestId, usage repeated
// verbatim) must be counted exactly once — the real-world 3-5x over-count trap.
func TestV3UsageDedupSameRequestIDWithinWindow(t *testing.T) {
	row := v3AsstLine("req_A", "claude-opus-4", 100, 20, 30, 7, false)
	path := v3WriteFile(t, row+row+row)

	perModel, newOff, reqIDs, err := readV3UsageWindow(path, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	c := perModel["claude-opus-4"]
	if c.Input != 100 || c.CacheCreation != 20 || c.CacheRead != 30 || c.Output != 7 {
		t.Errorf("duplicated requestId over-counted: %+v", c)
	}
	if c.CostUSD != 0 {
		t.Errorf("transcript has no cost source; CostUSD must be 0, got %v", c.CostUSD)
	}
	if len(reqIDs) != 1 {
		t.Errorf("want 1 requestId seen, got %d", len(reqIDs))
	}
	if want := int64(len(row) * 3); newOff != want {
		t.Errorf("newOffset=%d want %d", newOff, want)
	}
}

// A message's duplicate rows can straddle a window boundary: rows of req_A read
// in window 1 must suppress req_A rows appearing in window 2 (via prevReqIDs).
func TestV3UsageCrossWindowDedupViaPrevReqIDs(t *testing.T) {
	rowA := v3AsstLine("req_A", "m", 50, 0, 0, 5, false)
	path := v3WriteFile(t, rowA)

	pm1, off1, ids1, err := readV3UsageWindow(path, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pm1["m"].Input != 50 {
		t.Fatalf("window1: %+v", pm1)
	}

	// The same message's remaining content-block rows + a genuinely new request
	// arrive later.
	v3Append(t, path, rowA+rowA+v3AsstLine("req_B", "m", 7, 0, 0, 3, false))

	pm2, _, _, err := readV3UsageWindow(path, off1, ids1)
	if err != nil {
		t.Fatal(err)
	}
	c := pm2["m"]
	if c.Input != 7 || c.Output != 3 {
		t.Errorf("cross-window dedup failed (req_A recounted): %+v", c)
	}
}

// The offset must make each window an increment: window 2 starting at window
// 1's newOffset sees only rows appended after window 1.
func TestV3UsageOffsetIncrementalSemantics(t *testing.T) {
	path := v3WriteFile(t, v3AsstLine("req_A", "m", 10, 0, 0, 1, false))

	_, off1, _, err := readV3UsageWindow(path, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	v3Append(t, path, v3AsstLine("req_B", "m", 20, 0, 0, 2, false))

	pm2, off2, _, err := readV3UsageWindow(path, off1, map[string]struct{}{"req_A": {}})
	if err != nil {
		t.Fatal(err)
	}
	if c := pm2["m"]; c.Input != 20 || c.Output != 2 {
		t.Errorf("window2 must only see the appended row: %+v", c)
	}
	if fi, _ := os.Stat(path); off2 != fi.Size() {
		t.Errorf("off2=%d want file size %d", off2, fi.Size())
	}
}

// A trailing row without '\n' (a write in flight) must NOT be consumed:
// newOffset stops before it, and once completed it is read by the next window.
func TestV3UsagePartialLastLineNotConsumed(t *testing.T) {
	full := v3AsstLine("req_A", "m", 10, 0, 0, 1, false)
	partialRow := v3AsstLine("req_B", "m", 99, 0, 0, 9, false)
	half := partialRow[:len(partialRow)/2] // no trailing newline
	path := v3WriteFile(t, full+half)

	pm1, off1, _, err := readV3UsageWindow(path, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if c := pm1["m"]; c.Input != 10 || c.Output != 1 {
		t.Errorf("partial row must not be counted: %+v", c)
	}
	if want := int64(len(full)); off1 != want {
		t.Errorf("newOffset must stop before the partial row: got %d want %d", off1, want)
	}

	// Writer finishes the row → next window picks it up whole.
	v3Append(t, path, partialRow[len(partialRow)/2:])
	pm2, _, _, err := readV3UsageWindow(path, off1, map[string]struct{}{"req_A": {}})
	if err != nil {
		t.Fatal(err)
	}
	if c := pm2["m"]; c.Input != 99 || c.Output != 9 {
		t.Errorf("completed row not read in next window: %+v", c)
	}
}

// isSidechain=true rows are Task sub-agent calls — real billed API rounds,
// counted under the same requestId dedup rule.
func TestV3UsageSidechainRowsCounted(t *testing.T) {
	path := v3WriteFile(t,
		v3AsstLine("req_A", "m", 10, 0, 0, 1, false)+
			v3AsstLine("req_S", "m", 40, 0, 0, 4, true))

	pm, _, _, err := readV3UsageWindow(path, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if c := pm["m"]; c.Input != 50 || c.Output != 5 {
		t.Errorf("sidechain row must be counted: %+v", c)
	}
}

// Non-assistant rows and assistant rows without message.usage are skipped.
func TestV3UsageSkipsNonAssistantAndNoUsage(t *testing.T) {
	content := `{"type":"user","message":{"role":"user","content":"hi"}}` + "\n" +
		`{"type":"summary","summary":"x"}` + "\n" +
		`{"type":"assistant","requestId":"req_NU","message":{"role":"assistant","model":"m","content":[]}}` + "\n" + // no usage
		`{"type":"system","subtype":"info","content":"note about \"type\":\"assistant\" rows"}` + "\n" + // cheap filter matches, unmarshal disqualifies
		v3AsstLine("req_A", "m", 5, 0, 0, 2, false)
	path := v3WriteFile(t, content)

	pm, newOff, reqIDs, err := readV3UsageWindow(path, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(pm) != 1 {
		t.Fatalf("only the usage-bearing assistant row must count: %v", pm)
	}
	if c := pm["m"]; c.Input != 5 || c.Output != 2 {
		t.Errorf("aggregate wrong: %+v", c)
	}
	if _, ok := reqIDs["req_NU"]; ok {
		t.Errorf("usage-less row must not claim its requestId")
	}
	if want := int64(len(content)); newOff != want {
		t.Errorf("skipped rows must still advance the offset: got %d want %d", newOff, want)
	}
}

// A file shorter than the stored offset means rotation/truncation → reread
// from 0 instead of erroring or reading garbage.
func TestV3UsageRotationRereadsFromZero(t *testing.T) {
	longRow := v3AsstLine("req_A", "m", 1000, 0, 0, 100, false)
	path := v3WriteFile(t, longRow+longRow) // large file
	_, bigOff, _, err := readV3UsageWindow(path, 0, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Rotate: file replaced by shorter, fresh content.
	if err := os.WriteFile(path, []byte(v3AsstLine("req_NEW", "m", 3, 0, 0, 1, false)), 0o644); err != nil {
		t.Fatal(err)
	}
	pm, newOff, _, err := readV3UsageWindow(path, bigOff, nil)
	if err != nil {
		t.Fatal(err)
	}
	if c := pm["m"]; c.Input != 3 || c.Output != 1 {
		t.Errorf("rotation must reread from 0: %+v", c)
	}
	if fi, _ := os.Stat(path); newOff != fi.Size() {
		t.Errorf("newOffset after rotation reread: got %d want %d", newOff, fi.Size())
	}
}

// Usage lands under the model that actually served the request; an empty
// message.model falls back to "unknown".
func TestV3UsageMultiModelAttribution(t *testing.T) {
	path := v3WriteFile(t,
		v3AsstLine("req_A", "claude-opus-4", 100, 10, 20, 5, false)+
			v3AsstLine("req_B", "claude-haiku-4", 30, 0, 0, 3, true)+
			v3AsstLine("req_C", "", 7, 0, 0, 1, false))

	pm, _, _, err := readV3UsageWindow(path, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]openai.TokenCounts{
		"claude-opus-4":  {Input: 100, CacheCreation: 10, CacheRead: 20, Output: 5},
		"claude-haiku-4": {Input: 30, Output: 3},
		"unknown":        {Input: 7, Output: 1},
	}
	for model, w := range want {
		if got := pm[model]; got != w {
			t.Errorf("model %s: got %+v want %+v", model, got, w)
		}
	}
	if len(pm) != len(want) {
		t.Errorf("unexpected models: %v", pm)
	}
}

// A missing transcript file surfaces as an error (the harvester turns that
// into "not metered", never a zero).
func TestV3UsageMissingFileErrors(t *testing.T) {
	_, off, _, err := readV3UsageWindow(filepath.Join(t.TempDir(), "nope.jsonl"), 42, nil)
	if err == nil {
		t.Fatal("want error for missing file")
	}
	if off != 42 {
		t.Errorf("offset must be preserved on open failure: got %d", off)
	}
}
