package sessions

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// aged returns a timestamp comfortably older than the 72h test maxAge.
func aged() time.Time { return time.Now().Add(-100 * time.Hour) }

// TestCleanupKeepsSessionWithFreshChildFile locks in the CODEX-4 fix: a
// session directory whose OWN mtime is frozen at turn #1 (codex_thread.txt
// written once, attachments staged in the bot's working_dir instead) must NOT
// be reaped as long as a direct child file is fresh.
func TestCleanupKeepsSessionWithFreshChildFile(t *testing.T) {
	dir := t.TempDir()
	s := New(dir)

	sessDir := filepath.Join(dir, "active")
	if err := os.MkdirAll(sessDir, 0755); err != nil {
		t.Fatal(err)
	}
	// Fresh child (just written), then age the directory itself.
	if err := os.WriteFile(filepath.Join(sessDir, "codex_thread.txt"), []byte("tid-1"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(sessDir, aged(), aged()); err != nil {
		t.Fatal(err)
	}
	// Old sibling jsonl — must also survive because the SESSION is alive.
	logPath := filepath.Join(dir, "active.jsonl")
	if err := os.WriteFile(logPath, []byte("{}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(logPath, aged(), aged()); err != nil {
		t.Fatal(err)
	}

	s.cleanupOldSessions(72 * time.Hour)

	if _, err := os.Stat(filepath.Join(sessDir, "codex_thread.txt")); err != nil {
		t.Fatalf("live thread binding was deleted: %v", err)
	}
	if _, err := os.Stat(logPath); err != nil {
		t.Fatalf("jsonl of a live session was deleted: %v", err)
	}
}

// TestCleanupKeepsSessionWithFreshJsonl: old dir + old children but a fresh
// .jsonl sibling (session logged recently) must survive — the old code's dir
// branch would have removed both artifacts on the dir's mtime alone.
func TestCleanupKeepsSessionWithFreshJsonl(t *testing.T) {
	dir := t.TempDir()
	s := New(dir)

	sessDir := filepath.Join(dir, "logfresh")
	if err := os.MkdirAll(sessDir, 0755); err != nil {
		t.Fatal(err)
	}
	child := filepath.Join(sessDir, "codex_thread.txt")
	if err := os.WriteFile(child, []byte("tid-2"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{child, sessDir} {
		if err := os.Chtimes(p, aged(), aged()); err != nil {
			t.Fatal(err)
		}
	}
	// Fresh jsonl.
	logPath := filepath.Join(dir, "logfresh.jsonl")
	if err := os.WriteFile(logPath, []byte("{}\n"), 0644); err != nil {
		t.Fatal(err)
	}

	s.cleanupOldSessions(72 * time.Hour)

	if _, err := os.Stat(sessDir); err != nil {
		t.Fatalf("session dir with fresh jsonl sibling was deleted: %v", err)
	}
	if _, err := os.Stat(logPath); err != nil {
		t.Fatalf("fresh jsonl was deleted: %v", err)
	}
}

// TestCleanupRemovesFullyStaleSession: when dir, children AND jsonl are all
// past maxAge, everything is removed.
func TestCleanupRemovesFullyStaleSession(t *testing.T) {
	dir := t.TempDir()
	s := New(dir)

	sessDir := filepath.Join(dir, "stale")
	if err := os.MkdirAll(sessDir, 0755); err != nil {
		t.Fatal(err)
	}
	child := filepath.Join(sessDir, "codex_thread.txt")
	if err := os.WriteFile(child, []byte("tid-3"), 0600); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(dir, "stale.jsonl")
	if err := os.WriteFile(logPath, []byte("{}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	// Age child first, then dir (writing child bumps the dir mtime).
	for _, p := range []string{child, logPath, sessDir} {
		if err := os.Chtimes(p, aged(), aged()); err != nil {
			t.Fatal(err)
		}
	}

	s.cleanupOldSessions(72 * time.Hour)

	if _, err := os.Stat(sessDir); !os.IsNotExist(err) {
		t.Fatalf("stale session dir should be removed, stat err=%v", err)
	}
	if _, err := os.Stat(logPath); !os.IsNotExist(err) {
		t.Fatalf("stale jsonl should be removed, stat err=%v", err)
	}
}

// TestAppendCapsInMemoryEvents: the in-memory buffer must stay bounded for
// long-lived sessions, dropping the oldest half past the cap and marking the
// cut with a "truncated" event; the on-disk jsonl keeps everything.
func TestAppendCapsInMemoryEvents(t *testing.T) {
	dir := t.TempDir()
	s := New(dir)
	entry := s.GetOrCreate("longlived")

	total := maxBufferedEvents + 100
	for i := 0; i < total; i++ {
		data, _ := json.Marshal(map[string]string{"text": fmt.Sprintf("d%d", i)})
		entry.Append(Event{Timestamp: time.Now().Format(time.RFC3339Nano), Type: "response_delta", Data: data})
	}

	events := entry.Events()
	if len(events) > maxBufferedEvents {
		t.Fatalf("in-memory events not capped: len=%d > %d", len(events), maxBufferedEvents)
	}
	if events[0].Type != "truncated" {
		t.Fatalf("first buffered event should be the truncation marker, got %q", events[0].Type)
	}
	var marker struct {
		Dropped int `json:"dropped"`
	}
	if err := json.Unmarshal(events[0].Data, &marker); err != nil || marker.Dropped == 0 {
		t.Fatalf("truncation marker missing dropped count: data=%s err=%v", events[0].Data, err)
	}
	// Newest event survives the trim.
	last := events[len(events)-1]
	if !strings.Contains(string(last.Data), fmt.Sprintf("d%d", total-1)) {
		t.Fatalf("newest event lost in trim: %s", last.Data)
	}

	// Disk log is unaffected by the memory cap: all appends are on disk.
	data, err := os.ReadFile(filepath.Join(dir, "longlived.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if lines := strings.Count(string(data), "\n"); lines != total {
		t.Fatalf("disk log should keep all %d events, has %d lines", total, lines)
	}
}

// TestCleanupClosesEntryBeforeUnlink locks in the race fix: a handler that
// grabbed the *Entry before cleanup must not resurrect state — its late
// Appends are no-ops (closed flag, nil logFile), and the unlinked jsonl stays
// gone.
func TestCleanupClosesEntryBeforeUnlink(t *testing.T) {
	dir := t.TempDir()
	s := New(dir)

	entry := s.GetOrCreate("inflight")
	entry.Append(Event{Timestamp: time.Now().Format(time.RFC3339Nano), Type: "request", Data: json.RawMessage(`{}`)})

	logPath := filepath.Join(dir, "inflight.jsonl")
	if err := os.Chtimes(logPath, aged(), aged()); err != nil {
		t.Fatal(err)
	}

	s.cleanupOldSessions(72 * time.Hour)

	if s.Get("inflight") != nil {
		t.Fatal("cleaned-up session still in store map")
	}
	if _, err := os.Stat(logPath); !os.IsNotExist(err) {
		t.Fatalf("stale jsonl should be removed, stat err=%v", err)
	}

	// The handler still holds the old pointer and keeps logging.
	before := len(entry.Events())
	entry.Append(Event{Timestamp: time.Now().Format(time.RFC3339Nano), Type: "response_delta", Data: json.RawMessage(`{"text":"late"}`)})
	if got := len(entry.Events()); got != before {
		t.Fatalf("Append on a closed entry grew events: %d -> %d", before, got)
	}
	if _, err := os.Stat(logPath); !os.IsNotExist(err) {
		t.Fatalf("late Append resurrected the jsonl, stat err=%v", err)
	}
}
