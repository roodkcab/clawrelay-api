package sessions

import (
	"os"
	"path/filepath"
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
