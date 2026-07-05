package main

import (
	"os"
	"testing"
	"time"
)

// TestThreadMapSetRewritesFileForSameValue locks in the CODEX-4 double-fix:
// Set must rewrite the on-disk binding even when the value is unchanged, so
// the file's mtime acts as a liveness beacon for the sessions cleanup (codex
// keeps one thread_id for the whole conversation, so without this the file
// would never be touched after turn #1).
func TestThreadMapSetRewritesFileForSameValue(t *testing.T) {
	tm := newThreadMap(t.TempDir())
	tm.Set("sess", "tid-1")

	p := tm.path("sess")
	old := time.Now().Add(-48 * time.Hour)
	if err := os.Chtimes(p, old, old); err != nil {
		t.Fatal(err)
	}
	before, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}

	tm.Set("sess", "tid-1") // identical value — must still rewrite

	after, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	if !after.ModTime().After(before.ModTime()) {
		t.Fatalf("Set with identical value did not refresh mtime: before=%v after=%v",
			before.ModTime(), after.ModTime())
	}
	if got := tm.Get("sess"); got != "tid-1" {
		t.Fatalf("binding corrupted by rewrite: got %q", got)
	}
}

// TestThreadMapForgetRemovesMemoryAndDisk: after Forget, both the in-process
// cache and the on-disk file are gone, so Get returns "" (fresh session next).
func TestThreadMapForgetRemovesMemoryAndDisk(t *testing.T) {
	tm := newThreadMap(t.TempDir())
	tm.Set("sess", "tid-dead")

	tm.Forget("sess")

	if got := tm.Get("sess"); got != "" {
		t.Fatalf("expected empty binding after Forget, got %q", got)
	}
	if _, err := os.Stat(tm.path("sess")); !os.IsNotExist(err) {
		t.Fatalf("codex_thread.txt should be removed after Forget, stat err=%v", err)
	}
}
