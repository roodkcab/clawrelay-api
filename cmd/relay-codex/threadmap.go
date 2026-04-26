package main

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// threadMap persists the client_session_id → codex_thread_id binding so
// follow-up requests can `codex exec resume <thread_id>` instead of re-sending
// the full conversation history.
//
// Persistence is one tiny file per session: <sessionsDir>/<session_id>/codex_thread.txt
// — same directory we already use for that session's attachments. Surviving
// restarts is essentially free and we get atomic-write semantics from
// os.WriteFile.
type threadMap struct {
	mu      sync.RWMutex
	memory  map[string]string // session_id → thread_id (in-process cache)
	rootDir string
}

func newThreadMap(rootDir string) *threadMap {
	return &threadMap{
		memory:  make(map[string]string),
		rootDir: rootDir,
	}
}

// Get returns the codex thread_id bound to a client session_id, or "" if
// no binding exists. Reads the on-disk file lazily on first lookup.
func (t *threadMap) Get(sessionID string) string {
	if sessionID == "" {
		return ""
	}

	t.mu.RLock()
	tid, ok := t.memory[sessionID]
	t.mu.RUnlock()
	if ok {
		return tid
	}

	path := t.path(sessionID)
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	tid = strings.TrimSpace(string(data))
	if tid == "" {
		return ""
	}

	t.mu.Lock()
	t.memory[sessionID] = tid
	t.mu.Unlock()
	return tid
}

// Set binds a session_id to a codex thread_id and writes through to disk.
// Idempotent; the on-disk path is created lazily.
func (t *threadMap) Set(sessionID, threadID string) {
	if sessionID == "" || threadID == "" {
		return
	}

	t.mu.Lock()
	existing, had := t.memory[sessionID]
	t.memory[sessionID] = threadID
	t.mu.Unlock()
	if had && existing == threadID {
		return
	}

	dir := filepath.Join(t.rootDir, sessionID)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return
	}
	_ = os.WriteFile(t.path(sessionID), []byte(threadID), 0600)
}

// Forget removes the binding (e.g. when a resume fails because codex
// expired/discarded the thread). Next request will start a fresh thread.
func (t *threadMap) Forget(sessionID string) {
	if sessionID == "" {
		return
	}
	t.mu.Lock()
	delete(t.memory, sessionID)
	t.mu.Unlock()
	_ = os.Remove(t.path(sessionID))
}

func (t *threadMap) path(sessionID string) string {
	return filepath.Join(t.rootDir, sessionID, "codex_thread.txt")
}
