// Package sessions implements an on-disk + in-memory session log store with a
// WebSocket-streamed HTML viewer. It is shared by every relay binary so all
// completed conversations show up under the same `/sessions` UI regardless of
// which upstream CLI handled them.
package sessions

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"clawrelay-api/pkg/openai"
)

// Event is a single record in a session's append-only history.
type Event struct {
	Timestamp string          `json:"timestamp"`
	Type      string          `json:"type"` // "request", "response_delta", "response_done", "tool_use", "error"
	Data      json.RawMessage `json:"data"`
}

// Entry holds one session's in-memory state plus its log file handle.
type Entry struct {
	mu          sync.Mutex
	events      []Event
	subscribers map[chan Event]struct{}
	logFile     *os.File
}

// Store is the global registry of all known sessions for one relay binary.
// Sessions persist to <Dir>/<id>.jsonl on disk and (optionally) carry a
// per-session attachment subdirectory at <Dir>/<id>/files.
type Store struct {
	mu       sync.RWMutex
	sessions map[string]*Entry
	Dir      string
}

// New constructs a Store backed by a given directory. The directory is
// created lazily on first session.
func New(dir string) *Store {
	return &Store{
		sessions: make(map[string]*Entry),
		Dir:      dir,
	}
}

// AbsDir returns the absolute path of the storage root, useful for naming
// per-session attachment directories.
func (s *Store) AbsDir() string {
	if filepath.IsAbs(s.Dir) {
		return s.Dir
	}
	if abs, err := filepath.Abs(s.Dir); err == nil {
		return abs
	}
	return s.Dir
}

// GetOrCreate returns the Entry for a session, hydrating from disk if the
// session has prior history. Safe for concurrent use.
func (s *Store) GetOrCreate(sessionID string) *Entry {
	s.mu.RLock()
	entry, ok := s.sessions[sessionID]
	s.mu.RUnlock()
	if ok {
		return entry
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if entry, ok = s.sessions[sessionID]; ok {
		return entry
	}

	entry = &Entry{
		subscribers: make(map[chan Event]struct{}),
	}

	os.MkdirAll(s.Dir, 0755)
	logPath := filepath.Join(s.Dir, sessionID+".jsonl")
	if data, err := os.ReadFile(logPath); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			var ev Event
			if json.Unmarshal([]byte(line), &ev) == nil {
				entry.events = append(entry.events, ev)
			}
		}
	}
	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		log.Printf("Failed to open session log %s: %v", logPath, err)
	}
	entry.logFile = f

	s.sessions[sessionID] = entry
	return entry
}

// Get returns the Entry for an existing session or nil if unknown.
func (s *Store) Get(sessionID string) *Entry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.sessions[sessionID]
}

// Append adds an event to the session, persists it to the log file, and
// fans it out to live WebSocket subscribers.
func (e *Entry) Append(ev Event) {
	e.mu.Lock()
	e.events = append(e.events, ev)

	if e.logFile != nil {
		if data, err := json.Marshal(ev); err == nil {
			e.logFile.Write(append(data, '\n'))
		}
	}

	subs := make([]chan Event, 0, len(e.subscribers))
	for ch := range e.subscribers {
		subs = append(subs, ch)
	}
	e.mu.Unlock()

	for _, ch := range subs {
		select {
		case ch <- ev:
		default:
			// subscriber too slow; drop
		}
	}
}

// Subscribe attaches a channel to receive new events. The returned history
// snapshot is safe to send to clients before forwarding live events. Always
// call cancel() when done.
func (e *Entry) Subscribe() (history []Event, ch chan Event, cancel func()) {
	ch = make(chan Event, 256)
	e.mu.Lock()
	history = make([]Event, len(e.events))
	copy(history, e.events)
	e.subscribers[ch] = struct{}{}
	e.mu.Unlock()

	cancel = func() {
		e.mu.Lock()
		delete(e.subscribers, ch)
		e.mu.Unlock()
	}
	return
}

// Events returns a defensive copy of all events currently buffered in memory.
func (e *Entry) Events() []Event {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]Event, len(e.events))
	copy(out, e.events)
	return out
}

// ---- High-level logging helpers ----

// LogRequest appends a "request" event. Safe to call with empty sessionID
// (becomes a no-op).
func (s *Store) LogRequest(sessionID string, req *openai.ChatCompletionRequest) {
	if sessionID == "" {
		return
	}
	entry := s.GetOrCreate(sessionID)
	data, _ := json.Marshal(req)
	entry.Append(Event{
		Timestamp: time.Now().Format(time.RFC3339Nano),
		Type:      "request",
		Data:      data,
	})
}

// LogDelta appends a streaming text delta event.
func (s *Store) LogDelta(sessionID, text string) {
	if sessionID == "" || text == "" {
		return
	}
	entry := s.Get(sessionID)
	if entry == nil {
		return
	}
	data, _ := json.Marshal(map[string]string{"text": text})
	entry.Append(Event{
		Timestamp: time.Now().Format(time.RFC3339Nano),
		Type:      "response_delta",
		Data:      data,
	})
}

// LogToolUse appends a tool_use event with name, id, and serialized input.
func (s *Store) LogToolUse(sessionID, name, id, input string) {
	if sessionID == "" {
		return
	}
	entry := s.Get(sessionID)
	if entry == nil {
		return
	}
	data, _ := json.Marshal(map[string]string{"tool": name, "id": id, "input": input})
	entry.Append(Event{
		Timestamp: time.Now().Format(time.RFC3339Nano),
		Type:      "tool_use",
		Data:      data,
	})
}

// LogDone appends the terminating "response_done" event with usage.
func (s *Store) LogDone(sessionID string, usage *openai.UsageInfo) {
	if sessionID == "" {
		return
	}
	entry := s.Get(sessionID)
	if entry == nil {
		return
	}
	data, _ := json.Marshal(map[string]any{"usage": usage})
	entry.Append(Event{
		Timestamp: time.Now().Format(time.RFC3339Nano),
		Type:      "response_done",
		Data:      data,
	})
}

// LogError records an error against a session, creating it if missing so
// errors stay observable even when no prior request was logged.
func (s *Store) LogError(sessionID, errMsg string) {
	if sessionID == "" {
		return
	}
	entry := s.Get(sessionID)
	if entry == nil {
		entry = s.GetOrCreate(sessionID)
	}
	data, _ := json.Marshal(map[string]string{"error": errMsg})
	entry.Append(Event{
		Timestamp: time.Now().Format(time.RFC3339Nano),
		Type:      "error",
		Data:      data,
	})
}

// StartCleanup spawns a goroutine that prunes sessions older than maxAge
// at the given interval. Removes the .jsonl log, the attachment directory,
// and the in-memory entry.
func (s *Store) StartCleanup(maxAge, interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for range ticker.C {
			s.cleanupOldSessions(maxAge)
		}
	}()
}

func (s *Store) cleanupOldSessions(maxAge time.Duration) {
	entries, err := os.ReadDir(s.Dir)
	if err != nil {
		return
	}
	cutoff := time.Now().Add(-maxAge)
	for _, e := range entries {
		name := e.Name()
		var sessionID string
		var modTime time.Time

		if e.IsDir() {
			sessionID = name
			info, err := e.Info()
			if err != nil {
				continue
			}
			modTime = info.ModTime()
		} else if strings.HasSuffix(name, ".jsonl") {
			sessionID = strings.TrimSuffix(name, ".jsonl")
			info, err := e.Info()
			if err != nil {
				continue
			}
			modTime = info.ModTime()
		} else {
			continue
		}

		if modTime.After(cutoff) {
			continue
		}

		logPath := filepath.Join(s.Dir, sessionID+".jsonl")
		if err := os.Remove(logPath); err != nil && !os.IsNotExist(err) {
			log.Printf("session cleanup: failed to remove %s: %v", logPath, err)
		}
		filesDir := filepath.Join(s.Dir, sessionID)
		if err := os.RemoveAll(filesDir); err != nil {
			log.Printf("session cleanup: failed to remove %s: %v", filesDir, err)
		}

		s.mu.Lock()
		if entry, ok := s.sessions[sessionID]; ok {
			entry.mu.Lock()
			if entry.logFile != nil {
				entry.logFile.Close()
			}
			entry.mu.Unlock()
			delete(s.sessions, sessionID)
		}
		s.mu.Unlock()

		log.Printf("session cleanup: removed expired session %s (last modified: %s)", sessionID, modTime.Format(time.RFC3339))
	}
}
