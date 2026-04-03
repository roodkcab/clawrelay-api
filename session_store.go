package main

import (
	_ "embed"
	"encoding/json"
	"html/template"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

//go:embed session_viewer.html
var sessionViewerHTML string
var sessionViewerTmpl = template.Must(template.New("session").Parse(sessionViewerHTML))

// SessionEvent represents a single event in a session's conversation history.
type SessionEvent struct {
	Timestamp string          `json:"timestamp"`
	Type      string          `json:"type"`      // "request", "response_delta", "response_done", "error"
	Data      json.RawMessage `json:"data"`
}

// sessionEntry holds the in-memory state for one session.
type sessionEntry struct {
	mu          sync.Mutex
	events      []SessionEvent
	subscribers map[chan SessionEvent]struct{}
	logFile     *os.File
}

// sessionStore manages all active sessions.
type sessionStore struct {
	mu       sync.RWMutex
	sessions map[string]*sessionEntry
	dir      string // directory to store session log files
}

var globalSessionStore = &sessionStore{
	sessions: make(map[string]*sessionEntry),
	dir:      "sessions",
}

func (s *sessionStore) getOrCreate(sessionID string) *sessionEntry {
	s.mu.RLock()
	entry, ok := s.sessions[sessionID]
	s.mu.RUnlock()
	if ok {
		return entry
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	// Double-check
	if entry, ok = s.sessions[sessionID]; ok {
		return entry
	}

	entry = &sessionEntry{
		subscribers: make(map[chan SessionEvent]struct{}),
	}

	// Open log file (append mode), load existing events
	os.MkdirAll(s.dir, 0755)
	logPath := filepath.Join(s.dir, sessionID+".jsonl")
	if data, err := os.ReadFile(logPath); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			var ev SessionEvent
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

func (s *sessionStore) get(sessionID string) *sessionEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.sessions[sessionID]
}

// Append adds an event to a session, writes to log file, and broadcasts to subscribers.
func (e *sessionEntry) Append(ev SessionEvent) {
	e.mu.Lock()
	e.events = append(e.events, ev)

	// Write to log file
	if e.logFile != nil {
		if data, err := json.Marshal(ev); err == nil {
			e.logFile.Write(append(data, '\n'))
		}
	}

	// Copy subscribers to avoid holding lock during send
	subs := make([]chan SessionEvent, 0, len(e.subscribers))
	for ch := range e.subscribers {
		subs = append(subs, ch)
	}
	e.mu.Unlock()

	// Broadcast to subscribers (non-blocking)
	for _, ch := range subs {
		select {
		case ch <- ev:
		default:
			// subscriber too slow, skip
		}
	}
}

// Subscribe returns a channel of events and a cancel function.
func (e *sessionEntry) Subscribe() (history []SessionEvent, ch chan SessionEvent, cancel func()) {
	ch = make(chan SessionEvent, 256)
	e.mu.Lock()
	history = make([]SessionEvent, len(e.events))
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

// Events returns a snapshot of all events.
func (e *sessionEntry) Events() []SessionEvent {
	e.mu.Lock()
	defer e.mu.Unlock()
	result := make([]SessionEvent, len(e.events))
	copy(result, e.events)
	return result
}

// ---- Helper functions for recording session events ----

func sessionLogRequest(sessionID string, req *ChatCompletionRequest) {
	if sessionID == "" {
		return
	}
	entry := globalSessionStore.getOrCreate(sessionID)
	data, _ := json.Marshal(req)
	entry.Append(SessionEvent{
		Timestamp: time.Now().Format(time.RFC3339Nano),
		Type:      "request",
		Data:      data,
	})
}

func sessionLogDelta(sessionID string, text string) {
	if sessionID == "" || text == "" {
		return
	}
	entry := globalSessionStore.get(sessionID)
	if entry == nil {
		return
	}
	data, _ := json.Marshal(map[string]string{"text": text})
	entry.Append(SessionEvent{
		Timestamp: time.Now().Format(time.RFC3339Nano),
		Type:      "response_delta",
		Data:      data,
	})
}

func sessionLogToolUse(sessionID string, name string, id string, input string) {
	if sessionID == "" {
		return
	}
	entry := globalSessionStore.get(sessionID)
	if entry == nil {
		return
	}
	data, _ := json.Marshal(map[string]string{"tool": name, "id": id, "input": input})
	entry.Append(SessionEvent{
		Timestamp: time.Now().Format(time.RFC3339Nano),
		Type:      "tool_use",
		Data:      data,
	})
}

func sessionLogDone(sessionID string, usage *UsageInfo) {
	if sessionID == "" {
		return
	}
	entry := globalSessionStore.get(sessionID)
	if entry == nil {
		return
	}
	data, _ := json.Marshal(map[string]interface{}{"usage": usage})
	entry.Append(SessionEvent{
		Timestamp: time.Now().Format(time.RFC3339Nano),
		Type:      "response_done",
		Data:      data,
	})
}

func sessionLogError(sessionID string, errMsg string) {
	if sessionID == "" {
		return
	}
	entry := globalSessionStore.get(sessionID)
	if entry == nil {
		entry = globalSessionStore.getOrCreate(sessionID)
	}
	data, _ := json.Marshal(map[string]string{"error": errMsg})
	entry.Append(SessionEvent{
		Timestamp: time.Now().Format(time.RFC3339Nano),
		Type:      "error",
		Data:      data,
	})
}

// startSessionCleanup runs a background goroutine that periodically removes
// sessions (log files, attachment directories, and in-memory state) older than maxAge.
func (s *sessionStore) startSessionCleanup(maxAge time.Duration, interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for range ticker.C {
			s.cleanupOldSessions(maxAge)
		}
	}()
}

func (s *sessionStore) cleanupOldSessions(maxAge time.Duration) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return
	}
	cutoff := time.Now().Add(-maxAge)
	for _, e := range entries {
		name := e.Name()
		// Handle both session log files (.jsonl) and session directories
		var sessionID string
		var modTime time.Time

		if e.IsDir() {
			// Session attachment directory — check mod time
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

		// Remove log file
		logPath := filepath.Join(s.dir, sessionID+".jsonl")
		if err := os.Remove(logPath); err != nil && !os.IsNotExist(err) {
			log.Printf("session cleanup: failed to remove %s: %v", logPath, err)
		}
		// Remove attachment directory
		filesDir := filepath.Join(s.dir, sessionID)
		if err := os.RemoveAll(filesDir); err != nil {
			log.Printf("session cleanup: failed to remove %s: %v", filesDir, err)
		}

		// Remove from in-memory store
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

// ---- WebSocket + HTML handlers ----

var wsUpgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// sessionWSHandler handles WebSocket connections at /session/{id}/ws
func sessionWSHandler(w http.ResponseWriter, r *http.Request) {
	// Extract session ID from path: /session/{id}/ws
	parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
	if len(parts) < 3 || parts[0] != "session" || parts[2] != "ws" {
		http.NotFound(w, r)
		return
	}
	sessionID := parts[1]

	entry := globalSessionStore.get(sessionID)
	if entry == nil {
		// Create it so we can subscribe for future events
		entry = globalSessionStore.getOrCreate(sessionID)
	}

	conn, err := wsUpgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	history, ch, cancel := entry.Subscribe()
	defer cancel()

	// Send history
	for _, ev := range history {
		data, _ := json.Marshal(ev)
		if err := conn.WriteMessage(websocket.TextMessage, data); err != nil {
			return
		}
	}

	// Send a "history_end" marker
	marker, _ := json.Marshal(map[string]string{"type": "history_end"})
	conn.WriteMessage(websocket.TextMessage, marker)

	// Read from client (just for detecting close)
	go func() {
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				cancel()
				return
			}
		}
	}()

	// Stream new events
	for ev := range ch {
		data, _ := json.Marshal(ev)
		if err := conn.WriteMessage(websocket.TextMessage, data); err != nil {
			return
		}
	}
}

// sessionPageHandler serves the HTML viewer at /session/{id}
func sessionPageHandler(w http.ResponseWriter, r *http.Request) {
	// Extract session ID from path: /session/{id}
	parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
	if len(parts) < 2 || parts[0] != "session" {
		http.NotFound(w, r)
		return
	}
	sessionID := parts[1]

	// If it's a /ws path, delegate to websocket handler
	if len(parts) >= 3 && parts[2] == "ws" {
		sessionWSHandler(w, r)
		return
	}

	// Load session events and inline as JSON so content renders immediately
	// without waiting for WebSocket connection
	var eventsJSON string
	entry := globalSessionStore.get(sessionID)
	if entry == nil {
		entry = globalSessionStore.getOrCreate(sessionID)
	}
	events := entry.Events()
	if data, err := json.Marshal(events); err == nil {
		eventsJSON = string(data)
	} else {
		eventsJSON = "[]"
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	sessionViewerTmpl.Execute(w, struct {
		SessionID  string
		EventsJSON template.JS
	}{sessionID, template.JS(eventsJSON)})
}

// sessionListHandler serves a JSON list of all session IDs
func sessionListHandler(w http.ResponseWriter, r *http.Request) {
	files, err := os.ReadDir(globalSessionStore.dir)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("[]"))
		return
	}
	var sessions []map[string]interface{}
	for _, f := range files {
		if !f.IsDir() && strings.HasSuffix(f.Name(), ".jsonl") {
			id := strings.TrimSuffix(f.Name(), ".jsonl")
			info, _ := f.Info()
			sessions = append(sessions, map[string]interface{}{
				"session_id": id,
				"size":       info.Size(),
				"modified":   info.ModTime().Format(time.RFC3339),
			})
		}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(sessions)
}

