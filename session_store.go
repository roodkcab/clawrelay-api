package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

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

func sessionLogToolUse(sessionID string, name string, id string) {
	if sessionID == "" {
		return
	}
	entry := globalSessionStore.get(sessionID)
	if entry == nil {
		return
	}
	data, _ := json.Marshal(map[string]string{"tool": name, "id": id})
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

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	fmt.Fprintf(w, sessionViewerHTML, sessionID, sessionID)
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

const sessionViewerHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Session: %s</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace; background: #0d1117; color: #c9d1d9; }
  .header { background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 20px; display: flex; align-items: center; gap: 12px; position: sticky; top: 0; z-index: 10; }
  .header h1 { font-size: 16px; color: #58a6ff; }
  .status { font-size: 12px; padding: 3px 8px; border-radius: 12px; }
  .status.connected { background: #238636; color: #fff; }
  .status.disconnected { background: #da3633; color: #fff; }
  .container { max-width: 960px; margin: 0 auto; padding: 20px; }
  .event { margin-bottom: 8px; border-left: 3px solid #30363d; padding: 8px 12px; background: #161b22; border-radius: 0 6px 6px 0; }
  .event.request { border-left-color: #58a6ff; }
  .event.response_delta { border-left-color: #3fb950; }
  .event.response_done { border-left-color: #8b949e; }
  .event.tool_use { border-left-color: #d29922; }
  .event.error { border-left-color: #da3633; }
  .event-header { font-size: 11px; color: #8b949e; margin-bottom: 4px; display: flex; gap: 8px; }
  .event-type { font-weight: bold; text-transform: uppercase; }
  .event-type.request { color: #58a6ff; }
  .event-type.response_delta { color: #3fb950; }
  .event-type.tool_use { color: #d29922; }
  .event-type.error { color: #da3633; }
  .event-content { font-size: 13px; white-space: pre-wrap; word-break: break-word; line-height: 1.5; }
  .streaming-block { margin-bottom: 8px; border-left: 3px solid #3fb950; padding: 8px 12px; background: #161b22; border-radius: 0 6px 6px 0; }
  .streaming-block .event-header { font-size: 11px; color: #8b949e; margin-bottom: 4px; }
  .streaming-content { font-size: 13px; white-space: pre-wrap; word-break: break-word; line-height: 1.5; }
  .cursor { display: inline-block; width: 8px; height: 14px; background: #3fb950; animation: blink 1s infinite; vertical-align: text-bottom; }
  @keyframes blink { 0%%,50%% { opacity: 1; } 51%%,100%% { opacity: 0; } }
  .request-summary { color: #8b949e; }
  .request-summary .model { color: #d2a8ff; }
  .request-summary .msg-count { color: #58a6ff; }
</style>
</head>
<body>
<div class="header">
  <h1>Session: %s</h1>
  <span class="status disconnected" id="status">Connecting...</span>
</div>
<div class="container" id="events"></div>
<script>
const sessionId = location.pathname.split('/').filter(Boolean)[1];
const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = wsProto + '//' + location.host + '/session/' + sessionId + '/ws';
const eventsEl = document.getElementById('events');
const statusEl = document.getElementById('status');
let streamingEl = null;
let streamingContent = '';

function connect() {
  const ws = new WebSocket(wsUrl);
  ws.onopen = () => {
    statusEl.textContent = 'Connected';
    statusEl.className = 'status connected';
  };
  ws.onclose = () => {
    statusEl.textContent = 'Disconnected';
    statusEl.className = 'status disconnected';
    setTimeout(connect, 3000);
  };
  ws.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === 'history_end') return;
    renderEvent(ev);
  };
}

function renderEvent(ev) {
  const data = typeof ev.data === 'string' ? JSON.parse(ev.data) : ev.data;

  if (ev.type === 'response_delta') {
    if (!streamingEl) {
      streamingEl = document.createElement('div');
      streamingEl.className = 'streaming-block';
      streamingEl.innerHTML = '<div class="event-header"><span class="event-type response_delta">ASSISTANT</span></div><div class="streaming-content"></div>';
      eventsEl.appendChild(streamingEl);
      streamingContent = '';
    }
    streamingContent += (data.text || '');
    const contentEl = streamingEl.querySelector('.streaming-content');
    contentEl.textContent = streamingContent;
    // Add cursor
    let cursor = streamingEl.querySelector('.cursor');
    if (!cursor) { cursor = document.createElement('span'); cursor.className = 'cursor'; contentEl.appendChild(cursor); }
    autoScroll();
    return;
  }

  // Finalize streaming block
  if (streamingEl && (ev.type === 'response_done' || ev.type === 'request' || ev.type === 'tool_use')) {
    const cursor = streamingEl.querySelector('.cursor');
    if (cursor) cursor.remove();
    streamingEl = null;
    streamingContent = '';
  }

  const div = document.createElement('div');
  div.className = 'event ' + (ev.type || '');

  const ts = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : '';
  let content = '';

  switch(ev.type) {
    case 'request':
      const model = data.model || '?';
      const msgCount = (data.messages || []).length;
      const lastMsg = (data.messages || []).slice(-1)[0];
      let preview = '';
      if (lastMsg) {
        try {
          let c = lastMsg.content;
          if (typeof c === 'string') { preview = c; }
          else if (Array.isArray(c)) { preview = c.map(p => p.text || '').join(''); }
          else { preview = JSON.stringify(c); }
        } catch(e) { preview = JSON.stringify(lastMsg.content); }
      }
      if (preview.length > 300) preview = preview.substring(0, 300) + '...';
      content = '<span class="request-summary">Model: <span class="model">' + escHtml(model) + '</span> | Messages: <span class="msg-count">' + msgCount + '</span></span>\n' + escHtml(preview);
      break;
    case 'tool_use':
      content = 'Tool: ' + escHtml(data.tool || '') + ' (id: ' + escHtml(data.id || '') + ')';
      break;
    case 'response_done':
      const u = data.usage;
      content = u ? 'Tokens - prompt: ' + (u.prompt_tokens||0) + ', completion: ' + (u.completion_tokens||0) + ', total: ' + (u.total_tokens||0) : 'Done';
      break;
    case 'error':
      content = escHtml(data.error || JSON.stringify(data));
      break;
    default:
      content = JSON.stringify(data, null, 2);
  }

  div.innerHTML = '<div class="event-header"><span class="event-type ' + (ev.type||'') + '">' + (ev.type||'').toUpperCase() + '</span><span>' + ts + '</span></div><div class="event-content">' + content + '</div>';
  eventsEl.appendChild(div);
  autoScroll();
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function autoScroll() {
  if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 100) {
    window.scrollTo(0, document.body.scrollHeight);
  }
}

connect();
</script>
</body>
</html>
`
