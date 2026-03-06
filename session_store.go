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
<title>Session %s</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github-dark.min.css" id="hljs-theme">
<script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"></script>
<style>
  :root {
    --bg: #0c0e12; --bg-card: #14171e; --bg-card-alt: #191d27;
    --border: #252a35; --border-accent: #333a48;
    --text: #d4dae6; --text-muted: #6b7590; --text-dim: #4a5268;
    --accent-blue: #5b9ef5; --accent-green: #4ac78b; --accent-amber: #e5a84b;
    --accent-red: #e55b5b; --accent-purple: #b48efa;
    --user-bg: #1a2332; --user-border: #264070;
    --code-bg: #0a0d12;
    --shadow: 0 2px 12px rgba(0,0,0,0.3);
  }
  html[data-theme="light"] {
    --bg: #f4f2ee; --bg-card: #ffffff; --bg-card-alt: #faf9f6;
    --border: #e2dfd8; --border-accent: #d0ccc3;
    --text: #2c2c2c; --text-muted: #8a8578; --text-dim: #b0a99c;
    --accent-blue: #2f6fbf; --accent-green: #2a8c5a; --accent-amber: #b07c1e;
    --accent-red: #c03030; --accent-purple: #7c4dbd;
    --user-bg: #eef4fc; --user-border: #b8d4f0;
    --code-bg: #f6f5f2;
    --shadow: 0 2px 12px rgba(0,0,0,0.06);
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Source Serif 4', Georgia, serif; background: var(--bg); color: var(--text); line-height: 1.7; transition: background .3s, color .3s; }

  .header {
    background: var(--bg-card); border-bottom: 1px solid var(--border);
    padding: 14px 24px; display: flex; align-items: center; gap: 14px;
    position: sticky; top: 0; z-index: 100; backdrop-filter: blur(12px);
  }
  .header-title { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 500; color: var(--text-muted); letter-spacing: 0.02em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; flex-shrink: 1; }
  .header-title span { color: var(--accent-blue); }
  .header-right { margin-left: auto; display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%%; background: var(--accent-red); transition: background .3s; }
  .status-dot.connected { background: var(--accent-green); box-shadow: 0 0 8px rgba(74,199,139,0.4); }
  .status-label { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }

  .theme-toggle {
    background: none; border: 1px solid var(--border); color: var(--text-muted);
    width: 34px; height: 34px; border-radius: 8px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; transition: all .2s;
  }
  .theme-toggle:hover { border-color: var(--accent-blue); color: var(--accent-blue); }

  .container { max-width: 1300px; margin: 0 auto; padding: 24px 20px 28px; }

  /* Message blocks */
  .msg { margin-bottom: 20px; animation: fadeUp .35s ease both; }
  @keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

  .msg-user {
    background: var(--user-bg); border: 1px solid var(--user-border);
    border-radius: 12px; padding: 16px 20px;
  }
  .msg-assistant {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px 24px;
    box-shadow: var(--shadow);
  }
  .msg-tool {
    background: var(--bg-card-alt); border: 1px solid var(--border);
    border-left: 3px solid var(--accent-amber);
    border-radius: 4px 10px 10px 4px; padding: 10px 16px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
  }
  .msg-error {
    background: var(--bg-card); border: 1px solid var(--accent-red);
    border-left: 3px solid var(--accent-red);
    border-radius: 4px 10px 10px 4px; padding: 12px 16px;
    color: var(--accent-red);
  }
  .msg-done {
    background: transparent; border: 1px dashed var(--border);
    border-radius: 8px; padding: 8px 14px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: var(--text-dim); text-align: center;
  }

  .msg-label {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 8px; display: flex; align-items: center; gap: 8px;
  }
  .msg-label.user { color: var(--accent-blue); }
  .msg-label.assistant { color: var(--accent-green); }
  .msg-label.tool { color: var(--accent-amber); }
  .msg-label .ts { font-weight: 400; color: var(--text-dim); letter-spacing: 0; }
  .msg-label .model-tag { color: var(--accent-purple); font-weight: 400; }

  /* Markdown content */
  .md-content { font-size: 14.5px; line-height: 1.75; }
  .md-content p { margin-bottom: 12px; }
  .md-content p:last-child { margin-bottom: 0; }
  .md-content h1, .md-content h2, .md-content h3, .md-content h4 {
    font-family: 'Source Serif 4', Georgia, serif;
    margin: 20px 0 10px; font-weight: 600; line-height: 1.3;
  }
  .md-content h1 { font-size: 1.5em; }
  .md-content h2 { font-size: 1.3em; }
  .md-content h3 { font-size: 1.1em; }
  .md-content ul, .md-content ol { margin: 8px 0 12px 20px; }
  .md-content li { margin-bottom: 4px; }
  .md-content blockquote {
    border-left: 3px solid var(--accent-purple); padding: 4px 16px;
    margin: 12px 0; color: var(--text-muted); font-style: italic;
  }
  .md-content code {
    font-family: 'JetBrains Mono', monospace; font-size: 0.88em;
    background: var(--code-bg); padding: 2px 6px; border-radius: 4px;
    border: 1px solid var(--border);
  }
  .md-content pre {
    margin: 12px 0; border-radius: 8px; overflow-x: auto;
    border: 1px solid var(--border);
  }
  .md-content pre code {
    display: block; padding: 14px 18px; background: var(--code-bg);
    border: none; font-size: 13px; line-height: 1.6;
  }
  .md-content table { border-collapse: collapse; margin: 12px 0; width: 100%%; font-size: 13px; }
  .md-content th, .md-content td { border: 1px solid var(--border); padding: 8px 12px; text-align: left; }
  .md-content th { background: var(--bg-card-alt); font-weight: 600; }
  .md-content a { color: var(--accent-blue); text-decoration: none; border-bottom: 1px solid transparent; }
  .md-content a:hover { border-bottom-color: var(--accent-blue); }
  .md-content hr { border: none; border-top: 1px solid var(--border); margin: 20px 0; }
  .md-content img { max-width: 100%%; border-radius: 8px; }

  /* Streaming cursor */
  .cursor {
    display: inline-block; width: 2px; height: 1em; background: var(--accent-green);
    animation: pulse 1s ease infinite; vertical-align: text-bottom; margin-left: 2px;
  }
  @keyframes pulse { 0%%,100%% { opacity: 1; } 50%% { opacity: 0.2; } }

  /* Scroll-to-bottom button */
  .scroll-btn {
    position: fixed; bottom: 24px; right: 24px; z-index: 200;
    width: 44px; height: 44px; border-radius: 50%%;
    background: var(--accent-blue); color: #fff; border: none;
    cursor: pointer; display: none; align-items: center; justify-content: center;
    font-size: 22px; font-weight: bold; box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    animation: bounceArrow 1.2s ease infinite;
    transition: background .2s;
  }
  .scroll-btn:hover { background: var(--accent-purple); }
  .scroll-btn.show { display: flex; }
  @keyframes bounceArrow {
    0%%,100%% { transform: translateY(0); }
    50%% { transform: translateY(5px); }
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 3px; }
</style>
</head>
<body>
<div class="header">
  <div class="header-title">session <span>%s</span></div>
  <div class="header-right">
    <div class="status-dot" id="statusDot"></div>
    <span class="status-label" id="statusLabel">connecting</span>
    <button class="theme-toggle" id="themeBtn" title="Toggle theme">&#9790;</button>
  </div>
</div>
<div class="container" id="events"><div class="msg msg-done" id="autoLoadHint"></div></div>
<button class="scroll-btn" id="scrollBtn" title="New messages"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 10 12 16 18 10"/></svg></button>
<script>
marked.setOptions({
  highlight: function(code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try { return hljs.highlight(code, { language: lang }).value; } catch(e) {}
    }
    try { return hljs.highlightAuto(code).value; } catch(e) {}
    return code;
  },
  breaks: false,
  gfm: true
});

// Theme
const html = document.documentElement;
const themeBtn = document.getElementById('themeBtn');
const hljsLink = document.getElementById('hljs-theme');
function setTheme(t) {
  html.dataset.theme = t;
  localStorage.setItem('theme', t);
  themeBtn.innerHTML = t === 'light' ? '&#9728;' : '&#9790;';
  hljsLink.href = t === 'light'
    ? 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github.min.css'
    : 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github-dark.min.css';
}
setTheme(localStorage.getItem('theme') || 'dark');
themeBtn.onclick = () => setTheme(html.dataset.theme === 'dark' ? 'light' : 'dark');

const sessionId = location.pathname.split('/').filter(Boolean)[1];
const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = wsProto + '//' + location.host + '/session/' + sessionId + '/ws';
const eventsEl = document.getElementById('events');
const statusDot = document.getElementById('statusDot');
const statusLabel = document.getElementById('statusLabel');
const autoLoadHint = document.getElementById('autoLoadHint');
let streamingEl = null;
let streamingContent = '';

function renderMarkdown(text) {
  try { return marked.parse(text); } catch(e) { return escHtml(text); }
}

function connect() {
  const ws = new WebSocket(wsUrl);
  ws.onopen = () => {
    statusDot.classList.add('connected');
    statusLabel.textContent = 'live';
  };
  ws.onclose = () => {
    statusDot.classList.remove('connected');
    statusLabel.textContent = 'disconnected';
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
  const ts = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : '';

  if (ev.type === 'response_delta') {
    if (!streamingEl) {
      streamingEl = document.createElement('div');
      streamingEl.className = 'msg msg-assistant';
      streamingEl.innerHTML = '<div class="msg-label assistant"><span>Assistant</span><span class="ts">' + ts + '</span></div><div class="md-content streaming-target"></div>';
      eventsEl.insertBefore(streamingEl, autoLoadHint);
      streamingContent = '';
    }
    streamingContent += (data.text || '');
    const contentEl = streamingEl.querySelector('.streaming-target');
    contentEl.innerHTML = renderMarkdown(streamingContent);
    // Re-highlight code blocks
    contentEl.querySelectorAll('pre code').forEach(b => { if (!b.dataset.highlighted) { hljs.highlightElement(b); } });
    // Cursor
    let cursor = streamingEl.querySelector('.cursor');
    if (!cursor) { cursor = document.createElement('span'); cursor.className = 'cursor'; }
    contentEl.appendChild(cursor);
    autoScroll();
    return;
  }

  // Finalize streaming
  if (streamingEl && (ev.type === 'response_done' || ev.type === 'request' || ev.type === 'tool_use')) {
    const cursor = streamingEl.querySelector('.cursor');
    if (cursor) cursor.remove();
    // Final re-render with highlight
    const contentEl = streamingEl.querySelector('.streaming-target');
    if (contentEl && streamingContent) {
      contentEl.innerHTML = renderMarkdown(streamingContent);
      contentEl.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));
    }
    streamingEl = null;
    streamingContent = '';
  }

  const div = document.createElement('div');

  switch(ev.type) {
    case 'request': {
      div.className = 'msg msg-user';
      const model = data.model || '?';
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
      div.innerHTML = '<div class="msg-label user"><span>User</span><span class="model-tag">' + escHtml(model) + '</span><span class="ts">' + ts + '</span></div><div class="md-content">' + renderMarkdown(preview) + '</div>';
      break;
    }
    case 'tool_use':
      div.className = 'msg msg-tool';
      div.innerHTML = '<div class="msg-label tool"><span>Tool Call</span><span class="ts">' + ts + '</span></div>' + escHtml(data.tool || '') + ' <span style="color:var(--text-dim)">' + escHtml(data.id || '') + '</span>';
      break;
    case 'response_done':
      return; // handled by persistent footer hint
    case 'error':
      div.className = 'msg msg-error';
      div.innerHTML = '<div class="msg-label" style="color:var(--accent-red)"><span>Error</span><span class="ts">' + ts + '</span></div>' + escHtml(data.error || JSON.stringify(data));
      break;
    default:
      div.className = 'msg msg-assistant';
      div.innerHTML = '<div class="msg-label assistant"><span>Event</span><span class="ts">' + ts + '</span></div><div class="md-content"><pre><code>' + escHtml(JSON.stringify(data, null, 2)) + '</code></pre></div>';
  }

  eventsEl.insertBefore(div, autoLoadHint);
  autoScroll();
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

const scrollBtn = document.getElementById('scrollBtn');
let userAtBottom = true;

function isAtBottom() {
  return window.innerHeight + window.scrollY >= document.body.scrollHeight - 150;
}

window.addEventListener('scroll', () => {
  userAtBottom = isAtBottom();
  if (userAtBottom) scrollBtn.classList.remove('show');
});

scrollBtn.onclick = () => {
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
  scrollBtn.classList.remove('show');
};

function autoScroll() {
  if (userAtBottom) {
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
  } else {
    scrollBtn.classList.add('show');
  }
}

// Set auto-load hint based on browser language
(function() {
  const lang = (navigator.language || '').toLowerCase();
  let text = 'Auto-loading latest messages...';
  if (lang.startsWith('zh')) text = '自动加载最新消息中...';
  else if (lang.startsWith('ja')) text = '最新メッセージを自動読み込み中...';
  autoLoadHint.textContent = text;
})();

connect();
</script>
</body>
</html>
`
