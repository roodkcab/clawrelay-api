package sessions

import (
	_ "embed"
	"encoding/json"
	"html/template"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gorilla/websocket"
)

//go:embed session_viewer.html
var sessionViewerHTML string

var sessionViewerTmpl = template.Must(template.New("session").Parse(sessionViewerHTML))

var wsUpgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// WSHandler returns an http.HandlerFunc serving WebSocket sessions at
// /session/{id}/ws. Clients receive the full event history first, a marker
// event, then a live stream of new events.
func (s *Store) WSHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
		if len(parts) < 3 || parts[0] != "session" || parts[2] != "ws" {
			http.NotFound(w, r)
			return
		}
		sessionID := parts[1]

		entry := s.Get(sessionID)
		if entry == nil {
			entry = s.GetOrCreate(sessionID)
		}

		conn, err := wsUpgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("WebSocket upgrade error: %v", err)
			return
		}
		defer conn.Close()

		history, ch, cancel := entry.Subscribe()
		defer cancel()

		for _, ev := range history {
			data, _ := json.Marshal(ev)
			if err := conn.WriteMessage(websocket.TextMessage, data); err != nil {
				return
			}
		}

		marker, _ := json.Marshal(map[string]string{"type": "history_end"})
		conn.WriteMessage(websocket.TextMessage, marker)

		go func() {
			for {
				if _, _, err := conn.ReadMessage(); err != nil {
					cancel()
					return
				}
			}
		}()

		for ev := range ch {
			data, _ := json.Marshal(ev)
			if err := conn.WriteMessage(websocket.TextMessage, data); err != nil {
				return
			}
		}
	}
}

// PageHandler returns an http.HandlerFunc serving the HTML viewer at
// /session/{id}, transparently delegating to the WS handler when a `/ws`
// suffix is present.
func (s *Store) PageHandler() http.HandlerFunc {
	wsHandler := s.WSHandler()
	return func(w http.ResponseWriter, r *http.Request) {
		parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
		if len(parts) < 2 || parts[0] != "session" {
			http.NotFound(w, r)
			return
		}
		sessionID := parts[1]

		if len(parts) >= 3 && parts[2] == "ws" {
			wsHandler(w, r)
			return
		}

		// Inline events as JSON so the page renders without waiting on WS.
		// Use Get (not GetOrCreate) to avoid creating empty sessions on view.
		eventsJSON := "[]"
		if entry := s.Get(sessionID); entry != nil {
			events := entry.Events()
			if data, err := json.Marshal(events); err == nil {
				eventsJSON = string(data)
			}
		}

		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store, private")
		sessionViewerTmpl.Execute(w, struct {
			SessionID  string
			EventsJSON template.JS
		}{sessionID, template.JS(eventsJSON)})
	}
}

// ListHandler returns an http.HandlerFunc serving a JSON list of sessions
// at /sessions. Each entry includes session_id, size, and modified time.
func (s *Store) ListHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		files, err := os.ReadDir(s.Dir)
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte("[]"))
			return
		}
		var out []map[string]any
		for _, f := range files {
			if !f.IsDir() && strings.HasSuffix(f.Name(), ".jsonl") {
				id := strings.TrimSuffix(f.Name(), ".jsonl")
				info, _ := f.Info()
				out = append(out, map[string]any{
					"session_id": id,
					"size":       info.Size(),
					"modified":   info.ModTime().Format(time.RFC3339),
				})
			}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(out)
	}
}
