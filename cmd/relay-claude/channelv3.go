package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"clawrelay-api/pkg/openai"
	"clawrelay-api/pkg/proc"

	"github.com/creack/pty"
)

// Channel-mode V3: instead of driving a non-interactive (headless, token-
// restricted) `claude` over stdin/stdout NDJSON, V3 runs an INTERACTIVE claude
// (in a PTY, so it bills against the subscription) and pushes/receives messages
// through Claude Code's "channels" feature: the relay launches a Bun channel
// MCP server (bridge.ts) that Claude spawns over stdio. Inbound user messages
// are injected as <channel> events; Claude answers by calling the bridge's
// `reply` tool. The relay bridges those to/from its OpenAI HTTP/SSE frontend,
// keyed by session id — the external interface is unchanged.

// v3Config governs the V3 manager.
type v3Config struct {
	BridgeDir   string        // dir containing bridge.ts + node_modules (deployed)
	IdleTTL     time.Duration // kill an interactive claude session after this idle
	MaxSessions int           // cap on concurrent interactive claude processes
	ReplyTotal  time.Duration // overall wait for a turn's reply
}

// v3Msg is one inbound user turn handed to the bridge.
type v3Msg struct {
	ReqID   string `json:"req_id"`
	Content string `json:"content"`
}

// v3ReplyEvent is one piece of a turn routed back to the frontend:
//   - thinking=true: a live progress note (bridge `progress` tool), emitted as a
//     thinking delta — shown as transient progress, NOT part of the answer.
//   - final=false (and thinking=false): a streaming answer chunk (bridge
//     `reply_chunk` tool), emitted as an SSE content delta.
//   - final=true: the terminal reply (bridge `reply` tool), which ends the turn.
type v3ReplyEvent struct {
	text     string
	final    bool
	thinking bool
}

// v3Waiter receives a turn's reply events. ch is buffered and NEVER closed —
// senders guard on done instead, so a late bridge POST can never send on a
// closed channel (the V2 send-on-closed class of bug). done is closed once by
// the frontend (via unregisterWaiter) when it stops reading.
type v3Waiter struct {
	ch   chan v3ReplyEvent
	done chan struct{}
}

// v3Session wraps one interactive claude process (+ its channel bridge).
type v3Session struct {
	sid       string
	cmd       *exec.Cmd
	ptmx      *os.File
	mcpFile   string // per-session --mcp-config path (cleaned up on stop)
	dead      atomic.Bool
	lastUsed  atomic.Int64
	inTurn    atomic.Bool
	deadCh    chan struct{}
	ready     chan struct{} // closed once the channel is registered + event loop is live
	readyOnce sync.Once
	ptyLog    *os.File // optional raw PTY capture for debugging

	bound string // spawn-bound model (for change warnings)
}

// v3DebugPTY, when true, dumps each session's raw claude TUI to /tmp/v3pty-<sid>.log.
var v3DebugPTY = os.Getenv("V3_DEBUG_PTY") == "1"

func (s *v3Session) markUsed()           { s.lastUsed.Store(time.Now().UnixNano()) }
func (s *v3Session) LastUsed() time.Time { return time.Unix(0, s.lastUsed.Load()) }
func (s *v3Session) Dead() bool          { return s.dead.Load() }

// v3Manager owns the interactive claude sessions plus the localhost control
// server the bridges talk to.
type v3Manager struct {
	cfg     v3Config
	ctrlURL string // e.g. http://127.0.0.1:54321 (passed to bridges as RELAY_CTRL)

	mu       sync.Mutex
	sessions map[string]*v3Session
	inbox    map[string]chan v3Msg // sid -> queued inbound turns (drained by /v3/next)
	waiters  map[string]*v3Waiter  // req_id -> reply waiter (chunks + final reply)
}

func newV3Manager(cfg v3Config) *v3Manager {
	if cfg.IdleTTL <= 0 {
		cfg.IdleTTL = 20 * time.Minute
	}
	if cfg.MaxSessions <= 0 {
		cfg.MaxSessions = 30
	}
	if cfg.ReplyTotal <= 0 {
		cfg.ReplyTotal = 10 * time.Minute
	}
	return &v3Manager{
		cfg:      cfg,
		sessions: make(map[string]*v3Session),
		inbox:    make(map[string]chan v3Msg),
		waiters:  make(map[string]*v3Waiter),
	}
}

// startControlServer binds a localhost-only control listener that the bridges
// poll for inbound turns (/v3/next) and post replies to (/v3/reply).
func (m *v3Manager) startControlServer() error {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return fmt.Errorf("v3 control listen: %w", err)
	}
	m.ctrlURL = "http://" + ln.Addr().String()

	mux := http.NewServeMux()
	mux.HandleFunc("/v3/next", m.handleNext)
	mux.HandleFunc("/v3/reply", m.handleReply)
	mux.HandleFunc("/v3/reply_chunk", m.handleReplyChunk)
	mux.HandleFunc("/v3/progress", m.handleProgress)
	srv := &http.Server{Handler: mux}
	go func() {
		if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
			log.Printf("[v3] control server error: %v", err)
		}
	}()
	log.Printf("[v3] control server on %s", m.ctrlURL)
	return nil
}

// handleNext is the bridge's long-poll for the next inbound turn of a session.
func (m *v3Manager) handleNext(w http.ResponseWriter, r *http.Request) {
	sid := r.URL.Query().Get("session")
	if sid == "" {
		http.Error(w, "missing session", http.StatusBadRequest)
		return
	}
	q := m.inboxFor(sid)
	select {
	case msg := <-q:
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(msg)
	case <-time.After(50 * time.Second):
		w.WriteHeader(http.StatusNoContent) // long-poll timeout; bridge re-polls
	case <-r.Context().Done():
		w.WriteHeader(http.StatusNoContent)
	}
}

// handleReply routes the TERMINAL reply (from the bridge's reply tool) to the
// waiting frontend; it ends the turn.
func (m *v3Manager) handleReply(w http.ResponseWriter, r *http.Request) {
	m.routeReply(w, r, v3ReplyEvent{final: true})
}

// handleReplyChunk routes a streaming answer chunk (from the bridge's
// reply_chunk tool) so the frontend can flush it as an SSE content delta; it
// does NOT end the turn.
func (m *v3Manager) handleReplyChunk(w http.ResponseWriter, r *http.Request) {
	m.routeReply(w, r, v3ReplyEvent{})
}

// handleProgress routes a live status note (from the bridge's progress tool),
// emitted as a thinking delta — transient progress, not part of the answer.
func (m *v3Manager) handleProgress(w http.ResponseWriter, r *http.Request) {
	m.routeReply(w, r, v3ReplyEvent{thinking: true})
}

func (m *v3Manager) routeReply(w http.ResponseWriter, r *http.Request, ev v3ReplyEvent) {
	var body struct {
		ReqID string `json:"req_id"`
		Text  string `json:"text"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	ev.text = body.Text
	if m.deliver(body.ReqID, ev) {
		w.WriteHeader(http.StatusOK)
	} else {
		http.Error(w, "no waiter", http.StatusNotFound)
	}
}

// deliver hands a reply event to the waiting frontend. It blocks only until the
// frontend accepts it (the channel is buffered, so normally instant) or the
// frontend has gone away — so a streaming chunk is never silently dropped while
// the frontend is still reading.
func (m *v3Manager) deliver(reqID string, ev v3ReplyEvent) bool {
	m.mu.Lock()
	wt := m.waiters[reqID]
	m.mu.Unlock()
	if wt == nil {
		return false
	}
	select {
	case wt.ch <- ev:
		return true
	case <-wt.done:
		return false
	}
}

func (m *v3Manager) inboxFor(sid string) chan v3Msg {
	m.mu.Lock()
	defer m.mu.Unlock()
	q := m.inbox[sid]
	if q == nil {
		q = make(chan v3Msg, 8)
		m.inbox[sid] = q
	}
	return q
}

func (m *v3Manager) enqueue(sid string, msg v3Msg) {
	m.inboxFor(sid) <- msg
}

func (m *v3Manager) registerWaiter(reqID string) *v3Waiter {
	wt := &v3Waiter{ch: make(chan v3ReplyEvent, 256), done: make(chan struct{})}
	m.mu.Lock()
	m.waiters[reqID] = wt
	m.mu.Unlock()
	return wt
}

func (m *v3Manager) unregisterWaiter(reqID string) {
	m.mu.Lock()
	wt := m.waiters[reqID]
	delete(m.waiters, reqID)
	m.mu.Unlock()
	if wt != nil {
		close(wt.done) // release any sender blocked in deliver()
	}
}

// v3SpawnParams is the spawn-time config derived from the first request.
type v3SpawnParams struct {
	model        string
	systemPrompt string
	workingDir   string
	envVars      map[string]string
	permission   string
	allowedTools string
	addDirs      []string
}

// acquire returns a live interactive session for sid, launching one if needed.
func (m *v3Manager) acquire(sid string, p v3SpawnParams) (*v3Session, error) {
	m.mu.Lock()
	if s := m.sessions[sid]; s != nil && !s.Dead() {
		s.markUsed()
		m.mu.Unlock()
		return s, nil
	}
	delete(m.sessions, sid)
	if m.cfg.MaxSessions > 0 {
		alive := 0
		for _, s := range m.sessions {
			if !s.Dead() && !s.inTurn.Load() {
				alive++
			}
		}
		if alive >= m.cfg.MaxSessions {
			m.evictOldestLocked()
		}
	}
	m.mu.Unlock()

	s, err := m.launch(sid, p)
	if err != nil {
		return nil, err
	}
	m.mu.Lock()
	m.sessions[sid] = s
	m.mu.Unlock()
	return s, nil
}

// launch starts an interactive claude (in a PTY) with the channel bridge.
func (m *v3Manager) launch(sid string, p v3SpawnParams) (*v3Session, error) {
	// The bridge must be registered as a PROJECT MCP server (.mcp.json in the
	// cwd) so the --dangerously-load-development-channels flag recognizes it as
	// a channel. A --mcp-config server is NOT channel-eligible (claude reports
	// "no MCP server configured with that name"). RELAY_CTRL/RELAY_SESSION reach
	// the bridge via the inherited claude env (per-process); the .mcp.json env
	// only overrides the proxy so the bridge's loopback fetches bypass the
	// chroot's HTTP_PROXY (which answers localhost with 503).
	bridge := filepath.Join(m.cfg.BridgeDir, "bridge.ts")
	mcpFile := filepath.Join(p.workingDir, ".mcp.json")
	if err := ensureBridgeInMCPJSON(mcpFile, bridge); err != nil {
		return nil, fmt.Errorf("write .mcp.json: %w", err)
	}

	args := []string{
		"--dangerously-load-development-channels", "server:relaybridge",
		"--dangerously-skip-permissions",
		"--model", p.model,
	}
	if p.systemPrompt != "" {
		args = append(args, "--append-system-prompt", p.systemPrompt)
	}
	if p.permission != "" {
		args = append(args, "--permission-mode", p.permission)
	}
	if p.allowedTools != "" {
		args = append(args, "--allowedTools", p.allowedTools)
	}
	for _, d := range p.addDirs {
		if d != "" {
			args = append(args, "--add-dir", d)
		}
	}

	cmd := exec.Command("claude", args...)
	if p.workingDir != "" {
		cmd.Dir = p.workingDir
	}
	cmd.Env = append(cleanEnv(p.envVars),
		"RELAY_CTRL="+m.ctrlURL,
		"RELAY_SESSION="+sid,
		"TERM=xterm-256color",
	)

	ptmx, err := pty.StartWithSize(cmd, &pty.Winsize{Rows: 50, Cols: 160})
	if err != nil {
		os.Remove(mcpFile)
		return nil, fmt.Errorf("pty start claude: %w", err)
	}

	s := &v3Session{
		sid:     sid,
		cmd:     cmd,
		ptmx:    ptmx,
		mcpFile: mcpFile,
		deadCh:  make(chan struct{}),
		ready:   make(chan struct{}),
		bound:   p.model,
	}
	if v3DebugPTY {
		if f, err := os.Create(filepath.Join(os.TempDir(), "v3pty-"+sid+".log")); err == nil {
			s.ptyLog = f
		}
	}
	s.markUsed()

	go s.drivePTY()
	go func() {
		_ = cmd.Wait()
		s.dead.Store(true)
		close(s.deadCh)
		ptmx.Close()
		// .mcp.json lives in the bot working dir and is reused across this bot's
		// sessions — do not delete it.
		m.onSessionDie(sid)
	}()

	log.Printf("[v3] launched interactive claude session=%s model=%s pid=%d workdir=%s",
		sid, p.model, cmd.Process.Pid, p.workingDir)
	return s, nil
}

// ensureBridgeInMCPJSON merges the relaybridge channel server into the cwd's
// .mcp.json (creating it if absent, preserving any existing servers). The env
// only overrides the proxy: the bridge inherits RELAY_CTRL/RELAY_SESSION from
// claude's per-process env, but must NOT use the chroot's HTTP_PROXY for its
// loopback calls to the relay.
func ensureBridgeInMCPJSON(path, bridgeTS string) error {
	cfg := map[string]any{}
	if data, err := os.ReadFile(path); err == nil {
		_ = json.Unmarshal(data, &cfg)
	}
	servers, _ := cfg["mcpServers"].(map[string]any)
	if servers == nil {
		servers = map[string]any{}
	}
	servers["relaybridge"] = map[string]any{
		"command": "bun",
		"args":    []string{bridgeTS},
		"env": map[string]string{
			"NO_PROXY":    "127.0.0.1,localhost",
			"no_proxy":    "127.0.0.1,localhost",
			"HTTP_PROXY":  "",
			"HTTPS_PROXY": "",
			"http_proxy":  "",
			"https_proxy": "",
		},
	}
	cfg["mcpServers"] = servers
	data, _ := json.MarshalIndent(cfg, "", "  ")
	return os.WriteFile(path, data, 0o644)
}

// startup prompts that interactive claude shows; we answer each once by sending
// Enter (the safe/default option) into the PTY.
var v3Prompts = []struct {
	key string
	re  *regexp.Regexp
}{
	{"devchannels", regexp.MustCompile(`(?i)using this for local development`)},
	{"trust", regexp.MustCompile(`(?i)do you trust|trust the files`)},
	{"bypass", regexp.MustCompile(`(?i)yes, i accept|bypass permissions`)},
	{"mcpconsent", regexp.MustCompile(`(?i)new mcp server|use this mcp server`)},
}

var v3Ansi = regexp.MustCompile(`\x1b\[[0-9;?]*[a-zA-Z]`)

// drivePTY reads the claude TUI and auto-answers startup prompts. The TUI text
// itself is discarded (replies come through the channel, not the terminal).
func (s *v3Session) drivePTY() {
	buf := make([]byte, 8192)
	var roll []byte
	answered := map[string]bool{}
	for {
		n, err := s.ptmx.Read(buf)
		if n > 0 {
			if s.ptyLog != nil {
				s.ptyLog.Write(buf[:n])
			}
			roll = append(roll, buf[:n]...)
			if len(roll) > 16384 {
				roll = roll[len(roll)-16384:]
			}
			flat := string(v3Ansi.ReplaceAll(roll, []byte(" ")))
			for _, pr := range v3Prompts {
				if !answered[pr.key] && pr.re.MatchString(flat) {
					answered[pr.key] = true
					time.Sleep(300 * time.Millisecond)
					s.ptmx.Write([]byte("\r"))
					log.Printf("[v3] session=%s answered startup prompt %q", s.sid, pr.key)
				}
			}
			// The channel is live once claude prints the "messages from
			// server:relaybridge inject directly" notice. Give the event loop a
			// moment to settle, then release any queued turn.
			if strings.Contains(flat, "inject directly") {
				s.readyOnce.Do(func() {
					go func() {
						time.Sleep(2 * time.Second)
						close(s.ready)
						log.Printf("[v3] session=%s channel ready", s.sid)
					}()
				})
			}
		}
		if err != nil {
			return
		}
	}
}

func (s *v3Session) kill() {
	s.dead.Store(true)
	// claude was started via pty.Start (setsid → session/group leader), so a
	// group kill reaps the claude tree and its spawned bun bridge.
	proc.KillGroup(s.cmd)
	if s.ptmx != nil {
		s.ptmx.Close()
	}
}

func (m *v3Manager) onSessionDie(sid string) {
	m.mu.Lock()
	if s := m.sessions[sid]; s != nil && s.Dead() {
		delete(m.sessions, sid)
	}
	m.mu.Unlock()
}

func (m *v3Manager) evictOldestLocked() {
	var oldest string
	var t time.Time
	for k, s := range m.sessions {
		if s.inTurn.Load() {
			continue
		}
		if oldest == "" || s.LastUsed().Before(t) {
			oldest, t = k, s.LastUsed()
		}
	}
	if oldest == "" {
		return
	}
	s := m.sessions[oldest]
	delete(m.sessions, oldest)
	log.Printf("[v3] capacity evict session=%s", oldest)
	go s.kill()
}

// StartReaper kills sessions idle longer than IdleTTL.
func (m *v3Manager) StartReaper(ctx context.Context) {
	go func() {
		t := time.NewTicker(time.Minute)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				m.reapOnce()
			}
		}
	}()
}

func (m *v3Manager) reapOnce() {
	cutoff := time.Now().Add(-m.cfg.IdleTTL)
	m.mu.Lock()
	var toKill []*v3Session
	for k, s := range m.sessions {
		if s.Dead() {
			delete(m.sessions, k)
			continue
		}
		if !s.inTurn.Load() && s.LastUsed().Before(cutoff) {
			toKill = append(toKill, s)
			delete(m.sessions, k)
		}
	}
	m.mu.Unlock()
	for _, s := range toKill {
		log.Printf("[v3] reaping idle session=%s", s.sid)
		s.kill()
	}
}

// Stop kills every interactive session (shutdown).
func (m *v3Manager) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, s := range m.sessions {
		s.kill()
	}
	m.sessions = make(map[string]*v3Session)
}

func (m *v3Manager) snapshot() []map[string]any {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]map[string]any, 0, len(m.sessions))
	for k, s := range m.sessions {
		out = append(out, map[string]any{
			"session_id": k,
			"model":      s.bound,
			"last_used":  s.LastUsed().Format(time.RFC3339),
			"in_turn":    s.inTurn.Load(),
			"dead":       s.Dead(),
		})
	}
	return out
}

func v3ReqID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return "v3-" + hex.EncodeToString(b)
}

// handleChannelV3Response serves one /v1/chat/completions turn through an
// interactive (subscription-billed) claude session via the channel bridge.
func handleChannelV3Response(w http.ResponseWriter, r *http.Request, req *openai.ChatCompletionRequest, model string, includeUsage bool) {
	sid := req.SessionID
	if sid == "" {
		sid = newUUID() // ephemeral one-off session
	}
	systemPrompt := extractSystemPrompt(req.Messages)

	var sessionDir string
	if req.SessionID != "" {
		sessionDir = filepath.Join(sessionStore.AbsDir(), req.SessionID, "files")
	}
	content, tempFiles, ok := lastUserTurnContent(req.Messages, sessionDir)
	if len(tempFiles) > 0 {
		defer func() {
			for _, f := range tempFiles {
				os.Remove(f)
			}
		}()
	}
	if !ok || strings.TrimSpace(content) == "" {
		openai.WriteError(w, http.StatusBadRequest, "invalid_request_error", "no user message")
		return
	}

	chatID := openai.GenerateChatID()
	created := time.Now().Unix()

	p := v3SpawnParams{
		model:        model,
		systemPrompt: systemPrompt,
		workingDir:   req.WorkingDir,
		envVars:      req.EnvVars,
		permission:   req.PermissionMode,
		allowedTools: req.AllowedTools,
		addDirs:      req.AddDirs,
	}
	sess, err := v3Mgr.acquire(sid, p)
	if err != nil {
		log.Printf("[v3] acquire failed session=%s: %v", sid, err)
		openai.WriteError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	sess.inTurn.Store(true)
	defer func() { sess.inTurn.Store(false); sess.markUsed() }()

	reqID := v3ReqID()
	waiter := v3Mgr.registerWaiter(reqID)
	defer v3Mgr.unregisterWaiter(reqID)

	// SSE headers + initial ping (also keeps the client alive during a slow
	// first-launch readiness wait).
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	if flusher == nil {
		return
	}
	fmt.Fprintf(w, ": ping\n\n")
	flusher.Flush()

	// Instant feedback so the client never shows a frozen blank while the
	// session cold-starts or claude thinks before its first output.
	reqStart := time.Now()
	v3EmitThinkingDelta(w, flusher, chatID, created, model, "🤔 正在处理你的请求…")

	// Heartbeat: interactive claude can be silent for minutes during heavy work
	// (tool calls, big queries). A short visible "still working" thinking tick
	// keeps the client alive (covers the adapter's sock_read idle timeout) AND
	// shows movement so it never looks dead.
	heartbeat := time.NewTicker(15 * time.Second)
	defer heartbeat.Stop()

	// Phase 1: wait until the session's channel is live (first launch only).
	readyDeadline := time.After(2 * time.Minute)
waitReady:
	for {
		select {
		case <-sess.ready:
			break waitReady
		case <-heartbeat.C:
			v3EmitThinkingDelta(w, flusher, chatID, created, model,
				fmt.Sprintf("\n⏳ 正在准备会话…（已 %ds）", int(time.Since(reqStart).Seconds())))
		case <-sess.deadCh:
			fmt.Fprintf(w, "data: %s\n\n", v3ErrChunk(chatID, created, model, "claude session ended during startup"))
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		case <-r.Context().Done():
			return
		case <-readyDeadline:
			fmt.Fprintf(w, "data: %s\n\n", v3ErrChunk(chatID, created, model, "channel startup timeout"))
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			log.Printf("[v3] channel startup timeout session=%s", sid)
			return
		}
	}

	// Phase 2: send the turn into the live session.
	log.Printf("[v3] turn start session=%s req=%s model=%s chat=%s", sid, reqID, model, chatID)
	v3Mgr.enqueue(sid, v3Msg{ReqID: reqID, Content: content})

	// streamed tracks whether we already flushed ≥1 progressive chunk; if so, an
	// abnormal end (session death / timeout) finishes the SSE cleanly instead of
	// overwriting the partial answer with an error.
	streamed := false
	// Measure silence from the last visible output (the instant note at reqStart),
	// not from entering Phase 2 — otherwise the cold-start gap before claude's
	// first token wrongly counts as "recent activity" and skips the heartbeat.
	lastActivity := reqStart
	// Idle-based turn timeout: reset on every claude event so a long-but-
	// progressing task is never cut. Fires only if claude produces NOTHING
	// (no progress/chunk/reply) for ReplyTotal — a genuinely stuck session (the
	// relay heartbeat does NOT reset it). Keeping the SSE open instead of
	// erroring at an absolute cap is what lets wuji_tools' >10min "switch to
	// background + push when done" path actually deliver the eventual answer.
	idle := time.NewTimer(v3Mgr.cfg.ReplyTotal)
	defer idle.Stop()
	for {
		select {
		case ev := <-waiter.ch:
			lastActivity = time.Now()
			if !idle.Stop() {
				select {
				case <-idle.C:
				default:
				}
			}
			idle.Reset(v3Mgr.cfg.ReplyTotal)
			if ev.thinking {
				// live progress note (progress tool): emit as a thinking delta,
				// not part of the answer; do not mark streamed / log as answer.
				v3EmitThinkingDelta(w, flusher, chatID, created, model, ev.text)
				continue
			}
			if !ev.final {
				// progressive answer chunk (reply_chunk): flush as a content delta, keep waiting.
				v3EmitContentDelta(w, flusher, chatID, created, model, ev.text)
				if ev.text != "" {
					streamed = true
					sessionStore.LogDelta(req.SessionID, ev.text)
				}
				continue
			}
			// terminal reply: emit the final piece (if any) + finish + usage + [DONE].
			v3EmitClose(w, flusher, chatID, created, model, ev.text, includeUsage)
			if ev.text != "" {
				sessionStore.LogDelta(req.SessionID, ev.text)
			}
			sessionStore.LogDone(req.SessionID, nil)
			log.Printf("[v3] turn end session=%s req=%s streamed=%v finalLen=%d", sid, reqID, streamed, len(ev.text))
			return
		case <-heartbeat.C:
			// When claude has been silent (heavy work between progress notes),
			// emit a visible "still working" tick so the client shows movement
			// instead of looking dead; otherwise a cheap keepalive comment.
			if time.Since(lastActivity) >= 12*time.Second {
				v3EmitThinkingDelta(w, flusher, chatID, created, model,
					fmt.Sprintf("\n⏳ 处理中…（已 %ds）", int(time.Since(reqStart).Seconds())))
			} else {
				fmt.Fprintf(w, ": keepalive\n\n")
				flusher.Flush()
			}
		case <-sess.deadCh:
			if streamed {
				v3EmitClose(w, flusher, chatID, created, model, "", includeUsage)
			} else {
				fmt.Fprintf(w, "data: %s\n\n", v3ErrChunk(chatID, created, model, "claude session ended"))
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
			}
			sessionStore.LogDone(req.SessionID, nil)
			log.Printf("[v3] session=%s died mid-turn req=%s streamed=%v", sid, reqID, streamed)
			return
		case <-r.Context().Done():
			log.Printf("[v3] client disconnected session=%s req=%s (session kept alive)", sid, reqID)
			return
		case <-idle.C:
			// No claude activity at all for ReplyTotal → treat the session as stuck.
			if streamed {
				v3EmitClose(w, flusher, chatID, created, model, "", includeUsage)
			} else {
				fmt.Fprintf(w, "data: %s\n\n", v3ErrChunk(chatID, created, model, "任务长时间无响应"))
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
			}
			log.Printf("[v3] idle timeout (no activity %s) session=%s req=%s streamed=%v", v3Mgr.cfg.ReplyTotal, sid, reqID, streamed)
			return
		}
	}
}

// v3EmitThinkingDelta sends one live progress note as a thinking delta
// (delta.thinking, empty content). Clients render it as transient progress, not
// as part of the answer. Does NOT finish the turn.
func v3EmitThinkingDelta(w http.ResponseWriter, flusher http.Flusher, chatID string, created int64, model, text string) {
	if text == "" {
		return
	}
	msg := openai.NewChatMessage("assistant", "")
	msg.Thinking = text
	chunk := openai.ChatCompletionResponse{
		ID: chatID, Object: "chat.completion.chunk", Created: created, Model: model,
		Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: msg}},
	}
	data, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

// v3EmitContentDelta sends one progressive assistant content delta (a streaming
// chunk of the answer). It does NOT finish the turn — no finish/usage/[DONE].
func v3EmitContentDelta(w http.ResponseWriter, flusher http.Flusher, chatID string, created int64, model, text string) {
	if text == "" {
		return
	}
	chunk := openai.ChatCompletionResponse{
		ID: chatID, Object: "chat.completion.chunk", Created: created, Model: model,
		Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: openai.NewChatMessage("assistant", text)}},
	}
	data, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

// v3EmitClose ends the SSE turn: an optional final content delta, then the
// finish chunk, optional usage, and [DONE]. When the answer streamed via
// reply_chunk, finalText is the last remaining piece (often empty) and the
// earlier deltas have already gone out; when not streamed, finalText is the
// whole answer (the legacy single-chunk behavior).
func v3EmitClose(w http.ResponseWriter, flusher http.Flusher, chatID string, created int64, model, finalText string, includeUsage bool) {
	if finalText != "" {
		chunk := openai.ChatCompletionResponse{
			ID: chatID, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: openai.NewChatMessage("assistant", finalText)}},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
	}

	finish := "stop"
	fin := openai.ChatCompletionResponse{
		ID: chatID, Object: "chat.completion.chunk", Created: created, Model: model,
		Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: openai.NewChatMessage("", ""), FinishReason: &finish}},
	}
	data, _ := json.Marshal(fin)
	fmt.Fprintf(w, "data: %s\n\n", data)

	if includeUsage {
		// Channels don't surface token counts; report zeros so the shape holds.
		usage := openai.ChatCompletionResponse{
			ID: chatID, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []openai.ChatCompletionChoice{}, Usage: &openai.UsageInfo{},
		}
		data, _ = json.Marshal(usage)
		fmt.Fprintf(w, "data: %s\n\n", data)
	}
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func v3ErrChunk(chatID string, created int64, model, msg string) string {
	chunk := openai.ChatCompletionResponse{
		ID: chatID, Object: "chat.completion.chunk", Created: created, Model: model,
		Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: openai.NewChatMessage("assistant", "⚠️ "+msg)}},
	}
	data, _ := json.Marshal(chunk)
	return string(data)
}
