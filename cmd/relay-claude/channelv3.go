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
	BridgeDir        string        // dir containing bridge.ts + node_modules (deployed)
	IdleTTL          time.Duration // kill an interactive claude session after this idle
	MaxSessions      int           // cap on concurrent interactive claude processes
	ReplyTotal       time.Duration // overall wait for a turn's reply (idle-based, after first output)
	ColdStartTimeout time.Duration // cold-start watchdog: a turn must produce its FIRST output within this, else relaunch+retry
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
	sid      string
	cmd      *exec.Cmd
	ptmx     *os.File
	mcpFile  string // per-session --mcp-config path (cleaned up on stop)
	dead     atomic.Bool
	lastUsed atomic.Int64
	inTurn   atomic.Bool
	// turnSlot 是本会话 turn 的独占令牌（容量 1，launch 时放入一枚）。V2 用
	// turnMu 串行化同 session 请求，V3 此前没有等价物——并发的第二个请求会把
	// 消息 enqueue 给还在忙的 claude，等不到任何带自己 req_id 的事件，5 分钟后
	// 被冷启动 watchdog 误判为冻死，反手 SIGKILL 掉正在服务上一轮的进程。必须
	// 持有令牌才能 enqueue/等回复；用 channel 而非 mutex 是为了等待时能同时
	// select ctx.Done / deadCh / 心跳。
	turnSlot  chan struct{}
	reaped    atomic.Bool // cmd.Wait() 已返回：进程已收割、PID 可能复用，禁止再裸 kill 组
	deadCh    chan struct{}
	ready     chan struct{} // closed once the channel is registered + event loop is live
	readyOnce sync.Once
	ptyLog    *os.File // optional raw PTY capture for debugging

	bound string // spawn-bound model (for change warnings)

	// Transcript-based usage metering (see v3usage.go). These are only touched
	// from the turn-owning handler goroutine (V3 runs at most one turn per
	// session — inTurn semantics), but a small mutex guards them anyway so a
	// future concurrent reader can't corrupt the offset/dedup state.
	usageMu        sync.Mutex
	claudeSID      string              // claude-side session UUID (--session-id) == transcript basename
	transcriptPath string              // lazily resolved via glob by harvestV3Usage
	usageOffset    int64               // next transcript read position
	prevReqIDs     map[string]struct{} // requestIds already counted (cross-window dedup)
	usageWarned    bool                // "transcript not found" logged once per session
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
	// launching single-flights acquire() per sid（与 V2 chanManager.spawning 同构，
	// A8）：第一个请求登记 channel 并 launch，同 sid 并发请求等它完成后重查，
	// 而不是各自再起一个交互式 claude——败者会被覆盖成永久泄漏的重量级进程，
	// 且两个 bridge 会同时抢同一个 inbox 的消息。
	launching map[string]chan struct{}
}

func newV3Manager(cfg v3Config) *v3Manager {
	if cfg.IdleTTL <= 0 {
		cfg.IdleTTL = 20 * time.Minute
	}
	if cfg.MaxSessions <= 0 {
		cfg.MaxSessions = 30
	}
	if cfg.ReplyTotal <= 0 {
		// IDLE timeout (reset on every claude event), staggered well ABOVE
		// wuji_tools' agent_timeout (600s, when it switches to background +
		// push-when-done). If the relay closed the SSE at the same ~600s, it
		// would poison wuji's background path with a timeout error and drop the
		// genuine slow answer. 20min leaves that path ~10min of headroom to
		// deliver a single silent >10min step.
		cfg.ReplyTotal = 20 * time.Minute
	}
	if cfg.ColdStartTimeout <= 0 {
		// Cold-start watchdog: a fresh interactive claude that produces NO output
		// at all within this window is presumed frozen on a startup step (e.g.
		// stuck "Checking for updates") — kill + relaunch + retry once. Override
		// with V3_COLD_START_TIMEOUT (e.g. "10s" to exercise the retry path).
		cfg.ColdStartTimeout = 5 * time.Minute
		if v := os.Getenv("V3_COLD_START_TIMEOUT"); v != "" {
			if d, err := time.ParseDuration(v); err == nil && d > 0 {
				cfg.ColdStartTimeout = d
			}
		}
	}
	return &v3Manager{
		cfg:       cfg,
		sessions:  make(map[string]*v3Session),
		inbox:     make(map[string]chan v3Msg),
		waiters:   make(map[string]*v3Waiter),
		launching: make(map[string]chan struct{}),
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
	// 只查不建：inbox 与 session 同生共死（acquire 促升时创建、teardown 时删除）。
	// 之前这里会为已死/未知 sid 重建条目——被拆掉会话的 bridge 在退出前的最后一次
	// poll 就能让 map 无界增长，且残留条目会把过期消息重放进同 sid 的下一个会话。
	// 404 让孤儿 bridge 走它的 1s 退避重试，直到随 claude 进程组一起被收掉。
	m.mu.Lock()
	q := m.inbox[sid]
	m.mu.Unlock()
	if q == nil {
		http.Error(w, "unknown session", http.StatusNotFound)
		return
	}
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

// enqueue hands one inbound turn to the session's bridge. 裸 send 会在 inbox
// 满（bridge 挂死不 poll）时把 handler goroutine 永远吊住——turn 串行化下正常
// 永远不会满（同时最多一条在途），所以走到超时/无 inbox 本身就说明会话已坏，
// 交给调用方按 coldHang 处理。
func (m *v3Manager) enqueue(ctx context.Context, sid string, msg v3Msg) error {
	m.mu.Lock()
	q := m.inbox[sid]
	m.mu.Unlock()
	if q == nil {
		return fmt.Errorf("no inbox for session %s (torn down?)", sid)
	}
	select {
	case q <- msg:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(30 * time.Second):
		return fmt.Errorf("inbox for session %s stuck full (bridge not polling?)", sid)
	}
}

// killSession kills the interactive claude (+ its bun bridge) for sid and removes
// it from the manager (sessions + inbox). Used to tear down a one-off ephemeral
// session at turn end so it doesn't linger as a heavy unreusable process.
func (m *v3Manager) killSession(sid string) {
	m.mu.Lock()
	s := m.sessions[sid]
	delete(m.sessions, sid)
	delete(m.inbox, sid)
	m.mu.Unlock()
	if s != nil {
		m.flushV3Orphan(s, "kill-session") // 断连后烧掉的尾巴 token 最后一次入账机会
		s.kill()
	}
}

// drainInbox empties any undelivered inbound turns for a session — used before a
// cold-start retry (so the relaunched bridge doesn't inject a stale duplicate)
// and on client disconnect (turn 串行化下队列里只可能是断连请求自己那条还没被
// bridge 取走的消息，留着会污染同 session 的下一轮)。只查不建。
func (m *v3Manager) drainInbox(sid string) {
	m.mu.Lock()
	q := m.inbox[sid]
	m.mu.Unlock()
	if q == nil {
		return
	}
	for {
		select {
		case <-q:
		default:
			return
		}
	}
}

// v3FindTranscript globs for the interactive claude's transcript. The cwd
// encoding of the project dir is deliberately NOT reimplemented — the session
// UUID is globally unique, so ~/.claude/projects/*/<uuid>.jsonl matches at most
// one file.
func v3FindTranscript(claudeSID string) string {
	home, err := os.UserHomeDir()
	if err != nil || claudeSID == "" {
		return ""
	}
	matches, _ := filepath.Glob(filepath.Join(home, ".claude", "projects", "*", claudeSID+".jsonl"))
	if len(matches) == 0 {
		return ""
	}
	return matches[0]
}

// harvestV3Usage harvests this turn's token usage at a turn-end point: it
// lazily locates the transcript (one glob, then cached), reads the incremental
// window since the previous harvest, records the turn into /v1/stats, and
// returns the aggregate UsageInfo for the SSE usage chunk.
//
// Failure semantics: any failure (transcript not found, read/parse error) →
// returns nil so the caller emits NO usage chunk and downstream keeps the
// NULL = "not metered" meaning — a 0 would read as "free request". The request
// itself is still counted via stats.RecordTurn(model, nil) so /v1/stats request
// totals stop being blind to V3.
//
// graceRetry (used on the normal final-reply path only): if the transcript is
// missing or the window has no usage yet, wait 1.5s and retry once — the
// terminal reply can race claude's transcript flush by a moment.
func (m *v3Manager) harvestV3Usage(s *v3Session, model string, graceRetry bool) *openai.UsageInfo {
	s.usageMu.Lock()
	defer s.usageMu.Unlock()

	if s.transcriptPath == "" {
		p := v3FindTranscript(s.claudeSID)
		if p == "" && graceRetry {
			time.Sleep(1500 * time.Millisecond)
			p = v3FindTranscript(s.claudeSID)
		}
		if p == "" {
			if !s.usageWarned {
				s.usageWarned = true
				log.Printf("[v3] usage: transcript not found for claude_sid=%s session=%s — turn not metered", s.claudeSID, s.sid)
			}
			stats.RecordTurn(model, nil)
			return nil
		}
		s.transcriptPath = p
	}

	read := func() (map[string]openai.TokenCounts, error) {
		perModel, newOff, ids, err := readV3UsageWindow(s.transcriptPath, s.usageOffset, s.prevReqIDs)
		if err != nil {
			return nil, err
		}
		s.usageOffset = newOff
		if s.prevReqIDs == nil {
			s.prevReqIDs = make(map[string]struct{})
		}
		for id := range ids {
			s.prevReqIDs[id] = struct{}{}
		}
		return perModel, nil
	}

	perModel, err := read()
	if err != nil {
		log.Printf("[v3] usage: read transcript failed session=%s path=%s: %v — turn not metered", s.sid, s.transcriptPath, err)
		if os.IsNotExist(err) {
			s.transcriptPath = "" // vanished (cleanup?) → re-glob next turn
		}
		stats.RecordTurn(model, nil)
		return nil
	}
	if len(perModel) == 0 && graceRetry {
		// Window empty right at final-reply time: give the transcript writer a
		// moment (write ordering between the reply tool call and the jsonl flush
		// is not guaranteed), then read the follow-up increment.
		time.Sleep(1500 * time.Millisecond)
		if pm2, err2 := read(); err2 == nil {
			perModel = pm2
		}
	}

	if len(perModel) == 0 {
		// Still count the request so /v1/stats request totals include V3 turns.
		stats.RecordTurn(model, nil)
		return nil
	}

	stats.RecordTurn(model, perModel)
	var input, output, cacheRead, cacheCreation int
	for _, c := range perModel {
		input += c.Input
		output += c.Output
		cacheRead += c.CacheRead
		cacheCreation += c.CacheCreation
	}
	log.Printf("[v3] usage session=%s claude_sid=%s models=%d input=%d output=%d cache_read=%d cache_creation=%d",
		s.sid, s.claudeSID, len(perModel), input, output, cacheRead, cacheCreation)
	return openai.BuildUsageInfo(input, output, cacheRead, cacheCreation)
}

// flushV3Orphan sweeps any transcript increment that accrued OUTSIDE a metered
// turn window — tokens claude kept burning after a stop/idle-timeout, or the
// tail of a torn-down session. They belong to an already-counted turn, so they
// go into /v1/stats as orphan tokens (no request increment) and are NEVER
// attached to a later turn's usage chunk (that would inflate the next
// message's robot_chat_logs row). Quiet no-op when there is nothing to sweep.
func (m *v3Manager) flushV3Orphan(s *v3Session, where string) {
	if s == nil {
		return
	}
	s.usageMu.Lock()
	defer s.usageMu.Unlock()
	if s.transcriptPath == "" {
		if s.transcriptPath = v3FindTranscript(s.claudeSID); s.transcriptPath == "" {
			return
		}
	}
	perModel, newOff, ids, err := readV3UsageWindow(s.transcriptPath, s.usageOffset, s.prevReqIDs)
	if err != nil {
		return
	}
	s.usageOffset = newOff
	if s.prevReqIDs == nil {
		s.prevReqIDs = make(map[string]struct{})
	}
	for id := range ids {
		s.prevReqIDs[id] = struct{}{}
	}
	if len(perModel) == 0 {
		return
	}
	stats.RecordOrphanTokens(perModel)
	var input, output int
	for _, c := range perModel {
		input += c.Input
		output += c.Output
	}
	log.Printf("[v3] orphan usage swept (%s) session=%s claude_sid=%s input=%d output=%d — attributed to stats only, not to any turn",
		where, s.sid, s.claudeSID, input, output)
}

// v3TurnOutcome is what one runAttempt() of handleChannelV3Response resolves to.
type v3TurnOutcome int

const (
	v3OutcomeDone       v3TurnOutcome = iota // reply delivered / terminal handled — caller returns
	v3OutcomeColdHang                        // no output within ColdStartTimeout — caller may relaunch+retry
	v3OutcomeClientGone                      // client disconnected — caller returns (session kept alive)
)

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
// Launching is single-flighted per sid: concurrent requests that miss the
// registry wait for the in-progress launch and re-check instead of racing a
// second interactive claude onto the same sid（败者会被 map 覆盖成不可达的
// 永久泄漏进程，且两个 bridge 同时 poll 同一 inbox，消息被随机抢走）。ctx 让
// 已断连的客户端不再排队接盘下一次 launch。
func (m *v3Manager) acquire(ctx context.Context, sid string, p v3SpawnParams) (*v3Session, error) {
	for {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("acquire session %s: %w", sid, err)
		}
		m.mu.Lock()
		if s := m.sessions[sid]; s != nil && !s.Dead() {
			s.markUsed()
			m.mu.Unlock()
			return s, nil
		}
		// 死会话(或从没存在)的 inbox 一并清掉：否则 relaunch 失败时 acquire 从
		// map 摘掉死会话后直接返回错误，而 onSessionDie 又因 map 里已无此会话而
		// 不兜底删 inbox → 孤儿 inbox 泄漏。促升成功时下方会重建。
		delete(m.sessions, sid)
		delete(m.inbox, sid)
		if ch, ok := m.launching[sid]; ok {
			// Someone else is launching this sid: wait for it and re-check（它
			// 可能促升了活会话，也可能失败——那时本请求成为下一个 launcher）。
			m.mu.Unlock()
			select {
			case <-ch:
			case <-ctx.Done():
				return nil, fmt.Errorf("acquire session %s: %w", sid, ctx.Err())
			}
			continue
		}
		launchCh := make(chan struct{})
		m.launching[sid] = launchCh
		if m.cfg.MaxSessions > 0 {
			alive := 0
			for _, s := range m.sessions {
				if !s.Dead() {
					alive++ // count ALL live sessions (incl. inTurn) toward the cap
				}
			}
			if alive >= m.cfg.MaxSessions {
				m.evictOldestLocked() // frees an idle session if any; logs if all busy
			}
		}
		m.mu.Unlock()

		s, err := m.launch(sid, p)

		// Release the single-flight and promote in the SAME critical section,
		// so a waiter's re-check either sees the live session or finds
		// launching empty and takes over.
		m.mu.Lock()
		delete(m.launching, sid)
		close(launchCh)
		if err != nil {
			m.mu.Unlock()
			return nil, err
		}
		if existing := m.sessions[sid]; existing != nil && !existing.Dead() && existing != s {
			// 理论上单飞后到不了这里；防御：已有活会话就杀掉新 launch 的，
			// 保证同一 sid 永远只有一个交互式 claude + bridge。
			m.mu.Unlock()
			log.Printf("[v3] WARN session=%s: live session appeared during launch; killing the duplicate", sid)
			s.kill()
			return existing, nil
		}
		m.sessions[sid] = s
		// inbox 与 session 同生共死：促升时创建（bridge 起来后第一次 poll 就有
		// 东西可等），teardown（killSession/reap/evict/onSessionDie/Stop）时删除。
		if m.inbox[sid] == nil {
			m.inbox[sid] = make(chan v3Msg, 8)
		}
		m.mu.Unlock()
		return s, nil
	}
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

	// A fresh claude-side session UUID per launch (interactive claude supports
	// --session-id as a general flag): the transcript lands at
	// ~/.claude/projects/<encoded-cwd>/<uuid>.jsonl, which harvestV3Usage tails
	// for token metering. EVERY launch mints a NEW uuid — including a cold-start
	// relaunch of the same relay sid — because reusing a uuid whose transcript
	// file already exists would make claude refuse to start or unexpectedly
	// resume the old conversation. (Relaunch goes acquire→launch→new v3Session,
	// so the fresh uuid + zeroed usage offset come for free.)
	claudeSID := newUUID()
	args := []string{
		"--dangerously-load-development-channels", "server:relaybridge",
		"--dangerously-skip-permissions",
		"--session-id", claudeSID,
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
		// .mcp.json 是 bot working_dir 里的共享文件（可能含用户自己的 MCP
		// server），启动失败不能顺手删掉它——留着无害，下次 launch 还会复用。
		return nil, fmt.Errorf("pty start claude: %w", err)
	}

	s := &v3Session{
		sid:        sid,
		cmd:        cmd,
		ptmx:       ptmx,
		mcpFile:    mcpFile,
		deadCh:     make(chan struct{}),
		ready:      make(chan struct{}),
		turnSlot:   make(chan struct{}, 1),
		bound:      p.model,
		claudeSID:  claudeSID,
		prevReqIDs: make(map[string]struct{}),
	}
	s.turnSlot <- struct{}{} // 一枚令牌 = 同时最多一个 turn
	if v3DebugPTY {
		if f, err := os.Create(filepath.Join(os.TempDir(), "v3pty-"+sid+".log")); err == nil {
			s.ptyLog = f
		}
	}
	s.markUsed()

	go s.drivePTY()
	go func() {
		_ = cmd.Wait()
		s.reaped.Store(true) // 进程已收割：此后 PID 可能被复用，kill() 不得再裸 kill 组
		s.dead.Store(true)
		close(s.deadCh)
		ptmx.Close()
		// .mcp.json lives in the bot working dir and is reused across this bot's
		// sessions — do not delete it.
		m.onSessionDie(sid)
	}()

	log.Printf("[v3] launched interactive claude session=%s claude_sid=%s model=%s pid=%d workdir=%s",
		sid, claudeSID, p.model, cmd.Process.Pid, p.workingDir)
	return s, nil
}

// mcpJSONMu serializes read-modify-write of .mcp.json files: 同一 bot
// working_dir 的多个会话并发 launch 时，无锁的读改写会互相丢更新。写入频率
// 极低（每次 launch 一次），全局一把锁足够。
var mcpJSONMu sync.Mutex

// ensureBridgeInMCPJSON merges the relaybridge channel server into the cwd's
// .mcp.json (creating it if absent, preserving any existing servers). The env
// only overrides the proxy: the bridge inherits RELAY_CTRL/RELAY_SESSION from
// claude's per-process env, but must NOT use the chroot's HTTP_PROXY for its
// loopback calls to the relay.
func ensureBridgeInMCPJSON(path, bridgeTS string) error {
	mcpJSONMu.Lock()
	defer mcpJSONMu.Unlock()
	cfg := map[string]any{}
	if data, err := os.ReadFile(path); err == nil && strings.TrimSpace(string(data)) != "" {
		// 解析失败必须拒绝而不是静默清空：文件里可能是用户手写的 MCP 配置，
		// 一个多余的逗号不该换来整份配置被覆盖。launch 失败的报错会带上原因。
		if uerr := json.Unmarshal(data, &cfg); uerr != nil {
			return fmt.Errorf("existing %s is not valid JSON (%v) — fix or remove it before v3 can launch here", path, uerr)
		}
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
	// 原子替换：claude 启动时随时可能读这份文件，写一半的 JSON 会让它启动失败。
	tmp := path + ".relaytmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// startup prompts that interactive claude shows; we answer each once by sending
// Enter (the safe/default option) into the PTY.
var v3Prompts = []struct {
	key string
	re  *regexp.Regexp
}{
	{"devchannels", regexp.MustCompile(`(?i)using this for local development`)},
	{"trust", regexp.MustCompile(`(?i)do you trust|trust the files|trust this folder|trust this project|created or one you trust`)},
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
	// group kill reaps the claude tree and its spawned bun bridge. reaped 后
	// 禁止再发：进程已被 cmd.Wait() 收割，PID 可能已复用，kill(-pid) 会误杀
	// 无关进程组（pkg/proc WatchDisconnect 注释警告过的同一坑）。
	if !s.reaped.Load() {
		proc.KillGroup(s.cmd)
	}
	if s.ptmx != nil {
		s.ptmx.Close()
	}
}

func (m *v3Manager) onSessionDie(sid string) {
	m.mu.Lock()
	// 仅当 map 里就是这个死会话时才清理：冷启动重试会 kill 旧会话再促升新会话，
	// 旧会话的 onDie 可能晚到——那时 map 里已是活的新会话（连同新 inbox），不能误删。
	if s := m.sessions[sid]; s != nil && s.Dead() {
		delete(m.sessions, sid)
		delete(m.inbox, sid)
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
		// At capacity but every session is mid-turn — nothing safe to evict.
		// Launch anyway (brief over-cap); turn-end / reaper reclaim slots soon.
		log.Printf("[v3] at capacity (%d) but all sessions busy; launching over-cap", m.cfg.MaxSessions)
		return
	}
	s := m.sessions[oldest]
	delete(m.sessions, oldest)
	delete(m.inbox, oldest)
	log.Printf("[v3] capacity evict session=%s", oldest)
	go func() {
		m.flushV3Orphan(s, "evict")
		s.kill()
	}()
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
			delete(m.inbox, k)
			continue
		}
		if !s.inTurn.Load() && s.LastUsed().Before(cutoff) {
			toKill = append(toKill, s)
			delete(m.sessions, k)
			delete(m.inbox, k)
		}
	}
	m.mu.Unlock()
	for _, s := range toKill {
		log.Printf("[v3] reaping idle session=%s", s.sid)
		m.flushV3Orphan(s, "reap")
		s.kill()
	}
}

// Stop kills every interactive session (shutdown). 先摘表再收尸：flushV3Orphan
// 要做文件 IO，不能压着 m.mu 干；SIGTERM 后紧接 os.Exit，这里是重启前未收割
// 尾巴 token 的最后入账机会。
func (m *v3Manager) Stop() {
	m.mu.Lock()
	sessions := make([]*v3Session, 0, len(m.sessions))
	for _, s := range m.sessions {
		sessions = append(sessions, s)
	}
	m.sessions = make(map[string]*v3Session)
	m.inbox = make(map[string]chan v3Msg)
	m.mu.Unlock()
	for _, s := range sessions {
		m.flushV3Orphan(s, "stop")
		s.kill()
	}
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
	ephemeral := sid == "" // no session_id (e.g. cron) → one-off, never reusable
	if ephemeral {
		sid = newUUID()
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
	reqID := v3ReqID()
	waiter := v3Mgr.registerWaiter(reqID)
	defer v3Mgr.unregisterWaiter(reqID)

	// One-off (no-session, e.g. cron) sessions are never reusable — tear the
	// interactive claude + bun bridge down at turn end instead of leaking a heavy
	// process until the idle reaper fires. 必须在 acquireTurn 之前注册：launch
	// 成功后、拿到 turn 令牌前客户端就断开的话，这个 defer 是唯一的回收路径。
	// (sid is captured; a cold-start retry reassigns sess but keeps sid, so this
	// kills whatever is mapped to sid. 对未促升的 sid 是 no-op。)
	if ephemeral {
		defer v3Mgr.killSession(sid)
	}

	// SSE headers + initial ping BEFORE acquiring anything: 排队等同 session 上
	// 一轮、冷启动、launch 全都可能耗时，期间必须持续有字节可发（wuji adapter
	// sock_read=120s），头一旦发出，后续错误也只能走 SSE 错误块而非 JSON 错误。
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

	sseFail := func(msg string) {
		fmt.Fprintf(w, "data: %s\n\n", v3ErrChunk(chatID, created, model, msg))
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}

	// acquireTurn 拿到 sid 的活会话并独占其 turn 令牌。等待期间发可见的排队心跳；
	// 会话换代（冷启动重试杀旧拉新）时通过 deadCh/死令牌回到 re-acquire。返回
	// false 表示客户端已断开或 launch 失败（后者已发 SSE 错误块）。
	acquireTurn := func() (*v3Session, bool) {
		for {
			s2, err := v3Mgr.acquire(r.Context(), sid, p)
			if err != nil {
				if r.Context().Err() != nil {
					log.Printf("[v3] client gone while acquiring session=%s req=%s", sid, reqID)
					return nil, false
				}
				log.Printf("[v3] acquire failed session=%s req=%s: %v", sid, reqID, err)
				sseFail("session launch failed: " + err.Error())
				return nil, false
			}
		waitSlot:
			for {
				select {
				case <-s2.turnSlot:
					if s2.Dead() {
						s2.turnSlot <- struct{}{} // 别的等待者也要经此发现换代
						break waitSlot
					}
					return s2, true
				case <-s2.deadCh:
					break waitSlot // 会话换代，重新 acquire
				case <-r.Context().Done():
					return nil, false
				case <-heartbeat.C:
					v3EmitThinkingDelta(w, flusher, chatID, created, model,
						fmt.Sprintf("\n⏳ 同会话上一条消息仍在处理，排队中…（已 %ds）", int(time.Since(reqStart).Seconds())))
				}
			}
		}
	}

	sess, ok2 := acquireTurn()
	if !ok2 {
		return
	}
	// turnOwned/releaseOwned 管理令牌归还：正常返回、冷启动换代、give-up 拆会话
	// 都必须恰好归还一次（归还即放行队列里的下一条同 session 请求）。
	turnOwned := sess
	releaseOwned := func() {
		if turnOwned == nil {
			return
		}
		turnOwned.inTurn.Store(false)
		turnOwned.markUsed()
		turnOwned.turnSlot <- struct{}{}
		turnOwned = nil
	}
	defer releaseOwned()
	sess.inTurn.Store(true)

	// runAttempt drives ONE attempt at this turn on the given (already acquired)
	// session: Phase 1 waits for the channel to go live, Phase 2 sends the turn
	// and streams the reply. The cold-start watchdog requires the FIRST output to
	// arrive within ColdStartTimeout; once any output arrives the turn is
	// "committed" and the idle timeout (ReplyTotal) governs the rest (so genuinely
	// long work is never cut). If no output arrives in time (a frozen cold start),
	// it returns v3OutcomeColdHang so the caller can kill + relaunch + retry.
	runAttempt := func(sess *v3Session) v3TurnOutcome {
		// Single per-attempt deadline for the FIRST output (covers a never-ready
		// channel AND a ready-but-frozen claude).
		coldDeadline := time.After(v3Mgr.cfg.ColdStartTimeout)

		// Phase 1: wait until the session's channel is live.
	waitReady:
		for {
			select {
			case <-sess.ready:
				break waitReady
			case <-heartbeat.C:
				v3EmitThinkingDelta(w, flusher, chatID, created, model,
					fmt.Sprintf("\n⏳ 正在准备会话…（已 %ds）", int(time.Since(reqStart).Seconds())))
			case <-sess.deadCh:
				log.Printf("[v3] session died during startup session=%s req=%s", sid, reqID)
				return v3OutcomeColdHang
			case <-r.Context().Done():
				return v3OutcomeClientGone
			case <-coldDeadline:
				log.Printf("[v3] cold-start watchdog: channel not ready within %s session=%s req=%s", v3Mgr.cfg.ColdStartTimeout, sid, reqID)
				return v3OutcomeColdHang
			}
		}

		// Phase 2: send the turn into the live session and stream the reply.
		// 开窗前先把上一轮 stop/超时后 claude 继续写入的遗留增量单独入账（orphan，
		// 只进 /v1/stats 总账）——否则这些 token 会整段算进本轮的 usage chunk，
		// 污染下游 robot_chat_logs 的单条消息用量（B-3）。
		v3Mgr.flushV3Orphan(sess, "turn-start")
		log.Printf("[v3] turn start session=%s req=%s model=%s chat=%s", sid, reqID, model, chatID)
		if err := v3Mgr.enqueue(r.Context(), sid, v3Msg{ReqID: reqID, Content: content}); err != nil {
			if r.Context().Err() != nil {
				return v3OutcomeClientGone
			}
			// inbox 没了（会话正被拆）或卡满（bridge 不 poll）都说明会话已坏，
			// 按冷启动挂死处理：杀掉重拉。
			log.Printf("[v3] enqueue failed session=%s req=%s: %v", sid, reqID, err)
			return v3OutcomeColdHang
		}

		// streamed: flushed ≥1 content chunk (drives the clean-finish-vs-error
		// choice on abnormal end). committed: got ANY output, so the cold-start
		// watchdog is satisfied and the idle timeout takes over. lastActivity is
		// measured from the instant note so the heartbeat shows movement promptly.
		streamed := false
		committed := false
		lastActivity := reqStart
		idle := time.NewTimer(v3Mgr.cfg.ReplyTotal)
		defer idle.Stop()
		for {
			select {
			case ev := <-waiter.ch:
				committed = true
				lastActivity = time.Now()
				if !idle.Stop() {
					select {
					case <-idle.C:
					default:
					}
				}
				idle.Reset(v3Mgr.cfg.ReplyTotal)
				if ev.thinking {
					v3EmitThinkingDelta(w, flusher, chatID, created, model, ev.text)
					continue
				}
				if !ev.final {
					v3EmitContentDelta(w, flusher, chatID, created, model, ev.text)
					if ev.text != "" {
						streamed = true
						sessionStore.LogDelta(req.SessionID, ev.text)
					}
					continue
				}
				// Terminal reply → harvest this turn's transcript usage (with the
				// grace retry: the reply tool call can race the jsonl flush).
				usage := v3Mgr.harvestV3Usage(sess, model, true)
				v3EmitClose(w, flusher, chatID, created, model, ev.text, includeUsage, usage)
				if ev.text != "" {
					sessionStore.LogDelta(req.SessionID, ev.text)
				}
				sessionStore.LogDone(req.SessionID, usage)
				log.Printf("[v3] turn end session=%s req=%s streamed=%v finalLen=%d", sid, reqID, streamed, len(ev.text))
				return v3OutcomeDone
			case <-heartbeat.C:
				if time.Since(lastActivity) >= 12*time.Second {
					v3EmitThinkingDelta(w, flusher, chatID, created, model,
						fmt.Sprintf("\n⏳ 处理中…（已 %ds）", int(time.Since(reqStart).Seconds())))
				} else {
					fmt.Fprintf(w, ": keepalive\n\n")
					flusher.Flush()
				}
			case <-coldDeadline:
				if !committed {
					log.Printf("[v3] cold-start watchdog: no output within %s session=%s req=%s", v3Mgr.cfg.ColdStartTimeout, sid, reqID)
					return v3OutcomeColdHang
				}
				// committed → the turn is alive; the idle timer governs from here.
			case <-sess.deadCh:
				if !committed {
					log.Printf("[v3] session died before any output session=%s req=%s", sid, reqID)
					return v3OutcomeColdHang
				}
				// Meter what the dead claude wrote to its transcript regardless of
				// whether content streamed（B-1：只发过 progress 的长工具轮同样烧了
				// 真金白银的 token，进程死亡不能让它从 /v1/stats 消失）。No grace
				// retry: the writer is gone, nothing more will appear.
				usage := v3Mgr.harvestV3Usage(sess, model, false)
				if streamed {
					v3EmitClose(w, flusher, chatID, created, model, "", includeUsage, usage)
				} else {
					fmt.Fprintf(w, "data: %s\n\n", v3ErrChunk(chatID, created, model, "claude session ended"))
					fmt.Fprintf(w, "data: [DONE]\n\n")
					flusher.Flush()
				}
				sessionStore.LogDone(req.SessionID, usage)
				log.Printf("[v3] session=%s died mid-turn req=%s streamed=%v", sid, reqID, streamed)
				return v3OutcomeDone
			case <-r.Context().Done():
				// 断连（=stop）也要入账（B-2）：收割到目前为止的增量并计请求数；
				// 之后 claude 继续烧的部分由 turn-start/teardown 的 orphan sweep
				// 兜底。无 SSE 可发（客户端已走），usage 只进 /v1/stats。
				_ = v3Mgr.harvestV3Usage(sess, model, false)
				log.Printf("[v3] client disconnected session=%s req=%s (session kept alive)", sid, reqID)
				return v3OutcomeClientGone
			case <-idle.C:
				// No claude activity for ReplyTotal AFTER first output → stuck.
				// ALWAYS surface a visible marker (even mid-stream): never report a
				// stalled turn as a clean success.
				v3EmitContentDelta(w, flusher, chatID, created, model, "\n\n⚠️ 任务长时间无新进展，已停止等待。")
				// Meter whatever the stuck turn actually burned (no grace retry:
				// nothing has moved for ReplyTotal already).
				usage := v3Mgr.harvestV3Usage(sess, model, false)
				v3EmitClose(w, flusher, chatID, created, model, "", includeUsage, usage)
				log.Printf("[v3] idle timeout (no activity %s) session=%s req=%s streamed=%v", v3Mgr.cfg.ReplyTotal, sid, reqID, streamed)
				return v3OutcomeDone
			}
		}
	}

	// Cold-start watchdog + retry: if an attempt produces no output within
	// ColdStartTimeout, the cold-started claude is presumed frozen — kill it,
	// relaunch a fresh session, and retry the same turn (same reqID/waiter, same
	// sid → the new bridge re-injects to the same waiter). Give up after one retry.
	const maxColdRetries = 1
	for attempt := 0; ; attempt++ {
		switch runAttempt(sess) {
		case v3OutcomeDone:
			return
		case v3OutcomeClientGone:
			// turn 串行化下 inbox 里只可能是本请求还没被 bridge 取走的那条消息。
			// 清掉，否则同 session 的下一轮会先收到这条旧消息——claude 回答一个
			// 没人等的问题，新问题被压后。
			v3Mgr.drainInbox(sid)
			return
		case v3OutcomeColdHang:
			if attempt >= maxColdRetries {
				v3EmitContentDelta(w, flusher, chatID, created, model, "\n\n⚠️ 会话启动多次无响应，请稍后重发。")
				// sess here is the CURRENT (post-retry) session object; a frozen
				// cold start usually has an empty transcript, so this mostly just
				// records the request into /v1/stats (usage stays nil → NULL).
				usage := v3Mgr.harvestV3Usage(sess, model, false)
				v3EmitClose(w, flusher, chatID, created, model, "", includeUsage, usage)
				log.Printf("[v3] cold-start give up after %d retries session=%s req=%s", attempt, sid, reqID)
				// 冻死的会话不能留在 manager 里：留着的话下一条请求会原样再白等
				// 一轮 5 分钟 watchdog。拆掉（连同 inbox 与残留消息），下次重拉。
				// 顺序与重试路径一致：先 kill（标记 dead）再还令牌，这样并发同 sid
				// 的等待者拿到令牌时会看到 Dead()、回去重 acquire，而不是短暂"赢得"
				// 一个马上被销毁的会话。
				if !ephemeral {
					v3Mgr.killSession(sid)
					releaseOwned()
				}
				return
			}
			log.Printf("[v3] cold-start hang → kill+relaunch+retry (attempt %d) session=%s req=%s", attempt+1, sid, reqID)
			// 先杀再还令牌：排队的等待者拿到令牌会看到 Dead，回到 re-acquire 等
			// 新会话，不会误占尸体。
			sess.kill()
			releaseOwned()
			v3Mgr.drainInbox(sid)
			var ok3 bool
			sess, ok3 = acquireTurn()
			if !ok3 {
				log.Printf("[v3] cold-start relaunch not acquired session=%s req=%s", sid, reqID)
				return // 客户端已断开，或 relaunch 失败（acquireTurn 已发 SSE 错误）
			}
			turnOwned = sess
			sess.inTurn.Store(true)
			v3EmitThinkingDelta(w, flusher, chatID, created, model, "\n⏳ 会话启动异常，正在重启会话重试…")
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
//
// usage is the transcript-harvested aggregate (harvestV3Usage). Non-nil (and
// includeUsage) → a usage chunk goes out between the finish chunk and [DONE];
// nil → NO usage chunk, so downstream stores NULL ("not metered"), never a
// fake 0 ("free request").
func v3EmitClose(w http.ResponseWriter, flusher http.Flusher, chatID string, created int64, model, finalText string, includeUsage bool, usage *openai.UsageInfo) {
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

	if includeUsage && usage != nil {
		uc := openai.ChatCompletionResponse{
			ID: chatID, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []openai.ChatCompletionChoice{},
			Usage:   usage,
		}
		data, _ := json.Marshal(uc)
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
