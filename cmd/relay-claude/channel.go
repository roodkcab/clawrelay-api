package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"clawrelay-api/pkg/proc"
)

// newUUID returns a random RFC-4122 v4 UUID string. claude's --session-id
// requires a valid UUID, so ephemeral (no-session) channel runs mint one here.
func newUUID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant 10
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// Channel mode keeps one persistent `claude --print --input-format stream-json`
// process per session_id. Each inbound /v1/chat/completions request feeds its
// (single) new user message into the existing process via stdin and reads that
// turn's stdout up to the terminal `result` event — eliminating the per-request
// spawn + session reload + MCP/skill re-initialization cost that the legacy
// (per-request `claude -p`) path pays every time.
//
// The OpenAI HTTP/SSE contract is unchanged: the same sseTranslator that drives
// the legacy path drives the channel path, so upstream wuji_tools sees an
// identical byte stream.

var errWorkerDead = errors.New("channel worker is dead")

// chanWorker wraps one persistent claude subprocess bound to a single
// session_id. A single goroutine (drainStdout) reads the process stdout and
// fans each turn's lines to the active request via curLines; turns are
// serialized by turnMu so two requests can never interleave on one process.
type chanWorker struct {
	key     string // == session_id
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	workdir string

	sessionID atomic.Value // string, captured from the init/system event
	dead      atomic.Bool
	lastUsed  atomic.Int64 // unix nanos
	spawnedAt time.Time
	usedFlag  string // "--session-id" or "--resume" actually used at spawn

	// Spawn-time bound config, kept for change detection + logging. The
	// persistent process only honors these at startup; later requests that
	// differ are logged and ignored (see §4.3 of the design).
	boundSystemPrompt string
	boundModel        string

	ready      chan struct{} // closed once session_id is first captured
	readyOnce  sync.Once
	deadCh     chan struct{} // closed when the process exits
	startupErr atomic.Value  // string: a recognized startup-failure marker

	writeMu sync.Mutex // serializes stdin writes (user messages, interrupts)

	turnMu sync.Mutex  // held by the handler for the whole duration of a turn
	inTurn atomic.Bool // true while a turn is actively streaming (reap/evict must skip it)

	curMu sync.Mutex
	cur   *activeTurn // current turn's channels; nil between turns

	// meter diffs this persistent process's cumulative result.usage into
	// per-turn deltas. Its lifetime == the process's, so a respawned worker
	// (new struct) starts from a zero baseline. See usage_meter.go.
	meter *cumulativeMeter
}

// activeTurn bundles one turn's stdout-line channel with an abandon signal.
//
// Invariant that prevents "send on closed channel" panics: `lines` is closed
// ONLY by the drainStdout goroutine (via closeLines). Every other goroutine
// that wants to end a turn early (the cmd.Wait waiter, kill, beginTurn's error
// path) closes `quit` instead (via abandon); that wakes drainStdout's send
// select, which then performs the single close of `lines` itself. Since the one
// goroutine that sends on `lines` is also the only one that closes it, a send
// can never race a close.
type activeTurn struct {
	lines     chan string   // stdout sink; closed only by drainStdout
	quit      chan struct{} // closed to abandon the turn (any goroutine)
	quitOnce  sync.Once
	linesOnce sync.Once
}

func (a *activeTurn) abandon()    { a.quitOnce.Do(func() { close(a.quit) }) }
func (a *activeTurn) closeLines() { a.linesOnce.Do(func() { close(a.lines) }) }

func (w *chanWorker) SessionID() string   { v, _ := w.sessionID.Load().(string); return v }
func (w *chanWorker) Dead() bool          { return w.dead.Load() }
func (w *chanWorker) LastUsed() time.Time { return time.Unix(0, w.lastUsed.Load()) }
func (w *chanWorker) markUsed()           { w.lastUsed.Store(time.Now().UnixNano()) }
func (w *chanWorker) startupErrMarker() string {
	s, _ := w.startupErr.Load().(string)
	return s
}

// spawnChanWorker starts a persistent claude process. The caller must then
// waitStartup() to learn whether the chosen --session-id/--resume flag was
// accepted. onDie fires once when the process exits (used by the manager to
// drop the worker from its registry).
func spawnChanWorker(key string, args []string, workdir string, envVars map[string]string, usedFlag string, onDie func()) (*chanWorker, error) {
	cmd := exec.Command("claude", args...)
	proc.SetNewProcessGroup(cmd)
	cmd.Env = cleanEnv(envVars)
	if workdir != "" {
		cmd.Dir = workdir
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("stderr pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start claude: %w", err)
	}

	w := &chanWorker{
		key:       key,
		cmd:       cmd,
		stdin:     stdin,
		workdir:   workdir,
		spawnedAt: time.Now(),
		usedFlag:  usedFlag,
		ready:     make(chan struct{}),
		deadCh:    make(chan struct{}),
		meter:     &cumulativeMeter{},
	}
	w.markUsed()

	go w.drainStdout(stdout)
	go w.drainStderr(stderr)
	go func() {
		_ = cmd.Wait()
		w.dead.Store(true)
		_ = stdin.Close()
		close(w.deadCh)
		// Unblock any handler still ranging over the current turn channel: wake
		// a possibly-parked drainStdout send so it closes `lines` itself.
		w.abandonActiveTurn(nil)
		if onDie != nil {
			onDie()
		}
	}()

	log.Printf("[channel] spawned worker session=%s flag=%s pid=%d args=%v", key, usedFlag, cmd.Process.Pid, args)
	return w, nil
}

// abandonActiveTurn signals the active turn to end early (called by the waiter,
// kill, or beginTurn's error path — any goroutine). It only closes `quit`;
// drainStdout reacts by closing `lines`. only!=nil restricts to that exact turn.
func (w *chanWorker) abandonActiveTurn(only *activeTurn) {
	w.curMu.Lock()
	a := w.cur
	w.curMu.Unlock()
	if a == nil || (only != nil && a != only) {
		return
	}
	a.abandon()
}

// completeActiveTurn closes the turn's `lines` (the SOLE close site, always on
// the drainStdout goroutine) and clears it if still current. Idempotent.
func (w *chanWorker) completeActiveTurn(a *activeTurn) {
	w.curMu.Lock()
	if w.cur == a {
		w.cur = nil
	}
	w.curMu.Unlock()
	a.closeLines()
}

func (w *chanWorker) drainStdout(r io.Reader) {
	// On EOF, close whatever turn is still active (single close site).
	defer func() {
		w.curMu.Lock()
		a := w.cur
		w.cur = nil
		w.curMu.Unlock()
		if a != nil {
			a.closeLines()
		}
	}()

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			continue
		}
		var probe struct {
			Type      string `json:"type"`
			SessionID string `json:"session_id"`
		}
		_ = json.Unmarshal([]byte(line), &probe)
		if probe.SessionID != "" && w.SessionID() == "" {
			w.sessionID.Store(probe.SessionID)
			w.readyOnce.Do(func() { close(w.ready) })
		}

		// Forward to the active turn, if any. Lines that arrive between turns
		// (e.g. the init/system event at spawn) have no sink and are dropped.
		w.curMu.Lock()
		a := w.cur
		w.curMu.Unlock()
		if a == nil {
			continue
		}
		// Send-select: a parked send is rescued by quit (abandon), so a close
		// can never race the send. drainStdout is the only closer of a.lines.
		select {
		case a.lines <- line:
			if probe.Type == "result" {
				// Turn boundary: this user message has been fully answered.
				w.completeActiveTurn(a)
			}
		case <-a.quit:
			w.completeActiveTurn(a)
		}
	}
}

func (w *chanWorker) drainStderr(r io.Reader) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 4*1024), 256*1024)
	for sc.Scan() {
		line := sc.Text()
		log.Printf("[channel:%s/stderr] %s", w.key, line)
		switch {
		case strings.Contains(line, "is already in use"):
			w.startupErr.Store("already_in_use")
		case strings.Contains(line, "No conversation found with session ID"):
			w.startupErr.Store("no_conversation")
		}
	}
}

// startupOutcome is what waitStartup learned about the spawn.
type startupOutcome int

const (
	startupReady       startupOutcome = iota // init seen or alive past the window
	startupRetryResume                       // "--session-id <id> already in use" → use --resume
	startupRetryNew                          // "No conversation found" → use --session-id
	startupFatal                             // process died at startup for another reason
)

// waitStartup blocks until the worker proves healthy, dies with a recognized
// session-flag error, or the window elapses (alive ⇒ treated as healthy; the
// first turn will absorb any remaining MCP/skill init latency).
func (w *chanWorker) waitStartup(window time.Duration) startupOutcome {
	select {
	case <-w.ready:
		return startupReady
	case <-w.deadCh:
		switch w.startupErrMarker() {
		case "already_in_use":
			return startupRetryResume
		case "no_conversation":
			return startupRetryNew
		default:
			return startupFatal
		}
	case <-time.After(window):
		return startupReady
	}
}

// beginTurn acquires the worker exclusively, writes the new user message, and
// returns the channel of stdout lines for this turn (closed after the turn's
// `result`, or when the process dies). The caller MUST call endTurn when done.
func (w *chanWorker) beginTurn(content string) (<-chan string, error) {
	w.turnMu.Lock()
	if w.dead.Load() {
		w.turnMu.Unlock()
		return nil, errWorkerDead
	}
	a := &activeTurn{lines: make(chan string, 256), quit: make(chan struct{})}
	w.curMu.Lock()
	w.cur = a
	w.curMu.Unlock()
	w.inTurn.Store(true)

	envelope := map[string]any{
		"type": "user",
		"message": map[string]any{
			"role":    "user",
			"content": content,
		},
	}
	raw, _ := json.Marshal(envelope)
	raw = append(raw, '\n')

	w.writeMu.Lock()
	_, err := w.stdin.Write(raw)
	w.writeMu.Unlock()
	if err != nil {
		w.dead.Store(true)
		w.inTurn.Store(false)
		w.abandonActiveTurn(a) // wake drainStdout to close lines
		w.turnMu.Unlock()
		return nil, fmt.Errorf("write user message: %w", err)
	}
	// The write can succeed on a child that exited a moment ago (the stdin pipe
	// buffer still accepts it). If the process is already dead, drainStdout may
	// have exited without ever seeing this turn, so nothing would close a.lines.
	// Fail fast and let the caller re-acquire rather than hand out a dead turn.
	select {
	case <-w.deadCh:
		w.inTurn.Store(false)
		w.abandonActiveTurn(a)
		w.turnMu.Unlock()
		return nil, errWorkerDead
	default:
	}
	w.markUsed()
	return a.lines, nil
}

// endTurn releases the worker for the next request. It must be called after the
// turn's line channel has closed (the handler always drains to completion).
func (w *chanWorker) endTurn() {
	w.inTurn.Store(false)
	w.markUsed()
	w.turnMu.Unlock()
}

// interrupt writes a stdin interrupt control message, stopping the current turn
// while keeping the process (and its session context) alive.
func (w *chanWorker) interrupt() error {
	if w.dead.Load() {
		return errWorkerDead
	}
	w.writeMu.Lock()
	defer w.writeMu.Unlock()
	_, err := w.stdin.Write([]byte(`{"type":"interrupt"}` + "\n"))
	if err != nil {
		w.dead.Store(true)
	}
	return err
}

// kill force-terminates the process group (used as the interrupt-timeout
// backstop and for ephemeral teardown). The session is persisted on disk, so a
// later request --resumes it. It abandons the active turn first so a parked
// drainStdout send unblocks immediately rather than waiting for the pipe EOF.
func (w *chanWorker) kill() {
	w.dead.Store(true)
	w.abandonActiveTurn(nil)
	proc.KillGroup(w.cmd)
}

// ---- manager ----

// chanManagerConfig governs reaping + capacity.
type chanManagerConfig struct {
	IdleTTL     time.Duration
	MaxChannels int
}

// chanManager owns all channel workers, keyed by session_id, plus an `inflight`
// set of workers that have been spawned but are not (yet) in `workers`: a
// persistent worker during its waitStartup window, and an ephemeral worker for
// the duration of its single run. Tracking them means SIGTERM Stop() reaps them
// and /channels can see them, so no claude child is orphaned on shutdown.
type chanManager struct {
	cfg chanManagerConfig

	mu       sync.Mutex
	workers  map[string]*chanWorker
	inflight map[*chanWorker]struct{}
	// spawning single-flights acquire() per session_id: the first request
	// registers a channel here and spawns; concurrent requests for the same
	// session wait on it and re-check instead of double-spawning (two persistent
	// processes writing the same session jsonl). Closed + removed when the
	// spawn attempt finishes, success or not.
	spawning map[string]chan struct{}
}

func newChanManager(cfg chanManagerConfig) *chanManager {
	return &chanManager{
		cfg:      cfg,
		workers:  make(map[string]*chanWorker),
		inflight: make(map[*chanWorker]struct{}),
		spawning: make(map[string]chan struct{}),
	}
}

// trackInflight / untrackInflight register a spawned-but-not-promoted worker so
// shutdown reaps it and snapshot() reports it.
func (m *chanManager) trackInflight(w *chanWorker) {
	m.mu.Lock()
	m.inflight[w] = struct{}{}
	m.mu.Unlock()
}

func (m *chanManager) untrackInflight(w *chanWorker) {
	m.mu.Lock()
	delete(m.inflight, w)
	m.mu.Unlock()
}

// drop kills and removes the persistent worker for sessionID, if any. Used
// before a same-session request that must fall through to the legacy
// `claude --resume` path, so the two never write the session jsonl at once.
func (m *chanManager) drop(sessionID string) {
	m.mu.Lock()
	w := m.workers[sessionID]
	delete(m.workers, sessionID)
	m.mu.Unlock()
	if w != nil {
		log.Printf("[channel] dropping worker session=%s (legacy fall-through for same session)", sessionID)
		w.kill()
	}
}

// StartReaper kills workers idle longer than IdleTTL until ctx is cancelled.
func (m *chanManager) StartReaper(ctx context.Context) {
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

func (m *chanManager) reapOnce() {
	cutoff := time.Now().Add(-m.cfg.IdleTTL)
	m.mu.Lock()
	var toKill []*chanWorker
	for k, w := range m.workers {
		if w.Dead() {
			delete(m.workers, k)
			continue
		}
		// Never reap a worker mid-turn: a turn that streams longer than IdleTTL
		// has a stale lastUsed, but killing it would drop the live response.
		if !w.inTurn.Load() && w.LastUsed().Before(cutoff) {
			toKill = append(toKill, w)
			delete(m.workers, k)
		}
	}
	// Drop any dead worker that lingered in inflight (e.g. died during startup).
	for w := range m.inflight {
		if w.Dead() {
			delete(m.inflight, w)
		}
	}
	m.mu.Unlock()
	for _, w := range toKill {
		log.Printf("[channel] reaping idle worker session=%s idle=%s", w.key, time.Since(w.LastUsed()).Truncate(time.Second))
		w.kill()
	}
}

// spawnParams carries the spawn-time configuration derived from the first
// request for a session.
type spawnParams struct {
	args         []string // claude flags WITHOUT the trailing session flag
	workdir      string
	envVars      map[string]string
	systemPrompt string
	model        string
}

// acquire returns a live worker for sessionID, spawning one if necessary. On a
// fresh spawn it picks --session-id vs --resume from on-disk session presence,
// and retries with the opposite flag if claude rejects the first choice.
// Spawning is single-flighted per session (A8): concurrent requests that miss
// the registry wait for the in-progress spawn and re-check, instead of racing
// a second process onto the same session.
func (m *chanManager) acquire(sessionID string, p spawnParams) (*chanWorker, error) {
	var spawnCh chan struct{}
	for {
		m.mu.Lock()
		if w, ok := m.workers[sessionID]; ok && !w.Dead() {
			w.markUsed()
			// Warn (but honor spawn-time binding) if the model/system changed.
			if p.model != "" && w.boundModel != "" && p.model != w.boundModel {
				log.Printf("[channel] WARN session=%s model changed %q->%q; persistent process keeps spawn-time model", sessionID, w.boundModel, p.model)
			}
			if p.systemPrompt != w.boundSystemPrompt {
				log.Printf("[channel] WARN session=%s system_prompt changed since spawn; ignored (bound at spawn)", sessionID)
			}
			m.mu.Unlock()
			return w, nil
		}
		delete(m.workers, sessionID)
		if ch, ok := m.spawning[sessionID]; ok {
			// Another request is already spawning this session's worker: wait
			// for it and re-check (it may have promoted a live worker, or
			// failed — in which case this request becomes the next spawner).
			m.mu.Unlock()
			<-ch
			continue
		}
		spawnCh = make(chan struct{})
		m.spawning[sessionID] = spawnCh
		if m.cfg.MaxChannels > 0 {
			alive := 0
			for _, w := range m.workers {
				if !w.Dead() {
					alive++
				}
			}
			if alive >= m.cfg.MaxChannels {
				m.evictOldestLocked()
			}
		}
		m.mu.Unlock()
		break
	}

	w, err := m.spawnWithRetry(sessionID, p)

	// Release the single-flight (success or failure) and promote in the SAME
	// critical section, so a waiter's re-check either sees the live worker or
	// finds spawning empty and takes over.
	m.mu.Lock()
	delete(m.spawning, sessionID)
	close(spawnCh)
	if err != nil {
		m.mu.Unlock()
		return nil, err
	}

	// Promote from inflight to workers atomically. Register only if still alive:
	// the spawn ran outside m.mu, so its onDie could have fired-and-found-nothing
	// before this insert. A dead worker here is returned unregistered; beginTurn
	// will fail and the caller re-acquires.
	delete(m.inflight, w)
	if existing, ok := m.workers[sessionID]; ok && !existing.Dead() && existing != w {
		// 理论上单飞后到不了这里；防御：已有活 worker 就杀掉新 spawn 的，用旧的，
		// 保证同一 session 永远只有一个持久进程写它的 jsonl。
		m.mu.Unlock()
		log.Printf("[channel] WARN session=%s: live worker appeared during spawn; killing the duplicate", sessionID)
		w.kill()
		return existing, nil
	}
	if !w.Dead() {
		m.workers[sessionID] = w
	}
	m.mu.Unlock()
	return w, nil
}

func (m *chanManager) spawnWithRetry(sessionID string, p spawnParams) (*chanWorker, error) {
	useResume := claudeSessionExists(p.workdir, sessionID)
	flag := "--session-id"
	if useResume {
		flag = "--resume"
	}

	for attempt := 0; attempt < 2; attempt++ {
		args := append(append([]string{}, p.args...), flag, sessionID)
		w, err := spawnChanWorker(sessionID, args, p.workdir, p.envVars, flag, func() { m.onWorkerDie(sessionID) })
		if err != nil {
			return nil, err
		}
		w.boundSystemPrompt = p.systemPrompt
		w.boundModel = p.model
		// Track during the startup window so a SIGTERM Stop() can kill it
		// (it may be mid-MCP-init for several seconds). acquire promotes it.
		m.trackInflight(w)

		switch w.waitStartup(8 * time.Second) {
		case startupReady:
			return w, nil
		case startupRetryResume:
			log.Printf("[channel] session=%s: --session-id rejected (already in use), retrying with --resume", sessionID)
			m.untrackInflight(w)
			w.kill()
			flag = "--resume"
		case startupRetryNew:
			log.Printf("[channel] session=%s: --resume rejected (no conversation), retrying with --session-id", sessionID)
			m.untrackInflight(w)
			w.kill()
			flag = "--session-id"
		case startupFatal:
			m.untrackInflight(w)
			w.kill()
			return nil, fmt.Errorf("channel worker for session %s died at startup", sessionID)
		}
	}
	return nil, fmt.Errorf("channel worker for session %s failed to start after flag retry", sessionID)
}

func (m *chanManager) evictOldestLocked() {
	var oldestKey string
	var oldest time.Time
	for k, w := range m.workers {
		if w.inTurn.Load() {
			continue // don't evict a worker that is mid-turn
		}
		if oldestKey == "" || w.LastUsed().Before(oldest) {
			oldestKey = k
			oldest = w.LastUsed()
		}
	}
	if oldestKey == "" {
		return // all workers busy; allow the new spawn to exceed the soft cap
	}
	w := m.workers[oldestKey]
	delete(m.workers, oldestKey)
	log.Printf("[channel] capacity evict: session=%s", oldestKey)
	go w.kill()
}

func (m *chanManager) onWorkerDie(sessionID string) {
	m.mu.Lock()
	w := m.workers[sessionID]
	if w != nil && w.Dead() {
		delete(m.workers, sessionID)
	}
	m.mu.Unlock()
}

// Stop kills every worker — persistent AND inflight (in-spawn + ephemeral) — on
// shutdown, so no orphaned claude child is left behind.
func (m *chanManager) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, w := range m.workers {
		w.kill()
	}
	for w := range m.inflight {
		w.kill()
	}
	m.workers = make(map[string]*chanWorker)
	m.inflight = make(map[*chanWorker]struct{})
}

// snapshot returns a debug view of all live workers (persistent + inflight).
func (m *chanManager) snapshot() []map[string]any {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]map[string]any, 0, len(m.workers)+len(m.inflight))
	for k, w := range m.workers {
		out = append(out, map[string]any{
			"session_id": k,
			"claude_sid": w.SessionID(),
			"flag":       w.usedFlag,
			"last_used":  w.LastUsed().Format(time.RFC3339),
			"dead":       w.Dead(),
			"transient":  false,
		})
	}
	for w := range m.inflight {
		out = append(out, map[string]any{
			"session_id": w.key,
			"claude_sid": w.SessionID(),
			"flag":       w.usedFlag,
			"last_used":  w.LastUsed().Format(time.RFC3339),
			"dead":       w.Dead(),
			"transient":  true,
		})
	}
	return out
}

// claudeSessionExists reports whether claude already has a session jsonl for
// (workdir, sessionID). Claude stores them at
// ~/.claude/projects/<encoded-cwd>/<sessionID>.jsonl where the encoding is the
// realpath of the cwd with '/' and '_' replaced by '-'. This is a best-effort
// pre-check; spawnWithRetry corrects a wrong guess via the flag-swap retry, so
// an encoding miss is non-fatal.
func claudeSessionExists(workdir, sessionID string) bool {
	if sessionID == "" || workdir == "" {
		return false
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return false
	}
	path := filepath.Join(home, ".claude", "projects", encodeClaudeCwd(workdir), sessionID+".jsonl")
	_, statErr := os.Stat(path)
	return statErr == nil
}

// encodeClaudeCwd reproduces claude's project-directory encoding: the realpath
// of the cwd with '/' and '_' replaced by '-'. Empirically verified against
// claude 2.1.177 (e.g. /private/tmp/probe_resume_wd → -private-tmp-probe-resume-wd).
func encodeClaudeCwd(workdir string) string {
	abs := workdir
	if resolved, err := filepath.EvalSymlinks(workdir); err == nil {
		abs = resolved
	} else if a, err := filepath.Abs(workdir); err == nil {
		abs = a
	}
	return strings.NewReplacer("/", "-", "_", "-").Replace(abs)
}
