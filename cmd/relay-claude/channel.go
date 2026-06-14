package main

import (
	"bufio"
	"context"
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

	turnMu sync.Mutex // held by the handler for the whole duration of a turn

	curMu    sync.Mutex
	curLines chan string // current turn's line sink; only drainStdout closes it
}

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
	}
	w.markUsed()

	go w.drainStdout(stdout)
	go w.drainStderr(stderr)
	go func() {
		_ = cmd.Wait()
		w.dead.Store(true)
		_ = stdin.Close()
		close(w.deadCh)
		// Unblock any handler still ranging over the current turn channel.
		w.closeCurrentTurn(nil)
		if onDie != nil {
			onDie()
		}
	}()

	log.Printf("[channel] spawned worker session=%s flag=%s pid=%d args=%v", key, usedFlag, cmd.Process.Pid, args)
	return w, nil
}

// closeCurrentTurn closes the active turn's line channel and clears it. When
// only is non-nil, it only closes if that exact channel is still current
// (prevents closing a freshly started turn). Safe to call repeatedly.
func (w *chanWorker) closeCurrentTurn(only chan string) {
	w.curMu.Lock()
	defer w.curMu.Unlock()
	if w.curLines == nil {
		return
	}
	if only != nil && w.curLines != only {
		return
	}
	close(w.curLines)
	w.curLines = nil
}

func (w *chanWorker) drainStdout(r io.Reader) {
	defer w.closeCurrentTurn(nil) // process stdout ended → unblock the handler
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
		cur := w.curLines
		w.curMu.Unlock()
		if cur == nil {
			continue
		}
		cur <- line
		if probe.Type == "result" {
			// Turn boundary: this user message has been fully answered.
			w.closeCurrentTurn(cur)
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
	lines := make(chan string, 256)
	w.curMu.Lock()
	w.curLines = lines
	w.curMu.Unlock()

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
		w.closeCurrentTurn(lines)
		w.turnMu.Unlock()
		return nil, fmt.Errorf("write user message: %w", err)
	}
	w.markUsed()
	return lines, nil
}

// endTurn releases the worker for the next request. It must be called after the
// turn's line channel has closed (the handler always drains to completion).
func (w *chanWorker) endTurn() {
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
// backstop). The session is persisted on disk, so a later request --resumes it.
func (w *chanWorker) kill() {
	w.dead.Store(true)
	proc.KillGroup(w.cmd)
}

// ---- manager ----

// chanManagerConfig governs reaping + capacity.
type chanManagerConfig struct {
	IdleTTL     time.Duration
	MaxChannels int
}

// chanManager owns all channel workers, keyed by session_id.
type chanManager struct {
	cfg chanManagerConfig

	mu      sync.Mutex
	workers map[string]*chanWorker
}

func newChanManager(cfg chanManagerConfig) *chanManager {
	return &chanManager{cfg: cfg, workers: make(map[string]*chanWorker)}
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
		if w.LastUsed().Before(cutoff) {
			toKill = append(toKill, w)
			delete(m.workers, k)
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
func (m *chanManager) acquire(sessionID string, p spawnParams) (*chanWorker, error) {
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

	w, err := m.spawnWithRetry(sessionID, p)
	if err != nil {
		return nil, err
	}

	m.mu.Lock()
	m.workers[sessionID] = w
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

		switch w.waitStartup(8 * time.Second) {
		case startupReady:
			return w, nil
		case startupRetryResume:
			log.Printf("[channel] session=%s: --session-id rejected (already in use), retrying with --resume", sessionID)
			w.kill()
			flag = "--resume"
		case startupRetryNew:
			log.Printf("[channel] session=%s: --resume rejected (no conversation), retrying with --session-id", sessionID)
			w.kill()
			flag = "--session-id"
		case startupFatal:
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
		if oldestKey == "" || w.LastUsed().Before(oldest) {
			oldestKey = k
			oldest = w.LastUsed()
		}
	}
	if oldestKey == "" {
		return
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

// Stop kills every worker (called on shutdown).
func (m *chanManager) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, w := range m.workers {
		w.kill()
	}
	m.workers = make(map[string]*chanWorker)
}

// snapshot returns a debug view of tracked workers.
func (m *chanManager) snapshot() []map[string]any {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]map[string]any, 0, len(m.workers))
	for k, w := range m.workers {
		out = append(out, map[string]any{
			"session_id": k,
			"claude_sid": w.SessionID(),
			"flag":       w.usedFlag,
			"last_used":  w.LastUsed().Format(time.RFC3339),
			"dead":       w.Dead(),
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
