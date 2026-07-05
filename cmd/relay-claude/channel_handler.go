package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"clawrelay-api/pkg/attachments"
	"clawrelay-api/pkg/openai"
)

// channelInterruptBackstop bounds how long an interrupted turn (stop /
// AskUserQuestion) may keep draining in the background before the worker is
// force-killed. claude's interrupt for in-progress text only takes effect at
// the next message boundary, so a stopped turn finishes naturally within
// seconds-to-tens-of-seconds; this generous window lets that happen and keeps
// the hot process alive for the next request. Only a genuinely stuck turn hits
// the backstop kill — and even then its session is on disk, so the next request
// --resumes it. Overridable in tests.
var channelInterruptBackstop = 90 * time.Second

// channelQueuedPingInterval is how often a request queued behind another turn
// on the same session emits a `: queued` SSE comment. 上游 wuji_tools 的
// sock_read=120s 只看字节到达；排队期间必须持续有字节流出，连接才不会被掐。
// Overridable in tests.
var channelQueuedPingInterval = 15 * time.Second

// channelEmitErrClose ends an ALREADY-STARTED SSE stream with a visible error:
// SSE 头一旦发出（handleChannelStreamResponse 把它提前到了 acquire 之前），
// openai.WriteError 的状态码就写不进去了，只能以 content delta + finish +
// [DONE] 收尾，让上游把错误当普通回复展示，而不是挂死等一个永远不来的 [DONE]。
func channelEmitErrClose(w http.ResponseWriter, flusher http.Flusher, chatID string, created int64, model, msg string) {
	chunk := openai.ChatCompletionResponse{
		ID: chatID, Object: "chat.completion.chunk", Created: created, Model: model,
		Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: openai.NewChatMessage("assistant", "⚠️ "+msg)}},
	}
	data, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", data)

	finish := "stop"
	fin := openai.ChatCompletionResponse{
		ID: chatID, Object: "chat.completion.chunk", Created: created, Model: model,
		Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: openai.NewChatMessage("", ""), FinishReason: &finish}},
	}
	data, _ = json.Marshal(fin)
	fmt.Fprintf(w, "data: %s\n\n", data)
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// lockTurnWithQueuedPings ctx 感知地等 worker 的 turnMu，等待期间周期性写
// `: queued` SSE 注释——注释行对 OpenAI 客户端不可见，不污染回复，但让上游代理
// 看到连接还活着、请求只是在排队。拿到锁返回 nil；ctx 先取消返回错误（锁由
// lockTurn 内部归还，不泄漏）。
func lockTurnWithQueuedPings(ctx context.Context, w http.ResponseWriter, flusher http.Flusher, worker *chanWorker) error {
	done := make(chan error, 1)
	go func() { done <- worker.lockTurn(ctx) }()
	tick := time.NewTicker(channelQueuedPingInterval)
	defer tick.Stop()
	for {
		select {
		case err := <-done:
			return err
		case <-tick.C:
			fmt.Fprintf(w, ": queued\n\n")
			flusher.Flush()
		}
	}
}

// buildChannelArgs assembles the claude flags for a persistent stream-json
// process. The trailing session flag (--session-id / --resume) is appended by
// the manager. Unlike legacy buildClaudeArgs the prompt is NOT passed here —
// each turn's user message is written to stdin as a stream-json envelope.
func buildChannelArgs(req *openai.ChatCompletionRequest, model, systemPrompt string) []string {
	var args []string
	// No --print: the relay captures stdout via a pipe (not a TTY), which alone
	// puts claude into non-interactive mode (per `claude --help`: "via -p, or
	// when stdout is not a TTY"). stream-json input/output therefore works
	// without the explicit flag. Verified: multi-turn + interrupt behave
	// identically to the --print variant.
	args = append(args, "--input-format", "stream-json")
	args = append(args, "--output-format", "stream-json")
	args = append(args, "--include-partial-messages")
	args = append(args, "--verbose")

	if systemPrompt != "" {
		args = append(args, "--append-system-prompt", systemPrompt)
	}
	if req.SystemPromptFile != "" {
		args = append(args, "--append-system-prompt-file", req.SystemPromptFile)
	}
	args = append(args, "--model", model)

	permMode := "bypassPermissions"
	if req.PermissionMode != "" {
		permMode = req.PermissionMode
	}
	args = append(args, "--permission-mode", permMode)

	if req.AllowedTools != "" {
		args = append(args, "--allowedTools", req.AllowedTools)
	}
	for _, dir := range req.AddDirs {
		if dir != "" {
			args = append(args, "--add-dir", dir)
		}
	}

	maxTurns := 20
	if req.MaxTurns != nil {
		maxTurns = *req.MaxTurns
	}
	args = append(args, "--max-turns", openai.FmtInt(maxTurns))

	if req.Effort != "" {
		args = append(args, "--effort", req.Effort)
	}
	if req.Settings != "" {
		args = append(args, "--settings", req.Settings)
	}
	return args
}

// extractSystemPrompt returns the (last) system message content, matching
// buildPromptFromMessages' behavior.
func extractSystemPrompt(messages []openai.ChatMessage) string {
	var sp string
	for i := range messages {
		if messages[i].Role == "system" {
			sp = messages[i].ContentString()
		}
	}
	return sp
}

// lastUserTurnContent extracts the final user message as a single string,
// inlining any attachments as `[Image: /path]` / `[File: /path]` markers
// exactly as buildPromptFromMessages does for the legacy path. Channel mode
// only feeds this newest turn — prior turns already live in the persistent
// process's context. When sessionDir is empty, attachment temp files are
// returned for the caller to clean up.
func lastUserTurnContent(messages []openai.ChatMessage, sessionDir string) (content string, tempFiles []string, ok bool) {
	idx := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			idx = i
			break
		}
	}
	if idx < 0 {
		return "", nil, false
	}
	msg := messages[idx]
	text := msg.ContentString()

	files := attachments.ExtractAndSave(msg.Content, sessionDir, "claude-img", "claude-file")
	for _, a := range files {
		if sessionDir == "" {
			tempFiles = append(tempFiles, a.Path)
		}
	}
	if len(files) > 0 {
		var refs []string
		for _, a := range files {
			if a.IsImage {
				refs = append(refs, fmt.Sprintf("[Image: %s]", a.Path))
			} else {
				refs = append(refs, fmt.Sprintf("[File: %s]", a.Path))
			}
		}
		if text != "" {
			text += "\n"
		}
		text += strings.Join(refs, "\n")
	}
	return text, tempFiles, true
}

// handleChannelEphemeralStreamResponse serves a request that has NO session_id
// while in channel mode. Rather than degrade to the legacy `claude -p` path, it
// runs the request through the same stream-json channel mechanism on a fresh,
// throwaway process: mint a brand-new session_id, spawn one `--input-format
// stream-json` process, feed the full conversation, stream the turn, then kill
// the process. No reuse, no pooling — each independent (e.g. cron) request gets
// its own isolated run, so contexts never cross-contaminate.
//
// The conversation is flattened exactly as the legacy path would build it (no
// prior process context exists), so what claude receives is identical to
// legacy; only the delivery channel differs (stdin stream-json vs `-p`).
func handleChannelEphemeralStreamResponse(w http.ResponseWriter, r *http.Request, req *openai.ChatCompletionRequest, model string, includeUsage bool) {
	// No session → temp attachments, full-conversation flatten (== legacy, so
	// what claude receives is identical; only the delivery channel differs). No
	// empty-prompt rejection here: the legacy path accepts it, so we match it.
	prompt, systemPrompt, tempFiles := buildPromptFromMessages(req.Messages, "")
	if len(tempFiles) > 0 {
		defer func() {
			for _, f := range tempFiles {
				os.Remove(f)
			}
		}()
	}

	chatID := openai.GenerateChatID()
	created := time.Now().Unix()
	sid := newUUID()

	args := append(buildChannelArgs(req, model, systemPrompt), "--session-id", sid)
	worker, err := spawnChanWorker(sid, args, req.WorkingDir, req.EnvVars, "--session-id", func() {})
	if err != nil {
		log.Printf("[channel] ephemeral spawn failed: %v", err)
		openai.WriteError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	// Track so SIGTERM shutdown reaps it and /channels can see it.
	channelMgr.trackInflight(worker)

	lines, err := worker.beginTurn(prompt)
	if err != nil {
		channelMgr.untrackInflight(worker)
		worker.kill()
		openai.WriteError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	// recordEphemeralResult 解析一行残余输出：是 result 就给这轮 ephemeral 记账
	// （identityMeter 场景：bare/modelUsage 都是本轮值，perModelCounts 直接按实际
	// 消费模型归属）。返回是否记到（A9）。
	recordEphemeralResult := func(line string) bool {
		var event claudeEvent
		if json.Unmarshal([]byte(line), &event) != nil || event.Type != "result" {
			return false
		}
		pm := perModelCounts(&event, model)
		if pm == nil {
			return false
		}
		stats.RecordTurn(model, pm)
		log.Printf("[channel] ephemeral interrupted-turn usage recorded session=%s model=%s per_model=%+v", sid, model, pm)
		return true
	}

	// The throwaway process is always killed when this turn ends (completion,
	// stop, or AskUserQuestion). kill() abandons the active turn (unblocking the
	// stdout reader), and the background drain lets all worker goroutines exit.
	// drainWithGrace 在 deadCh 已触发后继续限时接收 lines：deadCh 关闭后它恒可
	// 读，与仍有缓冲的 lines 双就绪时 select 是伪随机选择——不加宽限期会有约一半
	// 概率把已缓冲的 result 连同 usage 一起丢掉（进程死亡与 result 送达几乎同时
	// 是 interrupt 的常规时序，不是边角）。
	drainWithGrace := func(fn func(string) bool) {
		grace := time.After(500 * time.Millisecond)
		for {
			select {
			case ln, ok := <-lines:
				if !ok {
					return
				}
				fn(ln)
			case <-grace:
				return
			}
		}
	}

	finished := false
	finish := func() {
		if finished {
			return
		}
		finished = true
		worker.kill()
		channelMgr.untrackInflight(worker)
		go func() {
			for {
				select {
				case ln, ok := <-lines:
					if !ok {
						return
					}
					// 机会主义收割（A9 minimal）：kill 已发出，SIGKILL 后 claude
					// 不会再补 result；但进程死前已缓冲进管道的 result 能捞到就
					// 捞（正常完成路径的 result 已被前台 feed 消费并记账，不会
					// 走到这里，故无重复计数）。
					recordEphemeralResult(ln)
				case <-worker.deadCh:
					drainWithGrace(recordEphemeralResult)
					return
				}
			}
		}()
	}
	defer finish()

	// finishWithHarvest 用于 AskUserQuestion 分支（A9 full）：该轮被我们主动
	// 截断，result 还没出来。先 stdin interrupt（claude 中止当前轮并 emit 带真实
	// usage 的 result）→ 限时 10s 收割记账 → 再 kill 兜底 + 排空。
	finishWithHarvest := func() {
		if finished {
			return
		}
		finished = true
		_ = worker.interrupt()
		go func() {
			deadline := time.After(10 * time.Second)
		harvest:
			for {
				select {
				case ln, ok := <-lines:
					if !ok {
						break harvest
					}
					if recordEphemeralResult(ln) {
						break harvest
					}
				case <-worker.deadCh:
					drainWithGrace(recordEphemeralResult) // 缓冲里可能还躺着 result
					break harvest
				case <-deadline:
					log.Printf("[channel] ephemeral usage harvest timed out (10s) session=%s", sid)
					break harvest
				}
			}
			// kill 之后才 untrack：收割窗口（最长 10s）内 worker 必须留在
			// inflight 里，否则 SIGTERM 时 Stop() 找不到它，claude 变孤儿进程。
			worker.kill()
			channelMgr.untrackInflight(worker)
			for {
				select {
				case _, ok := <-lines:
					if !ok {
						return
					}
				case <-worker.deadCh:
					return
				}
			}
		}()
	}

	log.Printf("[channel] ephemeral turn start session=%s model=%s chat=%s", sid, model, chatID)

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	flusher, isFlusher := w.(http.Flusher)
	if !isFlusher {
		log.Printf("[channel] ephemeral: streaming not supported; killing")
		return
	}

	fmt.Fprintf(w, ": ping\n\n")
	flusher.Flush()

	heartbeat := time.NewTicker(30 * time.Second)
	defer heartbeat.Stop()

	t := newSSETranslator(chatID, created, model, "", identityMeter{}, "") // no session → log no-ops
	for {
		select {
		case <-r.Context().Done():
			// Upstream disconnected = stop. 与 V1/持久 channel 口径一致：先
			// interrupt 让 claude 吐带真实 usage 的 result 并收割记账，再 kill
			// 兜底（直接 SIGKILL 的话被停轮的消耗永久漏计）。
			log.Printf("[channel] ephemeral stop session=%s (interrupt+harvest)", sid)
			finishWithHarvest()
			return
		case <-worker.deadCh:
			// Process died mid-turn; `lines` may never close. Finalize and exit
			// rather than block forever (finish kills + drains).
			t.flushAggLog()
			t.EmitFinishIfNoResult(w, flusher) // died without result → synthetic finish (A2)
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			log.Printf("[channel] ephemeral turn end session=%s (process died)", sid)
			return
		case <-heartbeat.C:
			fmt.Fprintf(w, ": keepalive\n\n")
			flusher.Flush()
		case line, lok := <-lines:
			if !lok {
				t.flushAggLog()
				t.EmitFinishIfNoResult(w, flusher) // closed without result → synthetic finish (A2)
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				log.Printf("[channel] ephemeral turn end session=%s (completed)", sid)
				return
			}
			if line == "" {
				continue
			}
			if t.feed(w, flusher, line, includeUsage) == outcomeAskUserDone {
				// No session to continue on; the card was emitted. Interrupt →
				// harvest the aborted turn's usage → kill (A9).
				log.Printf("[channel] ephemeral turn end session=%s (ask_user)", sid)
				finishWithHarvest()
				return
			}
		}
	}
}

// handleChannelStreamResponse serves one /v1/chat/completions turn through the
// persistent channel worker for req.SessionID. The SSE byte stream is produced
// by the shared sseTranslator, identical to the legacy path. The process is
// kept alive across turns; stop and AskUserQuestion interrupt (not kill) it.
func handleChannelStreamResponse(w http.ResponseWriter, r *http.Request, req *openai.ChatCompletionRequest, model string, includeUsage bool) {
	sessionID := req.SessionID
	systemPrompt := extractSystemPrompt(req.Messages)

	var sessionDir string
	if sessionID != "" {
		sessionDir = filepath.Join(sessionStore.AbsDir(), sessionID, "files")
	}
	content, tempFiles, ok := lastUserTurnContent(req.Messages, sessionDir)
	if len(tempFiles) > 0 {
		defer func() {
			for _, f := range tempFiles {
				os.Remove(f)
			}
		}()
	}
	if !ok {
		openai.WriteError(w, http.StatusBadRequest, "invalid_request_error", "no user message to feed channel worker")
		return
	}

	chatID := openai.GenerateChatID()
	created := time.Now().Unix()

	p := spawnParams{
		args:         buildChannelArgs(req, model, systemPrompt),
		workdir:      req.WorkingDir,
		envVars:      req.EnvVars,
		systemPrompt: systemPrompt,
		model:        model,
	}

	// SSE 头 + 首个 ping 在 acquire/等锁之前就发出去：同 session 的排队请求会
	// 阻塞在 turnMu 上，期间一个字节都不发的话，上游 wuji_tools 的 sock_read
	//（120s）会把还没轮到的连接掐掉。头发出去之后，下面所有错误路径都只能以
	// SSE 块收尾（channelEmitErrClose），WriteError 的状态码已经写不进去了。
	flusher, isFlusher := w.(http.Flusher)
	if !isFlusher {
		log.Printf("[channel] streaming not supported session=%s", sessionID)
		openai.WriteError(w, http.StatusInternalServerError, "server_error", "streaming unsupported by this connection")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, ": ping\n\n")
	flusher.Flush()

	worker, err := channelMgr.acquire(r.Context(), sessionID, p)
	if err != nil {
		log.Printf("[channel] acquire failed session=%s: %v", sessionID, err)
		channelEmitErrClose(w, flusher, chatID, created, model, err.Error())
		return
	}

	// ctx 感知地排队等 turnMu（期间发 `: queued` 注释保活）：客户端已断开的
	// 排队请求立刻放弃，消息不再注入 claude——否则它会进会话历史、随即被
	// interrupt，"进了历史但没被回答"。
	if err := lockTurnWithQueuedPings(r.Context(), w, flusher, worker); err != nil {
		// 客户端已经走了，没有对象可以收错误块；只留日志。
		log.Printf("[channel] queued request gave up session=%s: %v", sessionID, err)
		return
	}

	lines, err := worker.beginTurnLocked(content)
	if err != nil {
		// Worker died between acquire and beginTurn (e.g. reaped). One retry
		// with a fresh acquire (which will spawn/resume).
		log.Printf("[channel] beginTurn failed session=%s: %v; retrying acquire", sessionID, err)
		if worker, err = channelMgr.acquire(r.Context(), sessionID, p); err == nil {
			if err = worker.lockTurn(r.Context()); err == nil {
				lines, err = worker.beginTurnLocked(content)
			}
		}
		if err != nil {
			channelEmitErrClose(w, flusher, chatID, created, model, err.Error())
			return
		}
	}
	log.Printf("[channel] turn start session=%s claude_sid=%s flag=%s model=%s chat=%s",
		sessionID, worker.SessionID(), worker.usedFlag, model, chatID)

	// Stats attribution follows the worker's spawn-time model, not the request
	// model: after a hot model switch the persistent process keeps consuming
	// under the old model until respawn (A5).
	meterModel := worker.boundModel
	if meterModel == "" {
		meterModel = model
	}

	// Exactly one of these takes ownership of releasing the worker (endTurn):
	// the foreground path on normal completion, or a background drainer when we
	// interrupt (stop / AskUserQuestion). turnReleased guards against both.
	turnReleased := false
	releaseInBackground := func(reason string) {
		if turnReleased {
			return
		}
		turnReleased = true
		log.Printf("[channel] interrupt session=%s reason=%s (keeping process alive)", sessionID, reason)
		_ = worker.interrupt()
		// Hand the turn to a background drainer so the HTTP handler returns now
		// (the client is gone / has its [DONE]); the process is kept alive for
		// the next request on this session. A backstop kills only a genuinely
		// stuck turn — claude's interrupt for in-progress text lands at the next
		// message boundary, so normal turns finish well within the window and
		// their hot process is preserved.
		//
		// Scope the backstop to THIS turn (turnSeq): if this interrupted turn
		// drains right at the backstop boundary and the next queued request has
		// already taken the worker, killTurnSeq is a no-op instead of SIGKILLing
		// that innocent successor mid-stream.
		turnSeq := worker.turnSeqNow()
		timer := time.AfterFunc(channelInterruptBackstop, func() {
			if worker.killTurnSeq(turnSeq) {
				log.Printf("[channel] interrupt backstop fired session=%s; killed stuck turn", sessionID)
			}
		})
		go func() {
			for {
				select {
				case ln, ok := <-lines:
					if !ok {
						timer.Stop()
						worker.endTurn()
						return
					}
					// 中断轮记账（A7）：解析残余行中的 result，推进 meter 基线的
					// 同时把这轮的差分 usage 记入 stats（中断轮前台没有任何记账）。
					recordInterruptedTurnUsage(worker.meter, ln, meterModel)
				case <-worker.deadCh:
					// Process died; lines may never close. 先限时扫掉已缓冲的行
					// （interrupt 的 result 常与进程退出同刻到达，deadCh 与 lines
					// 双就绪时 select 伪随机，不扫会有约一半概率丢掉这轮记账），
					// 再释放 worker，避免泄漏本 goroutine。
					grace := time.After(500 * time.Millisecond)
				drain:
					for {
						select {
						case ln, ok := <-lines:
							if !ok {
								break drain
							}
							recordInterruptedTurnUsage(worker.meter, ln, meterModel)
						case <-grace:
							break drain
						}
					}
					timer.Stop()
					worker.endTurn()
					return
				}
			}
		}()
	}
	defer func() {
		if !turnReleased {
			worker.endTurn()
		}
	}()

	heartbeat := time.NewTicker(30 * time.Second)
	defer heartbeat.Stop()

	t := newSSETranslator(chatID, created, model, sessionID, worker.meter, worker.boundModel)
	ctxCh := r.Context().Done()

	for {
		select {
		case <-ctxCh:
			// Upstream disconnected = stop. Interrupt (not kill) and release in
			// the background; the process and its context survive.
			releaseInBackground("client_disconnect")
			return
		case <-worker.deadCh:
			// Process died mid-turn; `lines` may never close. Finalize and let
			// the deferred endTurn release the worker.
			t.flushAggLog()
			t.EmitFinishIfNoResult(w, flusher) // died without result → synthetic finish (A2)
			sessionStore.LogDone(sessionID, t.StreamUsage())
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			log.Printf("[channel] turn end session=%s (process died)", sessionID)
			return
		case <-heartbeat.C:
			fmt.Fprintf(w, ": keepalive\n\n")
			flusher.Flush()
		case line, lok := <-lines:
			if !lok {
				// Turn's result closed the channel: normal completion. (If the
				// channel closed WITHOUT a result — e.g. drainStdout scanner
				// error — the translator still owes a terminal chunk.)
				t.flushAggLog()
				t.EmitFinishIfNoResult(w, flusher)
				sessionStore.LogDone(sessionID, t.StreamUsage())
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				log.Printf("[channel] turn end session=%s (completed)", sessionID)
				return
			}
			if line == "" {
				continue
			}
			if t.feed(w, flusher, line, includeUsage) == outcomeAskUserDone {
				// §4.5: tool_call + finish + [DONE] already emitted. Interrupt
				// the hung turn but keep the process alive — the user's answer
				// arrives as a new request on the same session_id.
				releaseInBackground("ask_user_question")
				return
			}
		}
	}
}
