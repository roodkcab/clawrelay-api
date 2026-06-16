# 开发任务：为 clawrelay-api (relay-claude) 新增「Channel 常驻进程模式」

## 0. 一句话目标
在 `relay-claude` 里**新增一种接入模式**：一个 `session_id` 对应一个常驻 `claude --input-format stream-json` 进程，跨多轮对话从 stdin 持续喂消息，消除每轮的「spawn + session 重载 + MCP/skill 重新初始化」开销。通过**启动参数**在新模式（channel）与现有模式（legacy，即 `claude -p` 每请求一次）之间切换。**对上游 wuji_tools 的 HTTP/SSE 协议零改动。**
完成之后，更新10.0.100.173上的claude01服务的cmd/relay-claude进行测试，需要两种模式进行对比测试，都需要能够完美运行才算结束。

---

## 1. 背景与约束

- **仓库**：`/Users/wangxin/work/flutter/clawrelay/clawrelay-api`（Go，module `clawrelay-api`）。本任务只动 `cmd/relay-claude` 及 `pkg`，**不要碰 `cmd/relay-codex`**。
- **现状**：`relay-claude` 是 OpenAI 兼容 HTTP 前端，`POST /v1/chat/completions` → 每请求 fork 一次 `claude`（一次性 print 模式，prompt 走 stdin 纯文本，`--output-format stream-json --resume <sid>`，出完即退），把 stream-json 翻成 OpenAI SSE 回吐。
- **上游**：`wuji_tools`（Python，`src/adapters/claude_relay_adapter.py`）通过 SSE 调用本服务。它负责所有产品功能（stop/reset、思考展示、AskUserQuestion 卡片、引用上下文、公告拦截、对话日志、token 统计）。**relay 只是 claude 进程的消息中转，绝不承载这些功能。**
- **硬约束**：
  1. `/v1/chat/completions` 的**请求体和 SSE 响应格式一字不改**（OpenAI 兼容）。
  2. `--mode=legacy`（默认）行为与现状**完全一致**，零回归。
  3. 不改 `wuji_tools`，不改 `relay-codex`。

---

## 2. 已验证的关键技术事实（务必遵守，别重新发明轮子）

> 下列已基于 claude 2.1.177 实测 + 官方 issue 核实，cheat sheet 见 `/tmp/claude_stream_json_control_cheatsheet.md`。

- **中断不杀**：向常驻进程 stdin 写一行 `{"type":"interrupt"}` 即可**立即中断当前 turn 且进程保持存活**——无需 `request_id`、无 ack；紧接着写下一行 `{"type":"user",...}` 会被同一进程的新 turn 处理。（已实测验证两个独立 `result`）
- **喂消息**：一行 NDJSON `{"type":"user","message":{"role":"user","content":<string 或 OpenAI content blocks 数组>}}`。多模态用 content blocks（`text` / `image` base64）。
- **tool_result 不能通过 stdin 回喂**（GitHub #16712，未实现）。⇒ AskUserQuestion **不能**靠"进程挂起等 tool_result"，必须用第 5 节的 interrupt+留活方案。
- 进程参数仍需 `--include-partial-messages --verbose`。stdout 是 NDJSON stream-json，事件类型：`system`(init) / `assistant` / `stream_event`(含 `content_block_delta`→`text_delta`/`thinking_delta`) / `result`(turn 结束) / `rate_limit_event`。
- `session_id` 通过 `--session-id <id>`（首建）或 `--resume <id>`（续接已存在会话）在 spawn 时一次性确定，**不在每条消息里重复传**。
- 可选 `--replay-user-messages`：让进程把收到的 user 消息回显 stdout，可作"已接收"ack。按需。

---

## 3. 现有代码地图（要复用 / 参照的锚点）

| 文件:行 | 作用 | 在本任务里怎么用 |
|---------|------|------------------|
| `cmd/relay-claude/main.go:70` `chatCompletionsHandler` | 解析 `ChatCompletionRequest` → `buildPromptFromMessages` → `buildClaudeArgs` → 分发到 `handleStreamResponse` 等 | channel 模式在此分流到新路径 |
| `cmd/relay-claude/main.go:190` `main()` flags | port/proxy/model/sessions-dir/log-file/version | 加 `--mode` |
| `cmd/relay-claude/process.go:57` `buildClaudeArgs` | 拼 claude flags，返回 `(args, stdinData=prompt)` | channel 模式需要一个**变体**：不带 `--resume`/纯文本 prompt，改 `--input-format stream-json` + `--session-id` |
| `cmd/relay-claude/process.go:106` `launchClaude` / `:169` `startClaudeStream` | 一次性 spawn + resume→session-id 重试 | 参照其 stdout/stderr 管道与进程组处理 |
| `cmd/relay-claude/stream.go:17` `handleStreamResponse` | **stream-json 事件 → OpenAI SSE chunk 的翻译核心**（text/thinking/tool_calls/usage/result/`[DONE]`）；断连在 `:96` `r.Context().Done()`→`KillGroup`；turn 结束 `:253` emit usage+`[DONE]`+`KillGroup` | **把事件翻译逻辑抽成可复用函数**，channel 模式复用之；断连/结束语义改写（见 §4/§5） |
| `pkg/proc/proc.go` | `SetNewProcessGroup` / `KillGroup` / `WatchDisconnect` | reaper/shutdown 时仍用 KillGroup；断连**不再** KillGroup |
| `pkg/sessions/store.go` | relay 自己的会话日志（`/sessions` 查看 UI），`<Dir>/<id>.jsonl`；`LogRequest/LogDelta/LogToolUse/LogDone` | channel 模式照样调，保持 UI + 统计一致。**注意**：这与 claude 自身的 `~/.claude/projects/<encoded-cwd>/<uuid>.jsonl` 不是一回事 |
| `pkg/openai/types.go:16` `ChatCompletionRequest` | 字段：Model/Messages/Stream/StreamOptions/WorkingDir/EnvVars/MaxTurns/**SessionID**/Effort/SystemPromptFile/PermissionMode/AllowedTools/AddDirs/Settings/Tools | 入参不变 |
| **参考实现** `/tmp/clawrelay-v2/pkg/claude/worker.go` + `manager.go` | 已跑通的「常驻 stream-json 进程 + 按 key 复用 + FIFO turn 关联 + idle reaper + MaxPerBot evict + `--resume` + 磁盘 jsonl 校验」 | **直接移植改造**为 relay 的 channel 内核 |

---

## 4. 设计要点（务必落实）

### 4.1 模式开关
- `main.go` 加 `flag.String("mode", "legacy", "legacy=per-request claude -p；channel=常驻 stream-json 进程")`。
- legacy 走现有 `handleStreamResponse/...` 完全不变；channel 走新路径。

### 4.2 Channel 进程管理（移植 `manager.go`+`worker.go`，按 relay 语境改造）
- **key = `req.SessionID`**。
- **`session_id` 为空时的退化**（已替你拍板，可改）：channel 模式下 `session_id` 为空 ⇒ 退回 legacy 单轮一次性进程，避免产生无锚、无法复用、泄漏的常驻进程。
- **首次 spawn**：`claude --input-format stream-json --output-format stream-json --include-partial-messages --verbose --model <m> [--append-system-prompt(-file) ...] --permission-mode <p> [--allowedTools ...] [--add-dir ...] --max-turns <n> [--settings <json>] --session-id <sid>`。`workingDir`/`envVars` 同 legacy（`cleanEnv` 去 `CLAUDECODE`），`SetNewProcessGroup`。
- **复用**：进程活着 ⇒ 把本轮 user 消息写 stdin，流式读其 stdout 到**本轮 `result`** 为止。
- **重建**：进程已死/被 reap，再来消息时——若 `~/.claude/projects/<encoded-cwd>/<sid>.jsonl` 存在则用 `--resume <sid>` 续接历史，否则 `--session-id <sid>` 新建。（复用参考实现 `claudeSessionExists`）
- **生命周期**：idle reaper（默认 30m，可配 `--idle-ttl`）、每实例容量上限（默认如 50，evict 最旧）、shutdown 时 KillGroup 全部。

### 4.3 单请求 = 单 turn 的 SSE 映射
- channel 模式下，从 `req.Messages` 里只取**最后一条 `role:"user"` 消息**（连同其多模态 content blocks）作为「新一轮」，封成 `{"type":"user"}` 写进程 stdin。历史轮次已在进程上下文里，**不要重复喂**。
  - 复用 `buildPromptFromMessages`（`prompt.go`）的**附件落地 + 多模态处理**，但只针对这最后一轮。
- **system prompt 绑定时机**（已替你拍板，可改）：在**首次 spawn** 时用第一条请求的 `system_prompt` / `SystemPromptFile` 固定；后续请求的 system 变化忽略（如检测到变化打 warning）。理由：常驻进程的 system 只在启动时生效。
- 复用 §3 抽出的「事件→OpenAI chunk」翻译：从"写入本轮 user 消息后"开始读该进程 stdout，遇到本轮 `result` ⇒ emit usage（若 `include_usage`）+ `data: [DONE]`，**结束本次 HTTP 响应但进程留活**。
- **并发防护**：同一 worker 串行化（per-worker 锁/队列），杜绝两个请求交叉读同一 stdout。（wuji_tools 正常不会并发同会话，但要防御。）

### 4.4 stop = 中断不杀
- 本次请求的 `r.Context().Done()`（wuji_tools 断开 SSE 即 stop）触发时：向该 worker 的 stdin 写 `{"type":"interrupt"}`，**不要 `KillGroup`**；标记本轮结束、把 worker 释放给下一请求。进程与上下文保留。

### 4.5 AskUserQuestion（受"tool_result 不可回喂"约束）
- 检测到 assistant 发出 `AskUserQuestion` 工具调用时：照常把该工具调用事件**透传**进 SSE（wuji_tools 已能解析并弹卡片），随后对进程发 `{"type":"interrupt"}` 终止本轮挂起，结束本次 SSE（进程留活）。
- 用户答完后，wuji_tools 会带**同一 `session_id`** 再发一条普通 `/v1/chat/completions`（答案即新的 user 消息）⇒ 喂回同一活进程继续。
- 对 wuji_tools 完全透明：它只看到「工具调用事件」+「随后一次新请求」，与 legacy 的 terminate+resume 效果等价。

### 4.6 会话日志 / 可观测
- channel 模式同样调 `sessionStore.LogRequest/LogDelta/LogToolUse/LogDone`，保持 `/sessions` UI 与 token 统计一致。
- 日志显式标注：`mode=channel`、worker `spawn/reuse/interrupt/reap/evict`、session_id、turn 边界，便于线上排查。
- `/health` 不变。

---

## 5. 边界与非目标
- 不动 `relay-codex`、不动 `wuji_tools`、不改 OpenAI 请求/响应 schema。
- **不要**移植 clawrelay-wecom 的 `pkg/wecom`、`pkg/bridge`、`pkg/mcp`（那是它作为独立企微前端的部分，本任务不需要）——只移植 `pkg/claude`。
- 第三方 model（minimax/kimi/zhipu 等）按 model 透传即可，首期保证 claude 后端正确。

---

## 6. 验收标准（必须逐条可验证）
1. **零回归**：`--mode=legacy`（默认）下，现有 wuji_tools 调用行为与改动前一致。
2. **进程复用**：`--mode=channel` 下同一 `session_id` 连续多轮，进程只 spawn 一次（日志可证），第 2 轮起无 spawn/无 session 重载；给出**首字延迟 before/after 实测**。
3. **中断不杀**：请求中途断开 SSE → 进程仍存活（`ps` 可证）→ 紧接着同 session 新消息能继续且保留上下文。
4. **AskUserQuestion 全链路**：触发卡片 → 用户答 → 同会话继续，进程不重启。
5. **idle reaper**：空闲超时进程被回收；再来消息能 `--resume` 续接历史。
6. **多 session 隔离**：并发不同 session 互不串台。
7. `go build ./...` 通过；为 worker/manager 关键逻辑加单测（turn 边界切分、interrupt、复用、reaper、session_id 为空退化）。

---

## 7. 建议实施顺序
1. 把 `stream.go` 的「stream-json 事件 → OpenAI chunk」翻译抽成可复用函数（**不改 legacy 行为**，先做等价重构 + 回归）。
2. 新增 channel 内核（移植 `/tmp/clawrelay-v2/pkg/claude` 的 worker+manager，改 key=session_id、接 §1 的 SSE 翻译 + `sessionStore` 日志）。
3. `main.go` 加 `--mode`；`chatCompletionsHandler` 在 channel 模式分流。
4. 落实 §4.4 中断不杀 + §4.5 AskUserQuestion。
5. 测试 + 延迟 before/after 实测 + legacy 回归。

## 8. 必读参考
- `/tmp/clawrelay-v2/pkg/claude/worker.go`、`manager.go`（常驻进程参考实现）
- `/tmp/claude_stream_json_control_cheatsheet.md`（中断/控制面 cheat sheet）
- `cmd/relay-claude/{main.go,process.go,stream.go}`、`pkg/proc/proc.go`、`pkg/sessions/store.go`、`pkg/openai/types.go`
