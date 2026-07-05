# relay-claude Channel 常驻进程模式（v1.2.0）

> 实现规格见同目录 [`channel-mode-dev-prompt.md`](./channel-mode-dev-prompt.md)。
> 本文记录**已落地的实现、运维与上线状态**。

## 1. 它是什么

`relay-claude` 的一种接入模式：**一个 `session_id` 对应一个常驻 `claude --input-format stream-json` 进程**，跨多轮对话从 stdin 持续喂消息，消除每轮的「spawn + session 重载 + MCP/skill 重新初始化」开销。

> **不带 `--print`**：relay 用管道捕获子进程 stdout（非 TTY），claude 据此自动进入 non-interactive/headless 模式（`claude --help`：「via -p, **or when stdout is not a TTY**」），stream-json 输入/输出无需显式 `--print` 即生效。已实测多轮 + interrupt 行为与带 `--print` 完全一致。

通过启动参数 `--mode=channel` 开启；V1（旧称 legacy，`-p`/headless 每请求 fork `claude -p`；启动标志为 `--mode=v1`（默认）/默认裸 `--port=`，旧值 `--mode=legacy` 仍作别名兼容，`/health` 返回 `"mode":"v1"`）行为与改动前**字节级一致，零回归**。对上游 wuji_tools 的 OpenAI HTTP/SSE 协议**零改动**。

```
--mode=v1       (默认)  每请求 fork 一次 `claude -p`，出完即退（现状；旧值 legacy 仍兼容）
--mode=channel          每 session_id 一个常驻 stream-json 进程，多轮复用
```

## 2. 启动参数

| flag | 默认 | 说明 |
|------|------|------|
| `--mode` | `v1` | `v1`（别名 `legacy`）/ `channel` |
| `--idle-ttl` | `30m` | channel：进程空闲超过此时长被回收 |
| `--max-channels` | `50` | channel：最大常驻进程数，超出 evict 最旧 |

```bash
# channel 模式启动示例
claude_openai_api --port=50008 --mode=channel --idle-ttl=20m --max-channels=40
```

## 3. 分流规则（哪些请求走 channel）

channel 模式下，**所有 `stream=true` 且无 `tools` 的请求都走 stream-json 机制，不再降级 `claude -p`**：

| 请求形态 | 路径 |
|----------|------|
| `stream` + 有 `session_id` + 无 `tools` | **持久化 channel**：按 session_id 复用常驻进程（多轮省 spawn） |
| `stream` + 无 `session_id` + 无 `tools`（如无状态 cron） | **ephemeral channel**：全新 UUID 起独立进程，喂完整对话，跑完即 kill（见 §3.1） |
| 非流式 / 带 `tools` | V1（`handleNonStreamResponse` / `handleBufferedStreamResponse`，这些形态 wuji_tools 不会发） |

判定见 `isChannelEligible()`（`stream && 无 tools`）+ dispatch 按 `session_id` 有无分流（`cmd/relay-claude/main.go`）。

### 3.1 Ephemeral（无 session_id）独立运行

无 session_id 的请求每次都是一次**独立运行**：mint 一个全新 UUID → spawn 一个 `--input-format stream-json` 进程 → 用 `buildPromptFromMessages` 把整段对话扁平化后喂入（与 V1 收到的内容**完全一致**）→ 流式翻译返回 → 本轮 `result` 后 **kill 进程**。不复用、不入池，独立请求间上下文不串台。

> claude 在同一 cwd 下有跨 session 的 memory 行为；ephemeral 与 V1 在这点上表现一致（非回归）。
> ephemeral 进程在运行期间登记在 manager 的 `inflight` 集合里（供 SIGTERM 优雅关闭与 `/channels` 可见），跑完即注销并 kill。

## 4. 架构

| 文件 | 作用 |
|------|------|
| `translate.go` | `sseTranslator`：stream-json 事件 → OpenAI SSE chunk 翻译（V1 与 channel **共用**，等价重构） |
| `channel.go` | `chanWorker`（常驻进程，单 stdout reader 按 `result` 事件切 turn 边界，`turnMu` 串行化同 session 请求）+ `chanManager`（key=session_id、idle reaper、容量 evict、`--session-id`/`--resume` 双向 spawn 重试） |
| `channel_handler.go` | 单 turn 的 SSE 循环；stop / AskUserQuestion 的中断处理 |
| `main.go` | `--mode` 等 flag、dispatch 分流、`/channels` 端点、SIGTERM 优雅关闭 |

### turn 边界
常驻进程一条 stdout 流承载多轮。reader 把每行转发给「当前活跃 turn」的 channel，遇到 `result` 事件即关闭该 channel（turn 结束），进程留活等待下一轮。turn 之间到达的行（如 spawn 时的 init/system）无 sink，丢弃。

### spawn 决策（--session-id vs --resume）
首次 spawn 时：若 claude 已有该 session 的磁盘 jsonl（`~/.claude/projects/<encoded-cwd>/<sid>.jsonl`，编码 = realpath 后把 `/` 和 `_` 替换为 `-`）→ 用 `--resume <sid>` 续接历史；否则 `--session-id <sid>` 新建。猜错了靠**双向重试**纠正：
- `--session-id` 报 `is already in use` → 改 `--resume` 重试
- `--resume` 报 `No conversation found with session ID` → 改 `--session-id` 重试

### system prompt / env / model 绑定
常驻进程的 system/env/model **只在首次 spawn 生效**（用第一条请求的值固定）；后续请求若变化，打 warning 并忽略。同一 session（同一用户）这些值稳定，故安全。

## 5. stop 与 AskUserQuestion：中断不杀

- **stop**（wuji_tools 断开 SSE）→ 向进程 stdin 写 `{"type":"interrupt"}`，**不 KillGroup**，后台 drain 本轮到结束，进程与上下文保留。
- **AskUserQuestion** → 工具调用事件照常透传进 SSE（wuji_tools 弹卡片），随后 interrupt 终止挂起轮、结束本次 SSE，进程留活。用户答完后 wuji_tools 带同 `session_id` 再发一条普通请求，喂回同一活进程。

> ⚠️ claude 的 interrupt 对**纯文本生成不即时**（只在消息边界生效，实测约 ~10s 才出 `result`）。因此实现为「interrupt + 后台 drain 保活 + 90s 兜底 kill」：正常轮自然跑完即保留热进程，仅真卡死才兜底 kill（session 已落盘，下次请求 `--resume` 续接）。

## 6. 可观测与运维

- **`GET /health`** → `{"mode":"channel"|"v1", ...}`，可确认当前模式。
- **`GET /channels`** → 当前常驻 worker 列表（session_id、claude_sid、flag、last_used、dead），用于验证进程复用。
- 日志标注：`[channel] spawned/turn start/turn end/interrupt/reaping/capacity evict`。
- **SIGTERM 优雅关闭**：收到 SIGTERM/SIGINT 先 KillGroup 全部 worker 再退出，杜绝常驻 claude 子进程孤儿泄漏（V1 的每请求子进程会自行退出，channel 的不会）。重启用的 `kill` 即发 SIGTERM。

## 7. 二进制升级注意（chroot 实例）

`/usr/local/bin/claude_openai_api` 在生产 chroot（claude01–06）是**宿主机文件经 ro bind-mount 共享**。升级须：
1. 写**宿主机路径** `/usr/local/bin/claude_openai_api`（chroot 内只读）。
2. 用 **`mv`(rename)** 覆盖，**不能 `cp` 覆盖**正在执行的二进制（会 `ETXTBSY`）：
   ```bash
   sudo cp <new> /usr/local/bin/.cooa.new
   sudo mv -f /usr/local/bin/.cooa.new /usr/local/bin/claude_openai_api
   ```
3. 改它会影响 claude01–06 下次重启（V1 字节等价，安全）。

**回滚**：50008 重启时去掉 `--mode=channel` 即回 V1（v1.2.0 V1 == v1.1.4 V1），或恢复备份 `claude_openai_api.bak.v1.1.4`。

## 8. 上线状态与实测（2026-06-15）

- **claude01（10.0.100.173:50008）已切 channel 模式**（`--mode=channel --idle-ttl=20m --max-channels=40`），健康正常。
- 两种模式在真实环境对比（真实 `ritaoguoji` workdir，加载真实 MCP/skill）：

  | | turn1 冷启动 | turn2 |
  |---|---|---|
  | V1 | 19.2s | 4.96s（每轮重 spawn） |
  | channel | 22.3s | **1.76s（复用，省 ~20.5s）** |

- V1 真实 opus 流量零回归（token 统计完整）；本地 race + 单测 + 端到端集成全绿。
- 三轮对抗式审查共修 11 个并发/正确性问题（含 1 个会导致 relay 整体崩溃的 send-on-closed 竞争），最终审查 CLEAN。详见 §9。
- 上线后真实 opus 多轮会话已观测到 channel 复用（同 session 跨多轮复用同一进程），无 panic、内存稳定。

> **流量构成**：claude01 既有无状态 cron 巡检任务（无 `session_id` → ephemeral channel），也有多轮对话会话（有 `session_id` → 持久化复用，收益最大）。两类现在都走 channel，不再用 `-p`。

## 9. 并发设计要点（加固后）

- **per-turn 通道 `activeTurn{lines, quit}`**：`lines` **仅由 drainStdout goroutine 关闭**；其它 goroutine（cmd.Wait waiter / kill / beginTurn 错误路径）只关 `quit`（abandon），drainStdout 的发送用 `select{ case lines<-line; case <-quit }` 兜住——发送永不与关闭竞争，杜绝 "send on closed channel" panic。
- **deadCh 逃生**：beginTurn 写后复检 `deadCh` 快速失败；两个前台循环 + 两个后台 drain 都监听 `worker.deadCh`，进程中途死亡时绝不永久阻塞在「永不关闭的 lines」上。
- **inTurn 标志**：reaper / 容量 evict **跳过正在流式的 worker**（流式时长超 idle-ttl 也不会被误杀）。
- **inflight 集合**：spawn 出来但尚未 promote 的 worker（持久化 waitStartup 期 + ephemeral 运行期）都登记，SIGTERM `Stop()` 一并 kill，杜绝重启孤儿。

## 10. 稳定性加固（relay-claude 2.1.0，2026-07-05）

- **排队请求不再零字节等待**:SSE 头 + `: ping` 提前到 acquire 之前;同 session 排队等 turnMu 期间每 15s 发 `: queued` 注释,不再被上游 sock_read=120s 掐断;等待感知 ctx(客户端断开即放弃,消息不再"注入即被 interrupt"白进历史)。头发出后的错误改走 SSE 错误块(⚠️ 文本 + finish + [DONE]),不再是死流。
- **stdout scanner 错误不再产生僵尸 worker**:单行 >8MB(巨型工具结果)或读错误时,drainStdout 退出前标记 dead + KillGroup,后续请求走正常 respawn;旧行为是 worker 假活、写 stdin 永远无回应、claude 卡死在写满的管道上。
- **cmd.Wait 顺序修正**:两个 pipe reader 退出后才 Wait(os/exec 约定),interrupt 后缓冲里的 result 行不再被 Wait 关管道截断;deadCh 严格晚于 stderr 排空,waitStartup 不再可能错过 already_in_use/no_conversation 标记。
- **kill 带 reaped 防护**:进程已收割后不再裸 kill(-pid),防 PID 复用误杀。
- **drop() 尊重活跃 turn**:同 session 非流式/带 tools 请求 fall through V1 前,先 ctx 感知地等 turnMu 到手再 kill(不再当场杀断正在流式输出的 turn);等不到(ctx 取消)返回 503 "session busy"。
