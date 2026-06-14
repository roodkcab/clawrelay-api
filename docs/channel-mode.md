# relay-claude Channel 常驻进程模式（v1.2.0）

> 实现规格见同目录 [`channel-mode-dev-prompt.md`](./channel-mode-dev-prompt.md)。
> 本文记录**已落地的实现、运维与上线状态**。

## 1. 它是什么

`relay-claude` 的一种接入模式：**一个 `session_id` 对应一个常驻 `claude --print --input-format stream-json` 进程**，跨多轮对话从 stdin 持续喂消息，消除每轮的「spawn + session 重载 + MCP/skill 重新初始化」开销。

通过启动参数 `--mode=channel` 开启；`--mode=legacy`（默认）行为与改动前**字节级一致，零回归**。对上游 wuji_tools 的 OpenAI HTTP/SSE 协议**零改动**。

```
--mode=legacy   (默认)  每请求 fork 一次 `claude -p`，出完即退（现状）
--mode=channel          每 session_id 一个常驻 stream-json 进程，多轮复用
```

## 2. 启动参数

| flag | 默认 | 说明 |
|------|------|------|
| `--mode` | `legacy` | `legacy` / `channel` |
| `--idle-ttl` | `30m` | channel：进程空闲超过此时长被回收 |
| `--max-channels` | `50` | channel：最大常驻进程数，超出 evict 最旧 |

```bash
# channel 模式启动示例
claude_openai_api --port=50008 --mode=channel --idle-ttl=20m --max-channels=40
```

## 3. 分流规则（哪些请求走 channel）

只有 **`stream=true` + 有 `session_id` + 无 `tools`** 的请求走 channel（正是 wuji_tools 的请求形态）。其余一律降级 legacy，保证完全兼容：

- `session_id` 为空 → legacy（§4.2 退化）
- 非流式 → legacy（`handleNonStreamResponse`）
- 带 `tools` → legacy（`handleBufferedStreamResponse`）

判定见 `isChannelEligible()`（`cmd/relay-claude/main.go`）。

## 4. 架构

| 文件 | 作用 |
|------|------|
| `translate.go` | `sseTranslator`：stream-json 事件 → OpenAI SSE chunk 翻译（legacy 与 channel **共用**，等价重构） |
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

- **`GET /health`** → `{"mode":"channel"|"legacy", ...}`，可确认当前模式。
- **`GET /channels`** → 当前常驻 worker 列表（session_id、claude_sid、flag、last_used、dead），用于验证进程复用。
- 日志标注：`[channel] spawned/turn start/turn end/interrupt/reaping/capacity evict`。
- **SIGTERM 优雅关闭**：收到 SIGTERM/SIGINT 先 KillGroup 全部 worker 再退出，杜绝常驻 claude 子进程孤儿泄漏（legacy 的每请求子进程会自行退出，channel 的不会）。重启用的 `kill` 即发 SIGTERM。

## 7. 二进制升级注意（chroot 实例）

`/usr/local/bin/claude_openai_api` 在生产 chroot（claude01–06）是**宿主机文件经 ro bind-mount 共享**。升级须：
1. 写**宿主机路径** `/usr/local/bin/claude_openai_api`（chroot 内只读）。
2. 用 **`mv`(rename)** 覆盖，**不能 `cp` 覆盖**正在执行的二进制（会 `ETXTBSY`）：
   ```bash
   sudo cp <new> /usr/local/bin/.cooa.new
   sudo mv -f /usr/local/bin/.cooa.new /usr/local/bin/claude_openai_api
   ```
3. 改它会影响 claude01–06 下次重启（legacy 字节等价，安全）。

**回滚**：50008 重启时去掉 `--mode=channel` 即回 legacy（v1.2.0 legacy == v1.1.4 legacy），或恢复备份 `claude_openai_api.bak.v1.1.4`。

## 8. 上线状态与实测（2026-06-15）

- **claude01（10.0.100.173:50008）已切 channel 模式**（`--mode=channel --idle-ttl=20m --max-channels=40`），健康正常。
- 两种模式在真实环境对比（真实 `ritaoguoji` workdir，加载真实 MCP/skill）：

  | | turn1 冷启动 | turn2 |
  |---|---|---|
  | legacy | 19.2s | 4.96s（每轮重 spawn） |
  | channel | 22.3s | **1.76s（复用，省 ~20.5s）** |

- legacy 真实 opus 流量零回归（token 统计完整）；本地 race + 单测 + 端到端集成全绿。

> **注意**：claude01 的实际流量大多是无状态 cron 巡检任务（无 `session_id`），会正确降级 legacy，故 channel 在该实例收益有限。channel 的大收益只对**多轮对话型 bot** 显著。
