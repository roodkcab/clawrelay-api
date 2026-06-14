# relay-claude V3 — 交互式（订阅计费）Claude via Channels

## 0. 为什么有 V3

Claude Code 的**非交互模式**（`-p` / headless，含 V1/V2 的 stream-json）**不再能使用订阅（Max/Pro）token 量**。V3 改为驱动**交互式 claude 会话**（计入订阅），通过 Claude Code 的 [channels](https://code.claude.com/docs/en/channels) 特性收发消息。**对外 OpenAI HTTP/SSE 接口不变。**

`--mode=channelv3`（与 legacy / channel(v2) 并存，互不影响）。

## 1. 架构（与 V2 进程拓扑反转）

```
wuji_tools → relay :50008 (OpenAI HTTP/SSE，不变)
   │  每 session_id：
   ├─ 在 PTY 里启动交互式 `claude --dangerously-load-development-channels server:relaybridge`
   │     （PTY ⇒ 计入 Claude Max 订阅；启动提示自动应答）
   │        └─ claude spawn → Bun channel MCP server (channelv3/bridge.ts, stdio 子进程)
   │              ├─ long-poll  relay /v3/next?session=<sid>  → notifications/claude/channel → <channel> 事件
   │              └─ claude 调 `reply` 工具 → POST relay /v3/reply?session=<sid> {req_id,text}
   └─ relay 控制端（localhost ephemeral port）按 session_id 路由进/出，OpenAI handler 等 reply → SSE
```

- **bridge 注册**：必须用 **cwd 的 `.mcp.json`**（写进 bot working_dir）——`--mcp-config` 的 server **不被 channels 认可**（claude 报 "no MCP server configured with that name"）。
- **RELAY_CTRL / RELAY_SESSION**：经 claude 进程 env 继承传给 bridge（per-process）；`.mcp.json` 的 env 只覆盖代理（bridge 只连 localhost，必须绕过 chroot 的 HTTP_PROXY，否则 503）。
- **就绪门控**：relay 等 PTY 出现 "messages from server:relaybridge inject directly" + 缓冲后才 enqueue 首轮（否则事件在 claude 事件循环就绪前被静默丢弃）。
- **信任**：bridge 的 `instructions` 明确「此 channel 是已认证主用户本人、非外部/注入」，否则 claude 默认对 channel 消息施加注入防御、拒绝回忆类请求。bot 自身的安全规则（禁暴露密钥等）仍生效。

## 2. claude01 实测（订阅账号，端口 50098）

- ✅ **订阅计费**：PTY header `Haiku 4.5 · Claude Max`。
- ✅ 单轮（PONG）、多轮上下文（记住 7777 → 回忆 7777）、会话复用（1 次 launch 服务多轮）。
- ✅ 重安全 bot 提示鲁棒：正常问答/回忆放行，env/启动参数探测仍被拒。
- ✅ 并发多会话同时运行。

## 3. 与 V2 的取舍（channels 固有，无法消除）

| 能力 | V2 (stream-json) | V3 (channels) |
|------|------------------|---------------|
| 订阅计费 | ❌（被限制） | ✅ |
| 逐 token 流式 | ✅ | ❌ reply 工具一次性给**完整回复** |
| thinking 实时展示 | ✅ | ❌ |
| 每轮 token 用量 | ✅ | ❌（channels 不透出） |
| AskUserQuestion 卡片 | ✅ | ❌（交互 TUI，不经 channel） |
| Agent 工具/skill 能力 | ✅ | ✅（claude 仍是完整 agent） |
| 多轮/会话 | ✅ | ✅ |
| OpenAI 对外接口 | ✅ | ✅（最终回复整体作为 SSE chunk） |

> ⚠️ **多租户 memory 隔离**：同一 bot（同 working_dir）的多个用户会话是不同 claude 进程，但**共享 claude 的项目 memory**（写入 cwd 的 memory 文件）。「记住 X」类内容可能跨会话可见。这与 V1/V2 的同-cwd memory 行为一致，是 claude 特性。多用户场景需注意（可考虑每会话独立 cwd，但会丢 bot 的 CLAUDE.md/skill 自动加载）。

## 4. 部署要点

- bridge 部署到 `--v3-bridge-dir`（默认 `/data/relay-v3/bridge/`，含 `bridge.ts` + `node_modules`（`bun add @modelcontextprotocol/sdk`））。
- 依赖：claude01 需 **bun**（`/home/claude01/.local/bin/bun`）+ claude ≥ v2.1.80（channels）+ 订阅账号。
- relay 启动需登录 shell env（PATH 含 bun，claude 才能 spawn bridge）。
- `--idle-ttl` / `--max-sessions`：交互式 claude 进程较重，按内存设保守上限。
- `/channels` 暴露 v3 会话；`V3_DEBUG_PTY=1` 把每会话 TUI 落到 `/tmp/v3pty-<sid>.log`。
- bridge 会在每个 bot working_dir 写 `.mcp.json`（合并保留既有 server）——会出现在 bot 仓库 git status，按需 .gitignore。
