# relay-claude V3 — 交互式（订阅计费）Claude via Channels

> **发布版本：relay-claude v2.0.0**（channelv3 即 V3 的正式发布形态；早期 interim 构建曾以 1.2.0 运行）。运行在 channelv3 的实例 `/health` 返回 `"version":"2.0.0"` + `"mode":"channelv3"`。
>
> **当前生产 scope（2026-06-16 起，长期混合拓扑）**：channelv3 现仅用于 **claude01(50008) / claude02(50009)** 两个试点实例；其余订阅版 claude（claude03-05、claude07-09）已运行在更简单、已验证的 **legacy** 模式（`/health` 返回 `"mode":"legacy"`）。背景：V3 当初是为解决 headless `-p` 模式不能走订阅计费；2026-06-16 Anthropic 公告**恢复了 `-p` 模式的订阅计费**，故大多数订阅实例稳定回到 legacy，仅保留 claude01/02 作为 V3 试点。8 台订阅实例**共享同一份 relay-claude 2.0.0 二进制**，混版靠**启动标志**区分（见 §6.3），不是不同二进制。本文档作为 channelv3 的权威设计参考保留。

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
| 逐 token 流式 | ✅ | ❌ 无 token 级；但支持**段级渐进式流式**（`reply_chunk`，见 §5） |
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

## 5. 段级渐进式流式（reply_chunk，2026-06-15）

V3 做不到"真·逐 token"流式：交互 TUI 无结构化输出口，token 增量被 `claude --help` 明确绑死在 `--print` + `--output-format=stream-json`，而那是非交互、丢订阅计费的路。折中：bridge 增设 `reply_chunk` 工具，claude 边写边按句/段多次调用，relay 每收到一段就 flush 一个 OpenAI SSE delta；最后 `reply` 空串收尾。**对外 OpenAI SSE 接口不变**（就是变成多个 content delta，和 V2 token 流式同一条消费路径）。

- **协议**：三个 channel 工具 →三个端点：`progress`→`/v3/progress`（**thinking delta**，实时进度，不进答案）、`reply_chunk`→`/v3/reply_chunk`（content delta，答案分段）、`reply`→`/v3/reply`（终态，结束 turn）。waiter 用 `chan v3ReplyEvent{text,final,thinking}`；buffered `ch` **永不 close**，sender 在 `deliver()` 里用 `done` 兜底（沿用 V2 防 send-on-closed 不变量）。progress 走 `delta.thinking`（空 content），wuji_tools adapter `if thinking:` 解析为 ThinkingDelta、显示在思考块，不污染 `accumulated_text`（答案）。
- **效果**（claude01 实测，opus）：4 点结构化答案 → 4 段流式，首字 23.7s → **14.7s**；暖 session follow-up 首字 **4.6s**、3 段；短答案不过度切分（1 段）。bridge 日志可见 `N×reply_chunk + 1×reply(len=0)`。
- **liveness 心跳（防"感觉死掉了"）**：交互式 claude 在重任务（多次工具调用、大查询）期间会**静默数分钟**，且 wuji_tools adapter 的 `sock_read=120` 要求 relay 不能 120s 无字节。原来 relay 发的是 `: keepalive` SSE 注释（**企微不可见**）。现改为：请求一进来**立刻**发 `🤔 正在处理你的请求…`（thinking），之后 claude 静默 ≥12s 时每 15s 发一个**可见**的 `⏳ 处理中…（已 Ns）`（thinking，带秒数会动），有真实进度/答案就重置。这条心跳**独立于 claude**，是"卡死感"的兜底。配合指令让 claude 每次工具调用前都 `progress` 报一句。实测 opus 跑「今天总货值」类查询要 **7~8 分钟**（真在硬查 DB），靠心跳维持"在动"。
- **取舍**：段级 ≠ token 级；每段一次工具往返，**总时长略增**；**强依赖 bridge `instructions` 的强制措辞**——弱措辞下 opus 完全不分段（实测一次性 `reply`）。设计了**优雅降级**：不配合时退回单段（= 原 V3 整段行为），永不出错。
- **关键坑**：channel server 的 `instructions` 必须强制口吻（"MUST stream / 多点答案=多次 reply_chunk / 单段是错的"），否则 opus 默认一次性 `reply`，`reply_chunk` 形同虚设。

## 6. channelv3 试点部署 + 文件夹信任坑（重要）

> **拓扑（2026-06-16 起，长期混合）**：channelv3 仅用于 **claude01(50008) / claude02(50009)** 两个试点实例（均 chroot）。其余订阅版 claude——claude03(50010)/claude04(50011)/claude05(50012)（chroot）+ claude07(50014)/claude08(50015)/claude09(50016)（nspawn）——运行 **legacy** 模式（裸 `--port=<p>`，`claude -p` headless，新公告后仍计入订阅，`/health` 返回 `"mode":"legacy"`）。**排除** claude06=minimax(50013) v1.1.4 legacy（API key 计费，V3 订阅计费无意义、交互+channels 未必兼容 MiniMax 后端）、claude10/11=codex(50017/50018)（不同二进制 relay-codex）、QA(50009 2.0.0 legacy)。
>
> 下面 §6.1–§6.5 的前置条件、信任坑、启动命令均针对 **channelv3 实例（claude01/02）**；legacy 实例不需要 bridge/bun/folder-trust，启动只用裸 `--port=`（见 §6.3）。

### 6.1 前置条件（每 channelv3 实例）
- **二进制**：`/usr/local/bin/claude_openai_api` 是宿主机文件，**全实例共享**（chroot ro bind-mount；nspawn 映射到 `/home/<u>/.local/bin`）。更新一次即全实例可用（重启生效）。
- **bridge**：`/data/relay-v3/bridge/`（`bridge.ts` + `node_modules`）。`/data` 是**同一块 `data--vg-data` 设备**，bind 进每个 chroot/nspawn → bridge **全实例共享**，建一次全有。
- **bun**（`~/.local/bin/bun`）+ **claude ≥ v2.1.80**（channels）+ **订阅登录**（`.claude.json` 含 `oauthAccount`）——逐实例确认；nspawn 要 `nsenter` 进容器查（宿主机静态路径看不到容器内 `/home`、`/data` 的运行时挂载）。

### 6.2 ⚠️ 文件夹信任坑（最大的坑，必读）
V3 冷启动 claude 会弹「Is this a project you created or one you trust? / Yes, I trust this folder」**信任窗**。`--dangerously-skip-permissions` **不**跳过它。
- **症状**：未信任的目录 → drivePTY 旧正则 `do you trust|trust the files` 匹配不到这措辞 → 不自动确认 → **卡死 Phase 1**（`/health` 正常但每个请求 20min 后才 ⚠️）。即使 drivePTY 答了信任，**信任弹窗打断冷启动还会让 claude 把答案打在 TUI 里、不调 reply 工具 → Phase 2 也卡**。claude01 一直好，只因它的 bot 目录早被信任（`.claude.json` 的 `projects[/data/skills/<bot>].hasTrustDialogAccepted=true`）。
- **解法（两层，缺一不可）**：
  1. **drivePTY 正则放宽**（已加 `trust this folder|trust this project|created or one you trust`）——兜底任何残留弹窗。
  2. **预信任**（干净解，复刻 claude01）：每实例 `.claude.json` 把 `/data/skills/*` 全部设 `hasTrustDialogAccepted=true`。脚本 `/data/relay-v3/trust_edit.py`（编辑 `~/.claude.json`、glob `/data/skills/*`、key=`/data/skills/<bot>`）：
     - chroot：宿主机 `sudo python3` 改 `/data_ssd/<u>/home/<u>/.claude.json`，或 `chroot ... su - <u> -c 'python3 /data/relay-v3/trust_edit.py'`
     - nspawn：`nsenter -t <Leader> ... su - <u> -c 'python3 /data/relay-v3/trust_edit.py'`
  - **改 `.claude.json` 前先停 relay+会话**（避免运行中的 claude 回写覆盖）。预信任后冷启动干净、reply 正常、秒级返回（实测 claude02/kan、claude07/franky 13~16s）。
- **➡️ 新建 V3 实例 / 把 bot 迁到新实例，必须对其 working_dir 预信任**，否则首访卡死。

### 6.3 启动命令

**channelv3 实例（claude01/02，chroot）** —— 必须带 `--mode=channelv3` 全套 flag（裸 `--port=` 会静默回退 legacy）：
- chroot：`sudo chroot /data_ssd/<u> /bin/bash` → `su - <u> -c 'V3_DEBUG_PTY=1 nohup /usr/local/bin/claude_openai_api --port=<p> --mode=channelv3 --v3-bridge-dir=/data/relay-v3/bridge --idle-ttl=15m --max-channels=20 >> ~/claude_openai_api.log 2>&1 &'`
  - （`V3_DEBUG_PTY=1` 前缀可选，把每会话 TUI 落盘便于排查。）
- 重启前置检查 + 验证 `/health` 返回 `"mode":"channelv3"`，失败即止。

**legacy 实例（claude03-05 chroot / claude07-09 nspawn）** —— 裸 `--port=<p>`，无 channelv3 flag、无需 bridge/bun/folder-trust：
- chroot：`sudo chroot /data_ssd/<u> /bin/bash` → `su - <u> -c 'nohup /usr/local/bin/claude_openai_api --port=<p> >> ~/claude_openai_api.log 2>&1 &'`
- nspawn：`nsenter -t <Leader> -m -u -i -n -p -- su - <u> -c 'nohup /home/<u>/.local/bin/claude_openai_api --port <p> >> ~/claude_openai_api.log 2>&1 &'`
- 重启后验证 `/health` 返回 `"mode":"legacy"`。

### 6.4 `.mcp.json` 权限
bridge 往 bot working_dir 写 `.mcp.json`。`/data/skills/*` 多为 owner 各异但 **group=`claude`、组可写** → 跨实例可写。唯一会失败：已存在一个**非组可写的旧 `.mcp.json`**（如 claude01 早期 umask 建的）挡覆盖——窄边界，按需 chmod g+w 或删掉重建。

### 6.5 冷启动 watchdog（已实现）

预信任解决"信任类"冷启动卡死，但仍有**偶发的非信任卡死**（实测一次 claude 冷启动冻在 "Checking for updates"，旧版要 20min 才被 idle 兜底）。已加 **cold-start watchdog**（`ColdStartTimeout`，默认 **5min**，env `V3_COLD_START_TIMEOUT` 覆盖）：每个 turn 必须在该窗口内产出**首个输出**（覆盖 channel 一直 not-ready 和 ready 后冻住两种）；超时即判冷启动冻死 → **kill 会话 + drainInbox + relaunch + 重试同一 turn**（同 reqID/waiter/sid，新 bridge 重新注入同一 waiter），重试仍冻死才 `⚠️ 会话启动多次无响应，请稍后重发。`。一旦有任何输出即 **committed**，之后交给 idle 超时（`ReplyTotal`），长任务不受影响。实现：`handleChannelV3Response` 的 Phase1+Phase2 抽成 `runAttempt()` 闭包返回 done/coldHang/clientGone，外层重试。逻辑随 2.0.0 二进制全实例携带，仅在 channelv3 路径（claude01/02）生效（重试上限 `maxColdRetries=1`）。

### 6.6 仍未做（已知遗留）
- **超时"✅ 任务已完成"重复推送** 的 cosmetic（wuji_tools 侧后台推送 + 前台完成各推一次，导致 ⚠️ 被两条 ✅ 夹住、自相矛盾）。
