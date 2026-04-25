# clawrelay-api

**English** | [中文](#中文)

---

> A pair of lightweight Go relays that turn **Claude Code CLI** and **OpenAI Codex CLI** into **OpenAI-compatible HTTP APIs** — any OpenAI client (curl, ClawRelay desktop/iOS, WeCom/Feishu bots, etc.) can talk to either via the same SSE protocol, while each relay internally exploits its CLI's native features.

```
                                ┌──────────────────┐       ┌──────────────┐
                          ┌────▶│ relay-claude     │──────▶│  claude CLI  │
┌──────────────────────┐  │     │   (:50009)       │ spawn │  → Anthropic │
│      Clients         │  │     └──────────────────┘       └──────────────┘
│                      │  │
│  WeCom / Feishu Bot  │──┤
│  ClawRelay Desktop   │  │     ┌──────────────────┐       ┌──────────────┐
│  Any OpenAI client   │  └────▶│ relay-codex      │──────▶│  codex CLI   │
└──────────────────────┘ HTTP   │   (:50010)       │ spawn │   → OpenAI   │
                          SSE   └──────────────────┘       └──────────────┘
```

Both relays share the same OpenAI-compatible request/response shape and `/sessions` viewer. They differ in what's behind the curtain:

- **relay-claude** — full Claude Code feature surface: native tool_use, AskUserQuestion, --append-system-prompt, --resume, modelUsage sub-agent token rollup, token-level streaming.
- **relay-codex** — native OpenAI Codex features: thread-based resume (sends only the new user message on follow-up turns, big token savings on long conversations), native `-i FILE` multimodal attachments, `model_reasoning_effort` config override, sandbox/approval modes, command_execution surfaced as tool_calls for UI indicators.

## Clients

clawrelay-api is the shared backend. Pick a client that fits your workflow:

| Client | Platform | Repository |
|---|---|---|
| **ClawRelay** | macOS / Linux / Windows / iOS | [roodkcab/clawrelay](https://github.com/roodkcab/clawrelay) |
| **WeCom Bot** | WeCom (Enterprise WeChat) | [wxkingstar/clawrelay-wecom-server](https://github.com/wxkingstar/clawrelay-wecom-server) |
| **Feishu Bot** | Feishu (Lark) | [wxkingstar/clawrelay-feishu-server](https://github.com/wxkingstar/clawrelay-feishu-server) |
| **Any OpenAI client** | — | Just point `base_url` to `http://host:50009/v1` |

## Why?

Claude Code is a powerful agentic CLI, but it has no HTTP API. This server bridges the gap:

- Spawns a `claude` CLI subprocess per request
- Translates Claude's `stream-json` output into **OpenAI SSE format** in real time
- Supports **session persistence** — conversations resume across requests via `--resume`
- Handles **images & files** — decodes base64 attachments, saves to temp files, passes paths to Claude
- Tracks **token usage** per model
- Provides a built-in **session viewer** (WebSocket + HTML) for debugging
- **Kills the process** when the client disconnects (stop button support)

## Quick Start

### Prerequisites

- [Go 1.21+](https://go.dev/dl/)
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and on PATH
  ```bash
  npm install -g @anthropic-ai/claude-code
  ```
- A valid Anthropic API key configured for Claude Code

### Build & Run

```bash
git clone https://github.com/roodkcab/clawrelay-api.git
cd clawrelay-api

# Build whichever relays you need (both share pkg/openai + pkg/sessions).
go build -o relay-claude ./cmd/relay-claude
go build -o relay-codex  ./cmd/relay-codex

# Default ports: claude on 50009, codex on 50010 — run only what you need.
./relay-claude &
./relay-codex  &
```

**Common flags (apply to both binaries):**

| Flag | relay-claude default | relay-codex default | Description |
|---|---|---|---|
| `--port` | `50009` | `50010` | Port to listen on |
| `--model` | `claude-sonnet-4-6` | `gpt-5.4` | Default model when client omits one |
| `--proxy` | — | — | HTTP/HTTPS proxy URL |
| `--sessions-dir` | `sessions` | `sessions` | Where session logs + attachments live (point both relays at the same dir to share `/sessions` viewer) |
| `--log-file` | `relay-claude.log` | `relay-codex.log` | Log file (use `-` for stdout only) |

> **Use a different model provider:**
> ```bash
> ./clawrelay-api --model MiniMax-M2.7
> ```

> **Behind a proxy?**
> ```bash
> ./clawrelay-api --proxy http://127.0.0.1:7890
> ```

> **Run in background (Linux/macOS):**
> ```bash
> nohup ./clawrelay-api > clawrelay-api.log 2>&1 &
> ```

### Verify it works

```bash
curl http://localhost:50009/health
# {"status":"ok"}

curl http://localhost:50009/v1/models
# {"object":"list","data":[...]}
```

## API Reference

### `POST /v1/chat/completions`

OpenAI-compatible chat completions. Supports both streaming and non-streaming modes.

**Request body:**

```json
{
  "model": "claude-sonnet-4-6",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "working_dir": "/path/to/project",
  "session_id": "abc123",
  "max_turns": 200,
  "env_vars": {"MY_VAR": "value"}
}
```

| Field | Type | Description |
|---|---|---|
| `model` | string | Model name (see available models below) |
| `messages` | array | OpenAI-format messages (system/user/assistant/tool) |
| `stream` | bool | Enable SSE streaming (recommended) |
| `working_dir` | string | Directory where Claude CLI runs (loads `CLAUDE.md`, memory) |
| `session_id` | string | Resume a previous conversation (maps to `--resume`) |
| `max_turns` | int | Max tool-use turns per request (default: 200) |
| `env_vars` | object | Extra environment variables for the Claude process |
| `stream_options` | object | `{"include_usage": true}` to get token counts in stream |
| `tools` | array | OpenAI-format tool definitions (injected into system prompt) |

**Streaming response** (SSE):

```
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"thinking":"Let me think..."},"index":0}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"tool_calls":[{"id":"call_...","function":{"name":"Bash","arguments":"{...}"}}]},"index":0}]}

data: [DONE]
```

**Extended fields** (beyond OpenAI spec):
- `delta.thinking` — Claude's chain-of-thought reasoning text
- `delta.tool_calls[].function.name` = `"AskUserQuestion"` — human-in-the-loop prompt

### `GET /v1/models`

Returns available models.

```json
{
  "object": "list",
  "data": [
    {"id": "vllm/claude-sonnet-4-6", "object": "model", "owned_by": "anthropic"},
    {"id": "vllm/claude-opus-4-6", "object": "model", "owned_by": "anthropic"},
    {"id": "vllm/claude-haiku-4-5-20251001", "object": "model", "owned_by": "anthropic"},
    {"id": "minimax/MiniMax-M2.7", "object": "model", "owned_by": "minimax"},
    {"id": "minimax/MiniMax-M2.5", "object": "model", "owned_by": "minimax"},
    {"id": "kimi/kimi-k2.5", "object": "model", "owned_by": "moonshot"},
    {"id": "zhipu/glm-5.1", "object": "model", "owned_by": "zhipu"}
  ]
}
```

**Model aliases** (for OpenAI client compatibility):

| OpenAI model name | Maps to |
|---|---|
| `gpt-4` | `opus` |
| `gpt-4o` / `gpt-4-turbo` | `sonnet` |
| `gpt-3.5-turbo` / `gpt-4o-mini` | `haiku` |

### `GET /v1/stats`

Token usage statistics since server start.

```json
{
  "total_requests": 42,
  "input_tokens": 15000,
  "output_tokens": 8000,
  "total_tokens": 23000,
  "per_model": { "sonnet": { "requests": 30, "input_tokens": 12000, "output_tokens": 6000 } },
  "start_time": "2025-03-09T10:00:00Z",
  "uptime": "2h30m"
}
```

### `GET /health`

Returns `200 OK` with `{"status": "ok"}`.

### `GET /sessions`

Lists all sessions with their log file sizes and last modified times.

### `GET /session/{id}`

Opens a built-in HTML session viewer with real-time WebSocket streaming. Useful for debugging — shows requests, response deltas, tool calls, and token usage in a terminal-style UI.

WebSocket endpoint: `ws://host:50009/session/{id}/ws`

## How It Works

1. **Request arrives** at `/v1/chat/completions`
2. Messages are converted from OpenAI format to a prompt string
3. Images/files are decoded from base64 data URIs → saved to `/tmp/` → injected as `[Image: path]`
4. A `claude` CLI subprocess is spawned with flags:
   - `--model <model>` — selected model
   - `--system-prompt <prompt>` — system message
   - `--output-format stream-json` — machine-readable streaming output
   - `--resume <session_id>` — resume previous conversation (with auto-retry as `--session-id` on failure)
   - `--permission-mode bypassPermissions` — non-interactive mode
   - `--max-turns <n>` — limit autonomous tool-use loops
5. Claude's JSON stream is parsed line-by-line and translated to OpenAI SSE chunks
6. Tool calls from Claude's native `tool_use` content blocks are converted to OpenAI `tool_calls` format
7. On client disconnect, the Claude process is killed immediately
8. Temp files are cleaned up after the response completes

### Session Persistence

Sessions are stored as JSONL files in the `sessions/` directory. Each event (request, response delta, tool use, completion) is logged with a timestamp. The WebSocket viewer replays history and streams new events in real time.

## Project Structure

```
clawrelay-api/
├── pkg/
│   ├── openai/         OpenAI-compatible types, SSE/CORS helpers, lifetime token stats
│   ├── sessions/       Append-only session store + WebSocket-streamed HTML viewer
│   └── attachments/    base64 image/file decoder with content-hash dedup
├── cmd/
│   ├── relay-claude/   Claude Code CLI driver: stream-json parsing, native tool_use,
│   │                   AskUserQuestion handling, --resume retry, modelUsage rollup
│   └── relay-codex/    Codex CLI driver: native thread_id resume (token savings),
│                       -i FILE attachments, reasoning.effort, sandbox/approval modes
├── go.mod              Single Go module (1.24, gorilla/websocket)
└── go.sum
```

Both binaries link the same shared packages, so behavioral changes to session storage / OpenAI types only need a single edit.

## Configuration

Use command-line flags to customize the server at startup:

```bash
./clawrelay-api --port 8080 --model MiniMax-M2.7 --proxy http://127.0.0.1:7890
```

| Setting | Flag | Default |
|---|---|---|
| Port | `--port` | `50009` |
| Default model | `--model` | `claude-sonnet-4-6` |
| HTTP proxy | `--proxy` | — |
| Session log directory | — | `sessions/` |

The model name supports any provider — Claude, MiniMax, Kimi, GLM, etc. The client can also override the model per-request via the `model` field in the request body.

## License

[MIT](LICENSE) — Copyright (c) 2025 roodkcab

---

---

# 中文

**[English](#clawrelay-api)** | 中文

---

> 两个轻量级 Go 中继服务，分别将 **Claude Code CLI** 和 **OpenAI Codex CLI** 转化为 **OpenAI 兼容 HTTP API**。两个服务对外协议风格一致（同一套 OpenAI SSE），对内各自榨干所属 CLI 的原生特性。

```
                                ┌──────────────────┐       ┌──────────────┐
                          ┌────▶│ relay-claude     │──────▶│  claude CLI  │
┌──────────────────────┐  │     │   (:50009)       │ 子进程│   → Anthropic│
│        客户端         │  │     └──────────────────┘       └──────────────┘
│                      │  │
│  企业微信 / 飞书机器人  │──┤
│  ClawRelay 桌面端     │  │     ┌──────────────────┐       ┌──────────────┐
│  任意 OpenAI 客户端   │  └────▶│ relay-codex      │──────▶│  codex CLI   │
└──────────────────────┘ HTTP   │   (:50010)       │ 子进程│   → OpenAI   │
                          SSE   └──────────────────┘       └──────────────┘
```

两个服务共享同一套 OpenAI 兼容请求/响应结构和 `/sessions` 查看器。差异在底层：

- **relay-claude** — 完整保留 Claude Code 能力面：原生 tool_use、AskUserQuestion、--append-system-prompt、--resume、modelUsage 子代理 token 汇总、token 级流式。
- **relay-codex** — 原生 OpenAI Codex 特性：基于 thread_id 的 resume（后续轮次只发送最新 user message，长会话下大幅节省 token）、原生 `-i FILE` 多模态附件、`model_reasoning_effort` 配置透传、精确的 sandbox/approval 模式映射、command_execution 通过 tool_calls 上抛供 UI 显示进度。

## 客户端

clawrelay-api 是共享后端，选择适合你的客户端：

| 客户端 | 平台 | 仓库 |
|---|---|---|
| **ClawRelay** | macOS / Linux / Windows / iOS | [roodkcab/clawrelay](https://github.com/roodkcab/clawrelay) |
| **企业微信机器人** | 企业微信 | [wxkingstar/clawrelay-wecom-server](https://github.com/wxkingstar/clawrelay-wecom-server) |
| **飞书机器人** | 飞书 | [wxkingstar/clawrelay-feishu-server](https://github.com/wxkingstar/clawrelay-feishu-server) |
| **任意 OpenAI 客户端** | — | 将 `base_url` 指向 `http://host:50009/v1` 即可 |

## 为什么需要它？

Claude Code 是强大的智能体 CLI，但没有 HTTP API。本服务补上了这个缺口：

- 每个请求启动一个 `claude` CLI 子进程
- 将 Claude 的 `stream-json` 输出实时转换为 **OpenAI SSE 格式**
- 支持**会话持久化** — 通过 `--resume` 跨请求恢复对话
- 处理**图片和文件** — 解码 base64 附件，保存为临时文件，传递路径给 Claude
- 按模型追踪 **Token 用量**
- 内置**会话查看器**（WebSocket + HTML），方便调试
- 客户端断开时**立即终止进程**（支持停止按钮）

## 快速上手

### 环境要求

- [Go 1.21+](https://go.dev/dl/)
- 任选其一或全装：
  - [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)：`npm install -g @anthropic-ai/claude-code`（用于 relay-claude）
  - [OpenAI Codex CLI](https://github.com/openai/codex)：`brew install codex` 或参见上游文档（用于 relay-codex）

### 编译运行

```bash
git clone https://github.com/roodkcab/clawrelay-api.git
cd clawrelay-api

# 按需编译；两个 binary 都会复用 pkg/openai + pkg/sessions 共享代码。
go build -o relay-claude ./cmd/relay-claude
go build -o relay-codex  ./cmd/relay-codex

# 默认端口：claude 50009，codex 50010；只跑你需要的。
./relay-claude &
./relay-codex  &
```

**通用参数（两个 binary 都支持）：**

| 参数 | relay-claude 默认 | relay-codex 默认 | 说明 |
|---|---|---|---|
| `--port` | `50009` | `50010` | 监听端口 |
| `--model` | `claude-sonnet-4-6` | `gpt-5.4` | 客户端未指定时的默认模型 |
| `--proxy` | — | — | HTTP/HTTPS 代理地址 |
| `--sessions-dir` | `sessions` | `sessions` | 会话日志和附件目录（两个服务指向同一目录可共用 `/sessions` 查看器） |
| `--log-file` | `relay-claude.log` | `relay-codex.log` | 日志文件路径（设为 `-` 仅输出 stdout） |

> **使用其他模型：**
> ```bash
> ./clawrelay-api --model MiniMax-M2.7
> ```

> **需要代理？**
> ```bash
> ./clawrelay-api --proxy http://127.0.0.1:7890
> ```

> **后台运行（Linux/macOS）：**
> ```bash
> nohup ./clawrelay-api > clawrelay-api.log 2>&1 &
> ```

### 验证运行状态

```bash
curl http://localhost:50009/health
# {"status":"ok"}

curl http://localhost:50009/v1/models
# {"object":"list","data":[...]}
```

## API 参考

### `POST /v1/chat/completions`

OpenAI 兼容的聊天补全接口，支持流式和非流式模式。

**请求体：**

```json
{
  "model": "claude-sonnet-4-6",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "working_dir": "/path/to/project",
  "session_id": "abc123",
  "max_turns": 200,
  "env_vars": {"MY_VAR": "value"}
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | string | 模型名称（见下方可用模型列表） |
| `messages` | array | OpenAI 格式的消息（system/user/assistant/tool） |
| `stream` | bool | 启用 SSE 流式输出（推荐） |
| `working_dir` | string | Claude CLI 的运行目录（会加载 `CLAUDE.md`、记忆文件） |
| `session_id` | string | 恢复之前的对话（对应 `--resume`） |
| `max_turns` | int | 每次请求最大工具调用轮次（默认：200） |
| `env_vars` | object | 传给 Claude 进程的额外环境变量 |
| `stream_options` | object | `{"include_usage": true}` 在流中返回 Token 统计 |
| `tools` | array | OpenAI 格式的工具定义（注入到系统提示词中） |

**流式响应**（SSE）：

```
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"thinking":"Let me think..."},"index":0}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"tool_calls":[{"id":"call_...","function":{"name":"Bash","arguments":"{...}"}}]},"index":0}]}

data: [DONE]
```

**扩展字段**（OpenAI 规范之外）：
- `delta.thinking` — Claude 的思维链推理文本
- `delta.tool_calls[].function.name` = `"AskUserQuestion"` — 人机交互提示

### `GET /v1/models`

返回可用模型列表。

**模型别名**（兼容 OpenAI 客户端）：

| OpenAI 模型名 | 映射为 |
|---|---|
| `gpt-4` | `opus` |
| `gpt-4o` / `gpt-4-turbo` | `sonnet` |
| `gpt-3.5-turbo` / `gpt-4o-mini` | `haiku` |

### `GET /v1/stats`

返回服务启动以来的 Token 用量统计。

### `GET /health`

返回 `200 OK`，`{"status": "ok"}`。

### `GET /sessions`

列出所有会话及其日志文件大小和最后修改时间。

### `GET /session/{id}`

打开内置的 HTML 会话查看器，通过 WebSocket 实时展示请求、响应、工具调用和 Token 用量，方便调试。

WebSocket 端点：`ws://host:50009/session/{id}/ws`

## 工作原理

1. 请求到达 `/v1/chat/completions`
2. 消息从 OpenAI 格式转换为提示词字符串
3. 图片/文件从 base64 data URI 解码 → 保存到 `/tmp/` → 以 `[Image: path]` 注入提示词
4. 启动 `claude` CLI 子进程，参数包括：
   - `--model <model>` — 指定模型
   - `--system-prompt <prompt>` — 系统消息
   - `--output-format stream-json` — 机器可读的流式输出
   - `--resume <session_id>` — 恢复之前的对话（失败时自动切换为 `--session-id` 重试）
   - `--permission-mode bypassPermissions` — 非交互模式
   - `--max-turns <n>` — 限制自主工具调用轮次
5. 逐行解析 Claude 的 JSON 流，实时转换为 OpenAI SSE 格式
6. Claude 原生 `tool_use` 内容块转换为 OpenAI `tool_calls` 格式
7. 客户端断开连接时，立即终止 Claude 进程
8. 响应完成后清理临时文件

### 会话持久化

会话以 JSONL 文件存储在 `sessions/` 目录。每个事件（请求、响应增量、工具调用、完成）都带有时间戳记录。WebSocket 查看器会重放历史记录并实时推送新事件。

## 项目结构

```
clawrelay-api/
├── pkg/
│   ├── openai/         OpenAI 兼容类型、SSE/CORS 工具、token 用量统计
│   ├── sessions/       会话存储 + WebSocket 查看器 + HTML 界面
│   └── attachments/    base64 图片/文件解码 + 内容哈希去重
├── cmd/
│   ├── relay-claude/   Claude Code CLI 驱动：stream-json 解析、原生 tool_use、
│   │                   AskUserQuestion 处理、--resume 重试、modelUsage 子代理汇总
│   └── relay-codex/    Codex CLI 驱动：原生 thread_id resume（节省 token）、
│                       原生 -i FILE 多模态附件、reasoning.effort、sandbox/approval 模式
├── go.mod              单 Go module（1.24, gorilla/websocket）
└── go.sum
```

两个 binary 共用同一份共享包，会话存储 / OpenAI 类型等共性逻辑只需改一处即可。

## 配置

通过命令行参数自定义服务：

```bash
./clawrelay-api --port 8080 --model MiniMax-M2.7 --proxy http://127.0.0.1:7890
```

| 配置项 | 参数 | 默认值 |
|---|---|---|
| 端口 | `--port` | `50009` |
| 默认模型 | `--model` | `claude-sonnet-4-6` |
| HTTP 代理 | `--proxy` | — |
| 会话日志目录 | — | `sessions/` |

模型名称支持任意 provider — Claude、MiniMax、Kimi、GLM 等。客户端也可以通过请求体的 `model` 字段按请求覆盖默认模型。

## 许可证

[MIT](LICENSE) — Copyright (c) 2025 roodkcab
