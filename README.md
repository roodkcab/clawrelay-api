# clawrelay-api

**English** | [中文](#中文)

---

> A lightweight Go server that turns **Claude Code CLI** into an **OpenAI-compatible API** — just point any client at it and get full Claude Code power (file editing, Bash, web search, MCP, etc.) over HTTP/SSE.

```
┌──────────────────────────┐
│        Clients           │
│                          │
│  WeCom Bot               │
│  Feishu Bot              │       ┌──────────────┐       ┌──────────────┐
│  ClawRelay Desktop/iOS   │──────▶│ clawrelay-api│──────▶│  claude CLI  │
│  Any OpenAI client       │ HTTP  │   (:50009)   │ spawn │              │
│  curl                    │  SSE  │              │       │  Anthropic   │
└──────────────────────────┘       └──────────────┘       └──────────────┘
```

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
go build -o clawrelay-api .
./clawrelay-api
```

Output:

```
Starting Claude OpenAI-compatible API server on :50009
```

**Command-line flags:**

| Flag | Default | Description |
|---|---|---|
| `--port` | `50009` | Port to listen on |
| `--model` | `claude-sonnet-4-6` | Default model name (e.g. `MiniMax-M2.7`, `glm-5.1`) |
| `--proxy` | — | HTTP/HTTPS proxy URL (e.g. `http://127.0.0.1:7890`) |

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
├── claude_openai_api.go   Main server: types, handlers, stream translation, CLI launcher
├── session_store.go       Session persistence, WebSocket viewer, HTML UI
├── go.mod                 Module definition (Go 1.24, gorilla/websocket)
└── go.sum                 Dependency checksums
```

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

> 一个轻量级 Go 服务，将 **Claude Code CLI** 转化为 **OpenAI 兼容的 API** — 任何客户端直接对接即可获得完整的 Claude Code 能力（文件编辑、Bash、网络搜索、MCP 等）。

```
┌──────────────────────────┐
│          客户端           │
│                          │
│  企业微信机器人            │
│  飞书机器人               │       ┌──────────────┐       ┌──────────────┐
│  ClawRelay 桌面端/iOS     │──────▶│ clawrelay-api│──────▶│  claude CLI  │
│  任意 OpenAI 客户端       │ HTTP  │   (:50009)   │ 子进程│              │
│  curl                    │  SSE  │              │       │  Anthropic   │
└──────────────────────────┘       └──────────────┘       └──────────────┘
```

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
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) 已安装且在 PATH 中
  ```bash
  npm install -g @anthropic-ai/claude-code
  ```
- Claude Code 已配置有效的 Anthropic API Key

### 编译运行

```bash
git clone https://github.com/roodkcab/clawrelay-api.git
cd clawrelay-api
go build -o clawrelay-api .
./clawrelay-api
```

输出：

```
Starting Claude OpenAI-compatible API server on :50009
```

**命令行参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--port` | `50009` | 监听端口 |
| `--model` | `claude-sonnet-4-6` | 默认模型名称（如 `MiniMax-M2.7`、`glm-5.1`） |
| `--proxy` | — | HTTP/HTTPS 代理地址（如 `http://127.0.0.1:7890`） |

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
├── claude_openai_api.go   主服务：类型定义、请求处理、流式转换、CLI 启动器
├── session_store.go       会话持久化、WebSocket 查看器、HTML 界面
├── go.mod                 模块定义（Go 1.24, gorilla/websocket）
└── go.sum                 依赖校验
```

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
