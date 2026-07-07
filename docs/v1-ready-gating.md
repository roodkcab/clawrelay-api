# relay-claude V1「ready 门控」修复（2.2.0）

> 针对生产两条高频用户可见故障的根治：下游企微机器人（wuji-tools）频繁展示
> 「AI 已完成处理，但未生成文本回复」与「抱歉，AI 连接出现错误」。
> 2026-07-07 根因调查结论：两大主因都在 V1 的 ready 门控设计上，且 2.1.0（PR29）未覆盖。

## 根因

V1 路径（`--mode=v1`，每请求 fork `claude`）存在两个由同一设计（`<-ready` 门控）派生的缺陷：

1. **首字节前零字节窗口** → 下游 120s 读超时（「AI 连接出现错误」）
   `handleStreamResponse` 把 HTTP 200 头、`: ping`、30s keepalive 全部扣到
   `<-ready`（= claude 首行 stdout）之后。CLI 挂死（坏二进制劫持 PATH、
   auto-updater 卡死、代理黑洞）或 resume 嗅探重跑期间，下游连响应头都收不到，
   wuji-tools 的 aiohttp `sock_read=120` 在 120s 整把连接掐断。
   生产实证：近 30 天 error 最大类全是 `Timeout on reading data from socket`，
   latency 精确聚在 120.0~120.1s。V2 在 `channel_handler.go` 已独立修过同一问题，
   V1 一直没有。

2. **resume 嗅探丢弃错误 result + 盲重试空 200** →「未生成文本回复」
   wuji-tools 对所有会话都传 `session_id`，V1 全走 `--resume` 嗅探。旧逻辑：
   首个非 init/system 事件是 result（= CLI 只产出错误 result：API error/限流/
   登录失效）就把第一次运行的全部输出**连错误文本和 usage 一起丢弃**，盲目换
   `--session-id` 重跑；二次零输出仍 `ready <- nil`，返回一条延迟 15~120s 的
   **空 200 事件流**（0 事件、0 usage、直接 [DONE]）。下游 0 个 TextDelta →
   兜底文案，且以 status='success' 落库。该路径恰好绕过了 PR29 在
   `translate.go` 加的错误透出。
   生产实证：K8s 48h 内 38 条空回复 33 条同签名、38/38 无 Token 用量行；
   空回复率 7-04 起激增 8 倍且 7-06 升 2.1.0 后不降。

## 修复内容（2.2.0）

### stream.go
- `sseHandshake`：请求受理即发 200 头 + `: ping`（不再等 ready），V1 两个流式
  handler 共用；无 flusher 时仍走标准 500（头未发出）并后台杀进程。
- `waitClaudeReady`：pre-ready 期间跑 30s keepalive、响应客户端断连；启动失败
  以流内 `⚠️` 错误块 + finish + [DONE] 收尾（复用 `channelEmitErrClose`）。
- buffered（tools）路径补上错误 result 透出（镜像 translate.go），修复被转发的
  错误 result 在该路径被吞的问题。

### process.go
- **首行看门狗**：`launchClaude` 内置，CLI 静默 `firstLineTimeout`（默认 90s，
  `RELAY_FIRST_LINE_TIMEOUT_SECS` 可覆盖，必须 < 下游 sock_read=120s）无任何
  stdout 即 KillGroup；stdout EOF 时在 `cmd.Wait()` 收割前解除武装（防 PID
  复用误杀）。
- **嗅探看门狗**：resume 嗅探整体限时（同 `firstLineTimeout`）——init 行会先于
  API 调用到达并解除首行看门狗，其后挂死（代理死/CLI 卡重试）由嗅探看门狗兜底，
  否则提前发头 + keepalive 会把故障从 120s 掩盖到上游总超时（600s+）。
- **resume 嗅探重构**：全程 buffer 不丢弃；只有 stderr 出现
  `No conversation found with session ID`（`sessErrCh`）才降级 `--session-id`
  重试；会话存在且有 result → 原样转发（错误透出 + usage 记账生效）；任何零输出
  路径 → `ready <- error`（经 `waitClaudeReady` 变成用户可见的 ⚠️ 块），不再有
  空 200。零事件但 stdout 有非 JSON 文本时，错误信息附带最后一行输出。
- **procHandle**：互斥保护的进程句柄替换裸 `**exec.Cmd`——修复 pre-ready 断连
  读与生产者写的 data race，以及「断连后 retry 进程成为无人可杀的孤儿跑满整轮」
  的 TOCTOU（`publish()` 在 abort 后返回 false，生产者自杀新进程）。
- stderr scanner 加 8MB 行上限（与 stdout 一致）：超长 stderr 行不再吞掉
  missing-session 标记、不再让子进程阻塞在 stderr 写上挂死嗅探。
- `runClaude`（非流式）的 resume 盲重试同样加 `sessionMissing` 门控。

### 协议扩展
- 错误块（`channelEmitErrClose` 与错误 result 透出）携带顶层
  `"x_relay_error": true` 字段。OpenAI 客户端忽略未知字段；wuji-tools 后续可据此
  把此类轮次按 error 落库（当前会按普通文本展示 + status='success'，是已知的
  下游配套项）。

## 下游配套（wuji-tools，待办）
- adapter/orchestrator 识别 `x_relay_error`，按 error 落库并本地化文案；
- 零事件流（text=thinking=tool=0 且无 usage）按 error 处理；
- `error_message` 记录异常类型名（bare TimeoutError 的 str() 为空无法归因）。

## 行为变化注意
- 启动类失败（fork 失败、二进制缺失、零输出崩溃、看门狗击杀）从 **HTTP 500**
  变为 **200 + 流内 ⚠️ 错误块**：用户能看到具体原因，但下游在未适配
  `x_relay_error` 前会把这类轮次记为 success（V2 自 2.1.0 起已是此行为）。
- 会话存在但 CLI 只产出错误 result 时**不再自动重跑**——错误原样透出，用户自行
  重试；只有会话真缺失才重建重试（行为与修复前的「偶尔自愈」不同，但消除了
  重复执行带副作用 prompt 的风险）。

## 测试
`cmd/relay-claude/process_test.go` + `stream_test.go` 共 16 个用例，覆盖：
错误 result 透传不重试、session 缺失才重试、零输出各路径报错、首行/嗅探双看门狗、
pre-ready 断连杀进程（-race 验证）、断连不孤儿化 retry、超长 stderr 行、
错误文案带最后输出、提前发头时序、错误块含 x_relay_error、buffered 路径错误透出。
