# ClawRelay channel 模式 token usage 统计修复设计

日期: 2026-06-24
范围: `cmd/relay-claude/`（relay-claude 二进制）
影响实例: channel(V2) = claude02/50009；channelv3(V3) = claude01/50008
不改动: wuji_tools、数据库 schema、OpenAI API 签名

## 1. 问题与根因

下游 `robot_chat_logs` 的 token 统计在两种 channel 模式下都不对，已用生产数据确认。

### V2 (channel) — token 累计重复计数

`claude --input-format stream-json --output-format stream-json` 是按 session_id 复用的常驻进程。其 `result` 事件的 usage 是「该进程从启动到现在的累计值」，不是单轮增量。`translate.go` 的 `sseTranslator.feed()` 在 `result` 处理里调用 `effectiveUsage()` 后**直接透传**（无跨轮差分），每个 turn 把累计值 emit 成 OpenAI usage chunk，下游每轮入库一条 → 同一 session 的 token 被反复累加，`SUM()` 严重虚高。

生产证据: 同一 session 连续多轮 `input_tokens` 几乎固定、`output_tokens`/`cache_read_tokens` 单调递增（累计快照）。V1 每次 spawn 新进程，累计与单轮重合，所以正常。

### V3 (channelv3) — usage 全 0

`channelv3.go` 的 `v3EmitClose` 主动 emit 空 `Usage{}`（注释: "Channels don't surface token counts; report zeros so the shape holds"）。`bridge.ts` 是 MCP 工具处理器（只回传 `{req_id, text}`），interactive claude 跑在 PTY、不走 `--output-format stream-json`，PTY 输出被丢弃。**token 数据从源头不存在**。问题不是「拿到没传」，是整条链路没有这个数。下游因此把每条 V3 记录的 token 写成 `0`，被报表误当「零消耗」。

## 2. 设计：usageMeter 接口

引入一个抽象，把「raw usage 快照 → 单轮 usage」的策略按 mode 收敛到一个 seam（Codex 第二意见建议，已采纳）:

```
usageMeter: 输入本轮 raw usage 快照 → 输出本轮应上报的 usage（或「无」）
  - V1 (stream.go)   : identity meter — 透传（每次新进程，raw 本就是单轮）
  - V2 (channel)     : cumulative-diff meter — 状态 owned by chanWorker
  - V3 (channelv3)   : nil/unknown meter — 不 emit usage chunk
```

差分必须发生在 **emit SSE usage chunk 之前**，并同样作用于 `stats.Record` 与 `LogDone`。只在 `LogDone` 做差分是错的——下游入库的是 emit 出去的 SSE chunk，SSE 不修正等于没修。

## 3. V2: cumulative-diff meter

### 状态落点
在 `channel.go` 的 `chanWorker` struct 上挂 baseline（它的生命周期 == 常驻进程生命周期）:
- `lastInput / lastOutput / lastCacheRead / lastCacheCreation`（4 个 token 累计基线）
- `lastCostUSD`（cost 累计基线，见 §3.3）
- `lastSourceShape`（baseline 对应的 usage 形态: `usage` 裸值 / `modelUsage` 聚合，见 §3.2）
- mutex（保护以上字段，防 turn 与 reaper goroutine 并发）

### 差分逻辑（translate.go，`effectiveUsage()` 之后、emit 之前）
```
delta = cur - worker.baseline      // 逐字段
worker.baseline = cur
emit/Record/LogDone 用 delta
```

### 3.1 累计回退（auto-compact / clear）
进程内 auto-compact（已设 `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=50`）或 `/clear` 会让累计中途变小。用 `input_tokens` 的单调性做重置信号: `cur.input < last.input` 判为重置，该轮 `delta = cur`（不为负）。

### 3.2 source-shape 一致性（Codex 补强，必须做）
`effectiveUsage()` 在「裸 `usage`」与「`modelUsage` 聚合」之间切换。若 baseline 与本轮 diff 的形态不同（一轮有 modelUsage、下一轮没有），逐字段 diff 会产生假负数或虚高 delta，**直接破坏总量准确性**。

处理: baseline 记录它对应的 `sourceShape`。本轮 shape 与 baseline shape 不一致时，视为不可比，按重置处理（该轮 `delta = cur`），并 log 一条 source 切换。

### 3.3 cost 也 diff（已确认纳入）
`stats.Record(... event.TotalCostUSD)` 的 cost 若也是累计，会同样 double-count。一并 diff（`lastCostUSD`）。注意: `robot_chat_logs` 不存 cost，cost double-count 只影响 relay 的 `/v1/stats` 内存计数器，不污染 DB；修它是让 `/v1/stats` 准，零额外风险。

### 3.4 中断 / 无-result turn（已确认: 接受归属偏移）
turn 在 `result` 前出错/中断，或 result 不带 usage → **baseline 不推进**。被放弃的用量会被下一个成功 turn 的 delta 吸收。

关键不变量: 只要 claude 进程累计单调，`SUM(所有 delta) == 进程最后一次累计 == 真实总量`。**总量恒准**，代价仅是个别 turn 的单条 token 偏大或为 0（归属偏移）。对「修正统计让 SUM 准」的目标完全够用；不 kill worker，保留常驻进程性能。

### 3.5 后台 drain 推进 baseline（Codex 补强，推荐做）
client disconnect / AskUserQuestion 时 `releaseInBackground` 当前不调 `t.feed` 就 drain stdout，丢弃后续 `result`。若 worker 保活而后台不推进 baseline，下一前台 turn 的 delta 会把后台那轮的累计也算进去。

定位: 这不影响总量正确性（§3.4 的不变量仍成立，被丢的累计只是堆到下一个前台 delta），只影响归属程度——与 §3.4「接受归属偏移」同口径。让后台 drainer 解析 `result` 推进 baseline（不 emit SSE、不 LogDone）能让单轮值更合理，但属于优化项，不是总量正确性的前置条件。

### 3.6 gating bug（Codex 补强，顺手修）
当前 emit 条件是 `includeUsage && event.Usage != nil`，但 `t.streamUsage` 可能来自 `modelUsage`。新版 claude 若只发 `modelUsage`，会出现 stats/LogDone 有 usage 而 SSE 没有。改为 gate on `t.streamUsage != nil`。

### 3.7 进程重启
idle-ttl reaper 或 crash 杀掉进程后，`acquire()` 新建 `chanWorker` → baseline 自动归零；新进程累计也从 0 起 → 首轮 `delta = 首轮累计`，自洽。无需额外处理。

## 4. V3: nil meter

`channelv3.go` `v3EmitClose`: 删除 emit 空 `Usage{}` 的分支，改为**不 emit usage chunk**。

效果: 下游 adapter 收不到 usage → `usage_info=None` → `chat_logger` 把 token 字段写 `NULL`（字段本就 nullable）。语义: `NULL` = 未计量，与真实 `0` 区分；`SUM` 自动跳过 NULL、`AVG` 不被 0 拉低。报表/对账层把 NULL 当「未计量」，不计入用量、也不当零消耗。

不在 wuji_tools 侧判断 mode（DB 无 mode 字段，拓扑只在 `clawrelay_topology.sh`）。让 relay 在它确知是 V3 的代码路径表达「无 usage」最干净，单点改动。

## 5. 不做（YAGNI）

- 不强行把 V2/V3 统一到单一 usage 管道（V3 无权威源，强统一只会伪造数据或拖累 V2）。
- 不读 `~/.claude` usage 日志 / OTEL 旁路采集 V3 token（私有格式、race、难归属 req_id）。如需 V3 真实 token，作为**单独实验**评估 OTEL，不在本次范围。
- 不在 `robot_chat_logs` 加显式 flag 字段（NULL 已足够表达「未计量」）。

## 6. 测试

`channel_test.go` 新增用例:
- 连续两个累计递增的 `result` → 断言 emit 的是 delta 而非累计。
- 累计回退（cur < last）→ 断言该轮 delta = cur，不为负。
- source-shape 切换（modelUsage ↔ 裸 usage）→ 断言按重置处理，无假负数。
- result 缺 usage → 断言 baseline 不推进、不 panic。
- 进程重启（新 chanWorker）→ 断言 baseline 归零、首轮 delta = 首轮累计。

## 7. 部署与验证

- 重新编译 relay-claude，部署到 channel(claude02/50009) 与 channelv3(claude01/50008) 实例。
- 验证 V2: 部署后对照 50009 的 `robot_chat_logs`，同 session 内 `input_tokens` 固定时 `cache_read_tokens` 不再单调累计；逐轮值落回合理单轮规模。
- 验证 V3: 50008 的 token 字段写入 `NULL`（而非 0）。

## 8. 开放问题

- claude CLI 在 `--input-format stream-json` 常驻模式下 `result.usage` 是否确为「进程累计」，目前是从代码 + 生产数据反推（Codex 也无权威协议确认）。实现时应加一条 DEBUG 日志打印每轮 raw 累计 vs delta，部署到 claude02 后用真实流量交叉验证，再决定是否长期保留差分。
