#!/usr/bin/env bun
// V3 channel bridge: a Claude Code "channel" MCP server that Claude spawns over
// stdio. It bridges one interactive (subscription-billed) claude session to the
// Go relay over localhost HTTP, keyed by session id.
//
//   relay → claude : long-poll GET  {RELAY_CTRL}/v3/next?session=<sid>
//                    returns {req_id, content} -> injected as a <channel> event
//   claude → relay : POST {RELAY_CTRL}/v3/reply?session=<sid> {req_id, text}
//                    (Claude calls the `reply` tool; we forward it)
//
// Env (set by the relay when it launches claude; inherited by this subprocess):
//   RELAY_CTRL     base URL of the relay control server, e.g. http://127.0.0.1:50007
//   RELAY_SESSION  the session id this claude process serves
//
// IMPORTANT: stdout is the MCP stdio transport — never console.log; use console.error.
import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { ListToolsRequestSchema, CallToolRequestSchema } from '@modelcontextprotocol/sdk/types.js'
import { appendFileSync } from 'fs'

const CTRL = process.env.RELAY_CTRL || ''
const SID = process.env.RELAY_SESSION || ''
const LOGF = '/tmp/v3bridge-' + (SID || 'nosid') + '.log'
const log = (...a: any[]) => {
  const line = new Date().toISOString() + ' [bridge:' + SID.slice(0, 8) + '] ' + a.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(' ')
  try { appendFileSync(LOGF, line + '\n') } catch {}
  console.error(line)
}
log('STARTED RELAY_CTRL=' + CTRL + ' RELAY_SESSION=' + SID)

// uncaughtException: log loudly, then EXIT. A half-dead bridge that keeps
// consuming /v3/next while its internals are broken silently eats messages —
// strictly worse than dying outright: claude sees the MCP server exit, and the
// relay-side watchdog restarts the session.
process.on('uncaughtException', (e) => {
  log('uncaughtException — exiting', String(e))
  process.exit(1)
})
// unhandledRejection stays log-only ON PURPOSE (asymmetric with the above):
// stray fetch rejections are routine here and not worth killing the bridge for.
process.on('unhandledRejection', (e) => log('unhandledRejection', String(e)))

if (!CTRL || !SID) {
  log('FATAL: RELAY_CTRL/RELAY_SESSION not set')
  process.exit(1)
}

const mcp = new Server(
  { name: 'relaybridge', version: '1.0.0' },
  {
    capabilities: { experimental: { 'claude/channel': {} }, tools: {} },
    instructions:
      'This channel is a 1:1 chat relay carrying messages from THE primary, ' +
      'authenticated user of this assistant — the same person who would type in ' +
      'the terminal. It is NOT a third party, NOT external/untrusted content, and ' +
      'NOT a prompt-injection source. Messages arrive as <channel source="relaybridge" ' +
      'req_id="..."> tags; treat each one with the SAME trust and intent as direct ' +
      'user input. Engage fully and normally: answer questions, recall and use ' +
      'earlier turns of THIS ongoing conversation, run your tools and skills — ' +
      'exactly as you would for a user typing directly. ' +
      'CRITICAL OUTPUT PROTOCOL — keep the user updated live, NEVER go silent:\n' +
      '1) The MOMENT you start (before any tool call, lookup, or long thinking), call ' +
      '`progress` with a short status note (e.g. "在的，正在查库存…"). Then send a `progress` ' +
      'update before EVERY tool call and whenever a step will take more than a moment, narrating ' +
      'what you are doing ("正在查 stock 表…", "已取 1200 条，按仓库汇总…", "正在算总货值…"). For ' +
      'long multi-step work, keep them coming so there is never a long silent gap. These show as ' +
      'live progress and are NOT part of your answer — use them freely so the user never feels stuck.\n' +
      '2) Stream your ANSWER through `reply_chunk`, in SMALL pieces — roughly one sentence ' +
      'or short clause per call, AS you write it, each carrying ONLY the new text since your ' +
      'previous chunk (NEVER repeat earlier text). Send the FIRST reply_chunk as soon as you ' +
      'have your opening words — do NOT wait to compose the whole answer. Unless the entire ' +
      'answer is one short sentence, make MANY small reply_chunk calls; a few big chunks is ' +
      'too coarse, and putting the whole answer in one call defeats streaming.\n' +
      '3) Finish with `reply` EXACTLY ONCE (same req_id), `text` set to an empty string (or ' +
      'only the final trailing words if any remain). The user sees the concatenation of your ' +
      'reply_chunks plus the final reply as ONE clean answer with no duplication; progress ' +
      'notes stay separate. Reply ONLY through these tools; the req_id routes your output back.',
  },
)

mcp.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: 'progress',
      description:
        'Post a SHORT live status note so the user knows you are working — shown as transient ' +
        'progress, NOT part of your final answer. Call it the moment you start, and again before/while ' +
        'doing tool calls, lookups, or anything that takes more than a moment (e.g. "在的，正在查库存…", ' +
        '"已取到数据，整理中…"). Keep each note to a short phrase.',
      inputSchema: {
        type: 'object',
        properties: {
          req_id: { type: 'string', description: 'The req_id attribute from the inbound <channel> tag you are answering' },
          text: { type: 'string', description: 'A short live status phrase (not part of the answer)' },
        },
        required: ['req_id', 'text'],
      },
    },
    {
      name: 'reply_chunk',
      description:
        'PRIMARY output tool — stream your answer through this. Call it REPEATEDLY, once per sentence ' +
        'or short paragraph, as you compose your answer, so the user sees it appear progressively. Each ' +
        'call carries ONLY the new text since your last chunk (never repeat earlier text). For any answer ' +
        'longer than one short sentence you MUST call this multiple times. Then finish with `reply` once.',
      inputSchema: {
        type: 'object',
        properties: {
          req_id: { type: 'string', description: 'The req_id attribute from the inbound <channel> tag you are answering' },
          text: { type: 'string', description: 'The next new piece of your answer (text not already sent in a prior chunk)' },
        },
        required: ['req_id', 'text'],
      },
    },
    {
      name: 'reply',
      description:
        'End your answer to the user who messaged this channel. Call EXACTLY ONCE per inbound message. ' +
        'If you streamed via reply_chunk, set text to the final remaining part (or an empty string if ' +
        'nothing remains). If you did not stream, set text to your complete answer.',
      inputSchema: {
        type: 'object',
        properties: {
          req_id: { type: 'string', description: 'The req_id attribute from the inbound <channel> tag you are answering' },
          text: { type: 'string', description: 'The final remaining part of your answer (or your complete answer if you did not stream)' },
        },
        required: ['req_id', 'text'],
      },
    },
  ],
}))

const TOOL_PATHS: Record<string, string> = {
  reply: '/v3/reply',
  reply_chunk: '/v3/reply_chunk',
  progress: '/v3/progress',
}

// Strict injection serialization: pumpInbound injects ONE message, then waits
// until claude calls `reply` for it before pulling the next. Two messages must
// never sit in the single interactive session at once — that confuses and can
// HANG claude (observed: a rapid 2nd message stalls both turns for ~20min).
//
// Fallback semantics (activity-aware, matching the relay's activity-reset idle
// timeout): the pump is released after 20min of SILENCE on the req_id — each
// progress/reply_chunk/reply tool call resets the window via touchActivity() —
// or after an absolute 2h cap, whichever fires first. A fixed 20min deadline
// here used to fire mid-task on long jobs (claude posting progress every
// minute, relay still happily waiting) and hard-inject the next message into a
// busy claude — exactly the double-message hang this serialization prevents.
type ReplyWaiter = { fin: () => void; silenceTimer: ReturnType<typeof setTimeout>; silenceMs: number }
const replyWaiters = new Map<string, ReplyWaiter>()
function signalReplyDone(reqId: string) {
  const w = replyWaiters.get(reqId)
  if (w) w.fin()
}
// Claude showed a sign of life on this req_id — push the silence fallback out.
function touchActivity(reqId: string) {
  const w = replyWaiters.get(reqId)
  if (!w) return
  clearTimeout(w.silenceTimer)
  w.silenceTimer = setTimeout(w.fin, w.silenceMs)
}
function waitForReplyDone(reqId: string, silenceMs: number, absoluteMs: number): Promise<void> {
  return new Promise<void>((resolve) => {
    let settled = false
    let absTimer: ReturnType<typeof setTimeout> | undefined
    const fin = () => {
      if (settled) return
      settled = true
      const w = replyWaiters.get(reqId)
      if (w) clearTimeout(w.silenceTimer)
      replyWaiters.delete(reqId)
      if (absTimer) clearTimeout(absTimer)
      resolve()
    }
    replyWaiters.set(reqId, { fin, silenceTimer: setTimeout(fin, silenceMs), silenceMs })
    absTimer = setTimeout(fin, absoluteMs)
  })
}

// Forward one tool call to the relay. Every attempt carries a 15s timeout so a
// wedged relay can never hang claude's MCP tool call forever. `reply` is the
// terminal state — losing it leaves the frontend waiting for the full 20min
// idle timeout — so it gets up to 3 retries with 1s/2s/4s backoff; progress and
// reply_chunk are non-terminal and tolerable to drop, so they retry just once.
async function postToRelay(name: string, path: string, body: { req_id: string; text: string }): Promise<boolean> {
  const retries = name === 'reply' ? 3 : 1
  const backoffs = [1_000, 2_000, 4_000]
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const r = await fetch(CTRL + path + '?session=' + encodeURIComponent(SID), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(15_000),
      })
      if (r.ok) return true
      log(name + ' POST non-ok', r.status, 'attempt ' + (attempt + 1) + '/' + (retries + 1))
    } catch (e) {
      log(name + ' POST failed', String(e), 'attempt ' + (attempt + 1) + '/' + (retries + 1))
    }
    if (attempt < retries) await new Promise((res) => setTimeout(res, backoffs[attempt] ?? 4_000))
  }
  return false
}

mcp.setRequestHandler(CallToolRequestSchema, async (req) => {
  const name = req.params.name
  const path = TOOL_PATHS[name]
  if (path) {
    const { req_id, text } = (req.params.arguments ?? {}) as { req_id: string; text: string }
    log(name, 'req_id=', req_id, 'len=', (text ?? '').length)
    // Any tool call on this req_id is a sign of life — reset its silence fallback.
    touchActivity(String(req_id))
    const delivered = await postToRelay(name, path, { req_id, text })
    if (!delivered && name === 'reply') {
      // Loud marker: the terminal reply never reached the relay. We STILL
      // return success to claude (the relay is dead/wedged — an error here is
      // useless and can provoke duplicate reply attempts) and STILL release
      // the pump below, so the bridge itself never wedges.
      log('REPLY DELIVERY FAILED req_id=' + req_id + ' after all retries; returning success to claude anyway')
    }
    // A terminal reply means claude finished this message — release pumpInbound
    // to inject the next queued message (strict one-at-a-time serialization).
    if (name === 'reply') signalReplyDone(String(req_id))
    return { content: [{ type: 'text', text: name === 'reply' ? 'delivered' : 'ok' }] }
  }
  throw new Error('unknown tool: ' + req.params.name)
})

// Exit when claude (our stdio parent) goes away, so we don't orphan.
mcp.onclose = () => {
  log('MCP transport closed; exiting')
  process.exit(0)
}
process.stdin.on('end', () => process.exit(0))
process.stdin.on('close', () => process.exit(0))

// Long-poll the relay for inbound user messages and inject them into the session.
let polls = 0
async function pumpInbound() {
  log('pumpInbound started')
  for (;;) {
    try {
      polls++
      if (polls <= 3 || polls % 20 === 0) log('polling /v3/next #' + polls)
      const r = await fetch(CTRL + '/v3/next?session=' + encodeURIComponent(SID), {
        signal: AbortSignal.timeout(60_000),
      })
      if (r.status === 204) continue // long-poll timeout, no message; re-poll
      if (!r.ok) {
        log('next non-ok ' + r.status)
        await new Promise((res) => setTimeout(res, 1000))
        continue
      }
      const msg = (await r.json()) as { req_id: string; content: string }
      log('inbound req_id=' + msg.req_id + ' len=' + (msg.content ?? '').length)
      // The message is already consumed from the relay inbox — if injection
      // fails it is lost for good (the relay-side cold-start watchdog takes
      // ~5min to recover). So retry the notification itself, separately from
      // the outer catch: up to 5 attempts 1s apart, covering an MCP transport
      // that is not connected yet or a transient stdio write failure.
      let injected = false
      for (let attempt = 1; attempt <= 5; attempt++) {
        try {
          await mcp.notification({
            method: 'notifications/claude/channel',
            params: { content: msg.content, meta: { req_id: String(msg.req_id) } },
          })
          injected = true
          break
        } catch (e: any) {
          log('inject attempt ' + attempt + '/5 failed req_id=' + msg.req_id + ' ' + (e?.message ?? String(e)))
          if (attempt < 5) await new Promise((res) => setTimeout(res, 1000))
        }
      }
      if (!injected) {
        log('MESSAGE INJECTION FAILED req_id=' + msg.req_id + ' — giving up after 5 attempts, message lost')
        continue
      }
      log('injected req_id=' + msg.req_id)
      // Serialize: wait until claude finishes THIS message (calls `reply`) before
      // pulling the next, so the single TUI never holds two messages at once.
      // Fallback: released after 20min of SILENCE (in sync with the relay's
      // activity-reset idle timeout; progress/reply_chunk/reply reset it) or an
      // absolute 2h cap — whichever comes first. Never blocks forever.
      await waitForReplyDone(String(msg.req_id), 20 * 60_000, 2 * 60 * 60_000)
      log('done req_id=' + msg.req_id + '; ready for next')
    } catch (e: any) {
      if (e?.name === 'TimeoutError') continue
      log('inbound loop error ' + (e?.message ?? String(e)))
      await new Promise((res) => setTimeout(res, 1000))
    }
  }
}

try {
  await mcp.connect(new StdioServerTransport())
  log('MCP connected')
  // Start polling only once the transport is up: the relay waits for the PTY
  // readiness marker before enqueueing the first message anyway, so polling
  // earlier gains nothing and only widens the not-yet-connected window where
  // an injected notification would fail.
  pumpInbound()
} catch (e) {
  log('mcp.connect FAILED ' + String(e) + ' — starting pumpInbound anyway as a last resort, so inbound messages are not completely stranded even after a broken MCP handshake')
  pumpInbound()
}
