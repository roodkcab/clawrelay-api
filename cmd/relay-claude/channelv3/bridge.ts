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

process.on('uncaughtException', (e) => log('uncaughtException', String(e)))
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
      '`progress` with a short status note (e.g. "在的，正在查库存…"), and call it again ' +
      'with brief updates as you work ("已取到数据，整理中…"). These show as live progress ' +
      'and are NOT part of your answer — use them freely so the user never waits in silence.\n' +
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

mcp.setRequestHandler(CallToolRequestSchema, async (req) => {
  const name = req.params.name
  const path = TOOL_PATHS[name]
  if (path) {
    const { req_id, text } = (req.params.arguments ?? {}) as { req_id: string; text: string }
    log(name, 'req_id=', req_id, 'len=', (text ?? '').length)
    try {
      const r = await fetch(CTRL + path + '?session=' + encodeURIComponent(SID), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ req_id, text }),
      })
      if (!r.ok) log(name + ' POST non-ok', r.status)
    } catch (e) {
      log(name + ' POST failed', e)
    }
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
      await mcp.notification({
        method: 'notifications/claude/channel',
        params: { content: msg.content, meta: { req_id: String(msg.req_id) } },
      })
      log('injected req_id=' + msg.req_id)
    } catch (e: any) {
      if (e?.name === 'TimeoutError') continue
      log('inbound loop error ' + (e?.message ?? String(e)))
      await new Promise((res) => setTimeout(res, 1000))
    }
  }
}

// Start polling immediately (does not depend on the MCP handshake completing).
pumpInbound()

try {
  await mcp.connect(new StdioServerTransport())
  log('MCP connected')
} catch (e) {
  log('mcp.connect FAILED ' + String(e))
}
