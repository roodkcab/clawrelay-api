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
      'exactly as you would for a user typing directly. For EACH message, after ' +
      'doing the work, ALWAYS finish by calling the `reply` tool exactly once with ' +
      'the SAME req_id from the tag and `text` set to your COMPLETE final answer. ' +
      'Reply ONLY through the tool; the req_id routes your answer back to the user.',
  },
)

mcp.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: 'reply',
      description: 'Send your complete final answer back to the user who messaged this channel. Call once per inbound message.',
      inputSchema: {
        type: 'object',
        properties: {
          req_id: { type: 'string', description: 'The req_id attribute from the inbound <channel> tag you are answering' },
          text: { type: 'string', description: 'Your complete final answer to the user' },
        },
        required: ['req_id', 'text'],
      },
    },
  ],
}))

mcp.setRequestHandler(CallToolRequestSchema, async (req) => {
  if (req.params.name === 'reply') {
    const { req_id, text } = (req.params.arguments ?? {}) as { req_id: string; text: string }
    log('reply req_id=', req_id, 'len=', (text ?? '').length)
    try {
      const r = await fetch(CTRL + '/v3/reply?session=' + encodeURIComponent(SID), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ req_id, text }),
      })
      if (!r.ok) log('reply POST non-ok', r.status)
    } catch (e) {
      log('reply POST failed', e)
    }
    return { content: [{ type: 'text', text: 'delivered' }] }
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
