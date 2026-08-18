// Run from packages/sdk:  bun run examples/repro-prefill-slide-and-prompt-tokens.ts
// Repro for two llm-llamacpp/SDK issues, runnable against the installed SDK:
//
//   bun run kvcache-upstream/repro-prefill-slide-and-prompt-tokens.ts
//
// 1. SINGLE-STEP PREFILL SLIDE: trySlidePrefill discards n_discarded once and
//    throws if that one step is not enough — a prompt that fits comfortably in
//    ctx overflows on a warm cache even though 2+ discard steps would admit it.
// 2. promptTokens AMBIGUITY: ContextOverflowError.promptTokens is parsed from
//    the addon message; the slice-alone guard reports the slice, the
//    slide-failed guard reports nPast+slice. Same field, different quantity,
//    nothing says which fired. CompletionStats.promptTokens is always the slice.
import { loadModel, completion, deleteCache, unloadModel, ContextOverflowError , QWEN3_600M_INST_Q4 } from '@qvac/sdk'

// Default: the registry's Qwen3-0.6B (arch `qwen3`, reasoning family). Set
// QVAC_REPRO_MODEL=/path/to/model.gguf to use a local file instead.
const LOCAL = process.env['QVAC_REPRO_MODEL']
const CTX = 512
const FILLER = 'The quick brown fox jumps over the lazy dog near the quiet river bank. '

type Msg = { role: string; content: string }
const sys: Msg = { role: 'system', content: 'You are terse. Reply with the single word: ok' }
const user = (reps: number, tag: string): Msg => ({
  role: 'user',
  content: `${FILLER.repeat(reps)} Reply ok. (${tag})`
})

async function run(history: Msg[], kvCache: string) {
  const r = completion({
    modelId,
    history,
    stream: true,
    kvCache,
    generationParams: { reasoning_budget: 0 }
  })
  let text = ''
  for await (const t of r.tokenStream) text += t
  const stats = await r.stats
  return { text: text.trim(), stats }
}

// The batch-prefill overflow surfaces as a RAW RPC error (isQvacError: false),
// not the typed ContextOverflowError — the reconstructor has no entry for that
// path. Reportable on its own: typed-error consumers (instanceof checks) never
// see these. Match both shapes and parse the numbers out of the message.
const errInfo = (e: unknown) => {
  const message = e instanceof Error ? e.message : String(e)
  if (!/context overflow/i.test(message)) return null
  const typed = e instanceof ContextOverflowError
  const long = message.match(/prompt tokens (\d+),\s*max context tokens (\d+)/i)
  const short = message.match(/\((\d+) tokens,\s*max\s*(\d+)\)/i)
  const m = long ?? short
  return {
    typed,
    promptTokens: m ? Number(m[1]) : undefined,
    ctxSize: m ? Number(m[2]) : undefined,
    message: message.split('\n')[0]?.trim()
  }
}

const modelId = await loadModel({
  modelSrc: LOCAL ?? QWEN3_600M_INST_Q4,
  modelType: 'llm',
  modelConfig: { ctx_size: CTX, n_discarded: Math.floor(CTX / 10), verbosity: 0 }
})
console.log(`▸ loaded ${LOCAL?.split('/').pop() ?? 'QWEN3_600M_INST_Q4'} ctx=${CTX} n_discarded=${Math.floor(CTX / 10)}`)
await deleteCache({ kvCacheKey: 'p7-warm' }).catch(() => {})
await deleteCache({ kvCacheKey: 'p7-fresh' }).catch(() => {})
await deleteCache({ kvCacheKey: 'p7-fresh-a' }).catch(() => {})

// ---- warm the cache close to ctx ----
const h1: Msg[] = [sys, user(22, 'turn1')]
const t1 = await run(h1, 'p7-warm')
console.log(`▸ turn1 ok: cacheTokens=${t1.stats?.cacheTokens} promptTokens=${t1.stats?.promptTokens}`)

// ---- the same-size follow-up on the warm cache: expect case-B overflow ----
const h2: Msg[] = [...h1, { role: 'assistant', content: t1.text || 'ok' }, user(12, 'turn2')]
let caseB: ReturnType<typeof errInfo> | null = null
try {
  const t2 = await run(h2, 'p7-warm')
  console.log(`▸ turn2 unexpectedly OK (cacheTokens=${t2.stats?.cacheTokens}) — NOT reproduced`)
} catch (e) {
  caseB = errInfo(e)
  if (!caseB) throw e
  console.log(
    `▸ turn2 CASE B overflow: promptTokens=${caseB.promptTokens} ctx=${caseB.ctxSize} typedError=${caseB.typed}`
  )
}

// ---- identical last-exchange payload on a fresh key: fits → slice size ----
const t3 = await run([sys, h2[2]!, h2[3]!], 'p7-fresh')
console.log(`▸ same tail on fresh key OK: stats.promptTokens=${t3.stats?.promptTokens} cacheTokens=${t3.stats?.cacheTokens}`)

// ---- case A: one prompt alone above ctx on a fresh key ----
let caseA: ReturnType<typeof errInfo> | null = null
try {
  await run([sys, user(40, 'huge')], 'p7-fresh-a')
  console.log('▸ huge prompt unexpectedly OK — NOT reproduced')
} catch (e) {
  caseA = errInfo(e)
  if (!caseA) throw e
  console.log(
    `▸ CASE A overflow: promptTokens=${caseA.promptTokens} ctx=${caseA.ctxSize} typedError=${caseA.typed}`
  )
}

console.log('\n=== verdicts ===')
if (caseB && t3.stats?.promptTokens) {
  const slice = t3.stats.promptTokens
  const overflowBy = (caseB.promptTokens ?? 0) - CTX
  console.log(
    `single-step slide: warm turn overflowed at ${caseB.promptTokens} (ctx ${CTX}); ` +
      `the slice alone is ${slice} tokens and fits — ${overflowBy} tokens over, ` +
      `${Math.ceil(overflowBy / Math.floor(CTX / 10))} discard steps would have admitted it: ` +
      `${overflowBy > 0 && slice < CTX ? 'REPRODUCED' : 'not reproduced'}`
  )
  console.log(
    `promptTokens ambiguity: error field=${caseB.promptTokens} vs stats for the identical ` +
      `payload=${slice} (case B is nPast+slice)${caseA ? `; case A error field=${caseA.promptTokens} (slice alone)` : ''}: ` +
      `${caseB.promptTokens !== slice ? 'REPRODUCED' : 'not reproduced'}`
  )
  console.log(
    `typed-error contract: batch-prefill overflow arrived as ` +
      `${caseB.typed ? 'typed ContextOverflowError' : 'a RAW rpc error (instanceof fails)'}` +
      `${caseA ? `; case A: ${caseA.typed ? 'typed' : 'RAW'}` : ''}`
  )
}
await deleteCache({ kvCacheKey: 'p7-warm' }).catch(() => {})
await deleteCache({ kvCacheKey: 'p7-fresh' }).catch(() => {})
await deleteCache({ kvCacheKey: 'p7-fresh-a' }).catch(() => {})
await unloadModel({ modelId, clearStorage: false })
process.exit(0)
