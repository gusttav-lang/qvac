// Run from packages/sdk:  bun run examples/repro-reasoning-wipe.ts
// Repro: a generation-time context slide while a <think> span is open makes
// ReasoningBlockCompactor hard-fail the request and WIPE the sequence
// (Outcome::FailedKvWiped) — so any generation long enough to fill the context
// kills the chat whenever reasoning is enabled.
//
//   bun run kvcache-upstream/repro-reasoning-wipe.ts
//
// Expected failure: "ReasoningBlockCompactor::compact: generation-time context
// slide invalidated tracked reasoning state (pos=..., discarded=..., seqId=0)".
// Fix shape: shift the tracked span by `discarded` on a slide (the way the old
// tools anchor did) instead of clearing it and failing the turn.
import { loadModel, completion, deleteCache, unloadModel , QWEN3_600M_INST_Q4 } from '@qvac/sdk'

// Default: the registry's Qwen3-0.6B (arch `qwen3`, reasoning family). Set
// QVAC_REPRO_MODEL=/path/to/model.gguf to use a local file instead.
const LOCAL = process.env['QVAC_REPRO_MODEL']
const CTX = 512

// Optional: QVAC_REPRO_MMPROJ=/path/to.mmproj.gguf exercises the multimodal
// (Mtmd) context class — the production path for vision-paired chat models.
const MMPROJ = process.env['QVAC_REPRO_MMPROJ']
const modelId = await loadModel({
  modelSrc: LOCAL ?? QWEN3_600M_INST_Q4,
  modelType: 'llm',
  modelConfig: {
    ctx_size: CTX,
    n_discarded: Math.floor(CTX / 10),
    verbosity: 0,
    ...(MMPROJ ? { projectionModelSrc: MMPROJ } : {})
  }
})
console.log(`▸ loaded ${LOCAL?.split('/').pop() ?? 'QWEN3_600M_INST_Q4'} ctx=${CTX} n_discarded=${Math.floor(CTX / 10)} (reasoning ON)`)
await deleteCache({ kvCacheKey: 'p4-repro' }).catch(() => {})

const run = completion({
  modelId,
  history: [
    { role: 'system', content: 'You are a meticulous long-form storyteller.' },
    {
      role: 'user',
      content:
        'Before answering, think through AT LEAST fifty numbered plot ideas in ' +
        'your reasoning, elaborating each in several sentences — your thinking ' +
        'must be extremely long. Then write an extremely long, detailed story ' +
        'about a dragon. Never stop early.'
    }
  ],
  stream: true,
  kvCache: 'p4-repro',
  generationParams: { reasoning_budget: -1 }
})

let tokens = 0
try {
  for await (const _ of run.tokenStream) tokens++
  const stats = await run.stats
  console.log(`▸ completed without failure (${tokens} tokens, cacheTokens=${stats?.cacheTokens}) — NOT reproduced`)
} catch (e) {
  const msg = e instanceof Error ? e.message : String(e)
  const hit = /ReasoningBlockCompactor.*invalidated tracked reasoning state/s.test(msg)
  console.log(`▸ failed after ${tokens} streamed tokens`)
  console.log(`▸ ${msg.split('\n')[0]}`)
  console.log(`\n=== verdict ===\nreasoning-span wipe on generation slide: ${hit ? 'REPRODUCED' : `different failure — inspect: ${msg.slice(0, 200)}`}`)
}
await deleteCache({ kvCacheKey: 'p4-repro' }).catch(() => {})
await unloadModel({ modelId, clearStorage: false })
process.exit(0)
