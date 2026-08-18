# SDK / llm-llamacpp bug reports — KV cache

Verified reproductions, re-run 2026-08-18 **inside the qvac monorepo at main
(`1be383088`; SDK 0.17.1 source, addon 0.43.0 installed)** — no consumer-app
code in the chain. Each was also reproduced earlier the same day from a consumer
against the released `@qvac/sdk 0.17.0` → addon 0.39.4 with identical numbers.
Nothing here is from code reading alone. Excluded on purpose: anything
dynamic-tools related (feature retired upstream in qvac#3373 — decision, not a
bug) and everything consumer-side.

How to run — everything is on this branch, in place; the scripts import only
the `@qvac/sdk` public API (no consumer code involved). From `packages/sdk`:

- `bun run examples/repro-prefill-slide-and-prompt-tokens.ts` and
  `bun run examples/repro-reasoning-wipe.ts`. Default model is the registry's
  `QWEN3_600M_INST_Q4` (auto-download); set `QVAC_REPRO_MODEL=/path/to.gguf` to
  use any local `qwen3`-arch gguf instead. Rebuild `dist` first if stale
  (`node_modules/.bin/tsc -p tsconfig.json && bun run postcompile:aliases`).
- `npm run test:bare` — runs `test/bare/runtime/kv-cache-rollback-regression.test.ts`
  (no model, no GPU); it is the suite `pr-checks-sdk-pod.yml` runs on every
  SDK PR. Expected today: only that test fails.

---

## 1. Rollback destroys the previously committed chat cache (SDK)

Any failed turn — including a prefill rejection that never touched the cache,
e.g. a tool result larger than ctx — unlinks the chat's entire `.bin` and
forgets `savedCount`. The next turn re-primes cold.

Root cause: `committed` is tracked per `TurnHandle`, but the `.bin` is per
chat key. Turn N+1's handle has no memory of turn N's commit, so `runRollback`
deletes a file it never created. The discriminator already exists:
`savedCount === 0` at `beginTurn` means the turn primed the cache itself.

- Repro: `repro-rollback-wipe.test.ts` — **run today at qvac main**
  (`1be383088`, fresh install): `tests 152/153` — the only failure is this
  test: `not ok: the committed cache survives a failed turn`,
  `savedCount actual: undefined expected: 3`. The guard case (a self-primed
  cache is still cleaned up) passes, so the fix cannot over-correct.
- Fix shape: on rollback, only unlink what this turn primed; otherwise restore
  the last committed `savedCount` and keep the file.
- Observed in the wild: a 26 MB chat cache vanished after a web_search result
  overflowed a 2048 ctx (desktop, addon 0.36.4).

## 2. Reasoning-span wipe on a generation-time slide (addon)

With reasoning enabled (`reasoning_budget: -1`) and
`remove_thinking_from_context` at its default `true`, any generation long
enough to fill the context hard-fails and wipes the sequence: the decode-time
slide shifts every KV position, `markSpanInvalidatedByGenerationSlide`
**clears** the tracked `<think>` span instead of shifting it, and `compact()`
takes its `FailedKvWiped` branch (`clearSeqOnFailure` + throw).

- Repro: `repro-reasoning-wipe.ts` — **run today in the qvac monorepo at
  main (addon 0.43.0)** on BOTH shipped architectures: Qwen3-4B (`qwen3`,
  failed after 1,489 tokens) and Qwen3.5-2B (`qwen35`, failed after 3,726
  tokens), each with
  `[TextLlm] ReasoningBlockCompactor::compact: generation-time context slide
  invalidated tracked reasoning state (pos=512, discarded=51, seqId=0)`.
- Also hit organically on desktop at addon 0.36.4 (`[MtmdLlm]`, Qwen3.5-2B,
  ctx 2048, "tell me a story") — both context classes affected.
- Trigger note: the wipe fires only when the slide lands **while a think span
  is tracked**, so short-thinking runs sail past it — the repro prompt forces
  very long reasoning to make the trigger deterministic. The bug itself is
  architecture-independent (`ReasoningBlockCompactor` is shared; the reasoning
  family covers qwen3/qwen35/qwen36).
- Fix shape: shift the span start/end by `discarded` when the slide happens
  above the protected prefix; only a discard range that overlaps the span is
  genuinely unrecoverable (and then the overlapping reasoning is already gone).

## 3. Prefill slide is single-step (addon)

`trySlidePrefill` discards `n_discarded` once and gives up — it is not a loop —
so a warm turn overflows even when the incoming slice fits comfortably and two
discard steps would have admitted it.

- Repro: `repro-prefill-slide-and-prompt-tokens.ts` — **run today in the
  qvac monorepo at main (addon 0.43.0)**, identical numbers on Qwen3-4B
  (`qwen3`) and Qwen3.5-2B (`qwen35`), ctx 512, `n_discarded` 51: warm turn
  overflowed at 567 tokens
  (55 over); the identical payload on a fresh key is only **206 tokens** and
  succeeds; 2 discard steps would have admitted it.
- Fix shape: loop the discard while it makes progress and the protected prefix
  allows.

## 4. `promptTokens` on ContextOverflow is ambiguous (addon message + SDK parse)

The two prefill guards format **different quantities into the same parsed
field**: the slice-alone guard reports the slice; the slide-failed guard
reports `nPast + slice`. Nothing on the error says which fired, and
`CompletionStats.promptTokens` is always the slice — same name, three meanings.

- Repro: same script, same run: case B error field **567** vs **206** actual
  slice for the identical payload; case A error field **619** = slice alone.
- Fix shape: report both quantities (attempted total AND slice), or rename.

## 5. Batch-prefill overflow is not reconstructed as ContextOverflowError (SDK)

On the batch-prefill path (`[TextLlm] context overflow at batch prefill step`)
the error reaches the client as a **raw RPC error** — `isQvacError: false`,
`code: undefined`, `instanceof ContextOverflowError` fails. Any consumer doing
typed-error recovery (our context-recovery ladder does exactly this) silently
gets none on text-only models. Multimodal (`[MtmdLlm]`) overflows DO arrive
typed — observed on device — which makes the miss easy to ship.

- Repro: same script, same run: both case A and case B arrived RAW
  (`typedError=false`).
- Fix shape: add the batch-prefill message to the RPC error reconstructor's
  ContextOverflow detection (the non-batch path already matches).

---

Nice-to-have (not a bug): the addon counts slides per inference
(`runtimeStats().contextSlides`) but the SDK does not forward it —
a passthrough would make "did the sliding window fire?" observable without
DEBUG logs.

---

## Known but deliberately NOT reported (so nobody re-finds them)

- **Dynamic-slice defects** — the SDK's dynamic placement ignores the history it
  is given (turn-dropping retries send byte-identical payloads) and, on a cold
  cache, ships only the last exchange (`/* cacheExists */ true`). Both are real
  in the shipped SDK source but only bite when dynamic placement engages; the
  feature was retired upstream (qvac#3373) and we are switching to static, so
  these are not filed. Never run-verified — code reading only.
- **Tools anchor not persisted in cache metadata** — dynamic-only, moot with the
  feature retired. Code reading only.
- **`deleteCache` leaving the addon's in-memory KV stale** — already fixed
  upstream in addon 0.35.1 (qvac#3121); ships in our 0.36.4+.
