/**
 * Repro: cancelling an in-flight completion wipes the chat's KV cache, so the
 * NEXT prompt on an over-`ctx_size` conversation re-prefills the full history in
 * one batch and overflows — even though the new prompt is tiny and the sliding
 * window (`n_discarded`) would normally evict old context.
 *
 * Why this happens (server side):
 *   On abort, the deferred `session.rollback(turn)` runs `runRollback` in
 *   ops/kv-cache-session.ts, which does NOT just undo the aborted turn — it
 *   wipes the whole chat cache:
 *       await fsPromises.unlink(state.cachePath);     // deletes the .bin
 *       initializedCaches.delete(state.registryKey);  // forgets it's primed
 *       cachedMessageCounts.delete(state.cachePath);  // deletes savedCount
 *   So after a cancel, savedCount is gone. The next completion can't slice
 *   (decideCachedHistorySlice falls back to the full system-stripped history),
 *   and the prefill guard fires first:
 *       if (nTokens >= n_ctx || nPositions >= n_ctx) throw <context overflow>;
 *   ...before trySlidePrefill ever runs. Same end-state as a cold cache after
 *   an app restart, but triggered by cancel (a far more common action).
 *
 * The A/B below runs the same tiny "Say hi." turn before and after a cancelled
 * turn on one monotonically-growing history; the only difference is the cancel:
 *   Phase A: tiny turn            -> succeeds (warm cache slices + slides)
 *   Phase B: long turn, cancelled mid-stream
 *   Phase C: tiny turn again      -> context overflow  <-- the bug
 *
 * Note: on the batch-prefill path the overflow currently surfaces as a raw
 * RPCError ("[TextLlm] context overflow at batch prefill step: prompt tokens N,
 * max context tokens M"), not the typed ContextOverflowError — the RPC
 * reconstructor has no entry for it. `isOverflow` below matches both.
 *
 * Expected (correct) behavior: rollback should restore savedCount to the last
 * *committed* boundary (and keep the .bin) instead of wiping the cache, so
 * Phase C reuses the warm prefix and only prefills "Say hi.". Failing that, a
 * cold/full prefill should slide down to the most recent n_ctx-worth instead of
 * throwing, so overflow only happens when a single message exceeds ctx.
 *
 * Run:  bun run examples/kv-cache-cancel-overflow-repro.ts
 */

import {
  cancel,
  completion,
  type CompletionRun,
  ContextOverflowError,
  deleteCache,
  InferenceCancelledError,
  loadModel,
  QWEN3_600M_INST_Q4,
  unloadModel,
} from "@qvac/sdk";

const CACHE_KEY = "cancel-overflow-repro";
const CTX_SIZE = 1024;

// A large first user turn so the conversation exceeds ctx_size almost
// immediately — that's the precondition for the cold/full re-prefill to
// overflow. ~3.6k chars ≈ well over the 1024-token window on its own.
const BIG_FILLER = "The quick brown fox jumps over the lazy dog. ".repeat(80);

type Msg = { role: string; content: string };

function approxTokens(msgs: Msg[]): number {
  return Math.round(msgs.reduce((n, m) => n + m.content.length, 0) / 4);
}

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

// The overflow surfaces as a typed ContextOverflowError on some paths and a raw
// RPCError carrying the addon message on the batch-prefill path; match both.
function isOverflow(err: unknown): boolean {
  return err instanceof ContextOverflowError || /context overflow/i.test(errMessage(err));
}

async function drain(
  run: CompletionRun,
): Promise<{ text: string; cacheTokens: number | undefined }> {
  let text = "";
  for await (const tok of run.tokenStream) {
    text += tok;
  }
  const stats = await run.stats;
  return { text, cacheTokens: stats?.cacheTokens };
}

let reproduced = false;

try {
  const modelId = await loadModel({
    modelSrc: QWEN3_600M_INST_Q4,
    modelConfig: {
      ctx_size: CTX_SIZE,
      // Sliding window enabled, exactly like the downstream (Workbench) config.
      n_discarded: Math.floor(CTX_SIZE / 10),
    },
  });

  // ---- Build an over-ctx conversation, warm. History GROWS monotonically and
  // is never snapshotted/shrunk, so savedCount stays <= history.length and the
  // warm path keeps slicing + sliding (the precondition cache reuse needs). ----
  const history: Msg[] = [
    { role: "system", content: "You are a terse assistant. Answer in one short sentence." },
    { role: "user", content: `Summarize this in one word: ${BIG_FILLER}` },
  ];

  console.log("▸ Phase 0: build an over-ctx conversation (warm, sliding)\n");
  for (let turn = 1; turn <= 3; turn++) {
    const run = completion({ modelId, history, stream: true, kvCache: CACHE_KEY });
    const { text, cacheTokens } = await drain(run);
    history.push({ role: "assistant", content: text.trim() || "ok" });
    history.push({ role: "user", content: `Reply "ok" (turn ${turn}).` });
    console.log(`▸   turn ${turn}: ok, cacheTokens=${cacheTokens} (ctx=${CTX_SIZE})`);
  }
  console.log(
    `▸   full history is now ${history.length} messages, ~${approxTokens(history)} tokens >> ctx ${CTX_SIZE}\n`,
  );

  // ---- Phase A (control): a tiny turn succeeds while the cache is warm. ----
  console.log('▸ Phase A: a tiny "Say hi." turn on the WARM cache');
  history.push({ role: "user", content: "Say hi." });
  const warm = completion({ modelId, history, stream: true, kvCache: CACHE_KEY });
  const warmRes = await drain(warm);
  history.push({ role: "assistant", content: warmRes.text.trim() || "hi" });
  console.log(`▸   -> OK (cacheTokens=${warmRes.cacheTokens}); the slide handled it.\n`);

  // ---- Phase B: cancel an in-flight completion on the SAME chat. The aborted
  // turn's rollback wipes the chat cache (unlink .bin + delete savedCount). ----
  console.log("▸ Phase B: start a long completion and cancel it mid-stream");
  history.push({ role: "user", content: "Write a very long, detailed essay." });
  const victim = completion({ modelId, history, stream: true, kvCache: CACHE_KEY });
  let cancelled = false;
  let victimStop: string | undefined;
  for await (const event of victim.events) {
    if (event.type === "contentDelta" && !cancelled) {
      cancelled = true;
      await cancel({ requestId: victim.requestId });
      console.log("▸   cancel issued after first token");
    }
    if (event.type === "completionDone") {
      victimStop = event.stopReason;
    }
  }
  // Settle the aborted run so its deferred rollback has run before Phase C.
  await victim.text.catch((err) => {
    if (!(err instanceof InferenceCancelledError)) throw err;
  });
  console.log(`▸   cancelled turn settled (stopReason=${victimStop}); its rollback has run`);
  // Mirror the app: a fully-cancelled turn is hidden from model history.
  history.pop();
  console.log("");

  // ---- Phase C: the next tiny turn now overflows — the cancel wiped the warm
  // cache, so the full >ctx history is re-prefilled cold and the prefill guard
  // throws before the slide can run. Same shape as Phase A, opposite result. ----
  console.log('▸ Phase C: another tiny "Say hi." turn — same shape as Phase A');
  history.push({ role: "user", content: "Say hi again." });
  try {
    const after = completion({ modelId, history, stream: true, kvCache: CACHE_KEY });
    const afterRes = await drain(after);
    console.log(
      `▸   -> OK (cacheTokens=${afterRes.cacheTokens}). Cache survived the cancel — NOT reproduced.\n`,
    );
  } catch (err) {
    if (isOverflow(err)) {
      reproduced = true;
      console.log("▸   -> context overflow on a tiny turn that worked warm in Phase A.");
      console.log(`▸      ${errMessage(err)}\n`);
    } else {
      throw err;
    }
  }

  await deleteCache({ all: true });
  await unloadModel({ modelId, clearStorage: false });
} catch (error) {
  console.error("✖ unexpected error:", error);
  process.exit(1);
}

console.log(
  reproduced
    ? "✅ BUG REPRODUCED: a mid-stream cancel wiped the chat KV cache, so the next tiny prompt re-prefilled the full >ctx history and overflowed (the slide never ran)."
    : "❎ Not reproduced: the cache survived the cancel and Phase C succeeded.",
);
process.exit(reproduced ? 0 : 1);
