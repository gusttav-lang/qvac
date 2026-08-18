import test from "brittle";

// -----------------------------------------------------------------------------
// Regression guard for the QVAC-Workbench finding "a legitimate tool-output
// ContextOverflow destroys the chat's KV cache".
//
// The failing turn never touches the cache: the addon rejects the prompt at
// prefill (`nTokens >= ctxCeiling`) before anything is decoded. The cache
// committed by the PREVIOUS turn is still exactly as valid as it was a moment
// earlier — but `rollback` unlinks the `.bin` and forgets the saved count, so
// the next turn re-primes and (in dynamic-tools mode) ships only the tail.
//
// Requires no model, no addon and no GPU — same fake-prime approach as
// `kv-cache-session.test.ts`. Runs under Bare via `npm run test:bare`.
// -----------------------------------------------------------------------------

async function loadSession() {
  const fs = await import("bare-fs");
  const os = await import("bare-os");
  const path = await import("bare-path");
  const { default: env } = await import("bare-env");

  const testHome = fs.mkdtempSync(path.join(os.tmpdir(), "qvac-kvcache-rb-"));
  env["HOME"] = testHome;

  const mod = await import(
    "@/server/bare/plugins/llamacpp-completion/ops/kv-cache-session"
  );
  mod.__kvCacheSessionTestHooks.resetForTest();

  function cleanup() {
    try {
      fs.rmSync(testHome, { recursive: true, force: true });
    } catch {
      /* best-effort */
    }
  }

  function writeFakeCache(cachePath: string) {
    fs.mkdirSync(path.dirname(cachePath), { recursive: true });
    fs.writeFileSync(cachePath, "fake-kv-cache-bytes");
  }

  return { fs, mod, cleanup, writeFakeCache };
}

test(
  "kv-cache-session: a failed turn must not destroy the previously committed cache",
  async (t) => {
    const { fs, mod, cleanup, writeFakeCache } = await loadSession();
    try {
      const session = mod.createKvCacheSession("test-model");
      const configHash = mod.generateConfigHash("you are a helpful assistant.", []);
      const primeIfMissing = async (cachePath: string) => {
        writeFakeCache(cachePath);
      };

      // ---- Turn 1: an ordinary successful turn. Commits a real boundary. ----
      const first = await session.beginTurn({
        kind: "custom",
        customKey: "chat-1",
        configHash,
        primeIfMissing,
      });
      await session.commitTurn(
        first,
        // `toolBlockCached` exists only from 0.17.1; the cast keeps this file
        // compiling across SDK versions while staying type-adjacent.
        { kind: "static", messageCount: 3, toolBlockCached: false } as Parameters<
          typeof session.commitTurn
        >[1],
      );
      t.ok(
        fs.existsSync(first.cachePath),
        "precondition: turn 1 left a committed cache on disk",
      );

      // ---- Turn 2: the web_search result is larger than ctx_size, so the
      // addon throws ContextOverflow at prefill. Nothing was decoded and the
      // cache file was never written to, so the handler unwinds via the
      // deferred rollback. ----
      const second = await session.beginTurn({
        kind: "custom",
        customKey: "chat-1",
        configHash,
        primeIfMissing,
      });
      t.is(
        second.savedCount,
        3,
        "precondition: turn 2 opened against turn 1's committed boundary",
      );
      await session.rollback(second);

      // ---- The invariant. Turn 1's cache was valid before turn 2 ran and is
      // just as valid after it failed. ----
      t.ok(
        fs.existsSync(second.cachePath),
        "the committed cache survives a failed turn",
      );
      t.is(
        mod.__kvCacheSessionTestHooks.getSavedCount(second.cachePath),
        3,
        "savedCount falls back to the last committed boundary, not to zero",
      );
      t.ok(
        mod.__kvCacheSessionTestHooks.hasInitializedKey(
          "test-model",
          configHash,
          "chat-1",
        ),
        "the cache stays registered as primed, so the next turn does not re-prime",
      );
    } finally {
      cleanup();
    }
  },
);

test(
  "kv-cache-session: rollback still cleans up a turn that primed the cache itself",
  async (t) => {
    const { fs, mod, cleanup, writeFakeCache } = await loadSession();
    try {
      const session = mod.createKvCacheSession("test-model");
      const configHash = mod.generateConfigHash("sys", []);

      // No prior commit: this turn created the cache, so rolling it back must
      // remove it. This is the behaviour the fix above must NOT regress.
      const only = await session.beginTurn({
        kind: "custom",
        customKey: "chat-2",
        configHash,
        primeIfMissing: async (cachePath: string) => {
          writeFakeCache(cachePath);
        },
      });
      t.is(only.savedCount, 0, "a freshly primed cache has no committed boundary");

      await session.rollback(only);

      t.absent(
        fs.existsSync(only.cachePath),
        "a cache with no committed boundary is removed on rollback",
      );
      t.absent(
        mod.__kvCacheSessionTestHooks.hasInitializedKey(
          "test-model",
          configHash,
          "chat-2",
        ),
        "and its init-registry entry is cleared",
      );
    } finally {
      cleanup();
    }
  },
);
