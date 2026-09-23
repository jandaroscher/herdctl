/**
 * Tests for mid-run message injection into session-backed trigger jobs.
 *
 * Covers the full path: `trigger({ interactive: true })` opens a streaming
 * session, `sendToJob` pushes a turn into that session's live input queue while
 * the run is still going, and the handle is dropped once the job ends.
 */

import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * Turns injected into the live session, in arrival order. The mocked `query`
 * drains the streaming input iterable, so this is what the SDK would actually
 * have received.
 */
const injectedTurns: string[] = [];

/**
 * `SDKRuntime.execute()` (herdctl#458) now feeds `query()` a queue-backed
 * streaming-input iterable for EVERY run, not just `openSession()`-backed
 * ones — the old `typeof input === "string"` branch below is unreachable
 * post-#458, so it no longer distinguishes the "nobody will ever inject a
 * second turn" case. The non-interactive test opts into this instead: it has
 * no handle to inject through, so the mock must resolve turn 1 on its own
 * rather than blocking for a turn 2 that will never arrive.
 */
let singleTurnMode = false;

// Mock the SDK's streaming-input mode: emit an init message, then block on the
// input queue until a second turn arrives, then finish. Blocking on the queue is
// what makes the test deterministic — the job cannot complete before the test's
// sendToJob lands.
vi.mock("@anthropic-ai/claude-agent-sdk", () => ({
  query: vi.fn().mockImplementation((params: { prompt: unknown }) => {
    const input = params.prompt as AsyncIterable<{ message: { content: string } }>;

    const generator = (async function* () {
      yield { type: "system", subtype: "init", session_id: "interactive-session" };

      for await (const message of input) {
        injectedTurns.push(message.message.content);
        // First turn is the trigger prompt itself; the second is the injection.
        if (singleTurnMode || injectedTurns.length >= 2) break;
      }

      yield { type: "result", subtype: "success", result: "done" };
    })();

    // openSession() retains the Query handle for control requests.
    return Object.assign(generator, {
      interrupt: vi.fn().mockResolvedValue(undefined),
      supportedCommands: vi.fn().mockResolvedValue([]),
      setModel: vi.fn().mockResolvedValue(undefined),
    });
  }),
}));

import { JobExecutor } from "../../runner/job-executor.js";
import { FleetManager } from "../fleet-manager.js";

describe("FleetManager.sendToJob", () => {
  let tempDir: string;
  let configPath: string;
  let stateDir: string;

  beforeEach(async () => {
    injectedTurns.length = 0;
    singleTurnMode = false;
    tempDir = await mkdtemp(join(tmpdir(), "fleet-send-to-job-"));
    const configDir = join(tempDir, "config");
    stateDir = join(tempDir, ".herdctl");
    await mkdir(configDir, { recursive: true });

    const yaml = await import("yaml");
    await mkdir(join(configDir, "agents"), { recursive: true });
    await writeFile(
      join(configDir, "agents", "chatty.yaml"),
      yaml.stringify({ name: "chatty", description: "Interactive test agent" }),
    );

    configPath = join(configDir, "herdctl.yaml");
    await writeFile(
      configPath,
      yaml.stringify({
        version: 1,
        fleet: { name: "test-fleet" },
        agents: [{ path: "./agents/chatty.yaml" }],
      }),
    );
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  function createManager() {
    return new FleetManager({
      configPath,
      stateDir,
      logger: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
    });
  }

  it("delivers a message into a running session-backed job", async () => {
    const manager = createManager();
    await manager.initialize();

    let jobId: string | undefined;
    const run = manager.trigger("chatty", undefined, {
      prompt: "initial",
      interactive: true,
      onJobCreated: (id) => {
        jobId = id;
      },
    });

    // Poll until the session is registered — `false` means "not injectable yet".
    let delivered = false;
    for (let attempt = 0; attempt < 200 && !delivered; attempt++) {
      if (jobId) delivered = manager.sendToJob(jobId, "injected mid-run");
      if (!delivered) await new Promise((resolve) => setTimeout(resolve, 10));
    }
    expect(delivered).toBe(true);

    const result = await run;
    expect(result.success).toBe(true);
    expect(injectedTurns).toEqual(["initial", "injected mid-run"]);

    // Handle dropped once the job finished.
    expect(manager.sendToJob(jobId as string, "too late")).toBe(false);
  });

  it("passes injectionGraceMs through to the job executor", async () => {
    const executeSpy = vi.spyOn(JobExecutor.prototype, "execute");
    const manager = createManager();
    await manager.initialize();

    let jobId: string | undefined;
    const run = manager.trigger("chatty", undefined, {
      prompt: "initial",
      interactive: true,
      injectionGraceMs: 15_000,
      onJobCreated: (id) => {
        jobId = id;
      },
    });
    let delivered = false;
    for (let attempt = 0; attempt < 200 && !delivered; attempt++) {
      if (jobId) delivered = manager.sendToJob(jobId, "injected mid-run");
      if (!delivered) await new Promise((resolve) => setTimeout(resolve, 10));
    }
    await run;

    expect(executeSpy).toHaveBeenCalledWith(expect.objectContaining({ injectionGraceMs: 15_000 }));
    executeSpy.mockRestore();
  });

  it("returns false for an unknown job", async () => {
    const manager = createManager();
    await manager.initialize();

    expect(manager.sendToJob("job-2026-01-01-abcdef", "hello")).toBe(false);
    expect(manager.interruptJob("job-2026-01-01-abcdef")).toBe(false);
  });

  it("returns false once the run has stopped reading its input", async () => {
    const manager = createManager();
    await manager.initialize();

    let jobId: string | undefined;
    const run = manager.trigger("chatty", undefined, {
      prompt: "initial",
      interactive: true,
      onJobCreated: (id) => {
        jobId = id;
      },
    });

    let delivered = false;
    for (let attempt = 0; attempt < 200 && !delivered; attempt++) {
      if (jobId) delivered = manager.sendToJob(jobId, "injected mid-run");
      if (!delivered) await new Promise((resolve) => setTimeout(resolve, 10));
    }
    expect(delivered).toBe(true);

    await run;

    // The registry entry is dropped in trigger's finally, but the handle stops
    // accepting input the moment the terminal message lands — the window in
    // between must not report a delivery that MessageQueue would swallow.
    expect(manager.sendToJob(jobId as string, "after the result")).toBe(false);
    expect(manager.interruptJob(jobId as string)).toBe(false);
    expect(injectedTurns).toEqual(["initial", "injected mid-run"]);
  });

  it("returns false for a non-interactive job", async () => {
    // No handle is ever exposed to inject a second turn on this path — resolve
    // after the initial prompt instead of blocking for one that never comes.
    singleTurnMode = true;
    const manager = createManager();
    await manager.initialize();

    let jobId: string | undefined;
    const result = await manager.trigger("chatty", undefined, {
      prompt: "initial",
      onJobCreated: (id) => {
        jobId = id;
      },
    });

    expect(result.success).toBe(true);
    expect(manager.sendToJob(jobId as string, "hello")).toBe(false);
  });
});
