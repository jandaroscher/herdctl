import { mkdir, readFile, realpath, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ResolvedAgent } from "../../config/index.js";
import {
  getJob,
  getSessionInfo,
  initStateDirectory,
  readJobOutputAll,
  updateSessionInfo,
} from "../../state/index.js";
import { executeJob, JobExecutor } from "../job-executor.js";
import { MessageQueue } from "../runtime/message-queue.js";
import type { JobSessionHandle, SDKMessage } from "../types.js";

// =============================================================================
// Test Helpers
// =============================================================================

async function createTempDir(): Promise<string> {
  const baseDir = join(
    tmpdir(),
    `herdctl-executor-test-${Date.now()}-${Math.random().toString(36).slice(2)}`,
  );
  await mkdir(baseDir, { recursive: true });
  return await realpath(baseDir);
}

function createTestAgent(overrides: Partial<ResolvedAgent> = {}): ResolvedAgent {
  return {
    name: "test-agent",
    configPath: "/path/to/agent.yaml",
    fleetPath: [],
    qualifiedName: overrides.name ?? "test-agent",
    ...overrides,
  };
}

function createMockLogger() {
  return {
    warnings: [] as string[],
    errors: [] as string[],
    infos: [] as string[],
    warn: (_msg: string) => {
      // Suppress warnings during tests
    },
    error: (_msg: string) => {
      // Suppress errors during tests
    },
    info: (_msg: string) => {
      // Suppress info during tests
    },
  };
}

// Helper to create mock RuntimeInterface
function createMockRuntime(handler: (options: any) => AsyncIterableIterator<SDKMessage>): any {
  return {
    execute: handler,
  };
}

// Helper to create a mock runtime with predefined messages
function createMockRuntimeWithMessages(messages: SDKMessage[]): any {
  return createMockRuntime(async function* () {
    for (const message of messages) {
      yield message;
    }
  });
}

// Helper to create a mock runtime that yields messages with delays
function createDelayedMockRuntime(messages: SDKMessage[], delayMs: number = 10): any {
  return createMockRuntime(async function* () {
    for (const message of messages) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
      yield message;
    }
  });
}

// Helper to create a mock runtime that throws an error
function createErrorMockRuntime(error: Error): any {
  return createMockRuntime(async function* () {
    throw error;
  });
}

// =============================================================================
// JobExecutor tests
// =============================================================================

describe("JobExecutor", () => {
  let tempDir: string;
  let stateDir: string;

  beforeEach(async () => {
    tempDir = await createTempDir();
    stateDir = join(tempDir, ".herdctl");
    await initStateDirectory({ path: stateDir });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  describe("job lifecycle", () => {
    it("creates job record before execution", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Initialized" },
        { type: "assistant", content: "Done" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.jobId).toMatch(/^job-\d{4}-\d{2}-\d{2}-[a-z0-9]+$/);

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job).not.toBeNull();
      expect(job?.agent).toBe("test-agent");
    });

    it("updates job status to running", async () => {
      let _jobIdDuringExecution: string | undefined;

      const runtime = createMockRuntime(async function* (_options) {
        // During execution, we can check the job status
        yield { type: "system", content: "Running" };
      });

      const executor = new JobExecutor(runtime, {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      // Job should be completed now, but was running during execution
      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.status).toBe("completed");
    });

    it("updates job with final status on success", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Start" },
        { type: "assistant", content: "Task completed!" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.success).toBe(true);

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.status).toBe("completed");
      expect(job?.exit_reason).toBe("success");
      expect(job?.finished_at).toBeDefined();
      expect(job?.duration_seconds).toBeGreaterThanOrEqual(0);
    });

    it("persists per-model token accounting from the result message", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Start" },
        { type: "assistant", content: "Working" },
        {
          type: "result",
          subtype: "success",
          result: "Task completed!",
          num_turns: 4,
          total_cost_usd: 0.2571,
          usage: { input_tokens: 10, output_tokens: 20 },
          // SDK per-model breakdown: an Opus main agent + a Haiku subagent.
          modelUsage: {
            "claude-opus-4-8": {
              inputTokens: 800,
              outputTokens: 400,
              cacheCreationInputTokens: 100,
              cacheReadInputTokens: 6000,
              webSearchRequests: 0,
              costUSD: 0.25,
              contextWindow: 200000,
            },
            "claude-haiku-4-5": {
              inputTokens: 120,
              outputTokens: 40,
              cacheCreationInputTokens: 0,
              cacheReadInputTokens: 0,
              webSearchRequests: 0,
              costUSD: 0.0071,
              contextWindow: 200000,
            },
          },
        } as unknown as SDKMessage,
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.success).toBe(true);

      // Assert against the persisted job record on disk (real run path).
      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.usage).toBeDefined();
      expect(job?.usage?.num_turns).toBe(4);
      expect(job?.usage?.total_cost_usd).toBeCloseTo(0.2571);
      expect(Object.keys(job?.usage?.per_model ?? {})).toEqual([
        "claude-opus-4-8",
        "claude-haiku-4-5",
      ]);
      expect(job?.usage?.per_model["claude-opus-4-8"]).toEqual({
        input_tokens: 800,
        output_tokens: 400,
        cache_creation_input_tokens: 100,
        cache_read_input_tokens: 6000,
      });
      expect(job?.usage?.per_model["claude-haiku-4-5"].input_tokens).toBe(120);
      // No $/token price table is persisted in herdctl state.
      expect(job?.usage?.per_model["claude-opus-4-8"]).not.toHaveProperty("costUSD");
    });

    it("leaves usage null when the run produces no result message", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Start" },
        { type: "assistant", content: "Done, no result summary" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.usage).toBeNull();
    });

    it("updates job with failed status on error", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Start" },
        { type: "error", message: "Something went wrong" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.success).toBe(false);
      expect(result.error?.message).toContain("Something went wrong");

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.status).toBe("failed");
      expect(job?.exit_reason).toBe("error");
    });

    it("handles SDK query throwing an error", async () => {
      const executor = new JobExecutor(createErrorMockRuntime(new Error("SDK error")), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.success).toBe(false);
      expect(result.error?.message).toContain("SDK error");

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.status).toBe("failed");
    });
  });

  describe("streaming output", () => {
    it("writes all messages to job output", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Init" },
        { type: "assistant", content: "Hello" },
        { type: "tool_use", tool_name: "bash", input: { command: "ls" } },
        { type: "tool_result", result: "file1\nfile2", success: true },
        { type: "assistant", content: "Done" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
      expect(output).toHaveLength(5);
      expect(output[0].type).toBe("system");
      expect(output[1].type).toBe("assistant");
      expect(output[2].type).toBe("tool_use");
      expect(output[3].type).toBe("tool_result");
      expect(output[4].type).toBe("assistant");
    });

    it("writes output immediately without buffering", async () => {
      const _outputCountDuringExecution = 0;
      const jobsDir = join(stateDir, "jobs");

      const runtime = createMockRuntime(async function* (_options) {
        yield { type: "system", content: "First" };
        yield { type: "assistant", content: "Second" };
        yield { type: "assistant", content: "Third" };
      });

      // We can't easily verify real-time writing in a unit test,
      // but we can verify all messages are written
      const executor = new JobExecutor(runtime, {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const output = await readJobOutputAll(jobsDir, result.jobId);
      expect(output).toHaveLength(3);
    });

    it("preserves message content and metadata", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Session init", subtype: "session_start" },
        {
          type: "assistant",
          content: "Response",
          usage: { input_tokens: 100, output_tokens: 50 },
        },
        {
          type: "tool_use",
          tool_name: "read_file",
          tool_use_id: "tool-123",
          input: { path: "/etc/hosts" },
        },
        {
          type: "tool_result",
          tool_use_id: "tool-123",
          result: "localhost",
          success: true,
        },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);

      // Check system message
      expect(output[0].type).toBe("system");
      if (output[0].type === "system") {
        expect(output[0].content).toBe("Session init");
        expect(output[0].subtype).toBe("session_start");
      }

      // Check assistant message
      expect(output[1].type).toBe("assistant");
      if (output[1].type === "assistant") {
        expect(output[1].content).toBe("Response");
        expect(output[1].usage?.input_tokens).toBe(100);
        expect(output[1].usage?.output_tokens).toBe(50);
      }

      // Check tool_use message
      expect(output[2].type).toBe("tool_use");
      if (output[2].type === "tool_use") {
        expect(output[2].tool_name).toBe("read_file");
        expect(output[2].tool_use_id).toBe("tool-123");
        expect(output[2].input).toEqual({ path: "/etc/hosts" });
      }

      // Check tool_result message
      expect(output[3].type).toBe("tool_result");
      if (output[3].type === "tool_result") {
        expect(output[3].tool_use_id).toBe("tool-123");
        expect(output[3].result).toBe("localhost");
        expect(output[3].success).toBe(true);
      }
    });

    it("writes error message to output when SDK throws", async () => {
      const executor = new JobExecutor(createErrorMockRuntime(new Error("Connection failed")), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
      expect(output.some((m) => m.type === "error")).toBe(true);

      const errorMsg = output.find((m) => m.type === "error");
      if (errorMsg?.type === "error") {
        expect(errorMsg.message).toContain("Connection failed");
      }
    });
  });

  describe("message type handling", () => {
    it("handles all message types correctly", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "System message", subtype: "init" },
        { type: "assistant", content: "Assistant message", partial: false },
        { type: "tool_use", tool_name: "bash", input: { cmd: "test" } },
        { type: "tool_result", result: "output", success: true },
        { type: "error", message: "Error message", code: "ERR_TEST" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);

      expect(output[0].type).toBe("system");
      expect(output[1].type).toBe("assistant");
      expect(output[2].type).toBe("tool_use");
      expect(output[3].type).toBe("tool_result");
      expect(output[4].type).toBe("error");
    });

    it("handles partial assistant messages", async () => {
      const messages: SDKMessage[] = [
        { type: "assistant", content: "Part 1...", partial: true },
        { type: "assistant", content: "Part 1... Part 2...", partial: true },
        { type: "assistant", content: "Part 1... Part 2... Done!", partial: false },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);

      expect(output).toHaveLength(3);
      if (output[0].type === "assistant") {
        expect(output[0].partial).toBe(true);
      }
      if (output[2].type === "assistant") {
        expect(output[2].partial).toBe(false);
      }
    });

    it("handles tool_result with error", async () => {
      const messages: SDKMessage[] = [
        { type: "tool_use", tool_name: "read_file", input: { path: "/nope" } },
        { type: "tool_result", success: false, error: "File not found" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);

      if (output[1].type === "tool_result") {
        expect(output[1].success).toBe(false);
        expect(output[1].error).toBe("File not found");
      }
    });
  });

  describe("session handling", () => {
    it("extracts session ID from system message with init subtype", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Init", subtype: "init", session_id: "session-abc123" },
        { type: "assistant", content: "Done" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.sessionId).toBe("session-abc123");

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.session_id).toBe("session-abc123");
    });

    it("does not extract session ID from non-init system messages", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Progress", subtype: "progress", session_id: "should-ignore" },
        { type: "assistant", content: "Done" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.sessionId).toBeUndefined();

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.session_id).toBeUndefined();
    });

    it("persists session info via updateSessionInfo", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Init", subtype: "init", session_id: "session-persist-123" },
        { type: "assistant", content: "Done" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      await executor.execute({
        agent: createTestAgent({ name: "persist-agent" }),
        prompt: "Test prompt",
        stateDir,
      });

      // Verify session info was persisted
      const sessionInfo = await getSessionInfo(join(stateDir, "sessions"), "persist-agent");
      expect(sessionInfo).not.toBeNull();
      expect(sessionInfo?.session_id).toBe("session-persist-123");
      expect(sessionInfo?.agent_name).toBe("persist-agent");
      expect(sessionInfo?.job_count).toBe(1);
      expect(sessionInfo?.mode).toBe("autonomous");
    });

    it("increments job_count on subsequent runs with session", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Init", subtype: "init", session_id: "session-multi-123" },
        { type: "assistant", content: "Done" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      // Run twice
      await executor.execute({
        agent: createTestAgent({ name: "multi-agent" }),
        prompt: "First run",
        stateDir,
      });

      await executor.execute({
        agent: createTestAgent({ name: "multi-agent" }),
        prompt: "Second run",
        stateDir,
      });

      // Verify job_count was incremented
      const sessionInfo = await getSessionInfo(join(stateDir, "sessions"), "multi-agent");
      expect(sessionInfo?.job_count).toBe(2);
    });

    it("stores timestamps in session info", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Init", subtype: "init", session_id: "session-ts-123" },
        { type: "assistant", content: "Done" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      await executor.execute({
        agent: createTestAgent({ name: "ts-agent" }),
        prompt: "Test prompt",
        stateDir,
      });

      const sessionInfo = await getSessionInfo(join(stateDir, "sessions"), "ts-agent");
      expect(sessionInfo?.created_at).toBeDefined();
      expect(sessionInfo?.last_used_at).toBeDefined();

      // Verify they are valid ISO timestamps
      expect(() => new Date(sessionInfo!.created_at)).not.toThrow();
      expect(() => new Date(sessionInfo!.last_used_at)).not.toThrow();
    });

    it("returns session ID in RunnerResult for caller use", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Init", subtype: "init", session_id: "session-result-123" },
        { type: "assistant", content: "Done" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      // Session ID should be available in the result for caller use
      expect(result.sessionId).toBe("session-result-123");
    });

    it("passes resume option to SDK", async () => {
      // Create a valid session so resume isn't cleared
      const sessionsDir = join(stateDir, "sessions");
      await updateSessionInfo(sessionsDir, "test-agent", {
        session_id: "session-to-resume",
        mode: "autonomous",
      });

      let receivedOptions: Record<string, unknown> | undefined;

      const runtime = createMockRuntime(async function* (options) {
        receivedOptions = options;
        yield { type: "assistant", content: "Resumed" };
      });

      const executor = new JobExecutor(runtime, {
        logger: createMockLogger(),
      });

      await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
        resume: "session-to-resume",
      });

      expect(receivedOptions?.resume).toBe("session-to-resume");
    });

    it("honors an explicit resume when the agent has no session of its own (cross-agent adoption, #263)", async () => {
      // Regression for #263: agent B is asked to resume a session created by a
      // different agent A in the same process. B has NO agent-level session file
      // on disk. Previously this silently dropped the resume and forked a fresh
      // session; now the explicit caller-provided session ID is honored.
      const sessionsDir = join(stateDir, "sessions");

      // Sanity: agent "keeper-foo" has no session pointer on disk.
      const before = await getSessionInfo(sessionsDir, "keeper-foo");
      expect(before).toBeNull();

      let receivedOptions: Record<string, unknown> | undefined;
      const runtime = createMockRuntime(async function* (options) {
        receivedOptions = options;
        yield {
          type: "system",
          content: "Init",
          subtype: "init",
          session_id: "session-from-agent-a",
        };
        yield { type: "assistant", content: "Resumed" };
      });

      const executor = new JobExecutor(runtime, {
        logger: createMockLogger(),
      });

      await executor.execute({
        // SDK runtime (default): no transcript file probe required.
        agent: createTestAgent({ name: "keeper-foo" }),
        prompt: "Continue the relocated conversation",
        stateDir,
        resume: "session-from-agent-a",
      });

      // The runtime must receive the resume (not undefined → not a fork).
      expect(receivedOptions?.resume).toBe("session-from-agent-a");
      expect(receivedOptions?.fork).toBeUndefined();

      // An agent-level session pointer is now persisted so future runs and
      // restarts treat the session as owned by this agent.
      const after = await getSessionInfo(sessionsDir, "keeper-foo");
      expect(after?.session_id).toBe("session-from-agent-a");
    });

    it("does not adopt a CLI resume when the transcript file is missing", async () => {
      // For the CLI runtime, Claude Code keys session storage by spawn cwd. If the
      // transcript isn't physically present in the agent's working directory,
      // resuming would fail, so we start fresh instead of forwarding the resume.
      let receivedOptions: Record<string, unknown> | undefined;
      const runtime = createMockRuntime(async function* (options) {
        receivedOptions = options;
        yield { type: "assistant", content: "Fresh" };
      });

      const executor = new JobExecutor(runtime, {
        logger: createMockLogger(),
      });

      await executor.execute({
        agent: createTestAgent({
          name: "cli-keeper",
          runtime: "cli",
          // working_directory points at a real dir with no transcript for this id.
          working_directory: tempDir,
        }),
        prompt: "Try to resume a missing transcript",
        stateDir,
        resume: "nonexistent-cli-session",
      });

      // No file on disk → resume dropped, fresh session started.
      expect(receivedOptions?.resume).toBeUndefined();
    });

    it("passes fork option to SDK", async () => {
      let receivedOptions: Record<string, unknown> | undefined;

      const runtime = createMockRuntime(async function* (options) {
        receivedOptions = options;
        yield { type: "assistant", content: "Forked" };
      });

      const executor = new JobExecutor(runtime, {
        logger: createMockLogger(),
      });

      await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
        fork: "session-to-fork",
      });

      expect(receivedOptions?.fork).toBe(true);
    });

    it("resumes the fork source AND sets fork so the runtime forks from it", async () => {
      let receivedOptions: Record<string, unknown> | undefined;

      const runtime = createMockRuntime(async function* (options) {
        receivedOptions = options;
        yield { type: "assistant", content: "Forked" };
      });

      const executor = new JobExecutor(runtime, {
        logger: createMockLogger(),
      });

      await executor.execute({
        agent: createTestAgent(),
        prompt: "branch off",
        stateDir,
        fork: "session-to-fork",
      });

      // The runtime must resume the SOURCE session (so the CLI/SDK has a
      // transcript to fork from) while fork:true tells it to write new turns to
      // a brand-new session id. Seeding resume from the fork source is what makes
      // `--resume <src> --fork-session` (CLI) / resume+forkSession (SDK) work.
      expect(receivedOptions?.resume).toBe("session-to-fork");
      expect(receivedOptions?.fork).toBe(true);
    });

    it("creates job with trigger_type 'fork' and forked_from when forking", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Init", subtype: "init", session_id: "forked-session-123" },
        { type: "assistant", content: "Forked session started" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent({ name: "fork-agent" }),
        prompt: "Continue from where we left off",
        stateDir,
        fork: "original-session-id",
        forkedFrom: "job-2024-01-15-abc123",
      });

      expect(result.success).toBe(true);

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.trigger_type).toBe("fork");
      expect(job?.forked_from).toBe("job-2024-01-15-abc123");
    });

    it("sets trigger_type to fork even if triggerType option provided", async () => {
      const messages: SDKMessage[] = [{ type: "assistant", content: "Forked" }];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
        fork: "session-to-fork",
        triggerType: "manual", // Should be overridden by fork
      });

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.trigger_type).toBe("fork");
    });

    it("does not set forked_from when not forking", async () => {
      const messages: SDKMessage[] = [{ type: "assistant", content: "Normal run" }];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.trigger_type).toBe("manual");
      expect(job?.forked_from).toBeNull();
    });

    it("updates session info job_count after resume", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Init", subtype: "init", session_id: "resume-session-123" },
        { type: "assistant", content: "Resumed" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      // First run to establish the session
      await executor.execute({
        agent: createTestAgent({ name: "resume-test-agent" }),
        prompt: "First run",
        stateDir,
      });

      // Check initial job count
      let sessionInfo = await getSessionInfo(join(stateDir, "sessions"), "resume-test-agent");
      expect(sessionInfo?.job_count).toBe(1);

      // Resume the session
      await executor.execute({
        agent: createTestAgent({ name: "resume-test-agent" }),
        prompt: "Resume run",
        stateDir,
        resume: "resume-session-123",
      });

      // Check job count was incremented
      sessionInfo = await getSessionInfo(join(stateDir, "sessions"), "resume-test-agent");
      expect(sessionInfo?.job_count).toBe(2);
    });

    it("updates session info job_count after fork", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Init", subtype: "init", session_id: "fork-session-123" },
        { type: "assistant", content: "Forked" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      // First run to establish a session
      await executor.execute({
        agent: createTestAgent({ name: "fork-test-agent" }),
        prompt: "First run",
        stateDir,
      });

      // Check initial job count
      let sessionInfo = await getSessionInfo(join(stateDir, "sessions"), "fork-test-agent");
      expect(sessionInfo?.job_count).toBe(1);

      // Fork the session
      await executor.execute({
        agent: createTestAgent({ name: "fork-test-agent" }),
        prompt: "Fork run",
        stateDir,
        fork: "original-session-id",
        forkedFrom: "job-2024-01-15-parent",
      });

      // Check job count was incremented
      sessionInfo = await getSessionInfo(join(stateDir, "sessions"), "fork-test-agent");
      expect(sessionInfo?.job_count).toBe(2);
    });

    it("does not persist session info when no session ID is present", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "No session" },
        { type: "assistant", content: "Done" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      await executor.execute({
        agent: createTestAgent({ name: "no-session-agent" }),
        prompt: "Test prompt",
        stateDir,
      });

      // Verify no session info was created
      const sessionInfo = await getSessionInfo(join(stateDir, "sessions"), "no-session-agent");
      expect(sessionInfo).toBeNull();
    });
  });

  describe("summary extraction", () => {
    it("extracts summary from explicit summary field", async () => {
      const messages: SDKMessage[] = [
        { type: "assistant", content: "Working..." },
        { type: "assistant", content: "Done!", summary: "Task completed" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.summary).toBe("Task completed");

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.summary).toBe("Task completed");
    });

    it("extracts summary from short final assistant message", async () => {
      const messages: SDKMessage[] = [
        { type: "assistant", content: "Working..." },
        { type: "assistant", content: "All tasks finished successfully." },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.summary).toBe("All tasks finished successfully.");
    });

    it("returns undefined summary when no assistant messages exist", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Initialized" },
        { type: "tool_use", tool_name: "bash", input: { command: "ls" } },
        { type: "tool_result", result: "file1\nfile2", success: true },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.summary).toBeUndefined();

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.summary).toBeUndefined();
    });

    it("returns undefined summary when all assistant messages are partial", async () => {
      const messages: SDKMessage[] = [
        { type: "assistant", content: "Partial 1...", partial: true },
        { type: "assistant", content: "Partial 2...", partial: true },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.summary).toBeUndefined();
    });

    it("returns full summary for long assistant messages (no truncation)", async () => {
      const longContent = "x".repeat(5000);
      const messages: SDKMessage[] = [
        { type: "assistant", content: longContent },
        { type: "assistant", content: longContent },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      // No truncation at this layer - downstream consumers (Discord) handle their own limits
      expect(result.summary).toBeDefined();
      expect(result.summary).toBe(longContent);
      expect(result.summary?.length).toBe(5000);
    });

    it("returns full explicit summary (no truncation)", async () => {
      const longSummary = "x".repeat(5000);
      const messages: SDKMessage[] = [
        { type: "assistant", content: "Done!", summary: longSummary },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      // No truncation at this layer
      expect(result.summary).toBeDefined();
      expect(result.summary).toBe(longSummary);
      expect(result.summary!.length).toBe(5000);

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.summary).toBe(longSummary);
    });

    it("uses latest summary when multiple messages have summaries", async () => {
      const messages: SDKMessage[] = [
        { type: "assistant", content: "First", summary: "First summary" },
        { type: "assistant", content: "Second", summary: "Second summary" },
        { type: "assistant", content: "Third", summary: "Final summary" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.summary).toBe("Final summary");
    });

    it("handles empty message stream with undefined summary", async () => {
      const messages: SDKMessage[] = [];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.summary).toBeUndefined();
    });
  });

  describe("callbacks", () => {
    it("calls onMessage for each SDK message", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Init" },
        { type: "assistant", content: "Hello" },
        { type: "assistant", content: "World" },
      ];

      const receivedMessages: SDKMessage[] = [];
      const onMessage = vi.fn((msg: SDKMessage) => {
        receivedMessages.push(msg);
      });

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
        onMessage,
      });

      expect(onMessage).toHaveBeenCalledTimes(3);
      expect(receivedMessages).toHaveLength(3);
      expect(receivedMessages[0].type).toBe("system");
      expect(receivedMessages[1].type).toBe("assistant");
      expect(receivedMessages[2].type).toBe("assistant");
    });

    it("continues execution if onMessage throws", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Init" },
        { type: "assistant", content: "Continue" },
      ];

      const onMessage = vi.fn(() => {
        throw new Error("Callback error");
      });

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
        onMessage,
      });

      // Should still succeed despite callback error
      expect(result.success).toBe(true);
      expect(onMessage).toHaveBeenCalledTimes(2);
    });

    it("supports async onMessage callback", async () => {
      const messages: SDKMessage[] = [{ type: "assistant", content: "Hello" }];

      const onMessage = vi.fn(async () => {
        await new Promise((resolve) => setTimeout(resolve, 10));
      });

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
        onMessage,
      });

      expect(onMessage).toHaveBeenCalledTimes(1);
    });
  });

  describe("trigger types", () => {
    it("sets trigger type to manual by default", async () => {
      const messages: SDKMessage[] = [{ type: "assistant", content: "Done" }];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.trigger_type).toBe("manual");
    });

    it("sets trigger type from options", async () => {
      const messages: SDKMessage[] = [{ type: "assistant", content: "Done" }];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
        triggerType: "schedule",
        schedule: "daily-cleanup",
      });

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.trigger_type).toBe("schedule");
      expect(job?.schedule).toBe("daily-cleanup");
    });
  });

  describe("duration tracking", () => {
    it("calculates duration in seconds", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Start" },
        { type: "assistant", content: "End" },
      ];

      const executor = new JobExecutor(createDelayedMockRuntime(messages, 50), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.durationSeconds).toBeGreaterThanOrEqual(0);

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.duration_seconds).toBeGreaterThanOrEqual(0);
    });
  });
});

// =============================================================================
// executeJob convenience function tests
// =============================================================================

describe("executeJob", () => {
  let tempDir: string;
  let stateDir: string;

  beforeEach(async () => {
    tempDir = await createTempDir();
    stateDir = join(tempDir, ".herdctl");
    await initStateDirectory({ path: stateDir });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("executes job using convenience function", async () => {
    const messages: SDKMessage[] = [
      { type: "system", content: "Init" },
      { type: "assistant", content: "Done" },
    ];

    const result = await executeJob(
      createMockRuntimeWithMessages(messages),
      {
        agent: createTestAgent({ name: "convenience-agent" }),
        prompt: "Test prompt",
        stateDir,
      },
      { logger: createMockLogger() },
    );

    expect(result.success).toBe(true);
    expect(result.jobId).toBeDefined();

    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.agent).toBe("convenience-agent");
  });

  it("passes executor options", async () => {
    const messages: SDKMessage[] = [{ type: "assistant", content: "Done" }];
    const logger = createMockLogger();

    await executeJob(
      createMockRuntimeWithMessages(messages),
      {
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      },
      { logger },
    );

    // Logger should have been used (even though messages are suppressed)
  });
});

// =============================================================================
// Edge cases
// =============================================================================

describe("edge cases", () => {
  let tempDir: string;
  let stateDir: string;

  beforeEach(async () => {
    tempDir = await createTempDir();
    stateDir = join(tempDir, ".herdctl");
    await initStateDirectory({ path: stateDir });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("handles empty message stream", async () => {
    const executor = new JobExecutor(createMockRuntimeWithMessages([]), {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent(),
      prompt: "Test prompt",
      stateDir,
    });

    expect(result.success).toBe(true);

    const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
    expect(output).toHaveLength(0);
  });

  it("handles very long content in messages", async () => {
    const longContent = "x".repeat(100000);
    const messages: SDKMessage[] = [{ type: "assistant", content: longContent }];

    const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent(),
      prompt: "Test prompt",
      stateDir,
    });

    const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
    if (output[0].type === "assistant") {
      expect(output[0].content).toHaveLength(100000);
    }
  });

  it("handles unicode content", async () => {
    const messages: SDKMessage[] = [
      { type: "assistant", content: "Hello 世界! 🌍 Γεια σου κόσμε" },
    ];

    const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent(),
      prompt: "Test prompt",
      stateDir,
    });

    const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
    if (output[0].type === "assistant") {
      expect(output[0].content).toBe("Hello 世界! 🌍 Γεια σου κόσμε");
    }
  });

  it("handles special characters in content", async () => {
    const messages: SDKMessage[] = [
      {
        type: "assistant",
        content: 'Content with "quotes", \\backslashes\\, and\nnewlines',
      },
    ];

    const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent(),
      prompt: "Test prompt",
      stateDir,
    });

    const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
    if (output[0].type === "assistant") {
      expect(output[0].content).toBe('Content with "quotes", \\backslashes\\, and\nnewlines');
    }
  });

  it("handles rapid message stream", async () => {
    const messages: SDKMessage[] = Array(100)
      .fill(null)
      .map((_, i) => ({
        type: "assistant" as const,
        content: `Message ${i}`,
      }));

    const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent(),
      prompt: "Test prompt",
      stateDir,
    });

    const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
    expect(output).toHaveLength(100);
  });
});

// =============================================================================
// Enhanced error handling tests (US-7)
// =============================================================================

describe("error handling (US-7)", () => {
  let tempDir: string;
  let stateDir: string;

  beforeEach(async () => {
    tempDir = await createTempDir();
    stateDir = join(tempDir, ".herdctl");
    await initStateDirectory({ path: stateDir });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  describe("SDK initialization errors", () => {
    it("catches SDK initialization errors (e.g., missing API key)", async () => {
      // Simulates SDK throwing immediately when execute is called
      const runtime = createMockRuntime(() => {
        throw new Error("ANTHROPIC_API_KEY environment variable is not set");
      });

      const executor = new JobExecutor(runtime, {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent({ name: "api-key-agent" }),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.success).toBe(false);
      expect(result.error?.message).toContain("ANTHROPIC_API_KEY");
      expect(result.errorDetails?.type).toBe("initialization");
    });

    it("provides context (job ID, agent name) in initialization error", async () => {
      const runtime = createMockRuntime(() => {
        throw new Error("SDK init failed");
      });

      const executor = new JobExecutor(runtime, {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent({ name: "context-agent" }),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.error?.message).toContain("context-agent");
      expect(result.error?.message).toContain(result.jobId);
    });
  });

  describe("SDK streaming errors", () => {
    it("catches SDK streaming errors during execution", async () => {
      const runtime = createMockRuntime(async function* (_options) {
        yield { type: "system", content: "Init" };
        yield { type: "assistant", content: "Working..." };
        throw new Error("Connection reset by peer");
      });

      const executor = new JobExecutor(runtime, {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent({ name: "streaming-agent" }),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.success).toBe(false);
      expect(result.error?.message).toContain("Connection reset");
      expect(result.errorDetails?.type).toBe("streaming");
    });

    it("tracks messages received before streaming error", async () => {
      const runtime = createMockRuntime(async function* (_options) {
        yield { type: "system", content: "Init" };
        yield { type: "assistant", content: "Message 1" };
        yield { type: "assistant", content: "Message 2" };
        throw new Error("Stream interrupted");
      });

      const executor = new JobExecutor(runtime, {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.success).toBe(false);
      expect(result.errorDetails?.messagesReceived).toBe(3);
    });

    it("identifies recoverable errors (rate limit)", async () => {
      const runtime = createMockRuntime(async function* (_options) {
        yield { type: "system", content: "Init" };
        throw new Error("Rate limit exceeded, please retry");
      });

      const executor = new JobExecutor(runtime, {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.errorDetails?.recoverable).toBe(true);
    });

    it("identifies non-recoverable errors", async () => {
      const runtime = createMockRuntime(async function* (_options) {
        yield { type: "system", content: "Init" };
        throw new Error("Invalid request format");
      });

      const executor = new JobExecutor(runtime, {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.errorDetails?.recoverable).toBe(false);
    });
  });

  describe("error logging to job output", () => {
    it("logs error messages to job output as error type messages", async () => {
      const executor = new JobExecutor(
        createErrorMockRuntime(new Error("Test error for logging")),
        { logger: createMockLogger() },
      );

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
      const errorMessages = output.filter((m) => m.type === "error");

      expect(errorMessages.length).toBeGreaterThanOrEqual(1);
      expect(errorMessages[0].type).toBe("error");
      if (errorMessages[0].type === "error") {
        expect(errorMessages[0].message).toContain("Test error for logging");
      }
    });

    it("includes error code in job output when available", async () => {
      const errorWithCode = new Error("Network error") as NodeJS.ErrnoException;
      errorWithCode.code = "ECONNRESET";

      const executor = new JobExecutor(createErrorMockRuntime(errorWithCode), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
      const errorMsg = output.find((m) => m.type === "error");

      if (errorMsg?.type === "error") {
        expect(errorMsg.code).toBe("ECONNRESET");
      }
    });

    it("includes stack trace in job output", async () => {
      const error = new Error("Stack trace test");

      const executor = new JobExecutor(createErrorMockRuntime(error), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
      const errorMsg = output.find((m) => m.type === "error");

      if (errorMsg?.type === "error") {
        expect(errorMsg.stack).toBeDefined();
        expect(errorMsg.stack).toContain("Stack trace test");
      }
    });
  });

  describe("job status updates", () => {
    it("updates job status to failed with error exit_reason", async () => {
      const executor = new JobExecutor(createErrorMockRuntime(new Error("Failure")), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.status).toBe("failed");
      expect(job?.exit_reason).toBe("error");
    });

    it("sets exit_reason to timeout for timeout errors", async () => {
      const timeoutError = new Error("Request timed out");

      const executor = new JobExecutor(createErrorMockRuntime(timeoutError), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.exit_reason).toBe("timeout");
    });

    it("sets exit_reason to cancelled for abort errors", async () => {
      const abortError = new Error("Operation aborted by user");

      const executor = new JobExecutor(createErrorMockRuntime(abortError), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.exit_reason).toBe("cancelled");
    });

    it("sets exit_reason to max_turns for turn limit errors", async () => {
      const maxTurnsError = new Error("Maximum turns exceeded");

      const executor = new JobExecutor(createErrorMockRuntime(maxTurnsError), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      const job = await getJob(join(stateDir, "jobs"), result.jobId);
      expect(job?.exit_reason).toBe("max_turns");
    });
  });

  describe("error details in RunnerResult", () => {
    it("provides descriptive error message with context", async () => {
      const executor = new JobExecutor(createErrorMockRuntime(new Error("API connection failed")), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent({ name: "descriptive-agent" }),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.error).toBeDefined();
      expect(result.error?.message).toContain("API connection failed");
      expect(result.error?.message).toContain("descriptive-agent");
      expect(result.error?.message).toContain(result.jobId);
    });

    it("returns error details in RunnerResult", async () => {
      const errorWithCode = new Error("Network timeout") as NodeJS.ErrnoException;
      errorWithCode.code = "ETIMEDOUT";

      const executor = new JobExecutor(createErrorMockRuntime(errorWithCode), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.errorDetails).toBeDefined();
      expect(result.errorDetails?.message).toContain("Network timeout");
      expect(result.errorDetails?.code).toBe("ETIMEDOUT");
      expect(result.errorDetails?.stack).toBeDefined();
    });
  });

  describe("malformed SDK responses", () => {
    it("does not crash on malformed SDK messages", async () => {
      const runtime = createMockRuntime(async function* (_options) {
        yield { type: "system", content: "Init" };
        // Yield a malformed message (null)
        yield null as unknown as SDKMessage;
        yield { type: "assistant", content: "Continuing after malformed" };
      });

      const executor = new JobExecutor(runtime, {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      // Should complete without crashing
      expect(result.success).toBe(true);

      const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
      // Should have processed all 3 messages (including malformed one logged as system)
      expect(output.length).toBe(3);
      // The malformed message should be logged as a system warning
      const malformedMsg = output.find(
        (m) => m.type === "system" && m.subtype === "malformed_message",
      );
      expect(malformedMsg).toBeDefined();
    });

    it("handles messages with missing type field", async () => {
      const runtime = createMockRuntime(async function* (_options) {
        yield { type: "system", content: "Init" };
        yield { content: "Missing type" } as unknown as SDKMessage;
        yield { type: "assistant", content: "Done" };
      });

      const executor = new JobExecutor(runtime, {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.success).toBe(true);

      const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
      // Should contain a system message about unknown type
      const unknownTypeMsg = output.find(
        (m) => m.type === "system" && m.subtype === "unknown_type",
      );
      expect(unknownTypeMsg).toBeDefined();
    });

    it("handles messages with unexpected type values", async () => {
      const runtime = createMockRuntime(async function* (_options) {
        yield { type: "system", content: "Init" };
        yield { type: "unexpected_type", content: "Unknown" } as unknown as SDKMessage;
        yield { type: "assistant", content: "Done" };
      });

      const executor = new JobExecutor(runtime, {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.success).toBe(true);

      const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
      // Should handle gracefully
      expect(output.length).toBeGreaterThanOrEqual(2);
    });

    it("handles SDK error message type gracefully", async () => {
      const messages: SDKMessage[] = [
        { type: "system", content: "Start" },
        { type: "error", message: "SDK reported error", code: "SDK_ERR" },
      ];

      const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
        logger: createMockLogger(),
      });

      const result = await executor.execute({
        agent: createTestAgent(),
        prompt: "Test prompt",
        stateDir,
      });

      expect(result.success).toBe(false);
      expect(result.error?.message).toContain("SDK reported error");
      expect(result.errorDetails?.code).toBe("SDK_ERR");
    });
  });
});

// =============================================================================
// Output to file tests (US-9)
// =============================================================================

describe("outputToFile (US-9)", () => {
  let tempDir: string;
  let stateDir: string;

  beforeEach(async () => {
    tempDir = await createTempDir();
    stateDir = join(tempDir, ".herdctl");
    await initStateDirectory({ path: stateDir });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("creates job output directory when outputToFile is true", async () => {
    const messages: SDKMessage[] = [
      { type: "system", content: "Init" },
      { type: "assistant", content: "Hello" },
    ];

    const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent(),
      prompt: "Test prompt",
      stateDir,
      outputToFile: true,
    });

    // Verify directory was created
    const jobOutputDir = join(stateDir, "jobs", result.jobId);
    const dirStat = await stat(jobOutputDir);
    expect(dirStat.isDirectory()).toBe(true);
  });

  it("writes output.log file when outputToFile is true", async () => {
    const messages: SDKMessage[] = [
      { type: "system", content: "Init" },
      { type: "assistant", content: "Hello world" },
    ];

    const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent(),
      prompt: "Test prompt",
      stateDir,
      outputToFile: true,
    });

    // Verify output.log file exists
    const outputLogPath = join(stateDir, "jobs", result.jobId, "output.log");
    const logContent = await readFile(outputLogPath, "utf-8");
    expect(logContent).toContain("[SYSTEM] Init");
    expect(logContent).toContain("[ASSISTANT] Hello world");
  });

  it("does not create job output directory when outputToFile is false", async () => {
    const messages: SDKMessage[] = [
      { type: "system", content: "Init" },
      { type: "assistant", content: "Hello" },
    ];

    const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent(),
      prompt: "Test prompt",
      stateDir,
      outputToFile: false,
    });

    // Verify directory was NOT created
    const jobOutputDir = join(stateDir, "jobs", result.jobId);
    try {
      await stat(jobOutputDir);
      expect.fail("Directory should not exist");
    } catch (error) {
      expect((error as NodeJS.ErrnoException).code).toBe("ENOENT");
    }
  });

  it("does not create job output directory when outputToFile is not specified", async () => {
    const messages: SDKMessage[] = [
      { type: "system", content: "Init" },
      { type: "assistant", content: "Hello" },
    ];

    const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent(),
      prompt: "Test prompt",
      stateDir,
      // outputToFile not specified (defaults to false)
    });

    // Verify directory was NOT created
    const jobOutputDir = join(stateDir, "jobs", result.jobId);
    try {
      await stat(jobOutputDir);
      expect.fail("Directory should not exist");
    } catch (error) {
      expect((error as NodeJS.ErrnoException).code).toBe("ENOENT");
    }
  });

  it("writes all message types to output.log", async () => {
    const messages: SDKMessage[] = [
      { type: "system", content: "System message" },
      { type: "assistant", content: "Assistant response" },
      { type: "tool_use", tool_name: "read_file", input: { path: "/etc/hosts" } },
      { type: "tool_result", result: "localhost", success: true },
      { type: "error", message: "An error occurred" },
    ];

    const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent(),
      prompt: "Test prompt",
      stateDir,
      outputToFile: true,
    });

    const outputLogPath = join(stateDir, "jobs", result.jobId, "output.log");
    const logContent = await readFile(outputLogPath, "utf-8");

    expect(logContent).toContain("[SYSTEM] System message");
    expect(logContent).toContain("[ASSISTANT] Assistant response");
    expect(logContent).toContain("[TOOL] read_file");
    expect(logContent).toContain("[TOOL_RESULT] (OK) localhost");
    expect(logContent).toContain("[ERROR] An error occurred");
  });

  it("includes timestamps in output.log lines", async () => {
    const messages: SDKMessage[] = [{ type: "assistant", content: "Hello" }];

    const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent(),
      prompt: "Test prompt",
      stateDir,
      outputToFile: true,
    });

    const outputLogPath = join(stateDir, "jobs", result.jobId, "output.log");
    const logContent = await readFile(outputLogPath, "utf-8");

    // Check for ISO timestamp format: [YYYY-MM-DDTHH:MM:SS.sssZ]
    expect(logContent).toMatch(/\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z\]/);
  });

  it("still writes to JSONL even when outputToFile is true", async () => {
    const messages: SDKMessage[] = [
      { type: "system", content: "Init" },
      { type: "assistant", content: "Response" },
    ];

    const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent(),
      prompt: "Test prompt",
      stateDir,
      outputToFile: true,
    });

    // Verify JSONL still exists
    const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
    expect(output).toHaveLength(2);
    expect(output[0].type).toBe("system");
    expect(output[1].type).toBe("assistant");
  });

  it("handles failed tool results in output.log", async () => {
    const messages: SDKMessage[] = [
      { type: "tool_result", result: "File not found", success: false },
    ];

    const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent(),
      prompt: "Test prompt",
      stateDir,
      outputToFile: true,
    });

    const outputLogPath = join(stateDir, "jobs", result.jobId, "output.log");
    const logContent = await readFile(outputLogPath, "utf-8");

    expect(logContent).toContain("[TOOL_RESULT] (FAILED) File not found");
  });

  it("events still stream regardless of outputToFile setting", async () => {
    const receivedMessages: SDKMessage[] = [];
    const onMessage = vi.fn((msg: SDKMessage) => {
      receivedMessages.push(msg);
    });

    const messages: SDKMessage[] = [
      { type: "system", content: "Init" },
      { type: "assistant", content: "Hello" },
    ];

    const executor = new JobExecutor(createMockRuntimeWithMessages(messages), {
      logger: createMockLogger(),
    });

    await executor.execute({
      agent: createTestAgent(),
      prompt: "Test prompt",
      stateDir,
      outputToFile: true,
      onMessage,
    });

    // Events should still be emitted
    expect(onMessage).toHaveBeenCalledTimes(2);
    expect(receivedMessages).toHaveLength(2);
  });
});

// =============================================================================
// Session expiration handling tests (fixes unexpected logout bug)
// =============================================================================

describe("session expiration handling", () => {
  let tempDir: string;
  let stateDir: string;

  beforeEach(async () => {
    tempDir = await createTempDir();
    stateDir = join(tempDir, ".herdctl");
    await initStateDirectory({ path: stateDir });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("starts fresh session when existing session is expired", async () => {
    const sessionsDir = join(stateDir, "sessions");

    // Create an expired session (last used 25 hours ago, default timeout is 24h)
    const expiredLastUsed = new Date(Date.now() - 25 * 60 * 60 * 1000).toISOString();
    await updateSessionInfo(sessionsDir, "expiry-test-agent", {
      session_id: "old-expired-session",
      mode: "autonomous",
    });

    // Manually update last_used_at to make it expired
    const { writeFile } = await import("node:fs/promises");
    const sessionPath = join(sessionsDir, "expiry-test-agent.json");
    const sessionData = {
      agent_name: "expiry-test-agent",
      session_id: "old-expired-session",
      created_at: expiredLastUsed,
      last_used_at: expiredLastUsed,
      job_count: 1,
      mode: "autonomous",
    };
    await writeFile(sessionPath, JSON.stringify(sessionData));

    let receivedOptions: any;
    const runtime = createMockRuntime(async function* (options) {
      receivedOptions = options;
      yield { type: "system", content: "Init", subtype: "init", session_id: "new-fresh-session" };
      yield { type: "assistant", content: "Started fresh" };
    });

    const executor = new JobExecutor(runtime, {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent({ name: "expiry-test-agent" }),
      prompt: "Test prompt",
      stateDir,
      resume: "old-expired-session", // Try to resume expired session
    });

    expect(result.success).toBe(true);
    // Should NOT have passed resume option since session was expired
    expect(receivedOptions?.resume).toBeUndefined();
    // New session should be created
    expect(result.sessionId).toBe("new-fresh-session");
  });

  it("resumes valid session that is not expired", async () => {
    const sessionsDir = join(stateDir, "sessions");

    // Create a valid session (last used 1 hour ago, default timeout is 24h)
    await updateSessionInfo(sessionsDir, "valid-session-agent", {
      session_id: "valid-session-id",
      mode: "autonomous",
    });

    let receivedOptions: any;
    const runtime = createMockRuntime(async function* (options) {
      receivedOptions = options;
      yield { type: "assistant", content: "Resumed successfully" };
    });

    const executor = new JobExecutor(runtime, {
      logger: createMockLogger(),
    });

    await executor.execute({
      agent: createTestAgent({ name: "valid-session-agent" }),
      prompt: "Test prompt",
      stateDir,
      resume: "valid-session-id",
    });

    // Should have passed resume option since session is valid
    expect(receivedOptions?.resume).toBe("valid-session-id");
  });

  it("trusts caller-provided session ID when it differs from agent-level session", async () => {
    const sessionsDir = join(stateDir, "sessions");

    // Create a valid agent-level session with a DIFFERENT session_id than what options.resume will provide
    // This simulates per-thread Slack sessions where each thread has its own session ID
    // managed externally by the session manager, not the agent-level session file
    await updateSessionInfo(sessionsDir, "session-mismatch-agent", {
      session_id: "stored-session-abc123", // The agent-level session on disk
      mode: "autonomous",
    });

    let receivedOptions: any;
    const runtime = createMockRuntime(async function* (options) {
      receivedOptions = options;
      yield { type: "assistant", content: "Resumed with caller session" };
    });

    const executor = new JobExecutor(runtime, {
      logger: createMockLogger(),
    });

    await executor.execute({
      agent: createTestAgent({ name: "session-mismatch-agent" }),
      prompt: "Test prompt",
      stateDir,
      // Pass a DIFFERENT value than what's stored — this is a per-thread session ID
      // managed by the caller (e.g. SlackManager's session manager)
      resume: "per-thread-session-xyz789",
    });

    // Should trust the caller's session ID directly — it's a per-thread session
    // managed externally, not the agent-level session stored on disk
    expect(receivedOptions?.resume).toBe("per-thread-session-xyz789");
  });

  it("respects custom session timeout from agent config", async () => {
    const sessionsDir = join(stateDir, "sessions");

    // Create a session that is 2 hours old
    const twoHoursAgo = new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString();
    const { writeFile } = await import("node:fs/promises");
    const sessionPath = join(sessionsDir, "custom-timeout-agent.json");
    const sessionData = {
      agent_name: "custom-timeout-agent",
      session_id: "two-hour-old-session",
      created_at: twoHoursAgo,
      last_used_at: twoHoursAgo,
      job_count: 1,
      mode: "autonomous",
    };
    await writeFile(sessionPath, JSON.stringify(sessionData));

    let receivedOptions: any;
    const runtime = createMockRuntime(async function* (options) {
      receivedOptions = options;
      yield { type: "assistant", content: "Done" };
    });

    const executor = new JobExecutor(runtime, {
      logger: createMockLogger(),
    });

    // With 1 hour timeout, 2-hour-old session should be expired
    await executor.execute({
      agent: createTestAgent({
        name: "custom-timeout-agent",
        session: { timeout: "1h" },
      }),
      prompt: "Test prompt",
      stateDir,
      resume: "two-hour-old-session",
    });

    // Should NOT have passed resume since session exceeded custom timeout
    expect(receivedOptions?.resume).toBeUndefined();
  });

  it("writes system message to job output when session expires", async () => {
    const sessionsDir = join(stateDir, "sessions");

    // Create an expired session
    const expiredLastUsed = new Date(Date.now() - 25 * 60 * 60 * 1000).toISOString();
    const { writeFile } = await import("node:fs/promises");
    const sessionPath = join(sessionsDir, "output-test-agent.json");
    const sessionData = {
      agent_name: "output-test-agent",
      session_id: "expired-session",
      created_at: expiredLastUsed,
      last_used_at: expiredLastUsed,
      job_count: 1,
      mode: "autonomous",
    };
    await writeFile(sessionPath, JSON.stringify(sessionData));

    const runtime = createMockRuntime(async function* () {
      yield { type: "assistant", content: "Fresh start" };
    });

    const executor = new JobExecutor(runtime, {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent({ name: "output-test-agent" }),
      prompt: "Test prompt",
      stateDir,
      resume: "expired-session",
    });

    // Check job output for session expiry message
    const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
    const systemMsg = output.find((m) => m.type === "system" && m.content?.includes("session"));
    expect(systemMsg).toBeDefined();
  });

  it("clears expired session file when detected", async () => {
    const sessionsDir = join(stateDir, "sessions");

    // Create an expired session
    const expiredLastUsed = new Date(Date.now() - 25 * 60 * 60 * 1000).toISOString();
    const { writeFile } = await import("node:fs/promises");
    const sessionPath = join(sessionsDir, "clear-test-agent.json");
    const sessionData = {
      agent_name: "clear-test-agent",
      session_id: "expired-to-clear",
      created_at: expiredLastUsed,
      last_used_at: expiredLastUsed,
      job_count: 1,
      mode: "autonomous",
    };
    await writeFile(sessionPath, JSON.stringify(sessionData));

    // Verify session exists
    const sessionBefore = await getSessionInfo(sessionsDir, "clear-test-agent");
    expect(sessionBefore).not.toBeNull();

    const runtime = createMockRuntime(async function* () {
      yield { type: "system", content: "Init", subtype: "init", session_id: "new-session" };
      yield { type: "assistant", content: "Fresh" };
    });

    const executor = new JobExecutor(runtime, {
      logger: createMockLogger(),
    });

    await executor.execute({
      agent: createTestAgent({ name: "clear-test-agent" }),
      prompt: "Test prompt",
      stateDir,
      resume: "expired-to-clear",
    });

    // Session should now have the new ID (old one was cleared and new one created)
    const sessionAfter = await getSessionInfo(sessionsDir, "clear-test-agent");
    expect(sessionAfter?.session_id).toBe("new-session");
  });

  it("retries with fresh session when server-side session expiration detected", async () => {
    const sessionsDir = join(stateDir, "sessions");

    // Create a valid session (not locally expired)
    await updateSessionInfo(sessionsDir, "server-expiry-agent", {
      session_id: "valid-local-session",
      mode: "autonomous",
    });

    let attemptCount = 0;
    let lastResumeValue: string | undefined;

    // First attempt throws session expired error, second succeeds
    const runtime = createMockRuntime(async function* (options) {
      attemptCount++;
      lastResumeValue = options.resume;

      if (attemptCount === 1 && options.resume) {
        // First attempt with resume - server says session expired
        throw new Error("Session expired on server");
      }

      // Second attempt or fresh session - succeed
      yield { type: "system", content: "Init", subtype: "init", session_id: "new-server-session" };
      yield { type: "assistant", content: "Success after retry" };
    });

    const executor = new JobExecutor(runtime, {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent({ name: "server-expiry-agent" }),
      prompt: "Test prompt",
      stateDir,
      resume: "valid-local-session",
    });

    // Should have succeeded after retry
    expect(result.success).toBe(true);
    expect(result.sessionId).toBe("new-server-session");

    // Should have attempted twice: first with resume, then fresh
    expect(attemptCount).toBe(2);

    // Second attempt should NOT have resume (fresh session)
    expect(lastResumeValue).toBeUndefined();

    // Check job output includes retry message
    const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
    const retryMsg = output.find(
      (m) => m.type === "system" && m.content?.includes("Retrying with fresh session"),
    );
    expect(retryMsg).toBeDefined();
  });

  it("retries with fresh session when resume yields a session-not-found error (#126)", async () => {
    // Regression for #126: the CLI runtime does not THROW when `claude --resume`
    // can't find the session (e.g. the working_directory changed and the
    // transcript lives in the old project dir). It YIELDS a terminal error
    // message and the loop breaks normally — so the catch-block retry never ran.
    // The post-loop retry must recover from this yielded-error path too.
    const sessionsDir = join(stateDir, "sessions");

    await updateSessionInfo(sessionsDir, "yielded-error-agent", {
      session_id: "moved-session",
      mode: "autonomous",
    });

    let attemptCount = 0;
    let lastResumeValue: string | undefined;

    const runtime = createMockRuntime(async function* (options) {
      attemptCount++;
      lastResumeValue = options.resume;

      if (attemptCount === 1 && options.resume) {
        // First attempt with resume — runtime YIELDS a terminal error
        // (does not throw), mirroring the CLI runtime's resume failure.
        yield {
          type: "error",
          message: "No conversation found with session ID: moved-session",
          code: "EXIT_1",
        };
        return;
      }

      // Retry with a fresh session succeeds.
      yield { type: "system", content: "Init", subtype: "init", session_id: "fresh-after-retry" };
      yield { type: "assistant", content: "Recovered" };
    });

    const executor = new JobExecutor(runtime, {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent({ name: "yielded-error-agent" }),
      prompt: "Test prompt",
      stateDir,
      resume: "moved-session",
    });

    // Recovered after retrying fresh.
    expect(result.success).toBe(true);
    expect(result.sessionId).toBe("fresh-after-retry");
    expect(attemptCount).toBe(2);
    expect(lastResumeValue).toBeUndefined();

    const output = await readJobOutputAll(join(stateDir, "jobs"), result.jobId);
    const retryMsg = output.find(
      (m) => m.type === "system" && m.content?.includes("Retrying with fresh session"),
    );
    expect(retryMsg).toBeDefined();
  });

  it("does not retry infinitely on persistent server errors", async () => {
    const sessionsDir = join(stateDir, "sessions");

    await updateSessionInfo(sessionsDir, "no-infinite-retry-agent", {
      session_id: "some-session",
      mode: "autonomous",
    });

    let attemptCount = 0;

    // Always throw session expired error
    const runtime = createMockRuntime(async function* (_options) {
      attemptCount++;
      throw new Error("Session expired on server");
    });

    const executor = new JobExecutor(runtime, {
      logger: createMockLogger(),
    });

    const result = await executor.execute({
      agent: createTestAgent({ name: "no-infinite-retry-agent" }),
      prompt: "Test prompt",
      stateDir,
      resume: "some-session",
    });

    // Should fail after at most 2 attempts (initial + 1 retry)
    expect(result.success).toBe(false);
    expect(attemptCount).toBeLessThanOrEqual(2);
    expect(result.error?.message).toContain("Session expired");
  });

  it("updates last_used_at before execution to prevent mid-job session expiry", async () => {
    const sessionsDir = join(stateDir, "sessions");

    // Create a session that is 23 hours old (close to default 24h expiry)
    const twentyThreeHoursAgo = new Date(Date.now() - 23 * 60 * 60 * 1000).toISOString();
    const { writeFile } = await import("node:fs/promises");
    const sessionPath = join(sessionsDir, "refresh-test-agent.json");
    const sessionData = {
      agent_name: "refresh-test-agent",
      session_id: "almost-expired-session",
      created_at: twentyThreeHoursAgo,
      last_used_at: twentyThreeHoursAgo,
      job_count: 5,
      mode: "autonomous",
    };
    await writeFile(sessionPath, JSON.stringify(sessionData));

    // Record when the test starts
    const testStartTime = Date.now();

    const runtime = createMockRuntime(async function* () {
      // Simulate a delay to ensure last_used_at was updated BEFORE we got here
      yield { type: "assistant", content: "Working..." };
    });

    const executor = new JobExecutor(runtime, {
      logger: createMockLogger(),
    });

    await executor.execute({
      agent: createTestAgent({ name: "refresh-test-agent" }),
      prompt: "Test prompt",
      stateDir,
      resume: "almost-expired-session",
    });

    // Check that last_used_at was updated to a recent time (not 23 hours ago)
    const sessionAfter = await getSessionInfo(sessionsDir, "refresh-test-agent");
    expect(sessionAfter).not.toBeNull();

    const lastUsedMs = new Date(sessionAfter!.last_used_at).getTime();
    // Should be updated to approximately now (within last few seconds)
    expect(lastUsedMs).toBeGreaterThanOrEqual(testStartTime - 1000);
    // And not still 23 hours ago
    expect(lastUsedMs).toBeGreaterThan(new Date(twentyThreeHoursAgo).getTime());
  });
});

// =============================================================================
// Session-backed (interactive) jobs
// =============================================================================

/**
 * Mock runtime that implements `openSession`. The session yields the supplied
 * messages, records every injected turn, and reports whether it was closed.
 */
function createMockSessionRuntime(
  messages: SDKMessage[],
  options: {
    hang?: boolean;
    /** Push follow-up stream messages when the caller injects a turn. */
    onSend?: (text: string, queue: MessageQueue<SDKMessage>) => void;
  } = {},
) {
  const sent: string[] = [];
  const state = { opened: 0, closed: 0, interrupted: 0, sent, lastOptions: undefined as any };

  // A real session's stream does not end after a result — it stays open until
  // close(). MessageQueue gives exactly that: it blocks until more is pushed or
  // the queue ends, so a missing terminal-break hangs the test instead of
  // passing by accident, and a follow-up turn can be delivered mid-drain.
  const queue = new MessageQueue<SDKMessage>();
  for (const message of messages) queue.push(message);

  const runtime: any = {
    execute: async function* () {
      throw new Error("execute() must not be used when openSession() is available");
    },
    openSession: (sessionOptions: any) => {
      state.opened++;
      state.lastOptions = sessionOptions;
      return {
        messages: (async function* () {
          // Mirror the real runtime: close() aborts the session controller and
          // the underlying query stream rejects.
          const aborted = new Promise<never>((_resolve, reject) => {
            sessionOptions.abortController?.signal.addEventListener(
              "abort",
              () => reject(new Error("The operation was aborted")),
              { once: true },
            );
          });
          aborted.catch(() => {});

          const iterator = queue[Symbol.asyncIterator]();
          for (;;) {
            const next = await Promise.race([iterator.next(), aborted]);
            if (next.done) return;
            yield next.value;
          }
        })(),
        send: async (text: string) => {
          sent.push(text);
          options.onSend?.(text, queue);
        },
        interrupt: async () => {
          state.interrupted++;
        },
        listCommands: async () => [],
        setModel: async () => {},
        close: async () => {
          state.closed++;
          // A "hanging" session ignores the input close and only dies on the
          // abort — the wedged-subprocess case the timeout must recover from.
          if (!options.hang) queue.end();
          sessionOptions.abortController?.abort();
        },
      };
    },
  };

  return { runtime, state, queue };
}

describe("JobExecutor session-backed jobs", () => {
  let tempDir: string;
  let stateDir: string;

  beforeEach(async () => {
    tempDir = await createTempDir();
    stateDir = join(tempDir, ".herdctl");
    await initStateDirectory({ path: stateDir });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("drives the run through openSession, drains to result, and closes it", async () => {
    const { runtime, state } = createMockSessionRuntime([
      { type: "system", subtype: "init", session_id: "session-abc" } as SDKMessage,
      { type: "assistant", content: "Working" },
      { type: "result", subtype: "success", result: "All done" } as SDKMessage,
    ]);

    let handle: unknown;
    const executor = new JobExecutor(runtime, { logger: createMockLogger() });

    const result = await executor.execute({
      agent: createTestAgent({ name: "interactive-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
      onSessionOpen: (session) => {
        handle = session;
      },
    });

    expect(state.opened).toBe(1);
    expect(state.closed).toBe(1);
    expect(handle).toBeDefined();
    expect(result.success).toBe(true);

    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.interactive).toBe(true);
    expect(job?.status).toBe("completed");
  });

  it("closes the session when the run fails", async () => {
    const { runtime, state } = createMockSessionRuntime([
      { type: "error", message: "boom" } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });

    const result = await executor.execute({
      agent: createTestAgent({ name: "failing-interactive-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
    });

    expect(result.success).toBe(false);
    expect(state.closed).toBe(1);
  });

  it("falls back to execute() when the runtime has no openSession", async () => {
    const runtime = createMockRuntimeWithMessages([
      { type: "system", content: "Initialized" },
      { type: "result", subtype: "success", result: "Done" } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });

    const result = await executor.execute({
      agent: createTestAgent({ name: "cli-style-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
    });

    expect(result.success).toBe(true);

    // Not an error, just not session-backed — and the job record says so.
    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.interactive).toBeFalsy();
  });

  it("uses execute() when interactive is not requested", async () => {
    const { runtime, state } = createMockSessionRuntime([]);
    runtime.execute = async function* () {
      yield { type: "result", subtype: "success", result: "Done" } as SDKMessage;
    };

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });

    const result = await executor.execute({
      agent: createTestAgent({ name: "batch-agent" }),
      prompt: "Test prompt",
      stateDir,
    });

    expect(result.success).toBe(true);
    expect(state.opened).toBe(0);
  });
});

describe("JobExecutor session-backed failure and control paths", () => {
  let tempDir: string;
  let stateDir: string;

  beforeEach(async () => {
    tempDir = await createTempDir();
    stateDir = join(tempDir, ".herdctl");
    await initStateDirectory({ path: stateDir });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("fails the job when no terminal result arrives before sessionTimeoutMs", async () => {
    // Only an init message — the terminal result never comes.
    const { runtime, state } = createMockSessionRuntime([
      { type: "system", subtype: "init", session_id: "stuck-session" } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "stuck-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
      sessionTimeoutMs: 50,
    });

    expect(result.success).toBe(false);
    expect(result.errorDetails?.message).toMatch(/sessionTimeoutMs/);
    expect(state.closed).toBeGreaterThanOrEqual(1);

    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.status).toBe("failed");
  });

  it("recovers from a session that only dies on abort", async () => {
    const { runtime } = createMockSessionRuntime(
      [{ type: "system", subtype: "init", session_id: "wedged" } as SDKMessage],
      { hang: true },
    );

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "wedged-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
      sessionTimeoutMs: 50,
    });

    expect(result.success).toBe(false);
    expect(result.errorDetails?.message).toMatch(/sessionTimeoutMs/);
  });

  it("fails a job whose result reports is_error", async () => {
    const { runtime } = createMockSessionRuntime([
      { type: "result", subtype: "success", is_error: true, result: "tool blew up" } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "is-error-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
    });

    expect(result.success).toBe(false);
    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.status).toBe("failed");
  });

  it("fails a job whose result subtype is not success", async () => {
    const runtime = createMockRuntimeWithMessages([
      { type: "result", subtype: "error_max_turns" } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "max-turns-agent" }),
      prompt: "Test prompt",
      stateDir,
    });

    expect(result.success).toBe(false);
    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.status).toBe("failed");
    expect(job?.exit_reason).not.toBe("success");
  });

  it("records an interrupted run as cancelled, not failed", async () => {
    // The interrupt lands while the run is streaming; the runtime answers with
    // the terminal error result a real interrupt produces.
    const { runtime, state } = createMockSessionRuntime([
      { type: "system", subtype: "init", session_id: "interrupt-me" } as SDKMessage,
      { type: "result", subtype: "error_during_execution", is_error: true } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "interrupted-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
      onSessionOpen: (handle) => {
        expect(handle.interrupt()).toBe(true);
      },
    });

    expect(state.interrupted).toBe(1);
    expect(result.success).toBe(false);

    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.status).toBe("cancelled");
    expect(job?.exit_reason).toBe("cancelled");
  });

  it("stops accepting input once the terminal message arrives", async () => {
    const { runtime, state } = createMockSessionRuntime([
      { type: "system", subtype: "init", session_id: "closing" } as SDKMessage,
      { type: "result", subtype: "success", result: "done" } as SDKMessage,
    ]);

    let handle: JobSessionHandle | undefined;
    const executor = new JobExecutor(runtime, { logger: createMockLogger() });

    const result = await executor.execute({
      agent: createTestAgent({ name: "teardown-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
      injectionGraceMs: 20,
      onSessionOpen: (h) => {
        handle = h;
        // Still live at this point.
        expect(h.send("early")).toBe(true);
      },
    });

    expect(result.success).toBe(true);
    // After the terminal message the queue is no longer drained, so a send must
    // report false rather than silently dropping the text.
    expect(handle?.send("late")).toBe(false);
    expect(handle?.interrupt()).toBe(false);
    expect(state.sent).toEqual(["early"]);
  });

  it("records an aborted session-backed run as cancelled and closes the session", async () => {
    const abortController = new AbortController();
    const { runtime, state } = createMockSessionRuntime([
      { type: "system", subtype: "init", session_id: "abort-me" } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const run = executor.execute({
      agent: createTestAgent({ name: "aborted-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
      abortController,
      sessionTimeoutMs: 10_000,
    });

    // The job's abort is forwarded into the session, ending its stream.
    await new Promise((resolve) => setTimeout(resolve, 20));
    abortController.abort();

    const result = await run;
    expect(result.success).toBe(false);
    expect(state.closed).toBeGreaterThanOrEqual(1);

    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.status).toBe("cancelled");
  });

  it("closes the failed session before retrying after session expiry", async () => {
    const { runtime, state } = createMockSessionRuntime([
      { type: "error", message: "No conversation found with session ID: abc" } as SDKMessage,
    ]);

    await updateSessionInfo(join(stateDir, "sessions"), "retry-agent", {
      session_id: "abc",
      mode: "autonomous",
    });

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    await executor.execute({
      agent: createTestAgent({ name: "retry-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
      resume: "abc",
    });

    // Two sessions opened (original + retry), and both were closed.
    expect(state.opened).toBe(2);
    expect(state.closed).toBe(2);
  });
});

describe("JobExecutor injected-input drain (AI-406)", () => {
  let tempDir: string;
  let stateDir: string;

  beforeEach(async () => {
    tempDir = await createTempDir();
    stateDir = join(tempDir, ".herdctl");
    await initStateDirectory({ path: stateDir });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("keeps draining when the host answers injected input in a NEXT turn", async () => {
    // Host behaviour observed on claude 2.1.220: the pushed message does not
    // join the running turn, it starts its own once the first one has ended.
    const { runtime, state } = createMockSessionRuntime(
      [
        { type: "system", subtype: "init", session_id: "next-turn" } as SDKMessage,
        { type: "result", subtype: "success", result: "first turn" } as SDKMessage,
      ],
      {
        onSend: (text, queue) => {
          queue.push({ type: "assistant", content: `acting on: ${text}` });
          queue.push({ type: "result", subtype: "success", result: "second turn" } as SDKMessage);
        },
      },
    );

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "next-turn-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
      injectionGraceMs: 5_000,
      onSessionOpen: (handle) => {
        expect(handle.send("do this instead")).toBe(true);
      },
    });

    expect(state.sent).toEqual(["do this instead"]);
    expect(result.success).toBe(true);
    // The job's outcome is the LAST result, not the one that ended turn one.
    expect(result.summary).toBe("second turn");

    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.status).toBe("completed");
  });

  it("recovers a failed first turn when the injected turn succeeds", async () => {
    const { runtime } = createMockSessionRuntime(
      [{ type: "result", subtype: "error_during_execution", is_error: true } as SDKMessage],
      {
        onSend: (_text, queue) => {
          queue.push({ type: "result", subtype: "success", result: "recovered" } as SDKMessage);
        },
      },
    );

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "recovering-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
      injectionGraceMs: 5_000,
      onSessionOpen: (handle) => {
        handle.send("try again");
      },
    });

    expect(result.success).toBe(true);
    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.status).toBe("completed");
  });

  it("closes with a warning when no follow-up turn arrives within the grace window", async () => {
    const warnings: string[] = [];
    const logger = {
      ...createMockLogger(),
      warn: (message: string) => warnings.push(message),
    };

    // No onSend: the injected message is never answered by a second turn (the
    // host folded it into the turn that just ended, or dropped it).
    const { runtime, state } = createMockSessionRuntime([
      { type: "system", subtype: "init", session_id: "no-follow-up" } as SDKMessage,
      { type: "result", subtype: "success", result: "only turn" } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger });
    const result = await executor.execute({
      agent: createTestAgent({ name: "grace-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
      injectionGraceMs: 40,
      onSessionOpen: (handle) => {
        handle.send("never answered");
      },
    });

    expect(state.closed).toBeGreaterThanOrEqual(1);
    expect(warnings.some((w) => /no follow-up turn within 40ms/.test(w))).toBe(true);
    // An expected ending, not a failure: the last result still decides.
    expect(result.success).toBe(true);

    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.status).toBe("completed");
  });
});

// =============================================================================
// Background-task hold on the interactive path (LZS-347)
// =============================================================================
//
// Mirrors `SDKRuntime.execute()`'s own bg-wait fix (issue #458,
// runtime/__tests__/sdk-runtime-bg-wait.test.ts) for the openSession()/
// interactive path used by Jira/GitHub jobs: a `run_in_background` Agent-tool
// child spawned mid-turn must not be abandoned just because the turn's
// terminal result arrived while the child was still running.
//
// Operator decision after job-2026-08-31-okhjlg (2026-08-31): the fixed
// CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS-based ceiling that used to force-close
// this hold was removed entirely — it killed a legitimate >10min Figma build
// mid-work and the job still reported "success" with the stale pre-kill
// result. This path now holds unconditionally for as long as a background
// task is live (the SDK re-invokes the session on drain and produces a fresh
// terminal; only stream end/abort/the sessionTimeoutMs backstop exits the
// loop). The truthful-close tests below cover the other half of that fix:
// whenever the run DOES end while work was still live, the summary/log line
// says so — a killed build can never read as a clean, complete result.

describe("JobExecutor background-task hold (LZS-347)", () => {
  let tempDir: string;
  let stateDir: string;

  beforeEach(async () => {
    tempDir = await createTempDir();
    stateDir = join(tempDir, ".herdctl");
    await initStateDirectory({ path: stateDir });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
    vi.useRealTimers();
  });

  it("keeps draining past a terminal with a live background task, and the later re-invocation result wins", async () => {
    const { runtime, state } = createMockSessionRuntime([
      {
        type: "system",
        subtype: "background_tasks_changed",
        tasks: [{ task_id: "bg-1", task_type: "agent", description: "child" }],
      } as unknown as SDKMessage,
      { type: "result", subtype: "success", result: "first turn" } as SDKMessage,
      // The child reports back: an empty background_tasks_changed drain is
      // not itself a terminal message, so the loop must not release here —
      // it keeps consuming until the agent's own fresh terminal arrives.
      { type: "system", subtype: "background_tasks_changed", tasks: [] } as unknown as SDKMessage,
      { type: "assistant", content: "child finished, wrapping up" },
      { type: "result", subtype: "success", result: "second turn" } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "bg-hold-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
    });

    expect(state.opened).toBe(1);
    expect(state.closed).toBe(1);
    expect(result.success).toBe(true);
    // The job's outcome is the LAST result, not the one held open while the
    // background task was still live.
    expect(result.summary).toBe("second turn");

    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.status).toBe("completed");
  });

  it("releases immediately when no background task is ever reported", async () => {
    const { runtime, state } = createMockSessionRuntime([
      { type: "system", subtype: "background_tasks_changed", tasks: [] } as unknown as SDKMessage,
      { type: "result", subtype: "success", result: "only turn" } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "no-bg-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
    });

    expect(state.opened).toBe(1);
    expect(state.closed).toBe(1);
    expect(result.success).toBe(true);
    expect(result.summary).toBe("only turn");
  });

  it("holds the interactive run open with NO time limit while a background task stays live", async () => {
    // Regression for job-2026-08-31-okhjlg: a legitimate long-running build
    // must never be force-closed just because it outlives some fixed window.
    // Real timers throughout — this run does real fs I/O (createJob,
    // appendJobOutput), which fake timers can't fast-forward, and unlike
    // SDKRuntime.execute()'s one-shot grace this path has NO timer at all
    // left in its "still live" branch (see job-executor.ts) — a short real
    // wait is exactly as conclusive as a long one, there is nothing timer-based
    // for the wait to race against any more.
    const { runtime, state } = createMockSessionRuntime([
      {
        type: "system",
        subtype: "background_tasks_changed",
        tasks: [{ task_id: "bg-1", task_type: "agent", description: "long build" }],
      } as unknown as SDKMessage,
      { type: "result", subtype: "success", result: "held result" } as SDKMessage,
      // Never drains — nothing should time this out on its own.
    ]);

    const abortController = new AbortController();
    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    let settled = false;
    const run = executor
      .execute({
        agent: createTestAgent({ name: "bg-hold-forever-agent" }),
        prompt: "Test prompt",
        stateDir,
        interactive: true,
        abortController,
      })
      .then((r) => {
        settled = true;
        return r;
      });

    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(settled).toBe(false);
    expect(state.closed).toBe(0);

    // Clean up: end the run via abort rather than leaving it dangling.
    abortController.abort();
    await run;
    expect(state.closed).toBeGreaterThanOrEqual(1);
  });

  it("does not close on injection-grace expiry while a background task is still live — holds indefinitely, no ceiling to defer to", async () => {
    // Regression for the CodeRabbit finding on the vulpes-pack sibling patch
    // (PR #181): a terminal with BOTH pendingInjected>0 AND a live background
    // task used to take the pendingInjected branch only, leaving the (now
    // removed) bg-wait ceiling unarmed — the injection-grace timer then
    // closed the session on its own once it expired, killing the still-live
    // child. Now there is no ceiling for anything to leave unarmed: the grace
    // expiry just holds, unconditionally, same as the main terminal-message
    // branch does.
    const warnings: string[] = [];
    const logger = {
      ...createMockLogger(),
      warn: (message: string) => warnings.push(message),
    };

    const { runtime, state, queue } = createMockSessionRuntime([
      {
        type: "system",
        subtype: "background_tasks_changed",
        tasks: [{ task_id: "bg-1", task_type: "agent", description: "stuck child" }],
      } as unknown as SDKMessage,
      { type: "result", subtype: "success", result: "held result" } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger });
    let settled = false;
    const run = executor
      .execute({
        agent: createTestAgent({ name: "bg-and-injected-agent" }),
        prompt: "Test prompt",
        stateDir,
        interactive: true,
        injectionGraceMs: 20,
        onSessionOpen: (handle) => {
          handle.send("never answered");
        },
      })
      .then((r) => {
        settled = true;
        return r;
      });

    // The grace window (20ms) expired but the run stayed open — held instead
    // of closing, because the background task was still live.
    await new Promise((resolve) => setTimeout(resolve, 60));
    expect(
      warnings.some((w) => /injection grace expired but a background task is still live/.test(w)),
    ).toBe(true);
    expect(settled).toBe(false);
    expect(state.closed).toBe(0);

    // Now let the child actually finish — the run should still complete
    // normally on that fresh result, not on the stale held one.
    queue.push({
      type: "system",
      subtype: "background_tasks_changed",
      tasks: [],
    } as unknown as SDKMessage);
    queue.push({ type: "result", subtype: "success", result: "second turn" } as SDKMessage);

    const result = await run;
    expect(result.success).toBe(true);
    expect(result.summary).toBe("second turn");
  });

  it("marks the job summary when aborted while a background task is still live", async () => {
    // Truthful-close half of the job-2026-08-31-okhjlg fix: a run that ends
    // (for any reason) while background work was live must never read as a
    // clean result from its summary alone.
    const abortController = new AbortController();
    const { runtime } = createMockSessionRuntime([
      {
        type: "system",
        subtype: "background_tasks_changed",
        tasks: [{ task_id: "bg-1", task_type: "agent", description: "stuck child" }],
      } as unknown as SDKMessage,
      { type: "result", subtype: "success", result: "held result" } as SDKMessage,
      // Never drains — the abort below is what ends the run.
    ]);

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const run = executor.execute({
      agent: createTestAgent({ name: "abort-under-work-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
      abortController,
      sessionTimeoutMs: 10_000,
    });

    await new Promise((resolve) => setTimeout(resolve, 20));
    abortController.abort();

    const result = await run;
    expect(result.success).toBe(false);
    expect(result.summary).toContain("background work terminated before completion");

    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.status).toBe("cancelled");
  });

  it("marks the job summary when the sessionTimeoutMs backstop fires while a background task is still live", async () => {
    // The much longer (default 2h), unrelated "genuinely stuck stream"
    // backstop can still fire while a task is legitimately live — it must be
    // just as truthful about it as an explicit abort.
    const { runtime } = createMockSessionRuntime([
      {
        type: "system",
        subtype: "background_tasks_changed",
        tasks: [{ task_id: "bg-1", task_type: "agent", description: "stuck child" }],
      } as unknown as SDKMessage,
      { type: "result", subtype: "success", result: "held result" } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "timeout-under-work-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
      sessionTimeoutMs: 30,
    });

    expect(result.success).toBe(false);
    expect(result.summary).toContain("background work terminated before completion");

    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.status).toBe("failed");
  });
});

// =============================================================================
// Undelivered task_notification reinvocation (vulpes-pack#244, job-2026-09-01-yqhqzn)
// =============================================================================
//
// A background task can start AND drain to completion — its `task_notification`
// included — entirely WITHIN the turn that dispatched it, before that turn's
// own terminal. The LZS-347 hold above only covers a task still `live` AT the
// terminal; a task that already finished leaves `liveBackgroundTasks` empty,
// so the terminal used to be read as a genuine end and the notification's
// content was silently lost — the model never got a turn to act on it.

describe("JobExecutor undelivered-notification reinvocation (vulpes-pack#244)", () => {
  let tempDir: string;
  let stateDir: string;

  beforeEach(async () => {
    tempDir = await createTempDir();
    stateDir = join(tempDir, ".herdctl");
    await initStateDirectory({ path: stateDir });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("re-invokes once when a task_notification lands before its own turn's terminal, and the follow-up turn's result wins", async () => {
    const { runtime, state, queue } = createMockSessionRuntime([
      {
        type: "system",
        subtype: "background_tasks_changed",
        tasks: [{ task_id: "wf-1", task_type: "agent", description: "workflow" }],
      } as unknown as SDKMessage,
      // The task drains AND its notification lands entirely within this turn,
      // before the turn's own terminal — the model can't have consumed it.
      { type: "system", subtype: "background_tasks_changed", tasks: [] } as unknown as SDKMessage,
      {
        type: "system",
        subtype: "task_notification",
        task_id: "wf-1",
        status: "completed",
      } as unknown as SDKMessage,
      { type: "result", subtype: "success", result: "first turn (unaware)" } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const run = executor.execute({
      agent: createTestAgent({ name: "notif-reinvoke-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
      injectionGraceMs: 500,
    });

    // Give the reinvocation grace time to arm, then deliver the follow-up
    // turn the SDK produces once it reinvokes with the notification as input.
    await new Promise((resolve) => setTimeout(resolve, 20));
    queue.push({
      type: "assistant",
      content: "picked up the notification",
    } as unknown as SDKMessage);
    queue.push({
      type: "result",
      subtype: "success",
      result: "second turn (delivered)",
    } as SDKMessage);

    const result = await run;
    expect(result.success).toBe(true);
    // The job's outcome is the follow-up turn, not the stale one held while
    // the notification was still undelivered.
    expect(result.summary).toBe("second turn (delivered)");
    expect(state.opened).toBe(1);
    expect(state.closed).toBe(1);

    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.status).toBe("completed");
  });

  it("closes cleanly with no loop when the reinvocation grace elapses without a new notification", async () => {
    const warnings: string[] = [];
    const logger = {
      ...createMockLogger(),
      warn: (message: string) => warnings.push(message),
    };

    // No further queue pushes: the follow-up turn never actually shows up
    // (an isolated/irrelevant notification, or the model genuinely had
    // nothing left to say about it).
    const { runtime, state } = createMockSessionRuntime([
      { type: "system", subtype: "background_tasks_changed", tasks: [] } as unknown as SDKMessage,
      {
        type: "system",
        subtype: "task_notification",
        task_id: "wf-1",
        status: "completed",
      } as unknown as SDKMessage,
      { type: "result", subtype: "success", result: "only turn" } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger });
    const result = await executor.execute({
      agent: createTestAgent({ name: "notif-no-followup-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
      injectionGraceMs: 40,
    });

    expect(
      warnings.some((w) =>
        /no follow-up turn within 40ms after an undelivered task_notification/.test(w),
      ),
    ).toBe(true);
    // An expected ending, not a failure: the held result still decides, and
    // the session closes exactly once — no repeated reinvocation attempts.
    expect(result.success).toBe(true);
    expect(result.summary).toBe("only turn");
    expect(state.closed).toBe(1);

    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.status).toBe("completed");
  });

  it("does not arm an extra reinvocation when the notification is already delivered by a normal follow-up turn", async () => {
    const warnings: string[] = [];
    const logger = {
      ...createMockLogger(),
      warn: (message: string) => warnings.push(message),
    };

    // The notification lands while a DIFFERENT task is still live, so the
    // existing live-background-task hold (LZS-347) — not the new
    // notification grace — is what keeps the loop draining. By the time the
    // follow-up turn's own assistant message arrives, the notification is
    // already delivered (a new turn started), so the second turn's clean
    // terminal must not trigger a further, redundant reinvocation attempt.
    const { runtime, state } = createMockSessionRuntime([
      {
        type: "system",
        subtype: "background_tasks_changed",
        tasks: [{ task_id: "t1" }],
      } as unknown as SDKMessage,
      {
        type: "system",
        subtype: "task_notification",
        task_id: "wf-done-early",
        status: "completed",
      } as unknown as SDKMessage,
      { type: "result", subtype: "success", result: "first turn" } as SDKMessage,
      { type: "system", subtype: "background_tasks_changed", tasks: [] } as unknown as SDKMessage,
      { type: "assistant", content: "wrapping up" },
      { type: "result", subtype: "success", result: "second turn" } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger });
    const result = await executor.execute({
      agent: createTestAgent({ name: "notif-already-delivered-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
      injectionGraceMs: 30,
    });

    expect(result.success).toBe(true);
    expect(result.summary).toBe("second turn");
    expect(state.opened).toBe(1);
    expect(state.closed).toBe(1);
    // No notification-reinvocation grace was ever needed — the whole run
    // resolved off the preloaded queue, no grace timer had to fire.
    expect(
      warnings.some((w) => /notification-reinvocation|undelivered task_notification/.test(w)),
    ).toBe(false);
  });
});

// =============================================================================
// Empty-resume retry and dirty-marking (vulpes-pack#206, job-w11ho7)
// =============================================================================
//
// Resuming a session that was hard-closed while a background task was live
// can replay a stale `task_notification` straight into an immediate empty
// terminal result, before the injected prompt's own turn — job-executor used
// to treat that first terminal as the whole job and report a clean "success"
// with zero assistant turns and zero cost, silently swallowing the prompt.

describe("empty-resume retry (vulpes-pack#206)", () => {
  let tempDir: string;
  let stateDir: string;

  beforeEach(async () => {
    tempDir = await createTempDir();
    stateDir = join(tempDir, ".herdctl");
    await initStateDirectory({ path: stateDir });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("retries once on the SAME session when a resume emits an immediate empty terminal result", async () => {
    const sessionsDir = join(stateDir, "sessions");
    await updateSessionInfo(sessionsDir, "zombie-agent", {
      session_id: "zombie-session-id",
      mode: "autonomous",
    });

    const resumes: Array<string | undefined> = [];
    let resumeCallCount = 0;
    const runtime = createMockRuntime(async function* (options: { resume?: string }) {
      resumes.push(options.resume);
      if (options.resume) {
        resumeCallCount++;
        if (resumeCallCount === 1) {
          // The first resume: a stale task_notification replayed straight
          // into an empty terminal — zero assistant messages, zero turns,
          // no error at all.
          yield {
            type: "system",
            subtype: "task_notification",
            task_id: "t1",
          } as unknown as SDKMessage;
          yield { type: "result", subtype: "success", result: "" } as SDKMessage;
          return;
        }
        // The second resume of the SAME session: the notification has
        // already been replayed, so this turn answers the actual prompt.
        yield { type: "assistant", content: "Actually did the work" };
        yield {
          type: "result",
          subtype: "success",
          result: "Actually did the work",
          num_turns: 1,
        } as SDKMessage;
        return;
      }
      throw new Error("should not fall back to a fresh session in this test");
    });

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "zombie-agent" }),
      prompt: "carry on",
      stateDir,
      resume: "zombie-session-id",
    });

    expect(resumes).toEqual(["zombie-session-id", "zombie-session-id"]);
    expect(result.success).toBe(true);
    expect(result.summary).toBe("Actually did the work");

    // The session pointer must survive a same-session retry — clearing it
    // would destroy any Discord/channel context tied to that session.
    const sessionInfo = await getSessionInfo(sessionsDir, "zombie-agent");
    expect(sessionInfo?.session_id).toBe("zombie-session-id");
  });

  it("falls back to a fresh session when the same-session retry is also empty", async () => {
    const sessionsDir = join(stateDir, "sessions");
    await updateSessionInfo(sessionsDir, "double-zombie-agent", {
      session_id: "zombie-session-id",
      mode: "autonomous",
    });

    const resumes: Array<string | undefined> = [];
    const runtime = createMockRuntime(async function* (options: { resume?: string }) {
      resumes.push(options.resume);
      if (options.resume) {
        // Empty on both resume attempts (same session, twice) — a genuine
        // zombie, not a stale-notification replay.
        yield { type: "result", subtype: "success", result: "" } as SDKMessage;
      } else {
        // The fresh-session fallback: a real turn.
        yield { type: "assistant", content: "Actually did the work" };
        yield {
          type: "result",
          subtype: "success",
          result: "Actually did the work",
          num_turns: 1,
        } as SDKMessage;
      }
    });

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "double-zombie-agent" }),
      prompt: "carry on",
      stateDir,
      resume: "zombie-session-id",
    });

    // Same session retried once, then falls back to a fresh session.
    expect(resumes).toEqual(["zombie-session-id", "zombie-session-id", undefined]);
    expect(result.success).toBe(true);
    expect(result.summary).toBe("Actually did the work");

    // The fresh-session fallback DOES clear the stale pointer, unlike the
    // same-session retry above.
    const sessionInfo = await getSessionInfo(sessionsDir, "double-zombie-agent");
    expect(sessionInfo).toBeNull();
  });

  it("fails loudly (not success) when the fresh-session retry also produces zero assistant turns", async () => {
    const sessionsDir = join(stateDir, "sessions");
    await updateSessionInfo(sessionsDir, "triple-zombie-agent", {
      session_id: "zombie-session-id",
      mode: "autonomous",
    });

    const resumes: Array<string | undefined> = [];
    const runtime = createMockRuntime(async function* (options: { resume?: string }) {
      resumes.push(options.resume);
      // Empty every time — same session twice, then the fresh retry is
      // just as much a dead end.
      yield { type: "result", subtype: "success", result: "" } as SDKMessage;
    });

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "triple-zombie-agent" }),
      prompt: "carry on",
      stateDir,
      resume: "zombie-session-id",
    });

    // Three empty attempts: same session, same session again, then fresh.
    expect(resumes).toEqual(["zombie-session-id", "zombie-session-id", undefined]);
    expect(result.success).toBe(false);
    expect(result.error?.message).toContain("zero assistant turns");

    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.status).toBe("failed");
    expect(job?.exit_reason).toBe("error");
  });

  it("does not retry a fresh (non-resumed) run that legitimately produces zero turns", async () => {
    // Not this bug's failure mode — `resumeSessionId` is unset here, so the
    // empty-resume path must never engage at all.
    const runtime = createMockRuntimeWithMessages([
      { type: "result", subtype: "success", result: "" } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "fresh-empty-agent" }),
      prompt: "Test prompt",
      stateDir,
    });

    expect(result.success).toBe(true);
  });
});

describe("dirty-marking on hard close (vulpes-pack#206)", () => {
  let tempDir: string;
  let stateDir: string;

  beforeEach(async () => {
    tempDir = await createTempDir();
    stateDir = join(tempDir, ".herdctl");
    await initStateDirectory({ path: stateDir });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("clears the session pointer when the sessionTimeoutMs backstop fires while a background task is still live", async () => {
    const sessionsDir = join(stateDir, "sessions");
    await updateSessionInfo(sessionsDir, "dirty-close-agent", {
      session_id: "pre-existing-session",
      mode: "autonomous",
    });
    const before = await getSessionInfo(sessionsDir, "dirty-close-agent");
    expect(before).not.toBeNull();

    const { runtime } = createMockSessionRuntime([
      {
        type: "system",
        subtype: "background_tasks_changed",
        tasks: [{ task_id: "bg-1", task_type: "agent", description: "stuck child" }],
      } as unknown as SDKMessage,
      { type: "result", subtype: "success", result: "held result" } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "dirty-close-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
      sessionTimeoutMs: 30,
    });

    expect(result.success).toBe(false);

    // The next trigger must start fresh instead of resuming a zombie —
    // the pointer this run left dirty is gone, not silently republished.
    const after = await getSessionInfo(sessionsDir, "dirty-close-agent");
    expect(after).toBeNull();
  });

  it("does not clear the session pointer on a clean close with no live background work", async () => {
    const sessionsDir = join(stateDir, "sessions");

    const { runtime } = createMockSessionRuntime([
      { type: "system", subtype: "init", session_id: "clean-session" } as SDKMessage,
      { type: "result", subtype: "success", result: "all done", num_turns: 1 } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "clean-close-agent" }),
      prompt: "Test prompt",
      stateDir,
      interactive: true,
    });

    expect(result.success).toBe(true);
    const after = await getSessionInfo(sessionsDir, "clean-close-agent");
    expect(after).not.toBeNull();
    expect(after?.session_id).toBe("clean-session");
  });
});

// =============================================================================
// Non-interactive composition with an un-drained background task
// (vulpes-pack#206 follow-up: adversarial 13-agent review of vulpes-v2)
// =============================================================================
//
// The review noted the existing truthful-close/marker coverage only tested
// the interactive (openSession) path — this pins the plain execute() path,
// mirroring what SDKRuntime.execute()'s own maxHoldPromise backstop (BLOCKER
// 1) produces when a background child never reports drained: instead of
// hanging forever, a live background_tasks_changed followed by a synthetic
// terminal error. JobExecutor needs no special-case code for this — the same
// truthful-close marking that handles any other stream-level error already
// covers it, since the plain execute() path shares this loop with openSession.

describe("JobExecutor non-interactive composition with an un-drained background task (vulpes-pack#206 review)", () => {
  let tempDir: string;
  let stateDir: string;

  beforeEach(async () => {
    tempDir = await createTempDir();
    stateDir = join(tempDir, ".herdctl");
    await initStateDirectory({ path: stateDir });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("marks the job when the underlying runtime forcibly ends a non-interactive run over an un-drained background task", async () => {
    const runtime = createMockRuntimeWithMessages([
      {
        type: "system",
        subtype: "background_tasks_changed",
        tasks: [{ task_id: "bg-1" }],
      } as unknown as SDKMessage,
      {
        type: "error",
        message: "execute() timed out after holding open 7200000ms without concluding",
        code: "MAX_HOLD_ELAPSED",
      } as SDKMessage,
    ]);

    const executor = new JobExecutor(runtime, { logger: createMockLogger() });
    const result = await executor.execute({
      agent: createTestAgent({ name: "undrained-noninteractive-agent" }),
      prompt: "Test prompt",
      stateDir,
      // `interactive` intentionally omitted — the plain execute() path.
    });

    expect(result.success).toBe(false);
    expect(result.summary).toContain("background work terminated before completion");

    const job = await getJob(join(stateDir, "jobs"), result.jobId);
    expect(job?.status).toBe("failed");
  });
});
