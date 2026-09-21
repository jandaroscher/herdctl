/**
 * Job executor for running agents with streaming output to job logs
 *
 * Manages the lifecycle of agent execution including:
 * - Creating job records before execution
 * - Streaming all SDK messages to job output in real-time
 * - Updating job status and metadata on completion
 */

import { appendFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { resolveWorkingDirectory } from "../fleet-manager/working-directory-helper.js";
import {
  appendJobOutput,
  clearSession,
  cliSessionFileExists,
  createJob,
  getJobOutputPath,
  getSessionInfo,
  isSessionExpiredError,
  isTokenExpiredError,
  type JobMetadata,
  type RunUsage,
  type TriggerType,
  toSafeIdentifier,
  updateJob,
  updateSessionInfo,
  validateRuntimeContext,
  validateWorkingDirectory,
} from "../state/index.js";
import { createLogger } from "../utils/logger.js";
import {
  buildErrorMessage,
  classifyError,
  MalformedResponseError,
  type RunnerError,
  SDKInitializationError,
  SDKStreamingError,
  wrapError,
} from "./errors.js";
import {
  extractRunUsage,
  extractSummary,
  isErrorResult,
  isTerminalMessage,
  processSDKMessage,
} from "./message-processor.js";
import type { RuntimeInterface, RuntimeSession } from "./runtime/index.js";
import type {
  ProcessedMessage,
  RunnerErrorDetails,
  RunnerOptionsWithCallbacks,
  RunnerResult,
  SDKMessage,
} from "./types.js";
import { DEFAULT_INJECTION_GRACE_MS, DEFAULT_SESSION_TIMEOUT_MS } from "./types.js";

// =============================================================================
// Types
// =============================================================================

/**
 * Logger interface for job executor
 */
export interface JobExecutorLogger {
  warn: (message: string) => void;
  error: (message: string) => void;
  info?: (message: string) => void;
  debug?: (message: string) => void;
}

/**
 * Options for job executor
 */
export interface JobExecutorOptions {
  /** Logger for warnings and errors */
  logger?: JobExecutorLogger;
}

/**
 * SDK query function type (for dependency injection)
 * @deprecated Use RuntimeInterface instead. This type is kept for test compatibility.
 */
export type SDKQueryFunction = (params: {
  prompt: string;
  options?: Record<string, unknown>;
  abortController?: AbortController;
}) => AsyncIterable<SDKMessage>;

// =============================================================================
// Default Logger
// =============================================================================

const defaultLogger: JobExecutorLogger = createLogger("JobExecutor");

// =============================================================================
// Job Executor Class
// =============================================================================

/**
 * Executes agents with streaming output to job logs
 *
 * This class manages the complete lifecycle of agent execution:
 * 1. Creates a job record before starting
 * 2. Updates job status to 'running'
 * 3. Streams all SDK messages to job output in real-time
 * 4. Updates job with final status on completion
 *
 * @example
 * ```typescript
 * const runtime = RuntimeFactory.create(agent);
 * const executor = new JobExecutor(runtime);
 *
 * const result = await executor.execute({
 *   agent: resolvedAgent,
 *   prompt: "Fix the bug in auth.ts",
 *   stateDir: "/path/to/.herdctl",
 *   triggerType: "manual",
 * });
 *
 * console.log(`Job ${result.jobId} completed: ${result.success}`);
 * ```
 */
export class JobExecutor {
  private runtime: RuntimeInterface;
  private logger: JobExecutorLogger;

  /**
   * Create a new job executor
   *
   * @param runtime - The runtime interface to use for agent execution
   * @param options - Optional configuration
   */
  constructor(runtime: RuntimeInterface, options: JobExecutorOptions = {}) {
    this.runtime = runtime;
    this.logger = options.logger ?? defaultLogger;
  }

  /**
   * Execute an agent and stream output to job log
   *
   * @param options - Runner options including agent config and prompt
   * @returns Result of the execution including job ID and status
   */
  async execute(options: RunnerOptionsWithCallbacks): Promise<RunnerResult> {
    const {
      agent,
      prompt,
      stateDir,
      triggerType,
      schedule,
      onMessage,
      onJobCreated,
      outputToFile,
    } = options;

    // Session identity for this run. Defaults to the agent's qualified name (one
    // session per agent); callers can scope it narrower (e.g. one session per work
    // item) by passing `sessionKey`. Every session read/write below goes through
    // this, while the job record keeps `agent.qualifiedName` so jobs stay grouped
    // by agent regardless of session scoping.
    // Sanitized, not trusted: session storage rejects anything outside
    // SAFE_IDENTIFIER_PATTERN with a PathTraversalError, and by the time the first
    // session read happens the job record already exists — an unsanitized key
    // (e.g. "owner/repo#12") would strand an orphan job.
    const sessionKey = toSafeIdentifier(options.sessionKey ?? agent.qualifiedName);

    // Does this run actually get a long-lived streaming session? Only when the
    // caller asked AND the runtime can provide one (CLI/Docker cannot) — the
    // fallback is silent, so `interactive` is safe to set unconditionally. The
    // resolved value is recorded on the job so an out-of-process receiver can
    // tell which jobs accept injected messages.
    const openSession =
      options.interactive === true ? this.runtime.openSession?.bind(this.runtime) : undefined;
    const sessionBacked = openSession !== undefined;

    const jobsDir = join(stateDir, "jobs");
    let job: JobMetadata;
    let sessionId: string | undefined;
    let summary: string | undefined;
    let lastAssistantContent: string | undefined; // Track last assistant message for fallback summary
    let runUsage: RunUsage | undefined; // Per-run per-model token accounting from the terminal result message
    let lastError: RunnerError | undefined;
    let errorDetails: RunnerErrorDetails | undefined;
    let messagesReceived = 0;
    // Set when a caller interrupts the run through its session handle — the run
    // is then recorded as cancelled rather than failed.
    let interrupted = false;
    // Set when the run ends (abort, the sessionTimeoutMs backstop, or a
    // stream-level error) while a `run_in_background` child was still live per
    // the last known snapshot — so a killed-mid-work run never reads as a clean
    // completion from its summary/log line alone (job-2026-08-31-okhjlg: the
    // old bg-wait ceiling closed a session out from under a live Figma build
    // and the job recorded "success" with the stale pre-kill summary). Reset
    // per retry attempt below — reflects only the final attempt's outcome.
    let endedWithLiveBackgroundTasks = false;
    const sessionTimeoutMs = options.sessionTimeoutMs ?? DEFAULT_SESSION_TIMEOUT_MS;
    const injectionGraceMs = options.injectionGraceMs ?? DEFAULT_INJECTION_GRACE_MS;
    let outputLogPath: string | undefined;

    // Determine trigger type: use 'fork' if forking, otherwise use provided or default to 'manual'
    const effectiveTriggerType: TriggerType = options.fork
      ? "fork"
      : ((triggerType ?? "manual") as TriggerType);

    // Step 1: Create job record
    try {
      job = await createJob(jobsDir, {
        agent: agent.qualifiedName,
        trigger_type: effectiveTriggerType,
        prompt,
        schedule,
        forked_from: options.fork ? options.forkedFrom : undefined,
        interactive: sessionBacked ? true : undefined,
      });

      this.logger.info?.(`Created job ${job.id} for agent ${agent.name}`);

      // Notify caller of job ID immediately (before execution starts).
      // Pass the freshly-created record (status `pending`) and await the
      // callback so consumers can emit `job:created` up front — guaranteed to
      // complete before any `onMessage`/`job:output` streams below.
      if (onJobCreated) {
        await onJobCreated(job.id, job);
      }
    } catch (error) {
      this.logger.error(`Failed to create job: ${(error as Error).message}`);
      throw error;
    }

    // Step 2: Setup output log file if outputToFile is enabled
    if (outputToFile) {
      try {
        const jobOutputDir = join(jobsDir, job.id);
        await mkdir(jobOutputDir, { recursive: true });
        outputLogPath = join(jobOutputDir, "output.log");
        this.logger.info?.(`Output logging enabled for job ${job.id} at ${outputLogPath}`);
      } catch (error) {
        this.logger.warn(`Failed to create job output directory: ${(error as Error).message}`);
        // Continue execution - output logging is optional
      }
    }

    // Step 3: Update job status to 'running'
    try {
      await updateJob(jobsDir, job.id, {
        status: "running",
      });
    } catch (error) {
      this.logger.warn(`Failed to update job status to running: ${(error as Error).message}`);
      // Continue execution - job was created
    }

    // Step 3.5: Validate session if resuming
    // This prevents unexpected logouts by checking session expiration before attempting resume
    // Pass timeout to getSessionInfo so expired sessions are automatically cleared (consistent with schedule-runner.ts)
    let effectiveResume: string | undefined;
    if (options.resume) {
      const sessionsDir = join(stateDir, "sessions");
      // Default to 24h if not configured - prevents unexpected logouts from expired server-side sessions
      const sessionTimeout = agent.session?.timeout ?? "24h";

      // Read the agent-level session pointer WITHOUT applying the timeout, so we
      // can distinguish "this agent never owned a session" (adoption candidate,
      // issue #263) from "this agent had a session that just expired and was
      // cleared" (must start fresh). The timeout-aware read below deletes expired
      // files, which would otherwise make both cases look identical (null).
      const hadAgentSession = (await getSessionInfo(sessionsDir, sessionKey)) !== null;

      const existingSession = await getSessionInfo(sessionsDir, sessionKey, {
        timeout: sessionTimeout,
        logger: this.logger,
        runtime: agent.runtime ?? "sdk", // Pass runtime for correct validation
        // For the `cli` runtime that validation probes the filesystem, so it must
        // look in the home our runtime actually uses. Re-deriving `~/.claude`
        // here reports a valid session as `file_not_found` — and this call
        // CLEARS sessions it judges stale (herdctl#423).
        claudeHomePath: this.runtime.getClaudeHomePath?.(),
      });

      if (existingSession?.session_id && existingSession.session_id !== options.resume) {
        // Caller provided a different session ID than what's stored on disk for this agent.
        // This happens with per-thread Slack sessions — the caller manages session IDs
        // externally and passes the correct one for this specific thread.
        // Trust the caller's session ID directly; the agent-level session is irrelevant.
        effectiveResume = options.resume;
        this.logger.debug?.(
          `Using caller-provided session for ${agent.qualifiedName}: ${effectiveResume} (differs from agent-level session ${existingSession.session_id})`,
        );
      } else if (existingSession?.session_id) {
        // Caller's session matches the agent-level session — validate working dir and runtime
        const currentWorkingDirectory = resolveWorkingDirectory(agent);
        const wdValidation = validateWorkingDirectory(existingSession, currentWorkingDirectory);

        if (!wdValidation.valid) {
          this.logger.warn(
            `${wdValidation.message} - clearing stale session ${existingSession.session_id}`,
          );
          try {
            await clearSession(sessionsDir, sessionKey);
          } catch (clearError) {
            this.logger.warn(`Failed to clear stale session: ${(clearError as Error).message}`);
          }
          // Continue without resume - working directory changed
          effectiveResume = undefined;
        } else {
          // Validate that the runtime context hasn't changed since the session was created
          // Sessions are tied to specific runtime configurations (SDK vs CLI, Docker vs native)
          const currentRuntimeType = (agent.runtime as "sdk" | "cli") ?? "sdk";
          const currentDockerEnabled = agent.docker?.enabled ?? false;
          const runtimeValidation = validateRuntimeContext(
            existingSession,
            currentRuntimeType,
            currentDockerEnabled,
          );

          if (!runtimeValidation.valid) {
            this.logger.warn(
              `${runtimeValidation.message} - clearing stale session ${existingSession.session_id}`,
            );
            try {
              await clearSession(sessionsDir, sessionKey);
            } catch (clearError) {
              this.logger.warn(`Failed to clear stale session: ${(clearError as Error).message}`);
            }
            // Continue without resume - runtime context changed
            effectiveResume = undefined;
          } else {
            // Use the actual session ID from the stored session, not the original options.resume value
            // This ensures we always use the correct session ID stored on disk
            effectiveResume = existingSession.session_id;
            this.logger.info?.(
              `Found valid session for ${agent.qualifiedName}: ${effectiveResume}, will attempt to resume`,
            );

            // Update last_used_at NOW to prevent session from expiring during long-running jobs
            // This fixes the authentication bug where sessions could expire mid-execution
            try {
              await updateSessionInfo(sessionsDir, sessionKey, {
                session_id: existingSession.session_id,
                job_count: existingSession.job_count,
                mode: existingSession.mode,
                working_directory: currentWorkingDirectory,
                runtime_type: (agent.runtime as "sdk" | "cli") ?? "sdk",
                docker_enabled: agent.docker?.enabled ?? false,
              });
              this.logger.info?.(`Refreshed session timestamp for ${agent.name} before execution`);
            } catch (updateError) {
              this.logger.warn(
                `Failed to refresh session timestamp: ${(updateError as Error).message}`,
              );
              // Continue anyway - the session is still valid for now
            }
          }
        }
      } else {
        // No agent-level session pointer exists on disk for this agent. The
        // caller still passed an explicit `resume` session ID — this is an
        // authoritative request to continue a specific transcript that this
        // agent doesn't (yet) own. The most common case is "adopting" a session
        // created by a *different* agent in the same process: the transcript has
        // been relocated into this agent's working directory and re-attributed,
        // but this agent never created an agent-level session file of its own.
        //
        // Historically this branch silently dropped `options.resume` and started
        // fresh, which made the runtime fork a brand-new session instead of
        // continuing (issue #263 — cross-agent / runtime-added resume). We now
        // honor the caller's explicit session ID so a same-process resume reads
        // the transcript straight from disk, exactly as a process restart would.
        //
        // For the CLI runtime, the transcript must physically exist in the
        // agent's working directory (Claude Code keys session storage by spawn
        // cwd). If it's missing, resuming would fail anyway, so we fall back to a
        // fresh session. For the SDK runtime we trust the caller's ID directly
        // (the SDK owns session storage and there's no reliable file to probe).
        const currentRuntimeType = (agent.runtime as "sdk" | "cli") ?? "sdk";
        const currentWorkingDirectory = resolveWorkingDirectory(agent);
        const dockerEnabled = agent.docker?.enabled ?? false;

        // Only adopt when this agent has NEVER owned an agent-level session. If a
        // session file existed (now cleared by the timeout-aware read above), the
        // caller is trying to resume a session that just expired for this agent —
        // that must start fresh, not be force-adopted.
        let adopt = !hadAgentSession && (currentRuntimeType !== "cli" || dockerEnabled);
        if (
          !hadAgentSession &&
          currentRuntimeType === "cli" &&
          !dockerEnabled &&
          currentWorkingDirectory
        ) {
          // Native CLI: only adopt if the transcript exists where Claude Code
          // will look for it (the agent's working directory, under the home this
          // runtime resolved — NOT an assumed `~/.claude`, herdctl#423).
          adopt = await cliSessionFileExists(
            currentWorkingDirectory,
            options.resume,
            this.runtime.getClaudeHomePath?.(),
          );
        }

        if (adopt) {
          effectiveResume = options.resume;
          this.logger.info?.(
            `Adopting caller-provided session for ${agent.qualifiedName}: ${effectiveResume} ` +
              `(no agent-level session on disk; resuming explicitly requested transcript)`,
          );

          // Persist an agent-level session pointer so subsequent runs (and
          // restarts) treat this session as owned by this agent. Best-effort:
          // a failure here doesn't block the resume we're about to attempt.
          try {
            const sessionsDir = join(stateDir, "sessions");
            await updateSessionInfo(sessionsDir, sessionKey, {
              session_id: options.resume,
              mode: "autonomous",
              working_directory: currentWorkingDirectory,
              runtime_type: currentRuntimeType,
              docker_enabled: dockerEnabled,
            });
          } catch (adoptError) {
            this.logger.warn(
              `Failed to persist adopted session pointer: ${(adoptError as Error).message}`,
            );
          }
        } else {
          this.logger.info?.(
            `No valid session for ${agent.name} (expired or not found), starting fresh`,
          );

          // Write info to job output
          try {
            await appendJobOutput(jobsDir, job.id, {
              type: "system",
              content: `No valid session found (expired or missing). Starting fresh session.`,
            });
          } catch {
            // Ignore output write failures
          }

          // Don't resume - start fresh (effectiveResume stays undefined)
        }
      }
    }

    // Forking: resume the explicit source session but tell the runtime to fork
    // (write new turns to a brand-new session id). We deliberately bypass the
    // agent-level session adoption/validation above — the caller named a
    // specific source to fork, not the agent's own stored session — and let the
    // runtime find the source transcript by id under the agent's cwd.
    if (options.fork) {
      effectiveResume = options.fork;
    }

    // Step 4: Execute agent and stream output
    // Track whether we've already retried after a session expiration or token expiry
    let retriedAfterSessionExpiry = false;
    let retriedAfterTokenExpiry = false;
    // Track whether we've already retried a resume that produced zero
    // assistant turns (see `assistantMessageCount` below, vulpes-pack#206).
    let retriedAfterEmptyResume = false;
    // Track whether we've already retried an empty resume on the SAME
    // session id before falling back to a fresh one (vulpes-pack#626).
    let retriedSameSessionAfterEmptyResume = false;

    const executeWithRetry = async (resumeSessionId: string | undefined): Promise<void> => {
      // Session-backed runs hold the handle here so the retry paths and the
      // finally below can tear it down; the `execute()` path leaves it undefined.
      let session: RuntimeSession | undefined;
      // Whether the run is still consuming its input queue. Flipped to false
      // SYNCHRONOUSLY the moment the drain loop stops, before any await: pushing
      // into an ended MessageQueue is a silent no-op, so a `send` that lands in
      // the teardown window must report false rather than pretend delivery.
      let acceptingInput = false;
      let timedOut = false;
      let drainTimer: ReturnType<typeof setTimeout> | undefined;
      // Messages handed to the session that no turn has answered yet. Nonzero
      // means the drain loop must survive the next terminal result (see there).
      let pendingInjected = 0;
      // Set when the grace window below expired: an expected ending, not a
      // failure, so the last result's outcome stands.
      let graceClosed = false;
      let graceTimer: ReturnType<typeof setTimeout> | undefined;

      const clearGrace = (): void => {
        if (graceTimer) {
          clearTimeout(graceTimer);
          graceTimer = undefined;
        }
      };

      // Tracks the `background_tasks_changed` stream signal so a live
      // `run_in_background` child isn't abandoned when its parent turn's
      // terminal result arrives — same signal `SDKRuntime.execute()`'s own
      // bg-wait fix reads. See the terminal-message branch below.
      //
      // No ceiling defers to here any more (operator decision after
      // job-2026-08-31-okhjlg killed a legitimate Figma build mid-work): a
      // live background task holds the run open unconditionally. The
      // SessionReaper is the policy for when a session with no *pending job*
      // holding it should eventually close; this loop just keeps draining for
      // as long as the child is live, same as the reaper's own "keepAlive
      // while background work runs, no timer" rule.
      let liveBackgroundTasks: unknown[] = [];

      // How many `assistant` messages this attempt has actually seen — the
      // signal for the empty-resume case below (vulpes-pack#206,
      // job-w11ho7): resuming a session that was hard-closed while a
      // background task was live can replay a stale `task_notification`
      // straight into an empty terminal result, before the injected prompt
      // ever gets a model turn. Zero here (with the terminal reporting zero
      // turns too) on a run that WAS a resume is that pattern; a fresh
      // session with real content never looks like this. Per-attempt —
      // resets naturally on each `executeWithRetry` invocation.
      let assistantMessageCount = 0;
      // Whether an actual terminal message (result or error) was ever seen —
      // required for the empty-resume check below so a stream that simply
      // ended without ever producing ANY terminal (a different, pre-existing
      // situation, not the stale-task_notification bug) is left alone.
      let sawTerminalMessage = false;

      // vulpes-pack#244 (job-2026-09-01-yqhqzn): a background task can start
      // AND drain to completion — with its `task_notification` delivered on
      // the stream — entirely WITHIN the turn that dispatched it, before that
      // turn's own terminal. The existing `liveBackgroundTasks.length > 0`
      // hold above only covers a task still running AT the terminal; a task
      // that already finished has nothing to hold on, so the terminal was
      // read as a genuine end and the notification's content was silently
      // lost. Track whether the model has had a fresh turn to actually see a
      // notification: set on `task_notification`, cleared the moment any
      // follow-up turn actually starts (see `awaitingFollowUpTurn` below) —
      // that turn gets it as input, whether or not the model acts on it.
      // One reinvocation attempt per undelivered batch falls out of this same
      // boolean, no separate counter needed: the terminal-check branch below
      // that consults it can only ever run once per undelivered batch — the
      // very next message (any type, including a redundant terminal) resets
      // it back to false via `awaitingFollowUpTurn` BEFORE that branch is
      // reached again, so a model that keeps ending turns without consuming
      // it falls through to the real end instead of looping.
      let notificationUndelivered = false;
      // Set right before any `continue` that holds the drain loop open past a
      // terminal, waiting for the SDK to reinvoke with a follow-up turn —
      // covers all three hold reasons (injected input, a live background
      // task, and the notification-reinvocation grace below). The NEXT
      // message received, whatever it is, is that follow-up turn starting.
      let awaitingFollowUpTurn = false;

      // Bound the wait for the follow-up turn. A host that folded the injected
      // message into the turn that just ended will never produce a second
      // result, so without this the run would idle until sessionTimeoutMs.
      // `kind` only changes the log wording — the injected-input case keeps
      // its original phrasing verbatim (tests match on it).
      const armFollowUpGrace = (kind: "injected input" | "notification"): void => {
        clearGrace();
        graceTimer = setTimeout(() => {
          // A live background task can still be running when this grace
          // window expires (CodeRabbit finding on the vulpes-pack sibling
          // patch, PR #181). Closing here would kill it out from under the
          // still-live child — hold instead, unconditionally: the drain loop
          // keeps consuming until the child's own re-invocation produces a
          // fresh terminal, or the run is aborted/times out.
          if (liveBackgroundTasks.length > 0) {
            this.logger.warn(
              kind === "injected input"
                ? `Job ${job.id}: injection grace expired but a background task is still live — holding, draining for the child's re-invocation`
                : `Job ${job.id}: notification-reinvocation grace expired but a background task is still live — holding, draining for the child's re-invocation`,
            );
            return;
          }
          graceClosed = true;
          acceptingInput = false;
          this.logger.warn(
            kind === "injected input"
              ? `Job ${job.id}: no follow-up turn within ${injectionGraceMs}ms after injected input — closing session (the message may have been folded into the previous turn)`
              : `Job ${job.id}: no follow-up turn within ${injectionGraceMs}ms after an undelivered task_notification — closing session (the model's turn ended before it could act on the notification)`,
          );
          void closeSession();
        }, injectionGraceMs);
        graceTimer.unref?.();
      };
      const armInjectionGrace = (): void => armFollowUpGrace("injected input");
      const armNotificationGrace = (): void => armFollowUpGrace("notification");

      const closeSession = async (): Promise<void> => {
        acceptingInput = false;
        clearGrace();
        if (drainTimer) {
          clearTimeout(drainTimer);
          drainTimer = undefined;
        }
        const open = session;
        session = undefined;
        if (!open) return;
        try {
          await open.close();
        } catch (closeError) {
          this.logger.warn(`Failed to close session: ${(closeError as Error).message}`);
        }
      };

      const timeoutError = (): SDKStreamingError =>
        new SDKStreamingError(
          buildErrorMessage(
            `Session-backed run produced no terminal result within sessionTimeoutMs (${sessionTimeoutMs}ms); session closed`,
            { jobId: job.id, agentName: agent.name },
          ),
          { jobId: job.id, agentName: agent.name, code: "SESSION_TIMEOUT", messagesReceived },
        );

      const emptyResumeRetryFailedError = (): SDKStreamingError =>
        new SDKStreamingError(
          buildErrorMessage(
            "Resumed session produced zero assistant turns, and so did the same-session retry and the fresh-session retry",
            { jobId: job.id, agentName: agent.name },
          ),
          {
            jobId: job.id,
            agentName: agent.name,
            code: "EMPTY_RESUME_RETRY_FAILED",
            messagesReceived,
          },
        );

      try {
        let messages: AsyncIterable<SDKMessage>;

        // Catch runtime initialization errors
        try {
          const runtimeOptions = {
            prompt,
            agent: options.agent,
            resume: resumeSessionId,
            // Only fork when we actually have a source session to fork from. On a
            // retry that cleared the resume target (e.g. the source expired), fall
            // back to a plain fresh session rather than `--fork-session` with no
            // `--resume`, which the CLI can't satisfy.
            fork: options.fork && resumeSessionId ? true : undefined,
            abortController: options.abortController,
            injectedMcpServers: options.injectedMcpServers,
            systemPromptAppend: options.systemPromptAppend,
            onLifecycleSignal: options.onLifecycleSignal,
          };

          if (openSession) {
            // A session's close() aborts the controller it was handed, as a
            // teardown backstop. Handing it the JOB's controller would make
            // ordinary end-of-run cleanup indistinguishable from a cancellation
            // (the run would be recorded `cancelled`, not `completed`), so the
            // session gets its own and the job's abort is forwarded one-way.
            const sessionAbort = new AbortController();
            const jobSignal = options.abortController?.signal;
            const markIfLiveOnAbort = () => {
              if (liveBackgroundTasks.length > 0) endedWithLiveBackgroundTasks = true;
              sessionAbort.abort();
            };
            if (jobSignal?.aborted) {
              markIfLiveOnAbort();
            } else {
              jobSignal?.addEventListener("abort", markIfLiveOnAbort, { once: true });
            }

            const opened = openSession({ ...runtimeOptions, abortController: sessionAbort });
            session = opened;
            acceptingInput = true;

            // Stuck-run backstop: the stream never ends by itself, so without
            // this a missing terminal result drains forever and never releases
            // the job's concurrency slot. Closing the session ends the stream,
            // which is what unblocks the loop below. Unlike the removed
            // bg-wait ceiling, this is a much longer (default 2h), unrelated
            // "genuinely stuck stream" backstop — but it can still fire while a
            // background task is legitimately live, so mark that truthfully too.
            drainTimer = setTimeout(() => {
              timedOut = true;
              if (liveBackgroundTasks.length > 0) endedWithLiveBackgroundTasks = true;
              void closeSession();
            }, sessionTimeoutMs);
            drainTimer.unref?.();

            options.onSessionOpen?.({
              send: (text) => {
                if (!acceptingInput || !session) return false;
                pendingInjected++;
                void session.send(text).catch((sendError) => {
                  this.logger.warn(`Failed to send into session: ${(sendError as Error).message}`);
                });
                return true;
              },
              interrupt: () => {
                if (!acceptingInput || !session) return false;
                // Recorded as a cancellation, not a failure: the operator asked
                // for this ending. The interrupt's terminal message breaks the
                // drain loop, so the run really does end here.
                interrupted = true;
                if (liveBackgroundTasks.length > 0) endedWithLiveBackgroundTasks = true;
                void session.interrupt().catch((interruptError) => {
                  this.logger.warn(
                    `Failed to interrupt session: ${(interruptError as Error).message}`,
                  );
                });
                return true;
              },
            });
            messages = opened.messages;
          } else {
            // No session handle on this path, so no local abort listener to
            // hook a live-tasks check into today — the caller's abortController
            // goes straight into SDKRuntime.execute() and an abort there throws
            // out of the for-await below rather than yielding a message.
            // Truthful-close marking for that case is in the `catch` below,
            // which reads the same `liveBackgroundTasks` this loop tracks.
            messages = this.runtime.execute(runtimeOptions);
          }
        } catch (initError) {
          // Wrap initialization errors with context
          throw new SDKInitializationError(
            buildErrorMessage((initError as Error).message, {
              jobId: job.id,
              agentName: agent.name,
            }),
            {
              jobId: job.id,
              agentName: agent.name,
              cause: initError as Error,
            },
          );
        }

        for await (const sdkMessage of messages) {
          messagesReceived++;
          // The follow-up turn is producing output — stop counting down.
          clearGrace();

          // Whatever this message is, it's the first one to arrive since a
          // terminal-message branch below chose to hold for a follow-up turn
          // — that turn now has any previously-undelivered notification as
          // input, whether or not it acts on it (vulpes-pack#244). Reset
          // before this message's own type is inspected, so a task_notification
          // that itself is what unblocks the hold below still starts a fresh
          // undelivered batch of its own.
          if (awaitingFollowUpTurn) {
            awaitingFollowUpTurn = false;
            notificationUndelivered = false;
          }

          // Track live background tasks off the raw message (REPLACE
          // semantics per the SDK's payload) so the terminal-message branch
          // below knows whether a `run_in_background` child is still running.
          if (
            sdkMessage &&
            (sdkMessage as { type?: string }).type === "system" &&
            (sdkMessage as { subtype?: string }).subtype === "background_tasks_changed"
          ) {
            liveBackgroundTasks = (sdkMessage as { tasks?: unknown[] }).tasks ?? [];
          }

          // A background task can finish (and report its `task_notification`)
          // entirely within the turn that dispatched it, before that turn's
          // own terminal — the model, already mid-generation, cannot have
          // consumed it. Marks the batch undelivered until a follow-up turn
          // actually starts (reset above).
          if (
            sdkMessage &&
            (sdkMessage as { type?: string }).type === "system" &&
            (sdkMessage as { subtype?: string }).subtype === "task_notification"
          ) {
            notificationUndelivered = true;
          }

          if (sdkMessage && (sdkMessage as { type?: string }).type === "assistant") {
            assistantMessageCount++;
          }

          // Process the message safely (handles malformed responses)
          let processed: ProcessedMessage | undefined;
          try {
            processed = processSDKMessage(sdkMessage);
          } catch (processError) {
            // Log but don't crash on malformed messages
            this.logger.warn(`Malformed SDK message received: ${(processError as Error).message}`);

            // Write a warning to job output
            try {
              await appendJobOutput(jobsDir, job.id, {
                type: "error",
                message: `Malformed SDK message: ${(processError as Error).message}`,
                code: "MALFORMED_MESSAGE",
              });
            } catch {
              // Ignore output write failures for malformed message warnings
            }

            // Continue processing other messages
            continue;
          }

          // Write to job output immediately (no buffering)
          try {
            await appendJobOutput(jobsDir, job.id, processed.output);
          } catch (outputError) {
            this.logger.warn(`Failed to write job output: ${(outputError as Error).message}`);
            // Continue processing - don't fail execution due to logging issues
          }

          // Also write to output.log file if outputToFile is enabled
          if (outputLogPath) {
            try {
              const logLine = this.formatOutputLogLine(processed.output);
              if (logLine) {
                await appendFile(outputLogPath, `${logLine}\n`, "utf-8");
              }
            } catch (fileError) {
              this.logger.warn(
                `Failed to write to output log file: ${(fileError as Error).message}`,
              );
              // Continue processing - file logging is optional
            }
          }

          // Log error messages to console immediately
          if (processed.output.type === "error") {
            this.logger.error(`Job ${job.id} error: ${processed.output.message}`);
          }

          // Extract session ID if present
          if (processed.sessionId) {
            sessionId = processed.sessionId;
          }

          // Track last non-partial assistant message content for fallback summary
          // This ensures we capture the final response even if it's long
          if (
            processed.output.type === "assistant" &&
            !processed.output.partial &&
            processed.output.content
          ) {
            lastAssistantContent = processed.output.content;
          }

          // Extract explicit summary if present (summary field or result message)
          // Only track explicit summaries here - assistant content is tracked above
          // Guard against null/undefined messages from malformed SDK responses
          if (sdkMessage && (sdkMessage.summary || sdkMessage.type === "result")) {
            const messageSummary = extractSummary(sdkMessage);
            if (messageSummary) {
              summary = messageSummary;
            }
          }

          // Capture per-run per-model token accounting from the terminal result
          // message (SDK native or CLI-synthesized). Later result messages win.
          if (sdkMessage && sdkMessage.type === "result") {
            const usage = extractRunUsage(sdkMessage);
            if (usage) {
              runUsage = usage;
            }
          }

          // Call user's onMessage callback if provided
          if (onMessage) {
            try {
              await onMessage(sdkMessage);
            } catch (callbackError) {
              this.logger.warn(`onMessage callback error: ${(callbackError as Error).message}`);
            }
          }

          // Check for terminal messages
          if (isTerminalMessage(sdkMessage)) {
            sawTerminalMessage = true;
            if (sdkMessage.type === "error") {
              // A stream-level error is always the end — no further turn runs.
              acceptingInput = false;
              if (liveBackgroundTasks.length > 0) endedWithLiveBackgroundTasks = true;
              const errorMessage = (sdkMessage.message as string) ?? "Agent execution failed";
              lastError = new SDKStreamingError(
                buildErrorMessage(errorMessage, {
                  jobId: job.id,
                  agentName: agent.name,
                }),
                {
                  jobId: job.id,
                  agentName: agent.name,
                  code: sdkMessage.code as string | undefined,
                  messagesReceived,
                },
              );
              break;
            }

            // A `result` can report failure without being an `error` message:
            // `is_error`, or any subtype other than "success" (max turns,
            // error_during_execution, an interrupted turn). Treating those as
            // completed reported broken runs as green — same semantics the
            // message processor already applies to the output record.
            //
            // Recomputed per result (not just set): with an injected follow-up
            // turn the run can produce several, and the job's outcome is the
            // LAST one — an interrupted first turn followed by a clean second
            // must not leave the job failed.
            if (isErrorResult(sdkMessage)) {
              const resultMessage = sdkMessage as { subtype?: string; result?: string };
              lastError = new SDKStreamingError(
                buildErrorMessage(
                  resultMessage.result ??
                    `Agent run ended with result subtype "${resultMessage.subtype ?? "unknown"}"`,
                  { jobId: job.id, agentName: agent.name },
                ),
                {
                  jobId: job.id,
                  agentName: agent.name,
                  code: resultMessage.subtype,
                  messagesReceived,
                },
              );
            } else {
              lastError = undefined;
            }

            if (pendingInjected > 0) {
              // Injected input is still queued behind this turn. Hosts differ on
              // when they deliver a pushed streaming-input message: some fold it
              // into the running turn at a tool boundary, others start a NEW turn
              // for it after the current one ends (observed on claude 2.1.220,
              // AI-406). Breaking here closes the session out from under that
              // second turn and the message is lost — the caller was already told
              // it was delivered. So keep draining for one more result.
              //
              // Counting results, not user-echo events: whether a pushed message
              // is echoed back into the stream is host-specific and unverified,
              // while "a result ends a turn" holds everywhere. One extra turn
              // covers every message queued so far; anything injected during that
              // turn raises the counter again and earns another.
              pendingInjected = 0;
              awaitingFollowUpTurn = true;
              armInjectionGrace();
              continue;
            }

            // A live background task (tracked above) means this terminal
            // result is not really the end — the SDK will re-invoke the
            // model once the child reports back, producing a fresh terminal
            // that must win over this stale one (already guaranteed: summary/
            // runUsage/lastError above are recomputed per result, "later
            // wins", same as the injected-input case above). Only
            // re-evaluated here, at a fresh terminal, not on every message —
            // a `background_tasks_changed` drain to empty is not itself a
            // terminal message.
            //
            // Unconditional, no ceiling: keep draining for as long as the
            // child is live (operator decision after job-2026-08-31-okhjlg —
            // see the note on `liveBackgroundTasks` above). Only stream
            // end/abort/timeout exits the loop from here on.
            if (liveBackgroundTasks.length > 0) {
              awaitingFollowUpTurn = true;
              continue;
            }

            // vulpes-pack#244: a task's `task_notification` landed while this
            // turn was still generating, so this turn's own terminal can't be
            // proof the model ever saw it — give the SDK one bounded grace
            // window to reinvoke with it as input (same mechanism as the
            // live-task hold above, just for a task that already finished).
            if (notificationUndelivered) {
              awaitingFollowUpTurn = true;
              armNotificationGrace();
              continue;
            }

            // Nothing pending, no live background tasks, no undelivered
            // notification left to wait on: this really is the end.
            acceptingInput = false;
            break;
          }
        }

        // The timeout closed the session, which can end the stream cleanly
        // instead of throwing — record the real cause rather than a silent
        // "completed with no result".
        if (timedOut && !lastError) {
          lastError = timeoutError();
        }

        // Post-loop session-not-found retry (issue #126).
        //
        // The catch block below recovers from session expiry that is *thrown*.
        // But some runtimes — notably the CLI runtime — don't throw when a
        // `claude --resume <id>` can't find the session: they *yield* a terminal
        // `error` message and the loop breaks normally, setting `lastError`
        // without entering the catch. This happens when the transcript is
        // unfindable because the spawn cwd changed (a working_directory config
        // change), the session was migrated, or it was cleaned up out of band.
        // Mirror the catch-block recovery for that yielded-error path: clear the
        // stale pointer and retry once with a fresh session.
        if (
          lastError &&
          isSessionExpiredError(lastError) &&
          resumeSessionId &&
          !retriedAfterSessionExpiry
        ) {
          this.logger.warn(
            `Session not found for ${agent.name} (resume failed). Clearing session and retrying with fresh session.`,
          );

          try {
            const sessionsDir = join(stateDir, "sessions");
            await clearSession(sessionsDir, sessionKey);
            this.logger.info?.(`Cleared stale session for ${agent.qualifiedName}`);
          } catch (clearError) {
            this.logger.warn(`Failed to clear stale session: ${(clearError as Error).message}`);
          }

          try {
            await appendJobOutput(jobsDir, job.id, {
              type: "system",
              content: `Session not found. Retrying with fresh session.`,
            });
          } catch {
            // Ignore output write failures
          }

          retriedAfterSessionExpiry = true;
          lastError = undefined;
          messagesReceived = 0;
          endedWithLiveBackgroundTasks = false;
          // Tear the failed session down before the retry opens a new one, so
          // two `claude` processes never run for this job at once.
          await closeSession();
          await executeWithRetry(undefined);
          return;
        }

        // Empty-resume retry (vulpes-pack#206, job-w11ho7).
        //
        // A resumed session that was hard-closed while a background task was
        // live can replay a stale `task_notification` straight into an empty
        // terminal result — the model never gets the injected prompt's own
        // turn, and this loop (having no error to report) would otherwise
        // report a clean "success" on zero assistant turns and zero cost,
        // silently swallowing the prompt. Gated on `sawTerminalMessage`: a
        // stream that ends without ever producing ANY terminal (result or
        // error) is a different, pre-existing situation — not this bug — and
        // must not retry. Only ever true on a genuine resume
        // (`resumeSessionId` set) — a fresh, non-resumed run producing no
        // turns is a different, pre-existing situation, not this bug.
        //
        // Retries: the empty result is usually a stale `task_notification`
        // replay (a resume whose transcript ends with a prior run's live
        // background-task notice), not a zombie session — a second resume of
        // the SAME session answers with full context. So retry the same
        // session once first, WITHOUT clearing the pointer (preserves any
        // Discord/channel context tied to it). Only if that retry is ALSO
        // empty do we fall back to the original behavior: discard the
        // (likely genuinely zombie) session pointer and retry once with a
        // fresh session. If THAT retry (running with `resumeSessionId`
        // undefined, so it can't re-enter this branch on `resumeSessionId` —
        // `retriedAfterEmptyResume` is what catches it) is also empty, this is
        // a real failure — fail loudly (a real error, not cancelled/timeout)
        // rather than keep reporting success on nothing.
        if (
          sawTerminalMessage &&
          !lastError &&
          !interrupted &&
          !(options.abortController?.signal.aborted ?? false) &&
          assistantMessageCount === 0 &&
          (runUsage?.num_turns ?? 0) === 0
        ) {
          if (retriedAfterEmptyResume) {
            lastError = emptyResumeRetryFailedError();
          } else if (retriedSameSessionAfterEmptyResume) {
            if (resumeSessionId) {
              this.logger.warn(
                `Job ${job.id}: resumed session ${resumeSessionId} for ${agent.name} produced zero assistant turns again on the same session. Clearing session and retrying once with a fresh session.`,
              );

              try {
                const sessionsDir = join(stateDir, "sessions");
                await clearSession(sessionsDir, sessionKey);
                this.logger.info?.(`Cleared zombie session for ${agent.qualifiedName}`);
              } catch (clearError) {
                this.logger.warn(
                  `Failed to clear zombie session: ${(clearError as Error).message}`,
                );
              }

              try {
                await appendJobOutput(jobsDir, job.id, {
                  type: "system",
                  content:
                    "Resumed session produced no assistant turns again. Retrying with fresh session.",
                });
              } catch {
                // Ignore output write failures
              }

              retriedAfterEmptyResume = true;
              messagesReceived = 0;
              endedWithLiveBackgroundTasks = false;
              // Tear the zombie session down before the retry opens a new
              // one, so two `claude` processes never run for this job at once.
              await closeSession();
              await executeWithRetry(undefined);
              return;
            }
          } else if (resumeSessionId) {
            this.logger.warn(
              `Job ${job.id}: resumed session ${resumeSessionId} for ${agent.name} produced zero assistant turns (stale task_notification replay likely). Retrying once on the same session.`,
            );

            try {
              await appendJobOutput(jobsDir, job.id, {
                type: "system",
                content:
                  "Resumed session produced no assistant turns. Retrying once on the same session.",
              });
            } catch {
              // Ignore output write failures
            }

            retriedSameSessionAfterEmptyResume = true;
            messagesReceived = 0;
            endedWithLiveBackgroundTasks = false;
            // Tear the session down before the retry re-opens it, so two
            // `claude` processes never run for this job at once.
            await closeSession();
            await executeWithRetry(resumeSessionId);
            return;
          }
          // else: a genuinely fresh (non-resume) run with zero turns — not
          // this ticket's failure mode, leave the existing behavior alone.
        }
      } catch (error) {
        // Check if this is a session expiration error from the SDK
        // This can happen if the server-side session expired even though local validation passed
        if (
          isSessionExpiredError(error as Error) &&
          resumeSessionId &&
          !retriedAfterSessionExpiry
        ) {
          this.logger.warn(
            `Session expired on server for ${agent.name}. Clearing session and retrying with fresh session.`,
          );

          // Clear the expired session
          try {
            const sessionsDir = join(stateDir, "sessions");
            await clearSession(sessionsDir, sessionKey);
            this.logger.info?.(`Cleared expired session for ${agent.qualifiedName}`);
          } catch (clearError) {
            this.logger.warn(`Failed to clear expired session: ${(clearError as Error).message}`);
          }

          // Write info to job output about the retry
          try {
            await appendJobOutput(jobsDir, job.id, {
              type: "system",
              content: `Session expired on server. Retrying with fresh session.`,
            });
          } catch {
            // Ignore output write failures
          }

          // Retry with a fresh session (no resume)
          retriedAfterSessionExpiry = true;
          messagesReceived = 0; // Reset for fresh session
          endedWithLiveBackgroundTasks = false;
          // Tear the failed session down before the retry opens a new one, so
          // two `claude` processes never run for this job at once.
          await closeSession();
          await executeWithRetry(undefined);
          return;
        }

        // Check if this is an OAuth token expiry error
        // On retry, buildContainerEnv() re-reads the credentials file and
        // refreshes the token. The fresh credentials reach the container via a
        // new ephemeral container (created with the new env) or, for reused
        // persistent containers, by being injected into each docker exec — so
        // both cases pick up the refreshed token. See edspencer/herdctl#327.
        if (isTokenExpiredError(error as Error) && !retriedAfterTokenExpiry) {
          this.logger.warn(`OAuth token expired for ${agent.name}. Retrying with fresh token.`);

          // Write info to job output about the retry
          try {
            await appendJobOutput(jobsDir, job.id, {
              type: "system",
              content: `OAuth token expired. Refreshing token and retrying.`,
            });
          } catch {
            // Ignore output write failures
          }

          // Retry — buildContainerEnv() will refresh the token from the credentials file
          retriedAfterTokenExpiry = true;
          messagesReceived = 0;
          endedWithLiveBackgroundTasks = false;
          // Tear the failed session down before the retry opens a new one, so
          // two `claude` processes never run for this job at once.
          await closeSession();
          await executeWithRetry(undefined);
          return;
        }

        // An abort on the plain execute() path (no session handle to hook a
        // listener into, see the `else` branch above) surfaces here as a
        // thrown error rather than a stream message — the timedOut/graceClosed
        // cases already mark truthfully at their own throw sites, so only the
        // generic fallthrough (a genuine abort or unexpected failure) needs it.
        if (!timedOut && !graceClosed && liveBackgroundTasks.length > 0) {
          endedWithLiveBackgroundTasks = true;
        }

        // Wrap the error with context if not already a RunnerError. A timeout
        // surfaces here as the abort the session's close() raised — report the
        // timeout, not the abort it caused.
        lastError = timedOut
          ? timeoutError()
          : graceClosed
            ? // The grace window closed the session on purpose; the abort it
              // raised is the mechanism, not the outcome. Keep the last result's.
              lastError
            : wrapError(error, {
                jobId: job.id,
                agentName: agent.name,
                phase: messagesReceived === 0 ? "init" : "streaming",
              });

        // A grace-window close can leave no error at all — the run ended
        // normally on its last result and there is nothing to report.
        if (!lastError) return;

        // Add messages received count for streaming errors
        if (lastError instanceof SDKStreamingError && messagesReceived > 0) {
          (lastError as SDKStreamingError & { messagesReceived?: number }).messagesReceived =
            messagesReceived;
        }

        // Log the error with context
        this.logger.error(`${lastError.name}: ${lastError.message}`);

        // Write error to job output with full context
        try {
          await appendJobOutput(jobsDir, job.id, {
            type: "error",
            message: lastError.message,
            code:
              (lastError as SDKStreamingError).code ??
              (lastError.cause as NodeJS.ErrnoException)?.code,
            stack: lastError.stack,
          });
        } catch (outputError) {
          this.logger.warn(
            `Failed to write error to job output: ${(outputError as Error).message}`,
          );
        }
      } finally {
        // Always release the session — success, failure, or abort. A leaked
        // session keeps a `claude` process (and its RSS) alive indefinitely.
        await closeSession();
      }
    };

    await executeWithRetry(effectiveResume);

    // Build error details for programmatic access
    if (lastError) {
      errorDetails = {
        message: lastError.message,
        code:
          (lastError as SDKStreamingError).code ?? (lastError.cause as NodeJS.ErrnoException)?.code,
        stack: lastError.stack,
      };

      // Determine error type
      if (lastError instanceof SDKInitializationError) {
        errorDetails.type = "initialization";
        errorDetails.recoverable = lastError.isNetworkError();
      } else if (lastError instanceof SDKStreamingError) {
        errorDetails.type = "streaming";
        errorDetails.recoverable = lastError.isRecoverable();
        errorDetails.messagesReceived = lastError.messagesReceived;
      } else if (lastError instanceof MalformedResponseError) {
        errorDetails.type = "malformed_response";
        errorDetails.recoverable = false;
      } else {
        errorDetails.type = "unknown";
        errorDetails.recoverable = false;
      }
    }

    // Final summary logic:
    // 1. If an explicit summary was found (from summary field or result message), use it
    // 2. Otherwise, use the last assistant content
    // This ensures we capture the final response, not an early short message
    // Note: Truncation is handled by downstream consumers (e.g., Discord hook truncates to 4096)
    if (!summary && lastAssistantContent) {
      summary = lastAssistantContent;
    }

    // Truthful close (operator decision after job-2026-08-31-okhjlg): the run
    // ended (abort, the sessionTimeoutMs backstop, or a stream error) while a
    // background child was still live per the last known snapshot — mark the
    // summary/log line so it can never read as a clean, complete result on its
    // own, whatever the job's status field ends up being.
    if (endedWithLiveBackgroundTasks) {
      const marker = "background work terminated before completion";
      summary = summary ? `${summary} [${marker}]` : `[${marker}]`;
      try {
        await appendJobOutput(jobsDir, job.id, { type: "system", content: marker });
      } catch {
        // Ignore output write failures — the summary marker above is the
        // load-bearing signal; this is best-effort extra visibility.
      }
    }

    // Step 5: Update job with final status
    // A run that was aborted mid-flight (cancelJob → AbortController) is recorded
    // as "cancelled", not "failed": the CLI runtime surfaces the kill as a
    // terminal error, but that's an intentional cancellation, not a failure.
    // An operator-requested interrupt counts as a cancellation for the same
    // reason an abort does: the run was stopped on purpose, so its terminal
    // error is an intended ending, not a failure.
    const cancelled = (options.abortController?.signal.aborted ?? false) || interrupted;
    const success = !lastError && !cancelled;
    const finishedAt = new Date().toISOString();

    // Determine status + exit reason: cancelled > success > failed.
    const status: "completed" | "failed" | "cancelled" = cancelled
      ? "cancelled"
      : success
        ? "completed"
        : "failed";
    const exitReason = cancelled ? "cancelled" : success ? "success" : classifyError(lastError!);

    try {
      await updateJob(jobsDir, job.id, {
        status,
        finished_at: finishedAt,
        session_id: sessionId,
        summary,
        exit_reason: exitReason,
        output_file: getJobOutputPath(jobsDir, job.id),
        // Persist per-model token accounting when the run produced a terminal
        // result. Left untouched (stays null/absent) when there was none.
        ...(runUsage ? { usage: runUsage } : {}),
      });
    } catch (error) {
      this.logger.warn(`Failed to update job final status: ${(error as Error).message}`);
    }

    // Step 6: Persist session info for resume capability — UNLESS this run
    // ended (abort, sessionTimeoutMs backstop, or a stream error) while
    // background work was still live. Its transcript can hold a stale,
    // mid-flight `task_notification` that a later resume replays straight
    // into an empty terminal result before the next prompt's own turn
    // (vulpes-pack#206, job-w11ho7 — see the empty-resume retry above,
    // which recovers from this ONE level too late, after it's already
    // happened). Clear the pointer instead: dirty-marking here is the first
    // line of defense, so the next trigger starts fresh rather than resuming
    // a zombie session at all.
    if (endedWithLiveBackgroundTasks) {
      try {
        const sessionsDir = join(stateDir, "sessions");
        await clearSession(sessionsDir, sessionKey);
        this.logger.debug?.(
          `Cleared session pointer for ${agent.qualifiedName} — run ended with live background work`,
        );
      } catch (sessionError) {
        this.logger.warn(
          `Failed to clear dirty session pointer: ${(sessionError as Error).message}`,
        );
      }
    } else if (sessionId) {
      try {
        const sessionsDir = join(stateDir, "sessions");

        // Get existing session to determine if updating or creating
        const existingSession = await getSessionInfo(sessionsDir, sessionKey, {
          runtime: agent.runtime ?? "sdk",
        });

        // Store the current working directory with the session
        const currentWorkingDirectory = resolveWorkingDirectory(agent);

        await updateSessionInfo(sessionsDir, sessionKey, {
          session_id: sessionId,
          job_count: (existingSession?.job_count ?? 0) + 1,
          mode: existingSession?.mode ?? "autonomous",
          working_directory: currentWorkingDirectory,
          runtime_type: (agent.runtime as "sdk" | "cli") ?? "sdk",
          docker_enabled: agent.docker?.enabled ?? false,
        });

        this.logger.debug?.(`Persisted session ${sessionId} for agent ${agent.name}`);
      } catch (sessionError) {
        this.logger.warn(`Failed to persist session info: ${(sessionError as Error).message}`);
        // Continue - session persistence is non-fatal
      }
    }

    // Calculate duration
    const startTime = new Date(job.started_at).getTime();
    const endTime = new Date(finishedAt).getTime();
    const durationSeconds = Math.round((endTime - startTime) / 1000);

    return {
      success,
      jobId: job.id,
      sessionId,
      summary,
      error: lastError,
      errorDetails,
      durationSeconds,
    };
  }

  /**
   * Format a job output message as a human-readable log line
   *
   * Converts the structured JobOutputInput to a simple text format for the output.log file.
   *
   * @param output - The job output message to format
   * @returns Formatted log line, or null if message should not be logged
   */
  private formatOutputLogLine(output: {
    type: string;
    content?: string;
    message?: string;
    tool_name?: string;
    input?: unknown;
    result?: unknown;
    success?: boolean;
    [key: string]: unknown;
  }): string | null {
    const timestamp = new Date().toISOString();

    switch (output.type) {
      case "assistant":
        if (output.content) {
          return `[${timestamp}] [ASSISTANT] ${output.content}`;
        }
        break;

      case "tool_use":
        if (output.tool_name) {
          const inputStr = output.input ? ` ${JSON.stringify(output.input)}` : "";
          return `[${timestamp}] [TOOL] ${output.tool_name}${inputStr}`;
        }
        break;

      case "tool_result":
        if (output.result !== undefined) {
          const resultStr =
            typeof output.result === "string" ? output.result : JSON.stringify(output.result);
          const status = output.success === false ? "FAILED" : "OK";
          return `[${timestamp}] [TOOL_RESULT] (${status}) ${resultStr}`;
        }
        break;

      case "system":
        if (output.content || output.message) {
          return `[${timestamp}] [SYSTEM] ${output.content ?? output.message}`;
        }
        break;

      case "error":
        if (output.message || output.content) {
          return `[${timestamp}] [ERROR] ${output.message ?? output.content}`;
        }
        break;
    }

    return null;
  }
}

// =============================================================================
// Convenience Function
// =============================================================================

/**
 * Execute an agent with streaming output to job log
 *
 * This is a convenience function that creates a JobExecutor and runs
 * a single execution. For multiple executions, prefer creating a
 * JobExecutor instance directly.
 *
 * @param runtime - The runtime interface
 * @param options - Runner options including agent config and prompt
 * @param executorOptions - Optional executor configuration
 * @returns Result of the execution
 *
 * @example
 * ```typescript
 * import { RuntimeFactory } from "@herdctl/core";
 *
 * const runtime = RuntimeFactory.create(agent);
 * const result = await executeJob(runtime, {
 *   agent: resolvedAgent,
 *   prompt: "Fix the bug",
 *   stateDir: "/path/to/.herdctl",
 * });
 * ```
 */
export async function executeJob(
  runtime: RuntimeInterface,
  options: RunnerOptionsWithCallbacks,
  executorOptions: JobExecutorOptions = {},
): Promise<RunnerResult> {
  const executor = new JobExecutor(runtime, executorOptions);
  return executor.execute(options);
}
