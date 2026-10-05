/**
 * Tool output reference registry.
 *
 * When enabled via `RunConfig.toolOutputReferences.enabled`, ToolNode
 * stores each successful tool output under a stable key
 * (`tool<idx>turn<turn>`) where `idx` is the tool's position within a
 * ToolNode batch and `turn` is the batch index within the run
 * (incremented once per ToolNode invocation).
 *
 * Subsequent tool calls can pipe a previous output into their args by
 * embedding `{{tool<idx>turn<turn>}}` inside any string argument;
 * {@link ToolOutputReferenceRegistry.resolve} walks the args and
 * substitutes the placeholders immediately before invocation.
 *
 * The registry stores the *raw, untruncated* tool output so a later
 * `{{…}}` substitution pipes the full payload into the next tool —
 * even when the LLM only saw a head+tail-truncated preview in
 * `ToolMessage.content`. Outputs are stored without any annotation
 * (the `_ref` key or the `[ref: ...]` prefix seen by the LLM is
 * strictly a UX signal attached to `ToolMessage.content`). Keeping the
 * registry pristine means downstream bash/jq piping receives the
 * complete, verbatim output with no injected fields.
 */

import { ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import {
  calculateMaxTotalToolOutputSize,
  HARD_MAX_TOOL_RESULT_CHARS,
  HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE,
} from '@/utils/truncation';
import { isComputerCallOutputMessage } from '@/utils/toolContent';
import { INTENT_ARG } from '@/tools/intentArg';

/** Parses a stringified JSON object arg; undefined for anything else. */
function parseStringifiedArgsObject(
  value: string
): Record<string, unknown> | undefined {
  if (!value.trim().startsWith('{')) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(value) as unknown;
    if (
      parsed != null &&
      typeof parsed === 'object' &&
      !Array.isArray(parsed)
    ) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    return undefined;
  }
  return undefined;
}

/**
 * Non-global matcher for a single `{{tool<i>turn<n>}}` placeholder.
 * Exported for consumers that want to detect references (e.g., syntax
 * highlighting, docs). The stateful `g` variant lives inside the
 * registry so nobody trips on `lastIndex`.
 */
export const TOOL_OUTPUT_REF_PATTERN = /\{\{(tool\d+turn\d+)\}\}/;

/** Object key used when a parsed-object output has `_ref` injected. */
export const TOOL_OUTPUT_REF_KEY = '_ref';

/**
 * Object key used to carry unresolved reference warnings on a parsed-
 * object output. Using a dedicated field instead of a trailing text
 * line keeps the annotated `ToolMessage.content` parseable as JSON for
 * downstream consumers that rely on the object shape.
 */
export const TOOL_OUTPUT_UNRESOLVED_KEY = '_unresolved_refs';

/** Single-line prefix prepended to non-object tool outputs so the LLM sees the reference key. */
export function buildReferencePrefix(key: string): string {
  return `[ref: ${key}]`;
}

/** Stable registry key for a tool output. */
export function buildReferenceKey(toolIndex: number, turn: number): string {
  return `tool${toolIndex}turn${turn}`;
}

export type ToolOutputReferenceRegistryOptions = {
  /** Maximum characters stored per registered output. */
  maxOutputSize?: number;
  /** Maximum total characters retained across all registered outputs. */
  maxTotalSize?: number;
  /**
   * Upper bound on the number of concurrently-tracked runs. When
   * exceeded, the oldest run bucket is evicted (FIFO). Defaults to 32.
   */
  maxActiveRuns?: number;
};

export type ToolOutputReferenceProtection = {
  version: 1;
  toolName: string;
  toolCallId: string;
  protected: boolean;
};

export interface ToolOutputReferenceState {
  entries: Array<{ key: string; value: string; protection?: ToolOutputReferenceProtection }>;
  turnCounter: number;
  warnedNonStringTools: string[];
}

/**
 * Result of resolving placeholders in tool args.
 */
export type ResolveResult<T> = {
  /** Arguments with placeholders replaced. Same shape as the input. */
  resolved: T;
  /** Reference keys that were referenced but had no stored value. */
  unresolved: string[];
};

/**
 * Read-only view over a frozen registry snapshot. Returned by
 * {@link ToolOutputReferenceRegistry.snapshot} for callers that need
 * to resolve placeholders against the registry state at a specific
 * point in time, ignoring any subsequent registrations.
 */
export interface ToolOutputResolveView {
  resolve<T>(args: T, options?: ResolveOptions): ResolveResult<T>;
}

/**
 * Per-call resolution options. `substituteIntentKey` opts the top-level
 * `intent` key back INTO placeholder substitution: the exemption protects
 * the injected display label, but a tool whose own schema declares a
 * business parameter named `intent` (the injectors skip such tools) still
 * needs references piped into it like any other argument.
 */
export type ResolveOptions = {
  substituteIntentKey?: boolean;
};

/**
 * Pre-resolved arg map keyed by `toolCallId`. Used by the mixed
 * direct+event dispatch path to feed event calls' resolved args
 * (captured pre-batch) into the dispatcher without re-resolving
 * against the now-stale live registry.
 */
export type PreResolvedArgsMap = Map<
  string,
  { resolved: Record<string, unknown>; unresolved: string[] }
>;

/**
 * Per-call sink for resolved args, keyed by `toolCallId`. Threaded
 * as a per-batch local map so concurrent `ToolNode.run()` calls do
 * not race on shared sink state.
 */
export type ResolvedArgsByCallId = Map<string, Record<string, unknown>>;

const EMPTY_ENTRIES: ReadonlyMap<string, string> = new Map<string, string>();

/**
 * Per-run state bucket held inside the registry. Each distinct
 * `run_id` gets its own bucket so overlapping concurrent runs on a
 * shared registry cannot leak outputs, turn counters, or warn-memos
 * into one another.
 */
class RunStateBucket {
  entries: Map<string, string> = new Map();
  protections: Map<string, ToolOutputReferenceProtection> = new Map();
  totalSize: number = 0;
  turnCounter: number = 0;
  warnedNonStringTools: Set<string> = new Set();
}

/**
 * Anonymous (`run_id` absent) bucket key. Anonymous batches are
 * treated as fresh runs on every invocation — see `nextTurn`.
 */
const ANON_RUN_KEY = '\0anon';

/**
 * Default upper bound on the number of concurrently-tracked runs per
 * registry. When exceeded, the oldest run's bucket (by insertion
 * order) is evicted. Keeps memory bounded when a ToolNode is reused
 * across many runs without explicit `releaseRun` calls.
 */
const DEFAULT_MAX_ACTIVE_RUNS = 32;

/**
 * Ordered map of reference-key → stored output, partitioned by run so
 * concurrent / interleaved runs sharing one registry cannot leak
 * outputs between each other.
 *
 * Each public method takes a `runId` which selects the run's bucket.
 * Hosts typically get one registry per run via `Graph`, in which
 * case only a single bucket is ever populated; the partitioning
 * exists so the registry also behaves correctly when a single
 * instance is reused directly.
 */
export class ToolOutputReferenceRegistry {
  private runStates: Map<string, RunStateBucket> = new Map();
  private readonly maxOutputSize: number;
  private readonly maxTotalSize: number;
  private readonly maxActiveRuns: number;
  /**
   * Local stateful matcher used only by `replaceInString`. Kept
   * off-module so callers of the exported `TOOL_OUTPUT_REF_PATTERN`
   * never see a stale `lastIndex`.
   */
  private static readonly PLACEHOLDER_MATCHER = /\{\{(tool\d+turn\d+)\}\}/g;

  constructor(options: ToolOutputReferenceRegistryOptions = {}) {
    /**
     * Per-output default is the same ~400 KB budget as the standard
     * tool-result truncation (`HARD_MAX_TOOL_RESULT_CHARS`). This
     * keeps a single `{{…}}` substitution at a size that is safe to
     * pass through typical shell `ARG_MAX` limits and matches what
     * the LLM would otherwise have seen. Hosts that want larger per-
     * output payloads (API consumers, long JSON streams) can raise
     * the cap explicitly up to the 5 MB total budget.
     */
    const perOutput =
      options.maxOutputSize != null && options.maxOutputSize > 0
        ? options.maxOutputSize
        : HARD_MAX_TOOL_RESULT_CHARS;
    /**
     * Clamp a caller-supplied `maxTotalSize` to
     * `HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE` (5 MB) so the documented
     * absolute cap is enforced regardless of host config —
     * `calculateMaxTotalToolOutputSize` already applies the same
     * upper bound on its computed default, but the user-provided
     * branch was bypassing it.
     */
    const totalRaw =
      options.maxTotalSize != null && options.maxTotalSize > 0
        ? Math.min(options.maxTotalSize, HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE)
        : calculateMaxTotalToolOutputSize(perOutput);
    this.maxTotalSize = totalRaw;
    /**
     * The per-output cap can never exceed the per-run aggregate cap:
     * if a single entry were allowed to be larger than `maxTotalSize`,
     * the eviction loop would either blow the cap (to keep the entry)
     * or self-evict a just-stored value. Clamping here turns
     * `maxTotalSize` into a hard upper bound on *any* state the
     * registry retains per run.
     */
    this.maxOutputSize = Math.min(perOutput, totalRaw);
    this.maxActiveRuns =
      options.maxActiveRuns != null && options.maxActiveRuns > 0
        ? options.maxActiveRuns
        : DEFAULT_MAX_ACTIVE_RUNS;
  }

  private keyFor(runId: string | undefined): string {
    return runId ?? ANON_RUN_KEY;
  }

  private getOrCreate(runId: string | undefined): RunStateBucket {
    const key = this.keyFor(runId);
    let state = this.runStates.get(key);
    if (state == null) {
      state = new RunStateBucket();
      this.runStates.set(key, state);
      if (this.runStates.size > this.maxActiveRuns) {
        const oldest = this.runStates.keys().next().value;
        if (oldest != null && oldest !== key) {
          this.runStates.delete(oldest);
        }
      }
    }
    return state;
  }

  /** Registers (or replaces) the output stored under `key` for `runId`. */
  set(runId: string | undefined, key: string, value: string, protection?: ToolOutputReferenceProtection): void {
    const bucket = this.getOrCreate(runId);
    const clipped =
      value.length > this.maxOutputSize
        ? value.slice(0, this.maxOutputSize)
        : value;
    const existing = bucket.entries.get(key);
    if (existing != null) {
      bucket.totalSize -= existing.length;
      bucket.entries.delete(key);
    }
    bucket.entries.set(key, clipped);
    if (protection == null) bucket.protections.delete(key);
    else bucket.protections.set(key, { ...protection });
    bucket.totalSize += clipped.length;
    this.evictWithinBucket(bucket);
  }

  /** Returns the stored value for `key` in `runId`'s bucket, or `undefined`. */
  get(runId: string | undefined, key: string): string | undefined {
    return this.runStates.get(this.keyFor(runId))?.entries.get(key);
  }

  /**
   * Returns `true` when `key` is currently stored in `runId`'s bucket.
   * Used by {@link annotateMessagesForLLM} to gate transient annotation
   * on whether the registry still owns the referenced output (a stale
   * `_refKey` from a prior run silently no-ops here).
   */
  has(runId: string | undefined, key: string): boolean {
    return this.runStates.get(this.keyFor(runId))?.entries.has(key) ?? false;
  }

  /** Total number of registered outputs across every run bucket. */
  get size(): number {
    let n = 0;
    for (const bucket of this.runStates.values()) {
      n += bucket.entries.size;
    }
    return n;
  }

  /** Maximum characters retained per output (post-clip). */
  get perOutputLimit(): number {
    return this.maxOutputSize;
  }

  /** Maximum total characters retained *per run*. */
  get totalLimit(): number {
    return this.maxTotalSize;
  }

  /** Drops every run's state. */
  clear(): void {
    this.runStates.clear();
  }

  /**
   * Explicitly release `runId`'s state. Safe to call when a run has
   * finished. Hosts sharing one registry across runs should call this
   * to reclaim memory deterministically; otherwise LRU eviction kicks
   * in when `maxActiveRuns` runs accumulate.
   */
  releaseRun(runId: string | undefined): void {
    this.runStates.delete(this.keyFor(runId));
  }

  /**
   * Claims the next batch turn synchronously from `runId`'s bucket.
   *
   * Must be called once at the start of each ToolNode batch before
   * any `await`, so concurrent invocations within the same run see
   * distinct turn values (reads are effectively atomic by JS's
   * single-threaded execution of the sync prefix).
   *
   * If `runId` is missing the anonymous bucket is dropped and a
   * fresh one created so each anonymous call behaves as its own run.
   */
  nextTurn(runId: string | undefined): number {
    if (runId == null) {
      this.runStates.delete(ANON_RUN_KEY);
    }
    const bucket = this.getOrCreate(runId);
    return bucket.turnCounter++;
  }

  /**
   * Records that `toolName` has been warned about in `runId` (returns
   * `true` on the first call per run, `false` after). Used by
   * ToolNode to emit one log line per offending tool per run when a
   * `ToolMessage.content` isn't a string.
   */
  claimWarnOnce(runId: string | undefined, toolName: string): boolean {
    const bucket = this.getOrCreate(runId);
    if (bucket.warnedNonStringTools.has(toolName)) {
      return false;
    }
    bucket.warnedNonStringTools.add(toolName);
    return true;
  }

  /**
   * Walks `args` and replaces every `{{tool<i>turn<n>}}` placeholder in
   * string values with the stored output *from `runId`'s bucket*. Non-
   * string values and object keys are left untouched. Unresolved
   * references are left in-place and reported so the caller can
   * surface them to the LLM. When no placeholder appears anywhere in
   * the serialized args, the original input is returned without
   * walking the tree.
   */
  resolve<T>(
    runId: string | undefined,
    args: T,
    options?: ResolveOptions
  ): ResolveResult<T> {
    if (!hasAnyPlaceholder(args)) {
      return { resolved: args, unresolved: [] };
    }
    const bucket = this.runStates.get(this.keyFor(runId));
    return this.resolveAgainst(bucket?.entries ?? EMPTY_ENTRIES, args, options);
  }

  /**
   * Captures a frozen snapshot of `runId`'s current entries and
   * returns a view that resolves placeholders against *only* that
   * snapshot. The snapshot is decoupled from the live registry, so
   * subsequent `set()` calls (for example, same-turn direct outputs
   * registering while an event branch is still in flight) are
   * invisible to the snapshot's `resolve`. Used by the mixed
   * direct+event dispatch path to preserve same-turn isolation when
   * a `PreToolUse` hook rewrites event args after directs have
   * completed.
   */
  snapshot(runId: string | undefined): ToolOutputResolveView {
    const bucket = this.runStates.get(this.keyFor(runId));
    const entries: ReadonlyMap<string, string> = bucket
      ? new Map(bucket.entries)
      : EMPTY_ENTRIES;
    return {
      resolve: <T>(args: T, options?: ResolveOptions): ResolveResult<T> =>
        this.resolveAgainst(entries, args, options),
    };
  }

  /** Captures the checkpoint-safe state needed to resume one run bucket. */
  snapshotState(runId: string | undefined): ToolOutputReferenceState {
    const bucket = this.runStates.get(this.keyFor(runId));
    return {
      entries: [...(bucket?.entries ?? EMPTY_ENTRIES)].map(([key, value]) => ({
        key,
        value,
        ...(bucket?.protections.has(key) === true ? { protection: { ...bucket.protections.get(key)! } } : {}),
      })),
      turnCounter: bucket?.turnCounter ?? 0,
      warnedNonStringTools: [...(bucket?.warnedNonStringTools ?? [])],
    };
  }

  /** Restore batch inputs without rolling back graph-owned concurrent outputs. */
  resumeBatch(runId: string, state: ToolOutputReferenceState, canonicalInputs = false): ToolOutputResolveView {
    if (!this.runStates.has(runId)) {
      this.restoreState(runId, state);
    }
    const bucket = this.getOrCreate(runId);
    if (canonicalInputs) {
      for (const { key, value, protection } of state.entries) this.set(runId, key, value, protection);
    }
    bucket.turnCounter = Math.max(bucket.turnCounter, state.turnCounter + 1);
    const inputs = new ToolOutputReferenceRegistry({
      maxOutputSize: this.maxOutputSize,
      maxTotalSize: this.maxTotalSize,
      maxActiveRuns: 1,
    });
    inputs.restoreState(runId, state);
    return inputs.snapshot(runId);
  }

  /** Restores a checkpointed run bucket under the current resume scope. */
  restoreState(
    runId: string | undefined,
    state: ToolOutputReferenceState
  ): void {
    this.releaseRun(runId);
    const bucket = this.getOrCreate(runId);
    for (const { key, value, protection } of state.entries) {
      this.set(runId, key, value, protection);
    }
    bucket.turnCounter = state.turnCounter;
    for (const toolName of state.warnedNonStringTools) {
      bucket.warnedNonStringTools.add(toolName);
    }
  }

  private resolveAgainst<T>(
    entries: ReadonlyMap<string, string>,
    args: T,
    options?: ResolveOptions
  ): ResolveResult<T> {
    if (!hasAnyPlaceholder(args)) {
      return { resolved: args, unresolved: [] };
    }
    const exemptIntentKey = options?.substituteIntentKey !== true;
    const unresolved = new Set<string>();
    /**
     * Providers may deliver the args OBJECT as a JSON string. A plain
     * string-root transform would expand placeholders inside the `intent`
     * label too, bypassing the top-level exclusion below — parse, transform
     * key-aware, and re-serialize so the label stays verbatim while every
     * other field still substitutes. Only taken when an `intent` key is
     * actually present; other strings keep the fast raw-string path.
     */
    if (exemptIntentKey && typeof args === 'string') {
      const parsedRoot = parseStringifiedArgsObject(args);
      if (parsedRoot != null && INTENT_ARG in parsedRoot) {
        const resolved = JSON.stringify(
          this.transform(entries, parsedRoot, unresolved, true)
        ) as T;
        return { resolved, unresolved: Array.from(unresolved) };
      }
    }
    const resolved = this.transform(
      entries,
      args,
      unresolved,
      exemptIntentKey
    ) as T;
    return { resolved, unresolved: Array.from(unresolved) };
  }

  private transform(
    entries: ReadonlyMap<string, string>,
    value: unknown,
    unresolved: Set<string>,
    exemptRootIntent = false
  ): unknown {
    if (typeof value === 'string') {
      return this.replaceInString(entries, value, unresolved);
    }
    if (Array.isArray(value)) {
      return value.map((item) => this.transform(entries, item, unresolved));
    }
    if (value !== null && typeof value === 'object') {
      const source = value as Record<string, unknown>;
      const next: Record<string, unknown> = {};
      for (const [key, item] of Object.entries(source)) {
        /**
         * The top-level `intent` arg is a display label, never a data
         * channel: expanding a `{{tool<i>turn<n>}}` placeholder there would
         * dump a stored output (up to the registry cap) into a single-line
         * UI label and persist it with the message. Left verbatim instead.
         */
        next[key] =
          exemptRootIntent && key === INTENT_ARG
            ? item
            : this.transform(entries, item, unresolved);
      }
      return next;
    }
    return value;
  }

  private replaceInString(
    entries: ReadonlyMap<string, string>,
    input: string,
    unresolved: Set<string>
  ): string {
    if (input.indexOf('{{tool') === -1) {
      return input;
    }
    return input.replace(
      ToolOutputReferenceRegistry.PLACEHOLDER_MATCHER,
      (match, key: string) => {
        const stored = entries.get(key);
        if (stored == null) {
          unresolved.add(key);
          return match;
        }
        return stored;
      }
    );
  }

  private evictWithinBucket(bucket: RunStateBucket): void {
    if (bucket.totalSize <= this.maxTotalSize) {
      return;
    }
    for (const key of bucket.entries.keys()) {
      if (bucket.totalSize <= this.maxTotalSize) {
        return;
      }
      const entry = bucket.entries.get(key);
      if (entry == null) {
        continue;
      }
      bucket.totalSize -= entry.length;
      bucket.entries.delete(key);
      bucket.protections.delete(key);
    }
  }
}

/**
 * Cheap pre-check: returns true if any string value in `args` contains
 * the `{{tool` substring. Lets `resolve()` skip the deep tree walk (and
 * its object allocations) for the common case of plain args.
 */
function hasAnyPlaceholder(value: unknown): boolean {
  if (typeof value === 'string') {
    return value.indexOf('{{tool') !== -1;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      if (hasAnyPlaceholder(item)) {
        return true;
      }
    }
    return false;
  }
  if (value !== null && typeof value === 'object') {
    for (const item of Object.values(value as Record<string, unknown>)) {
      if (hasAnyPlaceholder(item)) {
        return true;
      }
    }
    return false;
  }
  return false;
}

/**
 * Annotates `content` with a reference key and/or unresolved-ref
 * warnings so the LLM sees both alongside the tool output.
 *
 * Behavior:
 *  - If `content` parses as a plain (non-array, non-null) JSON object
 *    and the object does not already have a conflicting `_ref` key,
 *    the reference key and (when present) `_unresolved_refs` array
 *    are injected as object fields, preserving JSON validity for
 *    downstream consumers that parse the output.
 *  - Otherwise (string output, JSON array/primitive, parse failure,
 *    or `_ref` collision), a `[ref: <key>]\n` prefix line is
 *    prepended and unresolved refs are appended as a trailing
 *    `[unresolved refs: …]` line.
 *
 * The annotated string is what the LLM sees as `ToolMessage.content`.
 * The *original* (un-annotated) value is what gets stored in the
 * registry, so downstream piping remains pristine.
 *
 * @param content     Raw (post-truncation) tool output.
 * @param key         Reference key for this output, or undefined when
 *                    there is nothing to register (errors etc.).
 * @param unresolved  Reference keys that failed to resolve during
 *                    argument substitution. Surfaced so the LLM can
 *                    self-correct its next tool call.
 */
export function annotateToolOutputWithReference(
  content: string,
  key: string | undefined,
  unresolved: string[] = []
): string {
  const hasRefKey = key != null;
  const hasUnresolved = unresolved.length > 0;
  if (!hasRefKey && !hasUnresolved) {
    return content;
  }
  const trimmed = content.trimStart();
  if (trimmed.startsWith('{')) {
    const annotated = tryInjectRefIntoJsonObject(content, key, unresolved);
    if (annotated != null) {
      return annotated;
    }
  }
  const prefix = hasRefKey ? `${buildReferencePrefix(key!)}\n` : '';
  const trailer = hasUnresolved
    ? `\n[unresolved refs: ${unresolved.join(', ')}]`
    : '';
  return `${prefix}${content}${trailer}`;
}

function tryInjectRefIntoJsonObject(
  content: string,
  key: string | undefined,
  unresolved: string[]
): string | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch {
    return null;
  }

  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return null;
  }

  const obj = parsed as Record<string, unknown>;
  const injectingRef = key != null;
  const injectingUnresolved = unresolved.length > 0;

  /**
   * Reject the JSON-injection path (fall back to prefix form) when
   * either of our keys collides with real payload data:
   *  - `_ref` collision: existing value is non-null and differs from
   *    the key we're about to inject.
   *  - `_unresolved_refs` collision: existing value is non-null and
   *    is not a deep-equal match for the array we'd inject.
   * This keeps us from silently overwriting legitimate tool output.
   */
  if (
    injectingRef &&
    TOOL_OUTPUT_REF_KEY in obj &&
    obj[TOOL_OUTPUT_REF_KEY] !== key &&
    obj[TOOL_OUTPUT_REF_KEY] != null
  ) {
    return null;
  }
  if (
    injectingUnresolved &&
    TOOL_OUTPUT_UNRESOLVED_KEY in obj &&
    obj[TOOL_OUTPUT_UNRESOLVED_KEY] != null &&
    !arraysShallowEqual(obj[TOOL_OUTPUT_UNRESOLVED_KEY], unresolved)
  ) {
    return null;
  }

  /**
   * Only strip the framework-owned key we're actually injecting —
   * leave everything else (including a pre-existing `_ref` on the
   * unresolved-only path, or a pre-existing `_unresolved_refs` on a
   * plain-annotation path) untouched so we annotate rather than
   * mutate downstream payload data. Our injected keys land first in
   * the serialized JSON so the LLM sees them before the body.
   */
  const omitKeys = new Set<string>();
  if (injectingRef) omitKeys.add(TOOL_OUTPUT_REF_KEY);
  if (injectingUnresolved) omitKeys.add(TOOL_OUTPUT_UNRESOLVED_KEY);
  const rest: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (!omitKeys.has(k)) {
      rest[k] = v;
    }
  }
  const injected: Record<string, unknown> = {};
  if (injectingRef) {
    injected[TOOL_OUTPUT_REF_KEY] = key;
  }
  if (injectingUnresolved) {
    injected[TOOL_OUTPUT_UNRESOLVED_KEY] = unresolved;
  }
  Object.assign(injected, rest);

  const pretty = /^\{\s*\n/.test(content);
  return pretty ? JSON.stringify(injected, null, 2) : JSON.stringify(injected);
}

function arraysShallowEqual(a: unknown, b: readonly string[]): boolean {
  if (!Array.isArray(a) || a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}

/**
 * Lazy projection that, given a registry and a runId, returns a new
 * `messages` array where each `ToolMessage` carrying ref metadata is
 * projected into a transient copy with annotated content (when the ref
 * is live in the registry) and with the framework-owned `additional_
 * kwargs` keys (`_refKey`, `_refScope`, `_unresolvedRefs`) stripped
 * regardless of whether annotation applied. The original input array
 * and its messages are never mutated.
 *
 * Annotation is gated on registry presence: a stale `_refKey` from a
 * prior run (e.g. one that survived in persisted history) silently
 * no-ops on the *content* side. The strip-metadata side still runs so
 * stale framework keys never leak onto the wire under any custom or
 * future provider serializer that might transmit `additional_kwargs`.
 * `_unresolvedRefs` is always meaningful and is not gated.
 *
 * **Feature-disabled fast path:** when the host hasn't enabled the
 * tool-output-reference feature, the registry is `undefined` and this
 * function returns the input array reference-equal *without iterating
 * a single message*. The loop is exclusive to the feature-enabled
 * code path.
 */
export function annotateMessagesForLLM(
  messages: BaseMessage[],
  registry: ToolOutputReferenceRegistry | undefined,
  runId: string | undefined
): BaseMessage[] {
  if (registry == null) return messages;

  /**
   * Lazy-allocate the output array so the common case (no ToolMessage
   * carries framework metadata) returns the input reference-equal with
   * zero allocations beyond the per-message predicate checks.
   */
  let out: BaseMessage[] | undefined;
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    if (m._getType() !== 'tool') continue;
    /**
     * `additional_kwargs` is untyped at the LangChain layer
     * (`Record<string, unknown>`), so persisted or client-supplied
     * ToolMessages can carry arbitrary shapes — including primitives
     * (a malformed serializer might write a string, or `null`).
     * Guard with a runtime object check before the `in` probes
     * because the `in` operator throws `TypeError` on primitives.
     * A single malformed message must never crash the provider call
     * path; skip its annotation/strip and continue.
     */
    const rawMeta = m.additional_kwargs as unknown;
    if (rawMeta == null || typeof rawMeta !== 'object') continue;
    const meta = rawMeta as Record<string, unknown>;
    const hasRefKey = '_refKey' in meta;
    const hasRefScope = '_refScope' in meta;
    const hasUnresolvedField = '_unresolvedRefs' in meta;
    if (!hasRefKey && !hasRefScope && !hasUnresolvedField) continue;
    if (isComputerCallOutputMessage(m)) {
      out ??= messages.slice();
      out[i] = cloneToolMessageWithContent(m as ToolMessage, m.content);
      continue;
    }

    const refKey = readRefKey(meta);
    const unresolved = readUnresolvedRefs(meta);

    /**
     * Prefer the message-stamped `_refScope` for the registry lookup.
     * For named runs it equals the current `runId`; for anonymous
     * invocations it carries the per-batch synthetic scope minted by
     * ToolNode (`\0anon-<n>`), which `runId` from config cannot
     * recover. Falling back to `runId` keeps backward compatibility
     * with messages stamped before this field existed.
     */
    const lookupScope = readRefScope(meta) ?? runId;
    const liveRef =
      refKey != null && registry.has(lookupScope, refKey) ? refKey : undefined;
    const annotates = liveRef != null || unresolved.length > 0;

    const tm = m as ToolMessage;
    let nextContent: ToolMessage['content'] = tm.content;

    if (annotates && typeof tm.content === 'string') {
      nextContent = annotateToolOutputWithReference(
        tm.content,
        liveRef,
        unresolved
      );
    } else if (annotates && Array.isArray(tm.content)) {
      /**
       * Array tool content. The string annotator can't run — this notably
       * includes a tail tool result that prompt caching rewrote from a string
       * into a text-block array to host its `cache_control` / `cachePoint`
       * marker (the `_refKey` survives on `additional_kwargs`). Project the
       * same markers the string path would, as leading text blocks: the live
       * `[ref: …]` prefix and/or the unresolved-refs warning. Without this the
       * common tool-result tail loses its reference marker once cached.
       *
       * `as unknown as ToolMessage['content']` is unavoidable: LangChain's
       * content union does not accept a freshly built mixed array literal even
       * though the structural shape is valid at runtime. The double-cast is
       * structurally safe — every original block is preserved and only
       * `{ type: 'text', text }` blocks (which all providers accept) are
       * prepended.
       */
      const prefixBlocks: Array<{ type: 'text'; text: string }> = [];
      if (liveRef != null) {
        prefixBlocks.push({
          type: 'text',
          text: buildReferencePrefix(liveRef),
        });
      }
      if (unresolved.length > 0) {
        prefixBlocks.push({
          type: 'text',
          text: `[unresolved refs: ${unresolved.join(', ')}]`,
        });
      }
      if (prefixBlocks.length > 0) {
        nextContent = [
          ...prefixBlocks,
          ...tm.content,
        ] as unknown as ToolMessage['content'];
      }
    }

    /**
     * Project unconditionally: even when no annotation applies (stale
     * `_refKey` or non-annotatable content), `cloneToolMessageWithContent`
     * runs `stripFrameworkRefMetadata` on `additional_kwargs` so the
     * framework-owned keys never reach the wire.
     */
    out ??= messages.slice();
    out[i] = cloneToolMessageWithContent(tm, nextContent);
  }

  return out ?? messages;
}

/**
 * Reads `_refKey` defensively from untyped `additional_kwargs`. Returns
 * undefined for non-string values so a malformed field cannot poison
 * the registry lookup or downstream string operations.
 */
function readRefKey(
  meta: Record<string, unknown> | undefined
): string | undefined {
  const v = meta?._refKey;
  return typeof v === 'string' ? v : undefined;
}

/**
 * Reads `_refScope` defensively from untyped `additional_kwargs`.
 * Mirrors {@link readRefKey} — non-string scopes are dropped (the
 * caller falls back to the run-derived scope) rather than passed into
 * the registry as a malformed key.
 */
function readRefScope(
  meta: Record<string, unknown> | undefined
): string | undefined {
  const v = meta?._refScope;
  return typeof v === 'string' ? v : undefined;
}

/**
 * Reads `_unresolvedRefs` defensively from untyped `additional_kwargs`.
 * Returns an empty array for any non-array value, and filters out
 * non-string entries from a real array. Without this guard, a hydrated
 * ToolMessage carrying e.g. `_unresolvedRefs: 'tool0turn0'` would crash
 * `attemptInvoke` on the eventual `.length` / `.join(...)` call.
 */
function readUnresolvedRefs(
  meta: Record<string, unknown> | undefined
): string[] {
  const v = meta?._unresolvedRefs;
  if (!Array.isArray(v)) return [];
  const out: string[] = [];
  for (const item of v) {
    if (typeof item === 'string') out.push(item);
  }
  return out;
}

/**
 * Builds a fresh `ToolMessage` that mirrors `tm`'s identity fields with
 * the supplied `content`. Every `ToolMessage` field but `content` is
 * carried over so the projection is structurally identical to the
 * original from a LangChain serializer's perspective.
 *
 * `additional_kwargs` is rebuilt with the framework-owned ref keys
 * stripped. Defensive: LangChain's standard provider serializers do not
 * transmit `additional_kwargs` to provider HTTP APIs, but a custom
 * adapter or future LangChain change could. Stripping keeps the
 * implementation correct under any serializer behavior at the cost of a
 * shallow object spread per annotated message.
 */
function cloneToolMessageWithContent(
  tm: ToolMessage,
  content: ToolMessage['content']
): ToolMessage {
  return new ToolMessage({
    id: tm.id,
    name: tm.name,
    status: tm.status,
    artifact: tm.artifact,
    tool_call_id: tm.tool_call_id,
    response_metadata: tm.response_metadata,
    additional_kwargs: stripFrameworkRefMetadata(tm.additional_kwargs),
    content,
  });
}

/**
 * Returns a copy of `kwargs` with `_refKey`, `_refScope`, and
 * `_unresolvedRefs` removed. Returns the input reference-equal when
 * none of those keys are present so the no-strip path stays cheap;
 * returns `undefined` when stripping leaves the object empty so the
 * caller can drop the field entirely.
 */
function stripFrameworkRefMetadata(
  kwargs: Record<string, unknown> | undefined
): Record<string, unknown> | undefined {
  if (kwargs == null) return undefined;
  if (
    !('_refKey' in kwargs) &&
    !('_refScope' in kwargs) &&
    !('_unresolvedRefs' in kwargs)
  ) {
    return kwargs;
  }
  const { _refKey, _refScope, _unresolvedRefs, ...rest } = kwargs as Record<
    string,
    unknown
  > & {
    _refKey?: unknown;
    _refScope?: unknown;
    _unresolvedRefs?: unknown;
  };
  void _refKey;
  void _refScope;
  void _unresolvedRefs;
  return Object.keys(rest).length === 0 ? undefined : rest;
}
