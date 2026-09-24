// src/llm/streamLimits.ts
import type { ToolCallChunk } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { getStreamedToolCallSeal } from '@/tools/streamedToolCallSeals';

/**
 * Circuit breakers for pathological model streams.
 *
 * A malformed generation can stream a single tool call's arguments for many
 * minutes at the provider's full token rate while the arguments never become
 * executable (observed live: one 149,923-char SQL argument streamed for 26
 * minutes before the 64k output-token ceiling finally ended the run). These
 * guards fail fast instead: when a limit trips, the stream handler throws
 * `StreamLimitExceededError` out of the run's `streamEvents` loop, which
 * tears down the in-flight provider request where it stands (see the
 * mid-flight halt notes in `Run.processStream`: leaving the loop cancels the
 * reader and langgraph aborts the model call).
 */

/** Default cap on a single streamed tool call's cumulative argument bytes (64 KiB). */
export const DEFAULT_MAX_TOOL_CALL_ARG_BYTES = 65_536;

/** Limits normalized by {@link resolveStreamLimits}; `0` uniformly means disabled. */
export interface ResolvedStreamLimits {
  maxToolCallArgBytes: number;
  /** Per-tool overrides of the byte cap, keyed by model-facing tool name. */
  maxToolCallArgBytesByTool?: Readonly<Record<string, number>>;
  maxDeltaEventsPerTurn: number;
  /** Precomputed once: whether ANY argument byte limit can fire. False when
   * the global cap is disabled and every per-tool entry is a zero-valued
   * disable — accounting must not allocate for guards that judge nothing. */
  hasEnforceableToolCallArgLimit: boolean;
}

/** Cumulative streamed argument bytes for one in-flight tool call. */
export interface StreamedToolCallArgTally {
  bytes: number;
  name?: string;
  /** The graph's breaker epoch when this tally was created. Entries from
   * the epoch that is ending survive one `resetValues` sweep, so producer
   * loops of straggling attempts — which are not behind the consumer-only
   * epoch gate — stay on their original budgets instead of receiving a
   * fresh allowance at every run start. */
  epoch?: number;
  /** Every tally-map key this tally is registered under: its primary key
   * (which can migrate through identifier transitions), the batch-position
   * fallback for id-bearing chunks, and the id for chunks carrying both
   * identifiers. Release deletes all of them — a call sealed through one
   * identity must not leave entries behind under another. */
  keys?: string[];
  /**
   * True when the previous chunk ended on an unpaired UTF-16 high surrogate.
   * Counting each half of a split surrogate pair alone yields 3 bytes per
   * half (the replacement-character encoding) versus 4 for the pair, so the
   * next chunk starting with the low surrogate reconciles by subtracting 2.
   */
  pendingHighSurrogate?: boolean;
}

/** Immutable snapshot of one run's breaker identity. Captured at
 * model/event/tool-batch entry and revalidated by REFERENCE after awaits —
 * one identity comparison proves no reset interleaved. */
export interface RunBreakerScope {
  readonly epoch: number;
  readonly controller: AbortController;
}

/** Streamed chunk events counted against one generation's event cap. */
export interface StreamDeltaEventTally {
  count: number;
  /** Creation-epoch tag, same grace semantics as
   * {@link StreamedToolCallArgTally.epoch}. */
  epoch?: number;
}

/**
 * The graph-owned state the guards read and write. Structural on purpose:
 * handler-level tests stub graphs with plain objects, and the guards lazily
 * create the tally maps so partial stubs need no extra setup.
 */
export interface StreamLimitState {
  streamLimits?: ResolvedStreamLimits;
  streamedToolCallArgTallies?: Map<string, StreamedToolCallArgTally>;
  streamDeltaEventCounts?: Map<string, StreamDeltaEventTally>;
  /** The graph's live breaker epoch; new accounting entries are tagged with
   * it so `resetValues` can sweep by age instead of clearing outright. */
  breakerEpoch?: number;
  /** Generation keys of model attempts still in flight. Each attempt leases
   * its generation at entry and releases it (with its accounting entries)
   * from its `finally`, so retention follows ATTEMPT LIFETIME — a
   * cancellation-ignoring straggler keeps its original budget no matter how
   * many runs start and reset while it drains. */
  activeStreamLimitGenerations?: Set<string>;
  /** Per-chunk-object, per-generation charge balance: producer visits
   * increment, consumer (handler echo) visits decrement, and a visit only
   * charges when the other side has not pre-charged the same emission.
   * Count-balancing rather than a lifetime set, because a streaming model
   * may mutate and re-yield the same chunk object; scoped by generation so
   * parallel generations sharing one reused object cannot cancel each
   * other's charges. */
  streamLimitChargeCredits?: WeakMap<object, Map<string, number>>;
}

function resolveLimit(value: number | undefined, fallback: number): number {
  if (value == null || Number.isNaN(value)) {
    return fallback;
  }
  if (!Number.isFinite(value)) {
    return 0;
  }
  const whole = Math.floor(value);
  return whole > 0 ? whole : 0;
}

/**
 * Normalizes host-supplied limits once at graph construction. `undefined`
 * applies the default for each guard (the tool-argument byte cap is ON by
 * default, the per-turn event cap is opt-in), `0` and negative values
 * disable a guard, `Infinity` means "no limit" and also disables, and `NaN`
 * falls back to the default.
 */
export function resolveStreamLimits(
  limits?: t.StreamLimits
): ResolvedStreamLimits {
  const maxToolCallArgBytes = resolveLimit(
    limits?.maxToolCallArgBytes,
    DEFAULT_MAX_TOOL_CALL_ARG_BYTES
  );
  let maxToolCallArgBytesByTool: Record<string, number> | undefined;
  if (limits?.maxToolCallArgBytesByTool != null) {
    for (const [name, value] of Object.entries(
      limits.maxToolCallArgBytesByTool
    )) {
      /** An unusable entry (NaN) falls back to the global cap by omission,
       * matching how the global field treats NaN. */
      if (name === '' || Number.isNaN(value)) {
        continue;
      }
      /** Prototype-free: bracket-assigning a `__proto__` tool name into a
       * plain object invokes the prototype setter instead of creating an own
       * property, silently dropping that tool's configured override. */
      maxToolCallArgBytesByTool ??= Object.create(null) as Record<
        string,
        number
      >;
      maxToolCallArgBytesByTool[name] = resolveLimit(
        value,
        maxToolCallArgBytes
      );
    }
  }
  const hasEnforceableToolCallArgLimit =
    maxToolCallArgBytes > 0 ||
    (maxToolCallArgBytesByTool != null &&
      Object.values(maxToolCallArgBytesByTool).some((value) => value > 0));
  return {
    maxToolCallArgBytes,
    ...(maxToolCallArgBytesByTool != null && { maxToolCallArgBytesByTool }),
    maxDeltaEventsPerTurn: resolveLimit(limits?.maxDeltaEventsPerTurn, 0),
    hasEnforceableToolCallArgLimit,
  };
}

const DEFAULT_RESOLVED_LIMITS: ResolvedStreamLimits = resolveStreamLimits();

export type StreamLimitKind = 'tool_call_args' | 'delta_events';

function buildLimitMessage(
  kind: StreamLimitKind,
  limit: number,
  toolName?: string
): string {
  if (kind === 'tool_call_args') {
    const named =
      toolName != null && toolName !== '' ? ` (tool call: ${toolName})` : '';
    return (
      `Streamed tool call arguments exceeded the ${limit}-byte safety limit${named}. ` +
      'The generation was aborted mid-stream: arguments growing this large without completing ' +
      'usually indicate a runaway or malformed tool call. Raise \'maxToolCallArgBytes\' (or the ' +
      'tool\'s \'maxToolCallArgBytesByTool\' entry) if your tools legitimately need larger arguments.'
    );
  }
  return (
    `Model stream exceeded the ${limit}-event safety limit for a single generation turn. ` +
    'The generation was aborted mid-stream: this usually indicates a looping or duplicated ' +
    'provider stream. Raise \'maxDeltaEventsPerTurn\' if legitimate generations need more stream events.'
  );
}

/**
 * Raised when a {@link t.StreamLimits} guard trips. Thrown from inside the
 * run's `streamEvents` loop, so the in-flight provider request is torn down
 * and `processStream` rejects with this error.
 */
export class StreamLimitExceededError extends Error {
  readonly kind: StreamLimitKind;
  readonly limit: number;
  readonly observed: number;
  readonly toolName?: string;

  constructor({
    kind,
    limit,
    observed,
    toolName,
  }: {
    kind: StreamLimitKind;
    limit: number;
    observed: number;
    toolName?: string;
  }) {
    super(buildLimitMessage(kind, limit, toolName));
    this.name = 'StreamLimitExceededError';
    this.kind = kind;
    this.limit = limit;
    this.observed = observed;
    this.toolName = toolName;
  }
}

/**
 * Identity of one model generation, derived from langgraph's node-execution
 * metadata. Deliberately NOT `Graph.getStepKey()`: the step key forks within
 * a single generation on reasoning transitions (`'reasoning'` /
 * `post-reasoning-<n>` suffixes in `getKeyList`) and on mid-turn server-tool
 * results (`invokedToolIds` count), which would hand a fresh budget to each
 * segment. One agent-node execution is one superstep, so
 * `checkpoint_ns + node + step` stays stable for the whole generation and
 * distinguishes parallel agents in the same superstep.
 *
 * The attempt stamp scopes attempts within one node execution:
 * `attemptInvoke` is the single funnel for primary, fallback, and
 * summarization model calls and stamps {@link STREAM_LIMIT_ATTEMPT_KEY}
 * with a unique sequence number into each attempt's callback metadata. A
 * fallback's chunks therefore key separately from the failed primary's,
 * even for two fallbacks configured with the same provider and model name,
 * and even when the decoupled `streamEvents` reader drains a failed
 * attempt's buffered chunks late. Those late chunks land in their own
 * attempt's bucket instead of polluting the next one's.
 */
export function resolveGenerationKey(
  metadata: Record<string, unknown> | undefined
): string {
  if (metadata == null) {
    return '';
  }
  const checkpointNs = metadata.langgraph_checkpoint_ns ?? '';
  const node = metadata.langgraph_node ?? '';
  const step = metadata.langgraph_step ?? '';
  const attempt = metadata[STREAM_LIMIT_ATTEMPT_KEY] ?? '';
  return `${checkpointNs}|${node}|${step}|${attempt}`;
}

/**
 * Metadata key carrying the unique per-model-attempt sequence number that
 * `attemptInvoke` stamps into every attempt's callback metadata. Part of
 * the generation key so budgets never alias across attempts.
 */
export const STREAM_LIMIT_ATTEMPT_KEY = 'lc_stream_limit_attempt';

/**
 * Event-metadata marker for a chunk the SDK re-dispatches inline after
 * transforming it (`attemptInvoke`'s OpenRouter final-reasoning replay). The
 * original wire chunk still reaches the handler through `streamEvents` and
 * is counted there, so the re-dispatch must not consume a second
 * event-budget slot.
 */
export const STREAM_LIMIT_REDISPATCH_KEY = 'lc_stream_limit_redispatch';

/**
 * Metadata key carrying the graph's breaker epoch at the time a model
 * attempt started. The stream handler trips the shared breaker only when
 * the event's epoch matches the live controller's — a straggling chunk from
 * a failed run that outlived `resetValues()` must fail its own (dead) run,
 * not abort the controller now serving the next one. A primitive rather
 * than the controller itself so attempt metadata stays serialization-safe
 * for tracing.
 */
export const STREAM_LIMIT_EPOCH_KEY = 'lc_stream_limit_epoch';

/**
 * Configurable key carrying the tool batch's entry-captured
 * {@link RunBreakerScope} to tools that spawn their own runs (subagents).
 * Captured BEFORE PreToolUse hooks, so a reset during a hook cannot rebind
 * the spawned child to the new run's controller. Stripped from host-facing
 * batch requests and from child-graph configurables.
 */
export const RUN_BREAKER_SCOPE_CONFIG_KEY = 'lc_run_breaker_scope';

/**
 * True when the event's `single` seal marks this chunk's own call as
 * complete. On the OpenAI Responses adapter the sealing chunk RESTATES the
 * full argument string (`response.function_call_arguments.done`), so summing
 * it would double-count every legitimate call; on Bedrock Converse the
 * sealing chunk carries empty args. Either way the call is finished: its
 * tally is replaced by the sealing chunk's own bytes, checked, and released.
 */
function sealsChunk(
  seal: ReturnType<typeof getStreamedToolCallSeal>,
  chunk: ToolCallChunk
): boolean {
  if (seal == null || seal.kind !== 'single') {
    return false;
  }
  /** Either supplied identifier agreeing is sufficient (matching the
   * eager-call seal handling in stream.ts): a mismatch on one identifier
   * must not veto a match on the other, or an OpenAI-style full-argument
   * restatement gets ADDED to the tally instead of replacing it, and an
   * empty seal leaves the tally unreleased. */
  if (
    seal.index != null &&
    chunk.index != null &&
    seal.index === chunk.index
  ) {
    return true;
  }
  const sealId = seal.id != null && seal.id !== '' ? seal.id : undefined;
  return sealId != null && sealId === chunkCallId(chunk);
}

function chunkToolName(chunk: ToolCallChunk): string | undefined {
  return chunk.name != null && chunk.name !== '' ? chunk.name : undefined;
}

/** The chunk's id with empty-string placeholders treated as absent —
 * OpenAI-compatible adapters emit `id: ''` on continuation deltas, and a
 * shared placeholder must not merge independent calls onto one identity
 * (mirrors `getEagerToolChunkKey`). */
function chunkCallId(chunk: ToolCallChunk): string | undefined {
  return chunk.id != null && chunk.id !== '' ? chunk.id : undefined;
}

function isHighSurrogate(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff;
}

function isLowSurrogate(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff;
}

/**
 * Accumulates the UTF-8 byte size of streamed tool-call argument chunks per
 * in-flight tool call and throws once a single call's cumulative bytes
 * exceed `maxToolCallArgBytes`. Runs once per streamed chunk event, before
 * complete tool calls are dispatched or eagerly executed and before chunks
 * are recorded, so a tripped limit stops the run without dispatching the
 * offending call.
 *
 * Calls are keyed by generation and chunk `index`, falling back to the
 * chunk `id` and then to the chunk's position within the event's batch when
 * a provider identifies chunks by neither (Google emits complete parallel
 * calls with optional ids and no index). A `kind: 'all'` arrival seal marks
 * every chunk in the event as its own complete call, so those are checked
 * standalone and never share a budget; a matching `kind: 'single'` seal
 * replaces the call's tally with the sealing chunk's own bytes (the OpenAI
 * Responses done-chunk restates the full argument string) and releases it.
 */
export function enforceStreamedToolCallArgLimit({
  graph,
  metadata,
  toolCallChunks,
  responseMetadata,
  parsedToolCalls,
}: {
  graph: StreamLimitState;
  metadata: Record<string, unknown> | undefined;
  toolCallChunks: ToolCallChunk[];
  responseMetadata?: Record<string, unknown>;
  /** Complete parsed calls from the same event, used to name anonymous raw
   * chunks so per-tool overrides are honored before the global cap trips. */
  parsedToolCalls?: CompleteToolCallLike[];
}): void {
  const resolved = graph.streamLimits ?? DEFAULT_RESOLVED_LIMITS;
  const globalLimit = resolved.maxToolCallArgBytes;
  const byTool = resolved.maxToolCallArgBytesByTool;
  if (!resolved.hasEnforceableToolCallArgLimit) {
    return;
  }
  /** An inline re-dispatch of a transformed chunk (OpenRouter final-reasoning
   * replay) duplicates tool-call chunks the original `streamEvents` event
   * already charged — the original always survives the content-specific
   * skips when it carries tool calls, so counting the marked copy would
   * double every legitimate argument byte. */
  if (metadata?.[STREAM_LIMIT_REDISPATCH_KEY] === true) {
    return;
  }
  const tallies: Map<string, StreamedToolCallArgTally> =
    (graph.streamedToolCallArgTallies ??= new Map<
      string,
      StreamedToolCallArgTally
    >());
  const generationKey = resolveGenerationKey(metadata);
  const seal = getStreamedToolCallSeal(responseMetadata);
  /** The complete call in this event sharing the chunk's id — the strongest
   * name evidence: adapters may stream raw names in FRAGMENTS, and an
   * id-correlated complete call states the full name outright. */
  const correlateParsedNameById = (id: string): string | undefined => {
    if (parsedToolCalls == null) {
      return undefined;
    }
    for (const parsed of parsedToolCalls) {
      if (parsed.id === id && parsed.name != null && parsed.name !== '') {
        return parsed.name;
      }
    }
    return undefined;
  };
  const generationPrefix = `${generationKey}:`;
  const getLiveGenerationTallies = (): StreamedToolCallArgTally[] => {
    const live = new Set<StreamedToolCallArgTally>();
    for (const [tallyKey, tally] of tallies) {
      if (tallyKey.startsWith(generationPrefix)) {
        live.add(tally);
      }
    }
    return [...live];
  };
  const resolveChunkName = (chunk: ToolCallChunk): string | undefined => {
    const chunkId = chunkCallId(chunk);
    const correlated =
      chunkId != null ? correlateParsedNameById(chunkId) : undefined;
    if (correlated != null) {
      return correlated;
    }
    const name = chunkToolName(chunk);
    if (name != null || parsedToolCalls == null || chunkId != null) {
      return name;
    }
    /** No id to correlate on; positional association is unambiguous only
     * when the event carries exactly one raw chunk AND one parsed call —
     * with several id-less raw chunks in flight, handing one parsed call's
     * name (and its override) to all of them would let an unrelated
     * still-partial call bypass the global cap. */
    if (parsedToolCalls.length === 1 && toolCallChunks.length === 1) {
      const only = parsedToolCalls[0];
      if (only.name != null && only.name !== '') {
        return only.name;
      }
    }
    return undefined;
  };
  const enforceTallyLimit = (target: StreamedToolCallArgTally): void => {
    const tallyLimit =
      byTool != null &&
      target.name != null &&
      Object.hasOwn(byTool, target.name)
        ? byTool[target.name]
        : globalLimit;
    if (tallyLimit > 0 && target.bytes > tallyLimit) {
      throw new StreamLimitExceededError({
        kind: 'tool_call_args',
        limit: tallyLimit,
        observed: target.bytes,
        toolName: target.name,
      });
    }
  };
  for (let i = 0; i < toolCallChunks.length; i++) {
    const chunk = toolCallChunks[i];
    const args = chunk.args;
    const hasArgs = typeof args === 'string' && args !== '';
    if (seal?.kind === 'all') {
      if (!hasArgs) {
        continue;
      }
      const bytes = Buffer.byteLength(args, 'utf8');
      const sealedName = resolveChunkName(chunk);
      const limit =
        byTool != null && sealedName != null && Object.hasOwn(byTool, sealedName)
          ? byTool[sealedName]
          : globalLimit;
      if (limit > 0 && bytes > limit) {
        throw new StreamLimitExceededError({
          kind: 'tool_call_args',
          limit,
          observed: bytes,
          toolName: sealedName,
        });
      }
      continue;
    }
    /** Keys are namespaced by identity kind (`i:` index, `c:` id, `#`
     * batch position) so an index and a string id with the same textual
     * value — index 0 and id "0" — cannot alias distinct calls onto one
     * tally. Empty-string ids are placeholders, not identities. */
    const chunkId = chunkCallId(chunk);
    const sealed = sealsChunk(seal, chunk);
    /** Applies this chunk's name contribution and reports whether the
     * effective name changed. Adapters may stream the name in FRAGMENTS
     * ("create_" then "file"), so unsealed fragments append — matching
     * langchain's own tool-call-chunk merge — while a sealing chunk's name
     * is a full restatement and replaces, mirroring how sealed args replace
     * the tally bytes. An id-correlated complete call in the same event
     * outranks both; positional parsed-call correlation fills in only when
     * the chunk carries no fragment. */
    const applyChunkName = (target: StreamedToolCallArgTally): boolean => {
      const correlated =
        chunkId != null ? correlateParsedNameById(chunkId) : undefined;
      const fragment = chunkToolName(chunk);
      let next = target.name;
      if (correlated != null) {
        /** An id-correlated complete call wins over fragment accumulation:
         * a raw name like "create_" must not hide the "create_file"
         * override behind the global cap. */
        next = correlated;
      } else if (fragment != null) {
        if (sealed || target.name == null) {
          next = fragment;
        } else if (fragment !== target.name) {
          /** A fragment identical to the accumulated name is a repeated
           * full-name delta (a common provider shape) and is a no-op; a
           * differing fragment is a continuation and appends. */
          next = target.name + fragment;
        }
      } else if (target.name == null) {
        next = resolveChunkName(chunk);
      }
      if (next === target.name) {
        return false;
      }
      target.name = next;
      return true;
    };
    /** A single identifier-less event while calls are live is a sparse
     * parallel continuation: its position is relative to this event, not to
     * the original batch, so `#0` cannot identify which call it belongs to.
     * With exactly one live call the association is unambiguous — charge
     * that tally, keeping its budget continuous (a fresh `#0` tally here
     * would RESET the sole remaining call's byte budget). With several,
     * charge every live candidate. Each tally is judged under its OWN
     * name-specific limit: comparing only against the global cap would let
     * a call with a lower per-tool override stream past it indefinitely
     * while its chunks stay anonymous. */
    if (
      chunk.index == null &&
      chunkId == null &&
      toolCallChunks.length === 1
    ) {
      const liveTallies = getLiveGenerationTallies();
      if (liveTallies.length > 0) {
        if (liveTallies.length === 1 && applyChunkName(liveTallies[0])) {
          /** A late name can select a lower override than the limit under
           * which prior bytes were accepted, so re-judge before charging
           * this continuation. */
          enforceTallyLimit(liveTallies[0]);
        }
        if (hasArgs) {
          const argBytes = Buffer.byteLength(args, 'utf8');
          for (const liveTally of liveTallies) {
            const reconcilesSplitPair =
              liveTally.pendingHighSurrogate === true &&
              isLowSurrogate(args.charCodeAt(0));
            liveTally.bytes += reconcilesSplitPair ? argBytes - 2 : argBytes;
            liveTally.pendingHighSurrogate = isHighSurrogate(
              args.charCodeAt(args.length - 1)
            );
            enforceTallyLimit(liveTally);
          }
        }
        continue;
      }
    }
    let key: string;
    if (chunk.index != null) {
      key = `${generationKey}:i:${chunk.index}`;
    } else if (chunkId != null) {
      key = `${generationKey}:c:${chunkId}`;
    } else {
      key = `${generationKey}:#${i}`;
    }
    /** Batch position is a stable identity only when the event carries a
     * single chunk: sparse parallel events reuse position 0 for whichever
     * call happens to continue, and a position alias would hand one call's
     * tally — and its per-tool override — to another live call's deltas. */
    const positionAliasSafe = toolCallChunks.length === 1;
    /** Secondary keys this call is also reachable under, so a later delta
     * that drops one or both identifiers still lands on the same tally:
     * id-bearing chunks register the batch-position fallback (single-chunk
     * events only), and chunks carrying both identifiers additionally
     * register the id. */
    const aliasCandidates: string[] = [];
    if (chunkId != null) {
      if (positionAliasSafe) {
        aliasCandidates.push(`${generationKey}:#${i}`);
      }
      if (chunk.index != null) {
        aliasCandidates.push(`${generationKey}:c:${chunkId}`);
      }
    }
    let tally = tallies.get(key);
    /** A later delta can also ADD an identifier, changing the primary key;
     * adopt the call's existing tally through a stronger prior identity
     * before allocating. An index maps to that call's original batch
     * position, never this sparse event's position. A newly arrived id can
     * adopt a position only when one anonymous tally is live; with several,
     * positional association is ambiguous. The old identity is recorded as
     * an alias and released together with the rest. */
    if (tally == null) {
      const adoptionCandidates: string[] = [];
      if (chunkId != null) {
        adoptionCandidates.push(`${generationKey}:c:${chunkId}`);
      }
      if (chunk.index != null) {
        /** An index is stable across sparse events; the loop position is
         * merely this event's position and may be 0 for call index 1. */
        const indexedPositionKey = `${generationKey}:#${chunk.index}`;
        adoptionCandidates.push(indexedPositionKey);
        /** A single anonymous call may first appear alone at event position
         * #0 and reveal a nonzero provider index only later. Exact indexed
         * position wins above; when it does not exist, one anonymous tally
         * is still an unambiguous fallback. */
        const liveTallies = getLiveGenerationTallies();
        const soleTally = liveTallies.length === 1 ? liveTallies[0] : undefined;
        if (
          soleTally?.keys?.[0]?.startsWith(`${generationPrefix}#`) === true
        ) {
          const anonymousKey = soleTally.keys[0];
          if (anonymousKey !== indexedPositionKey) {
            adoptionCandidates.push(anonymousKey);
          }
        }
      } else if (chunkId != null) {
        const anonymousTallies = getLiveGenerationTallies().filter(
          (liveTally: StreamedToolCallArgTally) =>
            liveTally.keys?.[0]?.startsWith(`${generationPrefix}#`) === true
        );
        if (anonymousTallies.length === 1) {
          const anonymousKey = anonymousTallies[0].keys?.[0];
          if (anonymousKey != null) {
            adoptionCandidates.push(anonymousKey);
          }
        }
      }
      for (const adoptionKey of adoptionCandidates) {
        if (adoptionKey === key) {
          continue;
        }
        const existing = tallies.get(adoptionKey);
        if (existing == null) {
          continue;
        }
        /** A position entry may only be adopted when the tally was CREATED
         * anonymous (its first key is the position): id-bearing calls also
         * alias their position, and adopting a live parallel call's alias
         * would merge distinct budgets. */
        if (
          adoptionKey.includes(':#') &&
          existing.keys?.[0] !== adoptionKey
        ) {
          continue;
        }
        tally = existing;
        tallies.set(key, existing);
        if (existing.keys?.includes(adoptionKey) !== true) {
          (existing.keys ??= []).push(adoptionKey);
        }
        if (!existing.keys.includes(key)) {
          existing.keys.push(key);
        }
        break;
      }
    }
    const registerAliasKeys = (
      target: StreamedToolCallArgTally,
      candidates: string[]
    ): void => {
      for (const aliasKey of candidates) {
        const currentOwner = tallies.get(aliasKey);
        if (currentOwner === target) {
          continue;
        }
        /** Parallel calls can contend for one batch position, so the newest
         * live call takes the alias over and the previous owner is disowned
         * — its seal must not delete a key it no longer holds. */
        if (currentOwner?.keys != null) {
          const remaining = currentOwner.keys.filter(
            (ownedKey: string) => ownedKey !== aliasKey
          );
          currentOwner.keys = remaining.length > 0 ? remaining : undefined;
        }
        tallies.set(aliasKey, target);
        (target.keys ??= []).push(aliasKey);
      }
    };
    const registerAliases = (target: StreamedToolCallArgTally): void => {
      registerAliasKeys(target, aliasCandidates);
    };
    /** Index-only calls register their batch position ONCE, at creation:
     * later deltas that drop the index then land on the same tally, while
     * the per-delta hot path (indexed id-less deltas) stays free of alias
     * work. */
    const registerCreationPositionAlias = (
      target: StreamedToolCallArgTally
    ): void => {
      if (chunkId != null || chunk.index == null || sealed) {
        return;
      }
      if (!positionAliasSafe) {
        return;
      }
      registerAliasKeys(target, [`${generationKey}:#${i}`]);
    };
    const releaseTally = (target: StreamedToolCallArgTally): void => {
      tallies.delete(key);
      if (target.keys == null) {
        return;
      }
      for (const aliasKey of target.keys) {
        if (tallies.get(aliasKey) === target) {
          tallies.delete(aliasKey);
        }
      }
    };
    if (!hasArgs) {
      if (tally == null) {
        if (!sealed) {
          tally = {
            bytes: 0,
            name: resolveChunkName(chunk),
            keys: [key],
            epoch: graph.breakerEpoch,
          };
          tallies.set(key, tally);
          registerAliases(tally);
          registerCreationPositionAlias(tally);
        }
        continue;
      }
      /** A sealing chunk is about to release this tally; taking the position
       * alias here would steal it from a still-live parallel call. */
      if (!sealed) {
        registerAliases(tally);
      }
      if (applyChunkName(tally)) {
        /** Bytes tallied under the previous (or absent) name were held
         * against that name's limit; a changed name — late arrival or a
         * completed fragment — must re-judge them, including on a sealing
         * chunk about to release the tally. */
        enforceTallyLimit(tally);
      }
      if (sealed) {
        releaseTally(tally);
      }
      continue;
    }
    if (tally == null) {
      tally = { bytes: 0, keys: [key], epoch: graph.breakerEpoch };
      tallies.set(key, tally);
      registerCreationPositionAlias(tally);
    }
    if (!sealed) {
      registerAliases(tally);
    }
    applyChunkName(tally);
    const argBytes = Buffer.byteLength(args, 'utf8');
    if (sealed) {
      tally.bytes = argBytes;
    } else {
      const reconcilesSplitPair =
        tally.pendingHighSurrogate === true &&
        isLowSurrogate(args.charCodeAt(0));
      tally.bytes += reconcilesSplitPair ? argBytes - 2 : argBytes;
    }
    tally.pendingHighSurrogate =
      !sealed && isHighSurrogate(args.charCodeAt(args.length - 1));
    const toolName = tally.name ?? resolveChunkName(chunk);
    const limit =
      byTool != null && toolName != null && Object.hasOwn(byTool, toolName)
        ? byTool[toolName]
        : globalLimit;
    if (limit > 0 && tally.bytes > limit) {
      throw new StreamLimitExceededError({
        kind: 'tool_call_args',
        limit,
        observed: tally.bytes,
        toolName,
      });
    }
    if (sealed) {
      releaseTally(tally);
    }
  }
}

/** Structural subset of a complete parsed tool call. */
interface CompleteToolCallLike {
  id?: string;
  name?: string;
  args?: unknown;
}

/**
 * Standalone byte check for complete parsed tool calls that arrive without a
 * raw chunk representation: a streaming custom or OpenAI-compatible
 * `ChatModel` can yield fully parsed `tool_calls` with empty
 * `tool_call_chunks`, which would otherwise dispatch without consuming any
 * byte budget. Complete calls are self-contained, so each is judged
 * standalone without tallying.
 */
export function enforceCompleteToolCallArgLimit({
  graph,
  metadata,
  toolCalls,
}: {
  graph: StreamLimitState;
  metadata: Record<string, unknown> | undefined;
  toolCalls: CompleteToolCallLike[];
}): void {
  const resolved = graph.streamLimits ?? DEFAULT_RESOLVED_LIMITS;
  const globalLimit = resolved.maxToolCallArgBytes;
  const byTool = resolved.maxToolCallArgBytesByTool;
  if (!resolved.hasEnforceableToolCallArgLimit) {
    return;
  }
  if (metadata?.[STREAM_LIMIT_REDISPATCH_KEY] === true) {
    return;
  }
  for (const toolCall of toolCalls) {
    const name =
      toolCall.name != null && toolCall.name !== '' ? toolCall.name : undefined;
    const limit =
      byTool != null && name != null && Object.hasOwn(byTool, name)
        ? byTool[name]
        : globalLimit;
    if (limit <= 0) {
      continue;
    }
    const args = toolCall.args;
    let serialized: string;
    if (typeof args === 'string') {
      serialized = args;
    } else if (args == null) {
      serialized = '';
    } else {
      serialized = JSON.stringify(args);
    }
    const bytes = Buffer.byteLength(serialized, 'utf8');
    if (bytes > limit) {
      throw new StreamLimitExceededError({
        kind: 'tool_call_args',
        limit,
        observed: bytes,
        toolName: name,
      });
    }
  }
}

/**
 * Whether a chunk needs charge accounting at all. With the event cap off
 * (the default) and no tool payload on the chunk, there is nothing a claim
 * could ever gate — skipping restores the documented zero-cost-disabled
 * behavior by avoiding a WeakMap entry and nested Map per ordinary text
 * delta. Both the producer and consumer paths use this same predicate on
 * the same chunk, so claim pairing is unaffected.
 */
export function requiresStreamLimitAccounting(
  graph: StreamLimitState,
  chunk: {
    tool_call_chunks?: unknown[];
    tool_calls?: unknown[];
    invalid_tool_calls?: unknown[];
  }
): boolean {
  const resolved = graph.streamLimits ?? DEFAULT_RESOLVED_LIMITS;
  if (resolved.maxDeltaEventsPerTurn > 0) {
    return true;
  }
  /** With no argument limit able to fire (byte cap disabled and every
   * per-tool entry a zero-valued disable), neither enforcement function can
   * trip — hosts that explicitly disable the guards for legitimate large
   * arguments must not pay per-chunk claim allocations (generation keys,
   * WeakMap entries, nested Maps) for bookkeeping that judges nothing. */
  if (!resolved.hasEnforceableToolCallArgLimit) {
    return false;
  }
  return (
    (chunk.tool_call_chunks?.length ?? 0) > 0 ||
    (chunk.tool_calls?.length ?? 0) > 0 ||
    (chunk.invalid_tool_calls?.length ?? 0) > 0
  );
}

/** True when any stream-limit guard can fire for this graph. The attempt
 * lease and per-chunk claims are both gated on it — fully disabled guards
 * must allocate no bookkeeping at all, per-attempt included. */
export function streamLimitAccountingEnabled(
  graph: StreamLimitState
): boolean {
  const resolved = graph.streamLimits ?? DEFAULT_RESOLVED_LIMITS;
  return (
    resolved.hasEnforceableToolCallArgLimit ||
    resolved.maxDeltaEventsPerTurn > 0
  );
}

/** Leases a model attempt's generation: entries under it are exempt from
 * the reset sweep until {@link releaseStreamLimitGeneration} runs from the
 * attempt's `finally`. */
export function registerActiveStreamLimitGeneration(
  graph: StreamLimitState,
  generationKey: string
): void {
  (graph.activeStreamLimitGenerations ??= new Set()).add(generationKey);
}

/** Ends an attempt's lease and deletes its accounting entries — the
 * authoritative retirement point for attempt-scoped state. */
export function releaseStreamLimitGeneration(
  graph: StreamLimitState,
  generationKey: string
): void {
  graph.activeStreamLimitGenerations?.delete(generationKey);
  deleteGenerationEntries(graph.streamedToolCallArgTallies, generationKey);
  deleteGenerationEntries(graph.streamDeltaEventCounts, generationKey);
}

function deleteGenerationEntries(
  entries: Map<string, unknown> | undefined,
  generationKey: string
): void {
  if (entries == null) {
    return;
  }
  const prefix = `${generationKey}:`;
  for (const key of entries.keys()) {
    if (key === generationKey || key.startsWith(prefix)) {
      entries.delete(key);
    }
  }
}

function isOwnedByActiveGeneration(
  key: string,
  activeGenerations: ReadonlySet<string>
): boolean {
  for (const generationKey of activeGenerations) {
    if (key === generationKey || key.startsWith(`${generationKey}:`)) {
      return true;
    }
  }
  return false;
}

/**
 * Deletes accounting entries older than the epoch that is ending, EXCEPT
 * entries leased by a still-active attempt. Called by `resetValues` instead
 * of clearing: producer loops of straggling attempts use the graph's maps
 * directly and are not behind the consumer-only epoch gate, so a clear
 * would hand a cancellation-ignoring provider a fresh allowance at every
 * run start. Leased entries live until their attempt's `finally` releases
 * them; unleased entries (direct callers with no attempt stamp) get one
 * grace reset via their epoch tag.
 */
export function sweepStaleStreamLimitEntries(
  entries: Map<string, { epoch?: number }>,
  endingEpoch: number,
  activeGenerations?: ReadonlySet<string>
): void {
  const hasActive = activeGenerations != null && activeGenerations.size > 0;
  for (const [key, value] of entries) {
    if (value.epoch === endingEpoch) {
      continue;
    }
    if (hasActive && isOwnedByActiveGeneration(key, activeGenerations)) {
      continue;
    }
    entries.delete(key);
  }
}

/** Links a secondary representation of a wire chunk (e.g. a provider
 * adapter's callback copy) to its canonical emission object, so claim
 * accounting treats both as one emission even though their identities
 * differ. Non-enumerable, so the link stays out of serialization and chunk
 * merges. */
const STREAM_LIMIT_CANONICAL: unique symbol = Symbol('streamLimitCanonical');

export function linkStreamLimitCanonical(
  copy: object,
  canonical: object
): void {
  Object.defineProperty(copy, STREAM_LIMIT_CANONICAL, {
    value: canonicalChunk(canonical),
    enumerable: false,
    configurable: true,
  });
}

function canonicalChunk(chunk: object): object {
  const linked = (chunk as Record<PropertyKey, unknown>)[
    STREAM_LIMIT_CANONICAL
  ];
  return typeof linked === 'object' && linked != null ? linked : chunk;
}

/**
 * Claims accounting ownership of one EMISSION of a wire chunk. LangChain can
 * hand the same chunk object to the decoupled `streamEvents` handler
 * (`consumer`) and to the dispatch loop (`producer`) in either order, and a
 * streaming model may mutate and re-yield the same object across emissions —
 * so dedup is a signed credit balance per object rather than a lifetime set.
 * Each producer visit adds a credit, each consumer visit removes one, and a
 * visit charges only when the other side has not already charged that
 * emission (positive balance = producer ahead, negative = consumer ahead).
 * Paths where only one side ever observes the chunk (summarization, local
 * replay-skip) charge every visit, since their balance never crosses zero
 * the other way. Balances are scoped by generation identity so parallel
 * generations sharing one reused chunk object cannot cancel each other's
 * charges. Non-object chunks cannot be identity-tracked and are always
 * claimable.
 */
export function claimStreamLimitCharge(
  graph: StreamLimitState,
  chunk: unknown,
  side: 'producer' | 'consumer',
  metadata: Record<string, unknown> | undefined
): boolean {
  if (typeof chunk !== 'object' || chunk == null) {
    return true;
  }
  const emission = canonicalChunk(chunk);
  const credits = (graph.streamLimitChargeCredits ??= new WeakMap());
  let byGeneration = credits.get(emission);
  if (byGeneration == null) {
    byGeneration = new Map();
    credits.set(emission, byGeneration);
  }
  const generationKey = resolveGenerationKey(metadata);
  const balance = byGeneration.get(generationKey) ?? 0;
  if (side === 'producer') {
    byGeneration.set(generationKey, balance + 1);
    return balance >= 0;
  }
  byGeneration.set(generationKey, balance - 1);
  return balance <= 0;
}

/**
 * Synchronous producer-side accounting for wire chunks that would otherwise
 * be judged only when the decoupled `streamEvents` reader catches up — or,
 * on replay-skipped and summarization chunks, not at all. A lagging reader
 * would let an oversized complete call return to LangGraph and reach
 * `ToolNode` before the queued handler throws; charging in the producer
 * loop keeps the breaker ahead of graph progression. Claim-based, so
 * whichever of this path and the handler echo sees the chunk object first
 * charges it and the other skips.
 */
export function enforceStreamLimitsForWireChunk({
  graph,
  metadata,
  chunk,
  side = 'producer',
}: {
  graph: StreamLimitState;
  metadata: Record<string, unknown> | undefined;
  chunk: {
    tool_call_chunks?: ToolCallChunk[];
    tool_calls?: CompleteToolCallLike[];
    invalid_tool_calls?: CompleteToolCallLike[];
    response_metadata?: Record<string, unknown>;
  };
  /** Claim side for the credit balance. The local dispatch branch charges
   * as `consumer` because its handler-handled and replay-skipped emissions
   * of one reused object ALTERNATE — mixed sides would pair them as
   * producer/echo and swallow a charge. */
  side?: 'producer' | 'consumer';
}): void {
  if (!requiresStreamLimitAccounting(graph, chunk)) {
    return;
  }
  if (!claimStreamLimitCharge(graph, chunk, side, metadata)) {
    return;
  }
  enforceStreamDeltaEventLimit({ graph, metadata });
  /** Combined view computed first so the raw-chunk guard can correlate
   * names from BOTH parsed and invalid calls in the same event; an unnamed
   * raw chunk twinned with a named invalid call must select that tool's
   * override, not the global cap. */
  const completeCalls = combineCompleteToolCalls(chunk);
  if (chunk.tool_call_chunks != null && chunk.tool_call_chunks.length > 0) {
    enforceStreamedToolCallArgLimit({
      graph,
      metadata,
      toolCallChunks: chunk.tool_call_chunks,
      responseMetadata: chunk.response_metadata,
      parsedToolCalls: completeCalls,
    });
  }
  /** Judged whenever parsed calls are present, not only when raw chunks are
   * absent: an adapter can pair an empty or partial raw chunk with a
   * complete parsed call, and the standalone check is stateless so the
   * common both-present case is not double-tallied. `invalid_tool_calls`
   * are included — ToolNode processes and promotes them, and a malformed
   * call streaming oversized arguments is the exact pathology this breaker
   * exists to stop. */
  if (completeCalls != null) {
    enforceCompleteToolCallArgLimit({ graph, metadata, toolCalls: completeCalls });
  }
}

/** Combined view of a chunk's parsed and invalid complete calls, avoiding
 * allocation on the common paths where one or both are absent. */
export function combineCompleteToolCalls(chunk: {
  tool_calls?: CompleteToolCallLike[];
  invalid_tool_calls?: CompleteToolCallLike[];
}): CompleteToolCallLike[] | undefined {
  const parsed = chunk.tool_calls;
  const invalid = chunk.invalid_tool_calls;
  const hasParsed = parsed != null && parsed.length > 0;
  const hasInvalid = invalid != null && invalid.length > 0;
  if (hasParsed && hasInvalid) {
    return [...parsed, ...invalid];
  }
  if (hasParsed) {
    return parsed;
  }
  return hasInvalid ? invalid : undefined;
}

/**
 * Counts streamed chunk events per model generation and throws once a single
 * generation exceeds `maxDeltaEventsPerTurn`. Opt-in defense in depth for
 * pathologies a byte cap cannot see, such as a provider stream looping on
 * empty chunks. Zero cost while disabled.
 */
export function enforceStreamDeltaEventLimit({
  graph,
  metadata,
}: {
  graph: StreamLimitState;
  metadata: Record<string, unknown> | undefined;
}): void {
  const limit = (graph.streamLimits ?? DEFAULT_RESOLVED_LIMITS)
    .maxDeltaEventsPerTurn;
  if (limit <= 0) {
    return;
  }
  if (metadata?.[STREAM_LIMIT_REDISPATCH_KEY] === true) {
    return;
  }
  const counts = (graph.streamDeltaEventCounts ??= new Map());
  const key = resolveGenerationKey(metadata);
  const existing = counts.get(key);
  const next = (existing?.count ?? 0) + 1;
  if (next > limit) {
    throw new StreamLimitExceededError({
      kind: 'delta_events',
      limit,
      observed: next,
    });
  }
  if (existing != null) {
    existing.count = next;
  } else {
    counts.set(key, { count: next, epoch: graph.breakerEpoch });
  }
}
