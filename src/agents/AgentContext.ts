/* eslint-disable no-console */
import { RunnableLambda } from '@langchain/core/runnables';
import {
  HumanMessage,
  SystemMessage,
  coerceMessageLikeToMessage,
} from '@langchain/core/messages';
import type {
  UsageMetadata,
  BaseMessage,
  BaseMessageFields,
  BaseMessageLike,
} from '@langchain/core/messages';
import type { RunnableConfig, Runnable } from '@langchain/core/runnables';
import type { ExactTokenCountCache } from '@/llm/contextPressureMeter';
import type * as t from '@/types';
import {
  type CallerCapabilityProjection,
  allowsToolCaller,
  applyCallerCapabilityDefinitionOverrides,
  createCallerCapabilityProjectionSnapshot,
  isToolDefinitionActive,
  isProgrammaticControlTool,
  mergeCallerCapabilityDefinitions,
  resolveCallerCapabilityProjection,
} from '@/tools/CallerCapabilities';
import {
  addTailCacheControl,
  addCacheControlToStablePrefixMessages,
  buildAnthropicCacheControl,
  buildBedrockCachePoint,
  resolvePromptCacheTtl,
  resolveBedrockPromptCacheTtl,
  cloneMessage,
  type PromptCacheTtl,
} from '@/messages/cache';
import {
  isProgrammaticRunnerAutoBound,
  isProgrammaticRunnerResolvedDirectly,
  resolveLocalImplementationNames,
  resolveLocalToolRegistry,
} from '@/tools/local/resolveLocalExecutionTools';
import {
  DEFAULT_RESERVE_RATIO,
  ORIGINAL_CONTENT_MAX_CHARS,
  clampCalibrationRatio,
  createPruneMessages,
  syncBudgetDerivedFields,
} from '@/messages';
import {
  ANTHROPIC_TOOL_TOKEN_MULTIPLIER,
  DEFAULT_TOOL_TOKEN_MULTIPLIER,
  ContentTypes,
  Constants,
  Providers,
} from '@/common';
import { isTokenCounterCacheCompatible } from '@/llm/tokenCounterCacheCompatibility';
import { snapshotCompactionSemanticIndex } from '@/summarization/semanticIndex';
import { createExactTokenCountCache } from '@/llm/contextPressureMeter';
import { buildSummaryCarrierText } from '@/summarization/shared';
import { createSchemaOnlyTools } from '@/tools/schema';
import { apportionTokenCounts } from '@/utils/tokens';
import { isThinkingEnabled } from '@/llm/request';
import { toJsonSchema } from '@/utils/schema';

type AgentSystemTextBlock = {
  type: 'text';
  text: string;
  cache_control?: { type: 'ephemeral'; ttl?: '1h' };
};

type AgentSystemContentBlock =
  | AgentSystemTextBlock
  | { cachePoint: { type: 'default'; ttl?: '1h' } };

type PromptCacheProvider = Providers.ANTHROPIC | Providers.OPENROUTER;

type ProgrammaticToolInstructionTarget = {
  name: string;
  codeGuidance: string;
  executesDirectly: boolean;
};

/**
 * Consecutive summarization attempts that may return no usable summary before
 * the run stops asking. Every such attempt spends a full model call over the
 * whole history and leaves the message set exactly as it was, so the cap
 * bounds spend as much as it bounds the compaction loop.
 */
const MAX_SUMMARIZATION_FAILURES = 3;

/**
 * Encapsulates agent-specific state that can vary between agents in a multi-agent system
 */
export class AgentContext {
  /**
   * Create an AgentContext from configuration with token accounting initialization
   */
  static fromConfig(
    agentConfig: t.AgentInputs,
    tokenCounter?: t.TokenCounter,
    indexTokenCountMap?: Record<string, number>,
    toolExecution?: t.ToolExecutionConfig
  ): AgentContext {
    const {
      agentId,
      endpoint,
      codeSessionKey,
      name,
      provider,
      clientOptions,
      langfuse,
      tools,
      toolMap,
      toolEnd,
      toolRegistry,
      toolDefinitions,
      instructions,
      additional_instructions,
      streamBuffer,
      maxContextTokens,
      reasoningKey,
      useLegacyContent,
      discoveredTools,
      summarizationEnabled,
      summarizeOnly,
      summarizationConfig,
      compactionSemanticIndex,
      initialSummary,
      contextPruningConfig,
      maxToolResultChars,
      toolSchemaTokens,
      subagentConfigs,
      maxSubagentDepth,
      graphTools,
    } = agentConfig;

    const agentContext = new AgentContext({
      agentId,
      endpoint,
      codeSessionKey,
      name: name ?? agentId,
      provider,
      clientOptions,
      langfuse,
      maxContextTokens,
      streamBuffer,
      tools,
      toolMap,
      toolRegistry,
      toolExecution,
      toolDefinitions,
      instructions,
      additionalInstructions: additional_instructions,
      reasoningKey,
      toolEnd,
      instructionTokens: 0,
      tokenCounter,
      useLegacyContent,
      discoveredTools,
      summarizationEnabled,
      summarizeOnly,
      summarizationConfig,
      compactionSemanticIndex,
      contextPruningConfig,
      maxToolResultChars,
    });

    agentContext._sourceInputs =
      compactionSemanticIndex == null
        ? agentConfig
        : {
          ...agentConfig,
          compactionSemanticIndex: agentContext.compactionSemanticIndex,
        };
    agentContext.subagentConfigs = subagentConfigs;
    agentContext.maxSubagentDepth = maxSubagentDepth;
    /**
     * Host-supplied direct tools (see `AgentInputs.graphTools`). Copied — never
     * aliased — because the SDK later pushes graph-managed tools (handoff /
     * subagent) into this same array and must not mutate the host's input.
     */
    if (graphTools && graphTools.length > 0) {
      agentContext.graphTools = [...graphTools];
    }

    if (initialSummary?.text != null && initialSummary.text !== '') {
      agentContext.setInitialSummary(
        initialSummary.text,
        initialSummary.tokenCount
      );
    }

    if (tokenCounter) {
      agentContext.initializeSystemRunnable();

      const tokenMap = indexTokenCountMap || {};
      agentContext.baseIndexTokenCountMap = { ...tokenMap };
      agentContext.indexTokenCountMap = tokenMap;

      if (toolSchemaTokens != null && toolSchemaTokens > 0) {
        /** Use pre-computed (cached) tool schema tokens — skip calculateInstructionTokens */
        agentContext.toolSchemaTokens = toolSchemaTokens;
        agentContext.tokenCalculationPromise = Promise.resolve();
        agentContext.updateTokenMapWithInstructions(tokenMap);
      } else {
        agentContext.tokenCalculationPromise = agentContext
          .calculateInstructionTokens(tokenCounter)
          .then(() => {
            agentContext.updateTokenMapWithInstructions(tokenMap);
          })
          .catch((err) => {
            console.error('Error calculating instruction tokens:', err);
          });
      }
    } else if (indexTokenCountMap) {
      agentContext.baseIndexTokenCountMap = { ...indexTokenCountMap };
      agentContext.indexTokenCountMap = indexTokenCountMap;
    }

    return agentContext;
  }

  /** Agent identifier */
  agentId: string;
  /** Logical endpoint selected by the host before provider resolution. */
  endpoint: string;
  /** Partition for this agent's transient code session and file refs. */
  codeSessionKey?: string;
  /** Human-readable name for this agent (used in handoff context). Falls back to agentId if not provided. */
  name?: string;
  /** Provider for this specific agent */
  provider: t.ProviderName;
  /** Client options for this agent */
  clientOptions?: t.ClientOptions;
  /** Per-agent Langfuse tracing configuration. */
  langfuse?: t.LangfuseConfig;
  /** Token count map indexed by message position */
  indexTokenCountMap: Record<string, number | undefined> = {};
  /** Canonical pre-run token map used to restore token accounting on reset */
  baseIndexTokenCountMap: Record<string, number> = {};
  /** Maximum context tokens for this agent */
  maxContextTokens?: number;
  /** Current usage metadata for this agent */
  currentUsage?: Partial<UsageMetadata>;
  /**
   * Usage from the most recent LLM call only (not accumulated).
   * Used for accurate provider calibration in pruning.
   */
  lastCallUsage?: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
    cacheRead?: number;
    cacheCreation?: number;
  };
  /**
   * Whether totalTokens data is fresh (set true when provider usage arrives,
   * false at the start of each turn before the LLM responds).
   * Prevents stale token data from driving pruning/trigger decisions.
   */
  totalTokensFresh: boolean = false;
  /** Context pruning configuration. */
  contextPruningConfig?: t.ContextPruningConfig;
  maxToolResultChars?: number;
  /** Prune messages function configured for this agent */
  pruneMessages?: ReturnType<typeof createPruneMessages>;
  /** Token counter function for this agent */
  tokenCounter?: t.TokenCounter;
  /** Exact stable-message counts reused by request-scoped context-pressure meters. */
  readonly contextPressureTokenCounts?: ExactTokenCountCache;
  /** Token count for the system message (instructions text). */
  systemMessageTokens: number = 0;
  /** Token count for instruction text emitted outside the system message. */
  dynamicInstructionTokens: number = 0;
  /** Token count for tool schemas only. */
  toolSchemaTokens: number = 0;
  /** Per-tool schema token counts (post-multiplier), keyed by tool name.
   *  `undefined` when not calculated (e.g. cached aggregate schema tokens). */
  toolTokenCounts?: Record<string, number>;
  /** Names of counted tools that are deferred (`defer_loading`) and discovered. */
  deferredToolNames: string[] = [];
  /** Running calibration ratio from the pruner — persisted across runs via contextMeta. */
  calibrationRatio: number = 1;
  /** Latched context-fading tier from the pruner — persisted across runs via contextMeta. */
  fadingTier?: t.FadingTier;
  /** Provider-observed instruction overhead from the pruner's best-variance turn. */
  resolvedInstructionOverhead?: number;
  private _pendingOriginalToolContent?: Map<number, string>;
  private pendingOriginalToolContentChars = 0;
  /** Provider-bound projection for this Run; graph messages remain canonical. */
  private providerProjectedMessages?: BaseMessage[];
  /** Canonical object identities corresponding to the projected slots. */
  private providerProjectionSources?: BaseMessage[];
  /** Canonical messages by ID, used to distinguish identity replay from replacement. */
  private providerProjectionSourcesById?: Map<string, BaseMessage>;
  /** Conservative counts retained by canonical identity across a rebuild. */
  private providerProjectionRecountFloors?: WeakMap<BaseMessage, number>;
  /** Recount after discarding a projection whose token map contains capped sizes. */
  private providerProjectionRequiresRecount = false;
  /** Pre-masking tool content keyed by message index, consumed by the summarize node. */
  get pendingOriginalToolContent(): Map<number, string> | undefined {
    return this._pendingOriginalToolContent;
  }
  set pendingOriginalToolContent(value: Map<number, string> | undefined) {
    this._pendingOriginalToolContent = value;
    this.pendingOriginalToolContentChars = 0;
    if (value != null) {
      for (const content of value.values()) {
        this.pendingOriginalToolContentChars += content.length;
      }
      this.enforcePendingOriginalContentCap();
    }
  }

  /** Total instruction overhead: system message + tool schemas + pending summary. */
  get instructionTokens(): number {
    const summaryOverhead =
      this._summaryLocation === 'user_message' ? this.summaryTokenCount : 0;
    return (
      this.systemMessageTokens +
      this.dynamicInstructionTokens +
      this.toolSchemaTokens +
      summaryOverhead
    );
  }
  /** The amount of time that should pass before another consecutive API call */
  streamBuffer?: number;
  /** Last stream call timestamp for rate limiting */
  lastStreamCall?: number;
  /** Tools available to this agent */
  tools?: t.GraphTools;
  /** Graph-managed tools (e.g., handoff tools created by MultiAgentGraph) that bypass event-driven dispatch */
  graphTools?: t.GraphTools;
  /** Tool map for this agent */
  toolMap?: t.ToolMap;
  /**
   * Tool definitions registry (includes deferred and programmatic tool metadata).
   * Used for tool search and programmatic tool calling.
   */
  toolRegistry?: t.LCToolRegistry;
  /** Run-scoped backend used to identify auto-bound programmatic runners. */
  private toolExecution?: t.ToolExecutionConfig;
  /**
   * Serializable tool definitions for event-driven execution.
   * When provided, ToolNode operates in event-driven mode.
   */
  toolDefinitions?: t.LCTool[];
  /** Set of tool names discovered via tool search (to be loaded) */
  discoveredToolNames: Set<string> = new Set();
  /** Original AgentInputs used to create this context — used for self-spawn subagent resolution. */
  _sourceInputs?: t.AgentInputs;
  /** Subagent configurations for hierarchical delegation. */
  subagentConfigs?: t.SubagentConfigEntry[];
  /** Maximum subagent nesting depth. */
  maxSubagentDepth?: number;
  /** Instructions for this agent */
  instructions?: string;
  /** Additional instructions for this agent */
  additionalInstructions?: string;
  /** Reasoning key for this agent */
  reasoningKey: 'reasoning_content' | 'reasoning' = 'reasoning_content';
  /** Last token for reasoning detection */
  lastToken?: string;
  /** Token type switch state */
  tokenTypeSwitch?: 'reasoning' | 'content';
  /** Tracks how many reasoning→text transitions have occurred (ensures unique post-reasoning step keys) */
  reasoningTransitionCount = 0;
  /** Current token type being processed */
  currentTokenType: ContentTypes.TEXT | ContentTypes.THINK | 'think_and_text' =
    ContentTypes.TEXT;
  /** Whether tools should end the workflow */
  toolEnd: boolean = false;
  /** Cached system runnable (created lazily) */
  private cachedSystemRunnable?: Runnable<
    BaseMessage[],
    (BaseMessage | SystemMessage)[],
    RunnableConfig<Record<string, unknown>>
  >;
  /** Whether system runnable needs rebuild (set when discovered tools change) */
  private systemRunnableStale: boolean = true;
  /** Promise for token calculation initialization */
  tokenCalculationPromise?: Promise<void>;
  /** Format content blocks as strings (for legacy compatibility) */
  useLegacyContent: boolean = false;
  /** Enables graph-level summarization for this agent */
  summarizationEnabled?: boolean;
  /** Summarize-only run: request a summary on the first model step and end after it */
  summarizeOnly?: boolean;
  /** Whether the summarize-only run has already issued its one request */
  private _manualSummarizationRequested: boolean = false;
  /** Summarization runtime settings used by graph pruning hooks */
  summarizationConfig?: t.SummarizationConfig;
  /** Host-supplied advisory guidance consumed only when compaction runs. */
  compactionSemanticIndex?: t.CompactionSemanticIndex;
  /** Current summary text produced by the summarize node, integrated into system message */
  private summaryText?: string;
  /** Token count of the current summary (tracked for token accounting) */
  private summaryTokenCount: number = 0;
  /**
   * Where the summary should be injected:
   * - `'system_prompt'`: cross-run summary, included in the dynamic system tail
   * - `'user_message'`: mid-run compaction, injected as HumanMessage on clean slate
   * - `'none'`: no summary present
   */
  private _summaryLocation: 'system_prompt' | 'user_message' | 'none' = 'none';
  /** Whether a mid-run summary must appear before every retained message. */
  private summaryPrecedesMessages: boolean = false;
  /**
   * Durable summary that survives reset() calls. Set from initialSummary
   * during fromConfig() and updated by setSummary() so that the latest
   * summary (whether cross-run or intra-run) is always restored after
   * processStream's resetValues() cycle.
   */
  private _durableSummaryText?: string;
  private _durableSummaryTokenCount: number = 0;
  private durableSummaryPrecedesMessages: boolean = false;
  /** Number of summarization cycles that have occurred for this agent context */
  private _summaryVersion: number = 0;
  /** Whether this run compacted canonical history and reset its fading tier. */
  private _fadingTierReset: boolean = false;
  /**
   * Message count at the time summarization was last triggered.
   * Used to prevent re-summarizing the same unchanged message set.
   * Summarization is allowed to fire again only when new messages appear.
   */
  private _lastSummarizationMsgCount: number = 0;
  /**
   * Consecutive summarization attempts that produced no usable summary.
   * An empty or failed summary leaves the message set exactly as it was, so
   * the next prune cycle would ask again on identical state. Cleared by
   * `setSummary` and by `reset()`.
   */
  private _summarizationFailures: number = 0;
  /**
   * Forced compactions performed after a provider rejected a prompt as too
   * large. Bounds the recovery loop so a model that keeps refusing cannot
   * make the run compact indefinitely.
   */
  private _overflowRecoveryAttempts: number = 0;
  /**
   * Budget in force before the first overflow correction of the current run.
   * Recorded so `reset()` can undo the correction for the next run without
   * disturbing a `maxContextTokens` that no correction ever touched.
   */
  private _preOverflowMaxContextTokens?: number;
  /**
   * Prompt size, normalized into the local counter's uncalibrated units, at
   * the last overflow correction. Keeping both measurements in the same units
   * lets a later overflow prove whether compaction changed anything even when
   * the provider observation updated calibration between attempts.
   */
  private _lastOverflowPromptTokens?: number;
  /**
   * Handoff context when this agent receives control via handoff.
   * Contains source and parallel execution info for system message context.
   */
  handoffContext?: {
    /** Source agent that transferred control */
    sourceAgentName: string;
    /** Names of sibling agents executing in parallel (empty if sequential) */
    parallelSiblings: string[];
  };

  constructor({
    agentId,
    endpoint,
    codeSessionKey,
    name,
    provider,
    clientOptions,
    langfuse,
    maxContextTokens,
    streamBuffer,
    tokenCounter,
    tools,
    toolMap,
    toolRegistry,
    toolExecution,
    toolDefinitions,
    instructions,
    additionalInstructions,
    reasoningKey,
    toolEnd,
    instructionTokens,
    useLegacyContent,
    discoveredTools,
    summarizationEnabled,
    summarizeOnly,
    summarizationConfig,
    compactionSemanticIndex,
    contextPruningConfig,
    maxToolResultChars,
  }: {
    agentId: string;
    endpoint?: string;
    codeSessionKey?: string;
    name?: string;
    provider: t.ProviderName;
    clientOptions?: t.ClientOptions;
    langfuse?: t.LangfuseConfig;
    maxContextTokens?: number;
    streamBuffer?: number;
    tokenCounter?: t.TokenCounter;
    tools?: t.GraphTools;
    toolMap?: t.ToolMap;
    toolRegistry?: t.LCToolRegistry;
    toolExecution?: t.ToolExecutionConfig;
    toolDefinitions?: t.LCTool[];
    instructions?: string;
    additionalInstructions?: string;
    reasoningKey?: 'reasoning_content' | 'reasoning';
    toolEnd?: boolean;
    instructionTokens?: number;
    useLegacyContent?: boolean;
    discoveredTools?: string[];
    summarizationEnabled?: boolean;
    summarizeOnly?: boolean;
    summarizationConfig?: t.SummarizationConfig;
    compactionSemanticIndex?: t.CompactionSemanticIndex;
    contextPruningConfig?: t.ContextPruningConfig;
    maxToolResultChars?: number;
  }) {
    this.agentId = agentId;
    this.endpoint = endpoint ?? provider;
    this.codeSessionKey = codeSessionKey;
    this.name = name;
    this.provider = provider;
    this.clientOptions = clientOptions;
    this.langfuse = langfuse;
    this.maxContextTokens = maxContextTokens;
    this.streamBuffer = streamBuffer;
    this.tokenCounter = tokenCounter;
    this.contextPressureTokenCounts =
      tokenCounter != null && isTokenCounterCacheCompatible(tokenCounter)
        ? createExactTokenCountCache(tokenCounter)
        : undefined;
    this.tools = tools;
    this.toolMap = toolMap;
    this.toolRegistry = resolveLocalToolRegistry({
      toolRegistry,
      toolExecution,
    });
    this.toolExecution = toolExecution;
    this.toolDefinitions = toolDefinitions;
    this.instructions = instructions;
    this.additionalInstructions = additionalInstructions;
    if (reasoningKey) {
      this.reasoningKey = reasoningKey;
    }
    if (toolEnd !== undefined) {
      this.toolEnd = toolEnd;
    }
    if (instructionTokens !== undefined) {
      this.systemMessageTokens = instructionTokens;
    }

    this.useLegacyContent = useLegacyContent ?? false;
    this.summarizationEnabled = summarizationEnabled;
    this.summarizeOnly = summarizeOnly;
    this.summarizationConfig = summarizationConfig;
    if (compactionSemanticIndex != null) {
      this.compactionSemanticIndex = snapshotCompactionSemanticIndex(
        compactionSemanticIndex
      );
    }
    this.contextPruningConfig = contextPruningConfig;
    this.maxToolResultChars = maxToolResultChars;

    if (discoveredTools && discoveredTools.length > 0) {
      for (const toolName of discoveredTools) {
        this.discoveredToolNames.add(toolName);
      }
    }
  }

  /** Builds the caller boundary and schemas for programmatic-only tools. */
  private buildProgrammaticOnlyToolsInstructions(): string {
    const programmaticTools = this.getProgrammaticToolInstructionTargets();
    if (programmaticTools.length === 0) return '';
    const directProgrammaticTools = programmaticTools.filter(
      (tool) => tool.executesDirectly
    );
    const eventProgrammaticTools = programmaticTools.filter(
      (tool) => !tool.executesDirectly
    );
    const groups: Array<{
      tools: ProgrammaticToolInstructionTarget[];
      capabilities: CallerCapabilityProjection;
      label: string;
    }> = [];
    if (directProgrammaticTools.length > 0) {
      groups.push({
        tools: directProgrammaticTools,
        capabilities: this.getDirectProgrammaticCapabilityProjection(),
        label: 'Direct programmatic runners',
      });
    }
    if (eventProgrammaticTools.length > 0) {
      groups.push({
        tools: eventProgrammaticTools,
        capabilities: this.getCallerCapabilityProjection(),
        label: 'Event-dispatched programmatic runners',
      });
    }
    if (groups.length === 0) {
      return '';
    }
    const showGroupLabels = groups.length > 1;
    return (
      '\n\n## Programmatic Tool Calling' +
      groups
        .map(
          ({ tools, capabilities, label }) =>
            (showGroupLabels ? `\n\n### ${label}` : '') +
            this.buildProgrammaticToolGroupInstructions(tools, capabilities)
        )
        .join('')
    );
  }

  private buildProgrammaticToolGroupInstructions(
    programmaticTools: ProgrammaticToolInstructionTarget[],
    capabilities: CallerCapabilityProjection
  ): string {
    const programmaticOnlyTools = capabilities.codeExecutionOnlyTools;
    const programmaticToolNames = capabilities.codeExecutionTools.map(
      (toolDef) => toolDef.name
    );
    const directOnlyToolNames = capabilities.directOnlyTools
      .map((toolDef) => toolDef.name)
      .filter((name) => !isProgrammaticControlTool(name));

    const programmaticRunnerNames = programmaticTools
      .map((tool) => `\`${tool.name}\``)
      .join(' or ');
    const quotedProgrammaticNames =
      programmaticToolNames.length > 0
        ? programmaticToolNames.map((name) => `\`${name}\``).join(', ')
        : 'none';
    const directOnlyBoundary =
      directOnlyToolNames.length > 0
        ? `\nCall these tools directly; never list them in the \`tool_manifest\` or reference them inside ${programmaticRunnerNames}: ${directOnlyToolNames
          .map((name) => `\`${name}\``)
          .join(
            ', '
          )}. Every ${programmaticRunnerNames} call must include a \`tool_manifest\` containing the exact registered names used by its code; the manifest is validated before execution starts.`
        : '';
    const boundary =
      '\n\n' +
      `Only these tools may be invoked inside ${programmaticRunnerNames}: ${quotedProgrammaticNames}.` +
      directOnlyBoundary;

    if (programmaticOnlyTools.length === 0) {
      return boundary;
    }

    const toolDescriptions = programmaticOnlyTools
      .map((tool) => {
        let desc = `- **${tool.name}**`;
        if (tool.description != null && tool.description !== '') {
          desc += `: ${tool.description}`;
        }
        if (tool.parameters) {
          desc += `\n  Parameters: ${JSON.stringify(tool.parameters, null, 2).replace(/\n/g, '\n  ')}`;
        }
        return desc;
      })
      .join('\n\n');

    return (
      boundary +
      '\n\n### Programmatic-Only Tools\n\n' +
      `The following tools are available exclusively through ${programmaticRunnerNames}. ` +
      `You cannot call these tools directly; instead, ${programmaticTools
        .map(
          (tool) =>
            `use \`${tool.name}\` with ${tool.codeGuidance} that invokes them`
        )
        .join(', or ')}.\n\n` +
      toolDescriptions
    );
  }

  private getProgrammaticToolInstructionTargets(): ProgrammaticToolInstructionTarget[] {
    const targets: ProgrammaticToolInstructionTarget[] = [];
    if (
      this.hasBoundTool(Constants.BASH_PROGRAMMATIC_TOOL_CALLING) ||
      isProgrammaticRunnerAutoBound(
        Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
        this.toolExecution
      )
    ) {
      targets.push({
        name: Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
        codeGuidance: 'Bash code',
        executesDirectly: this.isProgrammaticRunnerDirectlyBound(
          Constants.BASH_PROGRAMMATIC_TOOL_CALLING
        ),
      });
    }

    if (
      this.hasBoundTool(Constants.PROGRAMMATIC_TOOL_CALLING) ||
      isProgrammaticRunnerAutoBound(
        Constants.PROGRAMMATIC_TOOL_CALLING,
        this.toolExecution
      )
    ) {
      const localDefault =
        this.toolExecution?.engine === 'local' ||
        this.toolExecution?.engine === 'cloudflare-sandbox';
      targets.push({
        name: Constants.PROGRAMMATIC_TOOL_CALLING,
        codeGuidance: localDefault
          ? 'Bash code by default, or set `lang: "py"` to use Python code'
          : 'Python code',
        executesDirectly: this.isProgrammaticRunnerDirectlyBound(
          Constants.PROGRAMMATIC_TOOL_CALLING
        ),
      });
    }

    return targets;
  }

  /** Whether ToolNode executes this runner in-process instead of via an event. */
  private isProgrammaticRunnerDirectlyBound(name: string): boolean {
    return (
      isProgrammaticRunnerResolvedDirectly(
        name,
        this.toolExecution,
        this.toolDefinitions?.some((toolDef) => toolDef.name === name) === true
      ) ||
      this.graphTools?.some((tool) => 'name' in tool && tool.name === name) ===
        true
    );
  }

  /** Mirrors ToolNode's executable implementation gate for direct runners. */
  private getDirectProgrammaticCapabilityProjection(): CallerCapabilityProjection {
    const implementationNames = new Set<string>();
    const isEventDriven = (this.toolDefinitions?.length ?? 0) > 0;
    const resolverInputNames = new Set<string>();
    if (isEventDriven) {
      for (const toolDef of this.toolDefinitions ?? []) {
        resolverInputNames.add(toolDef.name);
      }
    } else {
      for (const tool of (this.tools as t.GenericTool[] | undefined) ?? []) {
        if ('name' in tool && typeof tool.name === 'string') {
          implementationNames.add(tool.name);
          resolverInputNames.add(tool.name);
        }
      }
    }
    for (const tool of (this.graphTools as t.GenericTool[] | undefined) ?? []) {
      if ('name' in tool && typeof tool.name === 'string') {
        implementationNames.add(tool.name);
        resolverInputNames.add(tool.name);
      }
    }
    for (const name of resolveLocalImplementationNames(
      resolverInputNames,
      this.toolExecution
    )) {
      implementationNames.add(name);
    }
    const activeCapabilities = this.getCallerCapabilityProjection();
    const executableCapabilities = resolveCallerCapabilityProjection(
      this.toolRegistry?.values() ?? [],
      (toolDef) =>
        implementationNames.has(toolDef.name) &&
        isToolDefinitionActive(toolDef, this.discoveredToolNames)
    );
    return {
      directTools: activeCapabilities.directTools,
      directOnlyTools: activeCapabilities.directOnlyTools,
      codeExecutionTools: executableCapabilities.codeExecutionTools,
      codeExecutionOnlyTools: executableCapabilities.codeExecutionOnlyTools,
    };
  }

  private hasBoundTool(name: string): boolean {
    return (
      this.getToolsForBinding()?.some(
        (tool) => 'name' in tool && tool.name === name
      ) === true
    );
  }

  /**
   * Gets the system runnable, creating it lazily if needed.
   * Includes stable instructions, dynamic additional instructions, and
   * programmatic-only tools documentation.
   * Only rebuilds when marked stale (via markToolsAsDiscovered).
   */
  get systemRunnable():
    | Runnable<
        BaseMessage[],
        (BaseMessage | SystemMessage)[],
        RunnableConfig<Record<string, unknown>>
      >
    | undefined {
    if (!this.systemRunnableStale && this.cachedSystemRunnable !== undefined) {
      return this.cachedSystemRunnable;
    }

    this.cachedSystemRunnable = this.buildSystemRunnable({
      stableInstructions: this.buildStableInstructionsString(),
      dynamicInstructions: this.buildDynamicInstructionsString(),
    });
    this.systemRunnableStale = false;
    return this.cachedSystemRunnable;
  }

  /**
   * Explicitly initializes the system runnable.
   * Call this before async token calculation to ensure system message tokens are counted first.
   */
  initializeSystemRunnable(): void {
    if (this.systemRunnableStale || this.cachedSystemRunnable === undefined) {
      this.cachedSystemRunnable = this.buildSystemRunnable({
        stableInstructions: this.buildStableInstructionsString(),
        dynamicInstructions: this.buildDynamicInstructionsString(),
      });
      this.systemRunnableStale = false;
    }
  }

  /**
   * Builds the cacheable instructions string (without creating SystemMessage).
   * Includes agent identity preamble and handoff context when available.
   */
  private buildStableInstructionsString(): string {
    const parts: string[] = [];

    const identityPreamble = this.buildIdentityPreamble();
    if (identityPreamble) {
      parts.push(identityPreamble);
    }

    if (this.instructions != null && this.instructions !== '') {
      parts.push(this.instructions);
    }

    const programmaticToolsDoc = this.buildProgrammaticOnlyToolsInstructions();
    if (programmaticToolsDoc) {
      parts.push(programmaticToolsDoc);
    }

    return parts.join('\n\n');
  }

  /**
   * Builds the dynamic system-tail string (without creating SystemMessage).
   * Keep this out of prompt-cache-marked content so volatile context does not
   * invalidate the stable prefix.
   */
  private buildDynamicInstructionsString(): string {
    const parts: string[] = [];

    if (
      this.additionalInstructions != null &&
      this.additionalInstructions !== ''
    ) {
      parts.push(this.additionalInstructions);
    }

    // Cross-run summary: include in the system tail so the model has context
    // from the prior run without invalidating the cacheable prefix. Mid-run
    // summaries are injected as a HumanMessage on the post-compaction clean
    // slate instead (see buildSystemRunnable).
    if (
      this._summaryLocation === 'system_prompt' &&
      this.summaryText != null &&
      this.summaryText !== ''
    ) {
      parts.push('## Conversation Summary\n\n' + this.summaryText);
    }

    return parts.join('\n\n');
  }

  /**
   * Builds the agent identity preamble including handoff context if present.
   * This helps the agent understand its role in the multi-agent workflow.
   */
  private buildIdentityPreamble(): string {
    if (!this.handoffContext) return '';

    const displayName = this.name ?? this.agentId;
    const { sourceAgentName, parallelSiblings } = this.handoffContext;
    const isParallel = parallelSiblings.length > 0;

    const lines: string[] = [];
    lines.push('## Multi-Agent Workflow');
    lines.push(
      `You are "${displayName}", transferred from "${sourceAgentName}".`
    );

    if (isParallel) {
      lines.push(`Running in parallel with: ${parallelSiblings.join(', ')}.`);
    }

    lines.push(
      'Execute only tasks relevant to your role. Routing is already handled if requested, unless you can route further.'
    );

    return lines.join('\n');
  }

  /**
   * Build system runnable from pre-built instructions string.
   * Only called when content has actually changed.
   */
  private buildSystemRunnable({
    stableInstructions,
    dynamicInstructions,
  }: {
    stableInstructions: string;
    dynamicInstructions: string;
  }):
    | Runnable<
        BaseMessage[],
        (BaseMessage | SystemMessage)[],
        RunnableConfig<Record<string, unknown>>
      >
    | undefined {
    const hasMidRunSummary =
      this._summaryLocation === 'user_message' &&
      this.summaryText != null &&
      this.summaryText !== '';

    if (!stableInstructions && !dynamicInstructions && !hasMidRunSummary) {
      this.systemMessageTokens = 0;
      this.dynamicInstructionTokens = 0;
      return undefined;
    }

    const promptCacheProvider = this.getPromptCacheProvider();
    /**
     * GPT-5.6 explicit caching needs the same structural split as Anthropic —
     * a stable system message with the volatile tail moved behind it — but
     * none of the Anthropic marker stamping: its breakpoints are attached to
     * the serialized request later, in the OpenAI client. So it drives the
     * relocation flag while leaving `promptCacheProvider` undefined.
     */
    const openAIExplicitCache = this.usesOpenAIExplicitPromptCache();
    const splitsDynamicInstructions =
      promptCacheProvider != null || openAIExplicitCache;
    const shouldMoveDynamicInstructions =
      splitsDynamicInstructions &&
      stableInstructions !== '' &&
      dynamicInstructions !== '';
    const systemMessage = this.buildSystemMessage({
      stableInstructions,
      dynamicInstructions,
      promptCacheProvider,
      shouldMoveDynamicInstructions,
    });

    if (this.tokenCounter) {
      this.systemMessageTokens = systemMessage
        ? this.tokenCounter(systemMessage)
        : 0;
      this.dynamicInstructionTokens = shouldMoveDynamicInstructions
        ? this.tokenCounter(new HumanMessage(dynamicInstructions))
        : 0;
    }

    return RunnableLambda.from((messages: BaseMessage[]) => {
      const prefix: BaseMessage[] = systemMessage ? [systemMessage] : [];

      // Build the non-system portion (summary + conversation), then apply
      // cache markers separately so addCacheControl doesn't strip the
      // SystemMessage's own cache_control breakpoint set above.
      const hasSummaryBody =
        this._summaryLocation === 'user_message' &&
        this.summaryText != null &&
        this.summaryText !== '';

      const bodyWithSummary =
        hasSummaryBody && !splitsDynamicInstructions
          ? [this.buildSummaryHumanMessage(promptCacheProvider), ...messages]
          : messages;
      const dynamicTail = this.buildPromptCacheDynamicTail({
        dynamicInstructions,
        hasSummaryBody,
        splitsDynamicInstructions,
        shouldMoveDynamicInstructions,
      });
      let body = this.buildBodyWithPromptCacheDynamicTail(
        bodyWithSummary,
        dynamicTail,
        promptCacheProvider
      );

      if (
        promptCacheProvider != null &&
        dynamicTail.length === 0 &&
        body.length >= 2
      ) {
        body = addTailCacheControl(
          body,
          this.getPromptCacheTtl(promptCacheProvider)
        );
      }
      return [...prefix, ...body];
    }).withConfig({ runName: 'prompt' });
  }

  private buildSummaryHumanMessage(
    promptCacheProvider: PromptCacheProvider | undefined
  ): HumanMessage {
    const wrappedSummary = buildSummaryCarrierText(this.summaryText as string);

    if (promptCacheProvider !== Providers.ANTHROPIC) {
      return new HumanMessage(wrappedSummary);
    }

    return new HumanMessage({
      content: [
        {
          type: 'text',
          text: wrappedSummary,
          cache_control: buildAnthropicCacheControl(
            this.getPromptCacheTtl(Providers.ANTHROPIC)
          ),
        },
      ],
    });
  }

  private buildPromptCacheDynamicTail({
    dynamicInstructions,
    hasSummaryBody,
    splitsDynamicInstructions,
    shouldMoveDynamicInstructions,
  }: {
    dynamicInstructions: string;
    hasSummaryBody: boolean;
    splitsDynamicInstructions: boolean;
    shouldMoveDynamicInstructions: boolean;
  }): BaseMessage[] {
    if (!splitsDynamicInstructions) {
      return [];
    }

    const dynamicTail = shouldMoveDynamicInstructions
      ? [new HumanMessage(dynamicInstructions)]
      : [];

    if (!hasSummaryBody) {
      return dynamicTail;
    }

    return [...dynamicTail, this.buildSummaryHumanMessage(undefined)];
  }

  private buildBodyWithPromptCacheDynamicTail(
    messages: BaseMessage[],
    tail: BaseMessage[],
    promptCacheProvider: PromptCacheProvider | undefined
  ): BaseMessage[] {
    if (tail.length === 0) {
      return messages;
    }

    const tailIndex =
      this._summaryLocation === 'user_message' && this.summaryPrecedesMessages
        ? 0
        : this.getPromptCacheDynamicTailIndex(messages, promptCacheProvider);
    const stablePrefix = messages.slice(0, tailIndex);
    const trailingMessages = messages.slice(tailIndex);
    /**
     * Anthropic-format markers only. On the OpenAI explicit path the prefix is
     * left untouched: its breakpoints are attached to the serialized request,
     * and a `cache_control` block here would be sent to a provider that has no
     * such field.
     */
    const cacheablePrefix =
      promptCacheProvider == null
        ? stablePrefix
        : this.addStablePromptCacheMarkers(
          stablePrefix,
          this.getPromptCacheTtl(promptCacheProvider)
        );

    return [...cacheablePrefix, ...tail, ...trailingMessages];
  }

  private getPromptCacheDynamicTailIndex(
    messages: BaseMessage[],
    promptCacheProvider: PromptCacheProvider | undefined
  ): number {
    const lastIndex = messages.length - 1;

    if (lastIndex < 0) {
      return 0;
    }

    if (promptCacheProvider === Providers.OPENROUTER && messages.length === 1) {
      return messages.length;
    }

    for (let index = lastIndex; index >= 0; index--) {
      if (messages[index].getType() === 'human') {
        if (promptCacheProvider === Providers.OPENROUTER && index === 0) {
          return 1;
        }
        return index;
      }
    }

    return messages.length;
  }

  private addStablePromptCacheMarkers(
    messages: BaseMessage[],
    ttl?: PromptCacheTtl
  ): BaseMessage[] {
    if (messages.length <= 1) {
      return messages;
    }

    return [
      messages[0],
      ...addCacheControlToStablePrefixMessages(messages.slice(1), 2, ttl),
    ];
  }

  private getPromptCacheProvider(): PromptCacheProvider | undefined {
    if (this.provider === Providers.ANTHROPIC) {
      const anthropicOptions = this.clientOptions as
        | t.AnthropicClientOptions
        | undefined;
      return anthropicOptions?.promptCache === true
        ? Providers.ANTHROPIC
        : undefined;
    }

    if (this.provider === Providers.OPENROUTER) {
      const openRouterOptions = this.clientOptions as
        | t.ProviderOptionsMap[Providers.OPENROUTER]
        | undefined;
      return openRouterOptions?.promptCache === true
        ? Providers.OPENROUTER
        : undefined;
    }

    return undefined;
  }

  /**
   * GPT-5.6 explicit prompt caching, on first-party OpenAI and Azure OpenAI.
   *
   * Unlike the providers above this adds no marker to the message content
   * here: `prompt_cache_breakpoint` is attached to the serialized request in
   * the OpenAI client, which selects the last system/developer message. That
   * selection is only worth anything if the system message stops at the
   * stable instructions, so this exists to drive the same dynamic-tail
   * relocation — nothing else.
   */
  private usesOpenAIExplicitPromptCache(): boolean {
    if (
      this.provider !== Providers.OPENAI &&
      this.provider !== Providers.AZURE
    ) {
      return false;
    }
    const openAIOptions = this.clientOptions as
      | { promptCacheExplicit?: boolean }
      | undefined;
    return openAIOptions?.promptCacheExplicit === true;
  }

  private hasBedrockPromptCache(): boolean {
    if (this.provider !== Providers.BEDROCK) {
      return false;
    }
    const bedrockOptions = this.clientOptions as
      | t.BedrockAnthropicClientOptions
      | undefined;
    // Nova accepts system/message cachePoints (only the tool checkpoint is
    // Claude-only), so this is gated on promptCache alone.
    return bedrockOptions?.promptCache === true;
  }

  /**
   * Resolved TTL for the active prompt-cache provider (Anthropic or OpenRouter).
   * Both expose `promptCacheTtl` and use the Anthropic `cache_control` format, so
   * the configured value resolves the same way (default `'1h'` extended cache).
   */
  private getPromptCacheTtl(
    provider: PromptCacheProvider | undefined
  ): PromptCacheTtl | undefined {
    if (provider == null) {
      return undefined;
    }
    return resolvePromptCacheTtl(
      (this.clientOptions as { promptCacheTtl?: PromptCacheTtl } | undefined)
        ?.promptCacheTtl
    );
  }

  /**
   * Resolved TTL for Bedrock prompt-cache checkpoints (default `'1h'` on Claude).
   * Claude models downgrade an unsupported 1h to 5m server-side; non-Claude
   * models (Nova) reject the extended TTL, so they are clamped to 5m.
   */
  private getBedrockPromptCacheTtl(): PromptCacheTtl {
    const bedrockOptions = this.clientOptions as
      | t.BedrockAnthropicClientOptions
      | undefined;
    return resolveBedrockPromptCacheTtl(
      bedrockOptions?.promptCacheTtl,
      (bedrockOptions as { model?: string } | undefined)?.model
    );
  }

  private buildSystemMessage({
    stableInstructions,
    dynamicInstructions,
    promptCacheProvider,
    shouldMoveDynamicInstructions,
  }: {
    stableInstructions: string;
    dynamicInstructions: string;
    promptCacheProvider: PromptCacheProvider | undefined;
    shouldMoveDynamicInstructions: boolean;
  }): SystemMessage | undefined {
    if (!stableInstructions && !dynamicInstructions) {
      return undefined;
    }

    if (promptCacheProvider === Providers.ANTHROPIC) {
      const content: AgentSystemContentBlock[] = [];
      if (stableInstructions) {
        content.push({
          type: 'text',
          text: stableInstructions,
          cache_control: buildAnthropicCacheControl(
            this.getPromptCacheTtl(promptCacheProvider)
          ),
        });
      }
      if (dynamicInstructions && !shouldMoveDynamicInstructions) {
        content.push({ type: 'text', text: dynamicInstructions });
      }
      return new SystemMessage({ content } as BaseMessageFields);
    }

    if (promptCacheProvider === Providers.OPENROUTER && !stableInstructions) {
      return new SystemMessage(dynamicInstructions);
    }

    if (promptCacheProvider === Providers.OPENROUTER) {
      return new SystemMessage({
        content: [
          {
            type: 'text',
            text: stableInstructions,
            cache_control: buildAnthropicCacheControl(
              this.getPromptCacheTtl(promptCacheProvider)
            ),
          },
        ],
      } as BaseMessageFields);
    }

    if (this.hasBedrockPromptCache() && stableInstructions) {
      const content: AgentSystemContentBlock[] = [
        { type: 'text', text: stableInstructions },
        { cachePoint: buildBedrockCachePoint(this.getBedrockPromptCacheTtl()) },
      ];
      if (dynamicInstructions) {
        content.push({ type: 'text', text: dynamicInstructions });
      }
      return new SystemMessage({ content } as BaseMessageFields);
    }

    /**
     * A relocated tail must not also appear here. On the GPT-5.6 explicit path
     * this is what leaves the system message holding only the stable prefix,
     * which is where the client then places the breakpoint.
     */
    return new SystemMessage(
      [
        stableInstructions,
        shouldMoveDynamicInstructions ? '' : dynamicInstructions,
      ]
        .filter((part) => part !== '')
        .join('\n\n')
    );
  }

  /**
   * Reset context for a new run
   */
  reset(options?: { preserveOriginalToolContent?: boolean }): void {
    this._fadingTierReset = false;
    this.systemMessageTokens = 0;
    this.dynamicInstructionTokens = 0;
    this.toolSchemaTokens = 0;
    this.toolTokenCounts = undefined;
    this.deferredToolNames = [];
    this.cachedSystemRunnable = undefined;
    this.systemRunnableStale = true;
    this.lastToken = undefined;
    this.indexTokenCountMap = { ...this.baseIndexTokenCountMap };
    this.currentUsage = undefined;
    this.pruneMessages = undefined;
    this.clearProviderProjection(false);
    this.lastStreamCall = undefined;
    this.tokenTypeSwitch = undefined;
    this.reasoningTransitionCount = 0;
    this.currentTokenType = ContentTypes.TEXT;
    this.discoveredToolNames.clear();
    this.handoffContext = undefined;
    if (options?.preserveOriginalToolContent !== true) {
      this.pendingOriginalToolContent = undefined;
    }

    this.summaryText = this._durableSummaryText;
    this.summaryTokenCount = this._durableSummaryTokenCount;
    this.summaryPrecedesMessages = this.durableSummaryPrecedesMessages;
    this._lastSummarizationMsgCount = 0;
    this._summarizationFailures = 0;
    this._manualSummarizationRequested = false;
    this.lastCallUsage = undefined;
    this.totalTokensFresh = false;
    this.restoreContextBudgetAfterOverflow();

    if (this.tokenCounter) {
      this.initializeSystemRunnable();
      const baseTokenMap = { ...this.baseIndexTokenCountMap };
      this.indexTokenCountMap = baseTokenMap;
      this.tokenCalculationPromise = this.calculateInstructionTokens(
        this.tokenCounter
      )
        .then(() => {
          this.updateTokenMapWithInstructions(baseTokenMap);
        })
        .catch((err) => {
          console.error('Error calculating instruction tokens:', err);
        });
    } else {
      this.tokenCalculationPromise = undefined;
    }
  }

  /**
   * Update the token count map from a base map.
   *
   * Previously this inflated index 0 with instructionTokens to indirectly
   * reserve budget for the system prompt.  That approach was imprecise: with
   * large tool-schema overhead (e.g. 26 MCP tools ~5 000 tokens) the first
   * conversation message appeared enormous and was always pruned, while the
   * real available budget was never explicitly computed.
   *
   * Now instruction tokens are passed to getMessagesWithinTokenLimit via
   * the `getInstructionTokens` factory param so the pruner subtracts them
   * from the budget directly.  The token map contains only real per-message
   * token counts.
   */
  updateTokenMapWithInstructions(baseTokenMap: Record<string, number>): void {
    this.indexTokenCountMap = { ...baseTokenMap };
  }

  /** Event definitions with matching runtime caller/defer metadata applied. */
  getEffectiveToolDefinitions(): t.LCTool[] | undefined {
    if (!this.toolDefinitions) {
      return undefined;
    }
    return applyCallerCapabilityDefinitionOverrides(
      this.toolDefinitions,
      this.toolRegistry?.values()
    );
  }

  /** Active tool definitions for token accounting (excludes deferred-and-undiscovered entries). */
  private getActiveToolDefinitions(): t.LCTool[] {
    const effectiveToolDefinitions = this.getEffectiveToolDefinitions();
    if (!effectiveToolDefinitions) {
      return [];
    }
    /**
     * Mirror `getEventDrivenToolsForBinding`'s gate: a definition is only
     * bound to the model when its `allowed_callers` include `'direct'` and
     * (if deferred) it has been discovered. Filtering by `defer_loading`
     * alone left programmatic-only definitions counted in
     * `toolSchemaTokens` even though they were never bound.
     */
    return resolveCallerCapabilityProjection(
      effectiveToolDefinitions,
      (toolDef) => isToolDefinitionActive(toolDef, this.discoveredToolNames)
    ).directTools;
  }

  /**
   * Single source of truth for "which entries of `this.tools` should be
   * treated as actually bound". Callers:
   *   - `getToolsForBinding` (non-event-driven branch)
   *   - `getEventDrivenToolsForBinding` (appends instance tools alongside
   *     schema-only definitions)
   *   - `calculateInstructionTokens` (counts schema bytes for accounting)
   *
   * In event-driven mode (`toolDefinitions` present) instance tools are
   * appended unfiltered; outside event-driven mode they pass through
   * `filterToolsForBinding`. Centralizing the decision here prevents the
   * accounting/binding paths from drifting apart, which was the root
   * cause of the original miscount.
   */
  private getEffectiveInstanceTools(): t.GraphTools | undefined {
    if (!this.tools) {
      return undefined;
    }
    const isEventDriven = (this.toolDefinitions?.length ?? 0) > 0;
    if (isEventDriven || !this.toolRegistry) {
      return this.tools;
    }
    return this.filterToolsForBinding(this.tools);
  }

  /**
   * Calculate tool tokens and add to instruction tokens
   * Note: System message tokens are calculated during systemRunnable creation
   */
  async calculateInstructionTokens(
    tokenCounter: t.TokenCounter
  ): Promise<void> {
    let toolTokens = 0;
    const countedToolNames = new Set<string>();
    /** Prototype-free: external tool names like `toString` must not hit
     *  inherited properties during accumulation */
    const rawToolTokenCounts: Record<string, number> = Object.create(null);
    const deferredCountedNames = new Set<string>();

    /**
     * Iterate both `tools` (user-provided instance tools) and `graphTools`
     * (graph-managed tools like handoff + subagent). `graphTools` is often
     * populated after `fromConfig()` kicks off the initial calculation, so
     * callers that mutate `graphTools` must re-trigger this method to
     * refresh `toolSchemaTokens`.
     *
     * Use `getEffectiveInstanceTools()` so accounting reflects exactly the
     * subset that `getToolsForBinding` would emit — preventing the
     * worst-case-ceiling miscount that triggered spurious `empty_messages`
     * preflight rejections at low `maxContextTokens`. Deferred and
     * non-`'direct'` `toolDefinitions` are excluded by
     * `getActiveToolDefinitions()` below.
     */
    const instanceTools: t.GraphTools = [
      ...((this.getEffectiveInstanceTools() as t.GenericTool[] | undefined) ??
        []),
      ...((this.graphTools as t.GenericTool[] | undefined) ?? []),
    ];

    if (instanceTools.length > 0) {
      for (const tool of instanceTools) {
        const genericTool = tool as Record<string, unknown>;
        if (
          genericTool.schema != null &&
          typeof genericTool.schema === 'object'
        ) {
          const toolName = (genericTool.name as string | undefined) ?? '';
          const jsonSchema = toJsonSchema(
            genericTool.schema,
            toolName,
            (genericTool.description as string | undefined) ?? ''
          );
          const schemaTokens = tokenCounter(
            new SystemMessage(JSON.stringify(jsonSchema))
          );
          toolTokens += schemaTokens;
          if (toolName) {
            countedToolNames.add(toolName);
            rawToolTokenCounts[toolName] =
              (rawToolTokenCounts[toolName] ?? 0) + schemaTokens;
          }
        }
      }
    }

    for (const def of this.getActiveToolDefinitions()) {
      if (countedToolNames.has(def.name)) {
        continue;
      }
      const schema = {
        type: 'function',
        function: {
          name: def.name,
          description: def.description ?? '',
          parameters: def.parameters ?? {},
        },
      };
      const schemaTokens = tokenCounter(
        new SystemMessage(JSON.stringify(schema))
      );
      toolTokens += schemaTokens;
      countedToolNames.add(def.name);
      rawToolTokenCounts[def.name] =
        (rawToolTokenCounts[def.name] ?? 0) + schemaTokens;
      if (def.defer_loading === true) {
        deferredCountedNames.add(def.name);
      }
    }

    const isAnthropic =
      this.provider !== Providers.BEDROCK &&
      (this.provider === Providers.ANTHROPIC ||
        /anthropic|claude/i.test(
          String(
            (this.clientOptions as { model?: string } | undefined)?.model ?? ''
          )
        ));
    const toolTokenMultiplier = isAnthropic
      ? ANTHROPIC_TOOL_TOKEN_MULTIPLIER
      : DEFAULT_TOOL_TOKEN_MULTIPLIER;
    this.toolSchemaTokens = Math.ceil(toolTokens * toolTokenMultiplier);

    /** Largest-remainder apportionment keeps the per-tool counts summing
     *  exactly to the aggregate despite per-entry rounding */
    const toolTokenCounts = apportionTokenCounts(
      rawToolTokenCounts,
      toolTokenMultiplier,
      this.toolSchemaTokens
    );
    const deferredToolNames: string[] = [];
    for (const name of Object.keys(rawToolTokenCounts)) {
      if (
        deferredCountedNames.has(name) ||
        this.toolRegistry?.get(name)?.defer_loading === true
      ) {
        deferredToolNames.push(name);
      }
    }
    this.toolTokenCounts = toolTokenCounts;
    this.deferredToolNames = deferredToolNames;
  }

  /**
   * Gets the tool registry for deferred tools (for tool search).
   * @param onlyDeferred If true, only returns tools with defer_loading=true
   * @returns LCToolRegistry with tool definitions
   */
  getDeferredToolRegistry(onlyDeferred: boolean = true): t.LCToolRegistry {
    const registry: t.LCToolRegistry = new Map();

    if (!this.toolRegistry) {
      return registry;
    }

    for (const [name, toolDef] of this.toolRegistry) {
      if (!onlyDeferred || toolDef.defer_loading === true) {
        registry.set(name, toolDef);
      }
    }

    return registry;
  }

  /**
   * Sets the handoff context for this agent.
   * Call this when the agent receives control via handoff from another agent.
   * Marks system runnable as stale to include handoff context in system message.
   * @param sourceAgentName - Name of the agent that transferred control
   * @param parallelSiblings - Names of other agents executing in parallel with this one
   */
  setHandoffContext(sourceAgentName: string, parallelSiblings: string[]): void {
    this.handoffContext = { sourceAgentName, parallelSiblings };
    this.systemRunnableStale = true;
  }

  /**
   * Clears any handoff context.
   * Call this when resetting the agent or when handoff context is no longer relevant.
   */
  clearHandoffContext(): void {
    if (this.handoffContext) {
      this.handoffContext = undefined;
      this.systemRunnableStale = true;
    }
  }

  setSummary(
    text: string,
    tokenCount: number,
    options?: { precedesMessages?: boolean }
  ): void {
    this.summaryText = text;
    this.summaryTokenCount = tokenCount;
    this._summaryLocation = 'user_message';
    this.summaryPrecedesMessages = options?.precedesMessages === true;
    this._durableSummaryText = text;
    this._durableSummaryTokenCount = tokenCount;
    this.durableSummaryPrecedesMessages = this.summaryPrecedesMessages;
    this._summaryVersion += 1;
    this._fadingTierReset = true;
    this._summarizationFailures = 0;
    this.systemRunnableStale = true;
    this.pruneMessages = undefined;
    this.fadingTier = undefined;
    this.clearProviderProjection(false);
  }

  /** Sets a cross-run summary that is injected into the system prompt. */
  setInitialSummary(text: string, tokenCount: number): void {
    this.summaryText = text;
    this.summaryTokenCount = tokenCount;
    this._summaryLocation = 'system_prompt';
    this.summaryPrecedesMessages = false;
    this._durableSummaryText = text;
    this._durableSummaryTokenCount = tokenCount;
    this.durableSummaryPrecedesMessages = false;
    this._summaryVersion += 1;
    this.systemRunnableStale = true;
    this.pruneMessages = undefined;
    this.fadingTier = undefined;
    this.clearProviderProjection(false);
  }

  /**
   * Replaces the indexTokenCountMap with a fresh map keyed to the surviving
   * context messages after summarization.  Called by the summarize node after
   * it emits RemoveMessage operations that shift message indices.
   */
  rebuildTokenMapAfterSummarization(newTokenMap: Record<string, number>): void {
    this.indexTokenCountMap = newTokenMap;
    this.baseIndexTokenCountMap = { ...newTokenMap };
    this._lastSummarizationMsgCount = Object.keys(newTokenMap).length;
    this.currentUsage = undefined;
    this.lastCallUsage = undefined;
    this.totalTokensFresh = false;
  }

  hasSummary(): boolean {
    return this.summaryText != null && this.summaryText !== '';
  }

  /** True when a mid-run compaction summary is ready to be injected as a HumanMessage. */
  hasPendingCompactionSummary(): boolean {
    return this._summaryLocation === 'user_message' && this.hasSummary();
  }

  getSummaryText(): string | undefined {
    return this.summaryText;
  }

  get summaryVersion(): number {
    return this._summaryVersion;
  }

  get fadingTierReset(): boolean {
    return this._fadingTierReset;
  }

  /**
   * Returns true when the message count hasn't changed since the last
   * summarization — re-summarizing would produce an identical result.
   * Oversized individual messages are handled by fit-to-budget truncation
   * in the pruner, which keeps them in context without triggering overflow.
   */
  shouldSkipSummarization(currentMsgCount: number): boolean {
    return (
      this._lastSummarizationMsgCount > 0 &&
      currentMsgCount <= this._lastSummarizationMsgCount
    );
  }

  /**
   * Records the message count at which summarization was triggered,
   * so subsequent calls with the same count are suppressed.
   */
  markSummarizationTriggered(msgCount: number): void {
    this._lastSummarizationMsgCount = msgCount;
  }

  /**
   * Claims the single summarization request a summarize-only run makes.
   * True exactly once per run, on the first model step; the step that gets
   * `false` is the one after the summary, which reports usage and ends
   * without a model call. Always false when `summarizeOnly` is off.
   */
  claimManualSummarization(): boolean {
    if (this.summarizeOnly !== true || this._manualSummarizationRequested) {
      return false;
    }
    this._manualSummarizationRequested = true;
    return true;
  }

  /**
   * Records a summarization attempt that produced no usable summary — an
   * empty model response, or a provider failure the run declined to paper
   * over with a metadata stub. Cleared by the next successful summary.
   */
  recordSummarizationFailure(): void {
    this._summarizationFailures += 1;
  }

  get summarizationFailures(): number {
    return this._summarizationFailures;
  }

  /**
   * True once consecutive no-progress attempts reach {@link MAX_SUMMARIZATION_FAILURES}.
   * A summarizer that has returned nothing this many times in a row will keep
   * returning nothing: each empty result leaves the history unchanged, so the
   * next prune cycle re-triggers on the same state and the run burns its
   * recursion budget on empty summary steps. Summarization stays off for the
   * remainder of the run; `reset()` restores it for the next one.
   */
  get summarizationExhausted(): boolean {
    return this._summarizationFailures >= MAX_SUMMARIZATION_FAILURES;
  }

  get overflowRecoveryAttempts(): number {
    return this._overflowRecoveryAttempts;
  }

  shouldSummarizeOverflow(): boolean {
    return (
      this.summarizationEnabled === true &&
      (this.tokenCounter == null ||
        this.maxContextTokens == null ||
        this._overflowRecoveryAttempts > 0)
    );
  }

  /** Preserves the earliest full tool output recorded for each message index. */
  preserveOriginalToolContent(
    originalToolContent: Map<number, string> | undefined
  ): void {
    if (originalToolContent == null || originalToolContent.size === 0) {
      return;
    }
    if (this.pendingOriginalToolContent == null) {
      this.pendingOriginalToolContent = new Map();
    }
    for (const [index, content] of originalToolContent) {
      if (!this.pendingOriginalToolContent.has(index)) {
        this.pendingOriginalToolContent.set(index, content);
        this.pendingOriginalToolContentChars += content.length;
      }
    }
    this.enforcePendingOriginalContentCap();
  }

  private enforcePendingOriginalContentCap(): void {
    const pending = this._pendingOriginalToolContent;
    if (pending == null) {
      return;
    }
    while (
      this.pendingOriginalToolContentChars > ORIGINAL_CONTENT_MAX_CHARS &&
      pending.size > 0
    ) {
      const oldest = pending.keys().next();
      if (oldest.done === true) {
        break;
      }
      const removed = pending.get(oldest.value);
      if (removed != null) {
        this.pendingOriginalToolContentChars -= removed.length;
      }
      pending.delete(oldest.value);
    }
  }

  /**
   * Retargets the context budget after a provider rejected the prompt as too
   * large, and clears the memoized pruner so the next call is planned against
   * the corrected budget rather than the one that was evidently wrong.
   *
   * Also clears the "already summarized at this message count" guard: that
   * guard exists to stop redundant summarization of an unchanged history, but
   * here the history has not changed and compaction is exactly what is
   * needed.
   */
  applyContextBudgetCorrection(
    budgetTokens: number | undefined,
    promptTokens?: number
  ): void {
    if (this._overflowRecoveryAttempts === 0) {
      this._preOverflowMaxContextTokens = this.maxContextTokens;
    }
    if (budgetTokens != null) {
      this.maxContextTokens = budgetTokens;
    }
    this.pruneMessages = undefined;
    this.clearProviderProjection(true);
    this._lastSummarizationMsgCount = 0;
    this._lastOverflowPromptTokens =
      promptTokens != null
        ? this.normalizePromptTokens(promptTokens)
        : promptTokens;
    this._overflowRecoveryAttempts += 1;
  }

  /**
   * Returns the mutable provider projection for this Run while preserving the
   * graph-owned messages as the canonical history. Appends are synchronized
   * incrementally so the pruner's watermarks remain valid. Any rewritten or
   * compacted canonical prefix starts a fresh projection and pruner.
   */
  getProviderProjectedMessages(messages: BaseMessage[]): BaseMessage[] {
    const sources = this.providerProjectionSources;
    let projection = this.providerProjectedMessages;
    const priorLength = sources?.length ?? 0;
    const prefixChanged =
      sources != null &&
      (messages.length < sources.length ||
        (messages.length === priorLength
          ? sources.some((source, index) => messages[index] !== source)
          : priorLength > 0 &&
            messages[priorLength - 1] !== sources[priorLength - 1]));

    if (projection == null || sources == null || prefixChanged) {
      if (prefixChanged) {
        this.pruneMessages = undefined;
        /** The originals map is keyed by index; a rewritten prefix would let
         * the summarizer restore one tool's bytes onto another message. */
        this.pendingOriginalToolContent = undefined;
        this.prepareProviderProjectionRecount(sources);
      }
      projection = messages.map((message) =>
        Array.isArray(message.content)
          ? cloneMessage(message, [...message.content])
          : message
      );
      this.providerProjectedMessages = projection;
      this.providerProjectionSources = [...messages];
      this.providerProjectionSourcesById = new Map(
        messages.flatMap((message) =>
          message.id == null ? [] : [[message.id, message] as const]
        )
      );
      if (this.providerProjectionRequiresRecount && this.tokenCounter != null) {
        const recounted: Record<string, number> = {};
        for (let i = 0; i < messages.length; i++) {
          const localCount = this.tokenCounter(messages[i]);
          const conservativeFloor = this.providerProjectionRecountFloors?.get(
            messages[i]
          );
          recounted[i] =
            conservativeFloor == null
              ? localCount
              : Math.max(localCount, conservativeFloor);
        }
        this.indexTokenCountMap = recounted;
        this.providerProjectionRequiresRecount = false;
        this.providerProjectionRecountFloors = undefined;
      }
      return projection;
    }

    for (let i = sources.length; i < messages.length; i++) {
      const message = messages[i];
      sources.push(message);
      if (message.id != null) {
        this.providerProjectionSourcesById?.set(message.id, message);
      }
      projection.push(
        Array.isArray(message.content)
          ? cloneMessage(message, [...message.content])
          : message
      );
    }
    return projection;
  }

  /**
   * Invalidates the provider projection when a graph reducer update rewrites
   * canonical history. Reducer updates expose replacements and removals before
   * they are folded into a potentially longer array, avoiding a prefix scan on
   * the normal append-only path.
   */
  invalidateProviderProjectionForMessageUpdates(
    updates:
      | BaseMessage
      | BaseMessageLike
      | Array<BaseMessage | BaseMessageLike | null | undefined>
      | null
      | undefined
  ): void {
    if (this.providerProjectionSources == null) {
      return;
    }
    const updateList = (Array.isArray(updates) ? updates : [updates]) as Array<
      BaseMessageLike | null | undefined
    >;
    for (const rawUpdate of updateList) {
      if (rawUpdate == null) {
        continue;
      }
      const update = coerceMessageLikeToMessage(rawUpdate);
      const priorSource =
        update.id == null
          ? undefined
          : this.providerProjectionSourcesById?.get(update.id);
      if (
        update.getType() === 'remove' ||
        (priorSource != null && priorSource !== update)
      ) {
        this.pruneMessages = undefined;
        this.pendingOriginalToolContent = undefined;
        this.clearProviderProjection(true);
        return;
      }
    }
  }

  private clearProviderProjection(recountOnNextUse: boolean): void {
    if (recountOnNextUse) {
      this.prepareProviderProjectionRecount(this.providerProjectionSources);
    } else {
      this.providerProjectionRequiresRecount = false;
      this.providerProjectionRecountFloors = undefined;
    }
    this.providerProjectedMessages = undefined;
    this.providerProjectionSources = undefined;
    this.providerProjectionSourcesById = undefined;
  }

  private prepareProviderProjectionRecount(
    sources: BaseMessage[] | undefined
  ): void {
    this.providerProjectionRequiresRecount = true;
    if (sources == null) {
      return;
    }
    const floors = new WeakMap<BaseMessage, number>();
    for (let i = 0; i < sources.length; i++) {
      const count = this.indexTokenCountMap[i];
      if (count != null && Number.isFinite(count) && count >= 0) {
        floors.set(sources[i], count);
      }
    }
    this.providerProjectionRecountFloors = floors;
  }

  /** Applies token calibration only when the observation came from this provider. */
  applyObservedOverflowCalibration(
    provider: t.ProviderName | undefined,
    observedCalibrationRatio: number | undefined
  ): void {
    if (
      provider !== this.provider ||
      observedCalibrationRatio == null ||
      observedCalibrationRatio <= 0
    ) {
      return;
    }
    this.calibrationRatio = clampCalibrationRatio(observedCalibrationRatio);
  }

  /**
   * True when a previous correction failed to make the prompt any smaller —
   * the signature of a state nothing can compact further (an emptied message
   * list carrying its content in an injected summary, for example). Retrying
   * from there resends a byte-identical prompt, so the caller should stop.
   */
  overflowRecoveryStalled(currentPromptTokens?: number): boolean {
    const previous = this._lastOverflowPromptTokens;
    if (
      previous == null ||
      currentPromptTokens == null ||
      !Number.isFinite(currentPromptTokens)
    ) {
      return false;
    }
    const rawCurrent = this.normalizePromptTokens(currentPromptTokens);
    return rawCurrent >= previous;
  }

  private normalizePromptTokens(promptTokens: number): number {
    if (this.calibrationRatio <= 0) {
      return promptTokens;
    }
    const messageTokens = Math.max(0, promptTokens - this.instructionTokens);
    return this.instructionTokens + messageTokens / this.calibrationRatio;
  }

  /**
   * Undoes overflow corrections so a reused context starts the next run with
   * the budget it was configured with and a fresh recovery allowance.
   *
   * Without this, a single overflow would permanently shrink the budget for
   * every later turn, and two would exhaust the per-run allowance for the
   * lifetime of the context.
   */
  private restoreContextBudgetAfterOverflow(): void {
    if (this._overflowRecoveryAttempts === 0) {
      return;
    }
    this.maxContextTokens = this._preOverflowMaxContextTokens;
    this._preOverflowMaxContextTokens = undefined;
    this._lastOverflowPromptTokens = undefined;
    this._overflowRecoveryAttempts = 0;
  }

  clearSummary(): void {
    if (this.summaryText != null) {
      this.summaryText = undefined;
      this.summaryTokenCount = 0;
      this._durableSummaryText = undefined;
      this._durableSummaryTokenCount = 0;
      this.summaryPrecedesMessages = false;
      this.durableSummaryPrecedesMessages = false;
      this._summaryLocation = 'none';
      this.systemRunnableStale = true;
    }
  }

  /**
   * Returns a structured breakdown of how the context token budget is consumed.
   * Useful for diagnostics when context overflow or pruning issues occur.
   *
   * Note: `markToolsAsDiscovered` re-triggers `calculateInstructionTokens`,
   * so `toolSchemaTokens`/`toolTokenCounts` refresh before the next call.
   */
  getTokenBudgetBreakdown(messages?: BaseMessage[]): t.TokenBudgetBreakdown {
    const maxContextTokens = this.maxContextTokens ?? 0;
    /**
     * Derive `toolCount` from `getToolsForBinding()` so the diagnostic stays
     * aligned with what is actually bound to the model — and with what
     * `calculateInstructionTokens` counts into `toolSchemaTokens`. Using raw
     * `this.tools.length` would inflate the count whenever the registry
     * marks instance tools as deferred-undiscovered or non-`'direct'`,
     * producing the same misleading "N tools" diagnostic this fix is meant
     * to eliminate.
     */
    const toolCount = this.getToolsForBinding()?.length ?? 0;
    const messageCount = messages?.length ?? 0;

    let messageTokens = 0;
    if (messages != null) {
      for (let i = 0; i < messages.length; i++) {
        messageTokens +=
          (this.indexTokenCountMap[i] as number | undefined) ?? 0;
      }
    }

    /** Mirror the pruner's reserve math so availableForMessages agrees
     *  with the contextBudget computed during pruning */
    const reserveRatio =
      this.summarizationConfig?.reserveRatio ?? DEFAULT_RESERVE_RATIO;
    const reserveTokens =
      reserveRatio > 0 && reserveRatio < 1
        ? Math.round(maxContextTokens * reserveRatio)
        : 0;
    const availableForMessages = Math.max(
      0,
      maxContextTokens - reserveTokens - this.instructionTokens
    );

    return {
      maxContextTokens,
      instructionTokens: this.instructionTokens,
      systemMessageTokens: this.systemMessageTokens,
      dynamicInstructionTokens: this.dynamicInstructionTokens,
      toolSchemaTokens: this.toolSchemaTokens,
      summaryTokens: this.summaryTokenCount,
      toolCount,
      messageCount,
      messageTokens,
      availableForMessages,
      toolTokenCounts:
        this.toolTokenCounts != null ? { ...this.toolTokenCounts } : undefined,
      deferredToolNames:
        this.deferredToolNames.length > 0
          ? [...this.deferredToolNames]
          : undefined,
    };
  }

  /**
   * Returns a human-readable string of the token budget breakdown
   * for inclusion in error messages and diagnostics.
   */
  formatTokenBudgetBreakdown(messages?: BaseMessage[]): string {
    const b = this.getTokenBudgetBreakdown(messages);
    const lines = [
      'Token budget breakdown:',
      `  maxContextTokens:    ${b.maxContextTokens}`,
      `  instructionTokens:   ${b.instructionTokens} (system: ${b.systemMessageTokens}, dynamic: ${b.dynamicInstructionTokens}, tools: ${b.toolSchemaTokens} [${b.toolCount} tools])`,
      `  summaryTokens:       ${b.summaryTokens}`,
      `  messageTokens:       ${b.messageTokens} (${b.messageCount} messages)`,
      `  availableForMessages: ${b.availableForMessages}`,
    ];
    return lines.join('\n');
  }

  /**
   * Projects the context-usage snapshot for an arbitrary message set WITHOUT
   * invoking the model — the pre-send / page-load / window-switch counterpart to
   * the live `ON_CONTEXT_USAGE` snapshot. Runs the same pruner + budget math the
   * graph uses (`createPruneMessages` → `getTokenBudgetBreakdown` →
   * `syncBudgetDerivedFields`) so projected numbers match a real call. Returns
   * null when the context lacks the tokenizer or window needed to prune. Omits
   * the live post-format reconciliation (provider-specific, invoke-time) — a
   * small, acceptable delta for a pre-send estimate.
   *
   * Safe to call off the hot path: the supplied `messages` are never mutated
   * (each is passed as a clone — the pruner both replaces tool-result slots and
   * unshifts reasoning blocks into AI content arrays in place), and this
   * context's own state is untouched apart from refreshing stale instruction
   * counts (idempotent, exactly what a real call does). Token counts are
   * recounted for the supplied messages (the context's `indexTokenCountMap` is
   * keyed to the live run's branch and would missum an arbitrary branch) unless
   * the caller passes a map it guarantees matches. Calibration is NOT re-derived
   * from this context's live usage (a fresh pruner would compare the prior
   * call's provider input against the whole projected branch); the learned
   * `calibrationRatio` is applied as a static seed, and callers may override it
   * with a persisted ratio via `opts.calibrationRatio`.
   */
  projectContextUsage(
    messages: BaseMessage[],
    opts?: {
      runId?: string;
      agentId?: string;
      calibrationRatio?: number;
      indexTokenCountMap?: Record<string, number | undefined>;
    }
  ): t.ContextUsageEvent | null {
    const tokenCounter = this.tokenCounter;
    if (tokenCounter == null || this.maxContextTokens == null) {
      return null;
    }
    /** Refresh stale system overhead (handoff/summary changes) so instruction
     *  tokens match the prompt a real call would send. */
    this.initializeSystemRunnable();
    /** Clone array-content messages: the pruner unshifts reasoning blocks into
     *  AI content arrays in place, which would otherwise corrupt the caller's
     *  history. (Slot replacements land on the mapped array, not the caller's.) */
    const projected = messages.map((message) =>
      Array.isArray(message.content)
        ? cloneMessage(message, [...message.content])
        : message
    );
    let indexTokenCountMap = opts?.indexTokenCountMap;
    if (indexTokenCountMap == null) {
      indexTokenCountMap = {};
      for (let i = 0; i < messages.length; i++) {
        indexTokenCountMap[String(i)] = tokenCounter(messages[i]);
      }
    }
    const prune = createPruneMessages({
      startIndex: 0,
      provider: this.provider,
      tokenCounter,
      maxTokens: this.maxContextTokens,
      maxToolResultChars: this.maxToolResultChars,
      thinkingEnabled: isThinkingEnabled(this.provider, this.clientOptions),
      indexTokenCountMap,
      contextPruningConfig: this.contextPruningConfig,
      summarizationEnabled: this.summarizationEnabled,
      reserveRatio: this.summarizationConfig?.reserveRatio,
      calibrationRatio: opts?.calibrationRatio ?? this.calibrationRatio,
      fadingTier: this.fadingTier,
      getInstructionTokens: () => this.instructionTokens,
    });
    const {
      context,
      prePruneContextTokens,
      remainingContextTokens,
      contextBudget,
      effectiveInstructionTokens,
      calibrationRatio,
    } = prune({
      messages: projected,
      usageMetadata: undefined,
      lastCallUsage: undefined,
      totalTokensFresh: false,
    });
    const breakdown = this.getTokenBudgetBreakdown(messages);
    breakdown.messageCount = context.length;
    const usage: t.ContextUsageEvent = {
      runId: opts?.runId,
      agentId: opts?.agentId,
      breakdown,
      contextBudget,
      effectiveInstructionTokens,
      prePruneContextTokens,
      remainingContextTokens,
      calibrationRatio,
    };
    syncBudgetDerivedFields(
      usage,
      context,
      this.contextPressureTokenCounts?.count ?? tokenCounter,
      undefined,
      this.provider
    );
    return usage;
  }

  /**
   * Updates the last-call usage with data from the most recent LLM response.
   * Unlike `currentUsage` which accumulates, this captures only the single call.
   */
  updateLastCallUsage(usage: Partial<UsageMetadata>): void {
    const baseInputTokens = Number(usage.input_tokens) || 0;
    const cacheCreation =
      Number(usage.input_token_details?.cache_creation) || 0;
    const cacheRead = Number(usage.input_token_details?.cache_read) || 0;

    const outputTokens = Number(usage.output_tokens) || 0;
    const cacheSum = cacheCreation + cacheRead;
    const cacheIsAdditive = cacheSum > 0 && cacheSum > baseInputTokens;
    const totalInputTokens = cacheIsAdditive
      ? baseInputTokens + cacheSum
      : baseInputTokens;

    this.lastCallUsage = {
      inputTokens: totalInputTokens,
      outputTokens,
      totalTokens: totalInputTokens + outputTokens,
      cacheRead: cacheRead || undefined,
      cacheCreation: cacheCreation || undefined,
    };
    this.totalTokensFresh = true;
  }

  /** Marks token data as stale before a new LLM call. */
  markTokensStale(): void {
    this.totalTokensFresh = false;
  }

  /** Returns a snapshot of the deferred tools discovered in this context. */
  getDiscoveredTools(): string[] {
    return Array.from(this.discoveredToolNames);
  }

  /** Returns the live projection shared by prompt and event execution. */
  private getCallerCapabilityProjection(): CallerCapabilityProjection {
    return resolveCallerCapabilityProjection(
      mergeCallerCapabilityDefinitions(
        this.toolDefinitions,
        this.toolRegistry?.values()
      ),
      (toolDef) => isToolDefinitionActive(toolDef, this.discoveredToolNames)
    );
  }

  /** Returns the SDK-owned active caller projection for event-driven hosts. */
  getCallerCapabilityProjectionSnapshot(): t.CallerCapabilityProjectionSnapshot {
    return createCallerCapabilityProjectionSnapshot(
      this.getCallerCapabilityProjection()
    );
  }

  /**
   * Marks tools as discovered via tool search.
   * Discovered tools will be included in the next model binding.
   * Only marks system runnable stale if NEW tools were actually added.
   * @param toolNames - Array of discovered tool names
   * @returns true if any new tools were discovered
   */
  markToolsAsDiscovered(toolNames: string[]): boolean {
    let hasNewDiscoveries = false;
    for (const name of toolNames) {
      if (!this.discoveredToolNames.has(name)) {
        this.discoveredToolNames.add(name);
        hasNewDiscoveries = true;
      }
    }
    if (hasNewDiscoveries) {
      this.systemRunnableStale = true;
      /** Refresh schema token accounting so the next call's budget and
       *  per-tool breakdown include the newly discovered tools; awaited
       *  via tokenCalculationPromise before the next model call */
      if (this.tokenCounter) {
        this.tokenCalculationPromise = this.calculateInstructionTokens(
          this.tokenCounter
        );
      }
    }
    return hasNewDiscoveries;
  }

  /**
   * Gets tools that should be bound to the LLM.
   * In event-driven mode (toolDefinitions present, tools empty), creates schema-only tools.
   * Otherwise filters tool instances based on:
   * 1. Non-deferred tools with allowed_callers: ['direct']
   * 2. Discovered tools (from tool search)
   * @returns Array of tools to bind to model
   */
  getToolsForBinding(): t.GraphTools | undefined {
    if (this.toolDefinitions && this.toolDefinitions.length > 0) {
      return this.getEventDrivenToolsForBinding();
    }

    const filtered = this.getEffectiveInstanceTools();

    if (this.graphTools && this.graphTools.length > 0) {
      return [...(filtered ?? []), ...this.graphTools];
    }

    return filtered;
  }

  /** Creates schema-only tools from toolDefinitions for event-driven mode, merged with native tools */
  private getEventDrivenToolsForBinding(): t.GraphTools {
    if (!this.toolDefinitions) {
      return this.graphTools ?? [];
    }

    const schemaTools = createSchemaOnlyTools(
      this.getActiveToolDefinitions()
    ) as t.GraphTools;

    const allTools = [...schemaTools];

    if (this.graphTools && this.graphTools.length > 0) {
      allTools.push(...this.graphTools);
    }

    const instanceTools = this.getEffectiveInstanceTools();
    if (instanceTools && instanceTools.length > 0) {
      allTools.push(...instanceTools);
    }

    return allTools;
  }

  /** Filters tool instances for binding based on registry config */
  private filterToolsForBinding(tools: t.GraphTools): t.GraphTools {
    return tools.filter((tool) => {
      if (!('name' in tool)) {
        return true;
      }

      const toolDef = this.toolRegistry?.get(tool.name);
      if (!toolDef) {
        return true;
      }

      return (
        allowsToolCaller(toolDef, 'direct') &&
        isToolDefinitionActive(toolDef, this.discoveredToolNames)
      );
    });
  }
}
