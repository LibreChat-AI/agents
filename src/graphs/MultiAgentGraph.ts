import { tool } from '@langchain/core/tools';
import { PromptTemplate } from '@langchain/core/prompts';
import {
  AIMessage,
  ToolMessage,
  HumanMessage,
  getBufferString,
} from '@langchain/core/messages';
import {
  END,
  START,
  Command,
  StateGraph,
  Annotation,
  messagesStateReducer,
} from '@langchain/langgraph';
import type { BaseMessage, AIMessageChunk } from '@langchain/core/messages';
import type { LangGraphRunnableConfig } from '@langchain/langgraph';
import type { ToolRuntime } from '@langchain/core/tools';
import type { ProviderMessageProvenancePart } from '@/messages/provenance';
import type { GraphFactoryDependencies } from '@/graphs/graphFactory';
import type * as t from '@/types';
import {
  hasBijectiveProviderContentPartMapping,
  inspectProviderMessageProvenance,
  inspectProviderSourceMessageIds,
  setInvalidProviderMessageProvenance,
  setProviderMessageProvenance,
  stampSyntheticProviderMessage,
} from '@/messages/provenance';
import {
  calculateMaxToolResultChars,
  HARD_MAX_TOOL_RESULT_CHARS,
} from '@/utils/truncation';
import { withInstructionlessHandoffCue } from '@/messages/handoffCue';
import { serializeToolContentBounded } from '@/utils/toolContent';
import { Constants, MULTI_AGENT_GRAPH_RUN_NAME } from '@/common';
import { StandardGraph } from './Graph';

/** Pattern to extract instructions from transfer ToolMessage content */
const HANDOFF_INSTRUCTIONS_PATTERN = /(?:Instructions?|Context):\s*(.+)/is;
const HANDOFF_INSTRUCTIONS_KEY = 'handoff_instructions';

/**
 * Handoff and fan-in prompts that route work between agents. Built in-run and
 * never persisted as standalone payload entries, so they are marked synthetic:
 * `messagesStateReducer` would otherwise give them a plain UUID and downstream
 * consumers — compaction coverage anchors — could not tell them apart from a
 * message replayed out of the payload.
 */
function buildRoutingPrompt(content: string): HumanMessage {
  return stampSyntheticProviderMessage(
    new HumanMessage({
      content,
      additional_kwargs: { role: 'user', isMeta: true, source: 'routing' },
    })
  );
}

function getHandoffInstructions(
  input: Record<string, unknown>,
  promptKey: string,
  hasHandoffInput: boolean
): string | null {
  if (
    !hasHandoffInput ||
    !Object.prototype.hasOwnProperty.call(input, promptKey)
  ) {
    return null;
  }
  const value = input[promptKey];
  return typeof value === 'string' ? value : null;
}

function formatHandoffPromptLabel(promptKey: string): string {
  return promptKey.charAt(0).toUpperCase() + promptKey.slice(1);
}

function extractLegacyHandoffInstructions(
  content: string,
  promptLabels: Set<string> | undefined
): string | null {
  let markerIndex = -1;
  let markerLength = 0;
  for (const label of promptLabels ?? []) {
    const marker = `\n\n${label}:`;
    const index = content.indexOf(marker);
    if (index >= 0 && (markerIndex < 0 || index < markerIndex)) {
      markerIndex = index;
      markerLength = marker.length;
    }
  }
  if (markerIndex >= 0) {
    return content.slice(markerIndex + markerLength).trim();
  }
  return content.match(HANDOFF_INSTRUCTIONS_PATTERN)?.[1]?.trim() ?? null;
}

/** Whether a tool name marks a handoff transfer (static or conditional). */
function isTransferToolName(name: unknown): boolean {
  return (
    typeof name === 'string' &&
    (name.startsWith(Constants.LC_TRANSFER_TO_) ||
      name === 'conditional_transfer')
  );
}

function graphToolName(tool: unknown): string | undefined {
  if (tool == null || typeof tool !== 'object' || !('name' in tool)) {
    return undefined;
  }
  const { name } = tool;
  return typeof name === 'string' ? name : undefined;
}

/**
 * Drop transfer `tool_use` content blocks from an AI message's array content.
 * Companion to the reception's tool-call filtering: array-content providers
 * (Anthropic) serialize retained blocks verbatim, so a transfer block whose
 * call/result the reception stripped — or a parallel sibling's transfer block,
 * whose result never reaches this recipient's state — would replay as an
 * unmatched `tool_use`. Matched by the gathered ids AND by transfer name
 * (sibling blocks have no collectable id here). String content passes through.
 */
function filterTransferToolUseBlocks(
  content: AIMessage['content'],
  transferToolCallIds: ReadonlySet<string>
): {
  content: AIMessage['content'];
  retainedContentPartIndices?: readonly number[];
} {
  if (!Array.isArray(content)) {
    return { content };
  }
  const filteredContent: typeof content = [];
  const retainedContentPartIndices: number[] = [];
  for (let index = 0; index < content.length; index++) {
    const block = content[index];
    if (
      typeof block === 'object' &&
      (block as { type?: string } | null)?.type === 'tool_use'
    ) {
      const toolUse = block as { id?: string; name?: string };
      if (
        (toolUse.id != null && transferToolCallIds.has(toolUse.id)) ||
        isTransferToolName(toolUse.name)
      ) {
        continue;
      }
    }
    filteredContent.push(block);
    retainedContentPartIndices.push(index);
  }
  return { content: filteredContent, retainedContentPartIndices };
}

/** Rebuild a handoff AI message without losing authorship for retained bytes. */
function copyFilteredAIMessageProvenance(
  source: AIMessage | AIMessageChunk,
  target: AIMessage,
  retainedContentPartIndices?: readonly number[]
): AIMessage {
  const sourceMessageIdsState = inspectProviderSourceMessageIds(source);
  const provenanceState = inspectProviderMessageProvenance(source);
  if (
    provenanceState.status === 'invalid' ||
    sourceMessageIdsState.status === 'invalid'
  ) {
    setInvalidProviderMessageProvenance(target);
    return target;
  }
  const sourceMessageIds =
    sourceMessageIdsState.status === 'valid'
      ? sourceMessageIdsState.sourceMessageIds
      : [];
  if (provenanceState.status === 'absent') {
    if (sourceMessageIds.length > 0) {
      setProviderMessageProvenance(
        target,
        sourceMessageIds.map((sourceMessageId) => ({
          attribution: 'model',
          sourceMessageId,
        }))
      );
    }
    return target;
  }
  const provenance = provenanceState.provenance;
  const typedSourceMessageIds = new Set<string>();
  for (const part of provenance.parts) {
    if (part.sourceMessageId != null) {
      typedSourceMessageIds.add(part.sourceMessageId);
    }
  }
  const legacyParts: ProviderMessageProvenancePart[] = [];
  for (const sourceMessageId of sourceMessageIds) {
    if (!typedSourceMessageIds.has(sourceMessageId)) {
      legacyParts.push({ attribution: 'model', sourceMessageId });
    }
  }
  const setFilteredProvenance = (
    parts: readonly ProviderMessageProvenancePart[]
  ): void => {
    if (legacyParts.length === 0) {
      setProviderMessageProvenance(target, parts);
      return;
    }
    const combinedParts = parts.slice();
    for (const part of legacyParts) {
      combinedParts.push(part);
    }
    setProviderMessageProvenance(target, combinedParts);
  };
  if (
    retainedContentPartIndices == null ||
    !Array.isArray(source.content) ||
    retainedContentPartIndices.length === source.content.length
  ) {
    setFilteredProvenance(provenance.parts);
    return target;
  }

  if (
    !hasBijectiveProviderContentPartMapping(
      provenance.parts,
      source.content.length
    )
  ) {
    /** An older or aggregate envelope cannot be mapped to current blocks.
     * Preserve its conservative attribution instead of dropping tool lineage. */
    setFilteredProvenance(provenance.parts);
    return target;
  }

  const retainedParts: ProviderMessageProvenancePart[] = [];
  const retainedContentPartIndexSet = new Set(retainedContentPartIndices);
  for (const part of provenance.parts) {
    if (part.sourceContentPartIndices == null) {
      retainedParts.push(part);
      continue;
    }
    const retainedSourceContentPartIndices: number[] = [];
    for (const sourceContentPartIndex of part.sourceContentPartIndices) {
      if (retainedContentPartIndexSet.has(sourceContentPartIndex)) {
        retainedSourceContentPartIndices.push(sourceContentPartIndex);
      }
    }
    if (retainedSourceContentPartIndices.length > 0) {
      retainedParts.push({
        ...part,
        sourceContentPartIndices: retainedSourceContentPartIndices,
      });
    }
  }
  setFilteredProvenance(
    retainedParts.length > 0 ? retainedParts : [{ attribution: 'model' }]
  );
  return target;
}

function isValidHandoffGroupId(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0;
}

function getLastNewAiMessage(
  messages: BaseMessage[],
  previousMessages: BaseMessage[]
): BaseMessage | undefined {
  const previousMessageObjects = new Set(previousMessages);
  const previousMessageIds = new Set<string>();
  for (const message of previousMessages) {
    if (message.id != null) {
      previousMessageIds.add(message.id);
    }
  }
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index];
    if (
      message.getType() === 'ai' &&
      !previousMessageObjects.has(message) &&
      (message.id == null || !previousMessageIds.has(message.id))
    ) {
      return message;
    }
  }
  return undefined;
}

function withHandoffGroupMetadata(
  config: LangGraphRunnableConfig | undefined,
  groupId: number | undefined
): LangGraphRunnableConfig {
  return {
    ...config,
    metadata: {
      ...config?.metadata,
      [Constants.HANDOFF_GROUP_ID]: groupId ?? null,
    },
  };
}

function withActiveAgentMetadata(
  config: LangGraphRunnableConfig | undefined,
  agentId: string,
  agentName: string | undefined
): LangGraphRunnableConfig {
  return {
    ...config,
    metadata: {
      ...config?.metadata,
      activeAgentId: agentId,
      ...(agentName == null ? {} : { activeAgentName: agentName }),
    },
  };
}

/**
 * MultiAgentGraph extends StandardGraph to support dynamic multi-agent workflows
 * with handoffs, fan-in/fan-out, and other composable patterns.
 *
 * Key behavior:
 * - Agents with ONLY handoff edges: Can dynamically route to any handoff destination
 * - Agents with ONLY direct edges: Always follow their direct edges
 * - Agents with BOTH: Use Command for exclusive routing (handoff OR direct, not both)
 *   - If handoff occurs: Only the handoff destination executes
 *   - If no handoff: Direct edges execute (potentially in parallel)
 *
 * This enables the common pattern where an agent either delegates (handoff)
 * OR continues its workflow (direct edges), but not both simultaneously.
 */
export class MultiAgentGraph extends StandardGraph {
  override readonly runName = MULTI_AGENT_GRAPH_RUN_NAME;
  private edges: t.GraphEdge[];
  private startingNodes: Set<string> = new Set();
  private directEdges: t.GraphEdge[] = [];
  private handoffEdges: t.GraphEdge[] = [];
  private handoffSourceIds = new Set<string>();
  private readonly resultAgentId?: string;
  private readonly memberRecursionLimit?: number;
  private handoffPromptLabels: Map<string, Set<string>> = new Map();
  /**
   * Map of agentId to parallel group info.
   * Contains groupId (incrementing number reflecting execution order) for agents in parallel groups.
   * Sequential agents (not in any parallel group) have undefined entry.
   *
   * Example for: researcher -> [analyst1, analyst2, analyst3] -> summarizer
   * - researcher: undefined (sequential, order 0)
   * - analyst1, analyst2, analyst3: { groupId: 1 } (parallel group, order 1)
   * - summarizer: undefined (sequential, order 2)
   */
  private agentParallelGroups: Map<string, number> = new Map();

  constructor(
    input: t.MultiAgentGraphInput,
    dependencies?: GraphFactoryDependencies
  ) {
    super(input, dependencies);
    this.edges = input.edges;
    this.resultAgentId = input.resultAgentId;
    this.memberRecursionLimit = input.memberRecursionLimit;
    if (
      this.memberRecursionLimit != null &&
      (!Number.isSafeInteger(this.memberRecursionLimit) ||
        this.memberRecursionLimit <= 0)
    ) {
      throw new Error(
        'MultiAgentGraph: memberRecursionLimit must be a positive safe integer.'
      );
    }
    if (
      this.resultAgentId != null &&
      !this.agentContexts.has(this.resultAgentId)
    ) {
      throw new Error(
        `MultiAgentGraph: resultAgentId "${this.resultAgentId}" is not present in agents.`
      );
    }
    this.validateEdgeAgents();
    this.categorizeEdges();
    this.validateCommandRoutedDirectEdges();
    this.analyzeGraph();
    this.createHandoffTools();
  }

  /**
   * Fails fast when an edge references an agent that is not in
   * `agentContexts`. Without this check, the underlying LangGraph
   * `StateGraph.compile()` would throw the opaque
   * `Found edge ending at unknown node "<id>"` error after graph
   * construction — far from the true root cause.
   *
   * This catches the common misuse of passing `edges` into a multi-agent
   * config without also passing the corresponding sub-agent configs in
   * `agents` (e.g. a host that forgot to pre-load handoff targets).
   */
  private validateEdgeAgents(): void {
    const known = new Set(this.agentContexts.keys());
    const unknown = new Set<string>();
    for (const edge of this.edges) {
      const participants = [
        ...(Array.isArray(edge.from) ? edge.from : [edge.from]),
        ...(Array.isArray(edge.to) ? edge.to : [edge.to]),
      ];
      for (const id of participants) {
        if (typeof id === 'string' && !known.has(id)) {
          unknown.add(id);
        }
      }
    }
    if (unknown.size === 0) {
      return;
    }
    const missing = Array.from(unknown)
      .map((id) => `"${id}"`)
      .join(', ');
    throw new Error(
      `MultiAgentGraph: edges reference agent(s) not present in agents: [${missing}]. ` +
        'Ensure every agent referenced by an edge is also included in the `agents` array, ' +
        'or filter orphaned edges before constructing the graph.'
    );
  }

  /**
   * Categorize edges into handoff and direct types
   */
  private categorizeEdges(): void {
    for (const edge of this.edges) {
      const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
      const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];
      const isDefaultDirect =
        edge.edgeType == null &&
        edge.condition == null &&
        sources.length === 1 &&
        destinations.length > 1;
      if (edge.edgeType === 'direct' || isDefaultDirect) {
        this.directEdges.push(edge);
        continue;
      }
      this.handoffEdges.push(edge);
      for (const source of sources) {
        this.handoffSourceIds.add(source);
      }
    }
  }

  /** Static waiting/prompt edges cannot also be driven by Command routing. */
  private validateCommandRoutedDirectEdges(): void {
    const destinationGroups = new Map<
      string,
      { hasPrompt: boolean; commandSource?: string }
    >();
    for (const edge of this.directEdges) {
      const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
      const commandSource = sources.find((source) =>
        this.handoffSourceIds.has(source)
      );
      if (commandSource != null && sources.length > 1) {
        throw new Error(
          'MultiAgentGraph: grouped direct edge cannot include command-routed ' +
            `source "${commandSource}". Split handoff routing from all-of fan-in.`
        );
      }
      const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];
      const hasPrompt = edge.prompt != null && edge.prompt !== '';
      for (const destination of destinations) {
        const group = destinationGroups.get(destination) ?? {
          hasPrompt: false,
        };
        group.hasPrompt ||= hasPrompt;
        group.commandSource ??= commandSource;
        destinationGroups.set(destination, group);
      }
    }
    for (const { hasPrompt, commandSource } of destinationGroups.values()) {
      if (!hasPrompt || commandSource == null) {
        continue;
      }
      throw new Error(
        'MultiAgentGraph: prompted direct edge cannot include command-routed ' +
          `source "${commandSource}". Move the prompt into the routed node.`
      );
    }
  }

  /**
   * Analyze graph structure to determine starting nodes and connections
   */
  private analyzeGraph(): void {
    const hasIncomingEdge = new Set<string>();

    // Track all nodes that have incoming edges
    for (const edge of this.edges) {
      const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];
      destinations.forEach((dest) => hasIncomingEdge.add(dest));
    }

    // Starting nodes are those without incoming edges
    for (const agentId of this.agentContexts.keys()) {
      if (!hasIncomingEdge.has(agentId)) {
        this.startingNodes.add(agentId);
      }
    }

    // If no starting nodes found, use the first agent
    if (this.startingNodes.size === 0 && this.agentContexts.size > 0) {
      this.startingNodes.add(this.agentContexts.keys().next().value!);
    }

    // Determine if graph has parallel execution capability
    this.computeParallelCapability();
  }

  /**
   * Compute parallel groups by traversing the graph in execution order.
   * Assigns incrementing group IDs that reflect the sequential order of execution.
   *
   * For: researcher -> [analyst1, analyst2, analyst3] -> summarizer
   * - researcher: no group (first sequential node)
   * - analyst1, analyst2, analyst3: groupId 1 (first parallel group)
   * - summarizer: no group (next sequential node)
   *
   * This allows frontend to render in order:
   * Row 0: researcher
   * Row 1: [analyst1, analyst2, analyst3] (grouped)
   * Row 2: summarizer
   */
  private computeParallelCapability(): void {
    let groupCounter = 1; // Start at 1, 0 reserved for "no group"

    // Check 1: Multiple starting nodes means parallel from the start (group 1)
    if (this.startingNodes.size > 1) {
      for (const agentId of this.startingNodes) {
        this.agentParallelGroups.set(agentId, groupCounter);
      }
      groupCounter++;
    }

    // Check 2: Traverse direct edges in order to find fan-out patterns
    // Build a simple execution order by following edges from starting nodes
    const visited = new Set<string>();
    const queue: string[] = [...this.startingNodes];

    while (queue.length > 0) {
      const current = queue.shift()!;
      if (visited.has(current)) continue;
      visited.add(current);

      // Find direct edges from this node
      for (const edge of this.directEdges) {
        const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
        if (!sources.includes(current)) continue;

        const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];

        // Fan-out: multiple destinations = parallel group
        if (destinations.length > 1) {
          for (const dest of destinations) {
            // Only set if not already in a group (first group wins)
            if (!this.agentParallelGroups.has(dest)) {
              this.agentParallelGroups.set(dest, groupCounter);
            }
            if (!visited.has(dest)) {
              queue.push(dest);
            }
          }
          groupCounter++;
        } else {
          // Single destination - add to queue for traversal
          for (const dest of destinations) {
            if (!visited.has(dest)) {
              queue.push(dest);
            }
          }
        }
      }

      // Also follow handoff edges for traversal (but they don't create parallel groups)
      for (const edge of this.handoffEdges) {
        const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
        if (!sources.includes(current)) continue;

        const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];
        for (const dest of destinations) {
          if (!visited.has(dest)) {
            queue.push(dest);
          }
        }
      }
    }
  }

  /**
   * Get the parallel group ID for an agent, if any.
   * Returns undefined if the agent is not part of a parallel group.
   * Group IDs are incrementing numbers reflecting execution order.
   */
  getParallelGroupId(agentId: string): number | undefined {
    return this.agentParallelGroups.get(agentId);
  }

  /**
   * Override to indicate this is a multi-agent graph.
   * Enables agentId to be included in RunStep for frontend agent labeling.
   */
  protected override isMultiAgentGraph(): boolean {
    return true;
  }

  /**
   * Override base class method to provide parallel group IDs for run steps.
   */
  protected override getParallelGroupIdForAgent(
    agentId: string
  ): number | undefined {
    return this.agentParallelGroups.get(agentId);
  }

  /**
   * Create handoff tools for agents based on handoff edges only
   */
  private createHandoffTools(): void {
    // Group handoff edges by source agent(s)
    const handoffsByAgent = new Map<string, t.GraphEdge[]>();
    const tokenAccountingRefresh = new Set<string>();

    /** Transfer tool names are SDK-reserved. Remove externally supplied or
     * stale transfer tools before deriving the source agent's allowed set
     * from the graph edges below. */
    for (const agentContext of this.agentContexts.values()) {
      const originalTools = agentContext.graphTools;
      const retainedTools = agentContext.graphTools?.filter(
        (graphTool) => !isTransferToolName(graphToolName(graphTool))
      );
      if (retainedTools?.length !== originalTools?.length) {
        tokenAccountingRefresh.add(agentContext.agentId);
      }
      agentContext.graphTools =
        retainedTools != null && retainedTools.length > 0
          ? retainedTools
          : undefined;
    }

    // Only process handoff edges for tool creation
    for (const edge of this.handoffEdges) {
      if (typeof edge.prompt === 'string') {
        const label = formatHandoffPromptLabel(
          edge.promptKey ?? 'instructions'
        );
        const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];
        for (const destination of destinations) {
          const labels =
            this.handoffPromptLabels.get(destination) ?? new Set<string>();
          labels.add(label);
          this.handoffPromptLabels.set(destination, labels);
        }
      }
      const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
      sources.forEach((source) => {
        if (!handoffsByAgent.has(source)) {
          handoffsByAgent.set(source, []);
        }
        handoffsByAgent.get(source)!.push(edge);
      });
    }

    // Create handoff tools for each agent
    for (const [agentId, edges] of handoffsByAgent) {
      const agentContext = this.agentContexts.get(agentId);
      if (!agentContext) continue;

      // Create handoff tools for this agent's outgoing edges
      const handoffTools: t.GenericTool[] = [];
      const sourceAgentName = agentContext.name ?? agentId;
      for (const edge of edges) {
        const edgeTools = this.createHandoffToolsForEdge(
          edge,
          agentId,
          sourceAgentName
        );
        for (const edgeTool of edgeTools) {
          handoffTools.push(edgeTool);
        }
      }

      if (!agentContext.graphTools) {
        agentContext.graphTools = [];
      }
      for (const handoffTool of handoffTools) {
        agentContext.graphTools.push(handoffTool);
      }
      if (handoffTools.length > 0) {
        tokenAccountingRefresh.add(agentId);
      }
    }

    for (const agentId of tokenAccountingRefresh) {
      const agentContext = this.agentContexts.get(agentId);
      if (agentContext?.tokenCounter == null) {
        continue;
      }
      const { tokenCounter, baseIndexTokenCountMap } = agentContext;
      agentContext.tokenCalculationPromise = agentContext
        .calculateInstructionTokens(tokenCounter)
        .then(() => {
          agentContext.updateTokenMapWithInstructions(baseIndexTokenCountMap);
        })
        .catch((err) => {
          // eslint-disable-next-line no-console
          console.error(
            'Error recalculating instruction tokens after handoff tool updates:',
            err
          );
        });
    }
  }

  /**
   * Create handoff tools for an edge (handles multiple destinations)
   * @param edge - The graph edge defining the handoff
   * @param sourceAgentId - The ID of the agent that will perform the handoff
   * @param sourceAgentName - The human-readable name of the source agent
   */
  private createHandoffToolsForEdge(
    edge: t.GraphEdge,
    sourceAgentId: string,
    sourceAgentName: string
  ): t.GenericTool[] {
    const tools: t.GenericTool[] = [];
    const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];

    /** If there's a condition, create a single conditional handoff tool */
    if (edge.condition != null) {
      const toolName = 'conditional_transfer';
      const toolDescription =
        edge.description ?? 'Conditionally transfer control based on state';

      /** Check if we have a prompt for handoff input */
      const hasHandoffInput =
        edge.prompt != null && typeof edge.prompt === 'string';
      const handoffInputDescription = hasHandoffInput ? edge.prompt : undefined;
      const promptKey = edge.promptKey ?? 'instructions';

      tools.push(
        tool(
          async (rawInput, runtime: ToolRuntime) => {
            const input = rawInput as Record<string, unknown>;
            const state = runtime.state as t.BaseGraphState;
            const toolCallId = runtime.toolCall?.id ?? 'unknown';

            /** Evaluated condition */
            const result = edge.condition!(state);
            let destination: string;

            if (typeof result === 'boolean') {
              /** If true, use first destination; if false, don't transfer */
              if (!result) return null;
              destination = destinations[0];
            } else if (typeof result === 'string') {
              destination = result;
            } else {
              /** Array of destinations - for now, use the first */
              destination = Array.isArray(result) ? result[0] : destinations[0];
            }

            const handoffInstructions = getHandoffInstructions(
              input,
              promptKey,
              hasHandoffInput
            );
            let content = `Conditionally transferred to ${destination}`;
            if (handoffInstructions !== null) {
              content += `\n\n${formatHandoffPromptLabel(promptKey)}: ${handoffInstructions}`;
            }

            const toolMessage = new ToolMessage({
              content,
              name: toolName,
              tool_call_id: toolCallId,
              additional_kwargs: {
                /** Store destination for programmatic access in handoff detection */
                handoff_destination: destination,
                /** Store source agent name for receiving agent to know who handed off */
                handoff_source_name: sourceAgentName,
                ...(handoffInstructions !== null && {
                  [HANDOFF_INSTRUCTIONS_KEY]: handoffInstructions,
                }),
              },
            });

            return new Command({
              goto: destination,
              update: { messages: state.messages.concat(toolMessage) },
              graph: Command.PARENT,
            });
          },
          {
            name: toolName,
            schema: hasHandoffInput
              ? {
                type: 'object',
                properties: {
                  [promptKey]: {
                    type: 'string',
                    description: handoffInputDescription as string,
                  },
                },
                required: [],
              }
              : { type: 'object', properties: {}, required: [] },
            description: toolDescription,
          }
        )
      );
    } else {
      /** Create individual tools for each destination */
      for (const destination of destinations) {
        const toolName = `${Constants.LC_TRANSFER_TO_}${destination}`;
        const toolDescription =
          edge.description ?? `Transfer control to agent '${destination}'`;

        /** Check if we have a prompt for handoff input */
        const hasHandoffInput =
          edge.prompt != null && typeof edge.prompt === 'string';
        const handoffInputDescription = hasHandoffInput
          ? edge.prompt
          : undefined;
        const promptKey = edge.promptKey ?? 'instructions';

        tools.push(
          tool(
            async (rawInput, runtime: ToolRuntime) => {
              const input = rawInput as Record<string, unknown>;
              const toolCallId = runtime.toolCall?.id ?? 'unknown';

              const handoffInstructions = getHandoffInstructions(
                input,
                promptKey,
                hasHandoffInput
              );
              let content = `Successfully transferred to ${destination}`;
              if (handoffInstructions !== null) {
                content += `\n\n${formatHandoffPromptLabel(promptKey)}: ${handoffInstructions}`;
              }

              const toolMessage = new ToolMessage({
                content,
                name: toolName,
                tool_call_id: toolCallId,
                additional_kwargs: {
                  /** Store source agent name for receiving agent to know who handed off */
                  handoff_source_name: sourceAgentName,
                  ...(handoffInstructions !== null && {
                    [HANDOFF_INSTRUCTIONS_KEY]: handoffInstructions,
                  }),
                },
              });

              const state = runtime.state as t.BaseGraphState;

              /**
               * For parallel handoff support:
               * Build messages that include ONLY this tool call's context.
               * This prevents errors when LLM calls multiple transfers simultaneously -
               * each destination gets a valid AIMessage with matching tool_call and tool_result.
               *
               * Strategy:
               * 1. Find the AIMessage containing this tool call
               * 2. Create a filtered AIMessage with ONLY this tool_call
               * 3. Include all messages before the AIMessage plus the filtered pair
               */
              const messages = state.messages;
              let filteredMessages: BaseMessage[];
              let aiMessageIndex = -1;

              /** Find the AIMessage containing this tool call */
              for (let i = messages.length - 1; i >= 0; i--) {
                const msg = messages[i];
                if (msg.getType() === 'ai') {
                  const aiMsg = msg as AIMessage;
                  const hasThisCall = aiMsg.tool_calls?.some(
                    (tc) => tc.id === toolCallId
                  );
                  if (hasThisCall === true) {
                    aiMessageIndex = i;
                    break;
                  }
                }
              }

              if (aiMessageIndex >= 0) {
                const originalAiMsg = messages[aiMessageIndex] as AIMessage;
                const thisToolCall = originalAiMsg.tool_calls?.find(
                  (tc) => tc.id === toolCallId
                );

                if (
                  thisToolCall != null &&
                  (originalAiMsg.tool_calls?.length ?? 0) > 1
                ) {
                  /**
                   * Multiple tool calls - create filtered AIMessage with ONLY this call.
                   * This ensures valid message structure for parallel handoffs.
                   */
                  const filteredAiMsg = copyFilteredAIMessageProvenance(
                    originalAiMsg,
                    new AIMessage({
                      content: originalAiMsg.content,
                      tool_calls: [thisToolCall],
                      id: originalAiMsg.id,
                    })
                  );

                  filteredMessages = [
                    ...messages.slice(0, aiMessageIndex),
                    filteredAiMsg,
                    toolMessage,
                  ];
                } else {
                  /** Single tool call - use messages as-is */
                  filteredMessages = messages.concat(toolMessage);
                }
              } else {
                /** Fallback - append tool message */
                filteredMessages = messages.concat(toolMessage);
              }

              return new Command({
                goto: destination,
                update: { messages: filteredMessages },
                graph: Command.PARENT,
              });
            },
            {
              name: toolName,
              schema: hasHandoffInput
                ? {
                  type: 'object',
                  properties: {
                    [promptKey]: {
                      type: 'string',
                      description: handoffInputDescription as string,
                    },
                  },
                  required: [],
                }
                : { type: 'object', properties: {}, required: [] },
              description: toolDescription,
            }
          )
        );
      }
    }

    return tools;
  }

  /**
   * Create a complete agent subgraph (similar to createReactAgent)
   */
  private createAgentSubgraph(agentId: string): t.CompiledAgentWorfklow {
    /** This is essentially the same as `createAgentNode` from `StandardGraph` */
    return this.createAgentNode(agentId);
  }

  /**
   * Detects if the current agent is receiving a handoff and processes the messages accordingly.
   * Returns filtered messages with the transfer tool call/message removed, plus any instructions,
   * source agent, and parallel sibling information extracted from the transfer.
   *
   * Supports both single handoffs (last message is the transfer) and parallel handoffs
   * (multiple transfer ToolMessages, need to find the one targeting this agent).
   *
   * @param messages - Current state messages
   * @param agentId - The agent ID to check for handoff reception
   * @returns Object with filtered messages, extracted instructions, source agent, and parallel siblings
   */
  /**
   * Character cap for text injected into an agent's context as a routing or
   * handoff prompt, derived from that agent's remaining message budget. Fading
   * only shrinks tool content, so a prompt above this cap could overflow the
   * agent's window or force the routing instruction out of its context.
   */
  private async resolveMaxRoutingPromptChars(agentId: string): Promise<number> {
    const agentContext = this.agentContexts.get(agentId);
    if (agentContext?.tokenCalculationPromise != null) {
      await agentContext.tokenCalculationPromise;
    }
    const availableMessageTokens =
      agentContext?.maxContextTokens == null
        ? undefined
        : agentContext.getTokenBudgetBreakdown().availableForMessages;
    if (availableMessageTokens == null) {
      return HARD_MAX_TOOL_RESULT_CHARS;
    }
    return availableMessageTokens <= 0
      ? 0
      : calculateMaxToolResultChars(availableMessageTokens);
  }

  private processHandoffReception(
    messages: BaseMessage[],
    agentId: string
  ): {
    filteredMessages: BaseMessage[];
    instructions: string | null;
    sourceAgentName: string | null;
    parallelSiblings: string[];
    parallelGroupId?: number;
  } | null {
    if (messages.length === 0) return null;

    /**
     * A handoff is active only while resolving the most recent assistant
     * tool-call round. Older transfer results remain in conversation history,
     * but must not be reused when this agent is reached later through a direct
     * edge or cycle.
     */
    const activeTransferToolCallIds = new Set<string>();
    let activeToolMessageStartIndex = messages.length;
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg.getType() === 'tool') continue;

      if (msg.getType() === 'ai') {
        const aiMsg = msg as AIMessage | AIMessageChunk;
        for (const toolCall of aiMsg.tool_calls ?? []) {
          if (
            toolCall.id != null &&
            (toolCall.name.startsWith(Constants.LC_TRANSFER_TO_) ||
              toolCall.name === 'conditional_transfer')
          ) {
            activeTransferToolCallIds.add(toolCall.id);
          }
        }
        activeToolMessageStartIndex = i + 1;
      }
      break;
    }

    if (activeTransferToolCallIds.size === 0) return null;

    /**
     * Search for a transfer ToolMessage targeting this agent.
     * For parallel handoffs, multiple transfer messages may exist - find ours.
     * Search backwards from the end to find the most recent transfer to this agent.
     */
    let toolMessage: ToolMessage | null = null;
    let toolMessageIndex = -1;

    for (let i = messages.length - 1; i >= activeToolMessageStartIndex; i--) {
      const msg = messages[i];
      if (msg.getType() !== 'tool') continue;

      const candidateMsg = msg as ToolMessage;
      const toolName = candidateMsg.name;

      if (
        typeof toolName !== 'string' ||
        !activeTransferToolCallIds.has(candidateMsg.tool_call_id)
      ) {
        continue;
      }

      /** Check for standard transfer pattern */
      const isTransferMessage = toolName.startsWith(Constants.LC_TRANSFER_TO_);
      const isConditionalTransfer = toolName === 'conditional_transfer';

      if (!isTransferMessage && !isConditionalTransfer) continue;

      /** Extract destination from tool name or additional_kwargs */
      let destinationAgent: string | null = null;

      if (isTransferMessage) {
        destinationAgent = toolName.replace(Constants.LC_TRANSFER_TO_, '');
      } else if (isConditionalTransfer) {
        const handoffDest = candidateMsg.additional_kwargs.handoff_destination;
        destinationAgent = typeof handoffDest === 'string' ? handoffDest : null;
      }

      /** Check if this transfer targets our agent */
      if (destinationAgent === agentId) {
        toolMessage = candidateMsg;
        toolMessageIndex = i;
        break;
      }
    }

    /** No transfer targeting this agent found */
    if (toolMessage === null || toolMessageIndex < 0) return null;

    /** Extract instructions from the ToolMessage content */
    const contentStr =
      typeof toolMessage.content === 'string'
        ? toolMessage.content
        : serializeToolContentBounded(
          toolMessage.content,
          HARD_MAX_TOOL_RESULT_CHARS
        );

    const structuredInstructions =
      toolMessage.additional_kwargs[HANDOFF_INSTRUCTIONS_KEY];
    const instructions =
      typeof structuredInstructions === 'string'
        ? structuredInstructions.trim()
        : extractLegacyHandoffInstructions(
          contentStr,
          this.handoffPromptLabels.get(agentId)
        );

    /** Extract source agent name from additional_kwargs */
    const handoffSourceName = toolMessage.additional_kwargs.handoff_source_name;
    const sourceAgentName =
      typeof handoffSourceName === 'string' ? handoffSourceName : null;

    /** Extract parallel siblings (set by ToolNode for parallel handoffs) */
    const rawSiblings = toolMessage.additional_kwargs.handoff_parallel_siblings;
    const siblingIds: string[] = Array.isArray(rawSiblings)
      ? rawSiblings.filter((s): s is string => typeof s === 'string')
      : [];
    /** Convert IDs to display names */
    const parallelSiblings = siblingIds.map((id) => {
      const ctx = this.agentContexts.get(id);
      return ctx?.name ?? id;
    });
    const storedParallelGroupId =
      toolMessage.additional_kwargs[Constants.HANDOFF_GROUP_ID];
    const parallelGroupId = isValidHandoffGroupId(storedParallelGroupId)
      ? storedParallelGroupId
      : undefined;

    /** Get the tool_call_id to find and filter the AI message's tool call */
    const toolCallId = toolMessage.tool_call_id;

    /**
     * Collect all transfer tool_call_ids to filter out.
     * For parallel handoffs, we filter ALL transfer messages (not just ours)
     * to give the receiving agent a clean context without handoff noise.
     */
    const transferToolCallIds = new Set<string>([toolCallId]);
    for (const msg of messages) {
      if (msg.getType() === 'tool') {
        const tm = msg as ToolMessage;
        const tName = tm.name;
        if (
          typeof tName === 'string' &&
          (tName.startsWith(Constants.LC_TRANSFER_TO_) ||
            tName === 'conditional_transfer')
        ) {
          transferToolCallIds.add(tm.tool_call_id);
        }
      } else if (msg.getType() === 'ai') {
        const aiMsg = msg as AIMessage | AIMessageChunk;
        for (const toolCall of aiMsg.tool_calls ?? []) {
          if (
            toolCall.id != null &&
            (toolCall.name.startsWith(Constants.LC_TRANSFER_TO_) ||
              toolCall.name === 'conditional_transfer')
          ) {
            transferToolCallIds.add(toolCall.id);
          }
        }
      }
    }

    /** Filter out all transfer messages */
    const filteredMessages: BaseMessage[] = [];

    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      const msgType = msg.getType();

      /** Skip transfer ToolMessages */
      if (msgType === 'tool') {
        const tm = msg as ToolMessage;
        if (transferToolCallIds.has(tm.tool_call_id)) {
          continue;
        }
      }

      if (msgType === 'ai') {
        /** Check if this AI message contains any transfer tool calls */
        const aiMsg = msg as AIMessage | AIMessageChunk;
        const toolCalls = aiMsg.tool_calls;

        if (toolCalls && toolCalls.length > 0) {
          /** Filter out all transfer tool calls */
          const remainingToolCalls = toolCalls.filter(
            (tc) => tc.id == null || !transferToolCallIds.has(tc.id)
          );

          const hasTransferCalls = remainingToolCalls.length < toolCalls.length;

          if (hasTransferCalls) {
            if (
              remainingToolCalls.length > 0 ||
              (typeof aiMsg.content === 'string' && aiMsg.content.trim())
            ) {
              /**
               * Keep the message but without transfer tool calls — AND
               * without their `tool_use` content blocks. Array-content
               * providers (Anthropic) serialize the retained blocks
               * verbatim, so a transfer block whose call/result this
               * filter just stripped would reach the receiving agent as
               * an unmatched `tool_use` and the provider rejects the
               * request. Filtered by transfer NAME as well as the
               * gathered ids: a parallel sibling's transfer block has no
               * result in THIS recipient's state, so its id is never
               * collected, but its name still marks it.
               */
              const filteredContent = filterTransferToolUseBlocks(
                aiMsg.content,
                transferToolCallIds
              );
              const filteredAiMsg = copyFilteredAIMessageProvenance(
                aiMsg,
                new AIMessage({
                  content: filteredContent.content,
                  tool_calls: remainingToolCalls,
                  id: aiMsg.id,
                }),
                filteredContent.retainedContentPartIndices
              );
              filteredMessages.push(filteredAiMsg);
            }
            /** If no remaining content or tool calls, skip this message entirely */
            continue;
          }
        }
      }

      /** Keep all other messages */
      filteredMessages.push(msg);
    }

    return {
      filteredMessages,
      instructions,
      sourceAgentName,
      parallelSiblings,
      parallelGroupId,
    };
  }

  /**
   * Create the multi-agent workflow with dynamic handoffs
   */
  override createWorkflow(): t.CompiledMultiAgentWorkflow {
    this.hasCompiledCheckpointer = this.compileOptions?.checkpointer != null;
    const StateAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: (a, b) => {
          if (!this.messages.length) {
            this.startIndex = a.length + b.length;
          }
          const result = messagesStateReducer(a, b);
          this.messages = result;
          return result;
        },
        default: () => [],
      }),
      /** Channel for passing filtered messages to agents when excludeResults is true */
      agentMessages: Annotation<BaseMessage[]>({
        /** Replaces state entirely */
        reducer: (a, b) => b,
        default: () => [],
      }),
      subagentResult: Annotation<t.SubagentGraphResult | undefined>({
        reducer: (_current, update) => update,
        default: () => undefined,
      }),
      /** Surfaced from the summarizing agent's subgraph on a summarize-only run. */
      manualSummary: Annotation<string | undefined>({
        reducer: (_current, update) => update,
        default: () => undefined,
      }),
      runStepState: this.createRunStepStateAnnotation(),
    });

    const builder = new StateGraph(StateAnnotation);
    const addDirectEdge = (sources: string[], destination: string): void => {
      if (sources.length === 0) {
        return;
      }
      const source = sources.length === 1 ? sources[0] : sources;
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      /** @ts-ignore */
      builder.addEdge(source, destination);
    };

    /**
     * A summarize-only run compiles to the opted-in agent alone: no other
     * node, no handoff or direct edge, START → agent → END. Anything else
     * would run the summarizer twice (once from START, once through its
     * predecessor) or append routing prompts to the checkpoint it just
     * produced.
     */
    const summarizeOnlyAgentId = this.summarizeOnlyAgentId;
    const agentIds =
      summarizeOnlyAgentId != null
        ? [summarizeOnlyAgentId]
        : [...this.agentContexts.keys()];
    const handoffEdges = summarizeOnlyAgentId != null ? [] : this.handoffEdges;
    const directEdges = summarizeOnlyAgentId != null ? [] : this.directEdges;

    // Add all agents as complete subgraphs
    for (const agentId of agentIds) {
      // Get all possible destinations for this agent
      const handoffDestinations = new Set<string>();
      const directDestinations = new Set<string>();

      // Check handoff edges for destinations
      for (const edge of handoffEdges) {
        const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
        if (sources.includes(agentId) === true) {
          const dests = Array.isArray(edge.to) ? edge.to : [edge.to];
          dests.forEach((dest) => handoffDestinations.add(dest));
        }
      }

      // Check direct edges for destinations
      for (const edge of directEdges) {
        const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
        if (sources.includes(agentId) === true) {
          const dests = Array.isArray(edge.to) ? edge.to : [edge.to];
          dests.forEach((dest) => directDestinations.add(dest));
        }
      }

      /** Check if this agent has BOTH handoff and direct edges */
      const hasHandoffEdges = handoffDestinations.size > 0;
      const hasDirectEdges = directDestinations.size > 0;
      const needsCommandRouting = hasHandoffEdges && hasDirectEdges;

      /** Collect all possible destinations for this agent */
      const allDestinations = new Set([
        ...handoffDestinations,
        ...directDestinations,
      ]);
      if (handoffDestinations.size > 0 || directDestinations.size === 0) {
        allDestinations.add(END);
      }

      /** Agent subgraph (includes agent + tools) */
      const agentSubgraph = this.createAgentSubgraph(agentId);

      /** Wrapper function that handles agentMessages channel, handoff reception, and conditional routing */
      const agentWrapper = async (
        state: t.MultiAgentGraphState,
        config?: LangGraphRunnableConfig
      ): Promise<t.MultiAgentGraphState | Command> => {
        let result: t.MultiAgentGraphState;
        let inputMessages = state.messages;
        const agentContext = this.agentContexts.get(agentId);
        const recursionLimitedConfig =
          this.memberRecursionLimit == null
            ? config
            : { ...config, recursionLimit: this.memberRecursionLimit };
        /**
         * Check if this agent is receiving a handoff.
         * If so, filter out the transfer messages and inject instructions as preamble.
         * This prevents the receiving agent from seeing the transfer as "completed work"
         * and prematurely producing an end token.
         */
        const handoffContext = this.processHandoffReception(
          state.messages,
          agentId
        );

        const memberConfig = withInstructionlessHandoffCue(
          withActiveAgentMetadata(
            recursionLimitedConfig,
            agentId,
            agentContext?.name
          ),
          handoffContext != null &&
            (handoffContext.instructions == null ||
              handoffContext.instructions === '')
            ? handoffContext.filteredMessages.at(-1)
            : undefined
        );

        if (
          handoffContext?.sourceAgentName != null &&
          handoffContext.sourceAgentName !== ''
        ) {
          agentContext?.setHandoffContext(
            handoffContext.sourceAgentName,
            handoffContext.parallelSiblings
          );
        } else {
          agentContext?.clearHandoffContext();
        }

        if (handoffContext !== null) {
          const { filteredMessages, instructions, parallelGroupId } =
            handoffContext;

          /** Build messages for the receiving agent */
          let messagesForAgent = filteredMessages;

          /**
           * If there are instructions, inject them as a HumanMessage to
           * ground the receiving agent.
           *
           * When the last filtered message is a ToolMessage (e.g. from a
           * non-handoff tool the router called before handing off), a
           * synthetic AIMessage is inserted first to satisfy the
           * tool → assistant role ordering required by chat APIs.  Without
           * this bridge, appending a HumanMessage directly after a
           * ToolMessage causes "400 Unexpected role 'user' after role
           * 'tool'" errors (see issue #54).
           */
          const hasInstructions = instructions !== null && instructions !== '';
          if (hasInstructions) {
            /** Bounded like a direct-edge routing prompt: a HumanMessage is
             * not tool content, so fading could never shrink it later. */
            const boundedInstructions = serializeToolContentBounded(
              instructions,
              await this.resolveMaxRoutingPromptChars(agentId)
            );
            const lastMsg =
              filteredMessages.length > 0
                ? filteredMessages[filteredMessages.length - 1]
                : null;

            if (lastMsg != null && lastMsg.getType() === 'tool') {
              messagesForAgent = [
                ...filteredMessages,
                stampSyntheticProviderMessage(
                  new AIMessage(
                    `[Processed tool result and transferring to ${agentId}]`
                  )
                ),
                buildRoutingPrompt(boundedInstructions),
              ];
            } else {
              messagesForAgent = [
                ...filteredMessages,
                buildRoutingPrompt(boundedInstructions),
              ];
            }
          }

          /** Update token map if we have a token counter */
          if (agentContext?.tokenCounter && hasInstructions) {
            const freshTokenMap: Record<string, number> = {};
            for (
              let i = 0;
              i < Math.min(filteredMessages.length, this.startIndex);
              i++
            ) {
              const tokenCount = agentContext.indexTokenCountMap[i];
              if (tokenCount !== undefined) {
                freshTokenMap[i] = tokenCount;
              }
            }
            /** Add tokens for the bridge AIMessage + instructions HumanMessage */
            for (
              let i = filteredMessages.length;
              i < messagesForAgent.length;
              i++
            ) {
              freshTokenMap[i] = agentContext.tokenCounter(messagesForAgent[i]);
            }
            agentContext.updateTokenMapWithInstructions(freshTokenMap);
          }

          const transformedState: t.MultiAgentGraphState = {
            ...state,
            messages: messagesForAgent,
          };
          inputMessages = messagesForAgent;
          result = await agentSubgraph.invoke(
            transformedState,
            withHandoffGroupMetadata(memberConfig, parallelGroupId)
          );
          result = {
            ...result,
            agentMessages: [],
          };
        } else if (
          state.agentMessages != null &&
          state.agentMessages.length > 0
        ) {
          /**
           * When using agentMessages (excludeResults=true), we need to update
           * the token map to account for the new prompt message
           */
          if (agentContext && agentContext.tokenCounter) {
            /** The agentMessages contains:
             * 1. Filtered messages (0 to startIndex) - already have token counts
             * 2. New prompt message - needs token counting
             */
            const freshTokenMap: Record<string, number> = {};

            /** Copy existing token counts for filtered messages (0 to startIndex) */
            for (let i = 0; i < this.startIndex; i++) {
              const tokenCount = agentContext.indexTokenCountMap[i];
              if (tokenCount !== undefined) {
                freshTokenMap[i] = tokenCount;
              }
            }

            /** Calculate tokens only for the new prompt message (last message) */
            const promptMessageIndex = state.agentMessages.length - 1;
            if (promptMessageIndex >= this.startIndex) {
              const promptMessage = state.agentMessages[promptMessageIndex];
              freshTokenMap[promptMessageIndex] =
                agentContext.tokenCounter(promptMessage);
            }

            /** Update the agent's token map with instructions added */
            agentContext.updateTokenMapWithInstructions(freshTokenMap);
          }

          /** Temporary state with messages replaced by `agentMessages` */
          const transformedState: t.MultiAgentGraphState = {
            ...state,
            messages: state.agentMessages,
          };
          inputMessages = state.agentMessages;
          result = await agentSubgraph.invoke(transformedState, memberConfig);
          result = {
            ...result,
            /** Clear agentMessages for next agent */
            agentMessages: [],
          };
        } else {
          result = await agentSubgraph.invoke(state, memberConfig);
        }
        result = this.propagateManualCompaction(result);

        if (this.resultAgentId === agentId) {
          result = {
            ...result,
            subagentResult: {
              agentId,
              message: getLastNewAiMessage(result.messages, inputMessages),
            },
          };
        }

        /** If agent has both handoff and direct edges, use Command for exclusive routing */
        if (needsCommandRouting) {
          /** Check if a handoff occurred */
          const lastMessage = result.messages[
            result.messages.length - 1
          ] as BaseMessage | null;
          if (
            lastMessage != null &&
            lastMessage.getType() === 'tool' &&
            typeof lastMessage.name === 'string' &&
            lastMessage.name.startsWith(Constants.LC_TRANSFER_TO_)
          ) {
            /** Handoff occurred - extract destination and navigate there exclusively */
            const handoffDest = lastMessage.name.replace(
              Constants.LC_TRANSFER_TO_,
              ''
            );
            return new Command({
              update: result,
              goto: handoffDest,
            });
          } else {
            /** No handoff - proceed with direct edges */
            const directDests = Array.from(directDestinations);
            if (directDests.length === 1) {
              return new Command({
                update: result,
                goto: directDests[0],
              });
            } else if (directDests.length > 1) {
              /** Multiple direct destinations - they'll run in parallel */
              return new Command({
                update: result,
                goto: directDests,
              });
            }
          }
        }

        /** No special routing needed - return state normally */
        return result;
      };

      /** Wrapped agent as a node with its possible destinations */
      builder.addNode(agentId, agentWrapper, {
        ends: Array.from(allDestinations),
      });
    }

    const startingNodes =
      summarizeOnlyAgentId != null
        ? [summarizeOnlyAgentId]
        : this.startingNodes;
    for (const startNode of startingNodes) {
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      /** @ts-ignore */
      builder.addEdge(START, startNode);
    }

    /**
     * Add direct edges for automatic transitions
     * Group edges by destination to handle fan-in scenarios
     */
    const edgesByDestination = new Map<string, t.GraphEdge[]>();

    for (const edge of directEdges) {
      const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];
      for (const destination of destinations) {
        if (!edgesByDestination.has(destination)) {
          edgesByDestination.set(destination, []);
        }
        edgesByDestination.get(destination)!.push(edge);
      }
    }

    for (const [destination, edges] of edgesByDestination) {
      /** Checks if this is a fan-in scenario with prompt instructions */
      const edgesWithPrompt = edges.filter(
        (edge) => edge.prompt != null && edge.prompt !== ''
      );

      if (edgesWithPrompt.length > 0) {
        /**
         * Single wrapper node for destination (Fan-in with prompt)
         */
        const wrapperNodeId = `fan_in_${destination}_prompt`;
        /**
         * First edge's `prompt`
         * (they should all be the same for fan-in)
         */
        const prompt = edgesWithPrompt[0].prompt;
        /**
         * First edge's `excludeResults` flag
         * (they should all be the same for fan-in)
         */
        const excludeResults = edgesWithPrompt[0].excludeResults;

        builder.addNode(wrapperNodeId, async (state: t.BaseGraphState) => {
          let promptText: string | undefined;
          let effectiveExcludeResults = excludeResults;
          const getMaxRoutingPromptChars = (): Promise<number> =>
            this.resolveMaxRoutingPromptChars(destination);

          if (typeof prompt === 'function') {
            const resolvedPrompt = await prompt(
              state.messages,
              this.startIndex
            );
            promptText =
              resolvedPrompt == null
                ? undefined
                : serializeToolContentBounded(
                  resolvedPrompt,
                  await getMaxRoutingPromptChars()
                );
          } else if (prompt != null) {
            if (prompt.includes('{results}')) {
              const resultsMessages = state.messages.slice(this.startIndex);
              const maxRoutingPromptChars = await getMaxRoutingPromptChars();
              const resultsString = serializeToolContentBounded(
                getBufferString(resultsMessages),
                maxRoutingPromptChars
              );
              const promptTemplate = PromptTemplate.fromTemplate(prompt);
              const result = await promptTemplate.invoke({
                results: resultsString,
              });
              promptText = serializeToolContentBounded(
                result.value,
                maxRoutingPromptChars
              );
              effectiveExcludeResults =
                excludeResults !== false && promptText !== '';
            } else {
              promptText = prompt;
            }
          }

          if (promptText != null && promptText !== '') {
            if (
              effectiveExcludeResults == null ||
              effectiveExcludeResults === false
            ) {
              return {
                messages: [buildRoutingPrompt(promptText)],
              };
            }

            /** When `excludeResults` is true, use agentMessages channel
             * to pass filtered messages + prompt to the destination agent
             */
            const filteredMessages = state.messages.slice(0, this.startIndex);
            const promptMessage = buildRoutingPrompt(promptText);
            return {
              messages: [promptMessage],
              agentMessages: messagesStateReducer(filteredMessages, [
                promptMessage,
              ]),
            };
          }

          /** No prompt needed, return empty update */
          return {};
        });

        /** Add edges from all sources to the wrapper, then wrapper to destination */
        for (const edge of edges) {
          const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
          addDirectEdge(sources, wrapperNodeId);
        }

        /** Single edge from wrapper to destination */
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        /** @ts-ignore */
        builder.addEdge(wrapperNodeId, destination);
      } else {
        /** No prompt instructions, add direct edges (skip if source uses Command routing) */
        for (const edge of edges) {
          const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
          const staticSources = sources.filter(
            (source) =>
              !this.handoffEdges.some((handoffEdge) => {
                const handoffSources = Array.isArray(handoffEdge.from)
                  ? handoffEdge.from
                  : [handoffEdge.from];
                return handoffSources.includes(source);
              })
          );
          addDirectEdge(staticSources, destination);
        }
      }
    }

    return builder.compile({
      ...this.compileOptions,
      name: MULTI_AGENT_GRAPH_RUN_NAME,
    } as unknown as never);
  }
}
