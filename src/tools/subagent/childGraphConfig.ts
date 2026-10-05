import type {
  AgentInputs,
  ExecutableSubagentConfig,
  ExecutableSubagentConfigEntry,
  GraphEdge,
  GraphSubagentConfig,
  ResolvedSingleAgentSubagentConfig,
  ResolvedSubagentConfig,
  ResolvedSubagentConfigEntry,
  SubagentConfig,
  SubagentConfigEntry,
} from '@/types';
import type { AgentContext } from '@/agents/AgentContext';
import {
  MAX_GRAPH_SUBAGENT_MEMBERS,
  MAX_GRAPH_SUBAGENT_TURNS,
} from './runtimeLimits';
import { getSubagentHostArgSpecs } from './hostArgs';

const RESERVED_GRAPH_MEMBER_IDS = new Set([
  '__start__',
  '__end__',
  'messages',
  'agentMessages',
  'subagentResult',
]);

type ChildGraphPlanBase = {
  subjectAgentId: string;
  checkpointNodeId: string;
  agents: AgentInputs[];
  memberInputs: ReadonlyMap<string, AgentInputs>;
};

export type ChildGraphPlan =
  | (ChildGraphPlanBase & {
      kind: 'agent';
    })
  | (ChildGraphPlanBase & {
      kind: 'graph';
      edges: GraphEdge[];
      entryAgentId: string;
      resultAgentId: string;
    });

type CreateChildGraphPlanParams = {
  config: ResolvedSubagentConfigEntry;
  executionSuffix: string;
  parentAgentId?: string;
  parentMaxDepth: number;
  keepToolDefinitions: boolean;
};

type UnvalidatedGraphSubagentConfig = Omit<
  GraphSubagentConfig,
  'allowNested' | 'edges'
> & { allowNested?: boolean; edges: GraphEdge[] };

type GraphSubagentConfigCandidate = {
  kind?: unknown;
  type?: unknown;
  name?: unknown;
  description?: unknown;
  maxTurns?: unknown;
  allowNested?: unknown;
  agents?: unknown;
  edges?: unknown;
  entryAgentId?: unknown;
  resultAgentId?: unknown;
  configId?: unknown;
  agentInputs?: unknown;
  resolveAgentInputs?: unknown;
  hostArgs?: unknown;
  self?: unknown;
};

function isObject(value: unknown): value is object {
  return value != null && typeof value === 'object';
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim() !== '';
}

export function isGraphSubagentConfig(
  config: SubagentConfigEntry
): config is GraphSubagentConfig {
  return (config as { kind?: unknown }).kind === 'graph';
}

function assertGraphSubagentConfigShape(
  input: unknown
): asserts input is UnvalidatedGraphSubagentConfig {
  if (!isObject(input)) {
    throw new Error('Graph subagent config must be an object.');
  }
  const candidate = input as GraphSubagentConfigCandidate;
  const configType = isNonEmptyString(candidate.type)
    ? candidate.type
    : '<unknown>';
  if (candidate.kind !== 'graph') {
    throw new Error(`Graph subagent "${configType}" must set kind to "graph".`);
  }
  for (const [field, value] of [
    ['type', candidate.type],
    ['name', candidate.name],
    ['description', candidate.description],
    ['entryAgentId', candidate.entryAgentId],
    ['resultAgentId', candidate.resultAgentId],
  ] as const) {
    if (!isNonEmptyString(value)) {
      throw new Error(
        `Graph subagent "${configType}" must define a non-empty ${field}.`
      );
    }
  }
  if (!Array.isArray(candidate.agents)) {
    throw new Error(`Graph subagent "${configType}" agents must be an array.`);
  }
  const agents: unknown[] = candidate.agents;
  for (const agent of agents) {
    if (!isObject(agent)) {
      throw new Error(
        `Graph subagent "${configType}" agents must contain objects.`
      );
    }
  }
  if (!Array.isArray(candidate.edges)) {
    throw new Error(`Graph subagent "${configType}" edges must be an array.`);
  }
  const edges: unknown[] = candidate.edges;
  for (const edge of edges) {
    if (!isObject(edge)) {
      throw new Error(
        `Graph subagent "${configType}" edges must contain objects.`
      );
    }
  }
  if (
    candidate.allowNested != null &&
    typeof candidate.allowNested !== 'boolean'
  ) {
    throw new Error(
      `Graph subagent "${configType}" allowNested must be a boolean.`
    );
  }
  if (
    candidate.maxTurns != null &&
    (typeof candidate.maxTurns !== 'number' ||
      !Number.isSafeInteger(candidate.maxTurns) ||
      candidate.maxTurns <= 0)
  ) {
    throw new Error(
      `Graph subagent "${configType}" maxTurns must be a positive safe integer.`
    );
  }
  if (
    candidate.agentInputs !== undefined ||
    candidate.resolveAgentInputs !== undefined ||
    candidate.configId !== undefined ||
    candidate.hostArgs !== undefined ||
    candidate.self !== undefined
  ) {
    throw new Error(
      `Graph subagent "${configType}" cannot define agentInputs or self, or lazy fields configId/resolveAgentInputs/hostArgs.`
    );
  }
}

function getEndpoints(endpoint: string | string[]): string[] {
  return Array.isArray(endpoint) ? endpoint : [endpoint];
}

function assertNonEmptyUniqueEndpoints(
  endpoints: string[],
  label: string,
  edgeIndex: number
): void {
  if (endpoints.length === 0) {
    throw new Error(
      `Graph subagent edge ${edgeIndex} must include at least one ${label} agent.`
    );
  }
  const seen = new Set<string>();
  for (const endpoint of endpoints) {
    if (typeof endpoint !== 'string' || endpoint.trim() === '') {
      throw new Error(
        `Graph subagent edge ${edgeIndex} contains a blank ${label} agent ID.`
      );
    }
    if (seen.has(endpoint)) {
      throw new Error(
        `Graph subagent edge ${edgeIndex} repeats ${label} agent "${endpoint}".`
      );
    }
    seen.add(endpoint);
  }
}

function visitReachable(
  start: string,
  adjacency: ReadonlyMap<string, ReadonlySet<string>>
): Set<string> {
  const visited = new Set<string>();
  const pending = [start];
  while (pending.length > 0) {
    const current = pending.pop();
    if (current == null || visited.has(current)) {
      continue;
    }
    visited.add(current);
    for (const next of adjacency.get(current) ?? []) {
      if (!visited.has(next)) {
        pending.push(next);
      }
    }
  }
  return visited;
}

export function validateGraphSubagentConfig(config: unknown): void {
  assertGraphSubagentConfigShape(config);
  if (config.agents.length === 0) {
    throw new Error(`Graph subagent "${config.type}" must include an agent.`);
  }
  if (config.agents.length > MAX_GRAPH_SUBAGENT_MEMBERS) {
    throw new Error(
      `Graph subagent "${config.type}" cannot exceed ${MAX_GRAPH_SUBAGENT_MEMBERS} agents.`
    );
  }
  if (config.maxTurns != null && config.maxTurns > MAX_GRAPH_SUBAGENT_TURNS) {
    throw new Error(
      `Graph subagent "${config.type}" maxTurns exceeds the safe graph recursion budget.`
    );
  }
  if (config.allowNested === true) {
    throw new Error(
      `Graph subagent "${config.type}" cannot enable allowNested because graph-member nesting and resume semantics are outside this MVP.`
    );
  }

  const agentIds = new Set<string>();
  for (const agent of config.agents) {
    if (typeof agent.agentId !== 'string' || agent.agentId.trim() === '') {
      throw new Error(
        `Graph subagent "${config.type}" contains a blank agent ID.`
      );
    }
    if (agentIds.has(agent.agentId)) {
      throw new Error(
        `Graph subagent "${config.type}" contains duplicate agent ID "${agent.agentId}".`
      );
    }
    if (
      RESERVED_GRAPH_MEMBER_IDS.has(agent.agentId) ||
      agent.agentId.includes('|') ||
      agent.agentId.includes(':')
    ) {
      throw new Error(
        `Graph subagent "${config.type}" agent ID "${agent.agentId}" is reserved by the graph runtime.`
      );
    }
    agentIds.add(agent.agentId);
  }

  if (!agentIds.has(config.entryAgentId)) {
    throw new Error(
      `Graph subagent "${config.type}" entryAgentId "${config.entryAgentId}" is not a configured agent.`
    );
  }
  if (!agentIds.has(config.resultAgentId)) {
    throw new Error(
      `Graph subagent "${config.type}" resultAgentId "${config.resultAgentId}" is not a configured agent.`
    );
  }

  const adjacency = new Map<string, Set<string>>();
  const reverseAdjacency = new Map<string, Set<string>>();
  const incomingEdgeSources = new Map<string, string[][]>();
  const promptedEdges: Array<{
    edgeIndex: number;
    destinations: string[];
  }> = [];
  const directedEdges = new Set<string>();
  for (const agentId of agentIds) {
    adjacency.set(agentId, new Set());
    reverseAdjacency.set(agentId, new Set());
  }

  for (let edgeIndex = 0; edgeIndex < config.edges.length; edgeIndex++) {
    const edge = config.edges[edgeIndex];
    if (edge.edgeType !== 'direct') {
      throw new Error(
        `Graph subagent "${config.type}" edge ${edgeIndex} must set edgeType to "direct".`
      );
    }
    if (edge.condition != null) {
      throw new Error(
        `Graph subagent "${config.type}" edge ${edgeIndex} cannot define a condition.`
      );
    }
    if (edge.promptKey != null) {
      throw new Error(
        `Graph subagent "${config.type}" edge ${edgeIndex} cannot define promptKey.`
      );
    }
    const runtimePrompt = (edge as { prompt?: unknown }).prompt;
    if (
      runtimePrompt != null &&
      typeof runtimePrompt !== 'string' &&
      typeof runtimePrompt !== 'function'
    ) {
      throw new Error(
        `Graph subagent "${config.type}" edge ${edgeIndex} prompt must be a string or function.`
      );
    }
    const runtimeExcludeResults = (edge as { excludeResults?: unknown })
      .excludeResults;
    if (
      runtimeExcludeResults != null &&
      typeof runtimeExcludeResults !== 'boolean'
    ) {
      throw new Error(
        `Graph subagent "${config.type}" edge ${edgeIndex} excludeResults must be a boolean.`
      );
    }

    const sources = getEndpoints(edge.from);
    const destinations = getEndpoints(edge.to);
    assertNonEmptyUniqueEndpoints(sources, 'source', edgeIndex);
    assertNonEmptyUniqueEndpoints(destinations, 'destination', edgeIndex);
    if (runtimePrompt != null && runtimePrompt !== '') {
      promptedEdges.push({ edgeIndex, destinations });
    } else if (runtimeExcludeResults != null) {
      throw new Error(
        `Graph subagent "${config.type}" edge ${edgeIndex} cannot define excludeResults without a prompt.`
      );
    }

    for (const endpoint of [...sources, ...destinations]) {
      if (!agentIds.has(endpoint)) {
        throw new Error(
          `Graph subagent "${config.type}" edge ${edgeIndex} references unknown agent "${endpoint}".`
        );
      }
    }

    for (const destination of destinations) {
      const incomingGroups = incomingEdgeSources.get(destination) ?? [];
      incomingGroups.push(sources);
      incomingEdgeSources.set(destination, incomingGroups);
      if (
        edge.prompt != null &&
        edge.prompt !== '' &&
        agentIds.has(`fan_in_${destination}_prompt`)
      ) {
        throw new Error(
          `Graph subagent "${config.type}" agent ID "fan_in_${destination}_prompt" conflicts with a generated prompt node.`
        );
      }
      for (const source of sources) {
        if (source === destination) {
          throw new Error(
            `Graph subagent "${config.type}" cannot contain self-edge "${source}".`
          );
        }
        const edgeKey = `${source}\u0000${destination}`;
        if (directedEdges.has(edgeKey)) {
          throw new Error(
            `Graph subagent "${config.type}" repeats edge "${source}" -> "${destination}".`
          );
        }
        directedEdges.add(edgeKey);
        adjacency.get(source)!.add(destination);
        reverseAdjacency.get(destination)!.add(source);
      }
    }
  }

  for (const [destination, incomingGroups] of incomingEdgeSources) {
    const incomingSources = reverseAdjacency.get(destination)!;
    if (incomingSources.size <= 1) {
      continue;
    }
    if (
      incomingGroups.length !== 1 ||
      incomingGroups[0].length !== incomingSources.size
    ) {
      throw new Error(
        `Graph subagent "${config.type}" fan-in to "${destination}" must use one edge with an array-valued from field.`
      );
    }
  }

  const isSimpleChain = [...agentIds].every(
    (agentId) =>
      adjacency.get(agentId)!.size <= 1 &&
      reverseAdjacency.get(agentId)!.size <= 1
  );
  for (const { edgeIndex, destinations } of promptedEdges) {
    if (destinations.length !== 1) {
      throw new Error(
        `Graph subagent "${config.type}" prompted edge ${edgeIndex} must have exactly one destination.`
      );
    }
    if (!isSimpleChain && destinations[0] !== config.resultAgentId) {
      throw new Error(
        `Graph subagent "${config.type}" prompted edge ${edgeIndex} must target resultAgentId "${config.resultAgentId}" after parallel branches converge.`
      );
    }
  }

  const roots = [...agentIds].filter(
    (agentId) => reverseAdjacency.get(agentId)!.size === 0
  );
  if (roots.length !== 1 || roots[0] !== config.entryAgentId) {
    throw new Error(
      `Graph subagent "${config.type}" must have entryAgentId "${config.entryAgentId}" as its only root.`
    );
  }
  const sinks = [...agentIds].filter(
    (agentId) => adjacency.get(agentId)!.size === 0
  );
  if (sinks.length !== 1 || sinks[0] !== config.resultAgentId) {
    throw new Error(
      `Graph subagent "${config.type}" must have resultAgentId "${config.resultAgentId}" as its only sink.`
    );
  }

  const remainingIncoming = new Map<string, number>();
  const ready: string[] = [];
  for (const agentId of agentIds) {
    const incomingCount = reverseAdjacency.get(agentId)!.size;
    remainingIncoming.set(agentId, incomingCount);
    if (incomingCount === 0) {
      ready.push(agentId);
    }
  }
  let visitedCount = 0;
  while (ready.length > 0) {
    const current = ready.pop()!;
    visitedCount++;
    for (const destination of adjacency.get(current) ?? []) {
      const incomingCount = remainingIncoming.get(destination)! - 1;
      remainingIncoming.set(destination, incomingCount);
      if (incomingCount === 0) {
        ready.push(destination);
      }
    }
  }
  if (visitedCount !== agentIds.size) {
    throw new Error(`Graph subagent "${config.type}" must be acyclic.`);
  }

  const reachableFromEntry = visitReachable(config.entryAgentId, adjacency);
  const canReachResult = visitReachable(config.resultAgentId, reverseAdjacency);
  if (
    reachableFromEntry.size !== agentIds.size ||
    canReachResult.size !== agentIds.size
  ) {
    throw new Error(
      `Every agent in graph subagent "${config.type}" must be reachable from entryAgentId and able to reach resultAgentId.`
    );
  }
}

function normalizeSingleAgentConfig(
  config: SubagentConfig,
  parentContext: AgentContext
): ExecutableSubagentConfig | null {
  const candidate = config as GraphSubagentConfigCandidate;
  if (
    candidate.kind === 'agent' &&
    (candidate.agents !== undefined ||
      candidate.edges !== undefined ||
      candidate.entryAgentId !== undefined ||
      candidate.resultAgentId !== undefined)
  ) {
    throw new Error(
      `Single-agent subagent "${config.type}" cannot define graph topology fields.`
    );
  }
  if (config.self === true && config.resolveAgentInputs != null) {
    throw new Error(
      `Subagent "${config.type}" cannot combine self with resolveAgentInputs.`
    );
  }
  if (
    getSubagentHostArgSpecs(config).length > 0 &&
    (config.resolveAgentInputs == null || config.agentInputs != null)
  ) {
    throw new Error(
      `Subagent "${config.type}" declares hostArgs, which only a lazy resolveAgentInputs config without eager agentInputs can receive.`
    );
  }
  if (config.agentInputs != null) {
    return config as ResolvedSingleAgentSubagentConfig;
  }
  if (config.self === true && parentContext._sourceInputs != null) {
    return {
      ...config,
      agentInputs: { ...parentContext._sourceInputs },
    };
  }
  if (config.resolveAgentInputs == null) {
    return null;
  }
  if (!isNonEmptyString(config.configId)) {
    throw new Error(
      `Lazy subagent "${config.type}" requires a non-empty "configId".`
    );
  }
  return config as ExecutableSubagentConfig;
}

/** Normalize eager, self, lazy, and graph subagent entries for advertisement. */
export function normalizeSubagentConfigEntries(
  configs: SubagentConfigEntry[],
  parentContext: AgentContext
): ExecutableSubagentConfigEntry[] {
  const normalized: ExecutableSubagentConfigEntry[] = [];
  const seenTypes = new Set<string>();
  for (const config of configs) {
    let executable: ExecutableSubagentConfigEntry | null;
    if (isGraphSubagentConfig(config)) {
      validateGraphSubagentConfig(config);
      executable = config;
    } else {
      executable = normalizeSingleAgentConfig(config, parentContext);
    }
    if (executable == null) {
      continue;
    }
    if (seenTypes.has(executable.type)) {
      throw new Error(
        `Duplicate subagent type "${executable.type}". Each SubagentConfig must have a unique "type" field.`
      );
    }
    seenTypes.add(executable.type);
    normalized.push(executable);
  }
  return normalized;
}

/** Resolve self-spawn inputs and validate explicit graph configs. */
export function resolveSubagentConfigEntries(
  configs: SubagentConfigEntry[],
  parentContext: AgentContext
): ResolvedSubagentConfigEntry[] {
  return normalizeSubagentConfigEntries(configs, parentContext).filter(
    (config): config is ResolvedSubagentConfigEntry =>
      isGraphSubagentConfig(config) || config.agentInputs != null
  );
}

/** Resolve legacy single-agent configs with the original public signature. */
export function resolveSubagentConfigs(
  configs: SubagentConfig[],
  parentContext: AgentContext
): ResolvedSubagentConfig[] {
  const graphConfig = configs.find(isGraphSubagentConfig);
  if (graphConfig != null) {
    throw new Error(
      `Graph subagent "${graphConfig.type}" requires resolveSubagentConfigEntries().`
    );
  }
  return resolveSubagentConfigEntries(
    configs,
    parentContext
  ) as ResolvedSubagentConfig[];
}

/** Normalize legacy single-agent configs while rejecting graph entries. */
export function normalizeSubagentConfigs(
  configs: SubagentConfig[],
  parentContext: AgentContext
): ExecutableSubagentConfig[] {
  const graphConfig = configs.find(isGraphSubagentConfig);
  if (graphConfig != null) {
    throw new Error(
      `Graph subagent "${graphConfig.type}" requires normalizeSubagentConfigEntries().`
    );
  }
  return normalizeSubagentConfigEntries(
    configs,
    parentContext
  ) as ExecutableSubagentConfig[];
}

function prepareChildInputs({
  agentInputs,
  agentId,
  self,
  allowNested,
  parentMaxDepth,
  keepToolDefinitions,
}: {
  agentInputs: AgentInputs;
  agentId: string;
  self: boolean;
  allowNested: boolean;
  parentMaxDepth: number;
  keepToolDefinitions: boolean;
}): AgentInputs {
  const childInputs: AgentInputs = {
    ...agentInputs,
    agentId,
    toolDefinitions: keepToolDefinitions
      ? agentInputs.toolDefinitions
      : undefined,
    initialSummary: undefined,
    discoveredTools: undefined,
    graphTools: self ? undefined : agentInputs.graphTools,
    compactionSemanticIndex: self
      ? undefined
      : agentInputs.compactionSemanticIndex,
  };
  if (allowNested) {
    childInputs.maxSubagentDepth = Math.max(0, parentMaxDepth - 1);
  } else {
    childInputs.subagentConfigs = undefined;
    childInputs.maxSubagentDepth = undefined;
  }
  return childInputs;
}

export function buildChildInputs(
  config: ResolvedSubagentConfig,
  childAgentId: string,
  parentMaxDepth: number,
  keepToolDefinitions: boolean = false
): AgentInputs {
  return prepareChildInputs({
    agentInputs: config.agentInputs,
    agentId: childAgentId,
    self: config.self === true,
    allowNested: config.allowNested === true,
    parentMaxDepth,
    keepToolDefinitions,
  });
}

export function buildGraphChildInputs(
  config: GraphSubagentConfig,
  parentMaxDepth: number,
  keepToolDefinitions: boolean = false
): AgentInputs[] {
  return config.agents.map((agentInputs) =>
    prepareChildInputs({
      agentInputs,
      agentId: agentInputs.agentId,
      self: false,
      allowNested: false,
      parentMaxDepth,
      keepToolDefinitions,
    })
  );
}

export function createChildGraphPlan({
  config,
  executionSuffix,
  parentAgentId,
  parentMaxDepth,
  keepToolDefinitions,
}: CreateChildGraphPlanParams): ChildGraphPlan {
  if (isGraphSubagentConfig(config)) {
    validateGraphSubagentConfig(config);
    const agents = buildGraphChildInputs(
      config,
      parentMaxDepth,
      keepToolDefinitions
    );
    return {
      kind: 'graph',
      subjectAgentId: `${parentAgentId ?? 'agent'}_subgraph_${executionSuffix}`,
      checkpointNodeId: config.resultAgentId,
      agents,
      memberInputs: new Map(
        agents.map((agentInputs) => [agentInputs.agentId, agentInputs])
      ),
      edges: config.edges.map((edge) => ({
        ...edge,
        from: Array.isArray(edge.from) ? [...edge.from] : edge.from,
        to: Array.isArray(edge.to) ? [...edge.to] : edge.to,
      })),
      entryAgentId: config.entryAgentId,
      resultAgentId: config.resultAgentId,
    };
  }

  const childAgentId =
    config.agentInputs.agentId ||
    `${parentAgentId ?? 'agent'}_sub_${executionSuffix}`;
  const childInputs = buildChildInputs(
    config,
    childAgentId,
    parentMaxDepth,
    keepToolDefinitions
  );
  return {
    kind: 'agent',
    subjectAgentId: childAgentId,
    checkpointNodeId: childAgentId,
    agents: [childInputs],
    memberInputs: new Map([[childAgentId, childInputs]]),
  };
}
