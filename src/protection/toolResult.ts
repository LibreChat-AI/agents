import { types } from 'node:util';
import { ToolMessage } from '@langchain/core/messages';
import { GraphInterrupt, ParentCommand } from '@langchain/langgraph';
import {
  StructuredTool,
  DynamicStructuredTool,
  DynamicTool,
  Tool,
} from '@langchain/core/tools';
import type { CallbackManagerForToolRun } from '@langchain/core/callbacks/manager';
import type { RunnableConfig } from '@langchain/core/runnables';
import type {
  ProviderTextProtection,
  ProviderTextProtectionResult,
  ProviderTextProtectionErrorCode,
} from './providerText';
import type {
  GenericTool,
  ToolExecuteResult,
  ToolExecuteBatchRequest,
  ToolCallRequest,
} from '@/types';
import type { ToolOutputReferenceState } from '@/tools/toolOutputReferences';
import {
  ProviderTextAttempt,
  ProviderTextProtectionError,
  validateProviderTextProtection,
} from './providerText';
import { PreparedSubagentError } from '@/tools/preparedSubagents';
import { StreamLimitExceededError } from '@/llm/streamLimits';

export const TOOL_RESULT_PROTECTION_VERSION = 1;
export type ToolResultProtectionResult = ProviderTextProtectionResult;

/** Host selects trusted synchronous plain-text tools. Structured/file/background outputs stay gated. */
export interface ToolResultProtection {
  readonly version: 1;
  readonly toolNames: readonly string[];
  readonly timeoutMs: number;
  readonly maxAttemptBytes: number;
  readonly maxBufferedBytes: number;
  readonly classify: (content: string) => 'prose' | 'unsupported';
  readonly inspect: (input: {
    readonly version: 1;
    readonly content: string;
    readonly toolName: string;
    readonly toolCallId: string;
    readonly target: {
      readonly source: 'tool_argument';
      readonly field: 'output';
      readonly provenance: 'tool';
      readonly outcome: 'success' | 'error';
    };
    readonly signal: AbortSignal;
  }) => ToolResultProtectionResult | Promise<ToolResultProtectionResult>;
}

export class ToolResultProtectionError extends ProviderTextProtectionError {
  constructor(code: ProviderTextProtectionErrorCode) {
    super(code);
    this.name = 'ToolResultProtectionError';
  }
}

const adapters = new WeakMap<ToolResultProtection, ProviderTextProtection>();
const approved = new WeakMap<
  object,
  {
    policy: ToolResultProtection;
    name: string;
    id: string;
    text: string;
    status: string;
    referenceContent?: string;
  }
>();
const errors = new Set<ProviderTextProtectionErrorCode>([
  'blocked',
  'unavailable',
  'timeout',
  'overflow',
  'cancelled',
  'unsupported',
  'incompatible',
]);

export function validateToolResultProtection(
  policy: ToolResultProtection
): void {
  const version: number = policy.version;
  if (version !== 1) throw new ToolResultProtectionError('incompatible');
  if (
    typeof policy.inspect !== 'function' ||
    typeof policy.classify !== 'function'
  )
    throw new ToolResultProtectionError('unavailable');
  if (
    !Array.isArray(policy.toolNames) ||
    policy.toolNames.length === 0 ||
    policy.toolNames.length > 128 ||
    policy.toolNames.some(
      (name) =>
        typeof name !== 'string' || name.trim() !== name || name.length === 0
    )
  )
    throw new ToolResultProtectionError('incompatible');
  budget(policy);
}

function budget(policy: ToolResultProtection): ProviderTextProtection {
  let adapter = adapters.get(policy);
  if (adapter == null) {
    adapter = {
      version: 1,
      timeoutMs: policy.timeoutMs,
      maxAttemptBytes: policy.maxAttemptBytes,
      maxBufferedBytes: policy.maxBufferedBytes,
      classify: (): 'prose' => 'prose',
      inspect: ({ content }): ProviderTextProtectionResult => ({
        version: 1,
        ok: true,
        value: { content, replacements: 0, categories: [] },
      }),
    };
    validateProviderTextProtection(adapter);
    adapters.set(policy, adapter);
  }
  return adapter;
}

export function requiresToolResultProtection(
  policy: ToolResultProtection | undefined,
  name: string
): policy is ToolResultProtection {
  return policy != null && policy.toolNames.includes(name);
}

function plainObject(
  value: object,
  allowed: readonly string[],
  checkpointData = false
): void {
  const candidate: unknown = value;
  if (
    candidate == null ||
    typeof candidate !== 'object' ||
    types.isProxy(value)
  )
    throw new ToolResultProtectionError('unsupported');
  const prototype = Object.getPrototypeOf(value);
  if (!checkpointData && prototype !== Object.prototype && prototype !== null)
    throw new ToolResultProtectionError('unsupported');
  const descriptors = Object.getOwnPropertyDescriptors(value);
  for (const [key, descriptor] of Object.entries(descriptors)) {
    if (!allowed.includes(key) || !('value' in descriptor))
      throw new ToolResultProtectionError('unsupported');
  }
}

function decision(result: ToolResultProtectionResult | undefined): string {
  if (result == null || typeof result !== 'object' || types.isProxy(result))
    throw new ToolResultProtectionError('incompatible');
  const version: number = result.version;
  if (version !== 1 || typeof result.ok !== 'boolean')
    throw new ToolResultProtectionError('incompatible');
  if (!result.ok)
    throw new ToolResultProtectionError(
      errors.has(result.error.code) ? result.error.code : 'incompatible'
    );
  if (
    typeof result.value.content !== 'string' ||
    !Number.isSafeInteger(result.value.replacements) ||
    result.value.replacements < 0 ||
    !Array.isArray(result.value.categories) ||
    result.value.categories.length > 32 ||
    result.value.categories.some(
      (entry) =>
        typeof entry.category !== 'string' ||
        !Number.isSafeInteger(entry.count) ||
        entry.count <= 0
    )
  )
    throw new ToolResultProtectionError('incompatible');
  return result.value.content;
}

async function release(
  policy: ToolResultProtection,
  name: string,
  id: string,
  text: unknown,
  status: 'success' | 'error',
  attempt: ProviderTextAttempt
): Promise<string> {
  attempt.check();
  if (typeof text !== 'string')
    throw new ToolResultProtectionError('unsupported');
  attempt.append(text);
  try {
    if (policy.classify(text) !== 'prose')
      throw new ToolResultProtectionError('unsupported');
    attempt.check();
    const result = await attempt.wait(
      Promise.resolve(
        policy.inspect({
          version: 1,
          content: text,
          toolName: name,
          toolCallId: id,
          target: {
            source: 'tool_argument',
            field: 'output',
            provenance: 'tool',
            outcome: status,
          },
          signal: attempt.signal,
        })
      )
    );
    attempt.check();
    const canonical = decision(result);
    attempt.check();
    attempt.retain(canonical.length * 2);
    return canonical;
  } catch (error) {
    attempt.check();
    if (error instanceof ProviderTextProtectionError)
      throw new ToolResultProtectionError(error.code);
    throw new ToolResultProtectionError('unavailable');
  }
}

export async function protectToolText(
  policy: ToolResultProtection | undefined,
  name: string,
  id: string,
  text: unknown,
  status: 'success' | 'error',
  signal?: AbortSignal
): Promise<unknown> {
  if (!requiresToolResultProtection(policy, name)) return text;
  validateToolResultProtection(policy);
  const attempt = new ProviderTextAttempt(budget(policy), signal);
  try {
    return await release(policy, name, id, text, status, attempt);
  } finally {
    attempt.finish();
  }
}

export function isReleasedToolError(
  policy: ToolResultProtection | undefined,
  name: string,
  id: string,
  error: Error
): boolean {
  const entry = approved.get(error);
  return (
    entry != null &&
    entry.policy === policy &&
    entry.name === name &&
    entry.id === id &&
    entry.status === 'error' &&
    entry.text === error.message
  );
}

/** Internal provenance only after an awaited release; never exposed from the root SDK entry. */
export function markReleasedToolMessage(
  policy: ToolResultProtection | undefined,
  name: string,
  id: string,
  message: ToolMessage,
  referenceContent?: string
): ToolMessage {
  if (
    requiresToolResultProtection(policy, name) &&
    typeof message.content === 'string'
  ) {
    approved.set(message, {
      policy,
      name,
      id,
      text: message.content,
      status: message.status ?? 'success',
      referenceContent:
        referenceContent ?? approved.get(message)?.referenceContent,
    });
  }
  return message;
}

export function hasReleasedToolReference(
  policy: ToolResultProtection,
  name: string,
  id: string,
  message: ToolMessage,
  referenceContent: string | undefined
): boolean {
  const entry = approved.get(message);
  return (
    entry != null &&
    entry.policy === policy &&
    entry.name === name &&
    entry.id === id &&
    entry.text === message.content &&
    entry.status === message.status &&
    entry.referenceContent === referenceContent
  );
}

function validateReplayReferenceMetadata(message: ToolMessage): void {
  const metadata = message.additional_kwargs;
  const key = metadata._refKey;
  const scope = metadata._refScope;
  const unresolved = metadata._unresolvedRefs;
  if (
    (key != null &&
      (typeof key !== 'string' || !/^tool\d+turn\d+$/.test(key))) ||
    (scope != null && typeof scope !== 'string') ||
    (unresolved != null &&
      (!Array.isArray(unresolved) ||
        unresolved.length > 128 ||
        unresolved.some(
          (ref) => typeof ref !== 'string' || !/^tool\d+turn\d+$/.test(ref)
        )))
  )
    throw new ToolResultProtectionError('unsupported');
}

export async function protectToolMessage(
  policy: ToolResultProtection | undefined,
  name: string,
  id: string,
  message: ToolMessage,
  signal?: AbortSignal,
  ownedReplayMetadata = false
): Promise<ToolMessage> {
  if (!requiresToolResultProtection(policy, name)) return message;
  if (types.isProxy(message))
    throw new ToolResultProtectionError('unsupported');
  for (const key of [
    'content',
    'artifact',
    'name',
    'id',
    'tool_call_id',
    'status',
    'additional_kwargs',
    'response_metadata',
  ]) {
    const descriptor = Object.getOwnPropertyDescriptor(message, key);
    if (descriptor != null && !('value' in descriptor))
      throw new ToolResultProtectionError('unsupported');
  }
  const status: string | undefined = message.status;
  if (
    types.isProxy(message) ||
    message.artifact != null ||
    message.tool_call_id !== id ||
    (message.name != null && message.name !== name) ||
    (status != null && status !== 'success' && status !== 'error')
  )
    throw new ToolResultProtectionError('unsupported');
  plainObject(message.additional_kwargs, [
    '_refKey',
    '_refScope',
    '_unresolvedRefs',
  ]);
  if (ownedReplayMetadata) validateReplayReferenceMetadata(message);
  if (
    approved.get(message) == null &&
    !ownedReplayMetadata &&
    (Object.keys(message.additional_kwargs).length > 0 ||
      (message.id != null && message.id !== id))
  )
    throw new ToolResultProtectionError('unsupported');
  if (Object.keys(message.response_metadata).length > 0)
    throw new ToolResultProtectionError('unsupported');
  const existing = approved.get(message);
  if (
    existing?.policy === policy &&
    existing.name === name &&
    existing.id === id &&
    existing.text === message.content &&
    existing.status === (message.status ?? 'success')
  )
    return message;
  const content = await protectToolText(
    policy,
    name,
    id,
    message.content,
    message.status === 'error' ? 'error' : 'success',
    signal
  );
  const safe = new ToolMessage({
    name: message.name ?? name,
    id: message.id,
    tool_call_id: id,
    status: message.status ?? 'success',
    content: content as string,
    additional_kwargs: message.additional_kwargs,
  });
  approved.set(safe, {
    policy,
    name,
    id,
    text: safe.content as string,
    status: safe.status ?? 'success',
  });
  return safe;
}

/** Validate host-owned envelopes before even reading their routing IDs. */
export function validateToolExecuteResults(
  results: ToolExecuteResult[],
  maxResults: number
): void {
  if (
    !Array.isArray(results) ||
    types.isProxy(results) ||
    Object.getPrototypeOf(results) !== Array.prototype ||
    results.length > maxResults
  )
    throw new ToolResultProtectionError('unsupported');
  for (let index = 0; index < results.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(results, String(index));
    if (descriptor == null || !('value' in descriptor))
      throw new ToolResultProtectionError('unsupported');
    plainObject(descriptor.value as ToolExecuteResult, [
      'toolCallId',
      'received_at',
      'content',
      'status',
      'errorMessage',
      'artifact',
      'injectedMessages',
      'outcome',
      'outcome_patch',
    ]);
  }
  for (const key of Reflect.ownKeys(results)) {
    if (
      key !== 'length' &&
      (typeof key !== 'string' || !/^(0|[1-9]\d*)$/.test(key))
    )
      throw new ToolResultProtectionError('unsupported');
  }
}

export async function protectToolExecuteResult(
  policy: ToolResultProtection | undefined,
  request: Pick<ToolCallRequest, 'id' | 'name'>,
  result: ToolExecuteResult,
  signal?: AbortSignal
): Promise<ToolExecuteResult> {
  request = checkedRequestIdentity(request);
  if (!requiresToolResultProtection(policy, request.name)) return result;
  plainObject(result, [
    'toolCallId',
    'received_at',
    'content',
    'status',
    'errorMessage',
    'artifact',
    'injectedMessages',
  ]);
  const status: string = result.status;
  const receivedAt = result.received_at;
  if (
    result.toolCallId !== request.id ||
    (status !== 'success' && status !== 'error') ||
    result.artifact != null ||
    (result.received_at != null &&
      (typeof result.received_at !== 'number' ||
        !Number.isFinite(result.received_at))) ||
    (result.injectedMessages?.length ?? 0) > 0 ||
    (status === 'success' && result.errorMessage != null)
  )
    throw new ToolResultProtectionError('unsupported');
  const text =
    status === 'error'
      ? (result.errorMessage ?? result.content)
      : result.content;
  const cached = approved.get(result);
  if (
    cached?.policy === policy &&
    cached.name === request.name &&
    cached.id === request.id &&
    cached.text === text &&
    cached.status === status
  )
    return result;
  const canonical = (await protectToolText(
    policy,
    request.name,
    request.id,
    text,
    status,
    signal
  )) as string;
  const safe: ToolExecuteResult = {
    toolCallId: request.id,
    status,
    content: status === 'error' ? '' : canonical,
    ...(status === 'error' ? { errorMessage: canonical } : {}),
    ...(receivedAt == null ? {} : { received_at: receivedAt }),
  };
  approved.set(safe, {
    policy,
    name: request.name,
    id: request.id,
    text: canonical,
    status: safe.status,
  });
  return safe;
}

/** Old checkpoints cannot establish which required source produced a reference. */
export function validateToolReferenceSources(
  policy: ToolResultProtection,
  state: ToolOutputReferenceState
): void {
  for (const entry of state.entries) {
    plainObject(entry, ['key', 'value', 'protection'], true);
    if (
      typeof Object.getOwnPropertyDescriptor(entry, 'key')?.value !==
        'string' ||
      typeof Object.getOwnPropertyDescriptor(entry, 'value')?.value !== 'string'
    )
      throw new ToolResultProtectionError('incompatible');
    const source: typeof entry.protection = Object.getOwnPropertyDescriptor(
      entry,
      'protection'
    )?.value;
    if (source == null) throw new ToolResultProtectionError('incompatible');
    plainObject(
      source,
      ['version', 'toolName', 'toolCallId', 'protected'],
      true
    );
    for (const field of ['version', 'toolName', 'toolCallId', 'protected']) {
      if (!Object.hasOwn(source, field))
        throw new ToolResultProtectionError('incompatible');
    }
    const version: number = source.version;
    if (
      version !== 1 ||
      typeof source.toolName !== 'string' ||
      source.toolName.length === 0 ||
      typeof source.toolCallId !== 'string' ||
      typeof source.protected !== 'boolean' ||
      (requiresToolResultProtection(policy, source.toolName) &&
        !source.protected)
    )
      throw new ToolResultProtectionError('incompatible');
  }
}

export async function protectToolReferenceState(
  policy: ToolResultProtection | undefined,
  state: ToolOutputReferenceState | undefined,
  signal?: AbortSignal
): Promise<ToolOutputReferenceState | undefined> {
  if (policy == null || state == null) return state;
  validateToolReferenceSources(policy, state);
  const entries = await Promise.all(
    state.entries.map(async (entry) => ({
      ...entry,
      value: (await protectToolText(
        policy,
        entry.protection!.toolName,
        entry.protection!.toolCallId,
        entry.value,
        'success',
        signal
      )) as string,
    }))
  );
  return { ...state, entries };
}

type ToolRequestIdentity = Readonly<Pick<ToolCallRequest, 'id' | 'name'>>;
type BoundToolExecuteRequest = {
  readonly policy: ToolResultProtection;
  readonly signal?: AbortSignal;
  readonly calls: readonly ToolRequestIdentity[];
  readonly required: boolean;
};
const hostPolicies = new WeakMap<
  ToolExecuteBatchRequest,
  BoundToolExecuteRequest
>();
const requestIdentities = new WeakMap<object, ToolRequestIdentity>();

function checkedRequestIdentity(
  request: ToolRequestIdentity
): ToolRequestIdentity {
  const identity = requestIdentities.get(request);
  if (identity == null) return request;
  if (
    types.isProxy(request) ||
    Object.getOwnPropertyDescriptor(request, 'id')?.value !== identity.id ||
    Object.getOwnPropertyDescriptor(request, 'name')?.value !== identity.name
  )
    throw new ToolResultProtectionError('unsupported');
  return identity;
}

function validateBoundToolRequests(
  request: ToolExecuteBatchRequest,
  binding: BoundToolExecuteRequest
): void {
  if (types.isProxy(request))
    throw new ToolResultProtectionError('unsupported');
  const calls: ToolCallRequest[] | undefined = Object.getOwnPropertyDescriptor(
    request,
    'toolCalls'
  )?.value;
  if (
    calls == null ||
    !Array.isArray(calls) ||
    types.isProxy(calls) ||
    calls.length !== binding.calls.length
  )
    throw new ToolResultProtectionError('unsupported');
  for (let index = 0; index < calls.length; index++) {
    const call: ToolCallRequest | undefined = Object.getOwnPropertyDescriptor(
      calls,
      String(index)
    )?.value;
    if (
      call == null ||
      types.isProxy(call) ||
      Object.getOwnPropertyDescriptor(call, 'id')?.value !==
        binding.calls[index].id ||
      Object.getOwnPropertyDescriptor(call, 'name')?.value !==
        binding.calls[index].name
    )
      throw new ToolResultProtectionError('unsupported');
  }
}

/** Internal snapshots bind selection to dispatched identities, never mutable host requests. */
export function bindToolExecuteProtection(
  request: ToolExecuteBatchRequest,
  policy: ToolResultProtection | undefined,
  signal?: AbortSignal
): void {
  if (policy == null) return;
  const calls = request.toolCalls.map((call) => {
    const identity = Object.freeze({ id: call.id, name: call.name });
    requestIdentities.set(call, identity);
    return identity;
  });
  hostPolicies.set(request, {
    policy,
    signal,
    calls,
    required: calls.some((call) =>
      requiresToolResultProtection(policy, call.name)
    ),
  });
}

export function inheritToolExecuteProtection(
  request: ToolExecuteBatchRequest,
  hostView: ToolExecuteBatchRequest
): void {
  const binding = hostPolicies.get(request);
  if (binding != null) hostPolicies.set(hostView, binding);
}

export function hasRequiredToolExecuteProtection(
  request: ToolExecuteBatchRequest
): boolean {
  return hostPolicies.get(request)?.required === true;
}

export function validateToolExecuteProtection(
  request: ToolExecuteBatchRequest,
  hostView = request
): void {
  const binding = hostPolicies.get(request);
  if (binding == null) return;
  validateBoundToolRequests(request, binding);
  validateBoundToolRequests(hostView, binding);
  if (binding.required && binding.signal?.aborted === true) {
    const reason: unknown = binding.signal.reason;
    if (
      reason instanceof ProviderTextProtectionError ||
      reason instanceof PreparedSubagentError ||
      reason instanceof StreamLimitExceededError
    )
      throw reason;
    throw new ToolResultProtectionError('cancelled');
  }
}

export function protectToolExecuteBatch(
  request: ToolExecuteBatchRequest,
  results: ToolExecuteResult[],
  hostView = request
): Promise<ToolExecuteResult[]> | undefined {
  const binding = hostPolicies.get(request);
  if (binding == null) return undefined;
  return protectBoundToolExecuteBatch(request, hostView, results, binding);
}

async function protectBoundToolExecuteBatch(
  request: ToolExecuteBatchRequest,
  hostView: ToolExecuteBatchRequest,
  results: ToolExecuteResult[],
  binding: BoundToolExecuteRequest
): Promise<ToolExecuteResult[]> {
  validateBoundToolRequests(request, binding);
  validateBoundToolRequests(hostView, binding);
  validateToolExecuteResults(results, binding.calls.length);
  const { policy, signal } = binding;
  const requests = new Map(binding.calls.map((call) => [call.id, call]));
  const seen = new Set<string>();
  for (const result of results) {
    if (!requests.has(result.toolCallId) || seen.has(result.toolCallId))
      throw new ToolResultProtectionError('unsupported');
    seen.add(result.toolCallId);
  }
  for (const call of binding.calls) {
    if (requiresToolResultProtection(policy, call.name) && !seen.has(call.id))
      throw new ToolResultProtectionError('unavailable');
  }
  const canonical = await Promise.all(
    results.map((result) =>
      protectToolExecuteResult(
        policy,
        requests.get(result.toolCallId)!,
        result,
        signal
      )
    )
  );
  validateBoundToolRequests(request, binding);
  validateBoundToolRequests(hostView, binding);
  return canonical;
}

/** Keep child/control APIs usable, but no raw body observations escape before release. */
function toolBodyCallbacks(
  manager: CallbackManagerForToolRun | undefined
): CallbackManagerForToolRun | undefined {
  if (manager == null) return manager;
  return Object.create(Object.getPrototypeOf(manager), {
    ...Object.getOwnPropertyDescriptors(manager),
    handlers: { value: [], enumerable: true },
    inheritableHandlers: { value: [], enumerable: true },
  }) as CallbackManagerForToolRun;
}

/** Intercepts StructuredTool before native tool-end callbacks, never mutating a shared tool. */
export function withToolResultBoundary(
  tool: GenericTool,
  policy: ToolResultProtection | undefined,
  id: string,
  config: RunnableConfig,
  requestName = tool.name
): GenericTool {
  if (policy != null) validateToolResultProtection(policy);
  if (
    !requiresToolResultProtection(policy, tool.name) &&
    !requiresToolResultProtection(policy, requestName)
  )
    return tool;
  if (requestName !== tool.name)
    throw new ToolResultProtectionError('unsupported');
  const toolName = tool.name;
  if (
    [
      'subagent',
      'run_tools_with_code',
      'run_tools_with_bash',
      'read_file',
      'skill',
    ].includes(tool.name)
  )
    throw new ToolResultProtectionError('unsupported');
  if (!(tool instanceof StructuredTool))
    throw new ToolResultProtectionError('unsupported');
  const responseFormat = tool.responseFormat;
  if (
    tool.invoke !== StructuredTool.prototype.invoke ||
    ![
      StructuredTool.prototype.call,
      DynamicStructuredTool.prototype.call,
      DynamicTool.prototype.call,
      Tool.prototype.call,
    ].includes(tool.call)
  )
    throw new ToolResultProtectionError('unsupported');
  const protectedTool = Object.create(
    Object.getPrototypeOf(tool),
    Object.getOwnPropertyDescriptors(tool)
  ) as StructuredTool;
  protectedTool.responseFormat = 'content';
  const admittedTool = tool;
  const invoke: (
    input: unknown,
    manager: CallbackManagerForToolRun | undefined,
    config: RunnableConfig
  ) => Promise<unknown> = Reflect.get(tool, '_call');
  Object.defineProperty(protectedTool, '_call', {
    value: async (
      input: unknown,
      manager: CallbackManagerForToolRun | undefined,
      effective?: RunnableConfig
    ): Promise<ToolMessage> => {
      const attempt = new ProviderTextAttempt(
        budget(policy),
        effective?.signal ?? config.signal
      );
      try {
        let raw: unknown;
        let status: 'success' | 'error' = 'success';
        try {
          raw = await attempt.wait(
            Promise.resolve(
              invoke.call(tool, input, toolBodyCallbacks(manager), {
                ...effective,
                callbacks: [],
              })
            )
          );
        } catch (error) {
          attempt.check();
          if (
            error instanceof ProviderTextProtectionError ||
            error instanceof PreparedSubagentError ||
            error instanceof StreamLimitExceededError ||
            error instanceof GraphInterrupt
          )
            throw error;
          if (error instanceof ParentCommand)
            throw new ToolResultProtectionError('unsupported');
          if (!(error instanceof Error))
            throw new ToolResultProtectionError('unsupported');
          raw = error.message;
          status = 'error';
        }
        attempt.check();
        if (admittedTool.name !== toolName || admittedTool.responseFormat !== responseFormat)
          throw new ToolResultProtectionError('unsupported');
        if (raw instanceof ToolMessage)
          return await protectToolMessage(
            policy,
            toolName,
            id,
            raw,
            attempt.signal
          );
        if (status === 'success' && responseFormat === 'content_and_artifact') {
          if (!Array.isArray(raw) || raw.length !== 2 || raw[1] != null)
            throw new ToolResultProtectionError('unsupported');
          raw = raw[0];
        }
        const content = await release(
          policy,
          toolName,
          id,
          raw,
          status,
          attempt
        );
        if (status === 'error') {
          const error = new Error(content);
          approved.set(error, {
            policy,
            name: toolName,
            id,
            text: content,
            status,
          });
          throw error;
        }
        const message = new ToolMessage({
          name: toolName,
          tool_call_id: id,
          content,
          status,
        });
        approved.set(message, {
          policy,
          name: toolName,
          id,
          text: content,
          status,
        });
        return message;
      } catch (error) {
        attempt.check();
        if (
          error instanceof ProviderTextProtectionError ||
          error instanceof PreparedSubagentError ||
          error instanceof StreamLimitExceededError ||
          error instanceof GraphInterrupt ||
          (error instanceof Error &&
            isReleasedToolError(policy, toolName, id, error))
        )
          throw error;
        throw new ToolResultProtectionError('unavailable');
      } finally {
        attempt.finish();
      }
    },
  });
  return protectedTool;
}
