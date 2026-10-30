import { nanoid } from 'nanoid';
import { types } from 'node:util';
import { RemoveMessage, ToolMessage } from '@langchain/core/messages';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import { BaseCheckpointSaver, GraphInterrupt, ParentCommand, Send } from '@langchain/langgraph';
import {
  StructuredTool,
  DynamicStructuredTool,
  DynamicTool,
  Tool,
} from '@langchain/core/tools';
import type { CallbackManagerForToolRun } from '@langchain/core/callbacks/manager';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type {
  GenericTool,
  RunStepResumeState,
  ToolExecuteResult,
  ToolExecuteBatchRequest,
  ToolCallRequest,
  ToolApprovalInterruptPayload,
} from '@/types';
import type {
  ProviderTextProtection,
  ProviderTextProtectionResult,
  ProviderTextProtectionErrorCode,
} from './providerText';
import type {
  ToolOutputReferenceRegistry,
  ToolOutputReferenceState,
} from '@/tools/toolOutputReferences';
import type { ToolCheckpointAdmission } from './toolCheckpoint';
import {
  ProviderTextAttempt,
  ProviderTextProtectionError,
  validateProviderTextProtection,
  isProviderTextRestartCancellation,
} from './providerText';
import { PreparedSubagentError } from '@/tools/preparedSubagents';
import { StreamLimitExceededError } from '@/llm/streamLimits';
import { isRunStepResumeState } from '@/tools/runStepResume';
import { ToolCheckpointSaver } from './toolCheckpoint';

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

const messageFields = [
  'content',
  'artifact',
  'name',
  'id',
  'tool_call_id',
  'status',
  'additional_kwargs',
  'response_metadata',
] as const;
type ToolMessageEnvelope = Pick<ToolMessage, (typeof messageFields)[number]>;

export type ToolMessageRelease = {
  readonly version: 1;
  readonly name: string;
  readonly id: string;
  readonly envelope: Omit<ToolMessageEnvelope, 'content'>;
};

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
    envelope?: ToolMessageEnvelope;
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

/** External exception objects are never proof of a raw-free failure. */
export function normalizeToolResultError(
  error: unknown,
  fallback: ProviderTextProtectionErrorCode = 'unavailable'
): ProviderTextProtectionError {
  if (
    error == null ||
    typeof error !== 'object' ||
    types.isProxy(error) ||
    !(error instanceof ProviderTextProtectionError)
  )
    return new ToolResultProtectionError(fallback);
  if (isProviderTextRestartCancellation(error)) return error;
  const code: unknown = Object.getOwnPropertyDescriptor(error, 'code')?.value;
  return new ToolResultProtectionError(
    typeof code === 'string' &&
    errors.has(code as ProviderTextProtectionErrorCode)
      ? (code as ProviderTextProtectionErrorCode)
      : 'incompatible'
  );
}

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
  for (const key of Reflect.ownKeys(descriptors)) {
    if (
      typeof key !== 'string' ||
      !allowed.includes(key) ||
      !('value' in descriptors[key])
    )
      throw new ToolResultProtectionError('unsupported');
  }
}

/** Copy indexed data only; checkpoint arrays may originate in the serializer's realm. */
function snapshotDataArray<T>(
  value: readonly T[],
  maxEntries: number,
  checkpointData = false
): T[] {
  if (!Array.isArray(value) || types.isProxy(value))
    throw new ToolResultProtectionError('unsupported');
  const prototype = Object.getPrototypeOf(value);
  const parent =
    prototype != null && !types.isProxy(prototype)
      ? Object.getPrototypeOf(prototype)
      : undefined;
  if (
    prototype !== Array.prototype &&
    (!checkpointData ||
      prototype == null ||
      types.isProxy(prototype) ||
      !Array.isArray(prototype) ||
      parent == null ||
      types.isProxy(parent) ||
      Object.getPrototypeOf(parent) !== null)
  )
    throw new ToolResultProtectionError('unsupported');
  const length: number = Object.getOwnPropertyDescriptor(
    value,
    'length'
  )?.value;
  if (!Number.isSafeInteger(length) || length < 0 || length > maxEntries)
    throw new ToolResultProtectionError('unsupported');
  const keys = Reflect.ownKeys(value);
  if (keys.length !== length + 1)
    throw new ToolResultProtectionError('unsupported');
  const result: T[] = [];
  for (const key of keys) {
    if (key === 'length') continue;
    if (
      typeof key !== 'string' ||
      !/^(0|[1-9]\d*)$/.test(key) ||
      Number(key) >= length
    )
      throw new ToolResultProtectionError('unsupported');
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (descriptor == null || !('value' in descriptor))
      throw new ToolResultProtectionError('unsupported');
    result[Number(key)] = descriptor.value as T;
  }
  return result;
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
      throw normalizeToolResultError(error);
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

export function detachReleasedToolError(policy: ToolResultProtection | undefined, name: string, id: string, error: Error): Error | undefined {
  const admission = approved.get(error);
  if (admission == null || admission.policy !== policy || admission.name !== name || admission.id !== id || admission.status !== 'error') return undefined;
  if (types.isProxy(error) || Object.getOwnPropertyDescriptor(error, 'message')?.value !== admission.text) throw new ToolResultProtectionError('unsupported');
  const detached = new Error(admission.text);
  approved.set(detached, { ...admission });
  return detached;
}

export function snapshotToolErrorText(error: unknown): string {
  if (error == null || typeof error !== 'object' || types.isProxy(error)) throw new ToolResultProtectionError('unsupported');
  const descriptor = Object.getOwnPropertyDescriptor(error, 'message');
  const value: unknown = descriptor?.value;
  if (descriptor != null && (!('value' in descriptor) || typeof value !== 'string')) throw new ToolResultProtectionError('unsupported');
  return typeof value === 'string' ? value : 'Tool execution failed';
}

/** Read external error text only from an own data property, never callback accessors. */
export async function protectToolErrorText(policy: ToolResultProtection | undefined, name: string, id: string, error: unknown, signal?: AbortSignal): Promise<string> {
  if (!requiresToolResultProtection(policy, name)) return (error as Error).message || 'Tool execution failed';
  const detached = detachReleasedToolError(policy, name, id, error as Error);
  if (detached != null) return detached.message;
  const content = snapshotToolErrorText(error);
  return await protectToolText(policy, name, id, content, 'error', signal) as string;
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
    !types.isProxy(error) && Object.getOwnPropertyDescriptor(error, 'message')?.value === entry.text
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
      envelope: snapshotToolMessageEnvelope(message),
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
    entry.referenceContent === referenceContent &&
    entry.envelope != null &&
    matchesToolMessageEnvelope(
      snapshotToolMessageEnvelope(message),
      entry.envelope
    )
  );
}

function validateReplayReferenceMetadata(
  metadata: ToolMessage['additional_kwargs']
): void {
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

function ownMessageValue(message: object, key: string): unknown {
  const candidate: unknown = message;
  if (candidate == null || typeof candidate !== 'object' || types.isProxy(message)) throw new ToolResultProtectionError('unsupported');
  const descriptor = Object.getOwnPropertyDescriptor(message, key);
  if (descriptor != null) {
    if (!('value' in descriptor)) throw new ToolResultProtectionError('unsupported');
    return descriptor.value;
  }
  let prototype = Object.getPrototypeOf(message);
  while (prototype != null) {
    if (types.isProxy(prototype) || Object.getOwnPropertyDescriptor(prototype, key) != null) throw new ToolResultProtectionError('unsupported');
    prototype = Object.getPrototypeOf(prototype);
  }
  return undefined;
}

/** Command updates preserve other channel payloads; only message writes are admitted here. */
export function snapshotToolResumeUpdates(update: unknown): Array<[string, unknown]> {
  if (update == null) return [];
  if (Array.isArray(update)) return snapshotDataArray(update, Number.MAX_SAFE_INTEGER).map((entry) => {
    const pair = snapshotDataArray(entry as readonly unknown[], 2);
    if (pair.length !== 2 || typeof pair[0] !== 'string') throw new ToolResultProtectionError('unsupported');
    return [pair[0], pair[1]];
  });
  if (typeof update !== 'object' || types.isProxy(update)) throw new ToolResultProtectionError('unsupported');
  const keys = Object.getOwnPropertyNames(update);
  plainObject(update, keys);
  return keys.map((key) => [key, Object.getOwnPropertyDescriptor(update, key)!.value]);
}

export function snapshotToolResumeMessages(value: unknown): { messages: BaseMessage[]; overwrite: boolean } {
  if (value == null || typeof value !== 'object' || types.isProxy(value)) throw new ToolResultProtectionError('unsupported');
  const replacement = ownMessageValue(value, '__overwrite__');
  const overwrite = Object.hasOwn(value, '__overwrite__');
  const messages: unknown = overwrite ? replacement : value;
  if (Array.isArray(messages)) return { messages: snapshotDataArray(messages, Number.MAX_SAFE_INTEGER, true), overwrite };
  if (messages == null || typeof messages !== 'object' || typeof ownMessageValue(messages, 'type') !== 'string') throw new ToolResultProtectionError('unsupported');
  return { messages: [messages as BaseMessage], overwrite };
}

/** Removal controls carry identity only and must never be certified as retained output. */
export function snapshotToolMessageRemoval(message: BaseMessage): RemoveMessage | undefined {
  if (ownMessageValue(message, 'type') !== 'remove') return undefined;
  const id = ownMessageValue(message, 'id');
  if (typeof id !== 'string' || id === '') throw new ToolResultProtectionError('unsupported');
  const content = ownMessageValue(message, 'content');
  if (content !== '' && (content == null || !Array.isArray(content) || snapshotDataArray(content, 0).length !== 0)) throw new ToolResultProtectionError('unsupported');
  for (const key of ['name', 'artifact', 'tool_call_id', 'status', 'metadata']) if (ownMessageValue(message, key) != null) throw new ToolResultProtectionError('unsupported');
  for (const key of ['additional_kwargs', 'response_metadata']) {
    const metadata = ownMessageValue(message, key);
    if (metadata != null) plainObject(metadata as object, []);
  }
  return new RemoveMessage({ id });
}

/** Shallow-copy SDK checkpoint containers without invoking inherited or own accessors. */
export function snapshotToolCheckpointRecord<T extends object>(value: T, fields: readonly string[]): T {
  const candidate: unknown = value;
  if (candidate == null || typeof candidate !== 'object' || types.isProxy(value)) throw new ToolResultProtectionError('unsupported');
  plainObject(value, Object.getOwnPropertyNames(value), true);
  for (const field of fields) ownMessageValue(value, field);
  return Object.fromEntries(Object.entries(Object.getOwnPropertyDescriptors(value)).map(([key, descriptor]) => [key, descriptor.value])) as T;
}

export function snapshotToolCheckpointData<T>(input: T, active = new WeakSet<object>(), depth = 0): T {
  if (typeof input === 'function' || typeof input === 'symbol' || depth > 64) throw new ToolResultProtectionError('unsupported');
  if (input == null || typeof input !== 'object') return input;
  if (types.isProxy(input) || active.has(input)) throw new ToolResultProtectionError('unsupported');
  active.add(input);
  try {
    if (Array.isArray(input)) return snapshotToolCheckpointArray(input).map((value) => snapshotToolCheckpointData(value, active, depth + 1)) as T;
    const prototype = Object.getPrototypeOf(input);
    if (prototype != null && (types.isProxy(prototype) || Object.getPrototypeOf(prototype) !== null)) throw new ToolResultProtectionError('unsupported');
    const record = snapshotToolCheckpointRecord(input, []);
    return Object.fromEntries(Object.entries(record).map(([key, value]) => [key, snapshotToolCheckpointData(value, active, depth + 1)])) as T;
  } finally { active.delete(input); }
}

export function snapshotToolRunStepState(value: RunStepResumeState): RunStepResumeState {
  const carrier = snapshotToolCheckpointRecord(value, ['toolResultReleases']);
  if (carrier.toolResultReleases != null) carrier.toolResultReleases = snapshotToolMessageReleases(carrier.toolResultReleases);
  const state = snapshotToolCheckpointData(carrier);
  if (!isRunStepResumeState(state)) throw new ToolResultProtectionError('incompatible');
  return state;
}

/** Detach routing identities, not arguments or opaque unselected payloads. */
export function snapshotToolInputMessages(messages: readonly BaseMessage[]): BaseMessage[] {
  return snapshotToolHistoryMessages(messages).map((message) => {
    ownMessageValue(message, 'type');
    const descriptors = Object.getOwnPropertyDescriptors(message);
    for (const key of Reflect.ownKeys(descriptors)) if (!('value' in descriptors[key as keyof typeof descriptors])) throw new ToolResultProtectionError('unsupported');
    for (const key of ['additional_kwargs', 'response_metadata', 'lc_kwargs']) {
      if (Object.hasOwn(descriptors, key) && descriptors[key].value != null) descriptors[key] = { ...descriptors[key], value: snapshotToolCheckpointRecord(descriptors[key].value as object, []) };
    }
    const calls = ownMessageValue(message, 'tool_calls');
    if (calls != null) {
      const detached = snapshotToolCheckpointArray(calls as readonly object[]).map((call) => snapshotToolCheckpointRecord(call, ['id', 'name', 'args']));
      descriptors.tool_calls = { ...descriptors.tool_calls, value: detached };
      if (Object.hasOwn(descriptors, 'lc_kwargs') && descriptors.lc_kwargs.value != null && Object.hasOwn(descriptors.lc_kwargs.value as object, 'tool_calls')) Object.defineProperty(descriptors.lc_kwargs.value, 'tool_calls', { value: detached, enumerable: true, writable: true, configurable: true });
    }
    const copy = Object.create(Object.getPrototypeOf(message), descriptors) as BaseMessage;
    const admission = approved.get(message);
    if (admission != null) approved.set(copy, admission);
    return copy;
  });
}

export function snapshotToolCheckpointConfig<C extends RunnableConfig>(input: C): C {
  const config = snapshotToolCheckpointRecord(input, ['configurable']);
  if (config.configurable == null) return config;
  const scope = snapshotToolCheckpointRecord(config.configurable, ['checkpoint_id', 'checkpoint_ns', 'thread_id', 'thread_ts', 'checkpoint_map']);
  if (scope.checkpoint_map != null) scope.checkpoint_map = snapshotToolCheckpointData(scope.checkpoint_map);
  return { ...config, configurable: scope };
}

export function snapshotToolSend(input: unknown): Send {
  if (input == null || typeof input !== 'object') throw new ToolResultProtectionError('unsupported');
  const packet = snapshotToolCheckpointRecord(input as Send, ['node', 'args', 'timeout']);
  if (typeof packet.node !== 'string' || packet.node === '' || packet.args == null || typeof packet.args !== 'object') throw new ToolResultProtectionError('unsupported');
  const args = snapshotToolCheckpointRecord(packet.args as { messages?: BaseMessage[]; agentMessages?: BaseMessage[]; runStepState?: RunStepResumeState }, ['messages', 'agentMessages', 'runStepState', '__pregel_tasks']);
  if (Object.hasOwn(args, '__pregel_tasks')) throw new ToolResultProtectionError('unsupported');
  if (args.messages != null) args.messages = snapshotToolInputMessages(args.messages);
  if (args.agentMessages != null) args.agentMessages = snapshotToolInputMessages(args.agentMessages);
  if (args.runStepState != null) args.runStepState = snapshotToolRunStepState(args.runStepState);
  return Object.assign(Object.create(Send.prototype) as Send, packet, { args, timeout: snapshotToolCheckpointData(packet.timeout) });
}

export function snapshotToolCommand<C extends object>(input: C): C {
  if (types.isProxy(input)) throw new ToolResultProtectionError('unsupported');
  return Object.fromEntries(['resume', 'goto', 'update', 'graph', 'lc_direct_tool_output'].map((field) => [field, ownMessageValue(input, field)])) as C;
}

/** Resume decisions bind to checkpoint-owned proposals, never caller-supplied names. */
export async function protectToolApprovalResume(policy: ToolResultProtection, value: unknown, approvals: ReadonlyMap<string, ToolApprovalInterruptPayload | undefined>, interruptId?: string, signal?: AbortSignal): Promise<unknown> {
  const decide = async (input: unknown, request: ToolApprovalInterruptPayload['action_requests'][number] | undefined): Promise<unknown> => {
    if (input == null || typeof input !== 'object') throw new ToolResultProtectionError('unsupported');
    const decision = snapshotToolCheckpointRecord(input as { type?: string; responseText?: string; reason?: string }, ['type', 'responseText', 'reason', 'updatedInput']);
    const fields = { approve: ['type'], reject: ['type', 'reason'], edit: ['type', 'updatedInput'], respond: ['type', 'responseText'] };
    const allowed: string[] | undefined = Object.getOwnPropertyDescriptor(fields, decision.type ?? '')?.value;
    if (allowed == null || Object.keys(decision).some((key) => !allowed.includes(key))) throw new ToolResultProtectionError('unsupported');
    if (decision.type !== 'respond' && decision.type !== 'reject') return decision;
    if (request == null) throw new ToolResultProtectionError('unsupported');
    if (!requiresToolResultProtection(policy, request.name)) return decision;
    const field = decision.type === 'respond' ? 'responseText' : 'reason';
    const text = decision[field];
    if (text == null && field === 'reason') return decision;
    if (typeof text !== 'string') throw new ToolResultProtectionError('unsupported');
    return { ...decision, [field]: await protectToolText(policy, request.name, request.tool_call_id, text, field === 'reason' ? 'error' : 'success', signal) };
  };
  const batch = async (input: unknown, payload: ToolApprovalInterruptPayload): Promise<unknown> => {
    const requests = snapshotToolCheckpointArray(payload.action_requests).map((request) => snapshotToolCheckpointRecord(request, ['name', 'tool_call_id']));
    for (const request of requests) if (typeof request.name !== 'string' || request.name === '' || typeof request.tool_call_id !== 'string' || request.tool_call_id === '') throw new ToolResultProtectionError('incompatible');
    if (Array.isArray(input)) {
      const decisions = snapshotToolCheckpointArray(input);
      if (decisions.length !== requests.length) throw new ToolResultProtectionError('unsupported');
      return Promise.all(decisions.map((decision, index) => decide(decision, requests[index])));
    }
    if (input == null || typeof input !== 'object') throw new ToolResultProtectionError('unsupported');
    const decisions = snapshotToolCheckpointRecord(input, []);
    const byId = new Map(requests.map((request) => [request.tool_call_id, request]));
    return Object.fromEntries(await Promise.all(Object.entries(decisions).map(async ([id, decision]) => [id, await decide(decision, byId.get(id))])));
  };
  if (approvals.size === 0) throw new ToolResultProtectionError('unsupported');
  if (value != null && typeof value === 'object' && !Array.isArray(value)) {
    const record = snapshotToolCheckpointRecord(value, []);
    if (Object.keys(record).some((key) => approvals.has(key))) {
      return Object.fromEntries(await Promise.all(Object.entries(record).map(async ([id, decisions]) => {
        const payload = approvals.get(id);
        if (!approvals.has(id)) throw new ToolResultProtectionError('unsupported');
        return [id, payload == null ? decisions : await batch(decisions, payload)];
      })));
    }
  }
  const payload = interruptId == null ? undefined : approvals.get(interruptId);
  if (interruptId == null || !approvals.has(interruptId) || approvals.size !== 1) throw new ToolResultProtectionError('unsupported');
  return payload == null ? value : batch(value, payload);
}

export function getToolResultCheckpointLookup<C extends RunnableConfig>(config: C): C {
  const explicit: unknown = config.configurable?.checkpoint_id;
  if (explicit != null) {
    if (typeof explicit !== 'string' || explicit === '') throw new ToolResultProtectionError('incompatible');
    return config;
  }
  const map: unknown = config.configurable?.checkpoint_map;
  if (map == null) return config;
  if (typeof map !== 'object' || Array.isArray(map)) throw new ToolResultProtectionError('unsupported');
  const namespace = config.configurable?.checkpoint_ns ?? '';
  if (typeof namespace !== 'string') throw new ToolResultProtectionError('incompatible');
  const snapshot = snapshotToolCheckpointRecord(map, [namespace]);
  const id: unknown = Object.getOwnPropertyDescriptor(snapshot, namespace)?.value;
  if (id == null) return config;
  if (typeof id !== 'string' || id === '') throw new ToolResultProtectionError('incompatible');
  return { ...config, configurable: { ...config.configurable, checkpoint_id: id } };
}

function toolCheckpointSource(config: RunnableConfig, saver: unknown): BaseCheckpointSaver {
  let source: unknown = config.configurable?.__pregel_checkpointer ?? saver;
  while (source instanceof ToolCheckpointSaver) source = source.source;
  if (!(source instanceof BaseCheckpointSaver)) throw new ToolResultProtectionError('unsupported');
  return source;
}

export function admitToolResultCheckpoints<C extends RunnableConfig>(config: C, saver: unknown, admit: ToolCheckpointAdmission): C {
  const thread: unknown = config.configurable?.thread_id;
  if (typeof thread !== 'string' || thread === '') throw new ToolResultProtectionError('incompatible');
  return { ...config, configurable: { ...config.configurable, __pregel_checkpointer: new ToolCheckpointSaver(toolCheckpointSource(config, saver), thread, new Map(), admit) } };
}

type AdmittedToolCheckpoint = {
  config?: RunnableConfig;
  values?: object;
  tasks?: readonly { state?: AdmittedToolCheckpoint | RunnableConfig }[];
};

/** Exact saver routing preserves native approval semantics; descendant map entries imply time travel. */
export function bindToolResultCheckpoint<C extends RunnableConfig>(config: C, snapshot: AdmittedToolCheckpoint, saver?: unknown, admit?: ToolCheckpointAdmission): C {
  const thread: unknown = config.configurable?.thread_id;
  if (typeof thread !== 'string' || thread === '') throw new ToolResultProtectionError('incompatible');
  const map: unknown = config.configurable?.checkpoint_map;
  if (map != null && (typeof map !== 'object' || Array.isArray(map))) throw new ToolResultProtectionError('unsupported');
  const requested = map == null ? {} : snapshotToolCheckpointRecord(map, []);
  const cursors = new Map<string, string>();
  const active = new WeakSet<object>();
  const visit = (state: AdmittedToolCheckpoint, parent?: string): void => {
    if (active.has(state)) throw new ToolResultProtectionError('unsupported');
    active.add(state);
    const source = state.config?.configurable;
    const namespace: unknown = source?.checkpoint_ns ?? (parent == null ? config.configurable?.checkpoint_ns ?? '' : undefined);
    if (typeof namespace !== 'string' || (parent == null ? namespace !== '' : namespace === parent || !namespace.startsWith(parent === '' ? '' : parent + '|')) || (source?.thread_id != null && source.thread_id !== thread)) throw new ToolResultProtectionError('unsupported');
    const id: unknown = source?.checkpoint_id ?? source?.thread_ts;
    if (id != null && (typeof id !== 'string' || id === '')) throw new ToolResultProtectionError('incompatible');
    if (id == null && (Object.keys(state.values ?? {}).length > 0 || (state.tasks?.length ?? 0) > 0)) throw new ToolResultProtectionError('incompatible');
    const explicit: unknown = Object.getOwnPropertyDescriptor(requested, namespace)?.value;
    if (explicit != null && (typeof explicit !== 'string' || explicit !== id)) throw new ToolResultProtectionError('unsupported');
    if (parent == null) {
      const rootId = getToolResultCheckpointLookup(config).configurable?.checkpoint_id;
      if (rootId != null && rootId !== id) throw new ToolResultProtectionError('incompatible');
    }
    const cursor = id ?? nanoid();
    if (cursors.has(namespace)) throw new ToolResultProtectionError('unsupported');
    cursors.set(namespace, cursor);
    for (const task of state.tasks ?? []) {
      if (task.state == null) continue;
      if (!Object.hasOwn(task.state, 'values') && !Object.hasOwn(task.state, 'tasks')) throw new ToolResultProtectionError('unsupported');
      visit(task.state as AdmittedToolCheckpoint, namespace);
    }
    active.delete(state);
  };
  visit(snapshot);
  for (const namespace of Object.keys(requested)) if (!cursors.has(namespace)) throw new ToolResultProtectionError('unsupported');
  return { ...config, configurable: { ...config.configurable, checkpoint_ns: '', checkpoint_map: { '': cursors.get('')! }, __pregel_checkpointer: new ToolCheckpointSaver(toolCheckpointSource(config, saver), thread, cursors, admit) } };
}

export function snapshotToolCheckpointArray<T>(value: readonly T[]): T[] {
  return snapshotDataArray(value, Number.MAX_SAFE_INTEGER, true);
}

export function snapshotToolHistoryMessages(messages: readonly BaseMessage[]): BaseMessage[] {
  return snapshotDataArray(messages, Number.MAX_SAFE_INTEGER, true);
}

function historyMessageType(message: BaseMessage): string {
  const type = ownMessageValue(message, 'type');
  if (typeof type !== 'string') throw new ToolResultProtectionError('unsupported');
  return type;
}

function historyIdentity(message: BaseMessage): { id?: string; tool_call_id: string; name?: string } {
  const id = ownMessageValue(message, 'id');
  const tool_call_id = ownMessageValue(message, 'tool_call_id');
  const name = ownMessageValue(message, 'name');
  if ((id != null && typeof id !== 'string') || typeof tool_call_id !== 'string' || (name != null && typeof name !== 'string')) throw new ToolResultProtectionError('unsupported');
  return { id: id ?? undefined, tool_call_id, name: name ?? undefined };
}

/** Receipt retention uses stable message/call identities, never mutable tool names. */
export function retainToolMessageReleases(messages: readonly BaseMessage[], releases: readonly ToolMessageRelease[]): ToolMessageRelease[] {
  const identities = new Set<string>();
  for (const message of snapshotToolHistoryMessages(messages)) {
    if (historyMessageType(message) !== 'tool') continue;
    const identity = historyIdentity(message);
    identities.add(JSON.stringify([identity.tool_call_id, identity.id]));
  }
  return releases.filter((release) => identities.has(JSON.stringify([release.id, release.envelope.id])));
}

function snapshotToolMessageEnvelope(
  message: ToolMessageEnvelope
): ToolMessageEnvelope {
  const values = Object.fromEntries(messageFields.map((key) => [key, ownMessageValue(message, key)])) as ToolMessageEnvelope;
  for (const key of ['content', 'tool_call_id', 'status', 'additional_kwargs', 'response_metadata']) {
    if (!Object.hasOwn(message, key)) throw new ToolResultProtectionError('unsupported');
  }
  plainObject(values.additional_kwargs, [
    '_refKey',
    '_refScope',
    '_unresolvedRefs',
  ]);
  plainObject(values.response_metadata, []);
  const unresolved = values.additional_kwargs._unresolvedRefs;
  const additional_kwargs = {
    ...values.additional_kwargs,
    ...(unresolved != null
      ? {
        _unresolvedRefs: snapshotDataArray(
            unresolved as readonly string[],
            128,
            true
        ),
      }
      : {}),
  };
  validateReplayReferenceMetadata(additional_kwargs);
  return {
    content: values.content,
    artifact: values.artifact,
    name: values.name,
    id: values.id,
    tool_call_id: values.tool_call_id,
    status: values.status,
    additional_kwargs,
    response_metadata: {},
  };
}

function matchesToolMessageEnvelope(
  current: ToolMessageEnvelope,
  envelope: ToolMessageEnvelope
): boolean {
  const metadata = envelope.additional_kwargs;
  const currentMetadata = current.additional_kwargs;
  const unresolved = metadata._unresolvedRefs as string[] | undefined;
  const currentUnresolved = currentMetadata._unresolvedRefs as
    | string[]
    | undefined;
  return (
    current.content === envelope.content &&
    current.artifact === envelope.artifact &&
    current.id === envelope.id &&
    current.name === envelope.name &&
    current.tool_call_id === envelope.tool_call_id &&
    current.status === envelope.status &&
    Object.keys(currentMetadata).length === Object.keys(metadata).length &&
    currentMetadata._refKey === metadata._refKey &&
    currentMetadata._refScope === metadata._refScope &&
    (currentUnresolved === unresolved ||
      (currentUnresolved != null &&
        unresolved != null &&
        currentUnresolved.length === unresolved.length &&
        currentUnresolved.every((ref, index) => ref === unresolved[index])))
  );
}

/** SDK projections may change content, never bless callback-mutated envelopes. */
export function replaceReleasedToolMessage(
  policy: ToolResultProtection,
  name: string,
  id: string,
  message: ToolMessage,
  content: ToolMessage['content'],
  referenceContent?: string
): ToolMessage {
  const entry = approved.get(message);
  const envelope = snapshotToolMessageEnvelope(message);
  if (
    entry?.policy !== policy ||
    entry.name !== name ||
    entry.id !== id ||
    entry.envelope == null ||
    !matchesToolMessageEnvelope(envelope, entry.envelope)
  )
    throw new ToolResultProtectionError('unsupported');
  return markReleasedToolMessage(
    policy,
    name,
    id,
    new ToolMessage({
      name: envelope.name,
      id: envelope.id,
      tool_call_id: envelope.tool_call_id,
      status: envelope.status,
      content,
      additional_kwargs: envelope.additional_kwargs,
    }),
    referenceContent
  );
}

/** Written only from live release provenance, never inferred from checkpoint ownership. */
export function getToolMessageRelease(policy: ToolResultProtection | undefined, name: string, id: string, message: ToolMessage): ToolMessageRelease | undefined {
  if (!requiresToolResultProtection(policy, name)) return undefined;
  const entry = approved.get(message);
  const envelope = snapshotToolMessageEnvelope(message);
  if (entry?.policy !== policy || entry.name !== name || entry.id !== id || entry.envelope == null || !matchesToolMessageEnvelope(envelope, entry.envelope)) throw new ToolResultProtectionError('unsupported');
  const { content: _content, ...identity } = envelope;
  return { version: 1, name, id, envelope: identity };
}

function validateToolReleaseEnvelope(envelope: ToolMessageRelease['envelope']): void {
  plainObject(envelope, ['artifact', 'name', 'id', 'tool_call_id', 'status', 'additional_kwargs', 'response_metadata'], true);
  for (const field of ['tool_call_id', 'status', 'additional_kwargs', 'response_metadata']) {
    if (!Object.hasOwn(envelope, field)) throw new ToolResultProtectionError('incompatible');
  }
  plainObject(envelope.additional_kwargs, ['_refKey', '_refScope', '_unresolvedRefs'], true);
  plainObject(envelope.response_metadata, [], true);
}

function validateToolMessageRelease(release: ToolMessageRelease, name: string, id: string, envelope: ToolMessageEnvelope): void {
  plainObject(release, ['version', 'name', 'id', 'envelope'], true);
  for (const field of ['version', 'name', 'id', 'envelope']) if (!Object.hasOwn(release, field)) throw new ToolResultProtectionError('incompatible');
  const version: number = release.version;
  if (version !== 1 || release.name !== name || release.id !== id) throw new ToolResultProtectionError('incompatible');
  validateToolReleaseEnvelope(release.envelope);
  const certified = snapshotToolMessageEnvelope({ ...release.envelope, content: envelope.content, additional_kwargs: { ...release.envelope.additional_kwargs }, response_metadata: {} });
  if (!matchesToolMessageEnvelope(envelope, certified)) throw new ToolResultProtectionError('unsupported');
}

export function getLiveToolMessageRelease(policy: ToolResultProtection, message: ToolMessage): ToolMessageRelease | undefined {
  const entry = approved.get(message);
  if (entry?.policy !== policy || entry.envelope == null) return undefined;
  return getToolMessageRelease(policy, entry.name, entry.id, message);
}

/** Only the SDK reducer may bind an ID generated while committing a live release. */
export function commitToolMessageRelease(policy: ToolResultProtection, message: ToolMessage, release: ToolMessageRelease, content: ToolMessage['content']): ToolMessageRelease {
  const envelope = snapshotToolMessageEnvelope(message);
  if (envelope.content !== content || (release.envelope.id != null && envelope.id !== release.envelope.id) || !matchesToolMessageEnvelope({ ...envelope, id: release.envelope.id }, { ...release.envelope, content })) throw new ToolResultProtectionError('unsupported');
  markReleasedToolMessage(policy, release.name, release.id, message);
  return getToolMessageRelease(policy, release.name, release.id, message)!;
}

export function snapshotToolMessageReleases(releases: readonly ToolMessageRelease[]): ToolMessageRelease[] {
  return snapshotDataArray(releases, Number.MAX_SAFE_INTEGER, true).map((release) => {
    plainObject(release, ['version', 'name', 'id', 'envelope'], true);
    for (const field of ['version', 'name', 'id', 'envelope']) if (!Object.hasOwn(release, field)) throw new ToolResultProtectionError('incompatible');
    const version: number = release.version;
    if (version !== 1 || typeof release.name !== 'string' || typeof release.id !== 'string') throw new ToolResultProtectionError('incompatible');
    validateToolReleaseEnvelope(release.envelope);
    const envelope = snapshotToolMessageEnvelope({ ...release.envelope, content: '', additional_kwargs: { ...release.envelope.additional_kwargs }, response_metadata: {} });
    validateToolMessageRelease(release, release.name, release.id, envelope);
    const { content: _content, ...identity } = envelope;
    return { version: 1, name: release.name, id: release.id, envelope: identity };
  });
}

/** A replay record's ownership cannot downgrade its embedded protected identity. */
export function validateToolReplayRelease(policy: ToolResultProtection | undefined, name: string, output: object, release?: ToolMessageRelease, id?: string): ToolMessageRelease | undefined {
  if (policy == null || release == null) return release;
  const certified = snapshotToolMessageReleases([release])[0];
  if (!requiresToolResultProtection(policy, certified.name)) throw new ToolResultProtectionError('incompatible');
  if (certified.name !== name || (id != null && certified.id !== id) || ownMessageValue(output, 'type') !== 'tool') throw new ToolResultProtectionError('unsupported');
  validateToolMessageRelease(certified, name, certified.id, snapshotToolMessageEnvelope(output as ToolMessage));
  return certified;
}

/** Enclosing proposals and receipts cannot be downgraded by a packet-local tool name. */
export function snapshotToolCallSources(groups: readonly (readonly BaseMessage[])[], releases: readonly ToolMessageRelease[] = [], enclosing?: ReadonlyMap<string, string>): Map<string, string> {
  const sources = new Map(enclosing);
  const bind = (id: string, name: string): void => {
    const previous = sources.get(id);
    if (previous != null && previous !== name) throw new ToolResultProtectionError('unsupported');
    sources.set(id, name);
  };
  for (const release of snapshotToolMessageReleases(releases)) bind(release.id, release.name);
  for (const group of groups) for (const message of snapshotToolHistoryMessages(group)) {
    if (historyMessageType(message) !== 'ai') continue;
    const calls = ownMessageValue(message, 'tool_calls');
    for (const call of snapshotToolCheckpointArray((calls ?? []) as readonly object[])) {
      const id = ownMessageValue(call, 'id'); const name = ownMessageValue(call, 'name');
      if ((id != null && typeof id !== 'string') || typeof name !== 'string') throw new ToolResultProtectionError('unsupported');
      if (id != null) bind(id, name);
    }
  }
  return sources;
}

/** Historical state is immutable: reject legacy or newly incompatible history before model reuse. */
export async function validateToolMessageHistory(policy: ToolResultProtection | undefined, messages: readonly BaseMessage[], releases: readonly ToolMessageRelease[], signal?: AbortSignal, sources?: ReadonlyMap<string, string>): Promise<BaseMessage[]> {
  if (policy == null) return [...messages];
  const history = snapshotToolHistoryMessages(messages);
  const byIdentity = new Map<string, ToolMessageRelease>();
  const byMessage = new Map<string, ToolMessageRelease>();
  for (const release of snapshotToolMessageReleases(releases)) {
    if (!requiresToolResultProtection(policy, release.name)) throw new ToolResultProtectionError('incompatible');
    const key = JSON.stringify([release.id, release.envelope.id]);
    const existing = byIdentity.get(key);
    if (existing != null && existing.name !== release.name) throw new ToolResultProtectionError('incompatible');
    if (!byIdentity.has(key)) byIdentity.set(key, release);
    if (release.envelope.id != null && !byMessage.has(release.envelope.id)) byMessage.set(release.envelope.id, release);
  }
  const calls = sources == null ? new Map<string, string>() : snapshotToolCallSources([history], releases, sources);
  for (let index = 0; index < history.length; index++) {
    const message = history[index];
    const type = historyMessageType(message);
    const messageId = ownMessageValue(message, 'id');
    if (messageId != null && typeof messageId !== 'string') throw new ToolResultProtectionError('unsupported');
    const knownRelease = messageId == null ? undefined : byMessage.get(messageId);
    if (knownRelease != null && requiresToolResultProtection(policy, knownRelease.name) && type !== 'tool') throw new ToolResultProtectionError('unsupported');
    if (type === 'ai') {
      const toolCalls = ownMessageValue(message, 'tool_calls');
      for (const call of snapshotDataArray((toolCalls ?? []) as readonly { id?: string; name: string }[], Number.MAX_SAFE_INTEGER, true)) {
        const id = ownMessageValue(call, 'id'); const name = ownMessageValue(call, 'name');
        if ((id != null && typeof id !== 'string') || typeof name !== 'string') throw new ToolResultProtectionError('unsupported');
        if (id != null) calls.set(id, name);
      }
      continue;
    }
    if (type !== 'tool') continue;
    const identity = historyIdentity(message);
    const callName = calls.get(identity.tool_call_id);
    const admission = approved.get(message);
    const certified = byIdentity.get(JSON.stringify([identity.tool_call_id, identity.id])) ?? (identity.id == null ? undefined : byMessage.get(identity.id));
    if (sources != null && callName == null && certified == null && admission == null) throw new ToolResultProtectionError('incompatible');
    const name = certified?.name ?? admission?.name ?? callName ?? identity.name;
    const selected = requiresToolResultProtection(policy, name ?? '') || requiresToolResultProtection(policy, callName ?? '') || requiresToolResultProtection(policy, identity.name ?? '');
    if (!selected) continue;
    if (name == null) throw new ToolResultProtectionError('incompatible');
    if ((callName != null && callName !== name) || (identity.name != null && identity.name !== name) || (certified != null && (certified.id !== identity.tool_call_id || certified.envelope.id !== identity.id))) throw new ToolResultProtectionError('unsupported');
    const toolMessage = message as ToolMessage;
    const live = getLiveToolMessageRelease(policy, toolMessage);
    const release = live ?? certified;
    if (release == null) throw new ToolResultProtectionError('incompatible');
    const canonical = await protectToolMessage(policy, name, identity.tool_call_id, toolMessage, signal, release);
    if (canonical.content !== ownMessageValue(message, 'content')) throw new ToolResultProtectionError('incompatible');
    history[index] = canonical;
  }
  return history;
}

export async function protectToolMessage(
  policy: ToolResultProtection | undefined,
  name: string,
  id: string,
  message: ToolMessage,
  signal?: AbortSignal,
  replayRelease?: ToolMessageRelease
): Promise<ToolMessage> {
  if (!requiresToolResultProtection(policy, name)) return message;
  const envelope = snapshotToolMessageEnvelope(message);
  const status: string | undefined = envelope.status;
  if (
    envelope.artifact != null ||
    envelope.tool_call_id !== id ||
    (envelope.name != null && envelope.name !== name) ||
    (status != null && status !== 'success' && status !== 'error')
  )
    throw new ToolResultProtectionError('unsupported');
  if (replayRelease != null) validateToolMessageRelease(replayRelease, name, id, envelope);
  if (
    approved.get(message) == null &&
    replayRelease == null &&
    (Object.keys(envelope.additional_kwargs).length > 0 ||
      (envelope.id != null && envelope.id !== id))
  )
    throw new ToolResultProtectionError('unsupported');
  const existing = approved.get(message);
  if (
    existing?.envelope != null &&
    !matchesToolMessageEnvelope(envelope, existing.envelope)
  )
    throw new ToolResultProtectionError('unsupported');
  if (
    existing?.policy === policy &&
    existing.name === name &&
    existing.id === id &&
    existing.text === message.content &&
    existing.status === (message.status ?? 'success')
  )
    return replaceReleasedToolMessage(policy, name, id, message, envelope.content, existing.referenceContent);
  const content = await protectToolText(
    policy,
    name,
    id,
    envelope.content,
    envelope.status === 'error' ? 'error' : 'success',
    signal
  );
  if (
    !matchesToolMessageEnvelope(snapshotToolMessageEnvelope(message), envelope)
  )
    throw new ToolResultProtectionError('unsupported');
  const safe = new ToolMessage({
    name: envelope.name ?? name,
    id: envelope.id,
    tool_call_id: id,
    status: envelope.status ?? 'success',
    content: content as string,
    additional_kwargs: envelope.additional_kwargs,
  });
  return markReleasedToolMessage(policy, name, id, safe);
}

/** Validate host-owned envelopes before even reading their routing IDs. */
export function validateToolExecuteResults(
  results: ToolExecuteResult[],
  maxResults: number
): void {
  const entries = snapshotDataArray(results, maxResults);
  for (const entry of entries) {
    plainObject(entry, [
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
): ToolOutputReferenceState {
  plainObject(state, ['entries', 'turnCounter', 'warnedNonStringTools'], true);
  if (
    !Object.hasOwn(state, 'entries') ||
    !Object.hasOwn(state, 'turnCounter') ||
    !Object.hasOwn(state, 'warnedNonStringTools') ||
    !Number.isSafeInteger(state.turnCounter) ||
    state.turnCounter < 0
  )
    throw new ToolResultProtectionError('incompatible');
  const entries = snapshotDataArray(
    state.entries,
    Number.MAX_SAFE_INTEGER,
    true
  );
  const warnedNonStringTools = snapshotDataArray(
    state.warnedNonStringTools,
    Number.MAX_SAFE_INTEGER,
    true
  );
  if (warnedNonStringTools.some((name) => typeof name !== 'string'))
    throw new ToolResultProtectionError('incompatible');
  const snapshot: ToolOutputReferenceState = {
    entries: [],
    turnCounter: state.turnCounter,
    warnedNonStringTools,
  };
  for (const entry of entries) {
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
      (requiresToolResultProtection(policy, source.toolName) !== source.protected)
    )
      throw new ToolResultProtectionError('incompatible');
    snapshot.entries.push({
      key: entry.key,
      value: entry.value,
      protection: {
        version: 1,
        toolName: source.toolName,
        toolCallId: source.toolCallId,
        protected: source.protected,
      },
    });
  }
  return snapshot;
}

export function needsToolReferenceInspection(
  policy: ToolResultProtection,
  state: ToolOutputReferenceState,
  registry?: ToolOutputReferenceRegistry,
  runId?: string
): boolean {
  const snapshot = validateToolReferenceSources(policy, state);
  return snapshot.entries.some(
    (entry) =>
      requiresToolResultProtection(policy, entry.protection!.toolName) &&
      registry?.isPolicyBound(runId, entry, policy) !== true
  );
}

export async function protectToolReferenceState(
  policy: ToolResultProtection | undefined,
  state: ToolOutputReferenceState | undefined,
  signal?: AbortSignal,
  liveRegistry?: ToolOutputReferenceRegistry,
  runId?: string
): Promise<ToolOutputReferenceState | undefined> {
  if (policy == null || state == null) return state;
  const snapshot = validateToolReferenceSources(policy, state);
  const entries = await Promise.all(
    snapshot.entries.map(async (entry) => ({
      ...entry,
      value:
        liveRegistry?.isPolicyBound(runId, entry, policy) === true
          ? entry.value
          : ((await protectToolText(
            policy,
              entry.protection!.toolName,
              entry.protection!.toolCallId,
              entry.value,
              'success',
              signal
          )) as string),
    }))
  );
  return { ...snapshot, entries };
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
    if (reason instanceof ProviderTextProtectionError)
      throw normalizeToolResultError(reason);
    if (
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
          const producerConfig = { ...effective, callbacks: [] };
          raw = await attempt.wait(
            AsyncLocalStorageProviderSingleton.runWithConfig(producerConfig, () =>
              invoke.call(tool, input, toolBodyCallbacks(manager), producerConfig)
            )
          );
        } catch (error) {
          attempt.check();
          if (error instanceof ProviderTextProtectionError)
            throw normalizeToolResultError(error);
          if (
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
        if (
          admittedTool.name !== toolName ||
          admittedTool.responseFormat !== responseFormat
        )
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
        return markReleasedToolMessage(policy, toolName, id, message);
      } catch (error) {
        attempt.check();
        if (error instanceof ProviderTextProtectionError)
          throw normalizeToolResultError(error);
        if (
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
