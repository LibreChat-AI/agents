import { z } from 'zod';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { ToolResultProtection } from '../toolResult';
import { withToolResultBoundary, ToolResultProtectionError } from '../toolResult';
import { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import { GraphEvents, Providers } from '@/common';
import { ToolNode } from '@/tools/ToolNode';
import { FakeChatModel } from '@/llm/fake';
import { Run } from '@/run';

const raw = 'native-envelope@example.invalid';
const policy = (): ToolResultProtection => ({ version: 1, toolNames: ['lookup'], timeoutMs: 10000, maxAttemptBytes: 65536, maxBufferedBytes: 262144, classify: () => 'prose', inspect: ({ content }) => ({ version: 1, ok: true, value: { content, replacements: 0, categories: [] } }) });
const selected = (native: boolean): object => native ? new ToolMessage({ name: 'lookup', tool_call_id: 'inner', content: raw }) : { type: 'tool', name: 'lookup', tool_call_id: 'inner', content: raw };
const safe = (value: unknown): void => { expect(String(value)).not.toContain(raw); expect(String(JSON.stringify(value))).not.toContain(raw); };
const input = () => ({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer', name: 'other', args: {} }] })] });

it.each((['text', 'tuple', 'record', 'blocks'] as const).flatMap((shape) => [false, true].flatMap((native) => (['before', 'producer', 'start-callback', 'control', 'default-off'] as const).map((mode) => [shape, native, mode] as const))))('gates formatter metadata on native %s output (native=%s/%s)', async (shape, native, mode) => {
  const metadata = { nested: { value: mode === 'before' ? selected(native) : 'Allowed metadata' } };
  let observations = 0, executions = 0; const observed: string[] = [], completions: string[] = [];
  const artifact = Buffer.from([1, 2, 3]);
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed control', schema: z.object({}), metadata, responseFormat: shape === 'tuple' ? 'content_and_artifact' : 'content', func: async () => {
    executions++; if (mode === 'producer' || mode === 'default-off') metadata.nested.value = selected(native);
    if (shape === 'tuple') return ['Allowed output', artifact];
    if (shape === 'record') return { count: 42 };
    if (shape === 'blocks') return [{ type: 'text', text: 'Allowed output' }];
    return 'Allowed output';
  } });
  const callback = BaseCallbackHandler.fromMethods({ handleToolStart: (): void => { if (mode === 'start-callback') metadata.nested.value = selected(native); }, handleToolEnd: (output): void => { observations++; observed.push(JSON.stringify(output)); }, handleCustomEvent: (event, data): void => { if (event === GraphEvents.ON_RUN_STEP_COMPLETED) completions.push(JSON.stringify(data)); } }); callback.awaitHandlers = true;
  const node = new ToolNode({ tools: [tool], trace: true, toolCallStepIds: new Map([['outer', 'step']]), ...(mode === 'default-off' ? {} : { toolResultProtection: policy() }) });
  const outcome = await node.invoke(input(), { callbacks: [callback] }).catch((error: Error) => error) as Error | { messages: ToolMessage[] };
  if (mode === 'default-off') { expect(observed.join('')).toContain(raw); expect(outcome).not.toBeInstanceOf(Error); return; }
  safe(observed); safe(completions); safe(outcome);
  if (mode === 'control') { expect(observations).toBe(1); expect(executions).toBe(1); expect(completions).toHaveLength(1); expect(outcome).not.toBeInstanceOf(Error); if (shape === 'tuple') expect((outcome as { messages: ToolMessage[] }).messages[0].artifact).toEqual(artifact); }
  else { expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(observations).toBe(0); expect(completions).toEqual([]); }
});

it.each(['content', 'metadata', 'additional_kwargs', 'response_metadata', 'serialization', 'extra', 'getter', 'name', 'control'] as const)('readmits the generated native envelope before compaction (%s)', async (field) => {
  let reads = 0; const refs = new ToolOutputReferenceRegistry(), completions: string[] = [], observed: string[] = [];
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed control', schema: z.object({}), func: async () => 'Allowed output' });
  const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (output): void => {
    observed.push(JSON.stringify(output)); const message = output as ToolMessage;
    if (field === 'content') message.content = [{ type: 'text', text: 'Allowed text', nested: selected(false) }];
    if (field === 'serialization') message.lc_kwargs.content = [{ type: 'text', text: 'Allowed text', nested: selected(false) }];
    if (field === 'name') message.name = 'lookup';
    if (field === 'getter') Object.defineProperty(message, 'content', { get: () => { reads++; throw new Error(raw); } });
    if (['metadata', 'additional_kwargs', 'response_metadata', 'extra'].includes(field)) Object.defineProperty(message, field, { value: { selected: selected(false) }, enumerable: true, configurable: true });
  }, handleCustomEvent: (event, data): void => { if (event === GraphEvents.ON_RUN_STEP_COMPLETED) completions.push(JSON.stringify(data)); } }); callback.awaitHandlers = true;
  const node = new ToolNode({ tools: [tool], trace: true, toolCallStepIds: new Map([['outer', 'step']]), toolOutputRegistry: refs, toolResultProtection: policy() });
  const outcome = await node.invoke(input(), { callbacks: [callback], configurable: { run_id: 'native-envelope' } }).catch((error: Error) => error);
  expect(reads).toBe(0); safe(observed); safe(completions); safe(outcome);
  if (field === 'control') { expect(outcome).not.toBeInstanceOf(Error); expect(completions).toHaveLength(1); expect(refs.size).toBe(1); }
  else { expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(completions).toEqual([]); expect(refs.size).toBe(0); }
});

it.each(['metadata', 'content', 'control'] as const)('detaches generated envelopes from retained native aliases (%s)', async (field) => {
  const metadata = { nested: { value: 'Allowed metadata' } }; let retained: ToolMessage | undefined;
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed control', schema: z.object({}), metadata, func: async () => 'Allowed output' });
  const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (output): void => { retained = output as ToolMessage; } }); callback.awaitHandlers = true;
  const node = new ToolNode({ tools: [tool], trace: true, toolResultProtection: policy() });
  const outcome = await node.invoke(input(), { callbacks: [callback] }) as { messages: ToolMessage[] };
  const before = JSON.stringify(outcome);
  if (field === 'metadata') { metadata.nested.value = raw; ((retained!.metadata as { nested: typeof metadata.nested }).nested).value = raw; }
  if (field === 'content') { retained!.content = raw; retained!.lc_kwargs.content = raw; }
  expect(JSON.stringify(outcome)).toBe(before); safe(outcome); expect(outcome.messages[0]).not.toBe(retained);
});

it.each(['producer-metadata', 'callback-content', 'control'] as const)('gates generated envelopes before real Run model reuse (%s)', async (mode) => {
  let modelCalls = 0; const completions: string[] = [], metadata: { nested: object | string } = { nested: 'Allowed metadata' };
  class Model extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { modelCalls++; yield* super._streamResponseChunks(...args); } }
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed control', schema: z.object({}), metadata, func: async () => { if (mode === 'producer-metadata') metadata.nested = selected(false); return 'Allowed output'; } });
  const run = await Run.create({ runId: 'generated-envelope-' + mode, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [tool] }, toolResultProtection: policy(), toolOutputReferences: { enabled: true }, customHandlers: { [GraphEvents.ON_RUN_STEP_COMPLETED]: { handle: (_event, data): void => { completions.push(JSON.stringify(data)); } } } });
  run.Graph!.overrideModel = new Model({ responses: ['', 'Allowed answer'], toolCalls: [{ id: 'outer', name: 'other', args: {} }] });
  const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (output): void => { if (mode === 'callback-content') (output as ToolMessage).content = [{ type: 'text', text: 'Allowed', nested: selected(false) }]; } }); callback.awaitHandlers = true;
  const outcome = await run.processStream({ messages: [new HumanMessage('Allowed input')] }, { version: 'v2', callbacks: [callback] }).catch((error: Error) => error);
  safe(outcome); safe(completions);
  if (mode === 'control') { expect(outcome).toBeUndefined(); expect(modelCalls).toBe(2); expect(completions).toHaveLength(1); }
  else { expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(modelCalls).toBe(1); expect(completions).toEqual([]); }
});

it.each(['getter', 'proxy', 'format-name', 'format-response'] as const)('rejects unsafe formatter fields without native result publication (%s)', async (mode) => {
  let reads = 0, observations = 0;
  const metadata = { nested: 'Allowed metadata' };
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed control', schema: z.object({}), metadata, func: async () => {
    if (mode === 'format-name') Object.defineProperty(tool, 'name', { value: 'lookup' });
    if (mode === 'format-response') tool.responseFormat = 'content_and_artifact';
    return 'Allowed output';
  } });
  if (mode === 'getter') Object.defineProperty(metadata, 'nested', { get: () => { reads++; throw new Error(raw); } });
  if (mode === 'proxy') tool.metadata = new Proxy(metadata, { getPrototypeOf: (): never => { reads++; throw new Error(raw); } });
  const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (): void => { observations++; } }); callback.awaitHandlers = true;
  const outcome = await new ToolNode({ tools: [tool], trace: true, toolResultProtection: policy() }).invoke(input(), { callbacks: [callback] }).catch((error: Error) => error);
  expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(observations).toBe(0); expect(reads).toBe(0); safe(outcome);
});

it('adopts a complete native envelope when no callbacks are configured', async () => {
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed control', schema: z.object({}), metadata: { custom: 'Allowed metadata' }, func: async () => 'Allowed output' });
  const invoke = tool.invoke, call = tool.call, metadata = tool.metadata;
  const node = new ToolNode({ tools: [tool], toolResultProtection: policy() });
  const outcome = await node.invoke(input()) as { messages: ToolMessage[] };
  expect(outcome.messages[0].metadata).toEqual(metadata); expect(outcome.messages[0].metadata).not.toBe(metadata);
  expect(tool.invoke).toBe(invoke); expect(tool.call).toBe(call); expect(tool.metadata).toBe(metadata);
});

it.each(['Allowed scalar', NaN, undefined, null, 42n])('preserves native direct-invoke scalar output (%s)', async (value) => {
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed control', schema: z.object({}), func: async () => value });
  const observed: unknown[] = [];
  const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (output): void => { observed.push(output); } }); callback.awaitHandlers = true;
  const result = await withToolResultBoundary(tool, policy(), '', {}).invoke({}, { callbacks: [callback] });
  expect(Object.is(result, value)).toBe(true); expect(observed).toHaveLength(1); expect(Object.is(observed[0], value)).toBe(true);
});
