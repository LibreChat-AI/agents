import { z } from 'zod';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { HumanMessage, AIMessageChunk } from '@langchain/core/messages';
import type { ProviderTextProtection } from '@/protection/providerText';
import type { ChatModel, EventHandler } from '@/types';
import { PreparedSubagentError } from '@/tools/preparedSubagents';
import { StreamLimitExceededError } from '@/llm/streamLimits';
import { CustomChatGoogleGenerativeAI } from '@/llm/google';
import { GraphEvents, Providers } from '@/common';
import { ChatModelStreamHandler } from '@/stream';
import { attemptInvoke } from '@/llm/invoke';
import { ChatOpenAI } from '@/llm/openai';
import { Run } from '@/run';

const raw = 'Contact a1.alice@example.invalid.';
const canonical = 'Contact [EMAIL_1].';
const policy = (inspect: ProviderTextProtection['inspect']): ProviderTextProtection => ({
  version: 1, timeoutMs: 10000, maxAttemptBytes: 65536, maxBufferedBytes: 262144,
  classify: () => 'prose', inspect,
});
const approve = (content: string) => ({ version: 1 as const, ok: true as const, value: { content, replacements: 1, categories: [{ category: 'EMAIL', count: 1 }] } });

afterEach(() => jest.restoreAllMocks());
async function verifyPublication(model: ChatModel, provider: Providers, protection: ProviderTextProtection): Promise<void> {
  for (const registered of [false, true]) {
    const events: string[] = [];
    const handlers: Record<string, EventHandler> = registered ? { [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler() } : {};
    const run = await Run.create({ runId: `codex-${provider}-${registered}`, graphConfig: { type: 'standard', llmConfig: { provider }, instructions: 'Allowed instructions.' }, providerTextProtection: protection, returnContent: true, skipCleanup: true, customHandlers: { ...handlers, [GraphEvents.ON_MESSAGE_DELTA]: { handle: (_event, data): void => { events.push(JSON.stringify(data)); } } } });
    run.Graph!.overrideModel = model;
    await run.processStream({ messages: [new HumanMessage('Allowed control')] }, { version: 'v2', configurable: { thread_id: `codex-${provider}-${registered}` } });
    expect(events.join('')).toContain(canonical);
    expect(events.join('')).not.toContain('a1.alice@example.invalid');
    expect(JSON.stringify(run.Graph!.getRunMessages())).toContain(canonical);
    expect(JSON.stringify(run.Graph!.getRunMessages())).not.toContain('a1.alice@example.invalid');
  }
}

it.each([false, true])('admits real Gemini prose and controls (native=%s)', async (native) => {
  const responses = [{ candidates: [{ content: { role: 'model', parts: [{ text: raw }] }, index: 0, finishReason: 'STOP', safetyRatings: [{ category: 'HARM_CATEGORY_HARASSMENT', probability: 'NEGLIGIBLE', blocked: false }] }], usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 } }];
  jest.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response(native ? JSON.stringify(responses[0]) : responses.map((response) => `data: ${JSON.stringify(response)}\n\n`).join(''), { headers: { 'content-type': native ? 'application/json' : 'text/event-stream' } }));
  const model = new CustomChatGoogleGenerativeAI({ model: 'gemini-2.5-flash', apiKey: 'synthetic-test-key', streaming: !native, _lc_stream_delay: 25 });
  model.disableStreaming = native;
  let inspected = false;
  const observed: string[] = [];
  const observer = BaseCallbackHandler.fromMethods({ handleLLMEnd: (output): void => { observed.push(JSON.stringify(output)); } });
  observer.awaitHandlers = true;
  const result = await attemptInvoke({ model, messages: [new HumanMessage('Allowed control')], provider: Providers.GOOGLE, onChunk: () => {}, providerTextProtection: policy(({ content }) => { expect(content).toBe(raw); inspected = true; return approve(canonical); }) }, { callbacks: [observer] });
  expect(inspected).toBe(true);
  expect(result.messages?.[0].content).toBe(canonical);
  expect(result.messages?.[0].additional_kwargs.__gemini_function_call_thought_signatures__).toEqual({});
  expect(result.messages?.[0].response_metadata).toMatchObject({ finishReason: 'STOP', safetyRatings: responses[0].candidates[0].safetyRatings });
  expect(observed.join('')).toContain(canonical);
  expect(observed.join('')).not.toContain('a1.alice@example.invalid');
  await verifyPublication(model, Providers.GOOGLE, policy(({ content }) => { expect(content).toBe(raw); return approve(canonical); }));
});

const response = {
  id: 'resp-control', created_at: 1, object: 'response', model: 'synthetic-model', status: 'completed', error: null,
  output: [{ id: 'msg-control', type: 'message', role: 'assistant', status: 'completed', content: [{ type: 'output_text', text: raw, annotations: [] }] }],
  output_text: raw, incomplete_details: null, instructions: null, metadata: {}, tools: [], tool_choice: 'auto', text: { format: { type: 'text' } },
  parallel_tool_calls: true, temperature: 1, top_p: 1, truncation: 'disabled', store: false,
  usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15, input_tokens_details: { cached_tokens: 0 }, output_tokens_details: { reasoning_tokens: 0 } },
};

it.each([false, true])('releases real Responses completion without raw aliases (native=%s)', async (native) => {
  const frames = [
    { type: 'response.created', response: { ...response, status: 'in_progress', output: [] }, sequence_number: 0 },
    { type: 'response.output_item.added', output_index: 0, item: { ...response.output[0], content: [], status: 'in_progress' }, sequence_number: 1 },
    { type: 'response.output_text.delta', item_id: 'msg-control', output_index: 0, content_index: 0, delta: raw.slice(0, 13), sequence_number: 2 },
    { type: 'response.output_text.delta', item_id: 'msg-control', output_index: 0, content_index: 0, delta: raw.slice(13), sequence_number: 3 },
    { type: 'response.completed', response, sequence_number: 4 },
  ];
  const observed: string[] = [];
  let inspected = false;
  const model = new ChatOpenAI({ model: 'synthetic-model', apiKey: 'synthetic-test-key', useResponsesApi: true, streaming: !native, _lc_stream_delay: 25, configuration: { apiKey: 'synthetic-test-key', fetch: async () => new Response(native ? JSON.stringify(response) : frames.map((frame) => `event: ${frame.type}\ndata: ${JSON.stringify(frame)}\n\n`).join(''), { headers: { 'content-type': native ? 'application/json' : 'text/event-stream' } }) } });
  model.disableStreaming = native;
  const observer = BaseCallbackHandler.fromMethods({ handleLLMNewToken: (_token, _indices, _run, _parent, _tags, fields): void => { observed.push(JSON.stringify(fields)); }, handleLLMEnd: (output): void => { observed.push(JSON.stringify(output)); } });
  observer.awaitHandlers = true;
  const result = await attemptInvoke({ model, messages: [new HumanMessage('Allowed control')], provider: Providers.OPENAI, onChunk: (chunk) => { observed.push(JSON.stringify(chunk)); }, providerTextProtection: policy(({ content }) => { expect(content).toBe(raw); inspected = true; return approve(canonical); }) }, { callbacks: [observer] });
  expect(inspected).toBe(true);
  expect(JSON.stringify(result)).toContain(canonical);
  expect(JSON.stringify(result)).not.toContain('a1.alice@example.invalid');
  expect(result.messages?.[0].response_metadata).toMatchObject({ status: 'completed', id: 'resp-control' });
  expect((result.messages?.[0] as AIMessageChunk).response_metadata.output).toEqual([{ ...response.output[0], content: [{ ...response.output[0].content[0], text: canonical }] }]);
  expect(observed.join('')).not.toContain('a1.alice@example.invalid');
  expect(observed.join('')).toContain(canonical);
  await verifyPublication(model, Providers.OPENAI, policy(({ content }) => { expect(content).toBe(raw); return approve(canonical); }));
});

it.each([
  new StreamLimitExceededError({ kind: 'tool_call_args', limit: 10, observed: 11 }),
  new PreparedSubagentError('Delegated work already started'),
])('preserves safety trip identity through a protected invocation (%#)', async (reason) => {
  const controller = new AbortController();
  const model = new FakeListChatModel({ responses: ['Allowed control'] });
  model._streamResponseChunks = () => { controller.abort(reason); throw new Error('Provider aborted'); };
  await expect(attemptInvoke({ model, messages: [], provider: Providers.OPENAI, providerTextProtection: policy(({ content }) => approve(content)), onChunk: () => {} }, { signal: controller.signal })).rejects.toBe(reason);
});

it.each([false, true])('preserves Gemini function signatures and arguments through real conversion (native=%s)', async (native) => {
  const answer = { candidates: [{ content: { role: 'model', parts: [{ text: raw }, { functionCall: { id: 'gemini-call', name: 'lookup', args: { count: 42 } }, thoughtSignature: 'signature-control' }] }, index: 0, finishReason: 'STOP' }], usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 } };
  jest.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response(native ? JSON.stringify(answer) : `data: ${JSON.stringify(answer)}\n\n`, { headers: { 'content-type': native ? 'application/json' : 'text/event-stream' } }));
  const model = new CustomChatGoogleGenerativeAI({ model: 'gemini-3-flash', apiKey: 'synthetic-test-key', streaming: !native });
  model.disableStreaming = native;
  const result = await attemptInvoke({ model, messages: [new HumanMessage('Allowed control')], provider: Providers.GOOGLE, onChunk: () => {}, providerTextProtection: policy(({ content }) => { expect(content).toBe(raw); return approve(canonical); }) });
  const message = result.messages?.[0] as AIMessageChunk;
  expect(message.tool_calls).toEqual([{ id: 'gemini-call', name: 'lookup', args: { count: 42 }, type: 'tool_call' }]);
  expect(message.additional_kwargs.__gemini_function_call_thought_signatures__).toEqual({ 'gemini-call': 'signature-control' });
  expect(JSON.stringify(message)).toContain(canonical);
  expect(JSON.stringify(message)).toContain('signature-control');
  expect(JSON.stringify(message)).not.toContain('a1.alice@example.invalid');
});

it.each([false, true])('preserves Responses reasoning/signatures and function control fields (native=%s)', async (native) => {
  const reasoning = { id: 'reason-control', type: 'reasoning', status: 'completed', encrypted_content: 'signature-control', summary: [{ type: 'summary_text', text: 'Reasoning control' }] };
  const call = { id: 'function-control', type: 'function_call', status: 'completed', call_id: 'call-control', name: 'lookup', arguments: '{"count":42}' };
  const tool = { type: 'function', name: 'lookup', description: 'Allowed control.', strict: true, parameters: { type: 'object', properties: { count: { type: 'number' } }, required: ['count'], additionalProperties: false } };
  const completed = { ...response, output: [reasoning, response.output[0], call], tools: [tool] };
  const frames = [
    { type: 'response.created', response: { ...completed, output: [], status: 'in_progress' }, sequence_number: 0 },
    { type: 'response.output_item.added', output_index: 0, item: { ...reasoning, summary: [] }, sequence_number: 1 },
    { type: 'response.reasoning_summary_text.delta', item_id: 'reason-control', output_index: 0, summary_index: 0, delta: 'Reasoning control', sequence_number: 2 },
    { type: 'response.output_item.done', output_index: 0, item: reasoning, sequence_number: 3 },
    { type: 'response.output_item.added', output_index: 1, item: { ...response.output[0], content: [], status: 'in_progress' }, sequence_number: 4 },
    { type: 'response.output_text.delta', item_id: 'msg-control', output_index: 1, content_index: 0, delta: raw, sequence_number: 5 },
    { type: 'response.output_item.added', output_index: 2, item: { ...call, arguments: '', status: 'in_progress' }, sequence_number: 6 },
    { type: 'response.function_call_arguments.delta', item_id: 'function-control', output_index: 2, delta: call.arguments, sequence_number: 7 },
    { type: 'response.completed', response: completed, sequence_number: 8 },
  ];
  const model = new ChatOpenAI({ model: 'synthetic-model', apiKey: 'synthetic-test-key', useResponsesApi: true, streaming: !native, configuration: { apiKey: 'synthetic-test-key', fetch: async () => new Response(native ? JSON.stringify(completed) : frames.map((frame) => `event: ${frame.type}\ndata: ${JSON.stringify(frame)}\n\n`).join(''), { headers: { 'content-type': native ? 'application/json' : 'text/event-stream' } }) } });
  model.disableStreaming = native;
  const result = await attemptInvoke({ model, messages: [], provider: Providers.OPENAI, onChunk: () => {}, providerTextProtection: policy(({ content }) => { expect(content).toBe(raw); return approve(canonical); }) });
  const message = result.messages?.[0] as AIMessageChunk;
  expect(message.response_metadata.output).toEqual([reasoning, { ...response.output[0], content: [{ ...response.output[0].content[0], text: canonical }] }, call]);
  if (!native) expect(message.response_metadata.tools).toEqual([tool]);
  expect(message.tool_calls).toEqual([{ id: 'call-control', name: 'lookup', args: { count: 42 }, type: 'tool_call' }]);
  expect(JSON.stringify(message)).toContain('signature-control');
  expect(JSON.stringify(message)).not.toContain('a1.alice@example.invalid');
});

it.each([
  { label: 'unknown alias', change: { raw_output: raw } },
  { label: 'mismatched output alias', change: { output_text: 'Unchecked alternate value' } },
  { label: 'structured format', change: { text: { format: { type: 'json_schema', schema: { type: 'object' } } } } },
  { label: 'unknown metadata', change: { metadata: { output: raw } } },
  { label: 'annotations', change: { output: [{ ...response.output[0], content: [{ ...response.output[0].content[0], annotations: [{ type: 'url_citation', title: raw, url: 'https://example.invalid' }] }] }] } },
])('rejects uncertified Responses $label without releasing aliases', async ({ change }) => {
  const frames = [
    { type: 'response.output_text.delta', item_id: 'msg-control', output_index: 0, content_index: 0, delta: raw },
    { type: 'response.completed', response: { ...response, ...change } },
  ];
  const model = new ChatOpenAI({ model: 'synthetic-model', apiKey: 'synthetic-test-key', useResponsesApi: true, streaming: true, configuration: { apiKey: 'synthetic-test-key', fetch: async () => new Response(frames.map((frame) => `event: ${frame.type}\ndata: ${JSON.stringify(frame)}\n\n`).join(''), { headers: { 'content-type': 'text/event-stream' } }) } });
  let inspected = false;
  await expect(attemptInvoke({ model, messages: [], provider: Providers.OPENAI, onChunk: () => {}, providerTextProtection: policy(() => { inspected = true; return approve(canonical); }) })).rejects.toMatchObject({ code: 'unsupported' });
  expect(inspected).toBe(false);
});

it('bounds retained Responses completion metadata before canonical release', async () => {
  const model = new ChatOpenAI({ model: 'synthetic-model', apiKey: 'synthetic-test-key', useResponsesApi: true, streaming: false, configuration: { apiKey: 'synthetic-test-key', fetch: async () => new Response(JSON.stringify(response), { headers: { 'content-type': 'application/json' } }) } });
  model.disableStreaming = true;
  const protection = policy(() => approve(canonical));
  Object.defineProperty(protection, 'maxAttemptBytes', { value: 1024 });
  await expect(attemptInvoke({ model, messages: [], provider: Providers.OPENAI, onChunk: () => {}, providerTextProtection: protection })).rejects.toMatchObject({ code: 'overflow' });
});

it.each([false, true])('reuses canonical Responses prose and exact function IDs after ToolNode (native=%s)', async (native) => {
  const call = { id: 'function-control', type: 'function_call', status: 'completed', call_id: 'call-control', name: 'lookup', arguments: '{"count":42}' };
  const completed = { ...response, output: [...response.output, call] };
  const final = { ...response, id: 'resp-final', output: [{ ...response.output[0], id: 'msg-final', content: [{ type: 'output_text', text: 'Final control', annotations: [] }] }], output_text: 'Final control' };
  const requests: string[] = [];
  const model = new ChatOpenAI({ model: 'synthetic-model', apiKey: 'synthetic-test-key', useResponsesApi: true, streaming: !native, configuration: { apiKey: 'synthetic-test-key', fetch: async (_url, init) => {
    requests.push(typeof init?.body === 'string' ? init.body : '');
    const terminal = requests.length === 1 ? completed : final;
    const frames = [
      { type: 'response.created', response: { ...terminal, output: [], status: 'in_progress' } },
      { type: 'response.output_item.added', output_index: 0, item: { ...terminal.output[0], content: [], status: 'in_progress' } },
      { type: 'response.output_text.delta', item_id: terminal.output[0].id, output_index: 0, content_index: 0, delta: requests.length === 1 ? raw : 'Final control' },
      ...(requests.length === 1 ? [{ type: 'response.output_item.added', output_index: 1, item: { ...call, arguments: '', status: 'in_progress' } }, { type: 'response.function_call_arguments.delta', output_index: 1, item_id: call.id, delta: call.arguments }] : []),
      { type: 'response.completed', response: terminal },
    ];
    return new Response(native ? JSON.stringify(terminal) : frames.map((frame) => `event: ${frame.type}\ndata: ${JSON.stringify(frame)}\n\n`).join(''), { headers: { 'content-type': native ? 'application/json' : 'text/event-stream' } });
  } } });
  model.disableStreaming = native;
  let executed = 0;
  const lookup = new DynamicStructuredTool({ name: 'lookup', description: 'Allowed control.', schema: z.object({ count: z.number() }), func: async ({ count }) => { expect(count).toBe(42); executed++; return 'Tool control'; } });
  const run = await Run.create({ runId: `responses-reuse-${native}`, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [lookup], instructions: 'Allowed instructions.' }, providerTextProtection: policy(({ content }) => approve(content === raw ? canonical : content)), returnContent: true, skipCleanup: true });
  run.Graph!.overrideModel = model;
  await run.processStream({ messages: [new HumanMessage('Allowed control')] }, { version: 'v2', configurable: { thread_id: `responses-reuse-${native}` } });
  expect(requests).toHaveLength(2);
  expect(executed).toBe(1);
  expect(requests[1]).toContain(canonical);
  expect(requests[1]).toContain('call-control');
  expect(requests[1]).toContain('Tool control');
  expect(requests[1]).not.toContain('a1.alice@example.invalid');
  expect(JSON.stringify(run.Graph!.getRunMessages())).toContain('Final control');
});
