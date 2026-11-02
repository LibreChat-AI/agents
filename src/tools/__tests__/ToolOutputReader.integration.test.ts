import { z } from 'zod';
import { writeFileSync } from 'node:fs';
import { tool } from '@langchain/core/tools';
import { HumanMessage } from '@langchain/core/messages';
import { context as otelContext } from '@opentelemetry/api';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { AgentInputs, EventHandler, ToolExecuteBatchRequest, LangfuseConfig } from '@/types';
import { resolveToolOutputTracingConfigForSpan } from '@/langfuseRuntimeScope';
import { ensureOpenTelemetryContextManager } from '@/instrumentation';
import { shouldRedactTool } from '@/langfuseToolOutputTracing';
import { TOOL_OUTPUT_READER } from '../ToolOutputReader';
import { GraphEvents, Providers } from '@/common';
import { StandardGraph } from '@/graphs/Graph';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

type Route = 'chat' | 'responses';
type Mode = 'local' | 'event';
type TextPart = { type: string; text?: string };
type WireMessage = { role?: string; type?: string; content?: string | TextPart[]; output?: string | TextPart[] };
type WireTool = { name?: string; parameters?: { properties?: object }; function?: { name: string; parameters: { properties?: object } } };
type WireBody = { stream: boolean; messages?: WireMessage[]; input?: WireMessage[]; tools?: WireTool[] };
type ToolInvocation = { id?: string; name: string; args: ToolCall['args'] };

function modelResponse(route: Route, invocation?: ToolInvocation): string {
  const args = invocation ? JSON.stringify(invocation.args) : '';
  if (route === 'chat') {
    const base = {
      id: `chat_${invocation?.id ?? (invocation ? 'synthetic' : 'final')}`,
      object: 'chat.completion.chunk',
      created: 1,
      model: 'gpt-5.6-sol',
    };
    const delta = invocation
      ? {
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            index: 0,
            id: invocation.id ?? 'call_remote',
            type: 'function',
            function: { name: invocation.name, arguments: args },
          },
        ],
      }
      : { role: 'assistant', content: 'OK' };
    const events = [
      { ...base, choices: [{ index: 0, delta, finish_reason: null }] },
      {
        ...base,
        choices: [{ index: 0, delta: {}, finish_reason: invocation ? 'tool_calls' : 'stop' }],
      },
      { ...base, choices: [], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } },
    ];
    return (
      events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('') + 'data: [DONE]\n\n'
    );
  }
  const item = invocation
    ? {
      id: invocation.id != null ? `fc_${invocation.id}` : 'fc_synthetic',
      type: 'function_call',
      status: 'completed',
      call_id: invocation.id ?? 'call_remote',
      name: invocation.name,
      arguments: args,
    }
    : {
      id: 'msg_synthetic',
      type: 'message',
      status: 'completed',
      role: 'assistant',
      content: [{ type: 'output_text', text: 'OK', annotations: [] }],
    };
  const response = {
    id: `resp_${invocation?.id ?? (invocation ? 'tool' : 'final')}`,
    object: 'response',
    created_at: 1,
    status: 'completed',
    model: 'gpt-5.6-sol',
    output: [item],
    usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
  };
  const events = [
    { type: 'response.created', response: { ...response, output: [], status: 'in_progress' } },
    {
      type: 'response.output_item.added',
      output_index: 0,
      item: {
        ...item,
        status: 'in_progress',
        ...(invocation ? { arguments: '' } : { content: [] }),
      },
    },
    invocation
      ? {
        type: 'response.function_call_arguments.delta',
        output_index: 0,
        item_id: item.id,
        delta: args,
      }
      : {
        type: 'response.output_text.delta',
        output_index: 0,
        item_id: item.id,
        content_index: 0,
        delta: 'OK',
      },
    { type: 'response.completed', response },
  ];
  return events
    .map(
      (event, sequence_number) =>
        `event: ${event.type}\ndata: ${JSON.stringify({ ...event, sequence_number })}\n\n`,
    )
    .join('');
}

function textOf(value?: string | TextPart[]): string {
  if (typeof value === 'string') return value;
  return value?.map((part) => part.text ?? '').join('') ?? '';
}

function toolTexts(body: WireBody, route: Route): string[] {
  return route === 'chat'
    ? (body.messages ?? [])
      .filter((message) => message.role === 'tool')
      .map((message) => textOf(message.content))
    : (body.input ?? [])
      .filter((message) => message.type === 'function_call_output')
      .map((message) => textOf(message.output));
}

const sentinel = 'RETAINED_COMMAND_SENTINEL';
const source = 'x'.repeat(20000) + sentinel + 'y'.repeat(20000);

async function capture(route: Route, mode: Mode, readTool?: boolean, finalOnly = false) {
  const requests: Array<{ raw: string; body: WireBody }> = [];
  let producerCalls = 0;
  const readerPolicies: boolean[] = [];
  const hooks = new HookRegistry();
  hooks.register('PreToolUse', { hooks: [async (input) => {
    if (input.toolName === TOOL_OUTPUT_READER) {
      const policy = resolveToolOutputTracingConfigForSpan(otelContext.active());
      readerPolicies.push(policy != null && shouldRedactTool(TOOL_OUTPUT_READER, policy));
    }
    return { decision: 'allow' };
  }] });
  const transport: typeof fetch = async (url, init) => {
    expect(String(url)).toBe(`http://synthetic.invalid/v1/${route === 'chat' ? 'chat/completions' : 'responses'}`);
    const raw = String(init?.body);
    const body = JSON.parse(raw) as WireBody;
    requests.push({ raw, body });
    expect(requests.length).toBeLessThanOrEqual(3);
    let invocation: ToolInvocation | undefined;
    if (!finalOnly && requests.length === 1) {
      invocation = { id: 'produce', name: 'synthetic_command', args: {} };
    }
    if (!finalOnly && requests.length === 2) {
      expect(toolTexts(body, route)[0]).not.toContain(sentinel);
      invocation = { id: 'read', name: TOOL_OUTPUT_READER, args: {
        reference: 'tool0turn0', start: 20000, maxChars: sentinel.length,
      } };
    }
    return new Response(modelResponse(route, invocation), { headers: { 'content-type': 'text/event-stream' } });
  };
  const producer = tool(async () => { producerCalls++; return source; }, {
    name: 'synthetic_command', description: 'Synthetic command output', schema: z.object({}),
  });
  const definition = { name: producer.name, description: producer.description, parameters: { type: 'object' as const, properties: {} } };
  const registry: NonNullable<AgentInputs['toolRegistry']> = new Map([[producer.name, definition]]);
  const langfuse: LangfuseConfig = { toolOutputTracing: { redactedToolNames: [producer.name] } };
  const agent: AgentInputs = {
    agentId: 'reader-agent',
    provider: Providers.OPENAI,
    clientOptions: {
      model: 'gpt-5.6-sol', apiKey: 'synthetic-key', maxRetries: 0,
      useResponsesApi: route === 'responses',
      configuration: { baseURL: 'http://synthetic.invalid/v1', fetch: transport },
    },
    tools: mode === 'local' ? [producer] : [],
    toolDefinitions: mode === 'event' ? [definition] : undefined,
    toolRegistry: registry,
    maxToolResultChars: 512,
    maxContextTokens: 100000,
  };
  const handler: EventHandler = { handle: (_event, data) => {
    const batch = data as ToolExecuteBatchRequest;
    expect(batch.toolCalls).toHaveLength(1);
    expect(batch.toolCalls[0].name).toBe(producer.name);
    producerCalls++;
    batch.resolve([{ toolCallId: batch.toolCalls[0].id, content: source, status: 'success' }]);
  } };
  const run = await Run.create({
    runId: 'synthetic-reader-run',
    graphConfig: { type: 'standard', agents: [agent] },
    toolOutputReferences: { enabled: true, ...(readTool === undefined ? {} : { readTool }) },
    customHandlers: { [GraphEvents.ON_TOOL_EXECUTE]: handler },
    hooks, langfuse, returnContent: true, skipCleanup: true,
  });
  expect(registry.has(TOOL_OUTPUT_READER)).toBe(false);
  expect(langfuse.toolOutputTracing?.redactedToolNames).toEqual([producer.name]);
  await run.processStream({ messages: [new HumanMessage('Inspect the retained command result.')] }, {
    version: 'v2', configurable: { thread_id: 'synthetic-reader-thread' },
  });
  const policy = readTool === undefined ? 'omitted' : String(readTool);
  const output = process.env.OUTPUT_READER_CAPTURE_DIR;
  if (output != null) {
    writeFileSync(`${output}/${route}-${mode}-${policy}.json`, JSON.stringify({ requests, producerCalls, readerPolicies }, null, 2));
  }
  return { requests, producerCalls, readerPolicies, run };
}

describe('retained reader through actual provider requests', () => {
  beforeAll(() => { ensureOpenTelemetryContextManager(); });

  it.each((['chat', 'responses'] as const).flatMap((route) =>
    (['local', 'event'] as const).map((mode) => ({ route, mode }))
  ))('$route/$mode recovers retained text without another command', async ({ route, mode }) => {
    const result = await capture(route, mode, true);
    expect(result.requests).toHaveLength(3);
    expect(result.producerCalls).toBe(1);
    expect(result.readerPolicies).toEqual([true]);
    const readers = result.requests[0].body.tools?.filter((entry) => (entry.function?.name ?? entry.name) === TOOL_OUTPUT_READER);
    expect(readers).toHaveLength(1);
    const definition = readers![0].function ?? readers![0];
    expect(Object.keys(definition.parameters?.properties ?? {})).toEqual(expect.arrayContaining(['reference', 'start', 'maxChars']));
    const outputs = toolTexts(result.requests[2].body, route);
    expect(outputs).toHaveLength(2);
    expect(outputs[1].length).toBeLessThanOrEqual(512);
    expect(JSON.parse(outputs[1])).toMatchObject({
      content: sentinel, start: 20000, end: 20000 + sentinel.length,
      retainedChars: source.length, sourceCompleteness: 'unknown',
    });
    expect(outputs[1]).not.toContain('[ref:');
    const context = result.run.Graph?.agentContexts.get('reader-agent');
    expect(context?.graphTools?.filter((entry) => 'name' in entry && entry.name === TOOL_OUTPUT_READER)).toHaveLength(1);
  });

  it.each(['chat', 'responses'] as const)('%s preserves complete requests when omitted/disabled', async (route) => {
    const omitted = await capture(route, 'event', undefined, true);
    const disabled = await capture(route, 'event', false, true);
    expect(omitted.requests).toHaveLength(1);
    expect(disabled.requests[0].raw).toBe(omitted.requests[0].raw);
    expect(omitted.requests[0].raw).not.toContain(TOOL_OUTPUT_READER);
    expect(omitted.producerCalls + disabled.producerCalls).toBe(0);
  });

  it.each(['tools', 'graphTools', 'toolDefinitions', 'toolMap', 'toolRegistry'] as const)('rejects a reserved-name collision in %s', (surface) => {
    const graph = new StandardGraph({ agents: [{ agentId: 'collision', provider: Providers.OPENAI, tools: [] }] });
    graph.toolOutputReferences = { enabled: true, readTool: true };
    const context = graph.agentContexts.get('collision')!;
    const conflicting = tool(async () => 'user tool', { name: TOOL_OUTPUT_READER, description: 'User tool', schema: z.object({}) });
    if (surface === 'tools' || surface === 'graphTools') context[surface] = [conflicting];
    if (surface === 'toolDefinitions') context.toolDefinitions = [{ name: TOOL_OUTPUT_READER }];
    if (surface === 'toolMap') context.toolMap = new Map([[TOOL_OUTPUT_READER, conflicting]]);
    if (surface === 'toolRegistry') context.toolRegistry = new Map([[TOOL_OUTPUT_READER, { name: TOOL_OUTPUT_READER }]]);
    expect(() => graph.createWorkflow()).toThrow('Reserved tool name');
    graph.clearHeavyState();
  });

  it('keeps one direct-only reader per graph across repeated compilation', () => {
    for (let index = 0; index < 2; index++) {
      const graph = new StandardGraph({ agents: [{ agentId: 'repeat', provider: Providers.OPENAI, tools: [] }] });
      graph.toolOutputReferences = { enabled: true, readTool: true };
      graph.createWorkflow();
      graph.createWorkflow();
      const context = graph.agentContexts.get('repeat')!;
      expect(context.graphTools?.filter((entry) => 'name' in entry && entry.name === TOOL_OUTPUT_READER)).toHaveLength(1);
      expect(context.toolRegistry?.get(TOOL_OUTPUT_READER)?.allowed_callers).toEqual(['direct']);
      graph.clearHeavyState();
    }
  });

  it.each([
    { type: 'function', function: { name: TOOL_OUTPUT_READER, parameters: {} } },
    { toolSpec: { name: TOOL_OUTPUT_READER, inputSchema: { json: {} } } },
    { functionDeclarations: [{ name: 'allowed' }, { name: TOOL_OUTPUT_READER }] },
  ])('rejects a reserved-name collision in a native tool schema (%#)', (schema) => {
    const graph = new StandardGraph({
      agents: [{ agentId: 'native-collision', provider: Providers.OPENAI, tools: [schema] }],
    });
    graph.toolOutputReferences = { enabled: true, readTool: true };
    try {
      expect(() => graph.createWorkflow()).toThrow('Reserved tool name');
    } finally {
      graph.clearHeavyState();
    }
  });

  it('keeps reader schema accounting failures from rejecting the pending model turn', async () => {
    const failure = new Error('Synthetic reader schema count failure');
    const errorSpy = jest.spyOn(console, 'error');
    const graph = new StandardGraph({
      agents: [{ agentId: 'count', provider: Providers.OPENAI, tools: [] }],
      tokenCounter: (message) => {
        if (String(message.content).includes(TOOL_OUTPUT_READER)) throw failure;
        return 1;
      },
    });
    graph.toolOutputReferences = { enabled: true, readTool: true };
    try {
      graph.createWorkflow();
      await expect(graph.agentContexts.get('count')!.tokenCalculationPromise).resolves.toBeUndefined();
      expect(errorSpy).toHaveBeenCalledWith(
        'Error recalculating instruction tokens after retained-output reader injection:', failure
      );
    } finally {
      graph.clearHeavyState();
      errorSpy.mockRestore();
    }
  });
});
