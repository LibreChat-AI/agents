import { RunnableLambda } from '@langchain/core/runnables';
import { tool } from '@langchain/core/tools';
import { HumanMessage, AIMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import {
  RequestBodyBudgetError,
  RequestBodyLimitError,
} from '@/llm/requestBodyBudget';
import * as bodyBudget from '@/llm/requestBodyBudget';
import { Providers } from '@/common';
import * as init from '@/llm/init';
import { Run } from '@/run';

type WireItem = {
  type?: string;
  role?: string;
  call_id?: string;
  arguments?: string;
  encrypted_content?: string;
  output?: string;
};
type WireBody = { input: WireItem[]; stream?: boolean };
const counter: t.TokenCounter = (message) =>
  Math.ceil(
    (typeof message.content === 'string'
      ? message.content.length
      : JSON.stringify(message.content).length) / 4
  ) + 4;
const args = {
  path: 'current.ts',
  description: 'Preserve these exact arguments',
};
const ciphertext = 'opaque-sealed-state-'.repeat(20);

function conversation(): BaseMessage[] {
  const messages: BaseMessage[] = [];
  for (let index = 0; index < 6; index++) {
    messages.push(
      new HumanMessage({
        id: `human-${index}`,
        content: `prior question ${index}`,
      })
    );
    messages.push(
      new AIMessage({ id: `ai-${index}`, content: '日本語'.repeat(667) })
    );
  }
  messages.push(
    new HumanMessage({
      id: 'current-task',
      content: 'Keep the latest tool exchange intact.',
    })
  );
  messages.push(
    new AIMessage({
      id: 'current-call',
      content: '',
      tool_calls: [{ id: 'call_current', name: 'read', args }],
      additional_kwargs: {
        reasoning: {
          id: 'rs_current',
          type: 'reasoning',
          summary: [],
          encrypted_content: ciphertext,
        },
      },
      response_metadata: { model_provider: 'openai' },
    })
  );
  messages.push(
    new ToolMessage({
      id: 'current-output',
      tool_call_id: 'call_current',
      content: 'current file contents',
    })
  );
  messages.push(new HumanMessage({ id: 'continue', content: 'Finish now.' }));
  return messages;
}

async function createRun(
  bodies: string[],
  cap?: number,
  options?: {
    instructions?: string;
    maxContextTokens?: number;
    tokenCounter?: t.TokenCounter;
    fallbacks?: t.FallbackConfig[];
  }
): Promise<Run<t.IState>> {
  const transport: typeof fetch = async (_url, request) => {
    if (typeof request?.body !== 'string')
      throw new Error('Expected JSON body');
    bodies.push(request.body);
    const response = {
      id: 'resp_local',
      object: 'response',
      created_at: 1,
      status: 'completed',
      model: 'gpt-4o',
      output: [
        {
          type: 'message',
          id: 'msg_local',
          status: 'completed',
          role: 'assistant',
          content: [
            { type: 'output_text', text: 'recovered', annotations: [] },
          ],
        },
      ],
    };
    return Response.json(response);
  };
  return Run.create<t.IState>({
    runId: 'local-body-budget',
    returnContent: true,
    skipCleanup: true,
    tokenCounter: options?.tokenCounter ?? counter,
    graphConfig: {
      type: 'standard',
      instructions: options?.instructions ?? 'Use the repository evidence.',
      maxContextTokens: options?.maxContextTokens ?? 4_000,
      llmConfig: {
        provider: Providers.OPENAI,
        model: 'gpt-4o',
        apiKey: 'offline-key',
        disableStreaming: true,
        streaming: false,
        streamUsage: false,
        _lc_stream_delay: 0,
        useResponsesApi: true,
        maxTokens: 32,
        maxRequestBodyBytes: cap,
        configuration: {
          baseURL: 'http://synthetic.invalid/v1',
          fetch: transport,
        },
        fallbacks: options?.fallbacks,
      },
      tools: [
        tool(async () => 'unused', {
          name: 'read',
          description: 'Read a file',
          schema: {
            type: 'object',
            properties: {
              path: { type: 'string' },
              description: { type: 'string' },
            },
          },
        }),
      ],
    },
  });
}

function captureCandidates(): string[] {
  const bodies: string[] = [];
  const original = bodyBudget.assertRequestBodyBudget;
  jest
    .spyOn(bodyBudget, 'assertRequestBodyBudget')
    .mockImplementation((body, limit) => {
      if (typeof body === 'string') bodies.push(body);
      original(body, limit);
    });
  return bodies;
}

const streamConfig = {
  configurable: { thread_id: 'request-body-budget' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

describe('Graph local serialized body recovery', () => {
  afterEach(() => jest.restoreAllMocks());

  it.each([false, true])('reprojects a token-fitting oversized body with summarization=%s', async (summarization) => {
    const candidates = captureCandidates();
    const bodies: string[] = [];
    const messages = conversation();
    const canonical = JSON.stringify(
      messages.map((message) => message.toDict())
    );
    expect(
      messages.reduce((total, message) => total + counter(message), 0)
    ).toBeLessThan(4_000);
    const run = await createRun(bodies, 35_000);
    const context = run.Graph?.agentContexts.get('default');
    if (context == null) throw new Error('Expected context');
    context.summarizationEnabled = summarization;
    const initialize = jest.spyOn(init, 'initializeModel');
    const calibration = context.calibrationRatio;
    const content = await run.processStream({ messages }, streamConfig);
    expect(content).toEqual([
      expect.objectContaining({ type: 'text', text: 'recovered' }),
    ]);
    expect(candidates.length).toBeGreaterThan(1);
    expect(candidates.length).toBeLessThanOrEqual(3);
    expect(Buffer.byteLength(candidates[0])).toBeGreaterThan(35_000);
    expect(bodies).toHaveLength(1);
    expect(Buffer.byteLength(bodies[0])).toBeLessThanOrEqual(35_000);
    expect(initialize).toHaveBeenCalledTimes(candidates.length);
    expect(initialize.mock.calls.every(([options]) => options.clientOptions != null &&
      'model' in options.clientOptions && options.clientOptions.model === 'gpt-4o')).toBe(true);
    expect(context.calibrationRatio).toBe(calibration);
    expect(context.overflowRecoveryAttempts).toBe(candidates.length - 1);
    expect(JSON.stringify(messages.map((message) => message.toDict()))).toBe(
      canonical
    );
    const wire = JSON.parse(bodies[0]) as WireBody;
    expect(wire.input.filter((item) => item.type === 'function_call')).toEqual([
      expect.objectContaining({
        call_id: 'call_current',
        arguments: JSON.stringify(args),
      }),
    ]);
    expect(
      wire.input.filter((item) => item.type === 'function_call_output')
    ).toEqual([
      expect.objectContaining({
        call_id: 'call_current',
        output: 'current file contents',
      }),
    ]);
    expect(
      wire.input.find((item) => item.type === 'reasoning')?.encrypted_content
    ).toBe(ciphertext);
  });

  it('preserves the unguarded request path by default', async () => {
    const candidates = captureCandidates();
    const bodies: string[] = [];
    const run = await createRun(bodies);
    await run.processStream({ messages: conversation() }, streamConfig);
    expect(candidates).toEqual(bodies);
    expect(bodies).toHaveLength(1);
    expect(Buffer.byteLength(bodies[0])).toBeGreaterThan(35_000);
    expect(
      run.Graph?.agentContexts.get('default')?.overflowRecoveryAttempts
    ).toBe(0);
  });

  it('stops an irreducible request without upstream, summaries or fallbacks', async () => {
    const candidates = captureCandidates();
    const bodies: string[] = [];
    const run = await createRun(bodies, 400, {
      instructions: 'Immutable instructions '.repeat(100),
      maxContextTokens: 16_000,
      fallbacks: [
        {
          provider: Providers.OPENAI,
          clientOptions: {
            model: 'fallback-not-called',
            apiKey: 'offline-key',
          },
        },
      ],
    });
    const initialize = jest.spyOn(init, 'initializeModel');
    const context = run.Graph?.agentContexts.get('default');
    if (context == null) throw new Error('Expected default context');
    context.summarizationEnabled = true;
    const messages = [new HumanMessage({ id: 'small-task', content: 'hello' })];
    await expect(
      run.processStream({ messages }, streamConfig)
    ).rejects.toBeInstanceOf(RequestBodyLimitError);
    expect(candidates).toHaveLength(2);
    expect(candidates[1]).toBe(candidates[0]);
    expect(context.overflowRecoveryAttempts).toBe(1);
    expect(initialize).toHaveBeenCalledTimes(2);
    expect(
      initialize.mock.calls.every(
        ([options]) =>
          options.clientOptions != null &&
          'model' in options.clientOptions &&
          options.clientOptions.model === 'gpt-4o'
      )
    ).toBe(true);
    expect(bodies).toEqual([]);
  });

  it('propagates an unsupported body without attempting projection recovery', async () => {
    const error = new RequestBodyBudgetError('unsupported fixture');
    const run = await createRun([], 1000);
    if (run.Graph == null) throw new Error('Expected graph');
    const primary = RunnableLambda.from((): never => {
      throw error;
    });
    const initialize = jest
      .spyOn(init, 'initializeModel')
      .mockReturnValue(primary);
    await expect(
      run.processStream({ messages: [new HumanMessage('hello')] }, streamConfig)
    ).rejects.toBe(error);
    expect(initialize).toHaveBeenCalledTimes(1);
    expect(
      run.Graph.agentContexts.get('default')?.overflowRecoveryAttempts
    ).toBe(0);
  });
});
