import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { MemorySaver } from '@langchain/langgraph';
import { Runnable } from '@langchain/core/runnables';
import { describe, expect, it, jest } from '@jest/globals';
import {
  AIMessageChunk,
  HumanMessage,
  AIMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { OVERFLOW_SIGNATURES } from '@/utils/__tests__/fixtures/contextOverflowSignatures';
import { ContentTypes, GraphEvents, Providers } from '@/common';
import { getProviderSourceMessageIds } from '@/messages/provenance';
import * as init from '@/llm/init';
import { Run } from '@/run';

/**
 * Every message is claimed to cost the same, so the budget the recovery
 * settles on maps directly onto a message count and the assertions can be
 * about behavior rather than arithmetic.
 */
const TOKENS_PER_MESSAGE = 40_000;

const tokenCounter: t.TokenCounter = () => TOKENS_PER_MESSAGE;

function signatureFor(model: string): Record<string, unknown> {
  const signature = OVERFLOW_SIGNATURES.find((s) => s.model === model);
  if (signature == null) {
    throw new Error(`missing fixture for ${model}`);
  }
  return signature.error;
}

/** Rebuilds a thrown provider error from a captured signature. */
function throwable(fields: Record<string, unknown>): Error {
  const error = new Error(String(fields.message));
  return Object.assign(error, fields);
}

/**
 * Fails the first N calls with a real captured provider rejection, then
 * answers normally — the shape of a run that overflows and then fits after
 * compaction.
 */
class OverflowThenSucceedModel extends Runnable<BaseMessage[], AIMessageChunk> {
  lc_namespace = ['tests'];
  readonly calls: BaseMessage[][] = [];

  constructor(
    private readonly error: Record<string, unknown>,
    private readonly failures = 1
  ) {
    super();
  }

  private record(messages: BaseMessage[]): void {
    this.calls.push(messages);
    if (this.calls.length <= this.failures) {
      throw throwable(this.error);
    }
  }

  async invoke(messages: BaseMessage[]): Promise<AIMessageChunk> {
    this.record(messages);
    return new AIMessageChunk({ content: 'recovered' });
  }
}

class SizeBoundModel extends Runnable<BaseMessage[], AIMessageChunk> {
  lc_namespace = ['tests'];
  readonly toolContentChars: number[] = [];

  constructor(
    private readonly maxToolContentChars: number,
    private readonly error: Record<string, unknown>
  ) {
    super();
  }

  async invoke(messages: BaseMessage[]): Promise<AIMessageChunk> {
    let toolContentChars = 0;
    for (const message of messages) {
      if (message.getType() !== 'tool') {
        continue;
      }
      toolContentChars +=
        typeof message.content === 'string'
          ? message.content.length
          : JSON.stringify(message.content).length;
    }
    this.toolContentChars.push(toolContentChars);
    if (toolContentChars > this.maxToolContentChars) {
      throw throwable(this.error);
    }
    return new AIMessageChunk({ content: 'recovered' });
  }
}

function buildConversation(turns: number): BaseMessage[] {
  const messages: BaseMessage[] = [];
  for (let i = 0; i < turns; i++) {
    messages.push(new HumanMessage(`question ${i}`));
    messages.push(new AIMessage(`answer ${i}`));
  }
  messages.push(new HumanMessage('final question'));
  return messages;
}

async function createRun(options: {
  runId: string;
  maxContextTokens: number;
  checkpointer?: boolean;
  provider?: Providers;
  tokenCounter?: t.TokenCounter;
  indexTokenCountMap?: Record<string, number>;
  tools?: t.GraphTools;
  maxToolResultChars?: number;
  model?: string;
  promptCache?: boolean;
  toolOutputReferences?: t.ToolOutputReferencesConfig;
  fallbacks?: t.FallbackConfig[];
}): Promise<Run<t.IState>> {
  return Run.create<t.IState>({
    runId: options.runId,
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: options.provider ?? Providers.ANTHROPIC,
        ...(options.model != null ? { model: options.model } : {}),
        ...(options.promptCache === true ? { promptCache: true } : {}),
        ...(options.fallbacks != null ? { fallbacks: options.fallbacks } : {}),
        disableStreaming: true,
        streamUsage: false,
      },
      maxContextTokens: options.maxContextTokens,
      maxToolResultChars: options.maxToolResultChars,
      tools: options.tools,
      compileOptions:
        options.checkpointer === true
          ? { checkpointer: new MemorySaver() }
          : undefined,
    },
    returnContent: true,
    skipCleanup: true,
    tokenCounter: options.tokenCounter ?? tokenCounter,
    indexTokenCountMap: options.indexTokenCountMap,
    toolOutputReferences: options.toolOutputReferences,
  });
}

const streamConfig = {
  configurable: { thread_id: 'context-overflow-recovery' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

async function captureFallback(
  messages: BaseMessage[],
  fallbackConfig: t.FallbackConfig,
  options: Partial<Parameters<typeof createRun>[0]> & { legacy?: boolean } = {}
): Promise<BaseMessage[]> {
  const run = await createRun({
    runId: 'fallback-serving-pipeline',
    maxContextTokens: 1_000_000,
    provider: Providers.ANTHROPIC,
    tokenCounter: () => 10,
    ...options,
    fallbacks: [fallbackConfig],
  });
  if (!run.Graph) {
    throw new Error('Expected graph');
  }
  const context = run.Graph.agentContexts.get('default');
  if (context != null && options.legacy === true) {
    context.useLegacyContent = true;
  }
  const primary = new OverflowThenSucceedModel({ message: '503 unavailable' });
  const fallback = new OverflowThenSucceedModel({}, 0);
  const spy = jest
    .spyOn(init, 'initializeModel')
    .mockReturnValue(fallback as ReturnType<typeof init.initializeModel>);
  run.Graph.overrideModel = primary;
  try {
    await run.processStream({ messages }, streamConfig);
    expect(primary.calls).toHaveLength(1);
    expect(fallback.calls).toHaveLength(1);
    return fallback.calls[0];
  } finally {
    spy.mockRestore();
  }
}

describe('context overflow recovery', () => {
  it.each([Providers.ANTHROPIC, Providers.OPENAI, Providers.GOOGLE])(
    'preserves fallback artifacts for %s',
    async (provider) => {
      const source = new ToolMessage({
        content: 'rendered',
        name: 'render',
        tool_call_id: 'render-1',
        artifact: {
          content: [
            { type: 'text', text: 'fallback-artifact-evidence' },
            {
              type: 'image_url',
              image_url: { url: 'data:image/png;base64,aW1hZ2U=' },
            },
          ],
        },
      });
      const result = await captureFallback(
        [
          new HumanMessage('Render'),
          new AIMessage({
            content: '',
            tool_calls: [{ id: 'render-1', name: 'render', args: {} }],
          }),
          source,
        ],
        { provider, maxContextTokens: 1_000_000 }
      );
      const wire = JSON.stringify(result.map((message) => message.content));
      expect(wire).toContain('fallback-artifact-evidence');
      expect(wire).toContain('aW1hZ2U=');
      expect(source.content).toBe('rendered');
    }
  );

  it('restores legacy string content for fallback endpoints', async () => {
    const result = await captureFallback(
      [
        new HumanMessage({
          content: [{ type: 'text', text: 'legacy question' }],
        }),
      ],
      { provider: Providers.OPENAI, maxContextTokens: 1_000_000 },
      { legacy: true }
    );
    expect(result.every((message) => typeof message.content === 'string')).toBe(
      true
    );
    expect(result[0].content).toContain('legacy question');
  });

  it('omits optional artifact expansion when the fallback budget cannot fit it', async () => {
    const result = await captureFallback(
      [
        new HumanMessage('Keep this question'),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'call', name: 'render', args: {} }],
        }),
        new ToolMessage({
          content: 'rendered',
          tool_call_id: 'call',
          artifact: { content: [{ type: 'text', text: 'oversized-artifact' }] },
        }),
      ],
      { provider: Providers.ANTHROPIC, maxContextTokens: 1_000 },
      {
        tokenCounter: (message) =>
          JSON.stringify(message.content).includes('oversized-artifact')
            ? 100_000
            : 10,
      }
    );
    const wire = JSON.stringify(result.map((message) => message.content));
    expect(wire).not.toContain('oversized-artifact');
    expect(wire).toContain('rendered');
    expect(wire).toContain('Keep this question');
  });

  it.each([Providers.ANTHROPIC, Providers.OPENROUTER, Providers.BEDROCK])(
    'adds serving fallback cache markers for %s',
    async (provider) => {
      const result = await captureFallback(
        [new HumanMessage('cache this history')],
        {
          provider,
          clientOptions: {
            promptCache: true,
            model: 'anthropic.claude-sonnet-4-5',
          },
          maxContextTokens: 1_000_000,
        }
      );
      expect(
        JSON.stringify(result.map((message) => message.content))
      ).toContain(
        provider === Providers.BEDROCK ? 'cachePoint' : 'cache_control'
      );
    }
  );

  it('sanitizes orphan tool results before a tool-enabled fallback', async () => {
    const result = await captureFallback(
      [
        new HumanMessage('Continue'),
        new ToolMessage({
          content: 'orphan evidence',
          tool_call_id: 'missing',
          name: 'lookup',
        }),
        new HumanMessage('Latest'),
      ],
      { provider: Providers.ANTHROPIC, maxContextTokens: 1_000_000 },
      {
        tools: [
          tool(async () => 'unused', {
            name: 'lookup',
            description: 'Lookup',
            schema: z.object({}),
          }),
        ],
      }
    );
    expect(result.some((message) => message.getType() === 'tool')).toBe(false);
  });

  it('compacts only synthetic fallback context to fit a smaller window', async () => {
    const result = await captureFallback(
      [
        new HumanMessage('Keep this question'),
        new AIMessage({
          content: '',
          tool_calls: [
            { id: 'call', name: 'lookup', args: { query: 'x'.repeat(5_000) } },
          ],
        }),
        new ToolMessage({ content: 'result', tool_call_id: 'call' }),
      ],
      { provider: Providers.ANTHROPIC, maxContextTokens: 1_000 },
      {
        tokenCounter: (message) => JSON.stringify(message.content).length,
      }
    );
    expect(JSON.stringify(result.map((message) => message.content))).toContain(
      'Keep this question'
    );
    expect(
      result.reduce(
        (sum, message) => sum + JSON.stringify(message.content).length,
        0
      )
    ).toBeLessThanOrEqual(1_000);
  });
  it.each([
    { primaryNative: false, intermediate: false },
    { primaryNative: true, intermediate: false },
    { primaryNative: false, intermediate: true },
  ])(
    'preserves source history across serving policy changes: %j',
    async ({ primaryNative, intermediate }) => {
      const source = new AIMessage({
        content: 'A drawing.',
        additional_kwargs: { sourceMessageId: 'original-drawing' },
        response_metadata: {
          model_provider: 'openai',
          preempted: true,
          output: [
            {
              type: 'image_generation_call',
              id: 'image-1',
              status: 'completed',
              result: '/9j/AA==',
            },
            {
              type: 'message',
              id: 'msg-1',
              content: [{ type: 'output_text', text: 'A drawing.' }],
            },
          ],
        },
      });
      const run = await createRun({
        runId: 'native-fallback-source-history',
        maxContextTokens: 1_000_000,
        provider: primaryNative ? Providers.OPENAI : Providers.ANTHROPIC,
        tokenCounter: () => 10,
        fallbacks: [
          ...(intermediate
            ? [{ provider: Providers.GOOGLE, maxContextTokens: 1_000_000 }]
            : []),
          {
            provider: primaryNative ? Providers.ANTHROPIC : Providers.OPENAI,
            maxContextTokens: 1_000_000,
          },
        ],
      });
      if (!run.Graph) {
        throw new Error('Expected graph');
      }
      const primary = new OverflowThenSucceedModel({
        message: '503 unavailable',
      });
      const fallback = new OverflowThenSucceedModel({}, 0);
      Object.assign(primary, {
        _useResponsesApi: (): boolean => primaryNative,
      });
      Object.assign(fallback, { _useResponsesApi: (): boolean => true });
      const initializeSpy = jest.spyOn(init, 'initializeModel');
      if (intermediate) {
        initializeSpy.mockReturnValueOnce(
          new OverflowThenSucceedModel({
            message: '503 intermediate unavailable',
          }) as ReturnType<typeof init.initializeModel>
        );
      }
      initializeSpy.mockReturnValue(
        fallback as ReturnType<typeof init.initializeModel>
      );
      run.Graph.overrideModel = primary;
      try {
        await run.processStream(
          {
            messages: [
              new HumanMessage('Draw'),
              source,
              new HumanMessage('Continue'),
            ],
          },
          streamConfig
        );
        expect(
          JSON.stringify(primary.calls[0]).includes(
            '[Previous tool interaction]'
          )
        ).toBe(!primaryNative);
        expect(fallback.calls).toHaveLength(1);
        const retained = fallback.calls[0].find((message) =>
          getProviderSourceMessageIds(message).includes('original-drawing')
        );
        expect(retained?.content).toEqual(
          expect.arrayContaining([
            expect.objectContaining({
              type: 'image',
              mimeType: 'image/jpeg',
              data: '/9j/AA==',
            }),
          ])
        );
        expect(
          JSON.stringify(fallback.calls[0]).includes(
            '[Previous tool interaction]'
          )
        ).toBe(primaryNative);
        expect(source.response_metadata.output).toBeDefined();
      } finally {
        initializeSpy.mockRestore();
      }
    }
  );
  it('projects structured OpenAI tool content before the final payload check', async () => {
    const toolCallId = 'tc-openai-structured';
    const toolMessage = new ToolMessage({
      content: [
        { type: ContentTypes.TEXT, text: 'rendered chart' },
        {
          type: 'image_url',
          image_url: {
            url: `data:image/png;base64,${'A'.repeat(2_000)}`,
          },
        },
      ],
      tool_call_id: toolCallId,
      name: 'render_chart',
    });
    const messages: BaseMessage[] = [
      new HumanMessage('render the chart'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: toolCallId,
            name: 'render_chart',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      toolMessage,
    ];
    const measuredToolContents: string[] = [];
    const projectionCounter: t.TokenCounter = (message) => {
      if (message.getType() === 'tool' && typeof message.content === 'string') {
        measuredToolContents.push(message.content);
      }
      return typeof message.content === 'string'
        ? Math.max(1, Math.ceil(message.content.length / 4))
        : 1;
    };
    const run = await createRun({
      runId: 'openai-structured-final-projection',
      maxContextTokens: 10_000,
      maxToolResultChars: 200,
      provider: Providers.OPENAI,
      tokenCounter: projectionCounter,
      indexTokenCountMap: {
        0: projectionCounter(messages[0]),
        1: projectionCounter(messages[1]),
        2: projectionCounter(messages[2]),
      },
      tools: [
        tool(async () => 'unused', {
          name: 'render_chart',
          description: 'Renders a chart',
          schema: z.object({}),
        }),
      ],
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(signatureFor('gpt-4o-mini'), 0);
    run.Graph.overrideModel = model;

    await run.processStream({ messages }, streamConfig);

    expect(model.calls).toHaveLength(1);
    const projectedTool = model.calls[0].find(
      (message) => message.getType() === 'tool'
    ) as ToolMessage | undefined;
    expect(typeof projectedTool?.content).toBe('string');
    expect((projectedTool?.content as string).length).toBeLessThanOrEqual(200);
    expect(measuredToolContents).toContain(projectedTool?.content);
    expect(Array.isArray(toolMessage.content)).toBe(true);
  });

  it('measures the bounded generic-provider tool payload that is invoked', async () => {
    const toolCallId = 'tc-google-structured';
    const structuredContent = Array.from({ length: 25_000 }, () => ({
      type: ContentTypes.TEXT,
      text: 'x',
    }));
    const toolMessage = new ToolMessage({
      content: structuredContent,
      tool_call_id: toolCallId,
      name: 'dense_result',
    });
    const messages: BaseMessage[] = [
      new HumanMessage('return the dense result'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: toolCallId,
            name: 'dense_result',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      toolMessage,
    ];
    const measuredToolContents: string[] = [];
    const projectionCounter: t.TokenCounter = (message) => {
      if (message.getType() === 'tool' && typeof message.content === 'string') {
        measuredToolContents.push(message.content);
      }
      return typeof message.content === 'string'
        ? Math.max(1, Math.ceil(message.content.length / 4))
        : 1;
    };
    const run = await createRun({
      runId: 'google-structured-final-projection',
      maxContextTokens: 10_000,
      maxToolResultChars: 2_000,
      provider: Providers.GOOGLE,
      tokenCounter: projectionCounter,
      indexTokenCountMap: {
        0: projectionCounter(messages[0]),
        1: projectionCounter(messages[1]),
        2: projectionCounter(messages[2]),
      },
      tools: [
        tool(async () => 'unused', {
          name: 'dense_result',
          description: 'Returns a dense structured result',
          schema: z.object({}),
        }),
      ],
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(signatureFor('gpt-4o-mini'), 0);
    run.Graph.overrideModel = model;

    await run.processStream({ messages }, streamConfig);

    expect(model.calls).toHaveLength(1);
    const projectedTool = model.calls[0].find(
      (message) => message.getType() === 'tool'
    ) as ToolMessage | undefined;
    expect(typeof projectedTool?.content).toBe('string');
    expect((projectedTool?.content as string).length).toBeLessThanOrEqual(
      2_000
    );
    expect(measuredToolContents).toContain(projectedTool?.content);
    expect(toolMessage.content).toBe(structuredContent);
  });

  it('guards a fallback-specific projection before invoking the fallback', async () => {
    const toolCallId = 'tc-fallback-structured';
    const structuredContent = Array.from({ length: 25_000 }, () => ({
      type: ContentTypes.TEXT,
      text: '',
    }));
    const messages: BaseMessage[] = [
      new HumanMessage('return the dense fallback result'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: toolCallId,
            name: 'dense_result',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: structuredContent,
        tool_call_id: toolCallId,
        name: 'dense_result',
      }),
    ];
    const projectionCounter: t.TokenCounter = (message) =>
      typeof message.content === 'string'
        ? Math.max(1, message.content.length)
        : 1;
    const run = await createRun({
      runId: 'fallback-structured-final-projection',
      maxContextTokens: 1_000_000,
      provider: Providers.ANTHROPIC,
      tokenCounter: projectionCounter,
      indexTokenCountMap: {
        0: projectionCounter(messages[0]),
        1: projectionCounter(messages[1]),
        2: projectionCounter(messages[2]),
      },
      fallbacks: [
        {
          provider: Providers.GOOGLE,
          maxContextTokens: 5_000,
        },
      ],
      tools: [
        tool(async () => 'unused', {
          name: 'dense_result',
          description: 'Returns a dense structured result',
          schema: z.object({}),
        }),
      ],
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const primary = new OverflowThenSucceedModel(
      { message: '503 primary unavailable' },
      1
    );
    let fallbackInvocations = 0;
    const fallback = {
      invoke: async (): Promise<AIMessageChunk> => {
        fallbackInvocations++;
        return new AIMessageChunk({ content: 'fallback should not run' });
      },
    } as unknown as ReturnType<typeof init.initializeModel>;
    const initializeSpy = jest
      .spyOn(init, 'initializeModel')
      .mockReturnValue(fallback);
    run.Graph.overrideModel = primary;

    try {
      const content = await run.processStream({ messages }, streamConfig);

      expect(content).toEqual([{ type: 'text', text: 'recovered' }]);
      expect(primary.calls).toHaveLength(2);
      expect(fallbackInvocations).toBe(0);
      expect(
        run.Graph.agentContexts.get('default')?.overflowRecoveryAttempts
      ).toBe(1);
    } finally {
      initializeSpy.mockRestore();
    }
  });

  it('projects unsafe tool-call args before measuring or invoking the provider', async () => {
    let toJSONCalls = 0;
    const toolCallId = 'tc-unsafe-input';
    const unsafeArgs = {
      query: 'safe',
      toJSON() {
        toJSONCalls++;
        return { query: 'x'.repeat(100_000) };
      },
    };
    const messages: BaseMessage[] = [
      new HumanMessage('run the lookup'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: toolCallId,
            name: 'lookup_records',
            args: unsafeArgs,
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'done',
        tool_call_id: toolCallId,
        name: 'lookup_records',
      }),
      new AIMessage('The lookup completed.'),
      new HumanMessage('continue'),
    ];
    const run = await createRun({
      runId: 'unsafe-tool-input-final-projection',
      maxContextTokens: 10_000,
      provider: Providers.OPENAI,
      tokenCounter: () => 1,
      indexTokenCountMap: { 0: 1, 1: 1, 2: 1, 3: 1, 4: 1 },
      tools: [
        tool(async () => 'unused', {
          name: 'lookup_records',
          description: 'Looks up records',
          schema: z.object({}),
        }),
      ],
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(signatureFor('gpt-4o-mini'), 0);
    run.Graph.overrideModel = model;

    await run.processStream({ messages }, streamConfig);

    const projectedCall = (
      model.calls[0].find((message) => message.getType() === 'ai') as AIMessage
    ).tool_calls?.[0];
    expect(projectedCall?.args).toEqual({ query: 'safe' });
    expect(toJSONCalls).toBe(0);
    expect(messages[1]).toBeInstanceOf(AIMessage);
    expect((messages[1] as AIMessage).tool_calls?.[0].args).toBe(unsafeArgs);
  });

  it('compacts cached structured tool output before the first provider call', async () => {
    const toolCallId = 'tc-structured';
    const messages: BaseMessage[] = [
      new HumanMessage('query the table'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: toolCallId,
            name: 'run_select_query',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: [
          {
            type: ContentTypes.TEXT,
            text: JSON.stringify(
              Array.from({ length: 240 }, (_, index) => ({
                id: index,
                value: `${'x'.repeat(100)}-${index}`,
              }))
            ),
          },
        ],
        tool_call_id: toolCallId,
        name: 'run_select_query',
      }),
      new AIMessage('The query returned 240 rows.'),
      new HumanMessage('compact context'),
    ];
    const structuredTokenCounter: t.TokenCounter = (message) => {
      const content =
        typeof message.content === 'string'
          ? message.content
          : JSON.stringify(message.content);
      return Math.ceil(content.length / 4);
    };
    const indexTokenCountMap: Record<string, number> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = i === 2 ? 0 : structuredTokenCounter(messages[i]);
    }
    const run = await createRun({
      runId: 'structured-output-preflight',
      maxContextTokens: 5_000,
      provider: Providers.BEDROCK,
      model: 'anthropic.claude-sonnet-4-5',
      tokenCounter: structuredTokenCounter,
      indexTokenCountMap,
      tools: [
        tool(async () => 'unused', {
          name: 'run_select_query',
          description: 'Queries ClickHouse',
          schema: z.object({}),
        }),
      ],
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new SizeBoundModel(
      1_500,
      signatureFor('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
    );
    run.Graph.overrideModel = model;

    const content = await run.processStream({ messages }, streamConfig);

    expect(content).toEqual([{ type: 'text', text: 'recovered' }]);
    expect(model.toolContentChars).toHaveLength(1);
    expect(model.toolContentChars[0]).toBeGreaterThan(0);
    expect(model.toolContentChars[0]).toBeLessThanOrEqual(1_500);
    expect(
      run.Graph.agentContexts.get('default')?.overflowRecoveryAttempts
    ).toBe(0);
  });

  it('compacts an unconsumed structured tool result before its first provider call', async () => {
    const toolCallId = 'tc-unconsumed-structured';
    const messages: BaseMessage[] = [
      new HumanMessage('query the table'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: toolCallId,
            name: 'run_select_query',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: [
          {
            type: ContentTypes.TEXT,
            text: JSON.stringify(
              Array.from({ length: 240 }, (_, index) => ({
                id: index,
                value: `${'x'.repeat(100)}-${index}`,
              }))
            ),
          },
        ],
        tool_call_id: toolCallId,
        name: 'run_select_query',
      }),
    ];
    const structuredTokenCounter: t.TokenCounter = (message) => {
      const content =
        typeof message.content === 'string'
          ? message.content
          : JSON.stringify(message.content);
      return Math.ceil(content.length / 4);
    };
    const run = await createRun({
      runId: 'unconsumed-structured-output-preflight',
      maxContextTokens: 5_000,
      maxToolResultChars: 1_500,
      provider: Providers.BEDROCK,
      tokenCounter: structuredTokenCounter,
      indexTokenCountMap: {
        0: structuredTokenCounter(messages[0]),
        1: structuredTokenCounter(messages[1]),
        2: 0,
      },
      tools: [
        tool(async () => 'unused', {
          name: 'run_select_query',
          description: 'Queries ClickHouse',
          schema: z.object({}),
        }),
      ],
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new SizeBoundModel(
      1_500,
      signatureFor('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
    );
    run.Graph.overrideModel = model;

    const content = await run.processStream({ messages }, streamConfig);

    expect(content).toEqual([{ type: 'text', text: 'recovered' }]);
    expect(model.toolContentChars).toHaveLength(1);
    expect(model.toolContentChars[0]).toBeGreaterThan(0);
    expect(model.toolContentChars[0]).toBeLessThanOrEqual(1_500);
    expect(
      run.Graph.agentContexts.get('default')?.overflowRecoveryAttempts
    ).toBe(0);
  });

  it('trusts the pruner baseline for an unchanged payload at high calibration', async () => {
    const messages: BaseMessage[] = [
      new HumanMessage('earlier question'),
      new AIMessage('provider-counted answer'),
      new HumanMessage('latest question'),
    ];
    const localCounter: t.TokenCounter = (message) =>
      message.getType() === 'ai' ? 120 : 5;
    const providerGroundedTokenMap = { 0: 5, 1: 10, 2: 5 };
    const run = await createRun({
      runId: 'provider-grounded-final-projection',
      maxContextTokens: 200,
      tokenCounter: localCounter,
      indexTokenCountMap: providerGroundedTokenMap,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const agentContext = run.Graph.agentContexts.get('default');
    if (agentContext == null) {
      throw new Error('Expected default agent context');
    }
    agentContext.calibrationRatio = 5;
    const model = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001'),
      0
    );
    run.Graph.overrideModel = model;

    const content = await run.processStream({ messages }, streamConfig);

    expect(content).toEqual([{ type: 'text', text: 'recovered' }]);
    expect(model.calls).toHaveLength(1);
    expect(model.calls[0]).toHaveLength(messages.length);
    expect(agentContext.overflowRecoveryAttempts).toBe(0);
  });

  it('does not let an unrelated provider-format shrink hide artifact growth', async () => {
    const toolCallId = 'tc-artifact-independent-growth';
    const artifactSentinel = 'ARTIFACT_INDEPENDENT_GROWTH_SENTINEL';
    const toolMessage = new ToolMessage({
      content: 'rendered',
      tool_call_id: toolCallId,
      name: 'render_report',
      artifact: {
        content: [
          {
            type: ContentTypes.TEXT,
            text: artifactSentinel,
          },
        ],
      },
    });
    const messages: BaseMessage[] = [
      new HumanMessage('render the report'),
      new AIMessageChunk({
        content: 'calling render_report',
        tool_calls: [
          {
            id: toolCallId,
            name: 'render_report',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      toolMessage,
    ];
    const transformCounter: t.TokenCounter = (message) => {
      if (
        message instanceof AIMessageChunk &&
        typeof message.content === 'string'
      ) {
        return 100;
      }
      if (
        message instanceof ToolMessage &&
        JSON.stringify(message.content).includes(artifactSentinel)
      ) {
        return 50;
      }
      return 1;
    };
    const run = await createRun({
      runId: 'artifact-independent-transform-growth',
      maxContextTokens: 50,
      provider: Providers.BEDROCK,
      model: 'anthropic.claude-sonnet-4-5',
      tokenCounter: transformCounter,
      indexTokenCountMap: { 0: 1, 1: 1, 2: 1 },
      tools: [
        tool(async () => 'unused', {
          name: 'render_report',
          description: 'Renders a report',
          schema: z.object({}),
        }),
      ],
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('us.anthropic.claude-sonnet-4-5-20250929-v1:0'),
      0
    );
    run.Graph.overrideModel = model;

    await run.processStream({ messages }, streamConfig);

    expect(model.calls).toHaveLength(1);
    expect(
      JSON.stringify(model.calls[0].map((message) => message.content))
    ).not.toContain(artifactSentinel);
    expect(toolMessage.artifact.content[0].text).toBe(artifactSentinel);
  });

  it('preserves provider attribution for an un-IDd AI clone during orphan sanitization', async () => {
    const droppedCallId = 'tc-dropped-ai';
    const missingCallId = 'tc-missing-result';
    const unrelatedResultId = 'tc-unrelated-result';
    const messages: BaseMessage[] = [
      new HumanMessage('earlier question'),
      new AIMessage({
        content: [
          {
            type: 'tool_use',
            id: droppedCallId,
            name: 'declared_tool',
            input: {},
          },
        ],
        tool_calls: [
          {
            id: droppedCallId,
            name: 'declared_tool',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      new AIMessage({
        content: 'provider-counted answer',
        tool_calls: [
          {
            id: missingCallId,
            name: 'declared_tool',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'orphaned result',
        tool_call_id: unrelatedResultId,
        name: 'declared_tool',
      }),
      new HumanMessage('latest question'),
    ];
    const skewedCounter: t.TokenCounter = (message) => {
      if (message.getType() !== 'ai') {
        return 5;
      }
      return Array.isArray(message.content) ? 1 : 120;
    };
    const run = await createRun({
      runId: 'orphan-sanitize-provider-origin',
      maxContextTokens: 1_000,
      provider: Providers.ANTHROPIC,
      promptCache: true,
      tokenCounter: skewedCounter,
      indexTokenCountMap: { 0: 5, 1: 80, 2: 10, 3: 5, 4: 5 },
      tools: [
        tool(async () => 'unused', {
          name: 'declared_tool',
          description: 'Declared only to prevent tool-less folding',
          schema: z.object({}),
        }),
      ],
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const agentContext = run.Graph.agentContexts.get('default');
    if (agentContext == null) {
      throw new Error('Expected default agent context');
    }
    agentContext.calibrationRatio = 5;
    const model = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001'),
      0
    );
    run.Graph.overrideModel = model;

    const content = await run.processStream({ messages }, streamConfig);

    expect(content).toEqual([{ type: 'text', text: 'recovered' }]);
    expect(model.calls).toHaveLength(1);
    expect(model.calls[0]).toHaveLength(3);
    expect(model.calls[0].some((message) => message.getType() === 'tool')).toBe(
      false
    );
    const sanitizedAI = model.calls[0].find(
      (message) => message.getType() === 'ai'
    ) as AIMessage | undefined;
    expect(sanitizedAI?.content).toBe('provider-counted answer');
    expect(sanitizedAI?.tool_calls ?? []).toHaveLength(0);
    expect(agentContext.overflowRecoveryAttempts).toBe(0);
  });

  it('reserves the reply primer before accepting fast-path context', async () => {
    const messages: BaseMessage[] = [
      new HumanMessage('earlier question'),
      new AIMessage('earlier answer'),
      new HumanMessage('latest question'),
    ];
    const run = await createRun({
      runId: 'fast-path-reply-primer',
      maxContextTokens: 100,
      tokenCounter: () => 31,
      indexTokenCountMap: { 0: 31, 1: 31, 2: 31 },
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001'),
      0
    );
    run.Graph.overrideModel = model;

    await run.processStream({ messages }, streamConfig);

    expect(model.calls).toHaveLength(1);
    expect(model.calls[0].length).toBeLessThan(messages.length);
    expect(3 + model.calls[0].length * 31).toBeLessThanOrEqual(95);
  });

  it('includes artifact expansion when it fits the post-prune budget', async () => {
    const toolCallId = 'tc-artifact-fits';
    const artifactSentinel = 'ARTIFACT_FITS_SENTINEL';
    const toolMessage = new ToolMessage({
      content: 'rendered',
      tool_call_id: toolCallId,
      name: 'render_report',
      artifact: {
        content: [
          {
            type: ContentTypes.TEXT,
            text: `${artifactSentinel}:complete`,
          },
        ],
      },
    });
    const messages: BaseMessage[] = [
      new HumanMessage('render the report'),
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: toolCallId,
            name: 'render_report',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      toolMessage,
    ];
    const artifactTokenCounter: t.TokenCounter = (message) => {
      const content =
        typeof message.content === 'string'
          ? message.content
          : JSON.stringify(message.content);
      return Math.ceil(content.length / 4);
    };
    const run = await createRun({
      runId: 'artifact-budget-control',
      maxContextTokens: 10_000,
      maxToolResultChars: 2_000,
      provider: Providers.BEDROCK,
      model: 'anthropic.claude-sonnet-4-5',
      tokenCounter: artifactTokenCounter,
      indexTokenCountMap: {
        0: artifactTokenCounter(messages[0]),
        1: artifactTokenCounter(messages[1]),
        2: artifactTokenCounter(messages[2]),
      },
      tools: [
        tool(async () => 'unused', {
          name: 'render_report',
          description: 'Renders a report',
          schema: z.object({}),
        }),
      ],
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('us.anthropic.claude-sonnet-4-5-20250929-v1:0'),
      0
    );
    run.Graph.overrideModel = model;

    await run.processStream({ messages }, streamConfig);

    expect(model.calls).toHaveLength(1);
    expect(
      JSON.stringify(model.calls[0].map((message) => message.content))
    ).toContain(artifactSentinel);
    expect(toolMessage.content).toBe('rendered');
    expect(toolMessage.artifact.content[0].text).toContain(artifactSentinel);
  });

  it('rechecks artifact expansion after provider message transforms', async () => {
    const toolCallId = 'tc-artifact-final-transform';
    const artifactSentinel = 'ARTIFACT_FINAL_TRANSFORM_SENTINEL';
    const toolMessage = new ToolMessage({
      content: 'rendered',
      tool_call_id: toolCallId,
      name: 'render_report',
      artifact: {
        content: [
          {
            type: ContentTypes.TEXT,
            text: `${artifactSentinel}:complete`,
          },
        ],
      },
    });
    const messages: BaseMessage[] = [
      new HumanMessage('render the report'),
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: toolCallId,
            name: 'render_report',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      toolMessage,
    ];
    const transformSensitiveCounter: t.TokenCounter = (message) => {
      const content =
        typeof message.content === 'string'
          ? message.content
          : JSON.stringify(message.content);
      if (
        message instanceof HumanMessage &&
        content.includes(artifactSentinel)
      ) {
        return 10_000;
      }
      return Math.max(1, Math.ceil(content.length / 4));
    };
    const run = await createRun({
      runId: 'artifact-final-transform-guard',
      maxContextTokens: 5_000,
      maxToolResultChars: 2_000,
      provider: Providers.BEDROCK,
      model: 'anthropic.claude-sonnet-4-5',
      tokenCounter: transformSensitiveCounter,
      indexTokenCountMap: {
        0: transformSensitiveCounter(messages[0]),
        1: transformSensitiveCounter(messages[1]),
        2: transformSensitiveCounter(messages[2]),
      },
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('us.anthropic.claude-sonnet-4-5-20250929-v1:0'),
      0
    );
    run.Graph.overrideModel = model;

    await run.processStream({ messages }, streamConfig);

    expect(model.calls).toHaveLength(1);
    const providerContent = JSON.stringify(
      model.calls[0].map((message) => message.content)
    );
    expect(providerContent).toContain('[Previous tool interaction]');
    expect(providerContent).not.toContain(artifactSentinel);
    expect(toolMessage.artifact.content[0].text).toContain(artifactSentinel);
  });

  it('compacts expanded synthetic context without an artifact', async () => {
    const toolCallId = 'tc-final-transform-without-artifact';
    const messages: BaseMessage[] = [
      new HumanMessage('query the table'),
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: toolCallId,
            name: 'run_select_query',
            args: { query: `SELECT '${'x'.repeat(5_000)}'` },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'query complete',
        tool_call_id: toolCallId,
        name: 'run_select_query',
      }),
    ];
    const transformSensitiveCounter: t.TokenCounter = (message) => {
      const content =
        typeof message.content === 'string'
          ? message.content
          : JSON.stringify(message.content);
      if (
        message instanceof HumanMessage &&
        content.includes('[Previous tool interaction]')
      ) {
        return content.length * 10;
      }
      return 1;
    };
    const run = await createRun({
      runId: 'final-transform-without-artifact',
      maxContextTokens: 500,
      provider: Providers.BEDROCK,
      model: 'anthropic.claude-sonnet-4-5',
      tokenCounter: transformSensitiveCounter,
      indexTokenCountMap: { 0: 1, 1: 1, 2: 1 },
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('us.anthropic.claude-sonnet-4-5-20250929-v1:0'),
      0
    );
    run.Graph.overrideModel = model;

    await run.processStream({ messages }, streamConfig);

    expect(model.calls).toHaveLength(1);
    const humanMessages = model.calls[0].filter(
      (message) => message instanceof HumanMessage
    );
    /**
     * Bedrock is a strict-alternation provider: after compaction shrinks the
     * synthetic context, `coalesceAdjacentUserTurns` merges it into the
     * adjacent user query — Converse rejects consecutive user turns, and the
     * `@langchain/aws` converter only merges toolResult-bearing ones. So one
     * user turn reaches the provider, carrying both the query and the
     * compacted note, and it must still be small.
     */
    expect(humanMessages).toHaveLength(1);
    const coalescedContent = JSON.stringify(humanMessages[0].content);
    expect(coalescedContent).toContain('query the table');
    /**
     * No closing bracket: compaction truncates at a character budget, and the
     * cut can land inside the `[Previous tool interaction]` header itself.
     */
    expect(coalescedContent).toContain('[Previous tool interaction');
    expect(coalescedContent.length).toBeLessThan(150);
    expect(coalescedContent).not.toContain('x'.repeat(1_000));
  });

  it('counts unresolved-reference annotations before invoking the provider', async () => {
    const toolCallId = 'tc-unresolved-projection';
    const unresolvedRefs = Array.from(
      { length: 1_200 },
      (_, index) => `missing_tool_${index}_turn_${index}`
    );
    const messages: BaseMessage[] = [
      new HumanMessage(`old question ${'q'.repeat(5_500)}`),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: toolCallId,
            name: 'lookup_records',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: `old result ${'r'.repeat(3_000)}`,
        tool_call_id: toolCallId,
        name: 'lookup_records',
        additional_kwargs: { _unresolvedRefs: unresolvedRefs },
      }),
      new AIMessage(`old answer ${'a'.repeat(4_500)}`),
      new HumanMessage(`latest question ${'n'.repeat(3_500)}`),
    ];
    const projectionCounter: t.TokenCounter = (message) => {
      const content =
        typeof message.content === 'string'
          ? message.content
          : JSON.stringify(message.content);
      return Math.max(1, content.length);
    };
    const indexTokenCountMap: Record<string, number> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = projectionCounter(messages[i]);
    }
    const run = await createRun({
      runId: 'unresolved-reference-final-projection',
      maxContextTokens: 20_000,
      provider: Providers.ANTHROPIC,
      tokenCounter: projectionCounter,
      indexTokenCountMap,
      toolOutputReferences: { enabled: true },
      tools: [
        tool(async () => 'unused', {
          name: 'lookup_records',
          description: 'Looks up records',
          schema: z.object({}),
        }),
      ],
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001'),
      0
    );
    run.Graph.overrideModel = model;

    const content = await run.processStream({ messages }, streamConfig);

    expect(content).toEqual([{ type: 'text', text: 'recovered' }]);
    expect(model.calls).toHaveLength(1);
    expect(
      run.Graph.agentContexts.get('default')?.overflowRecoveryAttempts
    ).toBeGreaterThan(0);
    const providerPayload = JSON.stringify(
      model.calls[0].map((message) => message.content)
    );
    expect(providerPayload).not.toContain(unresolvedRefs[0]);
    expect(providerPayload).not.toContain(unresolvedRefs.at(-1));
    const sentMessageTokens =
      3 +
      model.calls[0].reduce(
        (total, message) => total + projectionCounter(message),
        0
      );
    expect(sentMessageTokens).toBeLessThan(
      run.Graph.agentContexts.get('default')?.maxContextTokens ?? 0
    );
  });

  it('omits artifact expansion that would exceed the post-prune budget', async () => {
    const toolCallId = 'tc-artifact';
    const artifactSentinel = 'ARTIFACT_SENTINEL';
    const toolMessage = new ToolMessage({
      content: 'result'.repeat(100),
      tool_call_id: toolCallId,
      name: 'render_report',
      artifact: {
        content: [
          {
            type: ContentTypes.TEXT,
            text: `${artifactSentinel}:${'a'.repeat(5_000)}`,
          },
        ],
      },
    });
    const messages: BaseMessage[] = [
      new HumanMessage('h'.repeat(2_400)),
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: toolCallId,
            name: 'render_report',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      toolMessage,
    ];
    const artifactTokenCounter: t.TokenCounter = (message) => {
      const content =
        typeof message.content === 'string'
          ? message.content
          : JSON.stringify(message.content);
      return Math.ceil(content.length / 4);
    };
    const indexTokenCountMap: Record<string, number> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = artifactTokenCounter(messages[i]);
    }
    const run = await createRun({
      runId: 'artifact-budget-guard',
      maxContextTokens: 1_000,
      maxToolResultChars: 2_000,
      provider: Providers.BEDROCK,
      model: 'anthropic.claude-sonnet-4-5',
      tokenCounter: artifactTokenCounter,
      indexTokenCountMap,
      tools: [
        tool(async () => 'unused', {
          name: 'render_report',
          description: 'Renders a report',
          schema: z.object({}),
        }),
      ],
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('us.anthropic.claude-sonnet-4-5-20250929-v1:0'),
      0
    );
    run.Graph.overrideModel = model;

    await run.processStream({ messages }, streamConfig);

    expect(model.calls).toHaveLength(1);
    expect(
      JSON.stringify(model.calls[0].map((message) => message.content))
    ).not.toContain(artifactSentinel);
    expect(toolMessage.content).toBe('result'.repeat(100));
    expect(toolMessage.artifact.content[0].text).toContain(artifactSentinel);
  });

  it('preserves masked tool originals while checkpointed messages survive', async () => {
    const run = await createRun({
      runId: 'overflow-originals-checkpoint',
      maxContextTokens: 1_000_000,
      checkpointer: true,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const agentContext = run.Graph.agentContexts.get('default');
    if (agentContext == null) {
      throw new Error('Expected default agent context');
    }
    run.Graph.resetValues(undefined, '8:thread-a:');
    agentContext.preserveOriginalToolContent(new Map([[2, 'full output']]));

    run.Graph.resetValues(undefined, '8:thread-a:');

    expect(agentContext.pendingOriginalToolContent).toEqual(
      new Map([[2, 'full output']])
    );
  });

  it('does not share masked tool originals across checkpoint scopes', async () => {
    const run = await createRun({
      runId: 'overflow-originals-isolated',
      maxContextTokens: 1_000_000,
      checkpointer: true,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const agentContext = run.Graph.agentContexts.get('default');
    if (agentContext == null) {
      throw new Error('Expected default agent context');
    }
    run.Graph.resetValues(undefined, '8:thread-a:branch');
    agentContext.preserveOriginalToolContent(new Map([[2, 'thread A']]));

    run.Graph.resetValues(undefined, '8:thread-b:branch');

    expect(agentContext.pendingOriginalToolContent).toBeUndefined();
  });

  it('does not share masked tool originals across sibling checkpoint forks', async () => {
    const run = await createRun({
      runId: 'overflow-originals-branch-isolated',
      maxContextTokens: 1_000_000,
      checkpointer: true,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const agentContext = run.Graph.agentContexts.get('default');
    if (agentContext == null) {
      throw new Error('Expected default agent context');
    }
    run.Graph.resetValues(
      undefined,
      JSON.stringify(['thread-a', 'namespace', 'checkpoint-a', 1])
    );
    agentContext.preserveOriginalToolContent(new Map([[2, 'branch A']]));

    run.Graph.resetValues(
      undefined,
      JSON.stringify(['thread-a', 'namespace', 'checkpoint-a', 2])
    );

    expect(agentContext.pendingOriginalToolContent).toBeUndefined();
  });

  it('preserves snapshots with the auto-installed HITL checkpointer', async () => {
    const run = await Run.create<t.IState>({
      runId: 'overflow-originals-hitl-checkpointer',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.ANTHROPIC,
          disableStreaming: true,
          streamUsage: false,
        },
        maxContextTokens: 1_000_000,
        compileOptions: { interruptAfter: [] },
      },
      humanInTheLoop: { enabled: true },
      returnContent: true,
      skipCleanup: true,
      tokenCounter,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const agentContext = run.Graph.agentContexts.get('default');
    if (agentContext == null) {
      throw new Error('Expected default agent context');
    }
    expect(run.Graph.compileOptions?.checkpointer).toBeInstanceOf(MemorySaver);
    expect(run.Graph.hasCompiledCheckpointer).toBe(true);
    run.Graph.resetValues(undefined, 'checkpoint-scope');
    agentContext.preserveOriginalToolContent(new Map([[2, 'full output']]));

    run.Graph.resetValues(undefined, 'checkpoint-scope');

    expect(agentContext.pendingOriginalToolContent).toEqual(
      new Map([[2, 'full output']])
    );
  });

  it('compacts and retries instead of surfacing the provider error', async () => {
    const run = await createRun({
      runId: 'overflow-recovery-numbers',
      /** Deliberately wrong: far above the model's real 200k window. */
      maxContextTokens: 1_000_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const accepted: t.ModelResponseEvent[] = [];
    run.Graph.handlerRegistry?.register(GraphEvents.ON_MODEL_RESPONSE, {
      handle: (_event, data): void => {
        if (data != null && 'type' in data && data.type === 'model_response') {
          accepted.push(data);
        }
      },
    });
    const model = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001')
    );
    run.Graph.overrideModel = model;

    const content = await run.processStream(
      { messages: buildConversation(8) },
      streamConfig
    );

    expect(model.calls).toHaveLength(2);
    expect(content).toEqual([{ type: 'text', text: 'recovered' }]);
    expect(accepted).toHaveLength(1);
    expect(accepted[0].id).not.toBe('');
    expect(accepted[0].toolCalls).toEqual([]);
  });

  it('retargets the budget to the ceiling the provider reported', async () => {
    const run = await createRun({
      runId: 'overflow-recovery-budget',
      maxContextTokens: 1_000_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    run.Graph.overrideModel = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001')
    );

    await run.processStream({ messages: buildConversation(8) }, streamConfig);

    const agentContext = run.Graph.agentContexts.get('default');
    expect(agentContext?.maxContextTokens).toBeLessThan(1_000_000);
    expect(agentContext?.overflowRecoveryAttempts).toBe(1);
  });

  it('sends strictly less on the retry', async () => {
    const run = await createRun({
      runId: 'overflow-recovery-shrinks',
      maxContextTokens: 1_000_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001')
    );
    run.Graph.overrideModel = model;

    await run.processStream({ messages: buildConversation(8) }, streamConfig);

    expect(model.calls[1].length).toBeLessThan(model.calls[0].length);
  });

  it('recovers from a rejection that reported no numbers', async () => {
    const run = await createRun({
      runId: 'overflow-recovery-blind',
      maxContextTokens: 600_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
    );
    run.Graph.overrideModel = model;

    const content = await run.processStream(
      { messages: buildConversation(8) },
      streamConfig
    );

    expect(model.calls).toHaveLength(2);
    expect(content).toEqual([{ type: 'text', text: 'recovered' }]);
  });

  it('gives up after the bounded number of recoveries rather than looping', async () => {
    const run = await createRun({
      runId: 'overflow-recovery-bounded',
      maxContextTokens: 600_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('us.anthropic.claude-sonnet-4-5-20250929-v1:0'),
      Number.MAX_SAFE_INTEGER
    );
    run.Graph.overrideModel = model;

    await expect(
      run.processStream({ messages: buildConversation(8) }, streamConfig)
    ).rejects.toThrow(/too long/i);

    /** Initial call plus one retry per allowed recovery. */
    expect(model.calls.length).toBeLessThanOrEqual(4);
  });

  it('restores the configured budget and allowance for the next run', async () => {
    const run = await createRun({
      runId: 'overflow-recovery-reset',
      maxContextTokens: 1_000_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    run.Graph.overrideModel = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001')
    );

    await run.processStream({ messages: buildConversation(8) }, streamConfig);

    const agentContext = run.Graph.agentContexts.get('default');
    expect(agentContext?.maxContextTokens).toBeLessThan(1_000_000);
    expect(agentContext?.overflowRecoveryAttempts).toBe(1);

    /** What `processStream` runs at the start of every turn. */
    run.Graph.resetValues();

    expect(agentContext?.maxContextTokens).toBe(1_000_000);
    expect(agentContext?.overflowRecoveryAttempts).toBe(0);
  });

  it('recovers again on a later turn of the same run', async () => {
    const run = await createRun({
      runId: 'overflow-recovery-second-turn',
      maxContextTokens: 1_000_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const signature = signatureFor('claude-haiku-4-5-20251001');

    for (const turn of [1, 2, 3]) {
      run.Graph.overrideModel = new OverflowThenSucceedModel(signature);
      await run.processStream(
        { messages: buildConversation(8) },
        {
          ...streamConfig,
          configurable: { thread_id: `context-overflow-recovery-${turn}` },
        }
      );
    }

    const agentContext = run.Graph.agentContexts.get('default');
    /**
     * One recovery per turn, not three accumulated — an allowance that
     * carried across turns would have stopped recovering by turn three.
     */
    expect(agentContext?.overflowRecoveryAttempts).toBe(1);
  });

  it('does not retry when neither pruning nor summarization can shrink the prompt', async () => {
    /**
     * No token counter means no pruner, and summarization is off, so the
     * summarize node deliberately no-ops — a retry would resend a
     * byte-identical prompt.
     */
    const run = await Run.create<t.IState>({
      runId: 'overflow-recovery-nothing-to-shrink',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.ANTHROPIC,
          disableStreaming: true,
          streamUsage: false,
        },
        maxContextTokens: 1_000_000,
      },
      returnContent: true,
      skipCleanup: true,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001'),
      Number.MAX_SAFE_INTEGER
    );
    run.Graph.overrideModel = model;

    await expect(
      run.processStream({ messages: buildConversation(8) }, streamConfig)
    ).rejects.toThrow(/too long/i);

    expect(model.calls).toHaveLength(1);
    expect(
      run.Graph.agentContexts.get('default')?.overflowRecoveryAttempts
    ).toBe(0);
  });

  it('does not retry when recency preserves the entire first turn', async () => {
    const run = await Run.create<t.IState>({
      runId: 'overflow-recovery-recent-first-turn',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.ANTHROPIC,
          disableStreaming: true,
          streamUsage: false,
        },
        summarizationEnabled: true,
      },
      returnContent: true,
      skipCleanup: true,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('us.anthropic.claude-sonnet-4-5-20250929-v1:0'),
      Number.MAX_SAFE_INTEGER
    );
    run.Graph.overrideModel = model;

    await expect(
      run.processStream(
        { messages: [new HumanMessage('oversized first turn')] },
        streamConfig
      )
    ).rejects.toThrow(/too long/i);

    expect(model.calls).toHaveLength(1);
    expect(
      run.Graph.agentContexts.get('default')?.overflowRecoveryAttempts
    ).toBe(0);
  });

  it('does not retry without a configured pruning budget', async () => {
    const run = await Run.create<t.IState>({
      runId: 'overflow-recovery-no-pruning-budget',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.ANTHROPIC,
          disableStreaming: true,
          streamUsage: false,
        },
      },
      returnContent: true,
      skipCleanup: true,
      tokenCounter,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('us.anthropic.claude-sonnet-4-5-20250929-v1:0'),
      Number.MAX_SAFE_INTEGER
    );
    run.Graph.overrideModel = model;

    await expect(
      run.processStream({ messages: buildConversation(8) }, streamConfig)
    ).rejects.toThrow(/too long/i);

    expect(model.calls).toHaveLength(1);
    expect(
      run.Graph.agentContexts.get('default')?.overflowRecoveryAttempts
    ).toBe(0);
  });

  it('summarizes the first numberless overflow without a pruning budget', async () => {
    const summarizeStarts: unknown[] = [];
    const run = await Run.create<t.IState>({
      runId: 'overflow-recovery-summary-without-budget',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.ANTHROPIC,
          disableStreaming: true,
          streamUsage: false,
        },
        summarizationEnabled: true,
      },
      returnContent: true,
      skipCleanup: true,
      tokenCounter,
      customHandlers: {
        [GraphEvents.ON_SUMMARIZE_START]: {
          handle: (_event: string, data: t.StreamEventData): void => {
            summarizeStarts.push(data);
          },
        },
      },
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
    );
    run.Graph.overrideModel = model;

    const content = await run.processStream(
      { messages: buildConversation(8) },
      streamConfig
    );

    expect(content).toEqual([{ type: 'text', text: 'recovered' }]);
    expect(summarizeStarts).toHaveLength(1);
  });

  it('summarizes conversation messages pruned on the first recovery', async () => {
    /**
     * The initial overflow detour avoids an unconditional summarization call,
     * but the corrected prune must still summarize any conversation messages
     * it moves out of the retained tail.
     */
    const summarizeStarts: unknown[] = [];
    const run = await Run.create<t.IState>({
      runId: 'overflow-recovery-compress-first',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.ANTHROPIC,
          disableStreaming: true,
          streamUsage: false,
        },
        maxContextTokens: 1_000_000,
        summarizationEnabled: true,
      },
      returnContent: true,
      skipCleanup: true,
      tokenCounter,
      customHandlers: {
        [GraphEvents.ON_SUMMARIZE_START]: {
          handle: (_event: string, data: t.StreamEventData): void => {
            summarizeStarts.push(data);
          },
        },
      },
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001')
    );
    run.Graph.overrideModel = model;

    const content = await run.processStream(
      { messages: buildConversation(8) },
      streamConfig
    );

    expect(content).toEqual([{ type: 'text', text: 'recovered' }]);
    expect(model.calls).toHaveLength(2);
    expect(summarizeStarts).toHaveLength(1);
  });

  it('keeps the corrected budget in provider units and seeds calibration', async () => {
    /**
     * The provider counted 274,468 for a prompt it caps at 200,000 — 1.37×
     * over the provider ceiling. The corrected budget stays in provider units,
     * and the provider/local ratio is installed on the pruner so it is applied
     * exactly once on this retry and later tool-call turns.
     */
    const run = await createRun({
      runId: 'overflow-recovery-proportional',
      maxContextTokens: 1_000_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001')
    );
    run.Graph.overrideModel = model;

    const messages = buildConversation(8);
    const agentContext = run.Graph.agentContexts.get('default');
    if (agentContext == null) {
      throw new Error('Expected default agent context');
    }
    agentContext.calibrationRatio = 1.5;
    await run.processStream({ messages }, streamConfig);

    const uncalibratedPrompt = model.calls[0].length * TOKENS_PER_MESSAGE;
    expect(agentContext.maxContextTokens).toBe(Math.floor(200_000 * 0.95));
    expect(274_468 / uncalibratedPrompt).toBeLessThan(0.5);
    expect(agentContext.calibrationRatio).toBe(0.5);
  });

  it('does not intercept errors compaction cannot fix', async () => {
    const run = await createRun({
      runId: 'overflow-recovery-unrelated',
      maxContextTokens: 600_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      {
        name: 'AuthenticationError',
        status: 401,
        message: '401 Incorrect API key provided.',
      },
      Number.MAX_SAFE_INTEGER
    );
    run.Graph.overrideModel = model;

    await expect(
      run.processStream({ messages: buildConversation(2) }, streamConfig)
    ).rejects.toThrow(/Incorrect API key/);
    expect(model.calls).toHaveLength(1);
  });
});
