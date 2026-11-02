import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { HumanMessage } from '@langchain/core/messages';
import { LangfuseOtelSpanAttributes } from '@langfuse/tracing';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import { context as otelContext, trace as otelTrace } from '@opentelemetry/api';
import type { SpanProcessor } from '@opentelemetry/sdk-trace-base';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { Context } from '@opentelemetry/api';
import type { StopHookOutput } from '@/hooks';
import type * as t from '@/types';
import {
  registerLangfuseManagedSpan,
  resolveLangfuseDestinationKey,
  resolveLangfuseTraceAnchorParent,
} from '@/langfuseSpanRegistry';
import {
  Constants,
  ContentTypes,
  GraphEvents,
  Providers,
  TitleMethod,
} from '@/common';
import { LANGFUSE_OBSERVATION_METADATA_ARTIFACT_KEY } from '@/langfuseToolOutputTracing';
import { withLangfuseRuntimeScope } from '@/langfuseRuntimeScope';
import { initializeLangfuseTracing } from '@/instrumentation';
import { traceIdFromSeed } from '@/langfuseRuntimeContext';
import { createLangfuseHandler } from '@/langfuse';
import * as providers from '@/llm/providers';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

type ProcessorParams = {
  publicKey?: string;
  secretKey?: string;
  baseUrl?: string;
};

type SpanStartRecord = {
  name: string;
  params: ProcessorParams;
  traceId: string;
  spanId: string;
  parentSpanId?: string;
};

const spanStarts: SpanStartRecord[] = [];
const endedSpans: MockSpan[] = [];
let providerInput:
  | {
      spanProcessors?: SpanProcessor[];
      idGenerator?: {
        generateTraceId: () => string;
        generateSpanId: () => string;
      };
    }
  | undefined;

type MockSpan = {
  name: string;
  attributes: Record<string, unknown>;
  addEvent: jest.Mock;
  end: jest.Mock;
  isRecording: jest.Mock;
  recordException: jest.Mock;
  setAttribute: jest.Mock;
  setAttributes: jest.Mock;
  setStatus: jest.Mock;
  spanContext: jest.Mock;
  updateName: jest.Mock;
};

function isOtelContext(value: unknown): value is Context {
  return (
    value != null &&
    typeof value === 'object' &&
    typeof (value as { getValue?: unknown }).getValue === 'function'
  );
}

function createMockSpan(
  name: string,
  parentContext: Context = otelContext.active()
): MockSpan {
  const parentTraceId = otelTrace.getSpanContext(parentContext)?.traceId;
  const traceId =
    parentTraceId ??
    providerInput?.idGenerator?.generateTraceId() ??
    'trace-id';
  const spanId = providerInput?.idGenerator?.generateSpanId() ?? 'span-id';
  const attributes: Record<string, unknown> = {};
  const span = {} as MockSpan;
  Object.assign(span, {
    name,
    attributes,
    addEvent: jest.fn(),
    end: jest.fn(() => {
      endedSpans.push(span);
      for (const processor of providerInput?.spanProcessors ?? []) {
        processor.onEnd(span as never);
      }
    }),
    isRecording: jest.fn(() => true),
    recordException: jest.fn(),
    setAttribute: jest.fn((key: string, value: unknown) => {
      attributes[key] = value;
      return span;
    }),
    setAttributes: jest.fn((next: Record<string, unknown>) => {
      Object.assign(attributes, next);
      return span;
    }),
    setStatus: jest.fn(),
    spanContext: jest.fn(() => ({
      traceId,
      spanId,
      traceFlags: 1,
    })),
    updateName: jest.fn((nextName: string) => {
      span.name = nextName;
      return span;
    }),
  });

  for (const processor of providerInput?.spanProcessors ?? []) {
    processor.onStart(span as never, parentContext);
  }
  return span;
}

const startSpan = jest.fn((name: string, _options?: unknown, ctx?: unknown) =>
  createMockSpan(name, isOtelContext(ctx) ? ctx : otelContext.active())
);

function getParentContextFromStartActiveSpanArgs(args: unknown[]): Context {
  if (args.length >= 3 && isOtelContext(args[1])) {
    return args[1];
  }
  if (args.length >= 4 && isOtelContext(args[2])) {
    return args[2];
  }
  return otelContext.active();
}

const startActiveSpan = jest.fn((name: string, ...args: unknown[]) => {
  const callback = args[args.length - 1];
  const parentContext = getParentContextFromStartActiveSpanArgs(args);

  if (typeof callback !== 'function') {
    throw new Error('startActiveSpan mock expected a callback');
  }

  const span = createMockSpan(name, parentContext);
  const activeContext = otelTrace.setSpan(parentContext, span as never);
  return otelContext.with(activeContext, () => callback(span));
});

jest.mock('@langfuse/otel', () => ({
  LangfuseSpanProcessor: jest.fn().mockImplementation((params) => ({
    forceFlush: jest.fn(),
    onEnd: jest.fn(),
    onStart: jest.fn((span, parentContext) => {
      const spanContext = span.spanContext();
      const parentSpanId = otelTrace.getSpanContext(parentContext)?.spanId;
      spanStarts.push({
        name: span.name,
        params,
        traceId: spanContext.traceId,
        spanId: spanContext.spanId,
        ...(parentSpanId != null ? { parentSpanId } : {}),
      });
    }),
    shutdown: jest.fn(),
  })),
  isDefaultExportSpan: jest.fn(() => false),
}));

jest.mock('@opentelemetry/sdk-trace-base', () => ({
  BasicTracerProvider: jest.fn().mockImplementation((input) => {
    providerInput = input;
    return {
      forceFlush: jest.fn(),
      getTracer: jest.fn(() => ({
        startActiveSpan,
        startSpan,
      })),
      shutdown: jest.fn(),
    };
  }),
}));

const echoTool = tool(async ({ text }) => `echo:${text}`, {
  name: 'echo',
  description: 'Echoes text for routing tests.',
  schema: z.object({ text: z.string() }),
});

const callerConfig: Partial<RunnableConfig> & {
  version: 'v1' | 'v2';
  streamMode: string;
} = {
  configurable: { thread_id: 'routing-thread', user_id: 'routing-user' },
  streamMode: 'values',
  version: 'v2',
};

function tenantLangfuse(tenantId: string): t.LangfuseConfig {
  return {
    enabled: true,
    publicKey: `pk-${tenantId}`,
    secretKey: `sk-${tenantId}`,
    baseUrl: 'https://langfuse.proxy',
    deterministicTraceId: true,
    metadata: { tenantId },
    tags: [`tenant:${tenantId}`],
    toolOutputTracing: { enabled: true },
  };
}

function startsForTenant(tenantId: string): SpanStartRecord[] {
  return spanStarts.filter(
    (record) => record.params.publicKey === `pk-${tenantId}`
  );
}

function expectTenantCredentials(
  starts: SpanStartRecord[],
  tenantId: string
): void {
  expect(starts).toEqual(
    expect.arrayContaining([
      expect.objectContaining({
        params: expect.objectContaining({
          publicKey: `pk-${tenantId}`,
          secretKey: `sk-${tenantId}`,
          baseUrl: 'https://langfuse.proxy',
        }),
      }),
    ])
  );
}

function expectNamedSpansUseTraceId({
  starts,
  names,
  traceId,
}: {
  starts: SpanStartRecord[];
  names: string[];
  traceId: string;
}): void {
  for (const name of names) {
    const matching = starts.filter((record) => record.name === name);
    expect(matching).not.toHaveLength(0);
    expect(matching.map((record) => record.traceId)).toEqual(
      expect.arrayContaining([traceId])
    );
    expect(
      matching.filter((record) => record.traceId !== traceId)
    ).toHaveLength(0);
  }
}

function expectChildSpanParentName({
  starts,
  childName,
  parentNamePrefix,
}: {
  starts: SpanStartRecord[];
  childName: string;
  parentNamePrefix: string;
}): void {
  const children = starts.filter((record) => record.name === childName);
  expect(children).not.toHaveLength(0);
  for (const child of children) {
    const parent = starts.find(
      (record) => record.spanId === child.parentSpanId
    );
    expect(parent?.name.startsWith(parentNamePrefix)).toBe(true);
  }
}

function expectOnlyTraceIds(
  starts: SpanStartRecord[],
  allowedTraceIds: string[]
): void {
  const allowed = new Set(allowedTraceIds);
  expect(starts.filter((record) => !allowed.has(record.traceId))).toHaveLength(
    0
  );
}

function expectNoCrossTenantTrace({
  tenantId,
  otherTenantId,
  traceId,
}: {
  tenantId: string;
  otherTenantId: string;
  traceId: string;
}): void {
  expect(
    startsForTenant(otherTenantId).filter(
      (record) => record.traceId === traceId
    )
  ).toHaveLength(0);
  expect(startsForTenant(tenantId)).toEqual(
    expect.arrayContaining([expect.objectContaining({ traceId })])
  );
}

function createAgent(tenantId: string): t.AgentInputs {
  return {
    agentId: 'parent',
    name: `Parent ${tenantId}`,
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
    instructions: 'Use tools when asked.',
    maxContextTokens: 8000,
    tools: [echoTool],
    subagentConfigs: [
      {
        type: 'researcher',
        name: 'Researcher',
        description: 'Answers delegated research tasks.',
        agentInputs: {
          agentId: 'researcher',
          name: `Researcher ${tenantId}`,
          provider: Providers.OPENAI,
          clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
          instructions: 'Answer delegated tasks briefly.',
          maxContextTokens: 8000,
        },
      },
    ],
  };
}

function createGraphSubagentParent(tenantId: string): t.AgentInputs {
  const makeMember = (agentId: string): t.AgentInputs => ({
    agentId,
    name: `${agentId} ${tenantId}`,
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
    instructions: `Complete the ${agentId} stage.`,
    maxContextTokens: 8000,
  });
  return {
    ...makeMember('parent'),
    maxSubagentDepth: 1,
    subagentConfigs: [
      {
        kind: 'graph',
        type: 'research-team',
        name: 'Research Team',
        description: 'Runs a bounded research chain.',
        agents: [
          makeMember('entry'),
          makeMember('worker'),
          makeMember('result'),
        ],
        edges: [
          {
            from: 'entry',
            to: 'worker',
            edgeType: 'direct',
            prompt: 'Proceed to worker.',
          },
          {
            from: 'worker',
            to: 'result',
            edgeType: 'direct',
            prompt: 'Proceed to result.',
          },
        ],
        entryAgentId: 'entry',
        resultAgentId: 'result',
      },
    ],
  };
}

async function runTenantFlow(tenantId: string): Promise<void> {
  const runId = `routing-${tenantId}`;
  const run = await Run.create<t.IState>({
    runId,
    graphConfig: {
      type: 'standard',
      agents: [createAgent(tenantId)],
    },
    langfuse: tenantLangfuse(tenantId),
    returnContent: true,
    skipCleanup: true,
  });

  const toolCalls: ToolCall[] = [
    {
      id: `call_echo_${tenantId}`,
      name: 'echo',
      args: { text: `hello ${tenantId}` },
      type: 'tool_call',
    },
    {
      id: `call_subagent_${tenantId}`,
      name: Constants.SUBAGENT,
      args: {
        description: `Research ${tenantId}`,
        subagent_type: 'researcher',
      },
      type: 'tool_call',
    },
  ];
  run.Graph?.overrideTestModel(
    [`Using tools for ${tenantId}.`, `Final answer for ${tenantId}.`],
    1,
    toolCalls
  );

  await run.processStream(
    { messages: [new HumanMessage(`Use tools for ${tenantId}`)] },
    {
      ...callerConfig,
      configurable: {
        thread_id: `thread-${tenantId}`,
        user_id: `user-${tenantId}`,
      },
    }
  );

  await run.generateTitle({
    provider: Providers.OPENAI,
    inputText: `Use tools for ${tenantId}`,
    titleMethod: TitleMethod.COMPLETION,
    contentParts: [
      { type: ContentTypes.TEXT, text: `Final answer for ${tenantId}.` },
    ],
    chainOptions: {
      configurable: {
        thread_id: `thread-${tenantId}`,
        user_id: `user-${tenantId}`,
      },
    },
  });
}

async function runGraphSubagentTenantFlow(tenantId: string): Promise<void> {
  const runId = `routing-graph-${tenantId}`;
  const run = await Run.create<t.IState>({
    runId,
    graphConfig: {
      type: 'standard',
      agents: [createGraphSubagentParent(tenantId)],
    },
    langfuse: tenantLangfuse(tenantId),
    returnContent: true,
    skipCleanup: true,
  });
  run.Graph?.overrideTestModel(
    [
      `Delegating graph work for ${tenantId}.`,
      `Graph work complete for ${tenantId}.`,
    ],
    1,
    [
      {
        id: `call_graph_subagent_${tenantId}`,
        name: Constants.SUBAGENT,
        args: {
          description: `Run the research chain for ${tenantId}.`,
          subagent_type: 'research-team',
        },
        type: 'tool_call',
      },
    ]
  );

  await run.processStream(
    { messages: [new HumanMessage(`Run graph work for ${tenantId}`)] },
    {
      ...callerConfig,
      configurable: {
        thread_id: `graph-thread-${tenantId}`,
        user_id: `graph-user-${tenantId}`,
      },
    }
  );
}

const compactingTokenCounter: t.TokenCounter = (message) => {
  if (message._getType() === 'system') {
    return 1;
  }
  const content = message.content;
  return typeof content === 'string'
    ? content.length
    : JSON.stringify(content).length;
};

async function runTenantSummarizationFlow(tenantId: string): Promise<void> {
  const runId = `routing-summary-${tenantId}`;
  const run = await Run.create<t.IState>({
    runId,
    graphConfig: {
      type: 'standard',
      agents: [
        {
          agentId: 'parent',
          name: `Parent ${tenantId}`,
          provider: Providers.OPENAI,
          clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
          instructions: 'Summarize when context is full.',
          /* Under `compactingTokenCounter` a message costs its character
           * count, so this sits between the summary's own carrier (~272) and
           * the transcript (~397): high enough that the run survives its
           * checkpoint, low enough that compaction still fires. */
          maxContextTokens: 340,
          summarizationEnabled: true,
          summarizationConfig: {
            prompt: 'Write a checkpoint.',
            maxSummaryTokens: 32,
            retainRecent: { turns: 0 },
          },
        },
      ],
    },
    langfuse: tenantLangfuse(tenantId),
    tokenCounter: compactingTokenCounter,
    returnContent: true,
    skipCleanup: true,
  });

  run.Graph?.overrideTestModel([`After summary for ${tenantId}.`], 1);

  await run.processStream(
    {
      messages: [
        new HumanMessage(`${tenantId} old context `.repeat(8)),
        new HumanMessage(`${tenantId} more old context `.repeat(8)),
        new HumanMessage(`Continue for ${tenantId}`),
      ],
    },
    {
      ...callerConfig,
      configurable: {
        thread_id: `summary-thread-${tenantId}`,
        user_id: `summary-user-${tenantId}`,
      },
    }
  );
}

describe('Langfuse per-run routing integration', () => {
  let getChatModelClassSpy: jest.SpyInstance;
  const originalGetChatModelClass = providers.getChatModelClass;

  beforeEach(() => {
    jest.clearAllMocks();
    spanStarts.length = 0;
    endedSpans.length = 0;
    delete process.env.LANGFUSE_PUBLIC_KEY;
    delete process.env.LANGFUSE_SECRET_KEY;
    delete process.env.LANGFUSE_BASE_URL;
    delete process.env.LANGFUSE_BASEURL;
    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return class RoutingProviderFakeChatModel extends FakeListChatModel {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            constructor(_options: any) {
              super({ responses: ['provider response'] });
            }
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
          } as any;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);
  });

  afterEach(() => {
    getChatModelClassSpy.mockRestore();
  });

  it('keeps tool observations attached to the exported dispatch parent', async () => {
    await runTenantFlow('tenant-hierarchy');

    const starts = startsForTenant('tenant-hierarchy');
    expect(starts.some((record) => record.name === 'tool_batch')).toBe(false);
    expectChildSpanParentName({
      starts,
      childName: 'echo',
      parentNamePrefix: 'tools=',
    });
    expectChildSpanParentName({
      starts,
      childName: 'subagent',
      parentNamePrefix: 'tools=',
    });
  });

  it.each([false, true])(
    'routes parallel host-executed queries with metadata and redaction (eager=%s)',
    async (eager) => {
      const otherToolEnd = jest.fn();
      await Promise.all(
        ['host-a', 'host-b', 'host-disabled'].map(async (tenantId, index) => {
          const queryTool = tool(
            async () => [
              'private rows',
              {
                privateAttachment: 'private attachment',
                [LANGFUSE_OBSERVATION_METADATA_ARTIFACT_KEY]: {
                  query_returned_rows: index + 1,
                },
              },
            ],
            {
              name: 'query',
              description: 'Query fixture',
              schema: z.object({}),
              responseFormat: 'content_and_artifact',
            }
          );
          const run = await Run.create<t.IState>({
            runId: `routing-${tenantId}`,
            graphConfig: {
              type: 'standard',
              llmConfig: {
                provider: Providers.OPENAI,
                streaming: true,
                streamUsage: false,
              },
              toolDefinitions: [
                {
                  name: 'failed_query',
                  description: 'Failed query fixture',
                  parameters: { type: 'object', properties: {} },
                },
                {
                  name: 'query',
                  description: 'Query fixture',
                  parameters: { type: 'object', properties: {} },
                },
                {
                  name: 'unannotated',
                  description: 'Tool without safe metadata',
                  parameters: { type: 'object', properties: {} },
                },
              ],
            },
            langfuse: {
              ...tenantLangfuse(tenantId),
              enabled: tenantId !== 'host-disabled',
              toolOutputTracing: { enabled: index === 1 },
            },
            eagerEventToolExecution: { enabled: eager },
            returnContent: true,
            skipCleanup: true,
            customHandlers: {
              [GraphEvents.ON_TOOL_EXECUTE]: {
                handle: async (_event, rawData) => {
                  const data = rawData as t.ToolExecuteBatchRequest;
                  data.resolve(
                    await Promise.all(
                      data.toolCalls.map(async (call) => {
                        if (call.name === 'failed_query') {
                          return {
                            toolCallId: call.id,
                            status: 'error' as const,
                            content: 'private failed output',
                            errorMessage: 'private host error',
                            artifact: {
                              [LANGFUSE_OBSERVATION_METADATA_ARTIFACT_KEY]: {
                                query_error_category: 'timeout',
                              },
                            },
                          };
                        }
                        if (call.name === 'unannotated') {
                          return {
                            toolCallId: call.id,
                            status: 'success' as const,
                            content: 'private unannotated output',
                          };
                        }
                        const result = await queryTool.invoke(
                          {
                            name: call.name,
                            args: call.args,
                            id: call.id,
                            type: 'tool_call',
                          },
                          { callbacks: [], metadata: data.metadata }
                        );
                        return {
                          toolCallId: call.id,
                          status: 'success' as const,
                          content: result.content,
                          artifact: result.artifact,
                        };
                      })
                    )
                  );
                },
              },
            },
          });
          run.Graph?.overrideTestModel(['Querying', 'Done'], 1, [
            {
              id: `failed-${tenantId}`,
              name: 'failed_query',
              args: {},
              type: 'tool_call',
            },
            {
              id: `call-${tenantId}`,
              name: 'query',
              args: {},
              type: 'tool_call',
            },
            {
              id: `plain-${tenantId}`,
              name: 'unannotated',
              args: {},
              type: 'tool_call',
            },
          ]);
          await run.processStream(
            { messages: [new HumanMessage('Query')] },
            {
              ...callerConfig,
              callbacks: [
                { name: 'other-observer', handleToolEnd: otherToolEnd },
              ],
            }
          );
        })
      );
      expect(startsForTenant('host-disabled')).toHaveLength(0);
      expect(
        spanStarts.filter((span) => span.name === 'unannotated')
      ).toHaveLength(0);
      expect(otherToolEnd).not.toHaveBeenCalled();
      for (const [index, tenantId] of ['host-a', 'host-b'].entries()) {
        const failedStarts = startsForTenant(tenantId).filter(
          (span) => span.name === 'failed_query'
        );
        expect(failedStarts).toHaveLength(1);
        expect(failedStarts[0].traceId).toBe(
          traceIdFromSeed(`routing-${tenantId}`)
        );
        const failed = endedSpans.find(
          (entry) => entry.spanContext().spanId === failedStarts[0].spanId
        );
        expect(
          failed?.attributes[LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]
        ).toBe('ERROR');
        expect(
          failed?.attributes[
            LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE
          ]
        ).toBe('Error: Host tool execution failed');
        expect(JSON.stringify(failed?.attributes)).not.toContain('private');
        expect(
          failed?.attributes[
            'langfuse.observation.metadata.query_error_category'
          ]
        ).toContain('timeout');
        const starts = startsForTenant(tenantId).filter(
          (span) => span.name === 'query'
        );
        expect(starts).toHaveLength(1);
        expect(starts[0].traceId).toBe(traceIdFromSeed(`routing-${tenantId}`));
        const span = endedSpans.find(
          (entry) => entry.spanContext().spanId === starts[0].spanId
        );
        expect(
          span?.attributes[LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]
        ).not.toBe('ERROR');
        expect(
          JSON.parse(
            String(
              span?.attributes[
                'langfuse.observation.metadata.query_returned_rows'
              ]
            )
          )
        ).toBe(index + 1);
        expect(span?.attributes['langfuse.observation.output']).not.toContain(
          'private rows'
        );
        expect(JSON.stringify(span?.attributes)).not.toContain(
          'private attachment'
        );
      }
    }
  );

  it('keeps caller run names from replacing the graph operation name', async () => {
    const tenantId = 'tenant-caller-run-name';
    const run = await Run.create<t.IState>({
      runId: `routing-${tenantId}`,
      graphConfig: {
        type: 'standard',
        agents: [createAgent(tenantId)],
      },
      langfuse: tenantLangfuse(tenantId),
      returnContent: true,
      skipCleanup: true,
    });
    run.Graph?.overrideTestModel(['Caller name preserved.'], 1);

    await run.processStream(
      { messages: [new HumanMessage('Keep the operation name stable.')] },
      {
        ...callerConfig,
        runName: 'Host Agent Trace',
        configurable: {
          thread_id: `thread-${tenantId}`,
          user_id: `user-${tenantId}`,
        },
      }
    );

    const starts = startsForTenant(tenantId);
    expect(starts.some(({ name }) => name === 'AgentGraph')).toBe(true);
    expect(starts.some(({ name }) => name === 'Host Agent Trace')).toBe(false);
  });

  it('routes parallel root, model, tool, subagent, and title spans to each run config', async () => {
    await Promise.all([runTenantFlow('tenant-a'), runTenantFlow('tenant-b')]);

    for (const tenantId of ['tenant-a', 'tenant-b']) {
      const starts = startsForTenant(tenantId);
      const otherTenantId = tenantId === 'tenant-a' ? 'tenant-b' : 'tenant-a';
      const runTraceId = traceIdFromSeed(`routing-${tenantId}`);
      const titleTraceId = traceIdFromSeed(`title-routing-${tenantId}`);

      expectTenantCredentials(starts, tenantId);
      expectNamedSpansUseTraceId({
        starts,
        traceId: runTraceId,
        names: [
          'AgentGraph',
          'AgentModelCall',
          'FakeChatModel',
          'echo',
          'subagent',
        ],
      });
      expect(starts.some((record) => record.name === 'tool_batch')).toBe(false);
      expectChildSpanParentName({
        starts,
        childName: 'echo',
        parentNamePrefix: 'tools=',
      });
      expectNamedSpansUseTraceId({
        starts,
        traceId: titleTraceId,
        names: [`LibreChat Title: Parent ${tenantId}`, 'GenerateTitle'],
      });
      expectNoCrossTenantTrace({
        tenantId,
        otherTenantId,
        traceId: runTraceId,
      });
      expectNoCrossTenantTrace({
        tenantId,
        otherTenantId,
        traceId: titleTraceId,
      });
    }
  });

  it('keeps graph-subagent member spans in the owning tenant trace', async () => {
    await Promise.all([
      runGraphSubagentTenantFlow('graph-tenant-a'),
      runGraphSubagentTenantFlow('graph-tenant-b'),
    ]);

    for (const tenantId of ['graph-tenant-a', 'graph-tenant-b']) {
      const otherTenantId =
        tenantId === 'graph-tenant-a' ? 'graph-tenant-b' : 'graph-tenant-a';
      const starts = startsForTenant(tenantId);
      const traceId = traceIdFromSeed(`routing-graph-${tenantId}`);

      expectTenantCredentials(starts, tenantId);
      expectOnlyTraceIds(starts, [traceId]);
      expectNamedSpansUseTraceId({
        starts,
        traceId,
        names: ['AgentGraph', 'AgentModelCall', 'FakeChatModel', 'subagent'],
      });
      expect(
        starts.filter(
          (record) => record.name === 'RoutingProviderFakeChatModel'
        )
      ).toHaveLength(3);
      expectNoCrossTenantTrace({ tenantId, otherTenantId, traceId });
    }
  });

  it('routes parallel summarization spans to each run config', async () => {
    await Promise.all([
      runTenantSummarizationFlow('tenant-a'),
      runTenantSummarizationFlow('tenant-b'),
    ]);

    for (const tenantId of ['tenant-a', 'tenant-b']) {
      const otherTenantId = tenantId === 'tenant-a' ? 'tenant-b' : 'tenant-a';
      const starts = startsForTenant(tenantId);
      const summaryTraceId = traceIdFromSeed(`routing-summary-${tenantId}`);

      expectTenantCredentials(starts, tenantId);
      expectOnlyTraceIds(starts, [summaryTraceId]);
      expectNamedSpansUseTraceId({
        starts,
        traceId: summaryTraceId,
        names: [
          'AgentGraph',
          'summarize=parent',
          'summarization:cache_hit_compaction',
          'FakeChatModel',
        ],
      });
      expectNoCrossTenantTrace({
        tenantId,
        otherTenantId,
        traceId: summaryTraceId,
      });
    }
  });

  it('detaches root observations from foreign ambient spans', async () => {
    const tenantId = 'tenant-ambient';
    const foreignTraceId = 'f0e1d2c3b4a5968778695a4b3c2d1e0f';
    const foreignSpanId = 'a1b2c3d4e5f60718';
    initializeLangfuseTracing(tenantLangfuse(tenantId));

    // Simulates a host running agent code inside its own OpenTelemetry span
    // (HTTP server auto-instrumentation on the global provider): the span is
    // never exported to Langfuse, so inheriting it would orphan the trace
    // root, merge concurrent runs in one request into a single trace, and
    // bypass deterministic trace ids.
    const foreignSpan = otelTrace.wrapSpanContext({
      traceId: foreignTraceId,
      spanId: foreignSpanId,
      traceFlags: 1,
    });
    await otelContext.with(
      otelTrace.setSpan(otelContext.active(), foreignSpan),
      () => runTenantFlow(tenantId)
    );

    const starts = startsForTenant(tenantId);
    expect(starts).not.toHaveLength(0);
    expect(
      starts.filter(
        (record) =>
          record.traceId === foreignTraceId ||
          record.parentSpanId === foreignSpanId
      )
    ).toHaveLength(0);

    const agentRoot = starts.find((record) => record.name === 'AgentGraph');
    expect(agentRoot?.traceId).toBe(traceIdFromSeed(`routing-${tenantId}`));
    expect(agentRoot?.parentSpanId).toBeUndefined();

    const titleRoot = starts.find(
      (record) => record.name === `LibreChat Title: Parent ${tenantId}`
    );
    expect(titleRoot?.traceId).toBe(
      traceIdFromSeed(`title-routing-${tenantId}`)
    );
    expect(titleRoot?.parentSpanId).toBeUndefined();
  });

  it('keeps root observations nested under Langfuse-managed ambient spans', async () => {
    const tenantId = 'tenant-managed';
    const langfuse = tenantLangfuse(tenantId);
    initializeLangfuseTracing(langfuse);

    let hostSpan: MockSpan | undefined;
    await withLangfuseRuntimeScope({ langfuse }, async () => {
      hostSpan = createMockSpan('host-group');
      await otelContext.with(
        otelTrace.setSpan(otelContext.active(), hostSpan as never),
        () => runTenantFlow(tenantId)
      );
    });

    const hostSpanContext = hostSpan?.spanContext() as {
      traceId: string;
      spanId: string;
    };
    const agentRoot = startsForTenant(tenantId).find(
      (record) => record.name === 'AgentGraph'
    );
    expect(agentRoot?.traceId).toBe(hostSpanContext.traceId);
    expect(agentRoot?.parentSpanId).toBe(hostSpanContext.spanId);
  });

  it('detaches root observations from managed ambient spans of another destination', async () => {
    const hostTenantId = 'tenant-managed-a';
    const runTenantId = 'tenant-managed-b';
    const hostLangfuse = tenantLangfuse(hostTenantId);
    initializeLangfuseTracing(hostLangfuse);
    initializeLangfuseTracing(tenantLangfuse(runTenantId));

    let hostSpan: MockSpan | undefined;
    await withLangfuseRuntimeScope({ langfuse: hostLangfuse }, async () => {
      hostSpan = createMockSpan('host-group');
    });

    // A managed span is only a safe parent for runs exporting to the SAME
    // destination; nesting tenant-B under tenant-A's span would leave B's
    // trace dangling in B's project with A's trace id.
    await otelContext.with(
      otelTrace.setSpan(otelContext.active(), hostSpan as never),
      () => runTenantFlow(runTenantId)
    );

    const hostSpanContext = hostSpan?.spanContext() as {
      traceId: string;
      spanId: string;
    };
    const agentRoot = startsForTenant(runTenantId).find(
      (record) => record.name === 'AgentGraph'
    );
    expect(agentRoot?.traceId).toBe(traceIdFromSeed(`routing-${runTenantId}`));
    expect(agentRoot?.traceId).not.toBe(hostSpanContext.traceId);
    expect(agentRoot?.parentSpanId).toBeUndefined();
  });

  it('parents auxiliary label callbacks beneath the captured agent run', async () => {
    const tenantId = 'tenant-label-parent';
    const langfuse = tenantLangfuse(tenantId);
    const traceAnchor = {};
    initializeLangfuseTracing(langfuse);

    let agentRoot: MockSpan | undefined;
    withLangfuseRuntimeScope(
      { langfuse, traceAnchor, runId: 'source-run' },
      () => {
        agentRoot = createMockSpan('AgentGraph');
      }
    );
    const destinationKey = resolveLangfuseDestinationKey(langfuse) as string;
    const rootActiveContext = otelTrace.setSpan(
      otelContext.active(),
      agentRoot as never
    );
    const hostSpan = createMockSpan('managed-host', rootActiveContext);
    registerLangfuseManagedSpan(hostSpan as never, destinationKey);
    let graphChild: MockSpan | undefined;
    withLangfuseRuntimeScope(
      { langfuse, traceAnchor, runId: 'source-run' },
      () =>
        otelContext.with(rootActiveContext, () => {
          graphChild = createMockSpan('agent-child');
        })
    );
    const hostParent = otelContext.with(
      otelTrace.setSpan(otelContext.active(), hostSpan as never),
      () => resolveLangfuseTraceAnchorParent(traceAnchor, destinationKey)
    );
    const childParent = otelContext.with(
      otelTrace.setSpan(otelContext.active(), graphChild as never),
      () => resolveLangfuseTraceAnchorParent(traceAnchor, destinationKey)
    );
    expect(hostParent?.spanId).toBe(agentRoot?.spanContext().spanId);
    expect(childParent?.spanId).toBe(graphChild?.spanContext().spanId);
    const parentSpanContext = resolveLangfuseTraceAnchorParent(
      traceAnchor,
      destinationKey
    );
    const labelHandler = createLangfuseHandler({
      langfuse,
      runId: 'label-run',
      parentSpanContext,
      tags: ['librechat', 'activity-label'],
    });

    await withLangfuseRuntimeScope(
      { langfuse, runId: 'foreign-background-run' },
      () =>
        labelHandler?.handleChainStart(
          { id: ['StepLabel'] } as never,
          {},
          'label-chain'
        )
    );

    const label = startsForTenant(tenantId).find(
      (record) => record.name === 'StepLabel'
    );
    const rootContext = agentRoot?.spanContext() as {
      traceId: string;
      spanId: string;
    };
    expect(label?.traceId).toBe(rootContext.traceId);
    expect(label?.parentSpanId).toBe(rootContext.spanId);
  });

  it('restores an overlay agent lane after rejecting a foreign scope', async () => {
    const rootLangfuse = tenantLangfuse('tenant-anchor-root');
    const overlayLangfuse = tenantLangfuse('tenant-anchor-overlay');
    const traceAnchor = {};
    initializeLangfuseTracing(rootLangfuse);
    initializeLangfuseTracing(overlayLangfuse);

    withLangfuseRuntimeScope(
      { langfuse: rootLangfuse, traceAnchor, runId: 'source-run' },
      () => createMockSpan('AgentGraph')
    );
    const overlayHandler = createLangfuseHandler({
      langfuse: overlayLangfuse,
      traceAnchor,
      agentId: 'overlay-agent',
      runId: 'source-run',
      tags: ['librechat', 'agent'],
    });

    await withLangfuseRuntimeScope(
      { langfuse: rootLangfuse, runId: 'foreign-run' },
      () =>
        overlayHandler?.handleChainStart(
          { id: ['AgentModelCall'] } as never,
          {},
          'overlay-chain'
        )
    );

    const overlayParent = resolveLangfuseTraceAnchorParent(
      traceAnchor,
      resolveLangfuseDestinationKey(overlayLangfuse),
      'overlay-agent'
    );
    const overlayStart = startsForTenant('tenant-anchor-overlay').find(
      (record) => record.name === 'AgentModelCall'
    );
    expect(overlayParent?.spanId).toBe(overlayStart?.spanId);
    expect(overlayParent?.traceId).toBe(overlayStart?.traceId);
  });

  it('rotates the captured root between fresh executions of one run', async () => {
    const tenantId = 'tenant-anchor-rotation';
    const langfuse = tenantLangfuse(tenantId);
    const run = await Run.create<t.IState>({
      runId: `routing-${tenantId}`,
      graphConfig: {
        type: 'standard',
        agents: [createAgent(tenantId)],
      },
      langfuse,
      returnContent: true,
      skipCleanup: true,
    });
    const execute = async (response: string): Promise<void> => {
      run.Graph?.overrideTestModel([response]);
      await run.processStream(
        { messages: [new HumanMessage(response)] },
        callerConfig
      );
    };

    await execute('first execution');
    const firstTraceAnchor = run.Graph?.langfuseTraceAnchor;
    const firstScopeRunId = run.Graph?.langfuseScopeRunId;
    const firstParent = resolveLangfuseTraceAnchorParent(
      run.Graph?.langfuseTraceAnchor,
      resolveLangfuseDestinationKey(langfuse)
    );
    await execute('second execution');
    const secondTraceAnchor = run.Graph?.langfuseTraceAnchor;
    const secondScopeRunId = run.Graph?.langfuseScopeRunId;
    const secondParent = resolveLangfuseTraceAnchorParent(
      run.Graph?.langfuseTraceAnchor,
      resolveLangfuseDestinationKey(langfuse)
    );
    withLangfuseRuntimeScope(
      {
        langfuse,
        traceAnchor: firstTraceAnchor,
        runId: firstScopeRunId,
      },
      () => createMockSpan('late-first-execution-callback')
    );
    const secondParentAfterLateCallback = resolveLangfuseTraceAnchorParent(
      secondTraceAnchor,
      resolveLangfuseDestinationKey(langfuse)
    );
    const roots = startsForTenant(tenantId).filter(
      (record) => record.name === 'AgentGraph' && record.parentSpanId == null
    );

    expect(roots).toHaveLength(2);
    expect(firstParent?.spanId).toBe(roots[0].spanId);
    expect(secondParent?.spanId).toBe(roots[1].spanId);
    expect(secondParentAfterLateCallback?.spanId).toBe(roots[1].spanId);
    expect(secondParent?.spanId).not.toBe(firstParent?.spanId);
    expect(secondTraceAnchor).not.toBe(firstTraceAnchor);
    expect(secondScopeRunId).not.toBe(firstScopeRunId);
  });

  it('keeps warm terminal continuations under one trace root', async () => {
    const tenantId = 'tenant-warm-continuation';
    const registry = new HookRegistry();
    registry.register('Stop', {
      hooks: [
        async (input): Promise<StopHookOutput> =>
          input.continuationCount === 0
            ? {
              decision: 'block',
              injectedMessages: [
                { role: 'user', content: 'late steer', source: 'steer' },
              ],
            }
            : { decision: 'continue' },
      ],
    });
    const run = await Run.create<t.IState>({
      runId: `routing-${tenantId}`,
      graphConfig: {
        type: 'standard',
        agents: [createAgent(tenantId)],
      },
      langfuse: tenantLangfuse(tenantId),
      hooks: registry,
      returnContent: true,
      skipCleanup: true,
    });
    run.Graph?.overrideTestModel(['first answer', 'continued answer']);

    await run.processStream(
      { messages: [new HumanMessage('initial prompt')] },
      callerConfig
    );

    const roots = startsForTenant(tenantId).filter(
      (record) => record.name === 'AgentGraph' && record.parentSpanId == null
    );
    expect(roots).toHaveLength(1);
    expect(
      startsForTenant(tenantId).filter(
        (record) => record.parentSpanId === roots[0].spanId
      ).length
    ).toBeGreaterThan(0);
  });

  it('marks the deferred root as failed when terminal admission fails', async () => {
    const tenantId = 'tenant-terminal-admission-failure';
    const registry = new HookRegistry();
    registry.register('StopFinalize', {
      hooks: [
        async (): Promise<StopHookOutput> => {
          throw new Error('terminal admission unavailable');
        },
      ],
    });
    const run = await Run.create<t.IState>({
      runId: `routing-${tenantId}`,
      graphConfig: {
        type: 'standard',
        agents: [createAgent(tenantId)],
      },
      langfuse: tenantLangfuse(tenantId),
      hooks: registry,
      returnContent: true,
      skipCleanup: true,
    });
    run.Graph?.overrideTestModel(['first answer']);

    await expect(
      run.processStream(
        { messages: [new HumanMessage('initial prompt')] },
        callerConfig
      )
    ).rejects.toThrow(
      'StopFinalize terminal admission failed: terminal admission unavailable'
    );

    const root = endedSpans.find(
      (span) =>
        span.name === 'AgentGraph' &&
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_LEVEL] ===
          'ERROR'
    );
    expect(root).toBeDefined();
    expect(
      root?.attributes[LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE]
    ).toContain('terminal admission unavailable');
  });

  it('generates a scope stamp for directly-constructed graphs without a run id', async () => {
    const { StandardGraph } = await import('@/graphs/Graph');
    const buildGraph = (): { langfuseScopeRunId: string } =>
      new StandardGraph({
        agents: [createAgent('stampless')],
        langfuse: tenantLangfuse('stampless'),
      }) as unknown as { langfuseScopeRunId: string };

    const graphA = buildGraph();
    const graphB = buildGraph();
    expect(graphA.langfuseScopeRunId).toEqual(expect.stringMatching(/^graph:/));
    expect(graphB.langfuseScopeRunId).toEqual(expect.stringMatching(/^graph:/));
    expect(graphA.langfuseScopeRunId).not.toBe(graphB.langfuseScopeRunId);
  });

  it('stamps concurrent executions of the same public run id distinctly', async () => {
    const { StandardGraph } = await import('@/graphs/Graph');
    const buildGraph = (): { langfuseScopeRunId: string } =>
      new StandardGraph({
        runId: 'duplicate-run',
        agents: [createAgent('duplicate')],
        langfuse: tenantLangfuse('duplicate'),
      }) as unknown as { langfuseScopeRunId: string };

    const first = buildGraph();
    const second = buildGraph();
    expect(first.langfuseScopeRunId).toEqual(
      expect.stringMatching(/^duplicate-run:/)
    );
    expect(first.langfuseScopeRunId).not.toBe(second.langfuseScopeRunId);
  });

  it('routes spans from captured OTel context after ALS scope exits', () => {
    const langfuse = tenantLangfuse('tenant-otel');
    initializeLangfuseTracing(langfuse);

    let capturedContext: Context | undefined;
    withLangfuseRuntimeScope({ langfuse }, () => {
      capturedContext = otelContext.active();
    });

    expect(capturedContext).toBeDefined();
    createMockSpan('otel-context-only', capturedContext);

    expect(startsForTenant('tenant-otel')).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: 'otel-context-only',
          params: expect.objectContaining({
            publicKey: 'pk-tenant-otel',
            secretKey: 'sk-tenant-otel',
            baseUrl: 'https://langfuse.proxy',
          }),
        }),
      ])
    );
  });
});
