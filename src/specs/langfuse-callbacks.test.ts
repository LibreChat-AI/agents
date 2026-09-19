import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { LangfuseOtelSpanAttributes } from '@langfuse/tracing';
import { CallbackManager } from '@langchain/core/callbacks/manager';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import { context as otelContext, trace as otelTrace } from '@opentelemetry/api';
import {
  Command,
  END,
  GraphInterrupt,
  MemorySaver,
  MessagesAnnotation,
  NodeInterrupt,
  ParentCommand,
  START,
  StateGraph,
  isGraphInterrupt,
  isInterrupted,
} from '@langchain/langgraph';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { handleConverseStreamMetadata } from '@/llm/bedrock/utils/message_outputs';
import { initializeLangfuseTracing } from '@/instrumentation';
import { traceIdFromSeed } from '@/langfuseRuntimeContext';
import { NativeMediaError } from '@/llm/google/native';
import { createLangfuseHandler } from '@/langfuse';
import { Constants, Providers } from '@/common';
import { ToolNode } from '@/tools/ToolNode';
import { askUserQuestion } from '@/hitl';
import { Run } from '@/run';

const mockProcessorStarts: Array<{
  params: unknown;
  traceId: string;
}> = [];
const mockSpanAttributeSets: Array<Record<string, unknown>> = [];
const mockEndedSpanAttributes: Array<Record<string, unknown>> = [];
let mockSpansStarted = 0;
let mockSpansEnded = 0;
let mockProviderInput:
  | {
      spanProcessors?: Array<{
        onStart?: (span: unknown, parentContext: unknown) => void;
        onEnd?: (span: unknown) => void;
      }>;
      idGenerator?: {
        generateTraceId: () => string;
        generateSpanId: () => string;
      };
    }
  | undefined;

const createMockSpan = (traceIdOverride?: string, name = 'test-span') => {
  mockSpansStarted += 1;
  const traceId =
    traceIdOverride ??
    mockProviderInput?.idGenerator?.generateTraceId() ??
    'trace-id';
  const spanId = mockProviderInput?.idGenerator?.generateSpanId() ?? 'span-id';
  const span = {
    end: jest.fn(() => {
      mockSpansEnded += 1;
      for (const processor of mockProviderInput?.spanProcessors ?? []) {
        processor.onEnd?.(span);
      }
      mockEndedSpanAttributes.push({ ...span.attributes });
    }),
    spanContext: jest.fn(() => ({
      traceId,
      spanId,
      traceFlags: 1,
    })),
    setAttributes: jest.fn((attributes: Record<string, unknown>) => {
      mockSpanAttributeSets.push(attributes);
      Object.assign(span.attributes, attributes);
    }),
    setStatus: jest.fn(),
    attributes: {},
    name,
  };
  for (const processor of mockProviderInput?.spanProcessors ?? []) {
    processor.onStart?.(span, otelContext.active());
  }
  return span;
};

const mockStartSpan = jest.fn((name: string) =>
  createMockSpan(undefined, name)
);
const mockStartActiveSpan = jest.fn(
  (
    _name: string,
    _options: unknown,
    activeContext: Parameters<typeof otelTrace.getSpanContext>[0],
    callback: (span: ReturnType<typeof createMockSpan>) => unknown
  ) =>
    callback(
      createMockSpan(otelTrace.getSpanContext(activeContext)?.traceId, _name)
    )
);
const mockForceFlush = jest.fn();
const mockShutdown = jest.fn();

jest.mock('@langfuse/otel', () => ({
  LangfuseSpanProcessor: jest.fn().mockImplementation((params) => ({
    forceFlush: jest.fn(),
    onEnd: jest.fn(),
    onStart: jest.fn((span) => {
      mockProcessorStarts.push({
        params,
        traceId: span.spanContext().traceId,
      });
    }),
    shutdown: jest.fn(),
  })),
  isDefaultExportSpan: jest.fn(() => false),
}));

jest.mock('@opentelemetry/sdk-trace-base', () => ({
  BasicTracerProvider: jest.fn().mockImplementation((input) => {
    mockProviderInput = input;
    return {
      forceFlush: mockForceFlush,
      getTracer: jest.fn(() => ({
        startActiveSpan: mockStartActiveSpan,
        startSpan: mockStartSpan,
      })),
      shutdown: mockShutdown,
    };
  }),
}));

describe('Langfuse callback composition', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockProcessorStarts.length = 0;
    mockSpanAttributeSets.length = 0;
    mockEndedSpanAttributes.length = 0;
    mockSpansStarted = 0;
    mockSpansEnded = 0;
    delete process.env.LANGFUSE_PUBLIC_KEY;
    delete process.env.LANGFUSE_SECRET_KEY;
    delete process.env.LANGFUSE_BASE_URL;
    delete process.env.LANGFUSE_BASEURL;
    delete process.env.LANGFUSE_FORCE_FLUSH_ON_DISPOSE;
  });

  it('keeps native failure usage on each tenant generation without reporting successful output', async () => {
    const failures = [
      { tenant: 'native-a', input: 11, output: 1290 },
      { tenant: 'native-b', input: 22, output: 2580 },
    ];
    const handlers = failures.map(({ tenant }) => {
      const langfuse = {
        publicKey: `pk-${tenant}`,
        secretKey: `sk-${tenant}`,
        deterministicTraceId: true,
      };
      initializeLangfuseTracing(langfuse);
      return createLangfuseHandler({
        langfuse,
        runId: tenant,
        traceIdSeed: tenant,
      });
    });
    await Promise.all(
      handlers.map((handler, index) =>
        handler?.handleChatModelStart(
          { lc: 1, type: 'constructor', id: ['NativeGoogle'], kwargs: {} },
          [[new HumanMessage(failures[index].tenant)]],
          'model-run'
        )
      )
    );
    for (let index = handlers.length - 1; index >= 0; index--) {
      const failure = failures[index];
      await handlers[index]?.handleLLMError(
        new NativeMediaError(new Error('Storage unavailable'), {
          input_tokens: failure.input,
          output_tokens: failure.output,
          total_tokens: failure.input + failure.output,
          input_token_details: { cache_read: 3 },
        }),
        'model-run'
      );
    }
    expect(mockSpansStarted).toBe(2);
    expect(mockSpansEnded).toBe(2);
    expect(mockEndedSpanAttributes).toEqual(
      failures.toReversed().map((failure) =>
        expect.objectContaining({
          [LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]: 'ERROR',
          [LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE]:
            'Error: Storage unavailable',
          [LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS]:
            JSON.stringify({
              input: failure.input - 3,
              output: failure.output,
              total: failure.input + failure.output,
              input_cache_read: 3,
            }),
        })
      )
    );
    expect(
      mockEndedSpanAttributes.every(
        (attributes) =>
          attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == null
      )
    ).toBe(true);
  });

  it('runs explicit per-agent tracing when callbacks is a CallbackManager', async () => {
    const manager = CallbackManager.fromHandlers({
      handleCustomEvent: async (): Promise<void> => undefined,
    });
    const run = await Run.create<t.IState>({
      runId: 'test-langfuse-callback-manager',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'agent_abc123',
            name: 'DWAINE',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4' },
            tools: [],
            langfuse: {
              enabled: true,
              publicKey: 'pk-test',
              secretKey: 'sk-test',
            },
          },
        ],
      },
      skipCleanup: true,
    });

    run.Graph?.overrideTestModel(['hello']);

    const config = {
      callbacks: manager,
      configurable: { thread_id: 'thread-1', user_id: 'user-1' },
      streamMode: 'values' as const,
      version: 'v2' as const,
    };

    await run.processStream({ messages: [new HumanMessage('hello')] }, config);

    expect(mockStartActiveSpan).toHaveBeenCalled();
    expect(mockForceFlush).not.toHaveBeenCalled();
  });

  it('attaches Langfuse callbacks for direct graph invocations', async () => {
    const run = await Run.create<t.IState>({
      runId: 'test-langfuse-direct-graph',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'agent_abc123',
            name: 'DWAINE',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4' },
            tools: [],
            langfuse: {
              enabled: true,
              publicKey: 'pk-test',
              secretKey: 'sk-test',
            },
          },
        ],
      },
      skipCleanup: true,
    });

    run.Graph?.overrideTestModel(['hello']);
    const workflow = run.Graph?.createWorkflow();
    await workflow?.invoke(
      { messages: [new HumanMessage('hello')] },
      {
        callbacks: [],
        configurable: { thread_id: 'thread-1', user_id: 'user-1' },
      }
    );

    expect(mockStartActiveSpan).toHaveBeenCalled();
  });

  it('preserves per-agent Langfuse config when a stream callback already exists', async () => {
    const { LangfuseSpanProcessor } = await import('@langfuse/otel');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const { createLangfuseHandler } = await import('@/langfuse');
    initializeLangfuseTracing({
      publicKey: 'pk-run',
      secretKey: 'sk-run',
      baseUrl: 'https://langfuse.run',
    });
    const streamHandler = createLangfuseHandler({
      langfuse: {
        publicKey: 'pk-run',
        secretKey: 'sk-run',
        baseUrl: 'https://langfuse.run',
      },
    });
    const run = await Run.create<t.IState>({
      runId: 'test-langfuse-agent-callback-override',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'agent_abc123',
            name: 'DWAINE',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4' },
            tools: [],
            langfuse: {
              enabled: true,
              publicKey: 'pk-agent',
              secretKey: 'sk-agent',
              baseUrl: 'https://langfuse.agent',
            },
          },
        ],
      },
      skipCleanup: true,
    });

    run.Graph?.overrideTestModel(['hello']);
    const workflow = run.Graph?.createWorkflow();
    await workflow?.invoke(
      { messages: [new HumanMessage('hello')] },
      {
        callbacks: streamHandler != null ? [streamHandler] : [],
        configurable: { thread_id: 'thread-1', user_id: 'user-1' },
      }
    );

    expect(LangfuseSpanProcessor).toHaveBeenCalledWith(
      expect.objectContaining({
        publicKey: 'pk-agent',
        secretKey: 'sk-agent',
        baseUrl: 'https://langfuse.agent',
      })
    );
  });

  it('binds handler callback spans to their own Langfuse config and trace seed', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const tenantA = {
      publicKey: 'pk-tenant-a',
      secretKey: 'sk-tenant-a',
      baseUrl: 'https://langfuse.proxy',
      deterministicTraceId: true,
    };
    const tenantB = {
      publicKey: 'pk-tenant-b',
      secretKey: 'sk-tenant-b',
      baseUrl: 'https://langfuse.proxy',
      deterministicTraceId: true,
    };
    initializeLangfuseTracing(tenantA);
    initializeLangfuseTracing(tenantB);

    const handlerA = createLangfuseHandler({
      langfuse: tenantA,
      traceIdSeed: 'run-tenant-a',
    });
    const handlerB = createLangfuseHandler({
      langfuse: tenantB,
      traceIdSeed: 'run-tenant-b',
    });

    await Promise.all([
      handlerA?.handleChainStart(
        { lc: 1, type: 'not_implemented', id: ['TenantAChain'] },
        { input: 'tenant a' },
        'lc-run-a'
      ),
      handlerB?.handleChainStart(
        { lc: 1, type: 'not_implemented', id: ['TenantBChain'] },
        { input: 'tenant b' },
        'lc-run-b'
      ),
    ]);

    expect(mockProcessorStarts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          params: expect.objectContaining({
            publicKey: 'pk-tenant-a',
            secretKey: 'sk-tenant-a',
            baseUrl: 'https://langfuse.proxy',
          }),
          traceId: traceIdFromSeed('run-tenant-a'),
        }),
        expect.objectContaining({
          params: expect.objectContaining({
            publicKey: 'pk-tenant-b',
            secretKey: 'sk-tenant-b',
            baseUrl: 'https://langfuse.proxy',
          }),
          traceId: traceIdFromSeed('run-tenant-b'),
        }),
      ])
    );
  });

  it('preserves an active agent Langfuse runtime scope for callback-created spans', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const { withLangfuseRuntimeScope } = await import('@/langfuseRuntimeScope');
    const runLangfuse = {
      publicKey: 'pk-run',
      secretKey: 'sk-run',
      baseUrl: 'https://langfuse.run',
      deterministicTraceId: true,
    };
    const agentLangfuse = {
      publicKey: 'pk-agent',
      secretKey: 'sk-agent',
      baseUrl: 'https://langfuse.agent',
      deterministicTraceId: true,
    };
    initializeLangfuseTracing(runLangfuse);
    initializeLangfuseTracing(agentLangfuse);
    const streamHandler = createLangfuseHandler({
      langfuse: runLangfuse,
      traceIdSeed: 'run-seed',
    });

    await withLangfuseRuntimeScope(
      { langfuse: agentLangfuse, traceIdSeed: 'agent-seed' },
      () =>
        streamHandler?.handleChainStart(
          { lc: 1, type: 'not_implemented', id: ['AgentScopedChain'] },
          { input: 'agent scoped' },
          'lc-agent-run'
        )
    );

    expect(mockProcessorStarts).toContainEqual(
      expect.objectContaining({
        params: expect.objectContaining({
          publicKey: 'pk-agent',
          secretKey: 'sk-agent',
          baseUrl: 'https://langfuse.agent',
        }),
        traceId: traceIdFromSeed('agent-seed'),
      })
    );
    expect(mockProcessorStarts).not.toContainEqual(
      expect.objectContaining({
        params: expect.objectContaining({
          publicKey: 'pk-run',
          secretKey: 'sk-run',
          baseUrl: 'https://langfuse.run',
        }),
        traceId: traceIdFromSeed('run-seed'),
      })
    );
  });

  it('keeps callback spans on their own run when a foreign run scope is active', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const { withLangfuseRuntimeScope } = await import('@/langfuseRuntimeScope');
    const tenantA = {
      publicKey: 'pk-tenant-a',
      secretKey: 'sk-tenant-a',
      baseUrl: 'https://langfuse.tenant-a',
      deterministicTraceId: true,
    };
    const tenantB = {
      publicKey: 'pk-tenant-b',
      secretKey: 'sk-tenant-b',
      baseUrl: 'https://langfuse.tenant-b',
      deterministicTraceId: true,
    };
    initializeLangfuseTracing(tenantA);
    initializeLangfuseTracing(tenantB);
    const handlerB = createLangfuseHandler({
      langfuse: tenantB,
      traceIdSeed: 'run-b',
      runId: 'run-b',
    });

    // LangChain's background callback queue (`consumeCallback`) executes
    // non-awaited callbacks inside whichever concurrent run's async context
    // the queue drain happens to be on — here, tenant-A's. A scope stamped
    // with a different run must never reroute tenant-B's spans.
    await withLangfuseRuntimeScope(
      { langfuse: tenantA, traceIdSeed: 'run-a', runId: 'run-a' },
      () =>
        handlerB?.handleChainStart(
          { lc: 1, type: 'not_implemented', id: ['ForeignScopedChain'] },
          { input: 'foreign scoped' },
          'lc-foreign-run'
        )
    );

    expect(mockProcessorStarts).toContainEqual(
      expect.objectContaining({
        params: expect.objectContaining({
          publicKey: 'pk-tenant-b',
          secretKey: 'sk-tenant-b',
          baseUrl: 'https://langfuse.tenant-b',
        }),
        traceId: traceIdFromSeed('run-b'),
      })
    );
    expect(
      mockProcessorStarts.filter(
        (record) =>
          (record.params as { publicKey?: string }).publicKey ===
            'pk-tenant-a' || record.traceId === traceIdFromSeed('run-a')
      )
    ).toHaveLength(0);
  });

  it('adopts an agent overlay scope stamped with the handler run', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const { withLangfuseRuntimeScope } = await import('@/langfuseRuntimeScope');
    const runLangfuse = {
      publicKey: 'pk-run',
      secretKey: 'sk-run',
      baseUrl: 'https://langfuse.run',
      deterministicTraceId: true,
    };
    const agentLangfuse = {
      publicKey: 'pk-agent',
      secretKey: 'sk-agent',
      baseUrl: 'https://langfuse.agent',
      deterministicTraceId: true,
    };
    initializeLangfuseTracing(runLangfuse);
    initializeLangfuseTracing(agentLangfuse);
    const streamHandler = createLangfuseHandler({
      langfuse: runLangfuse,
      traceIdSeed: 'run-seed',
      runId: 'run-1',
    });

    await withLangfuseRuntimeScope(
      { langfuse: agentLangfuse, traceIdSeed: 'agent-seed', runId: 'run-1' },
      () =>
        streamHandler?.handleChainStart(
          { lc: 1, type: 'not_implemented', id: ['SameRunOverlayChain'] },
          { input: 'same run overlay' },
          'lc-overlay-run'
        )
    );

    expect(mockProcessorStarts).toContainEqual(
      expect.objectContaining({
        params: expect.objectContaining({
          publicKey: 'pk-agent',
          secretKey: 'sk-agent',
          baseUrl: 'https://langfuse.agent',
        }),
        traceId: traceIdFromSeed('agent-seed'),
      })
    );
  });

  it('scopes agent overlays to the callback agent in parallel fan-out', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const { withLangfuseRuntimeScope } = await import('@/langfuseRuntimeScope');
    const runLangfuse = {
      publicKey: 'pk-run',
      secretKey: 'sk-run',
      baseUrl: 'https://langfuse.run',
      deterministicTraceId: true,
    };
    const agentBLangfuse = {
      publicKey: 'pk-agent-b',
      secretKey: 'sk-agent-b',
      baseUrl: 'https://langfuse.agent-b',
      deterministicTraceId: true,
    };
    initializeLangfuseTracing(runLangfuse);
    initializeLangfuseTracing(agentBLangfuse);
    const streamHandler = createLangfuseHandler({
      langfuse: runLangfuse,
      traceIdSeed: 'run-seed',
      runId: 'run-1',
    });

    // Agent B's overlay scope is ambient (the background callback queue can
    // interleave concurrent fan-out agents), but this callback reports agent
    // A via inherited langgraph metadata — a sibling's overlay must not
    // capture it.
    await withLangfuseRuntimeScope(
      {
        langfuse: agentBLangfuse,
        traceIdSeed: 'agent-b-seed',
        runId: 'run-1',
        agentId: 'agent-b',
      },
      () =>
        streamHandler?.handleChainStart(
          { lc: 1, type: 'not_implemented', id: ['SiblingAgentChain'] },
          { input: 'sibling agent' },
          'lc-sibling-run',
          undefined,
          undefined,
          { langgraph_node: 'agent=agent-a' }
        )
    );

    expect(mockProcessorStarts).toContainEqual(
      expect.objectContaining({
        params: expect.objectContaining({ publicKey: 'pk-run' }),
        traceId: traceIdFromSeed('run-seed'),
      })
    );

    // The SAME agent's overlay scope is still adopted.
    await withLangfuseRuntimeScope(
      {
        langfuse: agentBLangfuse,
        traceIdSeed: 'agent-b-seed',
        runId: 'run-1',
        agentId: 'agent-b',
      },
      () =>
        streamHandler?.handleChainStart(
          { lc: 1, type: 'not_implemented', id: ['OwnAgentChain'] },
          { input: 'own agent' },
          'lc-own-agent-run',
          undefined,
          undefined,
          { langgraph_node: 'agent=agent-b' }
        )
    );

    expect(mockProcessorStarts).toContainEqual(
      expect.objectContaining({
        params: expect.objectContaining({ publicKey: 'pk-agent-b' }),
        traceId: traceIdFromSeed('agent-b-seed'),
      })
    );

    // Explicit `agentId` metadata (stamped by the graph's model path and
    // ToolNode) beats node-name parsing: agent `research`'s inner node is
    // named `agent=research`, indistinguishable by name from a sibling
    // literally named `agent=research` — the explicit identity disambiguates.
    await withLangfuseRuntimeScope(
      {
        langfuse: agentBLangfuse,
        traceIdSeed: 'prefixed-sibling-seed',
        runId: 'run-1',
        agentId: 'agent=research',
      },
      () =>
        streamHandler?.handleChainStart(
          { lc: 1, type: 'not_implemented', id: ['AmbiguousNodeChain'] },
          { input: 'ambiguous node' },
          'lc-ambiguous-node-run',
          undefined,
          undefined,
          { langgraph_node: 'agent=research', agentId: 'research' }
        )
    );

    expect(mockProcessorStarts).toContainEqual(
      expect.objectContaining({
        params: expect.objectContaining({ publicKey: 'pk-run' }),
        traceId: traceIdFromSeed('run-seed'),
      })
    );

    // An agent id that itself begins with an internal node prefix is carried
    // VERBATIM by its outer workflow node — still recognized as its own scope.
    await withLangfuseRuntimeScope(
      {
        langfuse: agentBLangfuse,
        traceIdSeed: 'prefixed-agent-seed',
        runId: 'run-1',
        agentId: 'agent=research',
      },
      () =>
        streamHandler?.handleChainStart(
          { lc: 1, type: 'not_implemented', id: ['PrefixedAgentChain'] },
          { input: 'prefixed agent' },
          'lc-prefixed-agent-run',
          undefined,
          undefined,
          { langgraph_node: 'agent=research' }
        )
    );

    expect(mockProcessorStarts).toContainEqual(
      expect.objectContaining({
        params: expect.objectContaining({ publicKey: 'pk-agent-b' }),
        traceId: traceIdFromSeed('prefixed-agent-seed'),
      })
    );
  });

  it('rejects a sibling agent overlay for tool callbacks', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const { withLangfuseRuntimeScope } = await import('@/langfuseRuntimeScope');
    const runLangfuse = {
      publicKey: 'pk-run',
      secretKey: 'sk-run',
      baseUrl: 'https://langfuse.run',
      deterministicTraceId: true,
    };
    const agentBLangfuse = {
      publicKey: 'pk-agent-b',
      secretKey: 'sk-agent-b',
      baseUrl: 'https://langfuse.agent-b',
      deterministicTraceId: true,
    };
    initializeLangfuseTracing(runLangfuse);
    initializeLangfuseTracing(agentBLangfuse);
    const streamHandler = createLangfuseHandler({
      langfuse: runLangfuse,
      traceIdSeed: 'run-seed',
      runId: 'run-1',
    });

    // Tool supersteps stamp their scope with the executing agent; a tool
    // callback reporting a different agent via `tools=<id>` metadata must
    // not adopt the sibling's overlay.
    await withLangfuseRuntimeScope(
      {
        langfuse: agentBLangfuse,
        traceIdSeed: 'agent-b-seed',
        runId: 'run-1',
        agentId: 'agent-b',
      },
      () =>
        streamHandler?.handleToolStart(
          { lc: 1, type: 'not_implemented', id: ['SiblingTool'] },
          'sibling tool input',
          'lc-sibling-tool-run',
          undefined,
          undefined,
          { langgraph_node: 'tools=agent-a' }
        )
    );

    expect(mockProcessorStarts).toContainEqual(
      expect.objectContaining({
        params: expect.objectContaining({ publicKey: 'pk-run' }),
        traceId: traceIdFromSeed('run-seed'),
      })
    );
    expect(
      mockProcessorStarts.filter(
        (record) =>
          (record.params as { publicKey?: string }).publicKey === 'pk-agent-b'
      )
    ).toHaveLength(0);
  });

  it('replaces foreign propagated identity attributes with its own', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const { withLangfuseRuntimeScope } = await import('@/langfuseRuntimeScope');
    const { propagateAttributes } = await import('@langfuse/tracing');
    const { getPropagatedAttributesFromContext } = await import(
      '@langfuse/core'
    );
    const tenantA = {
      publicKey: 'pk-tenant-a',
      secretKey: 'sk-tenant-a',
      baseUrl: 'https://langfuse.tenant-a',
    };
    const tenantB = {
      publicKey: 'pk-tenant-b',
      secretKey: 'sk-tenant-b',
      baseUrl: 'https://langfuse.tenant-b',
    };
    initializeLangfuseTracing(tenantA);
    initializeLangfuseTracing(tenantB);
    const handlerB = createLangfuseHandler({
      langfuse: tenantB,
      userId: 'user-b',
      runId: 'run-b',
    });

    let observedAttributes: Record<string, unknown> | undefined;
    mockStartActiveSpan.mockImplementationOnce(
      (_name, _options, activeContext, callback) => {
        observedAttributes = getPropagatedAttributesFromContext(
          otelContext.active()
        ) as Record<string, unknown>;
        return callback(
          createMockSpan(otelTrace.getSpanContext(activeContext)?.traceId)
        );
      }
    );

    // Run A's propagated identity (user, session) is ambient; B's spans must
    // carry B's identity — and NOT retain A's session where B has none.
    await propagateAttributes(
      { userId: 'user-a', sessionId: 'session-a' },
      async () =>
        withLangfuseRuntimeScope({ langfuse: tenantA, runId: 'run-a' }, () =>
          handlerB?.handleChainStart(
            { lc: 1, type: 'not_implemented', id: ['IdentityChain'] },
            { input: 'identity' },
            'lc-identity-run'
          )
        )
    );

    expect(observedAttributes?.['user.id']).toBe('user-b');
    expect(observedAttributes?.['session.id']).toBeUndefined();
  });

  it('clears a foreign scope for env-credential runs instead of inheriting it', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const { withLangfuseRuntimeScope } = await import('@/langfuseRuntimeScope');
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-env';
    process.env.LANGFUSE_SECRET_KEY = 'sk-env';
    const tenantA = {
      publicKey: 'pk-tenant-a',
      secretKey: 'sk-tenant-a',
      baseUrl: 'https://langfuse.tenant-a',
      deterministicTraceId: true,
    };
    initializeLangfuseTracing(tenantA);
    initializeLangfuseTracing();
    // Run B has no explicit config or seed of its own — it relies on env
    // credentials. Rejecting a foreign scope must CLEAR the foreign values
    // rather than let merge inheritance re-adopt them through `undefined`.
    const handlerB = createLangfuseHandler({ runId: 'run-b' });

    await withLangfuseRuntimeScope(
      { langfuse: tenantA, traceIdSeed: 'run-a', runId: 'run-a' },
      () =>
        handlerB?.handleChainStart(
          { lc: 1, type: 'not_implemented', id: ['EnvCredentialChain'] },
          { input: 'env credentials' },
          'lc-env-run'
        )
    );

    expect(mockProcessorStarts).toContainEqual(
      expect.objectContaining({
        params: expect.objectContaining({
          publicKey: 'pk-env',
          secretKey: 'sk-env',
        }),
      })
    );
    expect(
      mockProcessorStarts.filter(
        (record) =>
          (record.params as { publicKey?: string }).publicKey ===
            'pk-tenant-a' || record.traceId === traceIdFromSeed('run-a')
      )
    ).toHaveLength(0);
  });

  it('applies its own tool-output policy inside a foreign run scope', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const { withLangfuseRuntimeScope } = await import('@/langfuseRuntimeScope');
    const { resolveToolOutputTracingConfig } = await import('@/langfuseConfig');
    const { getLangfuseRuntimeToolOutputTracingConfig } = await import(
      '@/langfuseRuntimeContext'
    );
    const tenantA = {
      publicKey: 'pk-tenant-a',
      secretKey: 'sk-tenant-a',
      baseUrl: 'https://langfuse.tenant-a',
      toolOutputTracing: { enabled: true },
    };
    const tenantB = {
      publicKey: 'pk-tenant-b',
      secretKey: 'sk-tenant-b',
      baseUrl: 'https://langfuse.tenant-b',
      toolOutputTracing: { enabled: false },
    };
    initializeLangfuseTracing(tenantA);
    initializeLangfuseTracing(tenantB);
    const handlerB = createLangfuseHandler({
      langfuse: tenantB,
      runId: 'run-b',
    });

    let observedPolicy: { enabled: boolean } | undefined;
    mockStartActiveSpan.mockImplementationOnce(
      (_name, _options, activeContext, callback) => {
        observedPolicy = getLangfuseRuntimeToolOutputTracingConfig();
        return callback(
          createMockSpan(otelTrace.getSpanContext(activeContext)?.traceId)
        );
      }
    );

    // Tenant A permits tool output; tenant B redacts. B's callback running
    // inside A's scope must not inherit A's permissive policy.
    await withLangfuseRuntimeScope(
      {
        langfuse: tenantA,
        runId: 'run-a',
        toolOutputTracing: resolveToolOutputTracingConfig(tenantA),
      },
      () =>
        handlerB?.handleToolStart(
          { lc: 1, type: 'not_implemented', id: ['SensitiveTool'] },
          'sensitive input',
          'lc-tool-run'
        )
    );

    expect(observedPolicy?.enabled).toBe(false);
  });

  it('promotes trusted ToolMessage artifact metadata before redacting callback output', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const { LANGFUSE_OBSERVATION_METADATA_ARTIFACT_KEY } = await import(
      '@/langfuseToolOutputTracing'
    );
    const langfuse = {
      publicKey: 'pk-query-metadata',
      secretKey: 'sk-query-metadata',
      toolOutputTracing: {
        redactedToolNames: ['run_select_query'],
      },
    };
    initializeLangfuseTracing(langfuse);
    const handler = createLangfuseHandler({ langfuse });
    const queryTool = tool(
      async () => [
        '{"data":[{"private_column":"customer value"}]}',
        {
          [LANGFUSE_OBSERVATION_METADATA_ARTIFACT_KEY]: {
            query_database_system: 'clickhouse',
            query_status: 'success',
            query_returned_rows: 1,
          },
        },
      ],
      {
        name: 'run_select_query',
        description: 'Run a test query.',
        schema: z.object({ query: z.string() }),
        responseFormat: 'content_and_artifact',
      }
    );

    await queryTool.invoke(
      {
        name: 'run_select_query',
        args: { query: 'SELECT private_column FROM customer_table' },
        id: 'query-call',
        type: 'tool_call',
      },
      { callbacks: handler != null ? [handler] : [] }
    );

    expect(mockEndedSpanAttributes).toContainEqual(
      expect.objectContaining({
        [`${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.query_database_system`]:
          'clickhouse',
        [`${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.query_status`]:
          'success',
        [`${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.query_returned_rows`]: 1,
        [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]:
          '[tool output redacted]',
      })
    );
    expect(JSON.stringify(mockEndedSpanAttributes)).not.toContain(
      'customer value'
    );
  });

  it('attaches configured trace attributes to Langfuse callback spans', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const langfuse = {
      publicKey: 'pk-tenant-a',
      secretKey: 'sk-tenant-a',
      baseUrl: 'https://langfuse.proxy',
      librechatTraceAttributes: {
        'librechat.langfuse.tenant_export.enabled': 'true',
        'librechat.langfuse.destination': 'eu',
        ignored: '',
      },
    };
    initializeLangfuseTracing(langfuse);
    const handler = createLangfuseHandler({ langfuse });

    await handler?.handleChainStart(
      { lc: 1, type: 'not_implemented', id: ['TenantAChain'] },
      { input: 'tenant a' },
      'lc-run-a'
    );

    expect(mockSpanAttributeSets).toContainEqual({
      'librechat.langfuse.tenant_export.enabled': 'true',
      'librechat.langfuse.destination': 'eu',
    });
  });

  it('ends nested ParentCommand control flow without reporting an error', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const langfuse = {
      publicKey: 'pk-test',
      secretKey: 'sk-test',
    };
    initializeLangfuseTracing(langfuse);
    const handler = createLangfuseHandler({ langfuse });
    const runId = 'test-langfuse-parent-command';

    await handler?.handleChainStart(
      { lc: 1, type: 'not_implemented', id: ['HandoffChain'] },
      { input: 'handoff' },
      runId,
      'parent-run'
    );
    await handler?.handleChainError(
      new ParentCommand(
        new Command({ graph: 'source:graph', goto: 'destination' })
      ),
      runId,
      'parent-run'
    );

    expect(mockSpanAttributeSets).not.toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]: 'ERROR',
      })
    );
    expect(mockSpanAttributeSets).toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: JSON.stringify({
          controlFlow: 'ParentCommand',
        }),
      })
    );
  });

  it('ends nested GraphInterrupt control flow without reporting an error', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const langfuse = {
      publicKey: 'pk-graph-interrupt',
      secretKey: 'sk-graph-interrupt',
    };
    initializeLangfuseTracing(langfuse);
    const handler = createLangfuseHandler({ langfuse });
    const runId = 'test-langfuse-graph-interrupt';
    const sensitiveQuestion = 'sensitive question payload';

    await handler?.handleChainStart(
      { lc: 1, type: 'not_implemented', id: ['InterruptedChain'] },
      { input: 'ask' },
      runId,
      'parent-run'
    );
    await handler?.handleChainError(
      new GraphInterrupt([
        {
          value: {
            type: 'ask_user_question',
            question: sensitiveQuestion,
          },
        },
      ]),
      runId,
      'parent-run'
    );

    expect(mockSpanAttributeSets).not.toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]: 'ERROR',
      })
    );
    expect(mockSpanAttributeSets).toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: JSON.stringify({
          controlFlow: 'GraphInterrupt',
        }),
      })
    );
    expect(JSON.stringify(mockSpanAttributeSets)).not.toContain(
      sensitiveQuestion
    );
  });

  it('ends nested GraphInterrupt tool control flow without reporting an error', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const langfuse = {
      publicKey: 'pk-tool-graph-interrupt',
      secretKey: 'sk-tool-graph-interrupt',
    };
    initializeLangfuseTracing(langfuse);
    const handler = createLangfuseHandler({ langfuse });
    const runId = 'test-langfuse-tool-graph-interrupt';
    const sensitiveQuestion = 'sensitive tool question payload';

    await handler?.handleToolStart(
      { lc: 1, type: 'not_implemented', id: ['AskUserQuestion'] },
      '{}',
      runId,
      'parent-run'
    );
    await handler?.handleToolError(
      new GraphInterrupt([
        {
          value: {
            type: 'ask_user_question',
            question: sensitiveQuestion,
          },
        },
      ]),
      runId,
      'parent-run'
    );

    expect(mockSpanAttributeSets).not.toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]: 'ERROR',
      })
    );
    expect(mockSpanAttributeSets).toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: JSON.stringify({
          controlFlow: 'GraphInterrupt',
        }),
      })
    );
    expect(JSON.stringify(mockSpanAttributeSets)).not.toContain(
      sensitiveQuestion
    );
  });

  it('normalizes a real ask_user_question pause and preserves resume behavior', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const langfuse = {
      publicKey: 'pk-ask-user-question',
      secretKey: 'sk-ask-user-question',
    };
    initializeLangfuseTracing(langfuse);
    const handler = createLangfuseHandler({ langfuse });
    expect(handler).toBeDefined();

    const sensitiveQuestion = 'Should the workflow continue?';
    const askTool = tool(
      async () => {
        const { answer } = askUserQuestion({ question: sensitiveQuestion });
        return `answered: ${answer}`;
      },
      {
        name: 'ask_user_question',
        description: 'suspends to collect a human answer',
        schema: z.object({}).passthrough(),
      }
    );
    const toolNode = new ToolNode({
      tools: [askTool],
      eventDrivenMode: true,
      directToolNames: new Set(['ask_user_question']),
      interruptingToolNames: new Set(['ask_user_question']),
    });
    const graph = new StateGraph(MessagesAnnotation)
      .addNode('agent', () => ({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              { id: 'ask-call', name: 'ask_user_question', args: {} },
            ],
          }),
        ],
      }))
      .addNode('tools', toolNode)
      .addEdge(START, 'agent')
      .addEdge('agent', 'tools')
      .addEdge('tools', END)
      .compile({ checkpointer: new MemorySaver() });
    const chainErrorSpy = jest.spyOn(handler!, 'handleChainError');
    const toolErrorSpy = jest.spyOn(handler!, 'handleToolError');
    const config = {
      callbacks: [handler!],
      configurable: { thread_id: 'ask-user-question-langfuse' },
    };

    const first = await graph.invoke({ messages: [] }, config);

    expect(isInterrupted<t.HumanInterruptPayload>(first)).toBe(true);
    expect(
      (
        first as {
          __interrupt__?: Array<{
            value?: t.HumanInterruptPayload;
          }>;
        }
      ).__interrupt__?.[0]?.value
    ).toMatchObject({
      type: 'ask_user_question',
      question: { question: sensitiveQuestion },
    });
    expect(
      chainErrorSpy.mock.calls.some(
        ([error, , parentRunId]) =>
          isGraphInterrupt(error) && parentRunId != null
      )
    ).toBe(true);
    expect(
      toolErrorSpy.mock.calls.some(
        ([error, , parentRunId]) =>
          isGraphInterrupt(error) && parentRunId != null
      )
    ).toBe(true);
    const controlFlowOutputs = mockSpanAttributeSets.filter(
      (attributes) =>
        attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] ===
        JSON.stringify({ controlFlow: 'GraphInterrupt' })
    );
    expect(controlFlowOutputs.length).toBeGreaterThanOrEqual(2);
    expect(mockSpanAttributeSets).not.toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]: 'ERROR',
      })
    );
    expect(mockSpanAttributeSets).not.toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE]:
          expect.stringContaining('GraphInterrupt'),
      })
    );
    expect(mockSpansEnded).toBe(mockSpansStarted);

    const second = await graph.invoke(
      new Command({ resume: { answer: 'yes' } }),
      config
    );

    expect(isInterrupted<t.HumanInterruptPayload>(second)).toBe(false);
    const messages = (second as { messages: ToolMessage[] }).messages;
    expect(
      messages.find(
        (message) =>
          message instanceof ToolMessage && message.name === 'ask_user_question'
      )?.content
    ).toBe('answered: yes');
    expect(mockSpanAttributeSets).not.toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]: 'ERROR',
      })
    );
    expect(mockSpansEnded).toBe(mockSpansStarted);
  });

  it('defensively reports a synthetic root GraphInterrupt as an error', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const langfuse = {
      publicKey: 'pk-root-graph-interrupt',
      secretKey: 'sk-root-graph-interrupt',
    };
    initializeLangfuseTracing(langfuse);
    const handler = createLangfuseHandler({ langfuse });
    const runId = 'test-langfuse-root-graph-interrupt';

    await handler?.handleChainStart(
      { lc: 1, type: 'not_implemented', id: ['RootChain'] },
      { input: 'root interrupt' },
      runId
    );
    await handler?.handleChainError(new GraphInterrupt(), runId);

    expect(mockSpanAttributeSets).toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]: 'ERROR',
      })
    );
  });

  it('normalizes nested NodeInterrupt control flow', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const langfuse = {
      publicKey: 'pk-node-interrupt',
      secretKey: 'sk-node-interrupt',
    };
    initializeLangfuseTracing(langfuse);
    const handler = createLangfuseHandler({ langfuse });
    const runId = 'test-langfuse-node-interrupt';

    await handler?.handleChainStart(
      { lc: 1, type: 'not_implemented', id: ['InterruptedChain'] },
      { input: 'ask' },
      runId,
      'parent-run'
    );
    await handler?.handleChainError(
      new NodeInterrupt('pause'),
      runId,
      'parent-run'
    );

    expect(mockSpanAttributeSets).not.toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]: 'ERROR',
      })
    );
    expect(mockSpanAttributeSets).toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: JSON.stringify({
          controlFlow: 'GraphInterrupt',
        }),
      })
    );
  });

  it('continues reporting genuine nested tool failures as errors', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const langfuse = {
      publicKey: 'pk-genuine-tool-error',
      secretKey: 'sk-genuine-tool-error',
    };
    initializeLangfuseTracing(langfuse);
    const handler = createLangfuseHandler({ langfuse });
    const runId = 'test-langfuse-genuine-tool-error';

    await handler?.handleToolStart(
      { lc: 1, type: 'not_implemented', id: ['FailingTool'] },
      '{}',
      runId,
      'parent-run'
    );
    await handler?.handleToolError(new Error('tool boom'), runId, 'parent-run');

    expect(mockSpanAttributeSets).toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]: 'ERROR',
        [LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE]:
          'Error: tool boom',
      })
    );
  });

  it('reports a ParentCommand that escapes the root graph as an error', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const langfuse = {
      publicKey: 'pk-root-parent-command',
      secretKey: 'sk-root-parent-command',
    };
    initializeLangfuseTracing(langfuse);
    const handler = createLangfuseHandler({ langfuse });
    const runId = 'test-langfuse-root-parent-command';

    await handler?.handleChainStart(
      { lc: 1, type: 'not_implemented', id: ['RootChain'] },
      { input: 'root handoff' },
      runId
    );
    await handler?.handleChainError(
      new ParentCommand(
        new Command({ graph: Command.PARENT, goto: 'destination' })
      ),
      runId
    );

    expect(mockSpanAttributeSets).toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]: 'ERROR',
        [LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE]:
          'ParentCommand',
      })
    );
  });

  it('continues reporting genuine nested chain failures as errors', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const langfuse = {
      publicKey: 'pk-genuine-error',
      secretKey: 'sk-genuine-error',
    };
    initializeLangfuseTracing(langfuse);
    const handler = createLangfuseHandler({ langfuse });
    const runId = 'test-langfuse-genuine-error';

    await handler?.handleChainStart(
      { lc: 1, type: 'not_implemented', id: ['FailingChain'] },
      { input: 'fail' },
      runId,
      'parent-run'
    );
    await handler?.handleChainError(new Error('boom'), runId, 'parent-run');

    expect(mockSpanAttributeSets).toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]: 'ERROR',
        [LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE]: 'Error: boom',
      })
    );
  });

  it('delegates null chain errors without throwing during classification', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const langfuse = {
      publicKey: 'pk-null-error',
      secretKey: 'sk-null-error',
    };
    initializeLangfuseTracing(langfuse);
    const handler = createLangfuseHandler({ langfuse });
    const runId = 'test-langfuse-null-error';

    await handler?.handleChainStart(
      { lc: 1, type: 'not_implemented', id: ['NullErrorChain'] },
      { input: 'fail' },
      runId,
      'parent-run'
    );

    await expect(
      handler?.handleChainError(null, runId, 'parent-run')
    ).resolves.toBeUndefined();
  });

  it('delegates null tool errors without throwing during classification', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const langfuse = {
      publicKey: 'pk-null-tool-error',
      secretKey: 'sk-null-tool-error',
    };
    initializeLangfuseTracing(langfuse);
    const handler = createLangfuseHandler({ langfuse });
    const runId = 'test-langfuse-null-tool-error';

    await handler?.handleToolStart(
      { lc: 1, type: 'not_implemented', id: ['NullErrorTool'] },
      '{}',
      runId,
      'parent-run'
    );

    await expect(
      handler?.handleToolError(null, runId, 'parent-run')
    ).resolves.toBeUndefined();
  });

  it('traces a completed two-agent handoff without false error spans', async () => {
    const langfuse = {
      publicKey: 'pk-handoff',
      secretKey: 'sk-handoff',
    };
    const run = await Run.create<t.IState>({
      runId: 'test-langfuse-completed-handoff',
      graphConfig: {
        type: 'multi-agent',
        agents: [
          {
            agentId: 'agent_source',
            name: 'Source Agent',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4' },
            instructions: 'Transfer to the destination agent.',
            tools: [],
          },
          {
            agentId: 'agent_destination',
            name: 'Destination Agent',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4' },
            instructions: 'Complete the request.',
            tools: [],
          },
        ],
        edges: [
          {
            from: 'agent_source',
            to: 'agent_destination',
            edgeType: 'handoff',
          },
        ],
      },
      langfuse,
      skipCleanup: true,
    });
    run.Graph?.overrideTestModel(
      ['Transferring to destination', 'Destination complete'],
      10,
      [
        {
          id: 'handoff-call',
          name: `${Constants.LC_TRANSFER_TO_}agent_destination`,
          args: {},
        } as ToolCall,
      ]
    );

    await run.processStream(
      { messages: [new HumanMessage('Please hand this off')] },
      {
        configurable: {
          thread_id: 'test-langfuse-completed-handoff-thread',
        },
        version: 'v2',
      }
    );

    expect(
      run
        .getRunMessages()
        ?.some((message) => message.content === 'Destination complete')
    ).toBe(true);
    expect(mockSpanAttributeSets).toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: JSON.stringify({
          controlFlow: 'ParentCommand',
        }),
      })
    );
    expect(mockSpanAttributeSets).not.toContainEqual(
      expect.objectContaining({
        [LangfuseOtelSpanAttributes.OBSERVATION_LEVEL]: 'ERROR',
      })
    );
  });

  it('exports Bedrock prompt-cache usage buckets to Langfuse', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    initializeLangfuseTracing({
      publicKey: 'pk-test',
      secretKey: 'sk-test',
    });
    const handler = createLangfuseHandler({
      langfuse: {
        publicKey: 'pk-test',
        secretKey: 'sk-test',
      },
    });
    const runId = 'test-langfuse-bedrock-cache-usage';

    await handler?.handleChatModelStart(
      {
        lc: 1,
        type: 'constructor',
        id: ['LibreChatBedrockConverse'],
        kwargs: {},
      },
      [[new HumanMessage('hello')]],
      runId
    );

    const generation = handleConverseStreamMetadata(
      {
        usage: {
          inputTokens: 13,
          outputTokens: 5,
          totalTokens: 20849,
          cacheReadInputTokens: 10831,
          cacheWriteInputTokens: 10000,
        },
        metrics: { latencyMs: 1000 },
      },
      { streamUsage: true }
    );
    await handler?.handleLLMEnd({ generations: [[generation]] }, runId);

    const usageDetails = mockSpanAttributeSets
      .map(
        (attributes) =>
          attributes[LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS]
      )
      .find((value): value is string => typeof value === 'string');

    expect(usageDetails).toBe(
      JSON.stringify({
        input: 13,
        output: 5,
        total: 20849,
        input_cache_read: 10831,
        input_cache_creation: 10000,
      })
    );
  });

  it('uses deterministic trace ids when tracing is configured from env only', async () => {
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-env';
    process.env.LANGFUSE_SECRET_KEY = 'sk-env';
    process.env.LANGFUSE_BASE_URL = 'https://langfuse.env';

    const runId = 'test-langfuse-env-deterministic-run';
    const run = await Run.create<t.IState>({
      runId,
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'agent_abc123',
            name: 'DWAINE',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4' },
            tools: [],
          },
        ],
      },
      langfuse: {
        deterministicTraceId: true,
        metadata: { 'librechat.tenant.id': 'tenant-env' },
        tags: ['tenant:tenant-env'],
      },
      skipCleanup: true,
    });

    run.Graph?.overrideTestModel(['hello']);

    await run.processStream(
      { messages: [new HumanMessage('hello')] },
      {
        configurable: { thread_id: 'thread-1', user_id: 'user-1' },
        version: 'v2' as const,
      }
    );

    expect(mockProcessorStarts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          params: expect.objectContaining({
            publicKey: 'pk-env',
            secretKey: 'sk-env',
            baseUrl: 'https://langfuse.env',
          }),
          traceId: traceIdFromSeed(runId),
        }),
      ])
    );
  });

  it('adds current agent metadata when a stream Langfuse callback already exists', async () => {
    const metadataSpy = jest.fn();
    const { createLangfuseHandler } = await import('@/langfuse');
    const streamHandler = createLangfuseHandler({
      langfuse: {
        publicKey: 'pk-run',
        secretKey: 'sk-run',
        baseUrl: 'https://langfuse.run',
      },
    });
    const run = await Run.create<t.IState>({
      runId: 'test-langfuse-agent-metadata-with-stream-callback',
      graphConfig: {
        type: 'multi-agent',
        agents: [
          {
            agentId: 'agent_default',
            name: 'Default Agent',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4' },
            tools: [],
          },
          {
            agentId: 'agent_specialist',
            name: 'Specialist Agent',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4' },
            tools: [],
          },
        ],
        edges: [],
      },
      skipCleanup: true,
    });

    run.Graph?.overrideTestModel(['hello from specialist']);
    const agentNode = run.Graph?.createAgentNode('agent_specialist');
    await agentNode?.invoke(
      { messages: [new HumanMessage('hello')] },
      {
        callbacks: [
          ...(streamHandler != null ? [streamHandler] : []),
          {
            handleChatModelStart: async (
              _llm: unknown,
              _messages: unknown,
              _runId: string,
              _parentRunId?: string,
              _extraParams?: unknown,
              _tags?: string[],
              metadata?: Record<string, unknown>
            ): Promise<void> => {
              metadataSpy(metadata);
            },
          },
        ],
        configurable: { thread_id: 'thread-1', user_id: 'user-1' },
      }
    );

    expect(metadataSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        agentId: 'agent_specialist',
        agentName: 'Specialist Agent',
      })
    );
  });
});
