// src/hooks/__tests__/compactHooks.test.ts
import { FakeListChatModel } from '@langchain/core/utils/testing';
import { HumanMessage, AIMessage } from '@langchain/core/messages';
import type {
  HookCallback,
  PreCompactHookInput,
  PreCompactHookOutput,
  PostCompactHookInput,
  PostCompactHookOutput,
} from '../types';
import type * as t from '@/types';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { createTokenCounter } from '@/utils/tokens';
import { Providers, GraphEvents } from '@/common';
import { HookRegistry } from '../HookRegistry';
import * as providers from '@/llm/providers';
import { Run } from '@/run';

const SUMMARY_RESPONSE = '## Summary\nUser asked a question and got an answer.';

const callerConfig = {
  configurable: { thread_id: 'compact-test' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

const childExecutionContext: t.SubagentExecutionContext = {
  rootRunId: 'root-run',
  hookSessionId: 'compact-run',
  depth: 1,
  ancestry: [
    {
      subagentRunId: 'child-run',
      subagentType: 'worker',
      subagentKind: 'agent',
      subagentAgentId: 'worker',
      parentRunId: 'root-run',
    },
  ],
};

let getChatModelClassSpy: jest.SpyInstance;
const originalGetChatModelClass = providers.getChatModelClass;

function mockSummarizationModel(): void {
  getChatModelClassSpy = jest
    .spyOn(providers, 'getChatModelClass')
    .mockImplementation(((provider: Providers) => {
      if (provider === Providers.OPENAI) {
        return class extends FakeListChatModel {
          constructor(
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            _options: any
          ) {
            super({ responses: [SUMMARY_RESPONSE] });
          }
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } as any;
      }
      return originalGetChatModelClass(provider);
    }) as typeof providers.getChatModelClass);
}

function buildConversation(): t.IState {
  const messages: import('@langchain/core/messages').BaseMessage[] = [];
  for (let i = 0; i < 20; i++) {
    messages.push(new HumanMessage(`Question ${i}: ` + 'padding '.repeat(50)));
    messages.push(new AIMessage(`Answer ${i}: ` + 'response '.repeat(50)));
  }
  return { messages };
}

async function createCompactingRun(
  tokenCounter: t.TokenCounter,
  hooks?: HookRegistry,
  runId = 'compact-run',
  /**
   * Recency-window setting.  Defaults to `{ turns: 0 }` for these tests so
   * the post-compaction state is the legacy "remove-all only" shape, which
   * the original assertions were written against.  Tests that want to
   * exercise the recency-window path should pass an explicit value.
   */
  retainRecent: { turns: number } = { turns: 0 }
): Promise<Run<t.IState>> {
  const conversation = buildConversation();
  const indexTokenCountMap: Record<string, number> = {};
  for (let i = 0; i < conversation.messages.length; i++) {
    indexTokenCountMap[String(i)] = tokenCounter(conversation.messages[i]);
  }
  return Run.create<t.IState>({
    runId,
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: Providers.OPENAI,
        streaming: true,
        streamUsage: false,
      },
      instructions: 'Be concise.',
      maxContextTokens: 2_000,
      summarizationEnabled: true,
      summarizationConfig: {
        provider: Providers.OPENAI,
        retainRecent,
      },
    },
    returnContent: true,
    skipCleanup: true,
    customHandlers: {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    },
    hooks,
    tokenCounter,
    indexTokenCountMap,
  });
}

describe('Compaction hook integration', () => {
  jest.setTimeout(30_000);

  let tokenCounter: t.TokenCounter;

  beforeAll(async () => {
    tokenCounter = await createTokenCounter();
  });

  beforeEach(() => {
    mockSummarizationModel();
  });

  afterEach(() => {
    getChatModelClassSpy.mockRestore();
  });

  describe('PreCompact', () => {
    it('fires with messagesBeforeCount and trigger', async () => {
      const registry = new HookRegistry();
      let captured: PreCompactHookInput | undefined;
      const hook: HookCallback<'PreCompact'> = async (
        input
      ): Promise<PreCompactHookOutput> => {
        captured = input;
        return {};
      };
      registry.register('PreCompact', { hooks: [hook] });

      const run = await createCompactingRun(tokenCounter, registry);
      run.Graph!.overrideTestModel(['Final answer after compaction.']);
      const inputs = buildConversation();
      await run.processStream(inputs, callerConfig);

      expect(captured).toBeDefined();
      expect(captured!.hook_event_name).toBe('PreCompact');
      expect(captured!.messagesBeforeCount).toBeGreaterThan(0);
      expect(captured!.runId).toBe('compact-run');
      expect(captured!.threadId).toBe('compact-test');
      expect(captured!.trigger).toBe('default');
      expect(captured!.agentId).toBeDefined();
    });
  });

  it('carries child execution identity through compact hooks', async () => {
    const registry = new HookRegistry();
    let pre: PreCompactHookInput | undefined;
    let post: PostCompactHookInput | undefined;
    registry.register('PreCompact', {
      hooks: [async (input) => ((pre = input), {})],
    });
    registry.register('PostCompact', {
      hooks: [async (input) => ((post = input), {})],
    });
    const run = await createCompactingRun(tokenCounter, registry);
    Object.defineProperty(run.Graph, 'subagentExecutionContext', {
      value: childExecutionContext,
    });
    expect(run.Graph?.subagentExecutionContext).toEqual(childExecutionContext);
    run.Graph!.overrideTestModel(['Final answer after compaction.']);

    await run.processStream(buildConversation(), callerConfig);

    expect(pre?.executionContext).toEqual(childExecutionContext);
    expect(post?.executionContext).toEqual(childExecutionContext);
  });

  describe('PostCompact', () => {
    it('fires with summary text after compaction (legacy retainRecent.turns=0 shape)', async () => {
      const registry = new HookRegistry();
      let captured: PostCompactHookInput | undefined;
      const hook: HookCallback<'PostCompact'> = async (
        input
      ): Promise<PostCompactHookOutput> => {
        captured = input;
        return {};
      };
      registry.register('PostCompact', { hooks: [hook] });

      const run = await createCompactingRun(tokenCounter, registry);
      run.Graph!.overrideTestModel(['Final answer after compaction.']);
      const inputs = buildConversation();
      await run.processStream(inputs, callerConfig);

      expect(captured).toBeDefined();
      expect(captured!.hook_event_name).toBe('PostCompact');
      expect(captured!.threadId).toBe('compact-test');
      expect(captured!.summary).toBe(SUMMARY_RESPONSE);
      expect(captured!.messagesAfterCount).toBe(0);
      expect(captured!.agentId).toBeDefined();
    });

    it('reports the recency-tail length in messagesAfterCount when retainRecent.turns > 0', async () => {
      const registry = new HookRegistry();
      let captured: PostCompactHookInput | undefined;
      const hook: HookCallback<'PostCompact'> = async (
        input
      ): Promise<PostCompactHookOutput> => {
        captured = input;
        return {};
      };
      registry.register('PostCompact', { hooks: [hook] });

      const run = await createCompactingRun(
        tokenCounter,
        registry,
        'compact-recency-run',
        { turns: 2 }
      );
      run.Graph!.overrideTestModel(['Final answer after compaction.']);
      const inputs = buildConversation();
      await run.processStream(inputs, callerConfig);

      expect(captured).toBeDefined();
      expect(captured!.hook_event_name).toBe('PostCompact');
      // buildConversation produces 20 user-led turns of [user, ai].  With
      // retainRecent.turns=2, the tail is the last 2 turns = 4 messages.
      expect(captured!.messagesAfterCount).toBe(4);
    });
  });

  describe('error resilience', () => {
    it('throwing PreCompact hook does not crash compaction', async () => {
      const registry = new HookRegistry();
      const throwingHook: HookCallback<
        'PreCompact'
      > = async (): Promise<PreCompactHookOutput> => {
        throw new Error('pre hook crash');
      };
      registry.register('PreCompact', { hooks: [throwingHook] });

      const run = await createCompactingRun(tokenCounter, registry);
      run.Graph!.overrideTestModel(['Answer after compaction.']);
      const inputs = buildConversation();

      await expect(
        run.processStream(inputs, callerConfig)
      ).resolves.not.toThrow();
    });

    it('throwing PostCompact hook does not crash compaction', async () => {
      const registry = new HookRegistry();
      const throwingHook: HookCallback<
        'PostCompact'
      > = async (): Promise<PostCompactHookOutput> => {
        throw new Error('post hook crash');
      };
      registry.register('PostCompact', { hooks: [throwingHook] });

      const run = await createCompactingRun(tokenCounter, registry);
      run.Graph!.overrideTestModel(['Answer after compaction.']);
      const inputs = buildConversation();

      await expect(
        run.processStream(inputs, callerConfig)
      ).resolves.not.toThrow();
    });
  });

  describe('no-hooks baseline', () => {
    it('summarization works identically without hooks', async () => {
      const run = await createCompactingRun(tokenCounter);
      run.Graph!.overrideTestModel(['Answer.']);
      const inputs = buildConversation();
      const result = await run.processStream(inputs, callerConfig);

      expect(result).toBeDefined();
    });
  });
});
