import { randomUUID } from 'node:crypto';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
} from '@langchain/core/messages';
import type { ProviderTextProtection } from '@/protection/providerText';
import type { HostModelService, HostModelDescriptor } from './model';
import type { ModelInvocationContext } from '@/llm/invocation';
import { withProviderTextBoundary } from '@/llm/providerTextBoundary';
import { HostChatModel } from './model';

const descriptor: HostModelDescriptor = {
  version: 1,
  reference: 'primary',
  provider: 'openAI',
  model: 'fixture',
  streaming: true,
  tools: [],
};
const identity = {
  runtimeId: 'runtime',
  agentId: 'agent',
  userId: 'user',
  tenantId: 'tenant',
  runId: 'run',
};
const policy: ProviderTextProtection = {
  version: 1,
  timeoutMs: 1000,
  maxAttemptBytes: 4096,
  maxBufferedBytes: 8192,
  classify: () => 'prose',
  inspect: () => ({
    version: 1,
    ok: true,
    value: { content: 'approved', replacements: 1, categories: [] },
  }),
};
const messages = [new HumanMessage('hello')];

it.each([false, true])(
  'preserves native invocation identity across protection (stream=%s)',
  async (stream) => {
    const contexts: (Readonly<ModelInvocationContext> | undefined)[] = [];
    const service: HostModelService = {
      version: 1,
      invoke: async (_request, _signal, context) => {
        contexts.push(context);
        return new AIMessage('PRIVATE');
      },
      stream: async function* (_request, _signal, context) {
        contexts.push(context);
        await Promise.resolve();
        yield new AIMessageChunk('PRIVATE');
      },
    };
    const model = new HostChatModel({
      identity,
      descriptor: { ...descriptor, streaming: stream },
      service,
    });
    const protectedModel = withProviderTextBoundary(
      withProviderTextBoundary(model, policy),
      policy
    );
    const tokens: string[] = [];
    const callbacks = [
      {
        handleLLMNewToken: (token: string) => {
          tokens.push(token);
        },
      },
    ];
    const ids = [randomUUID(), randomUUID()];
    const answers = await Promise.all(
      ids.map(async (runId) => {
        if (!stream)
          return (await protectedModel.invoke(messages, { runId, callbacks }))
            .content;
        if (!protectedModel.stream) throw new Error('Streaming unavailable');
        let text = '';
        for await (const chunk of await protectedModel.stream(messages, {
          runId,
          callbacks,
        }))
          text += chunk.content;
        return text;
      })
    );
    expect(answers).toEqual(['approved', 'approved']);
    expect(contexts).toEqual(ids.map((runId) => ({ version: 1, runId })));
    expect(contexts.every(Object.isFrozen)).toBe(true);
    expect(tokens.join('')).not.toContain('PRIVATE');
    // Public/direct calls without a callback manager cannot inherit a previous call or forge context in options.
    const forged = {
      stop: [],
      metadata: { runId: ids[0] },
      configurable: { runId: ids[1] },
    };
    await model._generate(messages, forged);
    expect(contexts[2]).toBeUndefined();
  }
);

it('preserves context without exposing rejected provider output to callbacks', async () => {
  const seen: string[] = [];
  const runId = randomUUID();
  const invoke: HostModelService['invoke'] = jest.fn(
    async (_request, _signal, context) => {
      expect(context).toEqual({ version: 1, runId });
      return new AIMessage('PRIVATE');
    }
  );
  const model = new HostChatModel({
    identity,
    descriptor: { ...descriptor, streaming: false },
    service: { version: 1, invoke, stream: async function* () {} },
  });
  const protectedModel = withProviderTextBoundary(model, {
    ...policy,
    inspect: () => ({ version: 1, ok: false, error: { code: 'blocked' } }),
  });
  await expect(
    protectedModel.invoke(messages, {
      runId,
      callbacks: [
        {
          handleLLMNewToken: (token: string) => {
            seen.push(token);
          },
        },
      ],
    })
  ).rejects.toThrow();
  expect(invoke).toHaveBeenCalledTimes(1);
  expect(seen.join('')).not.toContain('PRIVATE');
});
