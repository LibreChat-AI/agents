import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import {
  createContextPressureMeter,
  createExactTokenCountCache,
} from './contextPressureMeter';
import { stampSyntheticProviderMessage } from '@/messages';

function contentLength(message: BaseMessage): number {
  return typeof message.content === 'string'
    ? message.content.length
    : JSON.stringify(message.content).length;
}

describe('createContextPressureMeter', () => {
  it.each([
    { length: 147, projectedMessageTokens: 150, fits: false },
    { length: 88, projectedMessageTokens: 91, fits: false },
    { length: 87, projectedMessageTokens: 90, fits: true },
  ])(
    'preserves baseline overage when projected content has $length tokens',
    ({ length, projectedMessageTokens, fits }) => {
      const retained = new HumanMessage('x'.repeat(147));
      const meter = createContextPressureMeter({
        tokenCounter: contentLength,
        sourceMessages: [retained],
        retainedMessages: [retained],
        indexTokenCountMap: { 0: 147 },
        contextUsage: {
          contextBudget: 100,
          effectiveInstructionTokens: 10,
          remainingContextTokens: -60,
          calibrationRatio: 1,
        },
        instructionTokens: 10,
        calibrationRatio: 1,
      });
      const projected =
        length === 147 ? retained : new HumanMessage('x'.repeat(length));
      meter.trackClone(retained, projected);

      expect(meter.measure([projected])).toMatchObject({
        fits,
        projectedMessageTokens,
        availableMessageTokens: 90,
      });
    }
  );

  it('preserves provider-grounded attribution while caching exact counts', () => {
    const source = [
      new HumanMessage({ id: 'first', content: 'a' }),
      new HumanMessage({ id: 'second', content: 'bb' }),
    ];
    const tokenCounter = jest.fn(contentLength);
    const meter = createContextPressureMeter({
      tokenCounter,
      sourceMessages: source,
      retainedMessages: source,
      indexTokenCountMap: { 0: 10, 1: 20 },
      contextUsage: {
        contextBudget: 100,
        effectiveInstructionTokens: 10,
        remainingContextTokens: 60,
        calibrationRatio: 1,
      },
      instructionTokens: 10,
      calibrationRatio: 1,
    });
    const projected = [
      new HumanMessage({ id: 'first', content: 'aaaa' }),
      source[1],
    ];
    meter.trackProjection(source, projected);

    expect(meter.measure(projected)).toEqual({
      fits: true,
      projectedMessageTokens: 33,
      availableMessageTokens: 90,
      contextBudget: 100,
      effectiveInstructionTokens: 10,
      toolMessageTokens: 0,
      toolMessageTokenCounts: undefined,
    });
    expect(meter.measure(projected).projectedMessageTokens).toBe(33);
    expect(tokenCounter).toHaveBeenCalledTimes(2);
  });

  it('does not recount unchanged rich content for tool attribution', () => {
    const retained = new HumanMessage({
      content: [{ type: 'text', text: 'rich retained content' }],
    });
    const tokenCounter = jest.fn(contentLength);
    const meter = createContextPressureMeter({
      tokenCounter,
      sourceMessages: [retained],
      retainedMessages: [retained],
      indexTokenCountMap: { 0: 20 },
      contextUsage: {
        contextBudget: 100,
        effectiveInstructionTokens: 10,
        remainingContextTokens: 70,
        calibrationRatio: 1,
      },
      instructionTokens: 10,
      calibrationRatio: 1,
    });

    expect(meter.measure([retained]).toolMessageTokens).toBe(0);
    expect(tokenCounter).not.toHaveBeenCalled();
  });

  it('avoids recounting provider-grounded messages across meters', () => {
    const retained = Array.from(
      { length: 100 },
      (_, index) =>
        new HumanMessage({ id: `message-${index}`, content: 'retained' })
    );
    const tokenCounter = jest.fn(contentLength);
    const createMeter = (messages: BaseMessage[]) =>
      createContextPressureMeter({
        tokenCounter,
        sourceMessages: messages,
        retainedMessages: messages,
        indexTokenCountMap: Object.fromEntries(
          messages.map((_, index) => [index, 8])
        ),
        contextUsage: {
          contextBudget: 10_000,
          effectiveInstructionTokens: 100,
          remainingContextTokens: 9_000,
          calibrationRatio: 1,
        },
        instructionTokens: 100,
        calibrationRatio: 1,
      });

    createMeter(retained).measure(retained);
    createMeter(retained).measure(retained);

    const appended = [...retained, new HumanMessage('new')];
    createMeter(appended).measure(appended);

    expect(tokenCounter).not.toHaveBeenCalled();
  });

  it('reuses exact counts across request-scoped meters when no stored counts exist', () => {
    const retained = Array.from(
      { length: 100 },
      (_, index) =>
        new HumanMessage({ id: `message-${index}`, content: 'retained' })
    );
    const tokenCounter = jest.fn(contentLength);
    const tokenCountCache = createExactTokenCountCache(tokenCounter);
    const createMeter = (messages: BaseMessage[]) =>
      createContextPressureMeter({
        tokenCounter,
        tokenCountCache,
        sourceMessages: messages,
        retainedMessages: messages,
        indexTokenCountMap: {},
        contextUsage: {
          contextBudget: 10_000,
          effectiveInstructionTokens: 100,
          remainingContextTokens: 9_000,
          calibrationRatio: 1,
        },
        instructionTokens: 100,
        calibrationRatio: 1,
      });

    createMeter(retained).measure(retained);
    createMeter(retained).measure(retained);

    const appended = [...retained, new HumanMessage('new')];
    createMeter(appended).measure(appended);

    expect(tokenCounter).toHaveBeenCalledTimes(101);
  });

  it('recounts stable messages after a token-relevant mutation', () => {
    const message = new HumanMessage('short');
    const tokenCounter = jest.fn(contentLength);
    const cache = createExactTokenCountCache(tokenCounter);

    expect(cache.count(message)).toBe(5);
    message.content = 'longer content';
    expect(cache.count(message)).toBe(14);
    expect(tokenCounter).toHaveBeenCalledTimes(2);
  });

  it('keeps mutable complex messages on the exact recount path', () => {
    const message = new HumanMessage({
      content: [{ type: 'text', text: 'mutable' }],
    });
    const tokenCounter = jest.fn(contentLength);
    const cache = createExactTokenCountCache(tokenCounter);

    cache.count(message);
    cache.count(message);

    expect(tokenCounter).toHaveBeenCalledTimes(2);
  });

  it('stops reusing an AI count when tool calls become mutable', () => {
    const message = new AIMessage({ content: 'answer', tool_calls: [] });
    const tokenCounter = jest.fn(contentLength);
    const cache = createExactTokenCountCache(tokenCounter);

    cache.count(message);
    message.tool_calls = new Proxy([], {});
    cache.count(message);

    expect(tokenCounter).toHaveBeenCalledTimes(2);
  });

  it('recounts a tool message when its provider type changes', () => {
    const message = new ToolMessage({
      content: 'result',
      tool_call_id: 'call-1',
    });
    const tokenCounter = jest.fn(contentLength);
    const cache = createExactTokenCountCache(tokenCounter);

    cache.count(message);
    message.additional_kwargs.type = 'computer_call_output';
    cache.count(message);

    expect(tokenCounter).toHaveBeenCalledTimes(2);
  });

  it('recounts when provider metadata gains an accessor', () => {
    const message = new HumanMessage('content');
    const tokenCounter = jest.fn(contentLength);
    const cache = createExactTokenCountCache(tokenCounter);

    cache.count(message);
    Object.defineProperty(message.additional_kwargs, 'type', {
      configurable: true,
      get: () => 'computer_call_output',
    });
    cache.count(message);

    expect(tokenCounter).toHaveBeenCalledTimes(2);
  });

  it('does not cache messages carrying mutable raw tool calls', () => {
    const message = new AIMessage('answer');
    const tokenCounter = jest.fn(contentLength);
    const cache = createExactTokenCountCache(tokenCounter);

    cache.count(message);
    message.additional_kwargs.tool_calls = [
      {
        id: 'raw-call',
        type: 'function',
        function: { name: 'lookup', arguments: '{}' },
      },
    ];
    cache.count(message);
    message.additional_kwargs.tool_calls[0]!.function.arguments =
      '{"query":"changed"}';
    cache.count(message);

    expect(tokenCounter).toHaveBeenCalledTimes(3);
  });

  it('does not invoke prototype proxy traps during cache checks', () => {
    const message = new AIMessage('answer');
    const tokenCounter = jest.fn(contentLength);
    const cache = createExactTokenCountCache(tokenCounter);
    let hasCalls = 0;
    Object.setPrototypeOf(
      message.additional_kwargs,
      new Proxy(
        {},
        {
          has: () => {
            hasCalls += 1;
            throw new Error('must not run');
          },
        }
      )
    );

    cache.count(message);
    cache.count(message);

    expect(hasCalls).toBe(0);
    expect(tokenCounter).toHaveBeenCalledTimes(2);
  });

  it('recounts inherited revoked parsed tool-call proxies safely', () => {
    const message = new AIMessage('answer');
    const tokenCounter = jest.fn(contentLength);
    const cache = createExactTokenCountCache(tokenCounter);
    const { proxy, revoke } = Proxy.revocable([], {});
    const inherited = Object.create(Object.getPrototypeOf(message));
    Object.defineProperty(inherited, 'tool_calls', { value: proxy });
    delete message.tool_calls;
    Object.setPrototypeOf(message, inherited);
    revoke();

    expect(() => cache.count(message)).not.toThrow();
    expect(() => cache.count(message)).not.toThrow();
    expect(tokenCounter).toHaveBeenCalledTimes(2);
  });

  it('matches forced recounts across append, replace, reorder, and mutation', () => {
    const first = new HumanMessage({ id: 'first', content: 'one' });
    const second = new HumanMessage({ id: 'second', content: 'two' });
    const third = new AIMessage({ id: 'third', content: 'three' });
    const tokenCountCache = createExactTokenCountCache(contentLength);
    const histories = [
      [first, second],
      [first, second, third],
      [first, new HumanMessage({ id: 'replacement', content: 'replacement' })],
      [second, first, third],
    ];

    for (const history of histories) {
      const indexTokenCountMap = Object.fromEntries(
        history.map((message, index) => [index, contentLength(message)])
      );
      const params = {
        tokenCounter: contentLength,
        sourceMessages: history,
        retainedMessages: history,
        indexTokenCountMap,
        contextUsage: {
          contextBudget: 100,
          effectiveInstructionTokens: 10,
          remainingContextTokens: 50,
          calibrationRatio: 1.25,
        },
        instructionTokens: 10,
        calibrationRatio: 1.25,
      };
      const cached = createContextPressureMeter({
        ...params,
        tokenCountCache,
      }).measure(history);
      const recounted = createContextPressureMeter(params).measure(history);

      expect(cached).toEqual(recounted);
    }

    first.content = 'mutated after caching';
    const mutatedParams = {
      tokenCounter: contentLength,
      sourceMessages: [first],
      retainedMessages: [first],
      indexTokenCountMap: { 0: contentLength(first) },
      contextUsage: {
        contextBudget: 100,
        effectiveInstructionTokens: 10,
        remainingContextTokens: 50,
        calibrationRatio: 1.25,
      },
      instructionTokens: 10,
      calibrationRatio: 1.25,
    };

    expect(
      createContextPressureMeter({
        ...mutatedParams,
        tokenCountCache,
      }).measure([first])
    ).toEqual(createContextPressureMeter(mutatedParams).measure([first]));
  });

  it('tokenizes only changed candidates across a compaction search', () => {
    const retained = Array.from(
      { length: 100 },
      (_, index) =>
        new HumanMessage({ id: `message-${index}`, content: 'retained' })
    );
    const tokenCounter = jest.fn(contentLength);
    const meter = createContextPressureMeter({
      tokenCounter,
      sourceMessages: retained,
      retainedMessages: retained,
      indexTokenCountMap: Object.fromEntries(
        retained.map((_, index) => [index, 8])
      ),
      contextUsage: {
        contextBudget: 10_000,
        effectiveInstructionTokens: 100,
        remainingContextTokens: 9_000,
        calibrationRatio: 1,
      },
      instructionTokens: 100,
      calibrationRatio: 1,
    });

    for (let i = 0; i < 13; i++) {
      meter.measure([
        ...retained,
        stampSyntheticProviderMessage(
          new HumanMessage({ content: 'x'.repeat(i + 1) })
        ),
      ]);
    }

    expect(tokenCounter).toHaveBeenCalledTimes(13);
  });

  it('uses a conservative ratio for fallback payloads', () => {
    const message = new HumanMessage('1234567890');
    const meter = createContextPressureMeter({
      tokenCounter: contentLength,
      sourceMessages: [message],
      retainedMessages: [message],
      indexTokenCountMap: { 0: 10 },
      instructionTokens: 5,
      calibrationRatio: 0.5,
    });

    expect(
      meter.measure([message], {
        contextBudget: 17,
        forceRawRecount: true,
      })
    ).toEqual({
      fits: false,
      projectedMessageTokens: 13,
      availableMessageTokens: 12,
      contextBudget: 17,
      effectiveInstructionTokens: 5,
      toolMessageTokens: 0,
      toolMessageTokenCounts: undefined,
    });
  });

  it('measures tool share during the provider-payload pass', () => {
    const messages = [
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'call-1', name: 'lookup', args: {} }],
      }),
      new ToolMessage({
        content: 'result',
        tool_call_id: 'call-1',
      }),
    ];
    const tokenCounter = jest.fn(() => 0.5);
    const meter = createContextPressureMeter({
      tokenCounter,
      sourceMessages: messages,
      retainedMessages: messages,
      indexTokenCountMap: {},
      contextUsage: {
        contextBudget: 100,
        effectiveInstructionTokens: 10,
        remainingContextTokens: 90,
        calibrationRatio: 1,
      },
      instructionTokens: 10,
      calibrationRatio: 1,
    });

    expect(meter.measure(messages)).toMatchObject({
      toolMessageTokens: 1,
      toolMessageTokenCounts: { lookup: 1 },
    });
    expect(tokenCounter).toHaveBeenCalledTimes(2);
  });

  it('surfaces invalid tool-share counts without invalidating the projection', () => {
    const message = new ToolMessage({
      content: 'result',
      tool_call_id: 'call-1',
    });
    const meter = createContextPressureMeter({
      tokenCounter: () => Number.NaN,
      sourceMessages: [message],
      retainedMessages: [message],
      indexTokenCountMap: { 0: 10 },
      contextUsage: {
        contextBudget: 100,
        effectiveInstructionTokens: 10,
        remainingContextTokens: 80,
        calibrationRatio: 1,
      },
      instructionTokens: 10,
      calibrationRatio: 1,
    });

    expect(meter.measure([message])).toMatchObject({
      fits: true,
      projectedMessageTokens: 10,
      toolMessageUsageError: expect.any(RangeError),
    });
  });
});
