import { Tokenizer } from 'ai-tokenizer';
import * as claudeEncoding from 'ai-tokenizer/encoding/claude';
import {
  AIMessage,
  ChatMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type { ContentBlock } from '@langchain/core/messages';
import {
  encodingForModel,
  createTokenCounter,
  getTokenCountForMessage,
  hasUnsafeStructuredSerialization,
  TokenEncoderManager,
  estimateImageBlockTokens,
  estimateDocumentBlockTokens,
  estimateTimedMediaBlockTokens,
  UnsafeTokenMeasurementError,
} from '@/utils/tokens';

/** Builds a minimal PNG data URI whose IHDR encodes the given dimensions. */
function pngDataUri(width: number, height: number): string {
  const buf = Buffer.alloc(48);
  buf[0] = 0x89;
  buf[1] = 0x50;
  buf[2] = 0x4e;
  buf[3] = 0x47;
  buf[4] = 0x0d;
  buf[5] = 0x0a;
  buf[6] = 0x1a;
  buf[7] = 0x0a;
  buf.writeUInt32BE(width, 16);
  buf.writeUInt32BE(height, 20);
  return `data:image/png;base64,${buf.toString('base64')}`;
}

describe('encodingForModel', () => {
  test('returns claude for Claude model strings', () => {
    expect(encodingForModel('claude-3-5-sonnet-20241022')).toBe('claude');
    expect(encodingForModel('claude-3-haiku-20240307')).toBe('claude');
  });

  test('handles Bedrock Claude ARNs', () => {
    expect(encodingForModel('anthropic.claude-3-5-sonnet-20241022-v2:0')).toBe(
      'claude'
    );
  });

  test('is case-insensitive', () => {
    expect(encodingForModel('CLAUDE-3-HAIKU')).toBe('claude');
    expect(encodingForModel('Claude-3-Opus')).toBe('claude');
  });

  test('returns o200k_base for non-Claude models', () => {
    expect(encodingForModel('gpt-4o')).toBe('o200k_base');
    expect(encodingForModel('gemini-2.0-flash')).toBe('o200k_base');
    expect(encodingForModel('mistral-large')).toBe('o200k_base');
  });

  test('returns o200k_base for empty string', () => {
    expect(encodingForModel('')).toBe('o200k_base');
  });
});

describe('createTokenCounter with different encodings', () => {
  beforeEach(() => {
    TokenEncoderManager.reset();
  });

  test('claude encoding produces valid token counts', async () => {
    const counter = await createTokenCounter('claude');
    const msg = new HumanMessage('Hello, world!');
    const count = counter(msg);
    expect(count).toBeGreaterThan(0);
  });

  test('claude correction keeps proxy-backed tool history safely measurable', async () => {
    let getCalls = 0;
    const args = new Proxy(
      { query: 'select * from reports' },
      {
        get(target, property, receiver) {
          getCalls++;
          return Reflect.get(target, property, receiver);
        },
      }
    );
    const message = {
      content: '',
      tool_calls: [{ id: 'proxy-call', name: 'query', args }],
      getType: () => 'ai',
    } as unknown as AIMessage;
    const counter = await createTokenCounter('claude');

    const count = counter(message);

    expect(Number.isSafeInteger(count)).toBe(true);
    expect(count).toBeGreaterThan(0);
    expect(count).toBeLessThan(1_000);
    expect(getCalls).toBe(0);
  });

  test('o200k_base encoding produces valid token counts', async () => {
    const counter = await createTokenCounter('o200k_base');
    const msg = new HumanMessage('Hello, world!');
    const count = counter(msg);
    expect(count).toBeGreaterThan(0);
  });

  test('both encodings can be initialized and used independently', async () => {
    const claudeCounter = await createTokenCounter('claude');
    const o200kCounter = await createTokenCounter('o200k_base');
    expect(TokenEncoderManager.isInitialized()).toBe(true);

    const msg = new HumanMessage('Test message for both encodings');
    expect(claudeCounter(msg)).toBeGreaterThan(0);
    expect(o200kCounter(msg)).toBeGreaterThan(0);
  });
});

describe('getTokenCountForMessage', () => {
  const messageWithToolArgs = (args: unknown): AIMessage =>
    ({
      content: '',
      tool_calls: [{ id: 'adversarial-call', name: 'stress', args }],
      getType: () => 'ai',
    }) as unknown as AIMessage;

  test('counts long plain-text messages close to the real Claude tokenizer result', () => {
    const tokenizer = new Tokenizer(claudeEncoding);
    const phrase = 'The quick brown fox jumps over the lazy dog. ';
    const content = phrase
      .repeat(Math.ceil(361_473 / phrase.length))
      .slice(0, 361_473);
    const exactCount = tokenizer.count(content) + 3;

    const count = getTokenCountForMessage(
      new HumanMessage(content),
      (text) => tokenizer.count(text),
      'claude'
    );

    expect(count).toBeGreaterThanOrEqual(exactCount);
    expect(count / exactCount).toBeLessThan(1.005);
  });

  test('counts a high-density suffix instead of extrapolating from low-density text', () => {
    const tokenizer = new Tokenizer(claudeEncoding);
    const phrase = 'The quick brown fox jumps over the lazy dog. ';
    const prose = phrase
      .repeat(Math.ceil(200_000 / phrase.length))
      .slice(0, 200_000);
    const content = `${prose}${'😀'.repeat(8_000)}`;
    const exactCount = tokenizer.count(content) + 3;

    const count = getTokenCountForMessage(
      new HumanMessage(content),
      (text) => tokenizer.count(text),
      'claude'
    );

    expect(count).toBeGreaterThanOrEqual(exactCount);
    expect(count / exactCount).toBeLessThan(1.005);
  });

  test('stays conservative when a chunk boundary changes tokenizer segmentation', () => {
    const tokenizer = new Tokenizer(claudeEncoding);
    const boundaryPrefix = '\n\n \n\n ';
    const boundarySuffix = '😀é a0é0漢,\n0.AAaA,Aééé0\n.😀.A😀,.aéa';
    const padding = 'word '
      .repeat(2_000)
      .slice(0, 8_192 - boundaryPrefix.length);
    const content = `${padding}${boundaryPrefix}${boundarySuffix}`;
    const exactCount = tokenizer.count(content) + 3;

    const count = getTokenCountForMessage(
      new HumanMessage(content),
      (text) => tokenizer.count(text),
      'claude'
    );

    expect(count).toBeGreaterThanOrEqual(exactCount);
  });

  test('keeps long instruction counts stable across prompt-cache placement', () => {
    const tokenizer = new Tokenizer(claudeEncoding);
    const phrase = 'The quick brown fox jumps over the lazy dog. ';
    const stable = phrase
      .repeat(Math.ceil(118_315 / phrase.length))
      .slice(0, 118_315);
    const dynamic = phrase
      .repeat(Math.ceil(361_473 / phrase.length))
      .slice(0, 361_473);
    const count = (message: HumanMessage | SystemMessage): number =>
      getTokenCountForMessage(
        message,
        (text) => tokenizer.count(text),
        'claude'
      );

    const splitCount =
      count(new SystemMessage(stable)) + count(new HumanMessage(dynamic));
    const joinedCount = count(new SystemMessage(`${stable}\n\n${dynamic}`));

    expect(Math.abs(splitCount - joinedCount)).toBeLessThanOrEqual(100);
  });

  test.each([
    -1,
    1.5,
    Number.MAX_SAFE_INTEGER,
    Number.MAX_SAFE_INTEGER + 1,
    Number.POSITIVE_INFINITY,
    Number.NaN,
  ])('rejects an unsafe tokenizer result of %s', (unsafeCount) => {
    expect(() =>
      getTokenCountForMessage(
        new HumanMessage('unsafe count'),
        () => unsafeCount
      )
    ).toThrow(UnsafeTokenMeasurementError);
  });

  test('bounds tokenizer input while fully counting direct string tool args', () => {
    const callbackLengths: number[] = [];
    const args = 'x'.repeat(300_000);

    const count = getTokenCountForMessage(messageWithToolArgs(args), (text) => {
      callbackLengths.push(text.length);
      return text.length;
    });

    expect(Math.max(...callbackLengths)).toBe(8_192);
    expect(count).toBeGreaterThan(args.length);
  });

  test('counts legacy function_call arguments when parsed tool_calls are absent', () => {
    const message = new AIMessage({
      content: '',
      additional_kwargs: {
        function_call: {
          name: 'legacy_lookup',
          arguments: `{"query":"${'x'.repeat(5_000)}"}`,
        },
      },
    });

    const count = getTokenCountForMessage(message, (text) => text.length);

    expect(message.tool_calls).toHaveLength(0);
    expect(count).toBeGreaterThan(5_000);
  });

  test('counts raw OpenAI tool calls when parsed tool_calls are absent', () => {
    const message = new AIMessage('');
    message.additional_kwargs.tool_calls = [
      {
        id: 'raw-call',
        type: 'function',
        function: {
          name: 'raw_lookup',
          arguments: `{"query":"${'x'.repeat(5_000)}"}`,
        },
      },
    ];

    const count = getTokenCountForMessage(message, (text) => text.length);

    expect(message.tool_calls).toHaveLength(0);
    expect(count).toBeGreaterThan(5_000);
  });

  test('does not double-count raw OpenAI calls mirrored in parsed tool_calls', () => {
    const parsed = new AIMessage({
      content: '',
      tool_calls: [
        { id: 'mirrored-call', name: 'lookup', args: { query: 'value' } },
      ],
    });
    const mirrored = new AIMessage({
      content: '',
      tool_calls: [
        { id: 'mirrored-call', name: 'lookup', args: { query: 'value' } },
      ],
      additional_kwargs: {
        tool_calls: [
          {
            id: 'mirrored-call',
            type: 'function',
            function: {
              name: 'lookup',
              arguments: '{"query":"value"}',
            },
          },
        ],
      },
    });

    expect(getTokenCountForMessage(mirrored, (text) => text.length)).toBe(
      getTokenCountForMessage(parsed, (text) => text.length)
    );
  });

  test('ignores stale raw OpenAI calls when parsed calls drive serialization', () => {
    const parsed = new AIMessage({
      content: '',
      tool_calls: [{ id: 'parsed-call', name: 'lookup', args: {} }],
    });
    const stale = new AIMessage({
      content: '',
      tool_calls: [{ id: 'parsed-call', name: 'lookup', args: {} }],
      additional_kwargs: {
        tool_calls: [
          {
            id: 'stale-call',
            type: 'function',
            function: {
              name: 'lookup',
              arguments: `{"query":"${'x'.repeat(5_000)}"}`,
            },
          },
        ],
      },
    });

    expect(getTokenCountForMessage(stale, (text) => text.length)).toBe(
      getTokenCountForMessage(parsed, (text) => text.length)
    );
  });

  test('counts duplicate ids within the raw OpenAI representation', () => {
    const single = new AIMessage('');
    const duplicate = new AIMessage('');
    const rawCall = {
      id: 'reused-call',
      type: 'function' as const,
      function: {
        name: 'lookup',
        arguments: `{"query":"${'x'.repeat(5_000)}"}`,
      },
    };
    single.additional_kwargs.tool_calls = [rawCall];
    duplicate.additional_kwargs.tool_calls = [rawCall, rawCall];

    const singleCount = getTokenCountForMessage(single, (text) => text.length);
    const duplicateCount = getTokenCountForMessage(
      duplicate,
      (text) => text.length
    );

    expect(duplicateCount).toBeGreaterThan(singleCount + 5_000);
  });

  test('counts non-enumerable raw OpenAI function arguments', () => {
    const message = new AIMessage('');
    const rawFunction = { name: 'lookup', arguments: '' };
    Object.defineProperty(rawFunction, 'arguments', {
      value: `{"query":"${'x'.repeat(5_000)}"}`,
    });
    message.additional_kwargs.tool_calls = [
      {
        id: 'non-enumerable-call',
        type: 'function',
        function: rawFunction,
      },
    ];

    expect(
      getTokenCountForMessage(message, (text) => text.length)
    ).toBeGreaterThan(5_000);
  });

  test('counts raw OpenAI custom tool inputs', () => {
    const message = new AIMessage('');
    Reflect.set(message.additional_kwargs, 'tool_calls', [
      {
        id: 'custom-call',
        type: 'custom',
        custom: {
          name: 'shell',
          input: 'x'.repeat(5_000),
        },
      },
    ]);

    expect(
      getTokenCountForMessage(message, (text) => text.length)
    ).toBeGreaterThan(5_000);
  });

  test('rejects raw tool-call arrays beyond the traversal limit', () => {
    const message = new AIMessage('');
    message.additional_kwargs.tool_calls = new Array(10_001);

    expect(() =>
      getTokenCountForMessage(message, (text) => text.length)
    ).toThrow(UnsafeTokenMeasurementError);
  });

  test('counts inherited raw OpenAI tool calls', () => {
    const message = new AIMessage('');
    Object.setPrototypeOf(message.additional_kwargs, {
      tool_calls: [
        {
          id: 'inherited-call',
          type: 'function',
          function: {
            name: 'lookup',
            arguments: `{"query":"${'x'.repeat(5_000)}"}`,
          },
        },
      ],
    });

    expect(
      getTokenCountForMessage(message, (text) => text.length)
    ).toBeGreaterThan(5_000);
  });

  test('rejects a revoked raw tool-call proxy with the typed error', () => {
    const message = new AIMessage('');
    const { proxy, revoke } = Proxy.revocable([], {});
    message.additional_kwargs.tool_calls = proxy;
    revoke();

    expect(() =>
      getTokenCountForMessage(message, (text) => text.length)
    ).toThrow(UnsafeTokenMeasurementError);
  });

  test('rejects raw tool-call array accessors without invoking them', () => {
    const message = new AIMessage('');
    const rawCalls: Array<{
      id: string;
      type: 'function';
      function: { name: string; arguments: string };
    }> = [];
    let getterCalls = 0;
    Object.defineProperty(rawCalls, '0', {
      configurable: true,
      get: () => {
        getterCalls += 1;
        throw new Error('must not run');
      },
    });
    rawCalls.length = 1;
    message.additional_kwargs.tool_calls = rawCalls;

    expect(() =>
      getTokenCountForMessage(message, (text) => text.length)
    ).toThrow(UnsafeTokenMeasurementError);
    expect(getterCalls).toBe(0);
  });

  test('rejects nested raw tool-call accessors without invoking them', () => {
    const message = new AIMessage('');
    const rawCall = {
      id: 'raw-call',
      type: 'function' as const,
      function: { name: 'lookup', arguments: '{}' },
    };
    let getterCalls = 0;
    Object.defineProperty(rawCall, 'function', {
      get: () => {
        getterCalls += 1;
        throw new Error('must not run');
      },
    });
    message.additional_kwargs.tool_calls = [rawCall];

    expect(() =>
      getTokenCountForMessage(message, (text) => text.length)
    ).toThrow(UnsafeTokenMeasurementError);
    expect(getterCalls).toBe(0);
  });

  test('keeps nested dense strings on the bounded structured path', () => {
    const callbackLengths: number[] = [];
    const payload = 'x'.repeat(4 * 1024 * 1024);

    const count = getTokenCountForMessage(
      messageWithToolArgs({ payload }),
      (text) => {
        callbackLengths.push(text.length);
        return text.length;
      }
    );

    expect(Math.max(...callbackLengths)).toBeLessThanOrEqual(200_000);
    // A traversal-work sentinel is unknown size, not real context usage.
    // The provider-input projection will compact the value before invoke.
    expect(count).toBeGreaterThan(0);
    expect(count).toBeLessThan(Number.MAX_SAFE_INTEGER);
  });

  test('counts an opaque proxy argument as its bounded provider-safe placeholder', () => {
    let prototypeCalls = 0;
    const args = new Proxy(
      { safe: true },
      {
        getPrototypeOf() {
          prototypeCalls++;
          throw new Error('prototype denied');
        },
      }
    );

    expect(hasUnsafeStructuredSerialization(args)).toBe(true);
    expect(
      getTokenCountForMessage(messageWithToolArgs(args), (text) => text.length)
    ).toBeLessThan(Number.MAX_SAFE_INTEGER);
    expect(prototypeCalls).toBeLessThanOrEqual(2);
  });

  test('does not recurse through self-referential prototype proxies', () => {
    let prototypeCalls = 0;
    const args: Record<string, unknown> = new Proxy<Record<string, unknown>>(
      {},
      {
        getPrototypeOf(): object | null {
          prototypeCalls++;
          return args;
        },
      }
    );

    expect(hasUnsafeStructuredSerialization(args)).toBe(true);
    expect(
      getTokenCountForMessage(messageWithToolArgs(args), (text) => text.length)
    ).toBeLessThan(Number.MAX_SAFE_INTEGER);
    expect(prototypeCalls).toBeLessThanOrEqual(2);
  });

  test('uses bounded placeholders when descriptor and own-key proxy traps throw', () => {
    const descriptorProxy = new Proxy(
      {},
      {
        getOwnPropertyDescriptor() {
          throw new Error('descriptor denied');
        },
      }
    );
    const ownKeysProxy = new Proxy(
      {},
      {
        ownKeys() {
          throw new Error('keys denied');
        },
      }
    );

    for (const args of [descriptorProxy, ownKeysProxy]) {
      expect(hasUnsafeStructuredSerialization(args)).toBe(true);
      expect(
        getTokenCountForMessage(
          messageWithToolArgs(args),
          (text) => text.length
        )
      ).toBeLessThan(Number.MAX_SAFE_INTEGER);
    }
  });

  test('does not invoke get traps or revoked proxies while measuring arguments', () => {
    let getCalls = 0;
    const getProxy = new Proxy(
      {},
      {
        get() {
          getCalls++;
          return { payload: 'x'.repeat(1_000_000) };
        },
      }
    );
    const revoked = Proxy.revocable({}, {});
    revoked.revoke();
    const inheritedProxy = Object.create(getProxy);

    for (const args of [
      getProxy,
      { nested: getProxy },
      inheritedProxy,
      revoked.proxy,
    ]) {
      expect(hasUnsafeStructuredSerialization(args)).toBe(true);
      expect(
        getTokenCountForMessage(
          messageWithToolArgs(args),
          (text) => text.length
        )
      ).toBeLessThan(Number.MAX_SAFE_INTEGER);
    }
    expect(getCalls).toBe(0);
  });

  test('reports a typed terminal error for a proxied content block', () => {
    let getCalls = 0;
    const contentBlock = new Proxy(
      { type: 'text', text: 'safe' },
      {
        get() {
          getCalls++;
          throw new Error('content read denied');
        },
      }
    );
    const message = {
      content: [contentBlock],
      getType: () => 'tool',
    } as unknown as ToolMessage;

    expect(() =>
      getTokenCountForMessage(message, (text) => text.length)
    ).toThrow(UnsafeTokenMeasurementError);
    try {
      getTokenCountForMessage(message, (text) => text.length);
    } catch (error) {
      expect(JSON.parse((error as Error).message)).toEqual({
        type: 'unsafe_token_measurement',
        reason: 'content_proxy',
        path: 'content[0]',
      });
    }
    expect(getCalls).toBe(0);
  });

  test('reports typed errors for unsafe message and metadata branches', () => {
    let getterCalls = 0;
    const messageProxy = new Proxy(new HumanMessage('secret payload'), {
      get() {
        getterCalls++;
        throw new Error('message read denied');
      },
    });
    const metadataProxy = new Proxy(
      {},
      {
        get() {
          getterCalls++;
          throw new Error('metadata read denied');
        },
      }
    );
    const accessorMetadata = {};
    Object.defineProperty(accessorMetadata, 'type', {
      get() {
        getterCalls++;
        return 'computer_call_output';
      },
    });
    const cases = [
      {
        message: messageProxy,
        reason: 'message_proxy',
        path: 'message',
      },
      {
        message: {
          content: '',
          additional_kwargs: metadataProxy,
          getType: () => 'tool',
        } as unknown as ToolMessage,
        reason: 'metadata_proxy',
        path: 'additional_kwargs',
      },
      {
        message: {
          content: '',
          additional_kwargs: accessorMetadata,
          getType: () => 'tool',
        } as unknown as ToolMessage,
        reason: 'metadata_accessor',
        path: 'additional_kwargs.type',
      },
    ];

    for (const testCase of cases) {
      try {
        getTokenCountForMessage(testCase.message, (text) => text.length);
        throw new Error('Expected token measurement to fail');
      } catch (error) {
        expect(error).toBeInstanceOf(UnsafeTokenMeasurementError);
        expect(JSON.parse((error as Error).message)).toEqual({
          type: 'unsafe_token_measurement',
          reason: testCase.reason,
          path: testCase.path,
        });
        expect((error as Error).message).not.toContain('secret payload');
      }
    }
    expect(getterCalls).toBe(0);
  });

  test('detects inherited accessors without invoking them', () => {
    let accessorCalls = 0;
    const prototype = {};
    Object.defineProperty(prototype, 'danger', {
      enumerable: true,
      get() {
        accessorCalls++;
        return 'x'.repeat(1_000_000);
      },
    });
    const args = Object.assign(Object.create(prototype), { safe: true });

    expect(hasUnsafeStructuredSerialization(args)).toBe(true);
    expect(
      getTokenCountForMessage(messageWithToolArgs(args), (text) => text.length)
    ).toBeLessThan(Number.MAX_SAFE_INTEGER);
    expect(accessorCalls).toBe(0);
  });

  test('counts safely serializable proxy tool-call history without invoking getters', () => {
    let getCalls = 0;
    const args = new Proxy(
      { query: 'select * from reports' },
      {
        get(target, property, receiver) {
          getCalls++;
          return Reflect.get(target, property, receiver);
        },
      }
    );

    const count = getTokenCountForMessage(
      messageWithToolArgs(args),
      (text) => text.length
    );

    expect(count).toBeGreaterThan(0);
    expect(count).toBeLessThan(Number.MAX_SAFE_INTEGER);
    expect(getCalls).toBe(0);
  });

  test('counts tool_calls-only names and arguments', () => {
    const message = new AIMessage({
      content: '',
      tool_calls: [
        {
          id: 'tool-call-only',
          name: 'evaluate_script',
          args: { code: 'x'.repeat(5_000) },
        },
      ],
    });

    const count = getTokenCountForMessage(message, (text) => text.length);

    expect(count).toBeGreaterThan(5_000);
  });

  test('counts adversarial structured tool-call args without invoking toJSON', () => {
    let toJSONCalls = 0;
    const message = new AIMessage({
      content: '',
      tool_calls: [
        {
          id: 'adversarial-tool-call',
          name: 'evaluate_script',
          args: {
            query: 'safe',
            toJSON() {
              toJSONCalls++;
              return { code: 'y'.repeat(2_000_000) };
            },
          },
        },
      ],
    });

    const count = getTokenCountForMessage(message, (text) => text.length);

    expect(count).toBeLessThan(Number.MAX_SAFE_INTEGER);
    expect(count).toBeGreaterThan(0);
    expect(toJSONCalls).toBe(0);
  });

  test('retains conservative pressure when structured args exceed the preview', () => {
    const message = new AIMessage({
      content: '',
      tool_calls: [
        {
          id: 'large-structured-tool-call',
          name: 'evaluate_script',
          args: { code: 'x'.repeat(300_000) },
        },
      ],
    });

    const count = getTokenCountForMessage(message, (text) => text.length);

    expect(count).toBeGreaterThan(300_000);
  });

  test('counts opaque structured tool-result blocks as serialized content', () => {
    const message = new ToolMessage({
      content: [
        {
          type: 'json',
          rows: Array.from({ length: 20 }, (_, index) => ({
            id: index,
            value: `row-${index}-${'x'.repeat(100)}`,
          })),
        },
      ],
      tool_call_id: 'select-query',
    });

    const count = getTokenCountForMessage(message, (text) => text.length);

    expect(count).toBeGreaterThan(2_000);
  });

  test('counts spoofed media types as serialized structured output', () => {
    const message = new ToolMessage({
      content: [
        {
          type: 'image_url',
          rows: [{ value: 'x'.repeat(5_000) }],
        },
      ],
      tool_call_id: 'spoofed-image',
    });

    const count = getTokenCountForMessage(message, (text) => text.length);

    expect(count).toBeGreaterThan(5_000);
  });

  test('prices Google image media blocks with the image estimator', () => {
    const message = new ToolMessage({
      content: [
        {
          type: 'media',
          mimeType: 'image/png',
          data: pngDataUri(1024, 768),
        },
      ],
      tool_call_id: 'google-image-media',
    });

    const count = getTokenCountForMessage(
      message,
      (text) => text.length,
      'o200k_base'
    );

    // 3 message tokens + ceil((85 + 4 * 170) * 1.05)
    expect(count).toBe(807);
  });

  test('prices computer screenshots as images instead of base64 text', () => {
    const screenshot = pngDataUri(1024, 768);
    const structured = new ToolMessage({
      content: [{ type: 'computer_screenshot', image_url: screenshot }],
      tool_call_id: 'computer-structured',
      additional_kwargs: { type: 'computer_call_output' },
    });
    const string = new ToolMessage({
      content: screenshot,
      tool_call_id: 'computer-string',
      additional_kwargs: { type: 'computer_call_output' },
    });

    expect(
      getTokenCountForMessage(structured, (text) => text.length, 'o200k_base')
    ).toBe(807);
    expect(
      getTokenCountForMessage(string, (text) => text.length, 'o200k_base')
    ).toBe(807);
    const file = new ToolMessage({
      content: [
        { type: 'input_image', file_id: 'file-screenshot123', detail: 'low' },
      ],
      tool_call_id: 'computer-file',
      additional_kwargs: { type: 'computer_call_output' },
    });
    expect(
      getTokenCountForMessage(file, (text) => text.length, 'o200k_base')
    ).toBe(93);
    const highDetailFile = new ToolMessage({
      content: [
        { type: 'input_image', file_id: 'file-screenshot456', detail: 'high' },
      ],
      tool_call_id: 'computer-file-high',
      additional_kwargs: { type: 'computer_call_output' },
    });
    expect(
      getTokenCountForMessage(
        highDetailFile,
        (text) => text.length,
        'o200k_base'
      )
    ).toBe(1079);
  });

  test('does not image-price malformed marked computer output', () => {
    const content = 'not a screenshot URL';
    const message = new ToolMessage({
      content,
      tool_call_id: 'computer-invalid',
      additional_kwargs: { type: 'computer_call_output' },
    });

    expect(
      getTokenCountForMessage(message, (text) => text.length, 'o200k_base')
    ).toBe(3 + content.length);
  });

  test('counts legacy calls on generic assistant messages', () => {
    const argumentsValue = `{"query":"${'x'.repeat(5_000)}"}`;
    const message = new ChatMessage({
      role: 'assistant',
      content: '',
      additional_kwargs: {
        function_call: {
          name: 'legacy_lookup',
          arguments: argumentsValue,
        },
      },
    });

    expect(
      getTokenCountForMessage(message, (text) => text.length, 'o200k_base')
    ).toBeGreaterThan(argumentsValue.length);
  });

  test('prices Google PDF media blocks per estimated page', () => {
    const message = new ToolMessage({
      content: [
        {
          type: 'media',
          mime_type: 'application/pdf',
          data: 'A'.repeat(150_000),
        },
      ],
      tool_call_id: 'google-pdf-media',
    });

    const count = getTokenCountForMessage(
      message,
      (text) => text.length,
      'o200k_base'
    );

    // 3 message tokens + ceil((2 pages * 1500) * 1.05)
    expect(count).toBe(3153);
  });

  test('continues pricing Google audio and video media as timed content', () => {
    const audio = new ToolMessage({
      content: [
        {
          type: 'media',
          mime_type: '  Audio/MP3 ; codecs=mp3 ',
          data: 'A'.repeat(320_000),
        },
      ],
      tool_call_id: 'google-audio-media',
    });
    const video = new ToolMessage({
      content: [
        {
          type: 'media',
          mimeType: 'video/mp4',
          data: 'A'.repeat(1_000_000),
        },
      ],
      tool_call_id: 'google-video-media',
    });

    expect(
      getTokenCountForMessage(audio, (text) => text.length, 'o200k_base')
    ).toBe(507);
    expect(
      getTokenCountForMessage(video, (text) => text.length, 'o200k_base')
    ).toBe(948);
  });

  test.each([
    ['mimeType', 'text/plain'],
    ['mime_type', 'application/json'],
  ])(
    'conservatively prices unknown Google media MIME via %s',
    (mimeKey, mimeType) => {
      let toJSONCalls = 0;
      const data = 'A'.repeat(100_000);
      const message = new ToolMessage({
        content: [
          {
            type: 'media',
            [mimeKey]: mimeType,
            data,
            toJSON() {
              toJSONCalls++;
              return { expanded: 'x'.repeat(1_000_000) };
            },
          },
        ],
        tool_call_id: 'google-unknown-media',
      });

      const count = getTokenCountForMessage(
        message,
        (text) => text.length,
        'o200k_base'
      );

      expect(count).toBeGreaterThan(data.length);
      expect(toJSONCalls).toBe(0);
    }
  );
});

describe('estimateImageBlockTokens', () => {
  test('claude: tokens = ceil(w*h/750), floored at 1024', () => {
    const block = {
      type: 'image_url',
      image_url: { url: pngDataUri(1024, 768) },
    };
    // 1024*768/750 = 1048.58 -> ceil 1049 (> 1024 floor)
    expect(estimateImageBlockTokens(block, 'claude')).toBe(1049);
  });

  test('openai: tokens = 85 + tiles*170 (512px tiles)', () => {
    const block = {
      type: 'image_url',
      image_url: { url: pngDataUri(1024, 768) },
    };
    // ceil(1024/512)*ceil(768/512) = 2*2 = 4 tiles -> 85 + 680 = 765
    expect(estimateImageBlockTokens(block, 'o200k_base')).toBe(765);
  });

  test('falls back to the Anthropic minimum (1024) without base64 data', () => {
    expect(
      estimateImageBlockTokens(
        { type: 'image_url', image_url: { url: 'https://example.com/a.png' } },
        'claude'
      )
    ).toBe(1024);
  });
});

describe('estimateDocumentBlockTokens', () => {
  const countChars = (text: string): number => text.length;

  test('text document is tokenized directly via getTokenCount', () => {
    const block = { type: 'file', source_type: 'text', text: 'hello world' };
    expect(estimateDocumentBlockTokens(block, 'o200k_base', countChars)).toBe(
      11
    );
  });

  test('base64 PDF is priced per estimated page', () => {
    const block = {
      type: 'file',
      source_type: 'base64',
      mime_type: 'application/pdf',
      data: 'x'.repeat(150_000),
    };
    // ceil(150000/75000) = 2 pages
    expect(estimateDocumentBlockTokens(block, 'claude', countChars)).toBe(4000); // 2 * 2000
    expect(estimateDocumentBlockTokens(block, 'o200k_base', countChars)).toBe(
      3000
    ); // 2 * 1500
  });

  test('base64 text document is counted by its decoded text', () => {
    const csv = 'id,name\n1,Zoë ☕\n'.repeat(50);
    const block = {
      type: 'file',
      source_type: 'base64',
      mime_type: 'text/csv',
      data: Buffer.from(csv).toString('base64'),
    };
    expect(estimateDocumentBlockTokens(block, 'o200k_base', countChars)).toBe(
      csv.length
    );
  });

  test('base64 binary document is counted by its decoded size', () => {
    const bytes = 40_000;
    const block = {
      type: 'file',
      source_type: 'base64',
      mime_type:
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      data: Buffer.alloc(bytes, 7).toString('base64'),
    };
    const count = jest.fn(countChars);
    expect(estimateDocumentBlockTokens(block, 'o200k_base', count)).toBe(
      bytes / 4
    );
    expect(count).not.toHaveBeenCalled();
  });

  test('url-referenced document uses the conservative fallback', () => {
    const block = { type: 'file', source_type: 'url', url: 'https://x/y.pdf' };
    expect(estimateDocumentBlockTokens(block, 'o200k_base', countChars)).toBe(
      2000
    );
  });

  test.each([
    ['audio/mpeg', 320_000],
    ['video/mp4', 1_000_000],
    ['Audio/WAV; rate=16000', 320_000],
  ])(
    '@scenario:timed-media-file-priced-by-duration prices %s in a file block by duration',
    (mimeType, chars) => {
      const data = 'A'.repeat(chars);
      const block = {
        type: 'file',
        source_type: 'base64',
        mime_type: mimeType,
        data,
      };
      expect(estimateDocumentBlockTokens(block, 'o200k_base', countChars)).toBe(
        estimateTimedMediaBlockTokens({ type: 'media', mimeType, data })
      );
    }
  );

  test('@scenario:timed-media-file-priced-by-duration counts a message carrying audio in a file block as it counts the same audio in a media part', () => {
    const data = Buffer.alloc(2 * 1024 * 1024, 7).toString('base64');
    const count = (part: ContentBlock): number =>
      getTokenCountForMessage(
        new HumanMessage({
          content: [{ type: 'text', text: 'Summarize the recording' }, part],
        }),
        countChars
      );

    const asFile = count({
      type: 'file',
      source_type: 'base64',
      mime_type: 'audio/mpeg',
      data,
    });
    expect(asFile).toBe(count({ type: 'media', mimeType: 'audio/mpeg', data }));
    expect(asFile).toBeLessThan(10_000);
  });

  test('remote audio in a file block uses the timed media fallback', () => {
    const block = {
      type: 'file',
      source_type: 'url',
      mime_type: 'audio/mpeg',
      url: 'https://x/memo.mp3',
    };
    expect(estimateDocumentBlockTokens(block, 'o200k_base', countChars)).toBe(
      960
    );
  });
});

describe('estimateTimedMediaBlockTokens', () => {
  const B64 = (chars: number): string => 'A'.repeat(chars);

  test('Google video (type=media, video/*): duration from size at ~300 tok/s', () => {
    // 1,000,000 b64 -> 750,000 bytes / 250,000 Bps = 3s * 300 = 900
    expect(
      estimateTimedMediaBlockTokens({
        type: 'media',
        mimeType: 'video/mp4',
        data: B64(1_000_000),
      })
    ).toBe(900);
  });

  test('Google audio (type=media, audio/*): 32 tok/s at ~16KB/s', () => {
    // 320,000 b64 -> 240,000 bytes / 16,000 Bps = 15s * 32 = 480
    expect(
      estimateTimedMediaBlockTokens({
        type: 'media',
        mimeType: 'audio/mp3',
        data: B64(320_000),
      })
    ).toBe(480);
  });

  test('OpenRouter input_audio: estimates from base64 data', () => {
    expect(
      estimateTimedMediaBlockTokens({
        type: 'input_audio',
        input_audio: { data: B64(320_000) },
      })
    ).toBe(480);
  });

  test('OpenRouter video_url with a data: URL estimates from size', () => {
    const url = `data:video/mp4;base64,${B64(1_000_000)}`;
    expect(
      estimateTimedMediaBlockTokens({ type: 'video_url', video_url: { url } })
    ).toBe(900);
  });

  test('bare remote URL falls back to the flat ~30s estimate', () => {
    expect(
      estimateTimedMediaBlockTokens({
        type: 'video_url',
        video_url: { url: 'https://x/v.mp4' },
      })
    ).toBe(9000);
    expect(
      estimateTimedMediaBlockTokens({ type: 'input_audio', input_audio: {} })
    ).toBe(960);
  });

  test('clamps to at least one second of tokens for tiny payloads', () => {
    expect(
      estimateTimedMediaBlockTokens({
        type: 'media',
        mimeType: 'audio/wav',
        data: 'AAAA',
      })
    ).toBe(32);
  });

  test('standard type=video / type=audio blocks (Bedrock converter shape)', () => {
    // 750,000 bytes video / 250,000 = 3s * 300 = 900
    expect(
      estimateTimedMediaBlockTokens({
        type: 'video',
        mimeType: 'video/mp4',
        data: B64(1_000_000),
      })
    ).toBe(900);
    // 240,000 bytes audio / 16,000 = 15s * 32 = 480
    expect(
      estimateTimedMediaBlockTokens({
        type: 'audio',
        mimeType: 'audio/mpeg',
        data: B64(320_000),
      })
    ).toBe(480);
  });

  test('reads Uint8Array data and base64 url payloads', () => {
    // 240,000-byte Uint8Array audio -> 15s * 32 = 480
    expect(
      estimateTimedMediaBlockTokens({
        type: 'audio',
        mimeType: 'audio/wav',
        data: new Uint8Array(240_000),
      })
    ).toBe(480);
    // base64 data url on a bare video block
    expect(
      estimateTimedMediaBlockTokens({
        type: 'video',
        mimeType: 'video/mp4',
        url: `data:video/mp4;base64,${B64(1_000_000)}`,
      })
    ).toBe(900);
  });

  test('non-video/audio media (image/document MIME) is NOT priced as video', () => {
    expect(
      estimateTimedMediaBlockTokens({
        type: 'media',
        mimeType: 'image/png',
        fileUri: 's3://x',
      })
    ).toBe(0);
    expect(
      estimateTimedMediaBlockTokens({
        type: 'media',
        mimeType: 'image/png',
        data: B64(400_000),
      })
    ).toBe(0);
    expect(
      estimateTimedMediaBlockTokens({
        type: 'media',
        mimeType: 'application/pdf',
        data: B64(400),
      })
    ).toBe(0);
  });

  test('fileId / bare URL with no size falls back to the ~30s estimate', () => {
    expect(
      estimateTimedMediaBlockTokens({
        type: 'video',
        mimeType: 'video/mp4',
        fileId: 's3://v',
      })
    ).toBe(9000);
    expect(
      estimateTimedMediaBlockTokens({
        type: 'audio',
        mimeType: 'audio/mp3',
        fileId: 's3://a',
      })
    ).toBe(960);
  });

  test('reads native Bedrock nested source.bytes (video/audio)', () => {
    expect(
      estimateTimedMediaBlockTokens({
        type: 'video',
        video: { source: { bytes: new Uint8Array(750_000) } },
      })
    ).toBe(900); // 750,000 / 250,000 = 3s * 300
    expect(
      estimateTimedMediaBlockTokens({
        type: 'audio',
        audio: { source: { bytes: new Uint8Array(240_000) } },
      })
    ).toBe(480); // 240,000 / 16,000 = 15s * 32
  });

  test('reads nested video/audio data-URL and data payloads, remote stays fallback', () => {
    // nested data URL sized by its bytes (750,000 -> 3s * 300)
    expect(
      estimateTimedMediaBlockTokens({
        type: 'video',
        video: { url: `data:video/mp4;base64,${B64(1_000_000)}` },
      })
    ).toBe(900);
    // nested base64 data (240,000 -> 15s * 32)
    expect(
      estimateTimedMediaBlockTokens({
        type: 'audio',
        audio: { data: B64(320_000) },
      })
    ).toBe(480);
    // nested REMOTE url has no size -> ~30s fallback
    expect(
      estimateTimedMediaBlockTokens({
        type: 'video',
        video: { url: 'https://x/v.mp4' },
      })
    ).toBe(9000);
  });

  test('non-data URI schemes (gs://, s3://) are treated as remote, not base64', () => {
    // gs:// audio must hit the ~30s remote fallback, not clamp to 32
    expect(
      estimateTimedMediaBlockTokens({
        type: 'audio',
        mimeType: 'audio/mp3',
        url: 'gs://bucket/a.mp3',
      })
    ).toBe(960);
    expect(
      estimateTimedMediaBlockTokens({
        type: 'media',
        mimeType: 'video/mp4',
        data: 's3://bucket/v.mp4',
      })
    ).toBe(9000);
  });

  test('classifies Google MIME-as-type blocks (type is the mime string)', () => {
    // { type: 'audio/wav', data } -> 240,000 bytes / 16,000 = 15s * 32 = 480
    expect(
      estimateTimedMediaBlockTokens({ type: 'audio/wav', data: B64(320_000) })
    ).toBe(480);
    // { type: 'video/mp4', data } -> 750,000 / 250,000 = 3s * 300 = 900
    expect(
      estimateTimedMediaBlockTokens({ type: 'video/mp4', data: B64(1_000_000) })
    ).toBe(900);
  });

  test('returns 0 for non-timed-media blocks', () => {
    expect(estimateTimedMediaBlockTokens({ type: 'text' })).toBe(0);
    expect(
      estimateTimedMediaBlockTokens({ type: 'image/png', data: B64(400) })
    ).toBe(0);
  });
});
