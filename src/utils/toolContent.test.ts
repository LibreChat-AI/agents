import { spawnSync } from 'child_process';
import type { BaseMessage } from '@langchain/core/messages';
import {
  compactToolContent,
  getBoundedCacheControlledTextToolContent,
  getBoundedSingleTextToolContent,
  getComputerCallOutputScreenshot,
  getToolContentCharLength,
  isAtomicToolContentBlock,
  serializeStructuredValue,
  serializeStructuredValueBounded,
  serializeToolContent,
  serializeToolContentBounded,
} from './toolContent';

type ToolContent = BaseMessage['content'];

describe('toolContent', () => {
  it('counts bounded inline images without exhausting structured character work', () => {
    const content: ToolContent = [
      {
        type: 'image_url',
        image_url: {
          url: `data:image/png;base64,${Buffer.alloc(2_477_915).toString('base64')}`,
        },
      },
    ];
    expect(getToolContentCharLength(content)).toBe(
      JSON.stringify(content).length
    );
    expect(compactToolContent(content, 4_000_000).changed).toBe(false);
    expect(compactToolContent(content, 400_000).changed).toBe(true);
  });

  it('retains character traversal protection for escaped and oversized strings', () => {
    for (const url of [
      `data:image/png;"base64,${'A'.repeat(1_100_000)}`,
      `data:image/png;base64,${'A'.repeat(5_000_004)}`,
      'A'.repeat(1_100_000),
    ]) {
      const content: ToolContent = [{ type: 'image_url', image_url: { url } }];
      expect(compactToolContent(content, 8_000_000).changed).toBe(true);
    }
  });

  it('normalizes a small opaque array to provider-neutral text', () => {
    const content = [{ type: 'json', rows: [{ id: 1 }] }] as ToolContent;

    const result = compactToolContent(content, 10_000);

    expect(result.changed).toBe(true);
    expect(result.content).toBe(JSON.stringify(content));
  });

  it('does not trust a media-looking type without a valid media payload', () => {
    const invalidBlocks = [
      {
        type: 'image_url',
        rows: [{ id: 1, value: 'not an image' }],
      },
      { type: 'image', text: 'ordinary row' },
      { type: 'file', content: [{ id: 1 }] },
      { type: 'media', uri: 'not-a-supported-media-payload' },
      { type: 'image_url', image_url: [{ id: 1 }] },
      {
        type: 'media',
        mimeType: 'application/octet-stream',
        data: [{ id: 1 }],
      },
    ];

    for (const block of invalidBlocks) {
      const content = [block] as ToolContent;
      const result = compactToolContent(content, 10_000);

      expect(result.changed).toBe(true);
      expect(result.content).toBe(JSON.stringify(content));
    }
  });

  it('recognizes native byte and resource payload shapes as atomic', () => {
    const bytes = new Uint8Array([1, 2, 3]);

    expect(
      isAtomicToolContentBlock({
        type: 'video',
        video: { source: { bytes } },
      })
    ).toBe(true);
    expect(
      isAtomicToolContentBlock({
        type: 'document',
        document: {
          source: { content: [{ type: 'text', text: 'page' }] },
        },
      })
    ).toBe(true);
    expect(
      isAtomicToolContentBlock({
        type: 'resource',
        resource: { uri: 'file:///result.bin', blob: bytes },
      })
    ).toBe(true);
    expect(
      isAtomicToolContentBlock({
        type: 'media',
        mimeType: 'application/octet-stream',
        data: new ArrayBuffer(8),
      })
    ).toBe(true);
    expect(
      isAtomicToolContentBlock({
        type: 'file',
        source_type: 'id',
        id: 'file_123',
      })
    ).toBe(true);
    expect(
      isAtomicToolContentBlock({
        type: 'computer_screenshot',
        image_url: 'data:image/png;base64,AAAA',
      })
    ).toBe(true);
    expect(
      isAtomicToolContentBlock({
        type: 'application/pdf',
        data: 'JVBERi0xLjQ=',
      })
    ).toBe(true);
  });

  it('normalizes a direct object result to provider-neutral text', () => {
    const content = { rows: [{ id: 1 }], count: 1 };

    const result = compactToolContent(content, 10_000);

    expect(result.changed).toBe(true);
    expect(result.content).toBe(JSON.stringify(content));
  });

  it('keeps valid text and small media blocks intact', () => {
    const image = {
      type: 'image_url',
      image_url: { url: 'https://example.com/chart.png' },
    };
    const content = [
      { type: 'text', text: 'Query result chart' },
      image,
    ] as ToolContent;

    const result = compactToolContent(content, 10_000);

    expect(result.changed).toBe(false);
    expect(result.content).toBe(content);
  });

  it('classifies each mixed content block once during compaction', () => {
    const blocks: Array<Record<string, unknown>> = [];
    const tracked = new Set<object>();
    for (let i = 0; i < 100; i++) {
      const text = { type: 'text', text: `row-${i}-${'x'.repeat(20)}` };
      const image = {
        type: 'image_url',
        image_url: { url: `https://example.com/chart-${i}.png` },
      };
      const opaque = { type: 'json', row: { id: i } };
      blocks.push(text, image, opaque);
      tracked.add(text);
      tracked.add(image);
      tracked.add(opaque);
    }
    const getOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
    let topLevelSafetyWalks = 0;
    const descriptorSpy = jest
      .spyOn(Object, 'getOwnPropertyDescriptor')
      .mockImplementation((target, property) => {
        if (property === 'toJSON' && tracked.has(target as object)) {
          topLevelSafetyWalks++;
        }
        return getOwnPropertyDescriptor(target, property);
      });

    const compacted = (() => {
      try {
        return compactToolContent(blocks as ToolContent, 1_000);
      } finally {
        descriptorSpy.mockRestore();
      }
    })();

    expect(topLevelSafetyWalks).toBe(blocks.length);
    expect(compacted.changed).toBe(true);
    expect(serializeToolContent(compacted.content).length).toBeLessThanOrEqual(
      1_000
    );
  });

  it('does not reclassify oversized text blocks while serializing', () => {
    const blocks: Array<Record<string, unknown>> = [];
    const tracked = new Set<object>();
    for (let i = 0; i < 300; i++) {
      const block = {
        type: 'text',
        text: `row-${i}-${'x'.repeat(20)}`,
      };
      blocks.push(block);
      tracked.add(block);
    }
    const getOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
    let topLevelSafetyWalks = 0;
    const descriptorSpy = jest
      .spyOn(Object, 'getOwnPropertyDescriptor')
      .mockImplementation((target, property) => {
        if (property === 'toJSON' && tracked.has(target as object)) {
          topLevelSafetyWalks++;
        }
        return getOwnPropertyDescriptor(target, property);
      });

    const compacted = (() => {
      try {
        return compactToolContent(blocks as ToolContent, 1_000);
      } finally {
        descriptorSpy.mockRestore();
      }
    })();

    expect(topLevelSafetyWalks).toBe(blocks.length);
    expect(compacted.changed).toBe(true);
    expect(typeof compacted.content).toBe('string');
    expect((compacted.content as string).length).toBeLessThanOrEqual(1_000);
  });

  it('includes forwarded text-block metadata in the compaction bound', () => {
    const citations = 'A'.repeat(1_000_000);
    const content = [
      {
        type: 'text',
        text: 'x',
        citations,
      },
    ] as ToolContent;

    const compacted = compactToolContent(content, 2_000);
    const serialized = serializeToolContent(compacted.content);

    expect(compacted.changed).toBe(true);
    expect(compacted.content).not.toBe(content);
    expect(serialized.length).toBeLessThanOrEqual(2_000);
    expect(serialized).not.toContain(citations);
    expect(compacted.originalChars).toBeGreaterThan(1_000_000);
  });

  it('does not trust callable serialization hooks on the content array', () => {
    const toJSON = jest.fn(() => ({
      expanded: 'x'.repeat(100_000),
    }));
    const content = [{ type: 'text', text: 'safe' }] as ToolContent;
    Object.defineProperty(content, 'toJSON', {
      enumerable: true,
      value: toJSON,
    });

    const compacted = compactToolContent(content, 100);

    expect(toJSON).not.toHaveBeenCalled();
    expect(compacted.changed).toBe(true);
    expect(typeof compacted.content).toBe('string');
    expect((compacted.content as string).length).toBeLessThanOrEqual(100);
  });

  it('uses one bounded fallback when nested opaque validation is exhausted', () => {
    const opaque = Array.from({ length: 10_001 }, (_, index) => index);
    const image = {
      type: 'image_url',
      image_url: { url: 'https://example.com/chart.png' },
    };
    const content: unknown[] = [];
    for (let i = 0; i < 25; i++) {
      content.push(image, opaque);
    }
    const getOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
    let descriptorReads = 0;
    const descriptorSpy = jest
      .spyOn(Object, 'getOwnPropertyDescriptor')
      .mockImplementation((target, property) => {
        if (target === opaque && property !== 'toJSON') {
          descriptorReads++;
        }
        return getOwnPropertyDescriptor(target, property);
      });

    const compacted = (() => {
      try {
        return compactToolContent(content as ToolContent, 1_000);
      } finally {
        descriptorSpy.mockRestore();
      }
    })();

    expect(descriptorReads).toBeLessThan(400_000);
    expect(compacted.changed).toBe(true);
    expect(typeof compacted.content).toBe('string');
    expect((compacted.content as string).length).toBeLessThanOrEqual(1_000);
  });

  it('preserves only the atomic occurrence that fits the budget', () => {
    const image = {
      type: 'image_url',
      image_url: { url: 'https://example.com/chart.png' },
    };
    const imageChars = JSON.stringify(image).length;
    const content = [
      image,
      image,
      { type: 'text', text: 'x'.repeat(1_000) },
    ] as ToolContent;

    const compacted = compactToolContent(content, imageChars * 2 + 20);

    expect(Array.isArray(compacted.content)).toBe(true);
    expect(
      (compacted.content as Exclude<ToolContent, string>).filter(
        (block) => block === image
      )
    ).toHaveLength(1);
    expect(serializeToolContent(compacted.content).length).toBeLessThanOrEqual(
      imageChars * 2 + 20
    );
  });

  it('accounts for non-finite numbers using their JSON null length', () => {
    const content = [
      {
        type: 'image_url',
        image_url: { url: 'https://example.com/chart.png' },
        metadata: [NaN, NaN, NaN, NaN, NaN],
      },
    ] as ToolContent;
    const maxChars = JSON.stringify(content).length - 1;

    const compacted = compactToolContent(content, maxChars);

    expect(compacted.changed).toBe(true);
    expect(serializeToolContent(compacted.content).length).toBeLessThanOrEqual(
      maxChars
    );
  });

  it('fails closed without invoking traps on an array proxy prototype', () => {
    let trapCalls = 0;
    const proxyPrototype = new Proxy(Array.prototype, {
      getOwnPropertyDescriptor() {
        trapCalls++;
        return undefined;
      },
      getPrototypeOf(target) {
        trapCalls++;
        return Reflect.getPrototypeOf(target);
      },
    });
    const nested = ['safe'];
    Object.setPrototypeOf(nested, proxyPrototype);

    const compacted = compactToolContent(
      [nested] as unknown as ToolContent,
      100
    );

    expect(trapCalls).toBe(0);
    expect(compacted.changed).toBe(true);
    expect(typeof compacted.content).toBe('string');
    expect((compacted.content as string).length).toBeLessThanOrEqual(100);
  });

  it('does not invoke overridden byteLength accessors on typed views', () => {
    let byteLengthCalls = 0;
    const bytes = new Uint8Array([1]);
    Object.defineProperty(bytes, 'byteLength', {
      get() {
        byteLengthCalls++;
        return 10_000_000;
      },
    });

    const compacted = compactToolContent(
      [{ type: 'video', data: bytes }] as ToolContent,
      100
    );

    expect(byteLengthCalls).toBe(0);
    expect(compacted.changed).toBe(true);
    expect(typeof compacted.content).toBe('string');
    expect((compacted.content as string).length).toBeLessThanOrEqual(100);
  });

  it('uses fixed native view names for exact cap accounting', () => {
    const bytes = new Uint8Array([1]);
    Object.defineProperty(bytes, 'constructor', {
      value: { name: 'X'.repeat(1_000) },
    });
    const content = [{ type: 'video', data: bytes }] as ToolContent;

    const compacted = compactToolContent(content, 60);
    const serialized = serializeToolContent(compacted.content);

    expect(compacted.changed).toBe(true);
    expect(serialized.length).toBeLessThanOrEqual(60);
    expect(serialized).not.toContain('X'.repeat(100));
  });

  it('caps aggregate character traversal across repeated opaque runs', () => {
    const payload = 'x'.repeat(100_000);
    const opaque = { type: 'json', payload };
    const image = {
      type: 'image_url',
      image_url: { url: 'https://example.com/chart.png' },
    };
    const content = Array.from({ length: 20 }, (_, index) =>
      index % 2 === 0 ? opaque : image
    ) as ToolContent;
    const charCodeAt = String.prototype.charCodeAt;
    let characterReads = 0;
    const charSpy = jest
      .spyOn(String.prototype, 'charCodeAt')
      .mockImplementation(function (this: string, index: number) {
        characterReads++;
        return charCodeAt.call(this, index);
      });

    const compacted = (() => {
      try {
        return compactToolContent(content, 200);
      } finally {
        charSpy.mockRestore();
      }
    })();

    expect(characterReads).toBeLessThanOrEqual(1_010_000);
    expect(compacted.changed).toBe(true);
    expect(serializeToolContent(compacted.content).length).toBeLessThanOrEqual(
      200
    );
    expect(compacted.originalChars).toBe(Number.MAX_SAFE_INTEGER);
  });

  it('bounds structured traversal across repeated block references', () => {
    const metadata = Array.from({ length: 1_000 }, (_, index) => index);
    const block = { type: 'text', text: 'safe', metadata };
    const content = new Array(100).fill(block) as ToolContent;
    const getOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
    let descriptorReads = 0;
    const descriptorSpy = jest
      .spyOn(Object, 'getOwnPropertyDescriptor')
      .mockImplementation((target, property) => {
        if (target === metadata && property !== 'toJSON') {
          descriptorReads++;
        }
        return getOwnPropertyDescriptor(target, property);
      });

    try {
      expect(serializeToolContent(content)).toContain('safe');
      expect(getToolContentCharLength(content)).toBeGreaterThan(0);
      expect(
        serializeToolContentBounded(content, 100).length
      ).toBeLessThanOrEqual(100);
    } finally {
      descriptorSpy.mockRestore();
    }

    expect(descriptorReads).toBeLessThan(10_000);
  });

  it('rejects oversized MIME-like types before scanning each distinct block', () => {
    const hugeType = `${'a'.repeat(100_000)}/b`;
    const content = Array.from({ length: 50 }, () => ({
      type: hugeType,
      data: 'x',
    })) as ToolContent;
    const indexOf = String.prototype.indexOf;
    let hugeTypeScans = 0;
    const indexOfSpy = jest
      .spyOn(String.prototype, 'indexOf')
      .mockImplementation(function (
        this: string,
        searchString: string,
        position?: number
      ) {
        if (String(this) === hugeType) {
          hugeTypeScans++;
        }
        return indexOf.call(this, searchString, position);
      });

    const compacted = (() => {
      try {
        return compactToolContent(content, 200);
      } finally {
        indexOfSpy.mockRestore();
      }
    })();
    const serialized = serializeToolContent(compacted.content);

    expect(hugeTypeScans).toBe(0);
    expect(compacted.changed).toBe(true);
    expect(serialized.length).toBeLessThanOrEqual(200);
    expect(serialized).not.toContain('a'.repeat(1_000));
  });

  it('bounds omitted atomic type labels before building the notice', () => {
    const boundedType = `${'a'.repeat(250)}/b`;
    const content = Array.from({ length: 50 }, () => ({
      type: boundedType,
      data: 'x',
    })) as ToolContent;

    const compacted = compactToolContent(content, 200);
    const serialized = serializeToolContent(compacted.content);

    expect(compacted.changed).toBe(true);
    expect(serialized.length).toBeLessThanOrEqual(200);
    expect(serialized).not.toContain(boundedType);
  });

  it('preserves safe provider-native tool-result blocks atomically', () => {
    const webSearchResult = {
      type: 'web_search_result',
      url: 'https://example.com/result',
      title: 'Example result',
      encrypted_content: 'encrypted',
    };
    const content = [
      {
        type: 'search_result',
        title: 'Knowledge base result',
        source: 'https://example.com/source',
        citations: { enabled: true },
        content: [{ type: 'text', text: 'Citation-preserving content' }],
      },
      {
        type: 'web_search_tool_result',
        tool_use_id: 'srvtoolu_search',
        content: [webSearchResult],
      },
      webSearchResult,
      {
        type: 'tool_result',
        tool_use_id: 'toolu_custom',
        content: 'done',
      },
      {
        type: 'server_tool_call_result',
        toolCallId: 'server_call',
        status: 'success',
        output: { stdout: 'done' },
      },
      {
        type: 'toolResponse',
        toolResponse: {
          id: 'google_search',
          name: 'google_search',
          response: { results: [] },
        },
      },
    ] as ToolContent;

    expect(
      (content as Exclude<ToolContent, string>).every(isAtomicToolContentBlock)
    ).toBe(true);
    const result = compactToolContent(content, 10_000);

    expect(result.changed).toBe(false);
    expect(result.content).toBe(content);
  });

  it('omits an oversized provider-native result as one atomic block', () => {
    const sentinel = 'OVERSIZED_NATIVE_RESULT';
    const content = [
      {
        type: 'search_result',
        title: 'Large result',
        source: 'https://example.com/large',
        citations: { enabled: true },
        content: [
          {
            type: 'text',
            text: `${sentinel}:${'x'.repeat(20_000)}`,
          },
        ],
      },
    ] as ToolContent;

    const result = compactToolContent(content, 200);
    const serialized = serializeToolContent(result.content);

    expect(result.changed).toBe(true);
    expect(serialized.length).toBeLessThanOrEqual(200);
    expect(serialized).toContain('omitted');
    expect(serialized).toContain('search_result');
    expect(serialized).not.toContain(sentinel);
  });

  it('omits an oversized inline media block instead of leaking past the cap', () => {
    const base64 = 'A'.repeat(10_000);
    const content = [
      {
        type: 'image_url',
        image_url: { url: `data:image/png;base64,${base64}` },
      },
    ] as ToolContent;

    const result = compactToolContent(content, 200);

    expect(result.changed).toBe(true);
    expect(serializeToolContent(result.content).length).toBeLessThanOrEqual(
      200
    );
    expect(serializeToolContent(result.content)).toContain('omitted');
    expect(serializeToolContent(result.content)).not.toContain(base64);
  });

  it('sizes large native byte payloads without JSON-expanding them', () => {
    const bytes = new Uint8Array(1_000_000);
    const content = [
      {
        type: 'video',
        video: { source: { bytes } },
      },
    ] as ToolContent;

    const uncapped = compactToolContent(content, Number.MAX_SAFE_INTEGER);
    const compacted = compactToolContent(content, 200);

    expect(uncapped.content).toBe(content);
    expect(compacted.changed).toBe(true);
    expect(compacted.originalChars).toBeGreaterThan(1_000_000);
    expect(serializeToolContent(compacted.content).length).toBeLessThanOrEqual(
      200
    );
    expect(serializeToolContent(compacted.content)).toContain('omitted');
  });

  it('preserves small document blocks while compacting adjacent text', () => {
    const document = {
      type: 'document',
      source: { type: 'url', url: 'https://example.com/report.pdf' },
    };
    const content = [
      { type: 'text', text: 'x'.repeat(2_000) },
      document,
    ] as ToolContent;

    const result = compactToolContent(content, 400);

    expect(result.changed).toBe(true);
    expect(Array.isArray(result.content)).toBe(true);
    expect(result.content).toContain(document);
    expect(serializeToolContent(result.content).length).toBeLessThanOrEqual(
      400
    );
  });

  it('repeats shared non-cyclic values and marks only true cycles', () => {
    const shared = { value: 'repeated' };
    expect(serializeStructuredValue([shared, shared])).toBe(
      '[{"value":"repeated"},{"value":"repeated"}]'
    );

    const cyclic: { self?: unknown } = {};
    cyclic.self = cyclic;
    expect(serializeStructuredValue(cyclic)).toBe('{"self":"[Circular]"}');
  });

  it('normalizes cyclic atomic and text blocks instead of returning them unchanged', () => {
    const cyclicPayload: {
      url: string;
      self?: unknown;
    } = {
      url: 'https://example.com/chart.png',
    };
    cyclicPayload.self = cyclicPayload;
    const atomicBlock = {
      type: 'image_url',
      image_url: cyclicPayload,
    };
    const textBlock: {
      type: 'text';
      text: string;
      metadata?: unknown;
    } = {
      type: 'text',
      text: 'done',
    };
    textBlock.metadata = textBlock;

    const atomicResult = compactToolContent(
      [atomicBlock] as ToolContent,
      1_000
    );
    const textResult = compactToolContent([textBlock] as ToolContent, 1_000);

    expect(isAtomicToolContentBlock(atomicBlock)).toBe(false);
    for (const result of [atomicResult, textResult]) {
      expect(result.changed).toBe(true);
      expect(typeof result.content).toBe('string');
      expect((result.content as string).length).toBeLessThanOrEqual(1_000);
      expect(result.content).toContain('[Circular]');
      expect(() => JSON.parse(result.content as string)).not.toThrow();
    }
  });

  it('still preserves atomic blocks with shared non-cyclic values', () => {
    const shared = { source: 'tool' };
    const block = {
      type: 'image_url',
      image_url: { url: 'https://example.com/chart.png' },
      first: shared,
      second: shared,
    };
    const content = [block] as ToolContent;

    expect(isAtomicToolContentBlock(block)).toBe(true);
    const compacted = compactToolContent(content, 1_000);

    expect(compacted.changed).toBe(false);
    expect(compacted.content).toBe(content);
  });

  it('bounds structured values without invoking custom toJSON', () => {
    let toJSONCalls = 0;
    const value = {
      safe: 'ok',
      toJSON() {
        toJSONCalls++;
        return 'x'.repeat(10_000);
      },
    };

    const serialized = serializeStructuredValueBounded(value, 10);
    const compacted = compactToolContent(value, 10);

    expect(toJSONCalls).toBe(0);
    expect(serialized.content.length).toBeLessThanOrEqual(10);
    expect(String(compacted.content).length).toBeLessThanOrEqual(10);
    expect(serialized.originalChars).toBe(
      JSON.stringify({ safe: 'ok' }).length
    );
  });

  it('does not preserve atomic blocks with callable toJSON methods', () => {
    let toJSONCalls = 0;
    const block = {
      type: 'image_url',
      image_url: { url: 'https://example.com/chart.png' },
      toJSON() {
        toJSONCalls++;
        return {
          type: 'image_url',
          image_url: { url: 'x'.repeat(10_000) },
        };
      },
    };

    const compacted = compactToolContent([block] as ToolContent, 200);

    expect(toJSONCalls).toBe(0);
    expect(typeof compacted.content).toBe('string');
    expect((compacted.content as string).length).toBeLessThanOrEqual(200);
    expect(compacted.changed).toBe(true);
  });

  it('does not invoke toJSON accessors while validating atomic blocks', () => {
    let getterCalls = 0;
    const source = { url: 'https://example.com/chart.png' };
    Object.defineProperty(source, 'toJSON', {
      enumerable: false,
      get() {
        getterCalls++;
        return (): string => 'unsafe';
      },
    });
    const block = { type: 'image_url', image_url: source };

    const compacted = compactToolContent([block] as ToolContent, 200);

    expect(getterCalls).toBe(0);
    expect(typeof compacted.content).toBe('string');
    expect((compacted.content as string).length).toBeLessThanOrEqual(200);
  });

  it('does not preserve ArrayBuffer payloads with custom toJSON methods', () => {
    let toJSONCalls = 0;
    const data = new ArrayBuffer(8) as ArrayBuffer & {
      toJSON: () => string;
    };
    data.toJSON = () => {
      toJSONCalls++;
      return 'x'.repeat(10_000);
    };
    const block = {
      type: 'media',
      mimeType: 'application/octet-stream',
      data,
    };

    const compacted = compactToolContent([block] as ToolContent, 200);

    expect(isAtomicToolContentBlock(block)).toBe(false);
    expect(toJSONCalls).toBe(0);
    expect(typeof compacted.content).toBe('string');
    expect((compacted.content as string).length).toBeLessThanOrEqual(200);
  });

  it('preserves only native Dates inside atomic blocks', () => {
    let toJSONCalls = 0;
    let toISOStringCalls = 0;
    const nativeBlock = {
      type: 'image_url',
      image_url: { url: 'https://example.com/chart.png' },
      capturedAt: new Date('2026-07-27T12:00:00.000Z'),
    };
    const dateWithToJSON = new Date('2026-07-27T12:00:00.000Z');
    Object.defineProperty(dateWithToJSON, 'toJSON', {
      value() {
        toJSONCalls++;
        return 'x'.repeat(10_000);
      },
    });
    const dateWithToISOString = new Date('2026-07-27T12:00:00.000Z');
    Object.defineProperty(dateWithToISOString, 'toISOString', {
      value() {
        toISOStringCalls++;
        return 'x'.repeat(10_000);
      },
    });
    class ExpandingDate extends Date {
      override toJSON(): string {
        toJSONCalls++;
        return 'x'.repeat(10_000);
      }
    }

    expect(isAtomicToolContentBlock(nativeBlock)).toBe(true);
    expect(
      isAtomicToolContentBlock({
        ...nativeBlock,
        capturedAt: dateWithToJSON,
      })
    ).toBe(false);
    expect(
      isAtomicToolContentBlock({
        ...nativeBlock,
        capturedAt: dateWithToISOString,
      })
    ).toBe(false);
    expect(
      isAtomicToolContentBlock({
        ...nativeBlock,
        capturedAt: new ExpandingDate('2026-07-27T12:00:00.000Z'),
      })
    ).toBe(false);
    expect(toJSONCalls).toBe(0);
    expect(toISOStringCalls).toBe(0);
  });

  it('rejects atomic blocks with enumerable accessors without invoking them', () => {
    let getterCalls = 0;
    const imageUrl = {};
    Object.defineProperty(imageUrl, 'url', {
      enumerable: true,
      get() {
        getterCalls++;
        return getterCalls === 1
          ? 'https://example.com/chart.png'
          : 'x'.repeat(10_000);
      },
    });

    expect(
      isAtomicToolContentBlock({
        type: 'image_url',
        image_url: imageUrl,
      })
    ).toBe(false);
    expect(getterCalls).toBe(0);
  });

  it('rejects inherited discriminators without invoking accessors', () => {
    for (const enumerable of [true, false]) {
      let getterCalls = 0;
      const prototype = {};
      Object.defineProperty(prototype, 'type', {
        enumerable,
        get() {
          getterCalls++;
          return 'image_url';
        },
      });
      const block = Object.create(prototype) as Record<string, unknown>;
      block.image_url = { url: 'https://example.com/chart.png' };

      expect(isAtomicToolContentBlock(block)).toBe(false);
      const compacted = compactToolContent([block] as ToolContent, 200);
      expect(getterCalls).toBe(0);
      expect(typeof compacted.content).toBe('string');
    }

    const inheritedDataBlock = Object.create({
      type: 'image_url',
      image_url: { url: 'https://example.com/chart.png' },
    }) as Record<string, unknown>;
    expect(isAtomicToolContentBlock(inheritedDataBlock)).toBe(false);
  });

  it('rejects inherited payloads without invoking accessors', () => {
    let getterCalls = 0;
    const imageUrlPrototype = {};
    Object.defineProperty(imageUrlPrototype, 'url', {
      enumerable: false,
      get() {
        getterCalls++;
        return `data:image/png;base64,${'A'.repeat(100_000)}`;
      },
    });
    const block = {
      type: 'image_url',
      image_url: Object.create(imageUrlPrototype) as Record<string, unknown>,
    };

    expect(isAtomicToolContentBlock(block)).toBe(false);
    const compacted = compactToolContent([block] as ToolContent, 200);

    expect(getterCalls).toBe(0);
    expect(compacted.changed).toBe(true);
    expect(typeof compacted.content).toBe('string');
    expect((compacted.content as string).length).toBeLessThanOrEqual(200);
  });

  it('compacts accessor-backed atomic shapes without invoking accessors', () => {
    let getterCalls = 0;
    const imageUrl = {};
    Object.defineProperty(imageUrl, 'url', {
      enumerable: true,
      get() {
        getterCalls++;
        return getterCalls === 1
          ? 'https://example.com/chart.png'
          : 'x'.repeat(10_000);
      },
    });

    const compacted = compactToolContent(
      [{ type: 'image_url', image_url: imageUrl }] as ToolContent,
      200
    );

    expect(getterCalls).toBe(0);
    expect(typeof compacted.content).toBe('string');
    expect((compacted.content as string).length).toBeLessThanOrEqual(200);
    expect(compacted.content).toContain('[Property accessor omitted]');
  });

  it('bounds generic accessor properties without invoking them', () => {
    let getterCalls = 0;
    const value = {};
    Object.defineProperty(value, 'payload', {
      enumerable: true,
      get() {
        getterCalls++;
        return 'x'.repeat(10_000);
      },
    });

    const serialized = serializeStructuredValueBounded(value, 200);

    expect(getterCalls).toBe(0);
    expect(serialized.content).toBe(
      '{"payload":"[Property accessor omitted]"}'
    );
  });

  it('serializes array holes without invoking inherited accessors', () => {
    let getterCalls = 0;
    const prototype = Object.create(Array.prototype) as unknown[];
    Object.defineProperty(prototype, '0', {
      get() {
        getterCalls++;
        return 'x'.repeat(10_000);
      },
    });
    const value: unknown[] = [];
    value.length = 1;
    Object.setPrototypeOf(value, prototype);

    const serialized = serializeStructuredValueBounded(value, 200);

    expect(getterCalls).toBe(0);
    expect(serialized.content).toBe('"[Unsafe array omitted]"');
  });

  it('fails closed when a proxy blocks own property descriptors', () => {
    let propertyReads = 0;
    const value = new Proxy<unknown[]>(['safe'], {
      get(target, property, receiver) {
        propertyReads++;
        return Reflect.get(target, property, receiver);
      },
      getOwnPropertyDescriptor(target, property) {
        if (property === '0') {
          throw new Error('blocked descriptor');
        }
        return Reflect.getOwnPropertyDescriptor(target, property);
      },
    });

    const serialized = serializeStructuredValueBounded(value, 200);

    expect(propertyReads).toBe(0);
    expect(serialized.content).toBe('"[Proxy value omitted]"');
    expect(serialized.originalChars).toBe(Number.MAX_SAFE_INTEGER);
    expect(serialized.truncated).toBe(true);
  });

  it('never returns stateful proxy blocks through structured fast paths', () => {
    const hugeValue = 'x'.repeat(1_000_000);
    let descriptorReads = 0;
    let propertyReads = 0;
    function statefulProxy<T extends object>(
      target: T,
      unstableKey: PropertyKey,
      expandedValue: unknown
    ): T {
      return new Proxy(target, {
        get(innerTarget, property, receiver) {
          propertyReads++;
          if (property === unstableKey && propertyReads > 2) {
            return expandedValue;
          }
          return Reflect.get(innerTarget, property, receiver);
        },
        getOwnPropertyDescriptor(innerTarget, property) {
          descriptorReads++;
          const descriptor = Reflect.getOwnPropertyDescriptor(
            innerTarget,
            property
          );
          if (
            descriptor != null &&
            property === unstableKey &&
            descriptorReads > 2
          ) {
            return { ...descriptor, value: expandedValue };
          }
          return descriptor;
        },
      });
    }
    const textBlock = statefulProxy(
      { type: 'text', text: 'ok' },
      'text',
      hugeValue
    );
    const atomicBlock = statefulProxy(
      {
        type: 'image_url',
        image_url: { url: 'https://example.com/chart.png' },
      },
      'image_url',
      { url: `data:image/png;base64,${hugeValue}` }
    );
    const content = [textBlock, atomicBlock] as ToolContent;

    const compacted = compactToolContent(content, 1_000);

    expect(descriptorReads).toBe(0);
    expect(propertyReads).toBe(0);
    expect(compacted.changed).toBe(true);
    expect(compacted.content).not.toBe(content);
    expect(typeof compacted.content).toBe('string');
    expect((compacted.content as string).length).toBeLessThanOrEqual(1_000);
    expect(compacted.content).toContain('[Proxy value omitted]');
    expect(() => JSON.parse(compacted.content as string)).not.toThrow();
  });

  it('does not traverse proxy arrays or proxy payloads while compacting', () => {
    const hugeValue = 'x'.repeat(1_000_000);
    let proxyTrapCalls = 0;
    let accessorCalls = 0;
    const payloadTarget = {};
    Object.defineProperty(payloadTarget, 'url', {
      enumerable: true,
      get() {
        accessorCalls++;
        return `data:image/png;base64,${hugeValue}`;
      },
    });
    const payload = new Proxy(payloadTarget, {
      get() {
        proxyTrapCalls++;
        return hugeValue;
      },
      getOwnPropertyDescriptor(target, property) {
        proxyTrapCalls++;
        return Reflect.getOwnPropertyDescriptor(target, property);
      },
    });
    const proxiedArray = new Proxy(
      [
        {
          type: 'image_url',
          image_url: payload,
        },
      ],
      {
        get(target, property, receiver) {
          proxyTrapCalls++;
          return Reflect.get(target, property, receiver);
        },
        getOwnPropertyDescriptor(target, property) {
          proxyTrapCalls++;
          return Reflect.getOwnPropertyDescriptor(target, property);
        },
      }
    ) as ToolContent;
    const callableProxy = new Proxy(() => 'safe', {
      get() {
        proxyTrapCalls++;
        return hugeValue;
      },
    });

    const arrayResult = compactToolContent(proxiedArray, 1_000);
    const callableResult = compactToolContent(callableProxy, 1_000);
    const nestedResult = compactToolContent(
      [
        {
          type: 'image_url',
          image_url: payload,
        },
      ] as ToolContent,
      1_000
    );

    expect(proxyTrapCalls).toBe(0);
    expect(accessorCalls).toBe(0);
    expect(arrayResult.changed).toBe(true);
    expect(callableResult.changed).toBe(true);
    expect(nestedResult.changed).toBe(true);
    expect(arrayResult.content).toBe('"[Proxy value omitted]"');
    expect(callableResult.content).toBe('"[Proxy value omitted]"');
    expect(typeof nestedResult.content).toBe('string');
    expect((nestedResult.content as string).length).toBeLessThanOrEqual(1_000);
    expect(nestedResult.content).toContain('[Proxy value omitted]');
    expect(() => JSON.parse(nestedResult.content as string)).not.toThrow();
  });

  it('counts lone surrogates before preserving atomic content', () => {
    const content = [
      {
        type: 'image_url',
        image_url: { url: '\ud800'.repeat(30) },
      },
    ] as ToolContent;

    const compacted = compactToolContent(content, 200);

    expect(compacted.changed).toBe(true);
    expect(serializeToolContent(compacted.content).length).toBeLessThanOrEqual(
      200
    );
  });

  it('serializes Buffer values as bounded byte placeholders', () => {
    const value = Buffer.alloc(10_000, 1);

    const serialized = serializeStructuredValueBounded(value, 15_000);
    const compacted = compactToolContent(value, 15_000);

    expect(serialized.content).toBe('"[Buffer: 10000 bytes]"');
    expect(serialized.content.length).toBeLessThanOrEqual(15_000);
    expect(compacted.content).toBe(serialized.content);
  });

  it('does not preserve Buffer toJSON expansion inside atomic blocks', () => {
    const content = [
      {
        type: 'image',
        data: Buffer.alloc(100, 255),
      },
    ] as ToolContent;

    const compacted = compactToolContent(content, 200);

    expect(compacted.changed).toBe(true);
    expect(typeof compacted.content).toBe('string');
    expect((compacted.content as string).length).toBeLessThanOrEqual(200);
    expect(compacted.content).toContain('[Buffer: 100 bytes]');
  });

  it('retains an exact registry prefix independently of the provider preview', () => {
    const value = { payload: 'x'.repeat(1_000), tail: 'done' };
    const exact = JSON.stringify(value);

    const serialized = serializeStructuredValueBounded(
      value,
      100,
      exact.length
    );

    expect(serialized.content.length).toBeLessThanOrEqual(100);
    expect(serialized.content).toContain('truncated');
    expect(serialized.prefix).toBe(exact);
  });

  it('emits JSON-exact strings across escape and surrogate chunk boundaries', () => {
    const value = {
      payload:
        `${'x'.repeat(4_095)}😀` +
        `\ud800${'y'.repeat(4_095)}\udc00` +
        '\b\t\n\f\r"\\',
    };
    const exact = JSON.stringify(value);

    const serialized = serializeStructuredValueBounded(value, exact.length);

    expect(serialized.content).toBe(exact);
    expect(serialized.originalChars).toBe(exact.length);
    expect(serialized.truncated).toBe(false);
  });

  it('bounds a 32 MiB primitive string within a 128 MiB heap', () => {
    const script = `
      const { serializeStructuredValueBounded } =
        await import('./src/utils/toolContent.ts');
      const payload = 'A'.repeat(32 * 1024 * 1024);
      const result = serializeStructuredValueBounded({ payload }, 4096);
      if (
        result.content.length !== 4096 ||
        result.originalChars !== Number.MAX_SAFE_INTEGER ||
        result.truncated !== true
      ) {
        throw new Error(JSON.stringify({
          contentLength: result.content.length,
          originalChars: result.originalChars,
          truncated: result.truncated,
        }));
      }
      process.stdout.write('ok');
    `;
    const benchmark = spawnSync(
      process.execPath,
      [
        '--max-old-space-size=128',
        '--import',
        'tsx',
        '--input-type=module',
        '--eval',
        script,
      ],
      {
        cwd: process.cwd(),
        encoding: 'utf8',
        env: { ...process.env, NODE_NO_WARNINGS: '1' },
        timeout: 30_000,
      }
    );

    if (benchmark.status !== 0) {
      throw new Error(
        `Constrained-heap serializer benchmark failed: ${benchmark.stderr}`
      );
    }
    expect(benchmark.stdout).toBe('ok');
  });

  it('bounds arrays with many small values using an exact segmented prefix', () => {
    const value = Array.from({ length: 25_000 }, (_, index) => index);
    const exact = JSON.stringify(value);

    const serialized = serializeStructuredValueBounded(value, 1_000, 2_000);

    expect(serialized.content.length).toBeLessThanOrEqual(1_000);
    expect(serialized.content).toContain('truncated');
    expect(serialized.prefix).toBe(exact.slice(0, 2_000));
    expect(serialized.originalChars).toBe(exact.length);
  });

  it('routes sparse text-looking arrays through bounded serialization', () => {
    const sparse: unknown[] = [];
    sparse.length = 100;

    const compacted = compactToolContent(sparse, 20);

    expect(typeof compacted.content).toBe('string');
    expect((compacted.content as string).length).toBeLessThanOrEqual(20);
    expect(compacted.changed).toBe(true);
  });

  it('fails closed without reading a hostile array proxy length', () => {
    const hostile = new Proxy<unknown[]>([], {
      get(target, property, receiver) {
        if (property === 'length') {
          throw new Error('blocked length');
        }
        return Reflect.get(target, property, receiver);
      },
    });

    const compacted = compactToolContent(hostile, 20);

    expect(typeof compacted.content).toBe('string');
    expect((compacted.content as string).length).toBeLessThanOrEqual(20);
    expect(compacted.originalChars).toBe(Number.MAX_SAFE_INTEGER);
  });

  it('caps traversal work for enormous sparse arrays', () => {
    const sparse: unknown[] = [];
    sparse.length = 0xffffffff;

    const serialized = serializeStructuredValueBounded(sparse, 200);
    const compacted = compactToolContent(sparse, 200);

    expect(serialized.content.length).toBeLessThanOrEqual(200);
    expect(serialized.originalChars).toBe(Number.MAX_SAFE_INTEGER);
    expect(serialized.truncated).toBe(true);
    expect(typeof compacted.content).toBe('string');
    expect((compacted.content as string).length).toBeLessThanOrEqual(200);
    expect(compacted.originalChars).toBe(Number.MAX_SAFE_INTEGER);
  });

  it('bounds canonical cache-controlled text without enumerating malformed arrays', () => {
    const canonical = getBoundedCacheControlledTextToolContent(
      [
        {
          type: 'text',
          text: 'result body',
          cache_control: { type: 'ephemeral' },
        },
      ],
      6
    );
    const ownKeys = jest.spyOn(Reflect, 'ownKeys');
    const malformed: unknown[] = [];
    malformed.length = 100_000;

    expect(canonical).toEqual([
      {
        type: 'text',
        text: 'result',
        cache_control: { type: 'ephemeral' },
      },
    ]);
    expect(
      getBoundedCacheControlledTextToolContent(malformed, 100)
    ).toBeUndefined();
    expect(ownKeys).not.toHaveBeenCalled();
    ownKeys.mockRestore();
  });

  it('rejects an accessor-backed cache ttl without invoking it', () => {
    let ttlReads = 0;
    const cacheControl = { type: 'ephemeral' };
    Object.defineProperty(cacheControl, 'ttl', {
      enumerable: true,
      get() {
        ttlReads++;
        return '1h';
      },
    });

    expect(
      getBoundedCacheControlledTextToolContent(
        [{ type: 'text', text: 'result', cache_control: cacheControl }],
        100
      )
    ).toBeUndefined();
    expect(ttlReads).toBe(0);

    const nonEnumerableTtl = { type: 'ephemeral' };
    Object.defineProperty(nonEnumerableTtl, 'ttl', { value: '1h' });
    expect(
      getBoundedCacheControlledTextToolContent(
        [{ type: 'text', text: 'result', cache_control: nonEnumerableTtl }],
        100
      )
    ).toBeUndefined();
  });

  it('only unwraps canonical text cache wrappers', () => {
    expect(
      getBoundedSingleTextToolContent(
        [{ type: 'text', text: 'result', extra: true }],
        100
      )
    ).toBeUndefined();
    expect(
      getBoundedCacheControlledTextToolContent(
        [
          {
            type: 'text',
            text: 'result',
            cache_control: { type: 'ephemeral', extra: true },
          },
        ],
        100
      )
    ).toBeUndefined();
  });

  it('rejects oversized screenshot URLs and file ids before scanning them', () => {
    const oversizedUrl = `https://example.com/${'a'.repeat(20_000)}`;
    const charCodeAt = jest.spyOn(String.prototype, 'charCodeAt');

    const urlResult = getComputerCallOutputScreenshot(oversizedUrl);
    const urlCharacterReads = charCodeAt.mock.calls.length;
    charCodeAt.mockRestore();

    expect(urlResult).toBeUndefined();
    expect(urlCharacterReads).toBe(0);
    expect(
      getComputerCallOutputScreenshot([
        {
          type: 'input_image',
          file_id: `file_${'a'.repeat(5_000)}`,
        },
      ])
    ).toBeUndefined();
  });

  it('preserves bounded native computer screenshots', () => {
    const screenshot = `data:image/png;base64,${'A'.repeat(100_000)}`;

    expect(getComputerCallOutputScreenshot(screenshot)).toEqual({
      type: 'input_image',
      image_url: screenshot,
    });
  });

  it('keeps large bounded head-tail collection linear', () => {
    const values = new Array<number>(500_000).fill(0);
    const serialized = serializeStructuredValueBounded(values, 400_000);

    expect(serialized.content).toHaveLength(400_000);
    expect(serialized.originalChars).toBe(1_000_001);
    expect(serialized.truncated).toBe(true);
  });

  it('bounds provider-facing tool content directly', () => {
    const serialized = serializeToolContentBounded(
      { payload: 'x'.repeat(1_000) },
      80
    );

    expect(serialized.length).toBeLessThanOrEqual(80);
    expect(serialized).toContain('truncated');
  });
});
