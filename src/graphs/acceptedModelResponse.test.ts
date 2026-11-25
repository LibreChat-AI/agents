import { AIMessageChunk } from '@langchain/core/messages';
import {
  snapshotValidatedModelChunk,
  detachValidatedModelToolCalls,
  snapshotAcceptedModelResponse,
} from './acceptedModelResponse';
import { cloneToolArguments } from '@/utils/acceptedToolArguments';
import { claimStreamLimitCharge } from '@/llm/streamLimits';

function response(): AIMessageChunk {
  return new AIMessageChunk({
    content: '',
    tool_calls: [{ id: 'provider', name: 'lookup', args: { city: 'Paris' } }],
  });
}

describe('graph-accepted tool snapshot', () => {
  it('rejects invalid diagnostics before cloning or accessing their fields', () => {
    const message = response();
    const getter = jest.fn(() => 'SENSITIVE'.repeat(1024));
    const diagnostic = { type: 'invalid_tool_call' as const, name: 'lookup' };
    Object.defineProperty(diagnostic, 'args', {
      enumerable: true,
      get: getter,
    });
    message.invalid_tool_calls = [diagnostic];
    expect(() =>
      snapshotAcceptedModelResponse(message, 'accepted', 'agent')
    ).toThrow('invalid tool calls');
    expect(getter).not.toHaveBeenCalled();
  });

  it.each(['tool_calls', 'invalid_tool_calls'] as const)(
    'rejects a %s accessor before reading it',
    (field) => {
      const message = response();
      const getter = jest.fn(() => []);
      Object.defineProperty(message, field, { enumerable: true, get: getter });
      expect(() =>
        snapshotAcceptedModelResponse(message, 'accepted', 'agent')
      ).toThrow('non-serializable tool calls');
      expect(getter).not.toHaveBeenCalled();
    }
  );

  it.each(['tool_calls', 'invalid_tool_calls'] as const)(
    'rejects a non-array %s instead of accepting an empty result',
    (field) => {
      const message = response();
      Object.defineProperty(message, field, { value: { length: 0 } });
      expect(() =>
        snapshotAcceptedModelResponse(message, 'accepted', 'agent')
      ).toThrow('non-serializable tool calls');
    }
  );

  it('rejects an oversized invalid-only response before enumerating it', () => {
    const message = response();
    message.tool_calls = [];
    message.invalid_tool_calls = Array(1_000_000);
    expect(() =>
      snapshotAcceptedModelResponse(message, 'accepted', 'agent')
    ).toThrow('invalid tool calls');
  });

  it('detaches JSON arguments without modifying the native call', () => {
    const message = response();
    const event = snapshotAcceptedModelResponse(message, 'accepted', 'agent');
    expect(event.toolCalls[0]).toMatchObject({
      id: 'provider',
      name: 'lookup',
      args: { city: 'Paris' },
    });
    event.toolCalls[0].args.city = 'changed';
    expect(message.tool_calls?.[0].args).toEqual({ city: 'Paris' });
  });

  it('classifies individual provider-executed IDs without treating prefixes as proof', () => {
    const message = new AIMessageChunk({
      content: '',
      tool_calls: [
        { id: 'srvtoolu_unconfirmed', name: 'web_search', args: {} },
        { id: 'executed', name: 'web_search', args: {} },
        { id: 'local', name: 'lookup', args: {} },
      ],
    });
    const accepted = snapshotAcceptedModelResponse(
      message,
      'response',
      'agent',
      new Set(['executed'])
    );
    expect(accepted.toolCallDispositions).toEqual(['sdk', 'provider', 'sdk']);
  });

  it('does not invoke argument, name or ID getters on an accepted call', () => {
    for (const key of ['args', 'name', 'id'] as const) {
      const message = response();
      const getter = jest.fn(() => 'SENSITIVE');
      Object.defineProperty(message.tool_calls![0], key, {
        enumerable: true,
        get: getter,
      });
      expect(() =>
        snapshotAcceptedModelResponse(message, 'accepted', 'agent')
      ).toThrow('non-serializable tool calls');
      expect(getter).not.toHaveBeenCalled();
    }
  });

  it('does not invoke a proxy trap before rejecting the tool-call object', () => {
    const message = response();
    const trap = jest.fn(() => {
      throw new Error('SENSITIVE');
    });
    message.tool_calls![0] = new Proxy(message.tool_calls![0], {
      get: trap,
      getOwnPropertyDescriptor: trap,
    });
    expect(() =>
      snapshotAcceptedModelResponse(message, 'accepted', 'agent')
    ).toThrow('non-serializable tool calls');
    expect(trap).not.toHaveBeenCalled();
  });

  it('rejects sparse and oversized tool-call lists before copying anything', () => {
    const message = response();
    message.tool_calls = Array(1);
    expect(() =>
      snapshotAcceptedModelResponse(message, 'accepted', 'agent')
    ).toThrow('non-serializable tool calls');
    message.tool_calls = Array(1025);
    expect(() =>
      snapshotAcceptedModelResponse(message, 'accepted', 'agent')
    ).toThrow('snapshot limits');
  });

  it('rejects an oversized argument and custom objects before normalization', () => {
    const message = response();
    message.tool_calls![0].args = { city: 'x'.repeat(4 * 1024 * 1024) };
    expect(() =>
      snapshotAcceptedModelResponse(message, 'accepted', 'agent')
    ).toThrow('buffer limit');
    message.tool_calls![0].args = { city: new Date() };
    expect(() =>
      snapshotAcceptedModelResponse(message, 'accepted', 'agent')
    ).toThrow('not JSON serializable');
  });
});

describe('execution descriptor validation without projection budgets', () => {
  it('clones deep JSON iteratively and preserves shared references without expanding them', () => {
    let tree: Record<string, unknown> = { leaf: true };
    for (let i = 0; i < 5000; i++) tree = { left: tree, right: tree };
    const copy = cloneToolArguments(tree);
    expect(copy).not.toBe(tree);
    let cursor = copy;
    for (let i = 0; i < 5000; i++) {
      expect(cursor.left).toBe(cursor.right);
      cursor = cursor.left as Record<string, unknown>;
    }
    expect(cursor).toEqual({ leaf: true });
  });

  it('still rejects cycles, sparse arrays, accessors, and proxy traps without reading them', () => {
    const cycle: Record<string, unknown> = {};
    cycle.self = cycle;
    const getter = jest.fn();
    const get = jest.fn();
    const accessor = Object.defineProperty({}, 'field', {
      get: getter,
      enumerable: true,
    });
    for (const value of [
      cycle,
      { sparse: new Array(2) },
      accessor,
      new Proxy({}, { get }),
    ]) {
      expect(() => cloneToolArguments(value)).toThrow('not JSON serializable');
    }
    expect(getter).not.toHaveBeenCalled();
    expect(get).not.toHaveBeenCalled();
  });

  it.each(['tool_call_chunks', 'invalid_tool_calls'] as const)(
    'detaches %s from provider-owned records',
    (field) => {
      const record = { id: 'a', name: 'lookup', args: '{' };
      const message = new AIMessageChunk('');
      Object.defineProperty(message, field, {
        value: [record],
        configurable: true,
      });
      detachValidatedModelToolCalls(message);
      record.id = 'changed';
      expect(message[field]![0].id).toBe('a');
    }
  );

  it.each(['tool_call_chunks', 'invalid_tool_calls'] as const)(
    'rejects indexed accessors and proxy records in %s',
    (field) => {
      const get = jest.fn();
      const list = [{}];
      Object.defineProperty(list, '0', { get });
      for (const source of [
        list,
        [new Proxy({}, { get })],
        new Proxy([], { get }),
      ]) {
        const message = new AIMessageChunk('');
        Object.defineProperty(message, field, {
          value: source,
          configurable: true,
        });
        expect(() => detachValidatedModelToolCalls(message)).toThrow(
          'non-serializable tool calls'
        );
      }
      expect(get).not.toHaveBeenCalled();
    }
  );
});

describe('stream chunk envelope snapshots', () => {
  it('keeps legacy descriptor coercion for primitive callback payloads', () => {
    const snapshot = (value: unknown): object =>
      Reflect.apply(snapshotValidatedModelChunk, undefined, [value]);
    const copy = snapshot('legacy');
    expect(Object.getPrototypeOf(copy)).toBe(String.prototype);
    expect(Object.getOwnPropertyDescriptor(copy, '0')).toEqual({
      value: 'l',
      enumerable: true,
      configurable: true,
      writable: true,
    });
    expect(() => snapshot(null)).toThrow(TypeError);
    expect(() => snapshot(undefined)).toThrow(TypeError);
  });

  it('thaws string fields and preserves symbol descriptors without reading getters', () => {
    const original = new AIMessageChunk('Original text');
    const symbol = Symbol('provider metadata');
    const getter = jest.fn();
    Object.defineProperty(original, 'providerField', { get: getter });
    Object.defineProperty(original, symbol, { value: 'provider' });
    Object.defineProperty(original.lc_kwargs, 'hidden', { value: 'metadata' });
    Object.defineProperty(original.lc_kwargs, symbol, { get: getter });
    Object.freeze(original.lc_kwargs);
    Object.freeze(original);

    const copy = snapshotValidatedModelChunk(original);
    expect(Object.getPrototypeOf(copy)).toBe(Object.getPrototypeOf(original));
    expect(copy.lc_kwargs).not.toBe(original.lc_kwargs);
    expect(Object.getOwnPropertyDescriptor(copy, 'providerField')).toEqual({
      get: getter,
      set: undefined,
      enumerable: false,
      configurable: true,
    });
    expect(Object.getOwnPropertyDescriptor(copy.lc_kwargs, 'hidden')).toEqual({
      value: 'metadata',
      enumerable: false,
      configurable: true,
      writable: true,
    });
    expect(Object.getOwnPropertyDescriptor(copy, symbol)).toEqual(
      Object.getOwnPropertyDescriptor(original, symbol)
    );
    expect(Object.getOwnPropertyDescriptor(copy.lc_kwargs, symbol)).toEqual(
      Object.getOwnPropertyDescriptor(original.lc_kwargs, symbol)
    );
    copy._updateId('snapshot');
    expect(copy.id).toBe('snapshot');
    expect(copy.lc_kwargs.id).toBe('snapshot');
    expect(original.id).toBeUndefined();
    expect(original.lc_kwargs.id).toBeUndefined();
    expect(getter).not.toHaveBeenCalled();
  });

  it('rejects unsafe metadata without invoking accessors or proxy traps', () => {
    const getter = jest.fn();
    const accessor = Object.defineProperty({}, 'field', { get: getter });
    for (const value of [null, accessor, new Proxy({}, { ownKeys: getter })]) {
      const message = new AIMessageChunk('Text');
      Object.defineProperty(message, 'lc_kwargs', { value });
      expect(() => snapshotValidatedModelChunk(message)).toThrow(
        'Invalid model response metadata'
      );
    }
    expect(getter).not.toHaveBeenCalled();
  });
});

describe('snapshot emission identity', () => {
  it('keeps nested copies on the original producer/consumer charge identity', () => {
    const original = new AIMessageChunk('');
    const first = snapshotValidatedModelChunk(original);
    const second = snapshotValidatedModelChunk(first);
    const graph = {};
    expect(claimStreamLimitCharge(graph, original, 'producer', {})).toBe(true);
    expect(claimStreamLimitCharge(graph, second, 'consumer', {})).toBe(false);
    expect(claimStreamLimitCharge(graph, first, 'consumer', {})).toBe(true);
    expect(claimStreamLimitCharge(graph, original, 'producer', {})).toBe(false);
  });
});
