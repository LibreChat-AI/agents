import { runInNewContext } from 'node:vm';
import { serializeToolArguments } from '../arguments';

const encode = (value: unknown, bytes = 10000): string =>
  serializeToolArguments(value, bytes);
const invalid = 'not JSON serializable';

describe('bounded tool argument encoding', () => {
  it('accepts plain JSON from another realm but not foreign built-in or custom types', () => {
    expect(encode(runInNewContext('({ value: [1, { city: "Paris" }] })'))).toBe(
      '{"value":[1,{"city":"Paris"}]}'
    );
    for (const expression of [
      'new Date()',
      'new Map()',
      'new (class {x=1})()',
    ]) {
      expect(() => encode({ value: runInNewContext(expression) })).toThrow(
        invalid
      );
    }
  });

  it('round-trips plain JSON, dense arrays, escapes, unicode and null-prototype objects', () => {
    const plain: Record<string, unknown> = Object.create(null);
    plain.key = {
      a: [true, false, null, 1.25, 1e-8, { x: 'é🌧\n"\\\u0000' }],
      empty: [],
    };
    const text = JSON.stringify(plain);
    expect(encode(plain)).toBe(text);
    expect(encode(plain, Buffer.byteLength(text))).toBe(text);
    expect(() => encode(plain, Buffer.byteLength(text) - 1)).toThrow(
      'buffer limit'
    );
  });

  it('preserves negative zero instead of silently replacing it', () => {
    expect(Object.is(JSON.parse(encode({ value: -0 })).value, -0)).toBe(true);
  });

  it('preserves own __proto__ keys without changing prototypes', () => {
    const obj: unknown = JSON.parse('{"__proto__":{"polluted":true}}');
    expect(encode(obj)).toBe('{"__proto__":{"polluted":true}}');
    expect(Object.prototype).not.toHaveProperty('polluted');
  });

  it.each([undefined, () => 1, Symbol('x'), 1n, NaN, Infinity])(
    'rejects non-JSON leaf %s',
    (value) => {
      expect(() => encode({ value })).toThrow(invalid);
    }
  );

  it.each([
    new Map(),
    new Set(),
    new Date(),
    /x/,
    new Uint8Array([1]),
    new ArrayBuffer(1),
    Object(1),
    Object('x'),
    new (class {
      value = 1;
    })(),
  ])('rejects non-plain object %s', (value) => {
    expect(() => encode({ value })).toThrow(invalid);
  });

  it('does not invoke accessors or serialization hooks', () => {
    const getter = jest.fn(() => 'SECRET');
    const toJSON = jest.fn(() => ({ okay: true }));
    expect(() => encode({ value: { toJSON } })).toThrow(invalid);
    expect(toJSON).not.toHaveBeenCalled();
    const value = Object.defineProperty({}, 'value', {
      get: getter,
      enumerable: true,
    });
    expect(() => encode(value)).toThrow(invalid);
    const list = Object.defineProperty([0], '0', {
      get: getter,
      enumerable: true,
    });
    expect(() => encode({ list })).toThrow(invalid);
    expect(getter).not.toHaveBeenCalled();
  });

  it('rejects proxies without invoking traps', () => {
    const trap = jest.fn(() => {
      throw new Error('SECRET');
    });
    const value = new Proxy(
      {},
      { get: trap, ownKeys: trap, getPrototypeOf: trap }
    );
    expect(() => encode({ value })).toThrow(invalid);
    expect(trap).not.toHaveBeenCalled();
  });

  it('rejects sparse arrays, array side properties, hidden properties and symbol keys', () => {
    expect(() => encode({ value: Array(2) })).toThrow(invalid);
    const value = Object.assign([1], { named: 2 });
    expect(() => encode({ value })).toThrow(invalid);
    expect(() =>
      encode({
        value: Object.defineProperty({}, 'hidden', { value: 'secret' }),
      })
    ).toThrow(invalid);
    expect(() => encode({ [Symbol('s')]: 1 })).toThrow(invalid);
  });

  it('detects cycles but permits repeated references and budgets their expansion', () => {
    const shared = { text: 'okay' };
    const value = { first: shared, second: shared };
    expect(encode(value)).toBe(JSON.stringify(value));
    const cyclic: Record<string, unknown> = {};
    cyclic.self = cyclic;
    expect(() => encode(cyclic)).toThrow(invalid);
    let exponential: unknown = {};
    for (let i = 0; i < 30; i++) exponential = [exponential, exponential];
    expect(() => encode({ value: exponential }, 1024)).toThrow('buffer limit');
  });

  it('bounds depth before exhausting the stack', () => {
    let value: unknown = 1;
    for (let i = 0; i < 10000; i++) value = { value };
    expect(() => encode(value)).toThrow(invalid);
  });

  it('checks output budget before visiting wide objects or escaping oversized strings', () => {
    expect(() => encode({ value: 'x'.repeat(10000) }, 10)).toThrow(
      'buffer limit'
    );
    expect(() => encode({ a: 1, b: 2, c: 3 }, 1)).toThrow('buffer limit');
    expect(() => encode({ value: '\u0000' }, 15)).toThrow('buffer limit');
  });

  it.each([null, 'object', 1, []])('rejects non-object root %s', (value) => {
    expect(() => encode(value)).toThrow('must be an object');
  });
});
