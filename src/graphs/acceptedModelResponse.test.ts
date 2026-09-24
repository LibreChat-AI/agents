import { AIMessageChunk } from '@langchain/core/messages';
import { snapshotAcceptedModelResponse } from './acceptedModelResponse';

function response(): AIMessageChunk {
  return new AIMessageChunk({
    content: '',
    tool_calls: [{ id: 'provider', name: 'lookup', args: { city: 'Paris' } }],
  });
}

describe('graph-accepted tool snapshot', () => {
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
