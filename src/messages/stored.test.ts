import { serialize, deserialize } from 'node:v8';
import {
  AIMessage,
  AIMessageChunk,
  ChatMessage,
  HumanMessage,
  ToolMessage,
  mapStoredMessagesToChatMessages,
} from '@langchain/core/messages';
import type { BaseMessage, StoredMessage } from '@langchain/core/messages';
import { toStoredMessage } from './stored';

function roundTrip(message: BaseMessage): BaseMessage {
  const stored = deserialize(
    serialize(toStoredMessage(message))
  ) as StoredMessage;
  return mapStoredMessagesToChatMessages([stored])[0];
}

it('retains native values nested inside messages for the caller codec', () => {
  const fields = {
    bytes: Uint8Array.from([1, 2, 255]),
    date: new Date('2026-01-01T00:00:00Z'),
    map: new Map([['key', new Set([1, 2])]]),
    large: BigInt('12345678901234567890'),
    missing: undefined,
    reserved: { lc: 1, type: 'constructor', $sdk: 'bytes', value: 'user data' },
  };
  const message = new HumanMessage({
    content: 'hello',
    additional_kwargs: fields,
  });
  expect(roundTrip(message).additional_kwargs).toEqual(fields);
});

it('uses live message fields, including values assigned after construction', () => {
  const message = new AIMessage('old');
  message.content = [{ type: 'text', text: 'current' }];
  message.id = 'current-id';
  message.name = 'current-name';
  message.response_metadata = { model_name: 'host-model' };
  message.additional_kwargs = { current: true };
  message.tool_calls = [{ id: 'call', name: 'inspect', args: { id: 1 } }];
  message.invalid_tool_calls = [
    { name: 'broken', args: '{', error: 'invalid' },
  ];
  message.usage_metadata = {
    input_tokens: 7,
    output_tokens: 3,
    total_tokens: 10,
  };
  expect(roundTrip(message)).toMatchObject({
    content: message.content,
    id: message.id,
    name: message.name,
    response_metadata: message.response_metadata,
    additional_kwargs: message.additional_kwargs,
    tool_calls: message.tool_calls,
    invalid_tool_calls: message.invalid_tool_calls,
    usage_metadata: message.usage_metadata,
  });
});

it('preserves tool artifacts, status and the live tool-call identity', () => {
  const message = new ToolMessage({
    content: 'result',
    tool_call_id: 'original',
  });
  message.tool_call_id = 'current';
  message.status = 'error';
  message.artifact = {
    bytes: Uint8Array.from([3, 4]),
    date: new Date('2026-01-01Z'),
  };
  expect(roundTrip(message)).toMatchObject({
    tool_call_id: 'current',
    status: 'error',
    artifact: message.artifact,
  });
});

it('preserves the role of generic chat messages', () => {
  const message = new ChatMessage({ role: 'custom', content: 'message' });
  expect(roundTrip(message)).toMatchObject({
    role: 'custom',
    content: 'message',
  });
});

it('preserves AI chunk fields for a chunk-aware transport', () => {
  const message = new AIMessageChunk({
    content: '',
    tool_call_chunks: [{ name: 'inspect', id: 'call', args: '{', index: 0 }],
    usage_metadata: { input_tokens: 7, output_tokens: 0, total_tokens: 7 },
  });
  const stored = toStoredMessage(message);
  expect(stored.data).toMatchObject({
    tool_call_chunks: message.tool_call_chunks,
    usage_metadata: message.usage_metadata,
  });
});

it('does not silently erase unsupported values before the bounded codec can reject them', () => {
  const message = new HumanMessage({
    content: 'hello',
    additional_kwargs: { closure: () => 'private' },
  });
  expect(() => serialize(toStoredMessage(message))).toThrow();
});

it('fails closed for custom secret-bearing message serializers', () => {
  class SecretMessage extends HumanMessage {
    override get lc_secrets() {
      return { 'additional_kwargs.token': 'TOKEN' };
    }
  }
  const message = new SecretMessage({
    content: 'hello', additional_kwargs: { token: 'PRIVATE' },
  });
  expect(() => toStoredMessage(message)).toThrow('secrets require an explicit codec');
});
