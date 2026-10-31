import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import {
  _convertLangChainToolCallToAnthropic,
  _convertMessagesToAnthropicPayload,
  normalizeAnthropicToolCallId,
} from './message_inputs';

describe('normalizeAnthropicToolCallId', () => {
  it('returns valid IDs unchanged', () => {
    expect(normalizeAnthropicToolCallId('toolu_01ABcdEFgh')).toBe(
      'toolu_01ABcdEFgh'
    );
    expect(normalizeAnthropicToolCallId('call_abc123XYZ')).toBe(
      'call_abc123XYZ'
    );
    expect(normalizeAnthropicToolCallId('a-b_c-d')).toBe('a-b_c-d');
  });

  it('sanitizes invalid characters and appends a hash suffix', () => {
    const out = normalizeAnthropicToolCallId(
      'fc_67abc1234def567|call_abc123def456ghi789jkl0mnopqrs'
    );
    expect(/^[a-zA-Z0-9_-]+$/.test(out)).toBe(true);
    expect(out.length).toBeLessThanOrEqual(64);
    expect(
      out.startsWith('fc_67abc1234def567_call_abc123def456ghi789jkl0mn')
    ).toBe(true);
    // Suffix is `_<10-hex-char hash>`
    expect(out).toMatch(/_[0-9a-f]{10}$/);
  });

  it('produces compliant output for IDs of any length', () => {
    const long = 'fc_' + 'a'.repeat(80);
    const out = normalizeAnthropicToolCallId(long);
    expect(out).toHaveLength(64);
    expect(/^[a-zA-Z0-9_-]+$/.test(out)).toBe(true);
  });

  it('produces uniquely distinguishable outputs for IDs that share a 64-char prefix', () => {
    const sharedPrefix = 'fc_' + 'a'.repeat(80);
    const idA = sharedPrefix + '|call_unique_A';
    const idB = sharedPrefix + '|call_unique_B';

    const outA = normalizeAnthropicToolCallId(idA);
    const outB = normalizeAnthropicToolCallId(idB);

    expect(outA).not.toBe(outB);
    expect(outA).toHaveLength(64);
    expect(outB).toHaveLength(64);
    expect(/^[a-zA-Z0-9_-]+$/.test(outA)).toBe(true);
    expect(/^[a-zA-Z0-9_-]+$/.test(outB)).toBe(true);
  });

  it('disambiguates short IDs that sanitize to the same value', () => {
    expect(normalizeAnthropicToolCallId('a|b')).not.toBe(
      normalizeAnthropicToolCallId('a.b')
    );
  });

  it('handles combined length and character violations', () => {
    const id = 'fc_' + 'x|'.repeat(100);
    const out = normalizeAnthropicToolCallId(id);
    expect(out).toHaveLength(64);
    expect(/^[a-zA-Z0-9_-]+$/.test(out)).toBe(true);
  });

  it('is deterministic — same input always yields same output', () => {
    const id = 'fc_a|b|c';
    expect(normalizeAnthropicToolCallId(id)).toBe(
      normalizeAnthropicToolCallId(id)
    );
  });

  it('passes through undefined for the optional overload', () => {
    expect(normalizeAnthropicToolCallId(undefined)).toBeUndefined();
  });

  it('handles empty string by producing a deterministic compliant output', () => {
    const out = normalizeAnthropicToolCallId('');
    expect(/^[a-zA-Z0-9_-]+$/.test(out)).toBe(true);
    expect(out.length).toBeLessThanOrEqual(64);
    expect(out).toBe(normalizeAnthropicToolCallId(''));
  });
});

describe('_convertMessagesToAnthropicPayload — cross-provider ID normalization', () => {
  it('flattens an Anthropic-native tool_result returned by a tool', () => {
    const toolCallId = 'toolu_native_result';
    const payload = _convertMessagesToAnthropicPayload([
      new HumanMessage('run the native tool'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: toolCallId,
            name: 'native_tool',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        tool_call_id: toolCallId,
        content: [
          {
            type: 'tool_result',
            tool_use_id: toolCallId,
            cache_control: { type: 'ephemeral', ttl: '1h' },
            is_error: true,
            content: [
              {
                type: 'text',
                text: 'native result',
                cache_control: { type: 'ephemeral' },
              },
              {
                type: 'image_url',
                image_url: 'data:image/png;base64,AA==',
              },
            ],
          },
        ],
      }),
    ]);

    const result = (
      payload.messages.find(
        (message) =>
          message.role === 'user' &&
          Array.isArray(message.content) &&
          message.content.some((block) => block.type === 'tool_result')
      )!.content as unknown as Array<Record<string, unknown>>
    ).find((block) => block.type === 'tool_result')!;

    expect(result).toEqual({
      type: 'tool_result',
      tool_use_id: toolCallId,
      cache_control: { type: 'ephemeral', ttl: '1h' },
      is_error: true,
      content: [
        { type: 'text', text: 'native result' },
        {
          type: 'image',
          source: {
            type: 'base64',
            media_type: 'image/png',
            data: 'AA==',
          },
        },
      ],
    });
    expect(
      (result.content as unknown as Array<Record<string, unknown>>).some(
        (block) => block.type === 'tool_result'
      )
    ).toBe(false);
  });

  it('flattens a content-less native error tool_result', () => {
    const toolCallId = 'toolu_empty_error';
    const payload = _convertMessagesToAnthropicPayload([
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: toolCallId,
            name: 'native_tool',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        tool_call_id: toolCallId,
        content: [
          {
            type: 'tool_result',
            tool_use_id: toolCallId,
            is_error: true,
          },
        ],
      }),
    ]);

    const result = (
      payload.messages.find((message) => message.role === 'user')!
        .content as unknown as Array<Record<string, unknown>>
    )[0];
    expect(result).toEqual({
      type: 'tool_result',
      tool_use_id: toolCallId,
      is_error: true,
    });
  });

  it('normalizes Responses-style IDs on tool_use AND matching tool_result', () => {
    const responsesId = 'fc_67abc1234def567|call_abc123def456ghi789jkl0mnopqrs';

    const payload = _convertMessagesToAnthropicPayload([
      new HumanMessage('weather?'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: responsesId,
            name: 'get_weather',
            args: { location: 'Tokyo' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        tool_call_id: responsesId,
        content: '{"temp": 21}',
      }),
    ]);

    const assistantMsg = payload.messages.find((m) => m.role === 'assistant')!;
    const userToolResultMsg = payload.messages.find(
      (m) =>
        m.role === 'user' &&
        Array.isArray(m.content) &&
        (m.content as Array<{ type: string }>)[0]?.type === 'tool_result'
    )!;

    const toolUseBlock = (
      assistantMsg.content as Array<{ type: string; id?: string }>
    ).find((b) => b.type === 'tool_use')!;
    const toolResultBlock = (
      userToolResultMsg.content as Array<{
        type: string;
        tool_use_id?: string;
      }>
    ).find((b) => b.type === 'tool_result')!;

    const expected = normalizeAnthropicToolCallId(responsesId);
    expect(toolUseBlock.id).toBe(expected);
    expect(toolResultBlock.tool_use_id).toBe(expected);
    expect(toolUseBlock.id).toBe(toolResultBlock.tool_use_id);
    expect(/^[a-zA-Z0-9_-]+$/.test(toolUseBlock.id!)).toBe(true);
    expect(toolUseBlock.id!.length).toBeLessThanOrEqual(64);
  });

  it('passes through Anthropic-native IDs unchanged', () => {
    const nativeId = 'toolu_01ABcdEFgh23ijKL';

    const payload = _convertMessagesToAnthropicPayload([
      new HumanMessage('hi'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: nativeId,
            name: 'noop',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        tool_call_id: nativeId,
        content: 'ok',
      }),
    ]);

    const assistantMsg = payload.messages.find((m) => m.role === 'assistant')!;
    const toolUseBlock = (
      assistantMsg.content as Array<{ type: string; id?: string }>
    ).find((b) => b.type === 'tool_use')!;

    expect(toolUseBlock.id).toBe(nativeId);
  });

  it('does not normalize server tool IDs (srvtoolu_ prefix)', () => {
    const serverId = 'srvtoolu_01abcXYZ';

    const block = _convertLangChainToolCallToAnthropic({
      id: serverId,
      name: 'web_search',
      args: { query: 'x' },
      type: 'tool_call',
    });

    expect(block.type).toBe('server_tool_use');
    expect(block.id).toBe(serverId);
  });

  it('keeps a tool result the same string once the tail marker has moved on', () => {
    const toolCallId = 'toolu_01ABcdEFgh';
    const resultOf = (content: ToolMessage['content']): unknown => {
      const payload = _convertMessagesToAnthropicPayload([
        new AIMessage({
          content: '',
          tool_calls: [
            { id: toolCallId, name: 'search', args: {}, type: 'tool_call' },
          ],
        }),
        new ToolMessage({ tool_call_id: toolCallId, content }),
      ]);
      const blocks = payload.messages[payload.messages.length - 1]
        .content as unknown as Array<Record<string, unknown>>;
      return blocks.find((block) => block.type === 'tool_result')!.content;
    };

    expect(resultOf('result body')).toBe('result body');
    expect(
      resultOf([
        {
          type: 'text',
          text: 'result body',
          cache_control: { type: 'ephemeral', ttl: '1h' },
        },
      ])
    ).toBe('result body');
  });
});
