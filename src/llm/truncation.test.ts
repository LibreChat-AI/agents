import { describe, expect, it } from '@jest/globals';
import { AIMessage, AIMessageChunk } from '@langchain/core/messages';
import {
  OutputTruncationError,
  assertNotTruncatedToolCall,
  getTruncationStopReason,
  getProviderStopReasons,
} from '@/llm/truncation';
import { Providers } from '@/common';

describe('getProviderStopReasons', () => {
  it.each([
    { stopReason: ' STOP_SEQUENCE ' },
    { messageStop: { stopReason: ' STOP_SEQUENCE ' } },
    { stop_reason: ' STOP_SEQUENCE ' },
    { finish_reason: ' STOP_SEQUENCE ' },
    { finishReason: ' STOP_SEQUENCE ' },
    { incomplete_details: { reason: ' STOP_SEQUENCE ' } },
  ])('normalizes native metadata: %j', (response_metadata) => {
    expect(
      getProviderStopReasons(new AIMessage({ content: '', response_metadata }))
    ).toEqual(['stop_sequence']);
  });

  it('reads Anthropic streamed stop reasons', () => {
    expect(
      getProviderStopReasons(
        new AIMessageChunk({
          content: '',
          additional_kwargs: { stop_reason: ' REFUSAL ' },
        })
      )
    ).toEqual(['refusal']);
  });

  it('does not let a normal reason hide a terminal reason in another field', () => {
    expect(
      getProviderStopReasons(
        new AIMessageChunk({
          content: '',
          response_metadata: {
            stopReason: 'end_turn',
            messageStop: { stopReason: 'stop_sequence' },
          },
        })
      )
    ).toEqual(['end_turn', 'stop_sequence']);
  });

  it('ignores absent, empty, and non-string fields', () => {
    expect(getProviderStopReasons(undefined)).toEqual([]);
    expect(getProviderStopReasons(null)).toEqual([]);
    expect(
      getProviderStopReasons(
        new AIMessage({
          content: '',
          response_metadata: { stopReason: 1, finishReason: ' ' },
        })
      )
    ).toEqual([]);
  });
});

describe('getTruncationStopReason', () => {
  it('returns null when there is no response metadata', () => {
    expect(getTruncationStopReason(new AIMessage('hi'))).toBeNull();
    expect(getTruncationStopReason(undefined)).toBeNull();
  });

  it('detects Bedrock Converse streaming shape (messageStop.stopReason)', () => {
    const message = new AIMessageChunk({
      content: '',
      response_metadata: { messageStop: { stopReason: 'max_tokens' } },
    });
    expect(getTruncationStopReason(message)).toBe('max_tokens');
  });

  it('detects Bedrock Converse non-streaming shape (stopReason)', () => {
    const message = new AIMessage({
      content: '',
      response_metadata: { stopReason: 'max_tokens' },
    });
    expect(getTruncationStopReason(message)).toBe('max_tokens');
  });

  it('detects Anthropic shape (stop_reason)', () => {
    const message = new AIMessage({
      content: '',
      response_metadata: { stop_reason: 'max_tokens' },
    });
    expect(getTruncationStopReason(message)).toBe('max_tokens');
  });

  it('detects OpenAI shape (finish_reason: length)', () => {
    const message = new AIMessage({
      content: '',
      response_metadata: { finish_reason: 'length' },
    });
    expect(getTruncationStopReason(message)).toBe('length');
  });

  it('detects Google shape (finishReason: MAX_TOKENS, case-insensitive)', () => {
    const message = new AIMessage({
      content: '',
      response_metadata: { finishReason: 'MAX_TOKENS' },
    });
    expect(getTruncationStopReason(message)).toBe('max_tokens');
  });

  it('detects Anthropic streaming shape (additional_kwargs.stop_reason)', () => {
    const message = new AIMessageChunk({
      content: '',
      additional_kwargs: { stop_reason: 'max_tokens' },
      response_metadata: { model_provider: 'anthropic' },
    });
    expect(getTruncationStopReason(message)).toBe('max_tokens');
  });

  it('detects OpenAI Responses shape (incomplete_details.reason)', () => {
    const message = new AIMessage({
      content: '',
      response_metadata: {
        status: 'incomplete',
        incomplete_details: { reason: 'max_output_tokens' },
      },
    });
    expect(getTruncationStopReason(message)).toBe('max_tokens');
  });

  it('returns null for a non-token incomplete reason (content_filter)', () => {
    const message = new AIMessage({
      content: '',
      response_metadata: {
        status: 'incomplete',
        incomplete_details: { reason: 'content_filter' },
      },
    });
    expect(getTruncationStopReason(message)).toBeNull();
  });

  it('returns null for normal stop reasons', () => {
    expect(
      getTruncationStopReason(
        new AIMessage({
          content: 'x',
          response_metadata: { stopReason: 'end_turn' },
        })
      )
    ).toBeNull();
    expect(
      getTruncationStopReason(
        new AIMessage({
          content: 'x',
          response_metadata: { finish_reason: 'stop' },
        })
      )
    ).toBeNull();
    expect(
      getTruncationStopReason(
        new AIMessage({
          content: 'x',
          response_metadata: { stopReason: 'tool_use' },
        })
      )
    ).toBeNull();
  });
});

describe('assertNotTruncatedToolCall', () => {
  it('no-ops for a normal completion', () => {
    expect(() =>
      assertNotTruncatedToolCall(
        new AIMessage({
          content: 'done',
          response_metadata: { stopReason: 'end_turn' },
        })
      )
    ).not.toThrow();
  });

  it('no-ops for a truncated plain-text turn (no tool calls)', () => {
    expect(() =>
      assertNotTruncatedToolCall(
        new AIMessage({
          content: 'partial...',
          response_metadata: { stopReason: 'max_tokens' },
        })
      )
    ).not.toThrow();
  });

  it('no-ops for a complete tool call (normal stop reason)', () => {
    const message = new AIMessage({
      content: '',
      tool_calls: [
        { id: '1', name: 'create_file', args: { path: 'a', content: 'b' } },
      ],
      response_metadata: { stopReason: 'tool_use' },
    });
    expect(() => assertNotTruncatedToolCall(message)).not.toThrow();
  });

  it('throws when a tool call is present and the turn was truncated', () => {
    const message = new AIMessage({
      content: '',
      tool_calls: [{ id: '1', name: 'create_file', args: { path: 'a' } }],
      response_metadata: { messageStop: { stopReason: 'max_tokens' } },
    });
    expect(() => assertNotTruncatedToolCall(message)).toThrow(
      OutputTruncationError
    );
    try {
      assertNotTruncatedToolCall(message);
    } catch (err) {
      expect(err).toBeInstanceOf(OutputTruncationError);
      expect((err as OutputTruncationError).stopReason).toBe('max_tokens');
      expect((err as OutputTruncationError).toolCallNames).toContain(
        'create_file'
      );
      expect((err as OutputTruncationError).message).toMatch(
        /max output tokens/i
      );
    }
  });

  it('throws when only incomplete tool_call_chunks survived truncation', () => {
    const message = new AIMessageChunk({
      content: '',
      tool_call_chunks: [
        {
          name: 'create_file',
          args: '{"path":"a"',
          index: 0,
          type: 'tool_call_chunk',
        },
      ],
      response_metadata: { messageStop: { stopReason: 'max_tokens' } },
    });
    expect(() => assertNotTruncatedToolCall(message)).toThrow(
      OutputTruncationError
    );
  });

  it('still throws for a streaming-arg provider (Anthropic)', () => {
    const message = new AIMessage({
      content: '',
      tool_calls: [{ id: '1', name: 'create_file', args: { path: 'a' } }],
      response_metadata: { stop_reason: 'max_tokens' },
    });
    expect(() =>
      assertNotTruncatedToolCall(message, Providers.ANTHROPIC)
    ).toThrow(OutputTruncationError);
  });

  it('throws for an Anthropic streaming tool call truncated via additional_kwargs', () => {
    const message = new AIMessageChunk({
      content: '',
      tool_calls: [{ id: '1', name: 'create_file', args: { path: 'a' } }],
      additional_kwargs: { stop_reason: 'max_tokens' },
      response_metadata: { model_provider: 'anthropic' },
    });
    expect(() =>
      assertNotTruncatedToolCall(message, Providers.ANTHROPIC)
    ).toThrow(OutputTruncationError);
  });

  it('throws for an OpenAI Responses tool call truncated via incomplete_details', () => {
    const message = new AIMessage({
      content: '',
      tool_calls: [{ id: '1', name: 'create_file', args: { path: 'a' } }],
      response_metadata: {
        status: 'incomplete',
        incomplete_details: { reason: 'max_output_tokens' },
      },
    });
    expect(() => assertNotTruncatedToolCall(message, Providers.OPENAI)).toThrow(
      OutputTruncationError
    );
  });

  it('does not throw for providers that deliver complete tool calls (Google/Vertex)', () => {
    const message = new AIMessage({
      content: '',
      tool_calls: [
        { id: '1', name: 'create_file', args: { path: 'a', content: 'b' } },
      ],
      response_metadata: { finishReason: 'MAX_TOKENS' },
    });
    expect(() =>
      assertNotTruncatedToolCall(message, Providers.GOOGLE)
    ).not.toThrow();
    expect(() =>
      assertNotTruncatedToolCall(message, Providers.VERTEXAI)
    ).not.toThrow();
  });
});
