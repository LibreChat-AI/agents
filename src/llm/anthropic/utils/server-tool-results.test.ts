import { AIMessageChunk } from '@langchain/core/messages';
import type Anthropic from '@anthropic-ai/sdk';
import { getProviderToolCallPartDescriptor } from '@/messages/toolResultTypes';
import { _makeMessageChunkFromAnthropicEvent } from './message_outputs';
import { _convertMessagesToAnthropicPayload } from './message_inputs';
import { modifyDeltaProperties } from '@/messages/core';
import { Providers } from '@/common';

type StreamBlock = Extract<
  Anthropic.Beta.BetaRawMessageStreamEvent,
  { type: 'content_block_start' }
>['content_block'];
const tool_use_id = 'srvtoolu_example';
const caller = {
  type: 'code_execution_20260120',
  tool_id: 'srvtoolu_parent',
} as const;
const results: StreamBlock[] = [
  {
    type: 'advisor_tool_result',
    tool_use_id,
    content: { type: 'advisor_tool_result_error', error_code: 'unavailable' },
  },
  {
    type: 'bash_code_execution_tool_result',
    tool_use_id,
    content: {
      type: 'bash_code_execution_tool_result_error',
      error_code: 'unavailable',
    },
  },
  {
    type: 'code_execution_tool_result',
    tool_use_id,
    content: {
      type: 'code_execution_tool_result_error',
      error_code: 'unavailable',
    },
  },
  {
    type: 'text_editor_code_execution_tool_result',
    tool_use_id,
    content: {
      type: 'text_editor_code_execution_tool_result_error',
      error_code: 'unavailable',
      error_message: null,
    },
  },
  {
    type: 'tool_search_tool_result',
    tool_use_id,
    content: {
      type: 'tool_search_tool_result_error',
      error_code: 'unavailable',
      error_message: null,
    },
  },
  {
    type: 'web_fetch_tool_result',
    tool_use_id,
    caller,
    content: { type: 'web_fetch_tool_result_error', error_code: 'unavailable' },
  },
  {
    type: 'web_search_tool_result',
    tool_use_id,
    caller,
    content: {
      type: 'web_search_tool_result_error',
      error_code: 'unavailable',
    },
  },
];

it.each(results)(
  'keeps $type only in Anthropic normalization and strips stream fields on replay',
  (block) => {
    const parsed = _makeMessageChunkFromAnthropicEvent(
      { type: 'content_block_start', index: 0, content_block: block },
      { streamUsage: false, coerceContentToString: false }
    );
    expect(parsed?.chunk.tool_call_chunks).toEqual([]);
    const final = modifyDeltaProperties(Providers.ANTHROPIC, parsed?.chunk);
    expect(final?.content).toEqual([{ ...block, index: 0 }]);
    expect(final?.lc_kwargs.content).toEqual(final?.content);
    expect(
      _convertMessagesToAnthropicPayload(final == null ? [] : [final])
        .messages[0]?.content
    ).toEqual([block]);

    const foreign = modifyDeltaProperties(
      Providers.OPENAI,
      new AIMessageChunk({ content: [{ ...block }] })
    );
    expect(foreign?.content).toEqual([{ ...block, type: 'text' }]);
  }
);

it('only recognizes a validated caller on an Anthropic native server call', () => {
  const call = {
    type: 'server_tool_call',
    id: tool_use_id,
    name: 'web_search',
    args: {},
    caller,
  };
  expect(getProviderToolCallPartDescriptor(call)?.kind).toBe(
    'anthropic-server'
  );
  expect(
    getProviderToolCallPartDescriptor({ ...call, caller: { type: 'custom' } })
  ).toBeUndefined();
  expect(
    getProviderToolCallPartDescriptor({ ...call, id: 'other_server' })
  ).toBeUndefined();
});
