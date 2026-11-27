import { AIMessage } from '@langchain/core/messages';
import { describe, it, expect } from '@jest/globals';
import { toolsCondition } from '../ToolNode';

/**
 * Anthropic server tools (`srvtoolu_`) run inside the provider response. The
 * final AI message keeps the call in `tool_calls` and holds the matching
 * `*_tool_result` block in `content`. Routing such a settled call to the tool
 * node runs nothing and makes the graph call the model again, which writes the
 * answer a second time.
 */

const END = '__end__';

const codeExecutionCall = {
  id: 'srvtoolu_A',
  name: 'code_execution',
  args: { code: 'print(1)' },
};
const webSearchCall = {
  id: 'srvtoolu_W',
  name: 'web_search',
  args: { query: 'x' },
};

const codeExecutionBlocks = [
  {
    type: 'server_tool_use',
    id: 'srvtoolu_A',
    name: 'code_execution',
    input: { code: 'print(1)' },
  },
  {
    type: 'code_execution_tool_result',
    tool_use_id: 'srvtoolu_A',
    content: { type: 'code_execution_result', stdout: '1\n', return_code: 0 },
  },
];
const answerBlock = { type: 'text', text: 'The answer is 1.' };

function route(message: AIMessage, invoked?: Set<string>): string {
  return toolsCondition({ messages: [message] }, 'tools', invoked);
}

describe('toolsCondition with settled Anthropic server tool calls', () => {
  it('ends when code_execution has its result block and nested web_search is invoked', () => {
    const message = new AIMessage({
      content: [
        {
          type: 'web_search_tool_result',
          tool_use_id: 'srvtoolu_W',
          content: [{ type: 'web_search_result', url: 'https://a.test' }],
        },
        ...codeExecutionBlocks,
        answerBlock,
      ],
      tool_calls: [webSearchCall, codeExecutionCall],
      response_metadata: { stop_reason: 'end_turn' },
    });
    expect(route(message, new Set(['srvtoolu_W']))).toBe(END);
  });

  it('ends for a settled code_execution call with empty or undefined invokedToolIds', () => {
    const message = new AIMessage({
      content: [...codeExecutionBlocks, answerBlock],
      tool_calls: [codeExecutionCall],
      response_metadata: { stop_reason: 'end_turn' },
    });
    expect(route(message, new Set())).toBe(END);
    expect(route(message, undefined)).toBe(END);
    expect(route(message)).toBe(END);
  });

  it('routes to the tool node when code_execution has no result block', () => {
    const message = new AIMessage({
      content: [codeExecutionBlocks[0], answerBlock],
      tool_calls: [codeExecutionCall],
    });
    expect(route(message, new Set())).toBe('tools');
    expect(route(message)).toBe('tools');
  });

  it('routes to the tool node until the client call next to a settled server call is invoked', () => {
    const message = new AIMessage({
      content: [
        ...codeExecutionBlocks,
        {
          type: 'tool_use',
          id: 'toolu_B',
          name: 'echo',
          input: { command: 'hi' },
        },
      ],
      tool_calls: [
        codeExecutionCall,
        { id: 'toolu_B', name: 'echo', args: { command: 'hi' } },
      ],
      response_metadata: { stop_reason: 'tool_use' },
    });
    expect(route(message, new Set())).toBe('tools');
    expect(route(message, new Set(['srvtoolu_A']))).toBe('tools');
    expect(route(message, new Set(['toolu_B']))).toBe(END);
  });

  it('routes to the tool node on pause_turn (response_metadata)', () => {
    const message = new AIMessage({
      content: codeExecutionBlocks,
      tool_calls: [codeExecutionCall],
      response_metadata: { stop_reason: 'pause_turn' },
    });
    expect(route(message, new Set())).toBe('tools');
  });

  it('routes to the tool node on pause_turn (additional_kwargs)', () => {
    const message = new AIMessage({
      content: codeExecutionBlocks,
      tool_calls: [codeExecutionCall],
      additional_kwargs: { stop_reason: 'pause_turn' },
    });
    expect(route(message, new Set())).toBe('tools');
  });

  it('ends for a web_search with an error result that is not in invokedToolIds', () => {
    const message = new AIMessage({
      content: [
        {
          type: 'web_search_tool_result',
          tool_use_id: 'srvtoolu_W',
          content: {
            type: 'web_search_tool_result_error',
            error_code: 'unavailable',
          },
        },
        answerBlock,
      ],
      tool_calls: [webSearchCall],
      response_metadata: { stop_reason: 'end_turn' },
    });
    expect(route(message, new Set())).toBe(END);
    expect(route(message)).toBe(END);
  });

  it('does not let a client tool_result block settle a client call', () => {
    const message = new AIMessage({
      content: [
        {
          type: 'tool_use',
          id: 'toolu_B',
          name: 'echo',
          input: { command: 'hi' },
        },
        { type: 'tool_result', tool_use_id: 'toolu_B', content: 'done' },
      ],
      tool_calls: [{ id: 'toolu_B', name: 'echo', args: { command: 'hi' } }],
    });
    expect(route(message, new Set())).toBe('tools');
  });

  it('does not let a result block of another server call settle this call', () => {
    const message = new AIMessage({
      content: [
        {
          type: 'code_execution_tool_result',
          tool_use_id: 'srvtoolu_OTHER',
          content: { type: 'code_execution_result', stdout: '' },
        },
      ],
      tool_calls: [codeExecutionCall],
    });
    expect(route(message, new Set())).toBe('tools');
  });

  it('keeps the previous behavior when there are no settled result blocks', () => {
    const message = new AIMessage({
      content: 'calling',
      tool_calls: [{ id: 'toolu_B', name: 'echo', args: { command: 'hi' } }],
    });
    expect(route(message, new Set(['toolu_B']))).toBe(END);
    expect(route(message, new Set(['toolu_other']))).toBe('tools');
    expect(route(message, new Set())).toBe('tools');
    expect(route(message)).toBe('tools');

    const noCalls = new AIMessage({ content: 'plain answer' });
    expect(route(noCalls)).toBe(END);
  });
});
