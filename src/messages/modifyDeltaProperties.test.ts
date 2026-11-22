import { AIMessageChunk } from '@langchain/core/messages';
import type { MessageContentComplex } from '@/types';
import { toLangChainContent } from './langchain';
import { modifyDeltaProperties } from './core';
import { Providers } from '@/common';

const serverToolResultTypes = [
  'web_search_tool_result',
  'code_execution_tool_result',
  'web_fetch_tool_result',
  'bash_code_execution_tool_result',
  'text_editor_code_execution_tool_result',
  'tool_search_tool_result',
];

const createChunk = (content: MessageContentComplex[]): AIMessageChunk =>
  new AIMessageChunk({ content: toLangChainContent(content) });

describe('modifyDeltaProperties', () => {
  describe('Anthropic server tool result blocks', () => {
    it('keeps a code_execution_tool_result block with encrypted content', () => {
      const result = {
        type: 'code_execution_tool_result',
        tool_use_id: 'srvtoolu_01ABC',
        content: {
          type: 'encrypted_code_execution_result',
          encrypted_stdout: 'EuYBCkQ=',
          stderr: '',
          return_code: 0,
          content: [],
        },
      };
      const chunk = createChunk([
        { type: 'text', text: 'Searching' },
        result as MessageContentComplex,
      ]);

      const modified = modifyDeltaProperties(Providers.ANTHROPIC, chunk);

      expect(modified?.content).toEqual([
        { type: 'text', text: 'Searching' },
        result,
      ]);
    });

    it.each(serverToolResultTypes)('keeps a %s block', (type) => {
      const block = {
        type,
        tool_use_id: 'srvtoolu_01ABC',
        content: [],
      } as MessageContentComplex;

      const modified = modifyDeltaProperties(
        Providers.ANTHROPIC,
        createChunk([block])
      );

      expect(modified?.content).toEqual([block]);
    });

    it('keeps the block in lc_kwargs.content too', () => {
      const block = {
        type: 'code_execution_tool_result',
        tool_use_id: 'srvtoolu_01ABC',
        content: { type: 'encrypted_code_execution_result' },
      } as MessageContentComplex;
      const chunk = createChunk([block]);

      const modified = modifyDeltaProperties(Providers.ANTHROPIC, chunk);

      expect(modified?.lc_kwargs.content).toEqual([block]);
    });

    it('still converts unknown block types to text', () => {
      const modified = modifyDeltaProperties(
        Providers.ANTHROPIC,
        createChunk([
          { type: 'unknown_block', tool_use_id: 'srvtoolu_01ABC' },
        ] as MessageContentComplex[])
      );

      expect(modified?.content).toEqual([
        { type: 'text', tool_use_id: 'srvtoolu_01ABC' },
      ]);
    });

    it('does not allow the server tool result types for other providers', () => {
      const modified = modifyDeltaProperties(
        Providers.OPENAI,
        createChunk([
          {
            type: 'code_execution_tool_result',
            tool_use_id: 'srvtoolu_01ABC',
          },
        ] as MessageContentComplex[])
      );

      expect(modified?.content).toEqual([
        { type: 'text', tool_use_id: 'srvtoolu_01ABC' },
      ]);
    });
  });
});
