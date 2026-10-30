import { z } from 'zod';
import { z as z4 } from 'zod/v4';
import { z as currentZod } from 'zod-v4';
import { tool } from '@langchain/core/tools';
import { SystemMessage } from '@langchain/core/messages';
import { FakeChatModel } from '@langchain/core/utils/testing';
import { convertToOpenAIFunction } from '@langchain/core/utils/function_calling';
import type { GraphTools, TokenCounter } from '@/types';
import {
  ANTHROPIC_TOOL_TOKEN_MULTIPLIER,
  DEFAULT_TOOL_TOKEN_MULTIPLIER,
} from '@/common/constants';
import { partitionAndMarkAnthropicToolCache } from '@/messages/anthropicToolCache';
import { countSummarizationToolTokens } from '../toolBudget';
import { registerProvider } from '@/llm/providerRegistry';
import { createTokenCounter } from '@/utils/tokens';
import { toJsonSchema } from '@/utils/schema';
import { Providers } from '@/common/enum';

const description = '搜索知识库中的中文资料，返回相关文档及其来源。'.repeat(20);
const parameters = {
  type: 'object' as const,
  properties: { query: { type: 'string' as const, description: '查询条件' } },
  required: ['query'],
};
const openAITool = {
  type: 'function',
  function: { name: 'search', description, parameters, strict: false },
};

let openAICount: TokenCounter;
let claudeCount: TokenCounter;

beforeAll(async () => {
  [openAICount, claudeCount] = await Promise.all([
    createTokenCounter('o200k_base'),
    createTokenCounter('claude'),
  ]);
});

describe('countSummarizationToolTokens', () => {
  it.each([undefined, []])(
    'returns zero for an empty inventory: %s',
    (tools) => {
      const count = jest.fn(openAICount);
      expect(
        countSummarizationToolTokens({
          tools,
          provider: Providers.OPENAI,
          count,
        })
      ).toBe(0);
      expect(count).not.toHaveBeenCalled();
    }
  );

  it.each([
    ['Zod 3', z.object({ query: z.string().describe('查询条件') })],
    ['early Zod 4', z4.object({ query: z4.string().describe('查询条件') })],
    [
      'current Zod 4',
      currentZod.object({ query: currentZod.string().describe('查询条件') }),
    ],
    ['JSON schema', parameters],
  ])(
    'converts %s without serializing tool implementation state',
    (_version, schema) => {
      const boundTool = tool(async () => 'result', {
        name: 'search',
        description,
        schema,
      });
      const count = jest.fn(openAICount);
      const expected = {
        type: 'function',
        function: convertToOpenAIFunction({
          name: 'search',
          description,
          schema: toJsonSchema(schema),
        }),
      };
      expect(
        countSummarizationToolTokens({
          tools: [boundTool],
          provider: Providers.OPENAI,
          count,
        })
      ).toBe(
        Math.ceil(
          openAICount(new SystemMessage(JSON.stringify(expected))) *
            DEFAULT_TOOL_TOKEN_MULTIPLIER
        )
      );
      expect(count).toHaveBeenCalledTimes(1);
      expect(count.mock.calls[0][0].content).toBe(JSON.stringify(expected));
      expect(JSON.stringify(expected.function.parameters)).toContain(
        '查询条件'
      );
      expect(JSON.stringify(expected.function.parameters)).toContain('query');
      expect(boundTool.schema).toBe(schema);
    }
  );

  it('uses the serving tokenizer and multiplier for a CJK bound inventory', () => {
    const allTools = Array.from({ length: 8 }, (_, index) => ({
      type: 'function',
      function: { ...openAITool.function, name: `search_${index}` },
    }));
    const tools = allTools.slice(0, 3);
    const rawCount = (count: TokenCounter): number =>
      tools.reduce(
        (total, entry) =>
          total + count(new SystemMessage(JSON.stringify(entry))),
        0
      );
    const openAI = countSummarizationToolTokens({
      tools,
      provider: Providers.OPENAI,
      count: openAICount,
    });
    const claude = countSummarizationToolTokens({
      tools,
      provider: Providers.ANTHROPIC,
      count: claudeCount,
    });
    expect(openAI).toBe(
      Math.ceil(rawCount(openAICount) * DEFAULT_TOOL_TOKEN_MULTIPLIER)
    );
    expect(claude).toBe(
      Math.ceil(rawCount(claudeCount) * ANTHROPIC_TOOL_TOKEN_MULTIPLIER)
    );
    expect(rawCount(claudeCount)).toBeGreaterThan(rawCount(openAICount));
    expect(claude).toBeGreaterThan(openAI);
    expect(
      countSummarizationToolTokens({
        tools: allTools,
        provider: Providers.OPENAI,
        count: openAICount,
      })
    ).toBeGreaterThan(openAI);
  });

  it.each([
    ['OpenAI functions', [openAITool]],
    [
      'Anthropic native',
      [
        {
          name: 'search',
          description,
          input_schema: parameters,
          cache_control: { type: 'ephemeral' },
          defer_loading: true,
        },
      ],
    ],
    [
      'Google declarations',
      [
        {
          functionDeclarations: [
            { name: 'search', description, parameters },
            { name: 'lookup', parameters },
          ],
        },
      ],
    ],
    ['Google server tools', [{ googleSearch: {} }, { codeExecution: {} }]],
    [
      'OpenAI built-ins',
      [
        { type: 'web_search_preview', search_context_size: 'high' },
        {
          type: 'mcp',
          server_label: 'knowledge',
          server_url: 'https://example.test/mcp',
        },
      ],
    ],
    [
      'Anthropic built-ins',
      [
        {
          type: 'web_search_20250305',
          name: 'web_search',
          max_uses: 4,
          cache_control: { type: 'ephemeral' },
        },
        {
          type: 'mcp_toolset',
          mcp_server_name: 'knowledge',
          default_config: { enabled: true },
        },
      ],
    ],
    [
      'Bedrock native',
      [
        {
          toolSpec: {
            name: 'search',
            description,
            inputSchema: { json: parameters },
          },
        },
        { cachePoint: { type: 'default' } },
      ],
    ],
  ])('counts every %s payload exactly once', (_name, inventory) => {
    const tools = inventory as GraphTools;
    const before = JSON.stringify(tools);
    const count = jest.fn(openAICount);
    const expected = tools.reduce(
      (total, entry) =>
        total + openAICount(new SystemMessage(JSON.stringify(entry))),
      0
    );
    expect(
      countSummarizationToolTokens({ tools, provider: Providers.OPENAI, count })
    ).toBe(Math.ceil(expected * DEFAULT_TOOL_TOKEN_MULTIPLIER));
    expect(count).toHaveBeenCalledTimes(tools.length);
    expect(count.mock.calls.map(([message]) => message.content)).toEqual(
      tools.map((entry) => JSON.stringify(entry))
    );
    expect(JSON.stringify(tools)).toBe(before);
  });

  it('keeps cache-control wrappers without counting the wrapped tool twice', () => {
    const boundTool = tool(async () => 'result', {
      name: 'search',
      description,
      schema: z.object({ query: z.string() }),
    });
    const tools = partitionAndMarkAnthropicToolCache([boundTool], () => false);
    const count = jest.fn(claudeCount);
    const before = JSON.stringify(tools);
    const expected = {
      name: boundTool.name,
      description,
      input_schema: toJsonSchema(boundTool.schema),
      cache_control: { type: 'ephemeral' },
    };
    expect(
      countSummarizationToolTokens({
        tools,
        provider: Providers.ANTHROPIC,
        count,
      })
    ).toBe(
      Math.ceil(
        claudeCount(new SystemMessage(JSON.stringify(expected))) *
          ANTHROPIC_TOOL_TOKEN_MULTIPLIER
      )
    );
    expect(count).toHaveBeenCalledTimes(1);
    expect(count.mock.calls[0][0].content).toBe(JSON.stringify(expected));
    expect(JSON.stringify(tools)).toBe(before);
    expect(boundTool.extras?.cache_control).toBeUndefined();
  });

  it('reads a structured schema getter only once', () => {
    const schema = jest.fn(() => parameters);
    const boundTool = {
      name: 'search',
      description,
      get schema() {
        return schema();
      },
    };
    countSummarizationToolTokens({
      tools: [boundTool],
      provider: Providers.OPENAI,
      count: openAICount,
    });
    expect(schema).toHaveBeenCalledTimes(1);
  });

  it('uses a provider tool definition without reading its unused schema getter', () => {
    const schema = jest.fn(() => parameters);
    const providerToolDefinition = {
      type: 'web_search_20250305',
      name: 'web_search',
      max_uses: 4,
    };
    const boundTool = {
      name: 'search',
      description,
      extras: { providerToolDefinition },
      get schema() {
        return schema();
      },
    };
    const count = jest.fn(claudeCount);
    countSummarizationToolTokens({
      tools: [boundTool],
      provider: Providers.ANTHROPIC,
      count,
    });
    expect(schema).not.toHaveBeenCalled();
    expect(count.mock.calls[0][0].content).toBe(
      JSON.stringify(providerToolDefinition)
    );
    expect(count).toHaveBeenCalledTimes(1);
  });

  it.each([
    [
      Providers.ANTHROPIC,
      { model: 'production' },
      ANTHROPIC_TOOL_TOKEN_MULTIPLIER,
    ],
    [
      Providers.OPENROUTER,
      { model: 'anthropic/claude-sonnet-4' },
      ANTHROPIC_TOOL_TOKEN_MULTIPLIER,
    ],
    [
      Providers.OPENROUTER,
      { modelName: 'anthropic/claude-sonnet-4' },
      ANTHROPIC_TOOL_TOKEN_MULTIPLIER,
    ],
    [
      Providers.OPENROUTER,
      { model: 'gpt-4.1', modelName: 'claude-sonnet-4' },
      DEFAULT_TOOL_TOKEN_MULTIPLIER,
    ],
    [
      Providers.BEDROCK,
      { model: 'anthropic.claude-sonnet-4' },
      DEFAULT_TOOL_TOKEN_MULTIPLIER,
    ],
  ])(
    'selects the serving multiplier for %s and %j',
    (provider, clientOptions, multiplier) => {
      const count = jest.fn(() => 11);
      expect(
        countSummarizationToolTokens({
          tools: [openAITool, openAITool],
          provider,
          clientOptions,
          count,
        })
      ).toBe(Math.ceil(22 * multiplier));
      expect(count).toHaveBeenCalledTimes(2);
    }
  );

  it.each(['anthropic', 'bedrock'] as const)(
    'honors a registered %s family behind an opaque model alias',
    (family) => {
      const provider = `tool-budget-${family}`;
      const dispose = registerProvider({
        provider,
        family,
        model: FakeChatModel,
      });
      try {
        const multiplier =
          family === 'anthropic'
            ? ANTHROPIC_TOOL_TOKEN_MULTIPLIER
            : DEFAULT_TOOL_TOKEN_MULTIPLIER;
        expect(
          countSummarizationToolTokens({
            tools: [openAITool],
            provider,
            clientOptions: { modelName: 'production' },
            count: () => 11,
          })
        ).toBe(Math.ceil(11 * multiplier));
      } finally {
        dispose();
      }
    }
  );

  it('does not mutate a frozen native inventory or cache marker', () => {
    const marker = Object.freeze({ type: 'ephemeral' });
    const entry = Object.freeze({ ...openAITool, cache_control: marker });
    const tools = [entry];
    Object.freeze(tools);
    expect(() =>
      countSummarizationToolTokens({
        tools,
        provider: Providers.OPENROUTER,
        count: openAICount,
      })
    ).not.toThrow();
    expect(entry.cache_control).toBe(marker);
    expect(entry.function).toBe(openAITool.function);
  });
});
