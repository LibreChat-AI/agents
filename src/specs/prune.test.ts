// src/specs/prune.test.ts
import { config } from 'dotenv';
config();
import {
  AIMessage,
  BaseMessage,
  ToolMessage,
  HumanMessage,
  isBaseMessage,
  SystemMessage,
  AIMessageChunk,
  ChatMessage,
} from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { UsageMetadata } from '@langchain/core/messages';
import type * as t from '@/types';
import {
  getMessagesWithinTokenLimit as realGetMessagesWithinTokenLimit,
  preFlightTruncateToolCallInputs,
  repairOrphanedToolMessages,
  sanitizeOrphanToolBlocks,
  enforceOriginalContentCap,
  ORIGINAL_CONTENT_MAX_CHARS,
  createPruneMessages,
  projectToolCallInputs,
  projectToolMessagesForProvider,
  TOOL_INPUT_ELISION_NOTE,
  serializeToolCallInput,
} from '@/messages/prune';
import { calculateMaxToolCallInputChars } from '@/utils/truncation';
import { projectToolStreamContentForProvider } from '@/messages/core';
import { getLLMConfig } from '@/utils/llmConfig';
import { ensureThinkingBlockInMessages } from '@/messages/format';
import { Providers, ContentTypes } from '@/common';
import { Run } from '@/run';

// Create a simple token counter for testing
const createTestTokenCounter = (): t.TokenCounter => {
  // This simple token counter just counts characters as tokens for predictable testing
  return (message: BaseMessage): number => {
    // Use type assertion to help TypeScript understand the type
    const content = message.content as
      | string
      | Array<t.MessageContentComplex | string>
      | undefined;

    // Handle string content
    if (typeof content === 'string') {
      return content.length;
    }

    // Handle array content
    if (Array.isArray(content)) {
      let totalLength = 0;

      for (const item of content) {
        if (typeof item === 'string') {
          totalLength += item.length;
        } else if (typeof item === 'object') {
          if ('text' in item && typeof item.text === 'string') {
            totalLength += item.text.length;
          }
          // Count tool_use input fields (serialized args contribute to token count)
          if ('input' in item && item.input != null) {
            const input = item.input;
            totalLength +=
              typeof input === 'string'
                ? input.length
                : JSON.stringify(input).length;
          }
        }
      }

      return totalLength;
    }

    // Default case - if content is null, undefined, or any other type
    return 0;
  };
};

// Since the internal functions in prune.ts are not exported, we'll reimplement them here for testing
// This is based on the implementation in src/messages/prune.ts
function calculateTotalTokens(usage: Partial<UsageMetadata>): UsageMetadata {
  const baseInputTokens = Number(usage.input_tokens) || 0;
  const cacheCreation = Number(usage.input_token_details?.cache_creation) || 0;
  const cacheRead = Number(usage.input_token_details?.cache_read) || 0;

  const totalInputTokens = baseInputTokens + cacheCreation + cacheRead;
  const totalOutputTokens = Number(usage.output_tokens) || 0;

  return {
    input_tokens: totalInputTokens,
    output_tokens: totalOutputTokens,
    total_tokens: totalInputTokens + totalOutputTokens,
  };
}

function getMessagesWithinTokenLimit({
  messages: _messages,
  maxContextTokens,
  indexTokenCountMap,
  startType,
}: {
  messages: BaseMessage[];
  maxContextTokens: number;
  indexTokenCountMap: Record<string, number>;
  startType?: string;
}): {
  context: BaseMessage[];
  remainingContextTokens: number;
  messagesToRefine: BaseMessage[];
  summaryIndex: number;
} {
  // Every reply is primed with <|start|>assistant<|message|>, so we
  // start with 3 tokens for the label after all messages have been counted.
  let currentTokenCount = 3;
  const instructions =
    _messages[0]?.getType() === 'system' ? _messages[0] : undefined;
  const instructionsTokenCount =
    instructions != null ? indexTokenCountMap[0] : 0;
  let remainingContextTokens = maxContextTokens - instructionsTokenCount;
  const messages = [..._messages];
  const context: BaseMessage[] = [];

  if (currentTokenCount < remainingContextTokens) {
    let currentIndex = messages.length;
    while (
      messages.length > 0 &&
      currentTokenCount < remainingContextTokens &&
      currentIndex > 1
    ) {
      currentIndex--;
      if (messages.length === 1 && instructions) {
        break;
      }
      const poppedMessage = messages.pop();
      if (!poppedMessage) continue;

      const tokenCount = indexTokenCountMap[currentIndex] || 0;

      if (currentTokenCount + tokenCount <= remainingContextTokens) {
        context.push(poppedMessage);
        currentTokenCount += tokenCount;
      } else {
        messages.push(poppedMessage);
        break;
      }
    }

    // If startType is specified, discard messages until we find one of the required type
    if (startType != null && startType && context.length > 0) {
      const requiredTypeIndex = context.findIndex(
        (msg) => msg.getType() === startType
      );

      if (requiredTypeIndex > 0) {
        // If we found a message of the required type, discard all messages before it
        const remainingMessages = context.slice(requiredTypeIndex);
        context.length = 0; // Clear the array
        context.push(...remainingMessages);
      }
    }
  }

  if (instructions && _messages.length > 0) {
    context.push(_messages[0] as BaseMessage);
    messages.shift();
  }

  const prunedMemory = messages;
  const summaryIndex = prunedMemory.length - 1;
  remainingContextTokens -= currentTokenCount;

  return {
    summaryIndex,
    remainingContextTokens,
    context: context.reverse(),
    messagesToRefine: prunedMemory,
  };
}

function checkValidNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value) && value > 0;
}

describe('Prune Messages Tests', () => {
  jest.setTimeout(30000);

  describe('calculateTotalTokens', () => {
    it('should calculate total tokens correctly with all fields present', () => {
      const usage: Partial<UsageMetadata> = {
        input_tokens: 100,
        output_tokens: 50,
        input_token_details: {
          cache_creation: 10,
          cache_read: 5,
        },
      };

      const result = calculateTotalTokens(usage);

      expect(result.input_tokens).toBe(115); // 100 + 10 + 5
      expect(result.output_tokens).toBe(50);
      expect(result.total_tokens).toBe(165); // 115 + 50
    });

    it('should handle missing fields gracefully', () => {
      const usage: Partial<UsageMetadata> = {
        input_tokens: 100,
        output_tokens: 50,
      };

      const result = calculateTotalTokens(usage);

      expect(result.input_tokens).toBe(100);
      expect(result.output_tokens).toBe(50);
      expect(result.total_tokens).toBe(150);
    });

    it('should handle empty usage object', () => {
      const usage: Partial<UsageMetadata> = {};

      const result = calculateTotalTokens(usage);

      expect(result.input_tokens).toBe(0);
      expect(result.output_tokens).toBe(0);
      expect(result.total_tokens).toBe(0);
    });
  });

  describe('getMessagesWithinTokenLimit', () => {
    it('should include all messages when under token limit', () => {
      const messages = [
        new SystemMessage('System instruction'),
        new HumanMessage('Hello'),
        new AIMessage('Hi there'),
      ];

      const indexTokenCountMap = {
        0: 17, // "System instruction"
        1: 5, // "Hello"
        2: 8, // "Hi there"
      };

      const result = getMessagesWithinTokenLimit({
        messages,
        maxContextTokens: 100,
        indexTokenCountMap,
      });

      expect(result.context.length).toBe(3);
      expect(result.context[0]).toBe(messages[0]); // System message
      expect(result.context[0].getType()).toBe('system'); // System message
      expect(result.remainingContextTokens).toBe(100 - 17 - 5 - 8 - 3); // -3 for the assistant label tokens
      expect(result.messagesToRefine.length).toBe(0);
    });

    it('should prune oldest messages when over token limit', () => {
      const messages = [
        new SystemMessage('System instruction'),
        new HumanMessage('Message 1'),
        new AIMessage('Response 1'),
        new HumanMessage('Message 2'),
        new AIMessage('Response 2'),
      ];

      const indexTokenCountMap = {
        0: 17, // "System instruction"
        1: 9, // "Message 1"
        2: 10, // "Response 1"
        3: 9, // "Message 2"
        4: 10, // "Response 2"
      };

      // Set a limit that can only fit the system message and the last two messages
      const result = getMessagesWithinTokenLimit({
        messages,
        maxContextTokens: 40,
        indexTokenCountMap,
      });

      // Should include system message and the last two messages
      expect(result.context.length).toBe(3);
      expect(result.context[0]).toBe(messages[0]); // System message
      expect(result.context[0].getType()).toBe('system'); // System message
      expect(result.context[1]).toBe(messages[3]); // Message 2
      expect(result.context[2]).toBe(messages[4]); // Response 2

      // Should have the first two messages in messagesToRefine
      expect(result.messagesToRefine.length).toBe(2);
      expect(result.messagesToRefine[0]).toBe(messages[1]); // Message 1
      expect(result.messagesToRefine[1]).toBe(messages[2]); // Response 1
    });

    it('should always include system message even when at token limit', () => {
      const messages = [
        new SystemMessage('System instruction'),
        new HumanMessage('Hello'),
        new AIMessage('Hi there'),
      ];

      const indexTokenCountMap = {
        0: 17, // "System instruction"
        1: 5, // "Hello"
        2: 8, // "Hi there"
      };

      // Set a limit that can only fit the system message
      const result = getMessagesWithinTokenLimit({
        messages,
        maxContextTokens: 20,
        indexTokenCountMap,
      });

      expect(result.context.length).toBe(1);
      expect(result.context[0]).toBe(messages[0]); // System message

      expect(result.messagesToRefine.length).toBe(2);
    });

    it('should start context with a specific message type when startType is specified', () => {
      const messages = [
        new SystemMessage('System instruction'),
        new AIMessage('AI message 1'),
        new HumanMessage('Human message 1'),
        new AIMessage('AI message 2'),
        new HumanMessage('Human message 2'),
      ];

      const indexTokenCountMap = {
        0: 17, // "System instruction"
        1: 12, // "AI message 1"
        2: 15, // "Human message 1"
        3: 12, // "AI message 2"
        4: 15, // "Human message 2"
      };

      // Set a limit that can fit all messages
      const result = getMessagesWithinTokenLimit({
        messages,
        maxContextTokens: 100,
        indexTokenCountMap,
        startType: 'human',
      });

      // All messages should be included since we're under the token limit
      expect(result.context.length).toBe(5);
      expect(result.context[0]).toBe(messages[0]); // System message
      expect(result.context[1]).toBe(messages[1]); // AI message 1
      expect(result.context[2]).toBe(messages[2]); // Human message 1
      expect(result.context[3]).toBe(messages[3]); // AI message 2
      expect(result.context[4]).toBe(messages[4]); // Human message 2

      // All messages should be included since we're under the token limit
      expect(result.messagesToRefine.length).toBe(0);
    });

    it('should keep all messages if no message of required type is found', () => {
      const messages = [
        new SystemMessage('System instruction'),
        new AIMessage('AI message 1'),
        new AIMessage('AI message 2'),
      ];

      const indexTokenCountMap = {
        0: 17, // "System instruction"
        1: 12, // "AI message 1"
        2: 12, // "AI message 2"
      };

      // Set a limit that can fit all messages
      const result = getMessagesWithinTokenLimit({
        messages,
        maxContextTokens: 100,
        indexTokenCountMap,
        startType: 'human',
      });

      // Should include all messages since no human messages exist to start from
      expect(result.context.length).toBe(3);
      expect(result.context[0]).toBe(messages[0]); // System message
      expect(result.context[1]).toBe(messages[1]); // AI message 1
      expect(result.context[2]).toBe(messages[2]); // AI message 2

      expect(result.messagesToRefine.length).toBe(0);
    });
  });

  describe('checkValidNumber', () => {
    it('should return true for valid positive numbers', () => {
      expect(checkValidNumber(5)).toBe(true);
      expect(checkValidNumber(1.5)).toBe(true);
      expect(checkValidNumber(Number.MAX_SAFE_INTEGER)).toBe(true);
    });

    it('should return false for zero, negative numbers, and NaN', () => {
      expect(checkValidNumber(0)).toBe(false);
      expect(checkValidNumber(-5)).toBe(false);
      expect(checkValidNumber(NaN)).toBe(false);
    });

    it('should return false for non-number types', () => {
      expect(checkValidNumber('5')).toBe(false);
      expect(checkValidNumber(null)).toBe(false);
      expect(checkValidNumber(undefined)).toBe(false);
      expect(checkValidNumber({})).toBe(false);
      expect(checkValidNumber([])).toBe(false);
    });
  });

  describe('createPruneMessages', () => {
    it('should return all messages when under token limit', () => {
      const tokenCounter = createTestTokenCounter();
      const messages = [
        new SystemMessage('System instruction'),
        new HumanMessage('Hello'),
        new AIMessage('Hi there'),
      ];

      const indexTokenCountMap = {
        0: tokenCounter(messages[0]),
        1: tokenCounter(messages[1]),
        2: tokenCounter(messages[2]),
      };

      const pruneMessages = createPruneMessages({
        maxTokens: 100,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap,
      });

      const result = pruneMessages({ messages });

      expect(result.context.length).toBe(3);
      expect(result.context).toEqual(messages);
      expect(result.messagesToRefine).toEqual([]);
      expect(result.remainingContextTokens).toBe(
        95 -
          Object.values(indexTokenCountMap).reduce(
            (total, count) => total + count,
            0
          ) -
          3
      );
    });

    it('recounts and compacts omitted structured tool results before sending', () => {
      const tokenCounter: t.TokenCounter = (message) => {
        const content =
          typeof message.content === 'string'
            ? message.content
            : JSON.stringify(message.content);
        return content.length;
      };
      const toolCallId = 'tc-structured';
      const messages: BaseMessage[] = [
        new HumanMessage('query the table'),
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: toolCallId,
              name: 'run_select_query',
              args: {},
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: [
            {
              type: ContentTypes.TEXT,
              text: JSON.stringify(
                Array.from({ length: 20 }, (_, index) => ({
                  id: index,
                  value: `${'x'.repeat(100)}-${index}`,
                }))
              ),
            },
          ],
          tool_call_id: toolCallId,
          name: 'run_select_query',
        }),
        new AIMessage('The query returned 20 rows.'),
        new HumanMessage('compact context'),
      ];
      const indexTokenCountMap: Record<string, number | undefined> = {
        0: tokenCounter(messages[0]),
        1: tokenCounter(messages[1]),
        3: tokenCounter(messages[3]),
        4: tokenCounter(messages[4]),
      };
      const pruneMessages = createPruneMessages({
        maxTokens: 2_000,
        startIndex: messages.length,
        tokenCounter,
        indexTokenCountMap,
        reserveRatio: 0,
      });

      const result = pruneMessages({ messages });

      const structuredResult = result.context.find(
        (message) => message.getType() === 'tool'
      );
      expect(structuredResult).toBeDefined();
      expect(typeof structuredResult?.content).toBe('string');
      expect(structuredResult?.content).toContain('truncated');
      expect(result.indexTokenCountMap[2]).toBeGreaterThan(0);
    });

    it('reconciles a stale cached legacy function-call count', () => {
      const tokenCounter: t.TokenCounter = (message) => {
        const functionCall = message.additional_kwargs.function_call;
        return (
          (typeof message.content === 'string' ? message.content.length : 0) +
          (functionCall != null ? JSON.stringify(functionCall).length : 0) +
          1
        );
      };
      const message = new AIMessage({
        content: '',
        additional_kwargs: {
          function_call: {
            name: 'legacy_lookup',
            arguments: `{"query":"${'x'.repeat(10_000)}"}`,
          },
        },
      });
      const messages: BaseMessage[] = [message];
      const pruneMessages = createPruneMessages({
        maxTokens: 2_000,
        startIndex: messages.length,
        tokenCounter,
        indexTokenCountMap: { 0: 1 },
        reserveRatio: 0,
      });

      const result = pruneMessages({ messages });
      const projectedFunctionCall = result.context[0].additional_kwargs
        .function_call as { arguments: string };

      expect(result.context[0]).not.toBe(message);
      expect(projectedFunctionCall.arguments.length).toBeLessThanOrEqual(
        calculateMaxToolCallInputChars(2_000)
      );
      expect(result.indexTokenCountMap[0]).toBe(
        tokenCounter(result.context[0])
      );
      expect(result.indexTokenCountMap[0]).toBeGreaterThan(1);
    });

    it('preserves provider output usage for an uncounted current legacy call', () => {
      const message = new AIMessage({
        content: '',
        additional_kwargs: {
          function_call: {
            name: 'legacy_lookup',
            arguments: '{}',
          },
        },
      });
      const pruneMessages = createPruneMessages({
        maxTokens: 2_000,
        startIndex: 0,
        tokenCounter: () => 123,
        indexTokenCountMap: {},
        reserveRatio: 0,
      });

      const result = pruneMessages({
        messages: [message],
        usageMetadata: {
          input_tokens: 10,
          output_tokens: 7,
          total_tokens: 17,
        },
      });

      expect(result.indexTokenCountMap[0]).toBe(7);
    });

    it('excludes a corrected tiny new tool count from calibration input', () => {
      const tokenCounter = createTestTokenCounter();
      const history: BaseMessage[] = [
        new HumanMessage('h'.repeat(50)),
        new AIMessage('a'.repeat(50)),
      ];
      const indexTokenCountMap: Record<string, number | undefined> = {
        0: 50,
        1: 50,
      };
      const pruneMessages = createPruneMessages({
        maxTokens: 10_000,
        startIndex: history.length,
        tokenCounter,
        indexTokenCountMap,
        reserveRatio: 0,
      });

      const first = pruneMessages({
        messages: history,
        usageMetadata: {
          input_tokens: 100,
          output_tokens: 10,
          total_tokens: 110,
        },
      });
      expect(first.calibrationRatio).toBe(1);

      const toolCallId = 'new-output-with-tiny-count';
      const secondMessages: BaseMessage[] = [
        ...history,
        new HumanMessage('q'.repeat(50)),
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: toolCallId,
              name: 'run_select_query',
              args: {},
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: 'r'.repeat(50),
          tool_call_id: toolCallId,
        }),
      ];
      indexTokenCountMap[2] = 50;
      indexTokenCountMap[4] = 1;

      const second = pruneMessages({
        messages: secondMessages,
        usageMetadata: {
          input_tokens: 100,
          output_tokens: 10,
          total_tokens: 110,
        },
      });

      expect(second.indexTokenCountMap[4]).toBe(50);
      expect(second.calibrationRatio).toBe(1);
    });

    it('preserves full structured output for summarization before masking', () => {
      const tokenCounter: t.TokenCounter = (message) => {
        const content =
          typeof message.content === 'string'
            ? message.content
            : JSON.stringify(message.content);
        return content.length;
      };
      const fullResult = JSON.stringify(
        Array.from({ length: 50 }, (_, index) => ({
          id: index,
          value: `${'x'.repeat(100)}-${index}`,
        }))
      );
      const messages: BaseMessage[] = [
        new HumanMessage('query the table'),
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'tc-summary-structured',
              name: 'run_select_query',
              args: {},
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: [{ type: ContentTypes.TEXT, text: fullResult }],
          tool_call_id: 'tc-summary-structured',
        }),
        new AIMessage('The query returned 50 rows.'),
        new HumanMessage('summarize the context'),
      ];
      const indexTokenCountMap: Record<string, number | undefined> = {};
      for (let i = 0; i < messages.length; i++) {
        indexTokenCountMap[i] = tokenCounter(messages[i]);
      }
      const pruneMessages = createPruneMessages({
        maxTokens: 500,
        startIndex: messages.length,
        tokenCounter,
        indexTokenCountMap,
        reserveRatio: 0,
        summarizationEnabled: true,
      });

      const result = pruneMessages({ messages });

      expect(result.newOriginalToolContent?.get(2)).toBe(fullResult);
      expect(messages[2].content).not.toEqual([
        { type: ContentTypes.TEXT, text: fullResult },
      ]);
    });

    it('should prune messages when over token limit', () => {
      const tokenCounter = createTestTokenCounter();
      const messages = [
        new SystemMessage('System instruction'),
        new HumanMessage('Message 1'),
        new AIMessage('Response 1'),
        new HumanMessage('Message 2'),
        new AIMessage('Response 2'),
      ];

      const indexTokenCountMap = {
        0: tokenCounter(messages[0]),
        1: tokenCounter(messages[1]),
        2: tokenCounter(messages[2]),
        3: tokenCounter(messages[3]),
        4: tokenCounter(messages[4]),
      };

      // Set a limit that can only fit the system message and the last two messages
      const pruneMessages = createPruneMessages({
        maxTokens: 40,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap,
        reserveRatio: 0,
      });

      const result = pruneMessages({ messages });

      // Should include system message and the last two messages
      expect(result.context.length).toBe(3);
      expect(result.context[0]).toBe(messages[0]); // System message
      expect(result.context[1]).toBe(messages[3]); // Message 2
      expect(result.context[2]).toBe(messages[4]); // Response 2
      expect(Array.isArray(result.messagesToRefine)).toBe(true);
      expect(result.messagesToRefine?.length).toBe(2);
      expect(typeof result.remainingContextTokens).toBe('number');
    });

    it('should return remaining tokens in calibrated units when pruning with calibration', () => {
      const tokenCounter = createTestTokenCounter();
      const messages = [
        new SystemMessage('System instruction'),
        new HumanMessage('Message 1'),
        new AIMessage('Response 1'),
        new HumanMessage('Message 2'),
        new AIMessage('Response 2'),
      ];

      const indexTokenCountMap = {
        0: tokenCounter(messages[0]),
        1: tokenCounter(messages[1]),
        2: tokenCounter(messages[2]),
        3: tokenCounter(messages[3]),
        4: tokenCounter(messages[4]),
      };

      const calibrationRatio = 2;
      const maxTokens = 80;
      const pruneMessages = createPruneMessages({
        maxTokens,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap,
        reserveRatio: 0,
        calibrationRatio,
      });

      const result = pruneMessages({ messages });

      expect(result.messagesToRefine?.length).toBeGreaterThan(0);

      /** Pruning selects within rawSpaceBudget = maxTokens / ratio (raw units,
       *  minus the 3-token assistant label); the returned remaining must be
       *  scaled back so `budget - remaining` reflects provider-space usage */
      const keptRaw = result.context.reduce(
        (sum, msg) => sum + tokenCounter(msg),
        0
      );
      const rawSpaceBudget = Math.round(maxTokens / calibrationRatio);
      const expectedRemaining =
        (rawSpaceBudget - keptRaw - 3) * calibrationRatio;
      expect(result.remainingContextTokens).toBe(expectedRemaining);
      expect(result.contextBudget).toBe(maxTokens);
    });

    it('should respect startType parameter', () => {
      const tokenCounter = createTestTokenCounter();
      const messages = [
        new SystemMessage('System instruction'),
        new AIMessage('AI message 1'),
        new HumanMessage('Human message 1'),
        new AIMessage('AI message 2'),
        new HumanMessage('Human message 2'),
      ];

      const indexTokenCountMap = {
        0: tokenCounter(messages[0]),
        1: tokenCounter(messages[1]),
        2: tokenCounter(messages[2]),
        3: tokenCounter(messages[3]),
        4: tokenCounter(messages[4]),
      };

      // Set a limit that can fit all messages
      const pruneMessages = createPruneMessages({
        maxTokens: 100,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap: { ...indexTokenCountMap },
      });

      const result = pruneMessages({
        messages,
        startType: 'human',
      });

      // All messages should be included since we're under the token limit
      expect(result.context.length).toBe(5);
      expect(result.context[0]).toBe(messages[0]); // System message
      expect(result.context[1]).toBe(messages[1]); // AI message 1
      expect(result.context[2]).toBe(messages[2]); // Human message 1
      expect(result.context[3]).toBe(messages[3]); // AI message 2
      expect(result.context[4]).toBe(messages[4]); // Human message 2
    });

    it('should update token counts when usage metadata is provided', () => {
      const tokenCounter = createTestTokenCounter();
      const messages = [
        new SystemMessage('System instruction'),
        new HumanMessage('Hello'),
        new AIMessage('Hi there'),
      ];

      const indexTokenCountMap = {
        0: tokenCounter(messages[0]),
        1: tokenCounter(messages[1]),
        2: tokenCounter(messages[2]),
      };

      const pruneMessages = createPruneMessages({
        maxTokens: 100,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap: { ...indexTokenCountMap },
      });

      // Provide usage metadata that indicates different token counts
      const usageMetadata: Partial<UsageMetadata> = {
        input_tokens: 50,
        output_tokens: 25,
        total_tokens: 75,
      };

      const result = pruneMessages({
        messages,
        usageMetadata,
      });

      // Map stays in raw tiktoken space — calibrationRatio captures the multiplier.
      // rawSum * calibrationRatio should approximate input_tokens (50).
      const rawSum = Object.values(result.indexTokenCountMap).reduce(
        (a = 0, b = 0) => a + b,
        0
      ) as number;
      const calibratedEstimate = Math.round(
        rawSum * (result.calibrationRatio ?? 1)
      );
      expect(Math.abs(calibratedEstimate - 50)).toBeLessThanOrEqual(3);
    });
  });

  describe('Tool Message Handling', () => {
    it('should drop orphan tool messages that no longer have matching AI tool calls', () => {
      const tokenCounter = createTestTokenCounter();
      const context = [
        new SystemMessage('System instruction'),
        new ToolMessage({
          content: 'Orphan result',
          tool_call_id: 'tool-orphan',
        }),
        new AIMessage({
          content: [
            { type: 'text', text: 'I will call a tool now' },
            {
              type: 'tool_use',
              id: 'tool-valid',
              name: 'read_file',
              input: '{"path":"README.md"}',
            },
          ],
        }),
        new ToolMessage({
          content: 'Valid result',
          tool_call_id: 'tool-valid',
        }),
      ];

      const indexTokenCountMap = {
        0: tokenCounter(context[0]),
        1: tokenCounter(context[1]),
        2: tokenCounter(context[2]),
        3: tokenCounter(context[3]),
      };

      const repaired = repairOrphanedToolMessages({
        context,
        allMessages: context,
        tokenCounter,
        indexTokenCountMap,
      });

      expect(repaired.context).toHaveLength(3);
      expect(repaired.context[0]).toBe(context[0]);
      expect(repaired.context[1]).toBe(context[2]);
      expect(repaired.context[2]).toBe(context[3]);
      expect(repaired.droppedOrphanCount).toBe(1);
      expect(repaired.reclaimedTokens).toBe(indexTokenCountMap[1]);
    });

    it('should strip orphan tool_use blocks from AI messages when ToolMessages are not in context', () => {
      const tokenCounter = createTestTokenCounter();
      const context = [
        new HumanMessage('Show me something cool'),
        new AIMessage({
          content: [
            { type: 'text', text: 'Let me create an animation.' },
            {
              type: 'tool_use',
              id: 'tool-navigate',
              name: 'navigate_page',
              input: '{"url":"about:blank"}',
            },
            {
              type: 'tool_use',
              id: 'tool-script',
              name: 'evaluate_script',
              input: '{"function":"' + 'x'.repeat(3000) + '"}',
            },
          ],
          tool_calls: [
            {
              id: 'tool-navigate',
              name: 'navigate_page',
              args: { url: 'about:blank' },
            },
            {
              id: 'tool-script',
              name: 'evaluate_script',
              args: { fn: 'x'.repeat(3000) },
            },
          ],
        }),
        // ToolMessages for both tool calls are NOT in context (pruned)
      ];

      const indexTokenCountMap = {
        0: tokenCounter(context[0]),
        1: tokenCounter(context[1]),
      };

      const repaired = repairOrphanedToolMessages({
        context,
        allMessages: context,
        tokenCounter,
        indexTokenCountMap,
      });

      // AI message should survive but with tool_use blocks stripped
      expect(repaired.context).toHaveLength(2);
      const repairedAI = repaired.context[1] as AIMessage;
      expect(repairedAI.getType()).toBe('ai');

      // Should only have the text block, no tool_use blocks
      const content = repairedAI.content as Array<{ type: string }>;
      expect(content).toHaveLength(1);
      expect(content[0].type).toBe('text');

      // tool_calls should be empty
      expect(repairedAI.tool_calls ?? []).toHaveLength(0);

      // Token savings from stripping the large tool_use blocks
      expect(repaired.reclaimedTokens).toBeGreaterThan(0);
    });

    it('strips orphan raw Responses computer calls with parsed tool calls', () => {
      const tokenCounter = createTestTokenCounter();
      const computerCall = new AIMessage({
        content: 'Taking a screenshot.',
        tool_calls: [
          {
            id: 'computer-orphan',
            name: 'computer_use',
            args: { action: { type: 'screenshot' } },
            type: 'tool_call',
          },
        ],
        response_metadata: {
          output: [
            {
              type: 'computer_call',
              id: 'computer-item',
              call_id: 'computer-orphan',
              action: { type: 'screenshot' },
            },
          ],
        },
      });
      const context: BaseMessage[] = [
        new HumanMessage('take a screenshot'),
        computerCall,
      ];

      const repaired = repairOrphanedToolMessages({
        context,
        allMessages: context,
        tokenCounter,
        indexTokenCountMap: {
          0: tokenCounter(context[0]),
          1: tokenCounter(context[1]),
        },
      });
      const repairedAI = repaired.context[1] as AIMessage;

      expect(repairedAI.tool_calls ?? []).toHaveLength(0);
      expect(repairedAI.response_metadata.output).toEqual([]);
    });

    it('strips orphan computer calls from the LangChain tool_outputs fallback', () => {
      const tokenCounter = createTestTokenCounter();
      const computerCall = new AIMessage({
        content: 'Taking a screenshot.',
        additional_kwargs: {
          tool_outputs: [
            {
              type: 'computer_call',
              call_id: 'computer-fallback-orphan',
              action: { type: 'screenshot' },
            },
          ],
        },
      });
      const context: BaseMessage[] = [
        new HumanMessage('take a screenshot'),
        computerCall,
      ];

      const repaired = repairOrphanedToolMessages({
        context,
        allMessages: context,
        tokenCounter,
        indexTokenCountMap: {
          0: tokenCounter(context[0]),
          1: tokenCounter(context[1]),
        },
      });
      const repairedAI = repaired.context[1] as AIMessage;

      expect(repairedAI.additional_kwargs.tool_outputs).toEqual([]);
    });

    it('retains an empty-content AI parent when one raw call remains paired', () => {
      const tokenCounter = createTestTokenCounter();
      const computerCall = new AIMessage({
        content: [],
        tool_calls: [
          {
            id: 'computer-keep',
            name: 'computer_use',
            args: { action: { type: 'screenshot' } },
            type: 'tool_call',
          },
          {
            id: 'computer-drop',
            name: 'computer_use',
            args: { action: { type: 'screenshot' } },
            type: 'tool_call',
          },
        ],
        response_metadata: {
          output: [
            {
              type: 'computer_call',
              call_id: 'computer-keep',
              action: { type: 'screenshot' },
            },
            {
              type: 'computer_call',
              call_id: 'computer-drop',
              action: { type: 'screenshot' },
            },
          ],
        },
      });
      const computerOutput = new ToolMessage({
        content: 'data:image/png;base64,AA==',
        tool_call_id: 'computer-keep',
        additional_kwargs: { type: 'computer_call_output' },
      });
      const context: BaseMessage[] = [computerCall, computerOutput];

      const repaired = repairOrphanedToolMessages({
        context,
        allMessages: context,
        tokenCounter,
        indexTokenCountMap: {
          0: tokenCounter(computerCall),
          1: tokenCounter(computerOutput),
        },
      });

      expect(repaired.context).toHaveLength(2);
      const repairedAI = repaired.context[0] as AIMessage;
      expect(repairedAI.tool_calls?.map((call) => call.id)).toEqual([
        'computer-keep',
      ]);
      expect(repairedAI.response_metadata.output).toEqual([
        expect.objectContaining({ call_id: 'computer-keep' }),
      ]);
      expect(repaired.context[1]).toBe(computerOutput);
    });

    it('recognizes raw computer calls on generic assistant messages', () => {
      const tokenCounter = createTestTokenCounter();
      const computerCall = new ChatMessage({
        role: 'assistant',
        content: '',
        response_metadata: {
          output: [
            {
              type: 'computer_call',
              call_id: 'generic-computer',
              action: { type: 'screenshot' },
            },
          ],
        },
      });
      const computerOutput = new ToolMessage({
        content: 'data:image/png;base64,AA==',
        tool_call_id: 'generic-computer',
        additional_kwargs: { type: 'computer_call_output' },
      });
      const context: BaseMessage[] = [computerCall, computerOutput];

      const repaired = repairOrphanedToolMessages({
        context,
        allMessages: context,
        tokenCounter,
        indexTokenCountMap: {
          0: tokenCounter(computerCall),
          1: tokenCounter(computerOutput),
        },
      });

      expect(repaired.context).toEqual(context);
      expect(repaired.droppedOrphanCount).toBe(0);
    });

    it('should drop AI message entirely when it has only tool_use blocks with no text', () => {
      const tokenCounter = createTestTokenCounter();
      const context = [
        new HumanMessage('Do something'),
        new AIMessage({
          content: [
            {
              type: 'tool_use',
              id: 'tool-only',
              name: 'some_tool',
              input: '{"query":"test"}',
            },
          ],
          tool_calls: [
            { id: 'tool-only', name: 'some_tool', args: { query: 'test' } },
          ],
        }),
      ];

      const indexTokenCountMap = {
        0: tokenCounter(context[0]),
        1: tokenCounter(context[1]),
      };

      const repaired = repairOrphanedToolMessages({
        context,
        allMessages: context,
        tokenCounter,
        indexTokenCountMap,
      });

      // AI message should be dropped since it had only tool_use blocks
      expect(repaired.context).toHaveLength(1);
      expect(repaired.context[0].getType()).toBe('human');
      expect(repaired.droppedOrphanCount).toBe(1);
    });

    it('should keep tool_use blocks when their ToolMessages ARE in context', () => {
      const tokenCounter = createTestTokenCounter();
      const context = [
        new HumanMessage('Do something'),
        new AIMessage({
          content: [
            { type: 'text', text: 'Calling tool' },
            {
              type: 'tool_use',
              id: 'tool-present',
              name: 'read_file',
              input: '{"path":"test.txt"}',
            },
          ],
          tool_calls: [
            {
              id: 'tool-present',
              name: 'read_file',
              args: { path: 'test.txt' },
            },
          ],
        }),
        new ToolMessage({
          content: 'File contents here',
          tool_call_id: 'tool-present',
        }),
      ];

      const indexTokenCountMap = {
        0: tokenCounter(context[0]),
        1: tokenCounter(context[1]),
        2: tokenCounter(context[2]),
      };

      const repaired = repairOrphanedToolMessages({
        context,
        allMessages: context,
        tokenCounter,
        indexTokenCountMap,
      });

      // Nothing should change — all tool_use blocks have matching ToolMessages
      expect(repaired.context).toHaveLength(3);
      expect(repaired.reclaimedTokens).toBe(0);
      expect(repaired.droppedOrphanCount).toBe(0);
    });

    it('should ensure context does not start with a tool message by finding an AI message', () => {
      const tokenCounter = createTestTokenCounter();
      const messages = [
        new SystemMessage('System instruction'),
        new AIMessage({
          content: [{ type: 'text', text: 'AI msg 1' }],
          tool_calls: [{ id: 'tool1', name: 'read_file', args: { p: '1' } }],
        }),
        new ToolMessage({ content: 'Tool result 1', tool_call_id: 'tool1' }),
        new AIMessage({
          content: [{ type: 'text', text: 'AI msg 2' }],
          tool_calls: [{ id: 'tool2', name: 'read_file', args: { p: '2' } }],
        }),
        new ToolMessage({ content: 'Tool result 2', tool_call_id: 'tool2' }),
      ];

      const indexTokenCountMap = {
        0: 17, // System instruction
        1: 12, // AI message 1
        2: 13, // Tool result 1
        3: 12, // AI message 2
        4: 13, // Tool result 2
      };

      // Create a pruneMessages function with a token limit that will only include the last few messages
      const pruneMessages = createPruneMessages({
        maxTokens: 58, // Only enough for system + last 3 messages + 3, but should not include a parent-less tool message
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap: { ...indexTokenCountMap },
      });

      const result = pruneMessages({ messages });

      // The context should include the system message, AI message 2, and Tool result 2
      // AI message 1 + Tool result 1 are pruned. Tool result 1 is orphaned (AI 1 pruned).
      expect(result.context.length).toBe(3);
      expect(result.context[0]).toBe(messages[0]); // System message
      expect(result.context[1].getType()).toBe('ai'); // AI message 2
      expect(result.context[2]).toBe(messages[4]); // Tool result 2
    });

    it('should ensure context does not start with a tool message by finding a human message', () => {
      const tokenCounter = createTestTokenCounter();
      const messages = [
        new SystemMessage('System instruction'),
        new HumanMessage('Human message 1'),
        new AIMessage('AI message 1'),
        new ToolMessage({ content: 'Tool result 1', tool_call_id: 'tool1' }),
        new HumanMessage('Human message 2'),
        // Tool result 2 has no parent AI tool_call — this is an orphan
        new ToolMessage({ content: 'Tool result 2', tool_call_id: 'tool2' }),
      ];

      const indexTokenCountMap = {
        0: 17, // System instruction
        1: 15, // Human message 1
        2: 12, // AI message 1
        3: 13, // Tool result 1
        4: 15, // Human message 2
        5: 13, // Tool result 2
      };

      // Create a pruneMessages function with a token limit that will only include the last few messages
      const pruneMessages = createPruneMessages({
        maxTokens: 48, // Only enough for system + last 2 messages
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap: { ...indexTokenCountMap },
        reserveRatio: 0,
      });

      const result = pruneMessages({ messages });

      // Tool result 2 is an orphan (no AI message with tool_call_id 'tool2' in context)
      // so it gets dropped. Context is system + human message 2.
      expect(result.context.length).toBe(2);
      expect(result.context[0]).toBe(messages[0]); // System message
      expect(result.context[1]).toBe(messages[4]); // Human message 2
    });

    it('should handle the case where a tool message is followed by an AI message', () => {
      const tokenCounter = createTestTokenCounter();
      const messages = [
        new SystemMessage('System instruction'),
        new HumanMessage('Human message'),
        new AIMessage('AI message with tool use'),
        new ToolMessage({ content: 'Tool result', tool_call_id: 'tool1' }),
        new AIMessage('AI message after tool'),
      ];

      const indexTokenCountMap = {
        0: 17, // System instruction
        1: 13, // Human message
        2: 22, // AI message with tool use
        3: 11, // Tool result
        4: 19, // AI message after tool
      };

      const pruneMessages = createPruneMessages({
        maxTokens: 50,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap: { ...indexTokenCountMap },
      });

      const result = pruneMessages({ messages });

      expect(result.context.length).toBe(2);
      expect(result.context[0]).toBe(messages[0]); // System message
      expect(result.context[1]).toBe(messages[4]); // AI message after tool
    });

    it('should handle the case where a tool message is followed by a human message', () => {
      const tokenCounter = createTestTokenCounter();
      const messages = [
        new SystemMessage('System instruction'),
        new HumanMessage('Human message 1'),
        new AIMessage('AI message with tool use'),
        new ToolMessage({ content: 'Tool result', tool_call_id: 'tool1' }),
        new HumanMessage('Human message 2'),
      ];

      const indexTokenCountMap = {
        0: 17, // System instruction
        1: 15, // Human message 1
        2: 22, // AI message with tool use
        3: 11, // Tool result
        4: 15, // Human message 2
      };

      const pruneMessages = createPruneMessages({
        maxTokens: 46,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap: { ...indexTokenCountMap },
      });

      const result = pruneMessages({ messages });

      expect(result.context.length).toBe(2);
      expect(result.context[0]).toBe(messages[0]); // System message
      expect(result.context[1]).toBe(messages[4]); // Human message 2
    });

    it('should handle complex sequence with multiple tool messages', () => {
      const tokenCounter = createTestTokenCounter();
      const messages = [
        new SystemMessage('System instruction'),
        new HumanMessage('Human message 1'),
        new AIMessage({
          content: [{ type: 'text', text: 'AI message 1' }],
          tool_calls: [{ id: 'tool1', name: 'read_file', args: { path: 'a' } }],
        }),
        new ToolMessage({ content: 'Tool result 1', tool_call_id: 'tool1' }),
        new AIMessage({
          content: [{ type: 'text', text: 'AI message 2' }],
          tool_calls: [{ id: 'tool2', name: 'read_file', args: { path: 'b' } }],
        }),
        new ToolMessage({ content: 'Tool result 2', tool_call_id: 'tool2' }),
        new AIMessage({
          content: [{ type: 'text', text: 'AI message 3' }],
          tool_calls: [{ id: 'tool3', name: 'read_file', args: { path: 'c' } }],
        }),
        new ToolMessage({ content: 'Tool result 3', tool_call_id: 'tool3' }),
      ];

      const indexTokenCountMap = {
        0: 17, // System instruction
        1: 15, // Human message 1
        2: 26, // AI message 1 with tool use
        3: 13, // Tool result 1
        4: 26, // AI message 2 with tool use
        5: 13, // Tool result 2
        6: 26, // AI message 3 with tool use
        7: 13, // Tool result 3
      };

      const pruneMessages = createPruneMessages({
        maxTokens: 111,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap: { ...indexTokenCountMap },
      });

      const result = pruneMessages({ messages });

      // AI messages 2 & 3 with their ToolMessages fit; AI1+Tool1 pruned; Tool1 is orphan (AI1 pruned)
      expect(result.context.length).toBe(5);
      expect(result.context[0]).toBe(messages[0]); // System message
      expect(result.context[1].getType()).toBe('ai'); // AI message 2
      expect(result.context[2]).toBe(messages[5]); // Tool result 2
      expect(result.context[3].getType()).toBe('ai'); // AI message 3
      expect(result.context[4]).toBe(messages[7]); // Tool result 3
    });
  });

  describe('projectToolCallInputs', () => {
    it('matches sequential stream and tool-input projection in one pass', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Use the tool.'),
        new AIMessageChunk({
          content: [
            { type: 'text', text: '' },
            {
              type: 'tool_use',
              id: 'call-1',
              name: 'search',
              input: { query: 'x'.repeat(2_000) },
            },
          ],
          tool_calls: [
            {
              id: 'call-1',
              name: 'search',
              args: { query: 'x'.repeat(2_000) },
            },
          ],
        }),
      ];

      const sequential = projectToolCallInputs(
        projectToolStreamContentForProvider(messages),
        200
      );
      const derived = projectToolMessagesForProvider(messages, 200);

      expect(derived).toEqual(sequential);
      expect(derived[1]).not.toBe(sequential[1]);
      expect(derived[1]).not.toBe(messages[1]);
    });

    it('returns the original array when stream content and inputs are safe', () => {
      const messages: BaseMessage[] = [
        new AIMessage({
          content: [{ type: 'text', text: 'Calling search.' }],
          tool_calls: [
            { id: 'call-1', name: 'search', args: { query: 'safe' } },
          ],
        }),
      ];

      expect(projectToolMessagesForProvider(messages, 200)).toBe(messages);
    });

    it('removes raw OpenAI calls from non-OpenAI provider projections', () => {
      const message = new AIMessage('');
      message.additional_kwargs.tool_calls = [
        {
          id: 'raw-call',
          type: 'function',
          function: {
            name: 'lookup',
            arguments: `{"query":"${'x'.repeat(5_000)}"}`,
          },
        },
      ];

      const [anthropicMessage] = projectToolMessagesForProvider(
        [message],
        200,
        Providers.ANTHROPIC
      );
      const [openAIMessage] = projectToolMessagesForProvider(
        [message],
        200,
        Providers.OPENAI
      );

      expect(anthropicMessage.additional_kwargs.tool_calls).toBeUndefined();
      expect(openAIMessage.additional_kwargs.tool_calls).toBeDefined();
      expect(message.additional_kwargs.tool_calls).toBeDefined();
    });

    it('uses the shared 15%-of-context cap with a 200K ceiling', () => {
      expect(calculateMaxToolCallInputChars()).toBe(200_000);
      expect(calculateMaxToolCallInputChars(0)).toBe(200_000);
      expect(calculateMaxToolCallInputChars(1_000)).toBe(600);
      expect(calculateMaxToolCallInputChars(1_000_000)).toBe(200_000);
      // Even below the envelope floor, serialized args stay a JSON object —
      // 'null' here poisoned replayed tool calls (Anthropic 400s a non-object
      // tool_use.input).
      expect(serializeToolCallInput(undefined, 4)).toBe('{}');
    });

    it('never nulls an input when the cap is below the truncation envelope', () => {
      // Regression: a tight summarization budget can shrink the per-input cap
      // below the elision envelope size.
      // The projection used to return `null` for BOTH the inline block input
      // and the tool_calls args; the nulls were written back into graph state
      // by preFlightTruncateToolCallInputs and later replayed to Anthropic as
      // `tool_use.input: null` → 400 "Input should be an object".
      const chunk = new AIMessageChunk({
        content: [
          {
            type: 'tool_use',
            id: 'tiny-cap-call',
            name: 'calculator',
            // Streaming leaves the raw JSON string on the block.
            input: '{"input": "670592745 / 99991"}',
          },
        ],
        tool_calls: [
          {
            id: 'tiny-cap-call',
            name: 'calculator',
            args: { input: '670592745 / 99991' },
          },
        ],
      });

      const [projected] = projectToolCallInputs([chunk], 20);
      const block = (
        (projected as AIMessageChunk).content as Array<Record<string, unknown>>
      )[0];
      expect(block.input).toEqual({});
      expect((projected as AIMessageChunk).tool_calls?.[0].args).toEqual({});
    });

    it('returns the original array when every input is already safe and bounded', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Run it'),
        new AIMessage({
          content: '',
          tool_calls: [
            { id: 'safe-call', name: 'lookup', args: { query: 'safe' } },
          ],
        }),
      ];

      const projected = projectToolCallInputs(messages, 1_000);

      expect(projected).toBe(messages);
      expect(projected[1]).toBe(messages[1]);
    });

    it('copy-on-write projects inline input and tool_calls args on AIMessageChunk', () => {
      let toJSONCalls = 0;
      const chunk = new AIMessageChunk({
        content: [
          { type: 'text', text: 'Running.' },
          {
            type: 'tool_use',
            id: 'chunk-call',
            name: 'evaluate_script',
            input: { code: 'x'.repeat(2_000) },
          },
        ],
        id: 'chunk-id',
        additional_kwargs: { marker: 'keep' },
        response_metadata: { model: 'chunk-model' },
        tool_calls: [
          {
            id: 'chunk-call',
            name: 'evaluate_script',
            args: {
              query: 'safe',
              toJSON() {
                toJSONCalls++;
                return { code: 'y'.repeat(2_000_000) };
              },
            },
          },
        ],
      });
      chunk.tool_call_chunks = [
        {
          id: 'chunk-call',
          index: 0,
          name: 'evaluate_script',
          args: '',
          type: 'tool_call_chunk',
        },
      ];
      const messages: BaseMessage[] = [chunk];

      const projected = projectToolCallInputs(messages, 200);

      expect(projected).not.toBe(messages);
      expect(projected[0]).not.toBe(chunk);
      expect(projected[0]).toBeInstanceOf(AIMessageChunk);
      expect(toJSONCalls).toBe(0);
      const projectedChunk = projected[0] as AIMessageChunk;
      expect(projectedChunk.id).toBe('chunk-id');
      expect(projectedChunk.additional_kwargs).toEqual({ marker: 'keep' });
      expect(projectedChunk.response_metadata).toEqual({
        model: 'chunk-model',
      });
      expect(projectedChunk.tool_call_chunks).toEqual(chunk.tool_call_chunks);
      const inlineToolUse = (
        projectedChunk.content as Array<Record<string, unknown>>
      ).find((block) => block.type === 'tool_use');
      expect(inlineToolUse?.input).toEqual({
        _note: TOOL_INPUT_ELISION_NOTE,
        _originalChars: expect.any(Number),
        _inputPrefix: expect.any(String),
      });
      expect(projectedChunk.tool_calls?.[0].args).toEqual({ query: 'safe' });
      expect(
        (
          (chunk.content as Array<Record<string, unknown>>)[1].input as {
            code: string;
          }
        ).code
      ).toHaveLength(2_000);
    });

    it('normalizes cyclic and bigint args into provider-safe JSON values', () => {
      const cyclic: Record<string, unknown> = { query: 'safe' };
      cyclic.self = cyclic;
      const messages: BaseMessage[] = [
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'non-json-call',
              name: 'lookup',
              args: { cyclic, count: 7n },
            },
          ],
        }),
      ];

      const projected = projectToolCallInputs(messages, 1_000);

      expect(projected).not.toBe(messages);
      const args = (projected[0] as AIMessage).tool_calls?.[0].args as {
        cyclic: { self: string };
        count: string;
      };
      expect(args.cyclic.self).toBe('[Circular]');
      expect(args.count).toBe('7');
      expect(() => JSON.stringify(args)).not.toThrow();
    });

    it('bounds legacy additional_kwargs.function_call arguments', () => {
      const message = new AIMessage({
        content: '',
        additional_kwargs: {
          marker: 'preserved',
          function_call: {
            name: 'legacy_lookup',
            arguments: `{"query":"${'x'.repeat(2_000)}"}`,
          },
        },
      });
      const messages: BaseMessage[] = [message];

      const projected = projectToolCallInputs(messages, 200);

      expect(projected).not.toBe(messages);
      expect(projected[0]).not.toBe(message);
      expect(projected[0].additional_kwargs.marker).toBe('preserved');
      const projectedFunctionCall = projected[0].additional_kwargs
        .function_call as {
        name: string;
        arguments: string;
      };
      expect(projectedFunctionCall.name).toBe('legacy_lookup');
      expect(projectedFunctionCall.arguments.length).toBeLessThanOrEqual(200);
      expect(() => JSON.parse(projectedFunctionCall.arguments)).not.toThrow();
      expect(
        (
          message.additional_kwargs.function_call as {
            arguments: string;
          }
        ).arguments.length
      ).toBeGreaterThan(2_000);
    });

    it('normalizes legacy function calls without invoking serialization hooks', () => {
      const toJSON = jest.fn(() => ({
        name: 'legacy_lookup',
        arguments: 'x'.repeat(500_000),
      }));
      const originalFunctionCall = {
        name: 'legacy_lookup',
        arguments: '{}',
        toJSON,
      };
      const message = new AIMessage({
        content: '',
        additional_kwargs: {
          function_call: originalFunctionCall,
        },
      });

      const [projected] = projectToolCallInputs([message], 200);
      const projectedFunctionCall = projected.additional_kwargs
        .function_call as {
        name: string;
        arguments: string;
      };
      const serialized = JSON.stringify(projected.additional_kwargs);

      expect(projected).not.toBe(message);
      expect(projectedFunctionCall).toEqual({
        name: 'legacy_lookup',
        arguments: '{}',
      });
      expect(Object.getPrototypeOf(projectedFunctionCall)).toBe(
        Object.prototype
      );
      expect(serialized.length).toBeLessThanOrEqual(300);
      expect(toJSON).not.toHaveBeenCalled();
      expect(projectToolCallInputs([projected], 200)[0]).toBe(projected);
    });

    it('omits a proxied legacy function call without propagating descriptor traps', () => {
      let descriptorCalls = 0;
      const legacyProxy = new Proxy(
        {},
        {
          getOwnPropertyDescriptor() {
            descriptorCalls++;
            throw new Error('descriptor trap');
          },
        }
      );
      const message = new AIMessage({
        content: '',
        additional_kwargs: {},
      });
      (message.additional_kwargs as Record<string, unknown>).function_call =
        legacyProxy;

      expect(() => projectToolCallInputs([message], 200)).not.toThrow();
      const [projected] = projectToolCallInputs([message], 200);

      expect(projected.additional_kwargs.function_call).toBeUndefined();
      expect(descriptorCalls).toBe(0);
      expect(projectToolCallInputs([projected], 200)[0]).toBe(projected);
    });

    it('shadows proxy-backed legacy metadata without invoking its get trap', () => {
      let getCalls = 0;
      const hugeFunctionCall = {
        name: 'legacy_lookup',
        arguments: 'x'.repeat(500_000),
      };
      const message = new AIMessage({
        content: '',
        additional_kwargs: {},
      });
      message.additional_kwargs = new Proxy(
        {},
        {
          get(_target, property) {
            getCalls++;
            return property === 'function_call' ? hugeFunctionCall : undefined;
          },
          getOwnPropertyDescriptor() {
            return undefined;
          },
          getPrototypeOf() {
            return null;
          },
        }
      );

      const [projected] = projectToolCallInputs([message], 200);

      expect(getCalls).toBe(0);
      expect(projected).not.toBe(message);
      expect(projected.additional_kwargs.function_call).toBeUndefined();
      expect(projectToolCallInputs([projected], 200)[0]).toBe(projected);
    });

    it('shadows legacy calls beyond the bounded prototype walk', () => {
      const inherited = {
        function_call: {
          name: 'legacy_lookup',
          arguments: 'x'.repeat(500_000),
        },
      };
      let additionalKwargs = inherited;
      for (let i = 0; i < 101; i++) {
        additionalKwargs = Object.create(additionalKwargs) as typeof inherited;
      }
      const message = new AIMessage({
        content: '',
        additional_kwargs: {},
      });
      message.additional_kwargs = additionalKwargs;

      const [projected] = projectToolCallInputs([message], 200);

      expect(projected).not.toBe(message);
      expect(
        Object.prototype.hasOwnProperty.call(
          projected.additional_kwargs,
          'function_call'
        )
      ).toBe(true);
      expect(projected.additional_kwargs.function_call).toBeUndefined();
      expect(projectToolCallInputs([projected], 200)[0]).toBe(projected);
    });

    it('stabilizes an omitted invalid legacy function call', () => {
      const message = new AIMessage({
        content: '',
        additional_kwargs: {},
      });
      (message.additional_kwargs as Record<string, unknown>).function_call =
        null;

      const [first] = projectToolCallInputs([message], 200);
      const [second] = projectToolCallInputs([first], 200);

      expect(first).not.toBe(message);
      expect(first.additional_kwargs.function_call).toBeUndefined();
      expect(second).toBe(first);
    });

    it('projects every provider-consumed tool-call representation to the same bounded wire args', () => {
      let getterCalls = 0;
      let toJSONCalls = 0;
      const adversarialArgs: Record<string, unknown> = {
        query: 'safe',
        payload: 'x'.repeat(2_000),
      };
      Object.defineProperty(adversarialArgs, 'expanded', {
        enumerable: true,
        get() {
          getterCalls++;
          return 'y'.repeat(2_000_000);
        },
      });
      adversarialArgs.self = adversarialArgs;
      adversarialArgs.toJSON = (): Record<string, string> => {
        toJSONCalls++;
        return { payload: 'z'.repeat(2_000_000) };
      };

      const message = new AIMessageChunk({
        content: [
          {
            type: 'tool_use',
            id: 'inline-use',
            name: 'evaluate',
            input: { code: 'i'.repeat(2_000) },
          },
          {
            type: 'tool_call',
            id: 'inline-call',
            name: 'evaluate',
            args: { code: 'a'.repeat(2_000) },
            tool_call: {
              type: 'tool_call',
              id: 'nested-call',
              name: 'evaluate',
              args: { code: 'n'.repeat(2_000) },
            },
          },
        ],
        tool_calls: [
          {
            id: 'shared-call',
            name: 'evaluate',
            args: adversarialArgs,
          },
          {
            id: 'custom-call',
            name: 'shell',
            args: { input: 'c'.repeat(2_000) },
          },
          {
            id: 'computer-call',
            name: 'computer_use',
            args: { action: { type: 'type', text: 't'.repeat(2_000) } },
          },
        ],
        additional_kwargs: {
          tool_calls: [
            {
              id: 'shared-call',
              type: 'function',
              function: {
                name: 'evaluate',
                arguments: `{"code":"${'r'.repeat(2_000)}"}`,
              },
            },
            {
              id: 'raw-only',
              type: 'function',
              function: {
                name: 'evaluate',
                arguments: `{"code":"${'o'.repeat(2_000)}"}`,
              },
            },
          ],
        },
        response_metadata: {
          output: [
            {
              type: 'function_call',
              id: 'fc_shared',
              call_id: 'shared-call',
              name: 'evaluate',
              arguments: `{"code":"${'w'.repeat(2_000)}"}`,
            },
            {
              type: 'function_call',
              id: 'fc_raw',
              call_id: 'response-only',
              name: 'evaluate',
              arguments: `{"code":"${'q'.repeat(2_000)}"}`,
            },
            {
              type: 'custom_tool_call',
              id: 'ctc_shared',
              call_id: 'custom-call',
              name: 'shell',
              input: 'c'.repeat(2_000),
            },
            {
              type: 'computer_call',
              id: 'cc_shared',
              call_id: 'computer-call',
              action: { type: 'type', text: 't'.repeat(2_000) },
            },
          ],
        },
      });
      Object.defineProperty(message, 'projectionMarker', {
        configurable: false,
        enumerable: false,
        value: 'preserved',
        writable: false,
      });
      const messages: BaseMessage[] = [message];

      const projected = projectToolCallInputs(messages, 200);

      expect(projected).not.toBe(messages);
      expect(projected[0]).toBeInstanceOf(AIMessageChunk);
      expect(getterCalls).toBe(0);
      expect(toJSONCalls).toBe(0);
      expect(
        Object.getOwnPropertyDescriptor(projected[0], 'projectionMarker')
      ).toEqual(Object.getOwnPropertyDescriptor(message, 'projectionMarker'));

      const projectedMessage = projected[0] as AIMessageChunk;
      const content = projectedMessage.content as Array<
        Record<string, unknown>
      >;
      const inlineUse = content[0];
      const inlineCall = content[1];
      const nestedCall = inlineCall.tool_call as Record<string, unknown>;
      for (const input of [
        inlineUse.input,
        inlineCall.args,
        nestedCall.args,
        projectedMessage.tool_calls?.[0].args,
      ]) {
        const serialized = serializeToolCallInput(input, 200);
        expect(serialized.length).toBeLessThanOrEqual(200);
        expect(() => JSON.parse(serialized)).not.toThrow();
      }

      const canonicalArguments = serializeToolCallInput(
        projectedMessage.tool_calls?.[0].args,
        200
      );
      const rawToolCalls = projectedMessage.additional_kwargs
        .tool_calls as Array<{
        id: string;
        function: { arguments: string };
      }>;
      expect(rawToolCalls[0].function.arguments).toBe(canonicalArguments);
      expect(rawToolCalls[1].function.arguments.length).toBeLessThanOrEqual(
        200
      );
      expect(() =>
        JSON.parse(rawToolCalls[1].function.arguments)
      ).not.toThrow();

      const responseOutput = projectedMessage.response_metadata
        .output as Array<{
        type: string;
        call_id: string;
        arguments?: string;
        input?: string;
        action?: unknown;
      }>;
      expect(responseOutput[0].arguments).toBe(canonicalArguments);
      expect(responseOutput[1].arguments?.length).toBeLessThanOrEqual(200);
      expect(() => JSON.parse(responseOutput[1].arguments ?? '')).not.toThrow();
      expect(responseOutput[2].input?.length).toBeLessThanOrEqual(200);
      expect(
        serializeToolCallInput(responseOutput[3].action, 200).length
      ).toBeLessThanOrEqual(200);

      expect(
        (
          message.additional_kwargs.tool_calls as Array<{
            function: { arguments: string };
          }>
        )[0].function.arguments.length
      ).toBeGreaterThan(2_000);
      expect(
        (
          message.response_metadata.output as Array<{
            arguments: string;
          }>
        )[0].arguments.length
      ).toBeGreaterThan(2_000);
    });
  });

  describe('preFlightTruncateToolCallInputs', () => {
    it('should truncate oversized tool_use input fields in AI messages', () => {
      const tokenCounter = createTestTokenCounter();
      const largeInput = '{"function":"' + 'x'.repeat(5000) + '"}';
      const messages: BaseMessage[] = [
        new HumanMessage('Run this script'),
        new AIMessage({
          content: [
            { type: 'text', text: 'I will execute the script.' },
            {
              type: 'tool_use',
              id: 'tool-exec',
              name: 'evaluate_script',
              input: largeInput,
            },
          ],
          tool_calls: [
            {
              id: 'tool-exec',
              name: 'evaluate_script',
              args: { function: 'x'.repeat(5000) },
            },
          ],
        }),
        new ToolMessage({ content: 'Result: OK', tool_call_id: 'tool-exec' }),
      ];

      const indexTokenCountMap: Record<string, number | undefined> = {
        0: tokenCounter(messages[0]),
        1: tokenCounter(messages[1]),
        2: tokenCounter(messages[2]),
      };

      const originalTokens = indexTokenCountMap[1] as number;
      expect(originalTokens).toBeGreaterThan(5000); // Large input counted

      // maxContextTokens: 1000 → maxInputChars = floor(1000 * 0.15) * 4 = 600
      const truncated = preFlightTruncateToolCallInputs({
        messages,
        maxContextTokens: 1000,
        indexTokenCountMap,
        tokenCounter,
      });

      expect(truncated).toBe(1);
      const newTokens = indexTokenCountMap[1] as number;
      expect(newTokens).toBeLessThan(originalTokens);

      // Verify the content block was truncated
      const aiMsg = messages[1] as AIMessage;
      const toolUseBlock = (
        aiMsg.content as Array<Record<string, unknown>>
      ).find((b) => b.type === 'tool_use');
      expect(toolUseBlock).toBeDefined();
      const truncatedInput = toolUseBlock!.input as {
        _note: string;
        _originalChars: number;
        _inputPrefix: string;
      };
      /** A completed call must never read as one that was cut off. */
      expect(truncatedInput._note).toBe(TOOL_INPUT_ELISION_NOTE);
      expect(JSON.stringify(truncatedInput)).not.toContain('truncated');
      expect(truncatedInput._originalChars).toBeGreaterThan(600);

      // Verify tool_calls args were also truncated
      expect(aiMsg.tool_calls).toBeDefined();
      const tc = aiMsg.tool_calls![0];
      expect(tc.args).toHaveProperty('_note', TOOL_INPUT_ELISION_NOTE);
    });

    it('truncates tool_calls-only args and preserves AI message metadata', () => {
      let recounts = 0;
      const tokenCounter: t.TokenCounter = () => {
        recounts++;
        return 17;
      };
      const messages: BaseMessage[] = [
        new AIMessage({
          content: '',
          id: 'ai-tool-call-only',
          name: 'assistant',
          additional_kwargs: { trace_marker: 'keep' },
          response_metadata: { model: 'test-model' },
          tool_calls: [
            {
              id: 'tool-exec',
              name: 'evaluate_script',
              args: { code: 'x'.repeat(5_000) },
            },
          ],
        }),
      ];
      const indexTokenCountMap: Record<string, number | undefined> = { 0: 1 };

      const truncated = preFlightTruncateToolCallInputs({
        messages,
        maxContextTokens: 1_000,
        indexTokenCountMap,
        tokenCounter,
      });

      expect(truncated).toBe(1);
      expect(recounts).toBe(1);
      expect(indexTokenCountMap[0]).toBe(17);
      const projected = messages[0] as AIMessage;
      expect(projected.content).toBe('');
      expect(projected.id).toBe('ai-tool-call-only');
      expect(projected.name).toBe('assistant');
      expect(projected.additional_kwargs).toEqual({ trace_marker: 'keep' });
      expect(projected.response_metadata).toEqual({ model: 'test-model' });
      expect(projected.tool_calls?.[0].args).toEqual({
        _note: TOOL_INPUT_ELISION_NOTE,
        _originalChars: expect.any(Number),
        _inputPrefix: expect.any(String),
      });
    });

    it('neutralizes adversarial tool-call args without invoking toJSON', () => {
      let toJSONCalls = 0;
      const messages: BaseMessage[] = [
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'tool-adversarial',
              name: 'lookup',
              args: {
                query: 'safe',
                toJSON() {
                  toJSONCalls++;
                  return { query: 'x'.repeat(2_000_000) };
                },
              },
            },
          ],
        }),
      ];
      const indexTokenCountMap: Record<string, number | undefined> = { 0: 1 };

      const changed = preFlightTruncateToolCallInputs({
        messages,
        maxContextTokens: 1_000,
        indexTokenCountMap,
        tokenCounter: () => 5,
      });

      expect(changed).toBe(1);
      expect(toJSONCalls).toBe(0);
      expect((messages[0] as AIMessage).tool_calls?.[0].args).toEqual({
        query: 'safe',
      });
    });

    it('should not truncate inputs that fit within the budget', () => {
      const tokenCounter = createTestTokenCounter();
      const messages: BaseMessage[] = [
        new HumanMessage('Read a file'),
        new AIMessage({
          content: [
            { type: 'text', text: 'Reading file.' },
            {
              type: 'tool_use',
              id: 'tool-read',
              name: 'read_file',
              input: '{"path":"test.txt"}',
            },
          ],
          tool_calls: [
            { id: 'tool-read', name: 'read_file', args: { path: 'test.txt' } },
          ],
        }),
      ];

      const indexTokenCountMap: Record<string, number | undefined> = {
        0: tokenCounter(messages[0]),
        1: tokenCounter(messages[1]),
      };

      const originalTokens = indexTokenCountMap[1];

      const truncated = preFlightTruncateToolCallInputs({
        messages,
        maxContextTokens: 1000,
        indexTokenCountMap,
        tokenCounter,
      });

      expect(truncated).toBe(0);
      expect(indexTokenCountMap[1]).toBe(originalTokens);
    });

    it('should skip non-AI messages', () => {
      const tokenCounter = createTestTokenCounter();
      const messages: BaseMessage[] = [
        new HumanMessage('Hello'),
        new ToolMessage({ content: 'x'.repeat(5000), tool_call_id: 'tool-1' }),
      ];

      const indexTokenCountMap: Record<string, number | undefined> = {
        0: tokenCounter(messages[0]),
        1: tokenCounter(messages[1]),
      };

      const truncated = preFlightTruncateToolCallInputs({
        messages,
        maxContextTokens: 1000,
        indexTokenCountMap,
        tokenCounter,
      });

      // Should not touch ToolMessages (that's preFlightTruncateToolResults' job)
      expect(truncated).toBe(0);
    });
  });

  describe('Instruction token budget reservation (getInstructionTokens)', () => {
    it('should reserve budget for instruction tokens when no system message is present', () => {
      const tokenCounter = createTestTokenCounter();
      // Agent flow: messages do NOT include a system message.
      // The system message is prepended later by buildSystemRunnable.
      const messages = [
        new HumanMessage('Hello there'), // 11 chars
        new AIMessage('Hi'), // 2 chars
        new HumanMessage('How are you?'), // 12 chars
        new AIMessage('Good'), // 4 chars
      ];

      const indexTokenCountMap: Record<string, number | undefined> = {};
      for (let i = 0; i < messages.length; i++) {
        indexTokenCountMap[i] = tokenCounter(messages[i]);
      }

      // Total message tokens: 11 + 2 + 12 + 4 = 29
      // Instruction tokens: 20 (simulating system prompt overhead)
      // Effective budget: 52 - 20 - 3 reply primer = 29 → exact fit
      const pruneMessages = createPruneMessages({
        maxTokens: 52,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap,
        reserveRatio: 0,
        getInstructionTokens: () => 20,
      });

      const result = pruneMessages({ messages });

      // All messages fit exactly with instructions and reply framing.
      expect(result.context.length).toBe(4);
      expect(result.context).toEqual(messages);
      expect(result.messagesToRefine).toEqual([]);
    });

    it('should prune when messages + instruction tokens exceed budget', () => {
      const tokenCounter = createTestTokenCounter();
      const messages = [
        new HumanMessage('Hello there'), // 11 chars
        new AIMessage('Hi'), // 2 chars
        new HumanMessage('How are you?'), // 12 chars
        new AIMessage('Good'), // 4 chars
      ];

      const indexTokenCountMap: Record<string, number | undefined> = {};
      for (let i = 0; i < messages.length; i++) {
        indexTokenCountMap[i] = tokenCounter(messages[i]);
      }

      // Total message tokens: 29
      // Instruction tokens: 25 (simulating large tool schema overhead)
      // Effective budget: 40 - 25 = 15 → must prune older messages
      const pruneMessages = createPruneMessages({
        maxTokens: 40,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap,
        getInstructionTokens: () => 25,
      });

      const result = pruneMessages({ messages });

      // Should prune older messages to fit within 15 available tokens.
      // Working backwards: "Good" (4) + "How are you?" (12) = 16 > 15
      // So only "Good" (4) fits, context starts on that AI message.
      // But startType may require a human message...
      // Actually with no startType and 3 tokens of overhead,
      // available = 15 - 3 = 12: "Good" (4) fits, "How are you?" (12) → 4+12=16 > 12
      // So only "Good" (4) fits.
      expect(result.context.length).toBeLessThan(4);
      expect(Array.isArray(result.messagesToRefine)).toBe(true);
      expect(result.messagesToRefine!.length).toBeGreaterThan(0);
    });

    it('should correctly account for instruction tokens in early-return path', () => {
      const tokenCounter = createTestTokenCounter();
      const messages = [
        new HumanMessage('Hi'), // 2 chars
        new AIMessage('Hello'), // 5 chars
      ];

      const indexTokenCountMap: Record<string, number | undefined> = {};
      for (let i = 0; i < messages.length; i++) {
        indexTokenCountMap[i] = tokenCounter(messages[i]);
      }

      // Message tokens: 7
      // Instruction tokens: 100 (simulating 26 MCP tools ~5000 chars)
      // Budget: 50 → 7 + 100 = 107 > 50, so early-return should NOT fire
      const pruneMessages = createPruneMessages({
        maxTokens: 50,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap,
        getInstructionTokens: () => 100,
      });

      const result = pruneMessages({ messages });

      // Even though messages alone (7) fit in 50, the instruction overhead (100)
      // means pruning must occur.  With only 50 - 100 = -50 effective budget,
      // nothing fits → all messages pruned.
      expect(result.messagesToRefine!.length).toBeGreaterThan(0);
    });

    it('should not double-subtract when messages include a system message', () => {
      const tokenCounter = createTestTokenCounter();
      const messages = [
        new SystemMessage('System'), // 6 chars
        new HumanMessage('Hello there'), // 11 chars
        new AIMessage('Hi'), // 2 chars
      ];

      const indexTokenCountMap: Record<string, number | undefined> = {};
      for (let i = 0; i < messages.length; i++) {
        indexTokenCountMap[i] = tokenCounter(messages[i]);
      }

      // When messages[0] IS a system message, getMessagesWithinTokenLimit uses
      // indexTokenCountMap[0] (6) to subtract from budget, ignoring instructionTokens.
      // getInstructionTokens is only used when no system message is at index 0.
      const pruneMessages = createPruneMessages({
        maxTokens: 30,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap,
        getInstructionTokens: () => 999, // Should be ignored for system message path
      });

      const result = pruneMessages({ messages });

      // Budget: 30 - 6 (system) = 24 available.
      // "Hi" (2) + "Hello there" (11) + 3 overhead = 16, fits in 24.
      // All messages should be kept.
      expect(result.context.length).toBe(3);
      expect(result.context[0]).toBe(messages[0]); // System message preserved
    });

    it('index 0 should NOT be inflated when getInstructionTokens is provided', () => {
      const tokenCounter = createTestTokenCounter();
      const messages = [
        new HumanMessage('Hello there'), // 11 chars
        new AIMessage('Hi'), // 2 chars
      ];

      const indexTokenCountMap: Record<string, number | undefined> = {};
      for (let i = 0; i < messages.length; i++) {
        indexTokenCountMap[i] = tokenCounter(messages[i]);
      }

      const pruneMessages = createPruneMessages({
        maxTokens: 50,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap,
        getInstructionTokens: () => 10,
      });

      // Before and after pruning, index 0 should remain 11 (the real token count)
      expect(indexTokenCountMap[0]).toBe(11);
      pruneMessages({ messages });
      // The returned indexTokenCountMap should still have the real count at index 0
      expect(indexTokenCountMap[0]).toBe(11);
    });

    it('pre-flight truncation uses effective budget after instruction overhead', () => {
      const tokenCounter = createTestTokenCounter();
      // Simulate the real scenario: AI message has a massive tool_call input
      // (like the chrome-devtools evaluate_script with a 7000-char JS payload)
      const hugeInput = 'x'.repeat(7000);
      const messages = [
        new HumanMessage('show me something'), // 17 chars
        new AIMessage({
          content: [
            { type: 'text', text: 'Creating animation' },
            {
              type: 'tool_use',
              id: 'tool_1',
              name: 'evaluate_script',
              input: { function: hugeInput },
            },
          ],
          tool_calls: [
            {
              id: 'tool_1',
              name: 'evaluate_script',
              args: { function: hugeInput },
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({ content: 'Script executed', tool_call_id: 'tool_1' }),
      ];

      const indexTokenCountMap: Record<string, number | undefined> = {};
      for (let i = 0; i < messages.length; i++) {
        indexTokenCountMap[i] = tokenCounter(messages[i]);
      }

      // Pre-flight truncation uses maxTokens for the truncation threshold:
      // Math.floor(8000*0.15)*4 = 4800 chars.  The AI message's tool_use
      // input (~7015 chars) shrinks to ~4800 chars, giving an AI token
      // count of ~4850.
      //
      // The effective pruning budget subtracts instruction overhead:
      // effectiveMax = 8000 - 2000 = 6000, which is enough for all three
      // messages (~4850 + 17 + 15 ≈ 4882).
      const instructionTokens = 2000;
      const pruneMessages = createPruneMessages({
        maxTokens: 8000,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap,
        getInstructionTokens: () => instructionTokens,
        reserveRatio: 0,
      });

      const result = pruneMessages({ messages });

      // The AI message should survive pruning (not be in messagesToRefine)
      // because pre-flight truncation used the effective budget
      const aiMessagesInContext = result.context.filter(
        (m) => m.getType() === 'ai'
      );
      expect(aiMessagesInContext.length).toBe(1);
      expect(result.context.length).toBe(3); // All 3 messages fit after truncation
    });

    it('emergency truncation recovers when initial prune produces empty context', () => {
      const tokenCounter = createTestTokenCounter();
      // Simulate post-summarization state: only 4 messages remain, but one
      // has a huge tool_call input that exceeds available budget alone.
      // With char-based counter, the AI message with 4000-char input is ~4000 tokens.
      // Available budget: 5000 - 4500 = 500. Nothing fits on first pass.
      const hugeInput = 'x'.repeat(4000);
      const messages = [
        new AIMessage({
          content: [
            { type: 'text', text: 'Running script' },
            {
              type: 'tool_use',
              id: 'tool_1',
              name: 'evaluate_script',
              input: { function: hugeInput },
            },
          ],
          tool_calls: [
            {
              id: 'tool_1',
              name: 'evaluate_script',
              args: { function: hugeInput },
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({
          content: 'Script ran successfully',
          tool_call_id: 'tool_1',
        }),
        new HumanMessage('that looks great'),
        new AIMessage('Thanks! Want more?'),
      ];

      const indexTokenCountMap: Record<string, number | undefined> = {};
      for (let i = 0; i < messages.length; i++) {
        indexTokenCountMap[i] = tokenCounter(messages[i]);
      }

      // Available budget is extremely tight: 500 tokens for messages.
      // The AI message alone is ~4000+ tokens. Initial prune: nothing fits.
      // Emergency truncation should reduce tool inputs to 150 chars,
      // making the AI message fit.
      const pruneMessages = createPruneMessages({
        maxTokens: 5000,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap,
        getInstructionTokens: () => 4500,
      });

      const result = pruneMessages({ messages });

      // Emergency truncation should have recovered — context is NOT empty
      expect(result.context.length).toBeGreaterThan(0);
      // At minimum, the newest messages should be present
      const types = result.context.map((m) => m.getType());
      expect(types).toContain('human');
    });

    it('emergency-truncates tool_calls-only args when content is empty', () => {
      const tokenCounter: t.TokenCounter = (message) => {
        let chars =
          typeof message.content === 'string' ? message.content.length : 0;
        if (message.getType() === 'ai') {
          for (const toolCall of (message as AIMessage).tool_calls ?? []) {
            chars +=
              toolCall.name.length + JSON.stringify(toolCall.args).length;
          }
        }
        return Math.max(1, Math.ceil(chars / 4));
      };
      const messages: BaseMessage[] = [
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'tool-call-only',
              name: 'evaluate_script',
              args: { code: 'x'.repeat(4_000) },
            },
          ],
        }),
        new ToolMessage({
          content: 'OK',
          tool_call_id: 'tool-call-only',
          name: 'evaluate_script',
        }),
      ];
      const indexTokenCountMap: Record<string, number | undefined> = {
        0: tokenCounter(messages[0]),
        1: tokenCounter(messages[1]),
      };
      const pruneMessages = createPruneMessages({
        maxTokens: 1_000,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap,
        getInstructionTokens: () => 900,
        reserveRatio: 0,
      });

      const result = pruneMessages({ messages });

      const projectedAI = result.context.find(
        (message) => message.getType() === 'ai'
      ) as AIMessage | undefined;
      expect(projectedAI).toBeDefined();
      expect(projectedAI?.content).toBe('');
      const projectedArgs = projectedAI?.tool_calls?.[0].args as
        | { _note?: string; _inputPrefix?: string; _originalChars?: number }
        | undefined;
      expect(projectedArgs?._note).toBe(TOOL_INPUT_ELISION_NOTE);
      expect(projectedArgs?._inputPrefix?.length).toBeLessThanOrEqual(200);
      expect(projectedArgs?._originalChars).toBeGreaterThan(200);
    });
  });

  describe('Empty messages guard', () => {
    it('returns empty context without crashing when messages array is empty', () => {
      const tokenCounter = createTestTokenCounter();
      const pruneMessages = createPruneMessages({
        maxTokens: 8000,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap: {},
        getInstructionTokens: () => 4000,
      });

      // Simulate post-summarization state where REMOVE_ALL left an empty messages array
      const result = pruneMessages({
        messages: [],
        usageMetadata: {
          input_tokens: 100,
          output_tokens: 50,
          total_tokens: 150,
        } as UsageMetadata,
      });

      expect(result.context).toEqual([]);
      expect(result.messagesToRefine).toEqual([]);
      expect(result.prePruneContextTokens).toBe(0);
      /** Reserve-adjusted budget minus instructions and reply primer */
      expect(result.contextBudget).toBe(7600);
      expect(result.effectiveInstructionTokens).toBe(4000);
      expect(result.remainingContextTokens).toBe(3597);
    });
  });

  describe('Dropped orphan ToolMessages appear in messagesToRefine', () => {
    it('appends orphan ToolMessage (whose parent AI was pruned) to messagesToRefine for summarization', () => {
      const tokenCounter = createTestTokenCounter();

      // Build messages where the large AI(evaluate) won't fit in a tight budget,
      // but its smaller ToolMessage(evaluate) does.  After backward iteration,
      // the ToolMessage lands in context while its parent AI is in prunedMemory.
      // repairOrphanedToolMessages then drops the orphan ToolMessage from context.
      // The fix: that dropped ToolMessage must appear in messagesToRefine so
      // summarization sees the tool result (otherwise summary says "in progress").
      const messages: BaseMessage[] = [
        new HumanMessage('Build me a solar system simulation'),
        new AIMessage({
          content: [
            {
              type: 'text',
              text: `I will write the code now. ${'Plan step. '.repeat(60)}`,
            },
            {
              type: 'tool_use',
              id: 'tc_eval',
              name: 'evaluate_script',
              // Large input that consumes most of the budget
              input: { code: 'x'.repeat(3000) },
            },
          ],
          tool_calls: [
            {
              id: 'tc_eval',
              name: 'evaluate_script',
              args: { code: 'x'.repeat(3000) },
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({
          // Small result — fits in budget individually
          content: 'Solar system simulation launched successfully!',
          tool_call_id: 'tc_eval',
          name: 'evaluate_script',
        }),
      ];

      const indexTokenCountMap: Record<string, number | undefined> = {};
      for (let i = 0; i < messages.length; i++) {
        indexTokenCountMap[i] = tokenCounter(messages[i]);
      }

      // Budget is tight enough that the large AI message won't fit
      // even after emergency truncation, but HumanMessage and ToolMessage
      // individually can.  Budget must be low enough that proportional
      // emergency truncation (budget / messages * 4 chars) still leaves
      // the AI message too large to fit.
      const pruneMessages = createPruneMessages({
        maxTokens: 100,
        startIndex: 0,
        tokenCounter,
        indexTokenCountMap,
        getInstructionTokens: () => 0,
      });

      const result = pruneMessages({ messages });

      // The orphan ToolMessage(evaluate) should NOT be in context
      // (its parent AI was pruned away)
      const contextToolMsgs = result.context.filter(
        (m) => m.getType() === 'tool'
      );
      const orphanInContext = contextToolMsgs.some(
        (m) => (m as ToolMessage).tool_call_id === 'tc_eval'
      );
      expect(orphanInContext).toBe(false);

      // The key assertion: the dropped ToolMessage MUST appear in messagesToRefine
      // so that summarization can see "Solar system simulation launched successfully!"
      expect(result.messagesToRefine).toBeDefined();
      const refineToolMsgs = result.messagesToRefine!.filter(
        (m) => m.getType() === 'tool'
      );
      const toolInRefine = refineToolMsgs.some(
        (m) => (m as ToolMessage).tool_call_id === 'tc_eval'
      );
      expect(toolInRefine).toBe(true);

      // The parent AI message should also be in messagesToRefine (from prunedMemory)
      const refineAiMsgs = result.messagesToRefine!.filter(
        (m) => m.getType() === 'ai'
      );
      const aiInRefine = refineAiMsgs.some((m) =>
        ((m as AIMessage).tool_calls ?? []).some((tc) => tc.id === 'tc_eval')
      );
      expect(aiInRefine).toBe(true);
    });
  });

  describe('Integration with Run', () => {
    it('should initialize Run with custom token counter and process messages', async () => {
      const provider = Providers.OPENAI;
      const llmConfig = getLLMConfig(provider);
      const tokenCounter = createTestTokenCounter();

      const run = await Run.create<t.IState>({
        runId: 'test-prune-run',
        graphConfig: {
          type: 'standard',
          llmConfig,
          instructions: 'You are a helpful assistant.',
          maxContextTokens: 1000,
        },
        returnContent: true,
        skipCleanup: true,
        tokenCounter,
        indexTokenCountMap: {},
      });

      // Override the model to use a fake LLM
      run.Graph?.overrideTestModel(['This is a test response'], 1);

      const messages = [new HumanMessage('Hello, how are you?')];

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-thread',
        },
        streamMode: 'values',
        version: 'v2' as const,
      };

      await run.processStream({ messages }, config);

      const finalMessages = run.getRunMessages();
      expect(finalMessages).toBeDefined();
      expect(finalMessages?.length).toBeGreaterThan(0);
    });
  });

  describe('enforceOriginalContentCap', () => {
    it('is a no-op when total chars are below the cap', () => {
      const map = new Map<number, string>([
        [0, 'a'.repeat(100)],
        [1, 'b'.repeat(200)],
      ]);
      enforceOriginalContentCap(map);
      expect(map.size).toBe(2);
      expect(map.get(0)?.length).toBe(100);
      expect(map.get(1)?.length).toBe(200);
    });

    it('evicts oldest entries (by Map insertion order) until under the cap', () => {
      const map = new Map<number, string>();
      // Insert 4 entries totaling well over the cap, in insertion order
      // 0, 1, 2, 3.  Each entry is roughly 700_000 chars (>1/3 of cap).
      const big = 'x'.repeat(700_000);
      map.set(0, big);
      map.set(1, big);
      map.set(2, big);
      map.set(3, big);

      // 4 * 700_000 = 2_800_000 > 2_000_000 cap.  Eviction should drop
      // the oldest entry (key 0) — leaving 3 * 700_000 = 2_100_000 still
      // > cap, so key 1 is also dropped — 2 * 700_000 = 1_400_000 ≤ cap.
      enforceOriginalContentCap(map);
      expect(map.has(0)).toBe(false);
      expect(map.has(1)).toBe(false);
      expect(map.has(2)).toBe(true);
      expect(map.has(3)).toBe(true);
    });

    it('exposes the cap as a constant for callers', () => {
      expect(ORIGINAL_CONTENT_MAX_CHARS).toBe(2_000_000);
    });
  });
});

describe('sanitizeOrphanToolBlocks', () => {
  it('strips orphan tool_use blocks from AI messages with no matching ToolMessage', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('Hello'),
      new AIMessage({
        content: [
          { type: 'text', text: 'Let me check.' },
          { type: 'tool_use', id: 'tool_1', name: 'calc', input: { x: 1 } },
        ],
        tool_calls: [
          { id: 'tool_1', name: 'calc', args: { x: 1 }, type: 'tool_call' },
        ],
      }),
      // No ToolMessage for tool_1 — orphan
    ];

    const result = sanitizeOrphanToolBlocks(messages);
    // The stripped AI message was the last message → dropped (incomplete tool call)
    expect(result).toHaveLength(1);
    expect(result[0].getType()).toBe('human');
  });

  it('drops orphan ToolMessages whose AI message is missing', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('Hello'),
      new ToolMessage({
        content: 'result',
        tool_call_id: 'tool_orphan',
        name: 'calc',
      }),
      new AIMessage('Some response'),
    ];

    const result = sanitizeOrphanToolBlocks(messages);
    expect(result).toHaveLength(2); // HumanMessage + AIMessage, orphan ToolMessage dropped
    expect(result[0].getType()).toBe('human');
    expect(result[1].getType()).toBe('ai');
  });

  it('preserves correctly paired tool_use and ToolMessages', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('Compute 1+1'),
      new AIMessage({
        content: [
          { type: 'text', text: 'Let me calculate.' },
          { type: 'tool_use', id: 'tool_a', name: 'calc', input: { x: 1 } },
        ],
        tool_calls: [
          { id: 'tool_a', name: 'calc', args: { x: 1 }, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: '2',
        tool_call_id: 'tool_a',
        name: 'calc',
      }),
      new AIMessage('The answer is 2.'),
    ];

    const result = sanitizeOrphanToolBlocks(messages);
    expect(result).toHaveLength(4); // All messages preserved
    expect(result.map((m) => m.getType())).toEqual([
      'human',
      'ai',
      'tool',
      'ai',
    ]);
  });

  it('drops AI message entirely when it only contained orphan tool_use blocks', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('Do something'),
      new AIMessage({
        content: [{ type: 'tool_use', id: 'tool_x', name: 'run', input: {} }],
        tool_calls: [
          { id: 'tool_x', name: 'run', args: {}, type: 'tool_call' },
        ],
      }),
      // No ToolMessage for tool_x
    ];

    const result = sanitizeOrphanToolBlocks(messages);
    // The AI message had only tool_use blocks, stripping them leaves nothing → dropped
    expect(result).toHaveLength(1);
    expect(result[0].getType()).toBe('human');
  });

  it('keeps stripped AI message in the middle but drops stripped trailing AI', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('First question'),
      new AIMessage({
        content: [
          { type: 'text', text: 'Let me use two tools.' },
          { type: 'tool_use', id: 'tool_a', name: 'calc', input: { x: 1 } },
          {
            type: 'tool_use',
            id: 'tool_orphan',
            name: 'search',
            input: { q: 'test' },
          },
        ],
        tool_calls: [
          { id: 'tool_a', name: 'calc', args: { x: 1 }, type: 'tool_call' },
          {
            id: 'tool_orphan',
            name: 'search',
            args: { q: 'test' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: '42',
        tool_call_id: 'tool_a',
        name: 'calc',
      }),
      // No ToolMessage for tool_orphan, but conversation continues:
      new AIMessage({
        content: [{ type: 'text', text: 'Got the calc result.' }],
        tool_calls: [
          { id: 'tool_b', name: 'run', args: {}, type: 'tool_call' },
        ],
      }),
      // tool_b is also orphan → stripped, and this AI is last → dropped
    ];

    const result = sanitizeOrphanToolBlocks(messages);
    // message[1]: AI has tool_orphan stripped but tool_a kept → stays (middle, not trailing)
    // message[3]: AI has tool_b stripped, is trailing → dropped
    expect(result).toHaveLength(3); // HumanMessage, stripped AI (kept tool_a), ToolMessage
    const ai = result[1] as AIMessage;
    expect(ai.tool_calls).toHaveLength(1);
    expect(ai.tool_calls![0].id).toBe('tool_a');
    expect(result[2].getType()).toBe('tool');
  });

  it('keeps unmodified trailing AI message (no orphan tool_use)', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('Hello'),
      new ToolMessage({
        content: 'result',
        tool_call_id: 'tool_orphan',
        name: 'calc',
      }),
      new AIMessage('Final response without tool calls.'),
    ];

    const result = sanitizeOrphanToolBlocks(messages);
    // orphan ToolMessage dropped, trailing AI kept (was not stripped)
    expect(result).toHaveLength(2);
    expect(result[0].getType()).toBe('human');
    expect(result[1].getType()).toBe('ai');
  });

  it('preserves BaseMessage prototype on stripped AIMessage instances', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('Hello'),
      new AIMessage({
        content: [
          { type: 'text', text: 'Let me search and calculate.' },
          {
            type: 'tool_use',
            id: 'tool_a',
            name: 'search',
            input: { q: 'test' },
          },
          { type: 'tool_use', id: 'tool_b', name: 'calc', input: { x: 1 } },
        ],
        tool_calls: [
          {
            id: 'tool_a',
            name: 'search',
            args: { q: 'test' },
            type: 'tool_call' as const,
          },
          {
            id: 'tool_b',
            name: 'calc',
            args: { x: 1 },
            type: 'tool_call' as const,
          },
        ],
      }),
      new ToolMessage({ content: 'result', tool_call_id: 'tool_b' }),
      // No ToolMessage for tool_a — orphan
    ];

    const result = sanitizeOrphanToolBlocks(messages);
    // AI message should survive (tool_a stripped, tool_b kept)
    expect(result).toHaveLength(3);

    // Every output message must pass isBaseMessage and have getType()
    for (const msg of result) {
      expect(isBaseMessage(msg)).toBe(true);
      expect(typeof msg.getType()).toBe('string');
    }
    expect(result[1].getType()).toBe('ai');
    expect(result[1]).toBeInstanceOf(AIMessage);
  });

  it('preserves AIMessageChunk prototype on stripped messages', () => {
    // Simulate what happens in real graph execution: model returns AIMessageChunk,
    // state passes through LangGraph, sanitizeOrphanToolBlocks strips orphan server tools.
    const chunk = new AIMessageChunk({
      content: [
        { type: 'text', text: 'Searching...' },
        { type: 'tool_use', id: 'srvtoolu_1', name: 'web_search', input: '' },
        { type: 'tool_use', id: 'toolu_2', name: 'calculator', input: '2+2' },
      ],
      tool_call_chunks: [
        { id: 'srvtoolu_1', index: 0, name: 'web_search', args: '' },
        { id: 'toolu_2', index: 2, name: 'calculator', args: '2+2' },
      ],
    });

    const messages: BaseMessage[] = [
      new HumanMessage('Search and calculate'),
      chunk,
      new ToolMessage({ content: '4', tool_call_id: 'toolu_2' }),
      // No ToolMessage for srvtoolu_1 — server tool, orphan
    ];

    const result = sanitizeOrphanToolBlocks(messages);
    expect(result).toHaveLength(3);

    // The AIMessageChunk must retain its prototype so LangChain's
    // coerceMessageLikeToMessage recognizes it as a BaseMessage.
    const aiMsg = result[1];
    expect(isBaseMessage(aiMsg)).toBe(true);
    expect(typeof aiMsg.getType()).toBe('string');
    expect(aiMsg.getType()).toBe('ai');
  });

  it('preserves prototype on plain-object messages with duck-typed patching', () => {
    // Simulate deserialized messages that still have a prototype (e.g. from
    // LangGraph subgraph state transfer) but aren't class instances.
    const proto = { _getType: (): string => 'ai', getType: (): string => 'ai' };
    const plainAi = Object.create(proto);
    Object.assign(plainAi, {
      role: 'assistant',
      content: [
        { type: 'text', text: 'checking' },
        { type: 'tool_use', id: 'orphan_1', name: 'tool', input: {} },
      ],
      tool_calls: [
        { id: 'orphan_1', name: 'tool', args: {}, type: 'tool_call' },
      ],
    });

    const messages = [plainAi] as BaseMessage[];
    sanitizeOrphanToolBlocks(messages);

    // Stripped AI was trailing → dropped. But if we add a human after:
    const messages2 = [
      new HumanMessage('hi'),
      plainAi,
      new HumanMessage('follow up'),
    ] as BaseMessage[];
    const result2 = sanitizeOrphanToolBlocks(messages2);

    // The patched message in the middle must still have _getType from proto
    const middleMsg = result2[1];
    expect(typeof middleMsg._getType).toBe('function');
    expect(middleMsg._getType()).toBe('ai');
  });

  it('handles plain objects (non-BaseMessage instances) via duck typing', () => {
    // Simulate messages that have lost their class instances (LangGraph state serialization)
    const plainMessages = [
      { role: 'user', content: 'Hello', _type: 'human' },
      {
        role: 'assistant',
        _type: 'ai',
        content: [
          { type: 'text', text: 'Let me check.' },
          { type: 'tool_use', id: 'tool_1', name: 'calc', input: { x: 1 } },
        ],
        tool_calls: [
          { id: 'tool_1', name: 'calc', args: { x: 1 }, type: 'tool_call' },
        ],
      },
      // No ToolMessage for tool_1 — orphan
    ] as unknown as BaseMessage[];

    // Should not throw "getType is not a function"
    const result = sanitizeOrphanToolBlocks(plainMessages);
    // The stripped AI message was the last message → dropped (incomplete tool call)
    expect(result).toHaveLength(1);
  });
});

describe('prunedMemory ordering with thinking enabled', () => {
  it('messagesToRefine preserves chronological order when thinking search pops multiple messages', () => {
    const tokenCounter = createTestTokenCounter();
    const messages: BaseMessage[] = [
      new HumanMessage('Hello'),
      new AIMessage({
        content: [
          {
            type: ContentTypes.REASONING_CONTENT,
            reasoningText: {
              text: 'Thinking about navigation...',
              signature: 'sig1',
            },
          },
          { type: 'text', text: 'Navigating now.' },
        ],
        tool_calls: [
          {
            id: 'tc_nav',
            name: 'navigate',
            args: { url: 'about:blank' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'Navigated to about:blank.',
        tool_call_id: 'tc_nav',
        name: 'navigate',
      }),
      new AIMessage({
        content: [
          {
            type: ContentTypes.REASONING_CONTENT,
            reasoningText: {
              text: 'Now I will write code...',
              signature: 'sig2',
            },
          },
          { type: 'text', text: 'Running script.' },
        ],
        tool_calls: [
          {
            id: 'tc_eval',
            name: 'evaluate',
            args: { code: 'x'.repeat(5000) },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'y'.repeat(5000), // large tool result
        tool_call_id: 'tc_eval',
        name: 'evaluate',
      }),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    // Use a very tight budget so the backward iteration must prune messages
    // The thinking search will cause the loop to `continue` past the large ToolMessage
    const result = realGetMessagesWithinTokenLimit({
      messages,
      maxContextTokens: 200, // very tight
      indexTokenCountMap,
      thinkingEnabled: true,
      tokenCounter,
      reasoningType: ContentTypes.REASONING_CONTENT,
    });

    // The key assertion: messagesToRefine must be in chronological order.
    // AI(evaluate) at index 3 must come BEFORE ToolMessage(evaluate) at index 4.
    for (let i = 0; i < result.messagesToRefine.length - 1; i++) {
      const current = result.messagesToRefine[i];
      const next = result.messagesToRefine[i + 1];
      // A ToolMessage should never come before its AI message
      if (next.getType() === 'ai' && current.getType() === 'tool') {
        const toolId = (current as ToolMessage).tool_call_id;
        const aiToolIds = ((next as AIMessage).tool_calls ?? []).map(
          (tc) => tc.id
        );
        expect(aiToolIds).not.toContain(toolId);
      }
    }

    // Verify the specific ordering: if both AI(evaluate) and Tool(evaluate) are in
    // messagesToRefine, AI must come first.
    const evalAiIdx = result.messagesToRefine.findIndex(
      (m) =>
        m.getType() === 'ai' &&
        ((m as AIMessage).tool_calls ?? []).some((tc) => tc.id === 'tc_eval')
    );
    const evalToolIdx = result.messagesToRefine.findIndex(
      (m) =>
        m.getType() === 'tool' && (m as ToolMessage).tool_call_id === 'tc_eval'
    );
    if (evalAiIdx >= 0 && evalToolIdx >= 0) {
      expect(evalAiIdx).toBeLessThan(evalToolIdx);
    }
  });
});

describe('thinking enabled — tail tool_use without a thinking block (issue #115)', () => {
  it('does not throw when the trailing AI message issued a tool call without a thinking block', () => {
    const tokenCounter = createTestTokenCounter();
    const messages: BaseMessage[] = [
      new HumanMessage('first turn'),
      new AIMessage({
        content: [
          {
            type: ContentTypes.THINKING,
            thinking: 'thinking about the first response',
            signature: 'sig0',
          },
          { type: 'text', text: 'first reply' },
        ],
      }),
      new HumanMessage('please read this doc and tell me X'),
      // Anthropic may emit a tool_use without an accompanying thinking block —
      // valid API behavior that the pruner must tolerate.
      new AIMessage({
        content: [
          {
            type: 'tool_use',
            id: 'tc_get_doc',
            name: 'get_doc_content',
            input: { docId: 'abc' },
          },
        ],
        tool_calls: [
          {
            id: 'tc_get_doc',
            name: 'get_doc_content',
            args: { docId: 'abc' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'a'.repeat(8000), // huge tool result that pushes us past budget
        tool_call_id: 'tc_get_doc',
        name: 'get_doc_content',
      }),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    expect(() =>
      realGetMessagesWithinTokenLimit({
        messages,
        maxContextTokens: 200, // tight budget so pruning actually runs
        indexTokenCountMap,
        thinkingEnabled: true,
        tokenCounter,
        reasoningType: ContentTypes.THINKING,
      })
    ).not.toThrow();
  });

  it('returns a prunable context for the [AI tool_use, Tool] tail without a thinking block', () => {
    const tokenCounter = createTestTokenCounter();
    const messages: BaseMessage[] = [
      new HumanMessage('please read this doc'),
      new AIMessage({
        content: [
          {
            type: 'tool_use',
            id: 'tc_get_doc',
            name: 'get_doc_content',
            input: { docId: 'abc' },
          },
        ],
        tool_calls: [
          {
            id: 'tc_get_doc',
            name: 'get_doc_content',
            args: { docId: 'abc' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'b'.repeat(6000),
        tool_call_id: 'tc_get_doc',
        name: 'get_doc_content',
      }),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    const result = realGetMessagesWithinTokenLimit({
      messages,
      maxContextTokens: 200,
      indexTokenCountMap,
      thinkingEnabled: true,
      tokenCounter,
      reasoningType: ContentTypes.THINKING,
    });

    expect(result.context).toBeDefined();
    expect(result.messagesToRefine.length).toBeGreaterThan(0);
    expect(result.thinkingStartIndex).toBeUndefined();
  });

  it('handles consecutive tool calls without any thinking block in the tail', () => {
    const tokenCounter = createTestTokenCounter();
    const messages: BaseMessage[] = [
      new HumanMessage('do two things'),
      new AIMessage({
        content: [
          {
            type: 'tool_use',
            id: 'tc_1',
            name: 'tool_a',
            input: { x: 1 },
          },
        ],
        tool_calls: [
          { id: 'tc_1', name: 'tool_a', args: { x: 1 }, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: 'result_a',
        tool_call_id: 'tc_1',
        name: 'tool_a',
      }),
      new AIMessage({
        content: [
          {
            type: 'tool_use',
            id: 'tc_2',
            name: 'tool_b',
            input: { y: 2 },
          },
        ],
        tool_calls: [
          { id: 'tc_2', name: 'tool_b', args: { y: 2 }, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: 'd'.repeat(6000),
        tool_call_id: 'tc_2',
        name: 'tool_b',
      }),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    const result = realGetMessagesWithinTokenLimit({
      messages,
      maxContextTokens: 200,
      indexTokenCountMap,
      thinkingEnabled: true,
      tokenCounter,
      reasoningType: ContentTypes.THINKING,
    });
    expect(result.thinkingStartIndex).toBeUndefined();
  });

  it('honors prior runThinkingStartIndex carry-over when the next call has a no-thinking tail', () => {
    // First call's tight budget forces pruning, which makes the closure
    // record the AI(thinking) message's index in runThinkingStartIndex.
    // Second call's tail is AI(tool_use) without a thinking block; the
    // pre-loaded thinkingBlock from the carry-over keeps the new guard
    // dormant and the existing reattachment path runs. Verifies the fix
    // doesn't disturb the carry-over interaction.
    const tokenCounter = createTestTokenCounter();
    const firstTurn: BaseMessage[] = [
      new HumanMessage('h'.repeat(120)),
      new AIMessage({
        content: [
          {
            type: ContentTypes.THINKING,
            thinking: 'planning the response',
            signature: 'sig-prior',
          },
          { type: 'text', text: 'hi' },
        ],
      }),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < firstTurn.length; i++) {
      indexTokenCountMap[i] = tokenCounter(firstTurn[i]);
    }

    const pruneMessages = createPruneMessages({
      maxTokens: 68,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap,
      thinkingEnabled: true,
      reserveRatio: 0,
    });

    const firstResult = pruneMessages({ messages: firstTurn });
    expect(firstResult.messagesToRefine?.length).toBeGreaterThan(0);
    expect(firstResult.context.some((m) => m.getType() === 'ai')).toBe(true);

    const secondTurn: BaseMessage[] = [
      ...firstTurn,
      new HumanMessage('please read the doc'),
      new AIMessage({
        content: [
          {
            type: 'tool_use',
            id: 'tc_get_doc',
            name: 'get_doc_content',
            input: { docId: 'abc' },
          },
        ],
        tool_calls: [
          {
            id: 'tc_get_doc',
            name: 'get_doc_content',
            args: { docId: 'abc' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'e'.repeat(40),
        tool_call_id: 'tc_get_doc',
        name: 'get_doc_content',
      }),
    ];

    let secondResult: ReturnType<typeof pruneMessages> | undefined;
    expect(() => {
      secondResult = pruneMessages({ messages: secondTurn });
    }).not.toThrow();

    // Carry-over reattachment: even though the trailing AI(tool_use) has
    // no thinking block of its own, the closure's runThinkingStartIndex
    // points at the prior AI(thinking) and that block gets prepended to
    // the surviving AI message in context.
    const trailingAi = secondResult!.context.find(
      (m) =>
        m.getType() === 'ai' &&
        Array.isArray(m.content) &&
        (m.content as t.ExtendedMessageContent[]).some(
          (c) => typeof c === 'object' && c.type === 'tool_use'
        )
    );
    expect(trailingAi).toBeDefined();
    expect(
      (trailingAi!.content as t.ExtendedMessageContent[]).some(
        (c) => typeof c === 'object' && c.type === ContentTypes.THINKING
      )
    ).toBe(true);
  });

  it('integrates with ensureThinkingBlockInMessages so the API-bound payload stays valid', () => {
    // Models the full Graph.ts pipeline: pruner runs first, then
    // ensureThinkingBlockInMessages on the pruned context. The pruner used
    // to throw on the issue #115 tail; with the fix it returns the
    // messages, and ensureThinkingBlockInMessages folds the orphan
    // AI(tool_use)+Tool tail into a `[Previous agent context]`
    // HumanMessage. The Tool size is tuned so the trailing sequence
    // actually survives pruning — otherwise the assertions would be
    // vacuous.
    const tokenCounter = createTestTokenCounter();
    const messages: BaseMessage[] = [
      new HumanMessage('please read this doc and tell me X'),
      new AIMessage({
        content: [
          {
            type: 'tool_use',
            id: 'tc_get_doc',
            name: 'get_doc_content',
            input: { docId: 'abc' },
          },
        ],
        tool_calls: [
          {
            id: 'tc_get_doc',
            name: 'get_doc_content',
            args: { docId: 'abc' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'f'.repeat(100),
        tool_call_id: 'tc_get_doc',
        name: 'get_doc_content',
      }),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    const pruneResult = realGetMessagesWithinTokenLimit({
      messages,
      maxContextTokens: 300,
      indexTokenCountMap,
      thinkingEnabled: true,
      tokenCounter,
      reasoningType: ContentTypes.THINKING,
    });

    expect(pruneResult.context.length).toBe(3);

    const finalMessages = ensureThinkingBlockInMessages(
      pruneResult.context,
      Providers.ANTHROPIC
    );

    // ensureThinkingBlockInMessages should fold the orphan AI(tool_use)+Tool
    // into a synthetic HumanMessage carrying the `[Previous agent context]`
    // marker, leaving no AI(tool_use) in the outgoing payload.
    expect(finalMessages.length).toBe(2);
    expect(finalMessages[0]).toBeInstanceOf(HumanMessage);
    expect(finalMessages[1]).toBeInstanceOf(HumanMessage);

    const folded = finalMessages[1] as HumanMessage;
    const foldedContent = folded.content;
    const foldedText = Array.isArray(foldedContent)
      ? (foldedContent as t.ExtendedMessageContent[])
        .filter((c) => typeof c === 'object' && c.type === 'text')
        .map((c) => String(c.text ?? ''))
        .join('\n')
      : String(foldedContent);
    expect(foldedText).toContain('[Previous agent context]');

    const hasOrphanToolUse = finalMessages.some((m) => {
      if (m.getType() !== 'ai') {
        return false;
      }
      const content = (m as AIMessage).content;
      if (!Array.isArray(content)) {
        return false;
      }
      return content.some(
        (c) => typeof c === 'object' && c.type === 'tool_use'
      );
    });
    expect(hasOrphanToolUse).toBe(false);
  });

  it('still preserves the thinking block when the trailing AI message has one', () => {
    const tokenCounter = createTestTokenCounter();
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: ContentTypes.THINKING,
            thinking: 'older thinking',
            signature: 'sig-old',
          },
          { type: 'text', text: 'older reply' },
        ],
      }),
      new HumanMessage('please read this doc'),
      new AIMessage({
        content: [
          {
            type: ContentTypes.THINKING,
            thinking: 'I will fetch the doc',
            signature: 'sig-new',
          },
          {
            type: 'tool_use',
            id: 'tc_get_doc',
            name: 'get_doc_content',
            input: { docId: 'abc' },
          },
        ],
        tool_calls: [
          {
            id: 'tc_get_doc',
            name: 'get_doc_content',
            args: { docId: 'abc' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'c'.repeat(6000),
        tool_call_id: 'tc_get_doc',
        name: 'get_doc_content',
      }),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    const result = realGetMessagesWithinTokenLimit({
      messages,
      maxContextTokens: 200,
      indexTokenCountMap,
      thinkingEnabled: true,
      tokenCounter,
      reasoningType: ContentTypes.THINKING,
    });

    expect(result.thinkingStartIndex).toBeGreaterThanOrEqual(0);
  });
});

describe('thinking enabled — non-Anthropic reasoning_content blocks (issue #191)', () => {
  it('locates a trailing reasoning_content block even when reasoningType defaults to THINKING (DeepSeek/Qwen)', () => {
    // DeepSeek-R1 and DashScope/Qwen-thinking route through the non-Bedrock
    // branch, so the caller passes reasoningType: THINKING — but their blocks
    // are tagged `reasoning_content` and are not normalized upstream. With a
    // system prompt at index 0 and an all-AI/tool tail, the consume loop never
    // pops a human to clear thinkingEndIndex (the issue #116 escape hatch), so
    // searching only for `thinking` missed the present block and threw a fatal
    // that permanently bricked the thread. The pruner must find the block by
    // its actual shape instead.
    const tokenCounter = createTestTokenCounter();
    const messages: BaseMessage[] = [
      new SystemMessage('you are a helpful assistant'),
      new AIMessage({
        content: [
          {
            type: ContentTypes.REASONING_CONTENT,
            reasoningText: {
              text: 'I will fetch the doc',
              signature: 'sig-new',
            },
          },
          {
            type: 'tool_use',
            id: 'tc_get_doc',
            name: 'get_doc_content',
            input: { docId: 'abc' },
          },
        ],
        tool_calls: [
          {
            id: 'tc_get_doc',
            name: 'get_doc_content',
            args: { docId: 'abc' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'c'.repeat(6000),
        tool_call_id: 'tc_get_doc',
        name: 'get_doc_content',
      }),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    let result: ReturnType<typeof realGetMessagesWithinTokenLimit> | undefined;
    expect(() => {
      result = realGetMessagesWithinTokenLimit({
        messages,
        maxContextTokens: 200,
        indexTokenCountMap,
        thinkingEnabled: true,
        tokenCounter,
        reasoningType: ContentTypes.THINKING,
      });
    }).not.toThrow();

    // thinkingStartIndex is only set when the reasoning block is actually
    // located — isolating the find fix (B) from the graceful-degradation
    // safety net (C), which would swallow the throw without finding anything.
    expect(result!.thinkingStartIndex).toBeGreaterThanOrEqual(0);
  });

  it('does not throw when a carried-over thinking sequence has no locatable block', () => {
    // Models a stale runThinkingStartIndex carry-over pointing at an assistant
    // message that has no reasoning block. The pruner cannot find a block, but
    // a trailing AI/tool sequence keeps thinkingEndIndex set, so it used to
    // reach the fatal "no thinking block found" throw. Defense in depth: a
    // misconfiguration upstream of the pruner must not be able to brick the
    // thread — degrade to the partially-pruned context instead.
    const tokenCounter = createTestTokenCounter();
    const messages: BaseMessage[] = [
      new HumanMessage('h'.repeat(100)),
      new AIMessage({
        content: [{ type: 'text', text: 'a reply with no reasoning block' }],
      }),
      new HumanMessage('please read the doc'),
      new AIMessage({
        content: [
          {
            type: 'tool_use',
            id: 'tc_get_doc',
            name: 'get_doc_content',
            input: { docId: 'abc' },
          },
        ],
        tool_calls: [
          {
            id: 'tc_get_doc',
            name: 'get_doc_content',
            args: { docId: 'abc' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'x'.repeat(150),
        tool_call_id: 'tc_get_doc',
        name: 'get_doc_content',
      }),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    let result: ReturnType<typeof realGetMessagesWithinTokenLimit> | undefined;
    expect(() => {
      result = realGetMessagesWithinTokenLimit({
        messages,
        maxContextTokens: 200,
        indexTokenCountMap,
        thinkingEnabled: true,
        tokenCounter,
        thinkingStartIndex: 1,
        reasoningType: ContentTypes.THINKING,
      });
    }).not.toThrow();

    expect(result!.context.length).toBeGreaterThan(0);
    expect(result!.messagesToRefine.length).toBeGreaterThan(0);
    // The stale carried-over index must NOT be propagated: createPruneMessages
    // persists it as runThinkingStartIndex, and a stale value would suppress
    // the trailing scan on later turns and miss a real reasoning block.
    expect(result!.thinkingStartIndex).toBeUndefined();
  });

  it('does not match an Anthropic thinking block for a Bedrock (reasoning_content) run', () => {
    // The cross-type fallback is one-directional: REASONING_CONTENT (Bedrock)
    // must not match a `thinking` block, since the Bedrock input converter
    // rejects `thinking` blocks and reattaching one would break the request.
    const tokenCounter = createTestTokenCounter();
    const messages: BaseMessage[] = [
      new SystemMessage('you are a helpful assistant'),
      new AIMessage({
        content: [
          {
            type: ContentTypes.THINKING,
            thinking: 'inherited Anthropic-style reasoning',
            signature: 'sig-anthropic',
          },
          {
            type: 'tool_use',
            id: 'tc_get_doc',
            name: 'get_doc_content',
            input: { docId: 'abc' },
          },
        ],
        tool_calls: [
          {
            id: 'tc_get_doc',
            name: 'get_doc_content',
            args: { docId: 'abc' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'c'.repeat(6000),
        tool_call_id: 'tc_get_doc',
        name: 'get_doc_content',
      }),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    let result: ReturnType<typeof realGetMessagesWithinTokenLimit> | undefined;
    expect(() => {
      result = realGetMessagesWithinTokenLimit({
        messages,
        maxContextTokens: 200,
        indexTokenCountMap,
        thinkingEnabled: true,
        tokenCounter,
        reasoningType: ContentTypes.REASONING_CONTENT,
      });
    }).not.toThrow();

    // The thinking block is intentionally not located for a Bedrock run, so no
    // index is reported and nothing gets reattached.
    expect(result!.thinkingStartIndex).toBeUndefined();
  });
});

describe('context fading of completed tool calls (issue: re-executed side effects)', () => {
  const lookupCalls = (count: number) =>
    Array.from({ length: count }, (_, i) => ({
      id: `call_lookup_${i}`,
      name: 'lookup',
      args: { query: `q${i}` },
      type: 'tool_call' as const,
    }));

  function fadingTierFor(messages: BaseMessage[]): t.FadingTier {
    const tokenCounter = createTestTokenCounter();
    const indexTokenCountMap: Record<string, number | undefined> = {};
    messages.forEach((message, index) => {
      indexTokenCountMap[index] = tokenCounter(message);
    });
    const pruneMessages = createPruneMessages({
      maxTokens: 200_000,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap,
      /** Mirrors the report: an inflated instruction estimate leaves ~35K. */
      getInstructionTokens: () => 154_703,
    });
    return pruneMessages({ messages }).fadingTier;
  }

  it('does not let a reloaded turn merged into one message drive the tier down', () => {
    const calls = lookupCalls(8);
    const previousTurn: BaseMessage[] = [
      new HumanMessage('Build the report'),
      /** Storage folds a turn's steps into one assistant message on reload. */
      new AIMessage({ content: '', tool_calls: calls }),
      ...calls.map(
        (call) =>
          new ToolMessage({
            content: 'ok',
            tool_call_id: call.id,
            name: 'lookup',
          })
      ),
    ];

    const reloaded = fadingTierFor([
      ...previousTurn,
      new HumanMessage('Send it'),
    ]);
    const liveWideTurn = fadingTierFor([
      new HumanMessage('Send it'),
      ...previousTurn.slice(1),
    ]);

    expect(reloaded.budgetTokens).toBeGreaterThan(liveWideTurn.budgetTokens);
  });

  it('re-caps a legacy truncation envelope into one that says the call completed', () => {
    const legacyArgs = {
      _truncated: `… [truncated]
${'{"subject":"Report","body":"'}${'x'.repeat(400)}`,
      _originalChars: 3810,
    };
    const message = new AIMessage({
      content: '',
      tool_calls: [
        {
          id: 'call_send',
          name: 'send_email',
          args: legacyArgs,
          type: 'tool_call',
        },
      ],
    });

    const [projected] = projectToolCallInputs([message], 200) as AIMessage[];
    const args = projected.tool_calls?.[0].args as Record<string, unknown>;

    expect(Object.keys(args)).toEqual([
      '_note',
      '_originalChars',
      '_inputPrefix',
    ]);
    expect(args._note).toBe(TOOL_INPUT_ELISION_NOTE);
    expect(args._originalChars).toBe(3810);
    expect(String(args._inputPrefix)).toMatch(/^\{"subject":"Report"/);
    expect(JSON.stringify(args)).not.toContain('truncated');
    expect(JSON.stringify(args).length).toBeLessThanOrEqual(200);
  });
});
