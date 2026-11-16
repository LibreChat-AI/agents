import type {
  BaseMessage,
  MessageContentComplex,
} from '@langchain/core/messages';
import {
  GEMINI_PARTS,
  isGeminiReplayParts,
} from '@/llm/google/utils/signatures';
import { ContentTypes } from '@/common';
import { cloneMessage } from './cache';

/**
 * Whether {@link formatContentStrings} will flatten this message's content:
 * a human/ai/system message whose content is an array of text-only blocks.
 */
export const isLegacyConvertible = (message: BaseMessage): boolean => {
  const messageType = message.getType();
  const isValidMessage =
    messageType === 'human' || messageType === 'ai' || messageType === 'system';
  if (!isValidMessage) {
    return false;
  }
  if (!Array.isArray(message.content)) {
    return false;
  }
  return message.content.every((block) => block.type === ContentTypes.TEXT);
};

/** Joins the text of {@link isLegacyConvertible} content blocks into the exact
 *  string {@link formatContentStrings} has always produced for them. */
export const flattenLegacyContent = (
  blocks: MessageContentComplex[]
): string => {
  const content = blocks.reduce((acc, curr) => {
    if (curr.type === ContentTypes.TEXT) {
      return `${acc}${curr[ContentTypes.TEXT] ?? ''}\n`;
    }
    return acc;
  }, '');
  return content.trim();
};

/**
 * Formats an array of messages for LangChain, making sure all content fields are strings
 * @param {Array<HumanMessage | AIMessage | SystemMessage | ToolMessage>} payload - The array of messages to format.
 * @returns {Array<HumanMessage | AIMessage | SystemMessage | ToolMessage>} - The array of formatted LangChain messages, including ToolMessages for tool calls.
 */
export const formatContentStrings = (
  payload: Array<BaseMessage>
): Array<BaseMessage> => {
  // Create a new array to store the processed messages
  const result: Array<BaseMessage> = [];

  for (const message of payload) {
    if (!isLegacyConvertible(message)) {
      result.push(message);
      continue;
    }

    const blocks = message.content as MessageContentComplex[];
    const preserveNativeText =
      message.getType() === 'ai' &&
      isGeminiReplayParts(message.additional_kwargs[GEMINI_PARTS]);
    const content = preserveNativeText
      ? blocks
        .map((block) => ('text' in block && typeof block.text === 'string' ? block.text : ''))
        .join('')
      : flattenLegacyContent(blocks);
    result.push(cloneMessage(message, content));
  }

  return result;
};
