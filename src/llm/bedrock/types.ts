/**
 * Type definitions for Bedrock Converse utilities.
 */
import type {
  Message as BedrockMessage,
  SystemContentBlock as BedrockSystemContentBlock,
  ContentBlock as BedrockContentBlock,
  ConverseResponse,
  ContentBlockDeltaEvent,
  ConverseStreamMetadataEvent,
  ContentBlockStartEvent,
  ReasoningContentBlock,
  ReasoningContentBlockDelta,
} from '@aws-sdk/client-bedrock-runtime';

/**
 * Reasoning content block type for LangChain messages.
 */
export interface MessageContentReasoningBlock {
  type: 'reasoning_content';
  [key: string]: unknown;
  reasoningText?: {
    text?: string;
    signature?: string;
  };
  redactedContent?: string;
}

export interface MessageContentReasoningBlockReasoningTextPartial {
  type: 'reasoning_content';
  reasoningText: {
    text?: string;
    signature?: string;
  };
}

export interface MessageContentReasoningBlockRedacted {
  type: 'reasoning_content';
  redactedContent: string;
}

export type {
  BedrockMessage,
  BedrockSystemContentBlock,
  BedrockContentBlock,
  ConverseResponse,
  ContentBlockDeltaEvent,
  ConverseStreamMetadataEvent,
  ContentBlockStartEvent,
  ReasoningContentBlock,
  ReasoningContentBlockDelta,
};
