// src/messages/toolDiscovery.ts
import { AIMessageChunk, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { findLastIndex } from './core';
import { Constants } from '@/common';

type ToolSearchArtifact = {
  tool_references?: Array<{ tool_name: string }>;
};

/**
 * Extracts discovered tool names from tool search results in the current turn.
 * Only processes tool search messages after the latest AI message with tool calls.
 *
 * Similar pattern to formatArtifactPayload - finds relevant messages efficiently
 * by identifying the latest AI parent and only processing subsequent tool messages.
 *
 * @param messages - All messages in the conversation
 * @returns Array of discovered tool names (empty if no new discoveries)
 */
export function extractToolDiscoveries(messages: BaseMessage[]): string[] {
  const lastMessage = messages[messages.length - 1];
  if (!(lastMessage instanceof ToolMessage)) return [];

  // Find the latest AIMessage with tool_calls that this tool message belongs to
  const latestAIParentIndex = findLastIndex(
    messages,
    (msg) =>
      (msg instanceof AIMessageChunk &&
        (msg.tool_calls?.length ?? 0) > 0 &&
        msg.tool_calls?.some((tc) => tc.id === lastMessage.tool_call_id)) ??
      false
  );

  if (latestAIParentIndex === -1) return [];

  // Collect tool_call_ids from the AI message
  const aiMessage = messages[latestAIParentIndex] as AIMessageChunk;
  const toolCallIds = new Set(aiMessage.tool_calls?.map((tc) => tc.id) ?? []);

  // Only process tool search results after the AI message that belong to this turn
  const discoveredNames: string[] = [];
  for (let i = latestAIParentIndex + 1; i < messages.length; i++) {
    const msg = messages[i];
    if (!(msg instanceof ToolMessage)) continue;
    if (
      msg.name !== Constants.TOOL_SEARCH &&
      msg.name !== Constants.SKILL_TOOL
    ) {
      continue;
    }
    if (!toolCallIds.has(msg.tool_call_id)) continue;

    // This is a tool search result from the current turn
    if (typeof msg.artifact === 'object' && msg.artifact != null) {
      const artifact = msg.artifact as ToolSearchArtifact;
      if (artifact.tool_references && artifact.tool_references.length > 0) {
        for (const ref of artifact.tool_references) {
          discoveredNames.push(ref.tool_name);
        }
      }
    }
  }

  return discoveredNames;
}

/**
 * Checks if the current turn has any tool search results.
 * Quick check to avoid full extraction when not needed.
 */
export function hasToolSearchInCurrentTurn(messages: BaseMessage[]): boolean {
  const lastMessage = messages[messages.length - 1];
  if (!(lastMessage instanceof ToolMessage)) return false;

  // Find the latest AIMessage with tool_calls
  const latestAIParentIndex = findLastIndex(
    messages,
    (msg) =>
      (msg instanceof AIMessageChunk &&
        (msg.tool_calls?.length ?? 0) > 0 &&
        msg.tool_calls?.some((tc) => tc.id === lastMessage.tool_call_id)) ??
      false
  );

  if (latestAIParentIndex === -1) return false;

  const aiMessage = messages[latestAIParentIndex] as AIMessageChunk;
  const toolCallIds = new Set(aiMessage.tool_calls?.map((tc) => tc.id) ?? []);

  // Check if any tool search results exist after the AI message
  for (let i = latestAIParentIndex + 1; i < messages.length; i++) {
    const msg = messages[i];
    if (
      msg instanceof ToolMessage &&
      (msg.name === Constants.TOOL_SEARCH ||
        msg.name === Constants.SKILL_TOOL) &&
      toolCallIds.has(msg.tool_call_id)
    ) {
      return true;
    }
  }

  return false;
}
