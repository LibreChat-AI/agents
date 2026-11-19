import { createHash } from 'node:crypto';
import { AIMessageChunk, isAIMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { GeminiPart } from '@langchain/google-common';
import type { Part } from '@google/generative-ai';

export const GEMINI_PARTS = '__gemini_parts__';
export const GEMINI_REPLAY_UNSUPPORTED = '__gemini_replay_unsupported__';
export const GEMINI_SIGNATURES = '__gemini_function_call_thought_signatures__';

export type GeminiReplayPart =
  | {
      type: 'text';
      length: number;
      hash: string;
      thought?: boolean;
      thoughtSignature?: string;
    }
  | {
      type: 'function_call';
      id: string;
      hash: string;
      thoughtSignature?: string;
    };

function hashText(text: string): string {
  return createHash('sha256').update(text).digest('hex');
}

export function isGeminiReplayParts(
  value: unknown
): value is GeminiReplayPart[] {
  return (
    Array.isArray(value) &&
    Array.from(value).every((value: unknown) => {
      if (value == null || typeof value !== 'object') return false;
      const part = value as Partial<GeminiReplayPart>;
      if (
        part.thoughtSignature !== undefined &&
        typeof part.thoughtSignature !== 'string'
      )
        return false;
      if (part.type === 'function_call')
        return (
          typeof part.id === 'string' &&
          part.id !== '' &&
          typeof part.hash === 'string' &&
          /^[a-f0-9]{64}$/.test(part.hash) &&
          Object.keys(part).every((key) =>
            ['type', 'id', 'hash', 'thoughtSignature'].includes(key)
          )
        );
      return (
        part.type === 'text' &&
        typeof part.length === 'number' &&
        Number.isSafeInteger(part.length) &&
        part.length >= 0 &&
        typeof part.hash === 'string' &&
        /^[a-f0-9]{64}$/.test(part.hash) &&
        (part.thought === undefined || typeof part.thought === 'boolean') &&
        Object.keys(part).every((key) =>
          ['type', 'length', 'hash', 'thought', 'thoughtSignature'].includes(
            key
          )
        )
      );
    })
  );
}

/** Captures native boundaries without duplicating prose. Known Gemini 1/2 models retain their existing signature formats for cross-model compatibility. */
export function captureGeminiReplayParts(
  message: BaseMessage,
  parts: readonly (Part | GeminiPart)[],
  model?: string
): void {
  if (!isAIMessage(message)) return;
  const sourceModel = model?.toLowerCase().split('/').pop() ?? '';
  if (/^gemini-(?:[12](?:[.-]|$)|pro(?:-|$))/.test(sourceModel)) return;
  message.additional_kwargs[GEMINI_REPLAY_UNSUPPORTED] = true;
  const replay: GeminiReplayPart[] = [];
  const signatures: Record<string, string> = {};
  let callIndex = 0;
  let reasoning = '';
  for (const part of parts) {
    const thoughtSignature =
      'thoughtSignature' in part && typeof part.thoughtSignature === 'string'
        ? part.thoughtSignature
        : undefined;
    const signature = thoughtSignature != null ? { thoughtSignature } : {};
    const thought =
      'thought' in part && typeof part.thought === 'boolean'
        ? part.thought
        : undefined;
    if ('text' in part && typeof part.text === 'string') {
      replay.push({
        type: 'text',
        length: part.text.length,
        hash: hashText(part.text),
        ...(thought != null ? { thought } : {}),
        ...signature,
      });
      if (thought === true) reasoning += part.text;
      continue;
    }
    if (!('functionCall' in part) || part.functionCall == null) continue;
    const index = callIndex++;
    const call = message.tool_calls?.[index];
    if (call?.id == null || call.id === '') continue;
    const nativeId =
      'id' in part.functionCall && typeof part.functionCall.id === 'string'
        ? part.functionCall.id
        : undefined;
    if (nativeId != null && nativeId !== '') {
      call.id = nativeId;
      if (
        message instanceof AIMessageChunk &&
        Array.isArray(message.tool_call_chunks) &&
        index < message.tool_call_chunks.length
      ) {
        message.tool_call_chunks[index].id = nativeId;
      }
    }
    replay.push({
      type: 'function_call',
      id: call.id,
      hash: hashText(JSON.stringify({ name: call.name, args: call.args })),
      ...signature,
    });
    if (thoughtSignature != null) signatures[call.id] = thoughtSignature;
  }
  message.additional_kwargs[GEMINI_SIGNATURES] = signatures;
  if (replay.length !== parts.length) return;
  message.additional_kwargs[GEMINI_PARTS] = replay;
  const hasThinkingContent =
    Array.isArray(message.content) &&
    message.content.some((block) => block.type === 'thinking');
  if (reasoning !== '' && !hasThinkingContent)
    message.additional_kwargs.reasoning = reasoning;
  delete message.additional_kwargs[GEMINI_REPLAY_UNSUPPORTED];
}

/** Returns original parts only when the current message still matches every text hash and call id. */
export function restoreGeminiReplayParts(
  message: BaseMessage,
  replay: unknown = message.additional_kwargs[GEMINI_PARTS]
): GeminiPart[] | undefined {
  if (
    message.additional_kwargs[GEMINI_REPLAY_UNSUPPORTED] === true ||
    !isAIMessage(message) ||
    !isGeminiReplayParts(replay)
  )
    return undefined;
  let text = '';
  let contentReasoning: string | undefined;
  if (typeof message.content === 'string') text = message.content;
  else
    for (const block of message.content) {
      if (block.type === 'text' && typeof block.text === 'string')
        text += block.text;
      else if (block.type === 'thinking' && typeof block.thinking === 'string')
        contentReasoning = (contentReasoning ?? '') + block.thinking;
      else if (
        block.type === 'reasoning' &&
        typeof block.reasoning === 'string'
      )
        contentReasoning = (contentReasoning ?? '') + block.reasoning;
      else if (!('functionCall' in block)) return undefined;
    }
  let reasoning = contentReasoning ?? '';
  if (typeof message.additional_kwargs.reasoning_content === 'string')
    reasoning = message.additional_kwargs.reasoning_content;
  if (typeof message.additional_kwargs.reasoning === 'string')
    reasoning = message.additional_kwargs.reasoning;
  if (contentReasoning != null && contentReasoning !== reasoning)
    return undefined;
  const calls = new Map(
    (message.tool_calls ?? []).map((call) => [call.id, call])
  );
  const parts: GeminiPart[] = [];
  let textOffset = 0;
  let reasoningOffset = 0;
  for (const descriptor of replay) {
    const signature =
      descriptor.thoughtSignature != null
        ? { thoughtSignature: descriptor.thoughtSignature }
        : {};
    if (descriptor.type === 'function_call') {
      const call = calls.get(descriptor.id);
      if (
        call == null ||
        hashText(JSON.stringify({ name: call.name, args: call.args })) !==
          descriptor.hash
      )
        return undefined;
      const functionCall = { id: call.id, name: call.name, args: call.args };
      parts.push({ functionCall, ...signature });
      calls.delete(descriptor.id);
      continue;
    }
    const offset = descriptor.thought === true ? reasoningOffset : textOffset;
    const source = descriptor.thought === true ? reasoning : text;
    const value = source.slice(offset, offset + descriptor.length);
    if (
      value.length !== descriptor.length ||
      hashText(value) !== descriptor.hash
    )
      return undefined;
    if (descriptor.thought === true) reasoningOffset += descriptor.length;
    else textOffset += descriptor.length;
    parts.push({
      text: value,
      ...(descriptor.thought != null ? { thought: descriptor.thought } : {}),
      ...signature,
    });
  }
  return textOffset === text.length &&
    reasoningOffset === reasoning.length &&
    calls.size === 0
    ? parts
    : undefined;
}
