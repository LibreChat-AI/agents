import { createHash } from 'node:crypto';
import { AIMessage, isAIMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { OpenAI as OpenAIClient } from 'openai';
import type * as t from '@/types';

/**
 * OpenAI Responses tool rounds replay their raw output items (ids, status,
 * reasoning) so a restored turn serializes as the live request did. The items
 * travel from the finished model response onto the stored tool call parts and
 * back onto the restored assistant message.
 */

type OutputItem = OpenAIClient.Responses.ResponseOutputItem;

/** `response_metadata` key: digest of the model and endpoint that produced `output`. */
export const OPENAI_RESPONSES_ORIGIN_KEY = '__openai_responses_origin__';

/**
 * Stored responses are kept 30 days. A reasoning item without
 * `encrypted_content` only references that stored copy and is rejected once
 * it expires, so older ones are left out of the replay.
 */
const REASONING_REFERENCE_MAX_AGE_SECONDS = 29 * 24 * 60 * 60;

/**
 * Item types a stored round may replay. Provider tool calls (web search,
 * computer use, MCP) belong to the run that made them and are never replayed
 * from storage.
 */
const REPLAYABLE_ITEM_TYPES: ReadonlySet<string> = new Set([
  'reasoning',
  'function_call',
  'message',
]);

/** Largest serialized size of the items stored for one tool call. */
const MAX_PROVIDER_ITEMS_CHARS = 512_000;

function isReplayableProviderItems(
  providerItems: t.ProviderOutputItems | undefined
): providerItems is t.ProviderOutputItems {
  if (
    providerItems == null ||
    typeof providerItems.origin !== 'string' ||
    !Array.isArray(providerItems.items) ||
    providerItems.items.length === 0
  ) {
    return false;
  }
  return (
    providerItems.items.every((item) =>
      REPLAYABLE_ITEM_TYPES.has((item as { type?: string } | null)?.type ?? '')
    ) && JSON.stringify(providerItems.items).length <= MAX_PROVIDER_ITEMS_CHARS
  );
}

export function getResponsesReplayOrigin(
  model: string,
  endpoint: string
): string {
  return createHash('sha256')
    .update(`${model}\n${endpoint}`)
    .digest('hex')
    .slice(0, 16);
}

/**
 * Splits a finished tool round's output after every `function_call`, so each
 * stored call part carries the items that precede it; trailing items stay
 * with the last call.
 */
export function sliceResponsesToolRound(
  message: AIMessage
): t.ProviderOutputItemsEvent['tool_calls'] | undefined {
  const metadata = message.response_metadata;
  const origin = metadata[OPENAI_RESPONSES_ORIGIN_KEY] as unknown;
  const output = metadata.output as unknown;
  if (typeof origin !== 'string' || !Array.isArray(output)) {
    return;
  }
  const createdAt =
    typeof metadata.created_at === 'number' ? metadata.created_at : undefined;
  const slices: t.ProviderOutputItemsEvent['tool_calls'] = [];
  let pending: OutputItem[] = [];
  for (const item of output as OutputItem[]) {
    pending.push(item);
    if (item.type !== 'function_call') {
      continue;
    }
    slices.push({
      id: item.call_id,
      providerItems: {
        origin,
        ...(createdAt != null && { created_at: createdAt }),
        items: pending,
      },
    });
    pending = [];
  }
  if (slices.length === 0) {
    return;
  }
  slices[slices.length - 1].providerItems.items.push(...pending);
  return slices.every(({ providerItems }) =>
    isReplayableProviderItems(providerItems)
  )
    ? slices
    : undefined;
}

/**
 * A stored call part can be in a different form than the items stored with it
 * (a host or the SDK rewrote the call after the model produced it, as the
 * live request carried it). The part is what the live request sent, so the
 * slice's `function_call` is rewritten to match it. Without parsed object
 * arguments (truncated in storage) the slice is kept as the model sent it.
 */
export function alignProviderItemsWithCall(
  providerItems: t.ProviderOutputItems | undefined,
  call: { id: string; name: string; args: unknown }
): t.ProviderOutputItems | undefined {
  if (
    providerItems == null ||
    call.args == null ||
    typeof call.args !== 'object' ||
    Array.isArray(call.args)
  ) {
    return providerItems;
  }
  const needsRewrite = (item: OutputItem): item is OutputItem & {
    type: 'function_call';
  } =>
    item.type === 'function_call' &&
    item.call_id === call.id &&
    item.name !== call.name;
  if (!providerItems.items.some(needsRewrite)) {
    return providerItems;
  }
  return {
    ...providerItems,
    items: providerItems.items.map((item) =>
      needsRewrite(item)
        ? { ...item, name: call.name, arguments: JSON.stringify(call.args) }
        : item
    ),
  };
}

/**
 * Puts a stored tool round back as the provider output of its assistant
 * message. All-or-nothing: every call must carry its slice from the same
 * origin, in the message's call order.
 */
export function restoreResponsesToolRound(
  message: AIMessage,
  slices: ReadonlyArray<t.ProviderOutputItems | undefined>
): void {
  const toolCalls = message.tool_calls ?? [];
  const first = slices[0];
  if (first == null || slices.length !== toolCalls.length) {
    return;
  }
  const items: OutputItem[] = [];
  for (const slice of slices) {
    if (
      slice == null ||
      slice.origin !== first.origin ||
      !isReplayableProviderItems(slice)
    ) {
      return;
    }
    items.push(...slice.items);
  }
  const callIds = items.flatMap((item) =>
    item.type === 'function_call' ? [item.call_id] : []
  );
  if (callIds.some((id, index) => id !== toolCalls[index].id)) {
    return;
  }
  const responseMetadata = {
    ...message.response_metadata,
    output: items,
    [OPENAI_RESPONSES_ORIGIN_KEY]: first.origin,
    ...(first.created_at != null && { created_at: first.created_at }),
  };
  message.response_metadata = responseMetadata;
  message.lc_kwargs.response_metadata = responseMetadata;
}

function isStaleReasoningReference(
  item: OutputItem,
  createdAt: unknown,
  nowSeconds: number
): boolean {
  return (
    item.type === 'reasoning' &&
    (item.encrypted_content == null || item.encrypted_content === '') &&
    typeof createdAt === 'number' &&
    nowSeconds - createdAt > REASONING_REFERENCE_MAX_AGE_SECONDS
  );
}

function withProviderOutput(
  message: AIMessage,
  output: OutputItem[] | undefined
): AIMessage {
  const responseMetadata = { ...message.response_metadata };
  delete responseMetadata.output;
  delete responseMetadata[OPENAI_RESPONSES_ORIGIN_KEY];
  const hasToolCalls = (message.tool_calls?.length ?? 0) > 0;
  return new AIMessage({
    id: message.id,
    name: message.name,
    content:
      output == null && hasToolCalls && message.content === ''
        ? []
        : message.content,
    additional_kwargs: message.additional_kwargs,
    response_metadata:
      output == null
        ? responseMetadata
        : {
          ...responseMetadata,
          output,
          [OPENAI_RESPONSES_ORIGIN_KEY]:
              message.response_metadata[OPENAI_RESPONSES_ORIGIN_KEY],
        },
    tool_calls: message.tool_calls,
    invalid_tool_calls: message.invalid_tool_calls,
    usage_metadata: message.usage_metadata,
  });
}

/**
 * Replays provider output only to the origin that produced it, and never
 * under ZDR (LangChain rebuilds those items without ids). A tool call message
 * left without output drops its empty text so no empty assistant message is
 * sent.
 */
export function gateResponsesProviderOutput(
  messages: BaseMessage[],
  origin: string,
  zdrEnabled: boolean,
  nowSeconds: number
): BaseMessage[] {
  let gated: BaseMessage[] | undefined;
  for (let index = 0; index < messages.length; index++) {
    const message = messages[index];
    const next = gateMessage(message, origin, zdrEnabled, nowSeconds);
    if (next !== message) {
      gated ??= [...messages];
      gated[index] = next;
    }
  }
  return gated ?? messages;
}

function gateMessage(
  message: BaseMessage,
  origin: string,
  zdrEnabled: boolean,
  nowSeconds: number
): BaseMessage {
  if (!isAIMessage(message)) {
    return message;
  }
  const metadata = message.response_metadata;
  const messageOrigin = metadata[OPENAI_RESPONSES_ORIGIN_KEY] as unknown;
  const output = metadata.output as OutputItem[] | undefined;
  if (messageOrigin == null) {
    const isEmptyToolRound =
      output == null &&
      message.content === '' &&
      (message.tool_calls?.length ?? 0) > 0;
    return isEmptyToolRound ? withProviderOutput(message, undefined) : message;
  }
  if (zdrEnabled || messageOrigin !== origin || !Array.isArray(output)) {
    return withProviderOutput(message, undefined);
  }
  const kept = output.filter(
    (item) => !isStaleReasoningReference(item, metadata.created_at, nowSeconds)
  );
  return kept.length === output.length
    ? message
    : withProviderOutput(message, kept);
}
