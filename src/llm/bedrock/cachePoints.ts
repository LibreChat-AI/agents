// Vendored from @langchain/aws@1.4.2 (utils/message_inputs.ts) because the
// upstream `applyCachePointsToConversePayload` export is internal.
import type {
  BedrockPromptCacheControl,
  ConverseCommandParams,
} from '@langchain/aws';
import type * as Bedrock from '@aws-sdk/client-bedrock-runtime';
import { modelDisallowsAssistantPrefill } from '@/llm/anthropic/utils/message_inputs';

function isConverseCachePoint(block: unknown): boolean {
  return Boolean(
    typeof block === 'object' &&
      block !== null &&
      'cachePoint' in block &&
      block.cachePoint &&
      typeof block.cachePoint === 'object' &&
      block.cachePoint !== null &&
      'type' in block.cachePoint
  );
}

function createConverseCachePointBlock(
  cacheControl: BedrockPromptCacheControl,
  isNovaModel: boolean
): { cachePoint: { type: 'default'; ttl?: '1h' } } {
  const ttl =
    !isNovaModel && cacheControl.ttl && cacheControl.ttl !== '5m'
      ? cacheControl.ttl
      : undefined;
  return {
    cachePoint: {
      type: 'default',
      ...(ttl ? { ttl } : {}),
    },
  };
}

export function applyCachePointsToConversePayload(fields: {
  cacheControl?: BedrockPromptCacheControl;
  system: Bedrock.SystemContentBlock[];
  messages: Bedrock.Message[];
  params?: Partial<ConverseCommandParams>;
  modelId: string;
}): void {
  const { cacheControl, system, messages, params, modelId } = fields;
  if (!cacheControl) {
    return;
  }

  const isNovaModel = modelId.toLowerCase().includes('amazon.nova');
  const cacheBlock = createConverseCachePointBlock(cacheControl, isNovaModel);

  if (
    system.length > 0 &&
    !system.some((block) => isConverseCachePoint(block))
  ) {
    system.push(cacheBlock);
  }

  const lastMessage = messages[messages.length - 1];
  const lastContent = lastMessage.content;
  if (Array.isArray(lastContent)) {
    const hasNovaToolBlock =
      isNovaModel &&
      lastContent.some(
        (block) =>
          typeof block === 'object' &&
          block !== null &&
          ('toolResult' in block || 'toolUse' in block)
      );
    if (
      !hasNovaToolBlock &&
      !lastContent.some((block) => isConverseCachePoint(block))
    ) {
      lastContent.push(cacheBlock);
    }
  }

  const tools = params?.toolConfig?.tools;
  if (
    !isNovaModel &&
    Array.isArray(tools) &&
    !tools.some((tool) => isConverseCachePoint(tool))
  ) {
    tools.push(cacheBlock as unknown as Bedrock.Tool);
  }
}

/**
 * Drop trailing assistant turns for Claude models that reject a prefill (4.6 and
 * later), mirroring `stripUnsupportedAssistantPrefill` for the Converse payload.
 * A cache point that rode a stripped turn is re-anchored on the new tail when no
 * surviving message carries one, so the request keeps its message caching.
 */
export function stripUnsupportedConversePrefill(
  messages: Bedrock.Message[],
  model?: string
): Bedrock.Message[] {
  if (
    !modelDisallowsAssistantPrefill(model) ||
    messages.length <= 1 ||
    messages[messages.length - 1]?.role !== 'assistant'
  ) {
    return messages;
  }

  const nextMessages = [...messages];
  let strippedCachePoint: Bedrock.ContentBlock | undefined;
  while (
    nextMessages.length > 1 &&
    nextMessages[nextMessages.length - 1]?.role === 'assistant'
  ) {
    const stripped = nextMessages.pop();
    strippedCachePoint ??= stripped?.content?.find((block) =>
      isConverseCachePoint(block)
    );
  }

  const survivorsHaveCachePoint = nextMessages.some((message) =>
    (message.content ?? []).some((block) => isConverseCachePoint(block))
  );
  if (strippedCachePoint == null || survivorsHaveCachePoint) {
    return nextMessages;
  }

  const lastIndex = nextMessages.length - 1;
  const tail = nextMessages[lastIndex];
  nextMessages[lastIndex] = {
    ...tail,
    content: [...(tail.content ?? []), strippedCachePoint],
  };
  return nextMessages;
}
