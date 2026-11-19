import type { Data, MessageContentComplex } from '@langchain/core/messages';
import type { ProviderMessageProjectionMode } from '@/llm/prepareProviderRequest';
import type { ProviderName } from './llm';

/** An attachment a message carries by reference: LangChain's `source_type: 'id'` data block. */
export type AttachmentReference = Data.IDContentBlock;

/** The provider request a reference is being resolved for. */
export interface AttachmentTarget {
  provider: ProviderName;
  modelId?: string;
  projectionMode: ProviderMessageProjectionMode;
}

/**
 * Supplies the content a referenced attachment stands for, as the block the
 * target provider accepts. Called for each reference while a provider request
 * is prepared. The result goes into that request only, so graph state and
 * checkpoints keep the reference and never hold the attachment's bytes.
 *
 * - `undefined`: not this host's reference (a provider-side file id); the block is left as it is.
 * - `null`: nothing to send to this provider; the block is dropped.
 * - a block: replaces the reference as returned, without further projection.
 *
 * Synchronous and repeatable: one model call can prepare its request more than
 * once (a fallback provider, an overflow retry), each for its own target.
 */
export type AttachmentResolver = (
  reference: AttachmentReference,
  target: AttachmentTarget
) => MessageContentComplex | null | undefined;
