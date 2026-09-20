import type { UsageMetadata } from '@langchain/core/messages';

export type NativeMediaPart =
  | { kind: 'text'; text: string; thoughtSignature?: string }
  | {
      kind: 'image';
      mimeType: string;
      data: string;
      thoughtSignature?: string;
    };
export type NativeMediaReference = { continuationRef: string };
export type NativeMediaRestoreInput = {
  file_id?: string;
  continuationRef?: string;
};
export type NativeMediaProviderOutcome = {
  kind: 'blocked' | 'invalid';
  code: string;
};

export type NativeMediaContent =
  | { type: 'text'; text: string; native_media?: NativeMediaReference }
  | {
      type: 'image_file';
      image_file: {
        file_id: string;
        filepath: string;
        filename: string;
        type: string;
        bytes: number;
        width?: number;
        height?: number;
      };
      native_media?: NativeMediaReference;
    };
export interface NativeMediaPort {
  /** Authorize the invocation before the provider request and select its modalities. */
  start(input: {
    modelRunId: string;
    model: string;
    signal?: AbortSignal;
  }): Promise<void | { responseModalities: string[] }>;
  /** Persist the original part before returning content safe to stream and serialize. */
  part(input: {
    modelRunId: string;
    chunkIndex: number;
    partIndex: number;
    part: NativeMediaPart;
  }): Promise<NativeMediaContent>;
  complete(input: { modelRunId: string }): Promise<void>;
  fail(input: {
    modelRunId: string;
    reason: 'aborted' | 'provider' | 'storage';
    /** Failure-only usage; successful calls use the normal model-end callback. */
    usage?: UsageMetadata;
    providerOutcome?: NativeMediaProviderOutcome;
  }): Promise<void>;
  /** Authorize and restore the exact signed provider part for a continuation. */
  restore(input: NativeMediaRestoreInput): Promise<NativeMediaPart>;
  /**
   * Restore all references in order, or reject the entire invocation. Hosts must
   * bound database batches and concurrent asset reads using their own limits.
   */
  restoreBatch?(input: {
    parts: readonly NativeMediaRestoreInput[];
    signal?: AbortSignal;
  }): Promise<NativeMediaPart[]>;
}
