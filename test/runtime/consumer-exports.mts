import { createRemoteSdkServer } from '@librechat/agents/runtime';
import {
  LANGFUSE_OBSERVATION_METADATA_ARTIFACT_KEY,
  MemorySaver,
  Run,
  toStoredMessage,
} from '@librechat/agents';

// @ts-expect-error Run must retain its configuration type.
Run.create(42);
// @ts-expect-error The runtime barrel must retain the server options type.
createRemoteSdkServer(42);
// @ts-expect-error The messages barrel must retain the input message type.
toStoredMessage(42);
// @ts-expect-error The named export is a string, not a symbol.
export const key: symbol = LANGFUSE_OBSERVATION_METADATA_ARTIFACT_KEY;
// @ts-expect-error The bare re-export is a constructor, not a number.
export const saver: number = MemorySaver;
