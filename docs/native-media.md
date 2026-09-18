# Native Google media

`CustomChatGoogleGenerativeAI` accepts a `NativeMediaPort` for models that return
images alongside text. The SDK calls the provider through its normal invocation
and streaming paths, including `streamEvents()` when the port is configured.
The host supplies authorization, durable storage and replay.

```typescript
import { CustomChatGoogleGenerativeAI } from '@librechat/agents/llm/google';
import type { NativeMediaPort } from '@librechat/agents';

function createImageModel(nativeMedia: NativeMediaPort) {
  return new CustomChatGoogleGenerativeAI({
    model: 'gemini-3-pro-image-preview',
    apiKey: process.env.GOOGLE_API_KEY,
    nativeMedia,
  });
}
```

The model exposes `nativeMediaProtocolVersion = 1` so a host can check support.
`NativeMediaPart`, `NativeMediaContent` and `NativeMediaReference` are exported
from the package root alongside `NativeMediaPort`.

| Callback   | Host responsibility                                                                                                                                               |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `start`    | Authorize the model invocation. Return the permitted `responseModalities`, such as `['TEXT', 'IMAGE']`. Rejection prevents a provider request.                    |
| `part`     | Persist each text or image part and its optional thought signature. Return visible text or an `image_file` reference only after persistence succeeds.             |
| `complete` | Record that the response finished and its visible parts were persisted.                                                                                           |
| `fail`     | Record an incomplete invocation, with an `aborted`, `provider` or `storage` reason. Reconcile any remote work and partial storage using the host's own lifecycle. |
| `restore`  | Authorize a continuation reference and return its original text or base64 image bytes with the exact thought signature.                                           |

Each invocation has a `modelRunId`; each persisted part also has a `chunkIndex`
and `partIndex`. The host can use these to identify writes. Its continuation
references must survive later invocations and process restarts if conversations
can be resumed. Scope reference lookup to the current user and conversation.

The host's `responseModalities` applies only to that invocation. If `start`
returns no selection, the constructor's `responseModalities` is used when a port
is configured; otherwise the provider chooses its defaults. Concurrent calls
keep their selections separate.

Text and images retain provider order through graph dispatch and aggregation.
Image content becomes an `image_file` with a file ID and stored-file metadata.
Private image bytes and thought signatures stay behind the host port; visible
content may carry an opaque `native_media.continuationRef`. Preserve that marker
and the content order when saving and restoring assistant messages.

Before a continuation request, the SDK restores signed provider parts, including
empty text carrying a signature, without mutating the caller's stored messages.
It rejects native continuation references
when no host port is configured. Existing tool-generated image files without a
native continuation marker continue through the ordinary message conversion path.

Without a port, ordinary text invocation and existing Google server-tool
signatures are preserved. Inline image output requires configured storage and is
rejected before it can be emitted. The host
controls supported models, file access, retention, accounting and recovery; an
aborted local stream does not establish that a remote generation stopped.

The deterministic tests in `src/llm/google/native.test.ts` exercise the real
Google wrapper and SDK graph with a controlled provider boundary. They cover
admission, ordered output, serialization, signed replay, storage failure,
cancellation and compatibility with ordinary text.
