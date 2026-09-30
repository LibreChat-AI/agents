# Classification

A `Classifier` asks typed boolean, choice, or score questions about one state. It is not a chat-model subclass. Construct one classifier per provider and tenant and pass signals, labels, state, and questions per call. This SDK does not cache credentials or host inference weights.

## Backends

- **Jev and Laya:** `createClassifier(settings, credential?, options?)` uses the same System One HTTP adapter. `classificationPreset('laya')` supplies only the dialect and optional-auth setting. Supply the full server `/v1/systemone` endpoint in `settings.baseURL`; omitting `settings.model` lets Laya route to a checkpoint. A supplied key is still sent as a bearer token. Jev, gateway, OpenRouter, Cloudflare, and generic HTTP settings require a credential unless `requiresAuth: false` is explicitly set.
- **Strict structured chat:** `createStructuredChatClassifier({ model, modelId, method: 'jsonSchema' | 'functionCalling' })` injects an already configured LangChain chat model. Choose a mode the provider enforces. Unsupported modes and score questions fail rather than falling back to JSON-mode prompting or asking a model for expected-value scores. Up to 32 compatible questions (128 choice options total) share one model invocation; larger batches fail explicitly.

```ts
const laya = createClassifier(
  {
    ...classificationPreset('laya'),
    baseURL: 'http://localhost:8000/v1/systemone',
  },
  undefined,
  { providerId: 'laya' }
);

const answer = (
  await laya.classify({
    state: 'Please refund the duplicate charge.',
    questions: { refund: booleanQuestion('Did the user request a refund?') },
  })
).answers.refund;

if (answer?.type === 'boolean' && answer.probability !== null) {
  // Apply only a threshold evaluated for this backend, checkpoint, and use case.
}
```

A System One boolean answer has a measured `probability: number`. A structured-chat boolean is an **unmeasured** `{ decision: boolean, probability: null }`. Choice distributions are `null` if unmeasured; present distributions must cover the rubric and sum to one within rounding tolerance. A measured choice must select an option with maximal probability; ties are valid. Boolean criteria may be omitted, a string, or an object containing only optional `true` and `false` text descriptions. Invalid criteria fail locally before credential minting or provider invocation. A missing HTTP answer is `undefined`; a missing strict-schema answer or any malformed answer throws `ClassificationError`. Usage is `null` when unknown, not zero. The `confidence` field on Jev and Laya choice/score answers has different mathematical meanings and is **not** a portable threshold. Score is a System One expected rubric level, not an LLM-selected ordinal level.

Use the request's `signal` and `timeoutMs` to bound a call. HTTP timeouts cover credential minting, retries, response reading, and backoff. Credential minters run once per call, plus one refresh after a 401; the refreshed key persists across retries. HTTP redirects fail rather than forwarding bearer keys. SDK errors and the `onAnswered(label, elapsedMs)` hook contain no response content or credentials; the caller supplies the label and must not put secrets in it.

The structured-chat adapter sends the original state to the provider. Its prompt carries a marker so Langfuse can drop the **entire state and question text** from generation inputs when any tool-output redaction policy is active. Selectively removing identifiable tool fields would still leak a private result quoted in free-form state or instructions. With no active redaction policy, prompts remain visible. Configure provider-side logging separately. No defaults should change until quality, calibration, latency, and cost have been evaluated for each backend and use case.
