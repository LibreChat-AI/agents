# Typed Decisions

A `DecisionModel` asks typed boolean, choice, or score questions about one state. It is not a chat-model subclass. Construct one decision model per provider and tenant and pass signals, labels, state, and questions per call. This SDK does not host inference weights, and caches a credential only through the opt-in `commandToken` helper.

## Backends

- **Jev and Laya:** `createDecisionModel(settings, credential?, options?)` uses the same System One HTTP adapter. `decisionPreset('laya')` supplies only the dialect and optional-auth setting. Supply the full server `/v1/systemone` endpoint in `settings.baseURL`; omitting `settings.model` lets Laya route to a checkpoint. A supplied key is still sent as a bearer token. Jev, gateway, OpenRouter, Cloudflare, and generic HTTP settings require a credential unless `requiresAuth: false` is explicitly set.
- **Strict structured chat:** `createStructuredChatDecisionModel({ model, modelId, method: 'jsonSchema' | 'functionCalling' })` injects an already configured LangChain chat model. The verified adapters are OpenAI and Azure OpenAI in either strict mode and Anthropic in strict `functionCalling` mode. Bedrock and other unverified adapters fail with `unsupported_mode` before invocation, even if `withStructuredOutput` silently accepts strict options. Unsupported modes and score questions fail rather than falling back to JSON-mode prompting or asking a model for expected-value scores. Up to 32 compatible questions (128 choice options total) share one model invocation; larger batches fail explicitly.

```ts
const laya = createDecisionModel(
  {
    ...decisionPreset('laya'),
    baseURL: 'http://localhost:8000/v1/systemone',
  },
  undefined,
  { providerId: 'laya' }
);

const answer = (
  await laya.decide({
    state: 'Please refund the duplicate charge.',
    questions: { refund: booleanQuestion('Did the user request a refund?') },
  })
).answers.refund;

if (answer?.type === 'boolean' && answer.probability !== null) {
  // Apply only a threshold evaluated for this backend, checkpoint, and use case.
}
```

A System One boolean answer has a measured `probability: number`. A structured-chat boolean is an **unmeasured** `{ decision: boolean, probability: null }`. Choice distributions are `null` if unmeasured; present distributions must cover the rubric and sum to one within rounding tolerance. Hosts in front of System One (the ClickHouse gateway, OpenRouter) round every number to two decimals, so a `systemone` answer whose probabilities and score all sit on the `0.01` grid is read under that rounding model: the distribution is accepted only when some measured distribution summing to exactly one rounds to it (each value within `0.005` and inside `[0, 1]`), and the score only when it is within `0.005` of an expectation such a distribution can have. Anything rounding cannot produce is rejected as inconsistent. Every `port` answer, and any `systemone` answer with a probability or score off that grid, keeps the four-decimal bounds for its distribution and its score alike. `src/decisions/fixtures/jev-1.13.0.json` holds recorded gateway answers that pin these bounds. A measured choice must select an option with maximal probability; ties are valid. Boolean criteria may be omitted, a string, or an object containing only optional `true` and `false` text descriptions. Invalid criteria fail locally before credential minting or provider invocation. A missing HTTP answer is `undefined`; a missing strict-schema answer or any malformed answer throws `DecisionError`. Usage is `null` when unknown, not zero. The `confidence` field on Jev and Laya choice/score answers has different mathematical meanings and is **not** a portable threshold. Score is a System One expected rubric level, not an LLM-selected ordinal level. Measured score maps use consecutive numeric levels starting at zero, including when the exported parser is called without an expected question.

Use the request's `signal` and `timeoutMs` to bound a call. HTTP timeouts cover credential minting, retries, response reading, and backoff. Credential minters run once per call, plus one refresh after a 401; the refreshed key persists across retries. The refresh call carries the refused key as `refused`, so a minter that caches can return a token minted since rather than discarding it. HTTP 402 is a `quota` failure (the credential is out of credits) and, like `bad_request`, is never retried. `unauthorized` is re-sent only once, after a 401 refresh. Backoff retries apply only to `rate_limited`, `server_error`, and `network`. HTTP redirects fail rather than forwarding bearer keys. SDK errors and the `onAnswered(label, elapsedMs)` hook contain no response content or credentials; the caller supplies the label and must not put secrets in it.

The structured-chat adapter sends the original state to the provider. Its prompt carries a marker so Langfuse can drop the **entire state and question text** from generation inputs when any tool-output redaction policy is active. Selectively removing identifiable tool fields would still leak a private result quoted in free-form state or instructions. With no active redaction policy, prompts remain visible. Configure provider-side logging separately. No defaults should change until quality, calibration, latency, and cost have been evaluated for each backend and use case.

## Credentials, Breakers, and Probes

`commandToken(command, { ttlMs, timeoutMs, killGraceMs })` from `@librechat/agents/decisions/node` turns a shell command that prints a short-lived bearer token (for example `dataplanectl chai auth token`) into a `DecisionCredential`. The last non-empty stdout line is the token; it is reused for `ttlMs` (default 50 minutes, counted from when the command finished), concurrent calls share one run, and the transport's refresh after a 401 runs the command again unless the refused token was already replaced, in which case it reuses the newer token (or joins the run minting it). A caller whose signal aborts, including at the decision deadline, stops waiting at once; the run is killed once every caller sharing it has aborted, so a token command slower than the decision timeout never finishes inside `decide()`. A kill, at `timeoutMs` or on abandonment, sends `SIGTERM` and then `SIGKILL` once `killGraceMs` (default two seconds) passes with the run still alive, so a command that ignores `SIGTERM` cannot outlive its run and a killed run overlaps the next one for at most that grace. Off Windows a kill reaches the command's whole process group, not only the shell; on Windows it ends only the shell, not the command the shell launched. The command reads no stdin. Give `probe` a longer `timeoutMs` to warm a slow first mint. A run that exits non-zero or prints nothing is an `unauthorized` `DecisionError`; a run killed at `timeoutMs` is a `timeout`, and a spawn failure or output overflow is `network`, so a slow token command never trips a breaker. The message names the exit status with a redacted stderr excerpt, never a line of stdout, a token (including a refused one whose refresh failed), the command, or any word of it (split at whitespace, quotes, and `=`), whatever their length; a short word may garble the excerpt instead. It needs `node:child_process`, so it lives on its own Node-only subpath.

```ts
import { commandToken } from '@librechat/agents/decisions/node';
import {
  createDecisionModel,
  decisionPreset,
  probe,
  withBreaker,
} from '@librechat/agents/decisions';

const model = withBreaker(
  createDecisionModel(
    decisionPreset('clickhouse')!,
    commandToken('dataplanectl chai auth token')
  )
);
const ready = await probe(model, { timeoutMs: 15_000 });
if (!ready.ok) {
  // ready.failure, ready.status, ready.detail (redacted); never thrown.
}
```

`unauthorized` and `quota` belong to the credential, not the request, so `withBreaker(model, breaker?)` opens on the first one: every later `decide()` rejects at once, without a request, with the same failure and status and `circuitOpen: true`, until `breaker.reset()`. Rate limits, server errors, timeouts, and network failures never trip it, and a fatal failure from a call that started before a `reset()` does not reopen it. Share one `createBreaker()` between models that share a credential. `probe(model)` asks one tiny boolean question and resolves `{ ok: true, model }` or `{ ok: false, failure, status?, circuitOpen, detail }`; a non-SDK exception reads as `network` and its message is dropped. `redact(text, secrets?)` removes listed secrets, bearer values, JWTs, and long opaque tokens from text bound for a log line or status row.

## Light Entry Point

`@librechat/agents/decisions` exports the System One HTTP stack (types, question builders, presets, `createHttpDecisionModel`, the dialect parser, `createTransport`, `failureForStatus`, `retryAfterMs`, `withDecisionDeadline`, `withBreaker`, `probe`, and `redact`) without loading LangChain, any other third-party package, or a Node built-in. `@librechat/agents/decisions/node` adds `commandToken` and reaches only `node:child_process`. `createStructuredChatDecisionModel` binds a LangChain chat model, so it is exported only from the root entry. The root `@librechat/agents` entry still exports everything. `npm run test:light-entries` loads each built subpath and fails if it reaches a non-built-in package, or, for the platform-neutral subpaths, any Node built-in.

## Structured Output Capabilities

The private `structuredOutput.ts` binding boundary owns adapter/method options. The caller explicitly chooses the method; there is no automatic mode fallback, public capability registry, provider-profile override, or extra provider SDK import at runtime.

| Adapter                           | `jsonSchema`                                                                 | `functionCalling`                      |
| --------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------- |
| OpenAI (`openai`)                 | Native schema with `strict: true`                                            | Tool schema with `strict: true`        |
| Azure OpenAI (`azure_openai`)     | Native schema with `strict: true`                                            | Tool schema with `strict: true`        |
| Anthropic (`anthropic`)           | Reject locally: the locked SDK translates choice enums into description text | Tool schema with `strict: true`        |
| Other adapters, including Bedrock | Reject locally with `unsupported_mode`                                       | Reject locally with `unsupported_mode` |

A model's `profile.structuredOutput` metadata is not proof that an adapter forwards the requested enforcement options. Contract tests exercise both strict modes through locked upstream and LibreChat OpenAI/Azure implementations, plus strict Anthropic tool calling, with controlled provider responses. Native Anthropic support stays disabled: its locked schema transformer moves `enum` constraints into description text, so a choice would no longer be provider-enforced. Local validation cannot replace that guarantee, and the SDK does not silently switch methods. Endpoint/checkpoint support remains provider-owned; remote rejections return sanitized typed failures. All answers are still validated locally, probabilities remain unmeasured for chat, and score questions remain unsupported.

## API Naming

`DecisionModel.decide()` is the provider-neutral contract. System One names the Jev/Laya HTTP dialect, not the strict-chat adapter. This API is introduced in this unreleased PR, so there are no classification compatibility aliases. LibreChat configuration keys and downstream migrations are separate changes.
