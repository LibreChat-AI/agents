# @librechat/agents

## Project Overview

`@librechat/agents` is a TypeScript library for LLM agent orchestration — tool calling, multi-agent graphs, message formatting, streaming, and provider abstraction (Anthropic, Bedrock, VertexAI, OpenAI, Google). Published as `@librechat/agents` on npm. This is a major backend dependency of [LibreChat](../LibreChat/CLAUDE.md) (same team).

| Path                                         | Purpose                                                                              |
| -------------------------------------------- | ------------------------------------------------------------------------------------ |
| `src/messages/`                              | Message formatting, caching, content processing                                      |
| `src/graphs/`                                | LangGraph-based agent graphs (single + multi-agent)                                  |
| `src/llm/`                                   | Provider-specific LLM wrappers and utilities                                         |
| `src/tools/`                                 | Tool definitions and search                                                          |
| `src/agents/`                                | Agent definitions and handoff logic                                                  |
| `src/types/`                                 | Shared TypeScript types                                                              |
| `src/common/`                                | Enums, constants                                                                     |
| `src/run.ts`                                 | Main run orchestration                                                               |
| `src/stream.ts`                              | Streaming logic                                                                      |
| `src/langfuse*.ts`, `src/instrumentation.ts` | Langfuse tracing integration (see [Langfuse Trace Shaping](#langfuse-trace-shaping)) |

---

## Code Style

### Structure and Clarity

- **Never-nesting**: early returns, flat code, minimal indentation. Break complex operations into well-named helpers.
- **Functional first**: pure functions, immutable data, `map`/`filter`/`reduce` over imperative loops. Only reach for OOP when it clearly improves domain modeling or state encapsulation.
- **No dynamic imports** unless absolutely necessary.

### DRY

- Extract repeated logic into utility functions.
- Parameterized helpers instead of near-duplicate functions.
- Constants for repeated values; configuration objects over duplicated init code.
- Shared validators, centralized error handling, single source of truth for business rules.
- Shared typing system with interfaces/types extending common base definitions.
- Abstraction layers for external API interactions.

### Iteration and Performance

- **Minimize looping** — especially over shared data structures like message arrays, which are iterated frequently throughout the codebase. Every additional pass adds up at scale.
- Consolidate sequential O(n) operations into a single pass whenever possible; never loop over the same collection twice if the work can be combined.
- Choose data structures that reduce the need to iterate (e.g., `Map`/`Set` for lookups instead of `Array.find`/`Array.includes`).
- Avoid unnecessary object creation; consider space-time tradeoffs.
- Prevent memory leaks: careful with closures, dispose resources/event listeners, no circular references.

### Type Safety

- **Never use `any`**. Explicit types for all parameters, return values, and variables.
- **Limit `unknown`** — avoid `unknown`, `Record<string, unknown>`, and `as unknown as T` assertions. A `Record<string, unknown>` almost always signals a missing explicit type definition.
- **Don't duplicate types** — before defining a new type, check whether it already exists in `src/types/`. Reuse and extend existing types (`MessageContentComplex`, `ExtendedMessageContent`, `TMessage`, etc.) rather than creating redundant definitions.
- Use union types, generics, and interfaces appropriately.
- All TypeScript and ESLint warnings/errors must be addressed — do not leave unresolved diagnostics.

### Comments and Documentation

- Write self-documenting code; no inline comments narrating what code does.
- JSDoc only for complex/non-obvious logic or intellisense on public APIs.
- Single-line JSDoc for brief docs, multi-line for complex cases.
- Avoid standalone `//` comments unless absolutely necessary.

### Import Order

Imports are organized into three sections:

1. **Package imports** — sorted shortest to longest line length.
2. **`import type` imports** — sorted longest to shortest (package types first, then local types; length resets between sub-groups).
3. **Local/project imports** — sorted longest to shortest.

Multi-line imports count total character length across all lines. Consolidate value imports from the same module. Always use standalone `import type { ... }` — never inline `type` inside value imports.

### JS/TS Loop Preferences

- **Limit looping as much as possible.** Prefer single-pass transformations and avoid re-iterating the same data.
- `for (let i = 0; ...)` for performance-critical or index-dependent operations.
- `for...of` for simple array iteration.
- `for...in` only for object property enumeration.

---

## Development Commands

| Command              | Purpose                            |
| -------------------- | ---------------------------------- |
| `npm run build`      | Build CJS + ESM + types via Rollup |
| `npx jest <pattern>` | Run tests matching pattern         |
| `npx tsc --noEmit`   | Type-check without emitting        |
| `npx eslint src/`    | Lint source files                  |

- Package manager: **npm** (`package-lock.json`, `packageManager: npm@10.5.2`)
- TypeScript with path aliases: `@/*` → `src/*`
- Test framework: Jest with `ts-jest`

---

## Dependency Management

- **Prefer bumping the parent over adding an `overrides` entry.** When a transitive dependency is flagged (e.g. by `npm audit`), first check whether a package we actually declare has a newer version whose range already pulls the patched dependency — bump that parent instead. Fixing at the source is always preferred to overriding.
- **Overrides are a last resort.** Only add one when no parent bump can supply the fix (e.g. an unmaintained parent, or a fix requiring a version the parent's range disallows). Every override is maintenance debt that must be revisited on future upgrades.
- **Hard-bump direct deps explicitly; do not rely on `npm audit fix`.** Set the target version in `package.json`, run `npm install`, then re-run `npm audit` to confirm. `npm audit fix` makes opaque, wide-reaching changes.
- **Keep overrides minimal and current.** A stale override that force-pins an old major can silently downgrade unrelated consumers into a semver-violating resolution — verify with `npm ls <pkg> --all` that a pin isn't harming packages that expect a newer major. Bump or delete overrides that are no longer needed.
- **Use reference overrides (`"$name"`) to dedupe** a transitive copy to the version we already declare (e.g. `"uuid": "$uuid"`) instead of hard-coding a duplicate version string.
- **After any dependency change**, verify the tree stays healthy: `npm install` → `npm audit` (expect 0) → `npx tsc --noEmit` → `npx eslint src/` → a `npx jest` smoke run, since overrides frequently affect dev/build tooling (eslint, jest, typescript-eslint).

---

## Testing

- Framework: **Jest**, run from project root: `npx jest <pattern>`.
- Test files: `*.test.ts` and `*.spec.ts` under `src/`.

### Philosophy

- **Real logic over mocks.** Exercise actual code paths with real dependencies. Mocking is a last resort.
- **Spies over mocks.** Assert that real functions are called with expected arguments and frequency without replacing underlying logic.
- **MCP**: use real `@modelcontextprotocol/sdk` exports for servers, transports, and tool definitions. Mirror real scenarios, don't stub SDK internals.
- Only mock what you cannot control: external HTTP APIs, rate-limited services, non-deterministic system calls.
- Heavy mocking is a code smell, not a testing strategy.

---

## Langfuse Trace Shaping

This library is LibreChat's tracing surface: every agent run it orchestrates is exported to Langfuse, and trace quality is a product feature. Any change that touches graphs, node naming, callbacks, tool execution, message serialization, streaming, or providers must keep traces well-shaped — ask "how will this look in Langfuse?" as part of the change, not after.

### Module Map

| Module                                                         | Responsibility                                                               |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `src/langfuseTraceShaping.ts`                                  | Export-time span rename/retype/drop rules (the shape itself)                 |
| `src/langfuseToolOutputTracing.ts`                             | Span processor applying shaping hooks + tool-output redaction                |
| `src/langfuseInlineMedia.ts`                                   | Export-time replacement of inline base64 media with type/size descriptors    |
| `src/langfuse.ts`                                              | Callback handler, identity/tags/metadata, control-flow + usage normalization |
| `src/instrumentation.ts`                                       | Tracer provider bootstrap, per-tenant routing, deterministic trace ids       |
| `src/langfuseConfig.ts`                                        | Config resolution/merging (env vs run vs agent level)                        |
| `src/langfuseRuntimeContext.ts`, `src/langfuseRuntimeScope.ts` | Per-run scoping so concurrent runs/tenants never cross-contaminate           |

### Invariants — what "well-shaped" means

These originated from direct Langfuse-team feedback (PRs #288, #316) and must survive all future changes:

- **Stable, operation-describing span names.** `agent`, `tool-dispatch`, `llm` — never leak ephemeral agent ids (`agent=<provider__model>`) or provider class names (`ChatOpenAI`) into span names. The model belongs on the generation's model attribute; name-based logic downstream must not break when models switch.
- **Correct observation types.** Agent trace roots and agent nodes are `agent` observations; tool dispatch is a `chain`; individual tool calls are `tool`; LLM calls are `generation`; title trace roots are `chain`.
- **No plumbing noise.** LangGraph internals (`__start__` channel seeds, anonymous `RunnableLambda` pass-throughs) are dropped at export. A trace tree should read like the run's story — if a new node adds noise, extend the shaping/drop rules rather than shipping it raw.
- **Root-observation input/output are the conversation, not the state.** Root input reduces to the user's question, output to the assistant's answer — never full serialized graph state. Tool-dispatch input is scoped to the pending tool calls. Do not emit deprecated trace-level input/output attributes.
- **Control flow is not an error.** `GraphInterrupt` and `ParentCommand` end their traces as successful with `controlFlow` outputs, not as error traces.
- **Usage/cost is accurate per provider.** e.g. Bedrock cache read/write tokens are folded into input tokens so Langfuse cost math is right.
- **Inline media is described, not exported.** Every span that serializes the conversation would otherwise carry its own copy of each attached file, so base64 payloads (raw, `data:` URIs while Langfuse media upload is off, serialized byte buffers) become a type-and-size descriptor unless a host opts in via `inlineMediaTracing`.
- **Redaction is honored everywhere tool output can surface** — tool spans, and any generation input that embeds tool results (e.g. the activity-label prompt).
- **Identity and metadata always propagate**: `userId`, `sessionId`, tags, environment, and trace metadata (`messageId`, `parentMessageId`, `agentId`, `agentName`) — including across LangChain callbacks that fire outside the caller's OTEL context.
- **Trace identity is self-contained.** Root observations never inherit trace ids or parents from foreign ambient OTEL spans (e.g. a host's HTTP auto-instrumentation): the callback handler detaches them so roots stay true roots, deterministic ids apply, and concurrent runs inside one request context (an agent run plus a title run) cannot merge into one trace. Spans created through the Langfuse tracer provider are honored as parents, so hosts can still group runs under their own Langfuse observations deliberately.
- **Runtime scopes are run-stamped.** LangChain executes non-awaited callbacks on a process-wide background queue, so a callback can run inside a DIFFERENT concurrent run's async context — the ambient scope at callback time is untrustworthy for tenant identity. Scopes and handlers carry a `runId`; a handler only adopts an ambient scope stamped with its own run (same-run agent overlays still win), otherwise its own configuration and seed apply. Without this, concurrent runs leak spans into each other's Langfuse projects.
- **Deterministic trace ids** when a run opts in (`LangfuseConfig.deterministicTraceId`), so host apps can attach scores/feedback by regenerating the id from the run id.

### Verifying

- Run the tracing specs after any change in this area: `npx jest langfuse deterministic-trace-id` (specs live in `src/specs/langfuse-*.test.ts` and `src/tools/__tests__/ToolNode.langfuse.test.ts`). New shaping rules get spec coverage in `langfuse-trace-shaping.test.ts`.
- For structural changes, verify against a real Langfuse project: set `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_BASE_URL`, run an agent, and inspect the observation tree in the UI or via the Observations API v2 filtered by trace ID — confirm span names, observation types, and root input/output match the invariants above.
- For nontrivial Langfuse SDK work, use Langfuse's own agent-facing resources instead of guessing SDK behavior: the [Langfuse skill](https://github.com/langfuse/skills/tree/main/skills/langfuse), the docs MCP server (`https://langfuse.com/api/mcp`), and [llms.txt](https://langfuse.com/llms.txt).

---

## Formatting

Fix all formatting lint errors (trailing spaces, tabs, newlines, indentation) using auto-fix when available. All TypeScript/ESLint warnings and errors **must** be resolved.
