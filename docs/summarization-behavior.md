# Summarization & Context Management Behavior

## Overview

LibreChat's agent context management uses a staged pipeline inspired by Claude Code's compaction approach. The behavior differs based on whether summarization is enabled or disabled for the agent.

Both paths share **observation masking** as the first line of defense. The key difference is what happens when masking alone isn't enough: summarization-enabled agents compact the full conversation via an LLM call, while summarization-disabled agents apply progressively aggressive mechanical truncation.

---

## Shared Behavior (Both Paths)

### Observation Masking (80%+ context pressure)

When the total message tokens exceed 80% of the pruning budget, consumed ToolMessages are replaced with tight head+tail truncations (~300 chars) that serve as informative placeholders.

**Consumed** means: a subsequent AI message exists with substantive text content (not purely tool calls). The model has already read and acted on the result.

- AI messages are **never masked** — they contain the model's own reasoning and conclusions, which prevents the model from repeating work after tool results are masked.
- **Unconsumed** tool results (the latest outputs the model hasn't responded to yet) are left intact.
- This runs every agent node turn when pressure is at or above 80%.

### Token Budget Anatomy

```
maxContextTokens (e.g. 8000)
  - reserveTokens (5% default)
  = pruningBudget
  - instructionTokens (system message + tool schemas)
  = effectiveMaxTokens (available for conversation messages)
```

`contextPressure = calibratedTotalTokens / pruningBudget`

### Calibration

Token counts from the local tokenizer (tiktoken) diverge from what providers actually count. The pruner maintains a **cumulative calibration ratio**:

```
calibrationRatio = cumulativeProviderReported / cumulativeRawSent
```

Updated each turn from `usageMetadata.input_tokens` returned by the provider. The ratio is persisted across runs via `contextMeta.calibrationRatio` so subsequent conversations start calibrated.

All budget comparisons multiply raw counts by `calibrationRatio` to approximate provider space, while the `indexTokenCountMap` stays in raw-token space for stability.

### Fading Tier

Every character cap applied to historical tool results (masking, pre-flight truncation, fit-to-budget) derives from one latched **fading tier** (`src/messages/fading.ts`): `{ v, budgetTokens, masked }`. The budget sits on a ladder that halves the context window per rung; the tier only ever shrinks and masking only ever activates. Because truncation is a pure function of (content, cap), a historical tool result maps to identical bytes on every call within a tier — which is what keeps prefix-based provider prompt caches (Anthropic's single tail breakpoint) valid from turn to turn. Only escalation and compaction rewrite the prefix.

- **Fit rung**: the shallowest rung at which the widest observed parallel tool exchange (call inputs plus fresh results) fits within the effective budget, so a complete exchange can never leave the context empty or lose its results to orphan repair.
- **Pressure-band rungs** (summarization disabled only): +1 rung at 85 %, +2 at 90 %, +4 at 99 %.
- **Masking** latches at 80 % pressure; consumed results then keep 10 % of the fresh cap (floor 300 chars).

Graph history stays canonical. The pruner keeps the original bytes of every capped tool result and input-capped AI message and builds a provider-facing projection per Run from them, so a deeper tier or masking re-derives from the original content and matches what a fresh Run seeded with the same tier derives from the host's stored messages. Restored tiers are validated at every boundary (`Run`, `StandardGraph`, `AgentSession`); an invalid or fresh seed starts fresh.

The tier is returned from every prune call. Single-agent hosts can use `Run.getFadingTier()` and `RunConfig.fadingTier`; multi-agent hosts should persist `Run.getFadingTiers()` and restore `RunConfig.fadingTiers` so each agent keeps its independent tier. The budget is absolute, so a mid-run budget correction or a return to the normal window keeps the tier; only compaction resets it, and `Run.didResetFadingTier()` / `Run.getFadingTierResetAgentIds()` report that reset to hosts that merge tiers.

**Instruction overhead calibration**: The pruner also tracks `bestInstructionOverhead` — the best observed instruction token count from provider feedback. When the variance between the estimated and calibrated `toolSchemaTokens` exceeds 15% (`CALIBRATION_VARIANCE_THRESHOLD`), the calibrated value is applied to `AgentContext.toolSchemaTokens`. This corrects the local tool-schema estimate (which uses a static multiplier) against real provider behavior. After intra-run summarization, the calibrated overhead is preserved and seeded into the recreated pruner.

---

## When Summarization is Enabled

### Pipeline (every agent node turn)

1. **Fit-to-budget truncation** (every turn): the fading tier's fit rung caps every tool result and tool-call input so the widest observed parallel exchange fits within the effective budget. The cap is latched, so a historical result keeps the same bytes on later turns.

2. **80%+ pressure — Observation masking**: Masking latches on the tier; consumed ToolMessages shrink to 10 % of the fresh cap (floor ~300 chars). Pre-masking snapshot saved so the summarizer can access un-masked originals later.

3. **Apply pass**: `applyFadingCaps` walks only the messages that arrived since the last call (or everything after an escalation) and rewrites the ones above their cap.

4. **Pruning split**: `getMessagesWithinTokenLimit` determines which messages fit (`context`) and which overflow (`messagesToRefine`). Messages are kept newest-first.

5. **Summarization trigger**: If `messagesToRefine` is non-empty, `shouldTriggerSummarization` evaluates the configured trigger (or defaults to "any pruned messages"). `shouldSkipSummarization` only blocks when the message count hasn't changed since the last summary (prevents re-summarizing identical content). If triggered: **full compaction** fires.

### Full Compaction

When summarization fires:

- The **entire conversation** (un-masked originals from the snapshot) is sent to the summarizer — not just the dropped messages.
- The summarizer produces a structured checkpoint covering the full conversation history.
- Graph state is wiped completely (`createRemoveAllMessage()`) — no surviving messages.
- The summary is stored on `AgentContext` but **not** injected into the system prompt (doesn't inflate `instructionTokens`).

### Post-Compaction Clean Slate

After compaction, the message array is empty. On the next agent node turn:

- The system runnable detects `messages.length === 0` with a mid-run summary present.
- It injects `[SystemMessage(instructions), HumanMessage(summary)]`.
- The model reads the checkpoint as a user message and continues naturally — making tool calls or responding.
- The summary competes for message budget rather than permanently reducing the instruction ceiling.

### Summarization Invocation

Raw conversation messages are sent to the LLM via `attemptInvoke` with the summarization instruction appended as the final HumanMessage. Tools are bound so providers that require tool definitions (e.g. Bedrock) accept the messages and cache-capable providers can reuse the tool-schema prefix. The summarization model does not currently pass through `AgentContext.systemRunnable`, so exact replay of the main request's full system + tools + messages prefix is not guaranteed.

If the primary call fails, fallback providers are attempted (via `tryFallbackProviders`). If all providers fail, a metadata stub is generated mechanically — no LLM call, just tool names and message counts.

### Summarization Prompt

The prompt is written in the tone of a user directing the assistant — assertive, first-person, active voice:

> "Hold on, before you continue I need you to write me a checkpoint of everything so far..."

This prevents the model from continuing to roleplay or respond to the conversation instead of producing a structured checkpoint.

### Emergency Truncation

If masking + fit-to-budget still produce an **empty context** (no messages fit at all), a deeper, temporary tier at which one complete tool exchange (result plus call input) fits within a per-message share of the effective budget is applied to a clone of the messages before pruning is retried. The latched tier is left alone: that share depends on the message count, and latching it would pin every future result to one transient event.

### Cross-Run Behavior

- `initialSummary` from the prior run is included in the **system prompt** via `buildInstructionsString`.
- `formatAgentMessages` drops messages before the summary boundary in the message chain.
- The model sees the system prompt (with summary) + the user's new message.
- Mid-run summaries do NOT go into the system prompt — they use the HumanMessage injection on clean slate.

---

## When Summarization is Disabled

### Pipeline (every agent node turn)

1. **< 80% pressure**: No modifications.

2. **80%+ pressure — Observation masking**: Same consumed-only masking as the summarization-enabled path. Consumed ToolMessages masked, unconsumed left intact, AI messages untouched.

3. **80%+ pressure — Context pressure fading**: The fading tier deepens by extra rungs on top of the fit rung, halving the cap budget per rung:

   | Pressure | Extra rungs | Budget factor |
   | -------- | ----------- | ------------- |
   | 80%      | 0           | 1.0           |
   | 85%      | 1           | 0.5           |
   | 90%      | 2           | 0.25          |
   | 99%      | 4           | 0.0625        |

   Every tool result shares one cap at a given tier; the consumed/fresh distinction from masking is the only per-message difference. The tier never relaxes, so a conversation hovering around a threshold does not flip between bands.

4. **Position-based context pruning** (if `contextPruningConfig.enabled`): Additional position-based degradation of old tool results.

5. **Pruning**: `getMessagesWithinTokenLimit` drops oldest messages to fit budget. Orphan repair strips unpaired tool_use/tool_result blocks.

6. **Emergency truncation** (if pruning produces empty context): A temporary tier at which one complete tool exchange fits within the per-message share of the effective budget is applied to a clone, then pruning is retried. The latched tier is not changed.

### Key Difference from Enabled Path

Messages that get pruned are **gone** — no summary captures them. The model loses context of what it did in earlier turns. This is acceptable for simpler conversations but problematic for long agentic runs with many tool calls.

---

## Summary Injection Locations

| Scenario                     | Where                                       | Why                                                        |
| ---------------------------- | ------------------------------------------- | ---------------------------------------------------------- |
| Mid-run post-compaction      | `HumanMessage` when `messages.length === 0` | Clean slate; doesn't inflate `instructionTokens`           |
| Mid-run subsequent turns     | Nowhere — already consumed                  | Model read the checkpoint and is working from it           |
| Cross-run (`initialSummary`) | System prompt via `buildInstructionsString` | One-time cost; model needs it alongside user's new message |
| No summary                   | N/A                                         | Normal `[SystemMessage, ...messages]`                      |

---

## Observation Masking Details

A ToolMessage is **consumed** when a subsequent AI message exists with substantive text content — meaning the model has read and acted on the result. Detection walks backwards from the end of the messages array:

1. Find the first AI message with non-empty text content (not just tool calls).
2. All ToolMessages before that point are consumed.
3. ToolMessages after that point are unconsumed.

Masking uses `truncateToolResultContent` with a ~300 char limit, producing head+tail truncations that preserve the beginning and end of the result. This is more informative than a synthetic placeholder — the model can still see what the tool returned at a glance.

---

## Key Design Decisions

1. **Summarization IS the pruning** — when enabled, no messages are hard-pruned without being captured in a summary first. The summary replaces dropped messages.

2. **Full compaction over rolling summary** — each compaction sees the entire conversation, avoiding compound information loss from summarizing summaries-of-summaries.

3. **Summary as user message, not system prompt** — mid-run summaries are injected as a HumanMessage to avoid inflating `instructionTokens` and shrinking the available budget for messages.

4. **Observation masking for both paths** — consumed tool results are masked regardless of whether summarization is enabled. The model's own AI message text preserves what it concluded from those results.

5. **No events XML** — with full compaction the LLM sees the entire conversation each time, making structured event extraction redundant with the checkpoint's markdown content.

6. **Computed `instructionTokens`** — `instructionTokens` is a getter (`systemMessageTokens + toolSchemaTokens`), not a manually tracked value. This eliminates the category of bugs where instruction overhead gets out of sync from increments/decrements in multiple places.

---

## Configuration Reference

### `librechat.yaml` — `summarization` block

| Field              | Type           | Default                    | Description                                                                            |
| ------------------ | -------------- | -------------------------- | -------------------------------------------------------------------------------------- |
| `enabled`          | `boolean`      | `true`                     | Top-level kill switch. Set `false` to disable summarization globally.                  |
| `provider`         | `string`       | Agent's own provider       | LLM provider for the summarizer (e.g. `anthropic`, `bedrock`).                         |
| `model`            | `string`       | Agent's own model          | Model for summarization calls.                                                         |
| `parameters`       | `object`       | `{}`                       | Extra LLM constructor params (temperature, etc.). Also accepts `maxSummaryTokens`.     |
| `prompt`           | `string`       | Built-in checkpoint prompt | Custom prompt for initial summarization.                                               |
| `updatePrompt`     | `string`       | Built-in update prompt     | Custom prompt for re-compaction when a prior summary exists. Falls back to `prompt`.   |
| `trigger`          | `object`       | Always on overflow         | When to fire summarization. See trigger types below.                                   |
| `reserveRatio`     | `number (0-1)` | `0.05`                     | Fraction of token budget reserved as headroom. Pruning triggers at `budget * (1 - r)`. |
| `maxSummaryTokens` | `number`       | Provider/client default    | Optional max output tokens for the summarization model.                                |
| `contextPruning`   | `object`       | disabled                   | Position-based context pruning (only applies when summarization is disabled).          |

### Trigger types (`trigger` field)

| Type                 | Value     | Behavior                                                                    |
| -------------------- | --------- | --------------------------------------------------------------------------- |
| `token_ratio`        | `0.0-1.0` | Fire when `1 - effectiveRemainingContextTokens / maxContextTokens >= value` |
| `remaining_tokens`   | `number`  | Fire when `effectiveRemainingContextTokens <= value`                        |
| `messages_to_refine` | `number`  | Fire when `messagesToRefine.length >= value`                                |
| _(not set)_          | —         | Fire whenever pruning drops any messages (default)                          |

### `contextPruning` sub-config (summarization-disabled path only)

| Field                  | Type            | Default | Description                                                 |
| ---------------------- | --------------- | ------- | ----------------------------------------------------------- |
| `enabled`              | `boolean`       | `false` | Enable position-based tool result degradation.              |
| `keepLastAssistants`   | `number (0-10)` | —       | Number of recent assistant turns to protect from pruning.   |
| `softTrimRatio`        | `number (0-1)`  | —       | Position threshold for head+tail soft-trim.                 |
| `hardClearRatio`       | `number (0-1)`  | —       | Position threshold for full content replacement.            |
| `minPrunableToolChars` | `number`        | —       | Minimum chars before a tool result is eligible for pruning. |

### `parameters` sub-fields (extracted before passing to LLM)

| Field              | Type     | Default                 | Description                                     |
| ------------------ | -------- | ----------------------- | ----------------------------------------------- |
| `maxSummaryTokens` | `number` | Provider/client default | Can also be set here (same as top-level field). |
