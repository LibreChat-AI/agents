/**
 * Summarization primitives shared by the in-run summarize node and by callers
 * that compact a conversation outside a run. Kept apart from `node.ts` so the
 * package can export them without exporting the graph node itself.
 */

/**
 * Instruction that follows the summary body inside the carrier. Private on
 * purpose: it is only ever correct alongside `buildSummaryCarrierText`, and a
 * caller that reaches for the text separately is a caller that can drift from
 * the accounting.
 */
const SUMMARY_CARRIER_INSTRUCTION =
  'This is your own checkpoint: you wrote it to preserve context after compaction. Pick up where you left off based on the summary above. Do not repeat prior tasks, information or acknowledge this checkpoint message directly.';

/**
 * Wraps a persisted summary in the carrier it is re-injected as, ahead of the
 * messages that survived compaction.
 *
 * A stored summary costs what this returns, not what its body costs, so a
 * caller budgeting for one measures this rather than adding a remembered
 * constant to the bare text. The wrapper alone is ~48 tokens on `o200k_base`
 * and more on Anthropic: too much to leave out of a context calculation, and
 * too easy to get wrong from memory once the instruction is edited.
 */
export function buildSummaryCarrierText(summaryText: string): string {
  return (
    '<summary>\n' +
    summaryText +
    '\n</summary>\n\n' +
    SUMMARY_CARRIER_INSTRUCTION
  );
}

/** Structured checkpoint prompt for fresh summarization (no prior summary). */
export const DEFAULT_SUMMARIZATION_PROMPT = `Hold on, before you continue I need you to write me a checkpoint of everything so far. Your context window is filling up and this checkpoint replaces the messages above, so capture everything you need to pick right back up.

Do not call tools, perform calculations, or continue the task. Summarize only the existing transcript.

Don't second-guess or fact-check anything you did, your tool results reflect exactly what happened. If a tool result appears truncated, that's just a display artifact from context management: the tool executed fully. Just record what you did and what you observed. Only the checkpoint, don't respond to me or continue the conversation.

## Checkpoint

## Goal
What I asked you to do and any sub-goals you identified.

## Constraints & Preferences
Any rules, preferences, or configuration I established.

## Progress
### Done
- What you completed and the outcomes

### In Progress
- What you're currently working on

## Key Decisions
Decisions you made and why.

## Next Steps
Concrete task actions remaining, in priority order.

## Critical Context
Exact identifiers, names, error messages, URLs, and details you need to preserve verbatim.

Rules:
- Record what you did and observed, don't judge or re-evaluate it
- For each tool call: the tool name, key inputs, and the outcome
- Preserve exact identifiers, names, errors, and references verbatim
- Short declarative sentences
- Skip empty sections`;

/** Prompt for re-compaction when a prior summary exists. */
export const DEFAULT_UPDATE_SUMMARIZATION_PROMPT = `Hold on again, update your checkpoint. Merge the new messages into your existing checkpoint and give me a single consolidated replacement.

Do not call tools, perform calculations, or continue the task. Summarize only the existing transcript.

Keep it roughly the same length as your last checkpoint. Compress older details to make room for what's new, don't just append. Give recent actions more detail, compress older items to one-liners.

Don't fact-check or second-guess anything, your tool results are ground truth. If a tool result appears truncated, that's just a display artifact: the tool executed fully. Only the checkpoint, don't respond to me or continue the conversation.

Rules:
- Merge new progress into existing sections, don't duplicate headers
- Compress older completed items into one-line entries
- Move items from "In Progress" to "Done" when you completed them
- Update "Next Steps" to reflect current task priorities.
- For each new tool call: the tool name, key inputs, and the outcome
- Preserve exact identifiers, names, errors, and references verbatim
- Skip empty sections`;

const SUMMARIZATION_PARAM_KEYS = new Set(['maxSummaryTokens']);

export function separateSummarizationParameters(
  parameters: Record<string, unknown>
): {
  llmParams: Record<string, unknown>;
  maxSummaryTokens?: number;
} {
  const llmParams: Record<string, unknown> = {};
  let maxSummaryTokens: number | undefined;

  for (const [key, value] of Object.entries(parameters)) {
    if (SUMMARIZATION_PARAM_KEYS.has(key)) {
      if (
        key === 'maxSummaryTokens' &&
        typeof value === 'number' &&
        value > 0
      ) {
        maxSummaryTokens = value;
      }
    } else {
      llmParams[key] = value;
    }
  }

  return { llmParams, maxSummaryTokens };
}

export function buildSummarizationInstruction(
  promptText: string,
  updatePromptText: string | undefined,
  priorSummaryText?: string,
  semanticIndexAppendix = '',
  priorSummaryInPrefix = false
): string {
  const prior = priorSummaryText?.trim() ?? '';
  const effectivePrompt = prior ? (updatePromptText ?? promptText) : promptText;
  const parts = semanticIndexAppendix
    ? [semanticIndexAppendix, '\n\n', effectivePrompt]
    : [effectivePrompt];
  if (prior) {
    parts.push(priorSummaryInPrefix
      ? '\n\n<previous-summary>The prior checkpoint is already present in the request prefix above; update that checkpoint using the conversation above.</previous-summary>'
      : `\n\n<previous-summary>\n${prior}\n</previous-summary>`);
  }
  return parts.join('');
}

/** Why a summarize-only run could not attempt its summary. */
export type ManualSummarizationSkipReason =
  | 'disabled'
  | 'exhausted'
  | 'instructions_exceed_budget'
  | 'nothing_to_summarize';

/**
 * A summarize-only run asked for a summary the graph could not attempt.
 * Thrown rather than ending the run quietly: the summary is the whole result
 * of such a run, so a host left with neither a summary nor an error would
 * have nothing to show the user.
 */
export class ManualSummarizationSkippedError extends Error {
  readonly reason: ManualSummarizationSkipReason;

  constructor(reason: ManualSummarizationSkipReason, message: string) {
    super(message);
    this.name = 'ManualSummarizationSkippedError';
    this.reason = reason;
  }
}
