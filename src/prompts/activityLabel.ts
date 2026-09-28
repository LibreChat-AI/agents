import type {
  ActivityLabelToolEntry,
  ActivityPhaseEntry,
} from '@/types/activityLabel';
import type { ResolvedLangfuseToolOutputTracingConfig } from '@/langfuseRuntimeContext';
import { shouldRedactTool } from '@/langfuseToolOutputTracing';

/**
 * Default system prompt for fast-model activity labeling.
 *
 * Style synthesized from Claude Code's tool-use summary prompt (git-subject
 * register, past tense, distinctive nouns) and claude.ai's observed group
 * headers (5–9 words describing a mixed reasoning + tool block, e.g.
 * "Synthesized version data and curated comparative framework").
 */
export const ACTIVITY_LABEL_PROMPT = `Write a short label describing what this block of agent activity accomplished. It appears as the header of a collapsed activity group in a chat UI.

Rules:
- 5 to 9 words, past-tense verb first
- Name the most distinctive subject (file, API, topic); drop articles and filler
- Describe outcomes, not mechanics; if something failed, say so plainly
- Output only the label — no quotes, no punctuation at the end, no preamble

Examples:
- Searched Node.js release notes and changelogs
- Compared runtime versions across official sources
- Fixed failing auth middleware tests
- Read project config and dependency manifests
- Attempted database migration, hit permission errors`;

/** Default system prompt for a run-wide parent activity phase. */
export const ACTIVITY_PHASE_LABEL_PROMPT = `Summarize what this phase of an agent run accomplished. The result appears as the header of one collapsed parent group containing several activities.

Rules:
- One line, 8 to 18 words, past tense
- Lead with the concrete outcome and name the most distinctive subject
- Synthesize the phase; do not enumerate, count, or restate individual activities
- Describe failures plainly when they are the phase's material outcome
- Never mention tool names, calls, arguments, reasoning, commentary, or activity counts
- Output only the summary — no quotes, no trailing punctuation, no preamble

Examples:
- Reconciled authentication behavior and fixed the failing session refresh path
- Compared deployment options and documented the safest production rollout
- Investigated database latency but could not confirm the suspected index regression

Bad examples:
- Used three tools to inspect files and run tests
- Searched code, read configuration, and updated middleware`;

/** Hard ceiling across every activity/context section in one phase request. */
export const ACTIVITY_PHASE_PROMPT_MAX_LENGTH = 12_000;

/** Truncates a serialized value for the label prompt. */
export function truncateForLabel(value: string, maxLength: number): string {
  if (value.length <= maxLength) {
    return value;
  }
  return value.slice(0, Math.max(0, maxLength - 1)) + '…';
}

/**
 * Reduces a committed label to bounded single-line data.
 *
 * Sections in this prompt are delimited by blank lines, so a label carrying
 * embedded newlines could otherwise forge an apparent entries section or
 * `Header:` cue. Unlike every other input here, previous labels re-enter
 * the prompt on EVERY later batch, so one malformed result — plain model
 * noncompliance, or injection surfacing through a tool result — would
 * persistently steer unrelated later labels rather than affecting one. The
 * clip bounds the same way `lastAssistantText` and reasoning excerpts are
 * bounded: oversized headers must not inflate later requests past the fast
 * model's window and starve the run of labels entirely.
 */
function sanitizePreviousLabel(label: string): string {
  return truncateForLabel(
    label.replace(/\s+/g, ' ').trim(),
    PREVIOUS_LABEL_LIMIT
  );
}

const ABORT_SERIALIZATION = Symbol('abort-label-serialization');

/**
 * Serializes a tool value for the prompt WITHOUT materializing huge JSON:
 * the output is clipped to a few hundred characters anyway, so a multi-
 * megabyte tool result must not be stringified in full on the label path.
 * Strings clip immediately; structured values serialize under a character
 * budget and degrade to a shape summary once it is exhausted.
 */
function serializeForLabel(value: unknown, limit: number): string {
  if (value == null) {
    return '';
  }
  if (typeof value === 'string') {
    return value.length > limit ? value.slice(0, limit + 1) : value;
  }
  let budget = limit * 4;
  try {
    return (
      JSON.stringify(value, (_key, nested: unknown) => {
        if (budget <= 0) {
          throw ABORT_SERIALIZATION;
        }
        if (typeof nested === 'string') {
          const clipped =
            nested.length > limit ? nested.slice(0, limit) : nested;
          budget -= clipped.length;
          return clipped;
        }
        budget -= 8;
        return nested;
      }) ?? ''
    );
  } catch (error) {
    if (error === ABORT_SERIALIZATION) {
      return Array.isArray(value) ? `[Array(${value.length})]` : '[Object]';
    }
    return String(value);
  }
}

const INPUT_CONTEXT_LIMIT = 200;
const MAX_THINKING_EXCERPTS = 4;
const MAX_PREVIOUS_LABELS = 3;
/** Per-label bound. A header is 5-9 words; anything past this is
 *  noncompliance or payload, and previous labels are the one input that
 *  RE-ENTERS the prompt on every later batch of the run. */
const PREVIOUS_LABEL_LIMIT = 200;
/** A label is 5-9 words; no batch needs more than this many entries to
 *  produce one, and the cap keeps a 200-call programmatic batch from
 *  building an enormous prompt out of per-field-bounded pieces. */
const MAX_PROMPT_ENTRIES = 12;
const MAX_PHASE_ACTIVITIES = 12;
const MAX_PHASE_CONTEXT = 3;
const MAX_PHASE_TOOL_ENTRIES = 6;
export const ACTIVITY_PHASE_LABEL_MAX_LENGTH = 160;

export type BuildActivityLabelPromptParams = {
  entries: ActivityLabelToolEntry[];
  charLimit: number;
  thinkingExcerpts?: string[];
  lastAssistantText?: string;
  /**
   * Headers already committed for earlier batches in this run, in run order
   * with the most recent last. Rendered ahead of the block context so the
   * label continues the run's story instead of restating a line the user is
   * already reading. Capped at {@link MAX_PREVIOUS_LABELS}.
   */
  previousLabels?: string[];
  /**
   * Resolved tool-output tracing policy. The label prompt becomes Langfuse
   * generation input, so outputs/errors excluded from tracing (global
   * disable or `redactedToolNames`) must never appear in it — the same
   * redaction the span processor applies to structured tool observations.
   */
  redaction?: ResolvedLangfuseToolOutputTracingConfig;
};

/**
 * Builds the user prompt for a fast-model activity label. Pure — exported
 * for direct testing of redaction and truncation behavior.
 */
export function buildActivityLabelPrompt({
  entries,
  charLimit,
  thinkingExcerpts,
  lastAssistantText,
  previousLabels,
  redaction,
}: BuildActivityLabelPromptParams): string {
  const clip = truncateForLabel;
  /** Reasoning and intent text can quote tool output verbatim — including
   *  output from EARLIER calls to a redacted tool that this batch does not
   *  contain — so any active policy (global disable or a configured
   *  redacted-name list) drops both wholesale. There is no reliable way to
   *  scrub a quoted fragment out of free-form model prose. */
  const excerptsRedacted =
    redaction != null &&
    (redaction.enabled === false || redaction.redactedToolNames.size > 0);
  const sections: string[] = [];
  /** Previous labels are free-form model prose too, and per-agent overlays
   *  mean an earlier header may have been generated under ANOTHER agent's
   *  weaker policy — so they share the excerpts' wholesale drop rather than
   *  letting a handoff leak a looser agent's phrasing into this trace. */
  if (
    !excerptsRedacted &&
    previousLabels != null &&
    previousLabels.length > 0
  ) {
    const recent = previousLabels
      .slice(-MAX_PREVIOUS_LABELS)
      .map(sanitizePreviousLabel)
      /** A label that sanitizes to nothing carries no story to continue;
       *  rendering it would leave a bare bullet implying a missing header. */
      .filter((label) => label.length > 0);
    if (recent.length > 0) {
      sections.push(
        'Previous headers in this run (most recent last):\n' +
          recent.map((label) => `- ${label}`).join('\n')
      );
    }
  }
  /** Intent text is free-form assistant prose that can quote a redacted
   *  tool result just as reasoning can, so it shares the excerpts' fate. */
  if (
    !excerptsRedacted &&
    lastAssistantText != null &&
    lastAssistantText.length > 0
  ) {
    sections.push(
      `Intent (assistant's last message): ${clip(lastAssistantText, INPUT_CONTEXT_LIMIT)}`
    );
  }
  if (
    !excerptsRedacted &&
    thinkingExcerpts != null &&
    thinkingExcerpts.length > 0
  ) {
    sections.push(
      'Reasoning excerpts:\n' +
        thinkingExcerpts
          .slice(0, MAX_THINKING_EXCERPTS)
          .map((excerpt) => `- ${clip(excerpt, charLimit)}`)
          .join('\n')
    );
  }
  if (entries.length > 0) {
    const shown = entries.slice(0, MAX_PROMPT_ENTRIES);
    const omitted = entries.length - shown.length;
    sections.push(
      /** Frames the list as reference material, not the thing to
       *  transcribe. Ported from LibreChat's fallback builder (its
       *  runtime.ts documents that without this the model "hands back a
       *  transcription" of the list) after the eval harness measured it
       *  across three independent sweeps: fewer template-redundancy and
       *  length violations than a bare `Tool calls:` heading, with no
       *  per-case regressions (agents #360). */
      'What it called, and what came back (do not restate these):\n' +
        shown
          .map((entry) => {
            const input = clip(
              serializeForLabel(entry.toolInput, charLimit),
              charLimit
            );
            const redacted =
              redaction != null && shouldRedactTool(entry.toolName, redaction);
            let outcome: string;
            if (redacted) {
              outcome = redaction.redactionText;
            } else if (entry.status === 'error') {
              outcome = `ERROR: ${clip(entry.error ?? 'unknown error', charLimit)}`;
            } else {
              outcome = clip(
                serializeForLabel(entry.toolOutput, charLimit),
                charLimit
              );
            }
            return `- ${entry.toolName}(${input}) → ${outcome}`;
          })
          .join('\n') +
        (omitted > 0
          ? `\n- …and ${omitted} more tool ${omitted === 1 ? 'call' : 'calls'}`
          : '')
    );
  }
  /** The fallback builder's terminal cue, measured alongside the heading
   *  (same sweeps). The default system prompt already describes the
   *  output as "the header of a collapsed activity group". */
  sections.push('Header:');
  return sections.join('\n\n');
}

export type BuildActivityPhaseLabelPromptParams = {
  activities: ActivityPhaseEntry[];
  totalActivityCount?: number;
  charLimit: number;
  assistantContext?: string[];
  redaction?: ResolvedLangfuseToolOutputTracingConfig;
  /**
   * Every child label was generated under a policy covering `redaction`, so
   * labels stay usable while reasoning and commentary remain suppressed.
   */
  childLabelsRedacted?: boolean;
};

/**
 * Builds bounded, redaction-aware evidence for a parent activity phase.
 * Committed child labels are preferred; raw tool/reasoning evidence is only
 * used when no child label exists.
 */
export function buildActivityPhaseLabelPrompt({
  activities,
  totalActivityCount,
  charLimit,
  assistantContext,
  redaction,
  childLabelsRedacted = false,
}: BuildActivityPhaseLabelPromptParams): string {
  const freeFormSuppressed =
    redaction != null &&
    (redaction.enabled === false || redaction.redactedToolNames.size > 0);
  const labelsAllowed = !freeFormSuppressed || childLabelsRedacted;
  const sections: string[] = [];
  if (
    !freeFormSuppressed &&
    assistantContext != null &&
    assistantContext.length > 0
  ) {
    const context = assistantContext
      .slice(-MAX_PHASE_CONTEXT)
      .map((text) =>
        truncateForLabel(text.replace(/\s+/g, ' ').trim(), charLimit)
      )
      .filter((text) => text.length > 0);
    if (context.length > 0) {
      sections.push(
        'Intermediate assistant context (do not quote or restate):\n' +
          context.map((text) => `- ${text}`).join('\n')
      );
    }
  }

  let hasDescribableEvidence = false;
  const activityLines = activities
    .slice(0, MAX_PHASE_ACTIVITIES)
    .map((activity, index) => {
      let status = 'completed';
      if (activity.status === 'error') {
        status = 'failed';
      } else if (activity.status === 'partial') {
        status = 'partial';
      }
      if (
        labelsAllowed &&
        activity.label != null &&
        activity.label.trim() !== ''
      ) {
        hasDescribableEvidence = true;
        return `${index + 1}. ${status}: ${truncateForLabel(activity.label.replace(/\s+/g, ' ').trim(), charLimit)}`;
      }

      const evidence: string[] = [];
      if (
        !freeFormSuppressed &&
        activity.thinkingExcerpts != null &&
        activity.thinkingExcerpts.length > 0
      ) {
        evidence.push(
          ...activity.thinkingExcerpts
            .slice(0, MAX_THINKING_EXCERPTS)
            .map((excerpt) =>
              truncateForLabel(excerpt.replace(/\s+/g, ' ').trim(), charLimit)
            )
            .filter((excerpt) => excerpt.length > 0)
            .map((excerpt) => `context=${excerpt}`)
        );
      }
      if (activity.entries != null && activity.entries.length > 0) {
        evidence.push(
          ...activity.entries.slice(0, MAX_PHASE_TOOL_ENTRIES).map((entry) => {
            const entryRedacted =
              redaction != null && shouldRedactTool(entry.toolName, redaction);
            const input = truncateForLabel(
              serializeForLabel(entry.toolInput, charLimit),
              charLimit
            );
            let outcome: string;
            if (entryRedacted) {
              outcome = redaction.redactionText;
            } else if (entry.status === 'error') {
              outcome = `ERROR: ${truncateForLabel(
                entry.error ?? 'unknown error',
                charLimit
              )}`;
            } else {
              outcome = truncateForLabel(
                serializeForLabel(entry.toolOutput, charLimit),
                charLimit
              );
            }
            return `${entry.toolName}(${input}) → ${outcome}`;
          })
        );
      }
      if (evidence.length > 0) {
        hasDescribableEvidence = true;
      }
      return `${index + 1}. ${status}${evidence.length > 0 ? `: ${evidence.join('; ')}` : ''}`;
    });

  if (!hasDescribableEvidence) {
    return '';
  }

  const activityCount = Math.max(activities.length, totalActivityCount ?? 0);
  if (activityCount > MAX_PHASE_ACTIVITIES) {
    activityLines.push(
      `${MAX_PHASE_ACTIVITIES + 1}. …and ${activityCount - MAX_PHASE_ACTIVITIES} more activities`
    );
  }
  sections.push(
    'Activities in this phase (synthesize; do not restate):\n' +
      activityLines.join('\n')
  );
  const terminalCue = '\n\nPhase summary:';
  const evidence = sections.join('\n\n');
  const prompt = evidence + terminalCue;
  if (prompt.length <= ACTIVITY_PHASE_PROMPT_MAX_LENGTH) {
    return prompt;
  }
  const evidenceLimit =
    ACTIVITY_PHASE_PROMPT_MAX_LENGTH - terminalCue.length - 1;
  return `${evidence.slice(0, evidenceLimit).trimEnd()}…${terminalCue}`;
}

/** Normalizes a model result for safe single-row persistence and display. */
export function normalizeActivityPhaseLabel(label: string): string {
  const normalized = label
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^["']|["']$/g, '')
    .replace(/[.!?]+$/g, '');
  return truncateForLabel(normalized, ACTIVITY_PHASE_LABEL_MAX_LENGTH);
}
