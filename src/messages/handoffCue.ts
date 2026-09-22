// src/messages/handoffCue.ts
import { HumanMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import { stampSyntheticProviderMessage } from './provenance';

const HANDOFF_CUE_MESSAGE_ID = '__handoff_cue_message_id';
export const INSTRUCTIONLESS_HANDOFF_CUE =
  'Continue as the receiving agent using the preceding user request and context.';

/** Scope transport-only grounding to this recipient's incoming assistant turn.
 * Every agent entry resets the marker, including direct edges and cycles. The
 * reducer has already assigned the source message's id, including on replay. */
export function withInstructionlessHandoffCue(
  config: RunnableConfig | undefined,
  tail?: BaseMessage
): RunnableConfig {
  return {
    ...config,
    metadata: {
      ...config?.metadata,
      [HANDOFF_CUE_MESSAGE_ID]:
        tail?.getType() === 'ai' ? (tail.id ?? null) : null,
    },
  };
}

/** Only provider projections get this cue. Matching the incoming turn's id
 * prevents it leaking into later tool iterations or deliberate assistant
 * prefill. It is independent of transport provider and run-produced ids, so
 * gateways and checkpoint replays follow the same handoff contract. */
export function appendInstructionlessHandoffCue(
  messages: BaseMessage[],
  config?: RunnableConfig
): BaseMessage[] {
  const tail = messages.at(-1);
  const sourceId = config?.metadata?.[HANDOFF_CUE_MESSAGE_ID];
  if (
    typeof sourceId !== 'string' ||
    sourceId === '' ||
    tail?.getType() !== 'ai' ||
    tail.id !== sourceId
  ) {
    return messages;
  }
  return [
    ...messages,
    stampSyntheticProviderMessage(
      new HumanMessage({
        content: INSTRUCTIONLESS_HANDOFF_CUE,
        additional_kwargs: { role: 'user', isMeta: true, source: 'routing' },
      })
    ),
  ];
}

/**
 * Bracketed-meta convention, like the handoff path's
 * `[Processed tool result and transferring to …]` bridge. The wording makes
 * two things unambiguous to the model: the assistant turn above is FINISHED,
 * and it belongs to a previous stage — so the successor answers as itself
 * instead of continuing someone else's sentence.
 */
export const PREDECESSOR_HANDOFF_CUE =
  '[The assistant message above is the completed output of a previous ' +
  'agent. Respond now according to your own role and instructions.]';

/**
 * Appends a user-turn handoff cue when a payload ends with an assistant turn
 * that THIS RUN produced — which only happens when a different agent in a
 * multi-agent workflow produced it (an agent's own self-loops always re-enter
 * on a tool result or an injected user turn).
 *
 * Why: providers with prefill semantics (Anthropic, Bedrock-Claude) treat a
 * trailing assistant message as a prefill and CONTINUE it. A bare direct-edge
 * successor therefore speaks in its predecessor's voice — or, when the
 * trailing turn reads complete (a preemption steer's short resume, say),
 * returns empty content (danny-avila/agents#345, reproduced live 3/3).
 * Handoff edges with instructions and prompt-instruction edges already break
 * the prefill with a user turn; this closes the same gap for bare edges.
 *
 * Fail-safe OFF by provenance: the trailing payload message must be one the
 * run itself produced (`isRunProduced`, backed by the graph's run-produced id
 * set — immune to summarization compaction, which rewrites the live array
 * and stales index-based boundaries). Host-supplied trailing assistant
 * turns (deliberate prefill flows) never match — the run has not produced
 * them — so single-agent prefill behavior is untouched. Wire-only: the cue is
 * appended to the provider projection, never to graph state or host history.
 */
export function appendPredecessorHandoffCue(
  messages: BaseMessage[],
  isRunProduced: ((message: BaseMessage) => boolean) | undefined
): BaseMessage[] {
  const last = messages.at(-1);
  if (last == null || last.getType() !== 'ai') {
    return messages;
  }
  if (isRunProduced == null || !isRunProduced(last)) {
    return messages;
  }
  return [
    ...messages,
    stampSyntheticProviderMessage(
      new HumanMessage({
        content: PREDECESSOR_HANDOFF_CUE,
        additional_kwargs: { role: 'user', isMeta: true, source: 'handoff' },
      })
    ),
  ];
}

/**
 * Strips a trailing handoff cue. The counterpart for the serving-provider
 * funnel: an Anthropic-like PRIMARY bakes the cue into its measured payload,
 * and a tolerant fallback (OpenAI, Mistral, Bedrock-Nova) re-sending that
 * payload must not ship the Claude-only synthetic turn. Identity on the
 * no-op path.
 */
export function removePredecessorHandoffCue(
  messages: BaseMessage[]
): BaseMessage[] {
  const last = messages.at(-1);
  if (
    last == null ||
    last.getType() !== 'human' ||
    last.additional_kwargs.source !== 'handoff' ||
    last.content !== PREDECESSOR_HANDOFF_CUE
  ) {
    return messages;
  }
  return messages.slice(0, -1);
}
