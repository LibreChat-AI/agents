import type { RunnableConfig } from '@langchain/core/runnables';
import type {
  ToolApprovalInterruptPayload,
  ToolApprovalRequest,
  ToolApprovalReviewConfig,
} from '@/types/hitl';
import { stableStringify } from '@/tools/eagerEventExecution';
import { isToolApprovalInterrupt } from '@/types/hitl';

/**
 * Private resume-only config entry populated from the checkpointed interrupt.
 * Hosts must never need to construct or inspect this value.
 */
export const TOOL_APPROVAL_REVIEW_CONFIG_KEY =
  '__librechat_tool_approval_review';

export interface ToolApprovalReviewEvidence {
  interruptId: string;
  payload: ToolApprovalInterruptPayload;
  owner?: string;
}

export interface ReviewedToolApproval {
  request: ToolApprovalRequest;
  reviewConfig: ToolApprovalReviewConfig;
}

const APPROVAL_DECISIONS = new Set(['approve', 'reject', 'edit', 'respond']);

/** Detach approval fields without deep-cloning checkpoint-owned metadata. */
export function cloneToolApprovalInterruptPayload<T>(payload: T): T {
  if (!isToolApprovalInterrupt(payload)) return payload;
  return {
    ...payload,
    ...structuredClone({
      action_requests: payload.action_requests,
      review_configs: payload.review_configs,
      ...(payload.subagent == null ? {} : { subagent: payload.subagent }),
    }),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function hasValidApprovalShape(payload: ToolApprovalInterruptPayload): boolean {
  if (
    !Array.isArray(payload.action_requests) ||
    !Array.isArray(payload.review_configs) ||
    payload.action_requests.length !== payload.review_configs.length
  ) {
    return false;
  }

  const toolCallIds = new Set<string>();
  return payload.action_requests.every((request, index) => {
    const reviewConfig = payload.review_configs[index];
    if (!isRecord(request) || !isRecord(reviewConfig)) {
      return false;
    }
    const toolCallId = request.tool_call_id;
    const toolName = request.name;
    const allowedDecisions = reviewConfig.allowed_decisions;
    if (
      typeof toolCallId !== 'string' ||
      toolCallId.length === 0 ||
      toolCallIds.has(toolCallId) ||
      typeof toolName !== 'string' ||
      toolName.length === 0 ||
      !isRecord(request.arguments) ||
      (request.description != null &&
        typeof request.description !== 'string') ||
      reviewConfig.tool_call_id !== toolCallId ||
      reviewConfig.action_name !== toolName ||
      !Array.isArray(allowedDecisions) ||
      !allowedDecisions.every(
        (decision) =>
          typeof decision === 'string' && APPROVAL_DECISIONS.has(decision)
      )
    ) {
      return false;
    }
    toolCallIds.add(toolCallId);
    return true;
  });
}

/** Build trusted review evidence from the interrupt restored by `Run`. */
export function createToolApprovalReviewEvidence(
  interruptId: string | undefined,
  payload: unknown,
  owner?: string
): ToolApprovalReviewEvidence | undefined {
  if (
    typeof interruptId !== 'string' ||
    interruptId.length === 0 ||
    !isToolApprovalInterrupt(payload) ||
    !hasValidApprovalShape(payload)
  ) {
    return undefined;
  }
  return {
    interruptId,
    payload: cloneToolApprovalInterruptPayload(payload),
    ...(owner == null ? {} : { owner }),
  };
}

/** Read only well-shaped evidence from a ToolNode's runnable config. */
export function getToolApprovalReviewEvidence(
  config: RunnableConfig,
  owner?: string
): ToolApprovalReviewEvidence | undefined {
  const candidate = config.configurable?.[TOOL_APPROVAL_REVIEW_CONFIG_KEY];
  if (candidate == null || typeof candidate !== 'object') {
    return undefined;
  }
  const {
    interruptId,
    payload,
    owner: evidenceOwner,
  } = candidate as {
    interruptId?: unknown;
    payload?: unknown;
    owner?: unknown;
  };
  if (
    evidenceOwner != null &&
    (typeof evidenceOwner !== 'string' ||
      (owner != null && owner !== evidenceOwner))
  ) {
    return undefined;
  }
  return createToolApprovalReviewEvidence(
    typeof interruptId === 'string' ? interruptId : undefined,
    payload,
    typeof evidenceOwner === 'string' ? evidenceOwner : undefined
  );
}

export function getReviewedToolApproval(
  payload: ToolApprovalInterruptPayload | undefined,
  toolCallId: string | undefined
): ReviewedToolApproval | undefined {
  if (payload == null || toolCallId == null || toolCallId === '') {
    return undefined;
  }
  const request = payload.action_requests.find(
    (candidate) => candidate.tool_call_id === toolCallId
  );
  const reviewConfig = payload.review_configs.find(
    (candidate) => candidate.tool_call_id === toolCallId
  );
  if (request == null || reviewConfig == null) {
    return undefined;
  }
  return { request, reviewConfig };
}

function decisionsEqual(
  left: ReadonlyArray<string>,
  right: ReadonlyArray<string>
): boolean {
  if (left.length !== right.length) {
    return false;
  }
  const sortedLeft = [...left].sort();
  const sortedRight = [...right].sort();
  return sortedLeft.every((value, index) => value === sortedRight[index]);
}

/**
 * Bind a resume decision to the exact proposal the reviewer saw. Description
 * text is deliberately excluded: it explains policy but cannot change the
 * side effect. Tool identity, normalized arguments and available decisions
 * are execution-authoritative.
 */
export function toolApprovalProposalMatches(
  current: ReviewedToolApproval,
  reviewed: ReviewedToolApproval
): boolean {
  return (
    current.request.tool_call_id === reviewed.request.tool_call_id &&
    current.request.name === reviewed.request.name &&
    current.reviewConfig.tool_call_id === reviewed.reviewConfig.tool_call_id &&
    current.reviewConfig.action_name === reviewed.reviewConfig.action_name &&
    stableStringify(current.request.arguments) ===
      stableStringify(reviewed.request.arguments) &&
    decisionsEqual(
      current.reviewConfig.allowed_decisions,
      reviewed.reviewConfig.allowed_decisions
    )
  );
}

/** Preserve batch order as part of the decision-to-request binding. */
export function toolApprovalPayloadMatches(
  current: ToolApprovalInterruptPayload,
  reviewed: ToolApprovalInterruptPayload
): boolean {
  if (
    current.action_requests.length !== reviewed.action_requests.length ||
    current.review_configs.length !== reviewed.review_configs.length
  ) {
    return false;
  }
  return current.action_requests.every((request, index) => {
    const currentReviewConfig = current.review_configs[index];
    const reviewedRequest = reviewed.action_requests[index];
    const reviewedReviewConfig = reviewed.review_configs[index];
    return toolApprovalProposalMatches(
      { request, reviewConfig: currentReviewConfig },
      { request: reviewedRequest, reviewConfig: reviewedReviewConfig }
    );
  });
}
