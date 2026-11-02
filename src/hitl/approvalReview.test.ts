import { describe, expect, it } from '@jest/globals';
import { createToolApprovalReviewEvidence, cloneToolApprovalInterruptPayload } from './approvalReview';

describe('approval review evidence', () => {
  const request = {
    tool_call_id: 'call_1',
    name: 'bash',
    arguments: { command: 'git status' },
  };
  const reviewConfig = {
    tool_call_id: 'call_1',
    action_name: 'bash',
    allowed_decisions: ['approve', 'reject'],
  };

  it('accepts a complete one-to-one approval payload', () => {
    expect(
      createToolApprovalReviewEvidence('interrupt_1', {
        type: 'tool_approval',
        action_requests: [request],
        review_configs: [reviewConfig],
      })
    ).toEqual({
      interruptId: 'interrupt_1',
      payload: {
        type: 'tool_approval',
        action_requests: [request],
        review_configs: [reviewConfig],
      },
    });
  });

  it('preserves an empty allowlist as fail-closed review evidence', () => {
    expect(
      createToolApprovalReviewEvidence('interrupt_1', {
        type: 'tool_approval',
        action_requests: [request],
        review_configs: [{ ...reviewConfig, allowed_decisions: [] }],
      })
    ).toBeDefined();
  });

  it('snapshots nested approval data before exposing it as evidence', () => {
    const payload = {
      type: 'tool_approval' as const,
      action_requests: [
        {
          ...request,
          arguments: { command: 'reviewed', options: { cwd: '/safe' } },
        },
      ],
      review_configs: [reviewConfig],
    };
    const evidence = createToolApprovalReviewEvidence('interrupt_1', payload);

    payload.action_requests[0].arguments.command = 'mutated';
    payload.action_requests[0].arguments.options.cwd = '/unsafe';

    expect(evidence?.payload.action_requests[0].arguments).toEqual({
      command: 'reviewed',
      options: { cwd: '/safe' },
    });
  });

  it.each([
    {
      label: 'null request',
      action_requests: [null],
      review_configs: [reviewConfig],
    },
    {
      label: 'missing arguments',
      action_requests: [{ tool_call_id: 'call_1', name: 'bash' }],
      review_configs: [reviewConfig],
    },
    {
      label: 'mismatched review identity',
      action_requests: [request],
      review_configs: [{ ...reviewConfig, tool_call_id: 'call_2' }],
    },
    {
      label: 'unknown decision',
      action_requests: [request],
      review_configs: [
        { ...reviewConfig, allowed_decisions: ['approve', 'execute_anyway'] },
      ],
    },
    {
      label: 'duplicate tool-call ids',
      action_requests: [request, request],
      review_configs: [reviewConfig, reviewConfig],
    },
  ])(
    'rejects a malformed $label payload',
    ({ action_requests, review_configs }) => {
      expect(
        createToolApprovalReviewEvidence('interrupt_1', {
          type: 'tool_approval',
          action_requests,
          review_configs,
        })
      ).toBeUndefined();
    }
  );

  it.each([
    { resolve: () => 'resolved' },
    { resolution: Symbol('resolved') },
    new Date(0),
  ])('detaches approval fields without cloning opaque checkpoint siblings: %p', (opaque) => {
    const manifest = { pendingInterrupts: [{ interruptId: 'sibling', payload: opaque }] };
    const payload = {
      type: 'tool_approval' as const,
      action_requests: [{ ...request, arguments: { command: 'reviewed', options: { cwd: '/safe' } } }],
      review_configs: [{ ...reviewConfig, allowed_decisions: ['approve', 'reject'] }],
      subagent: { run_id: 'child', agent_id: 'member', subagent_type: 'team' },
      __librechat_subagent_resume_manifest: manifest,
    };
    const cloned = cloneToolApprovalInterruptPayload(payload);
    expect(cloned.__librechat_subagent_resume_manifest).toBe(manifest);
    expect(cloned.__librechat_subagent_resume_manifest.pendingInterrupts[0].payload).toBe(opaque);
    cloned.action_requests[0].arguments.options.cwd = '/unsafe';
    cloned.review_configs[0].allowed_decisions.length = 0;
    cloned.subagent.agent_id = 'foreign';
    expect(payload.action_requests[0].arguments.options.cwd).toBe('/safe');
    expect(payload.review_configs[0].allowed_decisions).toEqual(['approve', 'reject']);
    expect(payload.subagent.agent_id).toBe('member');
  });

});
