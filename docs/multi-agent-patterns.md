# Multi-Agent Patterns

This document explains the different multi-agent patterns supported by the `MultiAgentGraph` class.

## Edge Types

The `MultiAgentGraph` supports two types of edges between agents:

### 1. Handoff Edges (Dynamic Routing)

**Use Case**: When an agent needs to dynamically decide which agent to call next based on the conversation context.

**How it works**: Creates transfer tools that agents can use to explicitly hand off control to another agent.

**Example**:

```typescript
const edges: t.GraphEdge[] = [
  {
    from: 'classifier',
    to: ['technical_expert', 'business_expert', 'general_assistant'],
    description: 'Route to appropriate expert based on query type',
    edgeType: 'handoff', // Optional - this is the default for conditional edges
    condition: (state) => {
      // Dynamic routing logic
      if (state.messages[0].content.includes('technical')) {
        return 'technical_expert';
      }
      // ... more logic
    },
  },
];
```

**Default behavior**:

- Single-to-single edges default to handoff
- Edges with conditions are always handoff
- Edges with `edgeType: 'handoff'` are handoff

### 2. Parallel Edges (Automatic Fan-out/Fan-in)

**Use Case**: When you want multiple agents to process simultaneously without explicit handoff logic.

**How it works**: Creates direct graph edges that cause automatic parallel execution.

**Example**:

```typescript
const edges: t.GraphEdge[] = [
  {
    from: 'researcher',
    to: ['analyst1', 'analyst2', 'analyst3'], // Fan-out
    description: 'Distribute to all analysts for parallel processing',
    edgeType: 'direct', // Explicit parallel execution
  },
  {
    from: ['analyst1', 'analyst2', 'analyst3'], // Fan-in
    to: 'summarizer',
    description: 'Aggregate results from all analysts',
    edgeType: 'direct',
  },
];
```

**Default behavior**:

- Single-to-multiple edges default to parallel (fan-out)
- Multiple-to-single edges should explicitly set `edgeType: 'direct'` for fan-in

## Common Patterns

### 1. Sequential Handoffs

```typescript
// Flight assistant can transfer to hotel assistant and vice versa
const edges = [
  { from: 'flight_assistant', to: 'hotel_assistant' },
  { from: 'hotel_assistant', to: 'flight_assistant' },
];
```

### 2. Supervisor Pattern (Handoff)

```typescript
// Supervisor decides which expert to route to
const edges = [
  {
    from: 'supervisor',
    to: ['expert1', 'expert2', 'expert3'],
    condition: (state) => decideExpert(state),
  },
  { from: 'expert1', to: 'supervisor' },
  { from: 'expert2', to: 'supervisor' },
  { from: 'expert3', to: 'supervisor' },
];
```

### 3. Map-Reduce Pattern (Parallel)

```typescript
// Distribute work and aggregate results
const edges = [
  {
    from: 'coordinator',
    to: ['worker1', 'worker2', 'worker3'],
    edgeType: 'direct', // Fan-out
  },
  {
    from: ['worker1', 'worker2', 'worker3'],
    to: 'aggregator',
    edgeType: 'direct', // Fan-in
  },
];
```

### 4. Hybrid Pattern

```typescript
// Mix of handoff and parallel
const edges = [
  // Classifier uses handoff to route
  {
    from: 'classifier',
    to: ['path_a', 'path_b'],
    condition: (state) => choosePath(state),
  },
  // Path A uses parallel processing
  {
    from: 'path_a',
    to: ['processor1', 'processor2'],
    edgeType: 'direct',
  },
  // Processors converge
  {
    from: ['processor1', 'processor2'],
    to: 'finalizer',
    edgeType: 'direct',
  },
];
```

## Important Notes

1. **Event Streaming**: When using parallel edges, you may see "Run ID not found in run map" errors in the console. These are harmless and can be ignored - they occur because LangGraph creates new run IDs for parallel executions that the event stream handler doesn't track.

2. **State Management**: All agents share the same state (messages). Parallel agents see the same state snapshot and their updates are merged.

3. **Tool Creation**:
   - Handoff edges create transfer tools (e.g., `transfer_to_agent_name`)
   - Parallel edges create direct graph connections (no tools)

4. **Performance**: Parallel execution can significantly speed up processing when agents perform independent work.

## Conversation-scoped handoffs (host opt-in)

An ordinary handoff changes execution within the current turn only. Set
`handoffScope: 'conversation'` on a handoff edge to request that a host continue
future user messages with the destination. This is trusted graph configuration,
not a model argument. It does **not** change a conversation or mutate an agent.

```typescript
const run = await Run.create({
  runId: responseMessageId,
  graphConfig: {
    type: 'multi-agent',
    agents,
    edges: [
      {
        from: 'router',
        to: 'specialist',
        edgeType: 'handoff',
        handoffScope: 'conversation',
      },
    ],
    entryAgentId: 'router',
    maxHandoffs: 8,
    compileOptions: { checkpointer },
  },
});
await run.processStream({ messages }, config);
const outcome = run.getHandoffOutcome();
if (outcome?.status === 'candidate') {
  // Host responsibility: authorize the target and commit with generation/revision
  // fencing. Only publish a changed client selection after the write succeeds.
  await commitAuthorizedConversationAgent(outcome);
}
```

`getHandoffOutcome()` is available after streaming, including normal cleanup:

- `candidate`: the last explicit conversation-scoped handoff in a successful,
  unambiguous top-level run. Includes `agentId` and an idempotent `transitionId`.
- `unchanged`: no conversation-scoped handoff occurred.
- `ambiguous`: there were conversation-scoped handoffs and parallel execution.
  There is no timing-based winner. Multiple inferred starting nodes and graphs
  containing direct fan-out conservatively prevent promotion, even if a dynamic
  handoff bypassed that fan-out.
- `incomplete`: interruption, cancellation, failure, budget exhaustion, a child
  scope, or incomplete legacy checkpoint provenance. Never auto-promote it.

Each outcome carries a logical-turn `executionId`, its initial `entryAgentId`,
and structured admitted `transitions`. These are execution facts, not proof of a
host-side database commit. A pause can contain already-admitted transitions but
still cannot produce a promotable candidate. Standard (non-multi-agent) runs
return `undefined`.

For A ⇒ B → C, where ⇒ is conversation-scoped and → is turn-scoped, the
candidate is B. For A ⇒ B ⇒ C it is C. Direct edges never promote their targets.
Isolated subagents do not inherit the parent's routing owner or handoff budget.

### Starting the next user turn

After the host commits B, construct a **fresh** run with `entryAgentId: 'B'`.
The explicit entry overrides topology-inferred roots, without changing the saved
edges or relying on agent array order. Only nodes reachable from that entry are
compiled. A grouped direct edge requiring an unreachable predecessor is rejected
rather than starting a workflow that cannot satisfy its join. Without an explicit
entry, existing inferred starting-node behavior is preserved.

Do not change entry or budget to resume a paused run. Rebuild its original graph,
with the same checkpointer and checkpoint namespace, then use `Run.resume()` or
`processStream(new Command({ resume: value }), config)`. The checkpoint records
routing state independently of displayed messages. Resume checks entry and budget
compatibility, preserves admitted transitions, and does not charge a replay twice.
A fresh `processStream({ messages }, ...)` resets the routing ledger, including
when reusing a checkpointed thread. Stop-hook continuations share its budget.

Never reconstruct routing state from transfer tool names, transcript order, or
agent-update events. Never accept client-supplied `handoffState` as authoritative.
Hosts remain responsible for rebuilding the original graph configuration and
serializing ownership of concurrent requests/resumes for a conversation.

### Bounding handoffs

`maxHandoffs` is an optional non-negative safe integer, shared by all members and
parallel branches of this graph within one logical turn. Zero forbids handoffs.
Absent preserves the existing recursion-only behavior. Set it independently of
`recursionLimit`: ordinary tool calls are not handoffs.

The budget is checked after the tool batch settles and before its routing Commands
schedule recipients. An oversized batch is rejected in full with the exported
`HandoffLimitError`; `getHaltReason()` reports `handoff_limit` and the outcome is
`incomplete`. Already executed ordinary sibling tools are not rolled back. False
conditional transfers consume no budget. Cycles are allowed, but a configured
finite budget bounds them. There is no conversation-lifetime cap.

### Checkpoint compatibility and rollout

Old hosts may ignore the additive outcome API and retain current behavior. They
must not enable automatic conversation switching until persistence, authorization,
and client reconciliation are implemented.

A new SDK can resume a legacy checkpoint without a handoff budget, but reports
`incomplete` / `legacy_checkpoint` rather than infer a candidate from partial
history. Enabling a budget on such a checkpoint fails closed because earlier
handoffs cannot be counted reliably. Finish that legacy turn before enabling the
feature. Rollback must not send feature-enabled paused runs to older SDK workers
that do not enforce their checkpointed budget.
