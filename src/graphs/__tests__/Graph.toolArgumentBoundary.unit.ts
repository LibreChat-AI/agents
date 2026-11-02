import { z } from 'zod';
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { writeFileSync } from 'node:fs';
import { tool } from '@langchain/core/tools';
import { MemorySaver } from '@langchain/langgraph';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import {
  TOOL_INPUT_ELISION_NOTE,
  projectToolMessagesForProvider,
} from '@/messages/prune';
import { GraphEvents, Providers } from '@/common';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

type Arguments = { command: string } | {
  _note: string;
  _originalChars: number;
  _inputPrefix: string;
};
type WireItem = {
  type?: string;
  call_id?: string;
  arguments?: string;
  output?: string;
};
type WireBody = { input: WireItem[]; stream?: boolean };
type CallSnapshot = { id?: string; name: string; arguments: string };
type Completion = CallSnapshot & { output?: string };
type Fixture = {
  create(hooks?: HookRegistry): Promise<Run<t.IState>>;
  config: t.RunStreamConfig;
  checkpointMessages(run: Run<t.IState>): Promise<BaseMessage[]>;
  requests: WireBody[];
  responses: string[];
  accepted: CallSnapshot[];
  completions: Completion[];
  executed: string[];
  checkpointDepths: number[];
};
type MessageSnapshot = {
  values: { messages?: BaseMessage[] };
  tasks?: Array<{ state?: MessageSnapshot | t.RunStreamConfig }>;
};

const toolName = 'record_command';
const freshId = 'call_fresh';
const originalArgs = { command: 'old-completed:' + 'x'.repeat(8000) };
const sentinel = { command: 'fresh-distinct-sentinel:' + 'y'.repeat(1600) };
const counter: t.TokenCounter = (message) => Math.ceil((
  JSON.stringify(message.content).length + (message.getType() === 'ai'
    ? JSON.stringify((message as AIMessage).tool_calls).length : 0)
) / 4) + 4;

function history(): BaseMessage[] {
  return [
    new HumanMessage({ id: 'old-user', content: 'Record the old value.' }),
    new AIMessage({ id: 'old-assistant', content: '', tool_calls: [
      { id: 'call_old', name: toolName, args: originalArgs },
    ] }),
    new ToolMessage({ id: 'old-result', tool_call_id: 'call_old', content: 'old value recorded' }),
    new HumanMessage({ id: 'new-user', content: 'Record a distinct new value.' }),
  ];
}

function canonical(messages: BaseMessage[]): string {
  return JSON.stringify(messages.map((message) => message.toDict()));
}

function callFrom(messages: BaseMessage[], id: string): string | undefined {
  for (const message of messages) {
    if (message.getType() !== 'ai') continue;
    const call = (message as AIMessage).tool_calls?.find((item) => item.id === id);
    if (call != null) return JSON.stringify(call.args);
  }
  return undefined;
}

function findCallMessages(snapshot: MessageSnapshot, id: string, depth = 0): { messages: BaseMessage[]; depth: number } | undefined {
  const messages = snapshot.values.messages;
  if (messages != null && callFrom(messages, id) != null) return { messages, depth };
  for (const task of snapshot.tasks ?? []) {
    const child = task.state;
    if (child == null || !('values' in child)) continue;
    const nested = findCallMessages(child, id, depth + 1);
    if (nested != null) return nested;
  }
  return undefined;
}

function archive(name: string, evidence: object): void {
  const directory = process.env.ARGUMENT_BOUNDARY_CAPTURE_DIR;
  if (directory != null) {
    writeFileSync(`${directory}/${name}.json`, JSON.stringify(evidence, null, 2));
  }
}

function fixture(args: Arguments): Fixture {
  const requests: WireBody[] = [];
  const responses: string[] = [];
  const accepted: CallSnapshot[] = [];
  const completions: Completion[] = [];
  const executed: string[] = [];
  const checkpointDepths: number[] = [];
  const saver = new MemorySaver();
  const transport: typeof fetch = async (url, request) => {
    assert.equal(String(url), 'http://synthetic.invalid/v1/responses');
    assert.equal(typeof request?.body, 'string');
    const body = JSON.parse(String(request?.body)) as WireBody;
    assert.equal(body.stream, false);
    requests.push(body);
    assert.ok((requests.length) <= 2);
    const output = requests.length === 1 ? [{
      type: 'function_call', id: 'fc_fresh', call_id: freshId,
      name: toolName, arguments: JSON.stringify(args), status: 'completed',
    }] : [{
      type: 'message', id: 'msg_final', status: 'completed', role: 'assistant',
      content: [{ type: 'output_text', text: 'Finished.', annotations: [] }],
    }];
    const response = {
      id: `resp_${requests.length}`, object: 'response', created_at: 1,
      status: 'completed', model: 'gpt-4o', output,
    };
    responses.push(JSON.stringify(response));
    return Response.json(response);
  };
  const recorder = tool(async ({ command }) => {
    executed.push(command);
    return 'fresh value recorded';
  }, {
    name: toolName, description: 'Record a synthetic string without executing it.',
    schema: z.object({ command: z.string() }),
  });
  const handlers: Record<string, t.EventHandler> = {
    [GraphEvents.ON_MODEL_RESPONSE]: { handle: (_event, data) => {
      for (const call of (data as t.ModelResponseEvent).toolCalls) {
        accepted.push({ id: call.id, name: call.name, arguments: JSON.stringify(call.args) });
      }
    } },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: { handle: (_event, data) => {
      const call = (data as { result?: t.ToolEndEvent }).result?.tool_call;
      if (call == null) return;
      completions.push({ id: call.id, name: call.name,
        arguments: typeof call.args === 'string' ? call.args : JSON.stringify(call.args),
        output: call.output });
    } },
  };
  const create = (hooks?: HookRegistry): Promise<Run<t.IState>> => Run.create<t.IState>({
    runId: 'argument-boundary', tokenCounter: counter,
    returnContent: true, skipCleanup: true, customHandlers: handlers, hooks,
    humanInTheLoop: { enabled: true },
    fadingTier: { v: 2, budgetTokens: 1000, masked: true, latched: true },
    graphConfig: {
      type: 'standard', compileOptions: { checkpointer: saver },
      agents: [{
        agentId: 'boundary-agent', provider: Providers.OPENAI,
        instructions: 'Record the requested synthetic string.',
        maxContextTokens: 16000, tools: [recorder],
        clientOptions: {
          model: 'gpt-4o', apiKey: 'offline-key', maxRetries: 0,
          disableStreaming: true, streaming: false, streamUsage: false,
          useResponsesApi: true, maxTokens: 32,
          configuration: { baseURL: 'http://synthetic.invalid/v1', fetch: transport },
        },
      }],
    },
  });
  const config: t.RunStreamConfig = {
    configurable: { thread_id: 'argument-boundary' }, version: 'v2',
  };
  const checkpointMessages = async (run: Run<t.IState>): Promise<BaseMessage[]> => {
    const workflow = run.graphRunnable as t.CompiledStateWorkflow & {
      getState(config: t.RunStreamConfig, options: { subgraphs: true }): Promise<MessageSnapshot>;
    };
    const snapshot = await workflow.getState(config, { subgraphs: true });
    const match = findCallMessages(snapshot, freshId);
    if (match == null) throw new Error('Pending call absent from all checkpoint levels');
    checkpointDepths.push(match.depth);
    return match.messages;
  };
  return { create, config, checkpointMessages, requests, responses, accepted, completions, executed, checkpointDepths };
}

function assertHistoricalProjection(request: WireBody): Arguments {
  const old = request.input.find((item) => item.type === 'function_call' && item.call_id === 'call_old');
  assert.notEqual(old?.arguments, undefined);
  const preview = JSON.parse(old!.arguments!) as Arguments;
  assert.ok('_note' in preview);
  assert.deepEqual(Object.keys(preview).sort(), ['_inputPrefix', '_note', '_originalChars']);
  assert.equal(preview._note, TOOL_INPUT_ELISION_NOTE);
  assert.equal(preview._originalChars, JSON.stringify(originalArgs).length);
  assert.equal(typeof preview._inputPrefix, 'string');
  assert.ok((old!.arguments!.length) < JSON.stringify(originalArgs).length);
  assert.equal(request.input.some((item) => item.type === 'function_call_output' && item.call_id === 'call_old'), true);
  return preview;
}

describe('tool argument boundaries after historical input elision', () => {
  it('executes fresh valid provider arguments unchanged while old history is projected', async () => {
    const capture = fixture(sentinel);
    const messages = history();
    const before = canonical(messages);
    const run = await capture.create();
    await run.processStream({ messages }, capture.config);
    assert.equal((capture.requests).length, 2);
    assertHistoricalProjection(capture.requests[0]);
    assert.deepEqual(capture.accepted, [{ id: freshId, name: toolName, arguments: JSON.stringify(sentinel) }]);
    assert.deepEqual(capture.executed, [sentinel.command]);
    assert.deepEqual(capture.completions, [{ id: freshId, name: toolName,
      arguments: JSON.stringify(sentinel), output: 'fresh value recorded' }]);
    const persisted = await capture.checkpointMessages(run);
    assert.equal(callFrom(persisted, freshId), JSON.stringify(sentinel));
    assert.equal(callFrom(persisted, 'call_old'), JSON.stringify(originalArgs));
    assert.equal(canonical(messages), before);
    archive('fresh-valid', { requests: capture.requests, responses: capture.responses,
      accepted: capture.accepted, executed: capture.executed, completions: capture.completions,
      canonicalBefore: before, canonicalAfter: canonical(persisted) });
  });

  it('rejects an explicit provider-returned historical preview without executing its prefix', async () => {
    const preview = { _note: TOOL_INPUT_ELISION_NOTE,
      _originalChars: JSON.stringify(originalArgs).length,
      _inputPrefix: JSON.stringify(originalArgs).slice(0, 120) };
    const capture = fixture(preview);
    const messages = history();
    const before = canonical(messages);
    const run = await capture.create();
    await run.processStream({ messages }, capture.config);
    assert.equal((capture.requests).length, 2);
    assertHistoricalProjection(capture.requests[0]);
    assert.deepEqual(capture.accepted, [{ id: freshId, name: toolName, arguments: JSON.stringify(preview) }]);
    assert.deepEqual(capture.executed, []);
    assert.equal((capture.completions).length, 1);
    assert.equal(capture.completions[0].arguments, JSON.stringify(preview));
    assert.equal(capture.completions[0].output?.includes('did not match expected schema'), true);
    const persisted = await capture.checkpointMessages(run);
    assert.equal(callFrom(persisted, freshId), JSON.stringify(preview));
    const result = persisted.find((message) => message.getType() === 'tool' && (message as ToolMessage).tool_call_id === freshId) as ToolMessage;
    assert.equal(result.status, 'error');
    assert.equal(capture.requests[1].input.find((item) => item.type === 'function_call_output' && item.call_id === freshId)?.output?.includes('did not match expected schema'), true);
    assert.equal(canonical(messages), before);
    archive('explicit-preview', { requests: capture.requests, responses: capture.responses,
      accepted: capture.accepted, executed: capture.executed, completions: capture.completions,
      canonicalBefore: before, canonicalAfter: canonical(persisted) });
  });

  it('restores pending valid arguments from a checkpoint after a separate historical projection', async () => {
    const capture = fixture(sentinel);
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'ask' })] });
    const messages = history();
    const before = canonical(messages);
    const original = await capture.create(hooks);
    await original.processStream({ messages }, capture.config);
    assert.notEqual(original.getInterrupt(), undefined);
    assert.equal((capture.requests).length, 1);
    assertHistoricalProjection(capture.requests[0]);
    assert.deepEqual(capture.executed, []);
    const pending = await capture.checkpointMessages(original);
    assert.ok(capture.checkpointDepths[0] > 0);
    const pendingBefore = canonical(pending);
    assert.equal(callFrom(pending, freshId), JSON.stringify(sentinel));
    const projected = projectToolMessagesForProvider(pending, 600, Providers.OPENAI);
    assert.notEqual(callFrom(projected, freshId), JSON.stringify(sentinel));
    assert.equal(canonical(pending), pendingBefore);
    const restored = await capture.create();
    await restored.resume([{ type: 'approve' }], capture.config);
    assert.equal((capture.requests).length, 2);
    assert.deepEqual(capture.accepted, [{ id: freshId, name: toolName, arguments: JSON.stringify(sentinel) }]);
    assert.deepEqual(capture.executed, [sentinel.command]);
    assert.deepEqual(capture.completions, [{ id: freshId, name: toolName,
      arguments: JSON.stringify(sentinel), output: 'fresh value recorded' }]);
    const persisted = await capture.checkpointMessages(restored);
    assert.equal(callFrom(persisted, freshId), JSON.stringify(sentinel));
    assert.equal(callFrom(persisted, 'call_old'), JSON.stringify(originalArgs));
    assert.equal(canonical(messages), before);
    archive('restored-pending', { requests: capture.requests, responses: capture.responses,
      accepted: capture.accepted, executed: capture.executed, completions: capture.completions,
      checkpointDepths: capture.checkpointDepths,
      checkpointBeforeProjection: pendingBefore, checkpointAfterProjection: canonical(pending),
      canonicalAfterResume: canonical(persisted) });
  });
});
