import assert from 'node:assert/strict';
import { fork } from 'node:child_process';
import { once } from 'node:events';
import { randomBytes, randomUUID } from 'node:crypto';
import { createRequire } from 'node:module';
import { setTimeout as delay } from 'node:timers/promises';
import {
  MemorySaver,
  HookRegistry,
  GraphEvents,
  HostModelAdmission,
  createHostModelService,
} from '@librechat/agents';
import { createRemoteSdkRun } from '@librechat/agents/runtime';
import {
  HumanMessage,
  AIMessageChunk,
} from '@librechat/agents/langchain/messages';
import { FakeChatModel } from '@langchain/core/utils/testing';
import { ChatGenerationChunk } from '@langchain/core/outputs';

const require = createRequire(import.meta.url);
assert.equal(
  typeof require('@librechat/agents/runtime').createRemoteSdkServer,
  'function'
);
assert.throws(() => require.resolve('@librechat/api'), {
  code: 'MODULE_NOT_FOUND',
});
const secret = randomBytes(32).toString('hex');
const worker = fork(new URL('./worker.cjs', import.meta.url), {
  env: { ...process.env, RUNTIME_TEST_SECRET: secret },
  stdio: ['ignore', 'inherit', 'inherit', 'ipc'],
});
const identity = () => ({
  runtimeId: 'package-test',
  agentId: 'agent',
  userId: 'user',
  conversationId: randomUUID(),
  runId: randomUUID(),
});
const options = (id, signal) => ({
  version: 'v2',
  signal,
  configurable: { thread_id: id.conversationId },
});
const input = {
  messages: [new HumanMessage('Inspect and review the fixture')],
};
const config = (id, instructions) => ({
  runId: id.runId,
  returnContent: true,
  tokenCounter: () => 1,
  graphConfig: {
    type: 'standard',
    agents: [
      {
        agentId: 'agent',
        provider: 'openAI',
        instructions,
        tools: [],
        clientOptions: { model: 'fixture', streaming: true },
      },
    ],
  },
});
try {
  const [{ port }] = await once(worker, 'message', {
    signal: AbortSignal.timeout(15000),
  });
  const remote = { url: `ws://127.0.0.1:${port}/sdk`, secret };
  const id = identity();
  const hooks = new HookRegistry();
  hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'ask' })] });
  const setup = config(id);
  setup.hooks = hooks;
  setup.humanInTheLoop = { enabled: true };
  setup.graphConfig.compileOptions = { checkpointer: new MemorySaver() };
  let deltas = 0;
  setup.customHandlers = {
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: () => {
        deltas++;
      },
    },
  };
  const run = await createRemoteSdkRun({ config: setup, identity: id, remote });
  await run.processStream(input, options(id));
  assert.equal(run.getInterrupt()?.payload.type, 'tool_approval');
  const resumed = await createRemoteSdkRun({
    config: setup,
    identity: id,
    remote,
    resumeState: {
      content: run.Graph.getContentParts() ?? [],
      steps: run.Graph.createRunStepResumeState().steps.map(({ step }) => step),
    },
  });
  const result = await resumed.resume(
    { 'example-inspection': { type: 'approve' } },
    options(id)
  );
  assert.equal(resumed.getInterrupt(), undefined);
  assert.match(JSON.stringify(result), /Custom graph complete/);
  assert.ok(deltas > 0);

  const hostId = identity();
  const model = new FakeChatModel({});
  let calls = 0;
  model.bindTools = () => model;
  model._generate = async () => {
    calls++;
    return {
      generations: [
        {
          text: '',
          message: new AIMessageChunk({
            content: 'Host-owned model reply',
            usage_metadata: {
              input_tokens: 1,
              output_tokens: 1,
              total_tokens: 2,
            },
          }),
        },
      ],
    };
  };
  model._streamResponseChunks = async function* () {
    const { generations } = await this._generate();
    yield new ChatGenerationChunk({
      text: generations[0].message.content,
      message: generations[0].message,
    });
  };
  const admission = new HostModelAdmission({
    maxConcurrentCalls: 2,
    maxRequestBytes: 1048576,
  });
  const hostRun = await createRemoteSdkRun({
    config: config(hostId),
    identity: hostId,
    remote,
    hostServices: {
      modelProfile: { provider: 'openAI', name: 'fixture', streaming: true },
      model: createHostModelService({
        identity: hostId,
        admission,
        model,
        descriptor: {
          version: 1,
          reference: 'primary',
          provider: 'openAI',
          model: 'fixture',
          streaming: true,
          tools: [],
        },
        limits: {
          timeoutMs: 10000,
          maxRequestBytes: 1048576,
          maxResponseBytes: 1048576,
        },
      }),
    },
  });
  const hostResult = await hostRun.processStream(input, options(hostId));
  assert.match(JSON.stringify(hostResult), /Host-owned model reply/);
  assert.equal(calls, 2);
  assert.equal(admission.snapshot().calls, 0);

  const cancelId = identity();
  const controller = new AbortController();
  const cancelled = await createRemoteSdkRun({
    config: config(cancelId, 'delay'),
    identity: cancelId,
    remote,
  });
  const pending = cancelled.processStream(
    input,
    options(cancelId, controller.signal)
  );
  const rejected = assert.rejects(pending);
  await delay(150);
  controller.abort();
  await rejected;

  const foreign = { ...identity(), agentId: 'foreign' };
  const denied = await createRemoteSdkRun({
    config: config(foreign),
    identity: foreign,
    remote,
  });
  await assert.rejects(denied.processStream(input, options(foreign)));
  console.log(
    'Packed runtime passed: CJS worker + ESM host, custom graph, streamed events, approval/reconstruction, host model, cancellation, identity denial.'
  );
} finally {
  if (worker.exitCode == null && worker.signalCode == null) {
    const exited = once(worker, 'exit');
    const deadline = setTimeout(() => worker.kill('SIGKILL'), 5000);
    worker.kill('SIGTERM');
    const [code, signal] = await exited.finally(() => clearTimeout(deadline));
    assert.equal(signal, null, 'Worker did not shut down gracefully');
    assert.equal(code, 0);
  }
}
