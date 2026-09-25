import { tmpdir } from 'os';
import { dirname, join } from 'path';
import { MemorySaver } from '@langchain/langgraph';
import { appendFile, mkdtemp, readFile, rm, writeFile } from 'fs/promises';
import {
  AIMessage,
  HumanMessage,
  RemoveMessage,
} from '@langchain/core/messages';
import type { Checkpoint, CheckpointMetadata } from '@langchain/langgraph';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { JsonlSessionStore, createAgentSession } from '@/session';
import { toJsonValue } from '@/session/messageSerialization';
import * as providers from '@/llm/providers';
import { GraphEvents } from '@/common';
import { Run } from '@/run';

type MockRun = {
  processStream: jest.MockedFunction<Run<t.IState>['processStream']>;
  resume: jest.MockedFunction<Run<t.IState>['resume']>;
  getRunMessages: jest.MockedFunction<Run<t.IState>['getRunMessages']>;
  getCalibrationRatio: jest.MockedFunction<
    Run<t.IState>['getCalibrationRatio']
  >;
  getFadingTier: jest.MockedFunction<Run<t.IState>['getFadingTier']>;
  getFadingTiers: jest.MockedFunction<Run<t.IState>['getFadingTiers']>;
  didResetFadingTier: jest.MockedFunction<Run<t.IState>['didResetFadingTier']>;
  getFadingTierResetAgentIds: jest.MockedFunction<
    Run<t.IState>['getFadingTierResetAgentIds']
  >;
  getInterrupt: jest.MockedFunction<Run<t.IState>['getInterrupt']>;
  getHaltReason: jest.MockedFunction<Run<t.IState>['getHaltReason']>;
  getChildCheckpointThreadIds: jest.MockedFunction<
    Run<t.IState>['getChildCheckpointThreadIds']
  >;
};

function createMockRun(outputText = 'ok'): MockRun {
  return {
    processStream: jest
      .fn<
        ReturnType<Run<t.IState>['processStream']>,
        Parameters<Run<t.IState>['processStream']>
      >()
      .mockResolvedValue([{ type: 'text', text: outputText }]),
    resume: jest
      .fn<
        ReturnType<Run<t.IState>['resume']>,
        Parameters<Run<t.IState>['resume']>
      >()
      .mockResolvedValue([{ type: 'text', text: outputText }]),
    getRunMessages: jest.fn(() => [new AIMessage(outputText)]),
    getCalibrationRatio: jest.fn(() => 1),
    getFadingTier: jest.fn(() => undefined),
    getFadingTiers: jest.fn(() => ({})),
    didResetFadingTier: jest.fn(() => false),
    getFadingTierResetAgentIds: jest.fn(() => []),
    getInterrupt: jest.fn(() => undefined),
    getHaltReason: jest.fn(() => undefined),
    getChildCheckpointThreadIds: jest.fn(() => []),
  };
}

function mockRunCreate(mockRun: MockRun): t.RunConfig[] {
  const capturedConfigs: t.RunConfig[] = [];
  jest.spyOn(Run, 'create').mockImplementation((async <
    T extends t.BaseGraphState,
  >(
    config: t.RunConfig
  ): Promise<Run<T>> => {
    capturedConfigs.push(config);
    return mockRun as unknown as Run<T>;
  }) as never);
  return capturedConfigs;
}

function getProcessedState(mockRun: MockRun): t.IState {
  expect(mockRun.processStream).toHaveBeenCalled();
  const input = mockRun.processStream.mock.calls[0][0];
  if (!('messages' in input)) {
    throw new Error('Expected processStream to receive message state');
  }
  return input;
}

function getProcessedMessages(mockRun: MockRun): BaseMessage[] {
  return getProcessedState(mockRun).messages;
}

async function putCheckpoint(params: {
  checkpointer: MemorySaver;
  threadId: string;
  id: string;
  checkpointNs?: string;
}): Promise<void> {
  const checkpoint: Checkpoint = {
    v: 4,
    id: params.id,
    ts: new Date().toISOString(),
    channel_values: {},
    channel_versions: {},
    versions_seen: {},
  };
  const metadata: CheckpointMetadata = {
    source: 'loop',
    step: 0,
    parents: {},
  };
  await params.checkpointer.put(
    {
      configurable: {
        thread_id: params.threadId,
        checkpoint_ns: params.checkpointNs ?? '',
      },
    },
    checkpoint,
    metadata
  );
}

function mockSummarizer(response: string): void {
  jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
    class {
      constructor() {
        return {
          invoke: jest.fn().mockResolvedValue({ content: response }),
        };
      }
    } as never
  );
}

describe('JsonlSessionStore', () => {
  let dir: string;

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'lc-agent-session-'));
  });

  afterEach(async () => {
    jest.restoreAllMocks();
    await rm(dir, { recursive: true, force: true });
  });

  it('uses cached session history across runs and falls back after mutable store exposure', async () => {
    const mockRun = createMockRun('answer');
    mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      sessionPath: join(dir, 'cached.jsonl'),
      runId: 'template-run',
      checkpointing: false,
      graphConfig: {
        type: 'standard',
        llmConfig: { provider: 'openAI' as never, model: 'test-model' },
        instructions: 'test',
      },
    });
    const getPath = jest.spyOn(JsonlSessionStore.prototype, 'getPath');
    await session.run('first');
    const firstMessages = getProcessedMessages(mockRun);
    firstMessages[0].content = 'run-local change';
    mockRun.processStream.mockClear();
    await session.run('second');
    expect(
      getProcessedMessages(mockRun).map((message) => message.content)
    ).toEqual(['first', 'answer', 'second']);
    expect(getPath).not.toHaveBeenCalled();
    const store = session.getSessionStore()!;
    const entry = store.getPath()[0];
    if (entry.type === 'message') {
      entry.data.message.content = 'external edit';
    }
    mockRun.processStream.mockClear();
    await session.run('third');
    expect(getProcessedMessages(mockRun)[0].content).toBe('external edit');
    expect(getPath).toHaveBeenCalledTimes(2);
  });

  it('keeps the unexposed cache coherent through compaction, HITL resume, branching, and reopening', async () => {
    mockSummarizer('summary of prior work');
    const mockRun = createMockRun('answer');
    const configs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      sessionPath: join(dir, 'cache-transitions.jsonl'),
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: { provider: 'openAI' as never, model: 'test-model' },
        instructions: 'test',
      },
    });
    await session.run('first');
    await session.compact({ retainRecentTurns: 0 });
    const getPath = jest.spyOn(JsonlSessionStore.prototype, 'getPath');
    await session.resumeInterrupt([]);
    const resumeConfig = configs.at(-1)!.graphConfig;
    expect(
      'initialSummary' in resumeConfig && resumeConfig.initialSummary
    ).toEqual({
      text: 'summary of prior work',
      tokenCount: expect.any(Number),
    });
    mockRun.processStream.mockClear();
    await session.run('after resume');
    expect(
      getProcessedMessages(mockRun).map((message) => message.content)
    ).toEqual(['answer', 'after resume']);
    expect(getPath).not.toHaveBeenCalled();

    const diskSnapshot = await JsonlSessionStore.openPath(session.sessionPath!);
    const summary = diskSnapshot
      .getPath()
      .find((entry) => entry.type === 'summary')!;
    await session.branch(summary.id);
    mockRun.processStream.mockClear();
    await session.run('alternate');
    expect(
      getProcessedMessages(mockRun).map((message) => message.content)
    ).toEqual(['alternate']);
    await session.resumeSession(session.sessionPath);
    mockRun.processStream.mockClear();
    await session.run('reopened');
    expect(
      getProcessedMessages(mockRun).map((message) => message.content)
    ).toEqual(['alternate', 'answer', 'reopened']);
  });

  it('opens a file whose session_state line carries null data', async () => {
    const path = join(dir, 'null-state.jsonl');
    const store = await JsonlSessionStore.create({
      path,
      cwd: dir,
      sessionId: 'session-null-state',
    });
    await store.appendMessage(new HumanMessage('hello'));
    await appendFile(
      path,
      `${JSON.stringify({
        type: 'session_state',
        id: 'state-null',
        parentId: null,
        timestamp: new Date().toISOString(),
        data: null,
      })}\n`,
      'utf8'
    );

    const reopened = await JsonlSessionStore.open(path);

    expect(reopened.hasFadingState()).toBe(false);
    expect(reopened.getFadingStates('session-null-state')).toEqual([]);
  });

  it('stores messages as an append-only tree and restores the active path', async () => {
    const path = join(dir, 'session.jsonl');
    const store = await JsonlSessionStore.create({
      path,
      cwd: dir,
      sessionId: 'session-a',
    });

    const user = await store.appendMessage(new HumanMessage('hello'));
    const assistant = await store.appendMessage(new AIMessage('hi'));

    const reopened = await JsonlSessionStore.open(path);

    expect(reopened.header.id).toBe('session-a');
    expect(reopened.getLeafEntry()?.id).toBe(assistant.id);
    expect(reopened.getPath().map((entry) => entry.id)).toEqual([
      user.id,
      assistant.id,
    ]);
    expect(reopened.getMessages().map((message) => message.content)).toEqual([
      'hello',
      'hi',
    ]);
  });

  it('round-trips remove messages in persisted sessions', async () => {
    const path = join(dir, 'remove.jsonl');
    const store = await JsonlSessionStore.create({ path, cwd: dir });

    await store.appendMessage(
      new HumanMessage({ id: 'message-a', content: 'a' })
    );
    await store.appendMessage(new RemoveMessage({ id: 'message-a' }));

    const reopened = await JsonlSessionStore.open(path);
    const messages = reopened.getMessages();

    expect(messages.map((message) => message._getType())).toEqual([
      'human',
      'remove',
    ]);
    expect((messages[1] as RemoveMessage).id).toBe('message-a');
  });

  it('fails when creating a session file that already exists', async () => {
    const path = join(dir, 'existing.jsonl');
    await JsonlSessionStore.create({
      path,
      cwd: dir,
      sessionId: 'session-a',
    });

    await expect(
      JsonlSessionStore.create({
        path,
        cwd: dir,
        sessionId: 'session-b',
      })
    ).rejects.toMatchObject({ code: 'EEXIST' });

    const raw = await readFile(path, 'utf8');
    expect(raw.match(/"type":"session"/g)).toHaveLength(1);
  });

  it('keeps default session roots distinct for similar cwd strings', async () => {
    const cwdA = join(dir, 'foo/bar');
    const cwdB = join(dir, 'foo-bar');
    const storeA = await JsonlSessionStore.create({
      cwd: cwdA,
      sessionId: 'cwd-a',
    });
    const storeB = await JsonlSessionStore.create({
      cwd: cwdB,
      sessionId: 'cwd-b',
    });

    try {
      const [itemsA, itemsB] = await Promise.all([
        JsonlSessionStore.list(cwdA),
        JsonlSessionStore.list(cwdB),
      ]);

      expect(dirname(storeA.path)).not.toBe(dirname(storeB.path));
      expect(itemsA.map((item) => item.id)).toContain('cwd-a');
      expect(itemsA.map((item) => item.id)).not.toContain('cwd-b');
      expect(itemsB.map((item) => item.id)).toContain('cwd-b');
      expect(itemsB.map((item) => item.id)).not.toContain('cwd-a');
    } finally {
      await Promise.all([
        rm(storeA.path, { force: true }),
        rm(storeB.path, { force: true }),
      ]);
    }
  });

  it('branches in place without deleting abandoned children', async () => {
    const store = await JsonlSessionStore.create({
      path: join(dir, 'branch.jsonl'),
      cwd: dir,
    });
    const first = await store.appendMessage(new HumanMessage('one'));
    const abandoned = await store.appendMessage(new AIMessage('abandoned'));

    await store.branch(first.id);
    const alternate = await store.appendMessage(new AIMessage('alternate'));

    expect(
      store
        .getChildren(first.id)
        .filter((entry) => entry.type === 'message')
        .map((entry) => entry.id)
        .sort()
    ).toEqual([abandoned.id, alternate.id].sort());
    expect(store.getPath().map((entry) => entry.id)).toEqual([
      first.id,
      alternate.id,
    ]);
  });

  it('clones and forks active paths into new session files', async () => {
    const store = await JsonlSessionStore.create({
      path: join(dir, 'source.jsonl'),
      cwd: dir,
    });
    const first = await store.appendMessage(new HumanMessage('first'));
    const second = await store.appendMessage(new AIMessage('second'));

    const clone = await store.clone({ cwd: dir });
    const fork = await store.fork(second.id, { cwd: dir, position: 'before' });

    expect(clone.header.parentSession).toBe(store.path);
    expect(clone.getPath().map((entry) => entry.id)).toEqual([
      first.id,
      second.id,
    ]);
    expect(fork.getPath().map((entry) => entry.id)).toEqual([first.id]);
  });

  it('tracks labels and compaction entries', async () => {
    const store = await JsonlSessionStore.create({
      path: join(dir, 'labels.jsonl'),
      cwd: dir,
    });
    const message = await store.appendMessage(new HumanMessage('hello'));

    await store.setLabel(message.id, 'checkpoint');
    const summary = await store.appendEntryForCompaction({
      text: 'summary',
      retainedEntryIds: [message.id],
      summarizedEntryIds: [],
    });
    const compaction = await store.appendCompactionEntry({
      summaryEntryId: summary.id,
      retainedEntryIds: [message.id],
      summarizedEntryIds: [],
    });

    expect(store.getLabel(message.id)).toBe('checkpoint');
    expect(summary.data.text).toBe('summary');
    expect(compaction.data.summaryEntryId).toBe(summary.id);
  });

  it('records LangGraph checkpoint references without moving the active leaf', async () => {
    const store = await JsonlSessionStore.create({
      path: join(dir, 'checkpoints.jsonl'),
      cwd: dir,
    });
    const message = await store.appendMessage(new HumanMessage('hello'));

    const checkpoint = await store.appendCheckpoint({
      source: 'run',
      threadId: store.header.id,
      runId: 'run_checkpoint',
      checkpointId: 'checkpoint_1',
      checkpointNs: '',
    });

    expect(checkpoint.data.provider).toBe('langgraph');
    expect(store.getLeafEntry()?.id).toBe(message.id);
    expect(store.getLatestCheckpoint(store.header.id)?.id).toBe(checkpoint.id);
  });

  it('treats reset checkpoints as latest checkpoint barriers', async () => {
    const store = await JsonlSessionStore.create({
      path: join(dir, 'checkpoint-reset.jsonl'),
      cwd: dir,
    });
    await store.appendCheckpoint({
      source: 'run',
      threadId: store.header.id,
      runId: 'run_before_reset',
      checkpointId: 'checkpoint_before_reset',
    });
    await store.appendCheckpoint({
      source: 'reset',
      threadId: store.header.id,
      reason: 'branch',
    });

    expect(store.getLatestCheckpoint(store.header.id)).toBeUndefined();

    const checkpoint = await store.appendCheckpoint({
      source: 'run',
      threadId: store.header.id,
      runId: 'run_after_reset',
      checkpointId: 'checkpoint_after_reset',
    });

    expect(store.getLatestCheckpoint(store.header.id)?.id).toBe(checkpoint.id);
  });

  it('preserves Error details in JSONL payloads', () => {
    const error = new Error('resume failed');
    const payload = toJsonValue(error);

    expect(payload).toMatchObject({
      name: 'Error',
      message: 'resume failed',
    });
    expect(
      typeof payload === 'object' &&
        payload != null &&
        !Array.isArray(payload) &&
        typeof payload.stack === 'string'
    ).toBe(true);
  });

  it('replaces circular object references in JSONL payloads', () => {
    interface CircularPayload {
      label: string;
      self?: CircularPayload;
      child?: { parent?: CircularPayload };
    }
    const circular: CircularPayload = { label: 'root' };
    circular.self = circular;
    circular.child = { parent: circular };

    expect(toJsonValue(circular)).toMatchObject({
      label: 'root',
      self: '[Circular]',
      child: { parent: '[Circular]' },
    });
  });

  it('replaces circular Error causes in JSONL payloads', () => {
    const error = new Error('request failed');
    Object.defineProperty(error, 'cause', {
      value: error,
      configurable: true,
    });

    expect(toJsonValue(error)).toMatchObject({
      name: 'Error',
      message: 'request failed',
      cause: '[Circular]',
    });
  });

  it('creates high-level sessions with a JSONL store by default', async () => {
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    expect(session.getSessionStore()?.header.cwd).toBe(dir);
    expect(session.sessionPath).toContain('.jsonl');
  });

  it('surfaces invalid explicit session files instead of replacing them', async () => {
    const sessionPath = join(dir, 'invalid.jsonl');
    await writeFile(sessionPath, 'not jsonl\n', 'utf8');

    await expect(
      createAgentSession({
        cwd: dir,
        sessionPath,
        runId: 'template-run',
        graphConfig: {
          type: 'standard',
          llmConfig: {
            provider: 'openAI' as never,
            model: 'test-model',
          },
          instructions: 'test',
        },
      })
    ).rejects.toThrow('Invalid session file');

    expect(await readFile(sessionPath, 'utf8')).toBe('not jsonl\n');
  });

  it('creates an explicit session path without fuzzy matching existing sessions', async () => {
    const existing = await JsonlSessionStore.create({
      path: join(dir, 'matching-existing.jsonl'),
      cwd: dir,
      sessionId: 'explicit-target-existing',
    });
    await existing.appendMessage(new HumanMessage('existing history'));
    const sessionPath = join(dir, 'explicit-target.jsonl');

    const session = await createAgentSession({
      cwd: dir,
      sessionPath,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    expect(session.sessionPath).toBe(sessionPath);
    expect(session.getSessionStore()?.header.id).not.toBe(existing.header.id);
    expect(session.getSessionStore()?.getMessages()).toEqual([]);
  });

  it('preserves non-message state while applying session history', async () => {
    const mockRun = createMockRun('stateful output');
    mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await session.getSessionStore()?.appendMessage(new HumanMessage('history'));
    const input: t.IState & { selectedAgent: string } = {
      messages: [new HumanMessage('fresh')],
      selectedAgent: 'subagent-a',
    };

    await session.run(input);

    const processedState = getProcessedState(mockRun) as t.IState & {
      selectedAgent?: string;
    };
    expect(processedState.selectedAgent).toBe('subagent-a');
    expect(processedState.messages.map((message) => message.content)).toEqual([
      'history',
      'fresh',
    ]);
  });

  it('restores persisted summary token counts into run config', async () => {
    const mockRun = createMockRun('with summary');
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await session.getSessionStore()?.appendEntryForCompaction({
      text: 'stored summary',
      tokenCount: 123,
      retainedEntryIds: [],
      summarizedEntryIds: [],
    });

    await session.run('fresh');

    const { graphConfig } = capturedConfigs[0];
    const initialSummary =
      'initialSummary' in graphConfig ? graphConfig.initialSummary : undefined;
    expect(initialSummary).toEqual({
      text: 'stored summary',
      tokenCount: 123,
    });
  });

  it('preserves custom handlers outside the session event adapter set', async () => {
    const mockRun = createMockRun('handled');
    const capturedConfigs = mockRunCreate(mockRun);
    const agentLogHandler: t.EventHandler = { handle: jest.fn() };
    const messageDeltaHandler: t.EventHandler = { handle: jest.fn() };
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
      customHandlers: {
        [GraphEvents.ON_AGENT_LOG]: agentLogHandler,
        [GraphEvents.ON_MESSAGE_DELTA]: messageDeltaHandler,
      },
    });

    await session.run('start');

    expect(capturedConfigs[0].customHandlers?.[GraphEvents.ON_AGENT_LOG]).toBe(
      agentLogHandler
    );
    expect(
      capturedConfigs[0].customHandlers?.[GraphEvents.ON_MESSAGE_DELTA]
    ).not.toBe(messageDeltaHandler);
  });

  it('shares a session-level LangGraph checkpointer for HITL resume', async () => {
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      humanInTheLoop: { enabled: true },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    expect(session.getCheckpointer()).toBeInstanceOf(MemorySaver);
  });

  it('keeps stores and checkpointing optional for high-level sessions', async () => {
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      ephemeral: true,
      checkpointing: false,
      humanInTheLoop: { enabled: true },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    expect(session.getSessionStore()).toBeUndefined();
    expect(session.getCheckpointer()).toBeUndefined();
  });

  it('removes graph checkpointers when checkpointing is disabled', async () => {
    const graphCheckpointer = new MemorySaver();
    const mockRun = createMockRun('disabled');
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: false,
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
        compileOptions: { checkpointer: graphCheckpointer },
      },
    });

    await session.run('start');

    expect(session.getCheckpointer()).toBeUndefined();
    expect(
      capturedConfigs[0].graphConfig.compileOptions?.checkpointer
    ).toBeUndefined();
  });

  it('reuses the session-level checkpointer across HITL resume', async () => {
    const mockRun = createMockRun('resumed');
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      humanInTheLoop: { enabled: true },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    await session.run('start');
    await session.resumeInterrupt([]);

    const checkpointer =
      capturedConfigs[0].graphConfig.compileOptions?.checkpointer;
    expect(checkpointer).toBeInstanceOf(MemorySaver);
    expect(session.getCheckpointer()).toBe(checkpointer);
    expect(capturedConfigs[1].graphConfig.compileOptions?.checkpointer).toBe(
      checkpointer
    );
  });

  it('preserves a caller-supplied session checkpointer', async () => {
    const checkpointer = new MemorySaver();
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    expect(session.getCheckpointer()).toBe(checkpointer);
  });

  it('injects the session checkpointer and replays JSONL history before checkpoints exist', async () => {
    const checkpointer = new MemorySaver();
    const mockRun = createMockRun('first output');
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await session.getSessionStore()?.appendMessage(new HumanMessage('history'));

    await session.run('next');

    expect(capturedConfigs[0].graphConfig.compileOptions?.checkpointer).toBe(
      checkpointer
    );
    expect(
      getProcessedMessages(mockRun).map((message) => message.content)
    ).toEqual(['history', 'next']);
  });

  it('does not replay session history when overriding thread id without checkpoint state', async () => {
    const checkpointer = new MemorySaver();
    const mockRun = createMockRun('override output');
    mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await session.getSessionStore()?.appendMessage(new HumanMessage('history'));

    await session.run('fresh turn', { threadId: 'thread_override' });

    expect(
      getProcessedMessages(mockRun).map((message) => message.content)
    ).toEqual(['fresh turn']);
    expect(
      session
        .getSessionStore()
        ?.getPath()
        .filter((entry) => entry.type === 'message')
        .map((entry) => entry.data.message.content)
    ).toEqual(['history']);
  });

  it('does not persist resumed override thread messages into the session path', async () => {
    const checkpointer = new MemorySaver();
    const mockRun = createMockRun('override resumed');
    mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await session.getSessionStore()?.appendMessage(new HumanMessage('history'));

    await session.resumeInterrupt([], { threadId: 'thread_override' });

    expect(mockRun.resume).toHaveBeenCalledWith(
      [],
      expect.objectContaining({
        configurable: expect.objectContaining({
          thread_id: 'thread_override',
        }),
      })
    );
    expect(
      session
        .getSessionStore()
        ?.getPath()
        .filter((entry) => entry.type === 'message')
        .map((entry) => entry.data.message.content)
    ).toEqual(['history']);
  });

  it('resumes isolated Run instances from the stored interrupted checkpoint', async () => {
    const checkpointer = new MemorySaver();
    const mockRun = createMockRun('resumed');
    mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await putCheckpoint({
      checkpointer,
      threadId: session.threadId,
      id: 'checkpoint_interrupted',
    });
    await session.getSessionStore()?.appendCheckpoint({
      source: 'run',
      threadId: session.threadId,
      runId: 'run_interrupted',
      checkpointId: 'checkpoint_interrupted',
      checkpointNs: '',
    });

    await session.resumeInterrupt([{ type: 'approve' }]);

    expect(mockRun.resume).toHaveBeenCalledWith(
      [{ type: 'approve' }],
      expect.objectContaining({
        configurable: expect.objectContaining({
          thread_id: session.threadId,
          checkpoint_id: 'checkpoint_interrupted',
          checkpoint_ns: '',
        }),
      })
    );
  });

  it('records a new interrupt checkpoint after resuming from an older checkpoint', async () => {
    const checkpointer = new MemorySaver();
    const mockRun = createMockRun('interrupted again');
    mockRun.getInterrupt.mockReturnValue({
      interruptId: 'interrupt_again',
      threadId: 'thread_new_interrupt',
      checkpointId: 'checkpoint_new_interrupt',
      checkpointNs: '',
      payload: {
        type: 'tool_approval',
        action_requests: [],
        review_configs: [],
      },
    });
    mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await putCheckpoint({
      checkpointer,
      threadId: session.threadId,
      id: 'checkpoint_resume_source',
    });
    await putCheckpoint({
      checkpointer,
      threadId: session.threadId,
      id: 'checkpoint_new_interrupt',
    });
    await session.getSessionStore()?.appendCheckpoint({
      source: 'run',
      threadId: session.threadId,
      runId: 'run_interrupted',
      checkpointId: 'checkpoint_resume_source',
      checkpointNs: '',
    });

    await session.resumeInterrupt([{ type: 'approve' }]);

    expect(
      session.getSessionStore()?.getLatestCheckpoint(session.threadId)?.data
    ).toMatchObject({
      source: 'resume',
      checkpointId: 'checkpoint_new_interrupt',
    });
  });

  it('resumes isolated Run instances from the requested checkpoint namespace', async () => {
    const checkpointer = new MemorySaver();
    const mockRun = createMockRun('resumed requested namespace');
    mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await session.getSessionStore()?.appendCheckpoint({
      source: 'run',
      threadId: session.threadId,
      runId: 'run_default',
      checkpointId: 'checkpoint_default_latest',
      checkpointNs: '',
    });
    await session.getSessionStore()?.appendCheckpoint({
      source: 'run',
      threadId: session.threadId,
      runId: 'run_requested',
      checkpointId: 'checkpoint_requested_latest',
      checkpointNs: 'requested',
    });
    await session.getSessionStore()?.appendCheckpoint({
      source: 'run',
      threadId: session.threadId,
      runId: 'run_default_again',
      checkpointId: 'checkpoint_default_newer',
      checkpointNs: '',
    });

    await session.resumeInterrupt([{ type: 'approve' }], {
      config: {
        configurable: {
          checkpoint_ns: 'requested',
        },
      },
    });

    expect(mockRun.resume).toHaveBeenCalledWith(
      [{ type: 'approve' }],
      expect.objectContaining({
        configurable: expect.objectContaining({
          thread_id: session.threadId,
          checkpoint_id: 'checkpoint_requested_latest',
          checkpoint_ns: 'requested',
        }),
      })
    );
  });

  it('resumes isolated Run instances from the explicitly requested default namespace', async () => {
    const checkpointer = new MemorySaver();
    const mockRun = createMockRun('resumed default namespace');
    mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await session.getSessionStore()?.appendCheckpoint({
      source: 'run',
      threadId: session.threadId,
      runId: 'run_default',
      checkpointId: 'checkpoint_default_latest',
      checkpointNs: '',
    });
    await session.getSessionStore()?.appendCheckpoint({
      source: 'run',
      threadId: session.threadId,
      runId: 'run_requested_newer',
      checkpointId: 'checkpoint_requested_newer',
      checkpointNs: 'requested',
    });

    await session.resumeInterrupt([{ type: 'approve' }], {
      config: {
        configurable: {
          checkpoint_ns: '',
        },
      },
    });

    expect(mockRun.resume).toHaveBeenCalledWith(
      [{ type: 'approve' }],
      expect.objectContaining({
        configurable: expect.objectContaining({
          thread_id: session.threadId,
          checkpoint_id: 'checkpoint_default_latest',
          checkpoint_ns: '',
        }),
      })
    );
  });

  it('uses only new input when LangGraph checkpoint state already exists', async () => {
    const checkpointer = new MemorySaver();
    const mockRun = createMockRun('checkpointed output');
    mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await session.getSessionStore()?.appendMessage(new HumanMessage('history'));
    await putCheckpoint({
      checkpointer,
      threadId: session.threadId,
      id: 'checkpoint_existing',
    });

    await session.run('fresh turn', { runId: 'run_checkpointed' });

    const checkpoints = session
      .getSessionStore()
      ?.getCheckpoints(session.threadId);
    expect(
      getProcessedMessages(mockRun).map((message) => message.content)
    ).toEqual(['fresh turn']);
    expect(checkpoints?.at(-1)?.data).toMatchObject({
      source: 'run',
      runId: 'run_checkpointed',
      checkpointId: 'checkpoint_existing',
    });
  });

  it('replays JSONL history when the requested checkpoint namespace has no state', async () => {
    const checkpointer = new MemorySaver();
    const mockRun = createMockRun('namespace output');
    mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await session.getSessionStore()?.appendMessage(new HumanMessage('history'));
    await putCheckpoint({
      checkpointer,
      threadId: session.threadId,
      id: 'checkpoint_other_namespace',
      checkpointNs: 'other',
    });

    await session.run('fresh turn', {
      config: { configurable: { checkpoint_ns: 'requested' } },
    });

    expect(
      getProcessedMessages(mockRun).map((message) => message.content)
    ).toEqual(['history', 'fresh turn']);
  });

  it('looks up the latest checkpoint in the requested namespace', async () => {
    const checkpointer = new MemorySaver();
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await putCheckpoint({
      checkpointer,
      threadId: session.threadId,
      id: 'checkpoint_requested_namespace',
      checkpointNs: 'requested',
    });

    await expect(session.getLatestCheckpoint()).resolves.toBeUndefined();
    await expect(
      session.getLatestCheckpoint({ checkpointNs: 'requested' })
    ).resolves.toMatchObject({
      checkpointId: 'checkpoint_requested_namespace',
      checkpointNs: 'requested',
    });
  });

  it('resets stale checkpoint state when branching changes the active JSONL path', async () => {
    const checkpointer = new MemorySaver();
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    const store = session.getSessionStore();
    const first = await store?.appendMessage(new HumanMessage('first'));
    await store?.appendMessage(new AIMessage('second'));
    await putCheckpoint({
      checkpointer,
      threadId: session.threadId,
      id: 'checkpoint_to_reset',
    });

    await session.branch(first?.id ?? '', { position: 'at' });

    const tuple = await checkpointer.getTuple({
      configurable: { thread_id: session.threadId },
    });
    expect(tuple).toBeUndefined();
    expect(store?.getCheckpoints(session.threadId).at(-1)?.data).toMatchObject({
      source: 'reset',
      reason: 'branch',
    });
  });

  it('records and resets child checkpoint threads owned by a run', async () => {
    const checkpointer = new MemorySaver();
    const childThreadId = 'subagent:owned-child';
    const mockRun = createMockRun('child result');
    mockRun.getChildCheckpointThreadIds.mockReturnValue([childThreadId]);
    mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await putCheckpoint({
      checkpointer,
      threadId: childThreadId,
      id: 'checkpoint_child',
    });

    await session.run('fresh turn', { runId: 'run_with_child' });

    expect(
      session.getSessionStore()?.getCheckpoints(childThreadId).at(-1)?.data
    ).toMatchObject({
      source: 'run',
      runId: 'run_with_child',
      threadId: childThreadId,
    });
    const firstMessage = session
      .getSessionStore()
      ?.getPath()
      .find((entry) => entry.type === 'message');
    await session.branch(firstMessage?.id ?? '', { position: 'at' });

    const tuple = await checkpointer.getTuple({
      configurable: { thread_id: childThreadId },
    });
    expect(tuple).toBeUndefined();
    expect(
      session.getSessionStore()?.getCheckpoints(childThreadId).at(-1)?.data
    ).toMatchObject({ source: 'reset', reason: 'branch' });
  });

  it('keeps checkpoint state when branching to the active JSONL leaf', async () => {
    const checkpointer = new MemorySaver();
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    const store = session.getSessionStore();
    const active = await store?.appendMessage(new HumanMessage('current'));
    await putCheckpoint({
      checkpointer,
      threadId: session.threadId,
      id: 'checkpoint_to_keep',
    });

    await session.branch(active?.id ?? '', { position: 'at' });

    const tuple = await checkpointer.getTuple({
      configurable: { thread_id: session.threadId },
    });
    expect(tuple?.checkpoint.id).toBe('checkpoint_to_keep');
    expect(
      store
        ?.getCheckpoints(session.threadId)
        .some((checkpoint) => checkpoint.data.source === 'reset')
    ).toBe(false);
  });

  it('resets overridden thread checkpoints when branching changes the active path', async () => {
    const checkpointer = new MemorySaver();
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    const store = session.getSessionStore();
    const first = await store?.appendMessage(new HumanMessage('first'));
    await store?.appendMessage(new AIMessage('second'));
    await putCheckpoint({
      checkpointer,
      threadId: 'thread_override',
      id: 'checkpoint_override',
    });
    await store?.appendCheckpoint({
      source: 'run',
      threadId: 'thread_override',
      runId: 'run_override',
      checkpointId: 'checkpoint_override',
    });

    await session.branch(first?.id ?? '', { position: 'at' });

    const tuple = await checkpointer.getTuple({
      configurable: { thread_id: 'thread_override' },
    });
    const reset = store
      ?.getCheckpoints('thread_override')
      .find((checkpoint) => checkpoint.data.source === 'reset');
    expect(tuple).toBeUndefined();
    expect(reset?.data.reason).toBe('branch');
  });

  it('ignores a malformed persisted tier instead of poisoning later merges', async () => {
    const sessionPath = join(dir, 'corrupt-tier.jsonl');
    const store = await JsonlSessionStore.create({
      path: sessionPath,
      cwd: dir,
      sessionId: 'corrupt-tier-session',
    });
    await store.appendFadingState({
      threadId: store.header.id,
      fadingTier: {
        v: 2,
        budgetTokens: 'abc',
        masked: true,
      } as unknown as t.FadingTier,
      fadingTiers: {
        default: {
          v: 2,
          budgetTokens: Number.NaN,
          masked: true,
        } as t.FadingTier,
      },
    });
    const fadingTier: t.FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    const mockRun = createMockRun('ok');
    mockRun.getFadingTier.mockReturnValue(fadingTier);
    mockRun.getFadingTiers.mockReturnValue({ default: fadingTier });
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      sessionPath,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    await session.run('first');
    await session.run('second');

    expect(capturedConfigs[0].fadingTier).toBeUndefined();
    expect(capturedConfigs[0].fadingTiers ?? {}).toEqual({});
    expect(capturedConfigs[1].fadingTier).toEqual(fadingTier);
    expect(capturedConfigs[1].fadingTiers).toEqual({ default: fadingTier });
  });

  it('persists a tier for an agent whose ID is an object prototype key', async () => {
    const fadingTier: t.FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    const mockRun = createMockRun('ok');
    mockRun.getFadingTiers
      .mockReturnValueOnce({ worker: fadingTier })
      .mockReturnValueOnce({ constructor: fadingTier });
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    await session.run('first');
    await session.run('second');
    await session.run('third');

    const tiers = capturedConfigs[2].fadingTiers ?? {};
    expect(Object.hasOwn(tiers, 'constructor')).toBe(true);
    expect(tiers.constructor).toEqual(fadingTier);
    expect(tiers.worker).toEqual(fadingTier);
  });

  it('records run.failed when resumeInterrupt throws', async () => {
    const mockRun = createMockRun('unused');
    const fadingTier: t.FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    mockRun.resume.mockRejectedValue(new Error('resume failed'));
    mockRun.getFadingTier.mockReturnValue(fadingTier);
    mockRun.getFadingTiers.mockReturnValue({ default: fadingTier });
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      humanInTheLoop: { enabled: true },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    await expect(
      session.resumeInterrupt([{ type: 'approve' }], {
        runId: 'run_resume_failure',
      })
    ).rejects.toThrow('resume failed');

    const events = session
      .getSessionStore()
      ?.getEntries()
      .filter((entry) => entry.type === 'run_event')
      .map((entry) => entry.data.event);
    expect(events).toEqual(['run.started', 'run.failed']);
    await session.run('after failed resume');
    expect(capturedConfigs[1].fadingTier).toEqual(fadingTier);
    expect(capturedConfigs[1].fadingTiers).toEqual({ default: fadingTier });
  });

  it('persists fading tiers when processStream throws', async () => {
    const mockRun = createMockRun('unused');
    const fadingTier: t.FadingTier = {
      v: 2,
      budgetTokens: 12_500,
      masked: true,
      latched: true,
    };
    mockRun.processStream.mockRejectedValueOnce(new Error('run failed'));
    mockRun.getFadingTier.mockReturnValue(fadingTier);
    mockRun.getFadingTiers.mockReturnValue({ default: fadingTier });
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    await expect(session.run('fail after fading')).rejects.toThrow(
      'run failed'
    );
    await session.run('continue');

    expect(capturedConfigs[1].fadingTier).toEqual(fadingTier);
    expect(capturedConfigs[1].fadingTiers).toEqual({ default: fadingTier });
  });

  it('compacts into a summary plus retained active path', async () => {
    mockSummarizer('summary of old work');
    const tokenCounter: t.TokenCounter = () => 7;
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      tokenCounter,
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    const store = session.getSessionStore();
    await store?.appendMessage(new HumanMessage('old'));
    await store?.appendMessage(new AIMessage('old answer'));
    await store?.appendMessage(new HumanMessage('recent'));

    await session.compact({
      instructions: 'summary of old work',
      retainRecentTurns: 1,
    });

    expect(store?.getMessages().map((message) => message.content)).toEqual([
      'summary of old work',
      'recent',
    ]);
    const summary = store
      ?.getEntries()
      .find((entry) => entry.type === 'summary');
    expect(summary?.data.tokenCount).toBeGreaterThan(0);
  });

  it('records no retained ids when compaction retains zero messages', async () => {
    mockSummarizer('summary of everything');
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    const store = session.getSessionStore();
    const user = await store?.appendMessage(new HumanMessage('old'));
    const assistant = await store?.appendMessage(new AIMessage('old answer'));

    await session.compact({ retainRecentTurns: 0 });

    const summary = store
      ?.getEntries()
      .find((entry) => entry.type === 'summary');
    const compaction = store
      ?.getEntries()
      .find((entry) => entry.type === 'compaction');
    expect(summary?.data.retainedEntryIds).toEqual([]);
    expect(summary?.data.summarizedEntryIds).toEqual([user?.id, assistant?.id]);
    expect(compaction?.data.retainedEntryIds).toEqual([]);
  });

  it('clears persisted fading tiers after session compaction', async () => {
    mockSummarizer('summary of prior work');
    const mockRun = createMockRun('continued');
    const capturedConfigs = mockRunCreate(mockRun);
    const fadingTier: t.FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      fadingTier,
      fadingTiers: { default: fadingTier },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    const store = session.getSessionStore();
    await store?.appendMessage(new HumanMessage('old'));
    await store?.appendMessage(new AIMessage('old answer'));

    await session.compact({ retainRecentTurns: 0 });
    await session.run('continue after compaction');

    expect(capturedConfigs[0].fadingTier).toBeUndefined();
    expect(capturedConfigs[0].fadingTiers).toBeUndefined();
  });

  it('carries a tier learned after compaction into the next summarized run', async () => {
    mockSummarizer('summary of prior work');
    const mockRun = createMockRun('continued');
    const fadingTier: t.FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    mockRun.getFadingTier.mockReturnValue(fadingTier);
    mockRun.getFadingTiers.mockReturnValue({ default: fadingTier });
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await session.getSessionStore()?.appendMessage(new HumanMessage('old'));
    await session.getSessionStore()?.appendMessage(new AIMessage('old answer'));

    await session.compact({ retainRecentTurns: 0 });
    await session.run('first post-compaction turn');
    await session.run('second post-compaction turn');

    expect(capturedConfigs[0].fadingTier).toBeUndefined();
    expect(capturedConfigs[1].fadingTier).toEqual(fadingTier);
    expect(capturedConfigs[1].fadingTiers).toEqual({ default: fadingTier });
  });

  it('carries calibration ratio forward after resumeInterrupt', async () => {
    const mockRun = createMockRun('resumed');
    mockRun.getCalibrationRatio.mockReturnValue(2);
    const fadingTier: t.FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    mockRun.getFadingTier.mockReturnValue(fadingTier);
    mockRun.getFadingTiers.mockReturnValue({ default: fadingTier });
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    await session.resumeInterrupt([]);
    await session.run('after resume');

    expect(capturedConfigs[1].calibrationRatio).toBe(2);
    expect(capturedConfigs[1].fadingTier).toEqual(fadingTier);
    expect(capturedConfigs[1].fadingTiers).toEqual({ default: fadingTier });
  });

  it('carries live fading tiers into cloned sessions', async () => {
    const mockRun = createMockRun('continued');
    const fadingTier: t.FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    mockRun.getFadingTier.mockReturnValue(fadingTier);
    mockRun.getFadingTiers.mockReturnValue({ default: fadingTier });
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    await session.run('advance tier');
    const cloned = await session.clone({ cwd: dir });
    await cloned.run('continue clone');

    expect(capturedConfigs[1].fadingTier).toEqual(fadingTier);
    expect(capturedConfigs[1].fadingTiers).toEqual({ default: fadingTier });
  });

  it('only carries live fading tiers into a fork that retains the active leaf', async () => {
    const mockRun = createMockRun('continued');
    const fadingTier: t.FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    mockRun.getFadingTier.mockReturnValue(fadingTier);
    mockRun.getFadingTiers.mockReturnValue({ default: fadingTier });
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    await session.run('advance tier');
    const leafId = session.getSessionStore()?.getLeafEntry()?.id ?? '';
    const rewound = await session.fork(leafId, { cwd: dir });
    await rewound.run('continue rewound fork');
    const full = await session.fork(leafId, { cwd: dir, position: 'at' });
    await full.run('continue full fork');

    expect(capturedConfigs[1].fadingTier).toBeUndefined();
    expect(capturedConfigs[1].fadingTiers).toBeUndefined();
    expect(capturedConfigs[2].fadingTier).toEqual(fadingTier);
    expect(capturedConfigs[2].fadingTiers).toEqual({ default: fadingTier });
  });

  it('allows graph compaction to clear a persisted fading tier', async () => {
    const mockRun = createMockRun('continued');
    const fadingTier: t.FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    mockRun.getFadingTier
      .mockReturnValueOnce(fadingTier)
      .mockReturnValue(undefined);
    mockRun.getFadingTiers
      .mockReturnValueOnce({ default: fadingTier })
      .mockReturnValue({});
    mockRun.didResetFadingTier
      .mockReturnValueOnce(false)
      .mockReturnValueOnce(true)
      .mockReturnValue(false);
    mockRun.getFadingTierResetAgentIds
      .mockReturnValueOnce([])
      .mockReturnValueOnce(['default'])
      .mockReturnValue([]);
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    await session.run('advance tier');
    await session.run('compact history');
    await session.run('after compaction');

    expect(capturedConfigs[1].fadingTier).toEqual(fadingTier);
    expect(capturedConfigs[2].fadingTier).toBeUndefined();
    expect(capturedConfigs[2].fadingTiers).toEqual({});
  });

  it('preserves prototype-named agent tiers across runs', async () => {
    const mockRun = createMockRun('continued');
    const fadingTier: t.FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    mockRun.getFadingTiers.mockReturnValue(
      Object.fromEntries([['__proto__', fadingTier]])
    );
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    await session.run('capture prototype tier');
    await session.run('reuse prototype tier');

    expect(
      Object.hasOwn(capturedConfigs[1].fadingTiers ?? {}, '__proto__')
    ).toBe(true);
    expect(capturedConfigs[1].fadingTiers?.['__proto__']).toEqual(fadingTier);
  });

  it('restores fading tiers when reopening a persisted session', async () => {
    const mockRun = createMockRun('continued');
    const fadingTier: t.FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    mockRun.getFadingTier.mockReturnValue(fadingTier);
    mockRun.getFadingTiers.mockReturnValue({ default: fadingTier });
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    await session.run('advance tier');
    const reopened = await createAgentSession({
      cwd: dir,
      sessionPath: session.sessionPath,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await reopened.run('continue reopened session');

    expect(capturedConfigs[1].fadingTier).toEqual(fadingTier);
    expect(capturedConfigs[1].fadingTiers).toEqual({ default: fadingTier });
  });

  it('keeps fading tiers isolated by checkpoint thread', async () => {
    const mockRun = createMockRun('continued');
    const mainTier: t.FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    const alternateTier: t.FadingTier = {
      v: 2,
      budgetTokens: 12_500,
      masked: true,
      latched: true,
    };
    mockRun.getFadingTier
      .mockReturnValueOnce(mainTier)
      .mockReturnValueOnce(alternateTier)
      .mockReturnValue(mainTier);
    mockRun.getFadingTiers
      .mockReturnValueOnce({ default: mainTier })
      .mockReturnValueOnce({ default: alternateTier })
      .mockReturnValue({ default: mainTier });
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    await session.run('main');
    await session.run('alternate', { threadId: 'alternate-thread' });
    await session.run('main again');
    await session.run('alternate again', { threadId: 'alternate-thread' });

    expect(capturedConfigs[1].fadingTier).toBeUndefined();
    expect(capturedConfigs[2].fadingTier).toEqual(mainTier);
    expect(capturedConfigs[3].fadingTier).toEqual(alternateTier);
  });

  it('keeps fading tiers isolated by checkpoint namespace', async () => {
    const mockRun = createMockRun('continued');
    const namespaceATier: t.FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    const namespaceBTier: t.FadingTier = {
      v: 2,
      budgetTokens: 12_500,
      masked: true,
      latched: true,
    };
    mockRun.getFadingTier
      .mockReturnValueOnce(namespaceATier)
      .mockReturnValueOnce(namespaceBTier)
      .mockReturnValue(namespaceATier);
    mockRun.getFadingTiers
      .mockReturnValueOnce({ default: namespaceATier })
      .mockReturnValueOnce({ default: namespaceBTier })
      .mockReturnValue({ default: namespaceATier });
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    const namespaceOptions = (checkpointNs: string) => ({
      config: { configurable: { checkpoint_ns: checkpointNs } },
    });

    await session.run('namespace A', namespaceOptions('namespace-a'));
    await session.run('namespace B', namespaceOptions('namespace-b'));
    const reopened = await createAgentSession({
      cwd: dir,
      sessionPath: session.sessionPath,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await reopened.run('namespace A again', namespaceOptions('namespace-a'));
    await reopened.run('namespace B again', namespaceOptions('namespace-b'));

    expect(capturedConfigs[0].fadingTier).toBeUndefined();
    expect(capturedConfigs[1].fadingTier).toBeUndefined();
    expect(capturedConfigs[2].fadingTier).toEqual(namespaceATier);
    expect(capturedConfigs[3].fadingTier).toEqual(namespaceBTier);
  });

  it('persists an interrupted tier under the interrupt checkpoint namespace', async () => {
    const checkpointer = new MemorySaver();
    const mockRun = createMockRun('interrupted');
    const nestedTier: t.FadingTier = {
      v: 2,
      budgetTokens: 12_500,
      masked: true,
      latched: true,
    };
    mockRun.getFadingTier.mockReturnValue(nestedTier);
    mockRun.getFadingTiers.mockReturnValue({ default: nestedTier });
    mockRun.getInterrupt.mockReturnValueOnce({
      interruptId: 'nested-interrupt',
      threadId: 'nested-thread',
      checkpointId: 'nested-checkpoint',
      checkpointNs: 'nested-namespace',
      payload: {
        type: 'tool_approval',
        action_requests: [],
        review_configs: [],
      },
    });
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await putCheckpoint({
      checkpointer,
      threadId: session.threadId,
      id: 'nested-checkpoint',
      checkpointNs: 'nested-namespace',
    });

    await session.run('pause in nested namespace');
    await session.resumeInterrupt([{ type: 'approve' }]);

    expect(capturedConfigs[1].fadingTier).toEqual(nestedTier);
    expect(capturedConfigs[1].fadingTiers).toEqual({ default: nestedTier });
  });

  it('merges overlapping run tiers monotonically', async () => {
    const mockRun = createMockRun('continued');
    const deepTier: t.FadingTier = {
      v: 2,
      budgetTokens: 12_500,
      masked: true,
      latched: true,
    };
    const staleTier: t.FadingTier = {
      v: 2,
      budgetTokens: 50_000,
      masked: false,
    };
    let markSlowStarted!: () => void;
    let releaseSlow!: () => void;
    const slowStarted = new Promise<void>((resolve) => {
      markSlowStarted = resolve;
    });
    const slowResult = new Promise<t.MessageContentComplex[]>((resolve) => {
      releaseSlow = () => resolve([{ type: 'text', text: 'slow' }]);
    });
    mockRun.processStream
      .mockImplementationOnce(async () => {
        markSlowStarted();
        return slowResult;
      })
      .mockResolvedValue([{ type: 'text', text: 'fast' }]);
    mockRun.getFadingTier
      .mockReturnValueOnce(deepTier)
      .mockReturnValueOnce(staleTier)
      .mockReturnValue(deepTier);
    mockRun.getFadingTiers
      .mockReturnValueOnce({ default: deepTier })
      .mockReturnValueOnce({ default: staleTier })
      .mockReturnValue({ default: deepTier });
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    const slowRun = session.run('slow run');
    await slowStarted;
    await session.run('fast run');
    releaseSlow();
    await slowRun;
    await session.run('after overlap');

    expect(capturedConfigs[0].fadingTier).toBeUndefined();
    expect(capturedConfigs[1].fadingTier).toBeUndefined();
    expect(capturedConfigs[2].fadingTier).toEqual(deepTier);
    expect(capturedConfigs[2].fadingTiers).toEqual({ default: deepTier });
  });

  it('rejects a stale capture that started before a concurrent compaction', async () => {
    const mockRun = createMockRun('continued');
    const staleTier: t.FadingTier = {
      v: 2,
      budgetTokens: 12_500,
      masked: true,
      latched: true,
    };
    let markResetStarted!: () => void;
    let markStaleStarted!: () => void;
    let releaseReset!: () => void;
    let releaseStale!: () => void;
    const resetStarted = new Promise<void>((resolve) => {
      markResetStarted = resolve;
    });
    const staleStarted = new Promise<void>((resolve) => {
      markStaleStarted = resolve;
    });
    const resetResult = new Promise<t.MessageContentComplex[]>((resolve) => {
      releaseReset = () => resolve([{ type: 'text', text: 'reset' }]);
    });
    const staleResult = new Promise<t.MessageContentComplex[]>((resolve) => {
      releaseStale = () => resolve([{ type: 'text', text: 'stale' }]);
    });
    mockRun.processStream
      .mockImplementationOnce(async () => {
        markResetStarted();
        return resetResult;
      })
      .mockImplementationOnce(async () => {
        markStaleStarted();
        return staleResult;
      })
      .mockResolvedValue([{ type: 'text', text: 'after' }]);
    mockRun.getFadingTier
      .mockReturnValueOnce(undefined)
      .mockReturnValueOnce(staleTier)
      .mockReturnValue(undefined);
    mockRun.getFadingTiers
      .mockReturnValueOnce({})
      .mockReturnValueOnce({ default: staleTier })
      .mockReturnValue({});
    mockRun.didResetFadingTier.mockReturnValueOnce(true).mockReturnValue(false);
    mockRun.getFadingTierResetAgentIds
      .mockReturnValueOnce(['default'])
      .mockReturnValue([]);
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    const resetRun = session.run('reset run');
    await resetStarted;
    const staleRun = session.run('stale run');
    await staleStarted;
    releaseReset();
    await resetRun;
    releaseStale();
    await staleRun;
    await session.run('after overlap');

    expect(capturedConfigs[2].fadingTier).toBeUndefined();
    expect(capturedConfigs[2].fadingTiers).toEqual({});
  });

  it('discards a capture from a run that started before an explicit history rewrite', async () => {
    const mockRun = createMockRun('continued');
    const preRewriteTier: t.FadingTier = {
      v: 2,
      budgetTokens: 12_500,
      masked: true,
      latched: true,
    };
    let markStaleStarted!: () => void;
    let releaseStale!: () => void;
    const staleStarted = new Promise<void>((resolve) => {
      markStaleStarted = resolve;
    });
    const staleResult = new Promise<t.MessageContentComplex[]>((resolve) => {
      releaseStale = () => resolve([{ type: 'text', text: 'stale' }]);
    });
    mockRun.processStream
      .mockResolvedValueOnce([{ type: 'text', text: 'seed' }])
      .mockImplementationOnce(async () => {
        markStaleStarted();
        return staleResult;
      })
      .mockResolvedValue([{ type: 'text', text: 'after' }]);
    mockRun.getFadingTier
      .mockReturnValueOnce(undefined)
      .mockReturnValueOnce(preRewriteTier)
      .mockReturnValue(undefined);
    mockRun.getFadingTiers
      .mockReturnValueOnce({})
      .mockReturnValueOnce({ default: preRewriteTier })
      .mockReturnValue({});
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    await session.run('seed run');
    const firstEntryId = session.getSessionStore()?.getPath()[0]?.id;
    expect(firstEntryId).toBeDefined();
    const staleRun = session.run('stale run');
    await staleStarted;
    await session.branch(firstEntryId as string);
    releaseStale();
    await staleRun;
    await session.run('after rewrite');

    expect(capturedConfigs[2].fadingTier).toBeUndefined();
    expect(capturedConfigs[2].fadingTiers ?? {}).toEqual({});
  });

  it('keeps a concurrent escalation from another agent when one agent resets', async () => {
    const mockRun = createMockRun('continued');
    const escalatedTier: t.FadingTier = {
      v: 2,
      budgetTokens: 12_500,
      masked: true,
      latched: true,
    };
    let markResetStarted!: () => void;
    let markEscalationStarted!: () => void;
    let releaseReset!: () => void;
    let releaseEscalation!: () => void;
    const resetStarted = new Promise<void>((resolve) => {
      markResetStarted = resolve;
    });
    const escalationStarted = new Promise<void>((resolve) => {
      markEscalationStarted = resolve;
    });
    const resetResult = new Promise<t.MessageContentComplex[]>((resolve) => {
      releaseReset = () => resolve([{ type: 'text', text: 'reset' }]);
    });
    const escalationResult = new Promise<t.MessageContentComplex[]>(
      (resolve) => {
        releaseEscalation = () =>
          resolve([{ type: 'text', text: 'escalated' }]);
      }
    );
    mockRun.processStream
      .mockImplementationOnce(async () => {
        markResetStarted();
        return resetResult;
      })
      .mockImplementationOnce(async () => {
        markEscalationStarted();
        return escalationResult;
      })
      .mockResolvedValue([{ type: 'text', text: 'after' }]);
    mockRun.getFadingTiers
      .mockReturnValueOnce({})
      .mockReturnValueOnce({ worker: escalatedTier })
      .mockReturnValue({});
    mockRun.getFadingTierResetAgentIds
      .mockReturnValueOnce(['planner'])
      .mockReturnValue([]);
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    const resetRun = session.run('planner compacts');
    await resetStarted;
    const escalationRun = session.run('worker escalates');
    await escalationStarted;
    releaseReset();
    await resetRun;
    releaseEscalation();
    await escalationRun;
    await session.run('after overlap');

    expect(capturedConfigs[2].fadingTiers).toEqual({ worker: escalatedTier });
  });

  it('bounds fading metadata retained for alternate checkpoint threads', async () => {
    const mockRun = createMockRun('continued');
    const fadingTier: t.FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    mockRun.getFadingTier.mockReturnValue(fadingTier);
    mockRun.getFadingTiers.mockReturnValue({ default: fadingTier });
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    await session.run('main');
    for (let i = 0; i < 65; i++) {
      await session.run(`alternate ${i}`, { threadId: `thread-${i}` });
    }
    expect(
      session.getSessionStore()?.getFadingStates(session.threadId, 64)
    ).toHaveLength(65);
    const reopened = await createAgentSession({
      cwd: dir,
      sessionPath: session.sessionPath,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await reopened.run('oldest again', { threadId: 'thread-0' });
    await reopened.run('newest again', { threadId: 'thread-64' });

    expect(capturedConfigs[66].fadingTier).toBeUndefined();
    expect(capturedConfigs[67].fadingTier).toEqual(fadingTier);
  });

  it('summarizes an abandoned branch before switching in place', async () => {
    mockSummarizer('summary of abandoned branch');
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    const store = session.getSessionStore();
    const first = await store?.appendMessage(new HumanMessage('first'));
    const abandoned = await store?.appendMessage(
      new AIMessage('abandoned answer')
    );

    await session.branch(first?.id ?? '', {
      position: 'at',
      summarizeAbandoned: {
        instructions: 'summarize abandoned branch',
      },
    });

    const activePath = store?.getPath();
    const summary = activePath?.at(-1);
    expect(activePath?.map((entry) => entry.id)).toEqual([
      first?.id,
      summary?.id,
    ]);
    expect(summary).toMatchObject({
      type: 'summary',
      parentId: first?.id,
      data: {
        text: 'summary of abandoned branch',
        summarizedEntryIds: [abandoned?.id],
        instructions: 'summarize abandoned branch',
      },
    });
    expect(
      store?.getEntries().some((entry) => entry.type === 'compaction')
    ).toBe(true);
  });

  it('summarizes a sibling branch before switching branches', async () => {
    mockSummarizer('summary of sibling branch');
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    const store = session.getSessionStore();
    const first = await store?.appendMessage(new HumanMessage('first'));
    const inactive = await store?.appendMessage(new AIMessage('inactive'));
    await store?.branch(first?.id ?? '');
    const activeSibling = await store?.appendMessage(new AIMessage('active'));

    await session.branch(inactive?.id ?? '', {
      position: 'at',
      summarizeAbandoned: true,
    });

    const activePath = store?.getPath();
    const summary = activePath?.at(-1);
    expect(activePath?.map((entry) => entry.id)).toEqual([
      first?.id,
      inactive?.id,
      summary?.id,
    ]);
    expect(summary).toMatchObject({
      type: 'summary',
      parentId: inactive?.id,
      data: {
        text: 'summary of sibling branch',
        summarizedEntryIds: [activeSibling?.id],
      },
    });
  });
});
