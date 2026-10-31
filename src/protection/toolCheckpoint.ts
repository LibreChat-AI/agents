import { BaseCheckpointSaver } from '@langchain/langgraph';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { CheckpointTuple } from '@langchain/langgraph';
import type { ToolCallSources } from './toolResult';

export type ToolCheckpointAdmission = (tuple: CheckpointTuple) => Promise<CheckpointTuple>;
export type ToolCheckpointAdmissionContext = {
  admit: ToolCheckpointAdmission;
  finish: () => Promise<ToolCallSources>;
};

export type ToolCheckpointWriteIdentity = {
  threadId: unknown;
  namespace: unknown;
  checkpointId: string;
};
export type ToolCheckpointReadIdentity = {
  config: RunnableConfig;
  threadId: unknown;
  namespace?: unknown;
  checkpointId?: unknown;
};
export type ToolCheckpointSafety = {
  identify: (config: RunnableConfig, history?: boolean) => ToolCheckpointReadIdentity;
  read: (tuple: CheckpointTuple, expected: ToolCheckpointReadIdentity) => CheckpointTuple;
  write: (config: RunnableConfig, expected: ToolCheckpointWriteIdentity) => RunnableConfig;
  delta: (channels: readonly string[]) => void;
};

/** Routes implicit reads to admitted state, then to checkpoints written by this execution. */
export class ToolCheckpointSaver extends BaseCheckpointSaver {
  private readonly cursors: Map<string, string>;
  override getNextVersion: BaseCheckpointSaver['getNextVersion'];

  constructor(
    readonly source: BaseCheckpointSaver,
    private readonly thread: string,
    cursors: ReadonlyMap<string, string>,
    private readonly safety: ToolCheckpointSafety,
    private readonly admit?: ToolCheckpointAdmission
  ) {
    super(source.serde);
    this.cursors = new Map(cursors);
    this.getNextVersion = source.getNextVersion.bind(source);
  }

  override async getTuple(config: RunnableConfig): ReturnType<BaseCheckpointSaver['getTuple']> {
    const selected = this.safety.identify(config);
    const lookup = this.selectCursor(selected.config);
    const expected = { ...selected, checkpointId: lookup.configurable?.checkpoint_id ?? selected.checkpointId };
    const returned = await this.source.getTuple(lookup);
    if (returned == null) return returned;
    const tuple = this.safety.read(returned, expected);
    return this.admit == null ? tuple : this.admit(tuple);
  }

  private selectCursor(config: RunnableConfig): RunnableConfig {
    const scope = config.configurable;
    const cursor = this.cursors.get(scope?.checkpoint_ns ?? '');
    return scope?.thread_id === this.thread && scope.checkpoint_id == null && scope.thread_ts == null && cursor != null
      ? { ...config, configurable: { ...scope, checkpoint_id: cursor } }
      : config;
  }

  override async *list(...args: Parameters<BaseCheckpointSaver['list']>): ReturnType<BaseCheckpointSaver['list']> {
    const expected = this.safety.identify(args[0], true);
    for await (const returned of this.source.list(expected.config, args[1])) {
      const tuple = this.safety.read(returned, expected);
      yield this.admit == null ? tuple : await this.admit(tuple);
    }
  }

  override async put(...args: Parameters<BaseCheckpointSaver['put']>): ReturnType<BaseCheckpointSaver['put']> {
    const expected = { threadId: args[0].configurable?.thread_id, namespace: args[0].configurable?.checkpoint_ns ?? '', checkpointId: args[1].id };
    const result = this.safety.write(await this.source.put(...args), expected);
    const scope = result.configurable;
    if (scope?.thread_id === this.thread && typeof scope.checkpoint_id === 'string') {
      this.cursors.set(scope.checkpoint_ns ?? '', scope.checkpoint_id);
    }
    return result;
  }

  override putWrites(...args: Parameters<BaseCheckpointSaver['putWrites']>): ReturnType<BaseCheckpointSaver['putWrites']> {
    return this.source.putWrites(...args);
  }

  override deleteThread(...args: Parameters<BaseCheckpointSaver['deleteThread']>): ReturnType<BaseCheckpointSaver['deleteThread']> {
    return this.source.deleteThread(...args);
  }

  override getDeltaChannelHistory(...args: Parameters<BaseCheckpointSaver['getDeltaChannelHistory']>): ReturnType<BaseCheckpointSaver['getDeltaChannelHistory']> {
    this.safety.delta(args[0].channels);
    return this.source.getDeltaChannelHistory({ ...args[0], config: this.selectCursor(this.safety.identify(args[0].config).config) });
  }

}
