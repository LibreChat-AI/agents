import { BaseCheckpointSaver } from '@langchain/langgraph';
import type { RunnableConfig } from '@langchain/core/runnables';

/** Routes implicit reads to admitted state, then to checkpoints written by this execution. */
export class ToolCheckpointSaver extends BaseCheckpointSaver {
  private readonly cursors: Map<string, string>;
  override getNextVersion: BaseCheckpointSaver['getNextVersion'];

  constructor(
    readonly source: BaseCheckpointSaver,
    private readonly thread: string,
    cursors: ReadonlyMap<string, string>
  ) {
    super(source.serde);
    this.cursors = new Map(cursors);
    this.getNextVersion = source.getNextVersion.bind(source);
  }

  override getTuple(config: RunnableConfig): ReturnType<BaseCheckpointSaver['getTuple']> {
    const scope = config.configurable;
    const cursor = this.cursors.get(scope?.checkpoint_ns ?? '');
    const lookup = scope?.thread_id === this.thread && scope.checkpoint_id == null && scope.thread_ts == null && cursor != null
      ? { ...config, configurable: { ...scope, checkpoint_id: cursor } }
      : config;
    return this.source.getTuple(lookup);
  }

  override list(...args: Parameters<BaseCheckpointSaver['list']>): ReturnType<BaseCheckpointSaver['list']> {
    return this.source.list(...args);
  }

  override async put(...args: Parameters<BaseCheckpointSaver['put']>): ReturnType<BaseCheckpointSaver['put']> {
    const result = await this.source.put(...args);
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
    return this.source.getDeltaChannelHistory(...args);
  }

}
