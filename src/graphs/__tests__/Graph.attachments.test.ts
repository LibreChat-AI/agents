import { MemorySaver } from '@langchain/langgraph';
import { Runnable } from '@langchain/core/runnables';
import { describe, expect, it, jest } from '@jest/globals';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
} from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { Providers } from '@/common';
import * as init from '@/llm/init';
import { Run } from '@/run';

/** Answers every call with the same text and remembers what it was sent. */
class RecordingModel extends Runnable<BaseMessage[], AIMessageChunk> {
  lc_namespace = ['tests'];
  readonly calls: BaseMessage[][] = [];

  async invoke(messages: BaseMessage[]): Promise<AIMessageChunk> {
    this.calls.push(messages);
    return new AIMessageChunk({ content: 'done' });
  }
}

const reference = {
  type: 'file' as const,
  source_type: 'id' as const,
  id: 'file-1',
  mime_type: 'application/pdf',
  metadata: { name: 'report.pdf' },
};
const text = { type: 'text', text: 'Summarize the report' };

describe('attachments carried by reference', () => {
  it('sends the host content to the model and keeps the reference in state and checkpoints', async () => {
    const payload = 'JVBERg=='.repeat(4096);
    const resolved = {
      type: 'document',
      source: { type: 'base64', media_type: 'application/pdf', data: payload },
    };
    const resolveAttachment = jest.fn<t.AttachmentResolver>(() => resolved);
    const checkpointer = new MemorySaver();
    const thread = { configurable: { thread_id: 'attachment-reference' } };
    const streamConfig = {
      ...thread,
      streamMode: 'values' as const,
      version: 'v2' as const,
    };
    const run = await Run.create<t.IState>({
      runId: 'attachment-reference',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.ANTHROPIC,
          disableStreaming: true,
          streamUsage: false,
        },
        compileOptions: { checkpointer },
      },
      resolveAttachment,
      returnContent: true,
      skipCleanup: true,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new RecordingModel();
    run.Graph.overrideModel = model;

    await run.processStream(
      { messages: [new HumanMessage({ content: [text, reference] })] },
      streamConfig
    );

    expect(model.calls).toHaveLength(1);
    expect(model.calls[0].at(-1)?.content).toEqual([text, resolved]);
    expect(resolveAttachment).toHaveBeenCalledWith(
      reference,
      expect.objectContaining({ provider: Providers.ANTHROPIC })
    );

    const saved = await checkpointer.getTuple(thread);
    const messages = saved?.checkpoint.channel_values.messages as BaseMessage[];
    expect(messages[0].content).toEqual([text, reference]);
    expect(JSON.stringify(messages)).not.toContain(payload);
  });

  it('resolves references in the history a summary request sends', async () => {
    const resolved = {
      type: 'document',
      source: {
        type: 'base64',
        media_type: 'application/pdf',
        data: 'JVBERg==',
      },
    };
    const llmConfig = {
      provider: Providers.ANTHROPIC,
      disableStreaming: true,
      streamUsage: false,
    };
    const run = await Run.create<t.IState>({
      runId: 'attachment-reference-summary',
      graphConfig: {
        type: 'standard',
        llmConfig,
        maxContextTokens: 100_000,
        summarizationEnabled: true,
        summarizeOnly: true,
      },
      resolveAttachment: () => resolved,
      tokenCounter: () => 10,
      returnContent: true,
      skipCleanup: true,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    run.Graph.overrideModel = new RecordingModel();
    const streamConfig = {
      configurable: { thread_id: 'attachment-reference-summary' },
      streamMode: 'values' as const,
      version: 'v2' as const,
    };
    const summarizer = new RecordingModel();
    const spy = jest.spyOn(init, 'initializeModel').mockReturnValue(summarizer);
    try {
      await run.processStream(
        {
          messages: [
            new HumanMessage({ content: [text, reference] }),
            new AIMessage('The report covers the third quarter.'),
            new HumanMessage('And the fourth?'),
            new AIMessage('It is not in the report.'),
            new HumanMessage('Thanks.'),
            new AIMessage('You are welcome.'),
          ],
        },
        streamConfig
      );
    } finally {
      spy.mockRestore();
    }

    expect(summarizer.calls).toHaveLength(1);
    expect(summarizer.calls[0][0].content).toEqual([text, resolved]);
  });
});
