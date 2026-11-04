import { Client } from 'langsmith';
import { randomUUID } from 'node:crypto';
import { HumanMessage } from '@langchain/core/messages';
import { awaitAllCallbacks } from '@langchain/core/callbacks/promises';
import { LangChainTracer } from '@langchain/core/tracers/tracer_langchain';
import { Providers } from '@/common';
import { Run } from '@/run';

describe('agent streaming with LangSmith tracing', () => {
  beforeEach(() => {
    jest.replaceProperty(process, 'env', {
      ...process.env,
      LANGCHAIN_TRACING: 'false',
      LANGCHAIN_TRACING_V2: 'false',
      LANGSMITH_TRACING_V2: 'false',
      LANGSMITH_TRACING: 'false',
      LANGCHAIN_API_KEY: 'test-key',
      LANGSMITH_API_KEY: 'test-key',
    });
    jest.spyOn(Client.prototype, 'createRun').mockResolvedValue(undefined);
    jest.spyOn(Client.prototype, 'updateRun').mockResolvedValue(undefined);
  });

  afterEach(async () => {
    await awaitAllCallbacks();
    jest.restoreAllMocks();
  });

  it.each([
    { tracing: 'disabled', background: 'true' },
    { tracing: 'explicit', background: 'true' },
    { tracing: 'environment', background: 'true' },
    { tracing: 'environment', background: 'false' },
  ])(
    'completes with $tracing tracing and background=$background',
    async ({ tracing, background }) => {
      process.env.LANGCHAIN_CALLBACKS_BACKGROUND = background;
      if (tracing === 'environment') {
        process.env.LANGCHAIN_TRACING_V2 = 'true';
      }

      const run = await Run.create({
        runId: randomUUID(),
        langfuse: { enabled: false },
        graphConfig: {
          type: 'standard',
          agents: [
            {
              agentId: 'agent',
              provider: Providers.OPENAI,
              clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
              instructions: 'Reply with the single word OK.',
              maxContextTokens: 8000,
            },
          ],
        },
      });
      run.Graph?.overrideTestModel(['OK'], 1);

      await run.processStream(
        { messages: [new HumanMessage('say OK')] },
        {
          version: 'v2',
          callbacks: tracing === 'explicit' ? [new LangChainTracer()] : [],
        }
      );
      await awaitAllCallbacks();

      expect(
        run.Graph?.getRunMessages()?.map((message) => message.content)
      ).toEqual(['OK']);
      if (tracing === 'disabled') {
        expect(Client.prototype.createRun).not.toHaveBeenCalled();
      } else {
        expect(Client.prototype.createRun).toHaveBeenCalled();
        expect(Client.prototype.updateRun).toHaveBeenCalled();
      }
    }
  );
});
