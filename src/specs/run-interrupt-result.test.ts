import { Providers } from '@/common';
import { Run } from '@/run';

class CustomQuestion {
  constructor(readonly question: string) {}
  describe(): string {
    return this.question;
  }
}

describe('public custom interrupt results', () => {
  test.each([
    { label: 'function', payload: { resolve: () => 'answer' } },
    { label: 'symbol', payload: { resolution: Symbol('resolution') } },
    { label: 'class', payload: new CustomQuestion('Continue?') },
    { label: 'primitive', payload: null },
  ])(
    'preserves custom $label payload semantics in pending entries',
    async ({ payload }) => {
      const run = await Run.create({
        runId: 'custom-interrupt-result',
        graphConfig: {
          type: 'standard',
          agents: [
            {
              agentId: 'agent',
              provider: Providers.OPENAI,
              clientOptions: { apiKey: 'unused', modelName: 'fake' },
            },
          ],
        },
      });
      Object.defineProperty(run, '_interrupt', {
        value: { interruptId: 'custom', payload },
      });
      const result = run.getInterrupt<typeof payload>();
      expect(result?.payload).toBe(payload);
      expect(result?.pendingInterrupts?.[0].payload).toBe(payload);
      if (payload instanceof CustomQuestion)
        expect(result?.pendingInterrupts?.[0].payload).toBeInstanceOf(
          CustomQuestion
        );
    }
  );
});
