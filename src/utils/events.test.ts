import { dispatchCustomEvent } from '@langchain/core/callbacks/dispatch';
import type { ToolExecuteBatchRequest, ToolExecuteResult } from '@/types/tools';
import { safeDispatchCustomEvent } from './events';
import { traceHostToolResults } from '@/langfuse';
import { GraphEvents } from '@/common';

jest.mock('@langchain/core/callbacks/dispatch', () => ({
  dispatchCustomEvent: jest.fn(),
}));
jest.mock('@/langfuse', () => ({ traceHostToolResults: jest.fn() }));

describe('host tool metadata completion', () => {
  it.each([false, true])(
    'settles the original batch after tracing (failure=%s)',
    async (failure) => {
      const results: ToolExecuteResult[] = [
        { toolCallId: 'call', status: 'success', content: 'rows' },
      ];
      let finishTracing!: () => void;
      jest.mocked(traceHostToolResults).mockImplementationOnce(
        () =>
          new Promise<void>((resolve, reject) => {
            finishTracing = () =>
              failure ? reject(new Error('telemetry unavailable')) : resolve();
          })
      );
      jest
        .mocked(dispatchCustomEvent)
        .mockImplementationOnce(async (_event, payload) => {
          (payload as ToolExecuteBatchRequest).resolve(results);
        });
      const resolve = jest.fn();
      const clock = jest.spyOn(Date, 'now').mockReturnValue(1_000);
      const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
      try {
        await safeDispatchCustomEvent(GraphEvents.ON_TOOL_EXECUTE, {
          toolCalls: [{ id: 'call', name: 'query', args: {} }],
          resolve,
          reject: jest.fn(),
        } as ToolExecuteBatchRequest);
        expect(resolve).not.toHaveBeenCalled();
        clock.mockReturnValue(9_000);
        finishTracing();
        await Promise.resolve();
        expect(resolve).toHaveBeenCalledTimes(1);
        expect(resolve).toHaveBeenCalledWith([
          { toolCallId: 'call', status: 'success', content: 'rows', received_at: 1_000 },
        ]);
        expect(results[0].received_at).toBeUndefined();
      } finally {
        clock.mockRestore();
        warn.mockRestore();
      }
    }
  );
});
