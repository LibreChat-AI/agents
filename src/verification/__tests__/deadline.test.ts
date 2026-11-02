import { withinDeadline } from '../deadline';

it('rejects an overdue result even when the timer callback has not run', async () => {
  let now = 0;
  const clock = jest.spyOn(performance, 'now').mockImplementation(() => now);
  const settled = jest.fn();
  try {
    const result = await withinDeadline(async () => {
      now = 11;
      return { passed: true };
    }, new AbortController().signal, 10, settled);
    expect(result.status).toBe('timed_out');
    expect(settled).toHaveBeenCalledTimes(1);
  } finally {
    clock.mockRestore();
  }
});
