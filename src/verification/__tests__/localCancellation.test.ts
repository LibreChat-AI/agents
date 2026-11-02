import { spawn } from 'child_process';
import type { ChildProcess } from 'child_process';
import type { LocalSpawn } from '@/types';
import { localCommandRunner } from '..';

describe('local verification cancellation', () => {
  it('does not spawn when cancellation already happened', async () => {
    const launcher = jest.fn<ReturnType<LocalSpawn>, Parameters<LocalSpawn>>();
    const controller = new AbortController();
    controller.abort();
    const runner = localCommandRunner({ spawn: launcher });
    await expect(
      runner('echo unused', {
        signal: controller.signal,
        timeoutMs: 5000,
        verifierId: 'tests',
      })
    ).rejects.toThrow();
    expect(launcher).not.toHaveBeenCalled();
  });

  it('terminates the real child process before the runner settles', async () => {
    const controller = new AbortController();
    const children: ChildProcess[] = [];
    const launcher: LocalSpawn = (command, args, options) => {
      const child = spawn(command, args, { ...options, stdio: 'pipe' });
      children.push(child);
      child.stdout.on('data', (chunk: Buffer) => {
        if (chunk.toString().includes('ready')) controller.abort();
      });
      return child;
    };
    const runner = localCommandRunner({
      spawn: launcher,
      sandbox: { enabled: false },
    });
    try {
      const outcome = await runner(
        'node -e \'process.stdout.write("ready"); setTimeout(() => {}, 30000)\'',
        {
          signal: controller.signal,
          timeoutMs: 5000,
          verifierId: 'tests',
        }
      );
      expect(controller.signal.aborted).toBe(true);
      expect(outcome).toMatchObject({ exitCode: null, timedOut: false });
      expect(children.length).toBeGreaterThan(0);
      expect(
        children.every(
          (child) => child.exitCode !== null || child.signalCode !== null
        )
      ).toBe(true);
    } finally {
      controller.abort();
    }
  }, 10000);
});
