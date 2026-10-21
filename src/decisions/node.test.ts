import path from 'node:path';
import { tmpdir } from 'node:os';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import type { DecisionFetch } from './core';
import {
  withBreaker,
  DecisionError,
  booleanQuestion,
  createHttpDecisionModel,
} from './core';
import { commandToken } from './node';

const answered = JSON.stringify({
  model: 'jev-1.13.0',
  answers: { q: { type: 'noul', noul: 0.9 } },
});

const request = {
  state: 'synthetic state',
  questions: { q: booleanQuestion('Is this synthetic?') },
};

let dir: string;

beforeEach(async () => {
  dir = await mkdtemp(path.join(tmpdir(), 'command-token-'));
});

afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
});

/**
 * A token command that counts its runs in a file, so tests see how often the
 * shell actually ran. `body` runs after the count with `n` in scope.
 */
async function script(body: string): Promise<string> {
  const counter = path.join(dir, 'runs');
  const file = path.join(dir, 'mint.js');
  await writeFile(
    file,
    [
      'const fs = require("node:fs");',
      `const counter = ${JSON.stringify(counter)};`,
      'const n = (fs.existsSync(counter) ? Number(fs.readFileSync(counter, "utf8")) : 0) + 1;',
      'fs.writeFileSync(counter, String(n));',
      body,
    ].join('\n')
  );
  return `"${process.execPath}" "${file}"`;
}

async function runs(): Promise<number> {
  try {
    return Number(await readFile(path.join(dir, 'runs'), 'utf8'));
  } catch {
    return 0;
  }
}

/** Whether a process is gone within two seconds; the command forked by the shell must not outlive a kill. */
async function exited(pid: number): Promise<boolean> {
  for (let i = 0; i < 100; i++) {
    try {
      process.kill(pid, 0);
    } catch {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  return false;
}

async function rejection(promise: Promise<unknown>): Promise<DecisionError> {
  const error = await promise.then(
    () => undefined,
    (thrown: unknown) => thrown
  );
  if (!(error instanceof DecisionError)) {
    throw new Error('expected a DecisionError');
  }
  return error;
}

describe('commandToken', () => {
  it('uses the last non-empty stdout line and caches it for ttlMs', async () => {
    const mint = commandToken(
      await script(
        'process.stdout.write("notice: minted\\n  token-" + n + "  \\n\\n");'
      )
    );
    await expect(mint({ refresh: false })).resolves.toBe('token-1');
    await expect(mint({ refresh: false })).resolves.toBe('token-1');
    expect(await runs()).toBe(1);
  });

  it('runs the command once for concurrent callers', async () => {
    const mint = commandToken(
      await script(
        'setTimeout(() => process.stdout.write("token-" + n + "\\n"), 150);'
      )
    );
    const tokens = await Promise.all([
      mint({ refresh: false }),
      mint({ refresh: false }),
      mint({ refresh: true }),
    ]);
    expect(tokens).toEqual(['token-1', 'token-1', 'token-1']);
    expect(await runs()).toBe(1);
  });

  it('mints again on refresh and after the ttl expires', async () => {
    /** Expiry runs on a skewed clock, so a slow spawn cannot expire a fresh token. */
    const realNow = Date.now.bind(Date);
    let skewMs = 0;
    const clock = jest
      .spyOn(Date, 'now')
      .mockImplementation(() => realNow() + skewMs);
    try {
      const mint = commandToken(
        await script('process.stdout.write("token-" + n + "\\n");'),
        { ttlMs: 60_000 }
      );
      await expect(mint({ refresh: false })).resolves.toBe('token-1');
      await expect(mint({ refresh: true })).resolves.toBe('token-2');
      await expect(mint({ refresh: false })).resolves.toBe('token-2');
      skewMs = 60_001;
      await expect(mint({ refresh: false })).resolves.toBe('token-3');
      expect(await runs()).toBe(3);
    } finally {
      clock.mockRestore();
    }
  });

  it('counts the ttl from when the command finished, not when it started', async () => {
    let now = 1_000_000;
    const clock = jest.spyOn(Date, 'now').mockImplementation(() => now);
    try {
      const mint = commandToken(
        await script('process.stdout.write("token-" + n + "\\n");'),
        { ttlMs: 5_000 }
      );
      const first = mint({ refresh: false });
      /** The command takes longer than the ttl. */
      now += 10_000;
      await expect(first).resolves.toBe('token-1');
      now += 4_000;
      await expect(mint({ refresh: false })).resolves.toBe('token-1');
      expect(await runs()).toBe(1);
      now += 1_001;
      await expect(mint({ refresh: false })).resolves.toBe('token-2');
      expect(await runs()).toBe(2);
    } finally {
      clock.mockRestore();
    }
  });

  it('redacts a refused token when the refresh that replaces it fails', async () => {
    const refused = 'tok-medium-secret-1234';
    const mint = commandToken(
      await script(
        [
          `if (n === 1) { process.stdout.write(${JSON.stringify(`${refused}\n`)}); process.exit(0); }`,
          `process.stderr.write(${JSON.stringify(`refresh failed for ${refused}\n`)});`,
          'process.exit(1);',
        ].join('\n')
      )
    );
    await expect(mint({ refresh: false })).resolves.toBe(refused);
    const error = await rejection(mint({ refresh: true }));
    expect(error.message).toBe(
      'token command exited with code 1: refresh failed for [REDACTED]'
    );
    expect(error.message).not.toContain(refused);
  });

  it('kills a run once its only caller aborts and starts fresh for the next call', async () => {
    const pidFile = path.join(dir, 'pid');
    const mint = commandToken(
      await script(
        [
          `if (n === 1) { fs.writeFileSync(${JSON.stringify(pidFile)}, String(process.pid)); setTimeout(() => process.stdout.write("token-1\\n"), 30_000); }`,
          'else process.stdout.write("token-" + n + "\\n");',
        ].join('\n')
      ),
      { timeoutMs: 60_000 }
    );
    const caller = new AbortController();
    const first = mint({ refresh: false, signal: caller.signal });
    while (!(await readFile(pidFile, 'utf8').catch(() => ''))) {
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    const pid = Number(await readFile(pidFile, 'utf8'));
    caller.abort();
    const error = await rejection(first);
    expect(error).toMatchObject({ failure: 'aborted', provider: 'command' });

    await expect(mint({ refresh: false })).resolves.toBe('token-2');
    expect(await runs()).toBe(2);
    expect(await exited(pid)).toBe(true);
  });

  it('keeps a shared run alive while another caller still waits on it', async () => {
    const mint = commandToken(
      await script(
        'setTimeout(() => process.stdout.write("token-" + n + "\\n"), 300);'
      )
    );
    const leaving = new AbortController();
    const left = mint({ refresh: false, signal: leaving.signal });
    const staying = mint({
      refresh: false,
      signal: new AbortController().signal,
    });
    leaving.abort();
    expect(await rejection(left)).toMatchObject({ failure: 'aborted' });
    await expect(staying).resolves.toBe('token-1');
    await expect(mint({ refresh: false })).resolves.toBe('token-1');
    expect(await runs()).toBe(1);
  });

  it('rejects an already-aborted caller without running the command', async () => {
    const mint = commandToken(
      await script('process.stdout.write("token-" + n + "\\n");')
    );
    const error = await rejection(
      mint({ refresh: false, signal: AbortSignal.abort() })
    );
    expect(error).toMatchObject({ failure: 'aborted' });
    expect(await runs()).toBe(0);
  });

  it('fails as unauthorized with a redacted stderr excerpt and never the token', async () => {
    const leaked = 'tok_live_0123456789';
    const mint = commandToken(
      await script(
        [
          `process.stdout.write(${JSON.stringify(`${leaked}\n`)});`,
          `process.stderr.write(${JSON.stringify(
            `login expired; last token ${leaked}\nAuthorization: Bearer abc.def\n`
          )});`,
          'process.exit(3);',
        ].join('\n')
      )
    );
    const error = await rejection(mint({ refresh: false }));
    expect(error.failure).toBe('unauthorized');
    expect(error.provider).toBe('command');
    expect(error.message).toBe(
      'token command exited with code 3: login expired; last token [REDACTED] Authorization: Bearer [REDACTED]'
    );
    expect(error.message).not.toContain(leaked);
    expect(error.message).not.toContain('mint.js');
  });

  it('does not cache a failed run', async () => {
    const mint = commandToken(
      await script(
        'if (n === 1) process.exit(1); process.stdout.write("token-" + n + "\\n");'
      )
    );
    const error = await rejection(mint({ refresh: false }));
    expect(error.message).toBe('token command exited with code 1');
    await expect(mint({ refresh: false })).resolves.toBe('token-2');
  });

  it('rejects a command that prints nothing', async () => {
    const mint = commandToken(await script('process.stdout.write("\\n");'));
    const error = await rejection(mint({ refresh: false }));
    expect(error).toMatchObject({
      failure: 'unauthorized',
      message: 'token command printed no token',
    });
  });

  it('reports a run killed at timeoutMs as a timeout that leaves a breaker closed', async () => {
    let requests = 0;
    const fetch: DecisionFetch = async () => {
      requests += 1;
      return new Response(answered, { status: 200 });
    };
    const model = withBreaker(
      createHttpDecisionModel({
        endpoint: 'https://decisions.example/v1/systemone',
        apiKey: commandToken(
          await script(
            [
              `fs.writeFileSync(${JSON.stringify(path.join(dir, 'pid'))}, String(process.pid));`,
              'setTimeout(() => process.stdout.write("token-" + n + "\\n"), 30_000);',
            ].join('\n')
          ),
          { timeoutMs: 300 }
        ),
        dialect: 'systemone',
        fetch,
        timeoutMs: 10_000,
      })
    );

    const error = await rejection(model.decide(request));
    expect(error).toMatchObject({ failure: 'timeout', provider: 'command' });
    expect(error.message).toMatch(/^token command timed out \(SIG[A-Z]+\)/);
    expect(model.breaker.open).toBe(false);
    expect(requests).toBe(0);
    expect(
      await exited(Number(await readFile(path.join(dir, 'pid'), 'utf8')))
    ).toBe(true);
  });

  it('reports output past the buffer limit as a network failure', async () => {
    const mint = commandToken(
      await script('process.stdout.write("x".repeat(2 * 1024 * 1024));')
    );
    const error = await rejection(mint({ refresh: false }));
    expect(error).toMatchObject({
      failure: 'network',
      message: 'token command failed',
    });
  });

  it('redacts the command when the shell echoes it into stderr', async () => {
    const base = await script(
      [
        'const line = [JSON.stringify(process.execPath), JSON.stringify(__filename), ...process.argv.slice(2)].join(" ");',
        'process.stderr.write("sh: -c: line 1: " + line + "\\n");',
        'process.stderr.write("unknown flag " + process.argv[2] + "\\n");',
        'process.exit(2);',
      ].join('\n')
    );
    const command = `${base} --key=sk_short123`;
    const error = await rejection(commandToken(command)({ refresh: false }));
    expect(error.failure).toBe('unauthorized');
    expect(error.message).toBe(
      'token command exited with code 2: sh: -c: line 1: [REDACTED] unknown flag --key=[REDACTED]'
    );
    expect(error.message).not.toContain('sk_short123');
    expect(error.message).not.toContain('mint.js');
  });

  it('drops the message of a spawn that throws, which quotes the command', async () => {
    const error = await rejection(
      commandToken('mint --key=sk_short123\0')({ refresh: false })
    );
    expect(error).toMatchObject({
      failure: 'network',
      message: 'token command failed',
    });
  });

  it('rejects an empty command and invalid timing at construction', () => {
    expect(() => commandToken('  ')).toThrow(DecisionError);
    expect(() => commandToken('true', { ttlMs: -1 })).toThrow(
      'invalid token command timing'
    );
    expect(() => commandToken('true', { timeoutMs: 0 })).toThrow(
      'invalid token command timing'
    );
    expect(() => commandToken('true', { killGraceMs: -1 })).toThrow(
      'invalid token command timing'
    );
  });

  it('force-kills a timed-out command that ignores SIGTERM after the shell that forked it dies', async () => {
    const pidFile = path.join(dir, 'pid');
    const mint = commandToken(
      `${await script(
        [
          'process.on("SIGTERM", () => {});',
          `fs.writeFileSync(${JSON.stringify(pidFile)}, String(process.pid));`,
          'setTimeout(() => process.stdout.write("token-" + n + "\\n"), 30_000);',
        ].join('\n')
      )}; true`,
      { timeoutMs: 500, killGraceMs: 200 }
    );
    const error = await rejection(mint({ refresh: false }));
    expect(error).toMatchObject({ failure: 'timeout', provider: 'command' });
    expect(await exited(Number(await readFile(pidFile, 'utf8')))).toBe(true);
  });

  it('force-kills an abandoned command that ignores SIGTERM', async () => {
    const pidFile = path.join(dir, 'pid');
    const mint = commandToken(
      await script(
        [
          'process.on("SIGTERM", () => {});',
          `if (n === 1) { fs.writeFileSync(${JSON.stringify(pidFile)}, String(process.pid)); setTimeout(() => process.stdout.write("token-1\\n"), 30_000); }`,
          'else process.stdout.write("token-" + n + "\\n");',
        ].join('\n')
      ),
      { killGraceMs: 200 }
    );
    const caller = new AbortController();
    const first = mint({ refresh: false, signal: caller.signal });
    while (!(await readFile(pidFile, 'utf8').catch(() => ''))) {
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    caller.abort();
    expect(await rejection(first)).toMatchObject({ failure: 'aborted' });
    expect(await exited(Number(await readFile(pidFile, 'utf8')))).toBe(true);
    await expect(mint({ refresh: false })).resolves.toBe('token-2');
  });

  it('re-mints through the transport after a 401, then opens the breaker on a second', async () => {
    const auth: Array<string | undefined> = [];
    const statuses = [401, 200, 401, 401];
    const fetch: DecisionFetch = async (_url, init) => {
      auth.push(init.headers.Authorization);
      const status = statuses[auth.length - 1];
      return new Response(status === 200 ? answered : 'refused', { status });
    };
    const model = withBreaker(
      createHttpDecisionModel({
        endpoint: 'https://decisions.example/v1/systemone',
        apiKey: commandToken(
          await script('process.stdout.write("token-" + n + "\\n");')
        ),
        dialect: 'systemone',
        fetch,
        timeoutMs: 10_000,
      })
    );

    await expect(model.decide(request)).resolves.toMatchObject({
      model: 'jev-1.13.0',
    });
    expect(auth).toEqual(['Bearer token-1', 'Bearer token-2']);

    const error = await rejection(model.decide(request));
    expect(error).toMatchObject({ failure: 'unauthorized', status: 401 });
    expect(auth).toEqual([
      'Bearer token-1',
      'Bearer token-2',
      'Bearer token-2',
      'Bearer token-3',
    ]);
    expect(model.breaker.open).toBe(true);

    const refused = await rejection(model.decide(request));
    expect(refused.circuitOpen).toBe(true);
    expect(auth).toHaveLength(4);
    expect(await runs()).toBe(3);
  });
});
