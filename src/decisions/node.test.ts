import path from 'node:path';
import { tmpdir } from 'node:os';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import type { DecisionFetch } from './core';
import {
  withBreaker,
  createBreaker,
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

/** Kills `pid` if it is still running; cleanup for a process a test leaves behind. */
function sweep(pid: number): void {
  try {
    process.kill(pid, 'SIGKILL');
  } catch {
    // Already gone.
  }
}

async function pidFrom(file: string): Promise<number> {
  while (!(await readFile(file, 'utf8').catch(() => ''))) {
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  return Number(await readFile(file, 'utf8'));
}

function activeTimeouts(): number {
  return process
    .getActiveResourcesInfo()
    .filter((resource) => resource === 'Timeout').length;
}

/** Runs `body` in a separate Node host with `commandToken` in scope and resolves its stdout once that host exits. */
async function host(body: string): Promise<string> {
  const file = path.join(dir, 'host.ts');
  await writeFile(
    file,
    [
      `import { commandToken } from ${JSON.stringify(path.join(__dirname, 'node'))};`,
      body,
    ].join('\n')
  );
  return new Promise<string>((resolve, reject) => {
    execFile(
      process.execPath,
      [require.resolve('tsx/cli'), file],
      { timeout: 20_000 },
      (error, stdout) => (error ? reject(error) : resolve(stdout))
    );
  });
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

  it('redacts a cached token shorter than eight characters when its refresh fails', async () => {
    const refused = 'Zq7xK2p';
    const mint = commandToken(
      await script(
        [
          `if (n === 1) { process.stdout.write(${JSON.stringify(`${refused}\n`)}); process.exit(0); }`,
          `process.stderr.write(${JSON.stringify(`refresh failed: token ${refused} was refused\n`)});`,
          'process.exit(1);',
        ].join('\n')
      )
    );
    await expect(mint({ refresh: false })).resolves.toBe(refused);
    const error = await rejection(mint({ refresh: true, refused }));
    expect(error.message).toBe(
      'token command exited with code 1: refresh failed: token [REDACTED] was refused'
    );
  });

  it('redacts a token a run detached by a breaker reset minted, though it was never cached', async () => {
    const breaker = createBreaker();
    const detached = 'Ab3dE6';
    const mint = commandToken(
      await script(
        [
          `if (n === 1) setTimeout(() => process.stdout.write(${JSON.stringify(`${detached}\n`)}), 300);`,
          'else if (n === 2) process.stdout.write("Qw9rT2\\n");',
          `else { process.stderr.write(${JSON.stringify(`cannot refresh ${detached}\n`)}); process.exit(1); }`,
        ].join('\n')
      ),
      { breaker }
    );
    const before = mint({ refresh: false });
    while ((await runs()) < 1) {
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    breaker.reset();
    await expect(mint({ refresh: false })).resolves.toBe('Qw9rT2');
    /** The command's own store now holds the token the detached run minted last. */
    await expect(before).resolves.toBe(detached);
    const error = await rejection(mint({ refresh: true, refused: 'Qw9rT2' }));
    expect(error.message).toBe(
      'token command exited with code 1: cannot refresh [REDACTED]'
    );
    expect(await runs()).toBe(3);
  });

  it('hands a refresh that refused an older token the one minted since', async () => {
    const mint = commandToken(
      await script('process.stdout.write("token-" + n + "\\n");')
    );
    await expect(mint({ refresh: false })).resolves.toBe('token-1');
    await expect(mint({ refresh: true, refused: 'token-1' })).resolves.toBe(
      'token-2'
    );
    await expect(mint({ refresh: true, refused: 'token-1' })).resolves.toBe(
      'token-2'
    );
    expect(await runs()).toBe(2);
    await expect(mint({ refresh: true, refused: 'token-2' })).resolves.toBe(
      'token-3'
    );
    expect(await runs()).toBe(3);
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

  it('redacts a stdout token shorter than eight characters when its first run fails', async () => {
    const leaked = 'Zq7xK2p';
    const mint = commandToken(
      await script(
        [
          `process.stdout.write(${JSON.stringify(`${leaked}\n`)});`,
          `process.stderr.write(${JSON.stringify(`verify failed: token ${leaked} rejected\n`)});`,
          'process.exit(1);',
        ].join('\n')
      )
    );
    const error = await rejection(mint({ refresh: false }));
    expect(error.message).toBe(
      'token command exited with code 1: verify failed: token [REDACTED] rejected'
    );
    expect(error.message).not.toContain(leaked);
  });

  it('redacts a short inline secret that the command quotes back', async () => {
    const base = await script(
      [
        'process.stderr.write("bad password " + process.argv[2].split("=")[1] + "\\n");',
        'process.exit(2);',
      ].join('\n')
    );
    const error = await rejection(
      commandToken(`${base} --pw=abc12`)({ refresh: false })
    );
    expect(error.message).toBe(
      'token command exited with code 2: bad password [REDACTED]'
    );
  });

  it('reports a forked command the shell saw killed by a signal as network, leaving a breaker closed', async () => {
    let requests = 0;
    const fetch: DecisionFetch = async () => {
      requests += 1;
      return new Response(answered, { status: 200 });
    };
    const model = withBreaker(
      createHttpDecisionModel({
        endpoint: 'https://decisions.example/v1/systemone',
        apiKey: commandToken('sh -c "kill -KILL \\$\\$" && echo token-1'),
        dialect: 'systemone',
        fetch,
        maxRetries: 0,
        timeoutMs: 10_000,
      })
    );

    const error = await rejection(model.decide(request));
    expect(error).toMatchObject({ failure: 'network', provider: 'command' });
    expect(error.message).toMatch(
      /^token command exited with code 137 \(killed by signal 9\)/
    );
    expect(model.breaker.open).toBe(false);
    expect(requests).toBe(0);

    const plain = await rejection(commandToken('exit 3')({ refresh: false }));
    expect(plain.failure).toBe('unauthorized');
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
      'token command exited with code 2: sh: -c: line 1: [REDACTED] unknown flag [REDACTED]=[REDACTED]'
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

  it('retries a stale 401 with the token another call already refreshed, without minting again', async () => {
    const auth: string[] = [];
    let refreshed!: () => void;
    const replaced = new Promise<void>((resolve) => {
      refreshed = resolve;
    });
    let staleSeen = 0;
    const fetch: DecisionFetch = async (_url, init) => {
      const header = init.headers.Authorization;
      auth.push(header);
      if (header === 'Bearer token-1') {
        staleSeen += 1;
        /** The second call's 401 lands only after the first has refreshed and succeeded. */
        if (staleSeen === 2) {
          await replaced;
        }
        return new Response('refused', { status: 401 });
      }
      const response = new Response(answered, { status: 200 });
      refreshed();
      return response;
    };
    const model = withBreaker(
      createHttpDecisionModel({
        endpoint: 'https://decisions.example/v1/systemone',
        apiKey: commandToken(
          await script(
            'if (n === 3) process.exit(1); process.stdout.write("token-" + n + "\\n");'
          )
        ),
        dialect: 'systemone',
        fetch,
        timeoutMs: 10_000,
      })
    );

    const outcomes = await Promise.allSettled([
      model.decide(request),
      model.decide(request),
    ]);
    expect(outcomes.map((outcome) => outcome.status)).toEqual([
      'fulfilled',
      'fulfilled',
    ]);
    expect(await runs()).toBe(2);
    expect(auth.filter((header) => header === 'Bearer token-2')).toHaveLength(
      2
    );
    expect(model.breaker.open).toBe(false);
  });

  it('rechecks a refreshed token refused late and sends the one another call minted since', async () => {
    const auth: string[] = [];
    let release!: () => void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    let holding!: () => void;
    const inFlight = new Promise<void>((resolve) => {
      holding = resolve;
    });
    let refreshedSent = 0;
    const fetch: DecisionFetch = async (_url, init) => {
      const header = init.headers.Authorization;
      auth.push(header);
      if (header === 'Bearer token-3') {
        release();
        return new Response(answered, { status: 200 });
      }
      /** The first call's request with its refreshed token-2 is answered only after the second call has replaced it. */
      if (header === 'Bearer token-2' && ++refreshedSent === 1) {
        holding();
        await held;
      }
      return new Response('refused', { status: 401 });
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

    const first = model.decide(request);
    await inFlight;
    const outcomes = await Promise.allSettled([model.decide(request), first]);
    expect(outcomes.map((outcome) => outcome.status)).toEqual([
      'fulfilled',
      'fulfilled',
    ]);
    expect(model.breaker.open).toBe(false);
    expect(auth).toEqual([
      'Bearer token-1',
      'Bearer token-2',
      'Bearer token-2',
      'Bearer token-3',
      'Bearer token-3',
    ]);
    expect(await runs()).toBe(3);
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

  it('drops the SIGKILL escalation once the killed group is gone, so it never holds the host open', async () => {
    const command = await script('setTimeout(() => {}, 30_000);');
    const out = await host(
      [
        'let rejectedAt = Date.now();',
        `commandToken(${JSON.stringify(command)}, { timeoutMs: 200, killGraceMs: 5_000 })({ refresh: false }).catch(() => { rejectedAt = Date.now(); });`,
        'process.on("exit", () => process.stdout.write(String(Date.now() - rejectedAt)));',
      ].join('\n')
    );
    expect(Number(out)).toBeLessThan(1_000);
  });

  it('clears the escalation timer of a killed run inside a vm context once its group is gone', async () => {
    const pidFile = path.join(dir, 'pid');
    const mint = commandToken(
      await script(
        [
          `fs.writeFileSync(${JSON.stringify(pidFile)}, String(process.pid));`,
          'setTimeout(() => {}, 30_000);',
        ].join('\n')
      ),
      { timeoutMs: 300, killGraceMs: 10_000 }
    );
    const baseline = activeTimeouts();
    expect(await rejection(mint({ refresh: false }))).toMatchObject({
      failure: 'timeout',
    });
    expect(await exited(await pidFrom(pidFile))).toBe(true);
    let left = activeTimeouts();
    for (let i = 0; i < 25 && left > baseline; i++) {
      await new Promise((resolve) => setTimeout(resolve, 20));
      left = activeTimeouts();
    }
    expect(left).toBeLessThanOrEqual(baseline);
  });

  it('kills a run still going when its host exits', async () => {
    const pidFile = path.join(dir, 'pid');
    const command = await script(
      [
        `fs.writeFileSync(${JSON.stringify(pidFile)}, String(process.pid));`,
        'setTimeout(() => process.stdout.write("token-1\\n"), 30_000);',
      ].join('\n')
    );
    await host(
      [
        'import { existsSync } from "node:fs";',
        `commandToken(${JSON.stringify(command)}, { timeoutMs: 60_000 })({ refresh: false }).catch(() => {});`,
        `const wait = (): void => { if (existsSync(${JSON.stringify(pidFile)})) process.exit(0); else setTimeout(wait, 20); };`,
        'wait();',
      ].join('\n')
    );
    const pid = await pidFrom(pidFile);
    try {
      expect(await exited(pid)).toBe(true);
    } finally {
      sweep(pid);
    }
  });

  it('settles when the shell exits, though a process it left behind still holds stdout', async () => {
    const pidFile = path.join(dir, 'left');
    const started = Date.now();
    const pids: number[] = [];
    try {
      const minted = commandToken(
        `sleep 3 & echo $! > "${pidFile}"; echo tok-1`,
        { timeoutMs: 1_500 }
      )({ refresh: false });
      await expect(minted).resolves.toBe('tok-1');
      expect(Date.now() - started).toBeLessThan(1_000);
      pids.push(await pidFrom(pidFile));
      /** The run ended on its own, so what it left behind is not killed. */
      expect(() => process.kill(pids[0], 0)).not.toThrow();

      await rm(pidFile);
      const refused = await rejection(
        commandToken(`sleep 3 & echo $! > "${pidFile}"; exit 3`, {
          timeoutMs: 1_500,
        })({ refresh: false })
      );
      pids.push(await pidFrom(pidFile));
      expect(refused).toMatchObject({
        failure: 'unauthorized',
        message: 'token command exited with code 3',
      });
    } finally {
      pids.forEach(sweep);
    }
  });

  it('reads an exit status outside refusalExitCodes as network, leaving a breaker closed', async () => {
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
            'if (n === 1) process.exit(7); process.stdout.write("token-" + n + "\\n");'
          ),
          { refusalExitCodes: [1, 2] }
        ),
        dialect: 'systemone',
        fetch,
        maxRetries: 0,
        timeoutMs: 10_000,
      })
    );

    const error = await rejection(model.decide(request));
    expect(error).toMatchObject({
      failure: 'network',
      message: 'token command exited with code 7',
    });
    expect(model.breaker.open).toBe(false);
    await expect(model.decide(request)).resolves.toMatchObject({
      model: 'jev-1.13.0',
    });
    expect(requests).toBe(1);

    const listed = await rejection(
      commandToken('exit 2', { refusalExitCodes: [1, 2] })({ refresh: false })
    );
    expect(listed.failure).toBe('unauthorized');
    for (const codes of [[0], [256], [1.5]]) {
      expect(() => commandToken('true', { refusalExitCodes: codes })).toThrow(
        'invalid token command refusal exit codes'
      );
    }
  });

  it('starts a fresh run for a call made after a breaker reset instead of joining one from before it', async () => {
    const flag = path.join(dir, 'logged-in');
    const breaker = createBreaker();
    const fetch: DecisionFetch = async () =>
      new Response(answered, { status: 200 });
    const model = withBreaker(
      createHttpDecisionModel({
        endpoint: 'https://decisions.example/v1/systemone',
        apiKey: commandToken(
          await script(
            [
              `if (fs.existsSync(${JSON.stringify(flag)})) process.stdout.write("token-" + n + "\\n");`,
              'else setTimeout(() => process.exit(1), 600);',
            ].join('\n')
          ),
          { breaker }
        ),
        dialect: 'systemone',
        fetch,
        timeoutMs: 10_000,
      }),
      breaker
    );

    expect(await rejection(model.decide(request))).toMatchObject({
      failure: 'unauthorized',
    });
    expect(breaker.open).toBe(true);
    breaker.reset();
    const inFlight = model.decide(request);
    while ((await runs()) < 2) {
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    /** The operator logs in while the run from before this reset is still failing. */
    await writeFile(flag, '');
    breaker.reset();
    const after = model.decide(request);

    expect(await rejection(inFlight)).toMatchObject({
      failure: 'unauthorized',
    });
    await expect(after).resolves.toMatchObject({ model: 'jev-1.13.0' });
    expect(breaker.open).toBe(false);
    expect(await runs()).toBe(3);
  });
  it('mints afresh after a reset instead of reusing a token cached before it, even one a 403 refused', async () => {
    const flag = path.join(dir, 'right-account');
    const breaker = createBreaker();
    const auth: Array<string | undefined> = [];
    const fetch: DecisionFetch = async (_url, init) => {
      auth.push(init.headers.Authorization);
      return init.headers.Authorization.startsWith('Bearer good-')
        ? new Response(answered, { status: 200 })
        : new Response('forbidden', { status: 403 });
    };
    const model = withBreaker(
      createHttpDecisionModel({
        endpoint: 'https://decisions.example/v1/systemone',
        apiKey: commandToken(
          await script(
            `process.stdout.write((fs.existsSync(${JSON.stringify(flag)}) ? "good-" : "bad-") + n + "\\n");`
          ),
          { breaker }
        ),
        dialect: 'systemone',
        fetch,
        timeoutMs: 10_000,
      }),
      breaker
    );

    expect(await rejection(model.decide(request))).toMatchObject({
      failure: 'unauthorized',
      status: 403,
    });
    expect(breaker.open).toBe(true);
    /** The operator logs in as the right account; a 403 never refreshes, so only the reset can retire `bad-1`. */
    await writeFile(flag, '');
    breaker.reset();

    await expect(model.decide(request)).resolves.toMatchObject({
      model: 'jev-1.13.0',
    });
    expect(auth).toEqual(['Bearer bad-1', 'Bearer good-2']);
    expect(breaker.open).toBe(false);
    expect(await runs()).toBe(2);
  });

  it('mints afresh after a reset that a run from before it outlived with no caller arriving meanwhile', async () => {
    const breaker = createBreaker();
    const mint = commandToken(
      await script(
        'setTimeout(() => process.stdout.write("token-" + n + "\\n"), 300);'
      ),
      { breaker }
    );

    const before = mint({ refresh: false });
    while ((await runs()) < 1) {
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    breaker.reset();
    await expect(before).resolves.toBe('token-1');

    await expect(mint({ refresh: false })).resolves.toBe('token-2');
    await expect(mint({ refresh: false })).resolves.toBe('token-2');
    expect(await runs()).toBe(2);
  });
});
