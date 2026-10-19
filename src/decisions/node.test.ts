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
    const mint = commandToken(
      await script('process.stdout.write("token-" + n + "\\n");'),
      { ttlMs: 40 }
    );
    await expect(mint({ refresh: false })).resolves.toBe('token-1');
    await expect(mint({ refresh: true })).resolves.toBe('token-2');
    await expect(mint({ refresh: false })).resolves.toBe('token-2');
    await new Promise((resolve) => setTimeout(resolve, 60));
    await expect(mint({ refresh: false })).resolves.toBe('token-3');
    expect(await runs()).toBe(3);
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

  it('rejects an empty command and invalid timing at construction', () => {
    expect(() => commandToken('  ')).toThrow(DecisionError);
    expect(() => commandToken('true', { ttlMs: -1 })).toThrow(
      'invalid token command timing'
    );
    expect(() => commandToken('true', { timeoutMs: 0 })).toThrow(
      'invalid token command timing'
    );
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
