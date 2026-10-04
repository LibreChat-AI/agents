import type { GitHubComparisonInput } from '@/tools/GitHubCompare';
import {
  GitHubCompareTool,
  GitHubCompareToolName,
  GitHubCompareToolSchema,
  GitHubCompareToolDefinition,
  compareGitHubCommits,
} from '@/tools/GitHubCompare';

const base = 'a'.repeat(40);
const head = 'b'.repeat(40);
const mergeBase = 'c'.repeat(40);
const input: GitHubComparisonInput = {
  owner: 'LibreChat-AI',
  repo: 'agents',
  base,
  head,
};
const metadata = {
  base_commit: { sha: base },
  merge_base_commit: { sha: mergeBase },
  status: 'diverged',
  ahead_by: 2,
  behind_by: 3,
  total_commits: 2,
};
const comparison = {
  base,
  head,
  mergeBase,
  status: 'diverged',
  aheadBy: 2,
  behindBy: 3,
  totalCommits: 2,
};
const response = (data: object = metadata): Response =>
  new Response(JSON.stringify(data));
const transport = (): jest.Mock<Promise<Response>, Parameters<typeof fetch>> =>
  jest.fn(async () => response());

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

afterEach(() => {
  jest.useRealTimers();
});

test('returns frozen ancestry through one metadata-only, fixed-host GET', async () => {
  const fetch = transport();
  expect(await compareGitHubCommits(input, { fetch })).toEqual(comparison);
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(fetch.mock.calls[0][0]).toBe(
    `https://api.github.com/repos/LibreChat-AI/agents/compare/${base}...${head}?per_page=1&page=2`
  );
  expect(fetch.mock.calls[0][1]).toMatchObject({
    method: 'GET',
    redirect: 'error',
    credentials: 'omit',
    headers: {
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
    },
  });
  expect(fetch.mock.calls[0][1]?.headers).not.toHaveProperty('Authorization');
  expect(fetch.mock.calls[0][1]?.signal).toBeInstanceOf(AbortSignal);
});

test.each([
  '../private',
  'https://evil.test',
  'repo?token=x',
  '--base',
  'user@host',
  'repo/name',
  '',
  '.git',
])(
  'rejects caller-controlled repository routes (%s) without I/O',
  async (repo) => {
    const fetch = transport();
    await expect(
      compareGitHubCommits({ ...input, repo }, { fetch })
    ).rejects.toMatchObject({ code: 'INVALID_INPUT' });
    await expect(
      compareGitHubCommits({ ...input, owner: repo }, { fetch })
    ).rejects.toMatchObject({ code: 'INVALID_INPUT' });
    expect(fetch).not.toHaveBeenCalled();
  }
);

test.each([
  'HEAD',
  'main',
  'a'.repeat(39),
  'A'.repeat(40),
  'g'.repeat(40),
  `${base}...${head}`,
])('rejects moving or malformed commit references (%s)', async (sha) => {
  const fetch = transport();
  await expect(
    compareGitHubCommits({ ...input, base: sha }, { fetch })
  ).rejects.toMatchObject({ code: 'INVALID_INPUT' });
  await expect(
    compareGitHubCommits({ ...input, head: sha }, { fetch })
  ).rejects.toMatchObject({ code: 'INVALID_INPUT' });
  expect(fetch).not.toHaveBeenCalled();
});

test.each([0, -1, 1.5, 30001, Infinity, NaN])(
  'rejects invalid timeout %s before I/O',
  async (timeoutMs) => {
    const fetch = transport();
    await expect(
      compareGitHubCommits(input, { fetch, timeoutMs })
    ).rejects.toMatchObject({ code: 'INVALID_CONFIGURATION' });
    expect(fetch).not.toHaveBeenCalled();
  }
);

test.each(['ahead', 'behind', 'identical', 'diverged'] as const)(
  'accepts the GitHub relation %s',
  async (status) => {
    const fetch = jest.fn(async () => response({ ...metadata, status }));
    expect(await compareGitHubCommits(input, { fetch })).toMatchObject({
      status,
    });
  }
);

test.each([
  {},
  { ...metadata, base_commit: { sha: head } },
  { ...metadata, merge_base_commit: null },
  { ...metadata, merge_base_commit: { sha: 'main' } },
  { ...metadata, status: 'invalid' },
  { ...metadata, ahead_by: '2' },
  { ...metadata, behind_by: -1 },
  { ...metadata, total_commits: 0.5 },
  { ...metadata, total_commits: Number.MAX_SAFE_INTEGER + 1 },
])('rejects malformed response metadata %#', async (data) => {
  const fetch = jest.fn(async () => response(data));
  await expect(compareGitHubCommits(input, { fetch })).rejects.toMatchObject({
    code: 'INVALID_RESPONSE',
  });
});

test.each(['null', '[]', '1', '"text"', '{secret'])(
  'rejects malformed JSON or primitive response %s safely',
  async (body) => {
    const fetch = jest.fn(async () => new Response(body));
    await expect(compareGitHubCommits(input, { fetch })).rejects.toMatchObject({
      code: 'INVALID_RESPONSE',
      message: 'GitHub comparison returned an invalid response.',
    });
  }
);

test.each([301, 403, 404, 429, 500])(
  'rejects HTTP %s without exposing provider bodies',
  async (status) => {
    const cancel = jest.fn();
    const body = new ReadableStream<Uint8Array>({ cancel });
    const fetch = jest.fn(async () => new Response(body, { status }));
    await expect(compareGitHubCommits(input, { fetch })).rejects.toMatchObject({
      code: 'HTTP_ERROR',
      status,
      message: `GitHub comparison unavailable (HTTP ${status}).`,
    });
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledTimes(1);
  }
);

test('discard failure does not mask an HTTP rejection', async () => {
  const fetch = jest.fn(
    async () =>
      new Response(
        new ReadableStream<Uint8Array>({
          cancel: () =>
            Promise.reject(new Error('secret body cleanup diagnostic')),
        }),
        { status: 403 }
      )
  );
  await expect(compareGitHubCommits(input, { fetch })).rejects.toMatchObject({
    code: 'HTTP_ERROR',
    status: 403,
  });
});

test('propagates safe operational failure rather than raw transport diagnostics', async () => {
  const fetch = jest.fn(async (): Promise<Response> => {
    throw new Error('secret transport diagnostic');
  });
  await expect(compareGitHubCommits(input, { fetch })).rejects.toMatchObject({
    code: 'TRANSPORT_ERROR',
    message: 'GitHub comparison transport failed.',
  });
});

test('handles synchronous transport failure safely', async () => {
  const fetch = jest.fn((): Promise<Response> => {
    throw new Error('secret synchronous diagnostic');
  });
  await expect(compareGitHubCommits(input, { fetch })).rejects.toMatchObject({
    code: 'TRANSPORT_ERROR',
  });
});

test('reads chunked metadata and cancels the response on completion', async () => {
  const encoded = new TextEncoder().encode(JSON.stringify(metadata));
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoded.slice(0, 11));
      controller.enqueue(encoded.slice(11));
      controller.close();
    },
  });
  const fetch = jest.fn(async () => new Response(body));
  expect(await compareGitHubCommits(input, { fetch })).toEqual(comparison);
  expect(body.locked).toBe(false);
});

test('cancels oversized streamed responses without draining', async () => {
  const cancel = jest.fn();
  let pulls = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      pulls++;
      controller.enqueue(new Uint8Array(1024 * 1024));
    },
    cancel,
  });
  const fetch = jest.fn(async () => new Response(body));
  await expect(compareGitHubCommits(input, { fetch })).rejects.toMatchObject({
    code: 'RESPONSE_TOO_LARGE',
  });
  expect(cancel).toHaveBeenCalledTimes(1);
  expect(body.locked).toBe(false);
  expect(pulls).toBeLessThanOrEqual(4);
});

test('treats body transport failure as a safe operational error', async () => {
  const body = new ReadableStream<Uint8Array>({
    pull() {
      throw new Error('secret body failure');
    },
  });
  const fetch = jest.fn(async () => new Response(body));
  await expect(compareGitHubCommits(input, { fetch })).rejects.toMatchObject({
    code: 'TRANSPORT_ERROR',
  });
  expect(body.locked).toBe(false);
});

test('rejects missing response bodies', async () => {
  const fetch = jest.fn(async () => new Response(null));
  await expect(compareGitHubCommits(input, { fetch })).rejects.toMatchObject({
    code: 'INVALID_RESPONSE',
  });
});

test('pre-aborted invocation does not dispatch', async () => {
  const abort = new AbortController();
  const reason = new DOMException('Stopped.', 'AbortError');
  abort.abort(reason);
  const fetch = transport();
  await expect(
    compareGitHubCommits(input, { fetch }, abort.signal)
  ).rejects.toBe(reason);
  expect(fetch).not.toHaveBeenCalled();
});

test('cancels an uncooperative transport and discards its late response', async () => {
  const abort = new AbortController();
  const pending = deferred<Response>();
  const fetch = jest.fn(() => pending.promise);
  const result = compareGitHubCommits(input, { fetch }, abort.signal);
  const reason = new DOMException('Stopped.', 'AbortError');
  abort.abort(reason);
  await expect(result).rejects.toBe(reason);
  const cancel = jest.fn();
  pending.resolve(new Response(new ReadableStream<Uint8Array>({ cancel })));
  await pending.promise;
  expect(cancel).toHaveBeenCalledTimes(1);
});

test('cancels a stalled body read and releases the reader', async () => {
  const abort = new AbortController();
  const reading = deferred<void>();
  const cancel = jest.fn();
  const body = new ReadableStream<Uint8Array>({
    pull() {
      reading.resolve();
    },
    cancel,
  });
  const fetch = jest.fn(async () => new Response(body));
  const result = compareGitHubCommits(input, { fetch }, abort.signal);
  await reading.promise;
  const reason = new DOMException('Stopped.', 'AbortError');
  abort.abort(reason);
  await expect(result).rejects.toBe(reason);
  expect(cancel).toHaveBeenCalledTimes(1);
  expect(body.locked).toBe(false);
});

test('bounds a stalled transport by the request timeout', async () => {
  jest.useFakeTimers();
  const fetch = jest.fn(() => new Promise<Response>(() => undefined));
  const result = compareGitHubCommits(input, { fetch, timeoutMs: 20 });
  const rejected = expect(result).rejects.toMatchObject({
    name: 'TimeoutError',
  });
  await jest.advanceTimersByTimeAsync(20);
  await rejected;
  expect(fetch.mock.calls).toHaveLength(1);
  expect(jest.getTimerCount()).toBe(0);
});

test('includes stalled body consumption in the same deadline', async () => {
  jest.useFakeTimers();
  const cancel = jest.fn();
  const body = new ReadableStream<Uint8Array>({ cancel });
  const fetch = jest.fn(async () => new Response(body));
  const result = compareGitHubCommits(input, { fetch, timeoutMs: 20 });
  const rejected = expect(result).rejects.toMatchObject({
    name: 'TimeoutError',
  });
  await jest.advanceTimersByTimeAsync(20);
  await rejected;
  expect(cancel).toHaveBeenCalledTimes(1);
  expect(body.locked).toBe(false);
  expect(jest.getTimerCount()).toBe(0);
});

test('clears the deadline after completion', async () => {
  jest.useFakeTimers();
  expect(await compareGitHubCommits(input, { fetch: transport() })).toEqual(
    comparison
  );
  expect(jest.getTimerCount()).toBe(0);
});

test('constructs a callable tool with the injected transport and isolated schema', async () => {
  const fetch = transport();
  const first = new GitHubCompareTool({ fetch });
  const second = new GitHubCompareTool({ fetch });
  expect(first.name).toBe(GitHubCompareToolName);
  expect(GitHubCompareToolDefinition.parameters).toBe(GitHubCompareToolSchema);
  expect(first.schema).not.toBe(second.schema);
  expect(first.schema).not.toBe(GitHubCompareToolSchema);
  expect(Object.keys(GitHubCompareToolSchema.properties)[0]).toBe('intent');
  expect(await first.invoke(input)).toBe(JSON.stringify(comparison));
  expect(fetch).toHaveBeenCalledTimes(1);
});

test('callable tool rejects HTTP failure instead of returning a success string', async () => {
  const fetch = jest.fn(
    async () => new Response('secret provider body', { status: 429 })
  );
  const tool = new GitHubCompareTool({ fetch });
  await expect(tool.invoke(input)).rejects.toMatchObject({
    code: 'HTTP_ERROR',
    status: 429,
  });
});

test('callable tool propagates the executing run cancellation signal', async () => {
  const pending = deferred<Response>();
  const invoked = deferred<void>();
  const fetch = jest.fn(() => {
    invoked.resolve();
    return pending.promise;
  });
  const abort = new AbortController();
  const tool = new GitHubCompareTool({ fetch });
  const result = tool.invoke(input, { signal: abort.signal });
  await invoked.promise;
  abort.abort();
  await expect(result).rejects.toMatchObject({ name: 'AbortError' });
});
