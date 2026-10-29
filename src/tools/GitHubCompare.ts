import { DynamicStructuredTool } from '@langchain/core/tools';
import { INTENT_PROPERTY } from '@/tools/intentArg';

export const GitHubCompareToolName = 'github_compare';
export const GitHubCompareToolDescription =
  'Read-only GitHub comparison of two full commit SHAs in a public repository. Returns the merge-base, relation and commit counts without Git, a worktree, or worker admission. Does not fetch code or execute commands. Private repositories and arbitrary URLs are not supported.';
export const GitHubCompareToolSchema = {
  type: 'object',
  properties: {
    intent: { ...INTENT_PROPERTY },
    owner: { type: 'string', description: 'GitHub repository owner.' },
    repo: { type: 'string', description: 'GitHub repository name.' },
    base: {
      type: 'string',
      description: 'Full lowercase base commit SHA, not a moving branch.',
    },
    head: {
      type: 'string',
      description: 'Full lowercase head commit SHA, not a moving branch.',
    },
  },
  required: ['owner', 'repo', 'base', 'head'],
} as const;
export const GitHubCompareToolDefinition = {
  name: GitHubCompareToolName,
  description: GitHubCompareToolDescription,
  parameters: GitHubCompareToolSchema,
} as const;

export interface GitHubComparisonInput {
  owner: string;
  repo: string;
  base: string;
  head: string;
}

export interface GitHubComparison {
  base: string;
  head: string;
  mergeBase: string;
  status: 'ahead' | 'behind' | 'identical' | 'diverged';
  aheadBy: number;
  behindBy: number;
  totalCommits: number;
}

export interface GitHubCompareOptions {
  /** Caller-owned transport, including proxy and instrumentation policy. */
  fetch: typeof fetch;
  /** Entire request budget, including body reads. Defaults to 10 seconds; maximum 30 seconds. */
  timeoutMs?: number;
}

export type GitHubComparisonErrorCode =
  | 'INVALID_INPUT'
  | 'INVALID_CONFIGURATION'
  | 'HTTP_ERROR'
  | 'INVALID_RESPONSE'
  | 'RESPONSE_TOO_LARGE'
  | 'TRANSPORT_ERROR';

export class GitHubComparisonError extends Error {
  constructor(
    public readonly code: GitHubComparisonErrorCode,
    message: string,
    public readonly status?: number
  ) {
    super(message);
    this.name = 'GitHubComparisonError';
  }
}

interface GitHubComparisonResponse {
  base_commit?: { sha?: string };
  merge_base_commit?: { sha?: string };
  status?: GitHubComparison['status'];
  ahead_by?: number;
  behind_by?: number;
  total_commits?: number;
}

const SHA = /^[a-f0-9]{40}$/;
const REPO = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$/;
const MAX_RESPONSE_BYTES = 2 * 1024 * 1024;

function awaitWithSignal<T>(
  operation: () => Promise<T>,
  signal: AbortSignal
): Promise<T> {
  signal.throwIfAborted();
  return new Promise<T>((resolve, reject) => {
    const onAbort = (): void => {
      signal.removeEventListener('abort', onAbort);
      reject(signal.reason);
    };
    signal.addEventListener('abort', onAbort, { once: true });
    try {
      operation().then(
        (value) => {
          signal.removeEventListener('abort', onAbort);
          resolve(value);
        },
        (error) => {
          signal.removeEventListener('abort', onAbort);
          reject(error);
        }
      );
    } catch (error) {
      signal.removeEventListener('abort', onAbort);
      reject(error);
    }
  });
}

function cancelBody(response: Response): void {
  void response.body?.cancel().catch(() => undefined);
}

async function readResponse(
  response: Response,
  signal: AbortSignal
): Promise<string> {
  const reader = response.body?.getReader();
  if (reader == null) {
    throw new GitHubComparisonError(
      'INVALID_RESPONSE',
      'GitHub comparison returned an invalid response.'
    );
  }
  const chunks: Uint8Array[] = [];
  let bytes = 0;
  try {
    for (;;) {
      const item = await awaitWithSignal(() => reader.read(), signal);
      if (item.done) {
        break;
      }
      bytes += item.value.byteLength;
      if (bytes > MAX_RESPONSE_BYTES) {
        throw new GitHubComparisonError(
          'RESPONSE_TOO_LARGE',
          'GitHub comparison response exceeded its bound.'
        );
      }
      chunks.push(item.value);
    }
    return Buffer.concat(chunks, bytes).toString('utf8');
  } finally {
    void reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

function parseComparison(
  text: string,
  input: GitHubComparisonInput
): GitHubComparison {
  let data: GitHubComparisonResponse | null;
  try {
    data = JSON.parse(text) as GitHubComparisonResponse | null;
  } catch {
    throw new GitHubComparisonError(
      'INVALID_RESPONSE',
      'GitHub comparison returned an invalid response.'
    );
  }
  const mergeBase = data?.merge_base_commit?.sha;
  const status = data?.status;
  const aheadBy = data?.ahead_by;
  const behindBy = data?.behind_by;
  const totalCommits = data?.total_commits;
  if (
    data?.base_commit?.sha !== input.base ||
    typeof mergeBase !== 'string' ||
    !SHA.test(mergeBase) ||
    (status !== 'ahead' &&
      status !== 'behind' &&
      status !== 'identical' &&
      status !== 'diverged') ||
    typeof aheadBy !== 'number' ||
    !Number.isSafeInteger(aheadBy) ||
    aheadBy < 0 ||
    typeof behindBy !== 'number' ||
    !Number.isSafeInteger(behindBy) ||
    behindBy < 0 ||
    typeof totalCommits !== 'number' ||
    !Number.isSafeInteger(totalCommits) ||
    totalCommits < 0
  ) {
    throw new GitHubComparisonError(
      'INVALID_RESPONSE',
      'GitHub comparison returned an invalid response.'
    );
  }
  return {
    base: input.base,
    head: input.head,
    mergeBase,
    status,
    aheadBy,
    behindBy,
    totalCommits,
  };
}

export async function compareGitHubCommits(
  input: GitHubComparisonInput,
  options: GitHubCompareOptions,
  signal?: AbortSignal
): Promise<GitHubComparison> {
  signal?.throwIfAborted();
  if (
    typeof input.owner !== 'string' ||
    !REPO.test(input.owner) ||
    typeof input.repo !== 'string' ||
    !REPO.test(input.repo) ||
    typeof input.base !== 'string' ||
    !SHA.test(input.base) ||
    typeof input.head !== 'string' ||
    !SHA.test(input.head)
  ) {
    throw new GitHubComparisonError(
      'INVALID_INPUT',
      'Supply a repository and full lowercase commit SHAs.'
    );
  }
  const timeoutMs = options.timeoutMs ?? 10000;
  if (
    typeof options.fetch !== 'function' ||
    !Number.isSafeInteger(timeoutMs) ||
    timeoutMs < 1 ||
    timeoutMs > 30000
  ) {
    throw new GitHubComparisonError(
      'INVALID_CONFIGURATION',
      'Supply a GitHub transport and a timeout between 1 and 30000 ms.'
    );
  }
  const timeout = new AbortController();
  const timer = setTimeout(
    () =>
      timeout.abort(
        new DOMException('GitHub comparison timed out.', 'TimeoutError')
      ),
    timeoutMs
  );
  timer.unref();
  let pendingResponse: Response | undefined;
  const requestSignal =
    signal == null ? timeout.signal : AbortSignal.any([signal, timeout.signal]);
  try {
    const response = await awaitWithSignal(
      () =>
        options
          .fetch(
            `https://api.github.com/repos/${encodeURIComponent(input.owner)}/${encodeURIComponent(input.repo)}/compare/${input.base}...${input.head}?per_page=1&page=2`,
            {
              method: 'GET',
              redirect: 'error',
              credentials: 'omit',
              signal: requestSignal,
              headers: {
                Accept: 'application/vnd.github+json',
                'X-GitHub-Api-Version': '2022-11-28',
              },
            }
          )
          .then((response) => {
            pendingResponse = response;
            if (requestSignal.aborted) {
              pendingResponse = undefined;
              cancelBody(response);
              requestSignal.throwIfAborted();
            }
            return response;
          }),
      requestSignal
    );
    if (!response.ok) {
      throw new GitHubComparisonError(
        'HTTP_ERROR',
        `GitHub comparison unavailable (HTTP ${response.status}).`,
        response.status
      );
    }
    pendingResponse = undefined;
    const text = await readResponse(response, requestSignal);
    requestSignal.throwIfAborted();
    return parseComparison(text, input);
  } catch (error) {
    requestSignal.throwIfAborted();
    if (error instanceof GitHubComparisonError) {
      throw error;
    }
    throw new GitHubComparisonError(
      'TRANSPORT_ERROR',
      'GitHub comparison transport failed.'
    );
  } finally {
    if (pendingResponse != null) {
      cancelBody(pendingResponse);
    }
    clearTimeout(timer);
  }
}

export class GitHubCompareTool extends DynamicStructuredTool {
  constructor(options: GitHubCompareOptions) {
    super({
      name: GitHubCompareToolName,
      description: GitHubCompareToolDescription,
      schema: structuredClone(GitHubCompareToolSchema),
      func: async (input: GitHubComparisonInput, _manager, config) =>
        JSON.stringify(
          await compareGitHubCommits(input, options, config?.signal)
        ),
    });
  }
}
