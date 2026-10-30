import { posix as path } from 'path';
import type * as t from '@/types';

const DEFAULT_API_PREFIX = '/v1';
const DEFAULT_WORKSPACE_ROOT = '/workspace';
const DEFAULT_SHELL = 'bash';

export type CloudflareBridgeRuntimeConfig = {
  /** Base URL of a Worker using `bridge()` from `@cloudflare/sandbox/bridge`. */
  baseURL: string;
  /** Bearer token stored in the bridge Worker as `SANDBOX_API_KEY`. */
  apiKey?: string;
  /** Existing sandbox id. If omitted, the adapter creates one lazily. */
  sandboxId?: string;
  /** Bridge API route prefix. Defaults to `/v1`. */
  apiRoutePrefix?: string;
  /** Workspace root used for path clamping. Defaults to `/workspace`. */
  workspaceRoot?: string;
  /** Optional bridge session id sent as `Session-Id`. */
  sessionId?: string;
  /** Shell used to run command strings over the bridge exec endpoint. */
  shell?: string;
  /** Optional fetch implementation. Defaults to global `fetch`. */
  fetch?: typeof fetch;
};

export type CloudflareBridgeRuntime = t.CloudflareSandboxRuntime & {
  getSandboxId(): Promise<string>;
};

type BridgeSSEEvent = {
  event: string;
  data: string;
};

function normalizeBaseURL(baseURL: string): string {
  return baseURL.replace(/\/+$/, '');
}

function normalizePrefix(prefix: string | undefined): string {
  const raw = prefix ?? DEFAULT_API_PREFIX;
  const withLeadingSlash = raw.startsWith('/') ? raw : `/${raw}`;
  return withLeadingSlash.replace(/\/+$/, '');
}

function normalizeWorkspaceRoot(workspaceRoot: string): string {
  const normalized = path.normalize(workspaceRoot);
  return normalized === '/' ? normalized : normalized.replace(/\/+$/, '');
}

function getFetch(config: CloudflareBridgeRuntimeConfig): typeof fetch {
  const fetchImpl = config.fetch ?? globalThis.fetch;
  return fetchImpl.bind(globalThis) as typeof fetch;
}

function quote(value: string): string {
  if (value === '') {
    return '\'\'';
  }
  if (/^[A-Za-z0-9_/:=.,@%+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replace(/'/g, '\'\\\'\'')}'`;
}

function encodeBridgePath(filePath: string): string {
  return filePath
    .replace(/^\/+/, '')
    .split('/')
    .map((part) => encodeURIComponent(part))
    .join('/');
}

function toSandboxPath(filePath: string, workspaceRoot: string): string {
  const raw = filePath === '' ? '.' : filePath;
  const root = normalizeWorkspaceRoot(workspaceRoot);
  const resolved = raw.startsWith('/')
    ? path.normalize(raw)
    : path.resolve(root, raw);
  if (root === '/') {
    return resolved;
  }
  if (resolved === root || resolved.startsWith(`${root}/`)) {
    return resolved;
  }
  throw new Error(
    `Path is outside the Cloudflare sandbox workspace: ${filePath}`
  );
}

function typeFromFind(value: string): t.CloudflareSandboxFileInfo['type'] {
  switch (value) {
  case 'd':
    return 'directory';
  case 'l':
    return 'symlink';
  case 'f':
    return 'file';
  default:
    return 'other';
  }
}

function decodeBase64(value: string): string {
  return Buffer.from(value, 'base64').toString('utf8');
}

async function readResponseText(response: Response): Promise<string> {
  try {
    return await response.text();
  } catch {
    return '';
  }
}

async function assertOk(response: Response, operation: string): Promise<void> {
  if (response.ok) {
    return;
  }
  const text = await readResponseText(response);
  throw new Error(
    `Cloudflare sandbox bridge ${operation} failed (${response.status}): ${text}`
  );
}

function createHeaders(
  config: CloudflareBridgeRuntimeConfig,
  extra?: HeadersInit
): Headers {
  const headers = new Headers(extra);
  if (config.apiKey != null && config.apiKey !== '') {
    headers.set('Authorization', `Bearer ${config.apiKey}`);
  }
  if (config.sessionId != null && config.sessionId !== '') {
    headers.set('Session-Id', config.sessionId);
  }
  return headers;
}

function envScript(env?: Record<string, string | undefined>): string {
  if (env == null) {
    return '';
  }
  return Object.entries(env)
    .filter((entry): entry is [string, string] => entry[1] != null)
    .map(([key, value]) => `export ${key}=${quote(value)}`)
    .join('\n');
}

function commandWithEnv(
  command: string,
  env?: Record<string, string | undefined>
): string {
  const exports = envScript(env);
  return exports === '' ? command : `${exports}\n${command}`;
}

async function collectReadableStream(
  stream: ReadableStream<Uint8Array>
): Promise<ReturnType<typeof Uint8Array.from>> {
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      total += value.byteLength;
    }
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function normalizeWriteBody(
  content: string | ReadableStream<Uint8Array>,
  options?: { encoding?: string }
): Promise<ReturnType<typeof Uint8Array.from>> {
  if (typeof content !== 'string') {
    return collectReadableStream(content);
  }
  if (options?.encoding === 'base64') {
    return Buffer.from(content, 'base64');
  }
  return Buffer.from(content, options?.encoding === 'utf8' ? 'utf8' : 'utf8');
}

function parseSSEChunk(buffer: string): {
  events: BridgeSSEEvent[];
  remainder: string;
} {
  const normalized = buffer.replace(/\r\n/g, '\n');
  const parts = normalized.split('\n\n');
  const remainder = parts.pop() ?? '';
  const events = parts
    .map((part) => {
      let event = 'message';
      const dataLines: string[] = [];
      for (const line of part.split('\n')) {
        if (line.startsWith('event:')) {
          event = line.slice('event:'.length).trim();
        } else if (line.startsWith('data:')) {
          dataLines.push(line.slice('data:'.length).trimStart());
        }
      }
      return { event, data: dataLines.join('\n') };
    })
    .filter((event) => event.data !== '' || event.event !== 'message');
  return { events, remainder };
}

function parseExitCode(data: string): number {
  try {
    const parsed = JSON.parse(data) as { exit_code?: number };
    return typeof parsed.exit_code === 'number' ? parsed.exit_code : 1;
  } catch {
    return 1;
  }
}

function parseBridgeError(data: string): string {
  try {
    const parsed = JSON.parse(data) as { error?: string };
    return parsed.error ?? data;
  } catch {
    return data;
  }
}

export function createCloudflareBridgeRuntime(
  config: CloudflareBridgeRuntimeConfig
): CloudflareBridgeRuntime {
  const baseURL = normalizeBaseURL(config.baseURL);
  const apiPrefix = normalizePrefix(config.apiRoutePrefix);
  const workspaceRoot = normalizeWorkspaceRoot(
    config.workspaceRoot ?? DEFAULT_WORKSPACE_ROOT
  );
  const shell = config.shell ?? DEFAULT_SHELL;
  const fetchImpl = getFetch(config);
  let sandboxIdPromise: Promise<string> | undefined =
    config.sandboxId != null ? Promise.resolve(config.sandboxId) : undefined;

  function bridgeURL(suffix: string): string {
    return `${baseURL}${apiPrefix}${suffix}`;
  }

  async function getSandboxId(): Promise<string> {
    if (sandboxIdPromise == null) {
      sandboxIdPromise = (async (): Promise<string> => {
        const response = await fetchImpl(bridgeURL('/sandbox'), {
          method: 'POST',
          headers: createHeaders(config),
        });
        await assertOk(response, 'sandbox create');
        const payload = (await response.json()) as { id?: string };
        if (typeof payload.id !== 'string' || payload.id === '') {
          throw new Error(
            'Cloudflare sandbox bridge create did not return an id.'
          );
        }
        return payload.id;
      })().catch((error: unknown) => {
        sandboxIdPromise = undefined;
        throw error;
      });
    }
    return sandboxIdPromise;
  }

  async function exec(
    command: string,
    options: t.CloudflareSandboxExecOptions = {}
  ): Promise<t.CloudflareSandboxExecResult> {
    const sandboxId = await getSandboxId();
    const sandboxPathId = encodeURIComponent(sandboxId);
    const response = await fetchImpl(
      bridgeURL(`/sandbox/${sandboxPathId}/exec`),
      {
        method: 'POST',
        headers: createHeaders(config, {
          'Content-Type': 'application/json',
        }),
        body: JSON.stringify({
          argv: [shell, '-lc', commandWithEnv(command, options.env)],
          cwd: toSandboxPath(options.cwd ?? workspaceRoot, workspaceRoot),
          timeout_ms: options.timeout,
        }),
        signal: options.signal,
      }
    );
    await assertOk(response, 'exec');
    if (response.body == null) {
      throw new Error(
        'Cloudflare sandbox bridge exec response did not include a body.'
      );
    }

    const decoder = new TextDecoder();
    const reader = response.body.getReader();
    let buffer = '';
    let stdout = '';
    let stderr = '';
    let exitCode: number | undefined;

    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parsed = parseSSEChunk(buffer);
        buffer = parsed.remainder;
        for (const event of parsed.events) {
          if (event.event === 'stdout') {
            const decoded = decodeBase64(event.data);
            stdout += decoded;
            options.onOutput?.('stdout', decoded);
          } else if (event.event === 'stderr') {
            const decoded = decodeBase64(event.data);
            stderr += decoded;
            options.onOutput?.('stderr', decoded);
          } else if (event.event === 'exit') {
            exitCode = parseExitCode(event.data);
          } else if (event.event === 'error') {
            throw new Error(parseBridgeError(event.data));
          }
        }
      }
      buffer += decoder.decode();
      const parsed = parseSSEChunk(buffer);
      buffer = parsed.remainder;
      for (const event of parsed.events) {
        if (event.event === 'stdout') {
          const decoded = decodeBase64(event.data);
          stdout += decoded;
          options.onOutput?.('stdout', decoded);
        } else if (event.event === 'stderr') {
          const decoded = decodeBase64(event.data);
          stderr += decoded;
          options.onOutput?.('stderr', decoded);
        } else if (event.event === 'exit') {
          exitCode = parseExitCode(event.data);
        } else if (event.event === 'error') {
          throw new Error(parseBridgeError(event.data));
        }
      }
    } finally {
      reader.releaseLock();
    }

    if (exitCode == null) {
      throw new Error(
        'Cloudflare sandbox bridge exec stream closed before an exit event.'
      );
    }

    return {
      success: exitCode === 0,
      exitCode,
      stdout,
      stderr,
      command,
    };
  }

  async function readFile(filePath: string): Promise<Buffer> {
    const sandboxId = await getSandboxId();
    const sandboxPathId = encodeURIComponent(sandboxId);
    const resolvedPath = toSandboxPath(filePath, workspaceRoot);
    const response = await fetchImpl(
      bridgeURL(
        `/sandbox/${sandboxPathId}/file/${encodeBridgePath(resolvedPath)}`
      ),
      {
        headers: createHeaders(config),
      }
    );
    await assertOk(response, 'readFile');
    return Buffer.from(await response.arrayBuffer());
  }

  async function writeFile(
    filePath: string,
    content: string | ReadableStream<Uint8Array>,
    options?: { encoding?: string }
  ): Promise<unknown> {
    const sandboxId = await getSandboxId();
    const sandboxPathId = encodeURIComponent(sandboxId);
    const resolvedPath = toSandboxPath(filePath, workspaceRoot);
    const body = await normalizeWriteBody(content, options);
    const response = await fetchImpl(
      bridgeURL(
        `/sandbox/${sandboxPathId}/file/${encodeBridgePath(resolvedPath)}`
      ),
      {
        method: 'PUT',
        headers: createHeaders(config, {
          'Content-Type': 'application/octet-stream',
        }),
        body,
      }
    );
    await assertOk(response, 'writeFile');
    return response.json().catch(() => ({ ok: true }));
  }

  async function mkdir(
    filePath: string,
    options?: { recursive?: boolean }
  ): Promise<unknown> {
    const resolvedPath = toSandboxPath(filePath, workspaceRoot);
    const command = `${options?.recursive === true ? 'mkdir -p' : 'mkdir'} -- ${quote(resolvedPath)}`;
    const result = await exec(command, { cwd: workspaceRoot });
    if (result.exitCode !== 0) {
      throw new Error(result.stderr || `mkdir failed for ${resolvedPath}`);
    }
    return { ok: true };
  }

  async function listFiles(
    filePath: string,
    options?: t.CloudflareSandboxListFilesOptions
  ): Promise<t.CloudflareSandboxFileInfo[]> {
    const resolvedPath = toSandboxPath(filePath, workspaceRoot);
    const maxDepth = options?.recursive === true ? '' : '-maxdepth 1';
    const hiddenFilter =
      options?.includeHidden === true ? '' : ' \\( -name \'.*\' -prune \\) -o';
    const quotedPath = quote(resolvedPath);
    const command =
      `[ -d ${quotedPath} ] || { printf '%s is not a directory\\n' ${quotedPath} >&2; exit 20; }; ` +
      `find ${quotedPath} -mindepth 1 ${maxDepth}${hiddenFilter} ` +
      '-printf \'%y\\t%s\\t%p\\n\'';
    const result = await exec(command, { cwd: workspaceRoot });
    if (result.exitCode !== 0) {
      throw new Error(result.stderr || `listFiles failed for ${resolvedPath}`);
    }
    return result.stdout
      .split('\n')
      .filter(Boolean)
      .map((line) => {
        const [rawType, rawSize, ...pathParts] = line.split('\t');
        const absolutePath = pathParts.join('\t');
        return {
          name: path.basename(absolutePath),
          absolutePath,
          relativePath: path.relative(resolvedPath, absolutePath),
          type: typeFromFind(rawType),
          size: Number.parseInt(rawSize, 10) || 0,
        };
      });
  }

  async function deleteFile(filePath: string): Promise<unknown> {
    const resolvedPath = toSandboxPath(filePath, workspaceRoot);
    const result = await exec(`rm -rf -- ${quote(resolvedPath)}`, {
      cwd: workspaceRoot,
    });
    if (result.exitCode !== 0) {
      throw new Error(result.stderr || `deleteFile failed for ${resolvedPath}`);
    }
    return { ok: true };
  }

  return {
    supportsExecSignal: true,
    getSandboxId,
    exec,
    readFile,
    writeFile,
    mkdir,
    listFiles,
    deleteFile,
  };
}
