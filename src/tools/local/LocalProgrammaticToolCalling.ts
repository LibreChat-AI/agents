import { createServer } from 'http';
import type {
  DynamicStructuredTool,
  ToolRunnableConfig,
} from '@langchain/core/tools';
import type { RunnableConfig } from '@langchain/core/runnables';
import { randomBytes, randomUUID, timingSafeEqual } from 'crypto';
import type { ProgrammaticInvocationParams } from '@/tools/ProgrammaticCallerPolicy';
import type { IncomingMessage, ServerResponse } from 'http';
import type { AddressInfo } from 'net';
import type * as t from '@/types';
import {
  executeTools,
  normalizeToPythonIdentifier,
  ProgrammaticToolCallingName,
  ProgrammaticToolCallingSchema,
} from '@/tools/ProgrammaticToolCalling';
import {
  BashProgrammaticToolCallingSchema,
  normalizeToBashIdentifier,
} from '@/tools/BashProgrammaticToolCalling';
import {
  resolveProgrammaticToolDefinitions,
  selectProgrammaticTools,
} from '@/tools/ProgrammaticCallerPolicy';
import {
  executeLocalBash,
  executeLocalCode,
  getLocalSessionId,
  getLocalCwd,
  shellQuote,
} from './LocalExecutionEngine';
import { executeHooks } from '@/hooks';
import { Constants } from '@/common';
import { createAwaitedTool } from '../awaitedCancellation';
import { capLocalOutput, formatLocalOutput } from './LocalExecutionTools';

const DEFAULT_TIMEOUT = 60000;
const LOCAL_MIN_TIMEOUT = 1000;
const LOCAL_MAX_TIMEOUT = 300000;

const LOCAL_WORKSPACE_GUIDANCE =
  'Execution starts in the configured working directory. Save task outputs with relative paths there. ' +
  'Project files persist between calls, including files written before a later failure. ' +
  'Each call starts a fresh process; variables and imports do not persist. ' +
  'Execution scratch is temporary. Saved project files are accessed through workspace tools; no artifact attachments are registered. ' +
  'Completed tool effects remain after a later failure, so check saved work before retrying. ' +
  'Print concise results for the model.';

const LOCAL_BASH_GUIDANCE =
  'Tools are predefined Bash functions accepting one JSON string argument. ' +
  'Capture output with raw=$(tool_name \'{"key":"value"}\'); printf \'%s\\n\' "$raw" > result.json. ' +
  'Tool output can be JSON or text; parse JSON only when its shape requires it. ' +
  'Independent calls can run with & followed by wait. Hyphens in tool names become underscores; reserved words get a _tool suffix.';

const LOCAL_PYTHON_GUIDANCE =
  'For Python, set lang to py or python. Code is wrapped in an async function; use await without asyncio.run(). ' +
  'Tools are predefined async functions. Call them with keyword arguments, for example await tool_name(key="value"). ' +
  'Results are Python values (dict, list or str); decode a JSON string only when needed. ' +
  'Use asyncio.gather for independent calls. Hyphens in tool names become underscores; keywords get a _tool suffix.';

const LOCAL_PROGRAMMATIC_DESCRIPTION = [
  'Compose registered tools in local orchestration code. Bash is the default runtime.',
  LOCAL_WORKSPACE_GUIDANCE,
  LOCAL_BASH_GUIDANCE,
  LOCAL_PYTHON_GUIDANCE,
].join('\n\n');

const LOCAL_BASH_PROGRAMMATIC_DESCRIPTION = [
  'Compose registered tools in local Bash orchestration code.',
  LOCAL_WORKSPACE_GUIDANCE,
  LOCAL_BASH_GUIDANCE,
].join('\n\n');

type LocalTimeoutSchema = {
  type: 'integer';
  minimum: number;
  maximum: number;
  default: number;
  description: string;
};

type LocalProgrammaticToolCallingJsonSchema = {
  type: 'object';
  properties: typeof ProgrammaticToolCallingSchema.properties & {
    timeout: LocalTimeoutSchema;
    lang: {
      type: 'string';
      enum: readonly ['py', 'python', 'bash', 'sh'];
      default: 'bash';
      description: string;
    };
  };
  required: readonly ['code'];
};

type LocalBashProgrammaticToolCallingJsonSchema = {
  type: 'object';
  properties: typeof BashProgrammaticToolCallingSchema.properties & {
    timeout: LocalTimeoutSchema;
  };
  required: readonly ['code'];
};

function normalizeLocalTimeout(timeoutMs: number | undefined): number {
  if (timeoutMs == null || !Number.isFinite(timeoutMs)) {
    return DEFAULT_TIMEOUT;
  }

  return Math.max(LOCAL_MIN_TIMEOUT, Math.floor(timeoutMs));
}

function formatLocalTimeout(timeoutMs: number): string {
  return timeoutMs % 1000 === 0
    ? `${timeoutMs / 1000} seconds`
    : `${timeoutMs} milliseconds`;
}

function createLocalTimeoutSchema(timeoutMs?: number): LocalTimeoutSchema {
  const defaultTimeout = normalizeLocalTimeout(timeoutMs);
  const maxTimeout = Math.max(LOCAL_MAX_TIMEOUT, defaultTimeout);
  const formattedDefault = formatLocalTimeout(defaultTimeout);
  const formattedMax = formatLocalTimeout(maxTimeout);

  return {
    type: 'integer',
    minimum: LOCAL_MIN_TIMEOUT,
    maximum: maxTimeout,
    default: defaultTimeout,
    description:
      'Maximum local execution time in milliseconds. ' +
      `Default: ${formattedDefault}. Max: ${formattedMax}.`,
  };
}

function createLocalProgrammaticToolCallingSchema(
  localConfig: t.LocalExecutionConfig = {}
): LocalProgrammaticToolCallingJsonSchema {
  return {
    ...ProgrammaticToolCallingSchema,
    properties: {
      ...ProgrammaticToolCallingSchema.properties,
      code: {
        ...ProgrammaticToolCallingSchema.properties.code,
        description: LOCAL_PROGRAMMATIC_DESCRIPTION,
      },
      timeout: createLocalTimeoutSchema(localConfig.timeoutMs),
      lang: {
        type: 'string',
        enum: ['py', 'python', 'bash', 'sh'],
        default: 'bash',
        description:
          'Local engine runtime for orchestration code. Defaults to bash; use py/python for Python orchestration.',
      },
    },
  } as const;
}

function createLocalBashProgrammaticToolCallingSchema(
  localConfig: t.LocalExecutionConfig = {}
): LocalBashProgrammaticToolCallingJsonSchema {
  return {
    ...BashProgrammaticToolCallingSchema,
    properties: {
      ...BashProgrammaticToolCallingSchema.properties,
      code: {
        ...BashProgrammaticToolCallingSchema.properties.code,
        description: LOCAL_BASH_PROGRAMMATIC_DESCRIPTION,
      },
      timeout: createLocalTimeoutSchema(localConfig.timeoutMs),
    },
  } as const;
}

type ToolBridge = {
  url: string;
  token: string;
  close: () => Promise<void>;
};

type ToolRequest = {
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
};

const BRIDGE_AUTH_HEADER = 'x-librechat-bridge-token';

function constantTimeEquals(a: string, b: string): boolean {
  const aBuf = Buffer.from(a, 'utf8');
  const bBuf = Buffer.from(b, 'utf8');
  if (aBuf.length !== bBuf.length) {
    return false;
  }
  return timingSafeEqual(aBuf, bBuf);
}

type LocalProgrammaticRuntime = 'python' | 'bash';

type LocalProgrammaticParams = ProgrammaticInvocationParams;

function resolveRuntime(
  params: LocalProgrammaticParams
): LocalProgrammaticRuntime {
  const rawRuntime = params.lang ?? params.runtime ?? params.language ?? 'bash';
  return rawRuntime === 'py' || rawRuntime === 'python' ? 'python' : 'bash';
}

function toSerializable(value: unknown): unknown {
  if (value === undefined) {
    return null;
  }
  return value;
}

async function readRequestBody(req: IncomingMessage): Promise<ToolRequest> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  const raw = Buffer.concat(chunks).toString('utf8');
  if (raw === '') {
    return {};
  }
  return JSON.parse(raw) as ToolRequest;
}

function writeJson(res: ServerResponse, status: number, value: unknown): void {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(value));
}

/**
 * Run the host's `PreToolUse` hook chain for a single bridge call.
 * Returns the (possibly rewritten) input and a `denyReason` if any
 * matcher returned `decision: 'deny'` or `'ask'`. `'ask'` collapses
 * to deny because the bridge can't raise a LangGraph interrupt from
 * inside an HTTP handler; fail-closed matches the rest of the SDK
 * when HITL is unavailable.
 *
 * @internal Exported for tests so the deny / allow / updatedInput /
 *   ask branches can be exercised without standing up the full HTTP
 *   bridge.
 */
export async function applyPreToolUseHooksForBridge(
  hookContext: t.ProgrammaticHookContext,
  toolName: string,
  toolUseId: string,
  toolInput: Record<string, unknown>,
  signal?: AbortSignal
): Promise<{ input: Record<string, unknown>; denyReason?: string }> {
  if (hookContext.registry == null) {
    return { input: toolInput };
  }
  const result = await executeHooks({
    registry: hookContext.registry,
    input: {
      hook_event_name: 'PreToolUse',
      runId: hookContext.runId,
      threadId: hookContext.threadId,
      agentId: hookContext.agentId,
      executionContext: hookContext.executionContext,
      executingAgentId: hookContext.executingAgentId,
      toolName,
      toolInput,
      toolUseId,
      stepId: '',
      turn: 0,
    },
    sessionId: hookContext.runId,
    matchQuery: toolName,
    signal,
  }).catch(() => undefined);
  if (
    hookContext.failClosedOnHookError === true &&
    (result == null || result.hasHookFailures === true)
  ) {
    return {
      input: toolInput,
      denyReason: `Approval policy could not be evaluated for "${toolName}" in a background subagent. Ask the parent to run it in the foreground.`,
    };
  }
  if (result == null) {
    return { input: toolInput };
  }
  const nextInput =
    result.updatedInput != null
      ? (result.updatedInput as Record<string, unknown>)
      : toolInput;
  if (result.decision === 'deny' || result.decision === 'ask') {
    return {
      input: nextInput,
      denyReason:
        result.decision === 'ask' && hookContext.failClosedOnHookError === true
          ? `Approval required for "${toolName}"; unavailable in a background subagent. Ask the parent to run it in the foreground.${result.reason == null ? '' : ` Reason: ${result.reason}`}`
          : (result.reason ??
            (result.decision === 'ask'
              ? `Tool "${toolName}" requires human approval; bridge cannot raise an interrupt, denying.`
              : `Tool "${toolName}" denied by PreToolUse hook.`)),
    };
  }
  return { input: nextInput };
}

async function executeBridgeCall(
  call: { id: string; name: string; input: NonNullable<ToolRequest['input']> },
  toolMap: t.ToolMap,
  signal: AbortSignal,
  programmaticToolName: string,
  hookContext?: t.ProgrammaticHookContext,
  config?: RunnableConfig
): Promise<t.PTCToolResult> {
  signal.throwIfAborted();
  const hookInput = {
    runId: hookContext?.runId ?? '',
    threadId: hookContext?.threadId,
    agentId: hookContext?.agentId,
    executionContext: hookContext?.executionContext,
    executingAgentId: hookContext?.executingAgentId,
    toolName: call.name,
    toolInput: call.input,
    toolUseId: call.id,
    stepId: '',
    turn: 0,
  };
  const hookOptions =
    hookContext?.registry == null
      ? undefined
      : {
        registry: hookContext.registry,
        sessionId: hookContext.runId,
        matchQuery: call.name,
      };
  let result: t.PTCToolResult;
  try {
    if (hookContext != null) {
      const gate = await applyPreToolUseHooksForBridge(
        hookContext,
        call.name,
        call.id,
        call.input,
        signal
      );
      hookInput.toolInput = gate.input;
      if (gate.denyReason != null) {
        if (hookOptions != null) {
          await executeHooks({
            ...hookOptions,
            input: {
              ...hookInput,
              hook_event_name: 'PermissionDenied',
              reason: gate.denyReason,
            },
          }).catch(() => undefined);
        }
        return {
          call_id: call.id,
          result: null,
          is_error: true,
          error_message: gate.denyReason,
        };
      }
    }
    [result] = await executeTools(
      [{ ...call, input: hookInput.toolInput }],
      toolMap,
      programmaticToolName,
      signal,
      config
    );
  } catch (error) {
    result = {
      call_id: call.id,
      result: null,
      is_error: true,
      error_message:
        error instanceof Error ? error.message : 'Tool execution cancelled',
    };
  }
  if (hookOptions == null) return result;
  // Completion hooks must still record the outcome when the request was aborted.
  if (result.is_error) {
    await executeHooks({
      ...hookOptions,
      input: {
        ...hookInput,
        hook_event_name: 'PostToolUseFailure',
        error: result.error_message ?? `Tool ${call.name} failed`,
      },
    }).catch(() => undefined);
    return result;
  }
  const post = await executeHooks({
    ...hookOptions,
    input: {
      ...hookInput,
      hook_event_name: 'PostToolUse',
      toolOutput: result.result,
    },
  }).catch(() => undefined);
  return post?.updatedOutput === undefined
    ? result
    : { ...result, result: post.updatedOutput };
}

async function createToolBridge(
  toolMap: t.ToolMap,
  hookContext: t.ProgrammaticHookContext | undefined,
  programmaticToolName: string,
  config?: RunnableConfig
): Promise<ToolBridge> {
  const signal = config?.signal;
  const token = randomBytes(32).toString('hex');
  const controllers = new Set<AbortController>();
  const pending = new Set<Promise<void>>();
  let closing = false;
  const abortRequests = (): void => {
    for (const controller of controllers) controller.abort(signal?.reason);
  };
  signal?.addEventListener('abort', abortRequests, { once: true });
  const server = createServer((req, res) => {
    if (closing || signal?.aborted === true) {
      res.writeHead(503);
      res.end('Tool bridge is closing');
      return;
    }
    const url = new URL(req.url ?? '/', 'http://127.0.0.1');
    const isTextMode = url.searchParams.get('mode') === 'text';
    if (req.method !== 'POST' || url.pathname !== '/tool') {
      if (isTextMode) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('Not found');
      } else {
        writeJson(res, 404, { error: 'Not found' });
      }
      return;
    }

    const presented = req.headers[BRIDGE_AUTH_HEADER];
    const presentedToken = Array.isArray(presented) ? presented[0] : presented;
    if (
      typeof presentedToken !== 'string' ||
      !constantTimeEquals(presentedToken, token)
    ) {
      if (isTextMode) {
        res.writeHead(401, { 'Content-Type': 'text/plain' });
        res.end('Unauthorized');
      } else {
        writeJson(res, 401, { error: 'Unauthorized' });
      }
      return;
    }

    const controller = new AbortController();
    controllers.add(controller);
    const disconnect = (): void => {
      if (!res.writableFinished)
        controller.abort(new Error('Tool bridge client disconnected'));
    };
    const cancelRequest = (): void => {
      req.destroy();
    };
    controller.signal.addEventListener('abort', cancelRequest, { once: true });
    res.once('close', disconnect);
    const operation = readRequestBody(req)
      .then(async (body) => {
        if (typeof body.name !== 'string' || body.name === '') {
          const message = 'Tool request is missing a tool name.';
          if (isTextMode) {
            res.writeHead(400, { 'Content-Type': 'text/plain' });
            res.end(message);
          } else {
            writeJson(res, 400, {
              call_id: body.id ?? 'invalid',
              result: null,
              is_error: true,
              error_message: message,
            });
          }
          return;
        }

        const result = await executeBridgeCall(
          {
            id: body.id ?? `local_call_${randomUUID()}`,
            name: body.name,
            input: body.input ?? {},
          },
          toolMap,
          controller.signal,
          programmaticToolName,
          hookContext,
          config
        );
        if (res.destroyed) return;

        if (isTextMode) {
          if (result.is_error === true) {
            res.writeHead(500, { 'Content-Type': 'text/plain' });
            res.end(result.error_message ?? `Tool ${body.name} failed`);
          } else {
            const value = toSerializable(result.result);
            const text =
              typeof value === 'string' ? value : JSON.stringify(value);
            res.writeHead(200, { 'Content-Type': 'text/plain' });
            res.end(text);
          }
          return;
        }

        writeJson(res, 200, {
          ...result,
          result: toSerializable(result.result),
        });
      })
      .catch((error: Error) => {
        if (res.destroyed) return;
        if (isTextMode) {
          res.writeHead(500, { 'Content-Type': 'text/plain' });
          res.end(error.message);
        } else {
          writeJson(res, 500, {
            call_id: 'error',
            result: null,
            is_error: true,
            error_message: error.message,
          });
        }
      })
      .finally(() => {
        res.removeListener('close', disconnect);
        controller.signal.removeEventListener('abort', cancelRequest);
        controllers.delete(controller);
        pending.delete(operation);
      });
    pending.add(operation);
  });

  try {
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject);
      server.listen(0, '127.0.0.1', resolve);
    });
  } catch (error) {
    signal?.removeEventListener('abort', abortRequests);
    throw error;
  }

  const address = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${address.port}/tool`,
    token,
    close: async (): Promise<void> => {
      closing = true;
      signal?.removeEventListener('abort', abortRequests);
      abortRequests();
      const closed = new Promise<void>((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()));
      });
      server.closeAllConnections();
      await Promise.all([closed, ...pending]);
    },
  };
}

function indent(code: string): string {
  return code
    .split('\n')
    .map((line) => `  ${line}`)
    .join('\n');
}

function createPythonProgram(
  code: string,
  toolDefs: t.LCTool[],
  bridgeUrl: string,
  bridgeToken: string
): string {
  const functionDefs = toolDefs
    .map((def) => {
      const pythonName = normalizeToPythonIdentifier(def.name);
      return [
        `async def ${pythonName}(**kwargs):`,
        `  return await __librechat_call_tool(${JSON.stringify(def.name)}, kwargs)`,
      ].join('\n');
    })
    .join('\n\n');

  return `
import asyncio
import json
import urllib.request

__LIBRECHAT_TOOL_BRIDGE = ${JSON.stringify(bridgeUrl)}
__LIBRECHAT_TOOL_TOKEN = ${JSON.stringify(bridgeToken)}

async def __librechat_call_tool(name, payload):
  body = json.dumps({"name": name, "input": payload}).encode("utf-8")
  headers = {
    "Content-Type": "application/json",
    ${JSON.stringify(BRIDGE_AUTH_HEADER)}: __LIBRECHAT_TOOL_TOKEN,
  }

  def request():
    req = urllib.request.Request(__LIBRECHAT_TOOL_BRIDGE, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as response:
      return response.read().decode("utf-8")

  raw = await asyncio.to_thread(request)
  result = json.loads(raw)
  if result.get("is_error"):
    raise RuntimeError(result.get("error_message") or f"Tool {name} failed")
  return result.get("result")

${functionDefs}

async def __librechat_main():
${indent(code)}

asyncio.run(__librechat_main())
`.trimStart();
}

export function _createBashProgramForTests(
  code: string,
  toolDefs: t.LCTool[],
  bridgeUrl: string,
  bridgeToken: string
): string {
  return createBashProgram(code, toolDefs, bridgeUrl, bridgeToken);
}

function createBashProgram(
  code: string,
  toolDefs: t.LCTool[],
  bridgeUrl: string,
  bridgeToken: string
): string {
  const functions = toolDefs
    .map((def) => {
      const bashName = normalizeToBashIdentifier(def.name);
      return [
        `${bashName}() {`,
        '  local payload="${1:-}"',
        '  if [ -z "$payload" ]; then payload=\'{}\'; fi',
        `  __librechat_call_tool ${shellQuote(def.name)} "$payload"`,
        '}',
      ].join('\n');
    })
    .join('\n\n');

  return `
__LIBRECHAT_TOOL_BRIDGE=${shellQuote(bridgeUrl)}
__LIBRECHAT_TOOL_HEADER=${shellQuote(BRIDGE_AUTH_HEADER)}
__LIBRECHAT_TOOL_TOKEN=${shellQuote(bridgeToken)}

# Bridge call helper. Tries curl first (universally available, no
# JSON parser needed thanks to the bridge's ?mode=text endpoint),
# falls back to python3 for environments without curl. Codex P2 #19
# flagged that the prior python3-only path broke minimal containers
# (and Windows hosts without a python3 binary on PATH). Tool names
# come from Constants.* and are always safe identifiers, so we can
# splice them into JSON without an escape pass.
__librechat_call_tool() {
  local tool_name="$1"
  local payload="$2"
  if command -v curl >/dev/null 2>&1; then
    local body="{\\"name\\":\\"$tool_name\\",\\"input\\":$payload}"
    local response
    local http_code
    response=$(curl -sS -X POST \
      -H "Content-Type: application/json" \
      -H "$__LIBRECHAT_TOOL_HEADER: $__LIBRECHAT_TOOL_TOKEN" \
      --data-binary "$body" \
      -w '\\n__LIBRECHAT_HTTP_CODE_%{http_code}__' \
      "$__LIBRECHAT_TOOL_BRIDGE?mode=text")
    http_code=$(printf '%s' "$response" | sed -n 's/.*__LIBRECHAT_HTTP_CODE_\\([0-9][0-9]*\\)__$/\\1/p')
    local body_only
    body_only=$(printf '%s' "$response" | sed 's/__LIBRECHAT_HTTP_CODE_[0-9][0-9]*__$//')
    if [ "$http_code" = "200" ]; then
      printf '%s' "$body_only"
      return 0
    fi
    printf '%s\\n' "$body_only" >&2
    return 1
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$__LIBRECHAT_TOOL_BRIDGE" "$tool_name" "$payload" "$__LIBRECHAT_TOOL_HEADER" "$__LIBRECHAT_TOOL_TOKEN" <<'PY'
import json
import sys
import urllib.request

url, name, payload, header, token = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
body = json.dumps({"name": name, "input": json.loads(payload)}).encode("utf-8")
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", header: token}, method="POST")
with urllib.request.urlopen(req, timeout=300) as response:
  result = json.loads(response.read().decode("utf-8"))
if result.get("is_error"):
  print(result.get("error_message") or f"Tool {name} failed", file=sys.stderr)
  sys.exit(1)
value = result.get("result")
if isinstance(value, str):
  print(value)
else:
  print(json.dumps(value))
PY
  else
    printf 'librechat: tool bridge needs either curl or python3 on PATH\\n' >&2
    return 1
  fi
}

${functions}

${code}
`.trimStart();
}

function getProgrammaticContext(config?: {
  toolCall?: unknown;
}): Partial<t.ProgrammaticCache> & { tools?: t.LCTool[] } {
  return (config?.toolCall ?? {}) as Partial<t.ProgrammaticCache> & {
    tools?: t.LCTool[];
  };
}

function createEffectiveToolMap(
  toolMap: t.ToolMap,
  effectiveTools: t.LCTool[]
): { effectiveTools: t.LCTool[]; effectiveMap: t.ToolMap } {
  const effectiveMap = new Map<string, t.GenericTool>(
    effectiveTools
      .map((def) => {
        const executable = toolMap.get(def.name);
        return executable == null
          ? undefined
          : ([def.name, executable] as [string, t.GenericTool]);
      })
      .filter((entry): entry is [string, t.GenericTool] => entry != null)
  );

  return { effectiveTools, effectiveMap };
}

async function runLocalProgrammaticTool(args: {
  params: LocalProgrammaticParams;
  config?: ToolRunnableConfig;
  localConfig: t.LocalExecutionConfig;
  runtime: LocalProgrammaticRuntime;
}): Promise<[string, t.ProgrammaticExecutionArtifact]> {
  const context = getProgrammaticContext(args.config);
  const signal = args.config?.signal;
  signal?.throwIfAborted();
  const { toolMap, disallowedToolDefs, programmaticToolName, hookContext } =
    context;
  const toolDefs = resolveProgrammaticToolDefinitions(context);

  const runnerName =
    programmaticToolName ??
    (args.runtime === 'bash'
      ? Constants.BASH_PROGRAMMATIC_TOOL_CALLING
      : Constants.PROGRAMMATIC_TOOL_CALLING);
  const selectedTools = selectProgrammaticTools({
    requestedToolNames: args.params.tool_manifest,
    allowedToolDefs: toolDefs,
    disallowedToolDefs,
    programmaticToolName: runnerName,
  });

  /* The local program builder supports zero stubs, so a call that selected no
   * tools needs neither a toolMap nor toolDefs. Injected context is still
   * required to conclude that. */
  const needsNoTools =
    selectedTools.length === 0 &&
    (args.params.tool_manifest?.length === 0 ||
      toolDefs != null ||
      disallowedToolDefs != null);

  if (!needsNoTools) {
    if (toolMap == null || toolMap.size === 0) {
      throw new Error('No toolMap provided for local programmatic execution.');
    }
    if (toolDefs == null || toolDefs.length === 0) {
      throw new Error(
        'No tool definitions provided for local programmatic execution.'
      );
    }
  }

  const { effectiveTools, effectiveMap } = createEffectiveToolMap(
    toolMap ?? new Map(),
    selectedTools
  );
  const bridge = await createToolBridge(
    effectiveMap,
    hookContext,
    runnerName,
    args.config
  );

  try {
    const timeoutMs =
      args.params.timeout ?? args.localConfig.timeoutMs ?? DEFAULT_TIMEOUT;
    const result =
      args.runtime === 'bash'
        ? await executeLocalBash(
          createBashProgram(
            args.params.code,
            effectiveTools,
            bridge.url,
            bridge.token
          ),
          { ...args.localConfig, timeoutMs },
          signal
        )
        : await executeLocalCode(
          {
            lang: 'py',
            code: createPythonProgram(
              args.params.code,
              effectiveTools,
              bridge.url,
              bridge.token
            ),
          },
          { ...args.localConfig, timeoutMs },
          signal
        );

    signal?.throwIfAborted();
    const output = formatLocalOutput(capLocalOutput(result, args.localConfig), getLocalCwd(args.localConfig));
    if (result.exitCode !== 0 || result.timedOut) {
      throw new Error(output);
    }

    return [output, {
      session_id: getLocalSessionId(args.localConfig),
      files: [],
    }];
  } finally {
    await bridge.close();
  }
}

function resolveProgrammaticOutputConfig(
  config: t.LocalExecutionConfig
): t.LocalExecutionConfig {
  const maxChars = config.programmaticResultMaxChars;
  if (maxChars === undefined) return config;
  if (!Number.isSafeInteger(maxChars) || maxChars < 0) {
    throw new RangeError('programmaticResultMaxChars must be a nonnegative safe integer.');
  }
  return { ...config, toolResultMaxChars: maxChars };
}

export function createLocalProgrammaticToolCallingTool(
  localConfig: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  const executionConfig = resolveProgrammaticOutputConfig(localConfig);
  return createAwaitedTool({
    func: async (
      rawParams,
      _manager,
      config
    ): Promise<[string, t.ProgrammaticExecutionArtifact]> => {
      const params = rawParams as LocalProgrammaticParams;
      return runLocalProgrammaticTool({
        params,
        config,
        localConfig: executionConfig,
        runtime: resolveRuntime(params),
      });
    },
    name: ProgrammaticToolCallingName,
    description: LOCAL_PROGRAMMATIC_DESCRIPTION,
    schema: createLocalProgrammaticToolCallingSchema(localConfig),
    responseFormat: Constants.CONTENT_AND_ARTIFACT,
  });
}

export function createLocalBashProgrammaticToolCallingTool(
  localConfig: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  const executionConfig = resolveProgrammaticOutputConfig(localConfig);
  return createAwaitedTool({
    func: async (
      rawParams,
      _manager,
      config
    ): Promise<[string, t.ProgrammaticExecutionArtifact]> => {
      const params = rawParams as LocalProgrammaticParams;
      return runLocalProgrammaticTool({
        params,
        config,
        localConfig: executionConfig,
        runtime: 'bash',
      });
    },
    name: Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
    description: LOCAL_BASH_PROGRAMMATIC_DESCRIPTION,
    schema: createLocalBashProgrammaticToolCallingSchema(localConfig),
    responseFormat: Constants.CONTENT_AND_ARTIFACT,
  });
}
