import { createServer } from 'http';
import { tool } from '@langchain/core/tools';
import { randomBytes, randomUUID, timingSafeEqual } from 'crypto';
import type { DynamicStructuredTool } from '@langchain/core/tools';
import type { IncomingMessage, ServerResponse } from 'http';
import type { AddressInfo } from 'net';
import type { ToolResultProtection } from '@/protection/toolResult';
import type * as t from '@/types';
import {
  executeTools,
  formatCompletedResponse,
  normalizeToPythonIdentifier,
  ProgrammaticToolCallingName,
  ProgrammaticToolCallingSchema,
  ProgrammaticToolCallingDescription,
} from '@/tools/ProgrammaticToolCalling';
import {
  BashProgrammaticToolCallingSchema,
  BashProgrammaticToolCallingDescription,
  normalizeToBashIdentifier,
} from '@/tools/BashProgrammaticToolCalling';
import {
  resolveProgrammaticToolDefinitions,
  selectProgrammaticTools,
  type ProgrammaticInvocationParams,
} from '@/tools/ProgrammaticCallerPolicy';
import {
  executeLocalBash,
  executeLocalCode,
  getLocalSessionId,
  shellQuote,
} from './LocalExecutionEngine';
import { ProviderTextProtectionError } from '@/protection/providerText';
import { protectToolText } from '@/protection/toolResult';
import { executeHooks } from '@/hooks';
import { Constants } from '@/common';

const DEFAULT_TIMEOUT = 60000;
const LOCAL_MIN_TIMEOUT = 1000;
const LOCAL_MAX_TIMEOUT = 300000;

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
      timeout: createLocalTimeoutSchema(localConfig.timeoutMs),
    },
  } as const;
}

type ToolBridge = {
  protectionFailure: () => ProviderTextProtectionError | undefined;
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
 * inside an HTTP handler — fail-closed matches the rest of the SDK
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
  toolInput: Record<string, unknown>
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
        result.decision === 'ask' &&
        hookContext.failClosedOnHookError === true
          ? `Approval required for "${toolName}"; unavailable in a background subagent. Ask the parent to run it in the foreground.${result.reason == null ? '' : ` Reason: ${result.reason}`}`
          : (result.reason ??
            (result.decision === 'ask'
              ? `Tool "${toolName}" requires human approval; bridge cannot raise an interrupt — denying.`
              : `Tool "${toolName}" denied by PreToolUse hook.`)),
    };
  }
  return { input: nextInput };
}

async function createToolBridge(
  toolMap: t.ToolMap,
  hookContext?: t.ProgrammaticHookContext,
  protection?: { policy?: ToolResultProtection; signal?: AbortSignal }
): Promise<ToolBridge> {
  let protectionFailure: ProviderTextProtectionError | undefined;
  const inFlight = new Set<Promise<void>>();
  const token = randomBytes(32).toString('hex');
  const server = createServer((req, res) => {
    // `?mode=text` returns the already-serialized result as the body
    // (or the error message at non-2xx). Python/Node callers stay on
    // JSON; bash callers using curl can avoid pulling in a JSON
    // parser dependency (Codex P2 #19 — `python3` was a hard
    // requirement for the bash bridge, breaking minimal containers).
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

    const work = readRequestBody(req)
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

        if (protectionFailure != null) throw protectionFailure;
        protection?.signal?.throwIfAborted();
        const callId = body.id ?? `local_call_${randomUUID()}`;
        let effectiveInput: Record<string, unknown> = body.input ?? {};
        if (hookContext != null) {
          const gate = await applyPreToolUseHooksForBridge(
            hookContext,
            body.name,
            callId,
            effectiveInput
          );
          if (gate.denyReason != null) {
            const denyMsg = await protectToolText(protection?.policy, body.name, callId, gate.denyReason, 'error', protection?.signal) as string;
            if (isTextMode) {
              res.writeHead(500, { 'Content-Type': 'text/plain' });
              res.end(denyMsg);
            } else {
              writeJson(res, 500, {
                call_id: callId,
                result: null,
                is_error: true,
                error_message: denyMsg,
              });
            }
            return;
          }
          effectiveInput = gate.input;
        }

        const [result] = await executeTools(
          [
            {
              id: callId,
              name: body.name,
              input: effectiveInput,
            },
          ],
          toolMap, Constants.PROGRAMMATIC_TOOL_CALLING, protection
        );

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
        if (error instanceof ProviderTextProtectionError) protectionFailure = error;
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
      });
    if (protection?.policy != null) {
      inFlight.add(work);
      void work.then(() => { inFlight.delete(work); }, () => { inFlight.delete(work); });
    }
  });

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });

  const address = server.address() as AddressInfo;
  let closing: Promise<void> | undefined;
  const close = (): Promise<void> => closing ??= new Promise<void>((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
  }).then(async () => { await Promise.allSettled(inFlight); });
  return {
    url: `http://127.0.0.1:${address.port}/tool`,
    token,
    protectionFailure: () => protectionFailure,
    close,
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
  config?: { toolCall?: unknown; signal?: AbortSignal };
  localConfig: t.LocalExecutionConfig;
  runtime: LocalProgrammaticRuntime;
}): Promise<[string, t.ProgrammaticExecutionArtifact]> {
  const context = getProgrammaticContext(args.config);
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
  const bridge = await createToolBridge(effectiveMap, hookContext, { policy: context.toolResultProtection, signal: args.config?.signal });

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
          { ...args.localConfig, timeoutMs }
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
          { ...args.localConfig, timeoutMs }
        );

    // Late accepted requests must settle before the enclosing result is accepted.
    await bridge.close();
    const protectionFailure = bridge.protectionFailure();
    if (protectionFailure != null) throw protectionFailure;
    if (result.exitCode !== 0 || result.timedOut) {
      throw new Error(
        result.stderr !== ''
          ? result.stderr
          : `Local ${args.runtime} programmatic execution exited with code ${
            result.exitCode ?? 'unknown'
          }`
      );
    }

    return formatCompletedResponse({
      status: 'completed',
      session_id: getLocalSessionId(args.localConfig),
      stdout: result.stdout,
      stderr: result.stderr,
      files: [],
    });
  } finally {
    await bridge.close();
  }
}

export function createLocalProgrammaticToolCallingTool(
  localConfig: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  return tool(
    async (rawParams, config) => {
      const params = rawParams as LocalProgrammaticParams;
      return runLocalProgrammaticTool({
        params,
        config,
        localConfig,
        runtime: resolveRuntime(params),
      });
    },
    {
      name: ProgrammaticToolCallingName,
      description: `${ProgrammaticToolCallingDescription}\n\nLocal engine: runs bash by default, or Python when \`lang\` is \`py\` or \`python\`, on the host machine and calls tools through an in-process localhost bridge.`,
      schema: createLocalProgrammaticToolCallingSchema(localConfig),
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export function createLocalBashProgrammaticToolCallingTool(
  localConfig: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  return tool(
    async (rawParams, config) => {
      const params = rawParams as LocalProgrammaticParams;
      return runLocalProgrammaticTool({
        params,
        config,
        localConfig,
        runtime: 'bash',
      });
    },
    {
      name: Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
      description: `${BashProgrammaticToolCallingDescription}\n\nLocal engine: runs this bash orchestration code on the host machine and calls tools through an in-process localhost bridge.`,
      schema: createLocalBashProgrammaticToolCallingSchema(localConfig),
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}
