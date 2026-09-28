// src/tools/ProgrammaticToolCalling.ts
import { config } from 'dotenv';
import { randomUUID } from 'node:crypto';
import fetch, { RequestInit } from 'node-fetch';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { ProgrammaticToolCallingJsonSchema } from './ptcTimeout';
import type * as t from '@/types';
import {
  CODE_ARTIFACT_PATH_GUIDANCE,
  addCodeApiExecutionProfileHeader,
  appendCodeSessionFileSummary,
  appendFailedExecutionFileReminder,
  buildCodeApiExecutionErrorMessage,
  buildCodeApiHttpErrorMessage,
  CodeApiRequestError,
  buildCodeApiEndpoint,
  emptyOutputMessage,
  getCodeBaseURL,
  appendTmpScratchReminder,
  normalizeCodeApiRequestError,
  resolveCodeApiAuthHeaders,
  selectRuntimeSessionHint,
} from './CodeExecutor';
import {
  assertUnambiguousIdentifiers,
  projectProgrammaticToolMap,
  resolveProgrammaticToolDefinitions,
  selectProgrammaticTools,
  type ProgrammaticInvocationParams,
} from './ProgrammaticCallerPolicy';
import {
  clampCodeApiRunTimeoutMs,
  createCodeApiRunTimeoutSchema,
  resolveCodeApiRunTimeoutMs,
} from './ptcTimeout';
import {
  appendArtifactDeliveryWarning,
  normalizeArtifactDeliveryFailure,
} from '@/tools/ArtifactDelivery';
import {
  appendArtifactTruncationWarning,
  normalizeArtifactTruncation,
} from '@/tools/ArtifactTruncation';
import {
  describeCodeApiError,
  logCodeApiDiagnostic,
} from '@/tools/diagnostics';
import { appendExecutionArtifactFileSummary } from './CodeSessionFileSummary';
import { resolveFetchProxyAgent } from '@/utils/proxy';
import { INTENT_PROPERTY } from '@/tools/intentArg';
import { Constants } from '@/common';

config();

/** Default max round-trips to prevent infinite loops */
const DEFAULT_MAX_ROUND_TRIPS = 20;

const DEFAULT_RUN_TIMEOUT_MS = resolveCodeApiRunTimeoutMs();
const CODE_API_PROGRAMMATIC_REQUEST_HEADER =
  'X-LibreChat-Code-Request-ID';
const CANCELLATION_REQUEST_TIMEOUT_MS = 2_000;

// ============================================================================
// Description Components (Single Source of Truth)
// ============================================================================

const STATELESS_WARNING = `CRITICAL - STATELESS EXECUTION:
Each call is a fresh Python interpreter. Variables, imports, and data do NOT persist between calls.
You MUST complete your entire workflow in ONE code block: query → process → output.
DO NOT split work across multiple calls expecting to reuse variables.`;

const CORE_RULES = `Rules:
- One call: state does not persist
- Auto-wrapped async; use await, no main()/asyncio.run()
- Tools are pre-defined—DO NOT write function definitions
- Call tools with keyword args only (await tool(arg=value), never pass a dict)
- Tool results are decoded Python values (dict/list/str)
- Only print() output returns to the model
- ${CODE_ARTIFACT_PATH_GUIDANCE}
- timeout caps one sandbox run/replay iteration, not the total multi-round-trip workflow`;

const ADDITIONAL_RULES =
  '- Tool names normalized: hyphens→underscores, keywords get `_tool` suffix';

const EXAMPLES = `Example (Complete workflow in one call):
  # Query data
  data = await query_database(sql="SELECT * FROM users")
  # Process it
  df = pd.DataFrame(data)
  summary = df.groupby('region').sum()
  # Output results
  await write_to_sheet(spreadsheet_id=sid, data=summary.to_dict())
  print(f"Wrote {len(summary)} rows")

Example (Parallel calls):
  sf, ny = await asyncio.gather(get_weather(city="SF"), get_weather(city="NY"))
  print(f"SF: {sf}, NY: {ny}")`;

// ============================================================================
// Schema
// ============================================================================

const CODE_PARAM_DESCRIPTION = `Python code that calls tools programmatically. Tools are available as async functions.

${STATELESS_WARNING}

Your code is auto-wrapped in async context. Just write logic with await—no boilerplate needed.

${EXAMPLES}

${CORE_RULES}`;

const TOOL_MANIFEST_DESCRIPTION =
  'Exact registered tool names used by the code. Required when direct-only tools are configured; ' +
  'validated before execution starts. Pass [] when the code calls no tools at all.';

export function createProgrammaticToolCallingSchema(
  maxRunTimeoutMs = DEFAULT_RUN_TIMEOUT_MS
): ProgrammaticToolCallingJsonSchema {
  return {
    type: 'object',
    properties: {
      intent: { ...INTENT_PROPERTY },
      code: {
        type: 'string',
        minLength: 1,
        description: CODE_PARAM_DESCRIPTION,
      },
      tool_manifest: {
        type: 'array',
        items: { type: 'string' },
        uniqueItems: true,
        description: TOOL_MANIFEST_DESCRIPTION,
      },
      timeout: createCodeApiRunTimeoutSchema(maxRunTimeoutMs),
    },
    required: ['code'],
  } as const;
}

export const ProgrammaticToolCallingSchema =
  createProgrammaticToolCallingSchema();

export const ProgrammaticToolCallingName = Constants.PROGRAMMATIC_TOOL_CALLING;

export const ProgrammaticToolCallingDescription = `
Run tools via Python code. Auto-wrapped in async context—just use \`await\` directly.

${STATELESS_WARNING}

${CORE_RULES}
${ADDITIONAL_RULES}

When to use: loops, conditionals, parallel (\`asyncio.gather\`), multi-step pipelines.

${EXAMPLES}
`.trim();

export const ProgrammaticToolCallingDefinition = {
  name: ProgrammaticToolCallingName,
  description: ProgrammaticToolCallingDescription,
  schema: ProgrammaticToolCallingSchema,
} as const;

// ============================================================================
// Helper Functions
// ============================================================================

/** Python reserved keywords that get `_tool` suffix in Code API */
const PYTHON_KEYWORDS = new Set([
  'False',
  'None',
  'True',
  'and',
  'as',
  'assert',
  'async',
  'await',
  'break',
  'class',
  'continue',
  'def',
  'del',
  'elif',
  'else',
  'except',
  'finally',
  'for',
  'from',
  'global',
  'if',
  'import',
  'in',
  'is',
  'lambda',
  'nonlocal',
  'not',
  'or',
  'pass',
  'raise',
  'return',
  'try',
  'while',
  'with',
  'yield',
]);

export type FetchSessionFilesScope =
  | { kind: 'skill'; id: string; version: number }
  | { kind: 'agent' | 'user'; id: string; version?: never };

type CodeApiSessionFileWire = {
  id?: unknown;
  name?: unknown;
  metadata?: unknown;
  resource_id?: unknown;
  storage_session_id?: unknown;
};

type CodeApiSessionFileMetadata = {
  'original-filename'?: unknown;
};

function isFetchSessionFilesScope(
  value: unknown
): value is FetchSessionFilesScope {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const scope = value as { kind?: unknown; id?: unknown; version?: unknown };
  if (
    (scope.kind === 'agent' || scope.kind === 'user') &&
    typeof scope.id === 'string'
  ) {
    return true;
  }
  return (
    scope.kind === 'skill' &&
    typeof scope.id === 'string' &&
    typeof scope.version === 'number'
  );
}

function isCodeApiAuthHeaders(
  value: string | t.CodeApiAuthHeaders | undefined
): value is t.CodeApiAuthHeaders {
  return value != null && typeof value !== 'string';
}

function isCodeApiSessionFileWire(
  value: unknown
): value is CodeApiSessionFileWire {
  return value != null && typeof value === 'object';
}

function isCodeApiSessionFileMetadata(
  value: unknown
): value is CodeApiSessionFileMetadata {
  return value != null && typeof value === 'object';
}

function normalizeSessionFile(
  file: CodeApiSessionFileWire,
  sessionId: string,
  scope?: FetchSessionFilesScope
): t.CodeEnvFile {
  const metadata = isCodeApiSessionFileMetadata(file.metadata)
    ? file.metadata
    : undefined;
  const rawName = typeof file.name === 'string' ? file.name : '';
  const nameParts = rawName.split('/');
  const fallbackId = nameParts.length > 1 ? nameParts[1].split('.')[0] : '';
  const id =
    typeof file.id === 'string' && file.id !== '' ? file.id : fallbackId;
  const originalFilename = metadata?.['original-filename'];
  const name =
    typeof originalFilename === 'string' ? originalFilename : rawName;
  const storage_session_id =
    typeof file.storage_session_id === 'string'
      ? file.storage_session_id
      : sessionId;
  const resource_id =
    typeof file.resource_id === 'string' && file.resource_id !== ''
      ? file.resource_id
      : (scope?.id ?? id);

  if (scope?.kind === 'skill') {
    return {
      storage_session_id,
      kind: 'skill',
      id,
      resource_id,
      name,
      version: scope.version,
    };
  }
  if (scope != null) {
    return {
      storage_session_id,
      kind: scope.kind,
      id,
      resource_id,
      name,
    };
  }
  return {
    storage_session_id,
    kind: 'user',
    id,
    resource_id: id,
    name,
  };
}

/**
 * Normalizes a tool name to Python identifier format.
 * Must match the Code API's `normalizePythonFunctionName` exactly:
 * 1. Replace hyphens and spaces with underscores
 * 2. Remove any other invalid characters
 * 3. Prefix with underscore if starts with number
 * 4. Append `_tool` if it's a Python keyword
 * @param name - The tool name to normalize
 * @returns Normalized Python-safe identifier
 */
export function normalizeToPythonIdentifier(name: string): string {
  let normalized = name.replace(/[-\s]/g, '_');

  normalized = normalized.replace(/[^a-zA-Z0-9_]/g, '');

  if (/^[0-9]/.test(normalized)) {
    normalized = '_' + normalized;
  }

  if (PYTHON_KEYWORDS.has(normalized)) {
    normalized = normalized + '_tool';
  }

  return normalized;
}

/**
 * Extracts tool names that are actually called in the Python code.
 * Handles hyphen/underscore conversion since Python identifiers use underscores.
 * @param code - The Python code to analyze
 * @param toolNameMap - Map from normalized Python name to original tool name
 * @returns Set of original tool names found in the code
 */
export function extractUsedToolNames(
  code: string,
  toolNameMap: Map<string, string>
): Set<string> {
  const usedTools = new Set<string>();

  for (const [pythonName, originalName] of toolNameMap) {
    const escapedName = pythonName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const pattern = new RegExp(`\\b${escapedName}\\s*\\(`, 'g');

    if (pattern.test(code)) {
      usedTools.add(originalName);
    }
  }

  return usedTools;
}

/**
 * Filters tool definitions to only include tools actually used in the code.
 * Handles the hyphen-to-underscore conversion for Python compatibility.
 * @param toolDefs - All available tool definitions
 * @param code - The Python code to analyze
 * @param debug - Enable debug logging
 * @returns Filtered array of tool definitions
 */
export function filterToolsByUsage(
  toolDefs: t.LCTool[],
  code: string,
  debug = false
): t.LCTool[] {
  const toolNameMap = new Map<string, string>();
  for (const tool of toolDefs) {
    const pythonName = normalizeToPythonIdentifier(tool.name);
    toolNameMap.set(pythonName, tool.name);
  }

  const usedToolNames = extractUsedToolNames(code, toolNameMap);

  if (debug) {
    // eslint-disable-next-line no-console
    console.log(
      `[PTC Debug] Tool filtering: found ${usedToolNames.size}/${toolDefs.length} tools in code`
    );
    if (usedToolNames.size > 0) {
      // eslint-disable-next-line no-console
      console.log(
        `[PTC Debug] Matched tools: ${Array.from(usedToolNames).join(', ')}`
      );
    }
  }

  if (usedToolNames.size === 0) {
    if (debug) {
      // eslint-disable-next-line no-console
      console.log(
        '[PTC Debug] No tools detected in code - sending all tools as fallback'
      );
    }
    return toolDefs;
  }

  return toolDefs.filter((tool) => usedToolNames.has(tool.name));
}

/**
 * Fetches files from a previous session to make them available for the current execution.
 * Files are returned as CodeEnvFile references to be included in the request.
 * @param baseUrl - The base URL for the Code API
 * @param sessionId - The session ID to fetch files from
 * @param scope - Resource scope used by CodeAPI to authorize the session
 * @param proxy - Optional HTTP proxy URL
 * @returns Array of CodeEnvFile references, or empty array if fetch fails
 */
export async function fetchSessionFiles(
  baseUrl: string,
  sessionId: string,
  proxy?: string,
  authHeaders?: t.CodeApiAuthHeaders
): Promise<t.CodeEnvFile[]>;
export async function fetchSessionFiles(
  baseUrl: string,
  sessionId: string,
  scope: FetchSessionFilesScope,
  proxyOrAuthHeaders?: string | t.CodeApiAuthHeaders,
  authHeaders?: t.CodeApiAuthHeaders
): Promise<t.CodeEnvFile[]>;
export async function fetchSessionFiles(
  baseUrl: string,
  sessionId: string,
  scopeOrProxy?: FetchSessionFilesScope | string,
  proxyOrAuthHeaders?: string | t.CodeApiAuthHeaders,
  scopedAuthHeaders?: t.CodeApiAuthHeaders
): Promise<t.CodeEnvFile[]> {
  try {
    const scope = isFetchSessionFilesScope(scopeOrProxy)
      ? scopeOrProxy
      : undefined;
    let proxy: string | undefined;
    let authHeaders: t.CodeApiAuthHeaders | undefined;
    if (scope == null) {
      proxy = typeof scopeOrProxy === 'string' ? scopeOrProxy : undefined;
      authHeaders = isCodeApiAuthHeaders(proxyOrAuthHeaders)
        ? proxyOrAuthHeaders
        : undefined;
    } else if (typeof proxyOrAuthHeaders === 'string') {
      proxy = proxyOrAuthHeaders;
      authHeaders = scopedAuthHeaders;
    } else {
      authHeaders = proxyOrAuthHeaders ?? scopedAuthHeaders;
    }
    const query = new URLSearchParams({ detail: 'full' });
    if (scope != null) {
      query.set('kind', scope.kind);
      query.set('id', scope.id);
      if (scope.kind === 'skill') {
        query.set('version', String(scope.version));
      }
    }
    const filesEndpoint = `${baseUrl}/files/${encodeURIComponent(sessionId)}?${query.toString()}`;
    const resolvedAuthHeaders = await resolveCodeApiAuthHeaders(authHeaders, {
      recoverable: true,
    });
    const fetchOptions: RequestInit = {
      method: 'GET',
      headers: {
        'User-Agent': 'LibreChat/1.0',
        ...resolvedAuthHeaders,
      },
    };

    const proxyAgent = resolveFetchProxyAgent(filesEndpoint, proxy);
    if (proxyAgent != null) {
      fetchOptions.agent = proxyAgent;
    }

    const response = await fetch(filesEndpoint, fetchOptions);
    if (!response.ok) {
      throw new Error(
        await buildCodeApiHttpErrorMessage('GET', filesEndpoint, response, {
          recoverable: true,
        })
      );
    }

    const files = await response.json();
    if (!Array.isArray(files) || files.length === 0) {
      return [];
    }

    return files
      .filter(isCodeApiSessionFileWire)
      .map((file) => normalizeSessionFile(file, sessionId, scope));
  } catch (error) {
    logCodeApiDiagnostic(
      'ProgrammaticToolCalling',
      'warn',
      'session file lookup failed; continuing without input files',
      describeCodeApiError(error)
    );
    return [];
  }
}

/**
 * Makes an HTTP request to the Code API.
 * @param endpoint - The API endpoint URL
 * @param body - The request body
 * @param proxy - Optional HTTP proxy URL
 * @returns The parsed API response
 */
export async function makeRequest(
  endpoint: string,
  body: Record<string, unknown>,
  proxy?: string,
  authHeaders?: t.CodeApiAuthHeaders,
  executionProfile?: t.CodeApiExecutionProfile,
  signal?: AbortSignal
): Promise<t.ProgrammaticExecutionResponse> {
  try {
    const resolvedAuthHeaders = await resolveCodeApiAuthHeaders(authHeaders);
    const requestId = signal != null ? randomUUID() : undefined;
    const fetchOptions: RequestInit = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'LibreChat/1.0',
        ...addCodeApiExecutionProfileHeader(
          resolvedAuthHeaders,
          executionProfile
        ),
        ...(requestId != null
          ? { [CODE_API_PROGRAMMATIC_REQUEST_HEADER]: requestId }
          : {}),
      },
      body: JSON.stringify(body),
      signal: signal as RequestInit['signal'],
    };

    const proxyAgent = resolveFetchProxyAgent(endpoint, proxy);
    if (proxyAgent != null) {
      fetchOptions.agent = proxyAgent;
    }

    let cancellationRequest: Promise<void> | undefined;
    const requestCancellation = (): void => {
      if (requestId == null || cancellationRequest != null) {
        return;
      }
      cancellationRequest = (async (): Promise<void> => {
        const cancellationController = new AbortController();
        const timeout = setTimeout(
          () => cancellationController.abort(),
          CANCELLATION_REQUEST_TIMEOUT_MS
        );
        try {
          const cancelOptions: RequestInit = {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'User-Agent': 'LibreChat/1.0',
              ...addCodeApiExecutionProfileHeader(
                resolvedAuthHeaders,
                executionProfile
              ),
            },
            body: JSON.stringify({ request_id: requestId }),
            signal: cancellationController.signal as RequestInit['signal'],
          };
          if (proxyAgent != null) {
            cancelOptions.agent = proxyAgent;
          }
          await fetch(`${endpoint}/cancel`, cancelOptions);
        } catch {
          // Cancellation is best effort for older Code API versions and
          // transient network failures. The original abort remains the public
          // result while the server-side execution deadline is the fallback.
        } finally {
          clearTimeout(timeout);
        }
      })();
    };
    signal?.addEventListener('abort', requestCancellation, { once: true });
    if (signal?.aborted === true) requestCancellation();

    let response;
    try {
      response = await fetch(endpoint, fetchOptions);
    } catch (error) {
      if (signal?.aborted === true) {
        requestCancellation();
        await cancellationRequest;
      }
      throw error;
    } finally {
      signal?.removeEventListener('abort', requestCancellation);
    }

    if (!response.ok) {
      throw new CodeApiRequestError(
        await buildCodeApiHttpErrorMessage('POST', endpoint, response, {
          profile: executionProfile,
        })
      );
    }

    return (await response.json()) as t.ProgrammaticExecutionResponse;
  } catch (error) {
    throw normalizeCodeApiRequestError(error);
  }
}

/**
 * Unwraps tool responses that may be formatted as tuples or content blocks.
 * MCP tools return [content, artifacts], we need to extract the raw data.
 * @param result - The raw result from tool.invoke()
 * @param isMCPTool - Whether this is an MCP tool (has mcp property)
 * @returns Unwrapped raw data (string, object, or parsed JSON)
 */
export function unwrapToolResponse(
  result: unknown,
  isMCPTool: boolean
): unknown {
  // Only unwrap if this is an MCP tool and result is a tuple
  if (!isMCPTool) {
    return result;
  }

  /**
   * Checks if a value is a content block object (has type and text).
   */
  const isContentBlock = (value: unknown): boolean => {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
      return false;
    }
    const obj = value as Record<string, unknown>;
    return typeof obj.type === 'string';
  };

  /**
   * Checks if an array is an array of content blocks.
   */
  const isContentBlockArray = (arr: unknown[]): boolean => {
    return arr.length > 0 && arr.every(isContentBlock);
  };

  /**
   * Extracts text from a single content block object.
   * Returns the text if it's a text block, otherwise returns null.
   */
  const extractTextFromBlock = (block: unknown): string | null => {
    if (typeof block !== 'object' || block === null) return null;
    const b = block as Record<string, unknown>;
    if (b.type === 'text' && typeof b.text === 'string') {
      return b.text;
    }
    return null;
  };

  /**
   * Extracts text from content blocks (array or single object).
   * Returns combined text or null if no text blocks found.
   */
  const extractTextFromContent = (content: unknown): string | null => {
    // Single content block object: { type: 'text', text: '...' }
    if (
      typeof content === 'object' &&
      content !== null &&
      !Array.isArray(content)
    ) {
      const text = extractTextFromBlock(content);
      if (text !== null) return text;
    }

    // Array of content blocks: [{ type: 'text', text: '...' }, ...]
    if (Array.isArray(content) && content.length > 0) {
      const texts = content
        .map(extractTextFromBlock)
        .filter((t): t is string => t !== null);
      if (texts.length > 0) {
        return texts.join('\n');
      }
    }

    return null;
  };

  /**
   * Tries to parse a string as JSON if it looks like JSON.
   */
  const maybeParseJSON = (str: string): unknown => {
    const trimmed = str.trim();
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      try {
        return JSON.parse(trimmed);
      } catch {
        return str;
      }
    }
    return str;
  };

  // Handle array of content blocks at top level FIRST
  // (before checking for tuple, since both are arrays)
  if (Array.isArray(result) && isContentBlockArray(result)) {
    const extractedText = extractTextFromContent(result);
    if (extractedText !== null) {
      return maybeParseJSON(extractedText);
    }
  }

  // Check if result is a tuple/array with [content, artifacts]
  if (Array.isArray(result) && result.length >= 1) {
    const [content] = result;

    // If first element is a string, return it (possibly parsed as JSON)
    if (typeof content === 'string') {
      return maybeParseJSON(content);
    }

    // Try to extract text from content blocks
    const extractedText = extractTextFromContent(content);
    if (extractedText !== null) {
      return maybeParseJSON(extractedText);
    }

    // If first element is an object (but not a text block), return it
    if (typeof content === 'object' && content !== null) {
      return content;
    }
  }

  // Handle single content block object at top level (not in tuple)
  const extractedText = extractTextFromContent(result);
  if (extractedText !== null) {
    return maybeParseJSON(extractedText);
  }

  // Not a formatted response, return as-is
  return result;
}

type ToolInputSchemaKind = {
  object: boolean;
  string: boolean;
};

function detectSchemaKind(schema: unknown): ToolInputSchemaKind {
  const kind: ToolInputSchemaKind = { object: false, string: false };

  if (schema == null || typeof schema !== 'object') {
    return kind;
  }

  const jsonSchemaType = (schema as { type?: unknown }).type;
  if (jsonSchemaType === 'object') {
    kind.object = true;
  } else if (jsonSchemaType === 'string') {
    kind.string = true;
  } else if (Array.isArray(jsonSchemaType)) {
    kind.object = jsonSchemaType.includes('object');
    kind.string = jsonSchemaType.includes('string');
  }

  const zodDef = (schema as { _def?: unknown })._def;
  if (zodDef == null || typeof zodDef !== 'object') {
    return kind;
  }

  const zodType = (zodDef as { type?: unknown; typeName?: unknown }).type;
  const zodTypeName = (zodDef as { type?: unknown; typeName?: unknown })
    .typeName;

  if (zodType === 'object' || zodTypeName === 'ZodObject') {
    kind.object = true;
  } else if (zodType === 'string' || zodTypeName === 'ZodString') {
    kind.string = true;
  }

  const innerSchema =
    (
      zodDef as {
        innerType?: unknown;
        schema?: unknown;
        type?: unknown;
      }
    ).innerType ?? (zodDef as { schema?: unknown }).schema;
  if (innerSchema != null) {
    const innerKind = detectSchemaKind(innerSchema);
    kind.object ||= innerKind.object;
    kind.string ||= innerKind.string;
  }

  const options = (zodDef as { options?: unknown }).options;
  if (Array.isArray(options)) {
    for (const option of options) {
      const optionKind = detectSchemaKind(option);
      kind.object ||= optionKind.object;
      kind.string ||= optionKind.string;
    }
  }

  return kind;
}

function getToolInputSchemaKind(tool: t.GenericTool): ToolInputSchemaKind {
  if (tool.constructor.name === 'DynamicTool') {
    return { object: false, string: true };
  }

  const schema = (tool as { schema?: unknown }).schema;
  return detectSchemaKind(schema);
}

function normalizeToolInput(
  input: t.PTCToolCall['input'],
  tool: t.GenericTool
): t.PTCToolCall['input'] {
  const schemaKind = getToolInputSchemaKind(tool);

  if (typeof input !== 'string') {
    if (!schemaKind.string || schemaKind.object) {
      return input;
    }

    const inputValue = (input as { input?: unknown }).input;
    if (typeof inputValue === 'string') {
      return input;
    }

    return JSON.stringify(input);
  }

  if (!schemaKind.object || schemaKind.string) {
    return input;
  }

  const trimmed = input.trim();
  if (!trimmed.startsWith('{')) {
    return input;
  }

  try {
    const parsed: unknown = JSON.parse(trimmed);
    if (
      typeof parsed === 'object' &&
      parsed !== null &&
      !Array.isArray(parsed)
    ) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    return input;
  }

  return input;
}

/**
 * Executes tools in parallel when requested by the API.
 * Uses Promise.all for parallel execution, catching individual errors.
 * Unwraps formatted responses (e.g., MCP tool tuples) to raw data.
 * @param toolCalls - Array of tool calls from the API
 * @param toolMap - Map of tool names to executable tools
 * @returns Array of tool results
 */
export async function executeTools(
  toolCalls: t.PTCToolCall[],
  toolMap: t.ToolMap,
  programmaticToolName = Constants.PROGRAMMATIC_TOOL_CALLING
): Promise<t.PTCToolResult[]> {
  const executions = toolCalls.map(async (call): Promise<t.PTCToolResult> => {
    const tool = toolMap.get(call.name);

    if (!tool) {
      return {
        call_id: call.id,
        result: null,
        is_error: true,
        error_message: `Tool '${call.name}' not found. Available tools: ${Array.from(toolMap.keys()).join(', ')}`,
      };
    }

    try {
      const result = await tool.invoke(normalizeToolInput(call.input, tool), {
        metadata: { [programmaticToolName]: true },
      });

      const isMCPTool = tool.mcp === true;
      const unwrappedResult = unwrapToolResponse(result, isMCPTool);

      return {
        call_id: call.id,
        result: unwrappedResult,
        is_error: false,
      };
    } catch (error) {
      return {
        call_id: call.id,
        result: null,
        is_error: true,
        error_message: (error as Error).message || 'Tool execution failed',
      };
    }
  });

  return await Promise.all(executions);
}

/**
 * Formats the completed response for the agent.
 *
 * Output includes stdout/stderr plus a compact session-file summary
 * when artifacts were persisted. The artifact still carries every
 * file so the host's session map stays in sync.
 *
 * @param response - The completed API response
 * @returns Tuple of [formatted string, artifact]
 */
export function formatCompletedResponse(
  response: t.ProgrammaticExecutionResponse,
  sourceCode = '',
  filePersistence: 'session' | 'execution' = 'session'
): [string, t.ProgrammaticExecutionArtifact] {
  let formatted = '';

  if (response.stdout != null && response.stdout !== '') {
    formatted += `stdout:\n${response.stdout}\n`;
  } else {
    formatted += emptyOutputMessage;
  }

  if (response.stderr != null && response.stderr !== '') {
    formatted += `stderr:\n${response.stderr}\n`;
  }

  const outputWithReminder = appendTmpScratchReminder(formatted, sourceCode);
  const artifactDelivery = normalizeArtifactDeliveryFailure(
    response.artifact_delivery
  );
  const outputWithDeliveryWarning = appendArtifactDeliveryWarning(
    outputWithReminder,
    artifactDelivery
  );
  const artifactTruncation = normalizeArtifactTruncation(
    response.artifact_truncation
  );
  const outputWithWarnings = appendArtifactTruncationWarning(
    outputWithDeliveryWarning,
    artifactTruncation
  );

  return [
    filePersistence === 'execution'
      ? appendExecutionArtifactFileSummary(outputWithWarnings, response.files)
      : appendCodeSessionFileSummary(outputWithWarnings, response.files),
    {
      session_id: response.session_id,
      files: response.files,
      ...(response.deleted_files != null
        ? { deleted_files: response.deleted_files }
        : {}),
      ...(artifactDelivery != null
        ? { artifact_delivery: artifactDelivery }
        : {}),
      ...(artifactTruncation != null
        ? { artifact_truncation: artifactTruncation }
        : {}),
      ...(response.runtime_session_id != null
        ? {
          runtime_session_id: response.runtime_session_id,
          runtime_status: response.runtime_status,
        }
        : {}),
    } satisfies t.ProgrammaticExecutionArtifact,
  ];
}

/**
 * Mirrors the Code API's programmatic Python wrapper (`wrapUserCodeInAsync`).
 *
 * The programmatic contract promises top-level `await` works; plain `/exec`
 * applies no wrapper, so forwarding source unchanged would turn valid
 * programmatic input into a SyntaxError. The wrapper is self-contained, so a
 * later divergence in the Code API's own copy cannot break this path.
 */
export function wrapPythonForPlainExecution(userCode: string): string {
  const indented = userCode
    .split('\n')
    .map((line) => (line.trim() === '' ? '' : `    ${line}`))
    .join('\n');

  return (
    'async def __user_main__():\n' +
    '    """Auto-generated wrapper for user code to support top-level await"""\n' +
    `${indented}\n` +
    '\n' +
    'if __name__ == "__main__":\n' +
    '    import asyncio\n' +
    '    asyncio.run(__user_main__())\n'
  );
}

/**
 * Runs code that selected no tools through plain `/exec`.
 *
 * `/exec/programmatic` rejects an empty tool manifest outright, and the sandbox
 * has nothing to inject for such a call anyway — no stubs, no replay round
 * trips. Routing it here makes a tool-free programmatic call behave exactly like
 * the equivalent `execute_code`, against every deployed Code API, instead of
 * failing with a rejection the model cannot act on.
 */
export async function runPlainExecution(args: {
  baseUrl: string;
  lang: 'bash' | 'py';
  code: string;
  /**
   * Forwarded so the model-supplied cap applies as soon as the Code API honors
   * it on `/exec`, which does not read `timeout` today. Session affinity and
   * input files are honored there, so those ride the same body.
   */
  timeout?: number;
  sessionId?: string;
  files?: t.CodeEnvFile[];
  runtimeSessionHint?: string;
  proxy?: string;
  authHeaders?: t.CodeApiAuthHeaders;
  executionProfile?: t.CodeApiExecutionProfile;
}): Promise<[string, t.ProgrammaticExecutionArtifact]> {
  const response = await makeRequest(
    buildCodeApiEndpoint(args.baseUrl, 'exec'),
    {
      lang: args.lang,
      code:
        args.lang === 'py' ? wrapPythonForPlainExecution(args.code) : args.code,
      ...(args.timeout != null ? { timeout: args.timeout } : {}),
      ...(args.sessionId != null && args.sessionId !== ''
        ? { session_id: args.sessionId }
        : {}),
      ...(args.files != null && args.files.length > 0
        ? { files: args.files }
        : {}),
      ...(args.runtimeSessionHint != null
        ? { runtime_session_hint: args.runtimeSessionHint }
        : {}),
    },
    args.proxy,
    args.authHeaders,
    args.executionProfile
  );

  return formatCompletedResponse(response, args.code);
}

// ============================================================================
// Tool Factory
// ============================================================================

/**
 * Creates a Programmatic Tool Calling tool for complex multi-tool workflows.
 *
 * This tool enables AI agents to write Python code that orchestrates multiple
 * tool calls programmatically, reducing LLM round-trips and token usage.
 *
 * The tool map must be provided at runtime via config.configurable.toolMap.
 *
 * @param params - Configuration parameters (baseUrl, maxRoundTrips, proxy)
 * @returns A LangChain DynamicStructuredTool for programmatic tool calling
 *
 * @example
 * const ptcTool = createProgrammaticToolCallingTool({ maxRoundTrips: 20 });
 *
 * const [output, artifact] = await ptcTool.invoke(
 *   { code, tools },
 *   { configurable: { toolMap } }
 * );
 */
export function createProgrammaticToolCallingTool(
  initParams: t.ProgrammaticToolCallingParams = {}
): DynamicStructuredTool {
  const baseUrl = initParams.baseUrl ?? getCodeBaseURL();
  const maxRoundTrips = initParams.maxRoundTrips ?? DEFAULT_MAX_ROUND_TRIPS;
  const maxRunTimeoutMs = resolveCodeApiRunTimeoutMs(initParams.runTimeoutMs);
  const proxy = initParams.proxy ?? process.env.PROXY;
  const debug = initParams.debug ?? process.env.PTC_DEBUG === 'true';
  const EXEC_ENDPOINT = buildCodeApiEndpoint(baseUrl, 'exec/programmatic');

  return tool(
    async (rawParams, config) => {
      const params = rawParams as ProgrammaticInvocationParams;
      const { code } = params;
      const timeout = clampCodeApiRunTimeoutMs(params.timeout, maxRunTimeoutMs);

      // Extra params injected by ToolNode (follows web_search pattern).
      const toolCall = (config.toolCall ?? {}) as ToolCall &
        Partial<t.ProgrammaticCache> & {
          session_id?: string;
          _injected_files?: t.CodeEnvFile[];
          _runtime_session_hint?: string;
        };
      const {
        toolMap,
        disallowedToolDefs,
        session_id,
        _injected_files,
        _runtime_session_hint,
      } = toolCall;
      const toolDefs = resolveProgrammaticToolDefinitions(
        toolCall as typeof toolCall & { tools?: t.LCTool[] }
      );

      const programmaticToolName =
        toolCall.programmaticToolName ??
        (typeof toolCall.name === 'string' && toolCall.name !== ''
          ? toolCall.name
          : Constants.PROGRAMMATIC_TOOL_CALLING);
      const effectiveTools = selectProgrammaticTools({
        requestedToolNames: params.tool_manifest,
        allowedToolDefs: toolDefs,
        disallowedToolDefs,
        programmaticToolName,
      });

      /* These guard the replay path only. A call that selected no tools never
       * replays, and event-driven ToolNode configurations legitimately inject
       * an empty toolMap/toolDefs. Injected context is still required to
       * conclude that — with nothing injected at all, an empty selection means
       * the host wired the tool up wrong. */
      const needsNoTools =
        effectiveTools.length === 0 &&
        (params.tool_manifest?.length === 0 ||
          toolDefs != null ||
          disallowedToolDefs != null);

      if (!needsNoTools) {
        if (toolMap == null || toolMap.size === 0) {
          throw new Error(
            'No toolMap provided. ' +
              'ToolNode should inject this from AgentContext when invoked through the graph.'
          );
        }

        if (toolDefs == null || toolDefs.length === 0) {
          throw new Error(
            'No tool definitions provided. ' +
              'Either pass tools in the input or ensure ToolNode injects toolDefs.'
          );
        }
      }

      assertUnambiguousIdentifiers(
        effectiveTools,
        normalizeToPythonIdentifier,
        programmaticToolName
      );

      const effectiveToolMap = projectProgrammaticToolMap(
        toolMap ?? new Map(),
        effectiveTools
      );

      let roundTrip = 0;

      try {
        // ====================================================================
        // Phase 1: Send the validated tool manifest with the initial request
        // ====================================================================

        if (debug) {
          // eslint-disable-next-line no-console
          console.log(
            `[PTC Debug] Sending ${effectiveTools.length} tools to API ` +
              `(selected from ${toolDefs?.length ?? 0})`
          );
        }

        /**
         * File injection: `_injected_files` from ToolNode session
         * context. The legacy `/files/<session_id>` HTTP fallback was
         * removed (see `CodeExecutor.ts`) — codeapi's sessionAuth now
         * requires kind/id query params unavailable at this point.
         */
        let files: t.CodeEnvFile[] | undefined;
        if (_injected_files && _injected_files.length > 0) {
          files = _injected_files;
        } else if (session_id != null && session_id.length > 0) {
          logCodeApiDiagnostic(
            'ProgrammaticToolCalling',
            'debug',
            'session carried no injected files; exec will run without input files',
            { files: 'none' }
          );
        }

        /* The hint rides the INITIAL request only; continuation_token binds
         * later round-trips. Prefer trusted per-agent factory context over
         * legacy ToolNode injection. Explicit default profiles always drop it.
         * PTC keeps its stateless runtime prompt in v1. */
        const selectedRuntimeSessionHint = selectRuntimeSessionHint(
          initParams.runtimeSessionHint,
          _runtime_session_hint
        );
        const runtimeSessionHint =
          initParams.executionProfile !== 'default' &&
          typeof selectedRuntimeSessionHint === 'string' &&
          selectedRuntimeSessionHint !== ''
            ? selectedRuntimeSessionHint
            : undefined;

        if (needsNoTools) {
          return await runPlainExecution({
            baseUrl,
            lang: 'py',
            code,
            timeout,
            sessionId: session_id,
            files,
            runtimeSessionHint,
            proxy,
            authHeaders: initParams.authHeaders,
            executionProfile: initParams.executionProfile,
          });
        }

        let response = await makeRequest(
          EXEC_ENDPOINT,
          {
            code,
            tools: effectiveTools,
            session_id,
            timeout,
            ...(files && files.length > 0 ? { files } : {}),
            ...(runtimeSessionHint != null
              ? { runtime_session_hint: runtimeSessionHint }
              : {}),
          },
          proxy,
          initParams.authHeaders,
          initParams.executionProfile,
          config.signal
        );

        // ====================================================================
        // Phase 2: Handle response loop
        // ====================================================================

        while (response.status === 'tool_call_required') {
          roundTrip++;

          if (roundTrip > maxRoundTrips) {
            throw new Error(
              `Exceeded maximum round trips (${maxRoundTrips}). ` +
                'This may indicate an infinite loop, excessive tool calls, ' +
                'or a logic error in your code.'
            );
          }

          if (debug) {
            // eslint-disable-next-line no-console
            console.log(
              `[PTC Debug] Round trip ${roundTrip}: ${response.tool_calls?.length ?? 0} tool(s) to execute`
            );
          }

          const toolResults = await executeTools(
            response.tool_calls ?? [],
            effectiveToolMap
          );

          response = await makeRequest(
            EXEC_ENDPOINT,
            {
              continuation_token: response.continuation_token,
              tool_results: toolResults,
            },
            proxy,
            initParams.authHeaders,
            initParams.executionProfile,
            config.signal
          );
        }

        // ====================================================================
        // Phase 3: Handle final state
        // ====================================================================

        if (response.status === 'completed') {
          return formatCompletedResponse(response, code);
        }

        if (response.status === 'error') {
          throw new Error(buildCodeApiExecutionErrorMessage(response));
        }

        throw new CodeApiRequestError();
      } catch (error) {
        const messageWithReminder = appendFailedExecutionFileReminder(
          (error as Error).message,
          code
        );
        const message = `Programmatic execution failed: ${messageWithReminder}`;
        if (error instanceof CodeApiRequestError) {
          throw new CodeApiRequestError(message);
        }
        throw new Error(message, { cause: error });
      }
    },
    {
      name: Constants.PROGRAMMATIC_TOOL_CALLING,
      description: ProgrammaticToolCallingDescription,
      schema: createProgrammaticToolCallingSchema(maxRunTimeoutMs),
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}
