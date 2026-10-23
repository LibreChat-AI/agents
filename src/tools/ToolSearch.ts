// src/tools/ToolSearch.ts
import { config } from 'dotenv';
import * as okapibm25Module from 'okapibm25';

type BM25Fn = (
  documents: string[],
  keywords: string[],
  constants?: { k1?: number; b?: number }
) => number[];

function getBM25Function(): BM25Fn {
  const mod = okapibm25Module as unknown as {
    default: BM25Fn | { default: BM25Fn } | undefined;
  };
  if (typeof mod === 'function') return mod;
  if (typeof mod.default === 'function') return mod.default;
  if (mod.default != null && typeof mod.default.default === 'function')
    return mod.default.default;
  throw new Error('Could not resolve BM25 function from okapibm25 module');
}

const BM25 = getBM25Function();
import fetch, { RequestInit } from 'node-fetch';
import { resolveFetchProxyAgent } from '@/utils/proxy';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type * as t from '@/types';
import { INTENT_PROPERTY } from '@/tools/intentArg';
import { buildCodeApiEndpoint, getCodeBaseURL } from './CodeExecutor';
import { Constants } from '@/common';

config();

/** Maximum allowed regex pattern length */
const MAX_PATTERN_LENGTH = 200;

export const ToolSearchToolName = Constants.TOOL_SEARCH;

export const ToolSearchToolDescription =
  'Searches deferred tools using BM25 ranking. Multi-word queries supported. Use mcp_server param to filter by server.';

const QUERY_DESCRIPTION_LOCAL =
  'Search term to find in tool names and descriptions. Case-insensitive substring matching. Optional if mcp_server is provided.';
const QUERY_DESCRIPTION_REGEX =
  'Regex pattern to search tool names and descriptions. Optional if mcp_server is provided.';
const FIELDS_DESCRIPTION =
  'Which fields to search. Default: name and description';
const MAX_RESULTS_DESCRIPTION = 'Maximum number of matching tools to return';
const DEFAULT_MAX_RESULTS = 5;
const MCP_SERVER_DESCRIPTION =
  'Filter to tools from specific MCP server(s). Can be a single server name or array of names. If provided without a query, lists all tools from those servers.';

export const ToolSearchToolSchema = {
  type: 'object',
  properties: {
    intent: { ...INTENT_PROPERTY },
    query: {
      type: 'string',
      maxLength: MAX_PATTERN_LENGTH,
      default: '',
      description: QUERY_DESCRIPTION_LOCAL,
    },
    fields: {
      type: 'array',
      items: { type: 'string', enum: ['name', 'description', 'parameters'] },
      default: ['name', 'description'],
      description: FIELDS_DESCRIPTION,
    },
    max_results: {
      type: 'integer',
      minimum: 1,
      maximum: 50,
      default: DEFAULT_MAX_RESULTS,
      description: MAX_RESULTS_DESCRIPTION,
    },
    mcp_server: {
      oneOf: [{ type: 'string' }, { type: 'array', items: { type: 'string' } }],
      description: MCP_SERVER_DESCRIPTION,
    },
  },
  required: [],
} as const;

export const ToolSearchToolDefinition = {
  name: ToolSearchToolName,
  description: ToolSearchToolDescription,
  schema: ToolSearchToolSchema,
} as const;

/** Maximum allowed regex nesting depth */
const MAX_REGEX_COMPLEXITY = 5;

/** Default search timeout in milliseconds */
const SEARCH_TIMEOUT = 5000;

/** JSON schema type for tool search parameters */
interface ToolSearchSchema {
  type: 'object';
  properties: Record<string, unknown>;
  required: string[];
}

/** Input params type for tool search */
interface ToolSearchParams {
  query?: string;
  fields?: ('name' | 'description' | 'parameters')[];
  max_results?: number;
  mcp_server?: string | string[];
}

/**
 * Creates the JSON schema with dynamic query description based on mode.
 * @param mode - The search mode determining query interpretation
 * @returns JSON schema for tool search parameters
 */
function createToolSearchSchema(mode: t.ToolSearchMode): ToolSearchSchema {
  const queryDescription =
    mode === 'local' ? QUERY_DESCRIPTION_LOCAL : QUERY_DESCRIPTION_REGEX;

  return {
    type: 'object',
    properties: {
      intent: { ...INTENT_PROPERTY },
      query: {
        type: 'string',
        maxLength: MAX_PATTERN_LENGTH,
        default: '',
        description: queryDescription,
      },
      fields: {
        type: 'array',
        items: { type: 'string', enum: ['name', 'description', 'parameters'] },
        default: ['name', 'description'],
        description: FIELDS_DESCRIPTION,
      },
      max_results: {
        type: 'integer',
        minimum: 1,
        maximum: 50,
        default: DEFAULT_MAX_RESULTS,
        description: MAX_RESULTS_DESCRIPTION,
      },
      mcp_server: {
        oneOf: [
          { type: 'string' },
          { type: 'array', items: { type: 'string' } },
        ],
        description: MCP_SERVER_DESCRIPTION,
      },
    },
    required: [],
  };
}

/**
 * Extracts the MCP server name from a tool name.
 * MCP tools follow the pattern: toolName_mcp_serverName
 * @param toolName - The full tool name
 * @returns The server name if it's an MCP tool, undefined otherwise
 */
function extractMcpServerName(toolName: string): string | undefined {
  const delimiterIndex = toolName.indexOf(Constants.MCP_DELIMITER);
  if (delimiterIndex === -1) {
    return undefined;
  }
  return toolName.substring(delimiterIndex + Constants.MCP_DELIMITER.length);
}

/**
 * Checks if a tool belongs to a specific MCP server.
 * @param toolName - The full tool name
 * @param serverName - The server name to match
 * @returns True if the tool belongs to the specified server
 */
function isFromMcpServer(toolName: string, serverName: string): boolean {
  const toolServer = extractMcpServerName(toolName);
  return toolServer === serverName;
}

/**
 * Canonical form used for lenient matching. It folds case and treats the
 * separators a host may rewrite as equivalent — LibreChat turns
 * `ClickHouse Cloud` into `ClickHouse_Cloud`, so a model asking for
 * `clickhouse-cloud` is naming the same server.
 *
 * **No character is ever discarded.** Every incorrect match this resolver has
 * produced came from deleting characters, because deleting merges two distinct
 * names onto one key: dropping non-ASCII made `éfoo` equal `foo`, dropping
 * combining marks made `İfoo` equal `ifoo`, and dropping symbols made `❤️`
 * equal `☀️`. Collapsing separators is the only equivalence intended here, so
 * it is the only one performed.
 *
 * Case is folded with `toLowerCase` alone. A round trip through uppercase
 * would additionally equate a Greek final sigma with its medial form, but it
 * also equates dotless `ı` with ASCII `i`, which Unicode case folding keeps
 * apart — and a false match hands over another server's tools while a missed
 * one only produces the message naming the servers that do exist. The safe
 * direction is the one taken here.
 */
function canonicalizeServerName(serverName: string): string {
  return serverName
    .trim()
    .toLowerCase()
    .normalize('NFC')
    .replace(/[-_. ]+/g, '-');
}

/**
 * Canonical form of a requested name with a leading `mcp` token removed, or
 * undefined when it had none. Servers are commonly keyed by the bare product
 * (`clickhouse`) while the package implementing them is `mcp-clickhouse`, and a
 * model that read the package name asks for that instead.
 *
 * The boundary is checked on the original string, before canonicalization drops
 * separators: `mcp-clickhouse` names the `clickhouse` server, but `mcparty` is
 * simply a server called `mcparty` and must never resolve to `arty`.
 */
function canonicalizeWithoutMcpPrefix(requested: string): string | undefined {
  const match = /^mcp[-_. ]+(.+)$/i.exec(requested.trim());
  if (match === null) {
    return undefined;
  }
  const key = canonicalizeServerName(match[1]);
  return key === '' ? undefined : key;
}

/**
 * Maps requested server names onto the servers that actually have tools
 * registered, so a near-miss filters instead of silently returning nothing.
 *
 * Exact names win, then case/separator-insensitive matches, then the
 * mcp-prefix-stripped form. Trying them in that order keeps a deployment that
 * runs both `github` and `mcp-github` unambiguous: each resolves to itself.
 *
 * Several registered names can still share one canonical form (`foo-bar` and
 * `foobar`). An inexact request cannot choose between them, so every candidate
 * is returned rather than whichever was seen first: a superset is recoverable
 * by reading the tool names, an arbitrary pick is not.
 *
 * @param requested - Server names as supplied by the caller
 * @param available - Server names present in the tool registry
 * @returns The registry names to filter by, and any request that matched none
 */
function resolveServerFilters(
  requested: string[],
  available: string[]
): { resolved: string[]; unresolved: string[] } {
  const exact = new Set(available);
  const canonical = new Map<string, string[]>();
  for (const name of available) {
    const key = canonicalizeServerName(name);
    /** A name with no ASCII alphanumerics — `---`, or a Unicode-only name —
     * canonicalizes to nothing. Indexing it would make every request that also
     * canonicalizes to nothing resolve to this server. */
    if (key === '') {
      continue;
    }
    const bucket = canonical.get(key);
    if (bucket === undefined) {
      canonical.set(key, [name]);
    } else {
      bucket.push(name);
    }
  }

  const resolved: string[] = [];
  const unresolved: string[] = [];
  for (const want of requested) {
    if (exact.has(want)) {
      resolved.push(want);
      continue;
    }
    const key = canonicalizeServerName(want);
    if (key === '') {
      unresolved.push(want);
      continue;
    }
    const direct = canonical.get(key);
    if (direct !== undefined) {
      resolved.push(...direct);
      continue;
    }
    const strippedKey = canonicalizeWithoutMcpPrefix(want);
    const stripped =
      strippedKey === undefined ? undefined : canonical.get(strippedKey);
    if (stripped !== undefined) {
      resolved.push(...stripped);
      continue;
    }
    unresolved.push(want);
  }

  return { resolved: Array.from(new Set(resolved)), unresolved };
}

/**
 * Checks if a tool belongs to any of the specified MCP servers.
 * @param toolName - The full tool name
 * @param serverNames - Array of server names to match
 * @returns True if the tool belongs to any of the specified servers
 */
function isFromAnyMcpServer(toolName: string, serverNames: string[]): boolean {
  const toolServer = extractMcpServerName(toolName);
  if (toolServer === undefined) {
    return false;
  }
  return serverNames.includes(toolServer);
}

/**
 * Normalizes server filter input to always be an array.
 * @param serverFilter - String, array of strings, or undefined
 * @returns Array of server names (empty if none specified)
 */
function normalizeServerFilter(
  serverFilter: string | string[] | undefined
): string[] {
  if (serverFilter === undefined) {
    return [];
  }
  if (typeof serverFilter === 'string') {
    return serverFilter === '' ? [] : [serverFilter];
  }
  return serverFilter.filter((s) => s !== '');
}

/**
 * Extracts all unique MCP server names from a tool registry.
 * @param toolRegistry - The tool registry to scan
 * @param onlyDeferred - If true, only considers deferred tools
 * @returns Array of unique server names, sorted alphabetically
 */
function getAvailableMcpServers(
  toolRegistry: t.LCToolRegistry | undefined,
  onlyDeferred: boolean = true
): string[] {
  if (!toolRegistry) {
    return [];
  }

  const servers = new Set<string>();
  for (const [, toolDef] of toolRegistry) {
    if (onlyDeferred && toolDef.defer_loading !== true) {
      continue;
    }
    const server = extractMcpServerName(toolDef.name);
    if (server !== undefined && server !== '') {
      servers.add(server);
    }
  }

  return Array.from(servers).sort();
}

/**
 * Escapes special regex characters in a string to use as a literal pattern.
 * @param pattern - The string to escape
 * @returns The escaped string safe for use in a RegExp
 */
function escapeRegexSpecialChars(pattern: string): string {
  return pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Counts the maximum nesting depth of groups in a regex pattern.
 * @param pattern - The regex pattern to analyze
 * @returns The maximum nesting depth
 */
function countNestedGroups(pattern: string): number {
  let maxDepth = 0;
  let currentDepth = 0;

  for (let i = 0; i < pattern.length; i++) {
    if (pattern[i] === '(' && (i === 0 || pattern[i - 1] !== '\\')) {
      currentDepth++;
      maxDepth = Math.max(maxDepth, currentDepth);
    } else if (pattern[i] === ')' && (i === 0 || pattern[i - 1] !== '\\')) {
      currentDepth = Math.max(0, currentDepth - 1);
    }
  }

  return maxDepth;
}

/**
 * Detects nested quantifiers that can cause catastrophic backtracking.
 * Patterns like (a+)+, (a*)*, (a+)*, etc.
 * @param pattern - The regex pattern to check
 * @returns True if nested quantifiers are detected
 */
function hasNestedQuantifiers(pattern: string): boolean {
  const nestedQuantifierPattern = /\([^)]*[+*][^)]*\)[+*?]/;
  return nestedQuantifierPattern.test(pattern);
}

/**
 * Checks if a regex pattern contains potentially dangerous constructs.
 * @param pattern - The regex pattern to validate
 * @returns True if the pattern is dangerous
 */
function isDangerousPattern(pattern: string): boolean {
  if (hasNestedQuantifiers(pattern)) {
    return true;
  }

  if (countNestedGroups(pattern) > MAX_REGEX_COMPLEXITY) {
    return true;
  }

  const dangerousPatterns = [
    /\.\{1000,\}/, // Excessive wildcards
    /\(\?=\.\{100,\}\)/, // Runaway lookaheads
    /\([^)]*\|\s*\){20,}/, // Excessive alternation (rough check)
    /\(\.\*\)\+/, // (.*)+
    /\(\.\+\)\+/, // (.+)+
    /\(\.\*\)\*/, // (.*)*
    /\(\.\+\)\*/, // (.+)*
  ];

  for (const dangerous of dangerousPatterns) {
    if (dangerous.test(pattern)) {
      return true;
    }
  }

  return false;
}

/**
 * Sanitizes a regex pattern for safe execution.
 * If the pattern is dangerous, it will be escaped to a literal string search.
 * @param pattern - The regex pattern to sanitize
 * @returns Object containing the safe pattern and whether it was escaped
 */
function sanitizeRegex(pattern: string): { safe: string; wasEscaped: boolean } {
  if (isDangerousPattern(pattern)) {
    return {
      safe: escapeRegexSpecialChars(pattern),
      wasEscaped: true,
    };
  }

  try {
    new RegExp(pattern);
    return { safe: pattern, wasEscaped: false };
  } catch {
    return {
      safe: escapeRegexSpecialChars(pattern),
      wasEscaped: true,
    };
  }
}

/**
 * Simplifies tool parameters for search purposes.
 * Extracts only the essential structure needed for parameter name searching.
 * @param parameters - The tool's JSON schema parameters
 * @returns Simplified parameters object
 */
function simplifyParametersForSearch(
  parameters?: t.JsonSchemaType
): t.JsonSchemaType | undefined {
  if (!parameters) {
    return undefined;
  }

  if (parameters.properties) {
    return {
      type: parameters.type,
      properties: Object.fromEntries(
        Object.entries(parameters.properties).map(([key, value]) => [
          key,
          { type: (value as t.JsonSchemaType).type },
        ])
      ),
    } as t.JsonSchemaType;
  }

  return { type: parameters.type };
}

/**
 * Splits one alphanumeric identifier segment on case boundaries without
 * emitting artificial one-character acronym fragments.
 */
function splitCaseSegment(segment: string): string[] {
  const splitTokens = segment
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);

  if (splitTokens.length < 2) return splitTokens;

  const mergedTokens: string[] = [];
  let prefix = '';
  for (const token of splitTokens) {
    if (token.length === 1) {
      prefix += token;
      continue;
    }
    mergedTokens.push(`${prefix}${token}`);
    prefix = '';
  }

  if (prefix && mergedTokens.length > 0) {
    mergedTokens[mergedTokens.length - 1] += prefix;
  }
  return mergedTokens.length > 0 ? mergedTokens : [prefix];
}

/**
 * Tokenizes a string into lowercase words for BM25.
 * Splits camelCase, underscores, and non-alphanumeric characters for consistent matching.
 * @param text - The text to tokenize
 * @returns Array of lowercase tokens
 */
function tokenize(text: string): string[] {
  return text
    .split(/[^a-zA-Z0-9]+/)
    .filter(Boolean)
    .flatMap(splitCaseSegment);
}

/**
 * Creates a searchable document string from tool metadata.
 * @param tool - The tool metadata
 * @param fields - Which fields to include
 * @returns Combined document string for BM25
 */
function createToolDocument(tool: t.ToolMetadata, fields: string[]): string {
  const parts: string[] = [];

  if (fields.includes('name')) {
    const baseName = tool.name.replace(/_/g, ' ');
    parts.push(baseName, baseName);
  }

  if (fields.includes('description') && tool.description) {
    parts.push(tool.description);
  }

  if (fields.includes('parameters') && tool.parameters?.properties) {
    const paramNames = Object.keys(tool.parameters.properties).join(' ');
    parts.push(paramNames);
  }

  return tokenize(parts.join(' ')).join(' ');
}

/**
 * Determines which field had the best match for a query.
 * @param tool - The tool to check
 * @param queryTokens - Tokenized query
 * @param fields - Fields to check
 * @returns The matched field and a snippet
 */
function findMatchedField(
  tool: t.ToolMetadata,
  queryTokens: string[],
  fields: string[]
): { field: string; snippet: string } {
  if (fields.includes('name')) {
    const nameLower = tool.name.toLowerCase();
    for (const token of queryTokens) {
      if (nameLower.includes(token)) {
        return { field: 'name', snippet: tool.name };
      }
    }
  }

  if (fields.includes('description') && tool.description) {
    const descLower = tool.description.toLowerCase();
    for (const token of queryTokens) {
      if (descLower.includes(token)) {
        return {
          field: 'description',
          snippet: tool.description.substring(0, 100),
        };
      }
    }
  }

  if (fields.includes('parameters') && tool.parameters?.properties) {
    const paramNames = Object.keys(tool.parameters.properties);
    const paramLower = paramNames.join(' ').toLowerCase();
    for (const token of queryTokens) {
      if (paramLower.includes(token)) {
        return { field: 'parameters', snippet: paramNames.join(', ') };
      }
    }
  }

  const fallbackSnippet = tool.description
    ? tool.description.substring(0, 100)
    : tool.name;
  return { field: 'unknown', snippet: fallbackSnippet };
}

/**
 * Performs BM25-based search for better relevance ranking.
 * Uses Okapi BM25 algorithm for term frequency and document length normalization.
 * If query is empty, returns all tools (up to maxResults) sorted alphabetically.
 * @param tools - Array of tool metadata to search
 * @param query - The search query (empty returns all tools)
 * @param fields - Which fields to search
 * @param maxResults - Maximum results to return
 * @returns Search response with matching tools ranked by BM25 score
 */
function performLocalSearch(
  tools: t.ToolMetadata[],
  query: string,
  fields: string[],
  maxResults: number
): t.ToolSearchResponse {
  if (tools.length === 0) {
    return {
      tool_references: [],
      total_tools_searched: 0,
      pattern_used: query,
    };
  }

  const queryTokens = tokenize(query);

  if (queryTokens.length === 0) {
    const allTools = tools
      .slice()
      .sort((a, b) => a.name.localeCompare(b.name))
      .slice(0, maxResults)
      .map((tool) => ({
        tool_name: tool.name,
        match_score: 1.0,
        matched_field: 'name',
        snippet: tool.description.substring(0, 100) || tool.name,
      }));

    return {
      tool_references: allTools,
      total_tools_searched: tools.length,
      pattern_used: query,
    };
  }

  const documents = tools.map((tool) => createToolDocument(tool, fields));
  const scores = BM25(documents, queryTokens, { k1: 1.5, b: 0.75 }) as number[];

  const maxScore = Math.max(...scores.filter((s) => s > 0), 1);
  const queryLower = query.toLowerCase().trim();
  const queryIdentifier = queryTokens.join('');
  const matchesIdentifiers = fields.includes('name');

  const results: Array<{
    result: t.ToolSearchResult;
    identifierPriority: number;
  }> = [];
  for (let i = 0; i < tools.length; i++) {
    const score = scores[i];
    const hasSearchScore = Number.isFinite(score) && score > 0;
    let identifierPriority = 0;
    let normalizedScore = hasSearchScore ? Math.min(score / maxScore, 1.0) : 0;

    if (matchesIdentifiers) {
      const rawBaseName = getBaseToolName(tools[i].name).toLowerCase();
      const rawFullName = tools[i].name.toLowerCase();
      const baseIdentifier = tokenize(rawBaseName).join('');
      const fullIdentifier = tokenize(rawFullName).join('');

      if (rawFullName === queryLower) {
        identifierPriority = 4;
        normalizedScore = 1.0;
      } else if (rawBaseName === queryLower) {
        identifierPriority = 3;
        normalizedScore = 1.0;
      } else if (
        baseIdentifier === queryIdentifier ||
        fullIdentifier === queryIdentifier
      ) {
        identifierPriority = 2;
        normalizedScore = 1.0;
      } else if (baseIdentifier.startsWith(queryIdentifier)) {
        identifierPriority = 1;
        normalizedScore = Math.max(normalizedScore, 0.95);
      }
    }

    if (!hasSearchScore && identifierPriority === 0) continue;

    const { field, snippet } = findMatchedField(tools[i], queryTokens, fields);
    results.push({
      result: {
        tool_name: tools[i].name,
        match_score: normalizedScore,
        matched_field: field,
        snippet,
      },
      identifierPriority,
    });
  }

  results.sort(
    (a, b) =>
      b.identifierPriority - a.identifierPriority ||
      b.result.match_score - a.result.match_score
  );
  const topResults = results.slice(0, maxResults).map(({ result }) => result);

  return {
    tool_references: topResults,
    total_tools_searched: tools.length,
    pattern_used: query,
  };
}

/**
 * Generates the JavaScript search script to be executed in the sandbox.
 * Uses plain JavaScript for maximum compatibility with the Code API.
 * @param deferredTools - Array of tool metadata to search through
 * @param fields - Which fields to search
 * @param maxResults - Maximum number of results to return
 * @param sanitizedPattern - The sanitized regex pattern
 * @returns The JavaScript code string
 */
function generateSearchScript(
  deferredTools: t.ToolMetadata[],
  fields: string[],
  maxResults: number,
  sanitizedPattern: string
): string {
  const lines = [
    '// Tool definitions (injected)',
    'var tools = ' + JSON.stringify(deferredTools) + ';',
    'var searchFields = ' + JSON.stringify(fields) + ';',
    'var maxResults = ' + maxResults + ';',
    'var pattern = ' + JSON.stringify(sanitizedPattern) + ';',
    '',
    '// Compile regex (pattern is sanitized client-side)',
    'var regex;',
    'try {',
    '  regex = new RegExp(pattern, \'i\');',
    '} catch (e) {',
    '  regex = new RegExp(pattern.replace(/[.*+?^${}()[\\]\\\\|]/g, "\\\\$&"), "i");',
    '}',
    '',
    '// Search logic',
    'var results = [];',
    '',
    'for (var j = 0; j < tools.length; j++) {',
    '  var tool = tools[j];',
    '  var bestScore = 0;',
    '  var matchedField = \'\';',
    '  var snippet = \'\';',
    '',
    '  // Search name (highest priority)',
    '  if (searchFields.indexOf(\'name\') >= 0 && regex.test(tool.name)) {',
    '    bestScore = 0.95;',
    '    matchedField = \'name\';',
    '    snippet = tool.name;',
    '  }',
    '',
    '  // Search description (medium priority)',
    '  if (searchFields.indexOf(\'description\') >= 0 && tool.description && regex.test(tool.description)) {',
    '    if (bestScore === 0) {',
    '      bestScore = 0.75;',
    '      matchedField = \'description\';',
    '      snippet = tool.description.substring(0, 100);',
    '    }',
    '  }',
    '',
    '  // Search parameter names (lower priority)',
    '  if (searchFields.indexOf(\'parameters\') >= 0 && tool.parameters && tool.parameters.properties) {',
    '    var paramNames = Object.keys(tool.parameters.properties).join(\' \');',
    '    if (regex.test(paramNames)) {',
    '      if (bestScore === 0) {',
    '        bestScore = 0.60;',
    '        matchedField = \'parameters\';',
    '        snippet = paramNames;',
    '      }',
    '    }',
    '  }',
    '',
    '  if (bestScore > 0) {',
    '    results.push({',
    '      tool_name: tool.name,',
    '      match_score: bestScore,',
    '      matched_field: matchedField,',
    '      snippet: snippet',
    '    });',
    '  }',
    '}',
    '',
    '// Sort by score (descending) and limit results',
    'results.sort(function(a, b) { return b.match_score - a.match_score; });',
    'var topResults = results.slice(0, maxResults);',
    '',
    '// Output as JSON',
    'console.log(JSON.stringify({',
    '  tool_references: topResults.map(function(r) {',
    '    return {',
    '      tool_name: r.tool_name,',
    '      match_score: r.match_score,',
    '      matched_field: r.matched_field,',
    '      snippet: r.snippet',
    '    };',
    '  }),',
    '  total_tools_searched: tools.length,',
    '  pattern_used: pattern',
    '}));',
  ];
  return lines.join('\n');
}

/**
 * Parses the search results from stdout JSON.
 * @param stdout - The stdout string containing JSON results
 * @returns Parsed search response
 */
function parseSearchResults(stdout: string): t.ToolSearchResponse {
  const jsonMatch = stdout.trim();
  const parsed = JSON.parse(jsonMatch) as t.ToolSearchResponse;
  return parsed;
}

/**
 * Formats search results as structured JSON for efficient parsing.
 * @param searchResponse - The parsed search response
 * @param nameFormat - Whether to show 'full' names (tool_mcp_server) or 'base' names (tool only)
 * @param notes - Diagnostics about requested servers that were not searched;
 * carried inside the JSON so the output stays parseable
 * @returns JSON string with search results
 */
function formatSearchResults(
  searchResponse: t.ToolSearchResponse,
  nameFormat: t.McpNameFormat = 'full',
  notes: string[] = []
): string {
  const { tool_references, total_tools_searched, pattern_used } =
    searchResponse;
  const useFullName = nameFormat === 'full';

  const output = {
    found: tool_references.length,
    tools: tool_references.map((ref) => ({
      name: useFullName ? ref.tool_name : getBaseToolName(ref.tool_name),
      score: Number(ref.match_score.toFixed(2)),
      matched_in: ref.matched_field,
      snippet: ref.snippet,
    })),
    total_searched: total_tools_searched,
    query: pattern_used,
    ...(notes.length > 0 ? { notes } : {}),
  };

  return JSON.stringify(output, null, 2);
}

/**
 * Extracts the base tool name (without MCP server suffix) from a full tool name.
 * @param toolName - The full tool name
 * @returns The base tool name without server suffix
 */
function getBaseToolName(toolName: string): string {
  const delimiterIndex = toolName.indexOf(Constants.MCP_DELIMITER);
  if (delimiterIndex === -1) {
    return toolName;
  }
  return toolName.substring(0, delimiterIndex);
}

/**
 * Checks whether a tool has any defined parameters in its JSON schema.
 * @param parameters - The tool's JSON schema parameters
 * @returns true if the tool has at least one parameter property
 */
function hasParams(parameters?: t.JsonSchemaType): boolean {
  return (
    parameters?.properties != null &&
    Object.keys(parameters.properties).length > 0
  );
}

/**
 * Generates a compact listing of deferred tools grouped by server.
 * Format: "server: tool1, tool2(\u2026), tool3"
 * Tools with parameters are annotated with (\u2026) to signal
 * that the LLM should discover the schema via tool_search before calling.
 * Non-MCP tools are grouped under "other".
 * @param toolRegistry - The tool registry
 * @param onlyDeferred - Whether to only include deferred tools
 * @returns Formatted string with tools grouped by server
 */
function getDeferredToolsListing(
  toolRegistry: t.LCToolRegistry | undefined,
  onlyDeferred: boolean
): string {
  if (!toolRegistry) {
    return '';
  }

  const toolsByServer: Record<string, string[]> = Object.create(null);

  for (const lcTool of toolRegistry.values()) {
    if (onlyDeferred && lcTool.defer_loading !== true) {
      continue;
    }

    const toolName = lcTool.name;
    const serverName = extractMcpServerName(toolName) ?? 'other';
    const baseName = getBaseToolName(toolName);
    const displayName = hasParams(lcTool.parameters)
      ? `${baseName}(\u2026)`
      : baseName;

    if (!(serverName in toolsByServer)) {
      toolsByServer[serverName] = [];
    }
    toolsByServer[serverName].push(displayName);
  }

  const serverNames = Object.keys(toolsByServer).sort((a, b) => {
    if (a === 'other') return 1;
    if (b === 'other') return -1;
    return a.localeCompare(b);
  });

  if (serverNames.length === 0) {
    return '';
  }

  const lines = serverNames.map(
    (server) => `${server}: ${toolsByServer[server].join(', ')}`
  );

  return lines.join('\n');
}

/**
 * Formats a server listing response as structured JSON.
 * NOTE: This is a PREVIEW only - tools are NOT discovered/loaded.
 * @param tools - Array of tool metadata from the server(s)
 * @param serverNames - The MCP server name(s)
 * @param nameFormat - Whether to show 'full' names (tool_mcp_server) or 'base' names (tool only)
 * @param notes - Diagnostics about requested servers that were not searched;
 * carried inside the JSON so the output stays parseable
 * @returns JSON string showing all tools grouped by server
 */
function formatServerListing(
  tools: t.ToolMetadata[],
  serverNames: string | string[],
  nameFormat: t.McpNameFormat = 'full',
  notes: string[] = []
): string {
  const servers = Array.isArray(serverNames) ? serverNames : [serverNames];
  const useFullName = nameFormat === 'full';

  if (tools.length === 0) {
    return JSON.stringify(
      {
        listing_mode: true,
        servers,
        total_tools: 0,
        tools_by_server: {},
        ...(notes.length > 0 ? { notes } : {}),
        hint: 'No tools found from the specified MCP server(s).',
      },
      null,
      2
    );
  }

  const toolsByServer: Record<
    string,
    Array<{ name: string; description: string }>
  > = Object.create(null);
  for (const tool of tools) {
    const server = extractMcpServerName(tool.name) ?? 'unknown';
    if (!(server in toolsByServer)) {
      toolsByServer[server] = [];
    }
    toolsByServer[server].push({
      name: useFullName ? tool.name : getBaseToolName(tool.name),
      description:
        tool.description.length > 100
          ? tool.description.substring(0, 97) + '...'
          : tool.description,
    });
  }

  const exampleToolName = useFullName
    ? (tools[0]?.name ?? 'tool_name')
    : getBaseToolName(tools[0]?.name ?? 'tool_name');

  const output = {
    listing_mode: true,
    servers,
    total_tools: tools.length,
    tools_by_server: toolsByServer,
    ...(notes.length > 0 ? { notes } : {}),
    hint: `To use a tool, search for it by name (e.g., query: "${exampleToolName}") to load it.`,
  };

  return JSON.stringify(output, null, 2);
}

/**
 * Creates a Tool Search tool for discovering tools from a large registry.
 *
 * This tool enables AI agents to dynamically discover tools from a large library
 * without loading all tool definitions into the LLM context window. The agent
 * can search for relevant tools on-demand.
 *
 * **Modes:**
 * - `code_interpreter` (default): Uses external sandbox for regex search. Safer for complex patterns.
 * - `local`: Uses safe substring matching locally. No network call, faster, completely safe from ReDoS.
 *
 * The tool registry can be provided either:
 * 1. At initialization time via params.toolRegistry
 * 2. At runtime via config.configurable.toolRegistry when invoking
 *
 * @param params - Configuration parameters for the tool (toolRegistry is optional)
 * @returns A LangChain DynamicStructuredTool for tool searching
 *
 * @example
 * // Option 1: Code interpreter mode (regex via sandbox)
 * const tool = createToolSearch({ toolRegistry });
 * await tool.invoke({ query: 'expense.*report' });
 *
 * @example
 * // Option 2: Local mode (safe substring search)
 * const tool = createToolSearch({ mode: 'local', toolRegistry });
 * await tool.invoke({ query: 'expense' });
 */
function createToolSearch(
  initParams: t.ToolSearchParams = {}
): DynamicStructuredTool {
  const mode: t.ToolSearchMode = initParams.mode ?? 'code_interpreter';
  const defaultOnlyDeferred = initParams.onlyDeferred ?? true;
  const mcpNameFormat: t.McpNameFormat = initParams.mcpNameFormat ?? 'full';
  const schema = createToolSearchSchema(mode);

  const baseEndpoint = initParams.baseUrl ?? getCodeBaseURL();
  const EXEC_ENDPOINT = buildCodeApiEndpoint(baseEndpoint, 'exec');

  const deferredToolsListing = getDeferredToolsListing(
    initParams.toolRegistry,
    defaultOnlyDeferred
  );

  const toolsListSection =
    deferredToolsListing.length > 0
      ? `

Deferred tools (search to load; \u2026 = has params, search first):
${deferredToolsListing}`
      : '';

  const mcpNote =
    deferredToolsListing.includes(Constants.MCP_DELIMITER) ||
    deferredToolsListing.split('\n').some((line) => !line.startsWith('other:'))
      ? `
- MCP tools use format: toolName${Constants.MCP_DELIMITER}serverName
- Use mcp_server param to filter by server`
      : '';

  const description =
    mode === 'local'
      ? `
Searches deferred tools using BM25 ranking. Multi-word queries supported.
${mcpNote}${toolsListSection}
`.trim()
      : `
Searches deferred tools by regex pattern.
${mcpNote}${toolsListSection}
`.trim();

  return tool(
    async (rawParams, config) => {
      const params = rawParams as ToolSearchParams;
      const {
        query = '',
        fields = ['name', 'description'],
        max_results = DEFAULT_MAX_RESULTS,
        mcp_server,
      } = params;

      const {
        toolRegistry: paramToolRegistry,
        onlyDeferred: paramOnlyDeferred,
        mcpServer: paramMcpServer,
      } = config.toolCall ?? {};

      const toolRegistry = paramToolRegistry ?? initParams.toolRegistry;
      const onlyDeferred =
        paramOnlyDeferred !== undefined
          ? paramOnlyDeferred
          : defaultOnlyDeferred;
      const rawServerFilter =
        mcp_server ?? paramMcpServer ?? initParams.mcpServer;
      const serverFilters = normalizeServerFilter(rawServerFilter);
      const hasServerFilter = serverFilters.length > 0;

      if (toolRegistry == null) {
        return [
          'Error: No tool registry provided. Configure toolRegistry at agent level or initialization.',
          {
            tool_references: [],
            metadata: {
              total_searched: 0,
              pattern: query,
              error: 'No tool registry provided',
            },
          },
        ];
      }

      const registeredServers = hasServerFilter ? new Set<string>() : undefined;
      const searchableServers = hasServerFilter ? new Set<string>() : undefined;
      /** The set to filter by is unknown until the whole registry has been
       * seen, so a server-filtered search has to stage its candidates. An
       * unfiltered search is the hot path: it builds the result during the same
       * single traversal, with no staging array and no copy of the registry. */
      const staged: t.LCTool[] | undefined = hasServerFilter ? [] : undefined;
      const deferredTools: t.ToolMetadata[] = [];
      let searchableCount = 0;

      const describeTool = (lcTool: t.LCTool): t.ToolMetadata => ({
        name: lcTool.name,
        description: lcTool.description ?? '',
        parameters: simplifyParametersForSearch(lcTool.parameters),
      });

      for (const lcTool of toolRegistry.values()) {
        let server: string | undefined;
        if (hasServerFilter) {
          server = extractMcpServerName(lcTool.name);
          if (server !== undefined && server !== '') {
            registeredServers?.add(server);
          } else {
            server = undefined;
          }
        }
        if (onlyDeferred === true && lcTool.defer_loading !== true) {
          continue;
        }
        searchableCount += 1;
        if (staged === undefined) {
          deferredTools.push(describeTool(lcTool));
          continue;
        }
        staged.push(lcTool);
        if (server !== undefined) {
          searchableServers?.add(server);
        }
      }

      const availableServers =
        searchableServers === undefined
          ? []
          : Array.from(searchableServers).sort();

      /** Resolve against every registered server, not just the searchable ones,
       * so a server whose tools are all loaded already is reported as having
       * nothing left to search instead of as a name that does not exist. */
      const { resolved: matchedServers, unresolved: unknownServers } =
        hasServerFilter
          ? resolveServerFilters(
            serverFilters,
            registeredServers === undefined
              ? []
              : Array.from(registeredServers)
          )
          : { resolved: [], unresolved: [] };

      const activeServers = matchedServers.filter(
        (name) => searchableServers?.has(name) === true
      );
      const idleServers = matchedServers.filter(
        (name) => searchableServers?.has(name) !== true
      );

      /** A filter naming several servers can resolve only partly. Saying so on
       * the successful path too stops "searched github and slack" being read
       * into a result that only ever covered github. */
      const filterNotes: string[] = [];
      if (unknownServers.length > 0) {
        filterNotes.push(
          `No MCP server matched: ${unknownServers.join(', ')}.`
        );
      }
      if (idleServers.length > 0) {
        filterNotes.push(
          `Registered with no tools left to search: ${idleServers.join(', ')}.`
        );
      }
      const filterNotice =
        filterNotes.length > 0 ? `Note: ${filterNotes.join(' ')}\n\n` : '';
      const filterMetadata = {
        ...(unknownServers.length > 0
          ? { unmatched_mcp_servers: unknownServers }
          : {}),
        ...(idleServers.length > 0 ? { idle_mcp_servers: idleServers } : {}),
      };

      if (staged !== undefined) {
        for (const lcTool of staged) {
          if (isFromAnyMcpServer(lcTool.name, activeServers)) {
            deferredTools.push(describeTool(lcTool));
          }
        }
      }

      if (deferredTools.length === 0) {
        /** Say which of the three it is, so an empty result is not read as an
         * outage: nothing is registered, the name did not match anything, or
         * the named server genuinely has no tools. */
        let message: string;
        /** The server diagnosis comes first: a registry holding only loaded
         * tools makes both conditions true, and naming the requested server is
         * the more useful of the two answers. */
        if (
          hasServerFilter &&
          (unknownServers.length > 0 || idleServers.length > 0)
        ) {
          const parts: string[] = [];
          if (unknownServers.length > 0) {
            parts.push(`No MCP server matched: ${unknownServers.join(', ')}.`);
          }
          if (idleServers.length > 0) {
            parts.push(
              `Registered with no tools left to search: ${idleServers.join(', ')}.`
            );
          }
          parts.push(
            `Server(s) with searchable tools: ${
              availableServers.length > 0
                ? availableServers.join(', ')
                : '(none)'
            }.`
          );
          message = parts.join(' ');
        } else if (searchableCount === 0) {
          message =
            'No tools available to search. The tool registry is empty or no deferred tools are registered.';
        } else {
          const serverMsg = hasServerFilter
            ? ` from MCP server(s): ${serverFilters.join(', ')}`
            : '';
          message = `No tools available to search${serverMsg}. The tool registry is empty or no matching deferred tools are registered.`;
        }
        return [
          message,
          {
            tool_references: [],
            metadata: {
              total_searched: 0,
              pattern: query,
              mcp_server: serverFilters,
              ...(hasServerFilter
                ? { available_mcp_servers: availableServers }
                : {}),
              ...filterMetadata,
            },
          },
        ];
      }

      const isServerListing = hasServerFilter && query === '';

      if (isServerListing) {
        /** The listing's contract is parseable JSON, so the diagnostics go
         * inside the payload rather than in front of it. */
        const formattedOutput = formatServerListing(
          deferredTools,
          activeServers,
          mcpNameFormat,
          filterNotes
        );

        return [
          formattedOutput,
          {
            tool_references: [],
            metadata: {
              total_available: deferredTools.length,
              mcp_server: serverFilters,
              listing_mode: true,
              ...filterMetadata,
            },
          },
        ];
      }

      if (mode === 'local') {
        const searchResponse = performLocalSearch(
          deferredTools,
          query,
          fields,
          max_results
        );
        /** Diagnostics ride inside the payload; this output is parsed. */
        const formattedOutput = formatSearchResults(
          searchResponse,
          mcpNameFormat,
          filterNotes
        );

        return [
          formattedOutput,
          {
            tool_references: searchResponse.tool_references,
            metadata: {
              total_searched: searchResponse.total_tools_searched,
              pattern: searchResponse.pattern_used,
              mcp_server: serverFilters.length > 0 ? serverFilters : undefined,
              ...filterMetadata,
            },
          },
        ];
      }

      const { safe: sanitizedPattern, wasEscaped } = sanitizeRegex(query);
      let warningMessage = '';
      if (wasEscaped) {
        warningMessage =
          'Note: The provided pattern was converted to a literal search for safety.\n\n';
      }

      const searchScript = generateSearchScript(
        deferredTools,
        fields,
        max_results,
        sanitizedPattern
      );

      const postData = {
        lang: 'js',
        code: searchScript,
        timeout: SEARCH_TIMEOUT,
      };

      try {
        const fetchOptions: RequestInit = {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'LibreChat/1.0',
          },
          body: JSON.stringify(postData),
        };

        const proxyAgent = resolveFetchProxyAgent(EXEC_ENDPOINT);
        if (proxyAgent) {
          fetchOptions.agent = proxyAgent;
        }

        const response = await fetch(EXEC_ENDPOINT, fetchOptions);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result: t.ExecuteResult = await response.json();

        if (result.stderr && result.stderr.trim()) {
          // eslint-disable-next-line no-console
          console.warn('[ToolSearch] stderr:', result.stderr);
        }

        if (!result.stdout || !result.stdout.trim()) {
          return [
            `${filterNotice}${warningMessage}No tools matched the pattern "${sanitizedPattern}".\nTotal tools searched: ${deferredTools.length}`,
            {
              tool_references: [],
              metadata: {
                total_searched: deferredTools.length,
                pattern: sanitizedPattern,
                ...filterMetadata,
              },
            },
          ];
        }

        const searchResponse = parseSearchResults(result.stdout);
        const formattedOutput = `${warningMessage}${formatSearchResults(searchResponse, mcpNameFormat, filterNotes)}`;

        return [
          formattedOutput,
          {
            tool_references: searchResponse.tool_references,
            metadata: {
              total_searched: searchResponse.total_tools_searched,
              pattern: searchResponse.pattern_used,
              ...filterMetadata,
            },
          },
        ];
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error('[ToolSearch] Error:', error);

        const errorMessage =
          error instanceof Error ? error.message : String(error);
        return [
          `Tool search failed: ${errorMessage}\n\nSuggestion: Try a simpler search pattern or search for specific tool names.`,
          {
            tool_references: [],
            metadata: {
              total_searched: 0,
              pattern: sanitizedPattern,
              error: errorMessage,
            },
          },
        ];
      }
    },
    {
      name: Constants.TOOL_SEARCH,
      description,
      schema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export {
  createToolSearch,
  performLocalSearch,
  extractMcpServerName,
  isFromMcpServer,
  isFromAnyMcpServer,
  resolveServerFilters,
  canonicalizeServerName,
  normalizeServerFilter,
  getAvailableMcpServers,
  getDeferredToolsListing,
  getBaseToolName,
  formatServerListing,
  sanitizeRegex,
  escapeRegexSpecialChars,
  isDangerousPattern,
  countNestedGroups,
  hasNestedQuantifiers,
};
