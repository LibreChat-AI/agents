import { basename, dirname } from 'path';
import { createTwoFilesPatch } from 'diff';
import { tool } from '@langchain/core/tools';
import type { DynamicStructuredTool } from '@langchain/core/tools';
import type * as t from '@/types';
import {
  commandAvailabilityEnvCacheKey,
  getSpawn,
  getWorkspaceFS,
  probeLocalCommandAvailability,
  resolveWorkspacePathSafe,
  setCommandAvailabilityCacheEntry,
  spawnLocalProcess,
  truncateLocalOutput,
} from './LocalExecutionEngine';
import {
  createLocalBashProgrammaticToolCallingTool,
  createLocalProgrammaticToolCallingTool,
} from './LocalProgrammaticToolCalling';
import {
  createLocalBashExecutionTool,
  createLocalCodeExecutionTool,
} from './LocalExecutionTools';
import {
  createCompileCheckTool,
  createCompileCheckToolDefinition,
} from './CompileCheckTool';
import { classifyAttachment, imageAttachmentContent } from './attachments';
import { createLocalFileCheckpointer } from './FileCheckpointer';
import { isWorkspaceClientTimeoutError } from './workspaceFS';
import { applyEdit, locateEdit } from './editStrategies';
import { decodeFile, encodeFile } from './textEncoding';
import { withIntent } from '@/tools/intentArg';
import { runPostEditSyntaxCheck } from './syntaxCheck';
import { capHeadTail, capLines } from './toolResultCaps';
import { Constants, LOCAL_CODING_BUNDLE_NAMES } from '@/common';

const MAX_READ_CHARS = 256000;
const DEFAULT_MAX_RESULTS = 200;
const DEFAULT_MAX_READ_BYTES = 10 * 1024 * 1024;
const BINARY_DETECTION_BYTES = 8000;

/**
 * Tool name aliases retained for back-compat with consumers that imported
 * the per-file `Local*ToolName` constants. The canonical names live on
 * `Constants.*` (see `src/common/enum.ts`); these aliases just point at
 * them so a typo upstream gets caught at the type level.
 */
export const LocalWriteFileToolName = Constants.WRITE_FILE;
export const LocalEditFileToolName = Constants.EDIT_FILE;
export const LocalGrepSearchToolName = Constants.GREP_SEARCH;
export const LocalGlobSearchToolName = Constants.GLOB_SEARCH;
export const LocalListDirectoryToolName = Constants.LIST_DIRECTORY;

export const LocalReadFileToolSchema: t.JsonSchemaType = withIntent({
  type: 'object',
  properties: {
    path: {
      type: 'string',
      description:
        'Path to a local file, relative to the configured cwd unless absolute paths are allowed.',
    },
    offset: {
      type: 'integer',
      description: 'Optional 1-indexed line offset for large files.',
    },
    limit: {
      type: 'integer',
      description: 'Optional maximum number of lines to return.',
    },
  },
  required: ['path'],
});

export const LocalWriteFileToolSchema: t.JsonSchemaType = withIntent({
  type: 'object',
  properties: {
    path: {
      type: 'string',
      description:
        'Path to write, relative to the configured cwd unless absolute paths are allowed.',
    },
    content: {
      type: 'string',
      description: 'Complete file contents to write.',
    },
  },
  required: ['path', 'content'],
});

export const LocalEditFileToolSchema: t.JsonSchemaType = withIntent({
  type: 'object',
  properties: {
    path: {
      type: 'string',
      description:
        'Path to edit, relative to the configured cwd unless absolute paths are allowed.',
    },
    old_text: {
      type: 'string',
      description: 'Exact text to replace. Must appear exactly once.',
    },
    new_text: {
      type: 'string',
      description: 'Replacement text.',
    },
    edits: {
      type: 'array',
      description:
        'Optional batch of exact replacements. Each old_text must appear exactly once in the original file.',
      items: {
        type: 'object',
        properties: {
          old_text: { type: 'string' },
          new_text: { type: 'string' },
        },
        required: ['old_text', 'new_text'],
      },
    },
  },
  required: ['path'],
});

export const LocalGrepSearchToolSchema: t.JsonSchemaType = withIntent({
  type: 'object',
  properties: {
    pattern: {
      type: 'string',
      description: 'Regex pattern to search for.',
    },
    path: {
      type: 'string',
      description: 'Directory or file to search. Defaults to cwd.',
    },
    glob: {
      type: 'string',
      description: 'Optional file glob passed to rg -g.',
    },
    max_results: {
      type: 'integer',
      description: 'Maximum matching lines to return.',
    },
  },
  required: ['pattern'],
});

export const LocalGlobSearchToolSchema: t.JsonSchemaType = withIntent({
  type: 'object',
  properties: {
    pattern: {
      type: 'string',
      description: 'File glob pattern, for example "src/**/*.ts".',
    },
    path: {
      type: 'string',
      description: 'Directory to search. Defaults to cwd.',
    },
    max_results: {
      type: 'integer',
      description: 'Maximum file paths to return.',
    },
  },
  required: ['pattern'],
});

export const LocalListDirectoryToolSchema: t.JsonSchemaType = withIntent({
  type: 'object',
  properties: {
    path: {
      type: 'string',
      description: 'Directory to list. Defaults to cwd.',
    },
  },
});

function lineWindow(
  content: string,
  offset?: number,
  limit?: number
): { text: string; truncated: boolean } {
  const start = Math.max((offset ?? 1) - 1, 0);
  // Avoid splitting the whole file when the caller asked for a small
  // window. For a 10 MB file with `offset: 1, limit: 10`, the prior
  // `content.split('\n')` allocated millions of strings to throw all
  // but 10 away. We walk newline indices directly: O(start + limit)
  // instead of O(file). When `limit` is omitted, fall back to the
  // simple split — it's the same amount of work either way.
  if (limit == null || limit <= 0) {
    const lines = content.split('\n');
    const selected = lines.slice(start);
    const numbered = selected
      .map(
        (line, index) =>
          `${String(start + index + 1).padStart(6, ' ')}\t${line}`
      )
      .join('\n');
    return {
      text: truncateLocalOutput(numbered, MAX_READ_CHARS),
      truncated: numbered.length > MAX_READ_CHARS,
    };
  }
  // Walk to the start line by counting newlines.
  let cursor = 0;
  for (let i = 0; i < start; i++) {
    const next = content.indexOf('\n', cursor);
    if (next === -1) {
      // File has fewer lines than `offset` — return empty window.
      return { text: '', truncated: false };
    }
    cursor = next + 1;
  }
  // Collect up to `limit` lines from `cursor`.
  const out: string[] = [];
  let pos = cursor;
  let exhausted = true;
  for (let k = 0; k < limit; k++) {
    const next = content.indexOf('\n', pos);
    if (next === -1) {
      out.push(content.slice(pos));
      break;
    }
    out.push(content.slice(pos, next));
    pos = next + 1;
    if (k === limit - 1 && pos < content.length) {
      exhausted = false;
    }
  }
  const numbered = out
    .map(
      (text, index) => `${String(start + index + 1).padStart(6, ' ')}\t${text}`
    )
    .join('\n');
  return {
    text: truncateLocalOutput(numbered, MAX_READ_CHARS),
    truncated: !exhausted || numbered.length > MAX_READ_CHARS,
  };
}

/** Reads whole lines from `offset` until `maxChars` or `limit` lines, then says how to continue. */
function cappedLineWindow(
  content: string,
  offset: number | undefined,
  limit: number | undefined,
  maxChars: number
): string {
  const start = Math.max(offset ?? 1, 1);
  // Walk newline indices instead of splitting the file, so a page costs
  // O(offset + page) rather than O(file).
  let cursor = 0;
  for (let line = 1; line < start; line++) {
    const next = content.indexOf('\n', cursor);
    if (next === -1) return '';
    cursor = next + 1;
  }
  const maxLines = limit != null && limit > 0 ? limit : Infinity;
  const kept: string[] = [];
  let used = 0;
  while (cursor <= content.length && kept.length < maxLines) {
    const newline = content.indexOf('\n', cursor);
    const lineEnd = newline === -1 ? content.length : newline;
    const text = content.slice(cursor, Math.min(lineEnd, cursor + maxChars));
    const full = `${String(start + kept.length).padStart(6, ' ')}\t${text}`;
    const numbered =
      lineEnd - cursor > maxChars || full.length > maxChars
        ? `${full.slice(0, maxChars)} [line truncated]`
        : full;
    if (used + numbered.length + 1 > maxChars && kept.length > 0) break;
    kept.push(numbered);
    used += numbered.length + 1;
    if (newline === -1) {
      cursor = content.length + 1;
      break;
    }
    cursor = newline + 1;
  }
  const end = start + kept.length - 1;
  return cursor <= content.length
    ? `${kept.join('\n')}\n[showing lines ${start}-${end}; continue with offset=${end + 1}]`
    : kept.join('\n');
}

const MAX_DIFF_CHARS = 4000;

type SyntaxRun =
  | {
      mode: 'auto' | 'strict';
      outcome: import('./syntaxCheck').SyntaxCheckOutcome;
    }
  | undefined;

async function maybeRunSyntaxCheck(
  path: string,
  config: t.LocalExecutionConfig
): Promise<SyntaxRun> {
  const mode = config.postEditSyntaxCheck ?? 'off';
  if (mode === 'off') return undefined;
  const outcome = await runPostEditSyntaxCheck(path, config);
  if (outcome == null) return undefined;
  return { mode, outcome };
}

function appendSyntaxCheckSummary(base: string, run: SyntaxRun): string {
  if (run == null) return base;
  if (run.outcome.ok) return base;
  const banner =
    run.mode === 'strict'
      ? `\n\n[syntax-check FAILED via ${run.outcome.checker}]\n`
      : `\n\n[syntax-check warning via ${run.outcome.checker}]\n`;
  return `${base}${banner}${run.outcome.output}`;
}

/**
 * Revert a write_file/edit_file mutation in `postEditSyntaxCheck:
 * 'strict'` mode after the post-write syntax check failed. Strict
 * mode advertises a safety gate, so leaving the corrupted file on
 * disk + throwing is a half-broken contract — the model "reacts" to
 * the error but the next call sees broken on-disk state. Codex P2
 * [49]. Best-effort: a swallowed error here means the workspace is
 * still in the bad post-write state, but we still throw the
 * original syntax-check error so the caller knows.
 *
 * - If the file existed pre-write: restore the previous bytes with
 *   the original encoding.
 * - If the file is brand-new: unlink it.
 */
async function revertStrictWrite(
  fs: import('./workspaceFS').WorkspaceFS,
  path: string,
  existed: boolean,
  before: string,
  encoding: { text: string; hasBom: boolean; newline: '\n' | '\r\n' }
): Promise<void> {
  try {
    if (existed) {
      // encodeFile uses encoding.{hasBom,newline} to restore the
      // on-disk shape; the `text` field is overridden by the
      // explicit `before` arg we pass in.
      await fs.writeFile(
        path,
        encodeFile(before, { ...encoding, text: before }),
        'utf8'
      );
    } else {
      await fs.unlink(path);
    }
  } catch (error) {
    // A timed-out revert leaves the rejected write on disk — the caller would
    // otherwise claim "reverted to pre-write state" while the bad bytes remain.
    // Surface it; for other failures stay best-effort (caller sees the original
    // syntax error).
    if (isWorkspaceClientTimeoutError(error)) {
      throw error;
    }
  }
}

function summariseDiff(
  filePath: string,
  before: string,
  after: string
): string {
  if (before === after) {
    return '(no textual changes)';
  }
  const name = basename(filePath);
  const patch = createTwoFilesPatch(name, name, before, after, '', '', {
    context: 3,
  });
  if (patch.length <= MAX_DIFF_CHARS) {
    return patch;
  }
  return (
    patch.slice(0, MAX_DIFF_CHARS) +
    `\n[... diff truncated, ${patch.length - MAX_DIFF_CHARS} more chars ...]`
  );
}

function normalizeEdits(input: {
  old_text?: string;
  new_text?: string;
  edits?: Array<{ old_text?: string; new_text?: string }>;
}): Array<{ oldText: string; newText: string }> {
  const edits = Array.isArray(input.edits)
    ? input.edits.map((edit) => ({
      oldText: edit.old_text ?? '',
      newText: edit.new_text ?? '',
    }))
    : [];

  if (input.old_text != null || input.new_text != null) {
    edits.push({
      oldText: input.old_text ?? '',
      newText: input.new_text ?? '',
    });
  }

  return edits;
}

function toolDefinition(
  name: string,
  description: string,
  parameters: t.JsonSchemaType
): t.LCTool {
  return {
    name,
    description,
    parameters,
    allowed_callers: ['direct', 'code_execution'],
    responseFormat: Constants.CONTENT_AND_ARTIFACT,
    toolType: 'builtin',
  };
}

async function looksBinary(
  path: string,
  fs: import('./workspaceFS').WorkspaceFS
): Promise<boolean> {
  let handle;
  try {
    handle = await fs.open(path, 'r');
    const sample = Buffer.alloc(BINARY_DETECTION_BYTES);
    const { bytesRead } = await handle.read(
      sample,
      0,
      BINARY_DETECTION_BYTES,
      0
    );
    for (let i = 0; i < bytesRead; i++) {
      if (sample[i] === 0) {
        return true;
      }
    }
    return false;
  } finally {
    await handle?.close();
  }
}

const DEFAULT_MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024;

export function createLocalReadFileTool(
  config: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  const fs = getWorkspaceFS(config);
  return tool(
    async (rawInput) => {
      const input = rawInput as {
        path: string;
        offset?: number;
        limit?: number;
      };
      const path = await resolveWorkspacePathSafe(
        input.path,
        config,
        'read'
      );
      const fileStat = await fs.stat(path);
      if (!fileStat.isFile()) {
        throw new Error(`Path is not a file: ${input.path}`);
      }
      const maxBytes = Math.max(
        config.maxReadBytes ?? DEFAULT_MAX_READ_BYTES,
        1
      );
      if (fileStat.size > maxBytes) {
        const stub = `File is ${fileStat.size} bytes, exceeds the ${maxBytes}-byte read cap. Read a slice via bash (e.g. head/sed) or raise local.maxReadBytes.`;
        return [stub, { path, bytes: fileStat.size, truncated: true }];
      }

      if (await looksBinary(path, fs)) {
        const attachmentMode = config.attachReadAttachments ?? 'off';
        if (attachmentMode !== 'off') {
          const attachment = await classifyAttachment({
            path,
            bytes: fileStat.size,
            mode: attachmentMode,
            maxBytes: config.maxAttachmentBytes ?? DEFAULT_MAX_ATTACHMENT_BYTES,
            // Route through the configured WorkspaceFS so a custom
            // engine sees the same path semantics as `read_file`
            // itself (manual review finding F).
            fs,
          });
          if (attachment.kind === 'image') {
            return [
              imageAttachmentContent(path, attachment),
              {
                path,
                bytes: fileStat.size,
                mime: attachment.mime,
                attachment: 'image',
              },
            ];
          }
          if (attachment.kind === 'pdf') {
            return [
              [
                {
                  type: 'text',
                  text: `Read ${path} (application/pdf, ${fileStat.size} bytes). PDF attached as base64 data URL; vision-capable models that accept PDF will render it.`,
                },
                {
                  type: 'image_url',
                  image_url: { url: attachment.dataUrl },
                },
              ],
              {
                path,
                bytes: fileStat.size,
                mime: attachment.mime,
                attachment: 'pdf',
              },
            ];
          }
          if (attachment.kind === 'oversize') {
            return [
              `Refusing to embed ${attachment.mime} attachment (${attachment.bytes} bytes exceeds ${attachment.maxBytes}-byte cap).`,
              {
                path,
                bytes: fileStat.size,
                mime: attachment.mime,
                attachment: 'oversize',
              },
            ];
          }
          if (attachment.kind === 'binary') {
            return [
              `Refusing to read binary file (${fileStat.size} bytes, ${attachment.mime}): ${path}`,
              {
                path,
                bytes: fileStat.size,
                mime: attachment.mime,
                binary: true,
              },
            ];
          }
          // text-or-unknown falls through to the text-read path below.
        } else {
          return [
            `Refusing to read binary file (${fileStat.size} bytes): ${path}`,
            { path, bytes: fileStat.size, binary: true },
          ];
        }
      }

      const content = await fs.readFile(path, 'utf8');
      if (config.toolResultMaxChars != null && config.toolResultMaxChars > 0) {
        return [
          cappedLineWindow(
            content,
            input.offset,
            input.limit,
            config.toolResultMaxChars
          ),
          { path, bytes: fileStat.size },
        ];
      }
      const result = lineWindow(content, input.offset, input.limit);
      return [
        result.truncated ? `${result.text}\n[truncated]` : result.text,
        { path, bytes: fileStat.size },
      ];
    },
    {
      name: Constants.READ_FILE,
      description:
        'Read a local text file from the configured working directory with line numbers. ' +
        'When `attachReadAttachments` is enabled (e.g. images-only), reading an image returns an ' +
        '`image_url` content block so vision-capable models can see the file directly.',
      schema: LocalReadFileToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export function createLocalWriteFileTool(
  config: t.LocalExecutionConfig = {},
  checkpointer?: t.LocalFileCheckpointer
): DynamicStructuredTool {
  const fs = getWorkspaceFS(config);
  return tool(
    async (rawInput) => {
      const input = rawInput as { path: string; content: string };
      if (config.readOnly === true) {
        throw new Error('write_file is blocked in read-only local mode.');
      }
      const path = await resolveWorkspacePathSafe(
        input.path,
        config,
        'write'
      );
      if (checkpointer != null) {
        await checkpointer.captureBeforeWrite(path);
      }

      let before = '';
      let encoding = { text: '', hasBom: false, newline: '\n' as const } as
        | ReturnType<typeof decodeFile>
        | { text: string; hasBom: false; newline: '\n' };
      let existed = false;
      try {
        const raw = await fs.readFile(path, 'utf8');
        const decoded = decodeFile(raw);
        before = decoded.text;
        encoding = decoded;
        existed = true;
      } catch (error) {
        // A stalled-RPC timeout is NOT "file absent" — treating it as such would
        // overwrite an existing file with fresh content. Surface it instead.
        if (isWorkspaceClientTimeoutError(error)) {
          throw error;
        }
        existed = false;
      }

      await fs.mkdir(dirname(path), { recursive: true });
      const finalText = encodeFile(input.content, encoding);
      await fs.writeFile(path, finalText, 'utf8');

      let syntax;
      try {
        syntax = await maybeRunSyntaxCheck(path, config);
      } catch (error) {
        // A validation-read timeout must still honor strict mode's revert
        // contract: restore the pre-write state before surfacing (best-effort —
        // the revert may itself time out on the same stalled container).
        if (
          isWorkspaceClientTimeoutError(error) &&
          config.postEditSyntaxCheck === 'strict'
        ) {
          await revertStrictWrite(fs, path, existed, before, encoding);
        }
        throw error;
      }

      const diff = existed
        ? summariseDiff(path, before, input.content)
        : `(new file, ${input.content.length} chars)`;
      const baseSummary = existed
        ? `Overwrote ${path} (${input.content.length} chars). Diff:\n${diff}`
        : `Created ${path} (${input.content.length} chars).`;
      const summary = appendSyntaxCheckSummary(baseSummary, syntax);
      if (syntax?.outcome.ok === false && syntax.mode === 'strict') {
        // Roll back the write so strict mode is an actual gate, not
        // "fail the call AND leave the corrupted file on disk".
        // Codex P2 [49].
        await revertStrictWrite(fs, path, existed, before, encoding);
        throw new Error(
          `write_file syntax check failed (${syntax.outcome.checker}); reverted to pre-write state.\n${syntax.outcome.output}`
        );
      }
      return [
        summary,
        {
          path,
          bytes: finalText.length,
          new_file: !existed,
          newline: encoding.newline === '\r\n' ? 'CRLF' : 'LF',
          had_bom: encoding.hasBom,
          ...(syntax != null && syntax.outcome.ok === false
            ? { syntax_error: syntax.outcome.checker }
            : {}),
        },
      ];
    },
    {
      name: LocalWriteFileToolName,
      description:
        'Create or overwrite a local text file in the configured working directory. ' +
        'Preserves the existing BOM and line endings when overwriting; defaults to LF without BOM for new files. ' +
        'Returns a unified diff of the changes when overwriting.',
      schema: LocalWriteFileToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export function createLocalEditFileTool(
  config: t.LocalExecutionConfig = {},
  checkpointer?: t.LocalFileCheckpointer
): DynamicStructuredTool {
  const fs = getWorkspaceFS(config);
  return tool(
    async (rawInput) => {
      const input = rawInput as {
        path: string;
        old_text?: string;
        new_text?: string;
        edits?: Array<{ old_text?: string; new_text?: string }>;
      };
      if (config.readOnly === true) {
        throw new Error('edit_file is blocked in read-only local mode.');
      }
      const edits = normalizeEdits(input);
      if (edits.length === 0) {
        throw new Error('edit_file requires old_text/new_text or edits[].');
      }

      const path = await resolveWorkspacePathSafe(
        input.path,
        config,
        'write'
      );
      const raw = await fs.readFile(path, 'utf8');
      const encoding = decodeFile(raw);
      const original = encoding.text;

      let next = original;
      const strategiesUsed: string[] = [];
      for (let i = 0; i < edits.length; i++) {
        const edit = edits[i];
        const match = locateEdit(next, edit.oldText);
        if (match == null) {
          throw new Error(
            `Edit ${i + 1}/${edits.length}: could not locate old_text in ${input.path}. ` +
              'Tried exact, line-trimmed, whitespace-normalized, and indentation-flexible matching. ' +
              'Re-read the file and copy the literal lines.'
          );
        }
        strategiesUsed.push(match.strategy);
        next = applyEdit(next, match, edit.newText);
      }

      if (checkpointer != null) {
        await checkpointer.captureBeforeWrite(path);
      }
      const finalText = encodeFile(next, encoding);
      await fs.writeFile(path, finalText, 'utf8');

      let syntax;
      try {
        syntax = await maybeRunSyntaxCheck(path, config);
      } catch (error) {
        // As in write_file: a validation-read timeout still triggers strict
        // mode's revert (edit_file always operates on an existing file).
        if (
          isWorkspaceClientTimeoutError(error) &&
          config.postEditSyntaxCheck === 'strict'
        ) {
          await revertStrictWrite(fs, path, true, original, encoding);
        }
        throw error;
      }

      const diff = summariseDiff(path, original, next);
      const fuzzy = strategiesUsed.some((s) => s !== 'exact');
      const baseSummary =
        `Applied ${edits.length} edit(s) to ${path}` +
        (fuzzy ? ` (strategies: ${strategiesUsed.join(', ')})` : '') +
        `. Diff:\n${diff}`;
      const summary = appendSyntaxCheckSummary(baseSummary, syntax);
      if (syntax?.outcome.ok === false && syntax.mode === 'strict') {
        // Restore the pre-edit bytes so strict mode is an actual
        // gate (Codex P2 [49]). edit_file always operates on an
        // existing file, so `existed = true` here.
        await revertStrictWrite(fs, path, true, original, encoding);
        throw new Error(
          `edit_file syntax check failed (${syntax.outcome.checker}); reverted to pre-edit state.\n${syntax.outcome.output}`
        );
      }
      return [
        summary,
        {
          path,
          edits: edits.length,
          strategies: strategiesUsed,
          newline: encoding.newline === '\r\n' ? 'CRLF' : 'LF',
          had_bom: encoding.hasBom,
          ...(syntax != null && syntax.outcome.ok === false
            ? { syntax_error: syntax.outcome.checker }
            : {}),
        },
      ];
    },
    {
      name: LocalEditFileToolName,
      description:
        'Apply exact text replacements to a local file. The matcher tries exact, line-trimmed, whitespace-normalized, and indentation-flexible strategies in order so common LLM whitespace mistakes are recoverable. Each old_text must still match exactly one location. Returns a unified diff of the changes.',
      schema: LocalEditFileToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

/**
 * Ripgrep availability cache, keyed on the *effective execution
 * backend* — whatever function `getSpawn(config)` returns. Without
 * the backend key, a Run that probes `rg` over Node's
 * `child_process.spawn` would poison subsequent Runs whose
 * `local.exec.spawn` routes to a remote sandbox or container that
 * doesn't have rg installed: the cached `true` would skip the probe,
 * the rg invocation would throw, and the Node fallback wouldn't be
 * reached. Per-backend caching avoids that without paying for a
 * spawn-per-search.
 */
// Per-backend × per-env cache. Codex P1 #34 — keying by spawn
// backend alone misses the case where two Runs share a backend but vary
// `local.env`. The inner map uses non-plaintext environment hashes and is
// bounded so stable worlds do not retain unbounded variants or raw secrets.
let ripgrepAvailabilityByBackend = new WeakMap<
  t.LocalSpawn,
  Map<string, ReturnType<typeof probeLocalCommandAvailability>>
>();

async function isRipgrepAvailable(
  config: t.LocalExecutionConfig
): Promise<boolean> {
  const backend = getSpawn(config);
  let envMap = ripgrepAvailabilityByBackend.get(backend);
  if (envMap == null) {
    envMap = new Map();
    ripgrepAvailabilityByBackend.set(backend, envMap);
  }
  const envKey = commandAvailabilityEnvCacheKey(config.env);
  let probePromise = envMap.get(envKey);
  if (probePromise == null) {
    probePromise = probeLocalCommandAvailability('rg', ['--version'], config);
    setCommandAvailabilityCacheEntry(envMap, envKey, probePromise);
  }
  const result = await probePromise;
  if (!result.cacheable && envMap.get(envKey) === probePromise) {
    envMap.delete(envKey);
  }
  if (result.cacheUntil != null && result.cacheUntil <= Date.now()) {
    if (envMap.get(envKey) === probePromise) {
      envMap.delete(envKey);
    }
    return isRipgrepAvailable(config);
  }
  return result.available;
}

/**
 * Test-only reset hook. Clears the ripgrep-availability cache so
 * tests can swap in mocked spawn backends and reprobe deterministically.
 *
 * @internal Not part of the public SDK surface; the leading underscore
 *   and `@internal` tag together signal that consumers should not call
 *   this. Tests import it via the module path directly.
 */
export function _resetRipgrepCacheForTests(): void {
  ripgrepAvailabilityByBackend = new WeakMap();
}

// Skipped by the Node-fallback walker (used when ripgrep is
// unavailable). Covers common build outputs, virtualenvs, and
// caches so a `grep_search`/`glob_search` on a large monorepo or a
// Python project with `.venv/` doesn't read every file under those
// trees. ripgrep itself respects .gitignore so it doesn't need this
// list. Audit follow-up from the comprehensive review (finding #3).
const SKIP_DIRS = new Set([
  '.git',
  '.svn',
  '.hg',
  'node_modules',
  '.next',
  '.nuxt',
  '.cache',
  '.parcel-cache',
  '.turbo',
  'dist',
  'build',
  'out',
  'target',
  'vendor',
  'coverage',
  '.nyc_output',
  '__pycache__',
  '.venv',
  'venv',
  'env',
  '.tox',
  '.mypy_cache',
  '.pytest_cache',
  '.ruff_cache',
]);

function globToRegExp(pattern: string): RegExp {
  let result = '^';
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i];
    if (c === '*') {
      if (pattern[i + 1] === '*') {
        result += '.*';
        i += 1;
        if (pattern[i + 1] === '/') {
          i += 1;
        }
      } else {
        result += '[^/]*';
      }
    } else if (c === '?') {
      result += '[^/]';
    } else if ('.+^$|(){}[]\\'.includes(c)) {
      result += '\\' + c;
    } else {
      result += c;
    }
  }
  result += '$';
  return new RegExp(result);
}

async function* walkFiles(
  root: string,
  fs: import('./workspaceFS').WorkspaceFS
): AsyncGenerator<string> {
  const stack: string[] = [root];
  while (stack.length > 0) {
    const dir = stack.pop() as string;
    let entries;
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch (error) {
      // A stalled-RPC timeout must surface, not be silently skipped as an
      // unreadable directory — that would corrupt search results ("no matches").
      if (isWorkspaceClientTimeoutError(error)) {
        throw error;
      }
      continue;
    }
    for (const entry of entries) {
      if (entry.name.startsWith('.git') || SKIP_DIRS.has(entry.name)) {
        continue;
      }
      const full = `${dir}/${entry.name}`;
      if (entry.isDirectory()) {
        stack.push(full);
      } else if (entry.isFile()) {
        yield full;
      }
    }
  }
}

/**
 * Catastrophic-backtracking guardrails for the fallback grep path.
 *
 * Without ripgrep we run the model-supplied pattern through Node's
 * `RegExp` engine, which uses a backtracking implementation. Patterns
 * with nested unbounded quantifiers (`(a+)+`, `(.*)*`, etc.) can
 * monopolise the event loop for arbitrary wall-clock time on
 * pathological input, and `setTimeout` cannot interrupt a synchronous
 * `RegExp.exec`. Manual review (finding D) flagged this as a real DoS.
 *
 * Mitigations applied here, in order of severity:
 *   1. Cap pattern length so an obviously oversize regex is rejected
 *      before compile.
 *   2. Reject patterns that contain a nested unbounded quantifier of
 *      the form `(...+|*)([+*]|{n,})` — the standard pathological
 *      shape. Still a heuristic (not a full safety proof), but blocks
 *      every common DoS construction we've seen in coding-agent logs.
 *   3. Wall-clock budget for the overall search: each file's regex
 *      pass is checked against a deadline; once exceeded the search
 *      bails with a partial result. Doesn't interrupt a stuck
 *      `exec()` call, but stops a slow pattern from making the whole
 *      Run hang once the first hung file finishes.
 *
 * Hosts that need bulletproof regex safety should install `rg` —
 * ripgrep uses RE2 internally and has no backtracking.
 */
const MAX_FALLBACK_PATTERN_LENGTH = 1024;
const FALLBACK_GREP_BUDGET_MS = 5000;
// Per-file byte cap. Codex P2 #41 — without it, the whole-file
// `readFile` + `split('\n')` for a multi-GB log is an unbounded
// allocation that the wall-clock budget (checked between files)
// can't interrupt. Hosts that need to grep large files should
// install ripgrep.
const FALLBACK_GREP_MAX_FILE_BYTES = 5 * 1024 * 1024;

/**
 * Heuristic: walks `pattern` to find any `(<contents>)<quant>` where
 * `<contents>` itself has an unbounded quantifier. Catches the
 * classic `(a+)+` form AND the double-nested `((a+)+)` form (which a
 * single-pass regex misses because `[^)]*` stops at the first inner
 * close-paren). Misses sufficiently obfuscated cases — bulletproof
 * ReDoS detection requires a real parser. The 5 s wall-clock budget
 * is the hard backstop for anything this slip past.
 */
function hasNestedUnboundedQuantifier(pattern: string): boolean {
  for (let i = 1; i < pattern.length - 1; i++) {
    if (pattern[i] !== ')') continue;
    if (pattern[i - 1] === '\\') continue;
    const next = pattern[i + 1];
    if (next !== '+' && next !== '*' && next !== '{') continue;
    // Walk back to find the matching opening paren (respecting depth
    // and `\(` escapes).
    let depth = 1;
    let j = i - 1;
    while (j >= 0) {
      const c = pattern[j];
      const escaped = j > 0 && pattern[j - 1] === '\\';
      if (!escaped) {
        if (c === ')') depth++;
        else if (c === '(') {
          depth--;
          if (depth === 0) break;
        }
      }
      j--;
    }
    if (j < 0) continue;
    const inner = pattern.slice(j + 1, i);
    if (/(?<!\\)[+*]/.test(inner)) return true;
  }
  return false;
}

class FallbackGrepError extends Error {
  readonly kind: 'pattern-too-long' | 'unsafe-pattern' | 'invalid-pattern';
  constructor(
    kind: 'pattern-too-long' | 'unsafe-pattern' | 'invalid-pattern',
    message: string
  ) {
    super(message);
    this.kind = kind;
  }
}

function compileFallbackRegex(pattern: string): RegExp {
  if (pattern.length > MAX_FALLBACK_PATTERN_LENGTH) {
    throw new FallbackGrepError(
      'pattern-too-long',
      `Pattern exceeds ${MAX_FALLBACK_PATTERN_LENGTH}-char fallback cap (install ripgrep for unbounded patterns).`
    );
  }
  if (hasNestedUnboundedQuantifier(pattern)) {
    throw new FallbackGrepError(
      'unsafe-pattern',
      'Pattern contains a nested unbounded quantifier (e.g. `(a+)+` or `((a+)+)`) which can cause catastrophic backtracking in the Node fallback. Install ripgrep for RE2-safe matching.'
    );
  }
  try {
    return new RegExp(pattern);
  } catch (e) {
    throw new FallbackGrepError(
      'invalid-pattern',
      `Invalid regex: ${(e as Error).message}`
    );
  }
}

/** Structured return so callers can count matches separately from
 * diagnostic skip-sentinels (Codex P2 [43]). */
type FallbackGrepResult = { matches: string[]; skipped: string[] };

/** Renders fallback-grep output: real matches first, skip diagnostics appended. */
function formatFallbackGrepDisplay(result: FallbackGrepResult): string {
  if (result.matches.length > 0) {
    return [...result.matches, ...result.skipped].join('\n');
  }
  if (result.skipped.length > 0) {
    return result.skipped.join('\n');
  }
  return 'No matches found.';
}

/**
 * Fallback grep display under `toolResultMaxChars`: skip notices get at most
 * a fifth of the budget (with a count of the ones left out), matches get the
 * rest, so neither can push the result past the cap or hide the other.
 */
function capFallbackGrepDisplay(
  matches: string[],
  skipped: string[],
  maxChars: number
): string {
  const skipText =
    skipped.length === 0
      ? ''
      : capLines(
        skipped.join('\n'),
        Math.max(1, Math.floor(maxChars / 5)),
        (omitted) => `${omitted} more files skipped`
      );
  if (matches.length === 0) {
    return skipText === '' ? 'No matches found.' : skipText;
  }
  const matchText = capLines(
    matches.join('\n'),
    Math.max(1, maxChars - skipText.length)
  );
  return skipText === '' ? matchText : `${matchText}\n${skipText}`;
}

async function fallbackGrep(
  root: string,
  pattern: string,
  globFilter: string | undefined,
  maxResults: number,
  fs: import('./workspaceFS').WorkspaceFS
): Promise<FallbackGrepResult> {
  const rx = compileFallbackRegex(pattern);
  const deadline = Date.now() + FALLBACK_GREP_BUDGET_MS;
  const globRx =
    globFilter != null && globFilter !== ''
      ? globToRegExp(globFilter)
      : undefined;
  const matches: string[] = [];
  // Track skipped (oversize) files separately so they don't consume
  // the maxResults budget. Codex P2 [43]: round 14's fix pushed skip
  // sentinels into `matches`, so a directory of one oversize non-
  // matching file falsely reported `matches: 1`, and enough
  // oversize files could fill the budget before any real match was
  // scanned. Now diagnostics are appended after real matches and
  // independent of the budget.
  const skippedDiagnostics: string[] = [];
  for await (const file of walkFiles(root, fs)) {
    if (Date.now() > deadline) {
      // Wall-clock budget exceeded — return partial results rather
      // than letting a slow pattern hang the Run.
      return { matches, skipped: skippedDiagnostics };
    }
    if (globRx != null) {
      const rel = file.startsWith(root + '/')
        ? file.slice(root.length + 1)
        : file;
      if (!globRx.test(rel)) {
        continue;
      }
    }
    // Skip files larger than the per-file cap and remember them as
    // diagnostics (NOT as matches). Codex P2 [41]: pre-fix
    // `fs.readFile` then `.split('\n')` allocated the whole file +
    // an array of every line, which a single multi-GB log could
    // turn into an OOM even after the regex DoS guards.
    let stat;
    try {
      stat = await fs.stat(file);
    } catch (error) {
      // A stalled-RPC timeout must surface, not be skipped — skipping a file can
      // report "no matches" even when the (unreadable-due-to-stall) file matched.
      if (isWorkspaceClientTimeoutError(error)) {
        throw error;
      }
      continue;
    }
    if (stat.size > FALLBACK_GREP_MAX_FILE_BYTES) {
      skippedDiagnostics.push(
        `${file}:0:[skipped: file > ${FALLBACK_GREP_MAX_FILE_BYTES} bytes; install ripgrep for unbounded grep]`
      );
      continue;
    }
    let content;
    try {
      content = await fs.readFile(file, 'utf8');
    } catch (error) {
      // As above: a client-timeout means a stalled read, not an unreadable file.
      if (isWorkspaceClientTimeoutError(error)) {
        throw error;
      }
      continue;
    }
    if (content.includes('\0')) {
      continue;
    }
    // Re-check the deadline AFTER the read — a slow disk on one
    // file can blow the budget without us noticing.
    if (Date.now() > deadline) {
      return { matches, skipped: skippedDiagnostics };
    }
    const lines = content.split('\n');
    for (let i = 0; i < lines.length; i++) {
      if (rx.test(lines[i])) {
        matches.push(`${file}:${i + 1}:${lines[i]}`);
        if (matches.length >= maxResults) {
          return { matches, skipped: skippedDiagnostics };
        }
      }
    }
  }
  return { matches, skipped: skippedDiagnostics };
}

async function fallbackGlob(
  root: string,
  pattern: string,
  maxResults: number,
  fs: import('./workspaceFS').WorkspaceFS
): Promise<string[]> {
  const rx = globToRegExp(pattern);
  const out: string[] = [];
  for await (const file of walkFiles(root, fs)) {
    const rel = file.startsWith(root + '/')
      ? file.slice(root.length + 1)
      : file;
    if (rx.test(rel)) {
      out.push(file);
      if (out.length >= maxResults) {
        break;
      }
    }
  }
  return out;
}

export function createLocalGrepSearchTool(
  config: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  const fs = getWorkspaceFS(config);
  return tool(
    async (rawInput) => {
      const input = rawInput as {
        pattern: string;
        path?: string;
        glob?: string;
        max_results?: number;
      };
      const target = await resolveWorkspacePathSafe(
        input.path ?? '.',
        config,
        'read'
      );
      const maxResults = Math.max(input.max_results ?? DEFAULT_MAX_RESULTS, 1);

      if (await isRipgrepAvailable(config)) {
        // Pass the pattern through `-e` so dash-prefixed patterns
        // like `-foo` are treated as the search regex, not as a
        // (probably-unknown) flag. `rg --help` explicitly requires
        // `-e/--regexp` (or `--`) for that case. Same trick avoids
        // any future flag-conflict if a user query happens to look
        // like an rg long option.
        const args = [
          '--line-number',
          '--column',
          '--hidden',
          '--glob',
          '!.git/**',
          ...(input.glob != null && input.glob !== ''
            ? ['--glob', input.glob]
            : []),
          '-e',
          input.pattern,
          target,
        ];
        const result = await spawnLocalProcess('rg', args, {
          ...config,
          timeoutMs: config.timeoutMs ?? 30000,
        });
        // ripgrep exit codes:
        //   0  → at least one match
        //   1  → no matches (clean — "No matches found.")
        //   2  → real error (bad regex, unreadable target, etc.)
        // Without this branch (Codex P2 #23 — same fix shape glob_search
        // got from P2 #13), exit-2 errors silently mapped to
        // `matches: 0`, so the agent treated tooling failures as a
        // genuine absence of matches.
        if (
          result.timedOut ||
          (result.exitCode != null && result.exitCode > 1)
        ) {
          const detail = result.stderr.trim() || `rg exited ${result.exitCode}`;
          const shown =
            config.toolResultMaxChars != null && config.toolResultMaxChars > 0
              ? capHeadTail(detail, config.toolResultMaxChars)
              : detail;
          return [
            `grep_search failed: ${shown}`,
            {
              matches: 0,
              engine: 'ripgrep',
              error: detail,
              exitCode: result.exitCode,
            },
          ];
        }
        const lines = result.stdout
          .split('\n')
          .filter(Boolean)
          .slice(0, maxResults);
        const joined =
          lines.length > 0
            ? lines.join('\n')
            : result.stderr.trim() || 'No matches found.';
        let output = joined;
        if (config.toolResultMaxChars != null && config.toolResultMaxChars > 0) {
          output =
            lines.length > 0
              ? capLines(joined, config.toolResultMaxChars)
              : capHeadTail(joined, config.toolResultMaxChars);
        }
        return [output, { matches: lines.length, engine: 'ripgrep' }];
      }

      try {
        const { matches, skipped } = await fallbackGrep(
          target,
          input.pattern,
          input.glob,
          maxResults,
          fs
        );
        // Artifact count: ONLY real matches (Codex P2 [43] —
        // skip sentinels used to inflate the count and the budget).
        const display =
          config.toolResultMaxChars != null && config.toolResultMaxChars > 0
            ? capFallbackGrepDisplay(matches, skipped, config.toolResultMaxChars)
            : formatFallbackGrepDisplay({ matches, skipped });
        return [
          display,
          {
            matches: matches.length,
            skipped: skipped.length,
            engine: 'node-fallback',
          },
        ];
      } catch (e) {
        if (e instanceof FallbackGrepError) {
          const reason =
            config.toolResultMaxChars != null && config.toolResultMaxChars > 0
              ? capHeadTail(e.message, config.toolResultMaxChars)
              : e.message;
          return [
            `grep_search refused the pattern: ${reason}`,
            {
              matches: 0,
              engine: 'node-fallback',
              error: e.message,
              kind: e.kind,
            },
          ];
        }
        throw e;
      }
    },
    {
      name: LocalGrepSearchToolName,
      description:
        'Search local files for a regex pattern (ripgrep when available, Node fallback otherwise).',
      schema: LocalGrepSearchToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export function createLocalGlobSearchTool(
  config: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  const fs = getWorkspaceFS(config);
  return tool(
    async (rawInput) => {
      const input = rawInput as {
        pattern: string;
        path?: string;
        max_results?: number;
      };
      const target = await resolveWorkspacePathSafe(
        input.path ?? '.',
        config,
        'read'
      );
      const maxResults = Math.max(input.max_results ?? DEFAULT_MAX_RESULTS, 1);

      if (await isRipgrepAvailable(config)) {
        const result = await spawnLocalProcess(
          'rg',
          [
            '--files',
            '--hidden',
            '--glob',
            '!.git/**',
            '--glob',
            input.pattern,
            target,
          ],
          { ...config, timeoutMs: config.timeoutMs ?? 30000 }
        );
        // rg --files exit codes:
        //   0  → at least one file matched
        //   1  → no files matched (clean — "No files found.")
        //   2  → real error (bad glob, unreadable target, etc.)
        // Without this branch, exit-2 errors used to silently map to
        // "No files found." — the agent then treats a tooling failure
        // as a real absence of matches.
        if (
          result.timedOut ||
          (result.exitCode != null && result.exitCode > 1)
        ) {
          const detail = result.stderr.trim() || `rg exited ${result.exitCode}`;
          return [
            `glob_search failed: ${detail}`,
            {
              files: [],
              engine: 'ripgrep',
              error: detail,
              exitCode: result.exitCode,
            },
          ];
        }
        const lines = result.stdout
          .split('\n')
          .filter(Boolean)
          .slice(0, maxResults);
        return [
          lines.length > 0 ? lines.join('\n') : 'No files found.',
          { files: lines, engine: 'ripgrep' },
        ];
      }

      const files = await fallbackGlob(target, input.pattern, maxResults, fs);
      return [
        files.length > 0 ? files.join('\n') : 'No files found.',
        { files, engine: 'node-fallback' },
      ];
    },
    {
      name: LocalGlobSearchToolName,
      description:
        'Find local files matching a glob pattern (ripgrep when available, Node fallback otherwise).',
      schema: LocalGlobSearchToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export function createLocalListDirectoryTool(
  config: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  const fs = getWorkspaceFS(config);
  return tool(
    async (rawInput) => {
      const input = rawInput as { path?: string };
      const path = await resolveWorkspacePathSafe(
        input.path ?? '.',
        config,
        'read'
      );
      const entries = await fs.readdir(path, { withFileTypes: true });
      const output = entries
        .map(
          (entry) => `${entry.isDirectory() ? 'dir ' : 'file'}\t${entry.name}`
        )
        .join('\n');
      return [output || 'Directory is empty.', { path, count: entries.length }];
    },
    {
      name: LocalListDirectoryToolName,
      description: 'List files and directories in a local directory.',
      schema: LocalListDirectoryToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export type LocalCodingToolBundle = {
  tools: DynamicStructuredTool[];
  /**
   * Present when `config.fileCheckpointing === true` or a `checkpointer`
   * was passed in. Callers can call `rewind()` to restore captured
   * pre-write contents.
   */
  checkpointer?: t.LocalFileCheckpointer;
};

export function createLocalCodingTools(
  config: t.LocalExecutionConfig = {},
  options: { checkpointer?: t.LocalFileCheckpointer } = {}
): DynamicStructuredTool[] {
  const checkpointer =
    options.checkpointer ??
    (config.fileCheckpointing === true
      ? createLocalFileCheckpointer({ fs: config.exec?.fs })
      : undefined);
  const create = (name: string): DynamicStructuredTool => {
    switch (name) {
    case Constants.READ_FILE:
      return createLocalReadFileTool(config);
    case Constants.WRITE_FILE:
      return createLocalWriteFileTool(config, checkpointer);
    case Constants.EDIT_FILE:
      return createLocalEditFileTool(config, checkpointer);
    case Constants.GREP_SEARCH:
      return createLocalGrepSearchTool(config);
    case Constants.GLOB_SEARCH:
      return createLocalGlobSearchTool(config);
    case Constants.LIST_DIRECTORY:
      return createLocalListDirectoryTool(config);
    case Constants.COMPILE_CHECK:
      return createCompileCheckTool(config);
    case Constants.BASH_TOOL:
      return createLocalBashExecutionTool({ config });
    case Constants.EXECUTE_CODE:
      return createLocalCodeExecutionTool(config);
    case Constants.PROGRAMMATIC_TOOL_CALLING:
      return createLocalProgrammaticToolCallingTool(config);
    default:
      return createLocalBashProgrammaticToolCallingTool(config);
    }
  };
  // Select by name before constructing, so omitted tools are never built.
  const names =
    config.codingToolNames == null ? undefined : new Set(config.codingToolNames);
  return LOCAL_CODING_BUNDLE_NAMES.filter(
    (name) => names == null || names.has(name)
  ).map(create);
}

/**
 * Variant of `createLocalCodingTools` that returns the bundle alongside
 * the file checkpointer so callers can later call
 * `bundle.checkpointer?.rewind()`.
 */
export function createLocalCodingToolBundle(
  config: t.LocalExecutionConfig = {},
  options: { checkpointer?: t.LocalFileCheckpointer } = {}
): LocalCodingToolBundle {
  const checkpointer =
    options.checkpointer ??
    (config.fileCheckpointing === true
      ? createLocalFileCheckpointer({ fs: config.exec?.fs })
      : undefined);
  return {
    tools: createLocalCodingTools(config, { checkpointer }),
    checkpointer,
  };
}

export function createLocalCodingToolDefinitions(): t.LCTool[] {
  return [
    toolDefinition(
      Constants.READ_FILE,
      'Read a local text file from the configured working directory with line numbers.',
      LocalReadFileToolSchema as t.JsonSchemaType
    ),
    toolDefinition(
      LocalWriteFileToolName,
      'Create or overwrite a local text file in the configured working directory.',
      LocalWriteFileToolSchema as t.JsonSchemaType
    ),
    toolDefinition(
      LocalEditFileToolName,
      'Apply exact text replacements to a local file.',
      LocalEditFileToolSchema as t.JsonSchemaType
    ),
    toolDefinition(
      LocalGrepSearchToolName,
      'Search local files with ripgrep and return matching lines.',
      LocalGrepSearchToolSchema as t.JsonSchemaType
    ),
    toolDefinition(
      LocalGlobSearchToolName,
      'Find local files matching a glob pattern.',
      LocalGlobSearchToolSchema as t.JsonSchemaType
    ),
    toolDefinition(
      LocalListDirectoryToolName,
      'List files and directories in a local directory.',
      LocalListDirectoryToolSchema as t.JsonSchemaType
    ),
    createCompileCheckToolDefinition(),
  ];
}

export function createLocalCodingToolRegistry(): t.LCToolRegistry {
  return new Map(
    createLocalCodingToolDefinitions().map((definition) => [
      definition.name,
      definition,
    ])
  );
}
