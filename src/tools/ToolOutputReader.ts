import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { ToolOutputResolveView } from '@/tools/toolOutputReferences';
import type { LCTool, LangfuseConfig } from '@/types';
import { HARD_MAX_TOOL_RESULT_CHARS } from '@/utils/truncation';

export const TOOL_OUTPUT_READER = 'read_tool_output';
const readers = new WeakSet<object>();
const description =
  'Read a bounded excerpt of an earlier retained tool result using its bare ref key. ' +
  'Offsets and maxChars count UTF-16 code units. No command is executed. ' +
  'Only retained text is available; upstream output may already be truncated.';
const schema = z.object({
  reference: z.string().describe('Bare key such as tool0turn0, without braces'),
  start: z.number().int().min(0).max(Number.MAX_SAFE_INTEGER).default(0),
  maxChars: z.number().int().min(1).max(HARD_MAX_TOOL_RESULT_CHARS).default(4000),
});

export const toolOutputReaderDefinition: LCTool = {
  name: TOOL_OUTPUT_READER,
  description,
  allowed_callers: ['direct'],
};

export function isToolOutputReader(candidate?: object): boolean {
  return candidate != null && readers.has(candidate);
}

export function withToolOutputReaderRedaction(
  config: LangfuseConfig | undefined,
  enabled: boolean
): LangfuseConfig | undefined {
  if (!enabled || config?.toolOutputTracing?.redactedToolNames?.includes(TOOL_OUTPUT_READER) === true) {
    return config;
  }
  return {
    ...config,
    toolOutputTracing: {
      ...config?.toolOutputTracing,
      redactedToolNames: [...(config?.toolOutputTracing?.redactedToolNames ?? []), TOOL_OUTPUT_READER],
    },
  };
}

function splitsSurrogatePair(text: string, offset: number): boolean {
  const before = text.charCodeAt(offset - 1);
  const after = text.charCodeAt(offset);
  return before >= 0xd800 && before <= 0xdbff && after >= 0xdc00 && after <= 0xdfff;
}

function readExcerpt(
  { reference, start, maxChars }: z.infer<typeof schema>,
  view: ToolOutputResolveView | undefined,
  budget: number
): string {
  if (reference.length > 64 || !/^tool\d+turn\d+$/.test(reference)) {
    return '{"error":"invalid_reference"}';
  }
  const result = view?.resolve({ text: `{{${reference}}}` });
  if (result == null || result.unresolved.length > 0) {
    return '{"error":"unavailable"}';
  }
  const source = result.resolved.text;
  if (start > source.length || splitsSurrogatePair(source, start)) {
    return '{"error":"invalid_range"}';
  }
  const encode = (end: number): string => JSON.stringify({
    reference,
    start,
    end,
    retainedChars: source.length,
    nextOffset: end < source.length ? end : null,
    unit: 'utf16',
    sourceCompleteness: 'unknown',
    content: source.slice(start, end),
  });
  const requestedEnd = start + Math.min(maxChars, source.length - start);
  const completeEnd = splitsSurrogatePair(source, requestedEnd) ? requestedEnd - 1 : requestedEnd;
  if (completeEnd === start && start < source.length) {
    return '{"error":"excerpt_budget_too_small"}';
  }
  const complete = encode(completeEnd);
  if (complete.length <= budget) {
    return complete;
  }
  let low = 0;
  let high = completeEnd - start - 1;
  let selected = '';
  let selectedEnd = start;
  while (low <= high) {
    const count = Math.floor((low + high) / 2);
    const proposedEnd = start + count;
    const end = splitsSurrogatePair(source, proposedEnd) ? proposedEnd - 1 : proposedEnd;
    const encoded = encode(end);
    if (encoded.length <= budget) {
      selected = encoded;
      selectedEnd = end;
      low = count + 1;
    } else {
      high = count - 1;
    }
  }
  if (selected === '' || (selectedEnd === start && start < source.length)) {
    return '{"error":"excerpt_budget_too_small"}';
  }
  return selected;
}

/** The invocation binds a frozen view; the model-bound instance owns no registry. */
export function createToolOutputReader(
  view?: ToolOutputResolveView,
  maxResultChars = HARD_MAX_TOOL_RESULT_CHARS
): StructuredToolInterface {
  if (!Number.isSafeInteger(maxResultChars) || maxResultChars < 256) {
    throw new Error('read_tool_output requires maxToolResultChars of at least 256');
  }
  const budget = Math.min(maxResultChars, HARD_MAX_TOOL_RESULT_CHARS);
  const reader = tool(async (args) => readExcerpt(args, view, budget), {
    name: TOOL_OUTPUT_READER,
    description,
    schema,
  });
  readers.add(reader);
  return reader;
}
