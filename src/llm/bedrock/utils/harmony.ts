/**
 * Plain-text OpenAI harmony control tokens in Bedrock requests to OpenAI models.
 *
 * OpenAI models on Bedrock render the conversation into the harmony chat format, and a
 * literal header such as `<|channel|>analysis` inside message text is not treated as text:
 * Bedrock answers `200 OK` with an event stream that closes without a single event (no
 * `messageStart`, no `messageStop`, no exception). It is deterministic for the request, so
 * every later turn that still carries the text fails the same way. Pasted transcripts,
 * attached text files, and fetched pages that document the format all carry such headers.
 *
 * The ASCII bars of each harmony special token (and only those, so other `<|name|>` text is
 * sent as written) are swapped for fullwidth bars (`<｜channel｜>`): the model
 * still reads the token, but nothing renders it as a control token. Zero-width characters
 * and spaces between `<`, `|` and the name do not help (verified live), so they are
 * tolerated inside the match and kept.
 */
import type { ConverseStreamCommandInput } from '@aws-sdk/client-bedrock-runtime';
import type {
  BedrockMessage,
  BedrockContentBlock,
  BedrockSystemContentBlock,
} from '../types';

/** The JSON document type Converse uses for `toolUse.input` and `toolResult` JSON. */
type DocumentType = Exclude<
  NonNullable<BedrockContentBlock['toolUse']>['input'],
  undefined
>;

const GAP = '[\\s\\u200B-\\u200D\\u2060\\uFEFF]*';
/** The special tokens of OpenAI's harmony (`o200k_harmony`) encoding. */
const HARMONY_TOKEN_NAMES =
  'startoftext|endoftext|endofprompt|start|end|message|channel|constrain|return|call|reserved_\\d+';
const HARMONY_CONTROL_TOKEN = new RegExp(
  `<${GAP}\\|${GAP}(?:${HARMONY_TOKEN_NAMES})${GAP}\\|${GAP}>`,
  'g'
);
const ASCII_BAR = /\|/g;
const FULLWIDTH_BAR = '｜';

type ToolResultContentBlock = NonNullable<
  NonNullable<BedrockContentBlock['toolResult']>['content']
>[number];

/** The text with every harmony control token rendered inert; the same string when none. */
export function neutralizeHarmonyControlTokens(text: string): string {
  if (!text.includes('|')) {
    return text;
  }
  return text.replace(HARMONY_CONTROL_TOKEN, (token) =>
    token.replace(ASCII_BAR, FULLWIDTH_BAR)
  );
}

/**
 * A JSON document with `rewrite` applied to its string values, given the key each one sits
 * under (an array item inherits its array's key); the same reference when nothing changed.
 * Keys are never rewritten: they are structure (argument and property names), and rewriting
 * them could merge two keys or rename an argument the tool expects.
 */
function rewriteDocument(
  value: DocumentType,
  rewrite: (text: string, key: string | undefined) => string,
  key?: string
): DocumentType {
  if (typeof value === 'string') {
    return rewrite(value, key);
  }
  if (Array.isArray(value)) {
    let next: DocumentType[] | undefined;
    for (let i = 0; i < value.length; i++) {
      const item = rewriteDocument(value[i], rewrite, key);
      if (item !== value[i]) {
        next ??= value.slice();
        next[i] = item;
      }
    }
    return next ?? value;
  }
  if (value == null || typeof value !== 'object') {
    return value;
  }
  let next: Record<string, DocumentType> | undefined;
  for (const entryKey of Object.keys(value)) {
    const entry = value[entryKey];
    const item = rewriteDocument(entry, rewrite, entryKey);
    if (item !== entry) {
      next ??= { ...value };
      next[entryKey] = item;
    }
  }
  return next ?? value;
}

/** Tool inputs and results replayed as history: every string value is free text. */
function neutralizeDocument(value: DocumentType): DocumentType {
  return rewriteDocument(value, neutralizeHarmonyControlTokens);
}

/** Schema annotations that carry prose and take no part in validation. */
const SCHEMA_TEXT_KEYS = new Set(['description', 'title', '$comment']);

/**
 * A tool's input schema is a contract the model's arguments must satisfy, so only its prose
 * is neutralized. Names, `enum`/`const` values, defaults and patterns are sent as written:
 * rewriting them would have the model produce values the bound tool does not accept.
 */
function neutralizeSchema(value: DocumentType): DocumentType {
  return rewriteDocument(value, (text, key) =>
    key != null && SCHEMA_TEXT_KEYS.has(key)
      ? neutralizeHarmonyControlTokens(text)
      : text
  );
}

function neutralizeToolResultBlock(
  block: ToolResultContentBlock
): ToolResultContentBlock {
  if (typeof block.text === 'string') {
    const text = neutralizeHarmonyControlTokens(block.text);
    return text === block.text ? block : { ...block, text };
  }
  if (block.json !== undefined) {
    const json = neutralizeDocument(block.json);
    return json === block.json ? block : { ...block, json };
  }
  return block;
}

/** Maps each item, returning the same array when no item changed. */
function mapChanged<T>(items: T[], map: (item: T) => T): T[] {
  let next: T[] | undefined;
  for (let i = 0; i < items.length; i++) {
    const item = map(items[i]);
    if (item !== items[i]) {
      next ??= items.slice();
      next[i] = item;
    }
  }
  return next ?? items;
}

function neutralizeContentBlock(
  block: BedrockContentBlock
): BedrockContentBlock {
  if (typeof block.text === 'string') {
    const text = neutralizeHarmonyControlTokens(block.text);
    return text === block.text ? block : { ...block, text };
  }
  if (block.toolUse != null) {
    const { input } = block.toolUse;
    const next = input === undefined ? input : neutralizeDocument(input);
    return next === input
      ? block
      : { ...block, toolUse: { ...block.toolUse, input: next } };
  }
  const content = block.toolResult?.content;
  if (block.toolResult != null && content != null) {
    const next = mapChanged(content, neutralizeToolResultBlock);
    return next === content
      ? block
      : { ...block, toolResult: { ...block.toolResult, content: next } };
  }
  return block;
}

function neutralizeSystemBlock(
  block: BedrockSystemContentBlock
): BedrockSystemContentBlock {
  if (typeof block.text !== 'string') {
    return block;
  }
  const text = neutralizeHarmonyControlTokens(block.text);
  return text === block.text ? block : { ...block, text };
}

type ToolConfiguration = NonNullable<ConverseStreamCommandInput['toolConfig']>;
type ToolEntry = NonNullable<ToolConfiguration['tools']>[number];

function neutralizeTool(tool: ToolEntry): ToolEntry {
  const spec = tool.toolSpec;
  if (spec == null) {
    return tool;
  }
  const description =
    spec.description == null
      ? spec.description
      : neutralizeHarmonyControlTokens(spec.description);
  const schema = spec.inputSchema?.json;
  const json = schema === undefined ? schema : neutralizeSchema(schema);
  if (description === spec.description && json === schema) {
    return tool;
  }
  return {
    ...tool,
    toolSpec: {
      ...spec,
      description,
      ...(spec.inputSchema != null ? { inputSchema: { json } } : {}),
    },
  } as ToolEntry;
}

/**
 * Tool descriptions and schema prose are rendered into the same prompt as messages, so a tool
 * that documents the format carries the same failure. Returns the same configuration when no tool
 * changed; a changed tool is copied, leaving the bound tools untouched.
 */
export function neutralizeHarmonyToolConfig(
  toolConfig: ToolConfiguration | undefined
): ToolConfiguration | undefined {
  const tools = toolConfig?.tools;
  if (toolConfig == null || tools == null) {
    return toolConfig;
  }
  const next = mapChanged(tools, neutralizeTool);
  return next === tools ? toolConfig : { ...toolConfig, tools: next };
}

/**
 * Neutralizes harmony control tokens across a converted request. Messages are updated in
 * place (the converter owns them); a block that changes is replaced by a copy, so content
 * shared with the LangChain input is never mutated.
 */
export function neutralizeHarmonyRequest(
  messages: BedrockMessage[],
  system: BedrockSystemContentBlock[]
): BedrockSystemContentBlock[] {
  for (const message of messages) {
    if (message.content != null) {
      message.content = mapChanged(message.content, neutralizeContentBlock);
    }
  }
  return mapChanged(system, neutralizeSystemBlock);
}
