import type { AIMessageChunk } from '@langchain/core/messages';
import type { ProviderTextAttempt } from '@/protection/providerText';
import { ProviderTextProtectionError } from '@/protection/providerText';

export const GEMINI_SIGNATURES = '__gemini_function_call_thought_signatures__';
export const RESPONSES_POSITIONS = '__openai_responses_replay_positions__';

export function keys(value: unknown, allowed: readonly string[]): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value) &&
    Object.keys(value).every((key) => allowed.includes(key));
}

export function stringMap(value: unknown): boolean {
  return value != null && typeof value === 'object' && !Array.isArray(value) &&
    Object.entries(value).length <= 128 && Object.values(value).every((entry) => typeof entry === 'string');
}

export function safetyRatings(value: unknown): boolean {
  return Array.isArray(value) && value.length <= 16 && value.every((entry: unknown) =>
    keys(entry, ['category', 'probability', 'blocked']) &&
    typeof entry.category === 'string' && /^HARM_CATEGORY_[A-Z_]+$/.test(entry.category) &&
    ['NEGLIGIBLE', 'LOW', 'MEDIUM', 'HIGH', 'HARM_PROBABILITY_UNSPECIFIED'].includes(String(entry.probability)) &&
    (entry.blocked == null || typeof entry.blocked === 'boolean'));
}

export function replayPositions(value: unknown): boolean {
  return Array.isArray(value) && value.length <= 128 && value.every((entry: unknown) =>
    keys(entry, ['itemId', 'kind', 'outputIndex', 'contentIndex']) && typeof entry.itemId === 'string' &&
    ['message', 'text', 'reasoning', 'output'].includes(String(entry.kind)) &&
    typeof entry.outputIndex === 'number' && Number.isSafeInteger(entry.outputIndex) && entry.outputIndex >= 0 &&
    (entry.contentIndex == null || (typeof entry.contentIndex === 'number' && Number.isSafeInteger(entry.contentIndex) && entry.contentIndex >= 0)));
}

const scalarKeys = new Set([
  'id', 'model', 'model_name', 'model_provider', 'created_at', 'object', 'status', 'user', 'service_tier',
  'max_output_tokens', 'max_tool_calls', 'parallel_tool_calls', 'temperature', 'top_p', 'truncation',
  'store', 'background', 'previous_response_id', 'prompt_cache_key', 'prompt_cache_retention',
  'safety_identifier', 'completed_at',
]);

interface OutputText {
  type: 'output_text';
  text: string;
  annotations: [];
}
interface MessageOutput {
  type: 'message';
  id: string;
  role: 'assistant';
  status: 'completed' | 'incomplete';
  phase?: 'commentary' | 'final_answer' | null;
  content: [OutputText];
}

function messageOutput(value: unknown): value is MessageOutput {
  if (!keys(value, ['id', 'type', 'role', 'status', 'phase', 'content']) || value.type !== 'message' ||
      typeof value.id !== 'string' || value.role !== 'assistant' || !['completed', 'incomplete'].includes(String(value.status)) ||
      (value.phase != null && !['commentary', 'final_answer'].includes(String(value.phase))) ||
      !Array.isArray(value.content) || value.content.length !== 1) return false;
  const part: unknown = value.content[0];
  return keys(part, ['type', 'text', 'annotations']) && part.type === 'output_text' &&
    typeof part.text === 'string' && Array.isArray(part.annotations) && part.annotations.length === 0;
}

/** One certified prose output. Unknown structured outputs and alternate aliases stay gated. */
export class ResponsesTextProjection {
  private metadata?: AIMessageChunk['response_metadata'];
  private hasOutputAlias = false;
  private output?: Array<MessageOutput | Record<string, unknown>>;

  constructor(private readonly attempt: ProviderTextAttempt) {}

  strip(metadata: AIMessageChunk['response_metadata']): AIMessageChunk['response_metadata'] {
    if (!Object.hasOwn(metadata, 'output')) return metadata;
    if (this.metadata != null || metadata.model_provider !== 'openai' || metadata.object !== 'response' ||
        !Array.isArray(metadata.output) || metadata.output.length > 32) {
      throw new ProviderTextProtectionError('unsupported');
    }
    let messageCount = 0;
    const output: Array<MessageOutput | Record<string, unknown>> = metadata.output.map((item: unknown) => {
      if (messageOutput(item)) {
        if (++messageCount > 1 || !this.attempt.matchesText(item.content[0].text)) throw new ProviderTextProtectionError('unsupported');
        return { ...item, content: [{ ...item.content[0], text: '' }] };
      }
      if (!responseOutputControl(item)) throw new ProviderTextProtectionError('unsupported');
      return item;
    });
    if ((messageCount === 0 && !this.attempt.matchesText('')) ||
        (metadata.output_text != null && (typeof metadata.output_text !== 'string' || !this.attempt.matchesText(metadata.output_text)))) {
      throw new ProviderTextProtectionError('unsupported');
    }
    for (const [key, value] of Object.entries(metadata)) {
      let valid: boolean;
      if (key === 'output' || key === 'output_text') continue;
      if (key === 'usage') valid = numericTree(value);
      else if (key === 'text') valid = keys(value, ['format', 'verbosity']) && keys(value.format, ['type']) && value.format.type === 'text' &&
        (value.verbosity == null || ['low', 'medium', 'high'].includes(String(value.verbosity)));
      else if (key === 'tools') valid = Array.isArray(value) && value.length <= 32 && value.every(responseTool);
      else if (key === 'tool_choice') valid = value === 'auto' || value === 'none' || value === 'required' || (keys(value, ['type', 'name']) && value.type === 'function' && typeof value.name === 'string');
      else if (key === 'reasoning') valid = value == null || (keys(value, ['effort', 'summary']) &&
        (value.effort == null || ['none', 'minimal', 'low', 'medium', 'high', 'xhigh'].includes(String(value.effort))) &&
        (value.summary == null || ['auto', 'concise', 'detailed'].includes(String(value.summary))));
      else if (key === 'metadata') valid = keys(value, []) || value == null;
      else if (key === 'instructions' || key === 'error') valid = value == null;
      else if (key === 'incomplete_details') valid = value == null || (keys(value, ['reason']) && ['max_output_tokens', 'content_filter'].includes(String(value.reason)));
      else valid = scalarKeys.has(key) && (value == null || typeof value === 'string' || typeof value === 'boolean' || (typeof value === 'number' && Number.isFinite(value)));
      if (!valid) throw new ProviderTextProtectionError('unsupported');
    }
    retainTree(metadata, this.attempt);
    this.output = output;
    this.hasOutputAlias = Object.hasOwn(metadata, 'output_text');
    const { output: _output, output_text: _alias, ...controls } = metadata;
    this.metadata = controls;
    return controls;
  }

  canonical(content: string): AIMessageChunk['response_metadata'] | undefined {
    if (this.metadata == null || this.output == null) return undefined;
    return {
      output: this.output.map((item) => messageOutput(item) ? { ...item, content: [{ ...item.content[0], text: content }] } : item),
      ...(this.hasOutputAlias ? { output_text: content } : {}),
    };
  }
}

function numericTree(value: unknown, depth = 4): boolean {
  if (value == null) return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (typeof value !== 'object' || Array.isArray(value) || depth === 0) return false;
  const entries = Object.values(value);
  return entries.length <= 64 && entries.every((entry) => numericTree(entry, depth - 1));
}

function responseOutputControl(value: unknown): value is Record<string, unknown> {
  if (!keys(value, ['id', 'type', 'status', 'call_id', 'name', 'arguments', 'summary', 'encrypted_content'])) return false;
  if (typeof value.id !== 'string' || (value.status != null && !['completed', 'incomplete', 'in_progress'].includes(String(value.status)))) return false;
  if (value.type === 'function_call') return keys(value, ['id', 'type', 'status', 'call_id', 'name', 'arguments']) &&
    typeof value.call_id === 'string' && typeof value.name === 'string' && typeof value.arguments === 'string';
  if (value.type !== 'reasoning' || !keys(value, ['id', 'type', 'status', 'summary', 'encrypted_content'])) return false;
  return (value.encrypted_content == null || typeof value.encrypted_content === 'string') &&
    Array.isArray(value.summary) && value.summary.length <= 32 && value.summary.every((entry: unknown) =>
    keys(entry, ['type', 'text']) && entry.type === 'summary_text' && typeof entry.text === 'string');
}

function responseTool(value: unknown): boolean {
  return keys(value, ['type', 'name', 'description', 'strict', 'parameters']) && value.type === 'function' &&
    typeof value.name === 'string' && (value.description == null || typeof value.description === 'string') &&
    (value.strict == null || typeof value.strict === 'boolean') && jsonControl(value.parameters);
}

function jsonControl(value: unknown, depth = 6): boolean {
  if (value == null || typeof value === 'string' || typeof value === 'boolean') return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (typeof value !== 'object' || depth === 0) return false;
  const entries = Object.values(value);
  return entries.length <= 64 && entries.every((entry) => jsonControl(entry, depth - 1));
}

function retainTree(value: unknown, attempt: ProviderTextAttempt, depth = 8): void {
  if (typeof value === 'string') { attempt.retain(value.length * 2 + 32); return; }
  attempt.retain(64);
  if (value == null || typeof value !== 'object') return;
  if (depth === 0) throw new ProviderTextProtectionError('unsupported');
  const entries = Object.entries(value);
  if (entries.length > 128) throw new ProviderTextProtectionError('unsupported');
  for (const [key, entry] of entries) {
    attempt.retain(key.length * 2 + 32);
    retainTree(entry, attempt, depth - 1);
  }
}
