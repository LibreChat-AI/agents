import { AIMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { OpenAI as OpenAIClient } from 'openai';
import {
  OPENAI_RESPONSES_ORIGIN_KEY,
  MAX_PROVIDER_ITEMS_CHARS,
  restoreResponsesToolRound,
  sliceResponsesToolRound,
  gateResponsesProviderOutput,
} from './responsesToolRounds';

type OutputItem = OpenAIClient.Responses.ResponseOutputItem;

const ORIGIN = 'origin-a';
const outputOf = (message: BaseMessage): OutputItem[] | undefined =>
  (message as AIMessage).response_metadata.output as OutputItem[] | undefined;
const reasoning = (id: string, encrypted?: string): OutputItem => ({
  id,
  type: 'reasoning',
  summary: [],
  ...(encrypted != null && { encrypted_content: encrypted }),
});
const functionCall = (index: number): OutputItem => ({
  id: `fc_${index}`,
  type: 'function_call',
  status: 'completed',
  arguments: '{}',
  call_id: `call_${index}`,
  name: 'lookup',
});
const toolRound = (output: OutputItem[], createdAt = 1_000): AIMessage =>
  new AIMessage({
    content: '',
    tool_calls: output.flatMap((item) =>
      item.type === 'function_call'
        ? [{ id: item.call_id, name: item.name, args: {} }]
        : []
    ),
    response_metadata: {
      output,
      created_at: createdAt,
      [OPENAI_RESPONSES_ORIGIN_KEY]: ORIGIN,
    },
  });

describe('Responses tool round items', () => {
  it('slices after each call and keeps trailing items with the last call', () => {
    const output = [reasoning('rs_1'), functionCall(1), functionCall(2), reasoning('rs_2')];
    const slices = sliceResponsesToolRound(toolRound(output));
    expect(slices?.map(({ id, providerItems }) => [id, providerItems.items.map((item) => item.id)])).toEqual([
      ['call_1', ['rs_1', 'fc_1']],
      ['call_2', ['fc_2', 'rs_2']],
    ]);
  });

  it('restores only when every call carries a matching slice', () => {
    const output = [functionCall(1), functionCall(2)];
    const slices = sliceResponsesToolRound(toolRound(output)) ?? [];
    const restored = new AIMessage({ content: '', tool_calls: toolRound(output).tool_calls });
    restoreResponsesToolRound(restored, [slices[0].providerItems, undefined]);
    expect(outputOf(restored)).toBeUndefined();
    restoreResponsesToolRound(
      restored,
      slices.map((slice) => slice.providerItems)
    );
    expect(outputOf(restored)).toEqual(output);
  });

  it('drops reasoning references older than the stored response retention', () => {
    const output = [reasoning('rs_ref'), reasoning('rs_enc', 'ENC'), functionCall(1)];
    const now = 1_000 + 31 * 24 * 60 * 60;
    const [fresh] = gateResponsesProviderOutput([toolRound(output, now)], ORIGIN, false, now);
    expect(outputOf(fresh)).toEqual(output);
    const [stale] = gateResponsesProviderOutput([toolRound(output)], ORIGIN, false, now);
    expect(outputOf(stale)?.map((item) => item.id)).toEqual([
      'rs_enc',
      'fc_1',
    ]);
  });
});

describe('Responses replay gate', () => {
  const now = 1_000 + 31 * 24 * 60 * 60;
  const output = [reasoning('rs_ref'), reasoning('rs_enc', 'ENC'), functionCall(1)];
  const gate = (
    message: AIMessage,
    options: { origin?: string; zdr?: boolean; at?: number } = {}
  ): AIMessage =>
    gateResponsesProviderOutput(
      [message],
      options.origin ?? ORIGIN,
      options.zdr ?? false,
      options.at ?? now
    )[0] as AIMessage;

  it('keeps provider output for the producing origin, outside ZDR', () => {
    const message = toolRound(output, now);
    expect(gate(message)).toBe(message);
  });

  it('drops provider output and its origin stamp under ZDR', () => {
    const gated = gate(toolRound(output, now), { zdr: true });
    expect(outputOf(gated)).toBeUndefined();
    expect(gated.response_metadata[OPENAI_RESPONSES_ORIGIN_KEY]).toBeUndefined();
    expect(gated.tool_calls).toHaveLength(1);
  });

  it('drops provider output produced by another model or endpoint', () => {
    const gated = gate(toolRound(output, now), { origin: 'origin-b' });
    expect(outputOf(gated)).toBeUndefined();
    expect(gated.response_metadata[OPENAI_RESPONSES_ORIGIN_KEY]).toBeUndefined();
  });

  it('drops a stamped message whose output is not an array', () => {
    const message = new AIMessage({
      content: 'text',
      response_metadata: { output: 'broken', [OPENAI_RESPONSES_ORIGIN_KEY]: ORIGIN },
    });
    expect(outputOf(gate(message))).toBeUndefined();
  });

  it('keeps an encrypted reasoning item however old, and a reference up to the retention limit', () => {
    const limit = 29 * 24 * 60 * 60;
    expect(outputOf(gate(toolRound(output), { at: 1_000 + limit }))?.map((item) => item.id)).toEqual([
      'rs_ref',
      'rs_enc',
      'fc_1',
    ]);
    expect(outputOf(gate(toolRound(output), { at: 1_000 + limit + 1 }))?.map((item) => item.id)).toEqual([
      'rs_enc',
      'fc_1',
    ]);
  });

  it('reads the age from the injected time, not the system clock', () => {
    const message = toolRound([reasoning('rs_ref'), functionCall(1)], 5_000_000_000);
    expect(outputOf(gate(message, { at: 5_000_000_000 + 40 * 24 * 60 * 60 }))?.map((item) => item.id)).toEqual(['fc_1']);
    expect(gate(message, { at: 5_000_000_000 + 60 })).toBe(message);
  });

  it('turns a tool round with no output and empty text into an empty content array', () => {
    const message = new AIMessage({
      content: '',
      tool_calls: [{ id: 'call_1', name: 'lookup', args: {} }],
    });
    expect(gate(message).content).toEqual([]);
    const withText = new AIMessage({
      content: 'text',
      tool_calls: [{ id: 'call_1', name: 'lookup', args: {} }],
    });
    expect(gate(withText)).toBe(withText);
  });
});

describe('Responses provider items trust boundary', () => {
  const webSearch = {
    id: 'ws_1',
    type: 'web_search_call',
    status: 'completed',
  } as unknown as OutputItem;
  const stored = (items: OutputItem[]) => ({ origin: ORIGIN, items });
  const restoreWith = (items: OutputItem[]): OutputItem[] | undefined => {
    const message = new AIMessage({
      content: '',
      tool_calls: [{ id: 'call_1', name: 'lookup', args: {} }],
    });
    restoreResponsesToolRound(message, [stored(items)]);
    return outputOf(message);
  };

  it('does not store a round that holds a provider tool item', () => {
    expect(sliceResponsesToolRound(toolRound([webSearch, functionCall(1)]))).toBeUndefined();
  });

  it('does not store a round over the size cap', () => {
    const big = reasoning('rs_big', 'x'.repeat(MAX_PROVIDER_ITEMS_CHARS));
    expect(sliceResponsesToolRound(toolRound([big, functionCall(1)]))).toBeUndefined();
  });

  it('restores nothing from items of a type outside the allowlist', () => {
    expect(restoreWith([functionCall(1)])).toEqual([functionCall(1)]);
    expect(restoreWith([webSearch, functionCall(1)])).toBeUndefined();
    expect(restoreWith([{ type: 'computer_call' } as unknown as OutputItem, functionCall(1)])).toBeUndefined();
  });

  it('restores nothing from malformed or oversized stored items', () => {
    expect(restoreWith([null as unknown as OutputItem, functionCall(1)])).toBeUndefined();
    expect(
      restoreWith([reasoning('rs_big', 'x'.repeat(MAX_PROVIDER_ITEMS_CHARS)), functionCall(1)])
    ).toBeUndefined();
  });
});
