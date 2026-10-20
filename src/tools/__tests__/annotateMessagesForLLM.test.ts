import { describe, it, expect } from '@jest/globals';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import {
  annotateMessagesForLLM,
  ToolOutputReferenceRegistry,
  TOOL_OUTPUT_REF_KEY,
  TOOL_OUTPUT_UNRESOLVED_KEY,
} from '../toolOutputReferences';

function makeToolMessage(fields: {
  content: ToolMessage['content'];
  name?: string;
  tool_call_id?: string;
  status?: 'success' | 'error';
  artifact?: unknown;
  additional_kwargs?: Record<string, unknown>;
}): ToolMessage {
  return new ToolMessage({
    name: fields.name ?? 'echo',
    tool_call_id: fields.tool_call_id ?? 'tc1',
    status: fields.status ?? 'success',
    artifact: fields.artifact,
    additional_kwargs: fields.additional_kwargs,
    content: fields.content,
  });
}

describe('annotateMessagesForLLM', () => {
  it('returns the input array reference when registry is undefined', () => {
    const messages = [
      new HumanMessage('hi'),
      makeToolMessage({
        content: 'data',
        additional_kwargs: { _refKey: 'tool0turn0' },
      }),
    ];
    const out = annotateMessagesForLLM(messages, undefined, 'r1');
    expect(out).toBe(messages);
  });

  it('does not iterate any message when the feature is disabled (registry undefined)', () => {
    /**
     * Hard guarantee: when the host hasn't enabled
     * `RunConfig.toolOutputReferences`, calling
     * `annotateMessagesForLLM` must short-circuit at O(1) without
     * touching a single ToolMessage. We assert by spying on
     * `_getType` — the first per-message call inside the loop — and
     * confirming it was never invoked.
     */
    const messages = [
      makeToolMessage({ content: 'a' }),
      makeToolMessage({ content: 'b' }),
      makeToolMessage({
        content: 'c',
        additional_kwargs: { _refKey: 'tool0turn0' },
      }),
    ];
    const spies = messages.map((m) => jest.spyOn(m, '_getType'));
    const out = annotateMessagesForLLM(messages, undefined, undefined);
    expect(out).toBe(messages);
    for (const spy of spies) {
      expect(spy).not.toHaveBeenCalled();
      spy.mockRestore();
    }
  });

  it('returns the input array reference when no ToolMessage carries metadata', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'stored');
    const messages = [
      new HumanMessage('hi'),
      makeToolMessage({ content: 'data' }),
      new AIMessage('answer'),
    ];
    const out = annotateMessagesForLLM(messages, registry, 'r1');
    expect(out).toBe(messages);
  });

  it('annotates string content when _refKey is live in the registry', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'stored-raw');
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    expect(out[0].content).toBe('[ref: tool0turn0]\noutput');
    expect(tm.content).toBe('output');
    expect(out).not.toBe([tm]);
    expect(out[0]).not.toBe(tm);
  });

  it('does not annotate native computer screenshot strings', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'stored-raw');
    const screenshot = `data:image/png;base64,${'A'.repeat(2_000)}`;
    const tm = makeToolMessage({
      content: screenshot,
      additional_kwargs: {
        type: 'computer_call_output',
        _refKey: 'tool0turn0',
      },
    });

    const out = annotateMessagesForLLM([tm], registry, 'r1');

    expect(out[0].content).toBe(screenshot);
    expect((out[0] as ToolMessage).additional_kwargs.type).toBe(
      'computer_call_output'
    );
    expect((out[0] as ToolMessage).additional_kwargs._refKey).toBeUndefined();
  });

  it('leaves native computer screenshots reference-equal without ref metadata', () => {
    const registry = new ToolOutputReferenceRegistry();
    const tm = makeToolMessage({
      content: 'data:image/png;base64,AAAA',
      additional_kwargs: { type: 'computer_call_output' },
    });
    const messages = [tm];

    expect(annotateMessagesForLLM(messages, registry, 'r1')).toBe(messages);
  });

  it('annotates a _refKey the registry no longer holds and strips framework metadata', () => {
    /**
     * The label is the output's persisted name, not a registry lookup,
     * so a restored message is labelled exactly like it was live. The
     * framework keys are still removed from `additional_kwargs` before
     * the bytes leave for the provider.
     */
    const registry = new ToolOutputReferenceRegistry();
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: { _refKey: 'tool0turn0', userField: 'preserved' },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    expect(out[0].content).toBe('[ref: tool0turn0]\noutput');
    expect(out[0]).not.toBe(tm);
    const projectedKwargs = (out[0] as ToolMessage).additional_kwargs;
    expect(projectedKwargs._refKey).toBeUndefined();
    expect(projectedKwargs.userField).toBe('preserved');
    expect(tm.additional_kwargs._refKey).toBe('tool0turn0');
  });

  it('always applies _unresolvedRefs even when there is no registry entry', () => {
    const registry = new ToolOutputReferenceRegistry();
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: { _unresolvedRefs: ['tool9turn9'] },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    expect(out[0].content).toBe('output\n[unresolved refs: tool9turn9]');
  });

  it('injects _ref into JSON-object string content', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', '{"a":1}');
    const tm = makeToolMessage({
      content: '{"a":1,"b":"x"}',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    const parsed = JSON.parse(out[0].content as string);
    expect(parsed[TOOL_OUTPUT_REF_KEY]).toBe('tool0turn0');
    expect(parsed.a).toBe(1);
    expect(parsed.b).toBe('x');
  });

  it('injects both _ref and _unresolved_refs into JSON-object content', () => {
    /**
     * Combined path: when a ToolMessage carries both a live `_refKey`
     * and unresolved-ref hints, JSON-object content should receive
     * both `_ref` and `_unresolved_refs` fields rather than falling
     * back to the prefix/trailer form. Exercises
     * `annotateToolOutputWithReference`'s collision-detection logic
     * through the projection entry point.
     */
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', '{"a":1}');
    const tm = makeToolMessage({
      content: '{"a":1,"b":"x"}',
      additional_kwargs: {
        _refKey: 'tool0turn0',
        _unresolvedRefs: ['tool9turn9'],
      },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    const parsed = JSON.parse(out[0].content as string);
    expect(parsed[TOOL_OUTPUT_REF_KEY]).toBe('tool0turn0');
    expect(parsed[TOOL_OUTPUT_UNRESOLVED_KEY]).toEqual(['tool9turn9']);
    expect(parsed.a).toBe(1);
    expect(parsed.b).toBe('x');
  });

  it('uses [ref: …] prefix for non-JSON string content', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'raw');
    const tm = makeToolMessage({
      content: 'plain output',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    expect(out[0].content).toBe('[ref: tool0turn0]\nplain output');
  });

  it('annotates a live ref on multi-part content (prompt-cache-rewritten tool tail)', () => {
    /**
     * A tail tool result that prompt caching rewrote from a string into a
     * text-block array (to host the cache_control / cachePoint marker) keeps
     * its `_refKey` on additional_kwargs. The live-ref marker must still be
     * projected as a leading text block; otherwise the common tool-result
     * tail silently loses its reference annotation once cached.
     */
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'raw');
    const tm = makeToolMessage({
      content: [
        { type: 'text', text: 'output', cache_control: { type: 'ephemeral' } },
      ] as unknown as ToolMessage['content'],
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    const blocks = out[0].content as Array<{
      type: string;
      text?: string;
      cache_control?: unknown;
    }>;
    expect(blocks).toHaveLength(2);
    expect(blocks[0]).toEqual({ type: 'text', text: '[ref: tool0turn0]' });
    // The original block (and its cache marker) is preserved after the prefix.
    expect(blocks[1].text).toBe('output');
    expect(blocks[1].cache_control).toEqual({ type: 'ephemeral' });
  });

  it('annotates both live ref and unresolved on multi-part content', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'raw');
    const tm = makeToolMessage({
      content: [
        { type: 'text', text: 'output' },
      ] as unknown as ToolMessage['content'],
      additional_kwargs: {
        _refKey: 'tool0turn0',
        _unresolvedRefs: ['tool9turn9'],
      },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    const blocks = out[0].content as Array<{ type: string; text?: string }>;
    expect(blocks.map((b) => b.text)).toEqual([
      '[ref: tool0turn0]',
      '[unresolved refs: tool9turn9]',
      'output',
    ]);
  });

  it('prepends an unresolved-refs warning text block to multi-part content', () => {
    const registry = new ToolOutputReferenceRegistry();
    const tm = makeToolMessage({
      content: [
        { type: 'text', text: 'data' },
        { type: 'image_url', image_url: { url: 'data:...' } },
      ] as unknown as ToolMessage['content'],
      additional_kwargs: { _unresolvedRefs: ['tool9turn9'] },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    const blocks = out[0].content as Array<{ type: string; text?: string }>;
    expect(blocks).toHaveLength(3);
    expect(blocks[0].type).toBe('text');
    expect(blocks[0].text).toBe('[unresolved refs: tool9turn9]');
    expect(blocks[1].type).toBe('text');
    expect(blocks[1].text).toBe('data');
    expect(blocks[2].type).toBe('image_url');
  });

  it('preserves artifact on the projected ToolMessage', () => {
    /**
     * Hosts attach `artifact` to ToolMessages via the
     * `content_and_artifact` response format (e.g. code execution
     * sessions, MCP tools that return structured side-data). The
     * projection must round-trip the artifact untouched so downstream
     * consumers (audit logs, code-session tracking) keep working.
     */
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'raw');
    const artifact = {
      session_id: 'abc',
      files: [{ id: 'f1', name: 'a.txt' }],
    };
    const tm = makeToolMessage({
      content: 'output',
      artifact,
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    const projected = out[0] as ToolMessage;
    expect(projected.artifact).toBe(artifact);
    expect(projected.content).toBe('[ref: tool0turn0]\noutput');
  });

  it('does not mutate the original ToolMessage instance or its content', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'raw');
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const originalContent = tm.content;
    const originalKwargs = { ...tm.additional_kwargs };
    annotateMessagesForLLM([tm], registry, 'r1');
    expect(tm.content).toBe(originalContent);
    expect(tm.additional_kwargs).toEqual(originalKwargs);
  });

  it('strips framework ref metadata from the projected additional_kwargs but preserves other fields', () => {
    /**
     * Defensive: even though LangChain's standard provider serializers
     * do not transmit `additional_kwargs`, a custom adapter or future
     * LangChain change could. Strip our three framework-owned keys on
     * the projection so the metadata never reaches the wire under any
     * serializer behavior. Non-framework fields stay put.
     */
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'raw');
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: {
        _refKey: 'tool0turn0',
        _refScope: 'r1',
        _unresolvedRefs: ['tool9turn9'],
        someOtherField: 'preserved',
      },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    const projected = out[0] as ToolMessage;
    expect(projected.additional_kwargs._refKey).toBeUndefined();
    expect(projected.additional_kwargs._refScope).toBeUndefined();
    expect(projected.additional_kwargs._unresolvedRefs).toBeUndefined();
    expect(projected.additional_kwargs.someOtherField).toBe('preserved');
    expect(tm.additional_kwargs._refKey).toBe('tool0turn0');
    expect(tm.additional_kwargs._refScope).toBe('r1');
  });

  it('leaves additional_kwargs empty when stripping removed every framework key', () => {
    /**
     * `stripFrameworkRefMetadata` returns `undefined` when no non-
     * framework keys remain, and the LangChain `ToolMessage`
     * constructor normalizes that to `{}` — so the projected message
     * exposes an empty object, not the original kwargs.
     */
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'raw');
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: {
        _refKey: 'tool0turn0',
        _refScope: 'r1',
      },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    const projected = out[0] as ToolMessage;
    expect(projected.additional_kwargs).toEqual({});
  });

  it('passes through non-ToolMessages unchanged in the projected array', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'raw');
    const human = new HumanMessage('hi');
    const ai = new AIMessage('answer');
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const out = annotateMessagesForLLM([human, ai, tm], registry, 'r1');
    expect(out[0]).toBe(human);
    expect(out[1]).toBe(ai);
    expect(out[2]).not.toBe(tm);
  });

  it('annotates by _refKey even when the registry holds only other keys', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool1turn0', 'somethingelse');
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const messages = [tm];
    const out = annotateMessagesForLLM(messages, registry, 'r1');
    expect(out).not.toBe(messages);
    expect(out[0].content).toBe('[ref: tool0turn0]\noutput');
    expect((out[0] as ToolMessage).additional_kwargs._refKey).toBeUndefined();
  });

  it('returns the input array reference-equal when no ToolMessage carries any framework metadata', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'stored');
    const messages = [
      new HumanMessage('hi'),
      makeToolMessage({
        content: 'output',
        additional_kwargs: { unrelated: 'value' },
      }),
      new AIMessage('answer'),
    ];
    const out = annotateMessagesForLLM(messages, registry, 'r1');
    expect(out).toBe(messages);
  });

  it('annotates only the live ref when both ref and unresolved are present', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'raw');
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: {
        _refKey: 'tool0turn0',
        _unresolvedRefs: ['tool9turn9'],
      },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    expect(out[0].content).toBe(
      '[ref: tool0turn0]\noutput\n[unresolved refs: tool9turn9]'
    );
  });

  it('labels by _refKey alone, without a runId or a stamped _refScope', () => {
    const registry = new ToolOutputReferenceRegistry();
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const out = annotateMessagesForLLM([tm], registry, undefined);
    expect(out[0].content).toBe('[ref: tool0turn0]\noutput');
  });

  it('coerces a non-array _unresolvedRefs to empty without throwing', () => {
    /**
     * Defensive against malformed hydrated messages:
     * `additional_kwargs._unresolvedRefs` is untyped at the LangChain
     * layer, so a persisted message could carry a string/object/null
     * by mistake. The transform must not crash the run on
     * `.length` / `.join` — coerce to an empty list, strip the
     * malformed key, and proceed.
     */
    const registry = new ToolOutputReferenceRegistry();
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: {
        _unresolvedRefs: 'tool9turn9' as unknown as string[],
      },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    expect(out[0].content).toBe('output');
    expect(
      (out[0] as ToolMessage).additional_kwargs._unresolvedRefs
    ).toBeUndefined();
  });

  it('filters non-string entries out of _unresolvedRefs', () => {
    const registry = new ToolOutputReferenceRegistry();
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: {
        _unresolvedRefs: [
          'tool9turn9',
          42,
          null,
          { not: 'a string' },
          'tool8turn8',
        ] as unknown as string[],
      },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    expect(out[0].content).toBe(
      'output\n[unresolved refs: tool9turn9, tool8turn8]'
    );
  });

  it('ignores a non-string _refKey rather than poisoning the registry lookup', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'raw');
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: {
        _refKey: { malformed: true } as unknown as string,
      },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    expect(out[0].content).toBe('output');
    expect((out[0] as ToolMessage).additional_kwargs._refKey).toBeUndefined();
  });

  it('skips a ToolMessage whose additional_kwargs is a primitive without throwing', () => {
    /**
     * The `in` operator throws `TypeError` on primitives, so without
     * a runtime object guard, a hydrated ToolMessage carrying e.g.
     * `additional_kwargs: 'not-an-object'` (from a buggy serializer)
     * would crash `attemptInvoke` before the provider call. Verify
     * we skip that message and process subsequent live-ref messages
     * normally.
     */
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'raw');

    const malformed = new ToolMessage({
      name: 'echo',
      tool_call_id: 'mal',
      status: 'success',
      content: 'malformed-output',
    });
    /* Force a primitive past LangChain's typed setter via a cast. */
    (malformed as unknown as { additional_kwargs: unknown }).additional_kwargs =
      'not-an-object' as unknown;

    const live = makeToolMessage({
      content: 'live-output',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });

    const out = annotateMessagesForLLM([malformed, live], registry, 'r1');
    /* Malformed message passes through unchanged; live ref still annotates. */
    expect(out[0]).toBe(malformed);
    expect(out[1].content).toBe('[ref: tool0turn0]\nlive-output');
  });

  it('combines the restored _refKey label with unresolved refs', () => {
    const registry = new ToolOutputReferenceRegistry();
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: {
        _refKey: 'tool0turn0',
        _unresolvedRefs: ['tool9turn9'],
      },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    expect(out[0].content).toBe(
      '[ref: tool0turn0]\noutput\n[unresolved refs: tool9turn9]'
    );
  });
});
