import { describe, it, expect } from '@jest/globals';
import {
  ToolOutputReferenceRegistry,
  annotateToolOutputWithReference,
  buildReferenceKey,
  buildReferencePrefix,
  TOOL_OUTPUT_REF_KEY,
  TOOL_OUTPUT_REF_PATTERN,
} from '../toolOutputReferences';

describe('ToolOutputReferenceRegistry', () => {
  describe('resumeBatch', () => {
    it.each([0, 5])('keeps shared state and applies current limits to frozen inputs (turns=%s)', (turns) => {
      const registry = new ToolOutputReferenceRegistry({ maxOutputSize: 4, maxTotalSize: 8 });
      registry.set('r', 'tool0turn4', 'live');
      for (let i = 0; i < turns; i++) registry.nextTurn('r');
      const snapshot = registry.resumeBatch('r', {
        entries: [{ key: 'tool0turn0', value: 'historical-long-value' }],
        turnCounter: 1, warnedNonStringTools: [],
      });
      const value = snapshot.resolve('{{tool0turn0}}').resolved;
      expect(value.length).toBeLessThanOrEqual(4);
      expect(snapshot.resolve('{{tool0turn4}}').unresolved).toEqual(['tool0turn4']);
      expect(registry.get('r', 'tool0turn4')).toBe('live');
      expect(registry.get('r', 'tool0turn0')).toBeUndefined();
      expect(registry.nextTurn('r')).toBe(Math.max(turns, 2));
    });
  });

  describe('buildReferenceKey', () => {
    it('formats keys as tool<idx>turn<turn>', () => {
      expect(buildReferenceKey(0, 0)).toBe('tool0turn0');
      expect(buildReferenceKey(3, 7)).toBe('tool3turn7');
    });
  });

  describe('set / get', () => {
    it('stores and retrieves outputs by key', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'hello world');
      expect(reg.get('r', 'tool0turn0')).toBe('hello world');
      expect(reg.size).toBe(1);
    });

    it('clips stored values to the per-output limit', () => {
      const reg = new ToolOutputReferenceRegistry({ maxOutputSize: 5 });
      reg.set('r', 'tool0turn0', 'abcdefghij');
      expect(reg.get('r', 'tool0turn0')).toBe('abcde');
    });

    it('replaces existing entries under the same key without double-counting size', () => {
      const reg = new ToolOutputReferenceRegistry({
        maxOutputSize: 100,
        maxTotalSize: 20,
      });
      reg.set('r', 'tool0turn0', 'hello');
      reg.set('r', 'tool0turn0', 'world-longer');
      expect(reg.get('r', 'tool0turn0')).toBe('world-longer');
      expect(reg.size).toBe(1);
    });
  });

  describe('clear / releaseRun', () => {
    it('clear() drops every run bucket', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('run-A', 'tool0turn0', 'A');
      reg.set('run-B', 'tool0turn0', 'B');
      expect(reg.size).toBe(2);
      reg.clear();
      expect(reg.size).toBe(0);
      expect(reg.get('run-A', 'tool0turn0')).toBeUndefined();
      expect(reg.get('run-B', 'tool0turn0')).toBeUndefined();
    });

    it('releaseRun() drops only the named run', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('run-A', 'tool0turn0', 'A');
      reg.set('run-B', 'tool0turn0', 'B');
      reg.releaseRun('run-A');
      expect(reg.get('run-A', 'tool0turn0')).toBeUndefined();
      expect(reg.get('run-B', 'tool0turn0')).toBe('B');
    });
  });

  describe('FIFO eviction', () => {
    it('evicts oldest entries when the aggregate cap is exceeded', () => {
      const reg = new ToolOutputReferenceRegistry({
        maxOutputSize: 10,
        maxTotalSize: 12,
      });
      reg.set('r', 'tool0turn0', '1234567'); // 7 chars
      reg.set('r', 'tool1turn0', '89'); // 9 total
      reg.set('r', 'tool2turn0', 'abc'); // 12 total — at limit
      reg.set('r', 'tool3turn0', 'XY'); // 14 → must evict oldest
      expect(reg.get('r', 'tool0turn0')).toBeUndefined();
      expect(reg.get('r', 'tool1turn0')).toBe('89');
      expect(reg.get('r', 'tool2turn0')).toBe('abc');
      expect(reg.get('r', 'tool3turn0')).toBe('XY');
    });

    it('keeps evicting oldest entries until the aggregate fits', () => {
      const reg = new ToolOutputReferenceRegistry({
        maxOutputSize: 10,
        maxTotalSize: 8,
      });
      reg.set('r', 'tool0turn0', 'aaa');
      reg.set('r', 'tool1turn0', 'bbb');
      reg.set('r', 'tool2turn0', 'ccccccc'); // total 3+3+7=13 > 8, evict aaa then bbb
      expect(reg.get('r', 'tool0turn0')).toBeUndefined();
      expect(reg.get('r', 'tool1turn0')).toBeUndefined();
      expect(reg.get('r', 'tool2turn0')).toBe('ccccccc');
    });

    it('clamps the per-output cap to maxTotalSize so no entry exceeds the aggregate', () => {
      const reg = new ToolOutputReferenceRegistry({
        maxOutputSize: 1000,
        maxTotalSize: 10,
      });
      reg.set('r', 'tool0turn0', 'x'.repeat(500));
      const stored = reg.get('r', 'tool0turn0');
      expect(stored).toBeDefined();
      expect(stored!.length).toBeLessThanOrEqual(10);
    });

    it('clamps a caller-supplied maxTotalSize to the documented hard cap', () => {
      // 50 MB requested; should be clamped to the 5 MB hard cap.
      const reg = new ToolOutputReferenceRegistry({
        maxTotalSize: 50_000_000,
      });
      // `totalLimit` getter exposes the effective post-clamp value.
      expect(reg.totalLimit).toBeLessThanOrEqual(5_000_000);
      // Per-output is also bound by the same effective total.
      expect(reg.perOutputLimit).toBeLessThanOrEqual(5_000_000);
    });

    it('evicts the oldest run bucket when maxActiveRuns is exceeded', () => {
      const reg = new ToolOutputReferenceRegistry({ maxActiveRuns: 2 });
      reg.set('run-A', 'tool0turn0', 'A');
      reg.set('run-B', 'tool0turn0', 'B');
      reg.set('run-C', 'tool0turn0', 'C');
      // run-A was the oldest insertion; LRU evicted it when run-C
      // pushed the bucket count above the cap.
      expect(reg.get('run-A', 'tool0turn0')).toBeUndefined();
      expect(reg.get('run-B', 'tool0turn0')).toBe('B');
      expect(reg.get('run-C', 'tool0turn0')).toBe('C');
    });
  });

  describe('resolve', () => {
    it('replaces placeholders in string args', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'HELLO');
      const { resolved, unresolved } = reg.resolve('r', 'echo {{tool0turn0}}');
      expect(resolved).toBe('echo HELLO');
      expect(unresolved).toEqual([]);
    });

    it('replaces placeholders in nested object args', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'DATA');
      const input = {
        command: 'cat {{tool0turn0}}',
        meta: { note: 'uses {{tool0turn0}} twice' },
      };
      const { resolved } = reg.resolve('r', input);
      expect(resolved).toEqual({
        command: 'cat DATA',
        meta: { note: 'uses DATA twice' },
      });
    });

    it('replaces placeholders inside array values', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool1turn2', '42');
      const { resolved } = reg.resolve('r', {
        args: ['--id', '{{tool1turn2}}', 'plain'],
      });
      expect(resolved).toEqual({ args: ['--id', '42', 'plain'] });
    });

    it('reports unresolved references and leaves the placeholder in place', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'known');
      const { resolved, unresolved } = reg.resolve(
        'r',
        'use {{tool0turn0}} and {{tool5turn9}}'
      );
      expect(resolved).toBe('use known and {{tool5turn9}}');
      expect(unresolved).toEqual(['tool5turn9']);
    });

    it('deduplicates repeated unresolved keys', () => {
      const reg = new ToolOutputReferenceRegistry();
      const { unresolved } = reg.resolve(
        'r',
        '{{tool7turn0}} and {{tool7turn0}} again'
      );
      expect(unresolved).toEqual(['tool7turn0']);
    });

    it('does not touch non-placeholder strings', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'X');
      const { resolved } = reg.resolve('r', 'nothing to see here');
      expect(resolved).toBe('nothing to see here');
    });

    it('leaves the top-level intent arg verbatim (display label, not a data channel)', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'HUGE-STORED-OUTPUT');
      const { resolved } = reg.resolve('r', {
        intent: 'Reusing {{tool0turn0}} for the next step',
        command: 'cat {{tool0turn0}}',
        nested: { intent: 'not top-level {{tool0turn0}}' },
      });
      expect(resolved).toEqual({
        intent: 'Reusing {{tool0turn0}} for the next step',
        command: 'cat HUGE-STORED-OUTPUT',
        nested: { intent: 'not top-level HUGE-STORED-OUTPUT' },
      });
    });

    it('preserves the intent key inside STRINGIFIED object args too', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'HUGE-STORED-OUTPUT');
      const { resolved } = reg.resolve(
        'r',
        '{"intent":"Reusing {{tool0turn0}}","command":"cat {{tool0turn0}}"}'
      );
      expect(JSON.parse(resolved as string)).toEqual({
        intent: 'Reusing {{tool0turn0}}',
        command: 'cat HUGE-STORED-OUTPUT',
      });
    });

    it('keeps plain (non-object) string args on the raw substitution path', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'DATA');
      const { resolved } = reg.resolve('r', 'echo {{tool0turn0}} intent');
      expect(resolved).toBe('echo DATA intent');
    });

    it('substitutes a BUSINESS intent arg when the caller opts in', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'STORED');
      const { resolved } = reg.resolve(
        'r',
        { intent: 'category {{tool0turn0}}', title: 'x {{tool0turn0}}' },
        { substituteIntentKey: true }
      );
      expect(resolved).toEqual({
        intent: 'category STORED',
        title: 'x STORED',
      });
    });

    it('substitutes a business intent inside stringified args when opted in', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'STORED');
      const { resolved } = reg.resolve(
        'r',
        '{"intent":"category {{tool0turn0}}"}',
        { substituteIntentKey: true }
      );
      expect(resolved).toBe('{"intent":"category STORED"}');
    });

    it('reports unresolved references from a business intent arg', () => {
      const reg = new ToolOutputReferenceRegistry();
      const { unresolved } = reg.resolve(
        'r',
        { intent: 'category {{tool9turn9}}' },
        { substituteIntentKey: true }
      );
      expect(unresolved).toEqual(['tool9turn9']);
    });

    it('snapshot views honor the opt-in too', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'STORED');
      const view = reg.snapshot('r');
      expect(view.resolve({ intent: 'x {{tool0turn0}}' }).resolved).toEqual({
        intent: 'x {{tool0turn0}}',
      });
      expect(
        view.resolve(
          { intent: 'x {{tool0turn0}}' },
          { substituteIntentKey: true }
        ).resolved
      ).toEqual({ intent: 'x STORED' });
    });

    it('passes through primitive values untouched', () => {
      const reg = new ToolOutputReferenceRegistry();
      const { resolved } = reg.resolve('r', {
        count: 3,
        enabled: true,
        note: null,
      });
      expect(resolved).toEqual({ count: 3, enabled: true, note: null });
    });
  });

  describe('snapshot', () => {
    it('restores entries and counters under a new run scope', () => {
      const source = new ToolOutputReferenceRegistry();
      source.set('source', 'tool0turn0', 'A');
      expect(source.nextTurn('source')).toBe(0);
      expect(source.claimWarnOnce('source', 'calculator')).toBe(true);

      const restored = new ToolOutputReferenceRegistry();
      restored.restoreState('branch', source.snapshotState('source'));

      expect(restored.get('branch', 'tool0turn0')).toBe('A');
      expect(restored.nextTurn('branch')).toBe(1);
      expect(restored.claimWarnOnce('branch', 'calculator')).toBe(false);
      expect(restored.get('source', 'tool0turn0')).toBeUndefined();
    });

    it('resolves against the captured state and ignores later mutations', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'OLD');
      const view = reg.snapshot('r');
      // Mutate after taking the snapshot.
      reg.set('r', 'tool0turn0', 'NEW');
      reg.set('r', 'tool1turn0', 'LATER');
      // Snapshot still resolves to the captured value and treats
      // post-snapshot additions as unresolved.
      expect(view.resolve('echo {{tool0turn0}}').resolved).toBe('echo OLD');
      const { resolved, unresolved } = view.resolve('see {{tool1turn0}}');
      expect(resolved).toBe('see {{tool1turn0}}');
      expect(unresolved).toEqual(['tool1turn0']);
    });

    it('returns an empty view for a runId with no bucket', () => {
      const reg = new ToolOutputReferenceRegistry();
      const view = reg.snapshot('never-touched');
      const { resolved, unresolved } = view.resolve('see {{tool0turn0}}');
      expect(resolved).toBe('see {{tool0turn0}}');
      expect(unresolved).toEqual(['tool0turn0']);
    });

    it('returns an isolated view per snapshot call', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'A');
      const view1 = reg.snapshot('r');
      reg.set('r', 'tool0turn0', 'B');
      const view2 = reg.snapshot('r');
      expect(view1.resolve('{{tool0turn0}}').resolved).toBe('A');
      expect(view2.resolve('{{tool0turn0}}').resolved).toBe('B');
    });
  });

  describe('annotateToolOutputWithReference', () => {
    it('injects _ref into plain JSON objects', () => {
      const content = '{"a":1,"b":"x"}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      const parsed = JSON.parse(annotated);
      expect(parsed[TOOL_OUTPUT_REF_KEY]).toBe('tool0turn0');
      expect(parsed.a).toBe(1);
      expect(parsed.b).toBe('x');
    });

    it('preserves pretty-printed formatting when the original was pretty', () => {
      const content = '{\n  "a": 1\n}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      expect(annotated).toContain('\n  "');
      const parsed = JSON.parse(annotated);
      expect(parsed[TOOL_OUTPUT_REF_KEY]).toBe('tool0turn0');
    });

    it('uses the [ref: …] prefix for JSON arrays', () => {
      const content = '[1,2,3]';
      const annotated = annotateToolOutputWithReference(content, 'tool1turn0');
      expect(annotated).toBe(`${buildReferencePrefix('tool1turn0')}\n[1,2,3]`);
    });

    it('uses the [ref: …] prefix for JSON primitives', () => {
      expect(annotateToolOutputWithReference('42', 'tool0turn0')).toBe(
        '[ref: tool0turn0]\n42'
      );
    });

    it('uses the [ref: …] prefix for plain strings', () => {
      expect(annotateToolOutputWithReference('hello', 'tool0turn0')).toBe(
        '[ref: tool0turn0]\nhello'
      );
    });

    it('falls back to the prefix on JSON _ref collision', () => {
      const content = '{"_ref":"other-value","data":1}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      expect(annotated.startsWith('[ref: tool0turn0]\n')).toBe(true);
    });

    it('injects when the existing _ref matches the target key', () => {
      const content = '{"_ref":"tool0turn0","data":1}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      const parsed = JSON.parse(annotated);
      expect(parsed._ref).toBe('tool0turn0');
      expect(parsed.data).toBe(1);
    });

    it('overwrites an existing _ref:null with the injected key', () => {
      const content = '{"_ref":null,"data":1}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      const parsed = JSON.parse(annotated);
      expect(parsed._ref).toBe('tool0turn0');
      expect(parsed.data).toBe(1);
    });

    it('places the injected _ref as the first key in the serialized JSON', () => {
      const content = '{"a":1,"b":"x"}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      expect(annotated.indexOf('"_ref"')).toBe(1);
      const annotatedFromNull = annotateToolOutputWithReference(
        '{"_ref":null,"a":1}',
        'tool0turn0'
      );
      expect(annotatedFromNull.indexOf('"_ref"')).toBe(1);
    });

    it('falls back to the prefix when parsing fails', () => {
      const content = '{ not actually json';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      expect(annotated).toBe(`[ref: tool0turn0]\n${content}`);
    });

    it('carries unresolved refs as a JSON field on parseable objects', () => {
      const content = '{"a":1}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0', [
        'tool9turn9',
        'tool7turn0',
      ]);
      const parsed = JSON.parse(annotated);
      expect(parsed._ref).toBe('tool0turn0');
      expect(parsed._unresolved_refs).toEqual(['tool9turn9', 'tool7turn0']);
      expect(parsed.a).toBe(1);
    });

    it('appends unresolved refs as a trailer line on non-object content', () => {
      const annotated = annotateToolOutputWithReference(
        'plain text',
        'tool0turn0',
        ['tool9turn9']
      );
      expect(annotated).toBe(
        '[ref: tool0turn0]\nplain text\n[unresolved refs: tool9turn9]'
      );
    });

    it('supports unresolved-only annotation (no ref key)', () => {
      const annotated = annotateToolOutputWithReference(
        'error text',
        undefined,
        ['tool9turn9']
      );
      expect(annotated).toBe('error text\n[unresolved refs: tool9turn9]');
    });

    it('keeps JSON parseable for unresolved-only annotation on object content', () => {
      const annotated = annotateToolOutputWithReference(
        '{"error":"bad"}',
        undefined,
        ['tool9turn9']
      );
      const parsed = JSON.parse(annotated);
      expect(parsed._unresolved_refs).toEqual(['tool9turn9']);
      expect(parsed.error).toBe('bad');
      expect(parsed._ref).toBeUndefined();
    });

    it('returns content unchanged when there is nothing to annotate', () => {
      expect(annotateToolOutputWithReference('plain', undefined, [])).toBe(
        'plain'
      );
    });

    it('preserves an existing _unresolved_refs payload when only injecting a ref key', () => {
      const content = '{"data":1,"_unresolved_refs":["user-supplied"]}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      const parsed = JSON.parse(annotated);
      expect(parsed._ref).toBe('tool0turn0');
      expect(parsed._unresolved_refs).toEqual(['user-supplied']);
      expect(parsed.data).toBe(1);
    });

    it('preserves an existing _ref payload on the unresolved-only path', () => {
      const content = '{"data":1,"_ref":"preserved-value"}';
      const annotated = annotateToolOutputWithReference(content, undefined, [
        'tool9turn9',
      ]);
      const parsed = JSON.parse(annotated);
      expect(parsed._ref).toBe('preserved-value');
      expect(parsed._unresolved_refs).toEqual(['tool9turn9']);
      expect(parsed.data).toBe(1);
    });

    it('falls back to the prefix when _unresolved_refs conflicts with a non-matching array', () => {
      const content = '{"data":1,"_unresolved_refs":["legacy"]}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0', [
        'tool9turn9',
      ]);
      expect(annotated.startsWith('[ref: tool0turn0]\n')).toBe(true);
      expect(annotated).toContain('[unresolved refs: tool9turn9]');
    });

    it('accepts a deep-equal existing _unresolved_refs array', () => {
      const content = '{"data":1,"_unresolved_refs":["tool9turn9"]}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0', [
        'tool9turn9',
      ]);
      const parsed = JSON.parse(annotated);
      expect(parsed._unresolved_refs).toEqual(['tool9turn9']);
      expect(parsed._ref).toBe('tool0turn0');
    });
  });

  describe('TOOL_OUTPUT_REF_PATTERN', () => {
    it('matches braced tool<N>turn<M> tokens and captures the key', () => {
      const match = '{{tool0turn0}}'.match(TOOL_OUTPUT_REF_PATTERN);
      expect(match?.[1]).toBe('tool0turn0');
    });

    it('rejects bare tool<N>turn<M> tokens without braces', () => {
      expect(TOOL_OUTPUT_REF_PATTERN.test('tool0turn0')).toBe(false);
    });

    it('is non-global so callers cannot trip on stale lastIndex', () => {
      expect(TOOL_OUTPUT_REF_PATTERN.flags).not.toContain('g');
    });
  });
});

it('round-trips reference source provenance and clears it on unprotected overwrite', () => {
  const registry = new ToolOutputReferenceRegistry();
  const protection = { version: 1 as const, toolName: 'lookup', toolCallId: 'source-control', protected: true };
  registry.set('source', 'tool0turn0', 'Allowed canonical control', protection);
  const snapshot = registry.snapshotState('source');
  expect(snapshot.entries[0].protection).toEqual(protection);
  snapshot.entries[0].protection!.toolName = 'changed-copy';
  expect(registry.snapshotState('source').entries[0].protection!.toolName).toBe('lookup');
  registry.restoreState('resume', registry.snapshotState('source'));
  expect(registry.snapshotState('resume').entries[0].protection).toEqual(protection);
  registry.set('resume', 'tool0turn0', 'New uncertified control');
  expect(registry.snapshotState('resume').entries[0].protection).toBeUndefined();
});

it('projects canonical replay inputs into the live registry without dropping unrelated concurrent outputs', () => {
  const registry = new ToolOutputReferenceRegistry();
  registry.set('resume', 'tool0turn0', 'Previous allowed control');
  registry.set('resume', 'tool0turn4', 'Concurrent allowed control');
  const protection = { version: 1 as const, toolName: 'lookup', toolCallId: 'source-control', protected: true };
  const view = registry.resumeBatch('resume', { entries: [{ key: 'tool0turn0', value: 'Canonical allowed control', protection }], turnCounter: 1, warnedNonStringTools: [] }, true);
  expect(view.resolve('{{' + 'tool0turn0' + '}}').resolved).toBe('Canonical allowed control');
  expect(registry.get('resume', 'tool0turn0')).toBe('Canonical allowed control');
  expect(registry.get('resume', 'tool0turn4')).toBe('Concurrent allowed control');
  expect(registry.snapshotState('resume').entries[1].protection).toEqual(protection);
});
