import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { PostToolUseHookOutput, PreToolUseHookOutput } from '@/hooks';
import type * as t from '@/types';
import {
  ToolOutputReferenceRegistry,
  annotateMessagesForLLM,
  nextReferenceTurn,
} from '../toolOutputReferences';
import { GraphEvents, Providers } from '@/common';
import { Run } from '@/run';
import * as events from '@/utils/events';
import { HookRegistry } from '@/hooks';
import { ToolNode } from '../ToolNode';

/**
 * Reads the lazy ref-metadata stamped onto a `ToolMessage` by ToolNode.
 * The metadata replaces the durable `[ref: …]` content mutation that the
 * earlier eager-annotation design used; the LLM-facing annotation is
 * applied at request time by `annotateMessagesForLLM` instead.
 */
function getRefKey(msg: ToolMessage): string | undefined {
  return (msg.additional_kwargs as { _refKey?: string } | undefined)?._refKey;
}
function getRefScope(msg: ToolMessage): string | undefined {
  return (msg.additional_kwargs as { _refScope?: string } | undefined)
    ?._refScope;
}
function getUnresolvedRefs(msg: ToolMessage): string[] {
  return (
    (msg.additional_kwargs as { _unresolvedRefs?: string[] } | undefined)
      ?._unresolvedRefs ?? []
  );
}

/**
 * Captures the `command` arg each time the tool is invoked and returns
 * a configurable string output. The tool shape matches a typical bash
 * executor: single required string arg, string response.
 */
function createEchoTool(options: {
  capturedArgs: string[];
  outputs: string[];
  name?: string;
}): StructuredToolInterface {
  const { capturedArgs, outputs, name = 'echo' } = options;
  let callCount = 0;
  return tool(
    async (input) => {
      const args = input as { command: string };
      capturedArgs.push(args.command);
      const output = outputs[callCount] ?? outputs[outputs.length - 1];
      callCount++;
      return output;
    },
    {
      name,
      description: 'Echo test tool',
      schema: z.object({ command: z.string() }),
    }
  ) as unknown as StructuredToolInterface;
}

function aiMsgWithCalls(
  calls: Array<{ id: string; name: string; command: string }>
): AIMessage {
  return new AIMessage({
    content: '',
    tool_calls: calls.map((c) => ({
      id: c.id,
      name: c.name,
      args: { command: c.command },
    })),
  });
}

async function invokeBatch(
  toolNode: ToolNode,
  calls: Array<{ id: string; name: string; command: string }>,
  runId: string = 'test-run'
): Promise<ToolMessage[]> {
  const aiMsg = aiMsgWithCalls(calls);
  const result = (await toolNode.invoke(
    { messages: [aiMsg] },
    { configurable: { run_id: runId } }
  )) as ToolMessage[] | { messages: ToolMessage[] };
  return Array.isArray(result) ? result : result.messages;
}

describe('ToolNode tool output references', () => {
  describe('disabled (default)', () => {
    it('does not annotate outputs or register anything when disabled', async () => {
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({
        capturedArgs,
        outputs: ['plain-output'],
      });
      const node = new ToolNode({ tools: [t1] });

      const [msg] = await invokeBatch(node, [
        { id: 'c1', name: 'echo', command: 'hello' },
      ]);

      expect(msg.content).toBe('plain-output');
      expect(node._unsafeGetToolOutputRegistry()).toBeUndefined();
    });

    it('does not substitute placeholders when disabled', async () => {
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({ capturedArgs, outputs: ['X'] });
      const node = new ToolNode({ tools: [t1] });

      await invokeBatch(node, [
        { id: 'c1', name: 'echo', command: 'raw {{tool0turn0}}' },
      ]);

      expect(capturedArgs).toEqual(['raw {{tool0turn0}}']);
    });

    it('bounds structured ToolMessage output before its first model call', async () => {
      const structuredTool = tool(
        async () =>
          new ToolMessage({
            status: 'success',
            content: [
              {
                type: 'json',
                rows: Array.from({ length: 20 }, (_, index) => ({
                  id: index,
                  value: `${'x'.repeat(100)}-${index}`,
                })),
              },
            ],
            artifact: { source: 'clickhouse' },
            metadata: { requestId: 'request-1' },
            name: 'run_select_query',
            tool_call_id: 'c1',
          }),
        {
          name: 'run_select_query',
          description: 'returns structured rows',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;
      const node = new ToolNode({
        tools: [structuredTool],
        maxToolResultChars: 200,
      });

      const [msg] = await invokeBatch(node, [
        {
          id: 'c1',
          name: 'run_select_query',
          command: 'SELECT * FROM events',
        },
      ]);

      expect(typeof msg.content).toBe('string');
      expect(msg.content).toContain('truncated');
      expect((msg.content as string).length).toBeLessThanOrEqual(200);
      expect(msg.tool_call_id).toBe('c1');
      expect(msg.name).toBe('run_select_query');
      expect(msg.status).toBe('success');
      expect(msg.artifact).toEqual({ source: 'clickhouse' });
      expect(msg.metadata).toEqual({ requestId: 'request-1' });
    });
  });

  describe('enabled', () => {
    it('keeps string outputs clean and stamps the ref key as metadata', async () => {
      const t1 = createEchoTool({
        capturedArgs: [],
        outputs: ['hello world'],
      });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(node, [
        { id: 'c1', name: 'echo', command: 'run' },
      ]);

      expect(msg.content).toBe('hello world');
      expect(getRefKey(msg)).toBe('tool0turn0');
      /**
       * `_refScope` is what lets `annotateMessagesForLLM` recover the
       * registry bucket at request time without re-deriving it from
       * `config.configurable.run_id` (which fails for anonymous
       * batches). For named runs it equals the run_id.
       */
      expect(getRefScope(msg)).toBe('test-run');
      expect(getUnresolvedRefs(msg)).toEqual([]);
    });

    it('keeps JSON-object string outputs unmodified and stamps ref metadata', async () => {
      const t1 = createEchoTool({
        capturedArgs: [],
        outputs: ['{"a":1,"b":"x"}'],
      });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(node, [
        { id: 'c1', name: 'echo', command: 'run' },
      ]);

      const parsed = JSON.parse(msg.content as string);
      expect(parsed.a).toBe(1);
      expect(parsed.b).toBe('x');
      expect(parsed._ref).toBeUndefined();
      expect(getRefKey(msg)).toBe('tool0turn0');
    });

    it('keeps JSON array outputs unmodified and stamps ref metadata', async () => {
      const t1 = createEchoTool({ capturedArgs: [], outputs: ['[1,2,3]'] });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(node, [
        { id: 'c1', name: 'echo', command: 'run' },
      ]);

      expect(msg.content).toBe('[1,2,3]');
      expect(getRefKey(msg)).toBe('tool0turn0');
    });

    it('registers the un-annotated output for piping into later calls', async () => {
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({
        capturedArgs,
        outputs: ['raw-payload', 'second-call'],
      });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true },
      });

      await invokeBatch(node, [{ id: 'c1', name: 'echo', command: 'first' }]);
      await invokeBatch(node, [
        {
          id: 'c2',
          name: 'echo',
          command: 'echo {{tool0turn0}}',
        },
      ]);

      expect(capturedArgs).toEqual(['first', 'echo raw-payload']);
    });

    it('keeps generated-file summaries out of registered outputs', async () => {
      const rawOutput = [
        'stdout:',
        '{"ok":true}',
        '',
        'Generated files:',
        'Session files: 1 persisted file(s) are available in /mnt/data, including 0 image(s). Use known /mnt/data paths directly in later code-tool calls. The app displays files/images automatically; do not invent download links or wrap generated images in Markdown.',
      ].join('\n');
      const cleanOutput = 'stdout:\n{"ok":true}';
      const capturedArgs: string[] = [];
      const filesTool = tool(
        async () =>
          new ToolMessage({
            status: 'success',
            content: rawOutput,
            name: 'files',
            tool_call_id: 'c1',
          }),
        {
          name: 'files',
          description: 'returns generated-file summary output',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;
      const echo = createEchoTool({
        capturedArgs,
        outputs: ['resolved'],
      });
      const node = new ToolNode({
        tools: [filesTool, echo],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(node, [
        { id: 'c1', name: 'files', command: 'first' },
      ]);
      await invokeBatch(node, [
        {
          id: 'c2',
          name: 'echo',
          command: 'echo {{tool0turn0}}',
        },
      ]);

      expect(msg.content).toBe(rawOutput);
      expect(getRefKey(msg)).toBe('tool0turn0');
      expect(
        node._unsafeGetToolOutputRegistry()!.get('test-run', 'tool0turn0')
      ).toBe(cleanOutput);
      expect(capturedArgs).toEqual([`echo ${cleanOutput}`]);
    });

    it('increments the turn counter per ToolNode batch', async () => {
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({
        capturedArgs,
        outputs: ['one', 'two', 'three'],
      });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true },
      });

      const [m0] = await invokeBatch(node, [
        { id: 'b1c1', name: 'echo', command: 'a' },
      ]);
      const [m1] = await invokeBatch(node, [
        { id: 'b2c1', name: 'echo', command: 'b' },
      ]);
      const [m2] = await invokeBatch(node, [
        { id: 'b3c1', name: 'echo', command: '{{tool0turn1}}' },
      ]);

      expect(getRefKey(m0)).toBe('tool0turn0');
      expect(getRefKey(m1)).toBe('tool0turn1');
      expect(getRefKey(m2)).toBe('tool0turn2');
      expect(capturedArgs[2]).toBe('two');
    });

    it('uses array index within a batch for the tool<idx> segment', async () => {
      const capturedA: string[] = [];
      const capturedB: string[] = [];
      const tA = createEchoTool({
        capturedArgs: capturedA,
        outputs: ['A-out'],
        name: 'alpha',
      });
      const tB = createEchoTool({
        capturedArgs: capturedB,
        outputs: ['B-out'],
        name: 'beta',
      });
      const node = new ToolNode({
        tools: [tA, tB],
        toolOutputReferences: { enabled: true },
      });

      const messages = await invokeBatch(node, [
        { id: 'c1', name: 'alpha', command: 'a' },
        { id: 'c2', name: 'beta', command: 'b' },
      ]);

      expect(getRefKey(messages[0])).toBe('tool0turn0');
      expect(getRefKey(messages[1])).toBe('tool1turn0');
    });

    it('reports unresolved placeholders after the output', async () => {
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({ capturedArgs, outputs: ['done'] });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(node, [
        {
          id: 'c1',
          name: 'echo',
          command: 'see {{tool9turn9}}',
        },
      ]);

      expect(capturedArgs[0]).toBe('see {{tool9turn9}}');
      expect(msg.content).toBe('done');
      expect(getUnresolvedRefs(msg)).toEqual(['tool9turn9']);
    });

    it('stores the raw untruncated output in the registry, independent of the LLM-visible truncation', async () => {
      const raw = 'X'.repeat(8_000);
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({
        capturedArgs,
        outputs: [raw, 'second'],
      });
      const node = new ToolNode({
        tools: [t1],
        maxToolResultChars: 200,
        toolOutputReferences: { enabled: true },
      });

      const [first] = await invokeBatch(
        node,
        [{ id: 'c1', name: 'echo', command: 'first' }],
        'raw-preservation'
      );

      expect((first.content as string).length).toBeLessThan(raw.length);
      expect(first.content).toContain('truncated');

      await invokeBatch(
        node,
        [{ id: 'c2', name: 'echo', command: 'echo {{tool0turn0}}' }],
        'raw-preservation'
      );

      expect(capturedArgs[1]).toBe(`echo ${raw}`);
      expect(
        node
          ._unsafeGetToolOutputRegistry()!
          .get('raw-preservation', 'tool0turn0')
      ).toBe(raw);
    });

    it('preserves structured output exactly up to the registry limit', async () => {
      const structured = {
        rows: Array.from({ length: 10 }, (_, index) => ({
          id: index,
          value: `row-${index}-${'x'.repeat(100)}`,
        })),
      };
      const exact = JSON.stringify(structured);
      const capturedArgs: string[] = [];
      let callCount = 0;
      const structuredTool = createEchoTool({
        capturedArgs: [],
        outputs: ['unused'],
        name: 'structured',
      });
      (
        structuredTool as unknown as {
          invoke: (input: {
            args: { command: string };
          }) => Promise<typeof structured | string>;
        }
      ).invoke = async (input) => {
        capturedArgs.push(input.args.command);
        return callCount++ === 0 ? structured : 'done';
      };
      const node = new ToolNode({
        tools: [structuredTool],
        maxToolResultChars: 100,
        toolOutputReferences: {
          enabled: true,
          maxOutputSize: exact.length,
        },
      });

      const [first] = await invokeBatch(
        node,
        [{ id: 'c1', name: 'structured', command: 'first' }],
        'structured-raw-preservation'
      );
      await invokeBatch(
        node,
        [
          {
            id: 'c2',
            name: 'structured',
            command: '{{tool0turn0}}',
          },
        ],
        'structured-raw-preservation'
      );

      expect((first.content as string).length).toBeLessThanOrEqual(100);
      expect(first.content).toContain('truncated');
      expect(capturedArgs[1]).toBe(exact);
      expect(
        node
          ._unsafeGetToolOutputRegistry()!
          .get('structured-raw-preservation', 'tool0turn0')
      ).toBe(exact);
    });

    it('uses each batch\'s own turn when ToolNode is invoked concurrently within a run', async () => {
      const gates: Record<string, () => void> = {};
      const slowTool = tool(
        async (input) => {
          const args = input as { command: string };
          await new Promise<void>((resolve) => {
            gates[args.command] = resolve;
          });
          return `output-${args.command}`;
        },
        {
          name: 'slow',
          description: 'awaits a per-command gate',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;

      const node = new ToolNode({
        tools: [slowTool],
        toolOutputReferences: { enabled: true },
      });

      // Two batches of the SAME run, started concurrently — batch A
      // captures turn 0 in its sync prefix, batch B captures turn 1.
      // If the turn were read from shared state after the awaits, the
      // reads would race and both batches would see the latest value.
      const first = node.invoke(
        {
          messages: [aiMsgWithCalls([{ id: 'a', name: 'slow', command: 'A' }])],
        },
        { configurable: { run_id: 'concurrent-run' } }
      );
      const second = node.invoke(
        {
          messages: [aiMsgWithCalls([{ id: 'b', name: 'slow', command: 'B' }])],
        },
        { configurable: { run_id: 'concurrent-run' } }
      );

      await new Promise<void>((resolve) => {
        const check = (): void => {
          if (
            Object.prototype.hasOwnProperty.call(gates, 'A') &&
            Object.prototype.hasOwnProperty.call(gates, 'B')
          ) {
            resolve();
          } else {
            setTimeout(check, 5);
          }
        };
        check();
      });
      // Release B first (the later-scheduled batch), then A. Under the
      // old code this would bake turn=1 into BOTH results because
      // `currentTurn` was overwritten during B's sync prefix.
      gates.B();
      gates.A();

      const [resA, resB] = (await Promise.all([first, second])) as Array<{
        messages: ToolMessage[];
      }>;

      expect(getRefKey(resA.messages[0])).toBe('tool0turn0');
      expect(resA.messages[0].content).toBe('output-A');
      expect(getRefKey(resB.messages[0])).toBe('tool0turn1');
      expect(resB.messages[0].content).toBe('output-B');

      const registry = node._unsafeGetToolOutputRegistry()!;
      expect(registry.get('concurrent-run', 'tool0turn0')).toBe('output-A');
      expect(registry.get('concurrent-run', 'tool0turn1')).toBe('output-B');
    });

    it('clips registered outputs to maxOutputSize', async () => {
      const t1 = createEchoTool({
        capturedArgs: [],
        outputs: ['{"payload":"' + 'y'.repeat(200) + '"}'],
      });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true, maxOutputSize: 40 },
      });

      await invokeBatch(node, [{ id: 'c1', name: 'echo', command: 'x' }]);

      const registry = node._unsafeGetToolOutputRegistry();
      expect(registry).toBeDefined();
      expect(
        registry!.get('test-run', 'tool0turn0')!.length
      ).toBeLessThanOrEqual(40);
    });

    it('honors maxTotalSize via FIFO eviction across batches', async () => {
      const t1 = createEchoTool({
        capturedArgs: [],
        outputs: ['aaaaa', 'bbbbb', 'ccccc'],
      });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: {
          enabled: true,
          maxOutputSize: 10,
          maxTotalSize: 10,
        },
      });

      await invokeBatch(node, [{ id: 'c1', name: 'echo', command: 'x' }]);
      await invokeBatch(node, [{ id: 'c2', name: 'echo', command: 'x' }]);
      await invokeBatch(node, [{ id: 'c3', name: 'echo', command: 'x' }]);

      const registry = node._unsafeGetToolOutputRegistry()!;
      expect(registry.get('test-run', 'tool0turn0')).toBeUndefined();
      expect(registry.get('test-run', 'tool0turn1')).toBe('bbbbb');
      expect(registry.get('test-run', 'tool0turn2')).toBe('ccccc');
    });

    it('does not register error outputs', async () => {
      const boom = tool(
        async () => {
          throw new Error('nope');
        },
        {
          name: 'boom',
          description: 'always errors',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;

      const node = new ToolNode({
        tools: [boom],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(node, [
        { id: 'c1', name: 'boom', command: 'x' },
      ]);

      expect(getRefKey(msg)).toBeUndefined();
      expect(
        node._unsafeGetToolOutputRegistry()!.get('test-run', 'tool0turn0')
      ).toBeUndefined();
    });

    it('surfaces unresolved refs on thrown-error ToolMessages', async () => {
      const boom = tool(
        async () => {
          throw new Error('nope');
        },
        {
          name: 'boom',
          description: 'always errors',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;

      const node = new ToolNode({
        tools: [boom],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(node, [
        { id: 'c1', name: 'boom', command: 'see {{tool9turn9}}' },
      ]);

      expect(msg.content).toContain('Error: nope');
      expect(msg.content as string).not.toContain('[unresolved refs:');
      expect(getUnresolvedRefs(msg)).toEqual(['tool9turn9']);
    });

    it('surfaces unresolved refs on tool-returned error ToolMessages', async () => {
      const errReturn = tool(
        async () =>
          new ToolMessage({
            status: 'error',
            content: 'handled failure',
            name: 'errReturn',
            tool_call_id: 'c1',
          }),
        {
          name: 'errReturn',
          description: 'returns error ToolMessage',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;

      const node = new ToolNode({
        tools: [errReturn],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(node, [
        { id: 'c1', name: 'errReturn', command: 'see {{tool9turn9}}' },
      ]);

      expect(msg.content).toBe('handled failure');
      expect(getUnresolvedRefs(msg)).toEqual(['tool9turn9']);
    });

    it('isolates state between overlapping runs on the same ToolNode', async () => {
      const sharedRegistry = new ToolOutputReferenceRegistry();
      const capturedArgs: string[] = [];
      const tl = createEchoTool({
        capturedArgs,
        outputs: ['out-A', 'out-B', 'resolved-in-B'],
      });

      const node = new ToolNode({
        tools: [tl],
        toolOutputRegistry: sharedRegistry,
      });

      // Run A records `tool0turn0` → 'out-A' in its bucket.
      await node.invoke(
        {
          messages: [
            aiMsgWithCalls([{ id: 'a1', name: 'echo', command: 'a' }]),
          ],
        },
        { configurable: { run_id: 'run-A' } }
      );
      expect(sharedRegistry.get('run-A', 'tool0turn0')).toBe('out-A');

      // Run B records `tool0turn0` → 'out-B' in its own bucket.
      // Under the old global-reset design, starting run B would have
      // wiped run A's registered output; with partitioning, A's
      // bucket survives untouched.
      await node.invoke(
        {
          messages: [
            aiMsgWithCalls([{ id: 'b1', name: 'echo', command: 'b' }]),
          ],
        },
        { configurable: { run_id: 'run-B' } }
      );
      expect(sharedRegistry.get('run-A', 'tool0turn0')).toBe('out-A');
      expect(sharedRegistry.get('run-B', 'tool0turn0')).toBe('out-B');

      // Run B's next batch resolves `{{tool0turn0}}` against its own
      // partition (out-B), not run A's partition (out-A).
      await node.invoke(
        {
          messages: [
            aiMsgWithCalls([
              { id: 'b2', name: 'echo', command: 'see {{tool0turn0}}' },
            ]),
          ],
        },
        { configurable: { run_id: 'run-B' } }
      );
      expect(capturedArgs[2]).toBe('see out-B');
    });

    it('gives concurrent anonymous invocations independent scopes', async () => {
      const gates: Record<string, () => void> = {};
      const slowTool = tool(
        async (input) => {
          const args = input as { command: string };
          await new Promise<void>((resolve) => {
            gates[args.command] = resolve;
          });
          return `out-${args.command}`;
        },
        {
          name: 'slow',
          description: 'awaits a per-command gate',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;

      const node = new ToolNode({
        tools: [slowTool],
        toolOutputReferences: { enabled: true },
      });

      // Two invocations without `run_id`, started concurrently. Before
      // the unique-anon-scope fix, the second invocation's sync prefix
      // would have deleted the shared anonymous bucket that the first
      // invocation's tool was about to register into.
      const first = node.invoke({
        messages: [aiMsgWithCalls([{ id: 'a1', name: 'slow', command: 'A' }])],
      });
      const second = node.invoke({
        messages: [aiMsgWithCalls([{ id: 'b1', name: 'slow', command: 'B' }])],
      });

      await new Promise<void>((resolve) => {
        const check = (): void => {
          if (
            Object.prototype.hasOwnProperty.call(gates, 'A') &&
            Object.prototype.hasOwnProperty.call(gates, 'B')
          ) {
            resolve();
          } else {
            setTimeout(check, 5);
          }
        };
        check();
      });
      gates.B();
      gates.A();

      const [resA, resB] = (await Promise.all([first, second])) as Array<{
        messages: ToolMessage[];
      }>;

      // Each invocation stamps its own ref metadata — neither's
      // registered tool0turn0 was clobbered by the other's sync-prefix
      // reset.
      expect(getRefKey(resA.messages[0])).toBe('tool0turn0');
      expect(resA.messages[0].content).toBe('out-A');
      expect(getRefKey(resB.messages[0])).toBe('tool0turn0');
      expect(resB.messages[0].content).toBe('out-B');

      /**
       * Each anonymous invocation stamps a distinct synthetic
       * `_refScope` so the lazy annotation transform can later look
       * up the right registry bucket — `config.configurable.run_id`
       * is undefined for both calls and would collapse them to the
       * same `\0anon` bucket without this stamping.
       */
      const scopeA = getRefScope(resA.messages[0]);
      const scopeB = getRefScope(resB.messages[0]);
      expect(scopeA).toMatch(/^\0anon-\d+$/);
      expect(scopeB).toMatch(/^\0anon-\d+$/);
      expect(scopeA).not.toBe(scopeB);
    });

    it('clears state on every batch when run_id is absent (anonymous caller)', async () => {
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({
        capturedArgs,
        outputs: ['first-anonymous', 'second-anonymous'],
      });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true },
      });

      await node.invoke({
        messages: [aiMsgWithCalls([{ id: 'a1', name: 'echo', command: 'a' }])],
      });
      const result = (await node.invoke({
        messages: [
          aiMsgWithCalls([
            { id: 'a2', name: 'echo', command: 'echo {{tool0turn0}}' },
          ]),
        ],
      })) as { messages: ToolMessage[] };

      expect(capturedArgs[1]).toBe('echo {{tool0turn0}}');
      expect(getUnresolvedRefs(result.messages[0])).toEqual(['tool0turn0']);
    });

    it('lets two ToolNodes sharing a registry resolve each other\'s refs', async () => {
      const sharedRegistry = new ToolOutputReferenceRegistry();
      const capturedA: string[] = [];
      const capturedB: string[] = [];
      const toolA = createEchoTool({
        capturedArgs: capturedA,
        outputs: ['agent-A-output'],
        name: 'alpha',
      });
      const toolB = createEchoTool({
        capturedArgs: capturedB,
        outputs: ['agent-B-output'],
        name: 'beta',
      });

      // Two independent ToolNodes (simulating one per agent in a
      // multi-agent graph) sharing one registry instance.
      const nodeA = new ToolNode({
        tools: [toolA],
        toolOutputRegistry: sharedRegistry,
      });
      const nodeB = new ToolNode({
        tools: [toolB],
        toolOutputRegistry: sharedRegistry,
      });

      await nodeA.invoke(
        {
          messages: [
            aiMsgWithCalls([{ id: 'a1', name: 'alpha', command: 'first' }]),
          ],
        },
        { configurable: { run_id: 'shared-run' } }
      );

      await nodeB.invoke(
        {
          messages: [
            aiMsgWithCalls([
              { id: 'b1', name: 'beta', command: 'see {{tool0turn0}}' },
            ]),
          ],
        },
        { configurable: { run_id: 'shared-run' } }
      );

      // nodeB resolved nodeA's tool0turn0 placeholder (cross-node),
      // and its own output landed under the *next* turn (1), not 0.
      expect(capturedB[0]).toBe('see agent-A-output');
      expect(sharedRegistry.get('shared-run', 'tool0turn0')).toBe(
        'agent-A-output'
      );
      expect(sharedRegistry.get('shared-run', 'tool0turn1')).toBe(
        'agent-B-output'
      );
    });

    it('emits resolved args in ON_RUN_STEP_COMPLETED, not the template', async () => {
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({
        capturedArgs,
        outputs: ['STORED', 'second'],
      });
      const stepCompletedArgs: string[] = [];
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event === 'on_run_step_completed') {
            const step = data as {
              result: { tool_call: { args: string } };
            };
            stepCompletedArgs.push(step.result.tool_call.args);
          }
        });

      const node = new ToolNode({
        tools: [t1],
        toolCallStepIds: new Map([
          ['a1', 'step_a1'],
          ['a2', 'step_a2'],
        ]),
        toolOutputReferences: { enabled: true },
      });

      await invokeBatch(
        node,
        [{ id: 'a1', name: 'echo', command: 'first' }],
        'resolved-args'
      );
      await invokeBatch(
        node,
        [
          {
            id: 'a2',
            name: 'echo',
            command: 'echo {{tool0turn0}}',
          },
        ],
        'resolved-args'
      );

      // Second step-completed event should reflect the post-
      // substitution command, not the `{{…}}` template.
      expect(stepCompletedArgs).toHaveLength(2);
      expect(JSON.parse(stepCompletedArgs[1]).command).toBe('echo STORED');
    });

    it('records unresolved refs on multipart ToolMessage content within the cap', async () => {
      const complexTool = tool(
        async () =>
          new ToolMessage({
            status: 'success',
            content: [
              { type: 'text', text: 'data' },
              { type: 'image_url', image_url: { url: 'data:...' } },
            ],
            name: 'complex',
            tool_call_id: 'c1',
          }),
        {
          name: 'complex',
          description: 'returns multi-part content',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;

      const node = new ToolNode({
        tools: [complexTool],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(
        node,
        [{ id: 'c1', name: 'complex', command: 'see {{tool9turn9}}' }],
        'non-string'
      );

      expect(Array.isArray(msg.content)).toBe(true);
      const blocks = msg.content as Array<{ type: string; text?: string }>;
      // In-budget multi-part content stays intact; the lazy transform handles
      // the unresolved-refs warning at request time.
      expect(blocks).toHaveLength(2);
      expect(blocks[0].type).toBe('text');
      expect(blocks[0].text).toBe('data');
      expect(blocks[1].type).toBe('image_url');
      expect(getUnresolvedRefs(msg)).toEqual(['tool9turn9']);
    });

    it('resets the registry and turn counter when the runId changes', async () => {
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({
        capturedArgs,
        outputs: ['from-run-A', 'from-run-B'],
      });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true },
      });

      const aiMsgA = aiMsgWithCalls([
        { id: 'a1', name: 'echo', command: 'first' },
      ]);
      await node.invoke(
        { messages: [aiMsgA] },
        { configurable: { run_id: 'run-A' } }
      );

      const aiMsgB = aiMsgWithCalls([
        {
          id: 'b1',
          name: 'echo',
          command: 'echo {{tool0turn0}}',
        },
      ]);
      const resultB = (await node.invoke(
        { messages: [aiMsgB] },
        { configurable: { run_id: 'run-B' } }
      )) as { messages: ToolMessage[] };

      expect(capturedArgs[1]).toBe('echo {{tool0turn0}}');
      expect(resultB.messages[0].content).toBe('from-run-B');
      expect(getRefKey(resultB.messages[0])).toBe('tool0turn0');
      expect(getUnresolvedRefs(resultB.messages[0])).toEqual(['tool0turn0']);
    });
  });

  describe('event-driven dispatch path', () => {
    afterEach(() => {
      jest.restoreAllMocks();
    });

    function mockEventDispatch(mockResults: t.ToolExecuteResult[]): void {
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== 'on_tool_execute') {
            return;
          }
          const request = data as Record<string, unknown>;
          if (typeof request.resolve === 'function') {
            (request.resolve as (r: t.ToolExecuteResult[]) => void)(
              mockResults
            );
          }
        });
    }

    function createSchemaStub(name: string): StructuredToolInterface {
      return tool(async () => 'unused', {
        name,
        description: 'schema-only stub; host executes via ON_TOOL_EXECUTE',
        schema: z.object({ command: z.string() }),
      }) as unknown as StructuredToolInterface;
    }

    it('keeps host-returned output clean and stamps the ref key as metadata', async () => {
      const node = new ToolNode({
        tools: [createSchemaStub('echo')],
        eventDrivenMode: true,
        agentId: 'agent-x',
        toolCallStepIds: new Map([['ec1', 'step_ec1']]),
        toolOutputReferences: { enabled: true },
      });

      mockEventDispatch([
        { toolCallId: 'ec1', content: 'host-output', status: 'success' },
      ]);

      const aiMsg = new AIMessage({
        content: '',
        tool_calls: [{ id: 'ec1', name: 'echo', args: { command: 'run' } }],
      });
      const result = (await node.invoke(
        { messages: [aiMsg] },
        { configurable: { run_id: 'run-host' } }
      )) as { messages: ToolMessage[] };

      expect(result.messages[0].content).toBe('host-output');
      expect(getRefKey(result.messages[0])).toBe('tool0turn0');
      expect(
        node._unsafeGetToolOutputRegistry()!.get('run-host', 'tool0turn0')
      ).toBe('host-output');
    });

    it('substitutes `{{…}}` in the request sent to the host', async () => {
      const node = new ToolNode({
        tools: [createSchemaStub('echo')],
        eventDrivenMode: true,
        agentId: 'agent-x',
        toolCallStepIds: new Map([
          ['ec1', 'step_ec1'],
          ['ec2', 'step_ec2'],
        ]),
        toolOutputReferences: { enabled: true },
      });

      mockEventDispatch([
        { toolCallId: 'ec1', content: 'FIRST', status: 'success' },
      ]);
      await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [{ id: 'ec1', name: 'echo', args: { command: 'a' } }],
            }),
          ],
        },
        { configurable: { run_id: 'run-subst' } }
      );

      jest.restoreAllMocks();
      const capturedRequests: t.ToolCallRequest[] = [];
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== 'on_tool_execute') {
            return;
          }
          const batch = data as t.ToolExecuteBatchRequest;
          for (const req of batch.toolCalls) {
            capturedRequests.push(req);
          }
          batch.resolve([
            { toolCallId: 'ec2', content: 'SECOND', status: 'success' },
          ]);
        });

      await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                {
                  id: 'ec2',
                  name: 'echo',
                  args: { command: 'see {{tool0turn0}}' },
                },
              ],
            }),
          ],
        },
        { configurable: { run_id: 'run-subst' } }
      );

      expect(capturedRequests).toHaveLength(1);
      expect(capturedRequests[0].args).toEqual({ command: 'see FIRST' });
    });

    it('surfaces unresolved refs on host-returned error results', async () => {
      const node = new ToolNode({
        tools: [createSchemaStub('echo')],
        eventDrivenMode: true,
        agentId: 'agent-x',
        toolCallStepIds: new Map([['ec1', 'step_ec1']]),
        toolOutputReferences: { enabled: true },
      });

      mockEventDispatch([
        {
          toolCallId: 'ec1',
          content: '',
          status: 'error',
          errorMessage: 'host failure',
        },
      ]);
      const result = (await node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              {
                id: 'ec1',
                name: 'echo',
                args: { command: 'see {{tool9turn9}}' },
              },
            ],
          }),
        ],
      })) as { messages: ToolMessage[] };

      expect(result.messages[0].content).toContain('Error: host failure');
      expect(result.messages[0].content as string).not.toContain(
        '[unresolved refs:'
      );
      expect(getUnresolvedRefs(result.messages[0])).toEqual(['tool9turn9']);
    });

    it('reports unresolved refs even when the host succeeds', async () => {
      const node = new ToolNode({
        tools: [createSchemaStub('echo')],
        eventDrivenMode: true,
        agentId: 'agent-x',
        toolCallStepIds: new Map([['ec1', 'step_ec1']]),
        toolOutputReferences: { enabled: true },
      });

      mockEventDispatch([
        { toolCallId: 'ec1', content: 'done', status: 'success' },
      ]);
      const result = (await node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              {
                id: 'ec1',
                name: 'echo',
                args: { command: 'see {{tool9turn9}}' },
              },
            ],
          }),
        ],
      })) as { messages: ToolMessage[] };

      expect(result.messages[0].content).toBe('done');
      expect(getUnresolvedRefs(result.messages[0])).toEqual(['tool9turn9']);
    });

    it('registers the post-hook output when PostToolUse replaces it', async () => {
      const hooks = new HookRegistry();
      hooks.register('PostToolUse', {
        hooks: [
          async (): Promise<{ updatedOutput: string }> => ({
            updatedOutput: 'hooked-output',
          }),
        ],
      });
      const node = new ToolNode({
        tools: [createSchemaStub('echo')],
        eventDrivenMode: true,
        agentId: 'agent-x',
        toolCallStepIds: new Map([['ec1', 'step_ec1']]),
        toolOutputReferences: { enabled: true },
        hookRegistry: hooks,
      });

      mockEventDispatch([
        { toolCallId: 'ec1', content: 'raw-output', status: 'success' },
      ]);
      const result = (await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: 'ec1', name: 'echo', args: { command: 'run' } },
              ],
            }),
          ],
        },
        { configurable: { run_id: 'run-posthook' } }
      )) as { messages: ToolMessage[] };

      expect(result.messages[0].content).toBe('hooked-output');
      expect(getRefKey(result.messages[0])).toBe('tool0turn0');
      expect(
        node._unsafeGetToolOutputRegistry()!.get('run-posthook', 'tool0turn0')
      ).toBe('hooked-output');
    });

    it('aborts event dispatch when a direct tool throws with handleToolErrors=false', async () => {
      const directBoom = tool(
        async () => {
          throw new Error('direct branch failed');
        },
        {
          name: 'directBoom',
          description: 'direct tool that throws',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;
      const eventStub = tool(async () => 'unused', {
        name: 'eventTool',
        description: 'schema-only stub',
        schema: z.object({ command: z.string() }),
      }) as unknown as StructuredToolInterface;

      let hostCalled = false;
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event === 'on_tool_execute') {
            hostCalled = true;
            (data as t.ToolExecuteBatchRequest).resolve([]);
          }
        });

      const node = new ToolNode({
        tools: [directBoom, eventStub],
        eventDrivenMode: true,
        handleToolErrors: false,
        agentId: 'agent-failfast',
        directToolNames: new Set(['directBoom']),
        toolCallStepIds: new Map([
          ['d1', 'step_d1'],
          ['e1', 'step_e1'],
        ]),
        toolOutputReferences: { enabled: true },
      });

      await expect(
        node.invoke(
          {
            messages: [
              new AIMessage({
                content: '',
                tool_calls: [
                  { id: 'd1', name: 'directBoom', args: { command: 'x' } },
                  { id: 'e1', name: 'eventTool', args: { command: 'y' } },
                ],
              }),
            ],
          },
          { configurable: { run_id: 'failfast-run' } }
        )
      ).rejects.toThrow('direct branch failed');

      expect(hostCalled).toBe(false);
    });

    it('isolates PreToolUse-injected refs from same-turn direct outputs in the mixed path', async () => {
      // PreToolUse hook rewrites the event call's args to include
      // `{{tool0turn0}}`. In the mixed direct+event path that
      // placeholder must NOT resolve to the same-turn direct
      // output that just registered — it should be reported as
      // unresolved (matching cross-batch resolution semantics).
      const directCapturedArgs: string[] = [];
      const directTool = createEchoTool({
        capturedArgs: directCapturedArgs,
        outputs: ['direct-same-turn'],
        name: 'directTool',
      });
      const eventStub = tool(async () => 'unused', {
        name: 'eventTool',
        description: 'schema-only stub',
        schema: z.object({ command: z.string() }),
      }) as unknown as StructuredToolInterface;

      const hooks = new HookRegistry();
      hooks.register('PreToolUse', {
        pattern: 'eventTool',
        hooks: [
          async (): Promise<{ updatedInput: { command: string } }> => ({
            updatedInput: { command: 'see {{tool0turn0}}' },
          }),
        ],
      });

      const hostCapturedArgs: Record<string, unknown>[] = [];
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== 'on_tool_execute') {
            return;
          }
          const batch = data as t.ToolExecuteBatchRequest;
          for (const req of batch.toolCalls) {
            hostCapturedArgs.push(req.args);
          }
          batch.resolve(
            batch.toolCalls.map((req) => ({
              toolCallId: req.id,
              content: 'event-out',
              status: 'success' as const,
            }))
          );
        });

      const node = new ToolNode({
        tools: [directTool, eventStub],
        eventDrivenMode: true,
        agentId: 'agent-snap',
        directToolNames: new Set(['directTool']),
        toolCallStepIds: new Map([
          ['d1', 'step_d1'],
          ['e1', 'step_e1'],
        ]),
        hookRegistry: hooks,
        toolOutputReferences: { enabled: true },
      });

      await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                {
                  id: 'd1',
                  name: 'directTool',
                  args: { command: 'first' },
                },
                {
                  id: 'e1',
                  name: 'eventTool',
                  args: { command: 'orig' },
                },
              ],
            }),
          ],
        },
        { configurable: { run_id: 'snap-run' } }
      );

      // Hook injected `{{tool0turn0}}`. The direct tool registered
      // `tool0turn0` in the same batch, but the snapshot was taken
      // pre-direct so the placeholder must remain unresolved.
      expect(hostCapturedArgs).toHaveLength(1);
      expect(hostCapturedArgs[0]).toEqual({
        command: 'see {{tool0turn0}}',
      });
    });

    it('keeps same-turn refs isolated in the mixed direct+event path', async () => {
      // Build a ToolNode with both a direct tool (via directToolNames)
      // and an event-driven schema stub. Share one registry across
      // both batches so refs only cross batch boundaries.
      const sharedRegistry = new ToolOutputReferenceRegistry();

      const directCapturedArgs: string[] = [];
      const directTool = createEchoTool({
        capturedArgs: directCapturedArgs,
        outputs: ['direct-A-output', 'direct-B-output'],
        name: 'directTool',
      });
      const eventStub = tool(async () => 'unused', {
        name: 'eventTool',
        description: 'schema-only stub',
        schema: z.object({ command: z.string() }),
      }) as unknown as StructuredToolInterface;

      const hostCapturedArgs: Record<string, unknown>[] = [];
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== 'on_tool_execute') {
            return;
          }
          const batch = data as t.ToolExecuteBatchRequest;
          for (const req of batch.toolCalls) {
            hostCapturedArgs.push(req.args);
          }
          batch.resolve(
            batch.toolCalls.map((req) => ({
              toolCallId: req.id,
              content: `event-${(req.args as { command: string }).command}`,
              status: 'success' as const,
            }))
          );
        });

      const node = new ToolNode({
        tools: [directTool, eventStub],
        eventDrivenMode: true,
        agentId: 'agent-mixed',
        directToolNames: new Set(['directTool']),
        toolCallStepIds: new Map([
          ['d1', 'step_d1'],
          ['e1', 'step_e1'],
          ['d2', 'step_d2'],
          ['e2', 'step_e2'],
        ]),
        toolOutputRegistry: sharedRegistry,
      });

      // Batch 1: mixed direct (index 0) + event (index 1). The event
      // call attempts `{{tool0turn0}}` — which points at the direct
      // call running *in the same batch*. Correct behavior: the
      // placeholder stays unresolved (cross-batch only), and the
      // event args received by the host carry the literal template
      // string. The unresolved-refs hint is stamped into the resulting
      // ToolMessage's `additional_kwargs._unresolvedRefs` so the lazy
      // annotation transform surfaces it to the LLM at request time.
      await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                {
                  id: 'd1',
                  name: 'directTool',
                  args: { command: 'first' },
                },
                {
                  id: 'e1',
                  name: 'eventTool',
                  args: { command: 'echo {{tool0turn0}}' },
                },
              ],
            }),
          ],
        },
        { configurable: { run_id: 'mixed-run' } }
      );

      expect(hostCapturedArgs).toHaveLength(1);
      expect(hostCapturedArgs[0]).toEqual({
        command: 'echo {{tool0turn0}}',
      });

      // Batch 2: ref across the boundary now resolves — direct's
      // registered output from batch 1 (tool0turn0) is available.
      await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                {
                  id: 'd2',
                  name: 'directTool',
                  args: { command: 'second' },
                },
                {
                  id: 'e2',
                  name: 'eventTool',
                  args: { command: 'echo {{tool0turn0}}' },
                },
              ],
            }),
          ],
        },
        { configurable: { run_id: 'mixed-run' } }
      );

      expect(hostCapturedArgs[1]).toEqual({
        command: 'echo direct-A-output',
      });
    });

    it('re-resolves placeholders when PreToolUse rewrites args', async () => {
      const hooks = new HookRegistry();
      hooks.register('PreToolUse', {
        hooks: [
          async (): Promise<{ updatedInput: { command: string } }> => ({
            updatedInput: { command: 'rewritten {{tool0turn0}}' },
          }),
        ],
      });
      const node = new ToolNode({
        tools: [createSchemaStub('echo')],
        eventDrivenMode: true,
        agentId: 'agent-x',
        toolCallStepIds: new Map([
          ['ec1', 'step_ec1'],
          ['ec2', 'step_ec2'],
        ]),
        toolOutputReferences: { enabled: true },
        hookRegistry: hooks,
      });

      mockEventDispatch([
        { toolCallId: 'ec1', content: 'STORED', status: 'success' },
      ]);
      await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: 'ec1', name: 'echo', args: { command: 'first' } },
              ],
            }),
          ],
        },
        { configurable: { run_id: 'run-hookresolve' } }
      );

      jest.restoreAllMocks();
      const capturedRequests: t.ToolCallRequest[] = [];
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== 'on_tool_execute') {
            return;
          }
          const batch = data as t.ToolExecuteBatchRequest;
          for (const req of batch.toolCalls) {
            capturedRequests.push(req);
          }
          batch.resolve([
            { toolCallId: 'ec2', content: 'done', status: 'success' },
          ]);
        });

      await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                {
                  id: 'ec2',
                  name: 'echo',
                  args: { command: 'input-without-placeholder' },
                },
              ],
            }),
          ],
        },
        { configurable: { run_id: 'run-hookresolve' } }
      );

      expect(capturedRequests).toHaveLength(1);
      expect(capturedRequests[0].args).toEqual({
        command: 'rewritten STORED',
      });
    });
  });

  describe('PostToolUse updatedOutput updates the registry (Codex P2 #17)', () => {
    it('subsequent {{tool0turn0}} substitutions deliver the post-hook content, not the pre-hook content', async () => {
      const capturedArgs: string[] = [];
      const echoT = createEchoTool({
        capturedArgs,
        outputs: ['original-output', 'second-call-result'],
      });

      const registry = new HookRegistry();
      registry.register('PostToolUse', {
        hooks: [
          // Replace the tool's content. Pre-fix the registry kept the
          // pre-hook string ("original-output"), so a later
          // {{tool0turn0}} substitution would deliver stale bytes.
          async (): Promise<PostToolUseHookOutput> => ({
            updatedOutput: 'redacted-by-hook',
          }),
        ],
      });

      const node = new ToolNode({
        tools: [echoT],
        toolOutputReferences: { enabled: true },
        hookRegistry: registry,
      });

      const [first] = await invokeBatch(
        node,
        [{ id: 'c1', name: 'echo', command: 'first' }],
        'run-posthook-ref'
      );
      // Sanity: the model sees the replaced content.
      expect(first.content).toBe('redacted-by-hook');

      // Second call references the first via {{tool0turn0}}. The
      // tool's `command` arg should resolve to the post-hook content.
      await invokeBatch(
        node,
        [{ id: 'c2', name: 'echo', command: 'value={{tool0turn0}}' }],
        'run-posthook-ref'
      );
      expect(capturedArgs).toEqual(['first', 'value=redacted-by-hook']);
      // Pre-fix: the second call would have seen 'value=original-output'
      // because the registry was never updated after the post-hook.
      expect(capturedArgs[1]).not.toContain('original-output');
    });
  });

  describe('direct-batch snapshot isolation (Codex P1 #18)', () => {
    it('does not let a slow PreToolUse hook on one call leak a sibling output into another call args', async () => {
      // Two direct calls in a single batch:
      //   c0: has a slow PreToolUse hook (await) + args containing
      //       `{{tool1turn0}}` (a same-turn placeholder).
      //   c1: no hook, returns 'sibling-output' instantly.
      //
      // Same-turn refs are intentionally isolated (the snapshot is
      // taken pre-batch). Pre-fix, runTool's late re-resolve against
      // the *live* registry meant c0 (waiting on its hook) saw c1's
      // already-registered output and substituted it into its args
      // — order-dependent leakage. With the snapshot, c0 sees the
      // placeholder unresolved.
      const capturedArgs: string[] = [];
      const echoT = createEchoTool({
        capturedArgs,
        outputs: ['c0-output', 'sibling-output'],
        name: 'echo',
      });

      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          // Slow hook gates ONLY c0; c1 has no hook to wait on. The
          // delay gives c1 time to finish and register its output
          // before c0's `runTool` runs.
          async (input): Promise<PreToolUseHookOutput> => {
            const cmd = (input.toolInput as { command?: string }).command ?? '';
            if (cmd.includes('{{tool1turn0}}')) {
              await new Promise<void>((resolve) => setTimeout(resolve, 50));
            }
            return { decision: 'allow' };
          },
        ],
      });

      const node = new ToolNode({
        tools: [echoT],
        toolOutputReferences: { enabled: true },
        hookRegistry: registry,
      });

      await invokeBatch(
        node,
        [
          { id: 'c0', name: 'echo', command: 'leak={{tool1turn0}}' },
          { id: 'c1', name: 'echo', command: 'instant' },
        ],
        'run-snapshot-iso'
      );

      // Pre-fix: capturedArgs[0] would have been 'leak=sibling-output'
      // because c1 won the race and c0's late re-resolve picked it up.
      // With the snapshot fix: same-turn isolation holds — the
      // placeholder stays unresolved in c0's args (and an
      // `[unresolved refs: …]` marker shows up downstream).
      const c0Index = capturedArgs.findIndex((a) => a.startsWith('leak='));
      expect(c0Index).toBeGreaterThanOrEqual(0);
      expect(capturedArgs[c0Index]).not.toContain('sibling-output');
    });
  });

  describe('labels persisted on the completed tool_call', () => {
    const echo = (): StructuredToolInterface =>
      tool(async (input) => (input as { command: string }).command, {
        name: 'echo',
        description: 'Echo test tool',
        schema: z.object({ command: z.string() }),
      }) as unknown as StructuredToolInterface;

    const capture = (): Array<{ result: { tool_call: t.ProcessedToolCall } }> => {
      const completions: Array<{ result: { tool_call: t.ProcessedToolCall } }> =
        [];
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data): Promise<void> => {
          if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
            completions.push(data as (typeof completions)[number]);
          } else if (event === GraphEvents.ON_TOOL_EXECUTE) {
            const batch = data as t.ToolExecuteBatchRequest;
            const results = batch.toolCalls.map((call) => ({
              toolCallId: call.id,
              status: 'success' as const,
              content: 'out',
            }));
            results.forEach((result) => batch.onResult?.(result));
            await new Promise((resolve) => setTimeout(resolve, 0));
            batch.resolve(results);
          }
        });
      return completions;
    };

    const run = async (
      node: ToolNode,
      id: string,
      command: string,
      runId = 'run-a'
    ): Promise<ToolMessage> => {
      const out = (await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [{ id, name: 'echo', args: { command } }],
            }),
          ],
        },
        { configurable: { run_id: runId } }
      )) as { messages: ToolMessage[] };
      return out.messages[0];
    };

    it('carries the key and unresolved refs of each result', async () => {
      const completions = capture();
      const node = new ToolNode({
        tools: [echo()],
        toolCallStepIds: new Map([
          ['c1', 's1'],
          ['c2', 's2'],
        ]),
        toolOutputReferences: { enabled: true },
      });

      await run(node, 'c1', 'first');
      await run(node, 'c2', 'use {{tool0turn9}}');

      expect(completions[0].result.tool_call).toMatchObject({
        ref: 'tool0turn0',
      });
      expect(completions[0].result.tool_call).not.toHaveProperty(
        'unresolved_refs'
      );
      expect(completions[1].result.tool_call).toMatchObject({
        ref: 'tool0turn1',
        unresolved_refs: ['tool0turn9'],
      });
    });

    it('carries the key on event-driven completions', async () => {
      const completions = capture();
      const node = new ToolNode({
        tools: [echo()],
        eventDrivenMode: true,
        toolCallStepIds: new Map([['c1', 's1']]),
        toolOutputReferences: { enabled: true },
      });

      await run(node, 'c1', 'a');

      expect(completions).toHaveLength(1);
      expect(completions[0].result.tool_call.ref).toBe('tool0turn0');
    });

    it('adds no ref fields when the feature is disabled', async () => {
      const completions = capture();
      const node = new ToolNode({
        tools: [echo()],
        toolCallStepIds: new Map([['c1', 's1']]),
      });

      await run(node, 'c1', 'first');

      expect(completions[0].result.tool_call).not.toHaveProperty('ref');
    });

    it('reports a placeholder naming an earlier run key as unresolved', async () => {
      capture();
      const node = new ToolNode({
        tools: [echo()],
        toolCallStepIds: new Map([['c1', 's1']]),
        toolOutputReferences: { enabled: true },
      });
      const previous = new ToolMessage({
        content: 'x',
        tool_call_id: 'old',
        additional_kwargs: { _refKey: 'tool0turn0' },
      });
      node
        ._unsafeGetToolOutputRegistry()!
        .seedTurn('run-b', nextReferenceTurn([previous]));

      const msg = await run(node, 'c1', 'echo {{tool0turn0}}', 'run-b');

      expect(getRefKey(msg)).toBe('tool0turn1');
      expect(getUnresolvedRefs(msg)).toEqual(['tool0turn0']);
      expect(msg.content).toBe('echo {{tool0turn0}}');
      const [wire] = annotateMessagesForLLM(
        [msg],
        node._unsafeGetToolOutputRegistry()
      );
      expect(wire.content).toContain('[unresolved refs: tool0turn0]');
    });

    it('seeds a run from the keys already in its input history', async () => {
      const created = await Run.create<t.IState>({
        runId: 'seed-run',
        graphConfig: {
          type: 'standard',
          agents: [
            {
              agentId: 'a',
              provider: Providers.OPENAI,
              clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
              instructions: 'x',
              maxContextTokens: 8000,
            },
          ],
        },
        toolOutputReferences: { enabled: true },
        returnContent: true,
        skipCleanup: true,
      });
      created.Graph?.overrideTestModel(['ok'], 10);

      await created.processStream(
        {
          messages: [
            new HumanMessage('q'),
            new AIMessage({
              content: '',
              tool_calls: [{ id: 't1', name: 'echo', args: {} }],
            }),
            new ToolMessage({
              content: 'x',
              tool_call_id: 't1',
              additional_kwargs: { _refKey: 'tool2turn6' },
            }),
            new HumanMessage('next'),
          ],
        },
        { configurable: { thread_id: 'seed-thread' }, version: 'v2' }
      );

      const registry = created.Graph?.getOrCreateToolOutputRegistry();
      expect(registry?.nextTurn('seed-run')).toBe(7);
    });
  });
});
