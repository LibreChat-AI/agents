import { z } from 'zod';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type { ToolResultProtection } from '../toolResult';
import type { ToolExecuteBatchRequest } from '@/types';
import { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import { ToolResultProtectionError } from '../toolResult';
import { ToolNode } from '@/tools/ToolNode';
import { HookRegistry } from '@/hooks';
import { GraphEvents } from '@/common';

const missing = '{' + '{tool7turn9}' + '}';
const policy = (inspect: ToolResultProtection['inspect']): ToolResultProtection => ({ version: 1, toolNames: ['lookup'], timeoutMs: 10000, maxAttemptBytes: 65536, maxBufferedBytes: 262144, classify: () => 'prose', inspect });

it.each([false, true].flatMap((event) => [false, true].map((hooks) => [event, hooks] as const)))('gates uncertified selected reference refusals before release (event=%s/hooks=%s)', async (event, hooks) => {
  let executions = 0, starts = 0, dispatches = 0, completions = 0;
  const source = new DynamicStructuredTool({ name: 'lookup', description: 'Allowed', schema: z.object({ value: z.string() }), func: async () => { executions++; return 'Allowed output'; } });
  const registry = new HookRegistry(); if (hooks) registry.register('PreToolUse', { hooks: [async () => ({ decision: 'allow' })] });
  const callback = BaseCallbackHandler.fromMethods({ handleToolStart: (): void => { starts++; }, handleCustomEvent: (name, data): void => {
    if (name === GraphEvents.ON_RUN_STEP_COMPLETED) completions++;
    if (name !== GraphEvents.ON_TOOL_EXECUTE) return;
    dispatches++; const request = data as ToolExecuteBatchRequest; request.resolve([{ toolCallId: 'outer', status: 'success', content: 'Must not run' }]);
  } }); callback.awaitHandlers = true;
  const inspect = jest.fn(() => ({ version: 1 as const, ok: false as const, error: { code: 'blocked' as const } }));
  const node = new ToolNode({ tools: [source], trace: true, eventDrivenMode: event, toolOutputRegistry: new ToolOutputReferenceRegistry(), hookRegistry: registry, toolResultProtection: policy(inspect), toolCallStepIds: new Map([['outer', 'step']]) });
  const result = await node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ name: 'lookup', id: 'outer', args: { value: missing } }] })] }, { callbacks: [callback] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect({ executions, starts, dispatches, completions }).toEqual({ executions: 0, starts: 0, dispatches: 0, completions: 0 });
});

it.each(['absent', 'unselected'] as const)('preserves upstream unresolved-reference refusal when policy is %s', async (mode) => {
  let executions = 0; const source = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({ value: z.string() }), func: async () => { executions++; return 'Must not run'; } });
  const node = new ToolNode({ tools: [source], toolOutputRegistry: new ToolOutputReferenceRegistry(), ...(mode === 'unselected' ? { toolResultProtection: policy(() => { throw new Error('Must not inspect'); }) } : {}) });
  const result = await node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ name: 'other', id: 'outer', args: { value: missing } }] })] }) as { messages: ToolMessage[] };
  expect(executions).toBe(0); expect(result.messages[0].status).toBe('error'); expect(result.messages[0].tool_call_id).toBe('outer'); expect(result.messages[0].content).toContain('nothing was run');
});
