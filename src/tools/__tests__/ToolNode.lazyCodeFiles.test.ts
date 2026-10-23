import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage } from '@langchain/core/messages';
import { CallbackManager } from '@langchain/core/callbacks/manager';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type * as t from '@/types';
import { Constants, GraphEvents } from '@/common';
import * as events from '@/utils/events';
import { ToolNode } from '../ToolNode';

const inputs: t.CodeEnvFile[] = [
  {
    id: 'order',
    name: 'order.pdf',
    resource_id: 'user',
    storage_session_id: 'user-storage',
    kind: 'user',
  },
  {
    id: 'discounts',
    name: 'discounts.csv',
    resource_id: 'worker',
    storage_session_id: 'agent-storage',
    kind: 'agent',
  },
];

const firstResults = [
  {
    label: 'HTTP-success execution with stderr and no output files',
    result: {
      status: 'success',
      artifact: { session_id: 'first-exec', files: [] },
    },
  },
  {
    label: 'execution without an artifact files field',
    result: { status: 'success', artifact: { session_id: 'first-exec' } },
  },
  {
    label: 'execution without an artifact',
    result: { status: 'success' },
  },
  {
    label: 'tool error without an artifact',
    result: { status: 'error', errorMessage: 'Code API unavailable' },
  },
  {
    label: 'tool error carrying unusable output files',
    result: {
      status: 'error',
      errorMessage: 'Code execution failed',
      artifact: {
        session_id: 'failed-exec',
        files: [{ id: 'failed-output', name: 'failed.csv' }],
      },
    },
  },
] satisfies Array<{
  label: string;
  result: Pick<t.ToolExecuteResult, 'status' | 'artifact' | 'errorMessage'>;
}>;

describe.each([Constants.EXECUTE_CODE, Constants.BASH_TOOL])(
  '%s lazy input retention',
  (toolName) => {
    afterEach(() => {
      jest.restoreAllMocks();
    });

    it.each(firstResults)(
      'preserves inputs after $label',
      async ({ result }) => {
        const sessions: t.ToolSessionMap = new Map([
          [
            'other-agent',
            {
              session_id: 'foreign-exec',
              files: [{ id: 'foreign', name: 'foreign.txt' }],
              lastUpdated: 1,
            },
          ],
        ]);
        const requests: t.ToolCallRequest[] = [];
        jest
          .spyOn(events, 'safeDispatchCustomEvent')
          .mockImplementation(async (event, data) => {
            if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
            const batch = data as t.ToolExecuteBatchRequest;
            const request = batch.toolCalls[0];
            requests.push(structuredClone(request));
            if (requests.length === 1) {
              request.codeSessionContext = {
                session_id: 'input-context',
                files: structuredClone(inputs),
              };
            }
            batch.resolve([
              {
                toolCallId: request.id,
                content: 'stderr: missing optional Python package',
                ...(requests.length === 1
                  ? result
                  : { status: 'success' as const }),
              },
            ]);
          });
        const node = new ToolNode({
          tools: [
            tool(async () => 'unused', {
              name: toolName,
              description: 'Code tool',
              schema: z.object({}),
            }),
          ],
          sessions,
          codeSessionKey: 'child-agent',
          eventDrivenMode: true,
        });
        for (const id of ['first', 'retry']) {
          await node.invoke({
            messages: [
              new AIMessage({
                content: '',
                tool_calls: [{ id, name: toolName, args: {} }],
              }),
            ],
          });
        }

        expect(requests[0].codeSessionContext).toBeUndefined();
        expect(requests[1].codeSessionContext?.files).toEqual(inputs);
        expect(sessions.get('child-agent')?.files).toEqual(inputs);
        expect(sessions.get('other-agent')).toEqual({
          session_id: 'foreign-exec',
          files: [{ id: 'foreign', name: 'foreign.txt' }],
          lastUpdated: 1,
        });
        if (result.status === 'error') {
          expect(sessions.get('child-agent')?.session_id).toBe('input-context');
        }
      }
    );

    it('keeps successful replacements when a later batch result carries older input refs', async () => {
      const sessions: t.ToolSessionMap = new Map();
      const updated: t.FileRef = {
        id: 'updated-discounts',
        name: 'discounts.csv',
        storage_session_id: 'output-storage',
      };
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
          const batch = data as t.ToolExecuteBatchRequest;
          for (const request of batch.toolCalls) {
            request.codeSessionContext = {
              session_id: 'input-context',
              files: structuredClone(inputs),
            };
          }
          batch.resolve(
            batch.toolCalls.map((request, index) => ({
              toolCallId: request.id,
              status: 'success',
              content: '',
              artifact: {
                session_id: `exec-${index}`,
                files: index === 0 ? [updated] : [],
              },
            }))
          );
        });
      const node = new ToolNode({
        tools: [
          tool(async () => 'unused', {
            name: toolName,
            description: 'Code tool',
            schema: z.object({}),
          }),
        ],
        sessions,
        codeSessionKey: 'child-agent',
        eventDrivenMode: true,
      });

      await node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: ['write', 'read'].map((id) => ({
              id,
              name: toolName,
              args: {},
            })),
          }),
        ],
      });

      expect(sessions.get('child-agent')?.files).toEqual([inputs[0], updated]);
      expect(sessions.get('child-agent')?.session_id).toBe('exec-1');
    });

    it('retains files provisioned into the actual eager request before its result settles', async () => {
      const sessions: t.ToolSessionMap = new Map();
      const request: t.ToolCallRequest = {
        id: 'first',
        name: toolName,
        args: {},
      };
      const eagerExecutions = new Map<string, t.EagerEventToolExecution>([
        [
          request.id,
          {
            toolCallId: request.id,
            toolName,
            args: {},
            request,
            promise: Promise.resolve().then(() => {
              request.codeSessionContext = {
                session_id: 'input-context',
                files: structuredClone(inputs),
              };
              return {
                results: [
                  {
                    toolCallId: request.id,
                    status: 'success',
                    content: '',
                    artifact: { session_id: 'first-exec', files: [] },
                  },
                ],
              };
            }),
          },
        ],
      ]);
      const captured: t.ToolCallRequest[] = [];
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
          const batch = data as t.ToolExecuteBatchRequest;
          captured.push(...batch.toolCalls);
          batch.resolve(
            batch.toolCalls.map((tc) => ({
              toolCallId: tc.id,
              status: 'success',
              content: '',
            }))
          );
        });
      const node = new ToolNode({
        tools: [
          tool(async () => 'unused', {
            name: toolName,
            description: 'Code tool',
            schema: z.object({}),
          }),
        ],
        sessions,
        codeSessionKey: 'child-agent',
        eventDrivenMode: true,
        eagerEventToolExecution: { enabled: true },
        eagerEventToolExecutions: eagerExecutions,
      });
      for (const id of ['first', 'retry']) {
        await node.invoke({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [{ id, name: toolName, args: {} }],
            }),
          ],
        });
      }

      expect(captured).toHaveLength(1);
      expect(captured[0].id).toBe('retry');
      expect(captured[0].codeSessionContext?.files).toEqual(inputs);
    });

    it('uses the eager request creation baseline for retained inputs', async () => {
      const stale = inputs[0];
      const refreshed = {
        ...stale,
        id: 'order-refreshed',
        storage_session_id: 'refreshed-storage',
      };
      const sessions: t.ToolSessionMap = new Map([
        [
          'child-agent',
          { session_id: 'new-exec', files: [refreshed], lastUpdated: 2 },
        ],
      ]);
      const request: t.ToolCallRequest = {
        id: 'eager',
        name: toolName,
        args: {},
        codeSessionContext: {
          session_id: 'old-exec',
          files: [stale],
        },
      };
      const eagerExecutions = new Map<string, t.EagerEventToolExecution>([
        [
          request.id,
          {
            toolCallId: request.id,
            toolName,
            args: {},
            request,
            codeSessionBaselineByName: new Map([
              [stale.name, `${stale.storage_session_id}\0${stale.id}`],
            ]),
            promise: Promise.resolve({
              results: [
                {
                  toolCallId: request.id,
                  status: 'success',
                  content: '',
                },
              ],
            }),
          },
        ],
      ]);
      const node = new ToolNode({
        tools: [
          tool(async () => 'unused', {
            name: toolName,
            description: 'Code tool',
            schema: z.object({}),
          }),
        ],
        sessions,
        codeSessionKey: 'child-agent',
        eventDrivenMode: true,
        eagerEventToolExecution: { enabled: true },
        eagerEventToolExecutions: eagerExecutions,
      });

      await node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [{ id: request.id, name: toolName, args: {} }],
          }),
        ],
      });

      expect(sessions.get('child-agent')?.files).toEqual([refreshed]);
    });

    it('retains refreshed eager inputs when the final tool identity differs', async () => {
      const stale = inputs[0];
      const refreshed = {
        ...stale,
        id: 'order-refreshed',
        storage_session_id: 'refreshed-storage',
      };
      const finalToolName = 'weather';
      const sessions: t.ToolSessionMap = new Map([
        [
          'child-agent',
          { session_id: 'old-exec', files: [stale], lastUpdated: 1 },
        ],
      ]);
      const request: t.ToolCallRequest = {
        id: 'eager',
        name: toolName,
        args: {},
        codeSessionContext: {
          session_id: 'old-exec',
          files: [stale],
        },
      };
      let finishEager = (_outcome: t.EagerEventToolExecutionOutcome): void =>
        undefined;
      const eagerPromise = new Promise<t.EagerEventToolExecutionOutcome>(
        (resolve) => {
          finishEager = resolve;
        }
      );
      const eagerExecutions = new Map<string, t.EagerEventToolExecution>([
        [
          request.id,
          {
            toolCallId: request.id,
            toolName,
            args: {},
            request,
            codeSessionBaselineByName: new Map([
              [stale.name, `${stale.storage_session_id}\0${stale.id}`],
            ]),
            promise: eagerPromise,
          },
        ],
      ]);
      const node = new ToolNode({
        tools: [
          tool(async () => 'unused', {
            name: toolName,
            description: 'Code tool',
            schema: z.object({}),
          }),
          tool(async () => 'unused', {
            name: finalToolName,
            description: 'Final code tool',
            schema: z.object({}),
          }),
        ],
        sessions,
        codeSessionKey: 'child-agent',
        eventDrivenMode: true,
        eagerEventToolExecution: { enabled: true },
        eagerEventToolExecutions: eagerExecutions,
      });

      const result = node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [{ id: request.id, name: finalToolName, args: {} }],
          }),
        ],
      });
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
      request.codeSessionContext = {
        session_id: 'input-context',
        files: [refreshed],
      };
      finishEager({ results: [] });
      await result;

      expect(sessions.get('child-agent')?.files).toEqual([refreshed]);
      expect(sessions.get('child-agent')?.session_id).toBe('input-context');
    });

    it('replaces a stale same-name session file with a refreshed input', async () => {
      const stale = inputs[0];
      const refreshed = {
        ...stale,
        id: 'order-refreshed',
        resource_id: 'user-refreshed',
        storage_session_id: 'refreshed-storage',
      };
      const sessions: t.ToolSessionMap = new Map([
        [
          'child-agent',
          {
            session_id: 'old-exec',
            files: [stale],
            lastUpdated: 1,
          },
        ],
      ]);
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
          const batch = data as t.ToolExecuteBatchRequest;
          batch.toolCalls[0].codeSessionContext = {
            session_id: 'input-context',
            files: [refreshed],
          };
          batch.reject(new Error('transport failed'));
        });
      const node = new ToolNode({
        tools: [
          tool(async () => 'unused', {
            name: toolName,
            description: 'Code tool',
            schema: z.object({}),
          }),
        ],
        sessions,
        codeSessionKey: 'child-agent',
        eventDrivenMode: true,
      });

      await expect(
        node.invoke({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [{ id: 'refresh', name: toolName, args: {} }],
            }),
          ],
        })
      ).rejects.toThrow('transport failed');
      expect(sessions.get('child-agent')?.files).toEqual([refreshed]);
      expect(sessions.get('child-agent')?.session_id).toBe('input-context');
    });

    it('retains an execution session refresh without file changes', async () => {
      const sessions: t.ToolSessionMap = new Map([
        [
          'child-agent',
          { session_id: 'old-exec', files: inputs, lastUpdated: 1 },
        ],
      ]);
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
          const batch = data as t.ToolExecuteBatchRequest;
          batch.toolCalls[0].codeSessionContext = {
            session_id: 'refreshed-exec',
            files: structuredClone(inputs),
          };
          batch.resolve([
            {
              toolCallId: batch.toolCalls[0].id,
              status: 'success',
              content: '',
            },
          ]);
        });
      const node = new ToolNode({
        tools: [
          tool(async () => 'unused', {
            name: toolName,
            description: 'Code tool',
            schema: z.object({}),
          }),
        ],
        sessions,
        codeSessionKey: 'child-agent',
        eventDrivenMode: true,
      });

      await node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [{ id: 'refresh-session', name: toolName, args: {} }],
          }),
        ],
      });

      expect(sessions.get('child-agent')).toMatchObject({
        session_id: 'refreshed-exec',
        files: inputs,
      });
    });

    it('prefers an explicit session refresh over a sibling file session', async () => {
      const refreshed = {
        ...inputs[0],
        id: 'order-refreshed',
        storage_session_id: 'refreshed-storage',
      };
      const sessions: t.ToolSessionMap = new Map([
        [
          'child-agent',
          { session_id: 'old-exec', files: inputs, lastUpdated: 1 },
        ],
      ]);
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
          const batch = data as t.ToolExecuteBatchRequest;
          batch.toolCalls[0].codeSessionContext = {
            session_id: 'refreshed-exec',
            files: structuredClone(inputs),
          };
          batch.toolCalls[1].codeSessionContext = {
            session_id: 'old-exec',
            files: [refreshed, inputs[1]],
          };
          batch.reject(new Error('transport failed'));
        });
      const node = new ToolNode({
        tools: [
          tool(async () => 'unused', {
            name: toolName,
            description: 'Code tool',
            schema: z.object({}),
          }),
        ],
        sessions,
        codeSessionKey: 'child-agent',
        eventDrivenMode: true,
      });

      await expect(
        node.invoke({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: ['session', 'file'].map((id) => ({
                id,
                name: toolName,
                args: {},
              })),
            }),
          ],
        })
      ).rejects.toThrow('transport failed');
      expect(sessions.get('child-agent')).toMatchObject({
        session_id: 'refreshed-exec',
        files: [refreshed, inputs[1]],
      });
    });

    it('ignores a later request carrying the original batch snapshot', async () => {
      const stale = inputs[0];
      const refreshed = {
        ...stale,
        id: 'order-refreshed',
        resource_id: 'user-refreshed',
        storage_session_id: 'refreshed-storage',
      };
      const sessions: t.ToolSessionMap = new Map([
        [
          'child-agent',
          {
            session_id: 'old-exec',
            files: structuredClone(inputs),
            lastUpdated: 1,
          },
        ],
      ]);
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
          const batch = data as t.ToolExecuteBatchRequest;
          batch.toolCalls[0].codeSessionContext = {
            session_id: 'input-context',
            files: [refreshed, inputs[1]],
          };
          batch.reject(new Error('transport failed'));
        });
      const node = new ToolNode({
        tools: [
          tool(async () => 'unused', {
            name: toolName,
            description: 'Code tool',
            schema: z.object({}),
          }),
        ],
        sessions,
        codeSessionKey: 'child-agent',
        eventDrivenMode: true,
      });

      await expect(
        node.invoke({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: ['refresh', 'stale'].map((id) => ({
                id,
                name: toolName,
                args: {},
              })),
            }),
          ],
        })
      ).rejects.toThrow('transport failed');
      expect(sessions.get('child-agent')?.files).toEqual([
        refreshed,
        inputs[1],
      ]);
    });

    it('uses the pre-dispatch baseline across concurrent batches', async () => {
      const stale = inputs[0];
      const refreshed = {
        ...stale,
        id: 'order-refreshed',
        storage_session_id: 'refreshed-storage',
      };
      const sessions: t.ToolSessionMap = new Map([
        [
          'child-agent',
          { session_id: 'old-exec', files: [stale], lastUpdated: 1 },
        ],
      ]);
      let finishStale = (): void => undefined;
      const staleGate = new Promise<void>((resolve) => {
        finishStale = resolve;
      });
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
          const batch = data as t.ToolExecuteBatchRequest;
          const request = batch.toolCalls[0];
          if (request.id === 'stale') {
            await staleGate;
          } else {
            request.codeSessionContext = {
              session_id: 'input-context',
              files: [refreshed],
            };
          }
          batch.resolve([
            { toolCallId: request.id, status: 'success', content: '' },
          ]);
        });
      const node = new ToolNode({
        tools: [
          tool(async () => 'unused', {
            name: toolName,
            description: 'Code tool',
            schema: z.object({}),
          }),
        ],
        sessions,
        codeSessionKey: 'child-agent',
        eventDrivenMode: true,
      });
      const invoke = (id: string) =>
        node.invoke({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [{ id, name: toolName, args: {} }],
            }),
          ],
        });

      const staleBatch = invoke('stale');
      await invoke('refreshed');
      finishStale();
      await staleBatch;

      expect(sessions.get('child-agent')?.files).toEqual([refreshed]);
    });

    it('retains refreshed inputs from read_file requests', async () => {
      const stale = inputs[0];
      const refreshed = {
        ...stale,
        id: 'order-refreshed',
        storage_session_id: 'refreshed-storage',
      };
      const sessions: t.ToolSessionMap = new Map([
        [
          'child-agent',
          { session_id: 'old-exec', files: [stale], lastUpdated: 1 },
        ],
      ]);
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
          const batch = data as t.ToolExecuteBatchRequest;
          const request = batch.toolCalls[0];
          if (request.name === Constants.READ_FILE) {
            request.codeSessionContext = {
              session_id: 'input-context',
              files: [refreshed],
            };
          } else {
            expect(request.codeSessionContext?.files).toEqual([refreshed]);
          }
          batch.resolve([
            { toolCallId: request.id, status: 'success', content: '' },
          ]);
        });
      const node = new ToolNode({
        tools: [
          tool(async () => 'unused', {
            name: toolName,
            description: 'Code tool',
            schema: z.object({}),
          }),
          tool(async () => 'unused', {
            name: Constants.READ_FILE,
            description: 'Read file',
            schema: z.object({}),
          }),
        ],
        sessions,
        codeSessionKey: 'child-agent',
        eventDrivenMode: true,
      });
      for (const [id, name] of [
        ['read', Constants.READ_FILE],
        ['execute', toolName],
      ]) {
        await node.invoke({
          messages: [
            new AIMessage({ content: '', tool_calls: [{ id, name, args: {} }] }),
          ],
        });
      }
    });

    it('retains lazily provisioned inputs when an event batch rejects', async () => {
      const sessions: t.ToolSessionMap = new Map();
      let attempt = 0;
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
          const batch = data as t.ToolExecuteBatchRequest;
          const request = batch.toolCalls[0];
          if (attempt++ === 0) {
            request.codeSessionContext = {
              session_id: 'input-context',
              files: structuredClone(inputs),
            };
            batch.reject(new Error('transport failed'));
            return;
          }
          expect(request.codeSessionContext?.files).toEqual(inputs);
          batch.resolve([
            { toolCallId: request.id, status: 'success', content: '' },
          ]);
        });
      const node = new ToolNode({
        tools: [
          tool(async () => 'unused', {
            name: toolName,
            description: 'Code tool',
            schema: z.object({}),
          }),
        ],
        sessions,
        codeSessionKey: 'child-agent',
        eventDrivenMode: true,
      });

      await expect(
        node.invoke({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [{ id: 'first', name: toolName, args: {} }],
            }),
          ],
        })
      ).rejects.toThrow('transport failed');
      await node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [{ id: 'retry', name: toolName, args: {} }],
          }),
        ],
      });
    });

    it('retains lazily provisioned inputs when eager execution rejects', async () => {
      const sessions: t.ToolSessionMap = new Map();
      const request: t.ToolCallRequest = {
        id: 'first',
        name: toolName,
        args: {},
      };
      const eagerExecutions = new Map<string, t.EagerEventToolExecution>([
        [
          request.id,
          {
            toolCallId: request.id,
            toolName,
            args: {},
            request,
            promise: Promise.resolve().then(() => {
              request.codeSessionContext = {
                session_id: 'input-context',
                files: structuredClone(inputs),
              };
              return { error: new Error('eager failed') };
            }),
          },
        ],
      ]);
      const node = new ToolNode({
        tools: [
          tool(async () => 'unused', {
            name: toolName,
            description: 'Code tool',
            schema: z.object({}),
          }),
        ],
        sessions,
        codeSessionKey: 'child-agent',
        eventDrivenMode: true,
        eagerEventToolExecution: { enabled: true },
        eagerEventToolExecutions: eagerExecutions,
      });

      await expect(
        node.invoke({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [{ id: 'first', name: toolName, args: {} }],
            }),
          ],
        })
      ).rejects.toThrow('eager failed');
      expect(sessions.get('child-agent')?.files).toEqual(inputs);
    });

    it('waits for peer execution input provisioning before retaining a rejected batch', async () => {
      const sessions: t.ToolSessionMap = new Map();
      const eagerRequest: t.ToolCallRequest = {
        id: 'eager',
        name: toolName,
        args: {},
      };
      const eagerExecutions = new Map<string, t.EagerEventToolExecution>([
        [
          eagerRequest.id,
          {
            toolCallId: eagerRequest.id,
            toolName,
            args: {},
            request: eagerRequest,
            promise: Promise.resolve({ error: new Error('eager failed') }),
          },
        ],
      ]);
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
          const batch = data as t.ToolExecuteBatchRequest;
          await Promise.resolve();
          batch.toolCalls[0].codeSessionContext = {
            session_id: 'input-context',
            files: structuredClone(inputs),
          };
          batch.resolve([
            {
              toolCallId: batch.toolCalls[0].id,
              status: 'success',
              content: '',
            },
          ]);
        });
      const node = new ToolNode({
        tools: [
          tool(async () => 'unused', {
            name: toolName,
            description: 'Code tool',
            schema: z.object({}),
          }),
        ],
        sessions,
        codeSessionKey: 'child-agent',
        eventDrivenMode: true,
        eagerEventToolExecution: { enabled: true },
        eagerEventToolExecutions: eagerExecutions,
      });

      await expect(
        node.invoke({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: 'eager', name: toolName, args: {} },
                { id: 'dispatched', name: toolName, args: {} },
              ],
            }),
          ],
        })
      ).rejects.toThrow('eager failed');
      expect(sessions.get('child-agent')?.files).toEqual(inputs);
    });
  }
);

describe('host-authorized lazy code file staging', () => {
  it.each([true, false])(
    'honors RunnableConfig callbacks while staging files only for authorized writers (authorized=%s)',
    async (authorized) => {
      const hostToolName = 'host_file_writer';
      const sessions: t.ToolSessionMap = new Map();
      const generatedFile: t.FileRef = {
        id: 'host-output',
        name: 'generated.csv',
        storage_session_id: 'host-output-storage',
      };
      const dispatchedIds: string[] = [];
      let secondRequestContext: t.ToolCallRequest['codeSessionContext'];
      const callbacks = new CallbackManager('host-file-stage-run');
      callbacks.addHandler(
        BaseCallbackHandler.fromMethods({
          handleCustomEvent(eventName: string, payload: unknown): void {
            if (eventName !== GraphEvents.ON_TOOL_EXECUTE) {
              return;
            }
            const batch = payload as t.ToolExecuteBatchRequest;
            const request = batch.toolCalls[0];
            dispatchedIds.push(request.id);
            if (request.id === 'first') {
              request.codeSessionContext = {
                session_id: 'host-input-session',
                files: [inputs[0]],
              };
            } else {
              secondRequestContext = request.codeSessionContext;
            }
            batch.resolve([
              {
                toolCallId: request.id,
                status: 'success',
                content: 'host execution completed',
                artifact: {
                  session_id: 'host-output-session',
                  files: [generatedFile],
                },
              },
            ]);
          },
        }),
        true,
      );
      const node = new ToolNode({
        tools: [
          tool(async () => 'unused', {
            name: hostToolName,
            description: 'Host file writer',
            schema: z.object({}),
          }),
        ],
        sessions,
        codeSessionKey: 'host-session',
        eventDrivenMode: true,
        codeSessionToolNames: authorized ? [hostToolName] : [],
      });

      for (const id of ['first', 'second']) {
        await node.invoke(
          {
            messages: [
              new AIMessage({
                content: '',
                tool_calls: [{ id, name: hostToolName, args: {} }],
              }),
            ],
          },
          { callbacks },
        );
      }

      expect(dispatchedIds).toEqual(['first', 'second']);
      if (authorized) {
        expect(secondRequestContext?.session_id).toBe('host-output-session');
        expect(secondRequestContext?.files?.map((file) => file.id)).toEqual(
          expect.arrayContaining([inputs[0].id, generatedFile.id]),
        );
        expect(sessions.get('host-session')?.files?.map((file) => file.id)).toEqual(
          expect.arrayContaining([inputs[0].id, generatedFile.id]),
        );
      } else {
        expect(secondRequestContext).toBeUndefined();
        expect(sessions.has('host-session')).toBe(false);
      }
    },
  );
});
