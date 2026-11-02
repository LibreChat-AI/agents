import { createHash } from 'crypto';
import type { OwnedJobManager, OwnedJobOutput } from '@/verification/ownedJobs';
import type * as t from '@/types';
import { executeLocalBash } from './LocalExecutionEngine';
import { createAwaitedTool } from '../awaitedCancellation';

/** The manager must be registered on the owning Run and drained in the host's finally block. */
export function createLocalBackgroundTools(
  manager: OwnedJobManager,
  config: t.LocalExecutionConfig = {}
): { tools: t.GenericTool[]; toolRegistry: t.LCToolRegistry } {
  const executionConfig = {
    ...config,
    toolResultMaxChars: Math.min(manager.maxResultChars,
      config.toolResultMaxChars != null && config.toolResultMaxChars > 0
        ? config.toolResultMaxChars : manager.maxResultChars),
  };
  const startSchema = {
    type: 'object' as const,
    properties: {
      id: { type: 'string' as const, pattern: '^[a-zA-Z0-9_.:-]{1,80}$', description: 'Unique operation identity. While this run accepts work, reusing the same id and command returns its existing result without executing again.' },
      command: { type: 'string' as const, description: 'Independent Bash command in the configured workspace. Completion is delivered automatically; do useful foreground work rather than polling.' },
    },
    required: ['id', 'command'],
    additionalProperties: false,
  };
  const cancelSchema = {
    type: 'object' as const,
    properties: { id: { type: 'string' as const } },
    required: ['id'],
    additionalProperties: false,
  };
  const tools = [
    createAwaitedTool({
      name: 'start_owned_command',
      description: 'Start an independent, bounded local command. The owning run delivers its result at a tool boundary or waits at completion. Do not background commands whose output or writes the next operation depends on.',
      schema: startSchema,
      func: async (raw, _callbacks, invocation): Promise<string> => {
        invocation?.signal?.throwIfAborted();
        const { id, command } = raw as { id: string; command: string };
        return JSON.stringify(manager.start({
          id,
          fingerprint: createHash('sha256').update(command).digest('hex'),
          execute: async (signal): Promise<OwnedJobOutput> => {
            const result = await executeLocalBash(command, executionConfig, signal);
            const output = result.stdout + result.stderr;
            return { output, failed: result.exitCode !== 0 || result.timedOut,
              exitCode: result.exitCode, timedOut: result.timedOut, signal: result.signal,
              overflowKilled: result.overflowKilled, fullOutputPath: result.fullOutputPath,
              outputOnHost: result.outputOnHost, outputTruncated: result.outputTruncated,
              outputRecoveryError: result.outputRecoveryError,
            };
          },
        }));
      },
    }),
    createAwaitedTool({
      name: 'cancel_owned_command',
      description: 'Request cancellation of an owned command. Cancelling is not terminal; final delivery occurs only after execution has actually stopped. Completed side effects remain.',
      schema: cancelSchema,
      func: async (raw): Promise<string> => JSON.stringify(manager.cancel((raw as { id: string }).id)),
    }),
  ];
  const schemas = [startSchema, cancelSchema];
  return {
    tools,
    toolRegistry: new Map(tools.map((bound, index) => [bound.name, {
      name: bound.name, description: bound.description, parameters: schemas[index], allowed_callers: ['direct', 'code_execution'],
    }])),
  };
}
