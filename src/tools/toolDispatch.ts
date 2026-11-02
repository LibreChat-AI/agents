import { AIMessage } from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { coerceRecordArgs } from './eagerEventExecution';
import { allowsToolCaller } from './CallerCapabilities';

export const TOOL_DISPATCHER_NAME = 'call_tool';

/** Normalizes `AgentInputs.toolDispatch`; `false` when the host did not opt in. */
export function resolveToolDispatch(
  config: boolean | t.ToolDispatchOptions | undefined
): false | t.ToolDispatchOptions {
  if (config == null || config === false) {
    return false;
  }
  return config === true ? {} : config;
}

/**
 * `arguments` is a JSON string rather than an object: an object property
 * without declared properties is rejected or dropped by some providers'
 * function-declaration formats.
 */
export function createToolDispatcherDefinition(): t.LCTool {
  return {
    name: TOOL_DISPATCHER_NAME,
    description:
      'Calls a tool that was made available during this conversation and is not in your tool list.',
    parameters: {
      type: 'object',
      properties: {
        name: {
          type: 'string',
          description: 'Exact name of the tool to call.',
        },
        arguments: {
          type: 'string',
          description:
            'The tool arguments as a JSON object, following its parameters schema.',
        },
      },
      required: ['name'],
    },
    allowed_callers: ['direct'],
    toolType: 'builtin',
  };
}

/** A tool call as the model sees it in the history. */
export type ModelToolCall = { name: string; args: Record<string, unknown> };

export type DispatchResolution =
  | {
      /** The call to run: the target for a dispatcher call, else the input. */
      call: ToolCall;
      /** Set when `call` runs as a dispatched call: the call the model's history keeps. */
      modelCall?: ModelToolCall;
      /** True when the model called a deferred tool directly and `modelCall` replaces that call. */
      rewritten?: boolean;
    }
  | { error: string };

type DefinitionLookup = (name: string) => t.LCTool | undefined;

function disabledError(name: string, tool: t.LCTool): string | undefined {
  if (tool.disabled == null) {
    return undefined;
  }
  const reason = tool.disabled.reason;
  return `Tool "${name}" is disabled in this conversation${
    reason != null && reason !== '' ? `: ${reason}` : '.'
  }`;
}

function isDeferredDirect(tool: t.LCTool): boolean {
  return tool.defer_loading === true && allowsToolCaller(tool, 'direct');
}

function isDispatchable(
  tool: t.LCTool,
  dispatch: t.ToolDispatchOptions
): boolean {
  return (
    isDeferredDirect(tool) &&
    (dispatch.canDispatch == null || dispatch.canDispatch(tool.name))
  );
}

function parseArguments(
  target: string,
  rawArgs: unknown
): { args: Record<string, unknown> } | { error: string } {
  const args =
    rawArgs == null || rawArgs === '' ? {} : coerceRecordArgs(rawArgs);
  return args == null
    ? { error: `The arguments for "${target}" must be a JSON object.` }
    : { args };
}

/**
 * Canonicalizes one model tool call before any other layer reads it. A
 * dispatcher call becomes the call of its target (same id, target name and
 * arguments). Only deferred tools the model may call directly are targets. A
 * direct call to such a tool runs the same way and carries the dispatcher call
 * that replaces it in the history. Calls to a `disabled` tool are refused
 * whichever way they arrive.
 */
export function resolveDispatchedCall(
  call: ToolCall,
  getDefinition: DefinitionLookup,
  dispatch: false | t.ToolDispatchOptions
): DispatchResolution {
  if (dispatch === false || call.name !== TOOL_DISPATCHER_NAME) {
    const tool = getDefinition(call.name);
    if (tool == null) {
      return { call };
    }
    const disabled = disabledError(call.name, tool);
    if (disabled != null) {
      return { error: disabled };
    }
    if (dispatch === false || !isDeferredDirect(tool)) {
      return { call };
    }
    if (!isDispatchable(tool, dispatch)) {
      return {
        error: `Tool "${call.name}" is not available through ${TOOL_DISPATCHER_NAME}.`,
      };
    }
    const parsed = parseArguments(call.name, call.args);
    if ('error' in parsed) {
      return parsed;
    }
    return {
      call: { ...call, args: parsed.args },
      modelCall: {
        name: TOOL_DISPATCHER_NAME,
        args: { name: call.name, arguments: JSON.stringify(parsed.args) },
      },
      rewritten: true,
    };
  }

  const target = call.args.name;
  const tool =
    typeof target === 'string' && target !== TOOL_DISPATCHER_NAME
      ? getDefinition(target)
      : undefined;
  if (typeof target !== 'string' || tool == null || !isDispatchable(tool, dispatch)) {
    return {
      error: `Tool ${JSON.stringify(target)} is not available through ${TOOL_DISPATCHER_NAME}.`,
    };
  }
  const disabled = disabledError(target, tool);
  if (disabled != null) {
    return { error: disabled };
  }
  const parsed = parseArguments(target, call.args.arguments);
  if ('error' in parsed) {
    return parsed;
  }
  return {
    call: { ...call, name: target, args: parsed.args },
    modelCall: { name: call.name, args: call.args },
  };
}

type ProviderCallItem = {
  id?: string;
  call_id?: string;
  input?: unknown;
  arguments?: unknown;
  function?: { name?: string; arguments?: string };
};

/**
 * Rewrites one provider-native copy of a call: an Anthropic-style content
 * block (`id`, `input`), a Responses output item (`call_id`, `arguments`), or
 * a raw Chat Completions call (`id`, `function`).
 */
function rewriteProviderItem(
  item: unknown,
  rewrites: ReadonlyMap<string, ModelToolCall>
): unknown {
  if (item == null || typeof item !== 'object') {
    return item;
  }
  const native = item as ProviderCallItem;
  const rewrite = rewrites.get(native.call_id ?? native.id ?? '');
  if (rewrite == null) {
    return item;
  }
  if (native.function != null) {
    return {
      ...native,
      function: {
        ...native.function,
        name: rewrite.name,
        arguments: JSON.stringify(rewrite.args),
      },
    };
  }
  return {
    ...native,
    name: rewrite.name,
    ...(native.input !== undefined && { input: rewrite.args }),
    ...(typeof native.arguments === 'string' && {
      arguments: JSON.stringify(rewrite.args),
    }),
  };
}

function rewriteProviderItems<V>(
  value: V,
  rewrites: ReadonlyMap<string, ModelToolCall>
): V {
  return Array.isArray(value)
    ? (value.map((item) => rewriteProviderItem(item, rewrites)) as V)
    : value;
}

/**
 * Replacement for an assistant message (same id, so the messages reducer
 * upserts it) whose direct calls to deferred tools become dispatcher calls in
 * `tool_calls` and in every provider-native copy the converters replay
 * (content blocks, raw calls, Responses output items).
 */
export function rewriteDispatchedToolCalls(
  message: AIMessage,
  rewrites: ReadonlyMap<string, ModelToolCall>
): AIMessage {
  const { additional_kwargs, response_metadata } = message;
  return new AIMessage({
    id: message.id,
    name: message.name,
    content: rewriteProviderItems(message.content, rewrites),
    additional_kwargs: {
      ...additional_kwargs,
      ...(additional_kwargs.tool_calls != null && {
        tool_calls: rewriteProviderItems(
          additional_kwargs.tool_calls,
          rewrites
        ),
      }),
    },
    response_metadata: {
      ...response_metadata,
      ...(response_metadata.output != null && {
        output: rewriteProviderItems(response_metadata.output, rewrites),
      }),
    },
    usage_metadata: message.usage_metadata,
    invalid_tool_calls: message.invalid_tool_calls,
    tool_calls: message.tool_calls?.map((toolCall) => {
      const rewrite =
        toolCall.id == null ? undefined : rewrites.get(toolCall.id);
      return rewrite == null ? toolCall : { ...toolCall, ...rewrite };
    }),
  });
}
