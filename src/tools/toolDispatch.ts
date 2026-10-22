import { Validator } from '@langchain/core/utils/json_schema';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { Command } from '@langchain/langgraph';
import type * as t from '@/types';
import { coerceArgsForSchema, coerceRecordArgs } from './eagerEventExecution';
import { serializeToolContentBounded } from '@/utils/toolContent';
import { allowsToolCaller } from './CallerCapabilities';

export const TOOL_DISPATCHER_NAME = 'call_tool';

const TOOL_DISPATCHER_DESCRIPTION =
  'Calls a tool that was made available during this conversation and is not in your tool list. Tools in your tool list are called directly.';

/** Validation errors quoted back to the model; the first few are enough to fix a call. */
const MAX_REPORTED_SCHEMA_ERRORS = 3;
/** Aggregate keywords that only repeat a nested error. */
const AGGREGATE_SCHEMA_KEYWORDS = new Set(['properties', 'items']);

export type ResolvedToolDispatch = {
  name: string;
  description: string;
  canDispatch?: (toolName: string) => boolean;
};

type DispatchSettings = Pick<ResolvedToolDispatch, 'name' | 'canDispatch'>;

/** Normalizes `AgentInputs.toolDispatch`; `undefined` when the host did not opt in. */
export function resolveToolDispatchConfig(
  config: true | t.ToolDispatchConfig | undefined
): ResolvedToolDispatch | undefined {
  if (config == null) {
    return undefined;
  }
  const options = config === true ? {} : config;
  return {
    name: options.name ?? TOOL_DISPATCHER_NAME,
    description: options.description ?? TOOL_DISPATCHER_DESCRIPTION,
    ...(options.canDispatch != null && { canDispatch: options.canDispatch }),
  };
}

/**
 * Schema of the dispatcher tool. `arguments` is a JSON string rather than an
 * object: an object property without declared properties is rejected or
 * dropped by some providers' function-declaration formats.
 */
export function createToolDispatcherDefinition(
  config: true | t.ToolDispatchConfig = true
): t.LCTool {
  const { name, description } = resolveToolDispatchConfig(config)!;
  return {
    name,
    description,
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

export type DispatchableToolCall = {
  id?: string;
  name: string;
  args: Record<string, unknown>;
};

/** A tool call as the model sees it in the history. */
export type ModelToolCall = { name: string; args: Record<string, unknown> };

export type ToolDispatchResolution<C extends DispatchableToolCall> =
  | {
      /** The call to run: the target for a dispatcher call, else the input. */
      call: C;
      /** Dispatcher name when `call` runs as a dispatched call. */
      dispatcher?: string;
      /**
       * Set when the model called a dispatchable tool directly: the
       * equivalent dispatcher call, which replaces the model's call in the
       * history so only declared tools are named there.
       */
      rewrite?: ModelToolCall;
    }
  | { error: string };

type ToolDefinitionLookup = Pick<ReadonlyMap<string, t.LCTool>, 'get'>;

const validators = new WeakMap<t.JsonSchemaType, Validator>();

function toStandardSchema(schema: unknown): unknown {
  if (Array.isArray(schema)) {
    return schema.map(toStandardSchema);
  }
  if (schema == null || typeof schema !== 'object') {
    return schema;
  }
  return Object.fromEntries(
    Object.entries(schema).map(([key, value]) => [
      key,
      key === 'type' && value === 'float' ? 'number' : toStandardSchema(value),
    ])
  );
}

function getValidator(schema: t.JsonSchemaType): Validator {
  let validator = validators.get(schema);
  if (validator == null) {
    validator = new Validator(
      toStandardSchema(schema) as ConstructorParameters<typeof Validator>[0],
      '2019-09',
      false
    );
    validators.set(schema, validator);
  }
  return validator;
}

function validateArgs(
  args: Record<string, unknown>,
  schema: t.JsonSchemaType | undefined
): string | undefined {
  if (schema == null) {
    return undefined;
  }
  let result: ReturnType<Validator['validate']>;
  try {
    result = getValidator(schema).validate(args);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    return `the tool's schema could not be applied (${reason})`;
  }
  const { valid, errors } = result;
  if (valid) {
    return undefined;
  }
  return errors
    .filter(({ keyword }) => !AGGREGATE_SCHEMA_KEYWORDS.has(keyword))
    .slice(0, MAX_REPORTED_SCHEMA_ERRORS)
    .map(({ instanceLocation, error }) => `${instanceLocation}: ${error}`)
    .join('; ');
}

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

function isDispatchable(tool: t.LCTool, dispatch: DispatchSettings): boolean {
  return (
    isDeferredDirect(tool) &&
    (dispatch.canDispatch == null || dispatch.canDispatch(tool.name))
  );
}

function resolveTarget<C extends DispatchableToolCall>(
  call: C,
  target: string,
  tool: t.LCTool,
  rawArgs: unknown
): { call: C; args: Record<string, unknown> } | { error: string } {
  const disabled = disabledError(target, tool);
  if (disabled != null) {
    return { error: disabled };
  }
  const record =
    rawArgs == null || rawArgs === '' ? {} : coerceRecordArgs(rawArgs);
  if (record == null) {
    return { error: `The arguments for "${target}" must be a JSON object.` };
  }
  const args = coerceArgsForSchema(record, tool.parameters);
  const invalid = validateArgs(args, tool.parameters);
  if (invalid != null) {
    return { error: `Invalid arguments for "${target}": ${invalid}` };
  }
  return { call: { ...call, name: target, args }, args };
}

/**
 * Canonicalizes one model tool call before any other layer reads it. A
 * dispatcher call becomes the call of its target (same id, target name,
 * target arguments coerced and validated against the target schema), so
 * hooks, approval, permission layers, and execution all see the real tool.
 * Only deferred registry tools that the model may call directly are
 * dispatchable. A direct call to such a tool runs the same way and carries
 * the dispatcher call that replaces it in the history. Calls to a `disabled`
 * tool are refused whichever way they arrive. Hosts with a permission layer
 * outside the SDK's ToolNode call this same function so every layer agrees
 * on the canonical call.
 */
export function resolveDispatchedToolCall<C extends DispatchableToolCall>(
  call: C,
  registry: ToolDefinitionLookup | undefined,
  dispatch?: DispatchSettings
): ToolDispatchResolution<C> {
  if (dispatch == null || call.name !== dispatch.name) {
    const tool = registry?.get(call.name);
    if (dispatch != null && tool != null && isDispatchable(tool, dispatch)) {
      const resolved = resolveTarget(call, call.name, tool, call.args);
      return 'error' in resolved
        ? resolved
        : {
          call: resolved.call,
          dispatcher: dispatch.name,
          rewrite: {
            name: dispatch.name,
            args: {
              name: call.name,
              arguments: JSON.stringify(resolved.args),
            },
          },
        };
    }
    const error = tool == null ? undefined : disabledError(call.name, tool);
    if (
      error == null &&
      dispatch?.canDispatch != null &&
      tool != null &&
      isDeferredDirect(tool) &&
      !dispatch.canDispatch(tool.name)
    ) {
      return {
        error: `Tool "${call.name}" is not available through ${dispatch.name}.`,
      };
    }
    return error == null ? { call } : { error };
  }

  const target = call.args.name;
  if (typeof target !== 'string' || target === '') {
    return {
      error: `${dispatch.name} requires "name": the exact name of a tool made available in this conversation.`,
    };
  }
  const tool = target === dispatch.name ? undefined : registry?.get(target);
  if (tool == null || !isDispatchable(tool, dispatch)) {
    const direct =
      tool != null && tool.defer_loading !== true
        ? ' It is in your tool list: call it directly.'
        : '';
    return {
      error: `Tool "${target}" is not available through ${dispatch.name}.${direct}`,
    };
  }
  const resolved = resolveTarget(call, target, tool, call.args.arguments);
  return 'error' in resolved
    ? resolved
    : { call: resolved.call, dispatcher: dispatch.name };
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
 * Per-node record of dispatched calls. Everything inside the ToolNode sees
 * the target; results, completion events, and rewritten assistant messages
 * leave in the model form, so the history only names declared tools.
 */
export class ToolDispatchTracker {
  private readonly modelCalls = new Map<string, ModelToolCall>();
  private readonly registry: ToolDefinitionLookup;

  constructor(
    getDefinition: (name: string) => t.LCTool | undefined,
    private readonly maxChars: number,
    private readonly dispatch?: DispatchSettings
  ) {
    this.registry = { get: getDefinition };
  }

  /** The enabled dispatcher's name. */
  get dispatcher(): string | undefined {
    return this.dispatch?.name;
  }

  /**
   * Resolves a batch: refused calls come back unchanged with their error,
   * direct calls to dispatchable tools come back with the dispatcher call
   * that replaces them in the history.
   */
  canonicalize<C extends DispatchableToolCall>(
    calls: C[]
  ): {
    calls: C[];
    refused: Map<string, string>;
    rewrites: Map<string, ModelToolCall>;
  } {
    const refused = new Map<string, string>();
    const rewrites = new Map<string, ModelToolCall>();
    const canonical = calls.map((call) => {
      if (call.id == null || call.id === '') {
        return call;
      }
      let resolution: ToolDispatchResolution<C>;
      try {
        resolution = resolveDispatchedToolCall(
          call,
          this.registry,
          this.dispatch
        );
      } catch (error) {
        const reason = error instanceof Error ? error.message : String(error);
        resolution = {
          error: `Tool call "${call.name}" could not be resolved: ${reason}`,
        };
      }
      if ('error' in resolution) {
        refused.set(call.id, resolution.error);
        return call;
      }
      if (resolution.dispatcher == null) {
        return resolution.call;
      }
      const modelCall = resolution.rewrite ?? {
        name: call.name,
        args: call.args,
      };
      this.modelCalls.set(call.id, modelCall);
      if (resolution.rewrite != null) {
        rewrites.set(call.id, resolution.rewrite);
      }
      return resolution.call;
    });
    return { calls: canonical, refused, rewrites };
  }

  has(callId: string | undefined): boolean {
    return callId != null && this.modelCalls.has(callId);
  }

  clear(): void {
    this.modelCalls.clear();
  }

  /** A dispatched call's completion keeps the model's form, with the target alongside. */
  toModelToolCall(toolCall: t.ProcessedToolCall): t.ProcessedToolCall {
    const modelCall = this.modelCalls.get(toolCall.id);
    if (modelCall == null) {
      return toolCall;
    }
    return {
      ...toolCall,
      name: modelCall.name,
      args: serializeToolContentBounded(modelCall.args, this.maxChars),
      dispatched: { name: toolCall.name, args: toolCall.args },
    };
  }

  /** Results of dispatched calls answer the dispatcher call under its name. */
  toModelMessages(
    outputs: (BaseMessage | Command)[]
  ): (BaseMessage | Command)[] {
    if (this.modelCalls.size === 0) {
      return outputs;
    }
    return outputs.map((output) => {
      if (!ToolMessage.isInstance(output)) {
        return output;
      }
      const name = this.modelCalls.get(output.tool_call_id)?.name;
      if (name == null || output.name === name) {
        return output;
      }
      return new ToolMessage({
        id: output.id,
        content: output.content,
        name,
        status: output.status,
        artifact: output.artifact,
        metadata: output.metadata,
        tool_call_id: output.tool_call_id,
        additional_kwargs: output.additional_kwargs,
        response_metadata: output.response_metadata,
      });
    });
  }
}

/**
 * Replacement for an assistant message (same id, so the messages reducer
 * upserts it) whose direct calls to dispatchable tools become dispatcher
 * calls in `tool_calls` and in every provider-native copy the converters
 * replay (content blocks, raw calls, Responses output items).
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
