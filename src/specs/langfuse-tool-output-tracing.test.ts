import { context } from '@opentelemetry/api';
import { LangfuseOtelSpanAttributes } from '@langfuse/tracing';
import { AIMessage, ToolMessage, HumanMessage } from '@langchain/core/messages';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import type { BaseMessage } from '@langchain/core/messages';
import type { Context } from '@opentelemetry/api';
import type { TPayload } from '@/types';
import {
  LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
  LANGFUSE_OBSERVATION_METADATA_ARTIFACT_KEY,
  classifyLangfuseToolNodeSpan,
  prepareLangfuseSpanForExport,
  redactLangfuseSpanToolOutputs,
  shouldTraceToolNodeForLangfuse,
} from '@/langfuseToolOutputTracing';
import {
  resolveLangfuseConfigForSpan,
  resolveLangfuseRuntimeScope,
  resolveToolOutputTracingConfigForSpan,
  withLangfuseRuntimeScope,
} from '@/langfuseRuntimeScope';
import {
  runWithLangfuseRuntimeContext,
  type ResolvedLangfuseToolOutputTracingConfig,
} from '@/langfuseRuntimeContext';
import {
  resolveLangfuseConfig,
  resolveToolOutputTracingConfig,
} from '@/langfuseConfig';
import { ensureOpenTelemetryContextManager } from '@/instrumentation';
import { projectToolStreamContentForProvider } from '@/messages/core';
import { formatAgentMessages } from '@/messages/format';
import { ContentTypes } from '@/common';

type SerializedLangfuseChatMessage = {
  content: BaseMessage['content'];
  role?: string;
  additional_kwargs?: BaseMessage['additional_kwargs'];
  tool_calls?:
    | NonNullable<AIMessage['tool_calls']>
    | NonNullable<BaseMessage['additional_kwargs']['tool_calls']>;
};

type RedactedMessage = {
  role?: string;
  content?: string;
  tool_calls?: Array<{
    id?: string;
    name?: string;
    args?: {
      query?: string;
    };
  }>;
};

function createSpan(
  name: string,
  attributes: Record<string, unknown>
): ReadableSpan {
  return { name, attributes } as unknown as ReadableSpan;
}

function createConfig(
  overrides: Partial<ResolvedLangfuseToolOutputTracingConfig> = {}
): ResolvedLangfuseToolOutputTracingConfig {
  return {
    enabled: true,
    redactedToolNames: new Set<string>(),
    redactedToolNameMatchMode: 'exact',
    redactionText: LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
    ...overrides,
  };
}

function serializeMessageForLangfuse(
  message: BaseMessage
): SerializedLangfuseChatMessage {
  if (message instanceof HumanMessage) {
    return { content: message.content, role: 'user' };
  }

  if (message instanceof AIMessage) {
    const response: SerializedLangfuseChatMessage = {
      content: message.content,
      role: 'assistant',
    };
    if (message.tool_calls != null && message.tool_calls.length > 0) {
      response.tool_calls = message.tool_calls;
    }
    if (message.additional_kwargs.tool_calls != null) {
      response.tool_calls = message.additional_kwargs.tool_calls;
    }
    return response;
  }

  if (message instanceof ToolMessage) {
    return {
      content: message.content,
      additional_kwargs: message.additional_kwargs,
      role: message.name,
    };
  }

  return message.name != null
    ? { content: message.content, role: message.name }
    : { content: message.content };
}

function readJsonAttribute<T>(span: ReadableSpan, key: string): T {
  return JSON.parse(span.attributes[key] as string) as T;
}

describe('Langfuse tool output tracing redaction', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('keeps internal ToolNode batch tracing opt-in', () => {
    delete process.env.LANGFUSE_SECRET_KEY;
    delete process.env.LANGFUSE_PUBLIC_KEY;
    delete process.env.LANGFUSE_BASE_URL;

    expect(shouldTraceToolNodeForLangfuse({})).toBe(false);
    expect(
      shouldTraceToolNodeForLangfuse({
        runLangfuse: {
          enabled: true,
          publicKey: 'pk-run',
          secretKey: 'sk-run',
        },
      })
    ).toBe(false);
    expect(
      shouldTraceToolNodeForLangfuse({
        agentLangfuse: {
          enabled: true,
          publicKey: 'pk-agent',
          secretKey: 'sk-agent',
          baseUrl: 'https://langfuse.test',
          toolNodeTracing: { enabled: true },
        },
      })
    ).toBe(true);

    process.env.LANGFUSE_SECRET_KEY = 'sk-test';
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-test';
    process.env.LANGFUSE_BASE_URL = 'https://langfuse.test';

    expect(shouldTraceToolNodeForLangfuse({})).toBe(false);
    expect(
      shouldTraceToolNodeForLangfuse({
        runLangfuse: { toolNodeTracing: { enabled: true } },
      })
    ).toBe(true);
    expect(
      shouldTraceToolNodeForLangfuse({
        runLangfuse: { toolNodeTracing: { enabled: false } },
      })
    ).toBe(false);
  });

  it('lets an agent explicitly opt into ToolNode batch tracing', () => {
    delete process.env.LANGFUSE_SECRET_KEY;
    delete process.env.LANGFUSE_PUBLIC_KEY;
    delete process.env.LANGFUSE_BASE_URL;

    expect(
      shouldTraceToolNodeForLangfuse({
        runLangfuse: {
          enabled: false,
        },
        agentLangfuse: {
          enabled: true,
          publicKey: 'pk-agent',
          secretKey: 'sk-agent',
          baseUrl: 'https://langfuse.test',
          toolNodeTracing: { enabled: true },
        },
      })
    ).toBe(true);
  });

  it('keeps ToolNode tracing disabled when resolved Langfuse is disabled', () => {
    process.env.LANGFUSE_SECRET_KEY = 'sk-test';
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-test';

    expect(
      shouldTraceToolNodeForLangfuse({
        runLangfuse: {
          enabled: false,
          toolNodeTracing: { enabled: true },
        },
      })
    ).toBe(false);
  });

  it('classifies LangGraph tool-node spans as Langfuse tool observations', () => {
    const span = createSpan('tool_batch', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'span',
      [`${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.langgraph_node`]:
        'tools=agent_1',
    });

    redactLangfuseSpanToolOutputs(span, createConfig());

    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE]).toBe(
      'tool'
    );
  });

  it('classifies LangGraph tool-node spans without requiring redaction config', () => {
    const span = createSpan('tool_batch', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'span',
      [`${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.langgraph_node`]:
        'tools=agent_1',
    });

    classifyLangfuseToolNodeSpan(span);

    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE]).toBe(
      'tool'
    );
  });

  it('does not reclassify non-tool LangGraph spans', () => {
    const span = createSpan('agent=agent_1', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'span',
      [`${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.langgraph_node`]:
        'agent=agent_1',
    });

    redactLangfuseSpanToolOutputs(span, createConfig());

    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE]).toBe(
      'span'
    );
  });

  it('redacts raw tool observation output when tool output tracing is disabled', () => {
    const span = createSpan('execute_sql', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'tool',
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: '{"query":"select 1"}',
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: 'secret rows',
    });

    redactLangfuseSpanToolOutputs(span, createConfig({ enabled: false }));

    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT]).toBe(
      '{"query":"select 1"}'
    );
  });

  it('redacts ToolMessage content inside serialized generation inputs', () => {
    const messages = [
      { role: 'user', content: 'show tables' },
      {
        role: 'execute_sql',
        content: 'private query result',
        additional_kwargs: {},
      },
    ];
    const span = createSpan('gpt-4o', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'generation',
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify(messages),
    });

    redactLangfuseSpanToolOutputs(span, createConfig({ enabled: false }));

    const redacted = JSON.parse(
      span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT] as string
    ) as Array<{ role: string; content: string }>;
    expect(redacted[0].content).toBe('show tables');
    expect(redacted[1].content).toBe(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
  });

  it('redacts captured Responses server-tool outputs wherever they are serialized', () => {
    const outputs = [
      {
        id: 'local_output',
        type: 'local_shell_call_output',
        status: 'completed',
        output: 'local shell secret',
      },
      {
        id: 'shell_output',
        call_id: 'shell_call',
        type: 'shell_call_output',
        status: 'completed',
        output: 'shell secret',
      },
      {
        id: 'patch_output',
        call_id: 'patch_call',
        type: 'apply_patch_call_output',
        status: 'completed',
        output: 'patch secret',
      },
      {
        id: 'program_output',
        call_id: 'program_call',
        type: 'program_output',
        status: 'completed',
        result: 'program secret',
      },
      {
        id: 'code_interpreter_call',
        type: 'code_interpreter_call',
        code: 'code input stays visible',
        container_id: 'container_1',
        status: 'completed',
        outputs: [{ type: 'logs', logs: 'code interpreter secret' }],
      },
      {
        id: 'mcp_call',
        type: 'mcp_call',
        name: 'private_mcp_tool',
        arguments: '{"query":"mcp input stays visible"}',
        server_label: 'mcp_server',
        status: 'failed',
        output: 'mcp output secret',
        error: 'mcp error secret',
      },
      {
        id: 'mcp_list',
        type: 'mcp_list_tools',
        server_label: 'mcp_server',
        tools: [{ name: 'listed secret tool', input_schema: {} }],
        error: 'mcp list error secret',
      },
      {
        id: 'image_generation_call',
        type: 'image_generation_call',
        status: 'completed',
        revised_prompt: 'image prompt stays visible',
        result: 'image generation secret',
      },
      {
        id: 'file_search_call',
        type: 'file_search_call',
        status: 'completed',
        queries: ['file query stays visible'],
        results: [{ text: 'file search secret' }],
      },
      {
        id: 'web_search_call',
        type: 'web_search_call',
        status: 'completed',
        action: {
          type: 'search',
          queries: ['web query stays visible'],
          sources: [{ type: 'url', url: 'web source secret' }],
        },
      },
      {
        id: 'tool_search_output',
        call_id: 'tool_search_call',
        type: 'tool_search_output',
        status: 'completed',
        tools: [{ type: 'function', name: 'tool definition secret' }],
      },
      {
        id: 'function_call',
        call_id: 'function_call_id',
        type: 'function_call',
        name: 'private_function',
        arguments: 'function input stays visible',
      },
      {
        id: 'function_output',
        call_id: 'function_call_id',
        type: 'function_call_output',
        output: 'function output secret',
      },
      {
        id: 'custom_call',
        call_id: 'custom_call_id',
        type: 'custom_tool_call',
        name: 'private_custom',
        input: 'custom input stays visible',
      },
      {
        id: 'custom_output',
        call_id: 'custom_call_id',
        type: 'custom_tool_call_output',
        output: 'custom output secret',
      },
      {
        id: 'computer_output',
        call_id: 'computer_call_id',
        type: 'computer_call_output',
        acknowledged_safety_checks: ['computer input stays visible'],
        output: 'computer output secret',
      },
    ];
    const serializedMessage = {
      role: 'assistant',
      content: outputs.map((value) => ({ type: 'non_standard', value })),
      content_blocks: outputs.map((value) => ({
        type: 'non_standard',
        value,
      })),
      additional_kwargs: { tool_outputs: outputs },
    };
    const span = createSpan('gpt-5.6', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'generation',
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: JSON.stringify([
        serializedMessage,
      ]),
    });

    redactLangfuseSpanToolOutputs(span, createConfig({ enabled: false }));

    const redacted = span.attributes[
      LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT
    ] as string;
    for (const secret of [
      'local shell secret',
      'shell secret',
      'patch secret',
      'program secret',
      'code interpreter secret',
      'mcp output secret',
      'mcp error secret',
      'listed secret tool',
      'mcp list error secret',
      'image generation secret',
      'file search secret',
      'web source secret',
      'tool definition secret',
      'function output secret',
      'custom output secret',
      'computer output secret',
    ]) {
      expect(redacted).not.toContain(secret);
    }
    expect(redacted).toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    for (const input of [
      'local_output',
      'shell_call',
      'patch_call',
      'program_call',
      'code input stays visible',
      'mcp input stays visible',
      'private_mcp_tool',
      'mcp_server',
      'image prompt stays visible',
      'file query stays visible',
      'web query stays visible',
      'tool_search_call',
      'function input stays visible',
      'custom input stays visible',
      'computer input stays visible',
    ]) {
      expect(redacted).toContain(input);
    }
  });

  it('selectively redacts captured Responses outputs by canonical tool name', () => {
    const outputs = [
      {
        type: 'local_shell_call_output',
        output: 'public local output',
      },
      { type: 'shell_call_output', output: 'private shell output' },
      { type: 'apply_patch_call_output', output: 'public patch output' },
      { type: 'program_output', result: 'private program output' },
      {
        type: 'code_interpreter_call',
        code: 'public code input',
        outputs: [{ type: 'logs', logs: 'private code output' }],
      },
      {
        type: 'mcp_call',
        name: 'private_mcp_tool',
        arguments: 'public mcp input',
        output: 'private named mcp output',
        error: 'private named mcp error',
      },
      {
        type: 'mcp_call',
        name: 'public_mcp_tool',
        output: 'public named mcp output',
      },
      {
        type: 'mcp_call',
        output: 'private fallback mcp output',
      },
      {
        type: 'mcp_list_tools',
        server_label: 'public server label',
        tools: [{ name: 'private listed tool' }],
        error: 'private list error',
      },
      {
        type: 'image_generation_call',
        revised_prompt: 'public image prompt',
        result: 'private generated image',
      },
      {
        type: 'file_search_call',
        queries: ['public file query'],
        results: [{ text: 'private file result' }],
      },
      {
        type: 'web_search_call',
        results: [{ title: 'private web result' }],
        action: {
          type: 'search',
          queries: ['public web query'],
          sources: [{ url: 'private web source' }],
        },
      },
      {
        type: 'tool_search_output',
        call_id: 'public tool search call id',
        tools: [{ name: 'private tool definition' }],
      },
      {
        type: 'computer_call_output',
        acknowledged_safety_checks: ['public computer input'],
        output: 'private computer output',
      },
    ];
    const span = createSpan('gpt-5.6', {
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify(outputs),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set([
          'shell',
          'program',
          'code_interpreter',
          'private_mcp_tool',
          'mcp',
          'mcp_list_tools',
          'image_generation',
          'file_search',
          'web_search',
          'tool_search',
          'computer_use',
        ]),
      })
    );

    const redacted = readJsonAttribute<Array<Record<string, unknown>>>(
      span,
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    );
    expect(redacted[0].output).toBe('public local output');
    expect(redacted[1].output).toBe(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(redacted[2].output).toBe('public patch output');
    expect(redacted[3].result).toBe(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(redacted[4]).toMatchObject({
      code: 'public code input',
      outputs: LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
    });
    expect(redacted[5]).toMatchObject({
      name: 'private_mcp_tool',
      arguments: 'public mcp input',
      output: LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
      error: LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
    });
    expect(redacted[6].output).toBe('public named mcp output');
    expect(redacted[7].output).toBe(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(redacted[8]).toMatchObject({
      server_label: 'public server label',
      tools: LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
      error: LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
    });
    expect(redacted[9]).toMatchObject({
      revised_prompt: 'public image prompt',
      result: LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
    });
    expect(redacted[10]).toMatchObject({
      queries: ['public file query'],
      results: LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
    });
    expect(redacted[11]).toMatchObject({
      results: LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
      action: {
        type: 'search',
        queries: ['public web query'],
        sources: LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
      },
    });
    expect(redacted[12]).toMatchObject({
      call_id: 'public tool search call id',
      tools: LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
    });
    expect(redacted[13]).toMatchObject({
      acknowledged_safety_checks: ['public computer input'],
      output: LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
    });
  });

  it('maps Responses function and custom outputs to their call names', () => {
    const outputs = [
      {
        type: 'function_call',
        call_id: 'private_function_call',
        name: 'execute_sql',
        arguments: 'private function input stays visible',
      },
      {
        type: 'function_call_output',
        call_id: 'private_function_call',
        output: 'private function output',
      },
      {
        type: 'function_call',
        call_id: 'public_function_call',
        name: 'public_function',
        arguments: 'public function input',
      },
      {
        type: 'function_call_output',
        call_id: 'public_function_call',
        output: 'public function output',
      },
      {
        type: 'custom_tool_call',
        call_id: 'private_custom_call',
        name: 'private_custom_tool',
        input: 'private custom input stays visible',
      },
      {
        type: 'custom_tool_call_output',
        call_id: 'private_custom_call',
        output: 'private custom output',
      },
      {
        type: 'custom_tool_call_output',
        call_id: 'unmatched_custom_call',
        output: 'unmatched custom output stays visible',
      },
    ];
    const span = createSpan('gpt-5.6', {
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify(outputs),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql', 'private_custom_tool']),
      })
    );

    const redacted = readJsonAttribute<Array<Record<string, unknown>>>(
      span,
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    );
    expect(redacted[0].arguments).toBe('private function input stays visible');
    expect(redacted[1].output).toBe(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(redacted[3].output).toBe('public function output');
    expect(redacted[4].input).toBe('private custom input stays visible');
    expect(redacted[5].output).toBe(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(redacted[6].output).toBe('unmatched custom output stays visible');
  });

  it('redacts standard server-tool results globally and by paired call name', () => {
    const blocks = [
      {
        type: 'server_tool_call',
        id: 'private_server_call',
        name: 'file_search',
        args: { query: 'private server input stays visible' },
      },
      {
        type: 'server_tool_call_result',
        toolCallId: 'private_server_call',
        status: 'success',
        output: { results: ['private standard server result'] },
      },
      {
        type: 'server_tool_call',
        id: 'public_server_call',
        name: 'web_search',
        args: { query: 'public server input' },
      },
      {
        type: 'server_tool_call_result',
        toolCallId: 'public_server_call',
        status: 'success',
        output: 'public standard server result',
      },
      {
        type: 'server_tool_call_result',
        toolCallId: 'unknown_server_call',
        status: 'success',
        output: 'unmatched standard server result',
      },
    ];
    const selectiveSpan = createSpan('gpt-5.6', {
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify(blocks),
    });

    redactLangfuseSpanToolOutputs(
      selectiveSpan,
      createConfig({ redactedToolNames: new Set(['file_search']) })
    );

    const selectivelyRedacted = readJsonAttribute<
      Array<Record<string, unknown>>
    >(selectiveSpan, LangfuseOtelSpanAttributes.OBSERVATION_INPUT);
    expect(selectivelyRedacted[0]).toMatchObject({
      args: { query: 'private server input stays visible' },
    });
    expect(selectivelyRedacted[1].output).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
    expect(selectivelyRedacted[3].output).toBe('public standard server result');
    expect(selectivelyRedacted[4].output).toBe(
      'unmatched standard server result'
    );

    const globalSpan = createSpan('gpt-5.6', {
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify(blocks),
    });
    redactLangfuseSpanToolOutputs(globalSpan, createConfig({ enabled: false }));
    const globallyRedacted = readJsonAttribute<Array<Record<string, unknown>>>(
      globalSpan,
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    );
    expect(globallyRedacted[1].output).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
    expect(globallyRedacted[3].output).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
    expect(globallyRedacted[4].output).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
  });

  it('redacts marked projected server-tool images without touching user images', () => {
    const message = new AIMessage({
      content: [{ type: 'text', text: 'Partial answer stays visible.' }],
      additional_kwargs: {
        tool_outputs: [
          {
            id: 'code_image_call',
            type: 'code_interpreter_call',
            code: 'public image-producing code',
            status: 'completed',
            outputs: [
              {
                type: 'image',
                url: 'data:image/png;base64,private-code-image-data',
              },
            ],
          },
          {
            id: 'generated_image_call',
            type: 'image_generation_call',
            status: 'completed',
            result: 'private-generated-image-data',
          },
        ],
      },
      response_metadata: {
        model_provider: 'openai',
        preempted: true,
      },
    });
    const [projected] = projectToolStreamContentForProvider(
      [message],
      'native'
    );
    const projectedJson = projected.toJSON();
    const serializedProjection = JSON.stringify(projectedJson);
    expect(serializedProjection).toContain('private-code-image-data');
    expect(serializedProjection).toContain('private-generated-image-data');
    expect(serializedProjection).toContain('librechatServerToolResult');

    const userImage = {
      type: 'image',
      mimeType: 'image/png',
      data: 'public-user-image-data',
    };
    const span = createSpan('gpt-5.6', {
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify([
        projectedJson,
        userImage,
      ]),
    });
    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['code_interpreter', 'image_generation']),
      })
    );

    const redacted = span.attributes[
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    ] as string;
    expect(redacted).toContain('Partial answer stays visible.');
    expect(redacted).toContain('public-user-image-data');
    expect(redacted).not.toContain('private-code-image-data');
    expect(redacted).not.toContain('private-generated-image-data');
    expect(redacted).toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
  });

  it.each([
    ['globally', createConfig({ enabled: false })],
    [
      'selectively',
      createConfig({ redactedToolNames: new Set(['image_generation']) }),
    ],
  ] as const)(
    'redacts serialized normalized generated images %s without touching user images',
    (_scope, config) => {
      const generatedById = {
        id: 'ig_normal_response',
        type: 'image_generation_call',
        status: 'completed',
        result: 'private-generated-image-by-id',
      };
      const generatedSecond = {
        id: 'ig_normal_response_second',
        type: 'image_generation_call',
        status: 'completed',
        result: 'private-generated-image-by-data',
      };
      const message = new AIMessage({
        content: [
          {
            type: 'image',
            mimeType: 'image/png',
            data: generatedById.result,
            id: generatedById.id,
            metadata: { status: generatedById.status },
          },
          {
            type: 'image',
            mimeType: 'image/png',
            data: generatedSecond.result,
            id: generatedSecond.id,
            metadata: { status: generatedSecond.status },
          },
          {
            type: 'image',
            mimeType: 'image/png',
            id: generatedById.id,
            url: 'private-generated-image-url',
            fileId: 'private-generated-image-file-id',
            metadata: { status: generatedById.status },
          },
          {
            type: 'image',
            mimeType: 'image/png',
            data: 'public-assistant-application-image',
            id: 'application-image',
            metadata: { status: 'completed' },
          },
        ],
        additional_kwargs: {
          tool_outputs: [generatedById, generatedSecond],
        },
        response_metadata: {
          model_provider: 'openai',
          output: [generatedById, generatedSecond],
        },
      });
      const serializedMessage = serializeMessageForLangfuse(message);
      const serializedUserMessage = serializeMessageForLangfuse(
        new HumanMessage({
          content: [
            {
              type: 'image',
              mimeType: 'image/png',
              data: 'public-user-image-data',
            },
          ],
        })
      );
      expect(JSON.stringify(serializedMessage)).not.toContain(
        'image_generation_call'
      );
      const span = createSpan('gpt-5.6', {
        [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: JSON.stringify([
          serializedMessage,
          serializedUserMessage,
        ]),
      });

      redactLangfuseSpanToolOutputs(span, config);

      const redacted = span.attributes[
        LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT
      ] as string;
      expect(redacted).not.toContain(generatedById.result);
      expect(redacted).not.toContain(generatedSecond.result);
      expect(redacted).not.toContain('private-generated-image-url');
      expect(redacted).not.toContain('private-generated-image-file-id');
      expect(redacted).toContain('public-assistant-application-image');
      expect(redacted).toContain('public-user-image-data');
      expect(redacted).toContain(generatedById.id);
      expect(redacted).toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    }
  );

  it('does not reclassify a marked code-interpreter image as image generation', () => {
    const codeImage = 'data:image/png;base64,public-code-image';
    const span = createSpan('gpt-5.6', {
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: JSON.stringify([
        {
          role: 'assistant',
          content: [
            {
              type: 'image',
              url: codeImage,
              extras: {
                librechatServerToolResult: {
                  toolName: 'code_interpreter',
                },
              },
            },
          ],
        },
      ]),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['image_generation']),
      })
    );

    expect(
      span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]
    ).toContain(codeImage);
    expect(
      span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]
    ).not.toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
  });

  it('fails closed when OpenTelemetry truncates a JSON-shaped attribute', () => {
    const secret = 'private truncated tool output';
    const serialized = JSON.stringify([
      {
        role: 'assistant',
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              serverToolResult: {
                toolName: 'shell',
                status: 'success',
                output: secret,
              },
            }),
          },
        ],
      },
    ]);
    const span = createSpan('gpt-5.6', {
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: serialized.slice(0, -1),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({ redactedToolNames: new Set(['shell']) })
    );

    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
    expect(JSON.stringify(span.attributes)).not.toContain(secret);
  });

  it('redacts marked bounded replay text without parsing its payload', () => {
    const truncatedSecret =
      '{"serverToolResult":{"librechatResponsesReplay":true,"status":"success","output":"secret head…secret tail"';
    const span = createSpan('gpt-5.6', {
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify([
        {
          role: 'assistant',
          content: [
            {
              type: 'text',
              text: truncatedSecret,
            },
          ],
        },
      ]),
    });

    redactLangfuseSpanToolOutputs(span, createConfig({ enabled: false }));

    const redacted = span.attributes[
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    ] as string;
    expect(redacted).not.toContain('secret head');
    expect(redacted).not.toContain('secret tail');
    expect(redacted).toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
  });

  it('redacts replay output before root-span answer shaping', () => {
    const rootSecret = 'private root replay result';
    const span = createSpan('LangGraph', {
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify([
        { role: 'user', content: 'Continue.' },
      ]),
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: JSON.stringify([
        {
          role: 'assistant',
          content: [
            { type: 'text', text: 'Partial answer. ' },
            {
              type: 'text',
              text: JSON.stringify({
                serverToolResult: {
                  toolName: 'shell',
                  status: 'success',
                  output: rootSecret,
                },
              }),
              extras: {
                librechatServerToolResult: { toolName: 'shell' },
              },
            },
          ],
        },
      ]),
    });

    prepareLangfuseSpanForExport(span, createConfig({ enabled: false }));

    const output = span.attributes[
      LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT
    ] as string;
    expect(output).toContain('Partial answer.');
    expect(output).toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(output).not.toContain(rootSecret);
    expect(span.attributes['langfuse.trace.output']).toBeUndefined();
  });

  it('redacts a root tool output without deriving deprecated trace output', () => {
    const rootSecret = 'private root tool result';
    const span = createSpan('shell', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'tool',
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: rootSecret,
      'langfuse.trace.output': rootSecret,
    });

    prepareLangfuseSpanForExport(span, createConfig({ enabled: false }));

    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
    expect(span.attributes['langfuse.trace.output']).toBeUndefined();
    expect(JSON.stringify(span.attributes)).not.toContain(rootSecret);
  });

  it('preserves ordinary text that only resembles a server-tool result', () => {
    const ordinaryText =
      '{"serverToolResult":{"status":"success","output":"public literal text"}}';
    const span = createSpan('gpt-5.6', {
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify([
        {
          role: 'assistant',
          content: [{ type: 'text', text: ordinaryText }],
        },
      ]),
    });

    redactLangfuseSpanToolOutputs(span, createConfig({ enabled: false }));

    expect(
      span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT]
    ).toContain('public literal text');
    expect(
      span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT]
    ).not.toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
  });

  it('recognizes serialized fallback results only inside assistant messages', () => {
    const userLiteral =
      '{"serverToolResult":{"status":"success","output":"public user literal"}}';
    const assistantResult =
      '{"serverToolResult":{"librechatResponsesReplay":true,"status":"success","output":"private replay result"}}';
    const span = createSpan('gpt-5.6', {
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify([
        {
          role: 'user',
          content: [{ type: 'text', text: userLiteral }],
        },
        {
          role: 'assistant',
          content: [{ type: 'text', text: assistantResult }],
        },
      ]),
    });

    redactLangfuseSpanToolOutputs(span, createConfig({ enabled: false }));

    const redacted = span.attributes[
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    ] as string;
    expect(redacted).toContain('public user literal');
    expect(redacted).not.toContain('private replay result');
    expect(redacted).toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
  });

  it('redacts neutralized replay results from resumed generation input', () => {
    const message = new AIMessage({
      content: [{ type: 'text', text: 'Partial answer.' }],
      additional_kwargs: {
        tool_outputs: [
          {
            id: 'shell_output_item',
            call_id: 'shell_call',
            type: 'shell_call_output',
            status: 'completed',
            output: 'resumed generation secret',
          },
        ],
      },
      response_metadata: {
        model_provider: 'openai',
        preempted: true,
      },
    });
    const [projected] = projectToolStreamContentForProvider(
      [message],
      'fallback'
    );
    expect(projected.content).toEqual([
      { type: 'text', text: 'Partial answer.' },
      expect.objectContaining({
        type: 'text',
        text: expect.stringContaining('resumed generation secret'),
      }),
    ]);
    expect(JSON.stringify(projected.content)).not.toContain('extras');
    const span = createSpan('gpt-5.6', {
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify([
        projected.toJSON(),
      ]),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({ redactedToolNames: new Set(['shell']) })
    );

    const redacted = span.attributes[
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    ] as string;
    expect(redacted).toContain('Partial answer.');
    expect(redacted).not.toContain('resumed generation secret');
    expect(redacted).toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
  });

  it('redacts only configured tool names when output tracing stays enabled', () => {
    const messages = [
      { role: 'execute_sql', content: 'private query result' },
      { role: 'bash', content: 'public build log' },
    ];
    const span = createSpan('LangGraph', {
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: JSON.stringify({
        messages,
      }),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql']),
      })
    );

    const redacted = JSON.parse(
      span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] as string
    ) as { messages: Array<{ role: string; content: string }> };
    expect(redacted.messages[0].content).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
    expect(redacted.messages[1].content).toBe('public build log');
  });

  it('uses nested ToolMessage names instead of generic tool role', () => {
    const messages = [
      {
        role: 'tool',
        content: 'private query result',
        kwargs: {
          name: 'execute_sql',
          tool_call_id: 'call_1',
        },
      },
    ];
    const span = createSpan('gpt-4o', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'generation',
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify(messages),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql']),
      })
    );

    const redacted = readJsonAttribute<Array<{ content: string }>>(
      span,
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    );
    expect(redacted[0].content).toBe(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
  });

  it('maps tool_call_id to the preceding tool call name for allowlisted redaction', () => {
    const messages = [
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          {
            id: 'call_sql',
            name: 'execute_sql',
            args: { query: 'select * from private_table' },
          },
        ],
      },
      {
        role: 'tool',
        tool_call_id: 'call_sql',
        content: 'sensitive row output',
      },
      {
        role: 'tool',
        tool_call_id: 'call_bash',
        content: 'public build log',
      },
    ];
    const span = createSpan('gpt-4o', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'generation',
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify(messages),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql']),
      })
    );

    const redacted = readJsonAttribute<RedactedMessage[]>(
      span,
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    );
    expect(redacted[0].tool_calls?.[0]?.args?.query).toBe(
      'select * from private_table'
    );
    expect(redacted[1].content).toBe(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(redacted[2].content).toBe('public build log');
  });

  it('does not redact partial tool name matches by default', () => {
    const span = createSpan('clickhouse_execute_sql_prod', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'tool',
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: 'secret rows',
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql']),
      })
    );

    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]).toBe(
      'secret rows'
    );
  });

  it('redacts configured partial tool name matches when enabled', () => {
    const span = createSpan('clickhouse_execute_sql_prod', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'tool',
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: 'secret rows',
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql']),
        redactedToolNameMatchMode: 'partial',
      })
    );

    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
  });

  it('redacts prior tool outputs from multi-turn generation inputs', () => {
    const messages = [
      { role: 'user', content: 'run the query' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          {
            id: 'call_sql',
            name: 'execute_sql',
            args: { query: 'select * from private_table' },
          },
        ],
      },
      {
        role: 'execute_sql',
        content: 'sensitive row output',
        additional_kwargs: {},
      },
      { role: 'assistant', content: 'I found the answer.' },
      { role: 'user', content: 'explain the first row' },
    ];
    const span = createSpan('gpt-4o', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'generation',
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify(messages),
    });

    redactLangfuseSpanToolOutputs(span, createConfig({ enabled: false }));

    const redacted = readJsonAttribute<RedactedMessage[]>(
      span,
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    );
    expect(redacted[0].content).toBe('run the query');
    expect(redacted[1].tool_calls?.[0]?.args?.query).toBe(
      'select * from private_table'
    );
    expect(redacted[2].content).toBe(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(redacted[3].content).toBe('I found the answer.');
    expect(redacted[4].content).toBe('explain the first row');
  });

  it('redacts tool outputs after formatAgentMessages rehydrates content parts', () => {
    const payload: TPayload = [
      { role: 'user', content: 'show me the private numbers' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I will query ClickHouse.',
            tool_call_ids: ['call_sql'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'call_sql',
              name: 'execute_sql',
              args: '{"query":"select secret_value from prod"}',
              output: 'secret_value: 12345',
            },
          },
        ],
      },
      { role: 'user', content: 'can you summarize it?' },
    ];
    const { messages } = formatAgentMessages(
      payload,
      undefined,
      new Set(['execute_sql'])
    );
    const serialized = messages.map(serializeMessageForLangfuse);
    const span = createSpan('gpt-4o', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'generation',
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]:
        JSON.stringify(serialized),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql']),
      })
    );

    const redacted = readJsonAttribute<RedactedMessage[]>(
      span,
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    );
    expect(redacted[1].tool_calls?.[0]?.args?.query).toBe(
      'select secret_value from prod'
    );
    expect(redacted[2].role).toBe('execute_sql');
    expect(redacted[2].content).toBe(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(JSON.stringify(redacted)).not.toContain('secret_value: 12345');
  });

  it('redacts constructor-serialized ToolMessages from rehydrated content parts', () => {
    const payload: TPayload = [
      { role: 'user', content: 'show the stored result' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I will query ClickHouse.',
            tool_call_ids: ['call_sql'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'call_sql',
              name: 'execute_sql',
              args: '{"query":"select constructor_path from prod"}',
              output: 'constructor path secret',
            },
          },
        ],
      },
    ];
    const { messages } = formatAgentMessages(
      payload,
      undefined,
      new Set(['execute_sql'])
    );
    const span = createSpan('gpt-4o', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'generation',
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify([
        messages,
      ]),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql']),
      })
    );

    const redacted = span.attributes[
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    ] as string;
    expect(redacted).toContain('select constructor_path from prod');
    expect(redacted).toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(redacted).not.toContain('constructor path secret');
  });

  it('redacts ToolMessage artifacts because they are tool output', () => {
    const messages = [
      {
        id: ['langchain_core', 'messages', 'ToolMessage'],
        kwargs: {
          name: 'execute_sql',
          tool_call_id: 'call_sql',
          content: 'safe display content',
          artifact: {
            rows: ['artifact secret row'],
          },
        },
      },
    ];
    const span = createSpan('gpt-4o', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'generation',
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify(messages),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql']),
      })
    );

    const redacted = readJsonAttribute<
      Array<{
        kwargs: {
          artifact: string;
          content: string;
        };
      }>
    >(span, LangfuseOtelSpanAttributes.OBSERVATION_INPUT);
    expect(redacted[0].kwargs.content).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
    expect(redacted[0].kwargs.artifact).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
    expect(JSON.stringify(redacted)).not.toContain('artifact secret row');
  });

  it('promotes trusted artifact fields before redacting a tool observation', () => {
    const output = new ToolMessage({
      name: 'run_select_query_mcp_ClickHouse',
      tool_call_id: 'call_query',
      content: 'customer row data',
      artifact: {
        [LANGFUSE_OBSERVATION_METADATA_ARTIFACT_KEY]: {
          query_database_system: 'clickhouse',
          query_status: 'success',
          query_returned_rows: 3,
          query_rows_read: 100,
          query_bytes_read: 2048,
        },
      },
    });
    const span = createSpan('run_select_query_mcp_ClickHouse', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'tool',
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: JSON.stringify(output),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['run_select_query']),
        redactedToolNameMatchMode: 'partial',
      })
    );

    expect(span.attributes).toMatchObject({
      [`${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.query_database_system`]:
        'clickhouse',
      [`${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.query_status`]:
        'success',
      [`${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.query_returned_rows`]: 3,
      [`${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.query_rows_read`]: 100,
      [`${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.query_bytes_read`]: 2048,
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]:
        LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
    });
    expect(JSON.stringify(span.attributes)).not.toContain('customer row data');
  });

  it('preserves callback identity when artifact metadata uses the same keys', () => {
    const metadataPrefix = LangfuseOtelSpanAttributes.OBSERVATION_METADATA;
    const span = createSpan('run_select_query', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'tool',
      [`${metadataPrefix}.agentId`]: 'agent-original',
      [`${metadataPrefix}.messageId`]: 'message-original',
      [`${metadataPrefix}.parentMessageId`]: 'parent-original',
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: JSON.stringify(
        new ToolMessage({
          tool_call_id: 'query-call',
          content: 'private row',
          artifact: {
            [LANGFUSE_OBSERVATION_METADATA_ARTIFACT_KEY]: {
              agentId: 'agent-replacement',
              messageId: 'message-replacement',
              parentMessageId: 'parent-replacement',
              query_status: 'success',
            },
          },
        })
      ),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({ redactedToolNames: new Set(['run_select_query']) })
    );

    expect(span.attributes).toMatchObject({
      [`${metadataPrefix}.agentId`]: 'agent-original',
      [`${metadataPrefix}.messageId`]: 'message-original',
      [`${metadataPrefix}.parentMessageId`]: 'parent-original',
      [`${metadataPrefix}.query_status`]: 'success',
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]:
        LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
    });
    expect(JSON.stringify(span.attributes)).not.toContain('private row');
  });

  it('does not promote metadata from an unredacted tool', () => {
    const output = new ToolMessage({
      name: 'untrusted_tool',
      tool_call_id: 'call_untrusted',
      content: 'output',
      artifact: {
        [LANGFUSE_OBSERVATION_METADATA_ARTIFACT_KEY]: {
          query_status: 'forged',
        },
      },
    });
    const span = createSpan('untrusted_tool', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'tool',
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: JSON.stringify(output),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({ redactedToolNames: new Set(['run_select_query']) })
    );

    expect(
      span.attributes[
        `${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.query_status`
      ]
    ).toBeUndefined();
  });

  it('merges run Langfuse defaults with agent redaction overrides', () => {
    const resolved = resolveLangfuseConfig(
      {
        enabled: true,
        publicKey: 'pk-run',
        secretKey: 'sk-run',
        baseUrl: 'https://langfuse.test',
        metadata: { tenantId: 'tenant-run' },
        librechatTraceAttributes: {
          'librechat.langfuse.destination': 'eu',
          'librechat.langfuse.tenant_export.enabled': true,
        },
        tags: ['tenant:tenant-run', 'shared'],
        deterministicTraceId: true,
        toolNodeTracing: { enabled: true },
        toolOutputTracing: {
          enabled: true,
          redactionText: '[redacted]',
        },
      },
      {
        publicKey: 'pk-agent',
        secretKey: 'sk-agent',
        baseUrl: 'https://langfuse.agent',
        metadata: { agentId: 'agent-1' },
        librechatTraceAttributes: {
          'librechat.langfuse.public_key': 'pk-agent',
        },
        tags: ['shared', 'agent:agent-1'],
        toolOutputTracing: {
          enabled: false,
          redactedToolNames: ['execute_sql'],
        },
      }
    );

    expect(resolved).toMatchObject({
      enabled: true,
      publicKey: 'pk-agent',
      secretKey: 'sk-agent',
      baseUrl: 'https://langfuse.agent',
      metadata: { tenantId: 'tenant-run', agentId: 'agent-1' },
      librechatTraceAttributes: {
        'librechat.langfuse.destination': 'eu',
        'librechat.langfuse.tenant_export.enabled': true,
        'librechat.langfuse.public_key': 'pk-agent',
      },
      tags: ['tenant:tenant-run', 'shared', 'agent:agent-1'],
      deterministicTraceId: true,
      toolNodeTracing: { enabled: true },
      toolOutputTracing: {
        enabled: false,
        redactedToolNames: ['execute_sql'],
        redactionText: '[redacted]',
      },
    });
  });

  it('keeps privacy run-level and fails closed on agent overlays', () => {
    const credentials = {
      publicKey: 'pk-run',
      secretKey: 'sk-run',
      baseUrl: 'https://langfuse.test',
    };
    expect(
      resolveLangfuseConfig(
        {
          ...credentials,
          privacy: { mode: 'metricsOnly', redactionText: '[run]' },
        },
        { privacy: { mode: 'full' } }
      )
    ).toMatchObject({
      privacy: { mode: 'metricsOnly', redactionText: '[run]' },
    });
    // An agent overlay asking for metricsOnly without a run-level policy
    // cannot be enforced on the shared trace, so export is disabled.
    expect(
      resolveLangfuseConfig(credentials, {
        privacy: { mode: 'metricsOnly', redactionText: '[agent]' },
      })
    ).toMatchObject({ enabled: false });
    expect(
      resolveLangfuseConfig(credentials, {
        privacy: { mode: 'metricsOnly', redactionText: '[agent]' },
      })?.privacy
    ).toBeUndefined();
    // Agent-only configs cannot carry privacy either.
    expect(
      resolveLangfuseConfig(undefined, {
        privacy: { mode: 'metricsOnly' },
      })
    ).toMatchObject({ enabled: false });
    expect(
      resolveLangfuseConfig(undefined, { privacy: { mode: 'full' } })?.privacy
    ).toBeUndefined();
  });

  it('merges run and agent custom headers with the agent winning collisions', () => {
    const resolved = resolveLangfuseConfig(
      {
        enabled: true,
        publicKey: 'pk-run',
        secretKey: 'sk-run',
        additionalHeaders: {
          'X-Proxy-Token': 'run-token',
          'X-Shared': 'run',
        },
      },
      {
        additionalHeaders: {
          'X-Shared': 'agent',
          'X-Agent-Only': 'agent',
        },
      }
    );

    expect(resolved).toMatchObject({
      additionalHeaders: {
        'X-Proxy-Token': 'run-token',
        'X-Shared': 'agent',
        'X-Agent-Only': 'agent',
      },
    });
  });

  it('lets an agent header override a differently cased run header', () => {
    const resolved = resolveLangfuseConfig(
      {
        additionalHeaders: {
          'X-Proxy-Token': 'run-token',
          'CF-Access-Client-Id': 'run-client',
        },
      },
      { additionalHeaders: { 'x-proxy-token': 'agent-token' } }
    );

    /** Both casings surviving would make fetch send a combined
     *  "run-token, agent-token" value instead of the agent's. */
    expect(resolved?.additionalHeaders).toEqual({
      'CF-Access-Client-Id': 'run-client',
      'x-proxy-token': 'agent-token',
    });
  });

  it('leaves custom headers untouched when only one side sets them', () => {
    expect(
      resolveLangfuseConfig(
        { additionalHeaders: { 'X-Proxy-Token': 'run-token' } },
        { publicKey: 'pk-agent' }
      )
    ).toMatchObject({
      additionalHeaders: { 'X-Proxy-Token': 'run-token' },
    });
    expect(
      resolveLangfuseConfig({ publicKey: 'pk-run' }, { publicKey: 'pk-agent' })
    ).not.toHaveProperty('additionalHeaders');
  });

  it('inherits deterministic trace ids when tenant config only supplies connection settings', () => {
    const resolved = resolveLangfuseConfig(
      {
        deterministicTraceId: true,
      },
      {
        publicKey: 'pk-tenant',
        secretKey: 'sk-tenant',
        baseUrl: 'https://langfuse.tenant',
      }
    );

    expect(resolved).toMatchObject({
      publicKey: 'pk-tenant',
      secretKey: 'sk-tenant',
      baseUrl: 'https://langfuse.tenant',
      deterministicTraceId: true,
    });
  });

  it('inherits application-level redaction when tenant config does not explicitly opt out', () => {
    process.env.LANGFUSE_REDACT_TOOL_OUTPUTS = 'true';
    process.env.LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT = '[app redacted]';

    const config = resolveToolOutputTracingConfig(
      {
        publicKey: 'pk-tenant',
        secretKey: 'sk-tenant',
        baseUrl: 'https://langfuse.tenant',
        toolOutputTracing: {
          redactionText: '[tenant redacted]',
        },
      },
      undefined
    );

    expect(config).toMatchObject({
      enabled: false,
      redactionText: '[tenant redacted]',
    });
  });

  it('keeps application redacted tool names when tenant adds its own names', () => {
    process.env.LANGFUSE_REDACT_TOOL_OUTPUT_NAMES = 'run_sql';

    const config = resolveToolOutputTracingConfig(
      {
        toolOutputTracing: {
          redactedToolNames: ['execute_sql'],
        },
      },
      {
        toolOutputTracing: {
          redactedToolNames: ['web_search'],
        },
      }
    );

    expect([...config.redactedToolNames].sort()).toEqual([
      'execute_sql',
      'run_sql',
      'web_search',
    ]);
  });

  it('keeps application partial redaction matching when tenant adds exact redaction config', () => {
    process.env.LANGFUSE_REDACT_TOOL_OUTPUT_NAME_MATCH_MODE = 'partial';

    const config = resolveToolOutputTracingConfig(
      {
        toolOutputTracing: {
          redactedToolNames: ['execute'],
        },
      },
      {
        toolOutputTracing: {
          redactedToolNameMatchMode: 'exact',
          redactedToolNames: ['web_search'],
        },
      }
    );

    expect(config.redactedToolNameMatchMode).toBe('partial');

    const span = createSpan('execute_sql', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'tool',
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: 'secret rows',
    });

    redactLangfuseSpanToolOutputs(span, config);

    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
  });

  it('lets tenant explicitly opt out of application-level redact-all outputs', () => {
    process.env.LANGFUSE_REDACT_TOOL_OUTPUTS = 'true';

    const config = resolveToolOutputTracingConfig(
      {
        publicKey: 'pk-tenant',
        secretKey: 'sk-tenant',
        toolOutputTracing: {
          enabled: true,
        },
      },
      undefined
    );

    expect(config.enabled).toBe(true);
    expect(config.redactedToolNames.size).toBe(0);
  });

  it('applies application-level redaction through tenant runtime scope unless tenant opts out', () => {
    ensureOpenTelemetryContextManager();
    process.env.LANGFUSE_REDACT_TOOL_OUTPUTS = 'true';
    process.env.LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT = '[app redacted]';
    let capturedContext: Context | undefined;

    withLangfuseRuntimeScope(
      resolveLangfuseRuntimeScope({
        runLangfuse: {
          publicKey: 'pk-tenant',
          secretKey: 'sk-tenant',
          baseUrl: 'https://langfuse.tenant',
        },
      }),
      () => {
        capturedContext = context.active();
      }
    );

    expect(capturedContext).toBeDefined();
    const config = resolveToolOutputTracingConfigForSpan(capturedContext!);
    expect(config).toMatchObject({
      enabled: false,
      redactionText: '[app redacted]',
    });

    const span = createSpan('execute_sql', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'tool',
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: 'tenant secret rows',
    });

    redactLangfuseSpanToolOutputs(span, config!);

    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]).toBe(
      '[app redacted]'
    );
  });

  it('keeps OTEL context fallback for spans outside callback runtime scope', () => {
    ensureOpenTelemetryContextManager();
    const langfuse = {
      publicKey: 'pk-context',
      secretKey: 'sk-context',
      baseUrl: 'https://langfuse.context',
    };
    let capturedContext: Context | undefined;

    withLangfuseRuntimeScope({ langfuse }, () => {
      capturedContext = context.active();
    });

    expect(capturedContext).toBeDefined();
    expect(resolveLangfuseConfigForSpan(capturedContext!)).toBe(langfuse);
  });

  it('keeps OTEL tool-output fallback for spans outside callback runtime scope', () => {
    ensureOpenTelemetryContextManager();
    let capturedContext: Context | undefined;

    withLangfuseRuntimeScope(
      { toolOutputTracing: createConfig({ enabled: false }) },
      () => {
        capturedContext = context.active();
      }
    );

    expect(capturedContext).toBeDefined();
    expect(
      resolveToolOutputTracingConfigForSpan(capturedContext!)
    ).toMatchObject({
      enabled: false,
    });
  });

  it('honors env-only tool-output redaction in runtime scope', () => {
    ensureOpenTelemetryContextManager();
    process.env.LANGFUSE_REDACT_TOOL_OUTPUTS = 'true';
    let capturedContext: Context | undefined;

    withLangfuseRuntimeScope(resolveLangfuseRuntimeScope({}), () => {
      capturedContext = context.active();
    });

    expect(capturedContext).toBeDefined();
    const config = resolveToolOutputTracingConfigForSpan(capturedContext!);
    expect(config).toMatchObject({
      enabled: false,
      redactedToolNameMatchMode: 'exact',
      redactionText: LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
    });
    expect(config?.redactedToolNames.size).toBe(0);
  });

  it('applies agent tool-output redaction override through runtime scope', () => {
    ensureOpenTelemetryContextManager();
    let capturedContext: Context | undefined;

    withLangfuseRuntimeScope(
      resolveLangfuseRuntimeScope({
        runLangfuse: {
          toolOutputTracing: {
            enabled: true,
            redactionText: '[agent redacted]',
          },
        },
        langfuseOverlay: {
          toolOutputTracing: {
            enabled: false,
          },
        },
      }),
      () => {
        capturedContext = context.active();
      }
    );

    expect(capturedContext).toBeDefined();
    const config = resolveToolOutputTracingConfigForSpan(capturedContext!);
    expect(config).toMatchObject({
      enabled: false,
      redactionText: '[agent redacted]',
    });

    const span = createSpan('execute_sql', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'tool',
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: 'secret rows',
    });

    redactLangfuseSpanToolOutputs(span, config!);

    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]).toBe(
      '[agent redacted]'
    );
  });

  it('prefers ALS runtime tenant config over OTEL fallback config', () => {
    ensureOpenTelemetryContextManager();
    const otelLangfuse = {
      publicKey: 'pk-otel',
      secretKey: 'sk-otel',
      baseUrl: 'https://langfuse.otel',
    };
    const runtimeLangfuse = {
      publicKey: 'pk-runtime',
      secretKey: 'sk-runtime',
      baseUrl: 'https://langfuse.runtime',
    };
    let capturedContext: Context | undefined;

    withLangfuseRuntimeScope({ langfuse: otelLangfuse }, () => {
      capturedContext = context.active();
    });

    runWithLangfuseRuntimeContext({ langfuse: runtimeLangfuse }, () => {
      expect(resolveLangfuseConfigForSpan(capturedContext!)).toBe(
        runtimeLangfuse
      );
    });
  });

  it('prefers ALS runtime tool-output config over OTEL fallback config', () => {
    ensureOpenTelemetryContextManager();
    const runtimeToolOutputTracing = {
      enabled: false,
      redactedToolNames: new Set(['runtime_tool']),
      redactedToolNameMatchMode: 'exact' as const,
      redactionText: '[runtime]',
    };
    let capturedContext: Context | undefined;

    withLangfuseRuntimeScope(
      { toolOutputTracing: createConfig({ enabled: true }) },
      () => {
        capturedContext = context.active();
      }
    );

    runWithLangfuseRuntimeContext(
      { toolOutputTracing: runtimeToolOutputTracing },
      () => {
        expect(resolveToolOutputTracingConfigForSpan(capturedContext!)).toBe(
          runtimeToolOutputTracing
        );
      }
    );
  });
});
