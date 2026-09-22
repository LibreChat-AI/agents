import { describe, expect, it, jest } from '@jest/globals';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import {
  assertPreparedProviderRequestFor,
  prepareProviderRequest,
} from '@/llm/prepareProviderRequest';
import {
  INSTRUCTIONLESS_HANDOFF_CUE,
  withInstructionlessHandoffCue,
} from '@/messages/handoffCue';
import {
  getProviderMessageProvenance,
  getProviderSourceMessageIds,
} from '@/messages/provenance';
import { createToolHistoryPreparation } from '@/messages/toolHistoryProjection';
import { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import { _convertMessagesToOpenAIParams } from '@/llm/openai/utils';
import { createContextPressureMeter } from './contextPressureMeter';
import { foldToolBlocksForToollessAgent } from '@/messages/format';
import { attemptInvoke } from '@/llm/invoke';
import { Providers } from '@/common';

type StubModel = {
  model?: string;
  _useResponsesApi?: (options?: unknown) => boolean;
  invoke: (messages: BaseMessage[]) => Promise<AIMessage>;
};

interface CapturingModel {
  model: StubModel;
  invocations: BaseMessage[][];
}

function createCapturingModel(): CapturingModel {
  const invocations: BaseMessage[][] = [];
  return {
    invocations,
    model: {
      model: 'prepared-model',
      invoke: jest.fn(async (messages: BaseMessage[]): Promise<AIMessage> => {
        invocations.push(messages);
        return new AIMessage('ok');
      }),
    },
  };
}

describe('prepareProviderRequest', () => {
  it.each([Providers.OPENAI, Providers.ANTHROPIC, Providers.BEDROCK])(
    'measures the instructionless handoff cue for %s without modifying its source',
    (provider) => {
      const { model } = createCapturingModel();
      const tail = new AIMessage({
        content: 'Transferring',
        id: 'handoff-tail',
      });
      const messages = [new HumanMessage('Analyze this'), tail];
      const config = withInstructionlessHandoffCue(undefined, tail);
      const measured = jest.fn((prepared: BaseMessage[]) => ({
        fits: prepared.length <= messages.length,
      }));
      const request = prepareProviderRequest({
        model: model as t.ChatModel,
        messages,
        provider,
        config,
        measure: measured,
      });
      expect(request.messages.at(-1)?.content).toBe(
        INSTRUCTIONLESS_HANDOFF_CUE
      );
      expect(measured).toHaveBeenCalledWith(request.messages);
      expect(request.measurement?.fits).toBe(false);
      expect(messages).toHaveLength(2);
      expect(tail.additional_kwargs).toEqual({});

      const fallback = prepareProviderRequest({
        model: model as t.ChatModel,
        messages: request.messages,
        provider: Providers.OPENAI,
        config,
      });
      expect(
        fallback.messages.filter(
          (message) => message.content === INSTRUCTIONLESS_HANDOFF_CUE
        )
      ).toHaveLength(1);
      const freshFallback = prepareProviderRequest({
        model: model as t.ChatModel,
        messages,
        provider: Providers.OPENAI,
        config,
      });
      expect(freshFallback.messages.at(-1)?.content).toBe(
        INSTRUCTIONLESS_HANDOFF_CUE
      );
    }
  );

  it('honors an explicit Chat override over Responses model defaults', () => {
    const { model } = createCapturingModel();
    model._useResponsesApi = (options?: unknown): boolean =>
      (options as { configurable?: { apiMode?: string } } | undefined)
        ?.configurable?.apiMode !== 'chat';
    expect(
      prepareProviderRequest({
        model: model as t.ChatModel,
        messages: [new HumanMessage('hello')],
        provider: Providers.OPENAI,
        config: { configurable: { apiMode: 'chat' } },
      }).projectionMode
    ).toBe('chat-messages');
    expect(
      prepareProviderRequest({
        model: model as t.ChatModel,
        messages: [new HumanMessage('hello')],
        provider: Providers.OPENAI,
      }).projectionMode
    ).toBe('openai-responses');
  });
  it('shares source facts without caching the destination serving policy', () => {
    const { model } = createCapturingModel();
    model._useResponsesApi = (): boolean => true;
    const message = new AIMessage({
      content: 'approval needed',
      additional_kwargs: {
        tool_outputs: [
          {
            type: 'mcp_approval_request',
            id: 'approval',
            name: 'lookup',
            arguments: '{}',
          },
        ],
      },
    });
    const messages = [message];
    const history = createToolHistoryPreparation();
    const facts = history.get(message);
    const native = foldToolBlocksForToollessAgent(
      messages,
      undefined,
      true,
      history
    );
    const fallback = foldToolBlocksForToollessAgent(
      messages,
      undefined,
      false,
      history
    );
    expect(native).toBe(messages);
    expect(fallback).not.toBe(messages);
    expect(JSON.stringify(fallback[0].content)).toContain(
      'mcp_approval_request'
    );
    expect(history.get(message)).toBe(facts);
    expect(
      prepareProviderRequest({
        model: model as t.ChatModel,
        messages: native,
        provider: Providers.OPENAI,
        toolHistory: history,
      }).projectionMode
    ).toBe('openai-responses');
    expect(
      prepareProviderRequest({
        model: model as t.ChatModel,
        messages: fallback,
        provider: Providers.ANTHROPIC,
        toolHistory: history,
      }).projectionMode
    ).not.toBe('openai-responses');
  });
  it('preserves original provenance when fallback folds before origin tracking', () => {
    const { model } = createCapturingModel();
    const message = new AIMessage({
      content: 'The answer is 4.',
      additional_kwargs: { sourceMessageId: 'original-assistant-row' },
      response_metadata: {
        output: [
          {
            type: 'code_interpreter_call',
            status: 'completed',
            outputs: [{ type: 'logs', logs: '4' }],
          },
          {
            type: 'message',
            content: [{ type: 'output_text', text: 'The answer is 4.' }],
          },
        ],
      },
    });
    const messages = [message];
    const meter = createContextPressureMeter({
      indexTokenCountMap: {},
      sourceMessages: messages,
      retainedMessages: messages,
      provider: Providers.OPENAI,
      tokenCounter: (entry) => JSON.stringify(entry.content).length,
      instructionTokens: 0,
      calibrationRatio: 1,
      contextUsage: { contextBudget: 10_000, effectiveInstructionTokens: 0 },
    });
    const toolHistory = createToolHistoryPreparation();
    const folded = foldToolBlocksForToollessAgent(
      messages,
      undefined,
      false,
      toolHistory
    );
    const request = prepareProviderRequest({
      model: model as t.ChatModel,
      provider: Providers.ANTHROPIC,
      messages: folded,
      toolHistory,
      measure: (prepared) =>
        meter.measure(meter.trackProjection(folded, prepared), {
          contextBudget: 10_000,
          forceRawRecount: true,
        }),
    });
    expect(getProviderSourceMessageIds(request.messages[0])).toEqual([
      'original-assistant-row',
    ]);
    expect(getProviderMessageProvenance(request.messages[0])?.parts).toEqual(
      expect.arrayContaining([
        { attribution: 'model', sourceMessageId: 'original-assistant-row' },
        { attribution: 'tool', sourceMessageId: 'original-assistant-row' },
      ])
    );
    expect(request.measurement?.toolMessageTokens).toBeGreaterThan(0);
    expect(message.content).toBe('The answer is 4.');
  });
  it.each([
    ['CSV', 'csv'],
    ['XLSX', 'xlsx'],
  ])(
    'recovers an existing Bedrock %s chat after its agent moves to an OpenAI-compatible endpoint',
    (_label, format) => {
      const { model } = createCapturingModel();
      const document = {
        type: 'document',
        document: {
          name: `sales.${format}`,
          format,
          source: {
            bytes: { type: 'Buffer', data: [99, 111, 108, 49, 10] },
          },
        },
      };
      const source = new HumanMessage({
        content: [
          { type: 'text', text: 'Attached document(s):\ncol1\nvalue' },
          document,
        ],
      });

      const bedrockRequest = prepareProviderRequest({
        model: model as t.ChatModel,
        messages: [source],
        provider: Providers.BEDROCK,
      });
      const request = prepareProviderRequest({
        model: model as t.ChatModel,
        messages: [source],
        provider: Providers.OPENAI,
      });

      expect(bedrockRequest.messages[0]).toBe(source);
      expect(bedrockRequest.messages[0].content[1]).toBe(document);
      expect(request.messages[0]).not.toBe(source);
      expect(request.messages[0].content).toEqual([
        { type: 'text', text: 'Attached document(s):\ncol1\nvalue' },
      ]);
      expect(_convertMessagesToOpenAIParams(request.messages)).toEqual([
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Attached document(s):\ncol1\nvalue' },
          ],
        },
      ]);
      expect(source.content).toEqual([
        { type: 'text', text: 'Attached document(s):\ncol1\nvalue' },
        document,
      ]);
    }
  );

  it('reprojects a persisted Bedrock PDF and preserves image content for OpenAI', () => {
    const { model } = createCapturingModel();
    const pdf = {
      type: 'document',
      document: {
        name: 'report.pdf',
        format: 'pdf',
        source: { bytes: { type: 'Buffer', data: [37, 80, 68, 70] } },
      },
    };
    const image = {
      type: 'image_url',
      image_url: { url: 'data:image/png;base64,AA==' },
    };
    const source = new HumanMessage({ content: [pdf, image] });

    const request = prepareProviderRequest({
      model: model as t.ChatModel,
      messages: [source],
      provider: Providers.OPENAI,
    });

    expect(request.messages[0].content).toEqual([
      {
        type: 'file',
        source_type: 'base64',
        mime_type: 'application/pdf',
        data: 'JVBERg==',
        metadata: { name: 'report.pdf' },
      },
      image,
    ]);
    expect(_convertMessagesToOpenAIParams(request.messages)).toEqual([
      {
        role: 'user',
        content: [
          {
            type: 'file',
            file: {
              file_data: 'data:application/pdf;base64,JVBERg==',
              filename: 'report.pdf',
            },
          },
          image,
        ],
      },
    ]);
    expect(source.content).toEqual([pdf, image]);
  });

  it('filters a canonical spreadsheet descriptor at the same final projection seam', () => {
    const { model } = createCapturingModel();
    const source = new HumanMessage({
      content: [
        { type: 'text', text: 'Attached document(s):\nRevenue: 42' },
        {
          type: 'file',
          source_type: 'base64',
          mime_type:
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
          data: 'UEs=',
          metadata: { name: 'sales.xlsx' },
        },
      ],
    });

    const request = prepareProviderRequest({
      model: model as t.ChatModel,
      messages: [source],
      provider: Providers.OPENAI,
    });

    expect(request.messages[0].content).toEqual([
      { type: 'text', text: 'Attached document(s):\nRevenue: 42' },
    ]);
    expect(source.content).toHaveLength(2);
  });

  it('retains canonical Anthropic PDF and image files with parameterized MIME types', () => {
    const { model } = createCapturingModel();
    const supportedFiles = [
      {
        type: 'file',
        source_type: 'base64',
        mime_type: 'application/pdf; charset=binary',
        data: 'JVBERg==',
      },
      {
        type: 'file',
        source_type: 'base64',
        mime_type: 'image/png; name=chart.png',
        data: 'iVBORw==',
      },
    ];
    const source = new HumanMessage({
      content: [
        ...supportedFiles,
        {
          type: 'file',
          source_type: 'base64',
          mime_type: 'application/vnd.ms-excel',
          data: '0M8R4A==',
        },
      ],
    });

    const request = prepareProviderRequest({
      model: model as t.ChatModel,
      messages: [source],
      provider: Providers.ANTHROPIC,
    });

    expect(request.messages[0].content).toEqual(supportedFiles);
    expect(source.content).toHaveLength(3);
  });

  it('does not scan or re-encode Bedrock-native documents for a Bedrock target', () => {
    const { model } = createCapturingModel();
    const document = {
      type: 'document',
      document: {
        name: 'sales.csv',
        format: 'csv',
        source: { bytes: new Uint8Array([99, 111, 108, 49]) },
      },
    };
    const source = new HumanMessage({ content: [document] });

    const request = prepareProviderRequest({
      model: model as t.ChatModel,
      messages: [source],
      provider: Providers.BEDROCK,
    });

    expect(request.messages[0]).toBe(source);
    expect(request.messages[0].content[0]).toBe(document);
  });

  it('keeps an attachment-only turn valid when its binary block is unsupported', () => {
    const { model } = createCapturingModel();
    const source = new HumanMessage({
      content: [
        { type: 'text', text: '' },
        {
          type: 'document',
          document: {
            name: 'sales.xlsx',
            format: 'xlsx',
            source: { bytes: { type: 'Buffer', data: [80, 75] } },
          },
        },
      ],
    });

    const request = prepareProviderRequest({
      model: model as t.ChatModel,
      messages: [source],
      provider: Providers.OPENAI,
    });

    expect(request.messages[0].content).toEqual([
      {
        type: 'text',
        text: '[Attachment omitted because its binary format is unsupported by this provider.]',
      },
    ]);
  });

  it('measures and sends the exact prepared message array without re-projection', async () => {
    const { model, invocations } = createCapturingModel();
    const source = [new HumanMessage('first'), new HumanMessage('second')];
    const measure = jest.fn((messages: BaseMessage[]) => ({
      fits: true,
      projectedMessageTokens: messages.length * 10,
    }));

    const request = prepareProviderRequest({
      model: model as t.ChatModel,
      messages: source,
      provider: Providers.MISTRAL,
      measure,
    });

    expect(Object.isFrozen(request)).toBe(true);
    const [brand] = Object.getOwnPropertySymbols(request);
    expect(Object.getOwnPropertyDescriptor(request, brand)).toMatchObject({
      configurable: false,
      enumerable: false,
      value: true,
      writable: false,
    });
    expect(Object.getOwnPropertySymbols({ ...request })).toHaveLength(0);
    expect(request.modelId).toBe('prepared-model');
    expect(request.messages).toHaveLength(1);
    expect(request.measurement).toEqual({
      fits: true,
      projectedMessageTokens: 10,
    });
    expect(measure).toHaveBeenCalledTimes(1);
    expect(measure).toHaveBeenCalledWith(request.messages);
    expect(source).toHaveLength(2);

    await attemptInvoke({ request });

    expect(invocations).toHaveLength(1);
    expect(invocations[0]).toBe(request.messages);
    expect(measure).toHaveBeenCalledTimes(1);
  });

  it('keeps tool-reference annotation transient and inside the measured request', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('run-1', 'tool0turn0', 'stored');
    const toolMessage = new ToolMessage({
      content: 'tool output',
      tool_call_id: 'call-1',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const { model } = createCapturingModel();
    const measure = jest.fn((messages: BaseMessage[]) => ({
      fits: messages.length > 0,
    }));

    const request = prepareProviderRequest({
      model: model as t.ChatModel,
      messages: [toolMessage],
      provider: Providers.ANTHROPIC,
      context: { getOrCreateToolOutputRegistry: () => registry },
      config: { configurable: { run_id: 'run-1' } },
      measure,
    });

    expect(request.messages[0].content).toBe('[ref: tool0turn0]\ntool output');
    expect(measure).toHaveBeenCalledWith(request.messages);
    expect(toolMessage.content).toBe('tool output');
    expect(toolMessage.additional_kwargs._refKey).toBe('tool0turn0');
  });

  it('includes serving-provider handoff shaping before measurement', () => {
    const { model } = createCapturingModel();
    const predecessor = new AIMessage({
      id: 'previous-agent',
      content: 'done',
    });
    const measure = jest.fn((messages: BaseMessage[]) => ({
      fits: messages.length > 0,
    }));

    const request = prepareProviderRequest({
      model: model as t.ChatModel,
      messages: [predecessor],
      provider: Providers.ANTHROPIC,
      context: { isRunProducedMessage: (message) => message === predecessor },
      measure,
    });

    expect(request.messages).toHaveLength(2);
    expect(request.messages[1].getType()).toBe('human');
    expect(measure).toHaveBeenCalledWith(request.messages);
  });

  it('fails closed when a prepared artifact is used for another model or provider', () => {
    const first = createCapturingModel();
    const second = createCapturingModel();
    const request = prepareProviderRequest({
      model: first.model as t.ChatModel,
      messages: [new HumanMessage('hello')],
      provider: Providers.OPENAI,
    });

    expect(() =>
      assertPreparedProviderRequestFor(
        request,
        second.model as t.ChatModel,
        Providers.OPENAI
      )
    ).toThrow('does not match serving model');
    expect(() =>
      assertPreparedProviderRequestFor(
        request,
        first.model as t.ChatModel,
        Providers.ANTHROPIC
      )
    ).toThrow('does not match serving provider');
  });

  it('rejects when invocation options switch the prepared OpenAI projection mode', async () => {
    const { model, invocations } = createCapturingModel();
    model._useResponsesApi = (options?: unknown): boolean =>
      (options as { configurable?: { apiMode?: string } } | undefined)
        ?.configurable?.apiMode === 'responses';
    const request = prepareProviderRequest({
      model: model as t.ChatModel,
      messages: [new HumanMessage('hello')],
      provider: Providers.OPENAI,
      config: { configurable: { apiMode: 'responses' } },
    });

    expect(request.projectionMode).toBe('openai-responses');
    await expect(
      attemptInvoke(
        { request },
        { configurable: { apiMode: 'chat-completions' } }
      )
    ).rejects.toThrow('does not match invocation options');
    expect(invocations).toHaveLength(0);
  });
});
