import type { StructuredChatDecisionModelOptions } from './index';
import { createStructuredChatDecisionModel, booleanQuestion } from './index';
import { AzureChatOpenAI } from '@/llm/openai';

interface CapturedRequest {
  url: string;
  body: {
    temperature?: number;
    top_p?: number;
    logprobs?: boolean;
    top_logprobs?: number;
    safety_identifier?: string;
    prompt_cache_options?: { mode: string; ttl: string };
    messages?: object[];
    input?: object[];
    tools?: object[];
    text?: { format?: object };
    response_format?: object;
  };
}

const modes: StructuredChatDecisionModelOptions['method'][] = [
  'jsonSchema',
  'functionCalling',
];
const answers = { q: { decision: true } };
const azureFields = {
  azureOpenAIApiKey: 'test',
  azureOpenAIApiVersion: '2024-10-21',
  azureOpenAIApiInstanceName: 'test-instance',
  azureOpenAIApiDeploymentName: 'test-deployment',
  model: 'gpt-6-astra',
  maxRetries: 0,
  temperature: 0.7,
  topP: 0.8,
  logprobs: true,
  topLogprobs: 2,
  safety_identifier: 'tenant-safety-id',
  promptCacheExplicit: true,
};

function captureAzureRequests(
  requests: CapturedRequest[]
): typeof globalThis.fetch {
  return async (input, init) => {
    const url = String(input);
    const body = JSON.parse(String(init?.body)) as CapturedRequest['body'];
    requests.push({ url, body });
    const isTool = body.tools != null;
    const responseTool = {
      type: 'function_call',
      id: 'fc_test',
      call_id: 'call_test',
      name: 'DecideQuestions',
      arguments: JSON.stringify({ answers }),
      status: 'completed',
    };
    const responseMessage = {
      type: 'message',
      id: 'msg_test',
      role: 'assistant',
      status: 'completed',
      content: [
        {
          type: 'output_text',
          text: JSON.stringify({ answers }),
          annotations: [],
        },
      ],
    };
    if (url.includes('/responses')) {
      return new Response(
        JSON.stringify({
          id: 'resp_test',
          object: 'response',
          created_at: 0,
          model: azureFields.model,
          status: 'completed',
          error: null,
          incomplete_details: null,
          output: isTool ? [responseTool] : [responseMessage],
          usage: {
            input_tokens: 12,
            input_tokens_details: { cached_tokens: 0 },
            output_tokens: 4,
            output_tokens_details: { reasoning_tokens: 0 },
            total_tokens: 16,
          },
        }),
        { headers: { 'content-type': 'application/json' } }
      );
    }
    const toolCall = {
      id: 'call_test',
      type: 'function',
      function: {
        name: 'DecideQuestions',
        arguments: JSON.stringify({ answers }),
      },
    };
    return new Response(
      JSON.stringify({
        id: 'chatcmpl_test',
        object: 'chat.completion',
        created: 0,
        model: azureFields.model,
        choices: [
          {
            index: 0,
            finish_reason: isTool ? 'tool_calls' : 'stop',
            logprobs: null,
            message: {
              role: 'assistant',
              content: isTool ? null : JSON.stringify({ answers }),
              tool_calls: isTool ? [toolCall] : undefined,
            },
          },
        ],
        usage: { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 },
      }),
      { headers: { 'content-type': 'application/json' } }
    );
  };
}

afterEach(() => jest.restoreAllMocks());

describe.each([
  { route: 'completions', useResponsesApi: false },
  { route: 'responses', useResponsesApi: true },
])(
  'LibreChat Azure $route delegates in decision binding',
  ({ useResponsesApi, route }) => {
    it.each(modes)(
      'retains Astra, safety and cache adaptations through %s binding',
      async (method) => {
        const requests: CapturedRequest[] = [];
        jest
          .spyOn(globalThis, 'fetch')
          .mockImplementation(captureAzureRequests(requests));
        const fields = {
          ...azureFields,
          firstPartyEndpoint: true,
          useResponsesApi,
        };
        const model = new AzureChatOpenAI(fields);
        const decisionModel = createStructuredChatDecisionModel({
          model,
          modelId: fields.model,
          method,
        });
        const result = await decisionModel.decide({
          state: 'Test evidence',
          questions: { q: booleanQuestion('Is this a test?') },
        });
        expect(result.answers.q).toEqual({
          type: 'boolean',
          decision: true,
          probability: null,
        });
        expect(requests).toHaveLength(1);
        const { url, body } = requests[0];
        expect(url).toContain(
          route === 'responses' ? '/responses' : '/chat/completions'
        );
        expect(body).not.toHaveProperty('temperature');
        expect(body).not.toHaveProperty('top_p');
        expect(body).not.toHaveProperty('top_logprobs');
        expect(body).not.toHaveProperty('logprobs');
        expect(body.safety_identifier).toBe('tenant-safety-id');
        expect(body.prompt_cache_options).toEqual({
          mode: 'explicit',
          ttl: '30m',
        });
        const inputPath = useResponsesApi
          ? 'input.0.content.0'
          : 'messages.0.content.0';
        expect(body).toHaveProperty(
          `${inputPath}.prompt_cache_breakpoint.mode`,
          'explicit'
        );
        const toolStrictPath = useResponsesApi
          ? 'tools.0.strict'
          : 'tools.0.function.strict';
        const schemaStrictPath = useResponsesApi
          ? 'text.format.strict'
          : 'response_format.json_schema.strict';
        const strictPath =
          method === 'functionCalling' ? toolStrictPath : schemaStrictPath;
        expect(body).toHaveProperty(strictPath, true);
      }
    );

    it('preserves the opt-out endpoint gate through repeated withConfig clones', async () => {
      const requests: CapturedRequest[] = [];
      jest
        .spyOn(globalThis, 'fetch')
        .mockImplementation(captureAzureRequests(requests));
      const fields = {
        ...azureFields,
        firstPartyEndpoint: false,
        useResponsesApi,
      };
      const model = new AzureChatOpenAI(fields);
      const bound = model.withConfig({}).withConfig({});
      await bound.invoke('Test evidence');
      expect(requests).toHaveLength(1);
      expect(requests[0].body).toMatchObject({
        temperature: 0.7,
        top_p: 0.8,
        safety_identifier: 'tenant-safety-id',
        prompt_cache_options: { mode: 'explicit', ttl: '30m' },
      });
    });
  }
);
