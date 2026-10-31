import { ChatOpenAI, AzureChatOpenAI } from './index';

const deployment = 'opaque-deployment';

const connection = {
  apiKey: 'offline-test',
  azureOpenAIApiKey: 'offline-test',
  azureOpenAIApiInstanceName: 'offline-instance',
  azureOpenAIApiDeploymentName: deployment,
  azureOpenAIApiVersion: '2024-10-21',
};

describe.each([ChatOpenAI, AzureChatOpenAI])(
  '%p deployment token parameters',
  (Model) => {
    it.each([
      { maxTokens: undefined, expected: 1536 },
      { maxTokens: 2048, expected: 2048 },
      { maxTokens: -1, expected: undefined },
    ])(
      'keeps the logical reasoning model contract for %j',
      ({ maxTokens, expected }) => {
        const modelKwargs = {
          model: deployment,
          max_completion_tokens: 1536,
        };
        const model = new Model({
          ...connection,
          model: 'gpt-6-astra',
          useResponsesApi: false,
          maxTokens,
          modelKwargs,
        });
        const request = JSON.parse(
          JSON.stringify(model.invocationParams())
        ) as {
          model: string;
          max_completion_tokens?: number;
          max_tokens?: number;
        };
        expect(request.model).toBe(deployment);
        expect(request.max_completion_tokens).toBe(expected);
        expect(request).not.toHaveProperty('max_tokens');
        expect(modelKwargs).toEqual({
          model: deployment,
          max_completion_tokens: 1536,
        });
      }
    );

    it('keeps ordinary Chat Completions limits unchanged', () => {
      const model = new Model({
        ...connection,
        model: 'gpt-4.1',
        useResponsesApi: false,
        maxTokens: 1024,
        modelKwargs: { model: deployment },
      });
      expect(model.invocationParams()).toMatchObject({
        model: deployment,
        max_tokens: 1024,
      });
      expect(model.invocationParams()).not.toHaveProperty(
        'max_completion_tokens'
      );
    });

    it('leaves Responses limits on their existing wire field', () => {
      const model = new Model({
        ...connection,
        model: 'gpt-6-astra',
        useResponsesApi: true,
        maxTokens: 1024,
        modelKwargs: { model: deployment },
      });
      expect(model.invocationParams()).toMatchObject({
        model: deployment,
        max_output_tokens: 1024,
      });
      expect(model.invocationParams()).not.toHaveProperty(
        'max_completion_tokens'
      );
    });
  }
);
