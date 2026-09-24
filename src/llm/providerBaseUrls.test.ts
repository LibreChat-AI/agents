import { afterEach, describe, expect, it } from '@jest/globals';
import { CustomAnthropic } from '@/llm/anthropic';
import { ChatOpenAI } from '@/llm/openai';

const originalOpenAIBaseURL = process.env.OPENAI_BASE_URL;
const originalAnthropicBaseURL = process.env.ANTHROPIC_BASE_URL;

function restoreEnv(
  name: 'OPENAI_BASE_URL' | 'ANTHROPIC_BASE_URL',
  value?: string
): void {
  if (value == null) {
    delete process.env[name];
  } else {
    process.env[name] = value;
  }
}

describe('provider base URLs from the environment', () => {
  afterEach(() => {
    restoreEnv('OPENAI_BASE_URL', originalOpenAIBaseURL);
    restoreEnv('ANTHROPIC_BASE_URL', originalAnthropicBaseURL);
  });

  it('routes OpenAI requests through OPENAI_BASE_URL', async () => {
    process.env.OPENAI_BASE_URL = 'https://openai-proxy.example/v1';
    const requestedURLs: string[] = [];
    const capturingFetch: typeof fetch = async (input) => {
      requestedURLs.push(String(input));
      throw new Error('Intercepted test request');
    };
    const model = new ChatOpenAI({
      model: 'gpt-4o-mini',
      apiKey: 'test-key',
      maxRetries: 0,
      configuration: { fetch: capturingFetch },
    });

    await expect(model.invoke('hello')).rejects.toThrow('Connection error.');
    expect(requestedURLs).toEqual([
      'https://openai-proxy.example/v1/chat/completions',
    ]);
  });

  it('routes Anthropic requests through ANTHROPIC_BASE_URL', async () => {
    process.env.ANTHROPIC_BASE_URL = 'https://anthropic-proxy.example';
    const requestedURLs: string[] = [];
    const capturingFetch: typeof fetch = async (input) => {
      requestedURLs.push(String(input));
      throw new Error('Intercepted test request');
    };
    const model = new CustomAnthropic({
      model: 'claude-haiku-4-5',
      apiKey: 'test-key',
      maxRetries: 0,
      clientOptions: { fetch: capturingFetch },
    });

    await expect(model.invoke('hello')).rejects.toThrow('Connection error.');
    expect(requestedURLs).toEqual([
      'https://anthropic-proxy.example/v1/messages',
    ]);
  });
});
