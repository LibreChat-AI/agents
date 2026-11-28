import { AIMessageChunk } from '@langchain/core/messages';
import { StandardGraph } from '../Graph';
import { Providers } from '@/common';

function createGraph(): StandardGraph {
  return new StandardGraph({
    runId: 'usage',
    agents: [
      {
        agentId: 'agent',
        provider: Providers.VERTEXAI,
        instructions: 'Answer the question.',
      },
    ],
  });
}

describe('StandardGraph context usage', () => {
  it('uses the initial prompt for long-decode context without changing billing usage', () => {
    const message = new AIMessageChunk({
      content: 'answer',
      usage_metadata: {
        input_tokens: 300,
        output_tokens: 65_538,
        total_tokens: 65_838,
        input_token_details: { cache_read: 250 },
        output_token_details: { reasoning: 65_533 },
      },
      response_metadata: { initial_prompt_token_count: 100 },
    });

    expect(createGraph().getUsageMetadata(message)).toEqual({
      input_tokens: 100,
      output_tokens: 65_538,
      total_tokens: 65_638,
      output_token_details: { reasoning: 65_533 },
    });
    expect(message.usage_metadata).toMatchObject({
      input_tokens: 300,
      output_tokens: 65_538,
      total_tokens: 65_838,
      input_token_details: { cache_read: 250 },
      output_token_details: { reasoning: 65_533 },
    });
  });

  it('preserves provider usage when no initial prompt override is present', () => {
    const message = new AIMessageChunk({
      content: 'answer',
      usage_metadata: {
        input_tokens: 100,
        output_tokens: 5,
        total_tokens: 105,
      },
    });

    expect(createGraph().getUsageMetadata(message)).toBe(
      message.usage_metadata
    );
  });
});
