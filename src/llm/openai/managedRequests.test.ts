import OpenAI from 'openai';

import {
  ChatOpenAI,
  addChatCacheBreakpoints,
  addResponseCacheBreakpoints,
  isGpt6AstraModel,
  shouldIncludeEncryptedReasoning,
} from './index';

describe('managed GPT-5.6 request fields', () => {
  it('marks the stable instruction message, not the relocated tail behind it', () => {
    /**
     * The tail is a second system message so it keeps its instruction role.
     * Marking the last one would put the breakpoint behind content that turns
     * over every turn, which is the invalidation the breakpoint exists to
     * avoid.
     */
    const messages = addChatCacheBreakpoints([
      { role: 'system', content: 'Stable instructions.' },
      { role: 'user', content: 'First question.' },
      { role: 'assistant', content: 'First answer.' },
      { role: 'system', content: 'Dynamic tail.' },
      { role: 'user', content: 'Current question.' },
    ]);

    /**
     * The instruction breakpoint is the one under test. The tail also carries
     * one here, from the separate rule that marks the history prefix before
     * the current turn; that marker is not this selection.
     */
    expect(JSON.stringify(messages[0])).toContain('prompt_cache_breakpoint');
  });

  it('places cache breakpoints after instructions and the prior history prefix', () => {
    const messages = addChatCacheBreakpoints([
      { role: 'system', content: 'Stable instructions.' },
      { role: 'user', content: 'First question.' },
      { role: 'assistant', content: 'First answer.' },
      { role: 'user', content: 'Current question.' },
    ]);

    expect(messages[0]).toMatchObject({
      content: [
        {
          type: 'text',
          text: 'Stable instructions.',
          prompt_cache_breakpoint: { mode: 'explicit' },
        },
      ],
    });
    expect(messages[2]).toMatchObject({
      content: [
        {
          type: 'text',
          text: 'First answer.',
          prompt_cache_breakpoint: { mode: 'explicit' },
        },
      ],
    });
    expect(JSON.stringify(messages[1])).not.toContain(
      'prompt_cache_breakpoint'
    );
    expect(JSON.stringify(messages[3])).not.toContain(
      'prompt_cache_breakpoint'
    );
  });

  it('uses supported Responses content blocks for the same stable prefixes', () => {
    const input = [
      {
        type: 'message',
        role: 'developer',
        content: [{ type: 'input_text', text: 'Stable instructions.' }],
      },
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'Prior question.' }],
      },
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'Current question.' }],
      },
    ] as unknown as OpenAI.Responses.ResponseInput;
    const result = addResponseCacheBreakpoints(input);

    expect(result).toMatchObject([
      {
        content: [
          {
            prompt_cache_breakpoint: { mode: 'explicit' },
          },
        ],
      },
      {
        content: [
          {
            prompt_cache_breakpoint: { mode: 'explicit' },
          },
        ],
      },
      {
        content: [{ type: 'input_text', text: 'Current question.' }],
      },
    ]);
  });

  it('does not mark replayed assistant output blocks as breakpoints', () => {
    const input = [
      {
        type: 'message',
        role: 'developer',
        content: [{ type: 'input_text', text: 'Stable instructions.' }],
      },
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'First question.' }],
      },
      {
        type: 'message',
        role: 'assistant',
        content: [{ type: 'output_text', text: 'Prior answer.' }],
      },
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'Current question.' }],
      },
    ] as unknown as OpenAI.Responses.ResponseInput;
    const result = addResponseCacheBreakpoints(input) as unknown as Array<{
      content: Array<Record<string, unknown>>;
    }>;

    // output_text is rejected with a 400 by OpenAI, so it must stay unmarked;
    // the stable prefix falls back to the prior user input message.
    expect(result[2].content[0]).not.toHaveProperty('prompt_cache_breakpoint');
    expect(result[1].content[0]).toHaveProperty('prompt_cache_breakpoint');
    expect(result[0].content[0]).toHaveProperty('prompt_cache_breakpoint');
  });

  it('marks string-content Responses messages by wrapping them in input_text', () => {
    const input = [
      { type: 'message', role: 'system', content: 'Stable system prompt.' },
      { type: 'message', role: 'user', content: 'Prior turn.' },
      { type: 'message', role: 'user', content: 'Current question.' },
    ] as unknown as OpenAI.Responses.ResponseInput;
    const result = addResponseCacheBreakpoints(input) as unknown as Array<{
      content: unknown;
    }>;

    expect(result[0].content).toEqual([
      {
        type: 'input_text',
        text: 'Stable system prompt.',
        prompt_cache_breakpoint: { mode: 'explicit' },
      },
    ]);
  });

  it('does not rewrite assistant string content as input_text', () => {
    const input = [
      { type: 'message', role: 'system', content: 'Stable system prompt.' },
      { type: 'message', role: 'user', content: 'First question.' },
      { type: 'message', role: 'assistant', content: 'Prior answer.' },
      { type: 'message', role: 'user', content: 'Current question.' },
    ] as unknown as OpenAI.Responses.ResponseInput;
    const result = addResponseCacheBreakpoints(input) as unknown as Array<{
      content: unknown;
    }>;

    // input_text under role:assistant is rejected with a 400, so assistant
    // string content must stay a string and the breakpoint falls back to the
    // prior user turn.
    expect(result[2].content).toBe('Prior answer.');
    expect(result[1].content).toEqual([
      {
        type: 'input_text',
        text: 'First question.',
        prompt_cache_breakpoint: { mode: 'explicit' },
      },
    ]);
  });

  it('requests encrypted reasoning whenever persisted or stateless replay may be needed', () => {
    expect(shouldIncludeEncryptedReasoning('gpt-5.6', {})).toBe(true);
    expect(
      shouldIncludeEncryptedReasoning('gpt-5.6', {
        reasoning: { context: 'all_turns' },
      })
    ).toBe(true);
    expect(
      shouldIncludeEncryptedReasoning('gpt-5.6', {
        reasoning: { context: 'current_turn' },
      })
    ).toBe(false);
    expect(
      shouldIncludeEncryptedReasoning('gpt-5.6', {
        store: false,
        reasoning: { context: 'current_turn' },
      })
    ).toBe(true);
    expect(shouldIncludeEncryptedReasoning('gpt-5.5', {})).toBe(false);
  });

  it('requests encrypted reasoning for GPT-6 Astra on a declared endpoint', () => {
    expect(shouldIncludeEncryptedReasoning('gpt-6-astra', {}, true)).toBe(true);
    expect(
      shouldIncludeEncryptedReasoning(
        'gpt-6-astra',
        { reasoning: { context: 'current_turn' } },
        true
      )
    ).toBe(false);
  });

  it('does not request encrypted reasoning for an undeclared Astra endpoint', () => {
    expect(shouldIncludeEncryptedReasoning('gpt-6-astra', {}, false)).toBe(
      false
    );
  });
});

describe('GPT-6 Astra model detection', () => {
  it('matches the documented id and its snapshots', () => {
    expect(isGpt6AstraModel('gpt-6-astra')).toBe(true);
    expect(isGpt6AstraModel('gpt-6-astra-2026-04-30')).toBe(true);
    expect(isGpt6AstraModel('GPT-6-Astra')).toBe(true);
  });

  it('does not widen to the rest of the gpt-6 family or near-miss ids', () => {
    expect(isGpt6AstraModel('gpt-6')).toBe(false);
    expect(isGpt6AstraModel('gpt-6-mini')).toBe(false);
    expect(isGpt6AstraModel('gpt-6-astral')).toBe(false);
    expect(isGpt6AstraModel('gpt-5.6')).toBe(false);
    expect(isGpt6AstraModel('not-gpt-6-astra')).toBe(false);
    expect(isGpt6AstraModel(undefined)).toBe(false);
    expect(isGpt6AstraModel('')).toBe(false);
  });

  /**
   * A `provider/` prefix means a proxy owns the request contract. `ChatOpenRouter`
   * extends `ChatOpenAI`, so matching a prefixed id would apply OpenAI's rules to
   * OpenRouter — where `effort: 'none'` is a supported value meaning "disable
   * reasoning", not an error to substitute away.
   */
  it('does not match proxy-routed ids', () => {
    expect(isGpt6AstraModel('openai/gpt-6-astra')).toBe(false);
    expect(isGpt6AstraModel('openrouter/openai/gpt-6-astra')).toBe(false);
  });
});

describe('GPT-6 Astra request constraints', () => {
  /**
   * The endpoint gate falls back to `OPENAI_BASE_URL` when no `baseURL` is
   * configured, so a developer or CI shell pointing at a compatibility gateway
   * would otherwise turn every gate off and fail these tests for a reason that
   * has nothing to do with the implementation. Isolated the same way the
   * sequential-tool-call suite does.
   */
  const ISOLATED_ENV_VARS = ['OPENAI_BASE_URL'];
  const originalEnv = new Map(
    ISOLATED_ENV_VARS.map((name) => [name, process.env[name]])
  );

  beforeEach(() => {
    for (const name of ISOLATED_ENV_VARS) {
      delete process.env[name];
    }
  });

  afterAll(() => {
    for (const [name, value] of originalEnv) {
      if (value == null) {
        delete process.env[name];
      } else {
        process.env[name] = value;
      }
    }
  });

  type AstraFields = ConstructorParameters<typeof ChatOpenAI>[0];

  /**
   * The request fields these tests assert on, across both APIs. Named rather
   * than probed through `Record<string, unknown>` so a renamed or mistyped
   * field is a compile error instead of a silently absent expectation.
   */
  type ProbedRequestParams = {
    temperature?: number | null;
    top_p?: number | null;
    logprobs?: boolean | null;
    top_logprobs?: number | null;
    max_output_tokens?: number | null;
    reasoning_effort?: string;
    reasoning?: { effort?: string };
    include?: OpenAI.Responses.ResponseIncludable[];
  };

  const astra = (fields: AstraFields = {}) =>
    new ChatOpenAI({
      model: 'gpt-6-astra',
      apiKey: 'test-key',
      firstPartyEndpoint: true,
      ...fields,
    });

  const tool = {
    type: 'function' as const,
    function: {
      name: 'get_weather',
      description: 'Get the weather',
      parameters: { type: 'object', properties: {} },
    },
  };

  it('strips sampling parameters the model rejects', () => {
    const model = astra({ temperature: 0.7, topP: 0.9, topLogprobs: 3 });
    for (const options of [{}, { tools: [tool] }]) {
      const params = model.invocationParams(options) as ProbedRequestParams;
      expect(params).not.toHaveProperty('temperature');
      expect(params).not.toHaveProperty('top_p');
      expect(params).not.toHaveProperty('top_logprobs');
    }
  });

  it('strips logprobs on Chat Completions', () => {
    const params = astra({ logprobs: true }).invocationParams(
      {}
    ) as ProbedRequestParams;
    expect(params).not.toHaveProperty('logprobs');
  });

  it('drops the rejected logprobs include on Responses, keeping the rest', () => {
    const model = astra({ useResponsesApi: true });
    const params = model.invocationParams({
      include: ['message.output_text.logprobs', 'reasoning.encrypted_content'],
    } as never) as ProbedRequestParams;
    expect(params.include).toEqual(
      expect.not.arrayContaining(['message.output_text.logprobs'])
    );
    expect(params.include).toEqual(
      expect.arrayContaining(['reasoning.encrypted_content'])
    );
  });

  /**
   * The two paths carry effort under different keys: Chat Completions emits the
   * scalar `reasoning_effort`, Responses the nested `reasoning.effort`. Which
   * API serves the turn is the caller's choice, so each shape is selected
   * explicitly here rather than inferred from the presence of tools.
   */
  const completionsEffort = (effort: string): string | undefined =>
    (
      astra().invocationParams({ reasoningEffort: effort } as never) as {
        reasoning_effort?: string;
      }
    ).reasoning_effort;

  const responsesEffort = (effort: string): string | undefined =>
    (
      astra({ useResponsesApi: true }).invocationParams({
        reasoningEffort: effort,
      } as never) as ProbedRequestParams
    ).reasoning?.effort;

  it('substitutes the reasoning efforts the model rejects on both APIs', () => {
    for (const effort of ['none', 'minimal'] as const) {
      expect(completionsEffort(effort)).toBe('low');
      expect(responsesEffort(effort)).toBe('low');
    }
  });

  it('passes supported reasoning efforts through unchanged on both APIs', () => {
    for (const effort of ['low', 'medium', 'high', 'xhigh', 'max'] as const) {
      expect(completionsEffort(effort)).toBe(effort);
      expect(responsesEffort(effort)).toBe(effort);
    }
  });

  it('leaves rejected efforts alone on other models', () => {
    const model = new ChatOpenAI({ model: 'gpt-5.6', apiKey: 'test-key' });
    const params = model.invocationParams({
      reasoningEffort: 'none',
    } as never) as ProbedRequestParams;
    expect(params.reasoning_effort).toBe('none');
  });

  it('leaves a proxy-routed Astra id alone on every gate', () => {
    const proxied = new ChatOpenAI({
      model: 'openai/gpt-6-astra',
      apiKey: 'test-key',
      temperature: 0.7,
    });
    const params = proxied.invocationParams({
      tools: [tool],
      reasoningEffort: 'none',
    } as never) as ProbedRequestParams;
    /** Chat Completions path retained, sampling kept, `none` not substituted. */
    expect('max_output_tokens' in params).toBe(false);
    expect(params.temperature).toBe(0.7);
    expect(params.reasoning_effort).toBe('none');
  });

  it('shapes nothing when the caller has not declared a first-party endpoint', () => {
    const undeclared = new ChatOpenAI({
      model: 'gpt-6-astra',
      apiKey: 'test-key',
      temperature: 0.7,
    });
    const params = undeclared.invocationParams({
      reasoningEffort: 'none',
    } as never) as ProbedRequestParams;
    /** Default off: sampling kept, `none` not rewritten. */
    expect(params.temperature).toBe(0.7);
    expect(params.reasoning_effort).toBe('none');
  });

  it('shapes the request once the endpoint is declared', () => {
    const params = astra({ temperature: 0.7 }).invocationParams({
      reasoningEffort: 'none',
    } as never) as ProbedRequestParams;
    expect(params).not.toHaveProperty('temperature');
    expect(params.reasoning_effort).toBe('low');
  });

  it('declaring the endpoint does not widen the gates to another model', () => {
    const other = new ChatOpenAI({
      model: 'gpt-5.6',
      apiKey: 'test-key',
      temperature: 0.7,
      firstPartyEndpoint: true,
    });
    const params = other.invocationParams({
      reasoningEffort: 'none',
    } as never) as ProbedRequestParams;
    expect(params.temperature).toBe(0.7);
    expect(params.reasoning_effort).toBe('none');
  });

  it('leaves other models untouched', () => {
    const model = new ChatOpenAI({
      model: 'gpt-5.6',
      apiKey: 'test-key',
      temperature: 0.7,
    });
    const params = model.invocationParams({}) as ProbedRequestParams;
    expect(params.temperature).toBe(0.7);
  });
});
