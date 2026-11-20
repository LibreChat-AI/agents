import { LangfuseOtelSpanAttributes } from '@langfuse/tracing';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import {
  ACTIVITY_LABEL_RUN_NAME,
  REASONING_LABEL_RUN_NAME,
  ACTIVITY_PHASE_LABEL_RUN_NAME,
} from '@/common';
import {
  shapeLangfuseSpan,
  shouldDropLangfuseSpan,
} from '@/langfuseTraceShaping';
import { LANGFUSE_OPERATION_METADATA_KEY } from '@/langfuseOperation';

type TestSpan = ReadableSpan & {
  name: string;
  attributes: Record<string, unknown>;
};

function createSpan(
  name: string,
  attributes: Record<string, unknown> = {},
  parentSpanId?: string
): TestSpan {
  return {
    name,
    attributes,
    ...(parentSpanId != null ? { parentSpanId } : {}),
  } as unknown as TestSpan;
}

const INPUT = LangfuseOtelSpanAttributes.OBSERVATION_INPUT;
const OUTPUT = LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT;
const DEPRECATED_TRACE_INPUT = 'langfuse.trace.input';
const DEPRECATED_TRACE_OUTPUT = 'langfuse.trace.output';
const OBSERVATION_TYPE = LangfuseOtelSpanAttributes.OBSERVATION_TYPE;
const TRACE_TAGS = LangfuseOtelSpanAttributes.TRACE_TAGS;
const METADATA_LANGGRAPH_NODE = `${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.langgraph_node`;
const METADATA_OPERATION = `${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.${LANGFUSE_OPERATION_METADATA_KEY}`;
const METADATA_COMPACTION_SEMANTIC_INDEX_ENTRIES = `${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.compaction_semantic_index_entries`;

/** The outer workflow node: a non-root LangGraph node span whose
 *  `langgraph_node` metadata equals its name. */
function createWorkflowNodeSpan(name: string): TestSpan {
  return createSpan(name, { [METADATA_LANGGRAPH_NODE]: name }, 'parent-1');
}

describe('shouldDropLangfuseSpan', () => {
  it('drops langgraph __start__ seed spans', () => {
    expect(shouldDropLangfuseSpan('__start__')).toBe(true);
  });

  it('drops anonymous RunnableLambda pass-throughs', () => {
    expect(shouldDropLangfuseSpan('RunnableLambda')).toBe(true);
  });

  it('keeps named observations', () => {
    expect(shouldDropLangfuseSpan('GenerateTitle')).toBe(false);
    expect(shouldDropLangfuseSpan('agent=openAI__gpt-5.4')).toBe(false);
    expect(shouldDropLangfuseSpan('ChatOpenAI')).toBe(false);
    expect(shouldDropLangfuseSpan('tool_batch')).toBe(false);
  });
});

describe('shapeLangfuseSpan', () => {
  it('strips the ephemeral agent id (provider__model) from agent node names', () => {
    const span = createSpan('agent=openAI__gpt-5.4', {}, 'parent-1');
    shapeLangfuseSpan(span);
    expect(span.name).toBe('agent');
    expect(span.attributes[OBSERVATION_TYPE]).toBe('agent');
  });

  it('names graph observations for the SDK runtime pattern', () => {
    const standard = createSpan(
      'LangGraph',
      { [METADATA_LANGGRAPH_NODE]: 'agent_primary' },
      'parent-1'
    );
    const multiAgent = createSpan('MultiAgentGraph', {
      [TRACE_TAGS]: JSON.stringify(['librechat', 'agent']),
    });

    shapeLangfuseSpan(standard);
    shapeLangfuseSpan(multiAgent);

    expect(standard.name).toBe('AgentGraph');
    expect(standard.attributes[OBSERVATION_TYPE]).toBe('agent');
    expect(multiAgent.name).toBe('MultiAgentGraph');
    expect(multiAgent.attributes[OBSERVATION_TYPE]).toBe('agent');
  });

  it('types agent-tagged graph roots nested beneath managed hosts', () => {
    const standard = createSpan(
      'AgentGraph',
      {
        [TRACE_TAGS]: JSON.stringify(['librechat', 'agent']),
        [INPUT]: JSON.stringify({
          messages: [{ type: 'human', content: 'Inspect the nested run' }],
        }),
        [OUTPUT]: JSON.stringify({
          messages: [{ type: 'ai', content: 'Nested run complete' }],
        }),
      },
      'managed-host-span'
    );

    shapeLangfuseSpan(standard);

    expect(standard.name).toBe('AgentGraph');
    expect(standard.attributes[OBSERVATION_TYPE]).toBe('agent');
    expect(standard.attributes[INPUT]).toBe('Inspect the nested run');
    expect(standard.attributes[OUTPUT]).toBe('Nested run complete');
    expect(standard.attributes[DEPRECATED_TRACE_INPUT]).toBeUndefined();
    expect(standard.attributes[DEPRECATED_TRACE_OUTPUT]).toBeUndefined();
  });

  it('names the agent prompt-to-model sequence as an SDK operation', () => {
    const span = createSpan(
      'RunnableSequence',
      { [METADATA_LANGGRAPH_NODE]: 'agent=openAI__gpt-5.4' },
      'parent-1'
    );

    shapeLangfuseSpan(span);

    expect(span.name).toBe('AgentModelCall');
    expect(span.attributes[OBSERVATION_TYPE]).toBe('chain');
  });

  it('reads metadata values that the Langfuse SDK stores as JSON', () => {
    const sequence = createSpan(
      'RunnableSequence',
      { [METADATA_LANGGRAPH_NODE]: JSON.stringify('agent=openAI__gpt-5.4') },
      'parent-1'
    );
    const reasoning = createSpan(
      'ChatOpenAI',
      {
        [OBSERVATION_TYPE]: 'generation',
        [TRACE_TAGS]: JSON.stringify(['librechat', 'reasoning-label']),
        [METADATA_OPERATION]: JSON.stringify(REASONING_LABEL_RUN_NAME),
      },
      'parent-1'
    );

    shapeLangfuseSpan(sequence);
    shapeLangfuseSpan(reasoning);

    expect(sequence.name).toBe('AgentModelCall');
    expect(reasoning.name).toBe('ReasoningLabel');
  });

  it('does not rename unrelated runnable sequences', () => {
    const span = createSpan('RunnableSequence', {}, 'parent-1');

    shapeLangfuseSpan(span);

    expect(span.name).toBe('RunnableSequence');
    expect(span.attributes[OBSERVATION_TYPE]).toBeUndefined();
  });

  it('shapes tool nodes as stable dispatch chains with scoped call inputs', () => {
    const messages = [
      { type: 'human', content: 'hello' },
      {
        type: 'ai',
        content: '',
        tool_calls: [
          {
            name: 'get_service_details',
            args: { path: 'organizations/1' },
            id: 'call_1',
          },
        ],
      },
    ];
    const span = createSpan(
      'tools=openAI__gpt-5.4',
      { [INPUT]: JSON.stringify({ messages }) },
      'parent-1'
    );
    shapeLangfuseSpan(span);
    expect(span.name).toBe('tool-dispatch');
    expect(span.attributes[OBSERVATION_TYPE]).toBe('chain');
    expect(JSON.parse(span.attributes[INPUT] as string)).toEqual([
      { name: 'get_service_details', args: { path: 'organizations/1' } },
    ]);
  });

  it('preserves every pending call in a multi-tool dispatch input', () => {
    const messages = [
      {
        type: 'ai',
        tool_calls: [
          { name: 'web_search', args: { q: 'a' }, id: '1' },
          { name: 'web_search', args: { q: 'b' }, id: '2' },
          { name: 'execute_code', args: { code: '1+1' }, id: '3' },
        ],
      },
    ];
    const span = createSpan(
      'tools=openAI__gpt-5.4',
      { [INPUT]: JSON.stringify({ messages }) },
      'parent-1'
    );
    shapeLangfuseSpan(span);
    expect(JSON.parse(span.attributes[INPUT] as string)).toEqual([
      { name: 'web_search', args: { q: 'a' } },
      { name: 'web_search', args: { q: 'b' } },
      { name: 'execute_code', args: { code: '1+1' } },
    ]);
  });

  it('reads tool calls from serialized langchain message kwargs', () => {
    const messages = [
      {
        lc: 1,
        type: 'constructor',
        id: ['langchain_core', 'messages', 'AIMessage'],
        kwargs: {
          content: '',
          tool_calls: [{ name: 'lookup', args: { id: 7 }, id: 'call_7' }],
        },
      },
    ];
    const span = createSpan(
      'tools=agent_abc',
      { [INPUT]: JSON.stringify({ messages }) },
      'parent-1'
    );
    shapeLangfuseSpan(span);
    expect(span.name).toBe('tool-dispatch');
  });

  it('counts id-bearing invalid_tool_calls in the dispatch input (mixed and invalid-only)', () => {
    /** ToolNode pairs attributable invalid calls with synthesized error
     *  results (and routes invalid-only turns on them alone), so the span
     *  input must include them — invalid-only used to find zero calls and
     *  keep the full serialized graph state as the input. */
    const mixed = [
      {
        type: 'ai',
        id: 'ai_mixed_span',
        tool_calls: [{ name: 'echo', args: { command: 'hi' }, id: 'tc_ok' }],
        invalid_tool_calls: [
          {
            name: 'echo',
            args: '"raw unparsed',
            id: 'tc_bad',
            error: 'Malformed args.',
            type: 'invalid_tool_call',
          },
          { name: 'echo', args: 'no-id — excluded', error: 'Malformed args.' },
          {
            name: 'echo',
            args: 'empty-id — excluded',
            id: '',
            error: 'Malformed args.',
          },
          {
            name: 'web_search',
            args: 'server-tool — excluded',
            id: 'srvtoolu_xyz',
            error: 'Malformed args.',
          },
          {
            args: 'nameless — included with the unknown fallback',
            id: 'tc_nameless',
            error: 'Malformed args.',
          },
        ],
      },
    ];
    const mixedSpan = createSpan(
      'tools=agent_abc',
      { [INPUT]: JSON.stringify({ messages: mixed }) },
      'parent-1'
    );
    shapeLangfuseSpan(mixedSpan);
    expect(JSON.parse(mixedSpan.attributes[INPUT] as string)).toEqual([
      { name: 'echo', args: { command: 'hi' } },
      { name: 'echo', args: '"raw unparsed' },
      {
        name: 'unknown',
        args: 'nameless — included with the unknown fallback',
      },
    ]);

    const invalidOnly = [
      {
        type: 'ai',
        id: 'ai_invalid_only_span',
        tool_calls: [],
        invalid_tool_calls: [
          {
            name: 'echo',
            args: 'garbage',
            id: 'tc_solo',
            error: 'Malformed args.',
            type: 'invalid_tool_call',
          },
        ],
      },
    ];
    const invalidOnlySpan = createSpan(
      'tools=agent_abc',
      { [INPUT]: JSON.stringify({ messages: invalidOnly }) },
      'parent-1'
    );
    shapeLangfuseSpan(invalidOnlySpan);
    expect(JSON.parse(invalidOnlySpan.attributes[INPUT] as string)).toEqual([
      { name: 'echo', args: 'garbage' },
    ]);
  });

  it('excludes invalid calls when ToolNode would skip them (array state / id-less message)', () => {
    /** Mirrors ToolNode's canPromoteInvalidCalls gate: a bare-array state
     *  returns a plain output list (invalid handling skipped) and an id-less
     *  message cannot take the reducer upsert — the span must not report
     *  those calls as pending work. Valid calls still count. */
    const invalidCall = {
      name: 'echo',
      args: 'garbage',
      id: 'tc_gated',
      error: 'Malformed args.',
      type: 'invalid_tool_call',
    };
    const arrayStateSpan = createSpan(
      'tools=agent_abc',
      {
        [INPUT]: JSON.stringify([
          {
            type: 'ai',
            id: 'ai_array_span',
            tool_calls: [
              { name: 'echo', args: { command: 'hi' }, id: 'tc_ok' },
            ],
            invalid_tool_calls: [invalidCall],
          },
        ]),
      },
      'parent-1'
    );
    shapeLangfuseSpan(arrayStateSpan);
    expect(JSON.parse(arrayStateSpan.attributes[INPUT] as string)).toEqual([
      { name: 'echo', args: { command: 'hi' } },
    ]);

    const idlessSpan = createSpan(
      'tools=agent_abc',
      {
        [INPUT]: JSON.stringify({
          messages: [
            {
              type: 'ai',
              tool_calls: [
                { name: 'echo', args: { command: 'hi' }, id: 'tc_ok' },
              ],
              invalid_tool_calls: [invalidCall],
            },
          ],
        }),
      },
      'parent-1'
    );
    shapeLangfuseSpan(idlessSpan);
    expect(JSON.parse(idlessSpan.attributes[INPUT] as string)).toEqual([
      { name: 'echo', args: { command: 'hi' } },
    ]);
  });

  it('excludes invalid calls already answered by a ToolMessage in the state', () => {
    /** Mirrors ToolNode's !toolMessageIds.has(id) execution filter: an
     *  answered invalid call is not pending work, even when the same turn
     *  still has a pending valid call. */
    const span = createSpan(
      'tools=agent_abc',
      {
        [INPUT]: JSON.stringify({
          messages: [
            {
              type: 'ai',
              id: 'ai_answered_invalid',
              tool_calls: [
                { name: 'echo', args: { command: 'hi' }, id: 'tc_pending' },
              ],
              invalid_tool_calls: [
                {
                  name: 'echo',
                  args: 'garbage',
                  id: 'tc_answered_invalid',
                  error: 'Malformed args.',
                  type: 'invalid_tool_call',
                },
              ],
            },
            {
              type: 'tool',
              tool_call_id: 'tc_answered_invalid',
              content: 'Error: Malformed args.',
            },
          ],
        }),
      },
      'parent-1'
    );
    shapeLangfuseSpan(span);
    expect(JSON.parse(span.attributes[INPUT] as string)).toEqual([
      { name: 'echo', args: { command: 'hi' } },
    ]);
  });

  it('keeps a stable tool-dispatch shape when no tool calls are found', () => {
    const original = JSON.stringify({
      messages: [{ type: 'human', content: 'hi' }],
    });
    const span = createSpan(
      'tools=agent_abc',
      { [INPUT]: original },
      'parent-1'
    );
    shapeLangfuseSpan(span);
    expect(span.name).toBe('tool-dispatch');
    expect(span.attributes[OBSERVATION_TYPE]).toBe('chain');
    expect(span.attributes[INPUT]).toBe(original);
  });

  it('reduces ephemeral workflow-agent node ids to agent observations named by sender', () => {
    const span = createWorkflowNodeSpan(
      'bedrock__claude-sonnet-5___ClickHouse Agent'
    );
    shapeLangfuseSpan(span);
    expect(span.name).toBe('ClickHouse Agent');
    expect(span.attributes[OBSERVATION_TYPE]).toBe('agent');
  });

  it('strips parallel-instance index suffixes from ephemeral agent ids', () => {
    const span = createWorkflowNodeSpan('openAI__gpt-4o___GPT-4o____1');
    shapeLangfuseSpan(span);
    expect(span.name).toBe('GPT-4o');
  });

  it('restores encoded colons in ephemeral agent sender names', () => {
    const span = createWorkflowNodeSpan('openAI__gpt-4o___alias__variant');
    shapeLangfuseSpan(span);
    expect(span.name).toBe('alias:variant');
  });

  it('keeps persisted agent ids and senderless ephemeral ids unchanged', () => {
    const persisted = createWorkflowNodeSpan('agent_okvkCroi6wXM4-7BY4ud1');
    shapeLangfuseSpan(persisted);
    expect(persisted.name).toBe('agent_okvkCroi6wXM4-7BY4ud1');

    const senderless = createWorkflowNodeSpan('openAI__gpt-4o');
    shapeLangfuseSpan(senderless);
    expect(senderless.name).toBe('openAI__gpt-4o');
  });

  it('does not rename tool observations whose names embed triple underscores', () => {
    const span = createSpan(
      'server__toolkit___lookup',
      { [OBSERVATION_TYPE]: 'tool' },
      'parent-1'
    );
    shapeLangfuseSpan(span);
    expect(span.name).toBe('server__toolkit___lookup');
  });

  it('only renames spans carrying matching langgraph node metadata', () => {
    const runName = createSpan('LibreChat Agent: Ops___EU', {}, 'parent-1');
    shapeLangfuseSpan(runName);
    expect(runName.name).toBe('LibreChat Agent: Ops___EU');

    const ordinaryChain = createSpan('pipeline__stage___EU', {}, 'parent-1');
    shapeLangfuseSpan(ordinaryChain);
    expect(ordinaryChain.name).toBe('pipeline__stage___EU');
    expect(ordinaryChain.attributes[OBSERVATION_TYPE]).toBeUndefined();

    const mismatchedNode = createSpan(
      'pipeline__stage___EU',
      { [METADATA_LANGGRAPH_NODE]: 'some-other-node' },
      'parent-1'
    );
    shapeLangfuseSpan(mismatchedNode);
    expect(mismatchedNode.name).toBe('pipeline__stage___EU');
  });

  it('never renames root observations, even with an encoded-id shape', () => {
    const span = createSpan('bedrock__claude-sonnet-5___ClickHouse Agent', {
      [TRACE_TAGS]: JSON.stringify(['librechat', 'agent']),
      [METADATA_LANGGRAPH_NODE]: 'bedrock__claude-sonnet-5___ClickHouse Agent',
    });
    shapeLangfuseSpan(span);
    expect(span.name).toBe('bedrock__claude-sonnet-5___ClickHouse Agent');
  });

  it('sets root-observation input/output without deprecated trace input/output', () => {
    const span = createSpan('LibreChat Agent', {
      [TRACE_TAGS]: JSON.stringify(['librechat', 'agent']),
      [INPUT]: JSON.stringify({
        messages: [
          { type: 'system', content: 'You are helpful.' },
          { type: 'human', content: 'What is ClickHouse?' },
        ],
      }),
      [OUTPUT]: JSON.stringify({
        messages: [
          { type: 'human', content: 'What is ClickHouse?' },
          { type: 'ai', content: 'A columnar OLAP database.' },
        ],
      }),
      [DEPRECATED_TRACE_INPUT]: 'legacy question',
      [DEPRECATED_TRACE_OUTPUT]: 'legacy answer',
    });
    shapeLangfuseSpan(span);
    expect(span.attributes[INPUT]).toBe('What is ClickHouse?');
    expect(span.attributes[OUTPUT]).toBe('A columnar OLAP database.');
    expect(span.attributes[DEPRECATED_TRACE_INPUT]).toBeUndefined();
    expect(span.attributes[DEPRECATED_TRACE_OUTPUT]).toBeUndefined();
  });

  it('extracts answer text from content part arrays', () => {
    const span = createSpan('LibreChat Agent', {
      [TRACE_TAGS]: JSON.stringify(['librechat', 'agent']),
      [INPUT]: JSON.stringify([{ type: 'human', content: 'hi' }]),
      [OUTPUT]: JSON.stringify({
        messages: [
          {
            id: ['langchain_core', 'messages', 'AIMessage'],
            kwargs: {
              content: [
                { type: 'text', text: 'Hello ' },
                { type: 'text', text: 'there.' },
              ],
            },
          },
        ],
      }),
    });
    shapeLangfuseSpan(span);
    expect(span.attributes[INPUT]).toBe('hi');
    expect(span.attributes[OUTPUT]).toBe('Hello there.');
  });

  it('does not rewrite non-root spans with message payloads', () => {
    const original = JSON.stringify({
      messages: [{ type: 'human', content: 'hi' }],
    });
    const span = createSpan('ChatOpenAI', { [INPUT]: original }, 'parent-1');
    shapeLangfuseSpan(span);
    expect(span.attributes[INPUT]).toBe(original);
  });

  it('redacts semantic-index content while retaining compaction trace counts', () => {
    const span = createSpan(
      'ChatOpenAI',
      {
        [OBSERVATION_TYPE]: 'generation',
        [METADATA_COMPACTION_SEMANTIC_INDEX_ENTRIES]: 1,
        [INPUT]: JSON.stringify({
          messages: [
            {
              type: 'human',
              content:
                '<compaction-semantic-index>raw history example</compaction-semantic-index>',
            },
            {
              type: 'human',
              content:
                '<compaction-semantic-index>\n- activity_phase: secret label\n</compaction-semantic-index>\n\nCheckpoint prompt with <compaction-semantic-index>custom example</compaction-semantic-index>',
            },
          ],
        }),
      },
      'parent-1'
    );

    shapeLangfuseSpan(span);

    expect(span.attributes[METADATA_COMPACTION_SEMANTIC_INDEX_ENTRIES]).toBe(1);
    expect(span.attributes[INPUT]).toContain(
      '<compaction-semantic-index>raw history example</compaction-semantic-index>'
    );
    expect(span.attributes[INPUT]).toContain('Checkpoint prompt');
    expect(span.attributes[INPUT]).toContain(
      '<compaction-semantic-index>custom example</compaction-semantic-index>'
    );
    expect(span.attributes[INPUT]).toContain(
      '<compaction-semantic-index redacted=\\"true\\" />'
    );
    expect(span.attributes[INPUT]).not.toContain('secret label');
  });

  it('does not redact a literal index example without positive metadata', () => {
    const original = JSON.stringify({
      messages: [
        {
          type: 'human',
          content:
            '<compaction-semantic-index>example</compaction-semantic-index>',
        },
      ],
    });
    const span = createSpan(
      'ChatOpenAI',
      { [OBSERVATION_TYPE]: 'generation', [INPUT]: original },
      'parent-1'
    );

    shapeLangfuseSpan(span);

    expect(span.attributes[INPUT]).toBe(original);
  });

  it('preserves root attributes when extraction finds nothing', () => {
    const span = createSpan('LibreChat Agent', { [INPUT]: 'plain text' });
    shapeLangfuseSpan(span);
    expect(span.attributes[INPUT]).toBe('plain text');
    expect(span.attributes[DEPRECATED_TRACE_INPUT]).toBeUndefined();
  });

  it('renames generation spans to a provider-agnostic name', () => {
    const span = createSpan(
      'ChatOpenAI',
      { [OBSERVATION_TYPE]: 'generation' },
      'parent-1'
    );
    shapeLangfuseSpan(span);
    expect(span.name).toBe('llm');
  });

  it('marks only agent-tagged root spans as agent observations', () => {
    const span = createSpan('LibreChat Agent', {
      [TRACE_TAGS]: JSON.stringify(['librechat', 'agent']),
      [INPUT]: JSON.stringify({
        messages: [{ type: 'human', content: 'hi' }],
      }),
    });
    shapeLangfuseSpan(span);
    expect(span.attributes[OBSERVATION_TYPE]).toBe('agent');
  });

  it('marks title-tagged root spans as chain observations', () => {
    const span = createSpan('LibreChat Title', {
      [TRACE_TAGS]: JSON.stringify(['librechat', 'title']),
      [INPUT]: 'Conversation text',
      [OUTPUT]: 'Conversation title',
    });

    shapeLangfuseSpan(span);

    expect(span.attributes[OBSERVATION_TYPE]).toBe('chain');
    expect(span.attributes[INPUT]).toBe('Conversation text');
    expect(span.attributes[OUTPUT]).toBe('Conversation title');
  });

  it('marks activity-phase roots as chains even when tagged as agent work', () => {
    const span = createSpan('MultiStepLabel', {
      [TRACE_TAGS]: JSON.stringify([
        'librechat',
        'activity-phase',
        'agent-run-summary',
        'agent',
      ]),
      [INPUT]: 'What changed?',
      [OUTPUT]: 'Reconciled the implementation and verified the fix',
    });

    shapeLangfuseSpan(span);

    expect(span.name).toBe('MultiStepLabel');
    expect(span.attributes[OBSERVATION_TYPE]).toBe('chain');
    expect(span.attributes[INPUT]).toBe('What changed?');
    expect(span.attributes[OUTPUT]).toBe(
      'Reconciled the implementation and verified the fix'
    );
  });

  it('does not treat a user-supplied activity-phase tag as an operation type', () => {
    const span = createSpan('LibreChat Agent', {
      [TRACE_TAGS]: JSON.stringify(['librechat', 'agent', 'activity-phase']),
      [INPUT]: 'Run an ordinary agent',
    });

    shapeLangfuseSpan(span);

    expect(span.attributes[OBSERVATION_TYPE]).toBe('agent');
  });

  it('does not classify untagged root spans as agents', () => {
    const span = createSpan('Custom root', { [INPUT]: 'input' });

    shapeLangfuseSpan(span);

    expect(span.attributes[OBSERVATION_TYPE]).toBeUndefined();
  });

  it('shapes standalone generation roots without replacing their type', () => {
    const span = createSpan('ChatOpenAI', {
      [OBSERVATION_TYPE]: 'generation',
      [TRACE_TAGS]: JSON.stringify(['librechat', 'title']),
      [INPUT]: 'Generate a title',
      [OUTPUT]: 'A useful title',
    });

    shapeLangfuseSpan(span);

    expect(span.name).toBe('llm');
    expect(span.attributes[OBSERVATION_TYPE]).toBe('generation');
    expect(span.attributes[INPUT]).toBe('Generate a title');
    expect(span.attributes[OUTPUT]).toBe('A useful title');
  });

  it('keeps a generation root\'s full observation input/output', () => {
    /** The activity-label path: a bare model.invoke traces the generation
     *  as its own root, so the observation input is the ONLY record of
     *  the system prompt. The exact shape @langfuse/langchain exports. */
    const originalInput = JSON.stringify([
      { role: 'system', content: 'Write a short label describing…' },
      { role: 'user', content: 'Tool calls:\n- bash(ls) → ok\n\nLabel:' },
    ]);
    const originalOutput = JSON.stringify({
      role: 'assistant',
      content: 'Confirmed /mnt/data persists',
    });
    const span = createSpan('LibreChat Activity Label', {
      [OBSERVATION_TYPE]: 'generation',
      [TRACE_TAGS]: JSON.stringify(['librechat', 'activity-label']),
      [METADATA_OPERATION]: ACTIVITY_LABEL_RUN_NAME,
      [INPUT]: originalInput,
      [OUTPUT]: originalOutput,
    });

    shapeLangfuseSpan(span);

    expect(span.name).toBe('StepLabel');
    expect(span.attributes[INPUT]).toBe(originalInput);
    expect(span.attributes[OUTPUT]).toBe(originalOutput);
    expect(span.attributes[DEPRECATED_TRACE_INPUT]).toBeUndefined();
    expect(span.attributes[DEPRECATED_TRACE_OUTPUT]).toBeUndefined();
  });

  it('names reasoning and phase-label generations by their role', () => {
    const reasoning = createSpan(
      'ChatOpenAI',
      {
        [OBSERVATION_TYPE]: 'generation',
        [TRACE_TAGS]: JSON.stringify(['librechat', 'reasoning-label']),
        [METADATA_OPERATION]: REASONING_LABEL_RUN_NAME,
      },
      'parent-1'
    );
    const phase = createSpan(
      'ChatOpenAI',
      {
        [OBSERVATION_TYPE]: 'generation',
        [TRACE_TAGS]: JSON.stringify(['librechat', 'activity-phase']),
        [METADATA_OPERATION]: ACTIVITY_PHASE_LABEL_RUN_NAME,
      },
      'parent-1'
    );

    shapeLangfuseSpan(reasoning);
    shapeLangfuseSpan(phase);

    expect(reasoning.name).toBe('ReasoningLabel');
    expect(phase.name).toBe('MultiStepLabelGeneration');
  });

  it('does not derive generation operation names from public tags alone', () => {
    const tags = ['activity-label', 'reasoning-label', 'activity-phase'];

    for (const tag of tags) {
      const span = createSpan(
        'ChatOpenAI',
        {
          [OBSERVATION_TYPE]: 'generation',
          [TRACE_TAGS]: JSON.stringify(['librechat', 'agent', tag]),
        },
        'parent-1'
      );

      shapeLangfuseSpan(span);

      expect(span.name).toBe('llm');
    }
  });

  it('still reduces observation input on non-generation roots', () => {
    const span = createSpan('LibreChat Agent', {
      [TRACE_TAGS]: JSON.stringify(['librechat', 'agent']),
      [INPUT]: JSON.stringify({
        messages: [
          { type: 'system', content: 'You are helpful.' },
          { type: 'human', content: 'What is ClickHouse?' },
        ],
      }),
    });

    shapeLangfuseSpan(span);

    expect(span.attributes[INPUT]).toBe('What is ClickHouse?');
    expect(span.attributes[DEPRECATED_TRACE_INPUT]).toBeUndefined();
  });
});

describe('shapeLangfuseSpan summarize-only output', () => {
  it('reports the manual summary as the root output when no assistant replied', () => {
    /** A summarize-only run ends with the retained tail (or nothing) in
     *  state and no assistant message; the summary is its answer. */
    const span = createSpan('LibreChat Agent', {
      [TRACE_TAGS]: JSON.stringify(['librechat', 'agent']),
      [INPUT]: JSON.stringify({
        messages: [{ type: 'human', content: 'Compact this conversation' }],
      }),
      [OUTPUT]: JSON.stringify({
        messages: [],
        manualSummary: 'CHECKPOINT: the user asked about ClickHouse.',
      }),
    });
    shapeLangfuseSpan(span);
    expect(span.attributes[OUTPUT]).toBe(
      'CHECKPOINT: the user asked about ClickHouse.'
    );
  });

  it('prefers the summary over an older reply the run retained', () => {
    /** With an explicit recency window the tail stays in state, so the last
     *  assistant message is history, not the compaction's result. */
    const span = createSpan('LibreChat Agent', {
      [TRACE_TAGS]: JSON.stringify(['librechat', 'agent']),
      [INPUT]: JSON.stringify({ messages: [{ type: 'human', content: 'hi' }] }),
      [OUTPUT]: JSON.stringify({
        messages: [{ type: 'ai', content: 'An older retained reply.' }],
        manualSummary: 'CHECKPOINT: the compaction result.',
      }),
    });
    shapeLangfuseSpan(span);
    expect(span.attributes[OUTPUT]).toBe('CHECKPOINT: the compaction result.');
  });

  it('ignores an empty summary channel', () => {
    const span = createSpan('LibreChat Agent', {
      [TRACE_TAGS]: JSON.stringify(['librechat', 'agent']),
      [INPUT]: JSON.stringify({ messages: [{ type: 'human', content: 'hi' }] }),
      [OUTPUT]: JSON.stringify({
        messages: [{ type: 'ai', content: 'A real answer.' }],
        manualSummary: '   ',
      }),
    });
    shapeLangfuseSpan(span);
    expect(span.attributes[OUTPUT]).toBe('A real answer.');
  });
});
