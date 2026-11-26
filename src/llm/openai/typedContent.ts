import type { OpenAIClient } from '@langchain/openai';

type Chunk = OpenAIClient.Chat.Completions.ChatCompletionChunk;
type Completion = OpenAIClient.Chat.Completions.ChatCompletion;
type Choice = Chunk['choices'][number];
type Delta = Choice['delta'] & { reasoning_content?: string };
/** Usage-only chunks carry no choice, Azure content-filter chunks no delta. */
type StreamChoice = Omit<Choice, 'delta'> & { delta?: Delta };
type Message = Completion['choices'][number]['message'] & {
  reasoning_content?: string;
};
type TypedPart = { type?: unknown; text?: unknown; thinking?: unknown };

/**
 * Mistral reasoning models send Chat Completions `content` as a list of typed
 * parts instead of a string: `thinking` parts, each holding a list of `text`
 * parts, carry the reasoning, and `text` parts carry the answer.
 */
function readTypedContent(content: TypedPart[]): {
  text: string;
  reasoning: string;
} {
  let text = '';
  let reasoning = '';
  for (const part of content) {
    if (part.type === 'text' && typeof part.text === 'string') {
      text += part.text;
      continue;
    }
    if (part.type !== 'thinking' || !Array.isArray(part.thinking)) {
      continue;
    }
    for (const thought of part.thinking as TypedPart[]) {
      if (typeof thought.text === 'string') {
        reasoning += thought.text;
      }
    }
  }
  return { text, reasoning };
}

/**
 * Rewrites a streamed chunk whose `delta.content` is a typed list as string
 * content with the reasoning in `reasoning_content`, or returns `undefined`
 * when the content is not a list. The chunk that closes the reasoning also
 * opens the answer or a tool call, so it splits in two: the reasoning first,
 * then the rest of the delta.
 */
export function splitTypedDeltaContent(chunk: Chunk): Chunk[] | undefined {
  const choice = chunk.choices[0] as StreamChoice | undefined;
  const delta = choice?.delta;
  const content: unknown = delta?.content;
  if (choice == null || delta == null || !Array.isArray(content)) {
    return undefined;
  }
  const { text, reasoning } = readTypedContent(content);
  const rest: Delta = { ...delta, content: text };
  const others = chunk.choices.slice(1);
  const continues = text !== '' || (rest.tool_calls?.length ?? 0) > 0;
  if (reasoning === '' || !continues) {
    if (reasoning !== '') {
      rest.reasoning_content = reasoning;
    }
    return [{ ...chunk, choices: [{ ...choice, delta: rest }, ...others] }];
  }
  const thought: Delta = { content: '', reasoning_content: reasoning };
  if (rest.role != null) {
    thought.role = rest.role;
  }
  return [
    {
      ...chunk,
      choices: [
        { ...choice, delta: thought, finish_reason: null, logprobs: null },
      ],
    },
    { ...chunk, choices: [{ ...choice, delta: rest }, ...others] },
  ];
}

function flattenMessage(message: Message | undefined): Message | undefined {
  const content: unknown = message?.content;
  if (message == null || !Array.isArray(content)) {
    return undefined;
  }
  const { text, reasoning } = readTypedContent(content);
  const flat: Message = { ...message, content: text };
  if (reasoning !== '') {
    flat.reasoning_content = reasoning;
  }
  return flat;
}

/** Rewrites the typed `message.content` lists of a non-streamed completion. */
export function flattenTypedMessageContent(completion: Completion): Completion {
  if (!Array.isArray(completion.choices)) {
    return completion;
  }
  let choices: Completion['choices'] | undefined;
  for (let i = 0; i < completion.choices.length; i++) {
    const choice = completion.choices[i];
    const message = flattenMessage(choice.message);
    if (message == null) {
      continue;
    }
    choices ??= [...completion.choices];
    choices[i] = { ...choice, message };
  }
  return choices == null ? completion : { ...completion, choices };
}
