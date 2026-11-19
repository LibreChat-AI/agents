import {
  FinishReason,
  GoogleGenerativeAIRequestInputError,
  GoogleGenerativeAIResponseError,
} from '@google/generative-ai';
import type {
  Part,
  FunctionCall,
  GenerateContentRequest,
  GenerateContentResponse,
  EnhancedGenerateContentResponse,
} from '@google/generative-ai';

export function formatRequest(
  input: GenerateContentRequest | string | (string | Part)[]
): GenerateContentRequest {
  if (typeof input !== 'string' && !Array.isArray(input)) return input;
  const parts: Part[] = [];
  let functionParts = 0;
  for (const value of typeof input === 'string' ? [input] : input) {
    const part = typeof value === 'string' ? { text: value } : value;
    if ('functionResponse' in part) functionParts++;
    parts.push(part);
  }
  if (parts.length === 0) {
    throw new GoogleGenerativeAIRequestInputError(
      'No content is provided for sending chat message.'
    );
  }
  if (functionParts !== 0 && functionParts !== parts.length) {
    throw new GoogleGenerativeAIRequestInputError(
      'Within a single message, FunctionResponse cannot be mixed with other types of part.'
    );
  }
  return {
    contents: [{ role: functionParts > 0 ? 'function' : 'user', parts }],
  };
}

/** Compatibility helpers promised by the inherited completionWithRetry API. */
export function enhanceResponse(
  response: GenerateContentResponse
): EnhancedGenerateContentResponse {
  function parts(): Part[] {
    const candidate = response.candidates?.[0];
    if (candidate == null && response.promptFeedback != null) {
      throw new GoogleGenerativeAIResponseError(
        'Response was blocked',
        response
      );
    }
    if (
      candidate?.finishReason === FinishReason.SAFETY ||
      candidate?.finishReason === FinishReason.RECITATION ||
      candidate?.finishReason === FinishReason.LANGUAGE
    ) {
      throw new GoogleGenerativeAIResponseError(
        `Candidate was blocked due to ${candidate.finishReason}`,
        response
      );
    }
    const content = candidate?.content;
    return content?.parts ?? [];
  }

  function functionCalls(): FunctionCall[] | undefined {
    const calls: FunctionCall[] = [];
    for (const part of parts()) {
      if (part.functionCall) calls.push(part.functionCall);
    }
    return calls.length > 0 ? calls : undefined;
  }

  return Object.assign(response, {
    text(): string {
      let text = '';
      for (const part of parts()) {
        text += part.text ?? '';
        if (part.executableCode) {
          text += `\n\`\`\`${part.executableCode.language}\n${part.executableCode.code}\n\`\`\`\n`;
        }
        if (part.codeExecutionResult) {
          text += `\n\`\`\`\n${part.codeExecutionResult.output}\n\`\`\`\n`;
        }
      }
      return text;
    },
    functionCall: (): FunctionCall | undefined => functionCalls()?.[0],
    functionCalls,
  });
}
