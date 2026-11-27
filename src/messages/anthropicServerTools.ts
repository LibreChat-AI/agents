export const ANTHROPIC_SERVER_TOOL_NAMES_BY_RESULT_TYPE = {
  advisor_tool_result: ['advisor'],
  bash_code_execution_tool_result: ['bash_code_execution'],
  code_execution_tool_result: ['code_execution'],
  text_editor_code_execution_tool_result: ['text_editor_code_execution'],
  tool_search_tool_result: ['tool_search_tool_regex', 'tool_search_tool_bm25'],
  web_fetch_tool_result: ['web_fetch'],
  web_search_tool_result: ['web_search'],
} as const;

type AnthropicServerToolResultType =
  keyof typeof ANTHROPIC_SERVER_TOOL_NAMES_BY_RESULT_TYPE;

export const ANTHROPIC_SERVER_TOOL_RESULT_TYPES = Object.keys(
  ANTHROPIC_SERVER_TOOL_NAMES_BY_RESULT_TYPE
) as AnthropicServerToolResultType[];

export function isAnthropicServerToolResultType(
  type: string | undefined
): type is AnthropicServerToolResultType {
  return (
    type != null && Object.hasOwn(ANTHROPIC_SERVER_TOOL_NAMES_BY_RESULT_TYPE, type)
  );
}
