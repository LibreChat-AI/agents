/**
 * Enum representing the various event types emitted during the execution of runnables.
 * These events provide real-time information about the progress and state of different components.
 *
 * @enum {string}
 */
export enum GraphEvents {
  /* Custom Events */

  /** Accepted graph model result, after fallback selection. Registry-only, not a provider callback. */
  ON_MODEL_RESPONSE = 'on_model_response',

  /** [Custom] Agent update event in multi-agent graph/workflow */
  ON_AGENT_UPDATE = 'on_agent_update',
  /** [Custom] Delta event for run steps (message creation and tool calls) */
  ON_RUN_STEP = 'on_run_step',
  /** [Custom] Delta event for run steps (tool calls) */
  ON_RUN_STEP_DELTA = 'on_run_step_delta',
  /** [Custom] Completed event for run steps (tool calls) */
  ON_RUN_STEP_COMPLETED = 'on_run_step_completed',
  /** [Custom] Terminal signal for a run step: closed with status + timestamps */
  ON_RUN_STEP_CLOSED = 'on_run_step_closed',
  /** [Custom] Delta events for messages */
  ON_MESSAGE_DELTA = 'on_message_delta',
  /** [Custom] Reasoning Delta events for messages */
  ON_REASONING_DELTA = 'on_reasoning_delta',
  /** [Custom] Request to execute tools - dispatched by ToolNode, handled by host */
  ON_TOOL_EXECUTE = 'on_tool_execute',
  /** [Custom] Emitted when the summarize node begins generating a summary */
  ON_SUMMARIZE_START = 'on_summarize_start',
  /** [Custom] Delta event carrying the completed summary content */
  ON_SUMMARIZE_DELTA = 'on_summarize_delta',
  /** [Custom] Emitted when the summarize node completes with the final summary */
  ON_SUMMARIZE_COMPLETE = 'on_summarize_complete',
  /** [Custom] Progress update from a running subagent (wraps child-graph events so hosts can display activity separately from parent). */
  ON_SUBAGENT_UPDATE = 'on_subagent_update',
  /** [Custom] Diagnostic logging event for context management observability */
  ON_AGENT_LOG = 'on_agent_log',
  /** [Custom] Per-model-call context window usage snapshot (post-prune token budget) */
  ON_CONTEXT_USAGE = 'on_context_usage',

  /* Official Events */

  /** Custom event, emitted by system */
  ON_CUSTOM_EVENT = 'on_custom_event',
  /** Emitted when a chat model starts processing. */
  CHAT_MODEL_START = 'on_chat_model_start',

  /** Emitted when a chat model streams a chunk of its response. */
  CHAT_MODEL_STREAM = 'on_chat_model_stream',

  /** Emitted when a chat model completes its processing. */
  CHAT_MODEL_END = 'on_chat_model_end',

  /** Emitted when a language model starts processing. */
  LLM_START = 'on_llm_start',

  /** Emitted when a language model streams a chunk of its response. */
  LLM_STREAM = 'on_llm_stream',

  /** Emitted when a language model completes its processing. */
  LLM_END = 'on_llm_end',

  /** Emitted when a chain starts processing. */
  CHAIN_START = 'on_chain_start',

  /** Emitted when a chain streams a chunk of its output. */
  CHAIN_STREAM = 'on_chain_stream',

  /** Emitted when a chain completes its processing. */
  CHAIN_END = 'on_chain_end',

  /** Emitted when a tool starts its operation. */
  TOOL_START = 'on_tool_start',

  /** Emitted when a tool completes its operation. */
  TOOL_END = 'on_tool_end',

  /** Emitted when a retriever starts its operation. */
  RETRIEVER_START = 'on_retriever_start',

  /** Emitted when a retriever completes its operation. */
  RETRIEVER_END = 'on_retriever_end',

  /** Emitted when a prompt starts processing. */
  PROMPT_START = 'on_prompt_start',

  /** Emitted when a prompt completes its processing. */
  PROMPT_END = 'on_prompt_end',
}

export enum Providers {
  OPENAI = 'openAI',
  VERTEXAI = 'vertexai',
  BEDROCK = 'bedrock',
  ANTHROPIC = 'anthropic',
  MISTRALAI = 'mistralai',
  MISTRAL = 'mistral',
  GOOGLE = 'google',
  AZURE = 'azureOpenAI',
  DEEPSEEK = 'deepseek',
  OPENROUTER = 'openrouter',
  XAI = 'xai',
  MOONSHOT = 'moonshot',
}

export enum GraphNodeKeys {
  TOOLS = 'tools=',
  AGENT = 'agent=',
  SUMMARIZE = 'summarize=',
  ROUTER = 'router',
  PRE_TOOLS = 'pre_tools',
  POST_TOOLS = 'post_tools',
}

export enum GraphNodeActions {
  TOOL_NODE = 'tool_node',
  CALL_MODEL = 'call_model',
  ROUTE_MESSAGE = 'route_message',
}

export enum CommonEvents {
  LANGGRAPH = 'LangGraph',
}

export enum StepTypes {
  TOOL_CALLS = 'tool_calls',
  MESSAGE_CREATION = 'message_creation',
}

export enum ContentTypes {
  TEXT = 'text',
  ERROR = 'error',
  THINK = 'think',
  TOOL_CALL = 'tool_call',
  IMAGE_URL = 'image_url',
  IMAGE_FILE = 'image_file',
  /** Anthropic */
  THINKING = 'thinking',
  /** Vertex AI / Google Common */
  REASONING = 'reasoning',
  /** Multi-Agent Switch */
  AGENT_UPDATE = 'agent_update',
  /** Framework-level conversation summary block */
  SUMMARY = 'summary',
  /** Bedrock */
  REASONING_CONTENT = 'reasoning_content',
  /** Mid-run user steer persisted inline in an assistant message; replayed as a user turn */
  STEER = 'steer',
  /** Fast-model activity label for a tool/reasoning block; UI-only, never model input */
  ACTIVITY_LABEL = 'activity_label',
}

export enum ToolCallTypes {
  FUNCTION = 'function',
  RETRIEVAL = 'retrieval',
  FILE_SEARCH = 'file_search',
  CODE_INTERPRETER = 'code_interpreter',
  /* Agents Tool Call */
  TOOL_CALL = 'tool_call',
}

export enum Callback {
  TOOL_ERROR = 'handleToolError',
  TOOL_START = 'handleToolStart',
  TOOL_END = 'handleToolEnd',
  CUSTOM_EVENT = 'handleCustomEvent',
  /*
  LLM_START = 'handleLLMStart',
  LLM_NEW_TOKEN = 'handleLLMNewToken',
  LLM_ERROR = 'handleLLMError',
  LLM_END = 'handleLLMEnd',
  CHAT_MODEL_START = 'handleChatModelStart',
  CHAIN_START = 'handleChainStart',
  CHAIN_ERROR = 'handleChainError',
  CHAIN_END = 'handleChainEnd',
  TEXT = 'handleText',
  AGENT_ACTION = 'handleAgentAction',
  AGENT_END = 'handleAgentEnd',
  RETRIEVER_START = 'handleRetrieverStart',
  RETRIEVER_END = 'handleRetrieverEnd',
  RETRIEVER_ERROR = 'handleRetrieverError',
  */
}

export enum Constants {
  OFFICIAL_CODE_BASEURL = 'https://api.librechat.ai/v1',
  EXECUTE_CODE = 'execute_code',
  TOOL_SEARCH = 'tool_search',
  PROGRAMMATIC_TOOL_CALLING = 'run_tools_with_code',
  WEB_SEARCH = 'web_search',
  CONTENT_AND_ARTIFACT = 'content_and_artifact',
  LC_TRANSFER_TO_ = 'lc_transfer_to_',
  HANDOFF_PARALLEL_BATCH = '__handoff_parallel_batch',
  HANDOFF_GROUP_ID = '__handoff_group_id',
  /** Delimiter for MCP tools: toolName_mcp_serverName */
  MCP_DELIMITER = '_mcp_',
  /** Anthropic server tool ID prefix (web_search, code_execution, etc.) */
  ANTHROPIC_SERVER_TOOL_PREFIX = 'srvtoolu_',
  SKILL_TOOL = 'skill',
  /**
   * Callback-metadata keys stamped by `attemptInvoke` /
   * `tryFallbackProviders` carrying the provider (SDK `Providers` enum
   * value) and configured model that actually served a model invocation.
   * Unlike `ls_provider` — which derived providers inherit from their base
   * class (e.g. DeepSeek/OpenRouter report `'openai'`) — these reflect the
   * SDK's own routing, including fallback-provider calls. Consumed by the
   * subagent usage-capture handler to tag billing events.
   */
  INVOKED_PROVIDER = '__invoked_provider',
  INVOKED_MODEL = '__invoked_model',
  READ_FILE = 'read_file',
  BASH_TOOL = 'bash_tool',
  BASH_PROGRAMMATIC_TOOL_CALLING = 'run_tools_with_bash',
  SUBAGENT = 'subagent',
  /**
   * Local-engine coding tool names. Promoted to `Constants.*` (rather
   * than left as per-file `*ToolName` consts) so consumer UIs — most
   * importantly LibreChat's `getToolIconType` map — can match against
   * canonical strings instead of guessing. Existing matched names:
   * `bash_tool`, `read_file`, `execute_code`, `run_tools_with_code`.
   * The rest below are new and currently fall through to the generic
   * tool icon; once LibreChat adds icons keyed on the same names, the
   * wiring will work without an SDK change.
   */
  WRITE_FILE = 'write_file',
  EDIT_FILE = 'edit_file',
  GREP_SEARCH = 'grep_search',
  GLOB_SEARCH = 'glob_search',
  LIST_DIRECTORY = 'list_directory',
  COMPILE_CHECK = 'compile_check',
}

/** Tool names that use the code execution environment (shared session, file tracking). */
export const CODE_EXECUTION_TOOLS: ReadonlySet<string> = new Set([
  Constants.EXECUTE_CODE,
  Constants.BASH_TOOL,
  Constants.PROGRAMMATIC_TOOL_CALLING,
  Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
]);

/**
 * Canonical names of the local-engine-specific coding tools — the
 * file/edit/search/typecheck surface that doesn't exist in the
 * remote (sandbox-API) engine. Single source of truth; the per-tool
 * factories, registry definitions, and `createWorkspacePolicyHook`
 * default extractors all key off these.
 *
 * `read_file` is on this list (the existing ReadFile tool is
 * remote-specific; the local engine's `read_file` is a parallel
 * implementation that shares the canonical name so consumer UIs
 * — most importantly LibreChat's `getToolIconType` — render both
 * with the same icon).
 */
export const LOCAL_CODING_TOOL_NAMES: readonly string[] = [
  Constants.READ_FILE,
  Constants.WRITE_FILE,
  Constants.EDIT_FILE,
  Constants.GREP_SEARCH,
  Constants.GLOB_SEARCH,
  Constants.LIST_DIRECTORY,
  Constants.COMPILE_CHECK,
];

/**
 * Every tool name the local coding bundle (`createLocalCodingTools`)
 * exposes — the local-specific tools above plus the bash/code/PTC
 * pair that the local engine wraps around the existing factories.
 * Tests pin against this so any addition/removal in the bundle is
 * accompanied by a deliberate canonical-name update here.
 */
export const LOCAL_CODING_BUNDLE_NAMES: readonly string[] = [
  ...LOCAL_CODING_TOOL_NAMES,
  Constants.BASH_TOOL,
  Constants.EXECUTE_CODE,
  Constants.PROGRAMMATIC_TOOL_CALLING,
  Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
];

export enum TitleMethod {
  STRUCTURED = 'structured',
  FUNCTIONS = 'functions',
  COMPLETION = 'completion',
}

export enum EnvVar {
  CODE_BASEURL = 'LIBRECHAT_CODE_BASEURL',
  CODE_API_RUN_TIMEOUT_MS = 'CODE_API_RUN_TIMEOUT_MS',
}
