import type * as t from '@/types';
import {
  LocalBashExecutionToolDescription,
  LocalBashExecutionToolSchema,
} from './LocalExecutionTools';
import { createLocalCodingToolDefinitions } from './LocalCodingTools';
import { allowsToolCaller } from '@/tools/CallerCapabilities';
import { createToolSearch } from '@/tools/ToolSearch';
import { Constants, LOCAL_CODING_BUNDLE_NAMES } from '@/common';

export type LocalProgrammaticProfile = {
  tools: t.GenericTool[];
  toolRegistry: t.LCToolRegistry;
  toolExecution: t.ToolExecutionConfig;
  instructions: string;
};

/** A stable outer surface backed by the existing local implementations and caller policy. */
export function createLocalProgrammaticProfile(options: {
  /** Hybrid lets the current model choose direct local calls or programs per operation. */
  mode?: 'compact' | 'hybrid';
  /** Optional bound on the initial discovery catalog; full search and schemas remain available. */
  maxCatalogChars?: t.ToolSearchParams['maxCatalogChars'];
  local?: t.LocalExecutionConfig;
  tools?: t.GenericTool[];
  toolRegistry?: t.LCToolRegistry;
} = {}): LocalProgrammaticProfile {
  const hybrid = options.mode === 'hybrid';
  const definitions = createLocalCodingToolDefinitions();
  definitions.push({
    name: Constants.BASH_TOOL,
    description: LocalBashExecutionToolDescription,
    parameters: { ...LocalBashExecutionToolSchema, required: [...LocalBashExecutionToolSchema.required] },
  });
  const definitionsByName = new Map(definitions.map((definition) => [definition.name, definition]));
  const selected = new Set(options.local?.codingToolNames ?? definitionsByName.keys());
  for (const name of selected) {
    if (!definitionsByName.has(name)) throw new Error(`Unsupported profile local tool: ${name}`);
  }
  const registry: t.LCToolRegistry = new Map();
  const localNames: string[] = [];
  const localCallers: t.AllowedCaller[] = hybrid ? ['direct', 'code_execution'] : ['code_execution'];
  for (const definition of definitions) {
    if (!selected.has(definition.name)) continue;
    const supplied = options.toolRegistry?.get(definition.name);
    if (supplied != null) {
      for (const caller of localCallers) {
        if (!allowsToolCaller(supplied, caller)) throw new Error(`Profile local tool does not permit ${caller}: ${definition.name}`);
      }
      if (hybrid && supplied.defer_loading === true) throw new Error(`Hybrid local tool requires immediate availability: ${definition.name}`);
    }
    localNames.push(definition.name);
    registry.set(definition.name, { ...definition, allowed_callers: [...localCallers], defer_loading: !hybrid });
  }
  const reserved = new Set<string>([Constants.TOOL_SEARCH, ...LOCAL_CODING_BUNDLE_NAMES]);
  for (const bound of options.tools ?? []) {
    if (reserved.has(bound.name)) throw new Error(`Reserved profile tool: ${bound.name}`);
    const definition = options.toolRegistry?.get(bound.name);
    if (definition?.parameters == null) throw new Error(`Profile tool requires a canonical schema: ${bound.name}`);
    if (!allowsToolCaller(definition, 'code_execution')) throw new Error(`Profile tool does not permit code execution: ${bound.name}`);
    registry.set(bound.name, { ...definition, allowed_callers: ['code_execution'], defer_loading: true });
  }
  return {
    tools: [...(options.tools ?? []), createToolSearch({
      mode: 'local', includeSchemas: true, toolRegistry: registry, maxCatalogChars: options.maxCatalogChars,
    })],
    toolRegistry: registry,
    toolExecution: {
      engine: 'local',
      programmaticToolInstructions: hybrid ? 'hybrid' : 'discovery',
      local: { ...options.local, includeCodingTools: true, codingToolNames: [...localNames, Constants.PROGRAMMATIC_TOOL_CALLING] },
    },
    instructions: [
      'You are a coding agent working in the configured repository.',
      ...(hybrid ? [
        `Choose direct tools or run_tools_with_code for each operation. These local tools are ready for both paths: ${localNames.join(', ') || 'none'}.`,
        'Use direct tools for isolated reads, edits, commands, and investigation that needs your judgment after each result.',
        'Use programs for predictable loops, dependent connector calls, or batches that benefit from filtering intermediate results.',
        'Do not wrap a single tool call in a program unless computation or output filtering justifies it.',
        'Use tool_search to discover connector tools and their complete schemas before calling them from a program.',
      ] : [
        'Use tool_search to discover relevant tools and their complete schemas, then run_tools_with_code to compose work.',
      ]),
      'A program can inspect and transform files, call discovered tools, and branch on results without another model request.',
      'Batch independent reads, filter large results in the program, and return concise evidence. Await dependent operations in order.',
      'Use the workspace for persistent intermediate data. Programs start fresh interpreters. Preserve useful evidence when an operation fails.',
      ...(hybrid ? [] : ['For simple shell work, use the Bash runtime with an empty tool_manifest. Prefer direct code over unnecessary tool indirection.']),
      'Finish only when the task is done; state what changed and how it was checked.',
    ].join(' '),
  };
}
