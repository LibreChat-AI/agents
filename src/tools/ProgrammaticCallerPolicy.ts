import type * as t from '@/types';

export type ProgrammaticInvocationParams = {
  code: string;
  tool_manifest?: string[];
  timeout?: number;
  lang?: string;
  runtime?: string;
  language?: string;
};

export function resolveProgrammaticToolDefinitions(
  context: Partial<t.ProgrammaticCache> & { tools?: t.LCTool[] }
): t.LCTool[] | undefined {
  return context.toolDefs ?? context.tools;
}

export function projectProgrammaticToolMap(
  toolMap: t.ToolMap,
  selectedToolDefs: readonly t.LCTool[]
): t.ToolMap {
  const selectedNames = new Set(selectedToolDefs.map((toolDef) => toolDef.name));
  return new Map(
    [...toolMap].filter(([name]) => selectedNames.has(name))
  );
}

export function assertDisallowedToolUsage(
  disallowedNames: ReadonlySet<string>,
  programmaticToolName: string
): void {
  if (disallowedNames.size === 0) {
    return;
  }

  const names = [...disallowedNames];
  throw new Error(
    `Tool${names.length === 1 ? '' : 's'} ${names
      .map((name) => `"${name}"`)
      .join(', ')} cannot be called from "${programmaticToolName}" because ` +
      `the${names.length === 1 ? ' tool is' : 'se tools are'} not marked for code_execution. ` +
      `Call ${names.length === 1 ? 'it' : 'them'} directly instead.`
  );
}

/**
 * Rejects a stub set holding two tools that collapse to one runtime identifier.
 *
 * Stub generation binds by identifier, so such tools are indistinguishable once
 * emitted — the generated program defines both and the later one wins. Failing
 * beats dispatching by declaration order.
 *
 * Called on the set each backend actually emits, not on the selection: a
 * definition dropped before codegen never binds an identifier, so a collision
 * among such definitions is not a conflict.
 */
export function assertUnambiguousIdentifiers(
  selectedToolDefs: readonly t.LCTool[],
  normalizeIdentifier: (name: string) => string,
  programmaticToolName: string
): void {
  const seen = new Map<string, string>();
  for (const toolDef of selectedToolDefs) {
    const identifier = normalizeIdentifier(toolDef.name);
    const prior = seen.get(identifier);
    if (prior != null && prior !== toolDef.name) {
      throw new Error(
        `Tools "${prior}" and "${toolDef.name}" both become "${identifier}" in ` +
          `the sandbox, so "${programmaticToolName}" cannot tell which one the ` +
          'code means. Call them directly, or rename one.'
      );
    }
    seen.set(identifier, toolDef.name);
  }
}

export function selectProgrammaticTools(args: {
  requestedToolNames?: string[];
  allowedToolDefs?: t.LCTool[];
  disallowedToolDefs?: t.LCTool[];
  programmaticToolName: string;
  getManifestAlias?: (name: string) => string | undefined;
}): t.LCTool[] {
  const allowedToolDefs = args.allowedToolDefs ?? [];
  const disallowedToolDefs = args.disallowedToolDefs ?? [];
  let requestedToolNames = args.requestedToolNames;

  if (requestedToolNames == null) {
    if (disallowedToolDefs.length > 0) {
      throw new Error(
        `"${args.programmaticToolName}" requires a tool_manifest when direct-only tools are configured. ` +
          'List every registered tool name used by the submitted code in the tool_manifest field.'
      );
    }
    return allowedToolDefs;
  }

  const disallowedNames = new Set(
    disallowedToolDefs.map((toolDef) => toolDef.name)
  );
  const allowedByName = new Map(
    allowedToolDefs.map((toolDef) => [toolDef.name, toolDef])
  );
  const getManifestAlias = args.getManifestAlias;
  if (getManifestAlias != null) {
    const aliases = new Map<string, string | null>();
    for (const name of [...allowedByName.keys(), ...disallowedNames]) {
      const alias = getManifestAlias(name);
      if (alias == null || alias === name) {
        continue;
      }
      const prior = aliases.get(alias);
      aliases.set(alias, prior === undefined || prior === name ? name : null);
    }
    requestedToolNames = requestedToolNames.map((name) => {
      if (allowedByName.has(name) || disallowedNames.has(name)) {
        return name;
      }
      const registeredName = aliases.get(name);
      if (registeredName === null) {
        throw new Error(
          `Tool identifier "${name}" is ambiguous in "${args.programmaticToolName}". ` +
            'Use an exact registered tool name in the tool_manifest field.'
        );
      }
      return registeredName ?? name;
    });
  }
  const requestedDisallowedNames = new Set(
    requestedToolNames.filter((name) => disallowedNames.has(name))
  );
  assertDisallowedToolUsage(
    requestedDisallowedNames,
    args.programmaticToolName
  );

  const unknownNames = new Set(
    requestedToolNames.filter((name) => !allowedByName.has(name))
  );
  if (unknownNames.size > 0) {
    throw new Error(
      `Tool${unknownNames.size === 1 ? '' : 's'} ${[...unknownNames]
        .map((name) => `"${name}"`)
        .join(
          ', '
        )} cannot be used by "${args.programmaticToolName}" because ` +
        `${unknownNames.size === 1 ? 'it is' : 'they are'} not available for code_execution.`
    );
  }

  return [...new Set(requestedToolNames)].map(
    (name) => allowedByName.get(name) as t.LCTool
  );
}
