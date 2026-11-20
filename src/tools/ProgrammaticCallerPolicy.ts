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
  normalizeIdentifier?: (name: string) => string;
}): t.LCTool[] {
  const allowedToolDefs = args.allowedToolDefs ?? [];
  const disallowedToolDefs = args.disallowedToolDefs ?? [];
  const requestedToolNames = args.requestedToolNames;

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
  const requestedDisallowedNames = new Set(
    requestedToolNames.filter((name) => disallowedNames.has(name))
  );
  assertDisallowedToolUsage(
    requestedDisallowedNames,
    args.programmaticToolName
  );

  const allowedByName = new Map(
    allowedToolDefs.map((toolDef) => [toolDef.name, toolDef])
  );
  const unknownNames = new Set(
    requestedToolNames.filter((name) => !allowedByName.has(name))
  );
  if (unknownNames.size > 0 && args.normalizeIdentifier != null) {
    const candidatesByIdentifier = new Map<string, t.LCTool[]>();
    for (const toolDef of allowedToolDefs) {
      const identifier = args.normalizeIdentifier(toolDef.name);
      if (!unknownNames.has(identifier)) {
        continue;
      }
      const candidates = candidatesByIdentifier.get(identifier) ?? [];
      candidates.push(toolDef);
      candidatesByIdentifier.set(identifier, candidates);
    }

    const disallowedAliases = new Set<string>();
    for (const toolDef of disallowedToolDefs) {
      const identifier = args.normalizeIdentifier(toolDef.name);
      if (unknownNames.has(identifier)) {
        disallowedAliases.add(toolDef.name);
      }
    }
    assertDisallowedToolUsage(disallowedAliases, args.programmaticToolName);

    for (const name of unknownNames) {
      const candidates = candidatesByIdentifier.get(name);
      if (candidates == null) {
        continue;
      }
      assertUnambiguousIdentifiers(
        candidates,
        args.normalizeIdentifier,
        args.programmaticToolName
      );
      allowedByName.set(name, candidates[0]);
      unknownNames.delete(name);
    }
  }

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

  return [
    ...new Set(
      requestedToolNames.map((name) => allowedByName.get(name) as t.LCTool)
    ),
  ];
}
