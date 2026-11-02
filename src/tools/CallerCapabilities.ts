import type * as t from '@/types';
import { Constants } from '@/common';

const PROGRAMMATIC_CONTROL_TOOL_NAMES = new Set<string>([
  Constants.PROGRAMMATIC_TOOL_CALLING,
  Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
  Constants.TOOL_SEARCH,
]);

export type CallerCapabilityProjection = {
  directTools: t.LCTool[];
  codeExecutionTools: t.LCTool[];
  directOnlyTools: t.LCTool[];
  codeExecutionOnlyTools: t.LCTool[];
};

/**
 * Combines event schemas with runtime capability metadata. Matching runtime
 * entries override caller/defer policy without replacing the event schema;
 * runtime-only definitions are appended intact.
 */
export function mergeCallerCapabilityDefinitions(
  toolDefs: Iterable<t.LCTool> | null | undefined,
  overrides: Iterable<t.LCTool> | null | undefined
): t.LCTool[] {
  const runtimeDefinitions = Array.from(overrides ?? []);
  const merged = new Map(
    applyCallerCapabilityDefinitionOverrides(
      toolDefs ?? [],
      runtimeDefinitions
    ).map((toolDef) => [toolDef.name, toolDef])
  );
  for (const override of runtimeDefinitions) {
    if (!merged.has(override.name)) {
      merged.set(override.name, override);
    }
  }
  return Array.from(merged.values());
}

/**
 * Applies runtime caller metadata to schema-only event definitions without
 * adding registry-only tools or replacing their model-facing schemas.
 */
export function applyCallerCapabilityDefinitionOverrides(
  toolDefs: Iterable<t.LCTool>,
  overrides: Iterable<t.LCTool> | null | undefined
): t.LCTool[] {
  const overridesByName = new Map<string, t.LCTool>();
  for (const override of overrides ?? []) {
    overridesByName.set(override.name, override);
  }
  return Array.from(toolDefs, (toolDef) => {
    const override = overridesByName.get(toolDef.name);
    if (override == null) {
      return toolDef;
    }
    return {
      ...toolDef,
      allowed_callers: override.allowed_callers,
      defer_loading: override.defer_loading,
    };
  });
}

/** Converts the live projection into the versioned event transport shape. */
export function createCallerCapabilityProjectionSnapshot(
  projection: CallerCapabilityProjection
): t.CallerCapabilityProjectionSnapshot {
  return {
    version: 1,
    directToolNames: projection.directTools.map((toolDef) => toolDef.name),
    codeExecutionToolNames: projection.codeExecutionTools.map(
      (toolDef) => toolDef.name
    ),
    directOnlyToolNames: projection.directOnlyTools.map(
      (toolDef) => toolDef.name
    ),
    codeExecutionOnlyToolNames: projection.codeExecutionOnlyTools.map(
      (toolDef) => toolDef.name
    ),
  };
}

export function getAllowedCallers(
  toolDef: t.LCTool
): readonly t.AllowedCaller[] {
  return toolDef.allowed_callers ?? ['direct'];
}

export function allowsToolCaller(
  toolDef: t.LCTool,
  caller: t.AllowedCaller
): boolean {
  return getAllowedCallers(toolDef).includes(caller);
}

export function isProgrammaticControlTool(name: string): boolean {
  return PROGRAMMATIC_CONTROL_TOOL_NAMES.has(name);
}

export function isToolDefinitionActive(
  toolDef: t.LCTool,
  discoveredToolNames: ReadonlySet<string>
): boolean {
  return (
    toolDef.defer_loading !== true || discoveredToolNames.has(toolDef.name)
  );
}

/** Execution must enforce the same policy as model-facing tool projection. */
export function getDirectToolCallError(
  definition: t.LCTool | undefined,
  discoveredNames: readonly string[]
): string | undefined {
  if (definition == null) return undefined;
  if (!allowsToolCaller(definition, 'direct')) return `Tool "${definition.name}" cannot be called directly`;
  if (definition.defer_loading === true && !discoveredNames.includes(definition.name)) return `Tool "${definition.name}" is not available before discovery`;
  return undefined;
}

export function resolveCallerCapabilityProjection(
  toolDefs: Iterable<t.LCTool>,
  isActive: (toolDef: t.LCTool) => boolean = () => true
): CallerCapabilityProjection {
  const directTools: t.LCTool[] = [];
  const codeExecutionTools: t.LCTool[] = [];
  const directOnlyTools: t.LCTool[] = [];
  const codeExecutionOnlyTools: t.LCTool[] = [];

  for (const toolDef of toolDefs) {
    if (!isActive(toolDef)) {
      continue;
    }
    const allowsDirect = allowsToolCaller(toolDef, 'direct');
    const allowsCodeExecution = allowsToolCaller(toolDef, 'code_execution');
    if (allowsDirect) {
      directTools.push(toolDef);
    }
    if (allowsCodeExecution) {
      codeExecutionTools.push(toolDef);
    }
    if (allowsDirect && !allowsCodeExecution) {
      directOnlyTools.push(toolDef);
    }
    if (allowsCodeExecution && !allowsDirect) {
      codeExecutionOnlyTools.push(toolDef);
    }
  }

  return {
    directTools,
    codeExecutionTools,
    directOnlyTools,
    codeExecutionOnlyTools,
  };
}
