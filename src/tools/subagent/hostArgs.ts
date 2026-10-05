import { createHash } from 'crypto';
import type {
  SubagentHostArgs,
  SubagentHostArgSpec,
  SubagentHostArgSpecs,
} from '@/types/graph';
import type { JsonSchemaType } from '@/types/tools';

/** Bounds on host-declared subagent call arguments and their values. */
export const SUBAGENT_HOST_ARG_LIMITS = Object.freeze({
  argsPerSubagent: 8,
  distinctArgs: 16,
  enumValues: 64,
  mergedEnumValues: 256,
  valueLength: 256,
  descriptionLength: 1024,
});

const HOST_ARG_NAME_PATTERN = /^[a-z][a-z0-9_]{0,63}$/;
/** Advertises the runtime's no-control-character rule; fixed and linear-time. */
const FREE_FORM_SCHEMA_PATTERN = '^[^\\u0000-\\u001f\\u007f]*$';
const RESERVED_HOST_ARG_NAMES: ReadonlySet<string> = new Set([
  'intent',
  'description',
  'subagent_type',
  'run_in_background',
  'subagent_thread_id',
]);

/** Any subagent entry that may declare host arguments. */
export type SubagentHostArgDeclaration = {
  type: string;
  hostArgs?: SubagentHostArgSpecs;
};

export type SubagentHostArgsResult =
  | { ok: true; hostArgs?: SubagentHostArgs }
  | { ok: false; message: string };

type HostArgEntry = readonly [string, SubagentHostArgSpec];

type MergedHostArg = {
  description: string;
  values: string[];
  seenValues: Set<string>;
  freeForm: boolean;
  maxLength: number;
};

/** A name a host may declare: lowercase snake case, not a built-in subagent argument. */
export function isSubagentHostArgName(name: string): boolean {
  return HOST_ARG_NAME_PATTERN.test(name) && !RESERVED_HOST_ARG_NAMES.has(name);
}

/** Length in Unicode code points, the unit JSON Schema `maxLength` counts. */
function countCodePoints(value: string): number {
  let count = 0;
  for (const _ of value) {
    count += 1;
  }
  return count;
}

function hasControlCharacter(value: string): boolean {
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code < 0x20 || code === 0x7f) {
      return true;
    }
  }
  return false;
}

function isBoundedText(value: unknown, maxLength: number): value is string {
  return (
    typeof value === 'string' &&
    value.trim().length > 0 &&
    countCodePoints(value) <= maxLength &&
    !hasControlCharacter(value)
  );
}

function getFreeFormMaxLength(spec: SubagentHostArgSpec): number {
  return spec.maxLength ?? SUBAGENT_HOST_ARG_LIMITS.valueLength;
}

function validateEnum(label: string, values: readonly string[]): void {
  if (
    !Array.isArray(values) ||
    values.length === 0 ||
    values.length > SUBAGENT_HOST_ARG_LIMITS.enumValues
  ) {
    throw new Error(
      `${label} enum must list 1-${SUBAGENT_HOST_ARG_LIMITS.enumValues} values.`
    );
  }
  const seen = new Set<string>();
  for (const value of values) {
    if (!isBoundedText(value, SUBAGENT_HOST_ARG_LIMITS.valueLength)) {
      throw new Error(
        `${label} enum values must be non-blank strings of at most ${SUBAGENT_HOST_ARG_LIMITS.valueLength} characters without control characters.`
      );
    }
    if (seen.has(value)) {
      throw new Error(`${label} enum values must be unique.`);
    }
    seen.add(value);
  }
}

function validateFreeForm(label: string, spec: SubagentHostArgSpec): void {
  const maxLength = getFreeFormMaxLength(spec);
  if (
    !Number.isSafeInteger(maxLength) ||
    maxLength < 1 ||
    maxLength > SUBAGENT_HOST_ARG_LIMITS.valueLength
  ) {
    throw new Error(
      `${label} maxLength must be an integer from 1 to ${SUBAGENT_HOST_ARG_LIMITS.valueLength}.`
    );
  }
}

function validateSpec(
  type: string,
  name: string,
  spec: SubagentHostArgSpec | undefined
): asserts spec is SubagentHostArgSpec {
  const label = `Subagent "${type}" host argument "${name}"`;
  if (!isSubagentHostArgName(name)) {
    throw new Error(
      `${label} must match ${HOST_ARG_NAME_PATTERN.source} and must not reuse a built-in subagent argument name.`
    );
  }
  if (spec == null) {
    throw new Error(`${label} must be an object.`);
  }
  if ((spec as { pattern?: unknown }).pattern !== undefined) {
    throw new Error(
      `${label} cannot declare a pattern; validate the format in the resolver.`
    );
  }
  if (
    typeof spec.description !== 'string' ||
    spec.description.trim().length === 0 ||
    spec.description.length > SUBAGENT_HOST_ARG_LIMITS.descriptionLength
  ) {
    throw new Error(
      `${label} needs a description of at most ${SUBAGENT_HOST_ARG_LIMITS.descriptionLength} characters.`
    );
  }
  if (spec.enum == null) {
    validateFreeForm(label, spec);
    return;
  }
  if (spec.maxLength != null) {
    throw new Error(`${label} cannot combine enum with maxLength.`);
  }
  validateEnum(label, spec.enum);
}

/**
 * Returns a subagent's validated host argument declarations in declaration
 * order. Throws on a malformed or oversized declaration.
 */
export function getSubagentHostArgSpecs(
  config: SubagentHostArgDeclaration
): readonly HostArgEntry[] {
  const { hostArgs } = config;
  if (hostArgs == null) {
    return [];
  }
  if (typeof hostArgs !== 'object' || Array.isArray(hostArgs)) {
    throw new Error(`Subagent "${config.type}" hostArgs must be an object.`);
  }
  const entries = Object.entries(hostArgs);
  if (entries.length > SUBAGENT_HOST_ARG_LIMITS.argsPerSubagent) {
    throw new Error(
      `Subagent "${config.type}" declares more than ${SUBAGENT_HOST_ARG_LIMITS.argsPerSubagent} host arguments.`
    );
  }
  for (const [name, spec] of entries) {
    validateSpec(config.type, name, spec);
  }
  return entries;
}

/** Every host argument name declared by at least one subagent. */
export function collectSubagentHostArgNames(
  configs: readonly SubagentHostArgDeclaration[]
): ReadonlySet<string> {
  const names = new Set<string>();
  for (const config of configs) {
    for (const [name] of getSubagentHostArgSpecs(config)) {
      names.add(name);
    }
  }
  if (names.size > SUBAGENT_HOST_ARG_LIMITS.distinctArgs) {
    throw new Error(
      `Subagents declare more than ${SUBAGENT_HOST_ARG_LIMITS.distinctArgs} distinct host arguments.`
    );
  }
  return names;
}

function mergeSpec(
  merged: Map<string, MergedHostArg>,
  name: string,
  spec: SubagentHostArgSpec
): void {
  let entry = merged.get(name);
  if (entry == null) {
    entry = {
      description: spec.description,
      values: [],
      seenValues: new Set(),
      freeForm: false,
      maxLength: 0,
    };
    merged.set(name, entry);
  }
  if (spec.enum == null) {
    entry.freeForm = true;
    entry.maxLength = Math.max(entry.maxLength, getFreeFormMaxLength(spec));
    return;
  }
  for (const value of spec.enum) {
    entry.maxLength = Math.max(entry.maxLength, countCodePoints(value));
    if (!entry.seenValues.has(value)) {
      entry.seenValues.add(value);
      entry.values.push(value);
    }
  }
  if (entry.values.length > SUBAGENT_HOST_ARG_LIMITS.mergedEnumValues) {
    throw new Error(
      `Subagent host argument "${name}" lists more than ${SUBAGENT_HOST_ARG_LIMITS.mergedEnumValues} distinct values across subagents.`
    );
  }
}

function toPropertySchema(entry: MergedHostArg): JsonSchemaType {
  if (!entry.freeForm) {
    return {
      type: 'string',
      description: entry.description,
      enum: entry.values,
    };
  }
  return {
    type: 'string',
    description: entry.description,
    maxLength: entry.maxLength,
    pattern: FREE_FORM_SCHEMA_PATTERN,
  };
}

function summarizeSpecs(entries: readonly HostArgEntry[]): string {
  return entries
    .map(
      ([name, spec]) =>
        `${name}: ${
          spec.enum == null
            ? `any text up to ${getFreeFormMaxLength(spec)} characters`
            : spec.enum.join(' | ')
        }`
    )
    .join('; ');
}

/**
 * Builds one optional tool property per declared host argument plus a
 * per-type summary of the values each subagent accepts. The first
 * declaration's description wins; enums are unioned across subagents.
 */
export function buildSubagentHostArgProperties(
  configs: readonly SubagentHostArgDeclaration[]
): {
  properties: Record<string, JsonSchemaType>;
  summaries: ReadonlyMap<string, string>;
} {
  const merged = new Map<string, MergedHostArg>();
  const summaries = new Map<string, string>();
  for (const config of configs) {
    const entries = getSubagentHostArgSpecs(config);
    if (entries.length === 0) {
      continue;
    }
    for (const [name, spec] of entries) {
      mergeSpec(merged, name, spec);
    }
    summaries.set(config.type, summarizeSpecs(entries));
  }
  if (merged.size > SUBAGENT_HOST_ARG_LIMITS.distinctArgs) {
    throw new Error(
      `Subagents declare more than ${SUBAGENT_HOST_ARG_LIMITS.distinctArgs} distinct host arguments.`
    );
  }
  const properties: Record<string, JsonSchemaType> = {};
  for (const [name, entry] of merged) {
    properties[name] = toPropertySchema(entry);
  }
  return { properties, summaries };
}

/**
 * Reads the declared host argument properties from raw tool input. Missing,
 * null, and blank values count as omitted; other non-string values are
 * rejected rather than coerced.
 */
export function pickSubagentHostArgInput(
  input: object,
  names: ReadonlySet<string>
): SubagentHostArgsResult {
  if (names.size === 0) {
    return { ok: true };
  }
  const values = input as Readonly<Record<string, unknown>>;
  const picked: Record<string, string> = {};
  let count = 0;
  for (const name of names) {
    if (!Object.prototype.hasOwnProperty.call(values, name)) {
      continue;
    }
    const value = values[name];
    if (value == null) {
      continue;
    }
    if (typeof value !== 'string') {
      return { ok: false, message: `Error: "${name}" must be a string.` };
    }
    if (value.trim() === '') {
      continue;
    }
    picked[name] = value;
    count += 1;
  }
  return count === 0 ? { ok: true } : { ok: true, hostArgs: picked };
}

function checkValue(
  type: string,
  name: string,
  spec: SubagentHostArgSpec,
  value: string
): string | undefined {
  const omit = `Omit "${name}" to let the host choose.`;
  if (spec.enum != null) {
    return spec.enum.includes(value)
      ? undefined
      : `Error: "${name}" for subagent "${type}" must be one of: ${spec.enum.join(', ')}. ${omit}`;
  }
  const maxLength = getFreeFormMaxLength(spec);
  return isBoundedText(value, maxLength)
    ? undefined
    : `Error: "${name}" for subagent "${type}" must be at most ${maxLength} characters without control characters. ${omit}`;
}

/**
 * Checks call values against the selected subagent's declarations and
 * returns them frozen in canonical key order. Rejects arguments the selected
 * subagent does not declare, including ones declared by a sibling.
 */
export function resolveSubagentHostArgs(
  config: SubagentHostArgDeclaration,
  hostArgs: SubagentHostArgs | undefined
): SubagentHostArgsResult {
  if (hostArgs == null) {
    return { ok: true };
  }
  const names = Object.keys(hostArgs).sort();
  if (names.length === 0) {
    return { ok: true };
  }
  const specs = new Map(getSubagentHostArgSpecs(config));
  const resolved: Record<string, string> = {};
  for (const name of names) {
    const spec = specs.get(name);
    if (spec == null) {
      return {
        ok: false,
        message: isSubagentHostArgName(name)
          ? `Error: Subagent "${config.type}" does not accept "${name}". Omit it for this subagent type.`
          : `Error: Subagent "${config.type}" received an unsupported argument.`,
      };
    }
    const value = hostArgs[name];
    if (typeof value !== 'string') {
      return { ok: false, message: `Error: "${name}" must be a string.` };
    }
    const error = checkValue(config.type, name, spec, value);
    if (error != null) {
      return { ok: false, message: error };
    }
    resolved[name] = value;
  }
  return { ok: true, hostArgs: Object.freeze(resolved) };
}

/** Opaque, value-free identity of a call's host arguments for replay records. */
export function getSubagentHostArgsDigest(
  hostArgs: SubagentHostArgs | undefined
): string | undefined {
  if (hostArgs == null) {
    return undefined;
  }
  const entries = Object.entries(hostArgs).sort(([left], [right]) =>
    left < right ? -1 : 1
  );
  if (entries.length === 0) {
    return undefined;
  }
  return createHash('sha256').update(JSON.stringify(entries)).digest('hex');
}
