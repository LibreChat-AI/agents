import type { SubagentDescriptor } from '@/types';

/** Lowercase words joined by `-` or `_`, short enough to copy reliably. */
const READABLE_TYPE = /^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$/;
const MAX_READABLE_TYPE_LENGTH = 40;

export type SubagentTypeAliases = {
  /** Model-facing name → configured `type`. Holds only types that got an alias. */
  toType: Map<string, string>;
  /** Configured `type` → the name advertised to the model. */
  toName: Map<string, string>;
};

export function isReadableSubagentType(type: string): boolean {
  return type.length <= MAX_READABLE_TYPE_LENGTH && READABLE_TYPE.test(type);
}

function slugify(name: string): string {
  return name
    .replace(/ß/g, 'ss')
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, MAX_READABLE_TYPE_LENGTH)
    .replace(/-+$/, '');
}

/**
 * Hosts often use opaque ids as `type` (database ids such as
 * `agent_bqFVwJV0dwKDSyqMnrIIC`). Models copy such strings into
 * `subagent_type` unreliably: they swap characters or invent an id, and the
 * call fails schema validation. Opaque types are therefore advertised under a
 * name derived from the subagent's display name (`exchange-readonly`), with
 * `-2`, `-3` on a clash. Readable types are advertised unchanged.
 *
 * Deterministic for a given config order, so the tool schema and the executor
 * derive the same names independently.
 */
export function buildSubagentTypeAliases(
  configs: Iterable<Pick<SubagentDescriptor, 'type' | 'name'>>
): SubagentTypeAliases {
  const list = [...configs];
  const toType = new Map<string, string>();
  const toName = new Map<string, string>();
  const taken = new Set(
    list.map((c) => c.type).filter((type) => isReadableSubagentType(type))
  );
  for (const { type, name } of list) {
    if (isReadableSubagentType(type)) {
      toName.set(type, type);
      continue;
    }
    const base = slugify(name) || 'subagent';
    let alias = base;
    for (let n = 2; taken.has(alias); n++) {
      alias = `${base}-${n}`;
    }
    taken.add(alias);
    toType.set(alias, type);
    toName.set(type, alias);
  }
  return { toType, toName };
}

/** Maps an advertised name back to its configured `type`; other values pass through. */
export function resolveSubagentType(
  aliases: SubagentTypeAliases,
  type: string | undefined
): string | undefined {
  return type == null ? type : (aliases.toType.get(type) ?? type);
}
