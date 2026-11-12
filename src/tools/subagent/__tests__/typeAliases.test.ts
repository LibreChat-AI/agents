import { describe, it, expect } from '@jest/globals';
import {
  buildSubagentTypeAliases,
  isReadableSubagentType,
  resolveSubagentType,
} from '../typeAliases';

describe('subagent type aliases', () => {
  it('keeps readable types', () => {
    expect(isReadableSubagentType('researcher')).toBe(true);
    expect(isReadableSubagentType('research-team')).toBe(true);
    expect(isReadableSubagentType('code_reviewer')).toBe(true);
    expect(isReadableSubagentType('agent_bqFVwJV0dwKDSyqMnrIIC')).toBe(false);
    expect(isReadableSubagentType('Researcher')).toBe(false);
    const { toName, toType } = buildSubagentTypeAliases([
      { type: 'researcher', name: 'Anything' },
      { type: 'self', name: 'Parent' },
    ]);
    expect(toName.get('researcher')).toBe('researcher');
    expect(toName.get('self')).toBe('self');
    expect(toType.size).toBe(0);
  });

  it('derives names from display names, numbering clashes', () => {
    const { toName } = buildSubagentTypeAliases([
      { type: 'exchange', name: 'Readable' },
      { type: 'agent_A1', name: 'Exchange' },
      { type: 'agent_B2', name: 'Exchange' },
      { type: 'agent_C3', name: 'Jira/Confluence' },
      { type: 'agent_D4', name: 'Größe Übersicht' },
      { type: 'agent_E5', name: '!!!' },
    ]);
    expect([...toName.values()]).toEqual([
      'exchange',
      'exchange-2',
      'exchange-3',
      'jira-confluence',
      'grosse-ubersicht',
      'subagent',
    ]);
  });

  it('resolves names back and passes other values through', () => {
    const aliases = buildSubagentTypeAliases([
      { type: 'agent_A1', name: 'Exchange' },
    ]);
    expect(resolveSubagentType(aliases, 'exchange')).toBe('agent_A1');
    expect(resolveSubagentType(aliases, 'agent_A1')).toBe('agent_A1');
    expect(resolveSubagentType(aliases, 'unknown')).toBe('unknown');
    expect(resolveSubagentType(aliases, undefined)).toBeUndefined();
  });
});
