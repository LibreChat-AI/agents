import type { ProviderTextProtection } from '@/protection/providerText';
import {
  createHostProviderTextService,
  createHostProviderTextProtection,
} from './providerText';
import { snapshotExecutionIdentity, createHostToolBindings } from './index';
import { ProviderTextProtectionError } from '@/protection/providerText';

const identity = {
  runtimeId: 'runtime',
  agentId: 'agent',
  runId: 'run',
  userId: 'user',
};
const policy: ProviderTextProtection = {
  version: 1,
  timeoutMs: 1000,
  maxAttemptBytes: 1024,
  maxBufferedBytes: 2048,
  classify: () => 'prose',
  inspect: ({ content }) => ({
    version: 1,
    ok: true,
    value: { content, replacements: 0, categories: [] },
  }),
};
const service = { version: 1 as const, invoke: async () => [] };

it.each(['runtimeId', 'agentId', 'runId', 'userId'] as const)(
  'validates required %s in every binding',
  (field) => {
    for (const value of ['', '  ', null, 42, undefined]) {
      const invalid = JSON.parse(
        JSON.stringify({ ...identity, [field]: value })
      ) as typeof identity;
      expect(() => snapshotExecutionIdentity(invalid)).toThrow(
        'Incomplete execution identity'
      );
      expect(() =>
        createHostToolBindings({ identity: invalid, tools: [], service })
      ).toThrow('Incomplete execution identity');
      expect(() =>
        createHostProviderTextService({ identity: invalid, policy })
      ).toThrow(ProviderTextProtectionError);
      expect(() =>
        createHostProviderTextProtection({
          identity: invalid,
          descriptor: policy,
          service: { version: 1, inspect: async () => '' },
        })
      ).toThrow(ProviderTextProtectionError);
    }
  }
);

it('snapshots only identity fields without trimming or normalizing optional nulls', () => {
  const source = { ...identity, userId: ' user ', extra: 'private' };
  const result = snapshotExecutionIdentity(source);
  source.userId = 'changed';
  expect(result).toEqual({ ...identity, userId: ' user ' });
  expect(Object.isFrozen(result)).toBe(true);
  const invalid = JSON.parse(
    JSON.stringify({ ...identity, tenantId: null })
  ) as typeof identity;
  expect(() => snapshotExecutionIdentity(invalid)).toThrow(
    'Incomplete execution identity'
  );
});
