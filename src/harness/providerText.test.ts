import { HumanMessage } from '@langchain/core/messages';
import type { ProviderTextProtection } from '@/protection/providerText';
import {
  createHostProviderTextService,
  createHostProviderTextProtection,
} from './providerText';
import {
  ProviderTextAttempt,
  ProviderTextProtectionError,
} from '@/protection/providerText';
import { ProviderTextAdmission } from '@/protection/providerTextAdmission';
import { Run, Providers, FakeChatModel, GraphEvents } from '../index';

const identity = {
  runtimeId: 'runtime',
  agentId: 'agent',
  runId: 'run',
  userId: 'user',
  tenantId: 'tenant',
  conversationId: 'thread',
};
const policy = (
  overrides: Partial<ProviderTextProtection> = {}
): ProviderTextProtection => ({
  version: 1,
  timeoutMs: 1000,
  maxAttemptBytes: 16384,
  maxBufferedBytes: 32768,
  classify: () => 'prose',
  inspect: ({ content }) => ({
    version: 1,
    ok: true,
    value: {
      content: content.replaceAll('PRIVATE', '[redacted]'),
      replacements: 1,
      categories: [],
    },
  }),
  ...overrides,
});

it('uses host classification and only releases canonical text through a real SDK Run', async () => {
  const classify = jest.fn(() => 'prose' as const);
  const admission = new ProviderTextAdmission({
    maxConcurrentInspections: 1,
    maxBufferedBytes: 16384,
  });
  const bound = createHostProviderTextService({
    identity,
    policy: policy({ classify }),
    admission,
    inspectionTimeoutMs: 500,
  });
  const inspect = jest.fn(bound.service.inspect);
  const protection = createHostProviderTextProtection({
    identity,
    descriptor: bound.descriptor,
    service: {
      version: 1,
      inspect: async (request, signal) =>
        inspect(JSON.parse(JSON.stringify(request)), signal),
    },
  });
  const visible: string[] = [];
  const run = await Run.create({
    runId: identity.runId,
    returnContent: true,
    tokenCounter: () => 1,
    providerTextProtection: protection,
    customHandlers: {
      [GraphEvents.ON_MESSAGE_DELTA]: {
        handle: (_event, data): void => {
          visible.push(JSON.stringify(data));
        },
      },
    },
    graphConfig: {
      type: 'standard',
      llmConfig: { provider: Providers.OPENAI },
      agents: [
        { agentId: identity.agentId, provider: Providers.OPENAI, tools: [] },
      ],
    },
  });
  run.Graph!.overrideModel = new FakeChatModel({
    responses: ['PRIVATE prose'],
    sleep: 1,
  });
  const result = await run.processStream(
    { messages: [new HumanMessage('go')] },
    { version: 'v2', configurable: { thread_id: 'thread' } }
  );
  expect(JSON.stringify(result)).toContain('[redacted]');
  expect(JSON.stringify(result)).not.toContain('PRIVATE');
  expect(visible.join('')).not.toContain('PRIVATE');
  expect(classify).toHaveBeenCalledWith('PRIVATE prose');
  expect(inspect).toHaveBeenCalledTimes(1);
  expect(admission.snapshot()).toMatchObject({
    inspections: 0,
    bufferedBytes: 0,
  });
});

it.each([
  'runtimeId',
  'agentId',
  'runId',
  'userId',
  'tenantId',
  'conversationId',
] as const)('rejects a foreign %s before policy inspection', async (key) => {
  const inspect = jest.fn(policy().inspect);
  const bound = createHostProviderTextService({
    identity,
    policy: policy({ inspect }),
  });
  await expect(
    bound.service.inspect({
      version: 1,
      identity: { ...identity, [key]: 'foreign' },
      content: 'PRIVATE',
    })
  ).rejects.toMatchObject({ code: 'incompatible' });
  expect(inspect).not.toHaveBeenCalled();
});

it('rejects unsupported host classification before inspection', async () => {
  const inspect = jest.fn(policy().inspect);
  const bound = createHostProviderTextService({
    identity,
    policy: policy({ classify: () => 'unsupported', inspect }),
  });
  const worker = createHostProviderTextProtection({ identity, ...bound });
  const attempt = new ProviderTextAttempt(worker);
  try {
    attempt.append('{"secret":"PRIVATE"}');
    await expect(attempt.release()).rejects.toMatchObject({
      code: 'unsupported',
    });
    expect(inspect).not.toHaveBeenCalled();
  } finally {
    attempt.finish();
  }
});

it('retains the shared host budget while cancelled inspection ignores its signal', async () => {
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  let started!: () => void;
  const ready = new Promise<void>((resolve) => {
    started = resolve;
  });
  let signal: AbortSignal | undefined;
  const bound = createHostProviderTextService({
    identity,
    policy: policy({
      maxAttemptBytes: 1024,
      maxBufferedBytes: 1024,
      inspect: async (input) => {
        signal = input.signal;
        started();
        await held;
        return {
          version: 1,
          ok: true,
          value: { content: 'safe', replacements: 0, categories: [] },
        };
      },
    }),
  });
  const controller = new AbortController();
  const request = { version: 1 as const, identity, content: 'text' };
  const first = bound.service.inspect(request, controller.signal);
  const rejected = first.catch((error: Error) => error);
  await ready;
  controller.abort();
  expect(await rejected).toMatchObject({ code: 'cancelled' });
  expect(signal?.aborted).toBe(true);
  await expect(bound.service.inspect(request)).rejects.toMatchObject({
    code: 'overflow',
  });
  release();
  await new Promise((resolve) => setImmediate(resolve));
  await expect(bound.service.inspect(request)).resolves.toBe('safe');
});

it('preserves structured policy failures and sanitizes transport failures', async () => {
  const descriptor = createHostProviderTextService({
    identity,
    policy: policy(),
  }).descriptor;
  for (const error of [
    new ProviderTextProtectionError('blocked'),
    new Error('PRIVATE credential'),
  ]) {
    const worker = createHostProviderTextProtection({
      identity,
      descriptor,
      service: {
        version: 1,
        inspect: async () => {
          throw error;
        },
      },
    });
    const attempt = new ProviderTextAttempt(worker);
    try {
      attempt.append('prose');
      await expect(attempt.release()).rejects.toMatchObject({
        code:
          error instanceof ProviderTextProtectionError
            ? 'blocked'
            : 'unavailable',
      });
    } finally {
      attempt.finish();
    }
  }
});

it('snapshots identity and rejects incompatible descriptors', async () => {
  const source = { ...identity };
  const bound = createHostProviderTextService({
    identity: source,
    policy: policy(),
  });
  const worker = createHostProviderTextProtection({
    identity: source,
    ...bound,
  });
  source.userId = 'foreign';
  const attempt = new ProviderTextAttempt(worker);
  try {
    attempt.append('PRIVATE');
    await expect(attempt.release()).resolves.toBe('[redacted]');
  } finally {
    attempt.finish();
  }
  expect(() =>
    createHostProviderTextProtection({
      identity,
      ...bound,
      descriptor: { ...bound.descriptor, timeoutMs: 0 },
    })
  ).toThrow();
});

it('enforces host limits even when a caller omits or enlarges its local policy', async () => {
  const inspect = jest.fn(policy().inspect);
  const bound = createHostProviderTextService({
    identity,
    policy: policy({ inspect, maxAttemptBytes: 1024 }),
  });
  await expect(
    bound.service.inspect({ version: 1, identity, content: 'x'.repeat(1000) })
  ).rejects.toMatchObject({ code: 'overflow' });
  expect(inspect).not.toHaveBeenCalled();
});

it('shares raw-text admission across independent bindings to the same host policy', async () => {
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  let started!: () => void;
  const ready = new Promise<void>((resolve) => {
    started = resolve;
  });
  const shared = policy({
    maxAttemptBytes: 1024,
    maxBufferedBytes: 1024,
    inspect: async () => {
      started();
      await held;
      return {
        version: 1,
        ok: true,
        value: { content: 'safe', replacements: 0, categories: [] },
      };
    },
  });
  const a = createHostProviderTextService({ identity, policy: shared });
  const other = { ...identity, runId: 'other' };
  const b = createHostProviderTextService({ identity: other, policy: shared });
  const request = { version: 1 as const, identity, content: 'text' };
  const first = a.service.inspect(request);
  try {
    await ready;
    await expect(
      b.service.inspect({ ...request, identity: other })
    ).rejects.toMatchObject({ code: 'overflow' });
  } finally {
    release();
    await first;
  }
});

it('times out an unavailable host inspector without releasing any candidate text', async () => {
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  const bound = createHostProviderTextService({
    identity,
    policy: policy({
      timeoutMs: 20,
      inspect: async () => {
        await held;
        return {
          version: 1,
          ok: true,
          value: { content: 'late PRIVATE', replacements: 0, categories: [] },
        };
      },
    }),
  });
  // Keep the process alive while the policy's unref'ed deadline runs.
  const timer = setTimeout(() => undefined, 1000);
  try {
    await expect(
      bound.service.inspect({ version: 1, identity, content: 'PRIVATE' })
    ).rejects.toMatchObject({ code: 'timeout' });
  } finally {
    release();
    clearTimeout(timer);
  }
});

it.each([
  null,
  {},
  { version: 1, content: 'x' },
  { version: 1, content: 'x', identity: null },
  { version: 2, identity, content: 'x' },
  { version: '1', identity, content: 'x' },
  { version: 1, identity, content: 42 },
  { version: 1, identity: { ...identity, tenantId: null }, content: 'x' },
])(
  'rejects malformed request %j with a typed contract failure',
  async (input) => {
    const inspect = jest.fn(policy().inspect);
    const bound = createHostProviderTextService({
      identity,
      policy: policy({ inspect }),
    });
    // Model deserialized input at a transport boundary, before its own schema checks.
    const request = JSON.parse(JSON.stringify(input)) as Parameters<
      typeof bound.service.inspect
    >[0];
    await expect(bound.service.inspect(request)).rejects.toBeInstanceOf(
      ProviderTextProtectionError
    );
    await expect(bound.service.inspect(request)).rejects.toMatchObject({
      code: 'incompatible',
    });
    expect(inspect).not.toHaveBeenCalled();
  }
);

it.each(['service', 'descriptor'] as const)(
  'rejects incompatible %s versions at worker binding',
  (field) => {
    const bound = createHostProviderTextService({ identity, policy: policy() });
    for (const version of [0, 2, '1', null]) {
      const options = JSON.parse(
        JSON.stringify({ identity, descriptor: bound.descriptor })
      ) as { identity: typeof identity; descriptor: typeof bound.descriptor };
      const changed = JSON.parse(JSON.stringify({ version })) as { version: 1 };
      expect(() =>
        createHostProviderTextProtection({
          ...options,
          descriptor:
            field === 'descriptor'
              ? { ...options.descriptor, ...changed }
              : options.descriptor,
          service: {
            ...bound.service,
            ...(field === 'service' ? changed : {}),
          },
        })
      ).toThrow(ProviderTextProtectionError);
    }
  }
);

it('forwards worker attempt cancellation to the service and discards a late reply', async () => {
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  let started!: () => void;
  const ready = new Promise<void>((resolve) => {
    started = resolve;
  });
  let serviceSignal: AbortSignal | undefined;
  const bound = createHostProviderTextService({ identity, policy: policy() });
  const worker = createHostProviderTextProtection({
    identity,
    descriptor: bound.descriptor,
    service: {
      version: 1,
      inspect: async (_request, signal) => {
        serviceSignal = signal;
        started();
        await held;
        return 'late PRIVATE';
      },
    },
  });
  const controller = new AbortController();
  const attempt = new ProviderTextAttempt(worker, controller.signal);
  try {
    attempt.append('PRIVATE');
    const result = attempt.release();
    const rejected = result.catch((error: Error) => error);
    await ready;
    expect(serviceSignal).toBe(attempt.signal);
    controller.abort();
    expect(serviceSignal?.aborted).toBe(true);
    expect(await rejected).toMatchObject({ code: 'cancelled' });
  } finally {
    release();
    attempt.finish();
  }
});
