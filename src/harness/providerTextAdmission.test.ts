import { runInNewContext } from 'node:vm';
import type {
  ProviderTextProtection,
  ProviderTextProtectionResult,
} from '@/protection/providerText';
import {
  ProviderTextAttempt,
  ProviderTextProtectionError,
  serializeProviderTextProtectionError,
  deserializeProviderTextProtectionError,
} from '@/protection/providerText';
import {
  createHostProviderTextService,
  createHostProviderTextProtection,
} from './providerText';
import { ProviderTextAdmission } from '@/protection/providerTextAdmission';

const identity = {
  runtimeId: 'runtime',
  agentId: 'agent',
  runId: 'run',
  userId: 'user',
};
const request = { version: 1 as const, identity, content: 'held' };
const approved = (content = 'safe'): ProviderTextProtectionResult => ({
  version: 1,
  ok: true,
  value: { content, replacements: 0, categories: [] },
});
const policy = (
  overrides: Partial<ProviderTextProtection> = {}
): ProviderTextProtection => ({
  version: 1,
  timeoutMs: 10000,
  maxAttemptBytes: 4096,
  maxBufferedBytes: 16384,
  classify: () => 'prose',
  inspect: () => approved(),
  ...overrides,
});
const deferred = <T>(): {
  promise: Promise<T>;
  resolve(value: T): void;
  reject(error: Error): void;
} => {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
};
const drained = (): Promise<void> =>
  new Promise((resolve) => setImmediate(resolve));
const empty = {
  inspections: 0,
  bufferedBytes: 0,
  retainedInspections: 0,
  retainedBytes: 0,
};

it.each(['resolve', 'reject'] as const)(
  'keeps cancelled ownership charged until the inspector %ss, without starving another scope',
  async (settlement) => {
    const held = deferred<ProviderTextProtectionResult>();
    const shared = policy({
      inspect: ({ content }) =>
        content === 'held' ? held.promise : approved(),
    });
    const admission = new ProviderTextAdmission({
      maxConcurrentInspections: 1,
      maxBufferedBytes: 4096,
    });
    const bound = createHostProviderTextService({
      identity,
      policy: shared,
      admission,
    });
    const controller = new AbortController();
    const result = bound.service
      .inspect(request, controller.signal)
      .catch((error: Error) => error);
    const charged = admission.snapshot();
    expect(charged.inspections).toBe(1);
    expect(charged.bufferedBytes).toBeGreaterThan(512);
    controller.abort();
    expect(await result).toMatchObject({ code: 'cancelled' });
    expect(admission.snapshot()).toEqual({
      ...charged,
      retainedInspections: 1,
      retainedBytes: charged.bufferedBytes,
    });

    const rebound = createHostProviderTextService({
      identity,
      policy: shared,
      admission,
    });
    await expect(rebound.service.inspect(request)).rejects.toMatchObject({
      code: 'overflow',
    });
    const otherIdentity = { ...identity, userId: 'other' };
    const otherAdmission = new ProviderTextAdmission({
      maxConcurrentInspections: 1,
      maxBufferedBytes: 4096,
    });
    const other = createHostProviderTextService({
      identity: otherIdentity,
      policy: shared,
      admission: otherAdmission,
    });
    await expect(
      other.service.inspect({
        ...request,
        identity: otherIdentity,
        content: 'other',
      })
    ).resolves.toBe('safe');
    expect(otherAdmission.snapshot()).toEqual(empty);
    expect(admission.snapshot().retainedInspections).toBe(1);

    if (settlement === 'resolve') held.resolve(approved('late PRIVATE'));
    else held.reject(new Error('PRIVATE failure'));
    await drained();
    expect(admission.snapshot()).toEqual(empty);
    await expect(
      rebound.service.inspect({ ...request, content: 'again' })
    ).resolves.toBe('safe');
  }
);

it('shares one admission scope across different runs and policies, including overlapping byte charges', async () => {
  const held = deferred<ProviderTextProtectionResult>();
  const limits = { maxConcurrentInspections: 10, maxBufferedBytes: 1024 };
  const admission = new ProviderTextAdmission(limits);
  limits.maxBufferedBytes = 100000;
  const a = createHostProviderTextService({
    identity,
    policy: policy({ inspect: () => held.promise }),
    admission,
  });
  const otherIdentity = { ...identity, runId: 'other' };
  const b = createHostProviderTextService({
    identity: otherIdentity,
    policy: policy(),
    admission,
  });
  const first = a.service.inspect(request);
  try {
    await expect(
      b.service.inspect({ ...request, identity: otherIdentity })
    ).rejects.toMatchObject({ code: 'overflow' });
    expect(admission.snapshot().inspections).toBe(1);
  } finally {
    held.resolve(approved());
    await first;
  }
  expect(admission.snapshot()).toEqual(empty);
  await expect(
    b.service.inspect({ ...request, identity: otherIdentity })
  ).resolves.toBe('safe');
});

it('enforces admission on canonical output as well as input and rolls back rejected attempts', async () => {
  const inspect = jest.fn(() => approved('x'.repeat(300)));
  const admission = new ProviderTextAdmission({
    maxConcurrentInspections: 1,
    maxBufferedBytes: 1024,
  });
  const bound = createHostProviderTextService({
    identity,
    policy: policy({ inspect }),
    admission,
  });
  await expect(bound.service.inspect(request)).rejects.toMatchObject({
    code: 'overflow',
  });
  expect(inspect).toHaveBeenCalledTimes(1);
  expect(admission.snapshot()).toEqual(empty);
  await expect(
    bound.service.inspect({ ...request, content: 'x'.repeat(300) })
  ).rejects.toMatchObject({ code: 'overflow' });
  expect(inspect).toHaveBeenCalledTimes(1);
  expect(admission.snapshot()).toEqual(empty);
});

it('uses an independent host deadline without cloning the shared policy budget', async () => {
  jest.useFakeTimers();
  const held = deferred<ProviderTextProtectionResult>();
  const admission = new ProviderTextAdmission({
    maxConcurrentInspections: 2,
    maxBufferedBytes: 4096,
  });
  let inspectorSignal: AbortSignal | undefined;
  const shared = policy({
    maxAttemptBytes: 1024,
    maxBufferedBytes: 1024,
    inspect: ({ signal }) => {
      inspectorSignal = signal;
      return held.promise;
    },
  });
  const bound = createHostProviderTextService({
    identity,
    policy: shared,
    admission,
    inspectionTimeoutMs: 20,
  });
  const other = createHostProviderTextService({ identity, policy: shared });
  expect(bound.descriptor.timeoutMs).toBe(10000);
  const result = bound.service.inspect(request).catch((error: Error) => error);
  try {
    await jest.advanceTimersByTimeAsync(20);
    expect(await result).toMatchObject({ code: 'timeout' });
    expect(inspectorSignal?.aborted).toBe(true);
    expect(admission.snapshot().retainedInspections).toBe(1);
    await expect(other.service.inspect(request)).rejects.toMatchObject({
      code: 'overflow',
    });
    held.resolve(approved());
    await jest.advanceTimersByTimeAsync(0);
    expect(admission.snapshot()).toEqual(empty);
    await expect(other.service.inspect(request)).resolves.toBe('safe');
  } finally {
    held.resolve(approved());
    jest.useRealTimers();
  }
});

it('cannot extend the policy deadline with a larger host inspection timeout', async () => {
  jest.useFakeTimers();
  const held = deferred<ProviderTextProtectionResult>();
  const bound = createHostProviderTextService({
    identity,
    policy: policy({ timeoutMs: 10, inspect: () => held.promise }),
    inspectionTimeoutMs: 1000,
  });
  const result = bound.service.inspect(request).catch((error: Error) => error);
  try {
    await jest.advanceTimersByTimeAsync(10);
    expect(await result).toMatchObject({ code: 'timeout' });
  } finally {
    held.resolve(approved());
    jest.useRealTimers();
  }
});

it.each(['missing', 'setup', 'cleanup'] as const)(
  'does not leak global or scoped admission on %s signal failure',
  async (failure) => {
    const admission = new ProviderTextAdmission({
      maxConcurrentInspections: 1,
      maxBufferedBytes: 512,
    });
    const shared = policy({ maxAttemptBytes: 512, maxBufferedBytes: 512 });
    const bound = createHostProviderTextService({
      identity,
      policy: shared,
      admission,
    });
    const signal =
      failure === 'missing'
        ? (JSON.parse('{}') as AbortSignal)
        : new AbortController().signal;
    if (failure === 'setup')
      jest.spyOn(signal, 'addEventListener').mockImplementation(() => {
        throw new Error('PRIVATE');
      });
    if (failure !== 'missing')
      jest.spyOn(signal, 'removeEventListener').mockImplementation(() => {
        throw new Error('PRIVATE');
      });
    const result = bound.service.inspect({ ...request, content: '' }, signal);
    if (failure === 'cleanup') await expect(result).resolves.toBe('');
    else await expect(result).rejects.toMatchObject({ code: 'incompatible' });
    expect(admission.snapshot()).toEqual(empty);
    await expect(
      bound.service.inspect({ ...request, content: '' })
    ).resolves.toBe('');
  }
);

it('does not acquire for malformed identity or pre-aborted calls and releases synchronous failures', async () => {
  const admission = new ProviderTextAdmission({
    maxConcurrentInspections: 1,
    maxBufferedBytes: 4096,
  });
  const inspect = jest.fn(() => {
    throw new Error('PRIVATE');
  });
  const bound = createHostProviderTextService({
    identity,
    policy: policy({ inspect }),
    admission,
  });
  await expect(
    bound.service.inspect({
      ...request,
      identity: { ...identity, userId: 'foreign' },
    })
  ).rejects.toMatchObject({ code: 'incompatible' });
  await expect(
    bound.service.inspect(request, AbortSignal.abort())
  ).rejects.toMatchObject({ code: 'cancelled' });
  expect(inspect).not.toHaveBeenCalled();
  await expect(bound.service.inspect(request)).rejects.toMatchObject({
    code: 'unavailable',
  });
  expect(admission.snapshot()).toEqual(empty);
});

it('rolls back an initial charge failure and forbids post-finish charges', () => {
  const admission = new ProviderTextAdmission({
    maxConcurrentInspections: 1,
    maxBufferedBytes: 511,
  });
  expect(
    () => new ProviderTextAttempt(policy(), undefined, { admission })
  ).toThrow(ProviderTextProtectionError);
  expect(admission.snapshot()).toEqual(empty);
  const valid = new ProviderTextAdmission({
    maxConcurrentInspections: 1,
    maxBufferedBytes: 1024,
  });
  const attempt = new ProviderTextAttempt(policy(), undefined, {
    admission: valid,
  });
  attempt.finish();
  attempt.finish();
  expect(() => attempt.append('late')).toThrow(ProviderTextProtectionError);
  expect(valid.snapshot()).toEqual(empty);
});

it.each([0, -1, 0.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1])(
  'rejects invalid admission limit %s',
  (value) => {
    for (const field of ['maxBufferedBytes', 'maxConcurrentInspections']) {
      expect(
        () =>
          new ProviderTextAdmission({
            maxBufferedBytes: 4096,
            maxConcurrentInspections: 1,
            [field]: value,
          })
      ).toThrow(ProviderTextProtectionError);
    }
  }
);

it.each([0, -1, 0.5, NaN, Infinity, 2147483648])(
  'rejects invalid host deadline %s at binding',
  (inspectionTimeoutMs) => {
    expect(() =>
      createHostProviderTextService({
        identity,
        policy: policy(),
        inspectionTimeoutMs,
      })
    ).toThrow(ProviderTextProtectionError);
  }
);

it('carries a safe typed policy failure across separate realms and a JSON transport', async () => {
  const error: Error = runInNewContext(
    'Object.assign(new Error(\'PRIVATE\'), { name: \'ProviderTextProtectionError\', code: \'blocked\' })'
  );
  expect(error).not.toBeInstanceOf(ProviderTextProtectionError);
  const wire = JSON.stringify(serializeProviderTextProtectionError(error));
  expect(wire).toBe('{"version":1,"ok":false,"error":{"code":"blocked"}}');
  const decoded = deserializeProviderTextProtectionError(JSON.parse(wire));
  expect(decoded).toBeInstanceOf(ProviderTextProtectionError);
  expect(decoded.code).toBe('blocked');
  const bound = createHostProviderTextService({ identity, policy: policy() });
  const worker = createHostProviderTextProtection({
    identity,
    descriptor: bound.descriptor,
    service: {
      version: 1,
      inspect: async () => {
        throw error;
      },
    },
  });
  const attempt = new ProviderTextAttempt(worker);
  try {
    attempt.append('PRIVATE');
    await expect(attempt.release()).rejects.toMatchObject({
      code: 'blocked',
      message: 'Provider text protection: blocked',
    });
  } finally {
    attempt.finish();
  }
});

it.each([
  null,
  {},
  { version: 2, ok: false, error: { code: 'blocked' } },
  { version: 1, ok: true, error: { code: 'blocked' } },
  { version: 1, ok: false, error: { code: 'PRIVATE' } },
])('fails closed on malformed wire failure %j', (failure) => {
  expect(deserializeProviderTextProtectionError(failure).code).toBe(
    'incompatible'
  );
});

it('never serializes arbitrary transport failures or hostile exception getters', () => {
  for (const error of [
    new Error('PRIVATE'),
    { code: 'blocked' },
    { name: 'ProviderTextProtectionError', code: 'PRIVATE' },
    {
      get name(): string {
        throw new Error('PRIVATE');
      },
    },
  ]) {
    expect(serializeProviderTextProtectionError(error)).toEqual({
      version: 1,
      ok: false,
      error: { code: 'unavailable' },
    });
  }
});
