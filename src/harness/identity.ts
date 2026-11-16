import { z } from 'zod';

const requiredIdentityField = z
  .string()
  .refine((value) => value.trim().length > 0);

/** Established by the authenticated host, never inferred from model arguments. */
export const executionIdentitySchema = z.object({
  runtimeId: requiredIdentityField,
  agentId: requiredIdentityField,
  runId: requiredIdentityField,
  userId: requiredIdentityField,
  tenantId: z.string().optional(),
  conversationId: z.string().optional(),
});

export type ExecutionIdentity = z.infer<typeof executionIdentitySchema>;
const identityKeys = executionIdentitySchema.keyof().options;

/** Validates and snapshots identity at the host binding boundary. */
export function snapshotExecutionIdentity(
  source: Readonly<ExecutionIdentity>
): Readonly<ExecutionIdentity> {
  const result = executionIdentitySchema.safeParse(source);
  if (!result.success) throw new Error('Incomplete execution identity');
  return Object.freeze(result.data);
}

export function matchesExecutionIdentity(
  a: Readonly<ExecutionIdentity>,
  b: Readonly<ExecutionIdentity>
): boolean {
  return identityKeys.every((key) => a[key] === b[key]);
}
