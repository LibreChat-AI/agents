// src/verification/index.ts
//
// Verified completion: an append-only run ledger written through hooks, and a
// Stop hook that closes a run only when its declared requirements pass.
export { JsonlRunLedger, MemoryRunLedger, sha256 } from './ledger';
export { registerLedgerHooks, finalAssistantText } from './ledgerHooks';
export { createCompletionGate } from './completionGate';
export { createOwnedJobManager } from './ownedJobs';
export type { OwnedJobManager, OwnedJobResult, OwnedJobOutput } from './ownedJobs';
export type { CompletionGate, CompletionGateOptions } from './completionGate';
export {
  artifactExists,
  jsonFileValid,
  commandPasses,
  localCommandRunner,
} from './verifiers';
export type { CommandOutcome, CommandRunner } from './verifiers';
export { codeApiVerdictRunner } from './codeApiVerdict';
export { createCodeApiVerificationSession } from './codeApiVerificationSession';
export type {
  CodeApiVerificationSession,
  CodeApiVerificationSessionOptions,
  CodeApiVerificationRecovery,
  CodeApiVerificationProgress,
} from './codeApiVerificationSession';
export type {
  CodeApiVerdictRunnerOptions,
  VerdictPublicKey,
} from './codeApiVerdict';
export type {
  LedgerKind,
  LedgerActor,
  LedgerArtifact,
  LedgerEntry,
  LedgerRow,
  LedgerPayload,
  RunLedger,
  CompletionRequirement,
  VerdictAttestation,
  VerifierVerdict,
  VerifierContext,
  VerifierDefinition,
  AdmittedMark,
  CompletionState,
  VerificationSnapshot,
  VerificationReceipt,
} from './types';
