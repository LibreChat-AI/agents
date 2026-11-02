/**
 * Host-owned evidence for declared completion checks, with optional tool
 * auditing and exact-state receipt reuse. The original marks and ledger
 * design draws on Octopus-SI-OS (`docs/contracts/VERIFIERS.md`).
 */

export type LedgerKind =
  | 'tool_call'
  | 'tool_result'
  | 'tool_failure'
  | 'permission_denied'
  | 'stop_requested'
  | 'verification_failed'
  | 'verification_receipt'
  | 'mark_admitted'
  | 'completion_unverified'
  | 'run_completed';

/** Who caused the row: the model's request, a tool's effect, or the runtime itself. */
export type LedgerActor = 'model' | 'tool' | 'runtime';

export interface LedgerArtifact {
  label: string;
  sha256: string;
}

export interface LedgerEntry {
  runId: string;
  kind: LedgerKind;
  actor: LedgerActor;
  artifact?: LedgerArtifact;
  metadata?: Record<string, unknown>;
}

export interface LedgerRow extends LedgerEntry {
  /** Contiguous from 0 within one ledger. */
  seq: number;
  recordedAt: string;
}

/** Content stored as a blob and referenced from the row by its sha256. */
export interface LedgerPayload {
  label: string;
  content: string;
}

export interface RunLedger {
  /** False once any write failed; an unhealthy ledger admits no marks. */
  readonly healthy: boolean;
  readonly failureReason?: string;
  append(entry: LedgerEntry, payload?: LedgerPayload): LedgerRow;
  rows(): LedgerRow[];
  blob(hash: string): string | undefined;
}

/** One requirement on the run's final answer: this mark, through this verifier. */
export interface CompletionRequirement {
  mark: string;
  verifier: string;
  /** Host-owned requirement version. Required for evidence reuse. */
  version?: string;
  /** Variables for the verifier. Never a command: commands live in verifier definitions. */
  params?: Record<string, unknown>;
}

/** Signed evidence returned by a remote verifier such as the Code API. */
export interface VerdictAttestation {
  issuer: string;
  keyId: string;
  token: string;
  trust: 'paired' | 'local';
}

export interface VerifierVerdict {
  passed: boolean;
  details?: Record<string, unknown>;
  /** Short text shown to the model when the check fails. */
  feedback?: string;
  attestation?: VerdictAttestation;
  /** Raw evidence retained outside model context in the ledger's artifact store. */
  artifact?: LedgerPayload;
}

export interface VerifierContext {
  runId: string;
  requirement: CompletionRequirement;
  params: Record<string, unknown>;
  /** The final assistant text the run is trying to finish with. */
  finalText: string | null;
  signal: AbortSignal;
}

export interface VerifierDefinition {
  id: string;
  /** Ceiling for one evaluation; `params.timeout_ms` may only narrow it. */
  timeoutMs?: number;
  defaultParams?: Record<string, unknown>;
  /** Opt in only for deterministic checks over the host's captured state. */
  reuse?: { version: string; includeFinalText?: boolean };
  run: (context: VerifierContext) => Promise<VerifierVerdict>;
}

export interface AdmittedMark {
  mark: string;
  verifier: string;
  seq: number | null;
  attempt: number;
  attestation?: VerdictAttestation;
  receiptId?: string;
  reused?: boolean;
}

/** A trusted host token covering dirty files, dependencies and external state. */
export interface VerificationSnapshot {
  tenantId: string;
  workspaceId: string;
  /** Monotonic mutation token or immutable snapshot identity, never just Git HEAD. */
  revision: string;
  environmentId: string;
}

export interface VerificationReceipt {
  schemaVersion: 1;
  id: string;
  inputHash: string;
  runId: string;
  mark: string;
  requirementVersion: string;
  verifier: string;
  verifierVersion: string;
  snapshot: VerificationSnapshot;
  attempt: number;
  verdict: VerifierVerdict;
}

export interface CompletionState {
  /** All declared checks passed in the admitted evaluation, possibly using current receipts. */
  verified: boolean;
  /** Evaluations run so far. */
  attempts: number;
  marks: AdmittedMark[];
  /** Requirements without a mark from the last evaluation. */
  missing: string[];
  /** Why the run ended unverified, when it did. */
  unverifiedReason?: string;
  /** Owned operations whose actual completion has not been observed. */
  pendingOperations?: string[];
  /** State observed at admission, not a promise that external state cannot change later. */
  snapshot?: VerificationSnapshot;
}
