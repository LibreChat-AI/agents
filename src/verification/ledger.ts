import { join } from 'path';
import { createHash } from 'crypto';
import {
  closeSync,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  writeFileSync,
  writeSync,
} from 'fs';
import type { LedgerEntry, LedgerPayload, LedgerRow, RunLedger } from './types';

export function sha256(content: string | Uint8Array): string {
  return createHash('sha256').update(content).digest('hex');
}

/**
 * Shared bookkeeping for ledgers: contiguous sequence numbers and the
 * fail-closed health flag. A ledger that failed one write stays unhealthy for
 * the rest of the run, so no later step can be admitted on a log with a gap.
 */
abstract class BaseRunLedger implements RunLedger {
  protected nextSeq: number;
  private failure: string | undefined;

  protected constructor(nextSeq: number) {
    this.nextSeq = nextSeq;
  }

  get healthy(): boolean {
    return this.failure === undefined;
  }

  get failureReason(): string | undefined {
    return this.failure;
  }

  append(entry: LedgerEntry, payload?: LedgerPayload): LedgerRow {
    if (this.failure !== undefined) {
      throw new Error(`ledger unavailable: ${this.failure}`);
    }
    try {
      const artifact =
        payload === undefined
          ? entry.artifact
          : { label: payload.label, sha256: this.storeBlob(payload.content) };
      const row: LedgerRow = {
        ...entry,
        ...(artifact !== undefined ? { artifact } : {}),
        seq: this.nextSeq,
        recordedAt: new Date().toISOString(),
      };
      this.writeRow(row);
      this.nextSeq += 1;
      return row;
    } catch (error) {
      this.failure = error instanceof Error ? error.message : String(error);
      throw error;
    }
  }

  abstract rows(): LedgerRow[];
  abstract blob(hash: string): string | undefined;
  protected abstract writeRow(row: LedgerRow): void;
  protected abstract storeBlob(content: string): string;
}

/** In-memory ledger for hosts without a filesystem and for tests. */
export class MemoryRunLedger extends BaseRunLedger {
  private readonly stored: LedgerRow[] = [];
  private readonly blobs = new Map<string, string>();

  constructor() {
    super(0);
  }

  rows(): LedgerRow[] {
    return [...this.stored];
  }

  blob(hash: string): string | undefined {
    return this.blobs.get(hash);
  }

  protected writeRow(row: LedgerRow): void {
    this.stored.push(row);
  }

  protected storeBlob(content: string): string {
    const hash = sha256(content);
    this.blobs.set(hash, content);
    return hash;
  }
}

/**
 * Append-only `ledger.jsonl` plus content-addressed `blobs/<sha256>` in `dir`.
 * Every row is flushed to disk before `append` returns. Reopening an existing
 * ledger continues its sequence and refuses a log whose rows are not
 * contiguous from 0.
 */
export class JsonlRunLedger extends BaseRunLedger {
  readonly path: string;
  private readonly blobDir: string;

  constructor(dir: string) {
    const path = join(dir, 'ledger.jsonl');
    mkdirSync(join(dir, 'blobs'), { recursive: true });
    super(JsonlRunLedger.verifiedLength(path));
    this.path = path;
    this.blobDir = join(dir, 'blobs');
  }

  /** Number of rows, after checking that `seq` runs 0..n-1 without gaps. */
  private static verifiedLength(path: string): number {
    if (!existsSync(path)) return 0;
    const lines = readFileSync(path, 'utf8')
      .split('\n')
      .filter((l) => l !== '');
    lines.forEach((line, index) => {
      const seq = (JSON.parse(line) as { seq?: unknown }).seq;
      if (seq !== index) {
        throw new Error(
          `ledger ${path} is not contiguous at row ${index} (seq ${String(seq)})`
        );
      }
    });
    return lines.length;
  }

  rows(): LedgerRow[] {
    if (!existsSync(this.path)) return [];
    return readFileSync(this.path, 'utf8')
      .split('\n')
      .filter((l) => l !== '')
      .map((l) => JSON.parse(l) as LedgerRow);
  }

  blob(hash: string): string | undefined {
    const path = join(this.blobDir, hash);
    if (!/^[0-9a-f]{64}$/.test(hash) || !existsSync(path)) return undefined;
    const content = readFileSync(path, 'utf8');
    // A blob is only returned when it still matches its name.
    return sha256(content) === hash ? content : undefined;
  }

  protected writeRow(row: LedgerRow): void {
    const fd = openSync(this.path, 'a');
    try {
      writeSync(fd, `${JSON.stringify(row)}\n`);
      fsyncSync(fd);
    } finally {
      closeSync(fd);
    }
  }

  protected storeBlob(content: string): string {
    const hash = sha256(content);
    const path = join(this.blobDir, hash);
    try {
      writeFileSync(path, content, { flag: 'wx' });
    } catch (error) {
      // Same name means same content: another call stored this blob first.
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
    }
    return hash;
  }
}
