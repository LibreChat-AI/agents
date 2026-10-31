import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';

const hasBunCheck =
  spawnSync('bun', ['check', '--help'], { stdio: 'ignore' }).status === 0;
let command = 'bun';
let args = ['check'];

if (!hasBunCheck) {
  try {
    const tsc = createRequire(import.meta.url).resolve('typescript/bin/tsc');
    command = process.execPath;
    args = [tsc, '--noEmit'];
  } catch {
    console.error(
      'typecheck: neither `bun check` nor typescript is available; run npm ci'
    );
    process.exit(1);
  }
  console.error(
    'typecheck: `bun check` is unavailable, falling back to tsc; install Bun 1.4.4 for faster checks'
  );
}

const result = spawnSync(command, [...args, ...process.argv.slice(2)], {
  stdio: 'inherit',
});
if (result.error) {
  console.error(`typecheck: ${result.error.message}`);
}
process.exit(result.status ?? 1);
