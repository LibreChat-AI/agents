import path from 'node:path';
import assert from 'node:assert/strict';
import test from 'node:test';
import { readdir, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { format } from 'prettier';

const workflowsDirectory = fileURLToPath(
  new URL('../.github/workflows/', import.meta.url)
);

test('workflow files have valid YAML and unique mapping keys', async () => {
  const files = (await readdir(workflowsDirectory)).filter((file) =>
    /\.ya?ml$/.test(file)
  );

  await Promise.all(
    files.map(async (file) => {
      const source = await readFile(
        path.join(workflowsDirectory, file),
        'utf8'
      );
      await assert.doesNotReject(format(source, { parser: 'yaml' }), file);
    })
  );
});

test('duplicate environment keys invalidate a workflow', async () => {
  await assert.rejects(
    format('env:\n  OPENAI_BASE_URL: first\n  OPENAI_BASE_URL: second\n', {
      parser: 'yaml',
    }),
    /Map keys must be unique/
  );
});
