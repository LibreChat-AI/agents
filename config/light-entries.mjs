import path from 'node:path';
import { promisify } from 'node:util';
import { readFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire, isBuiltin, registerHooks } from 'node:module';

const root = path.resolve(fileURLToPath(import.meta.url), '../..');
const packageName = '@librechat/agents';
const run = promisify(execFile);

/**
 * Subpaths that must load without any third-party package, so a host such as
 * a code-intelligence daemon can use typed decisions and reranking without
 * pulling LangChain or provider SDKs into its process.
 */
export const lightSubpaths = Object.freeze(['./decisions', './rerank']);

/**
 * Load one built package subpath through its own exports map in a fresh Node
 * process, recording every module resolution. `packages` lists bare
 * specifiers that are neither Node built-ins nor the entry itself.
 */
export async function probe(subpath, format) {
  const { stdout } = await run(
    process.execPath,
    [fileURLToPath(import.meta.url), '--probe', subpath, format],
    { cwd: root, maxBuffer: 16 * 1024 * 1024 }
  );
  return JSON.parse(stdout);
}

function record(subpath) {
  const entry = `${packageName}${subpath.slice(1)}`;
  const modules = new Set();
  const packages = new Set();
  registerHooks({
    resolve(specifier, context, nextResolve) {
      const result = nextResolve(specifier, context);
      if (result.url.startsWith('file:')) {
        modules.add(path.relative(root, fileURLToPath(result.url)));
      }
      const relative = specifier.startsWith('.') || specifier.startsWith('/');
      const external =
        !relative &&
        !specifier.startsWith('file:') &&
        !isBuiltin(specifier) &&
        specifier !== entry;
      if (external) {
        packages.add(specifier);
      }
      return result;
    },
  });
  return { entry, modules, packages };
}

async function runProbe(subpath, format) {
  const { entry, modules, packages } = record(subpath);
  if (format === 'cjs') {
    createRequire(path.join(root, 'package.json'))(entry);
  } else {
    await import(entry);
  }
  process.stdout.write(
    JSON.stringify({
      entry,
      format,
      modules: [...modules].sort(),
      packages: [...packages].sort(),
    })
  );
}

const declarationSpecifier =
  /(?:\bfrom\s+|\bimport\s*\(\s*|^\s*import\s+)['"]([^'"]+)['"]/gm;

async function resolveDeclaration(fromFile, specifier) {
  const base = path.resolve(path.dirname(fromFile), specifier);
  const candidates = [
    base.replace(/\.(?:m|c)?js$/, '.d.ts'),
    `${base}.d.ts`,
    path.join(base, 'index.d.ts'),
  ];
  for (const candidate of candidates) {
    try {
      await readFile(candidate);
      return candidate;
    } catch {
      continue;
    }
  }
  return null;
}

/**
 * Walk the declaration graph from a subpath's `types` file. `external` lists
 * every non-relative specifier, including unresolved `@/` path aliases that
 * consumers cannot resolve; `missing` lists relative imports with no file.
 */
export async function walkDeclarations(subpath) {
  const manifest = JSON.parse(
    await readFile(path.join(root, 'package.json'), 'utf8')
  );
  const start = path.join(root, manifest.exports[subpath].types);
  const files = new Set();
  const external = new Set();
  const missing = new Set();
  const queue = [start];
  while (queue.length > 0) {
    const file = queue.pop();
    if (files.has(file)) {
      continue;
    }
    files.add(file);
    const source = await readFile(file, 'utf8');
    for (const [, specifier] of source.matchAll(declarationSpecifier)) {
      if (!specifier.startsWith('.')) {
        if (!isBuiltin(specifier)) {
          external.add(specifier);
        }
        continue;
      }
      const resolved = await resolveDeclaration(file, specifier);
      if (resolved === null) {
        missing.add(`${path.relative(root, file)} -> ${specifier}`);
        continue;
      }
      queue.push(resolved);
    }
  }
  return {
    files: [...files].map((file) => path.relative(root, file)).sort(),
    external: [...external].sort(),
    missing: [...missing].sort(),
  };
}

const invokedDirectly =
  process.argv[1] !== undefined &&
  pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url;

if (invokedDirectly && process.argv[2] === '--probe') {
  await runProbe(process.argv[3], process.argv[4]);
} else if (invokedDirectly) {
  for (const subpath of lightSubpaths) {
    for (const format of ['esm', 'cjs']) {
      const result = await probe(subpath, format);
      console.log(
        `${result.entry} (${format}): ${result.modules.length} modules, ${result.packages.length} packages${result.packages.length > 0 ? `: ${result.packages.join(', ')}` : ''}`
      );
    }
    const types = await walkDeclarations(subpath);
    console.log(
      `${packageName}${subpath.slice(1)} (types): ${types.files.length} declaration files, ${types.external.length} external specifiers`
    );
  }
}
