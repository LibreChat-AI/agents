import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const KNOWN_EXTENSIONS = new Set(['.js', '.mjs', '.cjs', '.json', '.node']);

/**
 * Resolve an extensionless relative specifier inside a `.d.ts` file to its
 * Node16/NodeNext-compatible `.js` (or `/index.js`) specifier when the
 * corresponding declaration file exists on disk.
 */
export function resolveRelativeDtsSpecifier(fileDir, specifier) {
  if (!specifier.startsWith('./') && !specifier.startsWith('../')) {
    return specifier;
  }
  const ext = extname(specifier);
  if (KNOWN_EXTENSIONS.has(ext)) {
    return specifier;
  }

  const target = resolve(fileDir, specifier);
  if (existsSync(`${target}.d.ts`)) {
    return `${specifier}.js`;
  }
  if (existsSync(`${target}.d.mts`)) {
    return `${specifier}.mjs`;
  }
  if (existsSync(`${target}.d.cts`)) {
    return `${specifier}.cjs`;
  }
  if (existsSync(join(target, 'index.d.ts'))) {
    return `${specifier.replace(/\/$/, '')}/index.js`;
  }
  if (existsSync(join(target, 'index.d.mts'))) {
    return `${specifier.replace(/\/$/, '')}/index.mjs`;
  }
  return specifier;
}

const SPECIFIER_PATTERN =
  /(\b(?:from|import)\s*['"])(\.\.?\/[^'"]+)(['"])|(\bimport\s*\(\s*['"])(\.\.?\/[^'"]+)(['"]\s*\))/g;

export function rewriteDtsContent(content, fileDir) {
  return content.replace(
    SPECIFIER_PATTERN,
    (match, p1, s1, q1, p2, s2, q2) => {
      if (p1 !== undefined) {
        return `${p1}${resolveRelativeDtsSpecifier(fileDir, s1)}${q1}`;
      }
      return `${p2}${resolveRelativeDtsSpecifier(fileDir, s2)}${q2}`;
    },
  );
}

export function fixDtsExtensionsInDir(rootDir) {
  let filesUpdated = 0;
  const walk = (dir) => {
    for (const entry of readdirSync(dir)) {
      const fullPath = join(dir, entry);
      const st = statSync(fullPath);
      if (st.isDirectory()) {
        walk(fullPath);
      } else if (entry.endsWith('.d.ts') || entry.endsWith('.d.mts') || entry.endsWith('.d.cts')) {
        const original = readFileSync(fullPath, 'utf8');
        const updated = rewriteDtsContent(original, dirname(fullPath));
        if (updated !== original) {
          writeFileSync(fullPath, updated, 'utf8');
          filesUpdated += 1;
        }
      }
    }
  };
  if (existsSync(rootDir)) {
    walk(rootDir);
  }
  return filesUpdated;
}

const isMain =
  process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const targetDir = process.argv[2]
    ? resolve(process.argv[2])
    : resolve(dirname(fileURLToPath(import.meta.url)), '..', 'dist', 'types');
  fixDtsExtensionsInDir(targetDir);
}
