import ts from 'typescript';
import { dirname, join, resolve } from 'node:path';
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';

const relative = /^\.{1,2}(?:\/|$)/;
const explicitExtension = /\.(?:[cm]?[jt]s|[jt]sx|json)$/;

function moduleSpecifier(node) {
  if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) {
    return node.moduleSpecifier;
  }
  if (ts.isImportTypeNode(node) && ts.isLiteralTypeNode(node.argument)) {
    return node.argument.literal;
  }
  if (ts.isExternalModuleReference(node)) {
    return node.expression;
  }
  if (ts.isModuleDeclaration(node)) {
    return node.name;
  }
}

function suffixFor(specifier, directory, declarations) {
  if (!relative.test(specifier) || explicitExtension.test(specifier)) {
    return '';
  }
  const target = resolve(directory, specifier);
  const trailingSlash = specifier.endsWith('/');
  if (!trailingSlash && declarations.has(`${target}.d.ts`)) {
    return '.js';
  }
  if (declarations.has(join(target, 'index.d.ts'))) {
    return trailingSlash ? 'index.js' : '/index.js';
  }
  return '';
}

function rewrite(file, source, declarations) {
  const directory = dirname(file);
  const ast = ts.createSourceFile(file, source, ts.ScriptTarget.Latest);
  let cursor = 0;
  let result = '';

  function visit(node) {
    const token = moduleSpecifier(node);
    if (token && ts.isStringLiteral(token)) {
      const suffix = suffixFor(token.text, directory, declarations);
      if (suffix) {
        const offset = token.end - 1;
        result += source.slice(cursor, offset) + suffix;
        cursor = offset;
      }
    }
    ts.forEachChild(node, visit);
  }

  visit(ast);
  return result + source.slice(cursor);
}

const directory = resolve(process.argv[2]);
const declarations = new Set(
  readdirSync(directory, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith('.d.ts'))
    .map((entry) => join(entry.parentPath, entry.name))
);
for (const file of declarations) {
  const source = readFileSync(file, 'utf8');
  const result = rewrite(file, source, declarations);
  if (result !== source) {
    writeFileSync(file, result);
  }
}
