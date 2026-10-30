import { z } from 'zod';
import { z as z3 } from 'zod/v3';
import { $ZodString, globalRegistry } from 'zod/v4/core';
import { toJsonSchema } from './schema';

describe('toJsonSchema', () => {
  it('keeps Zod 4 parameters, tool names, and descriptions', () => {
    const schema = z.object({
      questions: z.array(z.object({ question: z.string() })).min(1),
    });

    expect(
      toJsonSchema(schema, 'ask_user_question', 'Ask related questions')
    ).toMatchObject({
      title: 'ask_user_question',
      description: 'Ask related questions',
      type: 'object',
      properties: {
        questions: {
          type: 'array',
          minItems: 1,
          items: {
            type: 'object',
            properties: { question: { type: 'string' } },
            required: ['question'],
          },
        },
      },
      required: ['questions'],
    });
  });

  it('converts Zod 4 inputs without requiring defaulted or optional fields', () => {
    const schema = z.object({
      query: z.string().transform((value) => value.length),
      limit: z.number().default(5),
      since: z.date().optional(),
    });

    expect(toJsonSchema(schema)).toMatchObject({
      $schema: 'http://json-schema.org/draft-07/schema#',
      properties: {
        query: { type: 'string' },
        limit: { type: 'number', default: 5 },
        since: {},
      },
      required: ['query'],
    });
  });

  it('overrides a Zod 4 description without changing the schema', () => {
    const schema = z
      .object({ value: z.string() })
      .describe('Original description');

    expect(toJsonSchema(schema, undefined, 'Tool description')).toMatchObject({
      description: 'Tool description',
    });
    expect(toJsonSchema(schema, '', '')).toMatchObject({
      description: 'Original description',
    });
    expect(schema.description).toBe('Original description');
  });

  it('keeps nested Zod 4 validation and descriptions unchanged', () => {
    const schema = z.object({
      filters: z.object({ query: z.string() }).describe('Search filters'),
    });
    const input = { filters: { query: 'files', extra: true } };

    expect(toJsonSchema(schema)).toMatchObject({
      properties: { filters: { description: 'Search filters' } },
    });
    expect(schema.parse(input)).toEqual({ filters: { query: 'files' } });
  });

  it('reads nested metadata from another Zod installation', () => {
    let separateZod = z;
    jest.isolateModules(() => {
      separateZod = jest.requireActual<typeof import('zod')>('zod').z;
    });
    const schema = separateZod.object({
      filters: separateZod
        .object({ query: separateZod.string().describe('Query text') })
        .meta({
          description: 'Search filters',
          examples: [{ query: 'files' }],
        }),
    });

    expect(toJsonSchema(schema, 'search')).toMatchObject({
      title: 'search',
      properties: {
        filters: {
          description: 'Search filters',
          examples: [{ query: 'files' }],
          properties: { query: { description: 'Query text' } },
        },
      },
    });
  });

  it('retains Zod 4 constraints', () => {
    const schema = z.object({
      id: z.string().regex(/^[a-z]+$/),
      count: z.number().int().min(1).max(4),
    });

    expect(toJsonSchema(schema)).toMatchObject({
      properties: {
        id: { type: 'string', pattern: '^[a-z]+$' },
        count: { type: 'integer', minimum: 1, maximum: 4 },
      },
    });
  });

  it.each([
    z.number().prefault(0),
    z.boolean().prefault(false),
    z.string().prefault(''),
    z.null().prefault(null),
  ])('keeps a falsy prefault in the input schema', (field) => {
    expect(toJsonSchema(z.object({ value: field }))).toMatchObject({
      properties: { value: { default: field.parse(undefined) } },
    });
  });

  it('keeps global metadata for core schemas without a meta accessor', () => {
    const schema = new $ZodString({ type: 'string' });
    globalRegistry.add(schema, { description: 'Core field' });

    expect(toJsonSchema(schema)).toMatchObject({
      type: 'string',
      description: 'Core field',
    });
  });

  it('keeps recursive Zod 4 references rooted when the tool is named', () => {
    type Tree = { name: string; children: Tree[] };
    const schema: z.ZodType<Tree> = z.lazy(() =>
      z.object({ name: z.string(), children: z.array(schema) })
    );

    expect(toJsonSchema(schema, 'tree')).toMatchObject({
      title: 'tree',
      type: 'object',
      properties: { children: { items: { $ref: '#' } } },
    });
  });

  it('preserves the named Zod 3 conversion contract', () => {
    expect(
      toJsonSchema(z3.object({ query: z3.string() }), 'search', 'Search files')
    ).toEqual({
      $ref: '#/definitions/search',
      definitions: {
        search: {
          type: 'object',
          properties: { query: { type: 'string' } },
          required: ['query'],
          additionalProperties: false,
          description: 'Search files',
        },
      },
      $schema: 'http://json-schema.org/draft-07/schema#',
    });
  });

  it('returns JSON Schema inputs unchanged', () => {
    const schema = {
      type: 'object',
      properties: { query: { type: 'string' } },
    };

    expect(toJsonSchema(schema, 'search', 'Search files')).toBe(schema);
  });
});
