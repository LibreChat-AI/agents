import { z } from 'zod';
import { z as z4 } from 'zod/v4';
import { z as currentZod } from 'zod-v4';
import { $ZodString, globalRegistry } from 'zod/v4/core';
import { toJsonSchema } from './schema';

describe('toJsonSchema', () => {
  it('keeps Zod 4 parameters, tool names, and descriptions', () => {
    const schema = z4.object({
      questions: z4.array(z4.object({ question: z4.string() })).min(1),
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
    const schema = z4.object({
      query: z4.string().transform((value) => value.length),
      limit: z4.number().default(5),
      since: z4.date().optional(),
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
    const schema = z4
      .object({ value: z4.string() })
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
    const schema = z4.object({
      filters: z4.object({ query: z4.string() }).describe('Search filters'),
    });
    const input = { filters: { query: 'files', extra: true } };

    expect(toJsonSchema(schema)).toMatchObject({
      properties: { filters: { description: 'Search filters' } },
    });
    expect(schema.parse(input)).toEqual({ filters: { query: 'files' } });
  });

  it('reads nested metadata from another Zod installation', () => {
    let separateZod = z4;
    jest.isolateModules(() => {
      separateZod = jest.requireActual<typeof import('zod/v4')>('zod/v4').z;
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

  it('retains current Zod 4 constraints', () => {
    const schema = currentZod.object({
      id: currentZod.string().regex(/^[a-z]+$/),
      count: currentZod.number().int().min(1).max(4),
    });

    expect(toJsonSchema(schema)).toMatchObject({
      properties: {
        id: { type: 'string', pattern: '^[a-z]+$' },
        count: { type: 'integer', minimum: 1, maximum: 4 },
      },
    });
  });

  it.each([
    z4.number().prefault(0),
    z4.boolean().prefault(false),
    z4.string().prefault(''),
    z4.null().prefault(null),
  ])('keeps a falsy early-Zod prefault in the input schema', (field) => {
    expect(toJsonSchema(z4.object({ value: field }))).toMatchObject({
      properties: { value: { default: field.parse(undefined) } },
    });
  });

  it('converts an early-Zod field inside a current-Zod object', () => {
    const query = z4.string().min(1).max(10).describe('Legacy query');
    const schema = currentZod.object({ query });

    expect(toJsonSchema(schema)).toMatchObject({
      properties: {
        query: {
          type: 'string',
          minLength: 1,
          maxLength: 10,
          description: 'Legacy query',
        },
      },
      required: ['query'],
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
    const schema: z4.ZodType<Tree> = z4.lazy(() =>
      z4.object({ name: z4.string(), children: z4.array(schema) })
    );

    expect(toJsonSchema(schema, 'tree')).toMatchObject({
      title: 'tree',
      type: 'object',
      properties: { children: { items: { $ref: '#' } } },
    });
  });

  it('preserves the named Zod 3 conversion contract', () => {
    expect(
      toJsonSchema(z.object({ query: z.string() }), 'search', 'Search files')
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
