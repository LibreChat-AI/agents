import { toJSONSchema } from 'zod/v4/core';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { isZodSchemaV4 } from '@langchain/core/utils/types';
import type { $ZodType } from 'zod/v4/core';
import type { ZodTypeAny } from 'zod';

/** Checks if a schema is a Zod schema by looking for the _def property */
export function isZodSchema(schema: unknown): schema is ZodTypeAny {
  return (
    schema != null && typeof schema === 'object' && '_def' in (schema as object)
  );
}

/**
 * Converts a schema to JSON schema format.
 * Handles both Zod schemas (converts) and JSON schemas (passthrough).
 */
export function toJsonSchema(
  schema: unknown,
  name?: string,
  description?: string
): Record<string, unknown> {
  if (isZodSchemaV4(schema)) {
    return {
      ...toJSONSchema(schema as $ZodType, {
        target: 'draft-7',
        io: 'input',
        unrepresentable: 'any',
      }),
      ...(name != null && name !== '' ? { title: name } : {}),
      ...(description != null && description !== '' ? { description } : {}),
    };
  }
  if (isZodSchema(schema)) {
    const zodSchema = schema as ZodTypeAny & {
      describe: (desc: string) => ZodTypeAny;
    };
    const described =
      description != null && description !== ''
        ? zodSchema.describe(description)
        : schema;
    return zodToJsonSchema(
      described as Parameters<typeof zodToJsonSchema>[0],
      name ?? ''
    );
  }
  return schema as Record<string, unknown>;
}
