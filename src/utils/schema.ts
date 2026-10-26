import { zodToJsonSchema } from 'zod-to-json-schema';
import { isZodSchemaV4 } from '@langchain/core/utils/types';
import { globalRegistry, registry, toJSONSchema } from 'zod/v4/core';
import type { $ZodType, GlobalMeta } from 'zod/v4/core';
import type { ZodType as ZodV4Type } from 'zod/v4';
import type { ZodTypeAny } from 'zod';

/** Separate Zod installations may not share their global metadata registry. */
const schemaMetadata = registry<GlobalMeta>();
schemaMetadata.get = (schema): GlobalMeta | undefined => {
  const annotated = schema as $ZodType & Partial<Pick<ZodV4Type, 'meta'>>;
  return typeof annotated.meta === 'function'
    ? annotated.meta()
    : globalRegistry.get(schema);
};

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
        metadata: schemaMetadata,
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
