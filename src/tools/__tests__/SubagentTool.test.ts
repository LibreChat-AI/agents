import { describe, it, expect } from '@jest/globals';
import type { SubagentConfig } from '@/types';
import {
  SubagentToolName,
  SubagentToolDescription,
  SubagentToolDefinition,
  SubagentToolSchema,
  createSubagentToolDefinition,
  buildSubagentToolParams,
} from '../SubagentTool';
import { Constants } from '@/common';

describe('SubagentTool', () => {
  describe('schema structure', () => {
    it('has description as required string property', () => {
      expect(SubagentToolSchema.properties.description.type).toBe('string');
      expect(SubagentToolSchema.required).toContain('description');
    });

    it('has subagent_type as required string property', () => {
      expect(SubagentToolSchema.properties.subagent_type.type).toBe('string');
      expect(SubagentToolSchema.required).toContain('subagent_type');
    });

    it('is an object type schema', () => {
      expect(SubagentToolSchema.type).toBe('object');
    });
  });

  describe('SubagentToolDefinition', () => {
    it('has correct name', () => {
      expect(SubagentToolDefinition.name).toBe(Constants.SUBAGENT);
    });

    it('references the same schema object', () => {
      expect(SubagentToolDefinition.parameters).toBe(SubagentToolSchema);
    });

    it('has a non-empty description', () => {
      expect(SubagentToolDefinition.description).toBe(SubagentToolDescription);
      expect(SubagentToolDefinition.description!.length).toBeGreaterThan(0);
    });
  });

  describe('SubagentToolName', () => {
    it('equals Constants.SUBAGENT', () => {
      expect(SubagentToolName).toBe('subagent');
      expect(SubagentToolName).toBe(Constants.SUBAGENT);
    });
  });

  describe('createSubagentToolDefinition', () => {
    const configs: SubagentConfig[] = [
      {
        type: 'researcher',
        name: 'Research Agent',
        description: 'Searches and summarizes information',
      },
      {
        type: 'coder',
        name: 'Coding Agent',
        description: 'Writes and reviews code',
      },
    ];

    it('populates subagent_type enum from configs', () => {
      const def = createSubagentToolDefinition(configs);
      const schema = def.parameters as Record<string, unknown>;
      const props = schema.properties as Record<
        string,
        Record<string, unknown>
      >;
      expect(props.subagent_type.enum).toEqual(['researcher', 'coder']);
    });

    it('includes type descriptions in tool description', () => {
      const def = createSubagentToolDefinition(configs);
      expect(def.description).toContain('"researcher" (Research Agent)');
      expect(def.description).toContain('"coder" (Coding Agent)');
      expect(def.description).toContain('Searches and summarizes information');
      expect(def.description).toContain('Writes and reviews code');
    });

    it('has correct name', () => {
      const def = createSubagentToolDefinition(configs);
      expect(def.name).toBe(Constants.SUBAGENT);
    });

    it('has required description and subagent_type fields', () => {
      const def = createSubagentToolDefinition(configs);
      const schema = def.parameters as Record<string, unknown>;
      expect(schema.required).toContain('description');
      expect(schema.required).toContain('subagent_type');
    });

    it('works with single config', () => {
      const def = createSubagentToolDefinition([configs[0]]);
      const schema = def.parameters as Record<string, unknown>;
      const props = schema.properties as Record<
        string,
        Record<string, unknown>
      >;
      expect(props.subagent_type.enum).toEqual(['researcher']);
    });
  });

  describe('buildSubagentToolParams', () => {
    const configs: SubagentConfig[] = [
      {
        type: 'researcher',
        name: 'Research Agent',
        description: 'Searches and summarizes information',
      },
      {
        type: 'coder',
        name: 'Coding Agent',
        description: 'Writes and reviews code',
      },
    ];

    it('returns name matching Constants.SUBAGENT', () => {
      const params = buildSubagentToolParams(configs);
      expect(params.name).toBe(Constants.SUBAGENT);
    });

    it('schema has enum populated from config types', () => {
      const params = buildSubagentToolParams(configs);
      const props = params.schema.properties as Record<
        string,
        Record<string, unknown>
      >;
      expect(props.subagent_type.enum).toEqual(['researcher', 'coder']);
    });

    it('description includes type listings', () => {
      const params = buildSubagentToolParams(configs);
      expect(params.description).toContain('"researcher" (Research Agent)');
      expect(params.description).toContain('"coder" (Coding Agent)');
    });

    it('only advertises detached execution when the host enables it', () => {
      const foreground = buildSubagentToolParams(configs);
      const background = buildSubagentToolParams(configs, {
        background: true,
      });
      const continuable = buildSubagentToolParams(configs, {
        background: true,
        threadContinuation: true,
      });
      const foregroundProperties = foreground.schema.properties as Record<
        string,
        Record<string, unknown>
      >;
      const backgroundProperties = background.schema.properties as Record<
        string,
        Record<string, unknown>
      >;
      const continuableProperties = continuable.schema.properties as Record<
        string,
        Record<string, unknown>
      >;

      expect(foregroundProperties.run_in_background).toBeUndefined();
      expect(foreground.description).not.toContain('BACKGROUND EXECUTION');
      expect(backgroundProperties.run_in_background).toMatchObject({
        type: 'boolean',
      });
      expect(backgroundProperties.subagent_thread_id).toBeUndefined();
      expect(background.description).toContain('BACKGROUND EXECUTION');
      expect(continuableProperties.subagent_thread_id).toMatchObject({
        type: 'string',
      });
      expect(continuable.description).toContain(
        'fresh execution from saved history'
      );
    });

    it('produces same schema as createSubagentToolDefinition', () => {
      const params = buildSubagentToolParams(configs);
      const def = createSubagentToolDefinition(configs);
      expect(params.name).toBe(def.name);
      expect(params.description).toBe(def.description);
      expect(params.schema).toEqual(def.parameters);
    });
  });

  describe('opaque subagent types', () => {
    const configs: SubagentConfig[] = [
      { type: 'researcher', name: 'Research Agent', description: 'Searches' },
      {
        type: 'agent_bqFVwJV0dwKDSyqMnrIIC',
        name: 'Exchange',
        description: 'Mailbox with write access',
      },
      {
        type: 'agent_4LDprR3w5AG934e_vaFU9',
        name: 'Exchange (read-only)',
        description: 'Mailbox, read-only',
      },
    ];

    it('advertises readable names instead of opaque ids', () => {
      const params = buildSubagentToolParams(configs);
      const prop = params.schema.properties!.subagent_type as {
        enum: string[];
        description: string;
      };
      expect(prop.description).toContain(
        'Available: researcher, exchange, exchange-read-only.'
      );
      expect(prop.description).not.toContain('agent_bqFVwJV0dwKDSyqMnrIIC');
      expect(params.description).toContain(
        '- "exchange" (Exchange): Mailbox with write access'
      );
      expect(params.description).not.toContain('agent_bqFVwJV0dwKDSyqMnrIIC');
      expect(params.description).not.toContain('agent_4LDprR3w5AG934e_vaFU9');
    });

    it('still accepts the configured ids', () => {
      const params = buildSubagentToolParams(configs);
      const prop = params.schema.properties!.subagent_type as {
        enum: string[];
      };
      expect(prop.enum).toEqual([
        'researcher',
        'exchange',
        'exchange-read-only',
        'agent_bqFVwJV0dwKDSyqMnrIIC',
        'agent_4LDprR3w5AG934e_vaFU9',
      ]);
    });
  });
});
