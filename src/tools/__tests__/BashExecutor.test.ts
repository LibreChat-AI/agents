import { describe, it, expect } from '@jest/globals';
import {
  BashExecutionToolDescription,
  BashToolOutputReferencesGuide,
  StatefulBashExecutionToolDescription,
  buildBashExecutionToolDescription,
} from '../BashExecutor';
import { CODE_ARTIFACT_PATH_GUIDANCE } from '../CodeExecutor';

describe('buildBashExecutionToolDescription', () => {
  it('returns the base description by default', () => {
    expect(buildBashExecutionToolDescription()).toBe(
      BashExecutionToolDescription
    );
    expect(buildBashExecutionToolDescription({})).toBe(
      BashExecutionToolDescription
    );
    expect(
      buildBashExecutionToolDescription({ enableToolOutputReferences: false })
    ).toBe(BashExecutionToolDescription);
  });

  it('warns about compact bash shell pitfalls', () => {
    expect(BashExecutionToolDescription).toContain('heredoc/printf');
    expect(BashExecutionToolDescription).toContain('not bare Python');
    expect(BashExecutionToolDescription).toContain(
      'Failed executions register nothing'
    );
    expect(BashExecutionToolDescription).toContain('`/tmp` never survives the call');
  });

  it('appends the tool-output references guide when enabled', () => {
    const composed = buildBashExecutionToolDescription({
      enableToolOutputReferences: true,
    });
    expect(composed.startsWith(BashExecutionToolDescription)).toBe(true);
    expect(composed).toContain(BashToolOutputReferencesGuide);
    expect(composed).toContain('{{tool<idx>turn<turn>}}');
  });

  it('nudges the model toward heredoc when payloads may contain shell metacharacters', () => {
    /**
     * Real-world failure observed against ClickHouse + bash piping:
     * the model emitted `echo '{{ref}}' | wc -c` and the substituted
     * binary payload contained literal single quotes, breaking the
     * shell. The model self-corrected to a heredoc on retry. Surface
     * the heredoc pattern upfront so the round-trip isn't burned to
     * rediscover it.
     */
    expect(BashToolOutputReferencesGuide).toContain('heredoc');
    expect(BashToolOutputReferencesGuide).toContain('<< \'EOF\'');
  });

  it('describes rejection of unknown keys and the literal escape', () => {
    expect(BashToolOutputReferencesGuide).toContain(
      'rejected before anything runs'
    );
    expect(BashToolOutputReferencesGuide).toContain('`\\{{tool0turn0}}`');
    expect(BashToolOutputReferencesGuide).not.toContain('left in place');
    expect(BashToolOutputReferencesGuide).toContain(
      '`C:\\out\\\\{{tool0turn0}}`'
    );
  });

  it('separates base and guide with a blank line', () => {
    const composed = buildBashExecutionToolDescription({
      enableToolOutputReferences: true,
    });
    expect(composed.includes(`${BashExecutionToolDescription}\n\n`)).toBe(true);
  });

  describe('stateful variant', () => {
    it('selects the stateful description when statefulSessions is on', () => {
      expect(
        buildBashExecutionToolDescription({ statefulSessions: true })
      ).toBe(StatefulBashExecutionToolDescription);
    });

    /* Filesystem-tier only: each call runs in a fresh sandbox (new process
     * tree + private /tmp), so background processes are reaped and non-
     * /mnt/data writes are discarded. The description must not promise
     * otherwise. */
    it('promises /mnt/data persistence WITHOUT promising surviving processes or /tmp', () => {
      const d = StatefulBashExecutionToolDescription;
      expect(d).toContain('same warm machine');
      expect(d).toContain('Only /mnt/data is durable');
      expect(d).toContain('background processes do NOT survive');
      expect(d).toContain('/tmp');
    });

    it('never claims /tmp or background processes persist between calls', () => {
      const d = StatefulBashExecutionToolDescription;
      expect(d).not.toContain('files (including /tmp)');
      expect(d).not.toContain(
        'background processes from earlier calls typically persist'
      );
    });

    it('keeps the artifact-path guidance in both variants', () => {
      expect(BashExecutionToolDescription).toContain(
        CODE_ARTIFACT_PATH_GUIDANCE
      );
      expect(StatefulBashExecutionToolDescription).toContain(
        CODE_ARTIFACT_PATH_GUIDANCE
      );
    });

    it('still composes with the output-references guide', () => {
      const composed = buildBashExecutionToolDescription({
        statefulSessions: true,
        enableToolOutputReferences: true,
      });
      expect(composed.startsWith(StatefulBashExecutionToolDescription)).toBe(
        true
      );
      expect(composed).toContain(BashToolOutputReferencesGuide);
    });
  });

  it('removes /mnt/data persistence claims from attached workspace parameters', async () => {
    const { buildBashExecutionToolSchema } = await import('../BashExecutor');
    const description = buildBashExecutionToolSchema({
      attachedWorkspace: true,
    }).properties.command.description;

    expect(description).toContain('selected persistent project');
    expect(description).toContain('Write anything needed later into the selected project');
    expect(description).not.toContain('Prior /mnt/data files');
    expect(description).not.toContain('Anything a later call needs');
  });
});
