import {
  buildSummaryCarrierText,
  separateSummarizationParameters,
  buildSummarizationInstruction,
} from '@/summarization';
import {
  DEFAULT_SUMMARIZATION_PROMPT,
  DEFAULT_UPDATE_SUMMARIZATION_PROMPT,
} from '@/summarization/shared';

describe('summarization primitives', () => {
  describe('buildSummaryCarrierText', () => {
    it('wraps the summary in the tags and instruction it is re-injected with', () => {
      const carrier = buildSummaryCarrierText('checkpoint body');
      expect(
        carrier.startsWith('<summary>\ncheckpoint body\n</summary>\n\n')
      ).toBe(true);
      expect(carrier).toContain('This is your own checkpoint');
    });

    /** A stored summary is budgeted for by measuring this, so the wrapper has
     *  to cost enough to be worth measuring: the literal it replaced claimed 33
     *  tokens for a carrier that had already grown well past that. */
    it('carries an instruction long enough to matter to a context budget', () => {
      expect(buildSummaryCarrierText('').length).toBeGreaterThan(200);
    });
  });

  describe('separateSummarizationParameters', () => {
    it('lifts maxSummaryTokens out of the parameters passed to the model', () => {
      const result = separateSummarizationParameters({
        maxSummaryTokens: 512,
        temperature: 0.2,
      });
      expect(result).toEqual({
        llmParams: { temperature: 0.2 },
        maxSummaryTokens: 512,
      });
    });

    it('ignores a maxSummaryTokens that is not a positive number', () => {
      expect(separateSummarizationParameters({ maxSummaryTokens: 0 })).toEqual({
        llmParams: {},
        maxSummaryTokens: undefined,
      });
      expect(
        separateSummarizationParameters({ maxSummaryTokens: '512' })
      ).toEqual({ llmParams: {}, maxSummaryTokens: undefined });
    });
  });

  describe('buildSummarizationInstruction', () => {
    it('sends the fresh prompt when there is no prior summary', () => {
      expect(
        buildSummarizationInstruction(
          DEFAULT_SUMMARIZATION_PROMPT,
          DEFAULT_UPDATE_SUMMARIZATION_PROMPT,
          undefined
        )
      ).toBe(DEFAULT_SUMMARIZATION_PROMPT);
    });

    it('sends the update prompt with the prior summary appended', () => {
      const result = buildSummarizationInstruction(
        'fresh',
        'update',
        'earlier checkpoint'
      );
      expect(result).toBe(
        'update\n\n<previous-summary>\nearlier checkpoint\n</previous-summary>'
      );
    });

    it('falls back to the fresh prompt when no update prompt is configured', () => {
      expect(buildSummarizationInstruction('fresh', undefined, 'prior')).toBe(
        'fresh\n\n<previous-summary>\nprior\n</previous-summary>'
      );
    });

    it('treats a blank prior summary as absent rather than consolidating nothing', () => {
      expect(buildSummarizationInstruction('fresh', 'update', '   \n ')).toBe(
        'fresh'
      );
    });

    /** The semantic index leads the instruction so the checkpoint request is
     *  the last thing the model reads, and the prior summary still trails it. */
    it('leads with the semantic index appendix when one is rendered', () => {
      expect(
        buildSummarizationInstruction('fresh', 'update', undefined, '<index/>')
      ).toBe('<index/>\n\nfresh');
      expect(
        buildSummarizationInstruction('fresh', 'update', 'prior', '<index/>')
      ).toBe(
        '<index/>\n\nupdate\n\n<previous-summary>\nprior\n</previous-summary>'
      );
    });

    it('references a prior checkpoint already carried by the cached prefix', () => {
      const result = buildSummarizationInstruction('fresh', 'update', 'earlier checkpoint', '<index/>', true);
      expect(result).toContain('<index/>\n\nupdate');
      expect(result).toContain('already present in the request prefix');
      expect(result).not.toContain('earlier checkpoint');
    });

    it('omits the appendix separator when no index was rendered', () => {
      expect(buildSummarizationInstruction('fresh', 'update', '', '')).toBe(
        'fresh'
      );
    });
  });
});
