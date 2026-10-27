import { describe, expect, test } from '@jest/globals';
import { CustomChatGoogleGenerativeAI } from './index';

const isMultimodal = (model: string): boolean =>
  new CustomChatGoogleGenerativeAI({ model, apiKey: 'test-key' })
    ._isMultimodalModel;

describe('CustomChatGoogleGenerativeAI._isMultimodalModel', () => {
  test('accepts Gemini 3 and later, including Gemini 4 Argon', () => {
    expect(isMultimodal('gemini-3-pro-preview')).toBe(true);
    expect(isMultimodal('gemini-3.8-flash')).toBe(true);
    expect(isMultimodal('gemini-4-argon')).toBe(true);
    expect(isMultimodal('models/gemini-4-argon')).toBe(true);
    expect(isMultimodal('gemini-4-argon-cyber-permissive')).toBe(true);
    expect(isMultimodal('gemini-10-pro')).toBe(true);
  });

  test('keeps the earlier generations and text-only models as they were', () => {
    expect(isMultimodal('gemini-1.5-pro')).toBe(true);
    expect(isMultimodal('gemini-2.5-flash')).toBe(true);
    expect(isMultimodal('gemma-3-27b-it')).toBe(true);
    expect(isMultimodal('gemma-3-1b-it')).toBe(false);
    expect(isMultimodal('gemini-pro')).toBe(false);
  });
});
