import type {
  Classifier,
  ClassificationCredential,
  ClassificationProviderSettings,
} from './types';
import type { ClassificationFetch } from './transport';
import { createHttpClassifier } from './http';

export const DEFAULT_API_KEY_ENV = 'CLASSIFIER_API_KEY';

/**
 * Known hosts, as settings rather than code. They all serve the same question shapes over HTTP
 * and differ only in URL, model name and how the body is wrapped, so a new one is an entry
 * here or the same fields supplied by the caller.
 */
export const CLASSIFICATION_PRESETS: Record<
  string,
  ClassificationProviderSettings
> = {
  http: {
    dialect: 'port',
    apiKeyEnv: DEFAULT_API_KEY_ENV,
  },
  typesafe: {
    baseURL: 'https://api.typesafe.ai/v1/systemone',
    model: 'jev-latest',
    dialect: 'systemone',
    apiKeyEnv: 'TYPESAFE_API_KEY',
  },
  clickhouse: {
    /** ClickHouse's inference gateway; the token is what `dataplanectl chai auth token` prints, hourly. */
    baseURL: 'https://inference-internal.clickhouse.cloud/v1/systemone',
    model: 'jev-latest',
    dialect: 'systemone',
    apiKeyEnv: 'CHAI_AUTH_TOKEN',
  },
  openrouter: {
    baseURL: 'https://openrouter.ai/api/alpha/decisions',
    model: '~typesafe/jev-latest',
    dialect: 'systemone',
    apiKeyEnv: 'OPENROUTER_KEY',
  },
  cloudflare: {
    /** No default URL: the account id is part of it. */
    model: 'typesafe/jev',
    dialect: 'systemone',
    requestKey: 'input',
    responseKey: 'result',
    apiKeyEnv: 'CLOUDFLARE_API_TOKEN',
  },
};

export function classificationPreset(
  name: string | undefined
): ClassificationProviderSettings | null {
  if (name == null || name === '') {
    return null;
  }
  return CLASSIFICATION_PRESETS[name] ?? null;
}

/** Caller settings win over the preset, field by field. */
export function mergeClassificationSettings(
  preset: ClassificationProviderSettings | null,
  configured: ClassificationProviderSettings | undefined
): ClassificationProviderSettings {
  return { ...(preset ?? {}), ...(configured ?? {}) };
}

export function classificationProviderNames(): string[] {
  return Object.keys(CLASSIFICATION_PRESETS);
}

export function createClassifier(
  settings: ClassificationProviderSettings,
  apiKey: ClassificationCredential,
  options?: {
    fetch?: ClassificationFetch;
    providerId?: string;
    onAnswered?: (label: string, ms: number) => void;
  }
): Classifier {
  return createHttpClassifier({
    providerId: options?.providerId,
    apiKey,
    endpoint: settings.baseURL ?? '',
    model: settings.model,
    dialect: settings.dialect,
    requestKey: settings.requestKey,
    responseKey: settings.responseKey,
    timeoutMs: settings.timeoutMs,
    maxRetries: settings.maxRetries,
    fetch: options?.fetch,
    onAnswered: options?.onAnswered,
  });
}
