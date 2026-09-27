import type {
  Classifier,
  ClassificationCredential,
  ClassificationProviderSettings,
} from './types';
import type { ClassificationFetch } from './transport';
import { createHttpClassifier } from './http';

export const DEFAULT_API_KEY_ENV = 'CLASSIFIER_API_KEY';

/**
 * Known HTTP hosts as settings, not new classifier implementations. Jev and Laya speak the
 * same wire protocol, but their confidence measures and checkpoint calibration differ.
 */
export const CLASSIFICATION_PRESETS: Record<
  string,
  ClassificationProviderSettings
> = {
  http: {
    dialect: 'port',
    requiresAuth: true,
    apiKeyEnv: DEFAULT_API_KEY_ENV,
  },
  typesafe: {
    baseURL: 'https://api.typesafe.ai/v1/systemone',
    model: 'jev-latest',
    dialect: 'systemone',
    requiresAuth: true,
    apiKeyEnv: 'TYPESAFE_API_KEY',
  },
  laya: {
    /** Supply the server's full /v1/systemone URL; no model lets Laya route by language. */
    dialect: 'systemone',
    requiresAuth: false,
  },
  clickhouse: {
    /** ClickHouse's inference gateway; the token is what `dataplanectl chai auth token` prints, hourly. */
    baseURL: 'https://inference-internal.clickhouse.cloud/v1/systemone',
    model: 'jev-latest',
    dialect: 'systemone',
    requiresAuth: true,
    apiKeyEnv: 'CHAI_AUTH_TOKEN',
  },
  openrouter: {
    baseURL: 'https://openrouter.ai/api/alpha/decisions',
    model: '~typesafe/jev-latest',
    dialect: 'systemone',
    requiresAuth: true,
    apiKeyEnv: 'OPENROUTER_KEY',
  },
  cloudflare: {
    /** No default URL: the account id is part of it. */
    model: 'typesafe/jev',
    dialect: 'systemone',
    requestKey: 'input',
    responseKey: 'result',
    requiresAuth: true,
    apiKeyEnv: 'CLOUDFLARE_API_TOKEN',
  },
};

export function classificationPreset(
  name: string | undefined
): ClassificationProviderSettings | null {
  if (name == null || name === '') {
    return null;
  }
  return Object.hasOwn(CLASSIFICATION_PRESETS, name)
    ? CLASSIFICATION_PRESETS[name]
    : null;
}

/** Caller settings win except that required preset authentication cannot be weakened. */
export function mergeClassificationSettings(
  preset: ClassificationProviderSettings | null,
  configured: ClassificationProviderSettings | undefined
): ClassificationProviderSettings {
  return {
    ...(preset ?? {}),
    ...(configured ?? {}),
    ...(preset?.requiresAuth === true ? { requiresAuth: true } : {}),
  };
}

export function classificationProviderNames(): string[] {
  return Object.keys(CLASSIFICATION_PRESETS);
}

export function createClassifier(
  settings: ClassificationProviderSettings,
  apiKey?: ClassificationCredential,
  options?: {
    fetch?: ClassificationFetch;
    providerId?: string;
    onAnswered?: (label: string, ms: number) => void;
  }
): Classifier {
  return createHttpClassifier({
    providerId: options?.providerId,
    apiKey,
    requiresAuth: settings.requiresAuth,
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
