import type {
  DecisionModel,
  DecisionCredential,
  DecisionProviderSettings,
} from './types';
import type { DecisionFetch } from './transport';
import { createHttpDecisionModel } from './http';

export const DEFAULT_API_KEY_ENV = 'DECISION_API_KEY';

/**
 * Known HTTP hosts as settings, not new decision model implementations. Jev and Laya speak the
 * same wire protocol, but their confidence measures and checkpoint calibration differ.
 */
export const DECISION_PRESETS: Readonly<
  Record<string, Readonly<DecisionProviderSettings>>
> = Object.freeze({
  http: Object.freeze({
    dialect: 'port',
    requiresAuth: true,
    apiKeyEnv: DEFAULT_API_KEY_ENV,
  }),
  typesafe: Object.freeze({
    baseURL: 'https://api.typesafe.ai/v1/systemone',
    model: 'jev-latest',
    dialect: 'systemone',
    requiresAuth: true,
    apiKeyEnv: 'TYPESAFE_API_KEY',
  }),
  laya: Object.freeze({
    /** Supply the server's full /v1/systemone URL; no model lets Laya route by language. */
    dialect: 'systemone',
    requiresAuth: false,
  }),
  clickhouse: Object.freeze({
    /** ClickHouse's inference gateway; the token is what `dataplanectl chai auth token` prints, hourly. */
    baseURL: 'https://inference-internal.clickhouse.cloud/v1/systemone',
    model: 'jev-latest',
    dialect: 'systemone',
    requiresAuth: true,
    apiKeyEnv: 'CHAI_AUTH_TOKEN',
  }),
  openrouter: Object.freeze({
    baseURL: 'https://openrouter.ai/api/alpha/decisions',
    model: '~typesafe/jev-latest',
    dialect: 'systemone',
    requiresAuth: true,
    apiKeyEnv: 'OPENROUTER_KEY',
  }),
  cloudflare: Object.freeze({
    /** No default URL: the account id is part of it. */
    model: 'typesafe/jev',
    dialect: 'systemone',
    requestKey: 'input',
    responseKey: 'result',
    requiresAuth: true,
    apiKeyEnv: 'CLOUDFLARE_API_TOKEN',
  }),
});

export function decisionPreset(
  name: string | undefined
): DecisionProviderSettings | null {
  if (name == null || name === '') {
    return null;
  }
  return Object.hasOwn(DECISION_PRESETS, name)
    ? { ...DECISION_PRESETS[name] }
    : null;
}

/** Caller settings win except that required preset authentication cannot be weakened. */
export function mergeDecisionSettings(
  preset: DecisionProviderSettings | null,
  configured: DecisionProviderSettings | undefined
): DecisionProviderSettings {
  return {
    ...(preset ?? {}),
    ...(configured ?? {}),
    ...(preset?.requiresAuth === true ? { requiresAuth: true } : {}),
  };
}

export function decisionProviderNames(): string[] {
  return Object.keys(DECISION_PRESETS);
}

export function createDecisionModel(
  settings: DecisionProviderSettings,
  apiKey?: DecisionCredential,
  options?: {
    fetch?: DecisionFetch;
    providerId?: string;
    onAnswered?: (label: string, ms: number) => void;
  }
): DecisionModel {
  return createHttpDecisionModel({
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
