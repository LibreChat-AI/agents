/**
 * Rate-limit retry probe.
 *
 * Drives a provider into a 429 on purpose and reports whether the run survived
 * it, so `src/utils/rateLimit.ts` stays grounded in what providers actually
 * send instead of in phrases someone expected them to send.
 *
 * The cheap way in is the concurrency limit: many tiny requests at once. That
 * costs a handful of tokens and trips the same 429 as the token bucket on
 * providers that answer every limit with one body - Scaleway being the reason
 * this exists.
 *
 * Run:
 *   OPENAI_LIKE_API_KEY=... OPENAI_LIKE_BASE_URL=https://api.scaleway.ai/v1 \
 *     node --loader ./tsconfig-paths-bootstrap.mjs \
 *     --experimental-specifier-resolution=node ./src/scripts/rate-limit-probe.ts
 *
 * Flags:
 *   --model <id>        model to call (default: glm-5.2)
 *   --concurrency <n>   simultaneous requests (default: 40)
 *   --filler <tokens>   pad each prompt, to drain a tokens-per-minute bucket
 *                       instead of the concurrency one. Costs real tokens.
 *   --no-retry          construct the model with LangChain's stock policy, to
 *                       show the failure this handler exists to prevent
 */
import { config as loadEnv } from 'dotenv';
import { ChatOpenAI as StockChatOpenAI } from '@langchain/openai';
import { ChatOpenAI } from '@/llm/openai';
import { extractErrorMessage } from '@/utils/errors';

loadEnv({ path: process.env.DOTENV_CONFIG_PATH ?? '.env' });

function flag(name: string, fallback: string): string {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 && process.argv[index + 1] != null
    ? process.argv[index + 1]
    : fallback;
}

const apiKey = process.env.OPENAI_LIKE_API_KEY ?? process.env.SCALEWAY_API_KEY;
const baseURL =
  process.env.OPENAI_LIKE_BASE_URL ??
  process.env.SCALEWAY_BASE_URL ??
  'https://api.scaleway.ai/v1';
const model = flag('model', 'glm-5.2');
const concurrency = parseInt(flag('concurrency', '40'), 10);
const fillerTokens = parseInt(flag('filler', '0'), 10);
const withoutRetry = process.argv.includes('--no-retry');

/** Roughly one token per repetition, which is close enough to aim a bucket at. */
const prompt = fillerTokens > 0 ? 'lorem ipsum '.repeat(fillerTokens) : 'hi';

if (apiKey == null || apiKey === '') {
  console.error('Set OPENAI_LIKE_API_KEY (or SCALEWAY_API_KEY) first.');
  process.exit(1);
}

/**
 * `--no-retry` builds the upstream class instead of ours, which is the honest
 * comparison: the handler cannot be switched off by passing `undefined`,
 * because that is indistinguishable from not passing it at all.
 */
const fields = { model, apiKey, configuration: { baseURL }, maxTokens: 1 };
const llm = withoutRetry ? new StockChatOpenAI(fields) : new ChatOpenAI(fields);

async function main(): Promise<void> {
  console.log(
    `${concurrency} concurrent requests to ${model} at ${baseURL} (${
      withoutRetry ? 'stock LangChain policy' : 'with retry handler'
    })`
  );

  const started = Date.now();
  const results = await Promise.all(
    Array.from({ length: concurrency }, async (_unused, index) => {
      try {
        await llm.invoke(prompt);
        return { index, ok: true, message: '' };
      } catch (error) {
        return { index, ok: false, message: extractErrorMessage(error) };
      }
    })
  );

  const failures = results.filter((result) => !result.ok);
  console.log(
    `succeeded ${results.length - failures.length}/${results.length} in ${
      Date.now() - started
    }ms`
  );
  for (const message of new Set(failures.map((failure) => failure.message))) {
    console.log(`  failure: ${message}`);
  }
  process.exit(failures.length > 0 ? 1 : 0);
}

void main();
