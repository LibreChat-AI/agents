import type { CapturedTurn, Json } from './harness';
import {
  endOfText,
  endOfTools,
  countMatches,
  toolSegments,
  runConversation,
  stripCacheMarkers,
  commonPrefixLength,
  serializeInCacheOrder,
} from './harness';
import { PROVIDER_CASES } from './providers';
import { SCENARIOS } from './scenarios';

jest.setTimeout(120_000);

const isResetTurn = ({ reset }: CapturedTurn): boolean => reset;

const marker = (value: Json): string =>
  JSON.stringify(stripCacheMarkers(value));

const escaped = (text: string): string => JSON.stringify(text).slice(1, -1);

/** System prompt without this turn's volatile part, which may differ per turn. */
const fixedSystem = ({ request, turn }: CapturedTurn): string => {
  const system = marker(request.system);
  return turn.additionalInstructions == null
    ? system
    : system.replace(escaped(turn.additionalInstructions), '');
};

/** Start of the segment that carries the volatile part, or the full length. */
const reusableLength = (serialized: string, { turn }: CapturedTurn): number => {
  if (turn.additionalInstructions == null) {
    return serialized.length;
  }
  const start = serialized.indexOf(escaped(turn.additionalInstructions));
  return start < 0 ? serialized.length : serialized.lastIndexOf('\n', start) + 1;
};

function roleOf(message: Json): string | undefined {
  if (message == null || typeof message !== 'object' || Array.isArray(message)) {
    return undefined;
  }
  const role = message.role;
  return typeof role === 'string' ? role : undefined;
}

describe.each(PROVIDER_CASES.map((providerCase) => [providerCase.name, providerCase]))(
  'prompt cache: %s',
  (_name, providerCase) => {
    describe.each(SCENARIOS.map((scenario) => [scenario.name, scenario]))(
      '%s',
      (_scenarioName, scenario) => {
        let conversations: CapturedTurn[][];

        beforeAll(async () => {
          conversations = [];
          for (const conversation of scenario.conversations) {
            conversations.push(
              await runConversation(providerCase, scenario, conversation)
            );
          }
        });

        const allTurns = (): CapturedTurn[] => conversations.flat();

        const consecutiveTurns = (
          select: (next: CapturedTurn) => boolean
        ): Array<[CapturedTurn, CapturedTurn]> =>
          conversations.flatMap((turns) =>
            turns
              .slice(1)
              .map((next, i): [CapturedTurn, CapturedTurn] => [turns[i], next])
              .filter(([, next]) => select(next))
          );

        it('keeps the conversation block in the system prompt', () => {
          for (const captured of allTurns()) {
            const system = JSON.stringify(captured.request.system);
            expect(system).toContain(escaped(captured.conversationInstructions));
            expect(system).toContain(escaped(captured.instructions));
          }
        });

        it('moves only the per-turn volatile part, and only where documented', () => {
          for (const { request, turn } of allTurns()) {
            if (turn.additionalInstructions == null) {
              continue;
            }
            const volatile = escaped(turn.additionalInstructions);
            const inSystem = JSON.stringify(request.system).includes(volatile);
            expect(inSystem).toBe(providerCase.movesDynamicPart !== true);
            expect(JSON.stringify(request.messages).includes(volatile)).toBe(
              !inSystem
            );
          }
        });

        it('keeps tools and system fixed between reset points', () => {
          for (const [previous, next] of consecutiveTurns(
            (turn) => !isResetTurn(turn)
          )) {
            expect(toolSegments(next.request).map(marker)).toEqual(
              toolSegments(previous.request).map(marker)
            );
            expect(fixedSystem(next)).toBe(fixedSystem(previous));
          }
        });

        it('starts every request with the previous one up to its volatile part', () => {
          for (const [previous, next] of consecutiveTurns(
            (turn) => !isResetTurn(turn)
          )) {
            const before = serializeInCacheOrder(previous.request);
            const after = serializeInCacheOrder(next.request);
            const required = next.followUp
              ? before.length
              : reusableLength(before, previous);
            expect(commonPrefixLength(before, after)).toBeGreaterThanOrEqual(
              required
            );
          }
        });

        it('keeps the cached prefix up to the declared boundary at reset points', () => {
          for (const [previous, next] of consecutiveTurns(isResetTurn)) {
            const before = serializeInCacheOrder(previous.request);
            const after = serializeInCacheOrder(next.request);
            const appendsTools =
              (next.turn.compaction?.appendTools?.length ?? 0) > 0;
            const boundary = appendsTools
              ? endOfTools(previous.request)
              : endOfText(before, previous.instructions);
            expect(boundary).toBeGreaterThan(0);
            expect(commonPrefixLength(before, after)).toBeGreaterThanOrEqual(
              boundary
            );
            if (next.turn.compaction != null) {
              expect(after).toContain(escaped(next.turn.compaction.summary));
            }
          }
        });

        if (scenario.conversations.length > 1) {
          it('shares tools and stable instructions across conversations', () => {
            const [first, second] = conversations.map((turns) =>
              serializeInCacheOrder(turns[0].request)
            );
            expect(commonPrefixLength(first, second)).toBeGreaterThanOrEqual(
              endOfText(first, scenario.instructions)
            );
          });
        }

        if (providerCase.breakpoints != null) {
          const { pattern, max } = providerCase.breakpoints;
          it(`uses at most ${max} cache breakpoints per request`, () => {
            for (const { request } of allTurns()) {
              const count = countMatches(request.raw, pattern);
              expect(count).toBeGreaterThan(0);
              expect(count).toBeLessThanOrEqual(max);
            }
          });
        }

        if (providerCase.strictAlternation === true) {
          it('never sends two user messages in a row', () => {
            for (const { request } of allTurns()) {
              const roles = request.messages.map(roleOf);
              for (let i = 1; i < roles.length; i++) {
                const pair = `${roles[i - 1]}>${roles[i]}`;
                expect(['user>user', 'human>human']).not.toContain(pair);
              }
            }
          });
        }
      }
    );
  }
);
