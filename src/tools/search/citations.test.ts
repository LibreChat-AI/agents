import type * as t from '@/types';
import {
  formatSearchAnchors,
  stripSearchAnchors,
  byteToCharOffset,
  SearchCitations,
  applyTextEdits,
} from './citations';
import { coerceGoogleGroundingResults } from './google';
import { coerceOpenAICitationResults } from './openai';
import { ContentTypes } from '@/common';

const A = 'https://a.example/weather';
const B = 'https://b.example/fog';

const cite = (url: string): Record<string, string> => ({
  type: 'web_search_result_location',
  url,
  title: url,
  cited_text: 'quoted',
  encrypted_index: 'opaque',
});

const textOf = (content: t.MessageContentComplex[]): string =>
  content
    .map((part) => {
      const { text } = part as { text?: unknown };
      return typeof text === 'string' ? text : '';
    })
    .join('');

describe('formatSearchAnchors', () => {
  it('writes one anchor, or a group for several sources', () => {
    expect(formatSearchAnchors([{ turn: 0, index: 2 }])).toBe(
      '\ue202turn0search2'
    );
    expect(
      formatSearchAnchors([
        { turn: 1, index: 0 },
        { turn: 0, index: 3 },
      ])
    ).toBe('\ue200\ue202turn1search0\ue202turn0search3\ue201');
  });
});

describe('stripSearchAnchors', () => {
  it('removes inserted markers but keeps the escaped anchors a model writes', () => {
    expect(
      stripSearchAnchors(
        'Dry\ue202turn0search0 and foggy\ue200\ue202turn0search1\ue202turn1search4\ue201.'
      )
    ).toBe('Dry and foggy.');
    expect(stripSearchAnchors('Dry.\\ue202turn0search0')).toBe(
      'Dry.\\ue202turn0search0'
    );
  });
});

describe('applyTextEdits', () => {
  it('applies insertions and replacements by position', () => {
    expect(
      applyTextEdits('Sunny (link). Dry.', [
        { start: 18, end: 18, text: '[2]' },
        { start: 6, end: 12, text: '[1]' },
      ])
    ).toBe('Sunny [1]. Dry.[2]');
  });

  it('skips edits outside the text or over an earlier edit', () => {
    expect(
      applyTextEdits('abcdef', [
        { start: 1, end: 4, text: 'X' },
        { start: 2, end: 3, text: 'Y' },
        { start: 4, end: 9, text: 'Z' },
        { start: -1, end: 0, text: 'W' },
      ])
    ).toBe('aXef');
  });
});

describe('offset conversion', () => {
  it('reads UTF-8 byte offsets, refusing one inside a character', () => {
    const text = 'Mensa öffnet 🍝 heute';
    expect(byteToCharOffset(text, 6)).toBe(6);
    expect(byteToCharOffset(text, 8)).toBe(7);
    expect(byteToCharOffset(text, 14)).toBe(13);
    expect(byteToCharOffset(text, 18)).toBe(15);
    expect(byteToCharOffset(text, Buffer.byteLength(text))).toBe(text.length);
    expect(byteToCharOffset(text, 7)).toBeUndefined();
    expect(byteToCharOffset(text, 16)).toBeUndefined();
    expect(byteToCharOffset(text, 999)).toBeUndefined();
  });
});

describe('SearchCitations anchors', () => {
  function setup(): SearchCitations {
    const citations = new SearchCitations();
    citations.register({
      turn: 0,
      organic: [
        { link: A, position: 1 },
        { link: B, position: 2 },
      ],
    });
    return citations;
  }

  it('writes a block anchor when the next block starts, whichever comes first in the block', () => {
    const citations = setup();
    const deltas: t.MessageContentComplex[][] = [
      [{ type: ContentTypes.TEXT, index: 2, text: '', citations: [] }],
      [{ type: ContentTypes.TEXT, index: 2, citations: [cite(A)] }],
      [{ type: ContentTypes.TEXT, index: 2, text: 'Dry today' }],
      [{ type: ContentTypes.TEXT, index: 3, text: '. ' }],
      [{ type: ContentTypes.TEXT, index: 4, text: 'Fog later' }],
      [{ type: ContentTypes.TEXT, index: 4, citations: [cite(B), cite(A)] }],
      [{ type: ContentTypes.TEXT, index: 4, citations: [cite(B)] }],
      [{ type: ContentTypes.TEXT, index: 5, text: '.' }],
    ];
    const streamed = deltas.map(
      (content) => citations.anchor('step', content) ?? content
    );
    expect(streamed.map(textOf).join('')).toBe(
      'Dry today\ue202turn0search0. Fog later\ue200\ue202turn0search1\ue202turn0search0\ue201.'
    );
    expect(citations.release('step')).toBeUndefined();
  });

  it('places anchors between blocks that share one delta', () => {
    const citations = setup();
    const content: t.MessageContentComplex[] = [
      { type: ContentTypes.TEXT, index: 2, citations: [cite(A)] },
      { type: ContentTypes.TEXT, index: 2, text: 'Dry' },
      { type: ContentTypes.TEXT, index: 3, text: '.' },
    ];
    expect(textOf(citations.anchor('step', content) ?? [])).toBe(
      'Dry\ue202turn0search0.'
    );
  });

  it('releases the anchors of the last block when its step closes', () => {
    const citations = setup();
    const metadata = { langgraph_node: 'agent=a' };
    citations.anchor(
      'step',
      [{ type: ContentTypes.TEXT, index: 2, citations: [cite(B)] }],
      metadata
    );
    expect(citations.release('other')).toBeUndefined();
    expect(citations.release('step')).toEqual({
      text: '\ue202turn0search1',
      metadata,
    });
    expect(citations.release('step')).toBeUndefined();
  });

  it('leaves content alone when no citation resolves to a known result', () => {
    const citations = setup();
    const content: t.MessageContentComplex[] = [
      {
        type: ContentTypes.TEXT,
        index: 2,
        citations: [
          cite('https://unknown.example'),
          { type: 'char_location', cited_text: 'doc', document_index: 0 },
        ],
      },
      { type: ContentTypes.TEXT, index: 3, text: 'Next.' },
    ];
    expect(citations.anchor('step', content)).toBeUndefined();
    expect(citations.release('step')).toBeUndefined();
  });
});

describe('SearchCitations late edits', () => {
  it('maps call offsets back to the step and position that hold the text', () => {
    const citations = new SearchCitations();
    citations.track('', 'one', [
      { type: ContentTypes.TEXT, text: 'Earlier. ' },
    ]);
    citations.take('');
    citations.track('', 'one', [
      { type: ContentTypes.TEXT, index: 0, text: 'First.' },
    ]);
    citations.track('', 'two', [
      { type: ContentTypes.TEXT, index: 1, text: 'Second.' },
    ]);
    const call = citations.take('');
    expect(call?.text).toBe('First.Second.');
    expect(call?.blocks).toEqual(
      new Map([
        [0, 0],
        [1, 6],
      ])
    );
    const byStep = citations.locate(call!, [
      { start: 6, end: 6, text: '[a]' },
      { start: 13, end: 13, text: '[b]' },
      { start: 5, end: 8, text: '[x]' },
    ]);
    expect(byStep).toEqual(
      new Map([
        ['one', [{ start: 15, end: 15, text: '[a]' }]],
        ['two', [{ start: 7, end: 7, text: '[b]' }]],
      ])
    );
  });

  it('collects url citations from annotation parts', () => {
    const citations = new SearchCitations();
    citations.track('', 'step', [
      { type: ContentTypes.TEXT, index: 0, text: 'Sunny.' },
      {
        type: ContentTypes.TEXT,
        index: 0,
        text: '',
        annotations: [
          {
            type: 'citation',
            source: 'url_citation',
            url: A,
            title: 'Weather',
            startIndex: 0,
            endIndex: 6,
          },
          { type: 'citation', source: 'file_citation', title: 'a.pdf' },
        ],
      },
    ]);
    expect(citations.take('')?.citations).toEqual([
      { block: 0, url: A, title: 'Weather', start: 0, end: 6 },
    ]);
  });
});

describe('coerceGoogleGroundingResults', () => {
  const first = 'Die Mensa öffnet um 11 Uhr.';
  const second = 'Heute gibt es Pasta.';
  const text = `${first} ${second}`;
  const bytes = (value: string): number => Buffer.byteLength(value);

  it('anchors each grounded segment at its byte offset', () => {
    const results = coerceGoogleGroundingResults(
      {
        webSearchQueries: ['mensa heute'],
        groundingChunks: [
          { web: { uri: A, title: 'studentenwerk.example' } },
          { web: { uri: B, title: 'mensa.example', domain: 'mensa.example' } },
        ],
        groundingSupports: [
          {
            segment: { startIndex: 0, endIndex: bytes(first), text: first },
            groundingChunkIndices: [0],
          },
          {
            segment: {
              startIndex: bytes(`${first} `),
              endIndex: bytes(text),
              text: second,
            },
            groundingChunkIndices: [1, 0, 1],
          },
        ],
      },
      text
    );
    expect(results).toEqual({
      organic: [
        {
          link: A,
          position: 1,
          title: 'studentenwerk.example',
          attribution: 'studentenwerk.example',
        },
        {
          link: B,
          position: 2,
          title: 'mensa.example',
          attribution: 'mensa.example',
        },
      ],
      spans: [
        { start: first.length, end: first.length, sources: [0] },
        { start: text.length, end: text.length, sources: [1, 0] },
      ],
      queries: ['mensa heute'],
    });
  });

  it('finds a segment by its text when the offsets do not fit the streamed text', () => {
    const results = coerceGoogleGroundingResults(
      {
        groundingChunks: [{}, { web: { uri: A } }],
        groundingSupports: [
          {
            segment: { startIndex: 900, endIndex: 950, text: second },
            groundingChunkIndices: [1],
          },
          { segment: { text: 'missing' }, groundingChunkIndices: [1] },
          { segment: { text: first }, groundingChunkIndices: [0] },
        ],
      },
      text
    );
    expect(results?.organic).toHaveLength(1);
    expect(results?.spans).toEqual([
      { start: text.length, end: text.length, sources: [0] },
    ]);
  });

  it('anchors a later response part only where its text is unambiguous', () => {
    const repeated = `${second} ${second}`;
    const segment = (partIndex: number, value = second) => ({
      partIndex,
      startIndex: 0,
      endIndex: bytes(value),
      text: value,
    });
    const results = coerceGoogleGroundingResults(
      {
        groundingChunks: [{ web: { uri: A } }, { web: { uri: B } }],
        groundingSupports: [
          { segment: segment(1), groundingChunkIndices: [1] },
          {
            segment: segment(1, `${second} ${second}`),
            groundingChunkIndices: [0],
          },
        ],
      },
      repeated
    );
    expect(results?.spans).toEqual([
      { start: repeated.length, end: repeated.length, sources: [0] },
    ]);
  });

  it('returns nothing without web sources', () => {
    expect(
      coerceGoogleGroundingResults({ groundingChunks: [{}] }, text)
    ).toBeUndefined();
  });
});

describe('coerceOpenAICitationResults', () => {
  const url = 'https://weather.example/today?utm_source=openai';
  const link = `([weather.example](${url}))`;

  it('replaces the cited markdown link of each block', () => {
    const text = `Intro. It is sunny ${link}. Dry ([dry.example](${B})).`;
    const blockStart = 'Intro. '.length;
    const block = text.slice(blockStart);
    const results = coerceOpenAICitationResults(
      [
        {
          block: 1,
          url,
          title: 'Today',
          start: block.indexOf(link),
          end: block.indexOf(link) + link.length,
        },
        {
          block: 1,
          url: B,
          start: block.indexOf('[dry'),
          end: block.indexOf('[dry') + `[dry.example](${B})`.length - 1,
        },
        { block: 1, url, start: 0, end: 5 },
      ],
      text,
      new Map([[1, blockStart]])
    );
    expect(results?.organic).toEqual([
      {
        link: url,
        position: 1,
        title: 'Today',
        attribution: 'weather.example',
      },
      { link: B, position: 2, title: undefined, attribution: 'b.example' },
    ]);
    const edits = results!.spans.map(({ start, end, sources }) => ({
      start,
      end,
      text: `[${sources.join(',')}]`,
    }));
    expect(applyTextEdits(text, edits)).toBe(
      'Intro. It is sunny [0]. Dry [1].'
    );
  });

  it('reads code point offsets after astral characters', () => {
    const text = `🌞 Sunny ${link}`;
    const results = coerceOpenAICitationResults(
      [{ block: 0, url, start: 8, end: 8 + link.length }],
      text,
      new Map([[0, 0]])
    );
    expect(results?.spans).toEqual([
      { start: 9, end: 9 + link.length, sources: [0] },
    ]);
  });

  it('keeps a citation as a source when its span is not a link', () => {
    const results = coerceOpenAICitationResults(
      [
        { block: 0, url, start: 0, end: 5 },
        { block: 7, url: B },
      ],
      'Sunny today.',
      new Map([[0, 0]])
    );
    expect(results?.organic.map(({ link }) => link)).toEqual([url, B]);
    expect(results?.spans).toEqual([]);
  });
});
