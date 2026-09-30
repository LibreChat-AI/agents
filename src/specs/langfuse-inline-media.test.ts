import { randomBytes } from 'node:crypto';
import {
  BasicTracerProvider,
  InMemorySpanExporter,
} from '@opentelemetry/sdk-trace-base';
import {
  LangfuseOtelSpanAttributes,
  setLangfuseTracerProvider,
  startObservation,
} from '@langfuse/tracing';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import type { LangfuseSpanProcessorParams } from '@langfuse/otel';
import type * as t from '@/types';
import {
  resolveLangfuseConfig,
  resolveInlineMediaTracingEnabled,
  resolveLangfuseMediaUploadEnabled,
} from '@/langfuseConfig';
import {
  createLangfuseSpanProcessor,
  prepareLangfuseSpanForExport,
} from '@/langfuseToolOutputTracing';

const INPUT = LangfuseOtelSpanAttributes.OBSERVATION_INPUT;
const OUTPUT = LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT;
const PDF_BYTES = randomBytes(96 * 1024);
const PDF_BASE64 = PDF_BYTES.toString('base64');
const PNG_BASE64 = randomBytes(8 * 1024).toString('base64');
const PDF_DESCRIPTOR = `[inline media omitted from trace: application/pdf, ${PDF_BYTES.length} bytes]`;
const PNG_DESCRIPTOR = '[inline media omitted from trace: image/png, 8192 bytes]';

type SerializedPart = {
  type?: string;
  text?: string;
  data?: string;
  mimeType?: string;
  filename?: string;
  file_data?: string;
  image_url?: { url?: string };
  file?: { filename?: string; file_data?: string };
  source?: { type?: string; media_type?: string; data?: string };
  document?: {
    format?: string;
    name?: string;
    source?: { bytes?: Buffer | string };
  };
};

type SerializedMessage = {
  role?: string;
  content: SerializedPart[];
};

function userMessage(...content: SerializedPart[]): SerializedMessage {
  return { role: 'user', content };
}

function readMessages(span: ReadableSpan, key = INPUT): SerializedMessage[] {
  return JSON.parse(span.attributes[key] as string) as SerializedMessage[];
}

function createSpan(attributes: Record<string, string>): ReadableSpan {
  return { name: 'llm', attributes } as unknown as ReadableSpan;
}

async function exportGeneration({
  input,
  output,
  langfuse,
  params,
}: {
  input?: object;
  output?: object;
  langfuse?: t.LangfuseConfig;
  params?: Partial<LangfuseSpanProcessorParams>;
}): Promise<ReadableSpan> {
  const exporter = new InMemorySpanExporter();
  const provider = new BasicTracerProvider({
    spanProcessors: [
      createLangfuseSpanProcessor(
        {
          publicKey: 'pk-test',
          secretKey: 'sk-test',
          baseUrl: 'http://langfuse.invalid',
          exporter,
          exportMode: 'immediate',
          ...params,
        },
        langfuse
      ),
    ],
  });
  setLangfuseTracerProvider(provider);
  try {
    startObservation('llm', { input, output }, { asType: 'generation' }).end();
    await provider.forceFlush();
    const [span] = exporter.getFinishedSpans();
    return span;
  } finally {
    setLangfuseTracerProvider(null);
    await provider.shutdown();
  }
}

describe('Langfuse inline media omission', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
    delete process.env.LANGFUSE_TRACE_INLINE_MEDIA;
    delete process.env.LANGFUSE_MEDIA_UPLOAD_ENABLED;
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('replaces a raw base64 media part with a descriptor and keeps the text', async () => {
    const span = await exportGeneration({
      input: [
        userMessage(
          { type: 'text', text: 'Summarize the attached report.' },
          { type: 'media', mimeType: 'application/pdf', data: PDF_BASE64 }
        ),
      ],
    });

    expect(span.attributes[INPUT]).not.toContain(PDF_BASE64);
    expect(readMessages(span)[0].content).toEqual([
      { type: 'text', text: 'Summarize the attached report.' },
      { type: 'media', mimeType: 'application/pdf', data: PDF_DESCRIPTOR },
    ]);
  });

  it('omits media from span outputs as well as inputs', async () => {
    const message = userMessage({
      type: 'media',
      mimeType: 'application/pdf',
      data: PDF_BASE64,
    });
    const span = await exportGeneration({
      input: [message],
      output: [message],
    });

    expect(readMessages(span, OUTPUT)[0].content[0].data).toBe(PDF_DESCRIPTOR);
    expect(readMessages(span, INPUT)[0].content[0].data).toBe(PDF_DESCRIPTOR);
  });

  it('omits data URIs, keeping filenames, when Langfuse media upload is disabled', async () => {
    const span = await exportGeneration({
      params: { mediaUploadEnabled: false },
      input: [
        userMessage(
          {
            type: 'image_url',
            image_url: { url: `data:image/png;base64,${PNG_BASE64}` },
          },
          {
            type: 'input_file',
            filename: 'report.pdf',
            file_data: `data:application/pdf;base64,${PDF_BASE64}`,
          },
          {
            type: 'file',
            file: {
              filename: 'report.pdf',
              file_data: `data:application/pdf;base64,${PDF_BASE64}`,
            },
          }
        ),
      ],
    });

    expect(readMessages(span)[0].content).toEqual([
      { type: 'image_url', image_url: { url: PNG_DESCRIPTOR } },
      { type: 'input_file', filename: 'report.pdf', file_data: PDF_DESCRIPTOR },
      {
        type: 'file',
        file: { filename: 'report.pdf', file_data: PDF_DESCRIPTOR },
      },
    ]);
  });

  it('leaves data URIs for the Langfuse SDK to upload while media upload is enabled', () => {
    const dataUri = `data:image/png;base64,${PNG_BASE64}`;
    const span = createSpan({
      [INPUT]: JSON.stringify([
        userMessage(
          { type: 'image_url', image_url: { url: dataUri } },
          { type: 'media', mimeType: 'application/pdf', data: PDF_BASE64 }
        ),
      ]),
    });

    prepareLangfuseSpanForExport(span, undefined, { omitDataUris: false });

    expect(readMessages(span)[0].content).toEqual([
      { type: 'image_url', image_url: { url: dataUri } },
      { type: 'media', mimeType: 'application/pdf', data: PDF_DESCRIPTOR },
    ]);
  });

  it('omits Anthropic base64 sources and serialized byte buffers', async () => {
    const span = await exportGeneration({
      input: [
        userMessage(
          {
            type: 'document',
            source: {
              type: 'base64',
              media_type: 'application/pdf',
              data: PDF_BASE64,
            },
          },
          {
            type: 'document',
            document: {
              format: 'pdf',
              name: 'report',
              source: { bytes: PDF_BYTES },
            },
          }
        ),
      ],
    });

    const [anthropic, bedrock] = readMessages(span)[0].content;
    expect(anthropic.source).toEqual({
      type: 'base64',
      media_type: 'application/pdf',
      data: PDF_DESCRIPTOR,
    });
    expect(bedrock.document).toEqual({
      format: 'pdf',
      name: 'report',
      source: {
        bytes: `[inline media omitted from trace: unknown type, ${PDF_BYTES.length} bytes]`,
      },
    });
  });

  it('exports ordinary content byte-for-byte', async () => {
    const input = [
      userMessage(
        { type: 'text', text: 'word '.repeat(4000) },
        { type: 'text', text: 'a-b_c.'.repeat(1000) },
        { type: 'media', mimeType: 'image/png', data: PNG_BASE64.slice(0, 1024) }
      ),
    ];
    const span = await exportGeneration({ input });

    expect(span.attributes[INPUT]).toBe(JSON.stringify(input));
  });

  it('replaces an attribute that is itself an inline payload', () => {
    const span = createSpan({
      [INPUT]: `data:application/pdf;base64,${PDF_BASE64}`,
      [OUTPUT]: PDF_BASE64,
    });

    prepareLangfuseSpanForExport(span, undefined, { omitDataUris: true });

    expect(span.attributes[INPUT]).toBe(PDF_DESCRIPTOR);
    expect(span.attributes[OUTPUT]).toBe(
      `[inline media omitted from trace: unknown type, ${PDF_BYTES.length} bytes]`
    );
  });

  it('leaves an attribute unchanged when it is no longer parseable JSON', () => {
    const truncated = JSON.stringify([
      userMessage({ type: 'media', mimeType: 'image/png', data: PDF_BASE64 }),
    ]).slice(0, 50_000);
    const span = createSpan({ [INPUT]: truncated });

    expect(() =>
      prepareLangfuseSpanForExport(span, undefined, { omitDataUris: true })
    ).not.toThrow();
    expect(span.attributes[INPUT]).toBe(truncated);
  });

  it('exports inline media verbatim when a run opts in', async () => {
    const span = await exportGeneration({
      langfuse: { inlineMediaTracing: { enabled: true } },
      input: [
        userMessage({
          type: 'media',
          mimeType: 'application/pdf',
          data: PDF_BASE64,
        }),
      ],
    });

    expect(readMessages(span)[0].content[0].data).toBe(PDF_BASE64);
  });

  it('exports inline media verbatim when LANGFUSE_TRACE_INLINE_MEDIA is set', async () => {
    process.env.LANGFUSE_TRACE_INLINE_MEDIA = 'true';
    const span = await exportGeneration({
      input: [
        userMessage({
          type: 'media',
          mimeType: 'application/pdf',
          data: PDF_BASE64,
        }),
      ],
    });

    expect(readMessages(span)[0].content[0].data).toBe(PDF_BASE64);
  });

  it('lets agent config override the run inline media policy', () => {
    const run: t.LangfuseConfig = { inlineMediaTracing: { enabled: true } };
    const agent: t.LangfuseConfig = { inlineMediaTracing: { enabled: false } };

    expect(resolveInlineMediaTracingEnabled(run, agent)).toBe(false);
    expect(resolveLangfuseConfig(run, {})?.inlineMediaTracing).toEqual({
      enabled: true,
    });
    expect(resolveLangfuseConfig(run, agent)?.inlineMediaTracing).toEqual({
      enabled: false,
    });
  });

  it('resolves media upload the same way the Langfuse SDK does', () => {
    expect(resolveLangfuseMediaUploadEnabled()).toBe(true);
    process.env.LANGFUSE_MEDIA_UPLOAD_ENABLED = 'FALSE';
    expect(resolveLangfuseMediaUploadEnabled()).toBe(false);
    process.env.LANGFUSE_MEDIA_UPLOAD_ENABLED = '0';
    expect(resolveLangfuseMediaUploadEnabled()).toBe(false);
    process.env.LANGFUSE_MEDIA_UPLOAD_ENABLED = 'no';
    expect(resolveLangfuseMediaUploadEnabled()).toBe(true);
    expect(resolveLangfuseMediaUploadEnabled({ mediaUploadEnabled: false })).toBe(
      false
    );
  });
});
