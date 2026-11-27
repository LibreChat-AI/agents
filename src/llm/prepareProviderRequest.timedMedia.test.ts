import { describe, expect, it } from '@jest/globals';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage, ContentBlock } from '@langchain/core/messages';
import type * as t from '@/types';
import {
  _convertMessagesToOpenAIParams,
  _convertMessagesToOpenAIResponsesParams,
} from '@/llm/openai/utils';
import { _convertMessagesToAnthropicPayload } from '@/llm/anthropic/utils/message_inputs';
import { convertToConverseMessages } from '@/llm/bedrock/utils/message_inputs';
import { convertBaseMessagesToContent } from '@/llm/google/utils/common';
import { prepareProviderRequest } from '@/llm/prepareProviderRequest';
import { Providers } from '@/common';

const AUDIO = Buffer.from('ID3 recorded memo').toString('base64');
const VIDEO = Buffer.from('ftypmp42 recorded clip').toString('base64');
const prompt = { type: 'text', text: 'Summarize the attachment' };

/** Gemini's own parts, as a host builds them for a Gemini or Vertex agent. */
const geminiAudio = { type: 'media', mimeType: 'audio/mpeg', data: AUDIO };
const geminiVideo = { type: 'media', mimeType: 'video/mp4', data: VIDEO };
/** OpenAI-compatible parts, as a host builds them for an OpenRouter agent. */
const inputAudio = {
  type: 'input_audio',
  input_audio: { data: AUDIO, format: 'mp3' },
};
const videoUrl = {
  type: 'video_url',
  video_url: { url: `data:video/mp4;base64,${VIDEO}` },
};
/** Audio in a LangChain standard file block. */
const fileAudio = {
  type: 'file',
  source_type: 'base64',
  mime_type: 'audio/mpeg',
  data: AUDIO,
};

const timedMedia = [
  ['Gemini audio', geminiAudio],
  ['Gemini video', geminiVideo],
  ['OpenRouter audio', inputAudio],
  ['OpenRouter video', videoUrl],
  ['standard file audio', fileAudio],
] as const;

/** The messages `provider` receives for `messages`, as the run prepares its request. */
function prepare(
  provider: t.ProviderName,
  messages: BaseMessage[],
  { responses = false }: { responses?: boolean } = {}
): BaseMessage[] {
  const model: { model: string; _useResponsesApi?: () => boolean } = {
    model: 'prepared-model',
  };
  if (responses) {
    model._useResponsesApi = (): boolean => true;
  }
  return prepareProviderRequest({
    model: model as unknown as t.ChatModel,
    messages,
    provider,
  }).messages;
}

function attach(...parts: ContentBlock[]): HumanMessage {
  return new HumanMessage({ content: [prompt, ...parts] });
}

describe('audio and video in a request to a provider that cannot read them', () => {
  it.each([
    ['audio', geminiAudio],
    ['video', geminiVideo],
  ])(
    '@scenario:gemini-media-left-out-for-claude sends Claude the message without a Gemini agent\'s %s',
    (_kind, media) => {
      const source = attach(media);
      const messages = prepare(Providers.ANTHROPIC, [source]);

      expect(messages[0].content).toEqual([prompt]);
      expect(_convertMessagesToAnthropicPayload(messages).messages).toEqual([
        { role: 'user', content: [prompt] },
      ]);
      expect(source.content).toEqual([prompt, media]);
    }
  );

  it.each([
    Providers.OPENAI,
    Providers.AZURE,
    Providers.OPENROUTER,
    Providers.XAI,
    Providers.DEEPSEEK,
    Providers.MOONSHOT,
  ])(
    '@scenario:gemini-media-left-out-for-openai-chat sends %s Chat Completions the message without a Gemini agent\'s audio and video',
    (provider) => {
      const messages = prepare(provider, [attach(geminiAudio, geminiVideo)]);

      expect(messages[0].content).toEqual([prompt]);
      expect(_convertMessagesToOpenAIParams(messages)).toEqual([
        { role: 'user', content: [prompt] },
      ]);
    }
  );

  it.each(timedMedia)(
    '@scenario:timed-media-left-out-for-bedrock sends Bedrock the message without %s',
    (_kind, media) => {
      const messages = prepare(Providers.BEDROCK, [attach(media)]);

      expect(messages[0].content).toEqual([prompt]);
      expect(convertToConverseMessages(messages).converseMessages).toEqual([
        { role: 'user', content: [{ text: prompt.text }] },
      ]);
    }
  );

  it('@scenario:timed-media-left-out-for-bedrock keeps Bedrock documents beside dropped media', () => {
    const document = {
      type: 'document',
      document: {
        name: 'sales',
        format: 'csv',
        source: { bytes: new Uint8Array([99, 111, 108]) },
      },
    };
    const messages = prepare(Providers.BEDROCK, [attach(document, videoUrl)]);

    expect(messages[0].content).toEqual([prompt, document]);
    expect((messages[0].content as unknown[])[1]).toBe(document);
  });

  it.each([
    ['audio', inputAudio],
    ['video', videoUrl],
  ])(
    '@scenario:openai-media-left-out-for-claude-and-responses sends Claude and the Responses API the message without an OpenRouter agent\'s %s',
    (_kind, media) => {
      const source = attach(media);
      const claude = prepare(Providers.ANTHROPIC, [source]);
      const responses = prepare(Providers.OPENAI, [source], {
        responses: true,
      });

      expect(claude[0].content).toEqual([prompt]);
      expect(_convertMessagesToAnthropicPayload(claude).messages).toEqual([
        { role: 'user', content: [prompt] },
      ]);
      expect(responses[0].content).toEqual([prompt]);
      expect(_convertMessagesToOpenAIResponsesParams(responses)).toEqual([
        {
          type: 'message',
          role: 'user',
          content: [{ type: 'input_text', text: prompt.text }],
        },
      ]);
    }
  );

  it('leaves every timed shape out for the Responses API', () => {
    const messages = prepare(
      Providers.OPENAI,
      [attach(geminiAudio, geminiVideo, fileAudio)],
      { responses: true }
    );

    expect(messages[0].content).toEqual([prompt]);
  });

  it('keeps an attachment-only message valid when its only part is left out', () => {
    const messages = prepare(Providers.ANTHROPIC, [
      new HumanMessage({ content: [geminiVideo] }),
    ]);

    expect(messages[0].content).toEqual([
      {
        type: 'text',
        text: '[Attachment omitted because its binary format is unsupported by this provider.]',
      },
    ]);
  });

  it('moves a prompt-cache marker from a dropped part to the last cacheable block', () => {
    const marker = { type: 'ephemeral' };
    const messages = prepare(Providers.ANTHROPIC, [
      attach({ ...geminiAudio, cache_control: marker }),
    ]);

    expect(messages[0].content).toEqual([{ ...prompt, cache_control: marker }]);
  });

  it('reads MIME types with parameters and in upper case, and file URIs', () => {
    const messages = prepare(Providers.ANTHROPIC, [
      attach(
        { type: 'media', mimeType: 'Audio/WAV; rate=16000', data: AUDIO },
        {
          type: 'media',
          mimeType: 'video/mp4',
          fileUri: 'https://www.youtube.com/watch?v=abc',
        }
      ),
    ]);

    expect(messages[0].content).toEqual([prompt]);
  });

  it('only rewrites the messages that carried timed media', () => {
    const earlier = new HumanMessage({ content: [prompt] });
    const later = attach(geminiAudio);
    const messages = prepare(Providers.ANTHROPIC, [earlier, later]);

    expect(messages[0]).toBe(earlier);
    expect(messages[1]).not.toBe(later);
    expect(messages[1].content).toEqual([prompt]);
  });

  it('leaves tool results to their own tool-output projection', () => {
    const tool = new ToolMessage({
      tool_call_id: 'call_1',
      content: [prompt, inputAudio],
    });
    const messages = prepare(Providers.BEDROCK, [tool]);

    expect(messages[0].content).toEqual([prompt, inputAudio]);
  });
});

describe('audio and video in a request to a provider that reads them', () => {
  it.each([Providers.GOOGLE, Providers.VERTEXAI])(
    '@scenario:openai-media-sent-to-gemini-as-inline-media sends %s an OpenRouter agent\'s audio and video as its own media parts',
    (provider) => {
      const source = attach(inputAudio, videoUrl);
      const messages = prepare(provider, [source]);

      expect(messages[0].content).toEqual([
        prompt,
        { type: 'media', mimeType: 'audio/mp3', data: AUDIO },
        { type: 'media', mimeType: 'video/mp4', data: VIDEO },
      ]);
      expect(source.content).toEqual([prompt, inputAudio, videoUrl]);
    }
  );

  it('@scenario:openai-media-sent-to-gemini-as-inline-media gives Gemini\'s request the inline audio and video', () => {
    const messages = prepare(Providers.GOOGLE, [attach(inputAudio, videoUrl)]);

    expect(convertBaseMessagesToContent(messages, true)).toEqual([
      {
        role: 'user',
        parts: [
          { text: prompt.text },
          { inlineData: { mimeType: 'audio/mp3', data: AUDIO } },
          { inlineData: { mimeType: 'video/mp4', data: VIDEO } },
        ],
      },
    ]);
  });

  it.each([
    ['wav', 'audio/wav'],
    ['mp3', 'audio/mp3'],
    ['aac', 'audio/aac'],
    ['ogg', 'audio/ogg'],
    ['flac', 'audio/flac'],
    ['aiff', 'audio/aiff'],
    ['m4a', 'audio/mp4'],
    ['webm', 'audio/webm'],
    ['WAV', 'audio/wav'],
  ])('gives Gemini %s audio as %s', (format, mimeType) => {
    const messages = prepare(Providers.GOOGLE, [
      attach({ type: 'input_audio', input_audio: { data: AUDIO, format } }),
    ]);

    expect(messages[0].content).toEqual([
      prompt,
      { type: 'media', mimeType, data: AUDIO },
    ]);
  });

  it('reads a video_url given as a string and an x- video type', () => {
    const messages = prepare(Providers.GOOGLE, [
      attach({
        type: 'video_url',
        video_url: `data:video/x-matroska;codecs=avc1;base64,${VIDEO}`,
      }),
    ]);

    expect(messages[0].content).toEqual([
      prompt,
      { type: 'media', mimeType: 'video/x-matroska', data: VIDEO },
    ]);
  });

  it.each([
    [
      'raw PCM, which names no sample rate',
      { type: 'input_audio', input_audio: { data: AUDIO, format: 'pcm16' } },
    ],
    [
      'audio without data',
      { type: 'input_audio', input_audio: { format: 'mp3' } },
    ],
    [
      'a remote video URL',
      { type: 'video_url', video_url: { url: 'https://example.com/clip.mp4' } },
    ],
    [
      'a video data URL that is not base64',
      { type: 'video_url', video_url: { url: 'data:video/mp4,clip' } },
    ],
    ['a video part without a URL', { type: 'video_url', video_url: {} }],
  ])('leaves out %s for Gemini instead of sending it', (_kind, media) => {
    const messages = prepare(Providers.GOOGLE, [attach(media)]);

    expect(messages[0].content).toEqual([prompt]);
  });

  it.each([Providers.GOOGLE, Providers.VERTEXAI])(
    '@scenario:readable-timed-media-unchanged sends %s its own media parts and standard audio files as they are',
    (provider) => {
      const source = attach(geminiAudio, geminiVideo, fileAudio, {
        type: 'media',
        mimeType: 'video/mp4',
        fileUri: 'https://www.youtube.com/watch?v=abc',
      });
      const messages = prepare(provider, [source]);

      expect(messages[0]).toBe(source);
    }
  );

  it.each([
    Providers.OPENAI,
    Providers.OPENROUTER,
    Providers.XAI,
    Providers.MOONSHOT,
  ])(
    '@scenario:readable-timed-media-unchanged sends %s Chat Completions its input_audio and video_url parts as they are',
    (provider) => {
      const source = attach(inputAudio, videoUrl);
      const messages = prepare(provider, [source]);

      expect(messages[0]).toBe(source);
      expect(_convertMessagesToOpenAIParams(messages)).toEqual([
        { role: 'user', content: [prompt, inputAudio, videoUrl] },
      ]);
    }
  );

  it('@scenario:readable-timed-media-unchanged leaves Gemini media parts that are not audio or video alone', () => {
    const pdf = {
      type: 'media',
      mimeType: 'application/pdf',
      data: 'JVBERg==',
    };
    const source = attach(pdf);

    expect(prepare(Providers.ANTHROPIC, [source])[0]).toBe(source);
  });
});
