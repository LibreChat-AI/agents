import { tmpdir } from 'os';
import { join } from 'path';
import { createHash } from 'crypto';
import { AIMessage } from '@langchain/core/messages';
import { mkdtemp, rm, symlink, unlink, writeFile } from 'fs/promises';
import type { ToolMessage } from '@langchain/core/messages';
import type { WorkspaceFS } from '../local/workspaceFS';
import type { ToolSearchArtifact } from '@/types';
import {
  createLocalCodingToolDefinitions,
  createLocalReadFileTool,
} from '../local/LocalCodingTools';
import { createLocalProgrammaticProfile } from '../local/LocalProgrammaticProfile';
import { nodeWorkspaceFS } from '../local/workspaceFS';
import { ToolNode } from '../ToolNode';

function continuation(text: string): { body: string; cursor?: string } {
  const match = /\n\[continue with cursor=(v1:[a-f0-9]{64}:\d+)\]$/.exec(text);
  return match == null
    ? { body: text }
    : { body: text.slice(0, match.index), cursor: match[1] };
}

describe('bounded source recovery', () => {
  let cwd: string;

  beforeEach(async () => {
    cwd = await mkdtemp(join(tmpdir(), 'lc-source-recovery-'));
  });
  afterEach(async () => {
    await rm(cwd, { recursive: true, force: true });
  });

  it('recovers every character of a long line before advancing to the next line', async () => {
    const source = `${'0123456789'.repeat(300)}\nlast`;
    await writeFile(join(cwd, 'source.txt'), source);
    const read = createLocalReadFileTool({ cwd, toolResultMaxChars: 300 });
    let cursor: string | undefined;
    let recovered = '';
    for (let page = 0; page < 20; page++) {
      const output = String(
        await read.invoke({
          path: 'source.txt',
          ...(cursor == null ? {} : { cursor }),
          line_numbers: false,
        })
      );
      const next = continuation(output);
      recovered += next.body;
      cursor = next.cursor;
      if (cursor == null) break;
    }
    expect(cursor).toBeUndefined();
    expect(recovered).toBe(source);
  });

  it.each(['b'.repeat(900), 'short', ''])(
    'rejects a continuation after a content change (%#. replacement)',
    async (replacement) => {
      await writeFile(join(cwd, 'source.txt'), 'a'.repeat(900));
      const read = createLocalReadFileTool({ cwd, toolResultMaxChars: 300 });
      const { cursor } = continuation(
        String(await read.invoke({ path: 'source.txt' }))
      );
      expect(cursor).toBeDefined();
      await writeFile(join(cwd, 'source.txt'), replacement);
      await expect(read.invoke({ path: 'source.txt', cursor })).rejects.toThrow(
        /content changed/
      );
    }
  );

  it.each([
    { fileCheckpointing: false, eventDrivenMode: false },
    { fileCheckpointing: true, eventDrivenMode: false },
    { fileCheckpointing: false, eventDrivenMode: true },
    { fileCheckpointing: true, eventDrivenMode: true },
  ])(
    'preserves the managed page and cursor (checkpointing=$fileCheckpointing, event=$eventDrivenMode)',
    async ({ fileCheckpointing, eventDrivenMode }) => {
      const source = '0123456789'.repeat(200);
      await writeFile(join(cwd, 'source.txt'), source);
      const node = new ToolNode({
        tools: [],
        maxToolResultChars: 300,
        eventDrivenMode,
        toolCallStepIds: new Map([['read', 'read-step']]),
        toolExecution: {
          engine: 'local',
          local: {
            cwd,
            fileCheckpointing,
            includeCodingTools: true,
            codingToolNames: ['read_file'],
            toolResultMaxChars: 1000,
          },
        },
      });
      const result = (await node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              {
                id: 'read',
                name: 'read_file',
                args: { path: 'source.txt', line_numbers: false },
              },
            ],
          }),
        ],
      })) as { messages: ToolMessage[] };
      const output = String(result.messages[0].content);
      const page = continuation(output);
      expect(result.messages[0].status).toBe('success');
      expect(output.length).toBeLessThanOrEqual(300);
      expect(page.cursor).toBeDefined();
      expect(page.body).toBe(source.slice(0, page.body.length));
      expect(page.body.length).toBe(Number(page.cursor?.split(':')[2]));
    }
  );

  it.each([9, 10, 11, 17, 31, 128])(
    'covers numbered and unnumbered source boundaries at a %s-character body cap',
    async (cap) => {
      const source = 'a🙂\r\n\n12345🙂suffix\nlast';
      await writeFile(join(cwd, 'source.txt'), source);
      const read = createLocalReadFileTool({ cwd, toolResultMaxChars: cap });
      for (const numbered of [false, true]) {
        let cursor: string | undefined;
        let recovered = '';
        for (let pageIndex = 0; pageIndex < 40; pageIndex++) {
          const page = continuation(
            String(
              await read.invoke({
                path: 'source.txt',
                line_numbers: numbered,
                ...(cursor == null ? {} : { cursor }),
              })
            )
          );
          expect(page.body.length).toBeLessThanOrEqual(cap);
          recovered += numbered
            ? page.body.replace(/^ *\d+\t/gm, '')
            : page.body;
          if (page.cursor != null) expect(page.cursor).not.toBe(cursor);
          cursor = page.cursor;
          if (cursor == null) break;
        }
        expect(cursor).toBeUndefined();
        expect(Buffer.from(recovered)).toEqual(Buffer.from(source));
      }
    }
  );

  it.each(['', '\n', '\ufeffa\r\nb\r\n', '🙂界\r\n'.repeat(20), 'end🙂'])(
    'reconstructs UTF-8 source bytes including newlines and BOM (%j)',
    async (source) => {
      const bytes = Buffer.from(source);
      await writeFile(join(cwd, 'source.txt'), bytes);
      const read = createLocalReadFileTool({ cwd, toolResultMaxChars: 9 });
      let cursor: string | undefined;
      let recovered = '';
      for (let index = 0; index < 100; index++) {
        const message = (await read.invoke({
          type: 'tool_call',
          id: `page-${index}`,
          name: 'read_file',
          args: {
            path: 'source.txt',
            line_numbers: false,
            ...(cursor == null ? {} : { cursor }),
          },
        })) as ToolMessage;
        const page = continuation(String(message.content));
        expect(Buffer.from(page.body).toString()).toBe(page.body);
        expect(message.artifact).toMatchObject({
          bytes: bytes.length,
          recovery: {
            sha256: createHash('sha256').update(bytes).digest('hex'),
            start: recovered.length,
            end: recovered.length + page.body.length,
            bodyChars: page.body.length,
            unit: 'utf16',
            nextCursor: page.cursor,
          },
        });
        recovered += page.body;
        cursor = page.cursor;
        if (cursor == null) break;
      }
      expect(cursor).toBeUndefined();
      expect(Buffer.from(recovered)).toEqual(bytes);
    }
  );

  it('supports explicit line ranges and continuation even when the receiving tool has no local cap', async () => {
    await writeFile(join(cwd, 'source.txt'), 'one\r\ntwo\r\nthree');
    const read = createLocalReadFileTool({ cwd, toolResultMaxChars: 100 });
    const first = continuation(
      String(
        await read.invoke({
          path: 'source.txt',
          offset: 2,
          limit: 1,
          line_numbers: false,
        })
      )
    );
    expect(first.body).toBe('two\r\n');
    expect(first.cursor).toBeDefined();
    const uncapped = createLocalReadFileTool({ cwd });
    expect(
      await uncapped.invoke({
        path: 'source.txt',
        cursor: first.cursor,
        line_numbers: false,
      })
    ).toBe('three');
    expect(await read.invoke({ path: 'source.txt', offset: 20 })).toBe('');
  });

  it('rejects malformed, conflicting, out-of-range and mid-character cursors', async () => {
    const bytes = Buffer.from('a🙂b');
    const digest = createHash('sha256').update(bytes).digest('hex');
    await writeFile(join(cwd, 'source.txt'), bytes);
    const read = createLocalReadFileTool({ cwd, toolResultMaxChars: 100 });
    for (const cursor of [
      '',
      'v2:' + digest + ':1',
      'v1:' + digest + ':-1',
      'v1:' + digest + ':01',
      'v1:' + digest + ':9007199254740992',
      'v1:' + digest + ':8',
      'v1:' + digest + ':2',
    ]) {
      await expect(read.invoke({ path: 'source.txt', cursor })).rejects.toThrow(
        /cursor/
      );
    }
    await expect(
      read.invoke({ path: 'source.txt', cursor: `v1:${digest}:1`, offset: 1 })
    ).rejects.toThrow(/either cursor or offset/);
    expect(
      await read.invoke({
        path: 'source.txt',
        cursor: `v1:${digest}:4`,
        line_numbers: false,
      })
    ).toBe('');
  });

  it('rechecks workspace authorization on a continuation after a symlink replacement', async () => {
    const outside = await mkdtemp(join(tmpdir(), 'lc-source-outside-'));
    try {
      await writeFile(join(cwd, 'source.txt'), 'a'.repeat(1000));
      await writeFile(join(outside, 'source.txt'), 'a'.repeat(1000));
      const read = createLocalReadFileTool({ cwd, toolResultMaxChars: 100 });
      const first = continuation(
        String(await read.invoke({ path: 'source.txt' }))
      );
      expect(first.cursor).toBeDefined();
      await unlink(join(cwd, 'source.txt'));
      await symlink(join(outside, 'source.txt'), join(cwd, 'source.txt'));
      await expect(
        read.invoke({ path: 'source.txt', cursor: first.cursor })
      ).rejects.toThrow(/outside the local workspace/);
    } finally {
      await rm(outside, { recursive: true, force: true });
    }
  });

  it('reads captured bytes through WorkspaceFS and checks the actual size after stat', async () => {
    await writeFile(join(cwd, 'source.txt'), 'stub');
    const fs: WorkspaceFS = { ...nodeWorkspaceFS };
    const readFile = jest.spyOn(fs, 'readFile');
    const read = createLocalReadFileTool({
      cwd,
      toolResultMaxChars: 100,
      maxReadBytes: 20,
      exec: { fs },
    });
    expect(await read.invoke({ path: 'source.txt', line_numbers: false })).toBe(
      'stub'
    );
    expect(readFile).toHaveBeenCalledWith(join(cwd, 'source.txt'));
    readFile.mockImplementationOnce(async () => Buffer.from('x'.repeat(21)));
    await expect(read.invoke({ path: 'source.txt' })).rejects.toThrow(
      /grew beyond/
    );
  });

  it('rejects a continuation if the file has grown beyond the read-byte cap', async () => {
    await writeFile(join(cwd, 'source.txt'), 'x'.repeat(300));
    const read = createLocalReadFileTool({
      cwd,
      toolResultMaxChars: 100,
      maxReadBytes: 400,
    });
    const first = continuation(
      String(await read.invoke({ path: 'source.txt' }))
    );
    expect(first.cursor).toBeDefined();
    await writeFile(join(cwd, 'source.txt'), 'x'.repeat(401));
    await expect(
      read.invoke({ path: 'source.txt', cursor: first.cursor })
    ).rejects.toThrow(/read cap/);
  });

  it('rejects invalid UTF-8 and binary continuations without silently attaching or replacing content', async () => {
    await writeFile(join(cwd, 'source.txt'), 'x'.repeat(300));
    const read = createLocalReadFileTool({
      cwd,
      toolResultMaxChars: 100,
      attachReadAttachments: 'images-only',
    });
    const first = continuation(
      String(await read.invoke({ path: 'source.txt' }))
    );
    expect(first.cursor).toBeDefined();
    await writeFile(join(cwd, 'source.txt'), Buffer.from([0, 1, 2]));
    await expect(
      read.invoke({ path: 'source.txt', cursor: first.cursor })
    ).rejects.toThrow(/UTF-8 text/);
    await writeFile(join(cwd, 'source.txt'), Buffer.from([0xc3, 0x28]));
    await expect(
      read.invoke({ path: 'source.txt', line_numbers: false })
    ).rejects.toThrow(/encoded data/);
  });

  it('fails clearly when the cap cannot fit a cursor or a complete character, while allowing fitting complete data', async () => {
    await writeFile(join(cwd, 'source.txt'), 'x'.repeat(200));
    const read = createLocalReadFileTool(
      { cwd, toolResultMaxChars: 100 },
      { maxResultChars: 80 }
    );
    await expect(read.invoke({ path: 'source.txt' })).rejects.toThrow(
      /too small for a continuation cursor/
    );
    await writeFile(join(cwd, 'source.txt'), '🙂🙂');
    const tiny = createLocalReadFileTool({ cwd, toolResultMaxChars: 1 });
    await expect(
      tiny.invoke({ path: 'source.txt', line_numbers: false })
    ).rejects.toThrow(/too small for one source character/);
    await writeFile(join(cwd, 'source.txt'), '{"value":1}');
    expect(await read.invoke({ path: 'source.txt', line_numbers: false })).toBe(
      '{"value":1}'
    );
  });

  it.each(['compact', 'hybrid'] as const)(
    'recovers nested pages through the %s programmatic surface with a smaller outer cap',
    async (mode) => {
      const source = '123456789'.repeat(120) + '\nend';
      const digest = createHash('sha256').update(source).digest('hex');
      await writeFile(join(cwd, 'source.txt'), source);
      const profile = createLocalProgrammaticProfile({
        mode,
        local: {
          cwd,
          toolResultMaxChars: 1000,
          codingToolNames: ['read_file'],
          sandbox: { enabled: false },
        },
      });
      const definition = createLocalCodingToolDefinitions().find(
        (entry) => entry.name === 'read_file'
      );
      expect(definition?.parameters).toHaveProperty('properties.cursor');
      expect(profile.toolRegistry.get('read_file')?.parameters).toHaveProperty(
        'properties.cursor'
      );
      const discovered: string[] = [];
      const node = new ToolNode({
        ...profile,
        maxToolResultChars: 300,
        getDiscoveredToolNames: () => discovered,
      });
      if (mode === 'compact') {
        const discovery = (await node.invoke({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                {
                  id: 'discover',
                  name: 'tool_search',
                  args: { query: 'read_file' },
                },
              ],
            }),
          ],
        })) as { messages: ToolMessage[] };
        expect(discovery.messages[0].status).toBe('success');
        const artifact = discovery.messages[0].artifact as ToolSearchArtifact;
        discovered.push(
          ...artifact.tool_references.map((reference) => reference.tool_name)
        );
        expect(discovered).toContain('read_file');
      }
      const code = [
        'import re,hashlib',
        'args={"path":"source.txt","line_numbers":False}',
        'parts=[]',
        'for _ in range(20):',
        '    page=await read_file(**args)',
        '    match=re.search(r"\\n\\[continue with cursor=(v1:[a-f0-9]{64}:\\d+)\\]$",page)',
        '    parts.append(page[:match.start()] if match else page)',
        '    if not match: break',
        '    args["cursor"]=match.group(1)',
        'else: raise RuntimeError("incomplete recovery")',
        'print(hashlib.sha256("".join(parts).encode()).hexdigest())',
      ].join('\n');
      const result = (await node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              {
                id: 'program',
                name: 'run_tools_with_code',
                args: { lang: 'py', code, tool_manifest: ['read_file'] },
              },
            ],
          }),
        ],
      })) as { messages: ToolMessage[] };
      expect(String(result.messages[0].content)).toContain(digest);
      expect(result.messages[0].status).toBe('success');
      expect(String(result.messages[0].content)).not.toContain('exit_code:');
    }
  );
});
