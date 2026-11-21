/** Width and height written into the PNG header of {@link screenshotDataUrl}. */
export const SCREENSHOT_DIMENSIONS = { width: 1280, height: 720 } as const;

/**
 * A base64 PNG data URL with a real IHDR header (1280x720), padded to about
 * `base64Chars` characters, the size of a browser screenshot from an MCP tool.
 */
export function screenshotDataUrl(base64Chars: number): string {
  const header = Buffer.alloc(33);
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).copy(header, 0);
  header.writeUInt32BE(13, 8);
  header.write('IHDR', 12, 'ascii');
  header.writeUInt32BE(SCREENSHOT_DIMENSIONS.width, 16);
  header.writeUInt32BE(SCREENSHOT_DIMENSIONS.height, 20);
  header[24] = 8;
  header[25] = 6;
  const bytes = Math.ceil(base64Chars / 4) * 3;
  const png = Buffer.concat([
    header,
    Buffer.alloc(Math.max(0, bytes - header.length), 7),
  ]);
  return `data:image/png;base64,${png.toString('base64')}`;
}
