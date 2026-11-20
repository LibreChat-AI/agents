/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable no-console */
// content.test.ts
import * as fs from 'fs';
import { processContent } from './content';

describe('Link Processor', () => {
  afterAll(() => {
    if (fs.existsSync('./temp.html')) {
      fs.unlinkSync('./temp.html');
    }
    if (fs.existsSync('./temp.md')) {
      fs.unlinkSync('./temp.md');
    }
  });
  // Basic functionality tests
  test('should replace basic links with references', () => {
    const html = `
      <p>Test with <a href="https://example.com/link" title="Example">a link</a></p>
      <p>And an <img src="https://example.com/img.jpg" alt="image"></p>
      <p>Plus a <video src="https://example.com/video.mp4"></video></p>
    `;

    const markdown = `
      Test with [a link](https://example.com/link "Example")
      And an ![image](https://example.com/img.jpg)
      Plus a [video](https://example.com/video.mp4)
    `;

    const result = processContent(html, markdown);

    expect(result.links.length).toBe(1);
    expect(result.images.length).toBe(1);
    expect(result.videos.length).toBe(1);
    expect(result.markdown).toContain('link#1');
    expect(result.markdown).toContain('image#1');
    expect(result.markdown).toContain('video#1');
  });

  // Edge case tests
  test('should handle links with parentheses and special characters', () => {
    const html = `
      <a href="https://example.com/page(1).html" title="Parens">Link with parens</a>
      <a href="https://example.com/path?query=test&param=value">Link with query</a>
    `;

    const markdown = `
      [Link with parens](https://example.com/page(1).html "Parens")
      [Link with query](https://example.com/path?query=test&param=value)
    `;

    const result = processContent(html, markdown);

    expect(result.links.length).toBe(2);
    expect(result.markdown).toContain('link#1');
    expect(result.markdown).toContain('link#2');
  });

  test('prefers the longest known URL at a position, including URLs with parentheses @scenario:scraped-page-references-keep-their-previous-output', () => {
    const html = `
      <a href="https://w.org/wiki/Foo_(bar)">a</a>
      <a href="https://w.org/wiki/Foo_">b</a>
      <a href="https://w.org/wiki/Foo_(bar)_baz">c</a>
    `;
    const markdown =
      '[a](https://w.org/wiki/Foo_(bar)) [b](https://w.org/wiki/Foo_) [c](https://w.org/wiki/Foo_(bar)_baz "T x") (https://w.org/wiki/Foo_';

    const result = processContent(html, markdown);

    expect(result.markdown).toBe(
      '[a](link#1) [b](link#2) [c](link#3 "T x") (https://w.org/wiki/Foo_'
    );
  });

  test('handles a URL whose prefix repeats without a quadratic rescan or a regex limit', () => {
    /** A trie walk restarted at every inner parenthesis took ~6 s here, and one
     *  regex per URL threw "Invalid regular expression" for a URL this long. */
    const prefix = 'https://x/(';
    const html = `<a href="${prefix.repeat(5000)}a">x</a>`;
    const markdown = `(${prefix.repeat(5000)}b) [x](${prefix.repeat(5000)}a)`;

    const start = performance.now();
    const result = processContent(html, markdown);
    const elapsed = performance.now() - start;

    expect(result.markdown).toBe(`(${prefix.repeat(5000)}b) [x](link#1)`);
    expect(elapsed).toBeLessThan(1500);
  });

  test('keeps the image reference when a URL is both an image and a link', () => {
    const html =
      '<a href="https://x.io/a.png">a</a><img src="https://x.io/a.png" alt="i">';
    const markdown = '![i](https://x.io/a.png) [a](https://x.io/a.png "t")';

    const result = processContent(html, markdown);

    expect(result.markdown).toBe('![i](image#1) [a](image#1 "t")');
    expect(result.links).toHaveLength(1);
    expect(result.images).toHaveLength(1);
  });

  test('rewrites a link-heavy page in time linear in its size', () => {
    /** One regex per URL over the whole page grew with links times page size (4,000
     *  links took ~3 s, doubling the page quintupled it) and blocked the event loop. */
    let html = '<html><body>';
    let markdown = '';
    for (let i = 0; i < 6000; i++) {
      const url = `https://example.com/docs/section-${i}/page-${i * 7}`;
      html += `<p><a href="${url}">l${i}</a> <img src="${url}.png" alt="i${i}"></p>`;
      markdown += `text [l${i}](${url}) ![i${i}](${url}.png)\n`;
    }

    const start = performance.now();
    const result = processContent(html, markdown);
    const elapsed = performance.now() - start;

    expect(result.links).toHaveLength(6000);
    expect(result.images).toHaveLength(6000);
    expect(result.markdown).not.toMatch(/\]\(https:\/\/example\.com\//);
    expect(elapsed).toBeLessThan(1500);
  });

  test('rewrites links of many different lengths in time linear in the page @scenario:link-heavy-scraped-page-is-rewritten-in-linear-time', () => {
    /** A query that grows per entry gives every link its own URL length, which
     *  defeats any scheme that tries every known length at every position. */
    let html = '<html><body>';
    let markdown = '';
    for (let i = 0; i < 2000; i++) {
      const url = `https://example.com/search?q=${'a'.repeat(i)}`;
      html += `<p><a href="${url}">l${i}</a></p>`;
      markdown += `text [l${i}](${url} "t${i}")\n`;
    }

    const start = performance.now();
    const result = processContent(html, markdown);
    const elapsed = performance.now() - start;

    expect(result.links).toHaveLength(2000);
    expect(result.markdown).toContain('[l1999](link#2000 "t1999")');
    expect(result.markdown).not.toMatch(/\]\(https:\/\/example\.com\//);
    expect(elapsed).toBeLessThan(1500);
  });

  // Performance test with large files
  test('should process large files efficiently', () => {
    const html = fs.readFileSync('src/tools/search/test.html', 'utf-8');
    const markdown = fs.readFileSync('src/tools/search/test.md', 'utf-8');

    // const largeHtml = generateLargeHtml(1000); // 1000 links
    // fs.writeFileSync('./temp.html', largeHtml);

    // const largeMd = generateLargeMarkdown(1000); // 1000 links
    // fs.writeFileSync('./temp.md', largeMd);

    // const html = fs.readFileSync('./temp.html', 'utf-8');
    // const markdown = fs.readFileSync('./temp.md', 'utf-8');

    // Measure time taken to process
    const startTime = process.hrtime();
    const result = processContent(html, markdown);
    const elapsed = process.hrtime(startTime);
    const timeInMs = elapsed[0] * 1000 + elapsed[1] / 1000000;

    console.log(
      `Processed ${result.links.length} links, ${result.images.length} images, and ${result.videos.length} videos in ${timeInMs.toFixed(2)}ms`
    );

    // Basic validations for large file processing
    expect(result.links.length).toBeGreaterThan(0);
    expect(result.markdown).toContain('link#');

    // Check if all links were replaced (sample check)
    expect(result.markdown).not.toContain('https://example.com/link');
  });

  // Memory usage test
  test('should have reasonable memory usage', () => {
    const html = fs.readFileSync('src/tools/search/test.html', 'utf-8');
    const markdown = fs.readFileSync('src/tools/search/test.md', 'utf-8');

    const beforeMem = process.memoryUsage();
    processContent(html, markdown);
    const afterMem = process.memoryUsage();

    const heapUsed = (afterMem.heapUsed - beforeMem.heapUsed) / 1024 / 1024; // MB

    console.log(`Memory used: ${heapUsed.toFixed(2)} MB`);

    // This is a loose check - actual thresholds depend on your environment
    expect(heapUsed).toBeLessThan(100); // Should use less than 100MB additional heap
  });

  // Real-world file test (if available)
  test('should process real-world Wikipedia content', () => {
    // Try to find real-world test files if they exist
    const wikiHtml = 'src/tools/search/test.html';
    const wikiMd = 'src/tools/search/test.md';

    if (fs.existsSync(wikiHtml) && fs.existsSync(wikiMd)) {
      const html = fs.readFileSync(wikiHtml, 'utf-8');
      const markdown = fs.readFileSync(wikiMd, 'utf-8');

      const result = processContent(html, markdown);

      console.log(
        `Processed ${result.links.length} Wikipedia links, ${result.images.length} images, and ${result.videos.length} videos`
      );

      expect(result.links.length).toBeGreaterThan(10); // Wikipedia articles typically have many links
      expect(result.markdown).not.toMatch(/\]\(https?:\/\/[^\s")]+\)/); // No regular URLs should remain
    } else {
      console.log('Wikipedia test files not found, skipping this test');
    }
  });
});

// Helper function to generate large HTML test data
function generateLargeHtml(linkCount: number): string {
  let html = '<html><body>';

  for (let i = 1; i <= linkCount; i++) {
    html += `<p>Paragraph ${i} with <a href="https://example.com/link${i}" title="Link ${i}">link ${i}</a>`;

    if (i % 10 === 0) {
      html += ` and <img src="https://example.com/image${i / 10}.jpg" alt="Image ${i / 10}">`;
    }

    if (i % 50 === 0) {
      html += ` and <video src="https://example.com/video${i / 50}.mp4" title="Video ${i / 50}"></video>`;
    }

    html += '</p>';
  }

  html += '</body></html>';
  return html;
}

/** Helper function to generate large Markdown test data  */
function generateLargeMarkdown(linkCount: number): string {
  let markdown = '# Test Document\n\n';

  for (let i = 1; i <= linkCount; i++) {
    markdown += `Paragraph ${i} with [link ${i}](https://example.com/link${i} "Link ${i}")`;

    if (i % 10 === 0) {
      markdown += ` and ![Image ${i / 10}](https://example.com/image${i / 10}.jpg)`;
    }

    if (i % 50 === 0) {
      markdown += ` and [Video ${i / 50}](https://example.com/video${i / 50}.mp4 "Video ${i / 50}")`;
    }

    markdown += '\n\n';
  }

  return markdown;
}
