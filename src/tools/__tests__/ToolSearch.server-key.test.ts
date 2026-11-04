import { describe, expect, it } from '@jest/globals';
import type { LCToolRegistry, ToolMetadata } from '@/types';
import {
  createToolSearch,
  formatServerListing,
  getDeferredToolsListing,
} from '../ToolSearch';

function registry(server: string): LCToolRegistry {
  return new Map([
    [
      'weather_mcp_weather',
      {
        name: 'weather_mcp_weather',
        description: 'Read the weather forecast',
        defer_loading: true,
      },
    ],
    [
      `status_mcp_${server}`,
      {
        name: `status_mcp_${server}`,
        description: 'Read service status',
        defer_loading: true,
      },
    ],
  ]);
}

function metadata(server: string): ToolMetadata[] {
  return Array.from(registry(server).values(), (tool) => ({
    name: tool.name,
    description: tool.description ?? '',
  }));
}

async function expectWeatherSearch(server: string): Promise<void> {
  const search = createToolSearch({
    mode: 'local',
    toolRegistry: registry(server),
  });
  const result = await search.invoke({
    query: 'weather',
    mcp_server: 'weather',
  });
  expect(JSON.parse(result)).toMatchObject({
    found: 1,
    tools: [{ name: 'weather_mcp_weather', score: 1 }],
  });
}

describe('MCP server grouping accepts ordinary string identifiers', () => {
  it('keeps an ordinary two-server listing', () => {
    const listing = getDeferredToolsListing(registry('status'), true);
    expect(listing).toContain('weather: weather');
    expect(listing).toContain('status: status');
  });

  it('searches the healthy server in the ordinary control', async () => {
    await expectWeatherSearch('status');
  });

  describe.each(['constructor', 'toString', 'hasOwnProperty'])(
    'server %s',
    (server) => {
      it('can build the initial deferred-tool listing', () => {
        const listing = getDeferredToolsListing(registry(server), true);
        expect(listing).toContain(`${server}: status`);
        expect(listing).toContain('weather: weather');
      });

      it('can format a server preview', () => {
        const result = formatServerListing(metadata(server), [server, 'weather']);
        expect(JSON.parse(result)).toMatchObject({
          total_tools: 2,
          tools_by_server: {
            [server]: [{ name: `status_mcp_${server}` }],
            weather: [{ name: 'weather_mcp_weather' }],
          },
        });
      });

      it('does not prevent ranked search for another server', async () => {
        await expectWeatherSearch(server);
      });
    }
  );
});
