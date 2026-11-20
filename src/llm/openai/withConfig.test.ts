import { ChatMoonshot, ChatOpenAI } from './index';

class InspectableChatOpenAI extends ChatOpenAI {
  set firstParty(value: boolean) {
    this.firstPartyEndpoint = value;
  }

  get state() {
    return {
      firstPartyEndpoint: this.firstPartyEndpoint,
      completions: this.completions,
      responses: this.responses,
      options: this.defaultOptions,
    };
  }
}

const tool = {
  type: 'function' as const,
  function: {
    name: 'noop',
    parameters: { type: 'object', properties: {} },
  },
};

test('clones preserve runtime fields and delegates without mutating their source', () => {
  const model = new InspectableChatOpenAI({
    model: 'gpt-4.1',
    apiKey: 'offline-test',
  });
  model._lc_stream_delay = 71;
  model.firstParty = true;
  const configured = model.withConfig({ stop: ['END'] });
  const bound = configured.bindTools([tool], { tool_choice: 'none' });

  expect(bound).toBeInstanceOf(InspectableChatOpenAI);
  if (!(bound instanceof InspectableChatOpenAI))
    throw new Error('Lost subclass');
  expect(bound).not.toBe(configured);
  expect(configured).not.toBe(model);
  expect(bound._lc_stream_delay).toBe(71);
  expect(bound.state.firstPartyEndpoint).toBe(true);
  expect(bound.state.completions).toBe(model.state.completions);
  expect(bound.state.responses).toBe(model.state.responses);
  expect(bound.state.options).toMatchObject({
    stop: ['END'],
    tool_choice: 'none',
    tools: [tool],
  });
  expect(configured.state.options).toEqual({ stop: ['END'] });
  expect(model.state.options).toBeUndefined();
});

test.each([ChatOpenAI, ChatMoonshot])(
  'preserves %p through repeated configuration and tool binding',
  (Ctor) => {
    const model = new Ctor({ model: 'offline-model', apiKey: 'offline-test' });
    const configured = model.withConfig({ stop: ['END'] });
    const bound = configured.bindTools([tool]);
    expect(configured).toBeInstanceOf(Ctor);
    expect(bound).toBeInstanceOf(Ctor);
    expect(configured.invocationParams()).toMatchObject({ stop: ['END'] });
  }
);
