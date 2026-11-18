const { createRemoteSdkServer } = require('@librechat/agents/runtime');
const { createExampleRun } = require('./agent.cjs');
const runtime = createRemoteSdkServer({
  runtimeId: 'package-test',
  secret: process.env.RUNTIME_TEST_SECRET,
  authorize: (identity) => identity.agentId === 'agent',
  createRun: (input) =>
    createExampleRun(input, {
      stepDelayMs: input.agent.instructions === 'delay' ? 10000 : 0,
    }),
});
runtime.server.listen(0, '127.0.0.1', () => {
  process.send({ port: runtime.server.address().port });
});
process.once('SIGTERM', () => void runtime.close().then(() => process.exit(0)));
