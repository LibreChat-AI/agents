import {
  createRemoteSdkRun,
  createRemoteSdkServer,
} from '@librechat/agents/runtime';
import type {
  RemoteSdkFactoryInput,
  RemoteSdkServerOptions,
  SdkAgentIdentity,
} from '@librechat/agents/runtime';
import { Run, Providers } from '@librechat/agents';

const createRun = (
  input: RemoteSdkFactoryInput
): ReturnType<typeof Run.create> =>
  Run.create({
    ...input.config,
    graphConfig: {
      type: 'standard',
      agents: [
        {
          agentId: input.identity.agentId,
          provider: Providers.OPENAI,
          clientOptions: { model: 'fixture' },
        },
      ],
      compileOptions: input.compileOptions,
    },
  });
const authorize = (identity: Readonly<SdkAgentIdentity>): boolean =>
  identity.agentId === 'agent';
const options: RemoteSdkServerOptions = {
  runtimeId: 'example',
  secret: 'typecheck-only',
  authorize,
  createRun,
};
const server: ReturnType<typeof createRemoteSdkServer> =
  createRemoteSdkServer(options);
void server.close;
void createRemoteSdkRun;

// @ts-expect-error Published identity types must reject non-string users.
const invalidUser: RemoteSdkFactoryInput['identity']['userId'] = 123;
// @ts-expect-error Published model services must retain their invocation contract.
const invalidModel: NonNullable<RemoteSdkFactoryInput['hostModel']> = {};
void invalidUser;
void invalidModel;
