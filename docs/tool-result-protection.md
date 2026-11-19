# Tool result protection

`RunConfig.toolResultProtection` lets the host inspect the text a tool returns
before the SDK uses it. It is off unless a policy is supplied, and it applies
only to the tools the policy names.

```typescript
import { Run } from '@librechat/agents';
import type { ToolResultProtection } from '@librechat/agents';

const toolResultProtection: ToolResultProtection = {
  version: 1,
  toolNames: ['crm_lookup'],
  timeoutMs: 5000,
  maxAttemptBytes: 1024 * 1024,
  maxBufferedBytes: 4 * 1024 * 1024,
  classify: (content) => (looksLikeProse(content) ? 'prose' : 'unsupported'),
  inspect: async ({ content, toolName, toolCallId, target, signal }) => ({
    version: 1,
    ok: true,
    value: { content: redact(content), replacements: 0, categories: [] },
  }),
};

const run = await Run.create({ ...config, toolResultProtection });
```

## What the SDK guarantees

For a selected tool, the text the tool returned is replaced by the text
`inspect` returned before anything else in the SDK reads it. Post-tool hooks,
the error handler, run-step completion events, the tool output reference
registry, checkpoints and every later model request receive the inspected text.
This holds for tools the SDK runs in process, tools the host runs through
`ON_TOOL_EXECUTE`, error text from either, and the same tools inside subagents.

If inspection does not succeed, the run ends with a
`ToolResultProtectionError` and the model is not called again. Its `code` is
the inspector's own failure code, or one of:

| Code           | Cause                                                                |
| -------------- | -------------------------------------------------------------------- |
| `unsupported`  | `classify` rejected the text, or the result was not plain text       |
| `unavailable`  | `inspect` threw or rejected; its message is never forwarded          |
| `incompatible` | `inspect` returned a malformed result                                |
| `timeout`      | `inspect` did not settle within `timeoutMs`                          |
| `overflow`     | The text exceeded `maxAttemptBytes` or the shared `maxBufferedBytes` |
| `cancelled`    | The run was aborted before inspection finished                       |

Byte limits count more than payload length: raw text is charged at four bytes
per character and inspected text at two, plus per-attempt overhead. Size them
well above the largest expected result.

## Trust boundary

The tool's text is untrusted data. Code running in the host process is trusted.
The SDK does not defend against tools, hooks, event handlers, callbacks or
checkpoint savers that misbehave, and it does not inspect text they author:

- A `PostToolUse` replacement, a hook denial reason, a human reviewer's
  response and host-injected messages are host-authored and pass through.
- Message history the host supplies to a run is not inspected. The host only
  ever received inspected text for selected tools, so stored history is already
  inspected; history that predates the policy is the host's to handle.
- Tracing callbacks attached to an in-process tool observe its raw return. Use
  the tool-output tracing configuration to redact those tools in traces.

## Limits

- A selected tool must return plain text. Structured content, an artifact or a
  `Command` ends the run as `unsupported`.
- Selected tools are not offered to programmatic tool calling, and are listed
  as disallowed for it. When the host runs programmatic tool calling itself,
  the host is responsible for the nested calls.
- Selected tools are never started eagerly and do not use the per-call early
  completion path; their completion is published once, after inspection.
- A selected tool cannot also be client delegated, and remote SDK runs reject a
  policy. Both fail when the run is created.
- Tools the policy does not name are untouched, whatever they return.
