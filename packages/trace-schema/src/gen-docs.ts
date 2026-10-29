// Generates docs/events.md from the zod schema. Run: pnpm --filter trace-schema docs
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { z } from 'zod';
import { EVENT_DETAIL, PRIORITY, TraceEvent, AckMessage, SCHEMA_VERSION } from './index';

type JsonSchema = {
  type?: string | string[];
  properties?: Record<string, JsonSchema>;
  required?: string[];
  items?: JsonSchema | JsonSchema[];
  prefixItems?: JsonSchema[];
  enum?: unknown[];
  const?: unknown;
  anyOf?: JsonSchema[];
  additionalProperties?: JsonSchema | boolean;
};

function typeOf(s: JsonSchema): string {
  if (s.const !== undefined) return JSON.stringify(s.const);
  if (s.enum) return s.enum.map((e) => JSON.stringify(e)).join(' \\| ');
  if (s.anyOf) return s.anyOf.map(typeOf).join(' \\| ');
  if (s.prefixItems) return `[${s.prefixItems.map(typeOf).join(', ')}]`;
  if (s.type === 'array' && s.items && !Array.isArray(s.items)) return `Array<${typeOf(s.items)}>`;
  if (s.type === 'object' && s.properties) {
    const req = new Set(s.required ?? []);
    return `{ ${Object.entries(s.properties)
      .map(([k, v]) => `${k}${req.has(k) ? '' : '?'}: ${typeOf(v)}`)
      .join(', ')} }`;
  }
  if (s.type === 'object') return 'object';
  if (s.type === 'integer') return 'int';
  return Array.isArray(s.type) ? s.type.join(' \\| ') : (s.type ?? 'unknown');
}

const BASE = new Set(['v', 'req', 'seq', 't_ns', 'type']);

let md = `# Trace event reference (schema v${SCHEMA_VERSION})

_Generated from \`packages/trace-schema/src/index.ts\` by \`pnpm --filter trace-schema docs\`. Do not edit by hand._

Every event is one NDJSON line carrying the common envelope:

| field | type | meaning |
|---|---|---|
| \`v\` | \`1\` | schema version |
| \`req\` | string | request id (the app's \`X-Trace-Request-Id\`) |
| \`seq\` | int | strictly increasing per \`req\` |
| \`t_ns\` | number | monotonic emitter clock, nanoseconds |
| \`type\` | string | one of the types below |

`;

for (const opt of TraceEvent.options) {
  const type = opt.shape.type.value;
  const js = z.toJSONSchema(opt, { io: 'input' }) as JsonSchema;
  const req = new Set(js.required ?? []);
  md += `## \`${type}\`\n\npriority **${PRIORITY[type]}** · detail ≥ **${EVENT_DETAIL[type]}**\n\n`;
  const fields = Object.entries(js.properties ?? {}).filter(([k]) => !BASE.has(k));
  if (fields.length) {
    md += '| field | type | required |\n|---|---|---|\n';
    for (const [k, v] of fields) md += `| \`${k}\` | \`${typeOf(v)}\` | ${req.has(k) ? 'yes' : 'no'} |\n`;
  } else md += '_No payload._\n';
  md += '\n';
}

const ack = z.toJSONSchema(AckMessage) as JsonSchema;
md += `## Control: \`ack\` (app → producer)\n\n\`${typeOf(ack)}\`\n\nSent over the same socket in step mode; \`continue: true\` releases the request from step mode.\n`;

const out = resolve(dirname(fileURLToPath(import.meta.url)), '../../../docs/events.md');
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, md);
console.error(`wrote ${out}`);
