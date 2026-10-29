#!/usr/bin/env tsx
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DetailLevel, parseNdjson, serializeEvent } from 'trace-schema';
import { generateTrace, type GenerateOptions, type Granularity, type TraceItem } from './generate';
import { DEFAULT_HUB_URL, HubClient, playItems } from './hub';

export function parseArgs(argv: string[]): { pos: string[]; flags: Record<string, string | true> } {
  const pos: string[] = [];
  const flags: Record<string, string | true> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const [k, inline] = a.slice(2).split('=', 2);
      if (inline !== undefined) flags[k] = inline;
      else if (i + 1 < argv.length && !argv[i + 1].startsWith('--')) flags[k] = argv[++i];
      else flags[k] = true;
    } else pos.push(a);
  }
  return { pos, flags };
}

const num = (v: string | true | undefined, d?: number) =>
  v === undefined || v === true ? d : Number(v);
const str = (v: string | true | undefined) => (typeof v === 'string' ? v : undefined);

export function generateOptionsFromFlags(f: Record<string, string | true>): GenerateOptions {
  return {
    req: str(f.req),
    model: str(f.model),
    layers: num(f.layers),
    ctx: num(f.ctx),
    prompt: str(f.prompt),
    promptTokens: num(f['prompt-tokens']),
    responseTokens: num(f['response-tokens']),
    tokPerSec: num(f['tok-per-sec']),
    loadMs: num(f['load-ms']),
    batchSize: num(f.batch),
    detail: f.detail ? DetailLevel.parse(f.detail) : undefined,
    seed: num(f.seed),
    heads: num(f.heads),
    layerStride: num(f['layer-stride']),
  };
}

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');

function stripUndefined<T extends object>(o: T): T {
  return Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined)) as T;
}

async function main() {
  const [cmd, ...rest] = process.argv.slice(2);
  const { pos, flags } = parseArgs(rest);
  const url = str(flags.url) ?? process.env.TRACE_URL ?? DEFAULT_HUB_URL;

  switch (cmd) {
    case 'replay': {
      const file = pos[0];
      if (!file) throw new Error('usage: replay <file.ndjson> [--speed 1.0] [--loop]');
      const { events, errors } = parseNdjson(
        readFileSync(resolve(process.env.INIT_CWD ?? process.cwd(), file), 'utf8'),
      );
      if (errors.length) console.error(`[trace-mock] ${errors.length} invalid lines skipped`);
      const hub = new HubClient(url);
      const speed = num(flags.speed, 1)!;
      do {
        // fresh req id per loop iteration so the app treats it as a new run
        const req = str(flags.req) ?? crypto.randomUUID();
        const items: TraceItem[] = events.map((e) => ({ kind: 'event', event: { ...e, req } }));
        console.error(`[trace-mock] replaying ${events.length} events as ${req} at ${speed}×`);
        await playItems(items, hub, { speed, onEvent: (e) => hub.send(e) });
        await hub.flush();
      } while (flags.loop);
      hub.close();
      break;
    }
    case 'generate': {
      const hub = new HubClient(url);
      const opts = stripUndefined(generateOptionsFromFlags(flags));
      const t0Ns = Number(process.hrtime.bigint());
      const tr = generateTrace({ ...opts, t0Ns, seed: opts.seed ?? Date.now() % 1e9 });
      const step = (str(flags.step) as Granularity | undefined) ?? null;
      console.error(
        `[trace-mock] generating ${tr.events.length} events as ${tr.req}` +
          (step ? ` (step mode: ${step})` : ''),
      );
      if (!hub.connected) await new Promise((r) => hub.once('open', r));
      await playItems(tr.items, hub, {
        speed: num(flags.speed, 1),
        step,
        shiftForPauses: true,
        onEvent: (e) => hub.send(e),
      });
      await hub.flush();
      hub.close();
      if (flags.print) process.stdout.write(tr.responseText + '\n');
      break;
    }
    case 'gen-fixtures': {
      const dir = resolve(ROOT, 'traces/fixtures');
      mkdirSync(dir, { recursive: true });
      const fixtures: Array<[string, GenerateOptions]> = [
        [
          'short-answer.ndjson',
          {
            req: 'fixture-short-answer',
            model: 'mock-llama:1b',
            layers: 8,
            responseTokens: 20,
            promptTokens: 48,
            batchSize: 16,
            prefillTokPerSec: 400,
            tokPerSec: 83,
            loadMs: 800,
            detail: 'layers',
            seed: 7,
          },
        ],
        [
          'full-detail.ndjson',
          {
            req: 'fixture-full-detail',
            model: 'mock-llama:1b',
            layers: 8,
            responseTokens: 6,
            prompt: 'Why is the sky blue?',
            batchSize: 16,
            tokPerSec: 40,
            loadMs: 0,
            detail: 'full',
            seed: 11,
          },
        ],
      ];
      for (const [name, o] of fixtures) {
        const tr = generateTrace(o);
        writeFileSync(resolve(dir, name), tr.events.map(serializeEvent).join('\n') + '\n');
        writeFileSync(resolve(dir, name.replace('.ndjson', '.response.txt')), tr.responseText);
        console.error(
          `wrote ${name}: ${tr.events.length} events, ${tr.promptTokens.length} prompt tokens`,
        );
      }
      break;
    }
    default:
      console.error(`usage:
  trace-mock replay <file.ndjson> [--speed 1.0] [--loop] [--url ws://127.0.0.1:4711/ingest]
  trace-mock generate [--model llama3] [--layers 32] [--prompt-tokens 120] [--response-tokens 200]
                      [--tok-per-sec 60] [--detail basic|layers|full] [--step token|layer|sample] [--req id]
  trace-mock gen-fixtures`);
      process.exit(cmd ? 1 : 0);
  }
}

const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
