#!/usr/bin/env tsx
/**
 * mock-ollama-http: just enough of Ollama's public API (/api/chat streaming,
 * /api/tags, /api/show, /api/version) to drive the app's real-world path.
 * Each chat request also emits a correlated trace to the app's hub.
 */
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { randomUUID } from 'node:crypto';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DetailLevel, TRACE_HEADERS } from 'trace-schema';
import { parseArgs } from './cli';
import { generateTrace, type Granularity } from './generate';
import { DEFAULT_HUB_URL, HubClient, playItems } from './hub';
import { MOCK_MODELS, mockTokenizer } from './model';

export interface MockServerOptions {
  port?: number;
  hubUrl?: string;
  tokPerSec?: number;
  speed?: number;
  log?: (m: string) => void;
  /** called once per chat request with the exact generated response text */
  onResponse?: (req: string, text: string) => void;
}

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': `Content-Type, Authorization, ${Object.values(TRACE_HEADERS).join(', ')}`,
};

function json(res: ServerResponse, code: number, body: unknown) {
  res.writeHead(code, { 'Content-Type': 'application/json', ...CORS });
  res.end(JSON.stringify(body));
}

async function readBody(req: IncomingMessage): Promise<any> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  const s = Buffer.concat(chunks).toString('utf8');
  return s ? JSON.parse(s) : {};
}

export function startMockOllama(o: MockServerOptions = {}) {
  const log = o.log ?? ((m: string) => console.error(m));
  const hub = new HubClient(o.hubUrl ?? DEFAULT_HUB_URL, log);
  const loaded = new Set<string>();

  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? '/', 'http://x');
      if (req.method === 'OPTIONS') {
        res.writeHead(204, CORS);
        return res.end();
      }
      if (url.pathname === '/' || url.pathname === '/api/version') {
        return json(res, 200, { version: '0.0.0-mock' });
      }
      if (url.pathname === '/api/tags') {
        return json(res, 200, {
          models: Object.entries(MOCK_MODELS).map(([name, m]) => ({
            name,
            model: name,
            size: m.size,
            modified_at: new Date(0).toISOString(),
            details: {
              family: 'mock-llama',
              parameter_size: m.params,
              quantization_level: 'Q4_K_M',
            },
          })),
        });
      }
      if (url.pathname === '/api/show' && req.method === 'POST') {
        const body = await readBody(req);
        const m = MOCK_MODELS[body.model ?? body.name];
        if (!m) return json(res, 404, { error: `model '${body.model}' not found` });
        const { spec } = mockTokenizer();
        const info: Record<string, unknown> = {
          'general.architecture': 'llama',
          'llama.block_count': m.layers,
          'llama.context_length': 4096,
          'tokenizer.ggml.model': 'gpt2',
          'tokenizer.ggml.pre': spec.pre,
        };
        if (body.verbose) {
          info['tokenizer.ggml.tokens'] = spec.tokens;
          info['tokenizer.ggml.merges'] = spec.merges;
          info['tokenizer.ggml.token_type'] = spec.tokenTypes;
        }
        return json(res, 200, { model_info: info, details: { parameter_size: m.params } });
      }
      if (url.pathname === '/api/chat' && req.method === 'POST') {
        const body = await readBody(req);
        const name: string = body.model;
        const m = MOCK_MODELS[name];
        if (!m) return json(res, 404, { error: `model '${name}' not found, try pulling it first` });
        const messages: Array<{ role: string; content: string }> = body.messages ?? [];
        const prompt = [...messages].reverse().find((x) => x.role === 'user')?.content ?? '';
        const reqId =
          (req.headers[TRACE_HEADERS.requestId.toLowerCase()] as string) || randomUUID();
        const detailH = req.headers[TRACE_HEADERS.detail.toLowerCase()] as string | undefined;
        const detail = DetailLevel.safeParse(detailH).data ?? 'basic';
        const step =
          (req.headers[TRACE_HEADERS.step.toLowerCase()] as Granularity | undefined) || null;
        const opts = body.options ?? {};
        const tr = generateTrace({
          req: reqId,
          model: name,
          layers: m.layers,
          prompt,
          responseTokens: Math.min(
            Number(opts.num_predict) > 0 ? Number(opts.num_predict) : 160,
            2000,
          ),
          tokPerSec: o.tokPerSec ?? m.tokPerSec,
          loadMs: loaded.has(name) ? 0 : 800,
          detail,
          temperature: opts.temperature ?? 0.8,
          topK: opts.top_k ?? 40,
          topP: opts.top_p ?? 0.9,
          minP: opts.min_p ?? 0,
          seed: opts.seed ?? Math.floor(Math.random() * 1e9),
          t0Ns: Number(process.hrtime.bigint()),
        });
        loaded.add(name);
        o.onResponse?.(reqId, tr.responseText);
        log(
          `[mock-ollama] /api/chat ${name} req=${reqId} detail=${detail}${step ? ` step=${step}` : ''}`,
        );
        const stream = body.stream !== false;
        const ac = new AbortController();
        res.on('close', () => ac.abort());
        if (stream) res.writeHead(200, { 'Content-Type': 'application/x-ndjson', ...CORS });
        let content = '';
        await playItems(tr.items, hub, {
          speed: o.speed ?? 1,
          step,
          shiftForPauses: true,
          signal: ac.signal,
          onEvent: (e) => {
            hub.send(e);
            if (e.type === 'decode.token' && e.text) {
              content += e.text;
              if (stream)
                res.write(
                  JSON.stringify({
                    model: name,
                    created_at: new Date().toISOString(),
                    message: { role: 'assistant', content: e.text },
                    done: false,
                  }) + '\n',
                );
            }
            if (e.type === 'request.done') {
              const s = e.stats;
              const fin = {
                model: name,
                created_at: new Date().toISOString(),
                message: { role: 'assistant', content: stream ? '' : content },
                done: true,
                done_reason: 'stop',
                total_duration: Math.round(s.total_ms * 1e6),
                load_duration: Math.round(s.load_ms * 1e6),
                prompt_eval_count: s.prompt_eval_count,
                prompt_eval_duration: Math.round(s.prompt_eval_ms * 1e6),
                eval_count: s.eval_count,
                eval_duration: Math.round(s.eval_ms * 1e6),
              };
              if (stream) res.end(JSON.stringify(fin) + '\n');
              else json(res, 200, fin);
            }
          },
        });
        return;
      }
      json(res, 404, { error: 'not found' });
    } catch (e) {
      log(`[mock-ollama] error: ${(e as Error).message}`);
      if (!res.headersSent) json(res, 500, { error: (e as Error).message });
      else res.end();
    }
  });
  const port = o.port ?? 11434;
  server.listen(port, '127.0.0.1', () =>
    log(`[mock-ollama] listening on http://127.0.0.1:${port}`),
  );
  return {
    server,
    hub,
    close: () =>
      new Promise<void>((r) => {
        hub.close();
        server.closeAllConnections?.();
        server.close(() => r());
      }),
  };
}

const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const { flags } = parseArgs(process.argv.slice(2));
  startMockOllama({
    port: flags.port ? Number(flags.port) : undefined,
    hubUrl: typeof flags.url === 'string' ? flags.url : process.env.TRACE_URL,
    tokPerSec: flags['tok-per-sec'] ? Number(flags['tok-per-sec']) : undefined,
    speed: flags.speed ? Number(flags.speed) : undefined,
  });
}
