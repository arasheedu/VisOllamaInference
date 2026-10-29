import { TRACE_HEADERS, type DetailLevel } from 'trace-schema';
import type { TokenizerSpec } from 'tokenizer';

export interface ChatChunk {
  message?: { role: string; content: string; thinking?: string };
  done: boolean;
  error?: string;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

export class OllamaError extends Error {
  constructor(
    message: string,
    readonly kind: 'unreachable' | 'not_found' | 'http',
  ) {
    super(message);
  }
}

export async function listModels(base: string): Promise<string[]> {
  let res: Response;
  try {
    res = await fetch(`${base}/api/tags`);
  } catch {
    throw new OllamaError(`Ollama is not reachable at ${base}. Is it running?`, 'unreachable');
  }
  if (!res.ok) throw new OllamaError(`GET /api/tags failed: ${res.status}`, 'http');
  const j = (await res.json()) as { models?: Array<{ name: string }> };
  return (j.models ?? []).map((m) => m.name).sort();
}

/** Fetch the model's BPE vocab/merges from /api/show (verbose). */
export async function fetchTokenizer(base: string, model: string): Promise<TokenizerSpec | null> {
  const res = await fetch(`${base}/api/show`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, verbose: true }),
  });
  if (!res.ok) return null;
  const j = (await res.json()) as { model_info?: Record<string, unknown> };
  const info = j.model_info ?? {};
  if (info['tokenizer.ggml.model'] !== 'gpt2') return null; // only byte-level BPE supported
  const tokens = info['tokenizer.ggml.tokens'] as string[] | undefined;
  const merges = info['tokenizer.ggml.merges'] as string[] | undefined;
  if (!tokens?.length || !merges?.length) return null;
  return {
    tokens,
    merges,
    tokenTypes: info['tokenizer.ggml.token_type'] as number[] | undefined,
    pre: info['tokenizer.ggml.pre'] as string | undefined,
  };
}

export interface ChatRequest {
  base: string;
  model: string;
  prompt: string;
  req: string;
  detail: DetailLevel;
  step: string | null;
  options?: Record<string, unknown>;
  signal: AbortSignal;
}

/**
 * Stream /api/chat. Sends trace headers; if the browser rejects them (CORS on
 * stock Ollama), retries without them and reports `instrumentedHeaders=false`.
 */
export async function streamChat(
  r: ChatRequest,
  onChunk: (c: ChatChunk) => void,
): Promise<{ instrumentedHeaders: boolean }> {
  const body = JSON.stringify({
    model: r.model,
    messages: [{ role: 'user', content: r.prompt }],
    stream: true,
    options: r.options ?? {},
  });
  const traceHeaders: Record<string, string> = {
    [TRACE_HEADERS.requestId]: r.req,
    [TRACE_HEADERS.detail]: r.detail,
  };
  if (r.step) traceHeaders[TRACE_HEADERS.step] = r.step;

  let res: Response;
  let instrumentedHeaders = true;
  try {
    res = await fetch(`${r.base}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...traceHeaders },
      body,
      signal: r.signal,
    });
  } catch (e) {
    if (r.signal.aborted) throw e;
    // Either unreachable or CORS rejected our custom headers: retry plain.
    instrumentedHeaders = false;
    try {
      res = await fetch(`${r.base}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        signal: r.signal,
      });
    } catch {
      throw new OllamaError(`Ollama is not reachable at ${r.base}. Is it running?`, 'unreachable');
    }
  }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      msg = ((await res.json()) as { error?: string }).error ?? msg;
    } catch {
      /* keep status text */
    }
    throw new OllamaError(
      res.status === 404 ? `Model "${r.model}" not found. Try: ollama pull ${r.model}` : msg,
      res.status === 404 ? 'not_found' : 'http',
    );
  }
  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      const c = JSON.parse(line) as ChatChunk;
      if (c.error) throw new OllamaError(c.error, 'http');
      onChunk(c);
    }
  }
  if (buf.trim()) onChunk(JSON.parse(buf) as ChatChunk);
  return { instrumentedHeaders };
}
