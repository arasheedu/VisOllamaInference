import {
  EVENT_TYPES,
  HUB_PATHS,
  includedAtDetail,
  parseNdjson,
  serializeEvent,
  validateEventObject,
  type DetailLevel,
  type EventType,
  type TraceEvent,
  atLeast,
} from 'trace-schema';
import { generateTrace, MOCK_MODELS, mockTokenizer, type TraceItem } from 'trace-mock/browser';
import { Player, type StepKind } from './engine/player';
import type { ViewModel } from './engine/types';
import { fetchTokenizer, listModels, OllamaError, streamChat, type ChatChunk } from './api/ollama';
import { clockConfigFrom, mergeSettings, type AppSettings } from './settings';
import { addWarning, clearWarning, setUi, useUi, type HubStatus } from './store';
import { loadTokenizer, planMerges, tokenizerFor } from './tokenizer/merges';

type FrameFn = (now: number, vm: ViewModel, player: Player | null) => void;

const KNOWN = new Set<string>(EVENT_TYPES);

/** Tokenizer cache key: mock models use the bundled vocab. */
const tokKey = (model: string) => (model.startsWith('mock-') ? 'mock:' : 'ollama:') + model;

interface RunState {
  req: string;
  source: 'ollama' | 'mock';
  model: string;
  abort: AbortController;
  chatText: string;
  chatDone: boolean;
  traceDone: boolean;
  synthetic: boolean;
  step: boolean;
  startWall: number;
  /** in-app mock gate release */
  release: (() => void) | null;
  released: boolean;
  acksSent: number;
  eventsSinceAck: number;
  lastAckWall: number;
  validated: boolean;
  saved: boolean;
}

/**
 * Owns trace ingestion, the single requestAnimationFrame loop, and the run
 * lifecycle. React reads the view model from the store (at most one update per
 * frame); canvases subscribe via `onFrame` and draw from the same tick.
 */
export class Session {
  players = new Map<string, Player>();
  active: Player | null = null;
  run: RunState | null = null;
  private ws: WebSocket | null = null;
  private wsBackoff = 250;
  private port = 4711;
  private frameFns = new Set<FrameFn>();
  private raf = 0;
  private lastVm: ViewModel | null = null;
  private fpsFrames = 0;
  private fpsT = 0;
  settings: AppSettings = useUi.getState().settings;

  async init() {
    const api = window.visualizer;
    if (api) {
      this.settings = mergeSettings(await api.getSettings());
      setUi({ settings: this.settings });
      const st = await api.getStatus();
      this.onHubStatus(st);
      api.onStatus((s) => this.onHubStatus(s));
    }
    this.connect();
    void this.refreshModels();
    this.raf = requestAnimationFrame(this.frame);
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.ws?.close();
  }

  // ---- settings ---------------------------------------------------------

  async updateSettings(patch: Partial<AppSettings>) {
    this.settings = { ...this.settings, ...patch };
    setUi({ settings: this.settings });
    await window.visualizer?.setSettings(this.settings as unknown as Record<string, unknown>);
    if (patch.source || patch.ollamaUrl) void this.refreshModels();
    const cfg = clockConfigFrom(this.settings);
    for (const p of this.players.values()) p.clock.config = cfg;
  }

  async refreshModels() {
    const s = this.settings;
    clearWarning('ollama');
    if (s.source === 'mock') {
      const models = Object.keys(MOCK_MODELS);
      setUi((u) => ({ models, model: models.includes(u.model) ? u.model : 'mock-llama:8b' }));
      return;
    }
    try {
      const models = await listModels(s.ollamaUrl);
      setUi((u) => ({ models, model: models.includes(u.model) ? u.model : (models[0] ?? '') }));
      if (!models.length)
        addWarning({
          id: 'ollama',
          level: 'warn',
          text: 'No models installed. Try: ollama pull llama3.2',
        });
    } catch (e) {
      setUi({ models: [] });
      addWarning({ id: 'ollama', level: 'error', text: (e as Error).message });
    }
  }

  // ---- trace socket -----------------------------------------------------

  private onHubStatus(s: HubStatus) {
    setUi({ hub: s });
    if (s.port !== this.port) {
      this.port = s.port;
      this.ws?.close();
    }
  }

  private connect() {
    setUi({ socket: 'connecting' });
    const ws = new WebSocket(`ws://127.0.0.1:${this.port}${HUB_PATHS.view}`);
    this.ws = ws;
    ws.onopen = () => {
      this.wsBackoff = 250;
      setUi({ socket: 'open' });
    };
    ws.onmessage = (m) => this.ingestText(String(m.data));
    ws.onclose = () => {
      setUi({ socket: 'closed' });
      this.ws = null;
      setTimeout(() => this.connect(), this.wsBackoff);
      this.wsBackoff = Math.min(this.wsBackoff * 2, 4000);
    };
  }

  private parseErrors = 0;

  ingestText(text: string) {
    const now = performance.now();
    for (const line of text.split('\n')) {
      if (!line) continue;
      let obj: { type?: string; req?: string };
      try {
        obj = JSON.parse(line);
      } catch {
        this.parseErrors++;
        continue;
      }
      // cheap pre-filter by detail before full validation
      const p = obj.req ? this.players.get(obj.req) : undefined;
      if (p && obj.type && KNOWN.has(obj.type) && !p.wants(obj.type as EventType)) {
        p.filtered++;
        continue;
      }
      const r = validateEventObject(obj, line);
      if (!r.ok) {
        this.parseErrors++;
        continue;
      }
      this.ingest(r.event, now);
    }
  }

  ingest(e: TraceEvent, now = performance.now()) {
    let p = this.players.get(e.req);
    if (!p) {
      if (this.run && this.run.req !== e.req && !this.run.traceDone) {
        // another request while ours runs: keep it but don't switch view
        p = this.newPlayer(e.req, this.settings.detail, true);
      } else {
        p = this.newPlayer(e.req, this.settings.detail, true);
        if (!this.run || this.run.traceDone) this.activate(p, 'external');
      }
    }
    if (this.run && this.run.req === e.req) {
      if (this.run.synthetic) return; // already fell back to token-stream-only
      this.run.eventsSinceAck++;
      if (e.type === 'request.done' || e.type === 'request.error') this.run.traceDone = true;
    }
    if (e.type === 'request.received' && atLeast(p.detail, 'full')) this.loadTok(e.model);
    if (e.type === 'request.received' && this.run?.req === e.req && e.detail) {
      if (e.detail !== p.detail)
        addWarning({
          id: 'detail',
          level: 'warn',
          text:
            p.detail === 'deep'
              ? `This Ollama traced at "${e.detail}" detail: Deep needs the deep-tier build (ollama-trace-deep).`
              : `This Ollama traced at "${e.detail}" detail (requested "${p.detail}").`,
        });
      else clearWarning('detail');
    }
    p.push(e, now);
  }

  private loadTok(model: string) {
    void loadTokenizer(tokKey(model), async () =>
      model.startsWith('mock-')
        ? mockTokenizer().spec
        : fetchTokenizer(this.settings.ollamaUrl, model),
    );
  }

  private newPlayer(req: string, detail: DetailLevel, live: boolean): Player {
    const p = new Player(req, { detail, live, clock: clockConfigFrom(this.settings) });
    p.clock.extraHold = (e) => this.extraHold(p, e);
    this.players.set(req, p);
    if (this.players.size > 8) {
      const oldest = [...this.players.keys()].find((k) => this.players.get(k) !== this.active);
      if (oldest) this.players.delete(oldest);
    }
    return p;
  }

  private activate(
    p: Player,
    viewing: 'live' | 'file' | 'external',
    fileName: string | null = null,
  ) {
    this.active = p;
    this.lastVm = null;
    setUi({ viewing, fileName, merge: null, inspector: null });
  }

  /** Full detail: hold tokenize.done long enough for the merge animation. */
  private extraHold(p: Player, e: TraceEvent): number {
    if (e.type !== 'tokenize.done' || !atLeast(p.detail, 'full') || p !== this.active) return 0;
    const model = p.machine.current.model ?? '';
    const plan = planMerges(tokenizerFor(tokKey(model)), e.tokens);
    setUi({ merge: plan });
    if (!plan.ok) {
      addWarning({ id: 'merge', level: 'warn', text: `Merge animation skipped: ${plan.reason}` });
      return 0;
    }
    clearWarning('merge');
    return Math.min(12_000, 400 + plan.rounds * this.settings.minimums['tokenize.merge'] * 2);
  }

  // ---- frame loop -------------------------------------------------------

  onFrame(fn: FrameFn): () => void {
    this.frameFns.add(fn);
    return () => this.frameFns.delete(fn);
  }

  private frame = (now: number) => {
    this.raf = requestAnimationFrame(this.frame);
    const p = this.active;
    let vm = this.lastVm;
    if (p) {
      vm = p.tick(now);
      this.pumpAcks(now);
      this.checkRunComplete();
      if (vm !== this.lastVm) {
        this.lastVm = vm;
        setUi({ vm });
      }
    }
    for (const fn of this.frameFns) fn(now, vm ?? useUi.getState().vm, p);
    this.fpsFrames++;
    if (now - this.fpsT > 500) {
      setUi({ fps: Math.round((this.fpsFrames * 1000) / (now - this.fpsT)) });
      this.fpsFrames = 0;
      this.fpsT = now;
    }
  };

  // ---- run --------------------------------------------------------------

  async start(prompt: string, model: string) {
    this.stop();
    const s = this.settings;
    const req = crypto.randomUUID();
    const detail = s.detail;
    const step = useUi.getState().stepMode;
    const run: RunState = {
      req,
      source: s.source,
      model,
      abort: new AbortController(),
      chatText: '',
      chatDone: false,
      traceDone: false,
      synthetic: false,
      step,
      startWall: performance.now(),
      release: null,
      released: false,
      acksSent: 0,
      eventsSinceAck: 0,
      lastAckWall: 0,
      validated: false,
      saved: false,
    };
    this.run = run;
    const p = this.newPlayer(req, detail, true);
    this.activate(p, 'live');
    for (const id of ['mismatch', 'uninstrumented', 'run', 'merge']) clearWarning(id);
    setUi({ running: true, uninstrumented: false });

    // prefetch tokenizer for the merge animation
    if (atLeast(detail, 'full')) {
      this.loadTok(model);
    }
    try {
      if (s.source === 'mock') await this.runMock(run, prompt, model, detail);
      else await this.runOllama(run, prompt, model, detail);
    } catch (e) {
      if (run.abort.signal.aborted) return;
      const msg = (e as Error).message;
      addWarning({ id: 'run', level: 'error', text: msg });
      if (!run.traceDone) {
        p.push(
          {
            v: 1,
            req,
            seq: 1e9,
            t_ns: performance.now() * 1e6,
            type: 'request.error',
            message: msg,
          },
          performance.now(),
        );
      }
      run.chatDone = true;
      run.traceDone = true;
    }
  }

  stop() {
    if (this.run && !this.run.chatDone) this.run.abort.abort();
    this.run?.release?.();
    setUi({ running: false });
  }

  /** In-process mock: same ingestion path as the socket, no external process. */
  private async runMock(run: RunState, prompt: string, model: string, detail: DetailLevel) {
    const m = MOCK_MODELS[model] ?? MOCK_MODELS['mock-llama:8b'];
    const tr = generateTrace({
      req: run.req,
      model,
      layers: m.layers,
      gpuLayers: m.gpuLayers,
      experts: m.experts,
      prompt,
      responseTokens: 120,
      tokPerSec: this.settings.mockTokPerSec || m.tokPerSec,
      loadMs: this.loadedMock.has(model) ? 0 : 800,
      detail,
      heads: this.settings.attnHeads,
      seed: Math.floor(Math.random() * 1e9),
      t0Ns: performance.now() * 1e6,
    });
    this.loadedMock.add(model);
    await this.playLocal(run, tr.items);
    run.chatText = tr.responseText;
    run.chatDone = true;
  }
  private loadedMock = new Set<string>();

  private async playLocal(run: RunState, items: TraceItem[]) {
    const first = items.find((i) => i.kind === 'event');
    if (!first || first.kind !== 'event') return;
    const t0 = first.event.t_ns;
    const start = performance.now();
    let paused = 0;
    for (const it of items) {
      if (run.abort.signal.aborted) return;
      if (it.kind === 'gate') {
        if (!run.step || run.released || it.granularity !== this.settings.stepGranularity) continue;
        const before = performance.now();
        await new Promise<void>((resolve) => (run.release = resolve));
        run.release = null;
        paused += performance.now() - before;
        continue;
      }
      const target = start + (it.event.t_ns - t0) / 1e6 + paused;
      const d = target - performance.now();
      if (d > 4) await new Promise((r) => setTimeout(r, d));
      this.ingest({ ...it.event, t_ns: Math.round(it.event.t_ns + paused * 1e6) });
    }
  }

  private async runOllama(run: RunState, prompt: string, model: string, detail: DetailLevel) {
    const base = this.settings.ollamaUrl;
    let decided = false;
    let step = 0;
    const synth = (type: TraceEvent['type'], payload: Record<string, unknown>) => {
      const p = this.players.get(run.req)!;
      const seq = p.events.length ? p.events[p.events.length - 1].seq + 1 : 0;
      p.push({
        v: 1,
        req: run.req,
        seq,
        t_ns: Math.round(performance.now() * 1e6),
        type,
        ...payload,
      } as TraceEvent);
    };
    const onChunk = (c: ChatChunk) => {
      const p = this.players.get(run.req)!;
      if (!decided) {
        decided = true;
        if (p.events.length === 0) {
          // No trace events for our request: stock Ollama. Token stream only.
          run.synthetic = true;
          setUi({ uninstrumented: true });
          addWarning({
            id: 'uninstrumented',
            level: 'info',
            text: 'Running without instrumentation: token stream only (stock Ollama or trace socket not connected).',
          });
          synth('request.received', { model, prompt_chars: prompt.length, options: {} });
          synth('model.load', { phase: 'done', layers: 0, ctx: 0, vram_mb: 0 });
        }
      }
      // decode.token carries post-parser text: thinking and content, in arrival order
      const text = (c.message?.thinking ?? '') + (c.message?.content ?? '');
      run.chatText += text;
      if (run.synthetic) {
        if (text) {
          synth('decode.step.start', { step, kv_used: 0, kv_total: 0 });
          synth('decode.token', { step, text, is_eos: false });
          step++;
        }
        if (c.done) {
          const ms = (ns?: number) => (ns ?? 0) / 1e6;
          synth('request.done', {
            stats: {
              prompt_eval_count: c.prompt_eval_count ?? 0,
              prompt_eval_ms: ms(c.prompt_eval_duration),
              eval_count: c.eval_count ?? step,
              eval_ms: ms(c.eval_duration),
              load_ms: ms(c.load_duration),
              total_ms: ms(c.total_duration),
            },
          });
          run.traceDone = true;
        }
      }
    };
    const r = await streamChat(
      {
        base,
        model,
        prompt,
        req: run.req,
        detail,
        step: run.step ? this.settings.stepGranularity : null,
        signal: run.abort.signal,
      },
      onChunk,
    );
    if (!r.instrumentedHeaders && !run.synthetic) run.synthetic = true;
    run.chatDone = true;
  }

  /** Response from the public API must equal the concatenated decode.token texts. */
  private checkRunComplete() {
    const run = this.run;
    const p = this.active;
    if (!run || !p || p.req !== run.req || run.validated) return;
    if (!run.chatDone || !run.traceDone) return;
    if (!p.atEnd) return;
    run.validated = true;
    setUi({ running: false });
    const traceText = p.events
      .filter((e): e is Extract<TraceEvent, { type: 'decode.token' }> => e.type === 'decode.token')
      .map((e) => e.text)
      .join('');
    if (!run.synthetic && run.chatText && traceText !== run.chatText) {
      addWarning({
        id: 'mismatch',
        level: 'warn',
        text: `Trace tokens differ from /api/chat response (${traceText.length} vs ${run.chatText.length} chars)`,
      });
    }
    if (this.settings.autosave && !run.saved) {
      run.saved = true;
      const name = `${new Date().toISOString().replace(/[:.]/g, '-')}-${run.model.replace(/[:/]/g, '_')}.ndjson`;
      void window.visualizer?.autosaveTrace(name, this.serialize(p), run.chatText);
    }
  }

  // ---- transport --------------------------------------------------------

  pause() {
    this.active?.clock.pause();
  }
  resume() {
    this.active?.clock.resume();
  }
  setSpeed(x: number) {
    this.active?.clock.setSpeed(x);
  }
  seek(t: number) {
    this.active?.seek(t);
  }
  goLive() {
    this.active?.goLive();
  }
  restart() {
    const p = this.active;
    if (!p) return;
    p.seek(p.t0 - 1);
    p.clock.resume();
  }

  /** Step: play until the next event of `kind`; in live step mode, acks feed the runner. */
  step(kind: StepKind) {
    const p = this.active;
    if (!p) return;
    p.stepUntil(kind);
  }

  continue() {
    const run = this.run;
    if (run && run.step) {
      run.released = true;
      run.release?.();
      this.sendAck(run, true);
      setUi({ stepMode: false });
    }
    const c = this.active?.clock;
    if (c) {
      // drop any pending Step so playback doesn't pause again at the next match
      c.stopWhen = null;
      c.stoppingLabel = null;
      c.resume();
    }
  }

  private sendAck(run: RunState, cont = false) {
    run.acksSent++;
    run.eventsSinceAck = 0;
    run.lastAckWall = performance.now();
    if (run.source === 'mock') {
      run.release?.();
      return;
    }
    this.ws?.send(
      JSON.stringify({ v: 1, type: 'ack', req: run.req, ...(cont ? { continue: true } : {}) }),
    );
  }

  /**
   * Live step mode: when the player wants more events than are buffered, ack
   * the runner — once per gate (only after new events arrived since the last).
   */
  private pumpAcks(now: number) {
    const run = this.run;
    const p = this.active;
    if (!run || !run.step || run.released || !p || p.req !== run.req || run.traceDone) return;
    if (!p.clock.stopWhen || !p.atEnd) return;
    if (run.acksSent > 0 && run.eventsSinceAck === 0) {
      // runner may not have reached the first gate yet; re-ack after a while
      if (now - run.lastAckWall < 1500) return;
    }
    if (now - p.headWall() < 30 && run.acksSent > 0) return;
    this.sendAck(run);
  }

  // ---- files ------------------------------------------------------------

  serialize(p: Player | null = this.active): string {
    return p ? p.events.map(serializeEvent).join('\n') + '\n' : '';
  }

  async saveTrace() {
    const p = this.active;
    if (!p) return;
    const name = `${p.machine.current.model ?? 'trace'}-${p.req.slice(0, 8)}.ndjson`.replace(
      /[:/]/g,
      '_',
    );
    await window.visualizer?.saveTrace(name, this.serialize(p), p.machine.current.responseText);
  }

  async openTrace(path?: string) {
    const f = await window.visualizer?.openTrace(path);
    if (!f) return;
    this.loadTraceText(f.text, f.path.split(/[\\/]/).pop() ?? f.path);
  }

  loadTraceText(text: string, name: string) {
    this.stop();
    const { events, errors } = parseNdjson(text);
    if (!events.length) {
      addWarning({ id: 'file', level: 'error', text: `No valid events in ${name}` });
      return;
    }
    if (errors.length)
      addWarning({
        id: 'file',
        level: 'warn',
        text: `${errors.length} invalid lines skipped in ${name}`,
      });
    else clearWarning('file');
    const req = events[0].req;
    const hasDeep = events.some((e) => !includedAtDetail(e.type, 'full'));
    const hasFull = events.some((e) => !includedAtDetail(e.type, 'layers'));
    const hasLayers = events.some((e) => !includedAtDetail(e.type, 'basic'));
    const detail: DetailLevel = hasDeep
      ? 'deep'
      : hasFull
        ? 'full'
        : hasLayers
          ? 'layers'
          : 'basic';
    this.run = null;
    this.players.delete(req);
    const p = this.newPlayer(req, detail, false);
    for (const e of events.filter((x) => x.req === req)) p.push(e, 0);
    void this.updateSettings({ detail });
    const model = events.find((e) => e.type === 'request.received');
    if (atLeast(detail, 'full') && model?.type === 'request.received') {
      this.loadTok(model.model);
    }
    this.activate(p, 'file', name);
    p.seek(p.t0 - 1);
    p.clock.resume();
  }
}

export const session = new Session();
export { OllamaError };
