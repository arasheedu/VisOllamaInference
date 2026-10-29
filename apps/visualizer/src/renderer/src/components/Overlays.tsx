import { useEffect, useMemo } from 'react';
import type { TraceEvent } from 'trace-schema';
import { session } from '../session';
import { clearWarning, setUi, useUi, type InspectorTarget } from '../store';
import { DEFAULT_SETTINGS, type AppSettings } from '../settings';
import { dilationLabel } from './Timeline';
import { showToken } from '../theme';

// ---- warnings ---------------------------------------------------------------

export function Warnings() {
  const warnings = useUi((s) => s.warnings);
  if (!warnings.length) return null;
  return (
    <div className="warnings" data-testid="warnings">
      {warnings.map((w) => (
        <div key={w.id} className={`warning ${w.level}`} data-testid={`warning-${w.id}`}>
          <span style={{ flex: 1 }}>{w.text}</span>
          <button onClick={() => clearWarning(w.id)} title="Dismiss">
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}

// ---- debug panel (Ctrl+Shift+D) ------------------------------------------------

export function DebugPanel() {
  const open = useUi((s) => s.debugOpen);
  const vm = useUi((s) => s.vm);
  const hub = useUi((s) => s.hub);
  if (!open) return null;
  const p = session.active;
  const c = vm.clock;
  const tail = vm.recent
    .slice(-40)
    .map((e) => JSON.stringify(e).slice(0, 160))
    .join('\n');
  return (
    <div className="debug" data-testid="debug">
      <header>
        <span>phase {vm.phase}</span>
        <span>tp {((c.tp - c.t0) / 1e6).toFixed(1)} ms</span>
        <span>head {(((p?.headT() ?? 0) - c.t0) / 1e6).toFixed(1)} ms</span>
        <span>lag {c.lagMs.toFixed(0)} ms</span>
        <span>speed {c.speed.toFixed(2)}×</span>
        <span>dilation {dilationLabel(c.dilation)}</span>
        <span>drops {c.dropped}</span>
        <span>filtered {p?.filtered ?? 0}</span>
        <span>late {p?.late ?? 0}</span>
        <span>
          buffer {p?.position ?? 0}/{p?.events.length ?? 0}
        </span>
        <span>applied {vm.eventsApplied}</span>
        <span>
          hub {hub?.listening ? `:${hub.port}` : 'down'} p{hub?.producers ?? 0} v
          {hub?.consumers ?? 0} in{hub?.eventsIn ?? 0}
        </span>
        {c.catchingUp && <span style={{ color: 'var(--warn)' }}>CATCHING UP</span>}
        {c.degraded && <span style={{ color: 'var(--warn)' }}>DEGRADED</span>}
      </header>
      <pre>{tail}</pre>
    </div>
  );
}

// ---- inspector ----------------------------------------------------------------

function softmax(vs: number[]): number[] {
  const m = Math.max(...vs);
  const e = vs.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

function eventsFor(t: InspectorTarget): TraceEvent[] {
  const ev = session.active?.events ?? [];
  switch (t.kind) {
    case 'token':
      return ev.filter(
        (e) =>
          'step' in e &&
          e.step === t.step &&
          e.type.startsWith('decode.') &&
          e.type !== 'decode.op' &&
          e.type !== 'decode.layer' &&
          e.type !== 'decode.residual' &&
          e.type !== 'decode.attn' &&
          e.type !== 'decode.kv.write',
      );
    case 'prompt-token':
      return ev.filter((e) => e.type === 'tokenize.done');
    case 'layer':
      return ev.filter(
        (e) => 'step' in e && e.step === t.step && 'layer' in e && e.layer === t.layer,
      );
    case 'op':
      return ev.filter(
        (e) => e.type === 'decode.op' && e.step === t.step && e.layer === t.layer && e.op === t.op,
      );
    case 'stage':
      return ev.filter(
        (e) => e.type === 'decode.sample.stage' && e.step === t.step && e.stage === t.stage,
      );
  }
}

function Derived({ t, events }: { t: InspectorTarget; events: TraceEvent[] }) {
  const all = session.active?.events ?? [];
  if (t.kind === 'token') {
    const sample = events.find((e) => e.type === 'decode.sample');
    const tok = events.find((e) => e.type === 'decode.token');
    const chosen = sample?.type === 'decode.sample' ? sample.chosen_id : undefined;
    const stages = events.filter(
      (e): e is Extract<TraceEvent, { type: 'decode.sample.stage' }> =>
        e.type === 'decode.sample.stage',
    );
    const logits = events.find((e) => e.type === 'decode.logits');
    return (
      <>
        <div className="kv">
          <span>step</span>
          <span>{t.step}</span>
          <span>text</span>
          <span>{tok?.type === 'decode.token' ? JSON.stringify(tok.text) : '–'}</span>
          <span>chosen id</span>
          <span>{chosen ?? '–'}</span>
          <span>entropy</span>
          <span>
            {logits?.type === 'decode.logits' ? logits.entropy.toFixed(3) + ' nats' : '–'}
          </span>
          {logits?.type === 'decode.logits' && (
            <>
              <span>p (top-k)</span>
              <span>{((logits.topk.find((c) => c.id === chosen)?.p ?? 0) * 100).toFixed(2)}%</span>
            </>
          )}
        </div>
        {stages.length > 0 && (
          <>
            <div style={{ color: 'var(--muted)', marginBottom: 4 }}>
              Probability of the chosen token at each stage
            </div>
            <div className="kv">
              {stages.map((s) => {
                const isProb = s.stage === 'softmax' || s.stage === 'draw';
                const kept =
                  s.cutoff !== undefined && !isProb
                    ? s.candidates.slice(0, s.cutoff)
                    : s.candidates;
                const probs = isProb ? kept.map((c) => c.v) : softmax(kept.map((c) => c.v));
                const i = kept.findIndex((c) => c.id === chosen);
                return [
                  <span key={s.stage + 'k'}>{s.stage}</span>,
                  <span key={s.stage + 'v'}>
                    {i >= 0
                      ? `${(probs[i] * 100).toFixed(2)}% (rank ${i + 1}/${kept.length})`
                      : 'pruned'}
                  </span>,
                ];
              })}
            </div>
          </>
        )}
      </>
    );
  }
  if (t.kind === 'layer') {
    const layerEvs = all.filter(
      (e): e is Extract<TraceEvent, { type: 'decode.layer' }> =>
        e.type === 'decode.layer' && e.step === t.step,
    );
    const total = layerEvs.reduce((s, e) => s + e.ms, 0);
    const me = layerEvs.find((e) => e.layer === t.layer);
    const res = events.find((e) => e.type === 'decode.residual');
    return (
      <div className="kv">
        <span>layer</span>
        <span>
          {t.layer} (step {t.step})
        </span>
        <span>time</span>
        <span>{me ? `${me.ms.toFixed(3)} ms` : '–'}</span>
        <span>share of step</span>
        <span>{me && total ? `${((me.ms / total) * 100).toFixed(1)}%` : '–'}</span>
        <span>residual norm</span>
        <span>
          {res?.type === 'decode.residual'
            ? `${res.norm.toFixed(2)} (+${res.delta_norm.toFixed(2)})`
            : (me?.norm?.toFixed(2) ?? '–')}
        </span>
      </div>
    );
  }
  if (t.kind === 'op') {
    const ops = all.filter(
      (e): e is Extract<TraceEvent, { type: 'decode.op' }> =>
        e.type === 'decode.op' && e.step === t.step && e.layer === t.layer,
    );
    const total = ops.reduce((s, e) => s + e.ms, 0);
    const me = ops.filter((e) => e.op === t.op)[t.slot];
    return (
      <div className="kv">
        <span>op</span>
        <span>
          {t.op} · layer {t.layer} · step {t.step}
        </span>
        <span>time</span>
        <span>{me ? `${me.ms.toFixed(4)} ms` : '–'}</span>
        <span>share of layer</span>
        <span>{me && total ? `${((me.ms / total) * 100).toFixed(1)}%` : '–'}</span>
      </div>
    );
  }
  if (t.kind === 'stage') {
    const s = events[0];
    if (s?.type !== 'decode.sample.stage') return null;
    return (
      <div className="kv">
        <span>stage</span>
        <span>{s.stage}</span>
        <span>candidates</span>
        <span>{s.candidates.length}</span>
        <span>cutoff</span>
        <span>{s.cutoff ?? '–'}</span>
        {s.draw !== undefined && (
          <>
            <span>draw u</span>
            <span>{s.draw.toFixed(5)}</span>
          </>
        )}
      </div>
    );
  }
  if (t.kind === 'prompt-token') {
    const tok = useUi.getState().vm.promptTokens[t.index];
    return (
      <div className="kv">
        <span>index</span>
        <span>{t.index}</span>
        <span>id</span>
        <span>{tok?.id}</span>
        <span>text</span>
        <span>{tok ? JSON.stringify(tok.text) : ''}</span>
      </div>
    );
  }
  return null;
}

export function Inspector() {
  const target = useUi((s) => s.inspector);
  const version = useUi((s) => s.vm.eventsApplied);
  // re-query the player buffer whenever new events have been applied
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const events = useMemo(() => (target ? eventsFor(target) : []), [target, version]);
  if (!target) return null;
  const raw =
    target.kind === 'prompt-token'
      ? (() => {
          const e = events[0];
          return e?.type === 'tokenize.done' ? { ...e, tokens: `[${e.tokens.length} tokens]` } : e;
        })()
      : events;
  return (
    <div className="panel inspector" data-testid="inspector">
      <h3>
        Inspector · {target.kind}
        <span className="spacer" />
        <button onClick={() => setUi({ inspector: null })} style={{ padding: '0 6px' }}>
          ✕
        </button>
      </h3>
      <div className="body">
        <Derived t={target} events={events} />
        <div style={{ color: 'var(--muted)', margin: '8px 0 4px' }}>
          Raw event{Array.isArray(raw) && raw.length !== 1 ? 's' : ''}
        </div>
        <pre data-testid="inspector-raw">{JSON.stringify(raw, null, 1)}</pre>
      </div>
    </div>
  );
}

// ---- settings -------------------------------------------------------------------

function Num({
  label,
  value,
  onChange,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
}) {
  return (
    <div className="row">
      <label>{label}</label>
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

export function SettingsModal() {
  const open = useUi((s) => s.settingsOpen);
  const s = useUi((u) => u.settings);
  if (!open) return null;
  const set = (patch: Partial<AppSettings>) => void session.updateSettings(patch);
  const setMin = (k: keyof AppSettings['minimums'], v: number) =>
    set({ minimums: { ...s.minimums, [k]: v } });
  return (
    <div className="modal-bg" onClick={() => setUi({ settingsOpen: false })}>
      <div className="modal" onClick={(e) => e.stopPropagation()} data-testid="settings">
        <h2>Settings</h2>
        <fieldset>
          <legend>Source</legend>
          <div className="row">
            <label>Source</label>
            <select
              value={s.source}
              onChange={(e) => set({ source: e.target.value as AppSettings['source'] })}
            >
              <option value="ollama">Ollama (HTTP API + trace socket)</option>
              <option value="mock">Mock (in-app generator)</option>
            </select>
          </div>
          <div className="row">
            <label>Ollama URL</label>
            <input
              type="text"
              value={s.ollamaUrl}
              onChange={(e) => set({ ollamaUrl: e.target.value })}
            />
          </div>
          <Num label="Trace socket port" value={s.port} onChange={(port) => set({ port })} />
          <div className="row">
            <label>Step-mode runner granularity</label>
            <select
              value={s.stepGranularity}
              onChange={(e) =>
                set({ stepGranularity: e.target.value as AppSettings['stepGranularity'] })
              }
            >
              <option value="token">token</option>
              <option value="layer">layer (mock / C++ patch only)</option>
              <option value="sample">sampling stage (mock / C++ patch only)</option>
            </select>
          </div>
        </fieldset>
        <fieldset>
          <legend>Stage minimum on-screen durations (ms)</legend>
          {(Object.keys(s.minimums) as Array<keyof AppSettings['minimums']>).map((k) => (
            <Num key={k} label={k} value={s.minimums[k]} onChange={(v) => setMin(k, v)} />
          ))}
        </fieldset>
        <fieldset>
          <legend>Playback</legend>
          <Num
            label="Max catch-up speed (×)"
            value={s.maxSpeed}
            onChange={(maxSpeed) => set({ maxSpeed })}
          />
          <Num
            label="Catch-up lag threshold (ms)"
            value={s.lagThresholdMs}
            onChange={(lagThresholdMs) => set({ lagThresholdMs })}
          />
          <Num
            label="Full-detail auto-degrade threshold (ms)"
            value={s.fullDetailLagThresholdMs}
            onChange={(fullDetailLagThresholdMs) => set({ fullDetailLagThresholdMs })}
          />
          <div className="row">
            <label>Reduced motion</label>
            <select
              value={s.reducedMotion}
              onChange={(e) =>
                set({ reducedMotion: e.target.value as AppSettings['reducedMotion'] })
              }
            >
              <option value="system">Follow system</option>
              <option value="on">On</option>
              <option value="off">Off</option>
            </select>
          </div>
          <Num
            label="Max prompt chips before collapsing"
            value={s.maxPromptChips}
            onChange={(maxPromptChips) => set({ maxPromptChips })}
          />
          <div className="row">
            <label>Auto-save every run</label>
            <input
              type="checkbox"
              checked={s.autosave}
              onChange={(e) => set({ autosave: e.target.checked })}
            />
          </div>
        </fieldset>
        <fieldset>
          <legend>Mock generator / capture</legend>
          <Num
            label="Per-head attention (heads, 0 = averaged)"
            value={s.attnHeads}
            onChange={(attnHeads) => set({ attnHeads })}
          />
          <Num
            label="Mock tokens/sec (0 = model default)"
            value={s.mockTokPerSec}
            onChange={(mockTokPerSec) => set({ mockTokPerSec })}
          />
          <p style={{ color: 'var(--muted)', fontSize: 12 }}>
            Layer stride and attention-layer subsets for real Ollama are runner settings: set{' '}
            <code>OLLAMA_TRACE_LAYER_STRIDE</code> / <code>OLLAMA_TRACE_ATTN_LAYERS</code> /{' '}
            <code>OLLAMA_TRACE_ATTN_HEADS</code> in Ollama's environment (see
            ollama-patch/build.md).
          </p>
        </fieldset>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <button
            onClick={() =>
              set({ ...DEFAULT_SETTINGS, source: s.source, ollamaUrl: s.ollamaUrl, port: s.port })
            }
          >
            Reset to defaults
          </button>
          <button className="primary" onClick={() => setUi({ settingsOpen: false })}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

// ---- guided tour ------------------------------------------------------------------

interface TourStep {
  title: string;
  text: string;
  action?: () => void;
  next?: string;
}

const TOUR_PROMPT = 'Why is the sky blue?';

const TOUR: TourStep[] = [
  {
    title: 'Welcome',
    text: 'This tour runs a canned prompt in Full detail with step mode on, so every stage waits for you. Press Start to send the prompt.',
    next: 'Start',
    action: () => {
      setUi({ stepMode: true, prompt: TOUR_PROMPT, inspector: null });
      void session.updateSettings({ detail: 'full' }).then(() => {
        const s = useUi.getState();
        void session.start(TOUR_PROMPT, s.model || s.models[0]);
      });
    },
  },
  {
    title: 'Loading & tokenization',
    text: 'First the model weights are loaded (or reused if already resident). Then the prompt text, wrapped in the chat template, is split into tokens: byte-pair merges glue adjacent byte pieces together, round by round, until only vocabulary entries remain. Each final piece becomes an integer id.',
  },
  {
    title: 'Prefill',
    text: 'All prompt tokens go through the network in parallel batches. Their keys and values are written into the KV cache so later steps never recompute them. Watch the prompt chips light up batch by batch and the KV bar fill.',
    next: 'Step one token',
    action: () => session.step('layer'),
  },
  {
    title: 'Attention',
    text: 'Decoding processes one new position. In each layer, attention compares the new query against every cached key. The arcs show which earlier tokens this layer attends to most. The first token is often an "attention sink", and recent tokens usually matter a lot.',
    next: 'Next layer',
    action: () => session.step('layer'),
  },
  {
    title: 'MLP & residual stream',
    text: 'After attention, the MLP (up-projection → activation → down-projection) transforms the hidden state. Both blocks add into the residual stream on the right. Its norm grows layer by layer as each layer writes its contribution.',
    next: 'Run to sampling',
    action: () => session.step('sample'),
  },
  {
    title: 'Sampling',
    text: 'The final hidden state becomes logits over the vocabulary. The sampler funnel then applies repetition penalty, temperature, then top-k and top-p cut-offs, and a softmax. Step through each stage and watch the candidates get pruned.',
    next: 'Next stage',
    action: () => session.step('sample'),
  },
  {
    title: 'The draw',
    text: 'Finally a random number lands somewhere in the cumulative distribution; whichever token owns that slice is emitted. Click any token, layer, op or stage to inspect the raw numbers.',
    next: 'Finish the token',
    action: () => session.step('token'),
  },
  {
    title: 'Streaming',
    text: 'The chosen token is detokenized into bytes and streamed as soon as they form complete UTF-8. Then the loop repeats with one more position in the KV cache. Press Continue to let the rest of the answer run.',
    next: 'Continue',
    action: () => session.continue(),
  },
];

export function Tour() {
  const i = useUi((s) => s.tour);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setUi({ tour: null });
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  if (i === null) return null;
  const s = TOUR[i];
  const last = i === TOUR.length - 1;
  return (
    <div className="tour" data-testid="tour-card">
      <h4>
        {i + 1}/{TOUR.length} · {s.title}
      </h4>
      <p>{s.text}</p>
      <div className="actions">
        <button onClick={() => setUi({ tour: null })}>Close</button>
        {i > 0 && <button onClick={() => setUi({ tour: i - 1 })}>Back</button>}
        <button
          className="primary"
          onClick={() => {
            s.action?.();
            setUi({ tour: last ? null : i + 1 });
          }}
        >
          {s.next ?? 'Next'}
        </button>
      </div>
    </div>
  );
}

export { showToken };
