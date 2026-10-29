import { useEffect, useRef, useState } from 'react';
import { Application, Graphics, Text } from 'pixi.js';
import { session } from '../session';
import { setUi, useUi } from '../store';
import { PHASE_HEX, prefersReducedMotion } from '../theme';
import type { ViewModel } from '../engine/types';

const PAD_L = 34;
const PAD_R = 10;
const PAD_T = 6;
const KV_H = 34;
const BASE = 0x1a1f2c;
const LABEL_STYLE = { fontFamily: 'Consolas, monospace', fontSize: 10, fill: 0x8a93a8 };

interface Layout {
  w: number;
  h: number;
  n: number;
  rowsH: number;
  rowH: number;
}

type Mode = 'ms' | 'norm';

/**
 * Layer stack (one row per layer) plus KV-cache bar. Drawn from the shared
 * frame loop with a single Graphics object redrawn per frame.
 */
export function LayerCanvas() {
  const host = useRef<HTMLDivElement>(null);
  const layout = useRef<Layout>({ w: 0, h: 0, n: 0, rowsH: 0, rowH: 0 });
  const modeRef = useRef<Mode>('ms');
  const [mode, setMode] = useState<Mode>('ms');
  const [hover, setHover] = useState<{ x: number; y: number; layer: number } | null>(null);
  const vmRef = useRef<ViewModel | null>(null);
  const hoverRef = useRef<number>(-1);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    const el = host.current!;
    let disposed = false;
    let inited = false;
    let unsub: (() => void) | null = null;
    const app = new Application();
    void app
      .init({
        resizeTo: el,
        antialias: true,
        backgroundAlpha: 0,
        autoStart: false,
        preference: 'webgl',
        resolution: window.devicePixelRatio || 1,
        autoDensity: true,
      })
      .then(() => {
        inited = true;
        if (disposed) {
          app.destroy(true, { children: true });
          return;
        }
        app.ticker?.stop();
        el.appendChild(app.canvas);
        app.canvas.style.position = 'absolute';
        app.canvas.style.inset = '0';
        const g = new Graphics();
        app.stage.addChild(g);
        const labels: Text[] = [];
        const kvLabel = new Text({ text: '', style: LABEL_STYLE });
        app.stage.addChild(kvLabel);
        let kvText = '';
        let lastKey = '';

        const relayout = (n: number) => {
          const w = app.screen.width;
          const h = app.screen.height;
          const key = `${w}x${h}x${n}`;
          if (key === lastKey) return;
          lastKey = key;
          const rowsH = Math.max(10, h - PAD_T - KV_H - 8);
          const rowH = n > 0 ? rowsH / n : rowsH;
          layout.current = { w, h, n, rowsH, rowH };
          for (const t of labels) t.destroy();
          labels.length = 0;
          if (n > 0) {
            const every = Math.max(1, Math.ceil(14 / rowH));
            for (let i = 0; i < n; i += every) {
              const t = new Text({ text: `L${i}`, style: LABEL_STYLE });
              t.x = 4;
              t.y = PAD_T + i * rowH + rowH / 2 - 7;
              app.stage.addChild(t);
              labels.push(t);
            }
          }
          kvLabel.x = PAD_L;
          kvLabel.y = h - KV_H + 18;
        };

        unsub = session.onFrame((now, vm) => {
          vmRef.current = vm;
          const n = Math.max(vm.layersCount, vm.layers.length);
          relayout(n);
          const { w, h, rowH } = layout.current;
          const reduced = prefersReducedMotion();
          const flashMs = Math.max(250, useUi.getState().settings.minimums['decode.layer']);
          g.clear();
          const rw = Math.max(1, w - PAD_L - PAD_R);
          const gap = rowH > 5 ? 1 : 0;

          let maxV = 1e-9;
          for (const l of vm.layers)
            maxV = Math.max(maxV, modeRef.current === 'ms' ? l.lastMs : l.norm);

          const sweepIdx =
            vm.phase === 'prefill' && !reduced && n > 0
              ? Math.floor(((now - vm.prefill.wall) / 150) * n) % n
              : -1;

          for (let i = 0; i < n; i++) {
            const l = vm.layers[i];
            const y = PAD_T + i * rowH;
            const hh = Math.max(1, rowH - gap);
            g.rect(PAD_L, y, rw, hh).fill({ color: BASE, alpha: 1 });
            if (!l) continue;
            const v = modeRef.current === 'ms' ? l.lastMs : l.norm;
            const prefillRow = l.step === -1 && l.wall > -Infinity;
            const color = prefillRow ? PHASE_HEX.prefill : PHASE_HEX.decoding;
            if (v > 0) {
              const inten = v / maxV;
              g.rect(PAD_L, y, rw * (0.25 + 0.75 * inten), hh).fill({
                color,
                alpha: 0.12 + 0.4 * inten,
              });
            }
            const age = now - l.wall;
            if (reduced) {
              if (l.active) g.rect(PAD_L, y, rw, hh).fill({ color, alpha: 0.75 });
            } else if (age >= 0 && age < flashMs) {
              const a = 1 - age / flashMs;
              g.rect(PAD_L, y, rw, hh).fill({ color, alpha: 0.85 * a * a });
            }
            if (l.device) {
              // device tick: GPU cyan, CPU orange (partial offload is visible at a glance)
              const gpu = !/^cpu/i.test(l.device);
              g.rect(PAD_L - 4, y, 3, hh).fill({ color: gpu ? 0x22d3ee : 0xf97316, alpha: 0.9 });
            }
            if (i === sweepIdx)
              g.rect(PAD_L, y, rw, hh).fill({ color: PHASE_HEX.prefill, alpha: 0.5 });
          }
          const hl = hoverRef.current;
          if (hl >= 0 && hl < n) {
            g.rect(PAD_L, PAD_T + hl * rowH, rw, Math.max(1, rowH - gap)).stroke({
              width: 1,
              color: 0xffffff,
              alpha: 0.6,
            });
          }

          // KV cache bar
          const total = vm.kv.total || vm.ctx || 1;
          const used = Math.min(vm.kv.used, total);
          const promptLen = Math.min(vm.promptTokens.length || vm.prefill.to || 0, used);
          const ky = h - KV_H + 4;
          g.rect(PAD_L, ky, rw, 12).fill({ color: BASE });
          g.rect(PAD_L, ky, (rw * promptLen) / total, 12).fill({
            color: PHASE_HEX.prefill,
            alpha: 0.85,
          });
          g.rect(PAD_L + (rw * promptLen) / total, ky, (rw * (used - promptLen)) / total, 12).fill({
            color: PHASE_HEX.decoding,
            alpha: 0.85,
          });
          const t = `KV cache  ${used} / ${total}  (prompt ${promptLen} · generated ${Math.max(0, used - promptLen)})`;
          if (t !== kvText) {
            kvText = t;
            kvLabel.text = t;
          }
          app.render();
        });
      });
    return () => {
      disposed = true;
      unsub?.();
      if (inited) app.destroy(true, { children: true });
    };
  }, []);

  const layerAt = (clientY: number) => {
    const r = host.current!.getBoundingClientRect();
    const { rowH, n } = layout.current;
    const i = Math.floor((clientY - r.top - PAD_T) / (rowH || 1));
    return i >= 0 && i < n ? i : -1;
  };

  const l = hover ? vmRef.current?.layers[hover.layer] : undefined;
  return (
    <div
      ref={host}
      style={{ position: 'absolute', inset: 0 }}
      onPointerMove={(e) => {
        const r = host.current!.getBoundingClientRect();
        const layer = layerAt(e.clientY);
        hoverRef.current = layer;
        setHover(layer >= 0 ? { x: e.clientX - r.left, y: e.clientY - r.top, layer } : null);
      }}
      onPointerLeave={() => {
        hoverRef.current = -1;
        setHover(null);
      }}
      onClick={(e) => {
        const layer = layerAt(e.clientY);
        const vm = vmRef.current;
        if (layer < 0 || !vm) return;
        setUi({ inspector: { kind: 'layer', step: vm.currentStep, layer } });
        session.pause();
      }}
    >
      <div style={{ position: 'absolute', right: 10, bottom: 2, zIndex: 2 }} className="seg">
        {(['ms', 'norm'] as const).map((m) => (
          <button
            key={m}
            className={mode === m ? 'on' : ''}
            style={{ padding: '0 6px', fontSize: 10 }}
            onClick={(e) => {
              e.stopPropagation();
              setMode(m);
            }}
          >
            {m}
          </button>
        ))}
      </div>
      {hover && (
        <div
          className="tip"
          style={{
            position: 'absolute',
            left: Math.min(hover.x + 12, (host.current?.clientWidth ?? 300) - 180),
            top: hover.y + 12,
          }}
        >
          <div className="mono">layer {hover.layer}</div>
          <div className="mono">last {l && l.lastMs > 0 ? `${l.lastMs.toFixed(3)} ms` : '–'}</div>
          <div className="mono">norm {l && l.norm > 0 ? l.norm.toFixed(2) : '–'}</div>
          {l?.device && <div className="mono">device {l.device}</div>}
          <div className="mono" style={{ color: 'var(--muted)' }}>
            step {l?.step ?? '–'} · click to inspect
          </div>
        </div>
      )}
    </div>
  );
}
