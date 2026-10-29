import { useEffect, useRef, useState } from 'react';
import { Application, Graphics, Sprite, Text, Texture } from 'pixi.js';
import { session } from '../session';
import { setUi } from '../store';
import { OP_COLOR, PHASE_HEX, prefersReducedMotion, showToken } from '../theme';
import type { ViewModel } from '../engine/types';

/** Sub-block columns of one layer, in execution order. */
const COLS: Array<{ op: string; slot: number; label: string; color: number }> = [
  { op: 'norm', slot: 0, label: 'norm', color: OP_COLOR.norm },
  { op: 'attn_qkv', slot: 0, label: 'qkv', color: OP_COLOR.attn_qkv },
  { op: 'attn_scores', slot: 0, label: 'scores', color: OP_COLOR.attn_scores },
  { op: 'attn_out', slot: 0, label: 'attn out', color: OP_COLOR.attn_out },
  { op: 'residual', slot: 0, label: '+res', color: OP_COLOR.residual },
  { op: 'norm', slot: 1, label: 'norm', color: OP_COLOR.norm },
  { op: 'mlp_up', slot: 0, label: 'mlp up', color: OP_COLOR.mlp_up },
  { op: 'mlp_act', slot: 0, label: 'act', color: OP_COLOR.mlp_act },
  { op: 'mlp_down', slot: 0, label: 'mlp down', color: OP_COLOR.mlp_down },
  { op: 'residual', slot: 1, label: '+res', color: OP_COLOR.residual },
];
const NCOL = COLS.length;
function colOf(op: string, slot: number): number {
  switch (op) {
    case 'norm':
      return slot === 0 ? 0 : 5;
    case 'attn_qkv':
      return 1;
    case 'attn_scores':
      return 2;
    case 'attn_out':
      return 3;
    case 'mlp_up':
      return 6;
    case 'mlp_act':
      return 7;
    case 'mlp_down':
      return 8;
    default:
      return -1;
  }
}

const MAX_LAYERS = 256;
const BASE = 0x1a1f2c;
const STRIP_H = 58;
const ARC_H = 118;
const LABEL_W = 30;
const RES_W = 64;
const PAD = 8;
const FADE = 320;
const KV_COLS_MAX = 1024;
const SMALL = { fontFamily: 'Consolas, monospace', fontSize: 10, fill: 0x8a93a8 };
const HEAD_COLORS = Array.from({ length: 32 }, (_, i) => hsl((i * 137.5) % 360, 0.7, 0.62));

function hsl(h: number, s: number, l: number): number {
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return (f(0) << 16) | (f(8) << 8) | f(4);
}

interface Geo {
  w: number;
  h: number;
  n: number;
  rowsTop: number;
  rowsH: number;
  rowH: number;
  cellsX: number;
  cellW: number;
  resX: number;
  kvTop: number;
  kvH: number;
  arcBase: number;
  stripX0: number;
  stripX1: number;
}

interface Hover {
  x: number;
  y: number;
  layer: number;
  col: number;
}

/**
 * Full-detail forward pass: per-layer sub-blocks, residual stream, KV slot
 * grid and attention arcs over the context. All per-frame state lives in
 * preallocated typed arrays; a handful of Graphics objects are redrawn.
 */
export function ForwardPass() {
  const host = useRef<HTMLDivElement>(null);
  const geo = useRef<Geo | null>(null);
  const vmRef = useRef<ViewModel | null>(null);
  const hoverRef = useRef<Hover | null>(null);
  // per-cell (layer*NCOL+col) wall time and ms for the current step
  const cellWall = useRef(new Float64Array(MAX_LAYERS * NCOL).fill(-Infinity));
  const cellMs = useRef(new Float32Array(MAX_LAYERS * NCOL));
  const [hover, setHover] = useState<Hover | null>(null);

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

        // ---- scene ----
        const gRows = new Graphics();
        const gStrip = new Graphics();
        const gArcs = new Graphics();
        // KV slot grid texture (offscreen canvas, incrementally updated)
        const kvCanvas = document.createElement('canvas');
        kvCanvas.width = 1;
        kvCanvas.height = 1;
        const kvCtx = kvCanvas.getContext('2d', { willReadFrequently: false })!;
        let kvImg = kvCtx.createImageData(1, 1);
        let kvTex = Texture.from(kvCanvas);
        kvTex.source.scaleMode = 'nearest';
        const kvSprite = new Sprite(kvTex);
        app.stage.addChild(gRows, kvSprite, gStrip, gArcs);

        const stripNames = COLS.map((c) => {
          const t = new Text({ text: c.label, style: { ...SMALL, fill: 0xc9d1e4, fontSize: 11 } });
          app.stage.addChild(t);
          return t;
        });
        const stripMs = COLS.map(() => {
          const t = new Text({ text: '', style: SMALL });
          app.stage.addChild(t);
          return t;
        });
        const stripMsStr: string[] = COLS.map(() => '');
        const title = new Text({ text: '', style: { ...SMALL, fill: 0xe6e9f2, fontSize: 11 } });
        app.stage.addChild(title);
        let titleStr = '';
        const kvTitle = new Text({ text: 'KV cache (layers × positions)', style: SMALL });
        const ctxTitle = new Text({ text: 'context · attention', style: SMALL });
        const resTitle = new Text({ text: 'residual', style: SMALL });
        app.stage.addChild(kvTitle, ctxTitle, resTitle);
        const topLabels = [0, 1, 2].map(() => {
          const t = new Text({ text: '', style: { ...SMALL, fill: 0xe6e9f2 } });
          t.anchor.set(0.5, 1);
          app.stage.addChild(t);
          return t;
        });
        const layerLabels: Text[] = [];

        // ---- per-run state ----
        let stepKey = -2;
        let opsDone = 0;
        let resDone = 0;
        let lastOpLayer = -1;
        let lastOpCol = -1;
        let runKey = '';
        let kvCols = 1;
        let kvRows = 1;
        let kvBin = 1;
        let kvFilledUpTo = 0;
        let kvWritesDone = 0;
        let kvDirty = false;
        // lit KV pixels fading to steady colour
        const LIT_CAP = 4096;
        const litIdx = new Int32Array(LIT_CAP);
        const litWall = new Float64Array(LIT_CAP);
        let litN = 0;
        let geoKey = '';
        let lastTopAttn: unknown = null;

        const FILLED = [52, 92, 78];
        const LIT = [190, 255, 220];

        const setPixel = (i: number, rgb: number[], a = 255) => {
          const d = kvImg.data;
          d[i * 4] = rgb[0];
          d[i * 4 + 1] = rgb[1];
          d[i * 4 + 2] = rgb[2];
          d[i * 4 + 3] = a;
          kvDirty = true;
        };
        const fillColumn = (pos: number) => {
          const c = Math.floor(pos / kvBin);
          if (c >= kvCols) return;
          for (let r = 0; r < kvRows; r++) setPixel(r * kvCols + c, FILLED);
        };

        const resetRun = (vm: ViewModel) => {
          const ctx = Math.max(1, vm.ctx || vm.kv.total || 4096);
          kvCols = Math.min(ctx, KV_COLS_MAX);
          kvBin = Math.ceil(ctx / kvCols);
          kvCols = Math.ceil(ctx / kvBin);
          kvRows = Math.max(1, Math.min(MAX_LAYERS, vm.layersCount || vm.layers.length || 1));
          kvCanvas.width = kvCols;
          kvCanvas.height = kvRows;
          kvImg = kvCtx.createImageData(kvCols, kvRows);
          const d = kvImg.data;
          for (let i = 0; i < kvCols * kvRows; i++) {
            d[i * 4] = 26;
            d[i * 4 + 1] = 31;
            d[i * 4 + 2] = 44;
            d[i * 4 + 3] = 255;
          }
          kvTex.destroy(true);
          kvTex = Texture.from(kvCanvas);
          kvTex.source.scaleMode = 'nearest';
          kvSprite.texture = kvTex;
          kvFilledUpTo = 0;
          kvWritesDone = 0;
          litN = 0;
          kvDirty = true;
          stepKey = -2;
        };

        const relayout = (n: number) => {
          const w = app.screen.width;
          const h = app.screen.height;
          const key = `${w}x${h}x${n}`;
          if (key === geoKey) return;
          geoKey = key;
          const kvH = Math.max(36, Math.min(90, n * 2));
          const rowsTop = PAD + STRIP_H + 6;
          const arcBase = h - 20;
          const kvTop = h - ARC_H - kvH - 10;
          const rowsH = Math.max(20, kvTop - 18 - rowsTop);
          const rowH = rowsH / Math.max(1, n);
          const cellsX = PAD + LABEL_W;
          const resX = w - PAD - RES_W;
          const cellW = Math.max(2, (resX - 8 - cellsX) / NCOL);
          geo.current = {
            w,
            h,
            n,
            rowsTop,
            rowsH,
            rowH,
            cellsX,
            cellW,
            resX,
            kvTop,
            kvH,
            arcBase,
            stripX0: cellsX,
            stripX1: w - PAD,
          };
          const stripCellW = (w - 2 * PAD) / NCOL;
          for (let c = 0; c < NCOL; c++) {
            stripNames[c].x = PAD + c * stripCellW + 6;
            stripNames[c].y = PAD + 20;
            stripMs[c].x = PAD + c * stripCellW + 6;
            stripMs[c].y = PAD + 36;
          }
          title.x = PAD;
          title.y = PAD;
          for (const t of layerLabels) t.destroy();
          layerLabels.length = 0;
          const every = Math.max(1, Math.ceil(13 / rowH));
          for (let i = 0; i < n; i += every) {
            const t = new Text({ text: `L${i}`, style: SMALL });
            t.x = PAD;
            t.y = rowsTop + i * rowH + rowH / 2 - 7;
            app.stage.addChild(t);
            layerLabels.push(t);
          }
          resTitle.x = resX;
          resTitle.y = rowsTop - 14;
          kvTitle.x = cellsX;
          kvTitle.y = kvTop - 14;
          ctxTitle.x = cellsX;
          ctxTitle.y = h - ARC_H - 6;
          kvSprite.x = cellsX;
          kvSprite.y = kvTop;
          kvSprite.width = w - PAD - cellsX;
          kvSprite.height = kvH;
        };

        unsub = session.onFrame((now, vm) => {
          vmRef.current = vm;
          const n = Math.min(MAX_LAYERS, Math.max(vm.layersCount, vm.layers.length));
          const rk = `${vm.req}|${vm.ctx}|${vm.layersCount}`;
          if (rk !== runKey) {
            runKey = rk;
            resetRun(vm);
          }
          relayout(n);
          const G = geo.current!;
          const reduced = prefersReducedMotion();
          const sd = vm.stepDetail;
          const walls = cellWall.current;
          const mss = cellMs.current;

          // ---- incremental ingest of the current step ----
          if (!sd || sd.step !== stepKey) {
            walls.fill(-Infinity);
            mss.fill(0);
            opsDone = 0;
            resDone = 0;
            kvWritesDone = 0;
            stepKey = sd ? sd.step : -2;
            lastOpLayer = -1;
            lastOpCol = -1;
          }
          if (sd) {
            for (; opsDone < sd.ops.length; opsDone++) {
              const o = sd.ops[opsDone];
              const c = colOf(o.op, o.slot);
              if (c < 0 || o.layer >= MAX_LAYERS) continue;
              const idx = o.layer * NCOL + c;
              walls[idx] = o.wall;
              mss[idx] = o.ms;
              // +res after attention lands when the second norm starts
              if (c === 5) walls[o.layer * NCOL + 4] = o.wall;
              lastOpLayer = o.layer;
              lastOpCol = c;
            }
            for (; resDone < sd.residual.length; resDone++) {
              const r = sd.residual[resDone];
              if (r.layer < MAX_LAYERS) walls[r.layer * NCOL + 9] = r.wall;
            }
            for (; kvWritesDone < sd.kvWrites.length; kvWritesDone++) {
              const k = sd.kvWrites[kvWritesDone];
              const c = Math.floor(k.pos / kvBin);
              if (k.layer >= kvRows || c >= kvCols) continue;
              const pi = k.layer * kvCols + c;
              setPixel(pi, reduced ? FILLED : LIT);
              if (!reduced && litN < LIT_CAP) {
                litIdx[litN] = pi;
                litWall[litN] = k.wall;
                litN++;
              }
            }
          }
          // positions written in earlier steps / prefill are steadily filled
          const promptLen = vm.promptTokens.length;
          const steadyTo = Math.max(
            vm.phase === 'decoding' || vm.phase === 'done' ? vm.kv.used : 0,
            Math.min(promptLen, vm.kv.used),
          );
          for (; kvFilledUpTo < steadyTo; kvFilledUpTo++) fillColumn(kvFilledUpTo);
          // fade lit pixels
          let w2 = 0;
          for (let i = 0; i < litN; i++) {
            const age = now - litWall[i];
            const pi = litIdx[i];
            if (age >= 600 || age < 0) {
              if (age >= 600) setPixel(pi, FILLED);
              if (age < 0) {
                litIdx[w2] = pi;
                litWall[w2] = litWall[i];
                w2++;
              }
              continue;
            }
            const t = age / 600;
            setPixel(pi, [
              LIT[0] + (FILLED[0] - LIT[0]) * t,
              LIT[1] + (FILLED[1] - LIT[1]) * t,
              LIT[2] + (FILLED[2] - LIT[2]) * t,
            ]);
            litIdx[w2] = pi;
            litWall[w2] = litWall[i];
            w2++;
          }
          litN = w2;
          if (kvDirty) {
            kvCtx.putImageData(kvImg, 0, 0);
            kvTex.source.update();
            kvDirty = false;
          }

          // ---- current-layer strip ----
          const curLayer = lastOpLayer >= 0 ? lastOpLayer : -1;
          const tStr = sd
            ? `step ${sd.step} · layer ${curLayer >= 0 ? curLayer : '–'} / ${n}`
            : vm.phase === 'decoding'
              ? 'waiting for op events…'
              : 'forward pass (Full detail)';
          if (tStr !== titleStr) {
            titleStr = tStr;
            title.text = tStr;
          }
          gRows.clear();
          const stripCellW = (G.w - 2 * PAD) / NCOL;
          for (let c = 0; c < NCOL; c++) {
            const x = PAD + c * stripCellW;
            const idx = curLayer >= 0 ? curLayer * NCOL + c : -1;
            const wl = idx >= 0 ? walls[idx] : -Infinity;
            const done = wl > -Infinity;
            gRows.roundRect(x + 2, PAD + 16, stripCellW - 4, STRIP_H - 18, 5).fill({
              color: done ? COLS[c].color : BASE,
              alpha: done ? 0.35 : 1,
            });
            const age = now - wl;
            if (done && c === lastOpCol && !reduced && age < FADE * 2) {
              gRows.roundRect(x + 2, PAD + 16, stripCellW - 4, STRIP_H - 18, 5).fill({
                color: COLS[c].color,
                alpha: 0.6 * (1 - age / (FADE * 2)),
              });
            }
            if (done && c === lastOpCol) {
              gRows.roundRect(x + 2, PAD + 16, stripCellW - 4, STRIP_H - 18, 5).stroke({
                width: 1.5,
                color: 0xffffff,
                alpha: 0.8,
              });
            }
            const ms =
              idx >= 0 && mss[idx] > 0
                ? `${mss[idx].toFixed(3)} ms`
                : c === 4 || c === 9
                  ? done
                    ? 'added'
                    : ''
                  : '';
            if (ms !== stripMsStr[c]) {
              stripMsStr[c] = ms;
              stripMs[c].text = ms;
            }
            if (c < NCOL - 1) {
              gRows
                .moveTo(x + stripCellW - 2, PAD + 16 + (STRIP_H - 18) / 2)
                .lineTo(x + stripCellW + 2, PAD + 16 + (STRIP_H - 18) / 2)
                .stroke({ width: 1, color: 0x4a5168 });
            }
          }

          // ---- layer rows × sub-blocks ----
          const gap = G.rowH > 5 ? 1 : 0;
          const hh = Math.max(1, G.rowH - gap);
          for (let l = 0; l < n; l++) {
            const y = G.rowsTop + l * G.rowH;
            for (let c = 0; c < NCOL; c++) {
              const x = G.cellsX + c * G.cellW;
              const idx = l * NCOL + c;
              const wl = walls[idx];
              if (wl === -Infinity) {
                gRows.rect(x, y, G.cellW - 1, hh).fill({ color: BASE });
                continue;
              }
              gRows.rect(x, y, G.cellW - 1, hh).fill({ color: COLS[c].color, alpha: 0.4 });
              const age = now - wl;
              if (!reduced && age >= 0 && age < FADE) {
                gRows
                  .rect(x, y, G.cellW - 1, hh)
                  .fill({ color: 0xffffff, alpha: 0.6 * (1 - age / FADE) });
              }
            }
          }
          if (curLayer >= 0 && lastOpCol >= 0) {
            gRows
              .rect(G.cellsX - 2, G.rowsTop + curLayer * G.rowH - 1, NCOL * G.cellW + 3, hh + 2)
              .stroke({
                width: 1,
                color: 0xffffff,
                alpha: 0.7,
              });
          }
          const hv = hoverRef.current;
          if (hv && hv.layer >= 0 && hv.col >= 0) {
            gRows
              .rect(G.cellsX + hv.col * G.cellW, G.rowsTop + hv.layer * G.rowH, G.cellW - 1, hh)
              .stroke({
                width: 1,
                color: 0xffffff,
                alpha: 0.9,
              });
          }

          // ---- residual stream ----
          gRows.rect(G.resX, G.rowsTop, RES_W, G.rowsH).fill({ color: 0x141824 });
          if (sd && sd.residual.length) {
            let maxN = 1e-9;
            let maxD = 1e-9;
            for (const r of sd.residual) {
              if (r.norm > maxN) maxN = r.norm;
              if (r.delta > maxD) maxD = r.delta;
            }
            for (const r of sd.residual) {
              if (r.layer >= n) continue;
              const f = r.norm / maxN;
              const bw = Math.max(2, RES_W * f);
              const y = G.rowsTop + r.layer * G.rowH;
              gRows
                .rect(G.resX + (RES_W - bw) / 2, y, bw, hh)
                .fill({ color: OP_COLOR.residual, alpha: 0.25 + 0.6 * f });
              const age = now - r.wall;
              if (!reduced && age >= 0 && age < 450) {
                const p = (1 - age / 450) * (r.delta / maxD);
                gRows.rect(G.resX + (RES_W - bw) / 2 - 3 * p, y, bw + 6 * p, hh).fill({
                  color: 0xffffff,
                  alpha: 0.55 * p,
                });
              }
            }
          }

          // ---- token strip + attention arcs ----
          gStrip.clear();
          gArcs.clear();
          const ctxUsed = Math.max(1, vm.kv.used + 1);
          const x0 = G.stripX0;
          const x1 = G.stripX1;
          const sw = x1 - x0;
          const xOf = (pos: number) => x0 + ((pos + 0.5) / ctxUsed) * sw;
          const base = G.arcBase;
          const pLen = Math.min(promptLen, ctxUsed);
          gStrip
            .rect(x0, base, (sw * pLen) / ctxUsed, 8)
            .fill({ color: PHASE_HEX.prefill, alpha: 0.55 });
          gStrip.rect(x0 + (sw * pLen) / ctxUsed, base, (sw * (ctxUsed - pLen)) / ctxUsed, 8).fill({
            color: PHASE_HEX.decoding,
            alpha: 0.55,
          });
          if (pLen > 0 && pLen < ctxUsed) {
            const bx = x0 + (sw * pLen) / ctxUsed;
            gStrip
              .moveTo(bx, base - 4)
              .lineTo(bx, base + 12)
              .stroke({ width: 1, color: 0xffffff, alpha: 0.6 });
          }
          // per-token ticks when there's room
          if (sw / ctxUsed >= 3) {
            for (let p = 0; p < ctxUsed; p++) {
              const x = x0 + (p / ctxUsed) * sw;
              gStrip.rect(x, base, 1, 8).fill({ color: 0x0b0d12, alpha: 0.6 });
            }
          }
          const cur = vm.kv.used;
          const xc = xOf(Math.min(cur, ctxUsed - 1));
          gStrip.circle(xc, base + 4, 4).fill({ color: 0xffffff });

          if (sd && sd.attn.length) {
            const attn = sd.attn;
            const latest = attn[attn.length - 1];
            const latestLayer = latest.layer;
            // draw the most recent ~16 attention events, newest brightest
            const from = Math.max(0, attn.length - 16);
            for (let k = from; k < attn.length; k++) {
              const a = attn[k];
              const rank = attn.length - 1 - k;
              const recency = Math.pow(0.82, rank);
              const age = now - a.wall;
              const fade = reduced ? 1 : Math.max(0.25, 1 - Math.max(0, age) / 2500);
              const color =
                a.head !== undefined
                  ? HEAD_COLORS[a.head % 32]
                  : a.layer === latestLayer
                    ? 0x93c5fd
                    : 0x818cf8;
              const top = a.top;
              const m = Math.min(8, top.length);
              for (let j = 0; j < m; j++) {
                const t = top[j];
                const xp = xOf(t.pos);
                const dx = Math.abs(xc - xp);
                const hgt = Math.min(ARC_H - 22, 10 + dx * 0.45);
                const alpha = Math.min(1, t.w * 2.2) * recency * fade;
                if (alpha < 0.02) continue;
                gArcs
                  .moveTo(xc, base)
                  .quadraticCurveTo((xc + xp) / 2, base - hgt * 2, xp, base)
                  .stroke({ width: 1 + 2.5 * t.w * recency, color, alpha });
              }
            }
            // labels for top-3 attended tokens of the latest event
            if (latest !== lastTopAttn) {
              lastTopAttn = latest;
              const sorted = latest.top.slice().sort((a, b) => b.w - a.w);
              for (let i = 0; i < 3; i++) {
                const t = sorted[i];
                const lbl = topLabels[i];
                if (!t) {
                  lbl.text = '';
                  continue;
                }
                const txt =
                  t.pos < promptLen
                    ? vm.promptTokens[t.pos]?.text
                    : vm.responseTokens[t.pos - promptLen]?.text;
                lbl.text = `${showToken(txt ?? `#${t.pos}`).slice(0, 14)} ${(t.w * 100).toFixed(0)}%`;
                lbl.x = Math.max(x0 + 30, Math.min(x1 - 30, xOf(t.pos)));
                lbl.y = base - 4 - i * 12;
              }
            }
          } else if (lastTopAttn !== null) {
            lastTopAttn = null;
            for (const l of topLabels) l.text = '';
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

  const hit = (clientX: number, clientY: number): Hover | null => {
    const G = geo.current;
    if (!G || !host.current) return null;
    const r = host.current.getBoundingClientRect();
    const x = clientX - r.left;
    const y = clientY - r.top;
    // current-layer strip
    if (y >= PAD + 16 && y <= PAD + STRIP_H - 2) {
      const sw = (G.w - 2 * PAD) / NCOL;
      const col = Math.floor((x - PAD) / sw);
      const sd = vmRef.current?.stepDetail;
      const layer = sd && sd.ops.length ? sd.ops[sd.ops.length - 1].layer : -1;
      if (col >= 0 && col < NCOL && layer >= 0) return { x, y, layer, col };
      return null;
    }
    if (
      y >= G.rowsTop &&
      y < G.rowsTop + G.rowsH &&
      x >= G.cellsX &&
      x < G.cellsX + NCOL * G.cellW
    ) {
      const layer = Math.floor((y - G.rowsTop) / G.rowH);
      const col = Math.floor((x - G.cellsX) / G.cellW);
      if (layer >= 0 && layer < G.n) return { x, y, layer, col };
    }
    return null;
  };

  const vm = vmRef.current;
  const tipMs =
    hover && hover.layer < MAX_LAYERS ? cellMs.current[hover.layer * NCOL + hover.col] : 0;
  const tipDone = hover ? cellWall.current[hover.layer * NCOL + hover.col] > -Infinity : false;
  return (
    <div
      ref={host}
      style={{ position: 'absolute', inset: 0 }}
      onPointerMove={(e) => {
        const h = hit(e.clientX, e.clientY);
        hoverRef.current = h;
        setHover(h);
      }}
      onPointerLeave={() => {
        hoverRef.current = null;
        setHover(null);
      }}
      onClick={(e) => {
        const h = hit(e.clientX, e.clientY);
        const v = vmRef.current;
        if (!h || !v) return;
        const step = v.stepDetail?.step ?? v.currentStep;
        const c = COLS[h.col];
        if (c.op === 'residual') setUi({ inspector: { kind: 'layer', step, layer: h.layer } });
        else setUi({ inspector: { kind: 'op', step, layer: h.layer, op: c.op, slot: c.slot } });
        session.pause();
      }}
    >
      {hover && (
        <div
          className="tip"
          style={{
            position: 'absolute',
            left: Math.min(hover.x + 12, (host.current?.clientWidth ?? 300) - 190),
            top: hover.y + 12,
          }}
        >
          <div className="mono">
            L{hover.layer} · {COLS[hover.col].op}
            {COLS[hover.col].op === 'norm' ? ` #${COLS[hover.col].slot + 1}` : ''}
          </div>
          <div className="mono">
            {tipMs > 0 ? `${tipMs.toFixed(3)} ms` : tipDone ? 'done' : 'not run yet'}
          </div>
          <div className="mono" style={{ color: 'var(--muted)' }}>
            step {vm?.stepDetail?.step ?? '–'} · click to inspect
          </div>
        </div>
      )}
    </div>
  );
}
