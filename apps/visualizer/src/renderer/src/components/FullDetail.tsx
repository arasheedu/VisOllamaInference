import { useEffect, useMemo, useState } from 'react';
import { motion } from 'motion/react';
import { symbolsToBytes } from 'tokenizer';
import { session } from '../session';
import { useUi } from '../store';
import { wordAtRound } from '../tokenizer/merges';
import { SPRING, showToken, useReducedMotion } from '../theme';

const MAX_WORDS = 300;
const dec = new TextDecoder();
const decode = (s: string) => dec.decode(symbolsToBytes(s));
const identity = (s: string) => s;

/** Tokenizer merge animation: adjacent pieces slide together round by round. */
export function MergeView() {
  const merge = useUi((s) => s.merge);
  const tokenizeWall = useUi((s) => s.vm.tokenizeWall);
  const mergeMs = useUi((s) => s.settings.minimums['tokenize.merge']);
  const rm = useReducedMotion();
  const rounds = merge?.rounds ?? 0;
  const roundAt = (now: number) =>
    Number.isFinite(tokenizeWall)
      ? Math.min(rounds, Math.max(0, Math.floor((now - tokenizeWall) / Math.max(1, mergeMs * 2))))
      : rounds;
  const [round, setRound] = useState(() => roundAt(performance.now()));

  useEffect(() => {
    setRound(roundAt(performance.now()));
    let last = -1;
    return session.onFrame((now) => {
      const r = roundAt(now);
      if (r !== last) {
        last = r;
        setRound(r);
      }
    });
    // roundAt is derived from exactly these values
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tokenizeWall, rounds, mergeMs]);

  const trace = merge?.ok ? merge.trace : null;

  const words = useMemo(() => {
    if (!trace) return [];
    const n = Math.min(trace.words.length, MAX_WORDS);
    const out = [];
    for (let i = 0; i < n; i++) {
      const w = trace.words[i];
      const syms = wordAtRound(trace, i, round, identity);
      const fresh = new Set(
        trace.merges
          .filter((m) => m.word === i && m.round === round - 1)
          .map((m) => m.left + m.right),
      );
      let offset = 0;
      const pieces = syms.map((s) => {
        const p = { key: offset, raw: s, text: w.special ? s : decode(s), fresh: fresh.has(s) };
        offset += w.special ? 1 : [...s].length;
        return p;
      });
      const complete = round >= rounds;
      out.push({
        i,
        special: w.special,
        pieces,
        ids: complete && w.ids.length === pieces.length ? w.ids : null,
      });
    }
    return out;
  }, [trace, round, rounds]);

  if (!trace) return null;
  const more = trace.words.length - MAX_WORDS;
  const trans = rm ? { duration: 0 } : SPRING;

  return (
    <div className="merge-view">
      <div
        style={{ width: '100%', fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--muted)' }}
      >
        BPE merges · round {Math.min(round, rounds)} / {rounds} · {trace.merges.length} merges ·{' '}
        {trace.ids.length} tokens
        {trace.byteFallback > 0 && ` · ${trace.byteFallback} byte-fallback`}
      </div>
      {words.map((w) => (
        <span key={w.i} className="merge-word">
          {w.pieces.map((p, j) => (
            <motion.span
              key={p.key}
              layout={!rm}
              transition={trans}
              style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center' }}
            >
              <span className={`merge-sym${w.special ? ' special' : p.fresh ? ' merged' : ''}`}>
                {showToken(p.text)}
              </span>
              {w.ids && (
                <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--muted)' }}>
                  {w.ids[j]}
                </span>
              )}
            </motion.span>
          ))}
        </span>
      ))}
      {more > 0 && <span style={{ color: 'var(--muted)' }}>+{more} more words</span>}
    </div>
  );
}

/** Byte-level streaming: chunks arrive as bytes; partial UTF-8 stays pending. */
export function StreamBytes() {
  const chunks = useUi((s) => s.vm.streamChunks);
  const recent = chunks.slice(-40);

  const lat = useMemo(() => {
    const d: number[] = [];
    for (let i = 1; i < recent.length; i++) d.push((recent[i].t_ns - recent[i - 1].t_ns) / 1e6);
    return d;
  }, [recent]);

  if (!recent.length) {
    return <div style={{ color: 'var(--muted)', padding: '4px 12px' }}>No stream chunks yet</div>;
  }
  const max = Math.max(...lat, 1);
  const SW = 160;
  const SH = 36;
  const pts = lat
    .map((v, i) => `${(i / Math.max(lat.length - 1, 1)) * SW},${SH - 2 - (v / max) * (SH - 6)}`)
    .join(' ');
  const avg = lat.length ? lat.reduce((a, b) => a + b, 0) / lat.length : 0;

  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', minHeight: 0 }}>
      <div className="bytes" style={{ flex: 1, overflow: 'auto', maxHeight: '100%' }}>
        {recent.map((c, i) => {
          // pending until a later chunk completes the sequence
          const pending = !c.complete && !recent.slice(i + 1).some((x) => x.complete);
          const enc = new TextEncoder().encode(c.text);
          const hex =
            enc.length === c.bytes ? Array.from(enc, (b) => b.toString(16).padStart(2, '0')) : null;
          return (
            <span
              key={`${c.step}-${i}`}
              title={`step ${c.step} · ${c.bytes} bytes · ${c.complete ? 'complete' : 'incomplete UTF-8'}`}
              style={{
                display: 'inline-flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 1,
              }}
            >
              <span style={{ display: 'inline-flex', gap: 1 }}>
                {Array.from({ length: Math.max(c.bytes, 1) }, (_, k) => (
                  <span key={k} className={`byte${pending ? ' pending' : ''}`}>
                    {hex ? hex[k] : '··'}
                  </span>
                ))}
              </span>
              <span
                style={{
                  fontFamily: 'var(--mono)',
                  fontSize: 10,
                  color: pending ? 'var(--warn)' : '#c9d1e4',
                }}
              >
                {pending ? '…' : showToken(c.text)}
              </span>
            </span>
          );
        })}
      </div>
      <div style={{ padding: '4px 8px', flex: 'none' }}>
        <svg width={SW} height={SH} style={{ display: 'block' }}>
          <polyline points={pts} fill="none" stroke="#34d399" strokeWidth={1.2} />
        </svg>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--muted)' }}>
          inter-chunk {avg.toFixed(1)} ms avg
        </div>
      </div>
    </div>
  );
}
