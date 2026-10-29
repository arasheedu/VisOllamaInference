import { motion } from 'motion/react';
import { useUi } from '../store';
import { showToken, SPRING, useReducedMotion } from '../theme';

const MAX_ENTROPY = Math.log(16); // nats; beyond this the step is "very uncertain"

export function EntropyGauge({ entropy }: { entropy: number }) {
  const reduced = useReducedMotion();
  const f = Math.max(0, Math.min(1, entropy / MAX_ENTROPY));
  const r = 30;
  const circ = Math.PI * r;
  const color = f < 0.25 ? '#34d399' : f < 0.55 ? '#fbbf24' : '#f87171';
  return (
    <svg
      width={80}
      height={48}
      viewBox="0 0 80 48"
      aria-label={`entropy ${entropy.toFixed(2)} nats`}
    >
      <path
        d={`M 10 40 A ${r} ${r} 0 0 1 70 40`}
        stroke="#232838"
        strokeWidth={7}
        fill="none"
        strokeLinecap="round"
      />
      <motion.path
        d={`M 10 40 A ${r} ${r} 0 0 1 70 40`}
        stroke={color}
        strokeWidth={7}
        fill="none"
        strokeLinecap="round"
        strokeDasharray={circ}
        initial={false}
        animate={{ strokeDashoffset: circ * (1 - f), stroke: color }}
        transition={reduced ? { duration: 0 } : SPRING}
      />
      <text x={40} y={38} textAnchor="middle" fontSize={12} fill="#e6e9f2" fontFamily="var(--mono)">
        {entropy.toFixed(2)}
      </text>
    </svg>
  );
}

export function SamplingPanel() {
  const s = useUi((u) => u.vm.sampling);
  const options = useUi((u) => u.vm.options);
  const reduced = useReducedMotion();
  const top = s.topk.slice(0, 10);
  const maxP = Math.max(1e-6, ...top.map((c) => c.p));
  const f = s.entropy / MAX_ENTROPY;
  const temp = s.temp || (options.temperature as number | undefined) || 0;
  const topP = s.top_p || (options.top_p as number | undefined) || 0;
  const topK = s.top_k || (options.top_k as number | undefined) || 0;

  return (
    <div
      className="sampling"
      data-testid="sampling"
      style={{ background: f > 0.55 ? 'linear-gradient(#2c121522, transparent)' : undefined }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div className="sparams">
          <span>
            T <b>{temp ? temp.toFixed(2) : '–'}</b>
          </span>
          <span>
            top_p <b>{topP ? topP.toFixed(2) : '–'}</b>
          </span>
          <span>
            top_k <b>{topK || '–'}</b>
          </span>
          <span>
            step <b>{s.step >= 0 ? s.step : '–'}</b>
          </span>
        </div>
        <div style={{ textAlign: 'center' }}>
          <EntropyGauge entropy={s.entropy} />
          <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: -4 }}>
            {s.step < 0
              ? 'entropy'
              : f < 0.25
                ? 'confident'
                : f < 0.55
                  ? 'some doubt'
                  : 'uncertain'}
          </div>
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {top.map((c) => (
          <motion.div
            key={c.id}
            layout={!reduced}
            transition={SPRING}
            className={`sbar ${c.id === s.chosen ? 'chosen' : ''}`}
            title={`id ${c.id} · p=${c.p.toFixed(4)}`}
          >
            <span className="label">{showToken(c.text)}</span>
            <div className="track">
              <motion.div
                className="fill"
                initial={false}
                animate={{ width: `${(c.p / maxP) * 100}%` }}
                transition={reduced ? { duration: 0 } : SPRING}
              />
            </div>
            <span className="pct">{(c.p * 100).toFixed(c.p < 0.1 ? 1 : 0)}%</span>
          </motion.div>
        ))}
        {top.length === 0 && (
          <div style={{ color: 'var(--muted)', padding: 8 }}>
            Top-k candidates appear here during decoding.
          </div>
        )}
      </div>
    </div>
  );
}
