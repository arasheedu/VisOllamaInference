import { useEffect, useRef } from 'react';
import { useUi } from '../store';

/**
 * Mixture-of-experts routing for the current token: rows = layers, columns =
 * experts, brightness = router weight. Shown only when the runner emits
 * decode.moe (MoE models). Redrawn only when routing changes.
 */
export function MoePanel() {
  const moe = useUi((s) => s.vm.stepDetail?.moe);
  const layers = useUi((s) => s.vm.layersCount);
  const step = useUi((s) => s.vm.stepDetail?.step ?? -1);
  const ref = useRef<HTMLCanvasElement>(null);

  const nExperts = moe?.length
    ? Math.max(...moe.flatMap((m) => m.experts.map((e) => e.id))) + 1
    : 0;
  const cols = nExperts <= 8 ? 8 : 2 ** Math.ceil(Math.log2(nExperts));

  useEffect(() => {
    const c = ref.current;
    if (!c || !moe?.length) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const rows = Math.max(layers, ...moe.map((m) => m.layer + 1));
    const cw = c.width / cols;
    const rh = c.height / rows;
    ctx.fillStyle = '#10131b';
    ctx.fillRect(0, 0, c.width, c.height);
    const last = moe[moe.length - 1];
    for (const m of moe) {
      for (const e of m.experts) {
        const a = Math.min(1, 0.25 + e.w * 2.5);
        ctx.fillStyle = m === last ? `rgba(251,191,36,${a})` : `rgba(129,140,248,${a})`;
        ctx.fillRect(e.id * cw, m.layer * rh, Math.max(1, cw - 0.5), Math.max(1, rh - 0.5));
      }
    }
  }, [moe, layers, cols]);

  if (!moe?.length) return null;
  const last = moe[moe.length - 1];
  return (
    <div
      data-testid="moe"
      style={{
        position: 'absolute',
        left: 10,
        bottom: 10,
        width: 300,
        background: '#0b0d12e6',
        border: '1px solid var(--line)',
        borderRadius: 8,
        padding: '6px 8px',
        zIndex: 4,
        pointerEvents: 'none',
      }}
    >
      <div
        style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 4, fontFamily: 'var(--mono)' }}
      >
        MoE routing · step {step} · {nExperts} experts × {layers} layers
      </div>
      <canvas
        ref={ref}
        width={cols * 2}
        height={Math.max(48, layers * 2)}
        style={{ width: '100%', height: 90, imageRendering: 'pixelated' }}
      />
      <div style={{ fontSize: 11, fontFamily: 'var(--mono)', marginTop: 4 }}>
        L{last.layer}:{' '}
        {last.experts.slice(0, 4).map((e) => (
          <span key={e.id} style={{ color: '#fbbf24', marginRight: 6 }}>
            e{e.id} {(e.w * 100).toFixed(0)}%
          </span>
        ))}
      </div>
    </div>
  );
}
