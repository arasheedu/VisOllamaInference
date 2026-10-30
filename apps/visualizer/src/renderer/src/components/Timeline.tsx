import { atLeast } from 'trace-schema';
import { useRef } from 'react';
import { session } from '../session';
import { useUi } from '../store';
import { fmtMs, PHASE_COLOR, PHASE_LABEL } from '../theme';
import { StepControls } from './StepControls';

export function dilationLabel(d: number): string {
  if (!isFinite(d) || d <= 0) return '0×';
  return d >= 10
    ? `${d.toFixed(0)}×`
    : d >= 1
      ? `${d.toFixed(1)}×`
      : `${d.toFixed(d < 0.1 ? 3 : 2)}×`;
}

export function Timeline() {
  const timeline = useUi((s) => s.vm.timeline);
  const clock = useUi((s) => s.vm.clock);
  const derived = useUi((s) => s.vm.derived);
  const stats = useUi((s) => s.vm.stats);
  const detail = useUi((s) => s.settings.detail);
  const stepMode = useUi((s) => s.stepMode);
  const viewing = useUi((s) => s.viewing);
  const fps = useUi((s) => s.fps);
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const t0 = clock.t0;
  const t1 = Math.max(clock.tEnd, clock.tp, t0 + 1);
  const span = t1 - t0;
  const x = (t: number) => `${((t - t0) / span) * 100}%`;

  const seekAt = (clientX: number) => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const f = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
    session.seek(t0 + f * span);
  };

  const shown = clock.paused ? 0 : clock.dilation;

  return (
    <div className="panel timeline" data-testid="timeline">
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <button
          className={clock.live ? 'active' : ''}
          onClick={() => session.goLive()}
          title={clock.live ? 'Following the live run' : 'Re-attach to live'}
          data-testid="live"
        >
          ● {viewing === 'file' ? 'End' : 'Live'}
        </button>
        <span className="dilation" title="Playback time ÷ wall time" data-testid="dilation">
          showing at {dilationLabel(shown)}
        </span>
        {clock.catchingUp && (
          <span className="badge hot">catching up {clock.speed.toFixed(1)}×</span>
        )}
        {clock.degraded && (
          <span className="badge hot" data-testid="fast-forward">
            fast-forwarding (Layers)
          </span>
        )}
        {clock.paused && <span className="badge">paused</span>}
        {(atLeast(detail, 'full') || stepMode) && <StepControls />}
      </div>
      <div
        ref={ref}
        className="track-wrap"
        onPointerDown={(e) => {
          dragging.current = true;
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
          seekAt(e.clientX);
        }}
        onPointerMove={(e) => dragging.current && seekAt(e.clientX)}
        onPointerUp={() => (dragging.current = false)}
        data-testid="scrubber"
      >
        <div
          style={{
            position: 'absolute',
            top: 6,
            height: 22,
            left: 0,
            right: 0,
            background: '#151925',
            borderRadius: 4,
          }}
        />
        {timeline.map((b, i) => {
          const to = b.to ?? Math.max(clock.tp, b.from);
          return (
            <div
              key={i}
              className="band"
              style={{
                left: x(b.from),
                width: `calc(${x(to)} - ${x(b.from)})`,
                background: PHASE_COLOR[b.phase],
                minWidth: 2,
              }}
              title={`${PHASE_LABEL[b.phase]} · ${fmtMs((to - b.from) / 1e6)}`}
            >
              {PHASE_LABEL[b.phase]}
            </div>
          );
        })}
        {span > 1 && <div className="playhead" style={{ left: x(clock.tp) }} />}
      </div>
      <div className="stats" data-testid="stats">
        <span>
          TTFT <b>{fmtMs(derived.ttftMs)}</b>
        </span>
        <span>
          prompt{' '}
          <b>{derived.promptTokPerSec ? `${derived.promptTokPerSec.toFixed(0)} tok/s` : '–'}</b>
        </span>
        <span>
          gen <b>{derived.genTokPerSec ? `${derived.genTokPerSec.toFixed(1)} tok/s` : '–'}</b>
        </span>
        <span>
          total <b>{fmtMs(stats?.total_ms ?? derived.totalMs)}</b>
        </span>
        <span title="Renderer frames per second">
          <b>{fps}</b> fps
        </span>
      </div>
    </div>
  );
}
