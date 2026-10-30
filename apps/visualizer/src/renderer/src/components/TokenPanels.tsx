import { atLeast } from 'trace-schema';
import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import type { ResponseToken } from '../engine/types';
import { session } from '../session';
import { setUi, useUi } from '../store';
import { showToken } from '../theme';
import { MergeView, StreamBytes } from './FullDetail';

export interface TipState {
  x: number;
  y: number;
  content: ReactNode;
}

export function Tip({ tip }: { tip: TipState | null }) {
  if (!tip) return null;
  return (
    <div
      className="tip"
      style={{ left: Math.min(tip.x + 14, window.innerWidth - 260), top: Math.max(8, tip.y - 10) }}
    >
      {tip.content}
    </div>
  );
}

const isSpecial = (t: string) => /^<\|?[^<>\s]+\|?>$/.test(t) || /^\[[A-Z_]+\]$/.test(t);
const isByte = (t: string) => t.includes('�');

/** Number of prompt chips revealed during the tokenize stagger. */
function useRevealCount(n: number): number {
  const [count, setCount] = useState(n);
  useEffect(() => {
    let last = -1;
    return session.onFrame((now, vm) => {
      const minMs = useUi.getState().settings.minimums.tokenize;
      let c = vm.promptTokens.length;
      if (vm.phase === 'tokenizing' && vm.tokenizeWall > -Infinity && c > 0) {
        c = Math.min(c, Math.floor(((now - vm.tokenizeWall) / (minMs * 0.8)) * c) + 1);
      }
      if (c !== last) {
        last = c;
        setCount(c);
      }
    });
  }, []);
  return Math.min(count, n);
}

export function PromptTokens() {
  const tokens = useUi((s) => s.vm.promptTokens);
  const phase = useUi((s) => s.vm.phase);
  const prefillTo = useUi((s) => s.vm.prefill.to);
  const merge = useUi((s) => s.merge);
  const detail = useUi((s) => s.settings.detail);
  const maxChips = useUi((s) => s.settings.maxPromptChips);
  const [tip, setTip] = useState<TipState | null>(null);
  const revealed = useRevealCount(tokens.length);
  const showMerge = atLeast(detail, 'full') && phase === 'tokenizing' && merge?.ok;

  const shown = Math.min(revealed, maxChips);
  const prefillLit = phase === 'prefill';
  const cached = phase === 'decoding' || phase === 'done';

  return (
    <div className="panel" data-testid="prompt-tokens">
      <h3>
        Prompt tokens <span className="mono">{tokens.length || ''}</span>
        <span className="spacer" />
        {prefillLit && <span style={{ color: 'var(--prefill)' }}>prefill → KV cache</span>}
      </h3>
      <div className="panel-body">
        {showMerge ? (
          <MergeView />
        ) : (
          <div className="chips">
            {tokens.slice(0, shown).map((t, i) => (
              <span
                key={i}
                className={[
                  'chip',
                  isSpecial(t.text) ? 'special' : '',
                  isByte(t.text) ? 'byte' : '',
                  prefillLit && i < prefillTo ? 'lit' : '',
                  cached ? 'cached' : '',
                ].join(' ')}
                onMouseMove={(e) =>
                  setTip({
                    x: e.clientX,
                    y: e.clientY,
                    content: (
                      <div className="mono">
                        #{i} · id <b>{t.id}</b>
                        <div>{JSON.stringify(t.text)}</div>
                      </div>
                    ),
                  })
                }
                onMouseLeave={() => setTip(null)}
                onClick={() => setUi({ inspector: { kind: 'prompt-token', index: i } })}
              >
                {showToken(t.text)}
              </span>
            ))}
            {tokens.length > maxChips && revealed >= maxChips && (
              <span className="chip more">+{tokens.length - maxChips} more</span>
            )}
            {tokens.length === 0 && (
              <span style={{ color: 'var(--muted)' }}>
                Prompt tokens appear here after tokenization.
              </span>
            )}
          </div>
        )}
      </div>
      <Tip tip={tip} />
    </div>
  );
}

const TokenSpan = memo(function TokenSpan({
  t,
  deep,
  onHover,
  onLeave,
}: {
  t: ResponseToken;
  /** has deep-tier data (logit lens, layer graph) */
  deep?: boolean;
  onHover: (e: React.MouseEvent, t: ResponseToken) => void;
  onLeave: () => void;
}) {
  if (t.isEos) return <span className="eos">EOS</span>;
  const low = t.entropy !== undefined && t.entropy > 1.4;
  return (
    <span
      className={`tok fresh ${low ? 'lowconf' : ''} ${deep ? 'has-deep' : ''}`}
      style={low ? { textDecoration: 'underline dotted #f59e0b' } : undefined}
      onMouseMove={(e) => onHover(e, t)}
      onMouseLeave={onLeave}
      onClick={() => {
        session.pauseToInspect();
        setUi({ inspector: { kind: 'token', step: t.step } });
      }}
    >
      {t.text}
    </span>
  );
});

export function ResponseStream() {
  const tokens = useUi((s) => s.vm.responseTokens);
  const deepList = useUi((s) => s.vm.deepSteps);
  const deepSteps = useMemo(() => new Set(deepList), [deepList]);
  const phase = useUi((s) => s.vm.phase);
  const detail = useUi((s) => s.settings.detail);
  const uninstrumented = useUi((s) => s.uninstrumented);
  const responseText = useUi((s) => s.vm.responseText);
  const [tip, setTip] = useState<TipState | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 80) el.scrollTop = el.scrollHeight;
  }, [tokens]);

  const onHover = (e: React.MouseEvent, t: ResponseToken) =>
    setTip({
      x: e.clientX,
      y: e.clientY,
      content: (
        <div>
          <div className="mono">
            step {t.step} · id {t.id ?? '?'}{' '}
            {t.entropy !== undefined && `· H=${t.entropy.toFixed(2)}`}
          </div>
          {t.topk && (
            <table>
              <tbody>
                {t.topk.slice(0, 6).map((c) => (
                  <tr key={c.id} style={{ color: c.id === t.id ? 'var(--decode)' : undefined }}>
                    <td className="mono">{showToken(c.text)}</td>
                    <td style={{ width: 70 }}>
                      <div
                        className="bar"
                        style={{
                          width: `${c.p * 100}%`,
                          background: c.id === t.id ? 'var(--decode)' : undefined,
                        }}
                      />
                    </td>
                    <td className="mono" style={{ textAlign: 'right' }}>
                      {(c.p * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      ),
    });

  return (
    <div className="panel" data-testid="response-panel">
      <h3>
        Response <span className="mono">{tokens.length || ''}</span>
        <span className="spacer" />
        {uninstrumented && <span className="badge">token stream only</span>}
      </h3>
      <div className="panel-body" style={{ display: 'flex', flexDirection: 'column' }}>
        <div
          className="stream"
          ref={ref}
          data-testid="response"
          data-text={responseText}
          style={{ flex: 1 }}
        >
          {tokens.map((t, i) => (
            <TokenSpan
              key={i}
              t={t}
              deep={detail === 'deep' && deepSteps.has(t.step)}
              onHover={onHover}
              onLeave={() => setTip(null)}
            />
          ))}
          {phase === 'decoding' && <span className="caret" />}
        </div>
        {atLeast(detail, 'full') && (
          <div style={{ height: 64, borderTop: '1px solid var(--line)' }}>
            <StreamBytes />
          </div>
        )}
      </div>
      <Tip tip={tip} />
    </div>
  );
}

/** Exact concatenation of decode.token texts (used by e2e). */
export function responseTextOf(tokens: ResponseToken[]) {
  return tokens.map((t) => t.text).join('');
}
