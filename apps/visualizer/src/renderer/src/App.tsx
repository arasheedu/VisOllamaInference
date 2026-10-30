import { useEffect } from 'react';
import { MotionConfig } from 'motion/react';
import { TopBar } from './components/TopBar';
import { PhasePipeline } from './components/PhasePipeline';
import { SamplingPanel } from './components/SamplingPanel';
import { SamplingFunnel } from './components/SamplingFunnel';
import { PromptTokens, ResponseStream } from './components/TokenPanels';
import { Timeline } from './components/Timeline';
import { DebugPanel, Inspector, SettingsModal, Tour, Warnings } from './components/Overlays';
import { LayerCanvas } from './viz/LayerCanvas';
import { ForwardPass } from './viz/ForwardPass';
import { MoePanel } from './components/MoePanel';
import { DeepView } from './components/DeepView';
import { atLeast } from 'trace-schema';
import { setUi, useUi } from './store';
import { useReducedMotion } from './theme';
import { dilationLabel } from './components/Timeline';

function CenterBadges() {
  const c = useUi((s) => s.vm.clock);
  const capabilities = useUi((s) => s.vm.capabilities);
  const detail = useUi((s) => s.settings.detail);
  const missingFull =
    detail !== 'basic' && capabilities.some((x) => x === 'no_layers' || x === 'no_ops');
  return (
    <div className="badges">
      {missingFull && (
        <span className="badge" title={capabilities.join(', ')}>
          runner has no per-layer data
        </span>
      )}
      {c.degraded && <span className="badge hot">fast-forwarding</span>}
      {c.catchingUp && <span className="badge hot">catch-up {c.speed.toFixed(1)}×</span>}
      {atLeast(detail, 'full') && !c.paused && (
        <span className="badge">slow motion {dilationLabel(c.dilation)}</span>
      )}
    </div>
  );
}

export function App() {
  const detail = useUi((s) => s.settings.detail);
  const inspector = useUi((s) => s.inspector);
  const deepTab = useUi((s) => s.deepTab);
  const reduced = useReducedMotion();
  const kvLabel = useUi((s) => (s.vm.kv.total ? `KV ${s.vm.kv.used}/${s.vm.kv.total}` : ''));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'D' || e.key === 'd')) {
        e.preventDefault();
        setUi((s) => ({ debugOpen: !s.debugOpen }));
      }
      if (e.key === 'Escape') setUi({ inspector: null, settingsOpen: false });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <MotionConfig reducedMotion={reduced ? 'always' : 'never'}>
      <div className={`app ${reduced ? 'reduced-motion' : ''}`}>
        <TopBar />
        <Warnings />
        <div className={`main ${inspector ? 'with-inspector' : ''}`}>
          <div className="panel">
            <h3>Pipeline</h3>
            <PhasePipeline />
          </div>
          <div className="panel" data-testid="layers-panel">
            <h3>
              {detail === 'deep' && deepTab === 'layer'
                ? 'Inside a layer · logit lens · attention heads'
                : atLeast(detail, 'full')
                  ? 'Forward pass · attention · residual · KV cache'
                  : 'Layer stack · KV cache'}
              {detail === 'deep' && (
                <span className="seg deep-tabs" style={{ marginLeft: 8 }}>
                  <button
                    className={deepTab === 'layer' ? 'on' : ''}
                    onClick={() => setUi({ deepTab: 'layer' })}
                    data-testid="deep-tab-layer"
                  >
                    Layer zoom
                  </button>
                  <button
                    className={deepTab === 'forward' ? 'on' : ''}
                    onClick={() => setUi({ deepTab: 'forward' })}
                    data-testid="deep-tab-forward"
                  >
                    Forward pass
                  </button>
                </span>
              )}
              <span className="spacer" />
              <span className="mono" style={{ textTransform: 'none' }}>
                {kvLabel}
              </span>
            </h3>
            <div className="panel-body">
              {detail === 'deep' && deepTab === 'layer' ? (
                <DeepView />
              ) : atLeast(detail, 'full') ? (
                <ForwardPass />
              ) : (
                <LayerCanvas />
              )}
              <CenterBadges />
              {atLeast(detail, 'full') && !(detail === 'deep' && deepTab === 'layer') && (
                <MoePanel />
              )}
            </div>
          </div>
          <div className="panel">
            <h3>{atLeast(detail, 'full') ? 'Sampling chain' : 'Sampling · top-k'}</h3>
            <div className="panel-body">
              {atLeast(detail, 'full') ? <SamplingFunnel /> : <SamplingPanel />}
            </div>
          </div>
          {inspector && <Inspector />}
        </div>
        <div className="tokens">
          <PromptTokens />
          <ResponseStream />
        </div>
        <Timeline />
        <DebugPanel />
        <SettingsModal />
        <Tour />
      </div>
    </MotionConfig>
  );
}
