import { DETAIL_LEVELS, type DetailLevel } from 'trace-schema';
import { session } from '../session';
import { setUi, useUi } from '../store';

const SPEEDS = [0.1, 0.25, 0.5, 1, 2, 4, 8];

export function TopBar() {
  const settings = useUi((s) => s.settings);
  const models = useUi((s) => s.models);
  const model = useUi((s) => s.model);
  const prompt = useUi((s) => s.prompt);
  const running = useUi((s) => s.running);
  const stepMode = useUi((s) => s.stepMode);
  const hub = useUi((s) => s.hub);
  const socket = useUi((s) => s.socket);
  const paused = useUi((s) => s.vm.clock.paused);
  const live = useUi((s) => s.vm.clock.live);
  const hasRun = useUi((s) => s.vm.req !== null);
  const speedNow = useUi((s) => s.vm.clock.speed);

  const run = () => {
    if (running) session.stop();
    else void session.start(prompt, model);
  };

  const hubOk = hub?.listening && socket === 'open';
  const hubTitle = hub
    ? hub.error
      ? `Trace hub error: ${hub.error}`
      : `Trace hub ws://127.0.0.1:${hub.port} · producers ${hub.producers} · viewers ${hub.consumers} · events ${hub.eventsIn}`
    : 'No trace hub (running outside Electron?)';

  return (
    <div className="topbar">
      <div className="title">
        LLM Inference Visualizer{' '}
        <span
          title={hubTitle}
          className={`status-dot ${hubOk ? 'ok' : 'warn'}`}
          data-testid="hub-status"
        />
      </div>
      <div className="seg" title="Where runs go">
        {(['ollama', 'mock'] as const).map((s) => (
          <button
            key={s}
            className={settings.source === s ? 'on' : ''}
            onClick={() => void session.updateSettings({ source: s })}
            data-testid={`source-${s}`}
          >
            {s === 'ollama' ? 'Ollama' : 'Mock'}
          </button>
        ))}
      </div>
      <select
        value={model}
        onChange={(e) => setUi({ model: e.target.value })}
        data-testid="model"
        style={{ maxWidth: 200 }}
      >
        {models.length === 0 && <option value="">(no models)</option>}
        {models.map((m) => (
          <option key={m}>{m}</option>
        ))}
      </select>
      <input
        className="prompt"
        type="text"
        value={prompt}
        placeholder="Type a prompt…"
        onChange={(e) => setUi({ prompt: e.target.value })}
        onKeyDown={(e) => e.key === 'Enter' && !running && run()}
        data-testid="prompt"
      />
      <div className="seg" title="Detail level: which events are requested and shown">
        {DETAIL_LEVELS.map((d: DetailLevel) => (
          <button
            key={d}
            className={settings.detail === d ? 'on' : ''}
            onClick={() => {
              void session.updateSettings({ detail: d });
              session.active?.setDetail(d);
              // deep data is meant to be walked through one op at a time
              if (d === 'deep') setUi({ stepMode: true });
            }}
            data-testid={`detail-${d}`}
          >
            {d === 'basic' ? 'Basic' : d === 'layers' ? 'Layers' : d === 'full' ? 'Full' : 'Deep'}
          </button>
        ))}
      </div>
      <label
        title="Step mode: the runner waits for you between tokens"
        style={{ display: 'flex', gap: 4, alignItems: 'center' }}
      >
        <input
          type="checkbox"
          checked={stepMode}
          onChange={(e) => setUi({ stepMode: e.target.checked })}
          data-testid="step-mode"
        />
        Step
      </label>
      <button className="primary" onClick={run} disabled={!model} data-testid="run">
        {running ? 'Stop' : 'Run'}
      </button>
      <div className="group">
        <button title="Restart playback" onClick={() => session.restart()} disabled={!hasRun}>
          ⏮
        </button>
        <button
          title={paused ? 'Resume' : 'Pause'}
          onClick={() => (paused ? session.resume() : session.pause())}
          disabled={!hasRun}
          data-testid="pause"
        >
          {paused ? '▶' : '⏸'}
        </button>
        <button
          title="Jump to live / end"
          onClick={() => session.goLive()}
          disabled={!hasRun}
          className={live ? '' : 'active'}
        >
          ⏭
        </button>
        <select
          title={`Playback speed (now ${speedNow.toFixed(2)}×)`}
          value={String(session.active?.clock.userSpeed ?? 1)}
          onChange={(e) => session.setSpeed(Number(e.target.value))}
        >
          {SPEEDS.map((s) => (
            <option key={s} value={String(s)}>
              {s}×
            </option>
          ))}
        </select>
      </div>
      <div className="group">
        <button title="Open a recorded trace (.ndjson)" onClick={() => void session.openTrace()}>
          Open…
        </button>
        <button
          title="Save this run's trace"
          onClick={() => void session.saveTrace()}
          disabled={!hasRun}
        >
          Save
        </button>
        <button title="Guided tour" onClick={() => setUi({ tour: 0 })} data-testid="tour">
          Tour
        </button>
        <button title="Settings" onClick={() => setUi({ settingsOpen: true })}>
          ⚙
        </button>
      </div>
    </div>
  );
}
