import { atLeast } from 'trace-schema';
import { session } from '../session';
import { useUi } from '../store';

export function StepControls() {
  const detail = useUi((s) => s.settings.detail);
  const stepping = useUi((s) => s.vm.clock.stepping);
  const hasRun = useUi((s) => s.vm.req !== null);
  return (
    <div className="step-controls">
      <button
        disabled={!hasRun}
        onClick={() => session.step('token')}
        title="Play until the next token is emitted"
      >
        Step token
      </button>
      {detail === 'deep' && (
        <button
          disabled={!hasRun}
          onClick={() => session.step('node')}
          title="Play until the next operation inside the layer"
        >
          Step op
        </button>
      )}
      <button
        disabled={!hasRun || detail === 'basic'}
        onClick={() => session.step('layer')}
        title="Play until the next layer completes"
      >
        Step layer
      </button>
      <button
        disabled={!hasRun || !atLeast(detail, 'full')}
        onClick={() => session.step('sample')}
        title="Play until the next sampler stage"
      >
        Step sampling stage
      </button>
      <button
        disabled={!hasRun}
        onClick={() => session.continue()}
        title="Leave step mode and play"
      >
        Continue
      </button>
      {stepping && <span className="badge hot">stepping: {stepping}</span>}
    </div>
  );
}
