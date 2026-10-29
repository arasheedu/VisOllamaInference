import { create } from 'zustand';
import type { DetailLevel } from 'trace-schema';
import { emptyViewModel } from './engine/stateMachine';
import type { ViewModel } from './engine/types';
import { DEFAULT_SETTINGS, type AppSettings } from './settings';
import type { MergePlan } from './tokenizer/merges';

export interface HubStatus {
  listening: boolean;
  port: number;
  producers: number;
  consumers: number;
  eventsIn: number;
  lastEventAt: number | null;
  error: string | null;
}

export type InspectorTarget =
  | { kind: 'token'; step: number }
  | { kind: 'prompt-token'; index: number }
  | { kind: 'layer'; step: number; layer: number }
  | { kind: 'op'; step: number; layer: number; op: string; slot: number }
  | { kind: 'stage'; step: number; stage: string };

export interface Warning {
  id: string;
  text: string;
  level: 'info' | 'warn' | 'error';
}

export interface UiState {
  vm: ViewModel;
  hub: HubStatus | null;
  socket: 'connecting' | 'open' | 'closed';
  settings: AppSettings;
  models: string[];
  model: string;
  prompt: string;
  stepMode: boolean;
  running: boolean;
  /** run is going through the public API only (no trace events) */
  uninstrumented: boolean;
  warnings: Warning[];
  debugOpen: boolean;
  settingsOpen: boolean;
  inspector: InspectorTarget | null;
  merge: MergePlan | null;
  tour: number | null;
  /** source of the visible run */
  viewing: 'live' | 'file' | 'external' | null;
  fileName: string | null;
  fps: number;
}

export const useUi = create<UiState>(() => ({
  vm: emptyViewModel(),
  hub: null,
  socket: 'connecting',
  settings: DEFAULT_SETTINGS,
  models: [],
  model: 'mock-llama:8b',
  prompt: 'Explain, step by step, how a large language model turns my prompt into an answer.',
  stepMode: false,
  running: false,
  uninstrumented: false,
  warnings: [],
  debugOpen: false,
  settingsOpen: false,
  inspector: null,
  merge: null,
  tour: null,
  viewing: null,
  fileName: null,
  fps: 0,
}));

export const setUi = useUi.setState;

export function addWarning(w: Warning) {
  setUi((s) => ({ warnings: [...s.warnings.filter((x) => x.id !== w.id), w] }));
}
export function clearWarning(id: string) {
  setUi((s) => ({ warnings: s.warnings.filter((x) => x.id !== id) }));
}

export const detailRankOf = (d: DetailLevel) => ({ basic: 0, layers: 1, full: 2 })[d];
