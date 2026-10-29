import type { DetailLevel } from 'trace-schema';
import {
  DEFAULT_CLOCK_CONFIG,
  DEFAULT_MINIMUMS,
  type ClockConfig,
  type StageMinimums,
} from './engine/clock';

export type Source = 'ollama' | 'mock';
export type StepGranularity = 'token' | 'layer' | 'sample';

export interface AppSettings {
  port: number;
  source: Source;
  ollamaUrl: string;
  detail: DetailLevel;
  minimums: StageMinimums;
  maxSpeed: number;
  lagThresholdMs: number;
  fullDetailLagThresholdMs: number;
  /** runner gating granularity requested in step mode (player can step finer from its buffer) */
  stepGranularity: StepGranularity;
  /** in-app mock: per-head attention */
  attnHeads: number;
  /** in-app mock: tokens/sec override (0 = model default) */
  mockTokPerSec: number;
  reducedMotion: 'system' | 'on' | 'off';
  autosave: boolean;
  maxPromptChips: number;
}

export const DEFAULT_SETTINGS: AppSettings = {
  port: 4711,
  source: 'mock',
  ollamaUrl: 'http://127.0.0.1:11434',
  detail: 'layers',
  minimums: DEFAULT_MINIMUMS,
  maxSpeed: DEFAULT_CLOCK_CONFIG.maxSpeed,
  lagThresholdMs: DEFAULT_CLOCK_CONFIG.lagThresholdMs,
  fullDetailLagThresholdMs: DEFAULT_CLOCK_CONFIG.fullDetailLagThresholdMs,
  stepGranularity: 'token',
  attnHeads: 0,
  mockTokPerSec: 0,
  reducedMotion: 'system',
  autosave: true,
  maxPromptChips: 600,
};

export function clockConfigFrom(s: AppSettings): ClockConfig {
  return {
    ...DEFAULT_CLOCK_CONFIG,
    minimums: { ...DEFAULT_MINIMUMS, ...s.minimums },
    maxSpeed: s.maxSpeed,
    lagThresholdMs: s.lagThresholdMs,
    fullDetailLagThresholdMs: s.fullDetailLagThresholdMs,
  };
}

export function mergeSettings(raw: Record<string, unknown> | null | undefined): AppSettings {
  const r = (raw ?? {}) as Partial<AppSettings>;
  return {
    ...DEFAULT_SETTINGS,
    ...r,
    minimums: { ...DEFAULT_MINIMUMS, ...(r.minimums ?? {}) },
  };
}
