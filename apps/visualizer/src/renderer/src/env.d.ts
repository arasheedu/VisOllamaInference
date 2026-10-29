/// <reference types="vite/client" />
import type { VisualizerApi } from '../../preload';

declare global {
  interface Window {
    visualizer?: VisualizerApi;
  }
}
