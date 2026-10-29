import { contextBridge, ipcRenderer } from 'electron';

export interface HubStatus {
  listening: boolean;
  port: number;
  producers: number;
  consumers: number;
  eventsIn: number;
  lastEventAt: number | null;
  error: string | null;
}

const api = {
  /** Trace hub connection status (the only trace-related IPC). */
  getStatus: (): Promise<HubStatus> => ipcRenderer.invoke('trace:status'),
  onStatus: (cb: (s: HubStatus) => void): (() => void) => {
    const h = (_e: unknown, s: HubStatus) => cb(s);
    ipcRenderer.on('trace:status', h);
    return () => ipcRenderer.removeListener('trace:status', h);
  },
  getSettings: (): Promise<Record<string, unknown>> => ipcRenderer.invoke('settings:get'),
  setSettings: (s: Record<string, unknown>): Promise<Record<string, unknown>> =>
    ipcRenderer.invoke('settings:set', s),
  saveTrace: (name: string, ndjson: string, response: string): Promise<string | null> =>
    ipcRenderer.invoke('file:save-trace', name, ndjson, response),
  autosaveTrace: (name: string, ndjson: string, response: string): Promise<string | null> =>
    ipcRenderer.invoke('file:autosave-trace', name, ndjson, response),
  openTrace: (path?: string): Promise<{ path: string; text: string } | null> =>
    ipcRenderer.invoke('file:open-trace', path),
};

export type VisualizerApi = typeof api;

contextBridge.exposeInMainWorld('visualizer', api);
