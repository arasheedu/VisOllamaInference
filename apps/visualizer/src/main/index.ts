import { app, BrowserWindow, dialog, ipcMain, shell } from 'electron';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { TraceHub, type HubStatus } from './hub';

const DEFAULT_PORT = Number(process.env.TRACE_PORT) || 4711;

// isolated profile for tests / multiple instances
if (process.env.VISUALIZER_USER_DATA) app.setPath('userData', process.env.VISUALIZER_USER_DATA);
// opt-in Chrome DevTools Protocol port, for attaching profilers / test drivers to a dev session
if (process.env.VISUALIZER_CDP_PORT)
  app.commandLine.appendSwitch('remote-debugging-port', process.env.VISUALIZER_CDP_PORT);

interface MainSettings {
  port: number;
  [k: string]: unknown;
}

const settingsPath = () => join(app.getPath('userData'), 'settings.json');

function loadSettings(): MainSettings {
  try {
    if (existsSync(settingsPath())) {
      const s = JSON.parse(readFileSync(settingsPath(), 'utf8'));
      return {
        port: DEFAULT_PORT,
        ...s,
        ...(process.env.TRACE_PORT ? { port: DEFAULT_PORT } : {}),
      };
    }
  } catch {
    /* fall through to defaults */
  }
  return { port: DEFAULT_PORT };
}

let settings: MainSettings;
let hub: TraceHub;
const windows = new Set<BrowserWindow>();

function broadcastStatus(s: HubStatus) {
  for (const w of windows) if (!w.isDestroyed()) w.webContents.send('trace:status', s);
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1600,
    height: 900,
    minWidth: 1100,
    minHeight: 680,
    backgroundColor: '#0b0d12',
    title: 'LLM Inference Visualizer',
    autoHideMenuBar: true,
    webPreferences: {
      preload: join(import.meta.dirname, '../preload/index.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  windows.add(win);
  win.on('closed', () => windows.delete(win));
  win.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url);
    return { action: 'deny' };
  });
  if (process.env.ELECTRON_RENDERER_URL) void win.loadURL(process.env.ELECTRON_RENDERER_URL);
  else void win.loadFile(join(import.meta.dirname, '../renderer/index.html'));
  return win;
}

app.whenReady().then(async () => {
  settings = loadSettings();
  hub = new TraceHub(settings.port, broadcastStatus);
  await hub.start();

  // The only trace-related IPC: connection status (push + pull).
  ipcMain.handle('trace:status', () => hub.status);

  ipcMain.handle('settings:get', () => settings);
  ipcMain.handle('settings:set', async (_e, next: Partial<MainSettings>) => {
    const portChanged = next.port !== undefined && next.port !== settings.port;
    settings = { ...settings, ...next };
    await mkdir(app.getPath('userData'), { recursive: true });
    await writeFile(settingsPath(), JSON.stringify(settings, null, 2));
    if (portChanged) await hub.restart(settings.port);
    return settings;
  });

  ipcMain.handle('file:save-trace', async (_e, name: string, ndjson: string, response: string) => {
    const dir = join(app.getPath('documents'), 'LLM Inference Visualizer', 'sessions');
    await mkdir(dir, { recursive: true });
    const r = await dialog.showSaveDialog({
      defaultPath: join(dir, name),
      filters: [{ name: 'Trace', extensions: ['ndjson'] }],
    });
    if (r.canceled || !r.filePath) return null;
    await writeFile(r.filePath, ndjson);
    await writeFile(r.filePath.replace(/\.ndjson$/, '') + '.response.txt', response);
    return r.filePath;
  });

  ipcMain.handle(
    'file:autosave-trace',
    async (_e, name: string, ndjson: string, response: string) => {
      const dir = join(app.getPath('userData'), 'sessions');
      await mkdir(dir, { recursive: true });
      const p = join(dir, name);
      await writeFile(p, ndjson);
      await writeFile(p.replace(/\.ndjson$/, '') + '.response.txt', response);
      return p;
    },
  );

  ipcMain.handle('file:open-trace', async (_e, path?: string) => {
    let file = path;
    if (!file) {
      const r = await dialog.showOpenDialog({
        properties: ['openFile'],
        filters: [{ name: 'Trace', extensions: ['ndjson', 'jsonl'] }],
      });
      if (r.canceled || !r.filePaths[0]) return null;
      file = r.filePaths[0];
    }
    return { path: file, text: await readFile(file, 'utf8') };
  });

  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', async () => {
  await hub?.stop();
  if (process.platform !== 'darwin') app.quit();
});
