// Records a scripted run of every detail level as MP4 (maximized window,
// captured with ffmpeg gdigrab), plus one combined video. Usage (from apps/visualizer,
// after `pnpm build`):
//   npx tsx e2e/record-modes.ts [--out <dir>] [--modes basic,layers,full,deep]
//                               [--mock] [--model llama3.2] [--ollama <ollama.exe>]
// Real runs spawn their own traced Ollama (default C:\ollama-trace-deep\ollama.exe)
// on port 11439 with the app's hub on 4799, so a running Ollama/app is not touched.
import { _electron as electron, type ElectronApplication, type Page } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { APP_DIR } from './helpers';

type Mode = 'basic' | 'layers' | 'full' | 'deep';

const args = process.argv.slice(2);
const opt = (name: string, def: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
const OUT = resolve(opt('out', resolve(APP_DIR, '../../recordings')));
const MODES = opt('modes', 'basic,layers,full,deep').split(',') as Mode[];
const MOCK = args.includes('--mock');
const MODEL = opt('model', MOCK ? 'mock-llama:1b' : 'llama3.2');
const OLLAMA = opt('ollama', 'C:\\ollama-trace-deep\\ollama.exe');
const FFMPEG = opt('ffmpeg', 'ffmpeg');
const HUB = 4799;
const HTTP = 11439;
const TITLE = 'LLM Inference Visualizer (recording)';
const PROMPT = 'Why is the sky blue? Answer in two sentences.';

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// ---- ffmpeg window capture ---------------------------------------------------

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Captures a screen region (the window's content area). Capturing the window
 * by title gives black frames: Chromium draws it on the GPU.
 */
function startCapture(file: string, r: Rect): ChildProcess {
  const even = (n: number) => Math.floor(n / 2) * 2;
  const ff = spawn(
    FFMPEG,
    [
      '-y',
      '-loglevel',
      'error',
      '-f',
      'gdigrab',
      '-framerate',
      '30',
      '-draw_mouse',
      '1',
      '-offset_x',
      String(r.x),
      '-offset_y',
      String(r.y),
      '-video_size',
      `${even(r.width)}x${even(r.height)}`,
      '-i',
      'desktop',
      '-c:v',
      'libx264',
      '-preset',
      'veryfast',
      '-crf',
      '20',
      '-pix_fmt',
      'yuv420p',
      file,
    ],
    { stdio: ['pipe', 'inherit', 'inherit'] },
  );
  return ff;
}

function stopCapture(ff: ChildProcess): Promise<void> {
  return new Promise((done) => {
    ff.once('exit', () => done());
    ff.stdin?.write('q');
    ff.stdin?.end();
    setTimeout(() => ff.kill(), 15_000);
  });
}

// ---- app ---------------------------------------------------------------------

async function launchApp(): Promise<{ app: ElectronApplication; page: Page; area: Rect }> {
  const userData = mkdtempSync(join(tmpdir(), 'llmviz-rec-'));
  writeFileSync(
    join(userData, 'settings.json'),
    JSON.stringify({
      source: MOCK ? 'mock' : 'ollama',
      ollamaUrl: `http://127.0.0.1:${HTTP}`,
      detail: 'basic',
      stepGranularity: 'layer',
      autosave: false,
    }),
  );
  const app = await electron.launch({
    args: [join(APP_DIR, 'out/main/index.js')],
    cwd: APP_DIR,
    env: {
      ...process.env,
      TRACE_PORT: String(HUB),
      VISUALIZER_USER_DATA: userData,
      ELECTRON_RENDERER_URL: '',
    },
  });
  const page = await app.firstWindow();
  await page.waitForLoadState('domcontentloaded');
  await app.evaluate(({ BrowserWindow }, title) => {
    const w = BrowserWindow.getAllWindows()[0];
    w.on('page-title-updated', (e) => e.preventDefault());
    w.setTitle(title);
    w.maximize();
    w.setAlwaysOnTop(true, 'screen-saver');
    w.show();
    w.focus();
  }, TITLE);
  await sleep(1200);
  // content area in physical pixels (gdigrab works on the physical desktop)
  const area = await app.evaluate(({ BrowserWindow, screen }) => {
    const w = BrowserWindow.getAllWindows()[0];
    const b = w.getContentBounds();
    const sf = screen.getDisplayMatching(b).scaleFactor;
    return {
      x: Math.round(b.x * sf),
      y: Math.round(b.y * sf),
      width: Math.round(b.width * sf),
      height: Math.round(b.height * sf),
    };
  });
  return { app, page, area };
}

/** Large caption at the top of the window, so viewers know what they are seeing. */
async function caption(page: Page, text: string) {
  await page.evaluate((t) => {
    let el = document.getElementById('rec-caption');
    if (!el) {
      el = document.createElement('div');
      el.id = 'rec-caption';
      el.style.cssText =
        'position:fixed;left:50%;top:44px;transform:translateX(-50%);z-index:9999;' +
        'background:#121a33f0;border:1px solid #3a4b85;color:#e6e9f2;border-radius:10px;' +
        'padding:10px 18px;font:600 18px Segoe UI,sans-serif;pointer-events:none;' +
        'box-shadow:0 8px 30px #000a;transition:opacity .3s';
      document.body.appendChild(el);
    }
    el.textContent = t;
    el.style.opacity = t ? '1' : '0';
  }, text);
}

const phase = (page: Page) =>
  page.evaluate(() => (window as any).__viz.useUi.getState().vm.phase as string);

async function waitPhase(page: Page, want: string[], timeout = 180_000) {
  const t0 = Date.now();
  while (!want.includes(await phase(page))) {
    if (Date.now() - t0 > timeout) throw new Error(`timed out waiting for ${want.join('/')}`);
    await sleep(250);
  }
}

async function click(page: Page, name: string, times: number, gapMs: number) {
  for (let i = 0; i < times; i++) {
    if ((await phase(page)) === 'done') return;
    await page.getByRole('button', { name, exact: true }).click();
    await sleep(gapMs);
  }
}

async function selectModel(page: Page) {
  let models: string[] = [];
  for (let i = 0; i < 40 && !models.some((m) => m.startsWith(MODEL)); i++) {
    models = await page.evaluate(async () => {
      await (window as any).__viz.session.refreshModels();
      return (window as any).__viz.useUi.getState().models as string[];
    });
    if (!models.some((m) => m.startsWith(MODEL))) await sleep(1000);
  }
  const pick = models.find((m) => m.startsWith(MODEL));
  if (!pick) throw new Error(`model ${MODEL} not found (have: ${models.join(', ')})`);
  await page.selectOption('[data-testid=model]', pick);
}

async function setMode(page: Page, mode: Mode, step: boolean) {
  await page.click(`[data-testid=detail-${mode}]`);
  const box = page.locator('[data-testid=step-mode]');
  if (step) await box.check();
  else await box.uncheck();
  await page.keyboard.press('Escape'); // close any inspector
}

// ---- scenarios -----------------------------------------------------------------

const SCENARIOS: Record<Mode, (page: Page) => Promise<void>> = {
  async basic(page) {
    await caption(page, 'Basic: phases, tokens, top-k sampling, streamed text');
    await page.click('[data-testid=run]');
    await waitPhase(page, ['done']);
    await sleep(3000);
  },

  async layers(page) {
    await caption(page, 'Layers: adds the per-layer stack and KV cache');
    await page.click('[data-testid=run]');
    await waitPhase(page, ['done']);
    await sleep(3000);
  },

  async full(page) {
    await caption(
      page,
      'Full: forward pass, attention, residual stream, sampler chain (step mode)',
    );
    await page.click('[data-testid=run]');
    await waitPhase(page, ['prefill', 'decoding']);
    await sleep(2500);
    await caption(page, 'Step layer: one transformer layer per click');
    await click(page, 'Step layer', 8, 900);
    await caption(page, 'Step sampling stage: raw → top-k → top-p → temperature → softmax → draw');
    await click(page, 'Step sampling stage', 7, 1400);
    await caption(page, 'Step token: next word');
    await click(page, 'Step token', 3, 1800);
    await caption(page, 'Click a word to inspect how it was picked');
    await page.locator('[data-testid=response] .tok').nth(1).click();
    await sleep(3500);
    await page.keyboard.press('Escape');
    await caption(page, 'Continue: play the rest of the answer');
    await page.getByRole('button', { name: 'Continue', exact: true }).click();
    await waitPhase(page, ['done']);
    await sleep(3000);
  },

  async deep(page) {
    await caption(page, 'Deep: every operation inside a layer, logit lens, attention heads');
    await page.click('[data-testid=run]');
    await waitPhase(page, ['prefill', 'decoding']);
    await sleep(2500);
    await caption(page, 'Step op: one graph node at a time, data flowing along the arrows');
    await click(page, 'Step op', 14, 900);
    await caption(page, 'Step layer: the logit lens column fills in, layer by layer');
    await click(page, 'Step layer', 27, 450);
    await sleep(1500);
    await caption(page, 'Pin a layer: its graph and per-head attention');
    await page.locator('.deep-lens-row').nth(20).click();
    await sleep(3000);
    await page.locator('.deep-lens-row').nth(20).click();
    await caption(page, 'Step sampling stage: how the next word is drawn');
    await click(page, 'Step sampling stage', 7, 1200);
    await caption(page, 'Step token: next word (layers sweep quickly)');
    await click(page, 'Step token', 3, 2000);
    await caption(page, 'Click a word: its logit lens, graph, heads and sampling');
    await page.locator('[data-testid=response] .tok').nth(1).click();
    await sleep(4500);
    await page.keyboard.press('Escape');
    await caption(page, 'Continue: play the rest of the answer');
    await page.getByRole('button', { name: 'Continue', exact: true }).click();
    await waitPhase(page, ['done']);
    await sleep(3000);
  },
};

// ---- main ------------------------------------------------------------------------

(async () => {
  mkdirSync(OUT, { recursive: true });
  let ollama: ChildProcess | null = null;
  if (!MOCK) {
    ollama = spawn(OLLAMA, ['serve'], {
      env: {
        ...process.env,
        OLLAMA_HOST: `127.0.0.1:${HTTP}`,
        OLLAMA_TRACE_URL: `ws://127.0.0.1:${HUB}/ingest`,
      },
      stdio: 'ignore',
    });
  }
  const { app, page, area } = await launchApp();
  console.log('capture area', JSON.stringify(area));
  const clips: string[] = [];
  try {
    await selectModel(page);
    await page.fill('[data-testid=prompt]', PROMPT);
    if (!MOCK) {
      // warm-up so model loading does not dominate the first clip
      await setMode(page, 'basic', false);
      await page.click('[data-testid=run]');
      await waitPhase(page, ['done']);
    }
    for (const mode of MODES) {
      await setMode(page, mode, mode === 'full' || mode === 'deep');
      await caption(page, '');
      const file = join(OUT, `${clips.length + 1}-${mode}.mp4`);
      const ff = startCapture(file, area);
      await sleep(1200);
      const t0 = Date.now();
      try {
        await SCENARIOS[mode](page);
      } finally {
        await stopCapture(ff);
      }
      clips.push(file);
      console.log(`${mode}: ${file} (${Math.round((Date.now() - t0) / 1000)} s)`);
    }
  } finally {
    await app.close().catch(() => {});
    ollama?.kill();
  }
  if (clips.length > 1) {
    const list = join(OUT, 'clips.txt');
    writeFileSync(list, clips.map((c) => `file '${c.replace(/\\/g, '/')}'`).join('\n'));
    const all = join(OUT, 'all-modes.mp4');
    await new Promise<void>((done) =>
      spawn(
        FFMPEG,
        ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', all],
        {
          stdio: 'inherit',
        },
      ).once('exit', () => done()),
    );
    console.log(`all: ${all}`);
  }
})();
