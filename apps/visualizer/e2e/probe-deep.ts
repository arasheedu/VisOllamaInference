// Ad-hoc: Deep detail + step mode screenshots (mock). Usage: npx tsx e2e/probe-deep.ts <outDir>
import { launch } from './helpers';

const out = process.argv[2] ?? '.';
(async () => {
  const { app, page } = await launch({
    port: 4797,
    settings: { source: 'mock', detail: 'deep', autosave: false, stepGranularity: 'layer' },
  });
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  try {
    await page.waitForTimeout(500);
    await page.selectOption('[data-testid=model]', 'mock-llama:1b');
    await page.fill('[data-testid=prompt]', 'Why is the sky blue?');
    await page.check('[data-testid=step-mode]');
    await page.click('[data-testid=run]');
    await page.screenshot({ path: `${out}/deep-start.png` });
    await page.waitForFunction(
      () => ['prefill', 'decoding'].includes((window as any).__viz.useUi.getState().vm.phase),
      null,
      { timeout: 30_000 },
    );
    await page.waitForTimeout(800);
    for (let i = 0; i < 9; i++) {
      await page.getByRole('button', { name: 'Step op' }).click();
      await page.waitForTimeout(250);
    }
    await page.screenshot({ path: `${out}/deep-ops.png` });
    for (let i = 0; i < 16; i++) {
      await page.getByRole('button', { name: 'Step op' }).click();
      await page.waitForTimeout(120);
    }
    for (let i = 0; i < 14; i++) {
      await page.getByRole('button', { name: 'Step layer' }).click();
      await page.waitForTimeout(250);
    }
    await page.waitForTimeout(600);
    await page.screenshot({ path: `${out}/deep-layer-end.png` });
    // pin layer 5 from the lens column, hover a heatmap cell
    await page.locator('.deep-lens-row').nth(5).click();
    await page.waitForTimeout(300);
    const hm = page.locator('[data-testid=attn-heads] .deep-heads-canvas');
    const box = await hm.boundingBox();
    if (box) await page.mouse.move(box.x + box.width * 0.02, box.y + box.height * 0.3);
    await page.waitForTimeout(300);
    await page.screenshot({ path: `${out}/deep-pinned.png` });
    const st = await page.evaluate(() => {
      const s = (window as any).__viz.useUi.getState();
      const sd = s.vm.stepDetail;
      return {
        step: sd?.step,
        nodes: sd?.nodes.length,
        lens: sd?.lens.length,
        heads: sd?.heads.length,
        pinned: s.deepLayer,
      };
    });
    console.log('STATE', JSON.stringify(st));
    await page.getByRole('button', { name: 'Continue' }).click();
    await page.waitForSelector('[data-testid=pipeline][data-phase=done]', { timeout: 120_000 });
    await page.screenshot({ path: `${out}/deep-done.png` });
    console.log('errors', errors.slice(0, 10));
  } finally {
    await app.close();
  }
})();
