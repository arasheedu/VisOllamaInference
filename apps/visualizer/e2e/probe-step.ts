// Ad-hoc: Full detail + step mode screenshots. Usage: npx tsx e2e/probe-step.ts <outDir>
import { launch } from './helpers';

const out = process.argv[2] ?? '.';
(async () => {
  const { app, page } = await launch({
    port: 4795,
    settings: { source: 'mock', detail: 'full', autosave: false },
  });
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  await page.waitForTimeout(500);
  await page.selectOption('[data-testid=model]', 'mock-llama:8b');
  await page.fill('[data-testid=prompt]', 'Why is the sky blue? Explain the scattering of light.');
  await page.check('[data-testid=step-mode]');
  await page.click('[data-testid=run]');
  await page.waitForSelector('[data-testid=pipeline][data-phase=tokenizing]');
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${out}/step-merge.png` });
  await page.waitForSelector('[data-testid=pipeline][data-phase=prefill]');
  await page.getByRole('button', { name: 'Step layer' }).click();
  await page.waitForTimeout(1500);
  for (let i = 0; i < 5; i++) {
    await page.getByRole('button', { name: 'Step layer' }).click();
    await page.waitForTimeout(500);
  }
  await page.screenshot({ path: `${out}/step-layers.png` });
  for (let i = 0; i < 6; i++) {
    await page.getByRole('button', { name: 'Step sampling stage' }).click();
    await page.waitForTimeout(i === 0 ? 4000 : 900);
  }
  await page.screenshot({ path: `${out}/step-sampling.png` });
  await page.getByRole('button', { name: 'Step token' }).click();
  await page.waitForTimeout(1500);
  await page.locator('[data-testid=response] .tok').first().click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${out}/step-inspector.png` });
  console.log('errors', errors.slice(0, 10));
  await app.close();
})();
