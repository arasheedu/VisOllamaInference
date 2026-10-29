import { expect, test } from '@playwright/test';
import { startMockOllama } from 'trace-mock';
import { launch } from './helpers';

const PORT = 4791;
const HTTP_PORT = 11501;

test('Run against mock Ollama (public API + trace socket): reaches Done, text matches byte-for-byte', async () => {
  const responses = new Map<string, string>();
  const mock = startMockOllama({
    port: HTTP_PORT,
    hubUrl: `ws://127.0.0.1:${PORT}/ingest`,
    tokPerSec: 200,
    log: () => {},
    onResponse: (req, text) => responses.set(req, text),
  });
  const { app, page } = await launch({
    port: PORT,
    settings: {
      source: 'ollama',
      ollamaUrl: `http://127.0.0.1:${HTTP_PORT}`,
      detail: 'layers',
      autosave: false,
    },
  });
  try {
    await expect(page.locator('[data-testid=model] option')).toHaveCount(3, { timeout: 15_000 });
    await page.selectOption('[data-testid=model]', 'mock-llama:1b');
    await page.click('[data-testid=run]');
    await expect(page.locator('[data-testid=pipeline]')).toHaveAttribute('data-phase', 'done', {
      timeout: 120_000,
    });
    await expect(page.locator('[data-testid=node-done]')).toHaveAttribute('data-state', 'done');
    expect(responses.size).toBe(1);
    const expected = [...responses.values()][0];
    await expect(page.locator('[data-testid=response]')).toHaveAttribute('data-text', expected);
    // no mismatch / uninstrumented warnings
    await expect(page.locator('[data-testid=warning-mismatch]')).toHaveCount(0);
    await expect(page.locator('[data-testid=warning-uninstrumented]')).toHaveCount(0);
  } finally {
    await app.close();
    await mock.close();
  }
});

test('Full detail + step mode: step one token through layers and sampling stages, inspector shows raw data', async () => {
  const { app, page } = await launch({
    port: PORT + 1,
    settings: { source: 'mock', detail: 'full', autosave: false },
  });
  try {
    await page.selectOption('[data-testid=model]', 'mock-llama:1b');
    await page.check('[data-testid=step-mode]');
    await page.click('[data-testid=run]');
    // runner holds before the first decode step; prefill completes on its own
    await expect(page.locator('[data-testid=pipeline]')).toHaveAttribute('data-phase', 'prefill', {
      timeout: 30_000,
    });

    // step one layer
    await page.getByRole('button', { name: 'Step layer' }).click();
    await expect(page.locator('[data-testid=pipeline]')).toHaveAttribute('data-phase', 'decoding', {
      timeout: 30_000,
    });
    await expect(page.getByText('stepping: layer')).toHaveCount(0, { timeout: 30_000 });

    // step sampling stages one at a time
    await page.getByRole('button', { name: 'Step sampling stage' }).click();
    await expect(page.locator('.stage.current')).toHaveCount(1, { timeout: 60_000 });
    const firstStage = await page.locator('.stage.current header').first().innerText();
    await page.getByRole('button', { name: 'Step sampling stage' }).click();
    await expect(page.locator('.stage.current header').first()).not.toHaveText(firstStage, {
      timeout: 30_000,
    });

    // finish the token
    await page.getByRole('button', { name: 'Step token' }).click();
    await expect(page.locator('[data-testid=response] .tok')).toHaveCount(1, { timeout: 60_000 });
    await page.waitForTimeout(300);
    await expect(page.locator('[data-testid=response] .tok')).toHaveCount(1);

    // inspector: click a sampling stage header
    await page.locator('.stage header').first().click();
    await expect(page.locator('[data-testid=inspector]')).toBeVisible();
    await expect(page.locator('[data-testid=inspector-raw]')).toContainText('decode.sample.stage');

    // continue: the rest of the answer runs to Done
    await page.getByRole('button', { name: 'Continue' }).click();
    await expect(page.locator('[data-testid=pipeline]')).toHaveAttribute('data-phase', 'done', {
      timeout: 120_000,
    });
  } finally {
    await app.close();
  }
});
