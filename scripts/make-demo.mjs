#!/usr/bin/env node
// Rebuilds the README demo from the latest recording (`pnpm record`):
//   docs/demo.gif      sped-up looping preview shown at the top of README.md
//   docs/all-modes.mp4 smaller copy of the full video the GIF links to
//
// Usage: pnpm demo [--in recordings/all-modes.mp4] [--speed 6] [--width 960]
//                  [--fps 10] [--ffmpeg path/to/ffmpeg]
import { spawnSync } from 'node:child_process';
import { existsSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : def;
};

const input = resolve(ROOT, opt('in', 'recordings/all-modes.mp4'));
const speed = Number(opt('speed', '6'));
const width = Number(opt('width', '960'));
const fps = Number(opt('fps', '10'));
const ffmpeg = opt('ffmpeg', 'ffmpeg');
const gif = resolve(ROOT, 'docs/demo.gif');
const mp4 = resolve(ROOT, 'docs/all-modes.mp4');

if (!existsSync(input)) {
  console.error(`No recording at ${input}. Run "pnpm record" first.`);
  process.exit(1);
}

function run(label, ffArgs) {
  console.log(label);
  const r = spawnSync(ffmpeg, ['-y', '-loglevel', 'error', ...ffArgs], { stdio: 'inherit' });
  if (r.error) {
    console.error(`Could not run ffmpeg (${r.error.message}). Put it on PATH or pass --ffmpeg.`);
    process.exit(1);
  }
  if (r.status !== 0) process.exit(r.status ?? 1);
}

// One palette for the whole clip keeps the file small; bayer dithering keeps text crisp.
run(`GIF preview (${speed}x, ${width} px, ${fps} fps)`, [
  '-i',
  input,
  '-vf',
  `setpts=PTS/${speed},fps=${fps},scale=${width}:-1:flags=lanczos,split[a][b];` +
    '[a]palettegen=max_colors=128:stats_mode=diff[p];' +
    '[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle',
  '-loop',
  '0',
  gif,
]);

run('Full video (1600 px, H.264)', [
  '-i',
  input,
  '-vf',
  'scale=1600:-2',
  '-c:v',
  'libx264',
  '-crf',
  '28',
  '-preset',
  'slow',
  '-pix_fmt',
  'yuv420p',
  '-movflags',
  '+faststart',
  '-an',
  mp4,
]);

const mb = (f) => (statSync(f).size / 1e6).toFixed(1) + ' MB';
console.log(`docs/demo.gif ${mb(gif)}, docs/all-modes.mp4 ${mb(mp4)}`);
