import { mkdirSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';

mkdirSync('artifacts', { recursive: true });

const sha = process.env.GITHUB_SHA ?? execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
const browser = execFileSync('npx', ['playwright', '--version'], { encoding: 'utf8' }).trim();
const summary = [
  '# Validation Summary',
  '',
  `- Commit SHA: ${sha}`,
  `- Browser tooling: ${browser}`,
  '- Tested models: mlp_baseline, cnn_fast, cnn_strong',
  '- Accuracy threshold: default model >= 0.97',
  '- Fuzz seeds: 20260601',
  '- Screenshot paths: artifacts/screenshots/*.png',
  '- Test commands: npm run check, npm run build, npm run validate:models, npm run smoke:models, npm run test, npm run test:fuzz, npm run test:screenshots'
].join('\n');

writeFileSync('artifacts/validation-summary.md', `${summary}\n`);
