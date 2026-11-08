import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const directory = mkdtempSync(join(tmpdir(), 'autograd-consumer-'));
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';

try {
  const packed = JSON.parse(execFileSync(npm, ['pack', '--json', '--pack-destination', directory], { encoding: 'utf8' }));
  writeFileSync(join(directory, 'package.json'), JSON.stringify({ private: true }));
  execFileSync(npm, ['install', '--ignore-scripts', '--no-audit', '--no-fund', '--offline', '--package-lock=false', join(directory, packed[0].filename)], { cwd: directory, stdio: 'inherit' });

  const assertions = `
const x = val(2);
const y = x.mul(x);
y.backward();
assert.equal(y.data, 4);
assert.equal(x.grad, 4);
assert.equal(typeof MLP, 'function');
assert.equal(typeof mse, 'function');
assert.equal(typeof sgd, 'function');
`;
  writeFileSync(join(directory, 'consumer.mjs'), "import assert from 'node:assert/strict';\nimport { val, MLP, mse, sgd } from 'autograd-ts';\n" + assertions);
  writeFileSync(join(directory, 'consumer.cjs'), "const assert = require('node:assert/strict');\nconst { val, MLP, mse, sgd } = require('autograd-ts');\n" + assertions);
  for (const file of ['consumer.mjs', 'consumer.cjs']) {
    execFileSync(process.execPath, [file], { cwd: directory, stdio: 'inherit' });
  }

  const typed = "import { val, Value, MLP, mse, sgd, type Activation } from 'autograd-ts';\nconst activation: Activation = 'tanh';\nconst model = new MLP(2, [1], { hiddenActivation: activation });\nconst value: Value = val(2);\nconst loss: Value = mse(model.forward([value.data, 1]), [0]);\nloss.backward();\nsgd(model.parameters(), 0.01);\n";
  for (const file of ['consumer.mts', 'consumer.cts']) writeFileSync(join(directory, file), typed);
  execFileSync(process.execPath, [resolve('node_modules/typescript/bin/tsc'), '--strict', '--noEmit', '--target', 'ES2020', '--module', 'NodeNext', '--moduleResolution', 'NodeNext', 'consumer.mts', 'consumer.cts'], { cwd: directory, stdio: 'inherit' });
  console.log('Packed package passes ESM, CommonJS, and TypeScript consumer checks.');
} finally {
  rmSync(directory, { recursive: true, force: true });
}
