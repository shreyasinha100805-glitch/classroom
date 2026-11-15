const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const htmlFiles = ['index.html', 'result.html', 'login.html', 'signup.html'];
const assetExtensions = /\.(?:css|js|html|png|jpe?g|webp|gif|svg|ico|woff2?|ttf)(?:[?#].*)?$/i;

function read(relativePath) {
  return fs.readFileSync(path.join(root, relativePath), 'utf8');
}

function verifyNodeSyntax(relativePath) {
  const result = spawnSync(process.execPath, ['--check', path.join(root, relativePath)], {
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, `${relativePath} failed node --check:\n${result.stderr}`);
}

for (const file of htmlFiles) {
  const source = read(file);
  const published = read(path.join('public', file));
  assert.equal(published, source, `public/${file} is stale; run node build.js`);
  assert.match(source, /name=["']viewport["'][^>]*width=device-width/i, `${file} needs a responsive viewport`);
  assert.match(source, /viewport-fit=cover/i, `${file} needs safe-area viewport support`);

  for (const [, reference] of source.matchAll(/\b(?:src|href)=["']([^"']+)["']/gi)) {
    if (!reference || /^(?:[a-z]+:|\/\/|#)/i.test(reference)) continue;
    if (!assetExtensions.test(reference)) continue;
    const localPath = decodeURIComponent(reference.split(/[?#]/, 1)[0]).replace(/^\/+/, '');
    const outputPath = path.join(root, 'public', localPath);
    assert.ok(fs.existsSync(outputPath), `${file} references missing public asset: ${reference}`);
  }

  for (const [, reference] of source.matchAll(/\bdata-sample-src=["']([^"']+)["']/gi)) {
    const localPath = decodeURIComponent(reference.split(/[?#]/, 1)[0]).replace(/^\/+/, '');
    assert.ok(fs.existsSync(path.join(root, 'public', localPath)), `${file} references missing sample: ${reference}`);
  }
}

assert.equal(read('js/script.js'), read('public/js/script.js'), 'public JavaScript is stale; run node build.js');
assert.equal(read('css/style.css'), read('public/css/style.css'), 'public CSS is stale; run node build.js');
assert.match(read('public/css/style.css'), /@media\s*\(/, 'responsive CSS is missing from public output');

const css = read('public/css/style.css').replace(/\/\*[\s\S]*?\*\//g, '');
assert.equal((css.match(/{/g) || []).length, (css.match(/}/g) || []).length, 'public CSS has unbalanced blocks');

verifyNodeSyntax('js/script.js');
verifyNodeSyntax('public/js/script.js');

console.log(`Frontend build assets verified: ${htmlFiles.length} HTML pages, local references, responsive CSS, and mirrored JavaScript.`);
