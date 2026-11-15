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

function loadVercelConfig(environment) {
  const childEnv = { ...process.env };
  for (const [key, value] of Object.entries(environment)) {
    if (value === undefined) delete childEnv[key];
    else childEnv[key] = value;
  }
  const result = spawnSync(
    process.execPath,
    ['--input-type=module', '-e', 'import { config } from "./vercel.mjs"; process.stdout.write(JSON.stringify(config));'],
    { cwd: root, env: childEnv, encoding: 'utf8' }
  );
  return result;
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

const productionConfig = loadVercelConfig({
  VERCEL_ENV: 'production',
  CLASSMIND_PREVIEW_BACKEND_URL: undefined,
});
assert.equal(productionConfig.status, 0, productionConfig.stderr);
const production = JSON.parse(productionConfig.stdout);
const productionBackend = 'https://classroom-backend-cp6o.onrender.com';
assert.equal(production.rewrites[0].destination, `${productionBackend}/api/:path*`);

const previewBackend = 'https://classroom-backend-preview.onrender.com';
const previewConfig = loadVercelConfig({
  VERCEL_ENV: 'preview',
  CLASSMIND_PREVIEW_BACKEND_URL: previewBackend,
});
assert.equal(previewConfig.status, 0, previewConfig.stderr);
const preview = JSON.parse(previewConfig.stdout);
assert.deepEqual(
  preview.rewrites.slice(0, 4).map(({ source, destination }) => [source, destination]),
  [
    ['/api/:path*', `${previewBackend}/api/:path*`],
    ['/upload', `${previewBackend}/upload`],
    ['/uploads/:path*', `${previewBackend}/uploads/:path*`],
    ['/pipeline_output/:path*', `${previewBackend}/pipeline_output/:path*`],
  ],
  'Preview rewrites must use the preview origin and preserve request path suffixes'
);

const invalidPreview = loadVercelConfig({
  VERCEL_ENV: 'preview',
  CLASSMIND_PREVIEW_BACKEND_URL: productionBackend,
});
assert.notEqual(invalidPreview.status, 0, 'Preview must reject the Production backend origin');
assert.match(invalidPreview.stderr, /separate Render preview backend/i);

console.log(`Frontend build assets verified: ${htmlFiles.length} HTML pages, local references, responsive CSS, and mirrored JavaScript.`);
console.log('Vercel Production and Preview rewrites verified against their separate backend origins.');
