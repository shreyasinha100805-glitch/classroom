const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const htmlFiles = ['index.html', 'result.html', 'login.html', 'signup.html'];
const assetExtensions = /\.(?:css|js|html|png|jpe?g|webp|gif|svg|ico|woff2?|ttf)(?:[?#].*)?$/i;
const retryPolicy = require('../js/retry_policy.js');

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
assert.equal(read('js/script.js'), read('static/js/script.js'), 'Flask static JavaScript is stale; run node build.js');
assert.equal(read('js/retry_policy.js'), read('public/js/retry_policy.js'), 'public retry policy is stale; run node build.js');
assert.equal(read('js/retry_policy.js'), read('static/js/retry_policy.js'), 'Flask retry policy is stale; run node build.js');
assert.equal(read('css/style.css'), read('public/css/style.css'), 'public CSS is stale; run node build.js');
assert.equal(read('css/style.css'), read('static/css/style.css'), 'Flask static CSS is stale; run node build.js');
assert.match(read('public/css/style.css'), /@media\s*\(/, 'responsive CSS is missing from public output');

const frontendScript = read('js/script.js');
assert.match(frontendScript, /function logRateLimitResponse\(/, '429 responses must emit correlation diagnostics');
assert.match(frontendScript, /function diagnosticFetch\(/, 'frontend fetches must check rate-limit responses');
assert.match(frontendScript, /X-ClassMind-Request-ID/, 'frontend diagnostics must read the Flask correlation ID');
assert.match(frontendScript, /Retry-After/, 'frontend diagnostics must preserve retry timing evidence');
const diagnosticsStart = frontendScript.indexOf('function diagnosticRoute(');
const diagnosticsEnd = frontendScript.indexOf("document.addEventListener('DOMContentLoaded'", diagnosticsStart);
const diagnosticsCode = frontendScript.slice(diagnosticsStart, diagnosticsEnd);
assert.doesNotMatch(diagnosticsCode, /Authorization|Cookie|document\.cookie|response\.text|response\.json/i, '429 diagnostics must not record credentials, cookies, or response bodies');
assert.match(frontendScript, /Text Capture Needs Review/, 'unverified OCR must use the review state');
assert.match(frontendScript, /STAGE 6\.0 NOTES WITHHELD/, 'unverified OCR must not be labeled as generated notes');
assert.match(frontendScript, /notesVerified \? `<div class="note-actions-group">/, 'copy and export actions must be conditional on verified notes');
assert.match(frontendScript, /notesVerified \? `<div class="note-points-box">/, 'unverified notes must not render a notes body');
assert.match(frontendScript, /const retakeInstructions =/, 'unverified OCR must show useful retake guidance');
assert.match(frontendScript, /id="btn-copy-ocr"/, 'raw OCR inspection must remain available');
assert.match(frontendScript, /res\.headers\.get\('Retry-After'\)/, '429 handling must read Retry-After');
assert.match(frontendScript, /err\.retryable = res\.status === 429 \|\| res\.status >= 500/, 'temporary failures must be marked retryable');
assert.match(frontendScript, /shouldRetryPoll\(\s*err\.status,\s*consecutivePollErrors\s*\)/, 'poll retries must stop at a bounded attempt count');
assert.match(frontendScript, /withJobPollLock\(jobId, async \(\) => \{[\s\S]*?\}, navigator\.locks\)/, 'same-origin tabs must coordinate polling for the same job');
assert.match(frontendScript, /computeRetryDelayMs\(\s*consecutivePollErrors,\s*err\.retryAfterMs\s*\)/, 'poll backoff must respect the server retry delay');
assert.match(frontendScript, /pollIntervalMs\(\s*document\.visibilityState === 'visible'\s*\)/, 'polling must slow down for hidden tabs');

const retryNow = Date.UTC(2026, 0, 1, 0, 0, 0);
assert.equal(retryPolicy.parseRetryAfterMs('3', retryNow), 3000);
assert.equal(retryPolicy.parseRetryAfterMs(new Date(retryNow + 7000).toUTCString(), retryNow), 7000);
assert.equal(retryPolicy.parseRetryAfterMs('not-a-date', retryNow), null);
assert.equal(retryPolicy.pollIntervalMs(true, () => 0), 5000);
assert.equal(retryPolicy.pollIntervalMs(false, () => 0.999), 15999);
assert.equal(retryPolicy.computeRetryDelayMs(1, null, () => 0), 4000);
assert.equal(retryPolicy.computeRetryDelayMs(2, null, () => 0.5), 10000);
assert.equal(retryPolicy.computeRetryDelayMs(1, 30000, () => 0), 30000);
const simulated429RetryAfterMs = retryPolicy.parseRetryAfterMs('9', retryNow);
assert.equal(retryPolicy.shouldRetryPoll(429, 1), true);
assert.equal(retryPolicy.computeRetryDelayMs(1, simulated429RetryAfterMs, () => 0), 9000);
assert.equal(retryPolicy.shouldRetryPoll(429, 5), true);
assert.equal(retryPolicy.shouldRetryPoll(429, 6), false);
assert.equal(retryPolicy.shouldRetryPoll(503, 1), true);
assert.equal(retryPolicy.shouldRetryPoll(400, 1), false);
const jobCreateStart = frontendScript.indexOf("accepted = await requestAnalysisJson('/api/analyze/jobs'");
const jobAcceptedMarker = frontendScript.indexOf('jobId = accepted.job_id', jobCreateStart);
assert.ok(jobCreateStart >= 0 && jobAcceptedMarker > jobCreateStart, 'async job creation code must remain identifiable');
assert.doesNotMatch(frontendScript.slice(jobCreateStart, jobAcceptedMarker), /shouldRetryPoll|computeRetryDelayMs/, 'job creation must not be automatically retried');

async function verifySameJobPollingIsSerialized() {
  const events = [];
  const names = [];
  let lockQueue = Promise.resolve();
  let activeTasks = 0;
  let peakActiveTasks = 0;
  const mockLocks = {
    request(name, options, task) {
      names.push(name);
      assert.deepEqual(options, { mode: 'exclusive' });
      const current = lockQueue.then(async () => {
        activeTasks += 1;
        peakActiveTasks = Math.max(peakActiveTasks, activeTasks);
        try {
          return await task({ name });
        } finally {
          activeTasks -= 1;
        }
      });
      lockQueue = current.catch(() => {});
      return current;
    },
  };

  await Promise.all([
    retryPolicy.withJobPollLock('job-123', async () => {
      events.push('tab-a-start');
      await new Promise(resolve => setTimeout(resolve, 5));
      events.push('tab-a-end');
    }, mockLocks),
    retryPolicy.withJobPollLock('job-123', async () => {
      events.push('tab-b-start');
      events.push('tab-b-end');
    }, mockLocks),
  ]);

  assert.deepEqual(names, ['classmind-analysis-poll:job-123', 'classmind-analysis-poll:job-123']);
  assert.equal(peakActiveTasks, 1, 'same-job tabs must not poll concurrently when Web Locks is available');
  assert.deepEqual(events, ['tab-a-start', 'tab-a-end', 'tab-b-start', 'tab-b-end']);
}

verifySameJobPollingIsSerialized().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

const css = read('public/css/style.css').replace(/\/\*[\s\S]*?\*\//g, '');
assert.equal((css.match(/{/g) || []).length, (css.match(/}/g) || []).length, 'public CSS has unbalanced blocks');

verifyNodeSyntax('js/script.js');
verifyNodeSyntax('public/js/script.js');
verifyNodeSyntax('js/retry_policy.js');

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
