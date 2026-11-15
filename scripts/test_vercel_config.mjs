import assert from 'node:assert/strict';

const productionBackend = 'https://classroom-backend-cp6o.onrender.com';
let caseId = 0;

async function loadConfig(environment, previewBackend) {
  process.env.VERCEL_ENV = environment;
  if (previewBackend === undefined) {
    delete process.env.CLASSMIND_PREVIEW_BACKEND_URL;
  } else {
    process.env.CLASSMIND_PREVIEW_BACKEND_URL = previewBackend;
  }

  const configUrl = new URL('../vercel.mjs', import.meta.url);
  configUrl.searchParams.set('test', String(++caseId));
  return (await import(configUrl.href)).config;
}

function verifyRewriteEntries(rewrites) {
  assert.ok(Array.isArray(rewrites), 'rewrites must be an array');

  for (const [index, rewrite] of rewrites.entries()) {
    assert.ok(
      rewrite && typeof rewrite === 'object' && !Array.isArray(rewrite),
      `rewrites[${index}] must be an object`,
    );
    assert.equal(typeof rewrite.source, 'string', `rewrites[${index}] must have a string source`);
    assert.ok(rewrite.source.trim(), `rewrites[${index}] must have a non-empty source`);
    assert.equal(
      typeof rewrite.destination,
      'string',
      `rewrites[${index}] must have a string destination`,
    );
    assert.ok(rewrite.destination.trim(), `rewrites[${index}] must have a non-empty destination`);
  }
}

function verifyConfig(config, backend, isPreview) {
  assert.ok(config && typeof config === 'object' && !Array.isArray(config));
  assert.equal(config.framework, null);
  assert.equal(config.buildCommand, 'node build.js');
  assert.equal(config.outputDirectory, 'public');
  assert.equal(config.cleanUrls, true);
  assert.equal(config.trailingSlash, false);
  verifyRewriteEntries(config.rewrites);
  assert.equal(config.rewrites.length, 7);

  const expectedDestinations = new Map([
    ['/api/:path*', `${backend}/api/:path*`],
    ['/upload', `${backend}/upload`],
    ['/uploads/:path*', `${backend}/uploads/:path*`],
    ['/pipeline_output/:path*', `${backend}/pipeline_output/:path*`],
    ['/results', '/result.html'],
    ['/login', '/login.html'],
    ['/signup', '/signup.html'],
  ]);
  for (const rewrite of config.rewrites) {
    assert.equal(rewrite.destination, expectedDestinations.get(rewrite.source), `unexpected rewrite for ${rewrite.source}`);
    if (/^https?:/.test(rewrite.destination)) {
      const destination = new URL(rewrite.destination);
      assert.equal(destination.protocol, 'https:');
      assert.ok(destination.hostname.endsWith('.onrender.com'));
      assert.ok(!destination.hostname.includes('localhost'));
      if (isPreview) assert.notEqual(destination.origin, productionBackend);
      else assert.equal(destination.origin, productionBackend);
    }
  }

  const responseHeaders = (config.headers || []).flatMap((rule) => rule.headers || []);
  assert.equal(
    responseHeaders.some((header) => header.key.toLowerCase() === 'access-control-allow-origin' && header.value === '*'),
    false,
    'Vercel must not add a wildcard CORS origin to credentialed backend responses',
  );
}

for (const invalidRewrites of [
  undefined,
  {},
  [undefined],
  [null],
  [{ source: '/api/:path*' }],
  [{ destination: 'https://backend.example.com/api/:path*' }],
  [{ source: '', destination: 'https://backend.example.com/api/:path*' }],
  [{ source: '/api/:path*', destination: '' }],
]) {
  assert.throws(() => verifyRewriteEntries(invalidRewrites));
}

const production = await loadConfig('production', 'not-a-production-setting');
verifyConfig(production, productionBackend, false);

// Test-only fixture; this URL is never used by a deployment.
const previewBackend = 'https://preview-test.onrender.com';
const preview = await loadConfig('preview', previewBackend);
verifyConfig(preview, previewBackend, true);

const development = await loadConfig('development', previewBackend);
verifyConfig(development, previewBackend, true);

await assert.rejects(
  () => loadConfig('preview', undefined),
  /Set CLASSMIND_PREVIEW_BACKEND_URL for Vercel Preview and Development deployments/,
);
await assert.rejects(
  () => loadConfig('preview', 'not-a-valid-url'),
  /must be a valid HTTPS origin/,
);
await assert.rejects(
  () => loadConfig('preview', 'http://preview-test.onrender.com'),
  /must be a plain HTTPS Render service origin/,
);
await assert.rejects(
  () => loadConfig('preview', 'https://preview-test.onrender.com/api'),
  /must be a plain HTTPS Render service origin/,
);
await assert.rejects(
  () => loadConfig('preview', 'https://user:secret@preview-test.onrender.com'),
  /must be a plain HTTPS Render service origin/,
);
await assert.rejects(
  () => loadConfig('preview', 'https://preview-test.onrender.com:8443'),
  /must be a plain HTTPS Render service origin/,
);
await assert.rejects(
  () => loadConfig('preview', 'https://preview-test.onrender.com/?env=preview'),
  /must be a plain HTTPS Render service origin/,
);
await assert.rejects(
  () => loadConfig('preview', 'https://preview-test.onrender.com/#preview'),
  /must be a plain HTTPS Render service origin/,
);
await assert.rejects(
  () => loadConfig('preview', 'https://preview-test.example.com'),
  /must be a plain HTTPS Render service origin/,
);
await assert.rejects(
  () => loadConfig('preview', productionBackend),
  /must use its separate Render preview backend/,
);

console.log('Vercel production, preview, development, missing-variable, and invalid-origin configurations validated.');
