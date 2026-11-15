const PRODUCTION_BACKEND_ORIGIN = 'https://classroom-backend-cp6o.onrender.com';

function getBackendOrigin() {
  if (process.env.VERCEL_ENV === 'production') {
    return PRODUCTION_BACKEND_ORIGIN;
  }

  const configured = process.env.CLASSMIND_PREVIEW_BACKEND_URL;
  if (!configured) {
    throw new Error(
      'Set CLASSMIND_PREVIEW_BACKEND_URL for Vercel Preview and Development deployments.'
    );
  }

  let backendUrl;
  try {
    backendUrl = new URL(configured);
  } catch {
    throw new Error('CLASSMIND_PREVIEW_BACKEND_URL must be a valid HTTPS origin.');
  }

  if (
    backendUrl.protocol !== 'https:' ||
    backendUrl.username ||
    backendUrl.password ||
    backendUrl.port ||
    backendUrl.pathname !== '/' ||
    backendUrl.search ||
    backendUrl.hash ||
    !backendUrl.hostname.endsWith('.onrender.com')
  ) {
    throw new Error(
      'CLASSMIND_PREVIEW_BACKEND_URL must be a plain HTTPS Render service origin.'
    );
  }

  if (backendUrl.origin === PRODUCTION_BACKEND_ORIGIN) {
    throw new Error('Vercel Preview must use its separate Render preview backend.');
  }

  return backendUrl.origin;
}

const backend = getBackendOrigin();

export const config = {
  $schema: 'https://openapi.vercel.sh/vercel.json',
  framework: null,
  buildCommand: 'node build.js',
  outputDirectory: 'public',
  cleanUrls: true,
  trailingSlash: false,
  rewrites: [
    { source: '/api/(.*)', destination: `${backend}/api/$1` },
    { source: '/upload', destination: `${backend}/upload` },
    { source: '/uploads/(.*)', destination: `${backend}/uploads/$1` },
    { source: '/pipeline_output/(.*)', destination: `${backend}/pipeline_output/$1` },
    { source: '/results', destination: '/result.html' },
    { source: '/login', destination: '/login.html' },
    { source: '/signup', destination: '/signup.html' },
  ],
};
