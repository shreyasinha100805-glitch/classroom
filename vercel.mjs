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

function createRewrite(source, destination) {
  if (typeof source !== 'string' || source.trim() === '') {
    throw new Error('Every Vercel rewrite must have a non-empty source.');
  }
  if (typeof destination !== 'string' || destination.trim() === '') {
    throw new Error(`Vercel rewrite for ${source} must have a non-empty destination.`);
  }

  return { source, destination };
}

export const config = {
  framework: null,
  buildCommand: 'node build.js',
  outputDirectory: 'public',
  cleanUrls: true,
  trailingSlash: false,
  rewrites: [
    createRewrite('/api/:path*', `${backend}/api/:path*`),
    createRewrite('/upload', `${backend}/upload`),
    createRewrite('/uploads/:path*', `${backend}/uploads/:path*`),
    createRewrite('/pipeline_output/:path*', `${backend}/pipeline_output/:path*`),
    createRewrite('/results', '/result.html'),
    createRewrite('/login', '/login.html'),
    createRewrite('/signup', '/signup.html'),
  ],
};
