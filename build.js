const fs = require('fs');
const path = require('path');

function copyRecursive(src, dest) {
  if (!fs.existsSync(src)) return;
  const stat = fs.statSync(src);
  if (stat.isDirectory()) {
    if (!fs.existsSync(dest)) fs.mkdirSync(dest, { recursive: true });
    for (const file of fs.readdirSync(src)) {
      copyRecursive(path.join(src, file), path.join(dest, file));
    }
  } else {
    const parent = path.dirname(dest);
    if (!fs.existsSync(parent)) fs.mkdirSync(parent, { recursive: true });
    fs.copyFileSync(src, dest);
  }
}

const htmlFiles = ['index.html', 'result.html', 'login.html', 'signup.html'];
const assetDirs = ['css', 'js', 'samples'];
const targetDirs = ['public', 'dist', 'build'];

console.log('--- Starting Classroom AI Frontend Build ---');

for (const target of targetDirs) {
  if (!fs.existsSync(target)) {
    fs.mkdirSync(target, { recursive: true });
  }

  // Copy HTML files
  for (const f of htmlFiles) {
    if (fs.existsSync(f)) {
      copyRecursive(f, path.join(target, f));
      console.log(`[Copied file] ${f} -> ${target}/${f}`);
    }
  }

  // Copy Asset directories
  for (const d of assetDirs) {
    if (fs.existsSync(d)) {
      copyRecursive(d, path.join(target, d));
      console.log(`[Copied directory] ${d} -> ${target}/${d}`);
    }
  }
}

console.log('--- Classroom AI Frontend Build Completed Successfully ---');
