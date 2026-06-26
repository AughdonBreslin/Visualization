// Runs in a Web Worker -- no DOM, no D3, no window.
// Receives a grid specification and dataset, computes KDE posteriors and GDA
// boundary lines, and posts results back. A token field lets the main thread
// discard stale responses when a newer render has already been dispatched.

function safeLog(x) { return x > 0 ? Math.log(x) : -Infinity; }

function logAddExp(a, b) {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

function logSumExp(vs) {
  let acc = -Infinity;
  for (const v of vs) acc = logAddExp(acc, v);
  return acc;
}

function kdePdf(points, bandwidth) {
  const inv2s2 = 1 / (2 * bandwidth * bandwidth);
  const norm = 1 / (2 * Math.PI * bandwidth * bandwidth);
  return function(x) {
    let sum = 0;
    for (const p of points) {
      const dx = x[0] - p[0]; const dy = x[1] - p[1];
      sum += Math.exp(-(dx * dx + dy * dy) * inv2s2);
    }
    return norm * (sum / points.length);
  };
}

function kdeLogPdf(points, bandwidth) {
  const n = points.length;
  if (!n) return () => -Infinity;
  const inv2s2 = 1 / (2 * bandwidth * bandwidth);
  const logNorm = -Math.log(2 * Math.PI) - 2 * Math.log(bandwidth);
  const logInvN = -Math.log(n);
  return function(x) {
    let logSum = -Infinity;
    for (const p of points) {
      const dx = x[0] - p[0]; const dy = x[1] - p[1];
      logSum = logAddExp(logSum, -(dx * dx + dy * dy) * inv2s2);
    }
    return logNorm + logInvN + logSum;
  };
}

function det2(m) { return m[0][0] * m[1][1] - m[0][1] * m[1][0]; }
function inv2(m) {
  const d = det2(m);
  if (Math.abs(d) < 1e-12) return null;
  return [[m[1][1] / d, -m[0][1] / d], [-m[1][0] / d, m[0][0] / d]];
}

function mvnLogPdf(x, mu, sigma) {
  const eps = 1e-6;
  const s = [[sigma[0][0] + eps, sigma[0][1]], [sigma[1][0], sigma[1][1] + eps]];
  const inv = inv2(s);
  const d = det2(s);
  if (!inv || d <= 0) return -Infinity;
  const dx = [x[0] - mu[0], x[1] - mu[1]];
  const q = dx[0] * (inv[0][0] * dx[0] + inv[0][1] * dx[1]) + dx[1] * (inv[1][0] * dx[0] + inv[1][1] * dx[1]);
  return -Math.log(2 * Math.PI) - 0.5 * Math.log(d) - 0.5 * q;
}

function mvnPdf(x, mu, sigma) {
  const eps = 1e-6;
  const s = [[sigma[0][0] + eps, sigma[0][1]], [sigma[1][0], sigma[1][1] + eps]];
  const inv = inv2(s);
  const d = det2(s);
  if (!inv || d <= 0) return 0;
  const dx = [x[0] - mu[0], x[1] - mu[1]];
  const q = dx[0] * (inv[0][0] * dx[0] + inv[0][1] * dx[1]) + dx[1] * (inv[1][0] * dx[0] + inv[1][1] * dx[1]);
  return Math.exp(-Math.log(2 * Math.PI) - 0.5 * Math.log(d) - 0.5 * q);
}

function gdaTopIdx(pt, gdaClasses, gdaParams, useLog) {
  let posts;
  if (useLog) {
    const logNums = gdaClasses.map(c => mvnLogPdf(pt, gdaParams[c].mu, gdaParams[c].sigma) + safeLog(gdaParams[c].prior));
    const logD = logSumExp(logNums);
    posts = logD === -Infinity
      ? logNums.map(() => 1 / Math.max(1, logNums.length))
      : logNums.map(ln => Math.exp(ln - logD));
  } else {
    const nums = gdaClasses.map(c => mvnPdf(pt, gdaParams[c].mu, gdaParams[c].sigma) * gdaParams[c].prior);
    const denom = nums.reduce((a, b) => a + b, 0);
    posts = denom === 0 ? nums.map(() => 1 / nums.length) : nums.map(v => v / denom);
  }
  let best = 0;
  for (let i = 1; i < posts.length; i++) if (posts[i] > posts[best]) best = i;
  return best;
}

self.onmessage = function({ data: msg }) {
  const {
    token, showKDE, showGDA,
    xmin, xmax, ymin, ymax, nx, ny,
    classPoints, classes, priors, bandwidth, useLogSpace,
    gdaFitted, gdaClasses, gdaParams
  } = msg;

  const xs = Array.from({ length: nx }, (_, i) => xmin + (xmax - xmin) * i / (nx - 1));
  const ys = Array.from({ length: ny }, (_, j) => ymin + (ymax - ymin) * j / (ny - 1));
  const grid = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) grid.push([xs[i], ys[j]]);

  let kdePostInfo = null;
  if (showKDE && classes.length > 0) {
    const pdfs = classPoints.map(arr => arr.length ? kdePdf(arr, bandwidth) : () => 0);
    const logPdfs = classPoints.map(arr => arr.length ? kdeLogPdf(arr, bandwidth) : () => -Infinity);
    kdePostInfo = grid.map(pt => {
      let posts;
      if (useLogSpace) {
        const logNums = classes.map((_, i) => logPdfs[i](pt) + safeLog(priors[i]));
        const logD = logSumExp(logNums);
        posts = logD === -Infinity
          ? logNums.map(() => 1 / Math.max(1, logNums.length))
          : logNums.map(ln => Math.exp(ln - logD));
      } else {
        const nums = classes.map((_, i) => pdfs[i](pt) * priors[i]);
        const denom = nums.reduce((a, b) => a + b, 0);
        posts = denom === 0 ? nums.map(() => 1 / Math.max(1, nums.length)) : nums.map(v => v / denom);
      }
      let maxIdx = 0;
      for (let i = 1; i < posts.length; i++) if (posts[i] > posts[maxIdx]) maxIdx = i;
      const sorted = posts.slice().sort((a, b) => b - a);
      return { x: pt[0], y: pt[1], maxIdx, maxVal: posts[maxIdx], isSoftBoundary: (sorted[0] - (sorted[1] || 0)) < 0.06 };
    });
  }

  let gdaBoundaryLines = null;
  if (showGDA && gdaFitted && gdaClasses.length > 0) {
    const topIdxGrid = grid.map(pt => gdaTopIdx(pt, gdaClasses, gdaParams, useLogSpace));
    const lines = [];
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const idx = j * nx + i;
        if (i < nx - 1 && topIdxGrid[idx] !== topIdxGrid[idx + 1]) {
          lines.push({ x1: grid[idx][0], y1: grid[idx][1], x2: grid[idx + 1][0], y2: grid[idx + 1][1] });
        }
        if (j < ny - 1 && topIdxGrid[idx] !== topIdxGrid[idx + nx]) {
          lines.push({ x1: grid[idx][0], y1: grid[idx][1], x2: grid[idx + nx][0], y2: grid[idx + nx][1] });
        }
      }
    }
    gdaBoundaryLines = lines;
  }

  self.postMessage({ token, kdePostInfo, gdaBoundaryLines });
};
