// Runs in a Web Worker -- no DOM, no D3, no window.
// Receives the sample, statistic key, bootstrap reps, seed, and confidence level.
// Returns sorted bootstrap stats, point estimate, SE, and CI bounds.

function mulberry32(seed) {
  let s = seed >>> 0;
  return function () {
    s += 0x6d2b79f5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function mean(xs) {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s / xs.length;
}

function median(xs) {
  const a = xs.slice().sort((a, b) => a - b);
  const mid = a.length >> 1;
  return a.length & 1 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}

function bootstrapStatistic(sample, statFn, B, rng) {
  const n = sample.length;
  const stats = new Array(B);
  const resample = new Array(n);
  for (let b = 0; b < B; b++) {
    for (let i = 0; i < n; i++) resample[i] = sample[Math.floor(rng() * n)];
    stats[b] = statFn(resample);
  }
  return stats;
}

function quantileSorted(sorted, p) {
  const n = sorted.length;
  if (!n) return NaN;
  if (p <= 0 || n < 2) return sorted[0];
  if (p >= 1) return sorted[n - 1];
  const i = (n - 1) * p;
  const lo = Math.floor(i);
  return sorted[lo] + (sorted[lo + 1] - sorted[lo]) * (i - lo);
}

self.onmessage = function ({ data: msg }) {
  const { token, sample, statKey, B, seed, level } = msg;
  const statFn = statKey === 'median' ? median : mean;
  const rng = mulberry32(seed + 1337);

  const thetaHat = statFn(sample);
  const stats = bootstrapStatistic(sample, statFn, B, rng);
  stats.sort((a, b) => a - b);

  const alpha = 1 - level;
  const ci = {
    lo: quantileSorted(stats, alpha / 2),
    hi: quantileSorted(stats, 1 - alpha / 2),
  };

  const se = Math.sqrt(mean(stats.map(s => (s - mean(stats)) ** 2)));

  self.postMessage({ token, stats, thetaHat, se, ci });
};
