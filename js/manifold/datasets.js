import { mulberry32, gaussian } from './rng.js';

function allocate(N) {
  return { X: new Float64Array(N * 3), t: new Float64Array(N), N };
}

function addNoise(X, noise, rand) {
  if (noise <= 0) return;
  for (let i = 0; i < X.length; i++) X[i] += noise * gaussian(rand);
}

export const SWISS_ROLL = {
  id: 'swiss_roll',
  label: 'Swiss roll',
  generate({ samples, noise, seed }) {
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const u = 1.5 * Math.PI * (1 + 2 * rand());
      const v = 21 * rand();
      out.X[i * 3 + 0] = u * Math.cos(u);
      out.X[i * 3 + 1] = v;
      out.X[i * 3 + 2] = u * Math.sin(u);
      out.t[i] = u;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const S_CURVE = {
  id: 's_curve',
  label: 'S-curve',
  generate({ samples, noise, seed }) {
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const t = (rand() - 0.5) * 3 * Math.PI;
      const sgn = t >= 0 ? 1 : -1;
      out.X[i * 3 + 0] = Math.sin(t);
      out.X[i * 3 + 1] = 4 * (rand() - 0.5);
      out.X[i * 3 + 2] = sgn * (Math.cos(t) - 1);
      out.t[i] = t;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

function projectToThreeViaPCA(rows) {
  const N = rows.length;
  if (N === 0) return { X: new Float64Array(0), t: new Float64Array(0), N: 0, empty: true };
  const d = rows[0].length;
  const mean = new Float64Array(d);
  for (const r of rows) for (let j = 0; j < d; j++) mean[j] += r[j];
  for (let j = 0; j < d; j++) mean[j] /= N;
  const centered = rows.map(r => r.map((x, j) => x - mean[j]));
  if (d <= 3) {
    const out = allocate(N);
    for (let i = 0; i < N; i++) {
      out.X[i * 3 + 0] = centered[i][0] || 0;
      out.X[i * 3 + 1] = centered[i][1] || 0;
      out.X[i * 3 + 2] = centered[i][2] || 0;
      out.t[i] = i / Math.max(1, N - 1);
    }
    return out;
  }
  if (typeof window === 'undefined' || !window.numeric) {
    throw new Error('CSV projection requires numeric.js (browser only)');
  }
  const C = [];
  for (let i = 0; i < d; i++) C.push(new Float64Array(d));
  for (const r of centered) {
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += r[i] * r[j];
  }
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] /= Math.max(1, N - 1);
  const Cm = C.map(row => Array.from(row));
  const eig = window.numeric.eig(Cm);
  const lam = eig.lambda.x.map((v, i) => ({ v: Math.abs(v), i }));
  lam.sort((a, b) => b.v - a.v);
  const V = eig.E.x;
  const out = allocate(N);
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < 3; k++) {
      let s = 0;
      const col = lam[k].i;
      for (let j = 0; j < d; j++) s += centered[i][j] * V[j][col];
      out.X[i * 3 + k] = s;
    }
    out.t[i] = i / Math.max(1, N - 1);
  }
  return out;
}

export const CSV_UPLOAD = {
  id: 'csv',
  label: 'Upload CSV...',
  generate({ csvRows }) {
    if (!csvRows || csvRows.length === 0) {
      return { X: new Float64Array(0), t: new Float64Array(0), N: 0, empty: true };
    }
    return projectToThreeViaPCA(csvRows);
  },
};

export function parseCSV(text) {
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
  if (lines.length === 0) return [];
  const first = lines[0].split(',').map(s => s.trim());
  const headerIsNumeric = first.every(c => c.length > 0 && Number.isFinite(Number(c)));
  const dataLines = headerIsNumeric ? lines : lines.slice(1);
  const rows = [];
  for (const line of dataLines) {
    const parts = line.split(',').map(s => Number(s.trim()));
    if (parts.length < 2) continue;
    if (!parts.every(v => Number.isFinite(v))) continue;
    rows.push(parts);
  }
  if (rows.length === 0) return [];
  const widths = rows.map(r => r.length);
  const mode = widths.sort((a, b) => widths.filter(x => x === a).length - widths.filter(x => x === b).length).pop();
  return rows.filter(r => r.length === mode);
}

export const DATASETS = [SWISS_ROLL, S_CURVE, CSV_UPLOAD];
export const DATASETS_BY_ID = Object.fromEntries(DATASETS.map(d => [d.id, d]));
