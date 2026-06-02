import { jacobiEigSym } from '../linalg.js';
import { allocate } from './shared.js';

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
  const C = [];
  for (let i = 0; i < d; i++) C.push(new Array(d).fill(0));
  for (const r of centered) {
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += r[i] * r[j];
  }
  const denom = Math.max(1, N - 1);
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] /= denom;
  const { vectors } = jacobiEigSym(C);
  const out = allocate(N);
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < 3; k++) {
      let s = 0;
      const vk = vectors[k];
      for (let j = 0; j < d; j++) s += centered[i][j] * vk[j];
      out.X[i * 3 + k] = s;
    }
    out.t[i] = i / Math.max(1, N - 1);
  }
  return out;
}

export const CSV_UPLOAD = {
  id: 'csv',
  label: 'Upload CSV...',
  params: [],
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
