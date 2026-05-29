import { gaussian } from '../rng.js';

export function allocate(N) {
  return { X: new Float64Array(N * 3), t: new Float64Array(N), N };
}

export function addNoise(X, noise, rand) {
  if (noise <= 0) return;
  for (let i = 0; i < X.length; i++) X[i] += noise * gaussian(rand);
}

export const CLUSTER_PALETTE = [
  '#ff6b6b', '#4ecdc4', '#ffd93d', '#6a8cff',
  '#c77dff', '#ff9f43', '#54e36b', '#ff7eb6',
];

export function fibonacciSpherePoints(count, radius) {
  const pts = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const y = count === 1 ? 0 : 1 - (i / (count - 1)) * 2;
    const ring = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = golden * i;
    pts.push([radius * Math.cos(theta) * ring, radius * y, radius * Math.sin(theta) * ring]);
  }
  return pts;
}
