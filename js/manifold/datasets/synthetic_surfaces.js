import { mulberry32 } from '../rng.js';
import { allocate, addNoise } from './shared.js';

function sampleSphere(rand) {
  const u = rand();
  const v = rand();
  const theta = 2 * Math.PI * v;
  const cosPhi = 1 - 2 * u;
  const phi = Math.acos(Math.max(-1, Math.min(1, cosPhi)));
  return { theta, phi };
}

function writeSpherePoint(out, i, theta, phi) {
  const s = Math.sin(phi);
  out.X[i * 3 + 0] = s * Math.cos(theta);
  out.X[i * 3 + 1] = s * Math.sin(theta);
  out.X[i * 3 + 2] = Math.cos(phi);
}

function hilbertPoint3D(h, order) {
  const n = 3;
  const X = [0, 0, 0];
  for (let p = 0; p < n * order; p++) {
    const bit = (h >> (n * order - 1 - p)) & 1;
    const dim = p % n;
    const lvl = order - 1 - ((p / n) | 0);
    if (bit) X[dim] |= (1 << lvl);
  }
  const N = 1 << order;
  const tg = X[n - 1] >> 1;
  for (let i = n - 1; i > 0; i--) X[i] ^= X[i - 1];
  X[0] ^= tg;
  for (let Q = 2; Q !== N; Q <<= 1) {
    const P = Q - 1;
    for (let i = n - 1; i >= 0; i--) {
      if (X[i] & Q) X[0] ^= P;
      else { const tmp = (X[0] ^ X[i]) & P; X[0] ^= tmp; X[i] ^= tmp; }
    }
  }
  return X;
}


export const TWIN_PEAKS = {
  id: 'twin_peaks',
  label: 'Twin peaks',
  params: [],
  generate({ samples, noise, seed }) {
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const x = 2 * rand() - 1;
      const y = 2 * rand() - 1;
      const z = Math.exp(-(((x - 0.5) ** 2) + ((y - 0.5) ** 2)) / 0.3)
              + Math.exp(-(((x + 0.5) ** 2) + ((y + 0.5) ** 2)) / 0.3);
      out.X[i * 3 + 0] = x;
      out.X[i * 3 + 1] = y;
      out.X[i * 3 + 2] = z;
      out.t[i] = x;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const SADDLE = {
  id: 'saddle',
  label: 'Saddle',
  params: [],
  generate({ samples, noise, seed }) {
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const x = 2 * rand() - 1;
      const y = 2 * rand() - 1;
      out.X[i * 3 + 0] = x;
      out.X[i * 3 + 1] = y;
      out.X[i * 3 + 2] = x * x - y * y;
      out.t[i] = x;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const CYLINDER = {
  id: 'cylinder',
  label: 'Cylinder',
  params: [{ name: 'height', type: 'float', default: 2, min: 0.5, max: 5 }],
  generate({ samples, noise, seed, height }) {
    const H = height || 2;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const theta = 2 * Math.PI * rand();
      const h = H * rand();
      out.X[i * 3 + 0] = Math.cos(theta);
      out.X[i * 3 + 1] = Math.sin(theta);
      out.X[i * 3 + 2] = h;
      out.t[i] = theta;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const SEVERED_SPHERE = {
  id: 'severed_sphere',
  label: 'Severed sphere',
  params: [{ name: 'cap', type: 'float', default: 0.35, min: 0, max: 0.9 }],
  generate({ samples, noise, seed, cap }) {
    const C = cap === undefined ? 0.35 : cap;
    const minPhi = C * Math.PI;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    let i = 0;
    while (i < samples) {
      const { theta, phi } = sampleSphere(rand);
      if (phi < minPhi) continue;
      writeSpherePoint(out, i, theta, phi);
      out.t[i] = theta;
      i++;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const HILBERT = {
  id: 'hilbert',
  label: 'Hilbert curve',
  params: [{ name: 'order', type: 'int', default: 4, min: 2, max: 5 }],
  generate({ samples, noise, seed, order }) {
    const ord = order || 4;
    const L = 1 << (3 * ord);
    const grid = (1 << ord) - 1;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const h = samples === 1 ? 0 : Math.round(i * (L - 1) / (samples - 1));
      const p = hilbertPoint3D(h, ord);
      out.X[i * 3 + 0] = (p[0] / grid) * 2 - 1;
      out.X[i * 3 + 1] = (p[1] / grid) * 2 - 1;
      out.X[i * 3 + 2] = (p[2] / grid) * 2 - 1;
      out.t[i] = h / (L - 1);
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const FULL_SPHERE = {
  id: 'full_sphere',
  label: 'Full sphere',
  params: [],
  generate({ samples, noise, seed }) {
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const { theta, phi } = sampleSphere(rand);
      writeSpherePoint(out, i, theta, phi);
      out.t[i] = theta;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};
