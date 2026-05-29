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

export const PUNCTURED_SPHERE = {
  id: 'punctured_sphere',
  label: 'Punctured sphere',
  params: [{ name: 'holeRadius', type: 'float', default: 0.4, min: 0, max: 1.5 }],
  generate({ samples, noise, seed, holeRadius }) {
    const HR = holeRadius === undefined ? 0.4 : holeRadius;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    let i = 0;
    while (i < samples) {
      const { theta, phi } = sampleSphere(rand);
      if (phi < HR) continue;
      writeSpherePoint(out, i, theta, phi);
      out.t[i] = theta;
      i++;
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
