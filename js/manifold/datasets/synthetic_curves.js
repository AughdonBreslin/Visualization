import { mulberry32 } from '../rng.js';
import { allocate, addNoise } from './shared.js';

export const SWISS_ROLL = {
  id: 'swiss_roll',
  label: 'Swiss roll',
  params: [],
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
  params: [],
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

export const HELIX = {
  id: 'helix',
  label: 'Helix',
  params: [{ name: 'turns', type: 'int', default: 3, min: 1, max: 8,
    label: 'Turns',
    desc: 'Number of full turns of the helix. More turns make a longer, tighter coil.' }],
  generate({ samples, noise, seed, turns }) {
    const T = turns || 3;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const u = 2 * Math.PI * T * rand();
      out.X[i * 3 + 0] = Math.cos(u);
      out.X[i * 3 + 1] = Math.sin(u);
      out.X[i * 3 + 2] = u / (2 * Math.PI);
      out.t[i] = u;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const TREFOIL_KNOT = {
  id: 'trefoil_knot',
  label: 'Trefoil knot',
  params: [],
  generate({ samples, noise, seed }) {
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const u = 2 * Math.PI * rand();
      out.X[i * 3 + 0] = Math.sin(u) + 2 * Math.sin(2 * u);
      out.X[i * 3 + 1] = Math.cos(u) - 2 * Math.cos(2 * u);
      out.X[i * 3 + 2] = -Math.sin(3 * u);
      out.t[i] = u;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const TOROIDAL_HELIX = {
  id: 'toroidal_helix',
  label: 'Toroidal helix',
  params: [{ name: 'q', type: 'int', default: 7, min: 2, max: 15,
    label: 'Winding (q)',
    desc: 'How many times the helix winds around the torus tube per loop around the ring. Higher q coils more tightly.' }],
  generate({ samples, noise, seed, q }) {
    const Q = q || 7;
    const R = 2, r = 0.7;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const u = 2 * Math.PI * rand();
      const ring = R + r * Math.cos(Q * u);
      out.X[i * 3 + 0] = ring * Math.cos(u);
      out.X[i * 3 + 1] = ring * Math.sin(u);
      out.X[i * 3 + 2] = r * Math.sin(Q * u);
      out.t[i] = u;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const SPIRAL_DISK = {
  id: 'spiral_disk',
  label: 'Spiral disk',
  params: [{ name: 'turns', type: 'int', default: 3, min: 1, max: 6,
    label: 'Turns',
    desc: 'Number of turns of the spiral arm. More turns wind it tighter toward the center.' }],
  generate({ samples, noise, seed, turns }) {
    const T = turns || 3;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    const thetaMax = 2 * Math.PI * T;
    const b = Math.log(40) / thetaMax;
    const r0 = 0.12;
    for (let i = 0; i < samples; i++) {
      const theta = thetaMax * rand();
      const r = r0 * Math.exp(b * theta);
      out.X[i * 3 + 0] = r * Math.cos(theta);
      out.X[i * 3 + 1] = r * Math.sin(theta);
      out.X[i * 3 + 2] = 0.2 * r * r;
      out.t[i] = theta;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};
