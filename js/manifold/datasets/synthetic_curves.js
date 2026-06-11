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
  params: [
    { name: 'turns', type: 'int', default: 3, min: 1, max: 8,
      label: 'Turns',
      desc: 'Number of full turns of the helix. More turns make a longer, tighter coil.' },
    { name: 'width', type: 'float', default: 0.4, min: 0, max: 1.2,
      label: 'Ribbon width',
      desc: 'Width of the helical ribbon across the coil. 0 leaves a 1D string; a positive width sweeps it into a 2D band that Isomap can unroll into a flat strip.' },
  ],
  generate({ samples, noise, seed, turns, width }) {
    const T = turns || 3;
    const W = width == null ? 0.4 : width;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const u = 2 * Math.PI * T * rand();
      // The horizontal radial direction (cos u, sin u, 0) is perpendicular to the
      // helix tangent, so offsetting along it sweeps a clean ribbon (a spiral ramp).
      const w = (rand() - 0.5) * W;
      out.X[i * 3 + 0] = Math.cos(u) * (1 + w);
      out.X[i * 3 + 1] = Math.sin(u) * (1 + w);
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
  params: [
    { name: 'width', type: 'float', default: 0.5, min: 0, max: 1.5,
      label: 'Ribbon width',
      desc: 'Width of the band swept along the knot. 0 leaves a 1D string; a positive width makes a 2D band. Keep it modest so the neighbor graph does not bridge between nearby strands of the knot.' },
  ],
  generate({ samples, noise, seed, width }) {
    const W = width == null ? 0.5 : width;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const u = 2 * Math.PI * rand();
      const cx = Math.sin(u) + 2 * Math.sin(2 * u);
      const cy = Math.cos(u) - 2 * Math.cos(2 * u);
      const cz = -Math.sin(3 * u);
      // Offset along the Frenet binormal B = normalize(C' x C''). This knot has
      // nonzero curvature everywhere, so B varies smoothly and the band is clean.
      const t1 = Math.cos(u) + 4 * Math.cos(2 * u);
      const t2 = -Math.sin(u) + 4 * Math.sin(2 * u);
      const t3 = -3 * Math.cos(3 * u);
      const a1 = -Math.sin(u) - 8 * Math.sin(2 * u);
      const a2 = -Math.cos(u) + 8 * Math.cos(2 * u);
      const a3 = 9 * Math.sin(3 * u);
      let bx = t2 * a3 - t3 * a2;
      let by = t3 * a1 - t1 * a3;
      let bz = t1 * a2 - t2 * a1;
      const bn = Math.hypot(bx, by, bz) || 1;
      bx /= bn; by /= bn; bz /= bn;
      const w = (rand() - 0.5) * W;
      out.X[i * 3 + 0] = cx + w * bx;
      out.X[i * 3 + 1] = cy + w * by;
      out.X[i * 3 + 2] = cz + w * bz;
      out.t[i] = u;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const TOROIDAL_HELIX = {
  id: 'toroidal_helix',
  label: 'Toroidal helix',
  params: [
    { name: 'q', type: 'int', default: 7, min: 2, max: 15,
      label: 'Winding (q)',
      desc: 'How many times the helix winds around the torus tube per loop around the ring. Higher q coils more tightly.' },
    { name: 'width', type: 'float', default: 0.4, min: 0, max: 1.0,
      label: 'Ribbon width',
      desc: 'Width of the band across the helix, swept around the tube so it stays on the torus surface. 0 leaves a 1D string; a positive width makes a 2D band Isomap can unroll.' },
  ],
  generate({ samples, noise, seed, q, width }) {
    const Q = q || 7;
    const R = 2, r = 0.7;
    const W = width == null ? 0.4 : width;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const u = 2 * Math.PI * rand();
      // Offset across the helix along the tube's angular direction (a band on the
      // torus surface). Width is geometric, converted to a tube angle by / r.
      const v = Q * u + ((rand() - 0.5) * W) / r;
      const ring = R + r * Math.cos(v);
      out.X[i * 3 + 0] = ring * Math.cos(u);
      out.X[i * 3 + 1] = ring * Math.sin(u);
      out.X[i * 3 + 2] = r * Math.sin(v);
      out.t[i] = u;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const SPIRAL_DISK = {
  id: 'spiral_disk',
  label: 'Spiral disk',
  params: [
    { name: 'turns', type: 'int', default: 3, min: 1, max: 6,
      label: 'Turns',
      desc: 'Number of turns of the spiral arm. More turns wind it tighter toward the center.' },
    { name: 'width', type: 'float', default: 0.35, min: 0, max: 0.9,
      label: 'Ribbon width',
      desc: 'Width of the band across the spiral arm, as a fraction of the local radius so it stays proportional as the arm grows. 0 leaves a 1D string; a positive width makes a 2D band Isomap can unroll.' },
  ],
  generate({ samples, noise, seed, turns, width }) {
    const T = turns || 3;
    const W = width == null ? 0.35 : width;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    const thetaMax = 2 * Math.PI * T;
    const b = Math.log(40) / thetaMax;
    const r0 = 0.12;
    for (let i = 0; i < samples; i++) {
      const theta = thetaMax * rand();
      const r = r0 * Math.exp(b * theta);
      // In-plane unit normal to the spiral arm; offset proportional to r so the
      // band stays proportionally wide as the arm grows (avoids center crowding).
      const tx = b * r * Math.cos(theta) - r * Math.sin(theta);
      const ty = b * r * Math.sin(theta) + r * Math.cos(theta);
      const tn = Math.hypot(tx, ty) || 1;
      const nx = -ty / tn, ny = tx / tn;
      const w = (rand() - 0.5) * W * r;
      out.X[i * 3 + 0] = r * Math.cos(theta) + w * nx;
      out.X[i * 3 + 1] = r * Math.sin(theta) + w * ny;
      out.X[i * 3 + 2] = 0.2 * r * r;
      out.t[i] = theta;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};
