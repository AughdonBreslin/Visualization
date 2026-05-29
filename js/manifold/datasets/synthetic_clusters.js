import { mulberry32, gaussian } from '../rng.js';
import { allocate, CLUSTER_PALETTE, fibonacciSpherePoints } from './shared.js';

export const CLUSTERS_3D = {
  id: 'clusters_3d',
  label: '3D Gaussian clusters',
  params: [
    { name: 'clusters', type: 'int', default: 5, min: 2, max: 8 },
    { name: 'sep', type: 'float', default: 2, min: 0.5, max: 5 },
  ],
  generate({ samples, noise, seed, clusters, sep }) {
    const K = clusters || 5;
    const S = sep || 2;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    const centers = fibonacciSpherePoints(K, S);
    const colors = new Array(samples);
    const spread = 0.25;
    for (let i = 0; i < samples; i++) {
      const c = i % K;
      const ctr = centers[c];
      out.X[i * 3 + 0] = ctr[0] + spread * gaussian(rand);
      out.X[i * 3 + 1] = ctr[1] + spread * gaussian(rand);
      out.X[i * 3 + 2] = ctr[2] + spread * gaussian(rand);
      out.t[i] = c;
      colors[i] = CLUSTER_PALETTE[c % CLUSTER_PALETTE.length];
    }
    if (noise > 0) {
      for (let j = 0; j < out.X.length; j++) out.X[j] += noise * gaussian(rand);
    }
    out.colors = colors;
    return out;
  },
};
