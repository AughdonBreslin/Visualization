// Dump a dataset's 3D points to JSON so the manim pipeline can reuse the exact
// same generators as the web sandbox (no Python reimplementation, no drift).
//
// Usage:
//   node manimexp/isomap/gen_points.mjs <datasetId> <samples> <seed> <outPath>
//
// Writes { "points": [[x,y,z], ...], "t": [...] }. Default dataset params are used.
import fs from 'fs';
import {
  SWISS_ROLL, S_CURVE, HELIX, TREFOIL_KNOT, TOROIDAL_HELIX, SPIRAL_DISK,
} from '../../js/manifold/datasets/synthetic_curves.js';
import {
  TWIN_PEAKS, SADDLE, CYLINDER, SEVERED_SPHERE, HILBERT, FULL_SPHERE,
} from '../../js/manifold/datasets/synthetic_surfaces.js';
import { CLUSTERS_3D } from '../../js/manifold/datasets/synthetic_clusters.js';

const REGISTRY = Object.fromEntries(
  [SWISS_ROLL, S_CURVE, HELIX, TREFOIL_KNOT, TOROIDAL_HELIX, SPIRAL_DISK,
    TWIN_PEAKS, SADDLE, CYLINDER, SEVERED_SPHERE, HILBERT, FULL_SPHERE,
    CLUSTERS_3D].map((d) => [d.id, d]),
);

const [, , id, samplesArg, seedArg, outPath] = process.argv;
const samples = parseInt(samplesArg || '1000', 10);
const seed = parseInt(seedArg || '0', 10);
const ds = REGISTRY[id];
if (!ds) {
  console.error(`unknown dataset '${id}'. known: ${Object.keys(REGISTRY).join(', ')}`);
  process.exit(1);
}

const params = {};
for (const p of (ds.params || [])) params[p.name] = p.default;
const out = ds.generate({ samples, noise: 0, seed, ...params });

const points = [];
for (let i = 0; i < out.N; i++) {
  points.push([out.X[i * 3 + 0], out.X[i * 3 + 1], out.X[i * 3 + 2]]);
}
fs.writeFileSync(outPath, JSON.stringify({ points, t: Array.from(out.t) }));
console.error(`wrote ${out.N} points for '${id}' to ${outPath}`);
