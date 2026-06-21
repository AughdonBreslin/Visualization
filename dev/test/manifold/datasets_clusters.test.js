import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CLUSTERS_3D } from '../../../js/manifold/datasets/synthetic_clusters.js';
import { CLUSTER_PALETTE } from '../../../js/manifold/datasets/shared.js';

test('clusters_3d yields 3N flat X and length-N t', () => {
  const out = CLUSTERS_3D.generate({ samples: 20, noise: 0, seed: 1, clusters: 5, sep: 2 });
  assert.equal(out.X.length, 60);
  assert.equal(out.t.length, 20);
  assert.ok(out.X instanceof Float64Array);
});

test('clusters_3d is deterministic for a fixed seed', () => {
  const a = CLUSTERS_3D.generate({ samples: 30, noise: 0, seed: 9, clusters: 4, sep: 2 });
  const b = CLUSTERS_3D.generate({ samples: 30, noise: 0, seed: 9, clusters: 4, sep: 2 });
  for (let i = 0; i < a.X.length; i++) assert.equal(a.X[i], b.X[i]);
});

test('clusters_3d emits a colors array drawn from the palette', () => {
  const out = CLUSTERS_3D.generate({ samples: 30, noise: 0, seed: 1, clusters: 3, sep: 2 });
  assert.ok(Array.isArray(out.colors));
  assert.equal(out.colors.length, 30);
  for (const c of out.colors) assert.ok(CLUSTER_PALETTE.includes(c));
});

test('clusters_3d partitions points roughly evenly across clusters', () => {
  const out = CLUSTERS_3D.generate({ samples: 30, noise: 0, seed: 1, clusters: 3, sep: 2 });
  const counts = [0, 0, 0];
  for (let i = 0; i < out.N; i++) counts[out.t[i]]++;
  for (const c of counts) assert.equal(c, 10);
});
