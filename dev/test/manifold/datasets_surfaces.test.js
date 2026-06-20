import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  TWIN_PEAKS, SADDLE, CYLINDER, SEVERED_SPHERE, HILBERT, FULL_SPHERE,
} from '../../../js/manifold/datasets/synthetic_surfaces.js';

const SURFACES = [TWIN_PEAKS, SADDLE, CYLINDER, SEVERED_SPHERE, HILBERT, FULL_SPHERE];

test('every surface yields 3N flat X and length-N t as Float64Array', () => {
  for (const ds of SURFACES) {
    const out = ds.generate({ samples: 20, noise: 0, seed: 1 });
    assert.equal(out.X.length, 60, `${ds.id} X length`);
    assert.equal(out.t.length, 20, `${ds.id} t length`);
    assert.ok(out.X instanceof Float64Array, `${ds.id} X type`);
    assert.ok(out.t instanceof Float64Array, `${ds.id} t type`);
  }
});

test('every surface is deterministic for a fixed seed', () => {
  for (const ds of SURFACES) {
    const a = ds.generate({ samples: 15, noise: 0, seed: 3 });
    const b = ds.generate({ samples: 15, noise: 0, seed: 3 });
    for (let i = 0; i < a.X.length; i++) assert.equal(a.X[i], b.X[i], `${ds.id} det`);
  }
});

test('full sphere points lie on the unit sphere', () => {
  const out = FULL_SPHERE.generate({ samples: 50, noise: 0, seed: 2 });
  for (let i = 0; i < out.N; i++) {
    const x = out.X[i * 3], y = out.X[i * 3 + 1], z = out.X[i * 3 + 2];
    assert.ok(Math.abs(Math.sqrt(x * x + y * y + z * z) - 1) < 1e-9);
  }
});

test('severed sphere removes the north polar cap', () => {
  const out = SEVERED_SPHERE.generate({ samples: 200, noise: 0, seed: 4, cap: 0.35 });
  const minZ = Math.cos(0.35 * Math.PI);
  for (let i = 0; i < out.N; i++) {
    assert.ok(out.X[i * 3 + 2] <= minZ + 1e-9, 'no point inside the removed cap');
  }
});

test('hilbert curve fills the cube and steps between adjacent cells', () => {
  const out = HILBERT.generate({ samples: 512, noise: 0, seed: 1, order: 3 });
  for (let i = 0; i < out.N; i++) {
    for (let d = 0; d < 3; d++) {
      const c = out.X[i * 3 + d];
      assert.ok(c >= -1 - 1e-9 && c <= 1 + 1e-9, 'point inside the unit cube');
    }
  }
  // consecutive samples (taken in curve order) stay close in 3D
  let maxStep = 0;
  for (let i = 1; i < out.N; i++) {
    let d2 = 0;
    for (let d = 0; d < 3; d++) {
      const delta = out.X[i * 3 + d] - out.X[(i - 1) * 3 + d];
      d2 += delta * delta;
    }
    maxStep = Math.max(maxStep, Math.sqrt(d2));
  }
  assert.ok(maxStep < 0.5, 'consecutive curve points are local');
});

test('surfaces expose expected ids', () => {
  assert.deepEqual(SURFACES.map(d => d.id),
    ['twin_peaks', 'saddle', 'cylinder', 'severed_sphere', 'hilbert', 'full_sphere']);
});
