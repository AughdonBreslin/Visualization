import { test } from 'node:test';
import assert from 'node:assert/strict';
import { allocate, addNoise, CLUSTER_PALETTE, fibonacciSpherePoints } from '../../../js/manifold/datasets/shared.js';
import { mulberry32 } from '../../../js/manifold/rng.js';

test('allocate returns typed arrays of the right size', () => {
  const out = allocate(10);
  assert.equal(out.N, 10);
  assert.ok(out.X instanceof Float64Array);
  assert.ok(out.t instanceof Float64Array);
  assert.equal(out.X.length, 30);
  assert.equal(out.t.length, 10);
});

test('addNoise is a no-op when noise is zero', () => {
  const out = allocate(4);
  const before = out.X.slice();
  addNoise(out.X, 0, mulberry32(1));
  for (let i = 0; i < before.length; i++) assert.equal(out.X[i], before[i]);
});

test('addNoise perturbs entries when noise is positive', () => {
  const out = allocate(4);
  addNoise(out.X, 0.5, mulberry32(1));
  let nonzero = 0;
  for (let i = 0; i < out.X.length; i++) if (out.X[i] !== 0) nonzero++;
  assert.ok(nonzero > 0);
});

test('CLUSTER_PALETTE has 8 distinct color strings', () => {
  assert.equal(CLUSTER_PALETTE.length, 8);
  assert.equal(new Set(CLUSTER_PALETTE).size, 8);
  for (const c of CLUSTER_PALETTE) assert.match(c, /^#[0-9a-fA-F]{6}$/);
});

test('fibonacciSpherePoints returns count points on the requested radius', () => {
  const pts = fibonacciSpherePoints(5, 2);
  assert.equal(pts.length, 5);
  for (const p of pts) {
    const r = Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    assert.ok(Math.abs(r - 2) < 1e-9);
  }
});
