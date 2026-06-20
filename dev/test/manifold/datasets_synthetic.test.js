import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SWISS_ROLL, S_CURVE, DATASETS, DATASETS_BY_ID } from '../../../js/manifold/datasets.js';

test('Swiss roll yields N points with length 3N flat array', () => {
  const out = SWISS_ROLL.generate({ samples: 50, noise: 0, seed: 1 });
  assert.equal(out.N, 50);
  assert.equal(out.X.length, 150);
  assert.equal(out.t.length, 50);
});

test('Swiss roll is reproducible for the same seed', () => {
  const a = SWISS_ROLL.generate({ samples: 30, noise: 0, seed: 7 });
  const b = SWISS_ROLL.generate({ samples: 30, noise: 0, seed: 7 });
  for (let i = 0; i < a.X.length; i++) assert.equal(a.X[i], b.X[i]);
});

test('Swiss roll changes with the seed', () => {
  const a = SWISS_ROLL.generate({ samples: 30, noise: 0, seed: 1 });
  const b = SWISS_ROLL.generate({ samples: 30, noise: 0, seed: 2 });
  let diff = 0;
  for (let i = 0; i < a.X.length; i++) if (a.X[i] !== b.X[i]) diff++;
  assert.ok(diff > 0);
});

test('S-curve also yields the right shape', () => {
  const out = S_CURVE.generate({ samples: 40, noise: 0, seed: 1 });
  assert.equal(out.N, 40);
  assert.equal(out.X.length, 120);
  assert.equal(out.t.length, 40);
});

test('DATASETS starts with swiss_roll and s_curve and ends with csv', () => {
  const ids = DATASETS.map(d => d.id);
  assert.equal(ids[0], 'swiss_roll');
  assert.equal(ids[1], 's_curve');
  assert.equal(ids[ids.length - 1], 'csv');
  assert.equal(DATASETS_BY_ID.swiss_roll.label, 'Swiss roll');
});
