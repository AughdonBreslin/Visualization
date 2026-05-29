import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  SWISS_ROLL, S_CURVE, HELIX, TREFOIL_KNOT, TOROIDAL_HELIX, SPIRAL_DISK,
} from '../../js/manifold/datasets/synthetic_curves.js';

const CURVES = [SWISS_ROLL, S_CURVE, HELIX, TREFOIL_KNOT, TOROIDAL_HELIX, SPIRAL_DISK];

test('every curve yields 3N flat X and length-N t as Float64Array', () => {
  for (const ds of CURVES) {
    const out = ds.generate({ samples: 20, noise: 0, seed: 1 });
    assert.equal(out.X.length, 60, `${ds.id} X length`);
    assert.equal(out.t.length, 20, `${ds.id} t length`);
    assert.ok(out.X instanceof Float64Array, `${ds.id} X type`);
    assert.ok(out.t instanceof Float64Array, `${ds.id} t type`);
  }
});

test('every curve is deterministic for a fixed seed', () => {
  for (const ds of CURVES) {
    const a = ds.generate({ samples: 15, noise: 0, seed: 3 });
    const b = ds.generate({ samples: 15, noise: 0, seed: 3 });
    for (let i = 0; i < a.X.length; i++) assert.equal(a.X[i], b.X[i], `${ds.id} det`);
  }
});

test('helix turns param changes the trace', () => {
  const a = HELIX.generate({ samples: 30, noise: 0, seed: 5, turns: 2 });
  const b = HELIX.generate({ samples: 30, noise: 0, seed: 5, turns: 6 });
  let diff = 0;
  for (let i = 0; i < a.X.length; i++) if (a.X[i] !== b.X[i]) diff++;
  assert.ok(diff > 0);
});

test('curves expose expected ids and labels', () => {
  assert.deepEqual(CURVES.map(d => d.id),
    ['swiss_roll', 's_curve', 'helix', 'trefoil_knot', 'toroidal_helix', 'spiral_disk']);
});
