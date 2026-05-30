import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DATASETS, DATASETS_BY_ID, parseCSV } from '../../js/manifold/datasets/index.js';

const EXPECTED_ORDER = [
  'swiss_roll', 's_curve', 'helix', 'trefoil_knot', 'toroidal_helix', 'spiral_disk',
  'twin_peaks', 'saddle', 'cylinder', 'severed_sphere', 'hilbert', 'full_sphere',
  'clusters_3d', 'csv',
];

test('DATASETS lists all 14 datasets in order', () => {
  assert.deepEqual(DATASETS.map(d => d.id), EXPECTED_ORDER);
});

test('DATASETS_BY_ID maps every id', () => {
  for (const id of EXPECTED_ORDER) assert.ok(DATASETS_BY_ID[id], `missing ${id}`);
});

test('parseCSV is re-exported from the aggregator', () => {
  assert.equal(typeof parseCSV, 'function');
  assert.deepEqual(parseCSV('1,2,3\n4,5,6\n'), [[1, 2, 3], [4, 5, 6]]);
});

test('the datasets.js shim re-exports the same DATASETS', async () => {
  const shim = await import('../../js/manifold/datasets.js');
  assert.deepEqual(shim.DATASETS.map(d => d.id), EXPECTED_ORDER);
  assert.equal(typeof shim.parseCSV, 'function');
});
