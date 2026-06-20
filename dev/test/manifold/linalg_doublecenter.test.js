import { test } from 'node:test';
import assert from 'node:assert/strict';
import { doubleCenterSquared } from '../../../js/manifold/linalg.js';

test('Double-centered Gram matrix has zero row/column sums', () => {
  const D = new Float64Array([
    0, 1, 2,
    1, 0, 1,
    2, 1, 0,
  ]);
  const B = doubleCenterSquared(D, 3);
  for (let i = 0; i < 3; i++) {
    let row = 0, col = 0;
    for (let j = 0; j < 3; j++) {
      row += B[i*3 + j];
      col += B[j*3 + i];
    }
    assert.ok(Math.abs(row) < 1e-9, `row ${i} not zero: ${row}`);
    assert.ok(Math.abs(col) < 1e-9, `col ${i} not zero: ${col}`);
  }
});
