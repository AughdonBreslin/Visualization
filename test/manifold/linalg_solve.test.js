import { test } from 'node:test';
import assert from 'node:assert/strict';
import { solveLinearSystem } from '../../js/manifold/linalg.js';

test('solveLinearSystem on a 3x3 identity returns the right-hand side', () => {
  const A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const b = [2, 3, 4];
  const x = solveLinearSystem(A, b);
  assert.deepEqual(x, [2, 3, 4]);
});

test('solveLinearSystem on a 3x3 with pivots returns the unique solution', () => {
  const A = [[2, 1, 1], [4, -6, 0], [-2, 7, 2]];
  const b = [5, -2, 9];
  const x = solveLinearSystem(A, b);
  assert.ok(Math.abs(x[0] - 1) < 1e-9);
  assert.ok(Math.abs(x[1] - 1) < 1e-9);
  assert.ok(Math.abs(x[2] - 2) < 1e-9);
});

test('solveLinearSystem returns null on a singular matrix', () => {
  const A = [[1, 2], [2, 4]];
  const b = [3, 6];
  const x = solveLinearSystem(A, b);
  assert.equal(x, null);
});
