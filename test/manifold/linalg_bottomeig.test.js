import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bottomKSymmetricEig } from '../../js/manifold/linalg.js';

test('bottomKSymmetricEig on a small diagonal matrix returns the smallest eigenvalues', () => {
  const N = 5;
  const M = new Float64Array(N * N);
  const expected = [0, 1, 3, 5, 7];
  for (let i = 0; i < N; i++) M[i * N + i] = expected[i];
  const { lambda, vectors } = bottomKSymmetricEig(M, N, 2);
  assert.ok(Math.abs(lambda[0] - 0) < 1e-6, 'first eigenvalue should be 0');
  assert.ok(Math.abs(lambda[1] - 1) < 1e-6, 'second eigenvalue should be 1');
  assert.equal(vectors.length, 2);
  assert.equal(vectors[0].length, N);
});

test('bottomKSymmetricEig with skipFirst skips trivial smallest', () => {
  const N = 5;
  const M = new Float64Array(N * N);
  const expected = [0, 1, 3, 5, 7];
  for (let i = 0; i < N; i++) M[i * N + i] = expected[i];
  const { lambda } = bottomKSymmetricEig(M, N, 2, { skipFirst: 1 });
  assert.ok(Math.abs(lambda[0] - 1) < 1e-6, 'first non-trivial eigenvalue should be 1');
  assert.ok(Math.abs(lambda[1] - 3) < 1e-6, 'second non-trivial eigenvalue should be 3');
});

test('bottomKSymmetricEig on a larger diagonal matrix uses shift-deflate path', () => {
  const N = 20;
  const M = new Float64Array(N * N);
  for (let i = 0; i < N; i++) M[i * N + i] = i + 1;
  const { lambda, vectors } = bottomKSymmetricEig(M, N, 3);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(lambda[i] - (i + 1)) < 1e-3, 'eigenvalue ' + i + ' approximately ' + (i + 1) + ', got ' + lambda[i]);
  }
  assert.equal(vectors[0].length, N);
});

test('bottomKSymmetricEig returns eigenvectors satisfying M v = lambda v', () => {
  const N = 6;
  const M = new Float64Array(N * N);
  for (let i = 0; i < N; i++) M[i * N + i] = (i + 1) * 0.5;
  M[0 * N + 1] = 0.1; M[1 * N + 0] = 0.1;
  M[2 * N + 4] = 0.2; M[4 * N + 2] = 0.2;
  const { lambda, vectors } = bottomKSymmetricEig(M, N, 2);
  for (let k = 0; k < 2; k++) {
    const v = vectors[k];
    const Mv = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let j = 0; j < N; j++) s += M[i * N + j] * v[j];
      Mv[i] = s;
    }
    for (let i = 0; i < N; i++) {
      assert.ok(Math.abs(Mv[i] - lambda[k] * v[i]) < 1e-4,
        'M v[' + i + '] should equal lambda v[' + i + '] for k=' + k);
    }
  }
});
