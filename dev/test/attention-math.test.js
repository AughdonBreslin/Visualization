// dev/test/attention-math.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  linearProject, dot, positionalEncoding, buildInputMatrix,
  projectAll, scoreMatrix, weightedSum, computePipeline,
} from '../../js/attention/math.js';

function closeTo(a, b, eps = 1e-9) {
  return Math.abs(a - b) < eps;
}

test('positionalEncoding(0, 4) is exactly [0, 1, 0, 1]', () => {
  assert.deepEqual(positionalEncoding(0, 4), [0, 1, 0, 1]);
});

test('positionalEncoding(1, 4) matches the sin/cos formula', () => {
  const pe = positionalEncoding(1, 4);
  assert.ok(closeTo(pe[0], Math.sin(1)));
  assert.ok(closeTo(pe[1], Math.cos(1)));
  assert.ok(closeTo(pe[2], Math.sin(1 / 100)));
  assert.ok(closeTo(pe[3], Math.cos(1 / 100)));
});

test('buildInputMatrix scales the embedding by sqrt(d) before adding position', () => {
  const X = buildInputMatrix([[0.2, 0.8, 0.1, 0.4]], 4);
  assert.ok(closeTo(X[0][0], 0.4));
  assert.ok(closeTo(X[0][1], 2.6));
  assert.ok(closeTo(X[0][2], 0.2));
  assert.ok(closeTo(X[0][3], 1.8));
});

test('buildInputMatrix gives an identical raw embedding different vectors at different positions', () => {
  const e = [0.2, 0.8, 0.1, 0.4];
  const X = buildInputMatrix([e, e, e, e], 4);
  assert.ok(!closeTo(X[0][0], X[3][0]) || !closeTo(X[0][1], X[3][1]));
  assert.ok(closeTo(X[3][0], 0.5411200080598672));
  assert.ok(closeTo(X[3][1], 0.6100075033995546));
  assert.ok(closeTo(X[3][2], 0.22999550033995098));
  assert.ok(closeTo(X[3][3], 1.7995500337489875));
});

test('projectAll applies a matrix to every row, indexed by position', () => {
  const identity4 = [[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0], [0, 0, 0, 5]];
  const out = projectAll([[1, 0, 0, 0], [0, 1, 0, 0]], identity4);
  assert.deepEqual(out[0], [2, 0, 0, 0]);
  assert.deepEqual(out[1], [0, 3, 0, 0]);
});

test('scoreMatrix dots every Q row against every K row by position', () => {
  const scores = scoreMatrix([[1, 0], [0, 1]], [[1, 0], [0, 1]]);
  assert.deepEqual(scores, [[1, 0], [0, 1]]);
});

test('weightedSum blends V rows by weight, indexed by position', () => {
  const out = weightedSum([[0.5, 0.5]], [[2, 0], [0, 2]], 2);
  assert.ok(closeTo(out[0][0], 1));
  assert.ok(closeTo(out[0][1], 1));
});

test('computePipeline gives a repeated token different Q, K, and V at each position (the fix)', () => {
  const tokens = ['the', 'dog', 'chased', 'the', 'cat'];
  const embeddings = [
    [0.2, 0.8, 0.1, 0.4],
    [1.0, 0.0, 0.9, 0.0],
    [0.2, 0.4, 1.0, 1.0],
    [0.2, 0.8, 0.1, 0.4],
    [0.9, 0.1, 0.6, 0.3],
  ];
  const weights = {
    WQ: [[1.5, 0.0, 0.8, 0.0], [0.0, 1.6, 0.0, 0.6], [0.8, 0.0, 1.5, 0.0], [0.0, 0.6, 0.0, 1.6]],
    WK: [[1.4, 0.3, 0.0, 0.0], [0.3, 1.4, 0.0, 0.0], [0.0, 0.0, 1.4, 0.3], [0.0, 0.0, 0.3, 1.4]],
    WV: [[0.9, 0.0, 0.0, 0.2], [0.0, 0.9, 0.2, 0.0], [0.0, 0.2, 0.9, 0.0], [0.2, 0.0, 0.0, 0.9]],
  };
  const result = computePipeline(tokens, embeddings, weights);
  assert.ok(!closeTo(result.Q[0][0], result.Q[3][0]) || !closeTo(result.Q[0][1], result.Q[3][1]));
  assert.ok(!closeTo(result.K[0][0], result.K[3][0]) || !closeTo(result.K[0][1], result.K[3][1]));
  assert.ok(!closeTo(result.V[0][0], result.V[3][0]) || !closeTo(result.V[0][1], result.V[3][1]));
});
