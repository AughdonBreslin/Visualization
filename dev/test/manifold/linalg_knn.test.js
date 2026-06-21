import { test } from 'node:test';
import assert from 'node:assert/strict';
import { knnGraph } from '../../../js/manifold/linalg.js';

test('kNN on a small line returns nearest neighbours', () => {
  const X = new Float64Array([
    0, 0, 0,
    1, 0, 0,
    2, 0, 0,
    3, 0, 0,
  ]);
  const { adj, edges } = knnGraph(X, 1);
  assert.equal(edges.length >= 3, true);
  assert.equal(adj.length, 4);
  for (const row of adj) assert.ok(row.length >= 1);
});

test('kNN edges are undirected and deduplicated', () => {
  const X = new Float64Array([0,0,0, 1,0,0, 2,0,0]);
  const { edges } = knnGraph(X, 1);
  const seen = new Set();
  for (const [a,b] of edges) {
    const k = a < b ? `${a},${b}` : `${b},${a}`;
    assert.ok(!seen.has(k), `duplicate edge ${k}`);
    seen.add(k);
  }
});
