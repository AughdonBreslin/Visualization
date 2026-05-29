import { test } from 'node:test';
import assert from 'node:assert/strict';
import { dijkstraAllPairs } from '../../js/manifold/linalg.js';

test('Dijkstra on a 3-node path returns expected distances', () => {
  const adj = [
    [[1, 1]],
    [[0, 1], [2, 1]],
    [[1, 1]],
  ];
  const D = dijkstraAllPairs(adj, 3);
  assert.equal(D[0*3 + 0], 0);
  assert.equal(D[0*3 + 1], 1);
  assert.equal(D[0*3 + 2], 2);
});

test('Dijkstra reports Infinity for disconnected nodes', () => {
  const adj = [
    [[1, 1]],
    [[0, 1]],
    [],
  ];
  const D = dijkstraAllPairs(adj, 3);
  assert.equal(D[0*3 + 2], Infinity);
});
