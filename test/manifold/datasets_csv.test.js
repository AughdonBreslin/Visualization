import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseCSV } from '../../js/manifold/datasets.js';

test('parseCSV handles numeric headerless rows', () => {
  const rows = parseCSV('1,2,3\n4,5,6\n7,8,9\n');
  assert.deepEqual(rows, [[1,2,3],[4,5,6],[7,8,9]]);
});

test('parseCSV skips a non-numeric header row', () => {
  const rows = parseCSV('x,y,z\n1,2,3\n4,5,6\n');
  assert.deepEqual(rows, [[1,2,3],[4,5,6]]);
});

test('parseCSV drops rows with mismatched column counts', () => {
  const rows = parseCSV('1,2,3\n4,5\n7,8,9\n');
  assert.deepEqual(rows, [[1,2,3],[7,8,9]]);
});

test('parseCSV returns empty array on empty input', () => {
  assert.deepEqual(parseCSV(''), []);
});

test('parseCSV tolerates blank lines and whitespace', () => {
  const rows = parseCSV(' 1, 2 ,3 \n\n4,5,6\n');
  assert.deepEqual(rows, [[1,2,3],[4,5,6]]);
});
