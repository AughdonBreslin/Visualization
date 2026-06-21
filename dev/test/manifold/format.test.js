import { test } from 'node:test';
import assert from 'node:assert/strict';
import { formatVec3, formatMatrix, formatTable } from '../../../js/manifold/format.js';

test('formatVec3 pads numbers to fixed width', () => {
  const s = formatVec3([1.234, -0.5, 12.3]);
  assert.equal(s, '( 1.234, -0.500, 12.300)');
});

test('formatMatrix renders a small numeric grid', () => {
  const M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
  const s = formatMatrix(M);
  assert.ok(s.includes('1.000'));
  assert.ok(s.includes('9.000'));
  const rows = s.split('\n');
  assert.equal(rows.length, 3);
});

test('formatMatrix supports a row count limit with ellipsis', () => {
  const M = [];
  for (let i = 0; i < 10; i++) M.push([i, i + 1, i + 2, i + 3]);
  const s = formatMatrix(M, { maxRows: 4 });
  const rows = s.split('\n');
  assert.equal(rows.length, 5);
  assert.ok(rows[4].includes('...'));
});

test('formatTable produces a header row and body rows', () => {
  const t = formatTable(
    ['i', 'x_i', 'x_i - mu'],
    [
      [5, '(1.230, 4.560, 7.890)', '(1.122, 4.448, 7.778)'],
      [10, '(2.100, 3.450, 6.780)', '(1.992, 3.338, 6.668)'],
    ]
  );
  const lines = t.split('\n');
  assert.equal(lines.length, 3);
  assert.ok(lines[0].includes('i'));
  assert.ok(lines[1].includes('5'));
  assert.ok(lines[2].includes('10'));
});
