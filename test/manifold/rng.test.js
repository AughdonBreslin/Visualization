import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, gaussian } from '../../js/manifold/rng.js';

test('mulberry32 is deterministic for a given seed', () => {
  const a = mulberry32(42);
  const b = mulberry32(42);
  const va = [a(), a(), a()];
  const vb = [b(), b(), b()];
  assert.deepEqual(va, vb);
});

test('mulberry32 produces different streams for different seeds', () => {
  const a = mulberry32(1)();
  const b = mulberry32(2)();
  assert.notEqual(a, b);
});

test('mulberry32 returns values in [0,1)', () => {
  const r = mulberry32(7);
  for (let i = 0; i < 1000; i++) {
    const v = r();
    assert.ok(v >= 0 && v < 1, `value out of range: ${v}`);
  }
});

test('gaussian returns finite numbers', () => {
  const r = mulberry32(11);
  for (let i = 0; i < 100; i++) {
    const v = gaussian(r);
    assert.ok(Number.isFinite(v));
  }
});
