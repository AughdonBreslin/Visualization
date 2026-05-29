import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CANONICAL_STEPS, canonicalOf, compareSubSteps, unionSubSteps } from '../../js/manifold/canonical_steps.js';

test('CANONICAL_STEPS has 7 entries 0..6', () => {
  assert.equal(CANONICAL_STEPS.length, 7);
  assert.deepEqual(CANONICAL_STEPS.map(s => s.id), ['0','1','2','3','4','5','6']);
});

test('canonicalOf strips the alphabetic suffix', () => {
  assert.equal(canonicalOf('2'), '2');
  assert.equal(canonicalOf('2a'), '2');
  assert.equal(canonicalOf('2b'), '2');
});

test('compareSubSteps orders by canonical then suffix', () => {
  assert.ok(compareSubSteps('0', '1') < 0);
  assert.ok(compareSubSteps('2', '2a') < 0);
  assert.ok(compareSubSteps('2a', '2b') < 0);
  assert.equal(compareSubSteps('3', '3'), 0);
});

test('unionSubSteps deduplicates and sorts', () => {
  assert.deepEqual(unionSubSteps(['0','2','2a'], ['0','2b','3']), ['0','2','2a','2b','3']);
});
