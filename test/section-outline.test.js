// test/section-outline.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { slugify, uniqueId, normalizeLabel } from '../js/section-outline.js';

test('slugify lowercases and hyphenates non-word runs', () => {
  assert.equal(slugify('PCA and SVD'), 'pca-and-svd');
});

test('slugify strips punctuation and trims edge hyphens', () => {
  assert.equal(slugify('  Mean squared error (MSE) '), 'mean-squared-error-mse');
});

test('slugify falls back to "section" for empty input', () => {
  assert.equal(slugify(''), 'section');
  assert.equal(slugify('   '), 'section');
});

test('uniqueId returns base when free, then numeric suffixes', () => {
  const used = new Set(['overview']);
  assert.equal(uniqueId('overview', used), 'overview-2');
  assert.equal(uniqueId('overview', used), 'overview-3');
  assert.equal(uniqueId('dataset', used), 'dataset');
});

test('normalizeLabel collapses whitespace and newlines', () => {
  assert.equal(normalizeLabel('  Active\n  Distributions '), 'Active Distributions');
});
