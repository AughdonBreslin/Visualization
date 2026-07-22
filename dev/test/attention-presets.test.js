// dev/test/attention-presets.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computePipeline } from '../../js/attention/math.js';
import { PRESETS, WEIGHTS, TOKEN_COLORS } from '../../js/attention/presets.js';

test('every preset has one color per token position', () => {
  for (const preset of PRESETS) {
    assert.ok(TOKEN_COLORS.length >= preset.tokens.length, `${preset.id} needs ${preset.tokens.length} colors, only ${TOKEN_COLORS.length} defined`);
  }
});

// The project's own quality bar: every preset's attention must be clearly peaked, not
// near-uniform. A prior preset ("dog-ran-fast") once shipped near-uniform (~0.05 above the
// 0.333 baseline) and had to be redone -- this guards the same regression for every preset,
// including after positional encoding shifted every preset's numbers, not just the new one's.
for (const preset of PRESETS) {
  test(`${preset.id}: every row's attention is clearly peaked, not uniform`, () => {
    const result = computePipeline(preset.tokens, preset.embeddings, WEIGHTS);
    const uniform = 1 / preset.tokens.length;
    result.weights.forEach((row, i) => {
      const peak = Math.max(...row);
      assert.ok(
        peak > uniform * 1.3,
        `${preset.id} row ${i} (${preset.tokens[i]}) peak ${peak.toFixed(3)} is too close to uniform ${uniform.toFixed(3)}`
      );
    });
  });
}

test('dog-chased-cat: the two "the" positions (0 and 3) attend to different tokens', () => {
  const preset = PRESETS.find((p) => p.id === 'dog-chased-cat');
  const result = computePipeline(preset.tokens, preset.embeddings, WEIGHTS);
  const argmax = (row) => row.indexOf(Math.max(...row));
  assert.notEqual(
    argmax(result.weights[0]),
    argmax(result.weights[3]),
    'both instances of "the" attend to the same token -- positional encoding is not disambiguating them'
  );
});
