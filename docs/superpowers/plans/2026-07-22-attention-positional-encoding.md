# Attention Positional Encoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real (pipeline-propagating) sinusoidal positional encoding to the Attention page, and fix the underlying bug it surfaces: `Q`/`K`/`V`/`embeddings` are keyed by token string instead of position, so repeated tokens silently collide.

**Architecture:** `math.js` gains a pure `positionalEncoding`/`buildInputMatrix` step run before projection; every data structure that was a `{tokenString: vector}` dict becomes a plain array indexed by position, matching how `scores`/`weights`/`output` already work. A third preset with a repeated word ("the dog chased the cat") proves the fix. Input's filmstrip gains two new stages (the addition itself, and a RoPE/ALiBi related-research aside) and its existing scale-stats stage is renumbered.

**Tech Stack:** Vanilla JS (ES modules), no build step, no framework. Tests use Node's built-in `node:test` + `node:assert/strict`, run via `node --test <file>` (no package.json, no test runner config — this matches `dev/test/section-outline.test.js`'s existing precedent).

## Global Constraints

- No em-dashes anywhere (literal `—` or `&mdash;`), in code, comments, or copy — check with `grep` after every edit.
- No new emphasis markup (`<b>`, bold, italics) in newly-written prose beyond what already exists at each edited call site.
- Every numeric value shown on the page must come from the live `PipelineResult`, never hardcoded — this file's own header comment already states this convention; follow it for every new line of UI code.
- Verify every visual change in headless Firefox per the `browser-preview` skill: 0 console errors, 0 horizontal overflow at 1400px and 375px, for every preset (not just the new one).
- Sinusoidal positional encoding formula (exact, from Vaswani et al. 2017, "Attention Is All You Need," section 3.5): `PE(pos, 2i) = sin(pos / 10000^(2i/d))`, `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`.
- Embeddings are scaled by `sqrt(d)` before adding positional encoding (same section 3.4 detail from the original paper) — verified computationally in this plan's own prep work to be necessary: without it, two of the three presets' attention patterns become too close to uniform once position is added.
- New preset's one new hand-picked embedding ("chased") is `[0.2, 0.4, 1.0, 1.0]` — verified (see Task 2) to keep every row of every preset's attention pattern clearly peaked (>1.3x the uniform baseline for that row count) and to make the two "the" positions (0 and 3) attend to different tokens, which is the whole point of the fix.

---

## Task 1: Positional encoding and position-indexed math

**Files:**
- Modify: `js/attention/math.js` (full rewrite)
- Test: `dev/test/attention-math.test.js` (new)

**Interfaces:**
- Produces: `positionalEncoding(pos, d)` → `number[]`. `buildInputMatrix(embeddings, d)` → `number[][]` (new). `linearProject(W, x)` and `dot(a, b)` unchanged. `projectAll(embeddings, W)` → `number[][]` (signature changed: drops the `tokens` parameter, `embeddings` is now an array not a dict). `scoreMatrix(Q, K)` → `number[][]` (signature changed: drops `tokens`). `scaleMatrix`, `NEG_INF`, `applyCausalMask`, `softmaxRow`, `softmaxMatrix` unchanged. `weightedSum(weights, V, d)` → `number[][]` (signature changed: drops `tokens`). `computePipeline(tokens, embeddings, weights, options)` — same call signature, but `embeddings` argument must now be an array of vectors (parallel to `tokens`), not a `{word: vector}` object, and the returned object gains a new `X` field (`number[][]`, the post-scaling-and-position input matrix) alongside `Q`/`K`/`V`/`output`, which are now arrays instead of dicts.
- Consumes: nothing from other tasks — this is the base pure-computation layer.

- [ ] **Step 1: Write the failing test**

Create `dev/test/attention-math.test.js`:

```js
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test dev/test/attention-math.test.js`
Expected: FAIL — `positionalEncoding` and `buildInputMatrix` are not exported from the current `math.js`, so the import throws.

- [ ] **Step 3: Write the implementation**

Replace the full contents of `js/attention/math.js`:

```js
// js/attention/math.js
// Pure computation for single-head scaled dot-product attention. No DOM access anywhere in
// this file — it is the only part of this page's code unit-testable with plain Node asserts.

export function linearProject(W, x) {
  return W.map((row) => row.reduce((sum, w, j) => sum + w * x[j], 0));
}

export function dot(a, b) {
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

// Standard sinusoidal positional encoding (Vaswani et al. 2017, "Attention Is All You Need",
// section 3.5). Computed exactly -- unlike the weight matrices, nothing here is hand-picked.
export function positionalEncoding(pos, d) {
  const out = new Array(d);
  for (let i = 0; i < d / 2; i++) {
    const divisor = Math.pow(10000, (2 * i) / d);
    out[2 * i] = Math.sin(pos / divisor);
    out[2 * i + 1] = Math.cos(pos / divisor);
  }
  return out;
}

// Embeddings are scaled by sqrt(d) before adding positional encoding, the same detail the
// original paper's embedding layer uses (section 3.4): without it, the positional signal
// (bounded to [-1, 1]) can overwhelm the hand-picked embedding values once added, flattening
// the resulting attention pattern toward uniform.
export function buildInputMatrix(embeddings, d) {
  const scale = Math.sqrt(d);
  return embeddings.map((embedding, pos) => {
    const pe = positionalEncoding(pos, d);
    return embedding.map((v, k) => v * scale + pe[k]);
  });
}

export function projectAll(embeddings, W) {
  return embeddings.map((x) => linearProject(W, x));
}

export function scoreMatrix(Q, K) {
  return Q.map((qi) => K.map((kj) => dot(qi, kj)));
}

export function scaleMatrix(scores, d) {
  const s = Math.sqrt(d);
  return scores.map((row) => row.map((v) => v / s));
}

export const NEG_INF = -1e9;

export function applyCausalMask(scaled) {
  return scaled.map((row, i) => row.map((v, j) => (j > i ? NEG_INF : v)));
}

export function softmaxRow(row) {
  const m = Math.max(...row);
  const exps = row.map((v) => Math.exp(v - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export function softmaxMatrix(matrix) {
  return matrix.map(softmaxRow);
}

export function weightedSum(weights, V, d) {
  return weights.map((row) => {
    const out = new Array(d).fill(0);
    row.forEach((w, j) => {
      V[j].forEach((vk, k) => {
        out[k] += w * vk;
      });
    });
    return out;
  });
}

export function computePipeline(tokens, embeddings, weights, options = {}) {
  const causal = !!options.causal;
  const d = embeddings[0].length;
  const X = buildInputMatrix(embeddings, d);
  const Q = projectAll(X, weights.WQ);
  const K = projectAll(X, weights.WK);
  const V = projectAll(X, weights.WV);
  const scores = scoreMatrix(Q, K);
  const scaled = scaleMatrix(scores, d);
  const masked = causal ? applyCausalMask(scaled) : scaled;
  const weightsOut = softmaxMatrix(masked);
  const output = weightedSum(weightsOut, V, d);
  return { tokens, embeddings, X, Q, K, V, scores, scaled, masked, weights: weightsOut, output, d, causal };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test dev/test/attention-math.test.js`
Expected: PASS, all 7 tests green.

- [ ] **Step 5: Commit**

```bash
git add js/attention/math.js dev/test/attention-math.test.js
git commit -m "feat(attention): add positional encoding, switch Q/K/V to position-indexed arrays"
```

---

## Task 2: Position-indexed presets, new repeated-word preset, extended token colors

**Files:**
- Modify: `js/attention/presets.js` (full rewrite)
- Test: `dev/test/attention-presets.test.js` (new)

**Interfaces:**
- Consumes: `computePipeline` from Task 1 (must accept an array-based `embeddings` argument).
- Produces: `PRESETS` (array of 3, each `embeddings` now an array not a dict), `WEIGHTS` (unchanged), `TOKEN_COLORS` (extended from 3 to 5 entries). Later tasks (`main.js`, `scenes.js`) read these the same way as before; only `embeddings`'s internal shape changed.

- [ ] **Step 1: Write the failing test**

Create `dev/test/attention-presets.test.js`:

```js
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test dev/test/attention-presets.test.js`
Expected: FAIL — `PRESETS`' current `embeddings` field is a `{word: vector}` object, so `computePipeline` (which now expects an array) throws when it calls `.map` on it.

- [ ] **Step 3: Write the implementation**

Replace the full contents of `js/attention/presets.js`:

```js
// js/attention/presets.js
// Worked-example data for the attention page. Three curated presets (no free-form editing in
// this phase - see docs/superpowers/specs/2026-07-19-attention-visualization-design.md).
// Weight matrices are shared across presets: hand-picked, not learned, chosen so the resulting
// attention pattern is clearly peaked (not near-uniform) after softmax. Embeddings are arrays
// indexed by position (parallel to `tokens`), not objects keyed by word: a repeated word (see
// "dog-chased-cat" below) gets one array entry per occurrence instead of colliding onto one.

export const PRESETS = [
  {
    id: 'cat-sat',
    label: '"the cat sat"',
    tokens: ['the', 'cat', 'sat'],
    embeddings: [
      [0.2, 0.8, 0.1, 0.4],
      [0.9, 0.1, 0.6, 0.3],
      [0.3, 0.5, 0.8, 0.2],
    ],
  },
  {
    id: 'dog-ran-fast',
    label: '"dog ran fast"',
    tokens: ['dog', 'ran', 'fast'],
    embeddings: [
      [1.0, 0.0, 0.9, 0.0],
      [0.0, 1.0, 0.0, 0.9],
      [0.95, 0.05, 0.85, 0.05],
    ],
  },
  {
    // Repeats "the" at positions 0 and 3 with an identical raw embedding, so positional
    // encoding is the only thing that can tell the two occurrences apart -- proving the
    // string-keyed-collision fix actually does something. Reuses "the"/"dog"/"cat" from the
    // two presets above; "chased" is the only genuinely new hand-picked embedding, tuned
    // (together with the sqrt(d) embedding scale in math.js) so every row of every preset
    // stays clearly peaked -- see dev/test/attention-presets.test.js.
    id: 'dog-chased-cat',
    label: '"the dog chased the cat"',
    tokens: ['the', 'dog', 'chased', 'the', 'cat'],
    embeddings: [
      [0.2, 0.8, 0.1, 0.4],
      [1.0, 0.0, 0.9, 0.0],
      [0.2, 0.4, 1.0, 1.0],
      [0.2, 0.8, 0.1, 0.4],
      [0.9, 0.1, 0.6, 0.3],
    ],
  },
];

export const WEIGHTS = {
  WQ: [
    [1.5, 0.0, 0.8, 0.0],
    [0.0, 1.6, 0.0, 0.6],
    [0.8, 0.0, 1.5, 0.0],
    [0.0, 0.6, 0.0, 1.6],
  ],
  WK: [
    [1.4, 0.3, 0.0, 0.0],
    [0.3, 1.4, 0.0, 0.0],
    [0.0, 0.0, 1.4, 0.3],
    [0.0, 0.0, 0.3, 1.4],
  ],
  WV: [
    [0.9, 0.0, 0.0, 0.2],
    [0.0, 0.9, 0.2, 0.0],
    [0.0, 0.2, 0.9, 0.0],
    [0.2, 0.0, 0.0, 0.9],
  ],
};

// Indexed by token POSITION (0-4), not by token identity, so recoloring stays stable when the
// preset changes which words are used, and so a repeated word (e.g. "the" in dog-chased-cat)
// gets two different colors, one per occurrence -- reinforcing that position, not word
// identity, is what's being tracked from here on.
export const TOKEN_COLORS = ['#7c8fff', '#e0b341', '#4fd1a5', '#c77dff', '#ff8fa3'];
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test dev/test/attention-presets.test.js`
Expected: PASS, all 5 tests green (2 presets x peakedness + colors + the-vs-the check; `dog-chased-cat`'s own peakedness test is one of the parametrized ones).

- [ ] **Step 5: Commit**

```bash
git add js/attention/presets.js dev/test/attention-presets.test.js
git commit -m "feat(attention): retrofit presets to position-indexed embeddings, add repeated-word preset"
```

---

## Task 3: CSS for 5-token grids

**Files:**
- Modify: `styles/attention.css:254-256` (desktop grid sizes) and `styles/attention.css:309-311` (mobile grid sizes)

**Interfaces:**
- Consumes: nothing (pure CSS, additive).
- Produces: `.mgrid.g5x4` and `.mgrid.g5x5` classes, so the new 5-token preset's embedding/Q/K/V matrices (5 rows x 4 cols) and score/mask/weights matrices (5x5) render at a sized grid instead of falling back to unstyled `display:grid` with no explicit cell size. `heatMatrixGrid` (in `scenes.js`) already generates class names as `g${matrix.length}x${cols}` dynamically — no JS changes needed, only the matching CSS rules.

- [ ] **Step 1: Add the desktop grid-size rules**

In `styles/attention.css`, find:

```css
.ui.attention .mgrid.g4x4 { grid-template-columns: repeat(4, 30px); grid-template-rows: repeat(4, 30px); }
.ui.attention .mgrid.g3x4 { grid-template-columns: repeat(4, 35px); grid-template-rows: repeat(3, 35px); }
.ui.attention .mgrid.g3x3 { grid-template-columns: repeat(3, 35px); grid-template-rows: repeat(3, 35px); }
```

Replace with:

```css
.ui.attention .mgrid.g4x4 { grid-template-columns: repeat(4, 30px); grid-template-rows: repeat(4, 30px); }
.ui.attention .mgrid.g3x4 { grid-template-columns: repeat(4, 35px); grid-template-rows: repeat(3, 35px); }
.ui.attention .mgrid.g3x3 { grid-template-columns: repeat(3, 35px); grid-template-rows: repeat(3, 35px); }
.ui.attention .mgrid.g5x4 { grid-template-columns: repeat(4, 35px); grid-template-rows: repeat(5, 35px); }
.ui.attention .mgrid.g5x5 { grid-template-columns: repeat(5, 35px); grid-template-rows: repeat(5, 35px); }
```

- [ ] **Step 2: Add the mobile grid-size rules**

In the same file, find (inside the `@media (max-width: 640px)` block):

```css
  .ui.attention .mgrid.g4x4 { grid-template-columns: repeat(4, 25px); grid-template-rows: repeat(4, 25px); }
  .ui.attention .mgrid.g3x4 { grid-template-columns: repeat(4, 27px); grid-template-rows: repeat(3, 27px); }
  .ui.attention .mgrid.g3x3 { grid-template-columns: repeat(3, 27px); grid-template-rows: repeat(3, 27px); }
```

Replace with:

```css
  .ui.attention .mgrid.g4x4 { grid-template-columns: repeat(4, 25px); grid-template-rows: repeat(4, 25px); }
  .ui.attention .mgrid.g3x4 { grid-template-columns: repeat(4, 27px); grid-template-rows: repeat(3, 27px); }
  .ui.attention .mgrid.g3x3 { grid-template-columns: repeat(3, 27px); grid-template-rows: repeat(3, 27px); }
  .ui.attention .mgrid.g5x4 { grid-template-columns: repeat(4, 27px); grid-template-rows: repeat(5, 27px); }
  .ui.attention .mgrid.g5x5 { grid-template-columns: repeat(5, 27px); grid-template-rows: repeat(5, 27px); }
```

- [ ] **Step 3: Verify visually**

This can't be checked until a preset with 5 tokens is selectable (Task 2 adds the data; Task 4/5 wire up the renderers that display it) — defer the actual visual check to Task 6's full verification pass. For now, confirm the CSS is syntactically valid:

Run: `grep -c "mgrid.g5x" styles/attention.css`
Expected: `4` (2 desktop + 2 mobile rules).

- [ ] **Step 4: Commit**

```bash
git add styles/attention.css
git commit -m "feat(attention): add grid CSS sizes for 5-token matrices"
```

---

## Task 4: Fix position-indexed lookups in QKV, Scores, and Weighted sum

**Files:**
- Modify: `js/attention/scenes.js:170-230` (`renderQkv`)
- Modify: `js/attention/scenes.js:232-320` (`renderScores`)
- Modify: `js/attention/scenes.js:383-418` (`renderWsum`)

**Interfaces:**
- Consumes: `result.X`, `result.Q`, `result.K`, `result.V` as position-indexed arrays (from Task 1's `computePipeline`); `result.qkvFocus`, `result.scoresQFocus`, `result.scoresKFocus`, `result.wsumFocus` (unchanged, already position indices, set in `main.js`).
- Produces: no new exports — these three renderers become correct for repeated tokens without any interface changes visible to `main.js` or other renderers.

- [ ] **Step 1: Fix `renderQkv`**

In `js/attention/scenes.js`, inside `renderQkv` (starts at line 170), find:

```js
  const xMatrix = result.tokens.map((t) => result.embeddings[t]);
  const qMatrix = result.tokens.map((t) => result.Q[t]);
  const kMatrix = result.tokens.map((t) => result.K[t]);
  const vMatrix = result.tokens.map((t) => result.V[t]);
```

Replace with:

```js
  const xMatrix = result.X;
  const qMatrix = result.Q;
  const kMatrix = result.K;
  const vMatrix = result.V;
```

`xMatrix` changes from `result.embeddings` to `result.X` deliberately, not just a rename: `Q`/`K`/`V` are now computed from `X` (embeddings scaled by `sqrt(d)` plus positional encoding), not from the raw embeddings, so the matrix labeled "$X$" in this stage must show the same values `Q`/`K`/`V` were actually derived from, or the stage would display a `Q`/`K`/`V` inconsistent with its own shown input.

A few lines down in the same function, find:

```js
       ${labeledVecBlock(`$q_{\\text{${t0}}}$`, result.Q[t0])}
       ${labeledVecBlock(`$k_{\\text{${t0}}}$`, result.K[t0])}
       ${labeledVecBlock(`$v_{\\text{${t0}}}$`, result.V[t0])}
```

Replace with:

```js
       ${labeledVecBlock(`$q_{\\text{${t0}}}$`, result.Q[focusIdx])}
       ${labeledVecBlock(`$k_{\\text{${t0}}}$`, result.K[focusIdx])}
       ${labeledVecBlock(`$v_{\\text{${t0}}}$`, result.V[focusIdx])}
```

(`t0` stays as the display label; only the data lookup switches from the string `t0` to the position index `focusIdx`, which this function already computes on its first line.)

- [ ] **Step 2: Fix `renderScores`**

In the same file, inside `renderScores` (starts at line 232), find:

```js
    `<div><div class="heatbar-block-title">$Q$: one row per token</div><div class="attn-row-select" data-role="scores-q-grid">${heatMatrixGrid(result.tokens.map((t) => result.Q[t]), { hiRow: qIdx, rowLabels: result.tokens })}</div></div>
     <div><div class="heatbar-block-title">$K$: one row per token</div><div class="attn-row-select" data-role="scores-k-grid">${heatMatrixGrid(result.tokens.map((t) => result.K[t]), { hiRow: kIdx, rowLabels: result.tokens })}</div></div>`,
```

Replace with:

```js
    `<div><div class="heatbar-block-title">$Q$: one row per token</div><div class="attn-row-select" data-role="scores-q-grid">${heatMatrixGrid(result.Q, { hiRow: qIdx, rowLabels: result.tokens })}</div></div>
     <div><div class="heatbar-block-title">$K$: one row per token</div><div class="attn-row-select" data-role="scores-k-grid">${heatMatrixGrid(result.K, { hiRow: kIdx, rowLabels: result.tokens })}</div></div>`,
```

A few lines down, find:

```js
    `<div><div class="heatbar-block-title">$q_{\\text{${tQ}}}$</div><div class="heatbar-list">${heatBarList(result.Q[tQ])}</div></div>
     <div><div class="heatbar-block-title">$k_{\\text{${tK}}}$</div><div class="heatbar-list">${heatBarList(result.K[tK])}</div></div>
     <div class="calc-line">${result.Q[tQ].map((qv, j) => `${qv.toFixed(2)}&times;${result.K[tK][j].toFixed(2)}`).join(' + ')} = <b>${result.scores[qIdx][kIdx].toFixed(2)}</b></div>`,
```

Replace with:

```js
    `<div><div class="heatbar-block-title">$q_{\\text{${tQ}}}$</div><div class="heatbar-list">${heatBarList(result.Q[qIdx])}</div></div>
     <div><div class="heatbar-block-title">$k_{\\text{${tK}}}$</div><div class="heatbar-list">${heatBarList(result.K[kIdx])}</div></div>
     <div class="calc-line">${result.Q[qIdx].map((qv, j) => `${qv.toFixed(2)}&times;${result.K[kIdx][j].toFixed(2)}`).join(' + ')} = <b>${result.scores[qIdx][kIdx].toFixed(2)}</b></div>`,
```

(`tQ`/`tK` stay as display labels; only the four `result.Q[tQ]`/`result.K[tK]` data lookups switch to `qIdx`/`kIdx`.)

- [ ] **Step 3: Fix `renderWsum`**

In the same file, inside `renderWsum` (starts at line 383), find:

```js
  const scaledVecs = result.tokens.map((t, j) => result.V[t].map((v) => v * rowWeights[j]));
```

Replace with:

```js
  const scaledVecs = result.V.map((v, j) => v.map((vk) => vk * rowWeights[j]));
```

A few lines down, find:

```js
     <div><div class="heatbar-block-title">$V$: one row per token</div>${heatMatrixGrid(result.tokens.map((t) => result.V[t]), { rowLabels: result.tokens })}</div>`,
```

Replace with:

```js
     <div><div class="heatbar-block-title">$V$: one row per token</div>${heatMatrixGrid(result.V, { rowLabels: result.tokens })}</div>`,
```

- [ ] **Step 4: Verify no string-keyed lookups remain outside `renderInput`**

Run: `grep -n "result\.Q\[t\|result\.K\[t\|result\.V\[t\|result\.embeddings\[t" js/attention/scenes.js`
Expected: only matches inside `renderInput` (lines 124-126), which Task 5 fixes. `renderQkv`, `renderScores`, `renderWsum` should have none.

- [ ] **Step 5: Verify in the browser**

Follow the `browser-preview` skill to serve the site locally and drive headless Firefox. For the `dog-chased-cat` preset specifically (select it via the preset picker), confirm:
- QKV's stage1 "$X$" matrix shows 5 distinct rows, with rows 0 and 3 (both "the") visibly different from each other (not identical).
- QKV's stage3 "$Q$"/"$K$"/"$V$" matrices are 5 rows each, rendered via the new `.mgrid.g5x4` CSS class from Task 3 (no unstyled/collapsed grid).
- Scores' stage1 "$Q$"/"$K$" matrices likewise show 5 distinct rows.
- Clicking Scores' Q-matrix row 3 (the second "the") and K-matrix row 0 (the first "the") updates stage2 to a cell with a real, non-zero computed value — confirming the two "the" positions produce genuinely different `Q`/`K` vectors, not a silent collision.
- Weighted sum's stage1 "$V$" matrix shows 5 distinct rows.
- 0 console errors, 0 horizontal overflow at 1400px and 375px.

- [ ] **Step 6: Commit**

```bash
git add js/attention/scenes.js
git commit -m "fix(attention): index Q/K/V/X by position in QKV, Scores, and Wsum renderers"
```

---

## Task 5: Input step rewrite — position addition, renumbered concept stage, RoPE/ALiBi research, blurb fixes

**Files:**
- Modify: `js/attention/scenes.js:123-152` (`renderInput`, full rewrite)
- Modify: `pages/attention.html:43` (Input's static blurb)

**Interfaces:**
- Consumes: `result.X`, `result.embeddings` (both arrays, from Task 1/2), `result.d`, `result.tokens`.
- Produces: no new exports; `renderInput`'s filmstrip goes from 2 stages (`01: STORAGE`, `02: CONCEPT`) to 4 (`01: STORAGE`, `02: TRANSFORM`, `03: CONCEPT`, `04: RELATED RESEARCH`).

- [ ] **Step 1: Replace `renderInput`**

In `js/attention/scenes.js`, replace the full current `renderInput` function (lines 123-152):

```js
function renderInput(container, stepId, result) {
  const storageBody = `<div class="heatbar-block-row">${result.tokens
    .map((t) => labeledVecBlock(`&quot;${t}&quot;`, result.embeddings[t]))
    .join('')}</div>`;
  const stage1 = stageCard(
    '01: STORAGE',
    'The three embeddings',
    `Every token in this worked example starts as a ${result.d}-number vector called an <b>embedding</b>. Stacked together, the embeddings below form $X$, the matrix the rest of this pipeline operates on.`,
    storageBody,
    `${result.tokens.length} tokens, ${result.d} numbers each, stored in full`,
    'stage-wide'
  );
  const stage2 = stageCard(
    '02: CONCEPT',
    'True attention head scale',
    null,
    `<div class="scale-stats">
       <div class="scale-stat"><div class="scale-stat-label">attention heads</div><div class="scale-stat-value">1 here. Real attention layers run many heads in parallel (12 in BERT-base, 32 to 96+ in larger models), each with its own smaller $W_Q$, $W_K$, $W_V$, concatenated back together through one more learned matrix, $W_O$. Multi-head attention is entirely absent from this page.</div></div>
       <div class="scale-stat"><div class="scale-stat-label">model dimension ($d$)</div><div class="scale-stat-value">${result.d} here. Real models run 768 (BERT-base, GPT-2 small) up to 12,288+ (large GPT-3-class models), or 4,096 to 16,384+ in modern LLMs.</div></div>
       <div class="scale-stat"><div class="scale-stat-label">layers</div><div class="scale-stat-value">1 here, this single attention operation. Real transformers stack dozens to over a hundred blocks (12 in BERT-base, 96+ in large GPT-3-class models, 80+ in the largest open models), each with its own attention and a feedforward layer.</div></div>
       <div class="scale-stat"><div class="scale-stat-label">sequence length</div><div class="scale-stat-value">${result.tokens.length} tokens here. Real context windows run from the low thousands historically up to hundreds of thousands or millions of tokens in current long-context models.</div></div>
       <div class="scale-stat"><div class="scale-stat-label">vocabulary</div><div class="scale-stat-value">${result.tokens.length} whole words here. Real models tokenize into tens to a few hundred thousand subword pieces instead.</div></div>
       <div class="scale-stat"><div class="scale-stat-label">parameters</div><div class="scale-stat-value">${3 * result.d * result.d} numbers here (three ${result.d}&times;${result.d} weight matrices). Real models run from millions up to hundreds of billions of parameters.</div></div>
       <div class="scale-stat"><div class="scale-stat-label">position</div><div class="scale-stat-value">Left out here to keep focus on attention itself. Real models always add positional information (sinusoidal, learned, or rotary/ALiBi-style schemes), since attention alone has no sense of token order.</div></div>
     </div>`,
    undefined,
    'stage-wide'
  );
  container.innerHTML = filmstrip([stage1, stage2]);
}
```

with:

```js
// Scans for the first two positions sharing an identical raw embedding -- i.e. a repeated
// word. Position-agnostic on purpose: only the "dog-chased-cat" preset has one, but this
// function makes no assumption about which preset is active, so the other two presets fall
// back to showing the general addition with no repeat comparison.
function findRepeatedPositions(embeddings) {
  for (let i = 0; i < embeddings.length; i++) {
    for (let j = i + 1; j < embeddings.length; j++) {
      if (embeddings[i].every((v, k) => v === embeddings[j][k])) return [i, j];
    }
  }
  return null;
}

function renderInput(container, stepId, result) {
  const storageBody = `<div class="heatbar-block-row">${result.tokens
    .map((t, i) => labeledVecBlock(`&quot;${t}&quot;`, result.embeddings[i]))
    .join('')}</div>`;
  const stage1 = stageCard(
    '01: STORAGE',
    'The three embeddings',
    `Every token in this worked example starts as a ${result.d}-number vector called an <b>embedding</b>. Stacked together, the embeddings below form $E$, the raw lookup-table rows this sentence selects, before anything else happens to them.`,
    storageBody,
    `${result.tokens.length} tokens, ${result.d} numbers each, stored in full`,
    'stage-wide'
  );
  const repeat = findRepeatedPositions(result.embeddings);
  const repeatHtml = repeat
    ? `<div class="heatbar-block-row">
         ${labeledVecBlock(`$X_{${repeat[0]}}$ (&quot;${result.tokens[repeat[0]]}&quot;)`, result.X[repeat[0]])}
         ${labeledVecBlock(`$X_{${repeat[1]}}$ (&quot;${result.tokens[repeat[1]]}&quot;)`, result.X[repeat[1]])}
       </div>`
    : '';
  const repeatNote = repeat
    ? `&quot;${result.tokens[repeat[0]]}&quot; appears at both position ${repeat[0]} and position ${repeat[1]} with an identical raw embedding; their final $X$ rows differ only because of position`
    : `every position gets the same treatment, whether or not a word repeats`;
  const stage2 = stageCard(
    '02: TRANSFORM',
    'Adding position',
    `Every embedding is scaled by $\\sqrt{d}$, then a positional encoding is added, so the model can tell tokens apart by where they sit and not just what they are.`,
    `<div class="formula">$$ X_i = \\sqrt{d}\\,E_i + PE(i) $$</div>
     <div class="formula">$$ PE(i)_{2k} = \\sin\\!\\left(\\frac{i}{10000^{2k/d}}\\right), \\quad PE(i)_{2k+1} = \\cos\\!\\left(\\frac{i}{10000^{2k/d}}\\right) $$</div>
     ${repeatHtml}
     <div class="heatbar-block-title">$X$: every position, after scaling and adding position</div>
     ${heatMatrixGrid(result.X, { rowLabels: result.tokens })}`,
    repeatNote,
    'stage-wide'
  );
  const stage3 = stageCard(
    '03: CONCEPT',
    'True attention head scale',
    null,
    `<div class="scale-stats">
       <div class="scale-stat"><div class="scale-stat-label">attention heads</div><div class="scale-stat-value">1 here. Real attention layers run many heads in parallel (12 in BERT-base, 32 to 96+ in larger models), each with its own smaller $W_Q$, $W_K$, $W_V$, concatenated back together through one more learned matrix, $W_O$. Multi-head attention is entirely absent from this page.</div></div>
       <div class="scale-stat"><div class="scale-stat-label">model dimension ($d$)</div><div class="scale-stat-value">${result.d} here. Real models run 768 (BERT-base, GPT-2 small) up to 12,288+ (large GPT-3-class models), or 4,096 to 16,384+ in modern LLMs.</div></div>
       <div class="scale-stat"><div class="scale-stat-label">layers</div><div class="scale-stat-value">1 here, this single attention operation. Real transformers stack dozens to over a hundred blocks (12 in BERT-base, 96+ in large GPT-3-class models, 80+ in the largest open models), each with its own attention and a feedforward layer.</div></div>
       <div class="scale-stat"><div class="scale-stat-label">sequence length</div><div class="scale-stat-value">${result.tokens.length} tokens here. Real context windows run from the low thousands historically up to hundreds of thousands or millions of tokens in current long-context models.</div></div>
       <div class="scale-stat"><div class="scale-stat-label">vocabulary</div><div class="scale-stat-value">${result.tokens.length} whole words here. Real models tokenize into tens to a few hundred thousand subword pieces instead.</div></div>
       <div class="scale-stat"><div class="scale-stat-label">parameters</div><div class="scale-stat-value">${3 * result.d * result.d} numbers here (three ${result.d}&times;${result.d} weight matrices). Real models run from millions up to hundreds of billions of parameters.</div></div>
       <div class="scale-stat"><div class="scale-stat-label">position</div><div class="scale-stat-value">Added here via sinusoidal positional encoding, computed exactly rather than learned. Real models add position the same way, or with a learned, rotary, or ALiBi-style variant instead.</div></div>
     </div>`,
    undefined,
    'stage-wide'
  );
  const stage4 = stageCard(
    '04: RELATED RESEARCH',
    'A rotation instead of an addition',
    null,
    `<p class="concept-box">The additive encoding above is the original approach, but it isn't what most current large language models use. <a href="https://arxiv.org/abs/2104.09864" target="_blank" rel="noopener">RoFormer: Enhanced Transformer with Rotary Position Embedding</a> (Su et al., 2021) introduced an alternative, RoPE, which rotates $Q$ and $K$ by an angle proportional to position instead of adding anything to the embedding. Rotating two vectors and then taking their dot product depends only on the angle between them, so every attention score ends up depending on relative distance rather than absolute position, structurally, not just in practice; that property, plus leaving vector magnitude untouched, is why RoPE is what LLaMA, GPT-NeoX, PaLM, and Falcon actually use instead of additive encoding. A third option, ALiBi, skips rotating $Q$/$K$ altogether and instead adds a distance-based penalty directly to the raw scores before softmax, the same place the causal mask already operates, just as a smooth decay instead of a hard cutoff.</p>`
  );
  container.innerHTML = filmstrip([stage1, stage2, stage3, stage4]);
}
```

Note the two other changes bundled into this replacement beyond adding stages 2 and 4:
- Stage 1's prose now calls the raw lookup rows `$E$` instead of `$X$` (matching the static blurb's own existing `$E$`/`$X$` distinction) and stage1's `storageBody` now maps with `(t, i) => result.embeddings[i]` instead of `(t) => result.embeddings[t]` — a position-indexed lookup, since `embeddings` is now an array (Task 2).
- Stage 3 (the renumbered scale-stats stage)'s "position" row wording changed from "Left out here..." to "Added here...", since it's no longer true that position is left out.

- [ ] **Step 2: Fix Input's static blurb**

In `pages/attention.html`, find (inside `<section class="panel" id="step-input">`):

```html
        <p>For an input to a trained model, the sentence is first broken into tokens, and each token needs to become a vector a matrix can act on. Embeddings live in a lookup table $E$, one row per vocabulary word: real ones run hundreds or thousands of dimensions per token, so a token can encode many independent aspects of itself at once (syntax, sense, tone, and whatever else training finds useful) instead of trying to collapse those aspects into only a handful of numbers. The tokens extracted from this sentence become $X$, where $X \subset E$: not a separate object, just the handful of $E$'s rows this particular sentence selects. Training treats every entry of $E$ as a learnable parameter: after each prediction, backpropagation nudges the rows used in that sentence a little, $E \leftarrow E - \eta \dfrac{\partial \mathcal{L}}{\partial E}$, so tokens used in similar ways drift toward similar vectors over many examples. For this walkthrough the tokens and values are hand-picked instead, with positional information set aside to keep focus on attention itself; every token keeps the same color everywhere it appears below. See the planned Phase 3 for the training loop worked through in full.</p>
```

Replace with:

```html
        <p>For an input to a trained model, the sentence is first broken into tokens, and each token needs to become a vector a matrix can act on. Embeddings live in a lookup table $E$, one row per vocabulary word: real ones run hundreds or thousands of dimensions per token, so a token can encode many independent aspects of itself at once (syntax, sense, tone, and whatever else training finds useful) instead of trying to collapse those aspects into only a handful of numbers. The tokens extracted from this sentence select rows of $E$, then get a positional encoding added so the model can tell where each token sits, not just what it is; the result is $X$, the matrix the rest of this pipeline operates on. Training treats every entry of $E$ as a learnable parameter: after each prediction, backpropagation nudges the rows used in that sentence a little, $E \leftarrow E - \eta \dfrac{\partial \mathcal{L}}{\partial E}$, so tokens used in similar ways drift toward similar vectors over many examples. For this walkthrough the tokens and values are hand-picked instead; each position keeps its own consistent color everywhere it appears below, so a repeated word (two occurrences of the same token) shows up in two different colors, one per position. See the planned Phase 3 for the training loop worked through in full.</p>
```

This fixes both gaps caught in spec self-review: the sentence no longer claims positional information is "set aside" (it now describes the addition that actually happens), and it no longer claims every *token* keeps one color — colors are position-keyed, so a repeated token now correctly gets two different colors, one per occurrence.

- [ ] **Step 3: Verify no stale "set aside" or "same color" claims remain**

Run: `grep -n "set aside\|same color everywhere" pages/attention.html js/attention/scenes.js`
Expected: no matches (both phrasings were removed/reworded in Steps 1-2).

- [ ] **Step 4: Verify in the browser**

Using the `browser-preview` skill, for each of the three presets:
- Input now shows 4 stages: `01: STORAGE`, `02: TRANSFORM`, `03: CONCEPT`, `04: RELATED RESEARCH`.
- For `cat-sat` and `dog-ran-fast` (no repeated word): stage2's note reads "every position gets the same treatment..." and no `repeatHtml` comparison block renders.
- For `dog-chased-cat`: stage2 shows the two "the" comparison blocks (position 0 and position 3), with visibly different numbers in each, and the note names both positions.
- Stage2's formulas render via MathJax with no raw `$` left in the DOM text.
- Stage4's RoPE link points to `https://arxiv.org/abs/2104.09864` and opens correctly (`target="_blank"`).
- 0 console errors, 0 horizontal overflow at 1400px and 375px, for all three presets.

- [ ] **Step 5: Commit**

```bash
git add js/attention/scenes.js pages/attention.html
git commit -m "feat(attention): add position-encoding and RoPE/ALiBi stages to Input, fix stale blurb claims"
```

---

## Task 6: Full verification pass

**Files:** none (verification only).

**Interfaces:** none — this task only confirms the finished feature works end to end.

- [ ] **Step 1: Run the full test suite**

Run: `node --test dev/test/attention-math.test.js dev/test/attention-presets.test.js`
Expected: PASS, all tests green (7 from Task 1 + 5 from Task 2).

- [ ] **Step 2: Full headless-browser pass, all three presets, both breakpoints**

Follow the `browser-preview` skill. For each preset (`cat-sat`, `dog-ran-fast`, `dog-chased-cat`) and each viewport (1400px, 375px):
- Load the page, step through all 8 pipeline stops (Input, Q/K/V, QKᵀ scores, Mask, Softmax, Weighted sum, Output) via the pipeline bar.
- Confirm 0 console errors and 0 `document.documentElement.scrollWidth > document.documentElement.clientWidth` at each stop.
- Confirm every interactive element still works: Mask's causal toggle, Scores' "compute all cells" (and that it still fills Scale's grid too), Wsum's clickable attention-weights matrix, Scores' clickable Q/K matrices, QKV's clickable X matrix — click at least one row in each on the `dog-chased-cat` preset specifically (5 rows available, including both "the" positions) and confirm the dependent stage updates correctly.
- Screenshot Input's new stage2 (`02: TRANSFORM`) for all three presets, confirming the repeat-comparison only appears for `dog-chased-cat`.

- [ ] **Step 3: Spot-check the numbers against Task 1/2's computed values**

For the `dog-chased-cat` preset, confirm via the rendered DOM (not just trusting the test suite) that:
- `the` at position 0's attention weights peak at position 0 (self), matching the verified `peak=0.789 at the(0)`.
- `the` at position 3's attention weights peak at position 2 (`chased`), matching the verified `peak=0.856 at chased(2)` — i.e., the two "the" positions genuinely attend differently.

- [ ] **Step 4: Fix anything found, then re-run Steps 1-3**

If any check fails, fix the specific issue (do not weaken a test or skip a check to make it pass) and re-run the full verification from Step 1.

- [ ] **Step 5: Update the progress ledger**

Append an entry to `.superpowers/sdd/progress.md` (git-ignored, this session's running record) summarizing what shipped, mirroring every other entry's level of detail: what changed, what was verified, and the final commit hashes for this feature.
