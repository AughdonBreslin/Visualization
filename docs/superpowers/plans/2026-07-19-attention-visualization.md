# Attention Mechanism Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the site's tenth page — a deep, interactive explainer for single-head scaled
dot-product attention (forward pass only), where a sticky pipeline diagram is the primary
navigation and each of its eight steps expands into a fully worked, granular, scrubbable
numeric walkthrough.

**Architecture:** Static site, no build step, no test framework (confirmed: no `package.json` in
the repo). `js/attention/` is an ES-module directory (following the `js/manifold/` precedent, not
`gradient-descent.js`'s single-file pattern, because this page is materially larger): a pure,
DOM-free math layer, pure SVG-string glyph builders, a sticky-pipeline-bar module, a per-step
scene-animation module, and an entry point. The rail reuses the site's existing
`js/section-outline.js` unmodified. All state lives in module-scope JS variables recomputed from
one `computePipeline()` call; there is no framework and no virtual DOM — scenes are re-rendered by
clearing and rebuilding their `innerHTML`, the same pattern `gradient-descent.js` and
`manifold_isomap.js` already use.

**Tech Stack:** Vanilla ES modules, inline SVG, MathJax v3 (CDN, `tex-svg.js`) for formulas, the
site's existing design tokens (`styles/tokens.css`, `styles/system.css`, `styles/components.css`,
`styles/article-ui.css`, `styles/section-outline.css`). No new dependencies.

## Global Constraints

- No build tooling exists in this repo and none is being added. "Tests" for the pure math and
  glyph modules are plain Node scripts run with `node`, asserting with the built-in `assert`
  module — there is no test runner, no `pytest`, no `jest`. Every other task's verification is
  manual, in-browser, using the headless Firefox + Playwright harness already proven to work in
  this environment for this project (see the gradient-descent QA precedent) — start a static
  server (`python3 -m http.server`) and script the checks; there is no interactive display in
  this environment, so verification never means "look at it yourself," it means script an
  assertion against the rendered DOM or a screenshot.
- Match `pages/gradient-descent.html` / `pages/manifold.html` conventions exactly for anything not
  specified here: `<html lang="en">`, the `.ui <pagename> has-section-outline` body classes, the
  `page-head` / `eyebrow` / `lede` header, `<main class="article-body">` wrapping
  `<section class="panel">` blocks each with a leading `<h2>`, `<footer class="site-footer">`.
- Every embedded number in the worked example must come from the verified computation in Task 1,
  never a hand-typed placeholder — Task 1's own test is the source of truth other tasks check
  their rendered numbers against.
- No em-dashes, no emphasis markup (bold/italic) in any HTML copy text (site-wide prose
  convention) — CSS `font-weight` is fine, `<b>`/`<strong>`/`<em>` in prose is not.
- The eight step ids, used verbatim as `data-step` / DOM id suffixes everywhere in this plan:
  `input`, `qkv`, `scores`, `scale`, `mask`, `softmax`, `wsum`, `output`.

---

## File structure

```
pages/attention.html          new — page shell, static prose, 8 <section class="panel"> scenes
styles/attention.css          new — sticky bar, scene layout, hero glyph, heat-grid, animations
js/attention/presets.js       new — worked-example token sets, embeddings, weight matrices
js/attention/math.js          new — pure step computations (no DOM); the only unit-testable layer
js/attention/glyphs.js        new — pure SVG-string builders for the 8 node icons + connectors
js/attention/pipeline.js      new — sticky bar, open-node behavior, rail integration
js/attention/scenes.js        new — per-step scene rendering + play/step/scrub animation
js/attention/main.js          new — entry point: wires presets.js + math.js into pipeline.js/scenes.js
index.html                    modify — add entry 10 card
```

---

### Task 1: Worked-example data and the pure math layer

**Files:**
- Create: `js/attention/presets.js`
- Create: `js/attention/math.js`
- Test: `/home/audie/Visualization/.scratch/test-math.mjs` (throwaway, not committed — Node has
  no test runner in this repo, so verification scripts live outside the repo and are deleted
  after use, per Global Constraints)

**Interfaces:**
- Produces (consumed by every later task):
  - `presets.js`: `export const PRESETS` — array of `{ id: string, label: string, tokens: string[], embeddings: {[token: string]: number[]} }`
  - `presets.js`: `export const WEIGHTS` — `{ WQ: number[][], WK: number[][], WV: number[][] }`
  - `presets.js`: `export const TOKEN_COLORS` — `string[]` of CSS color values, indexed by token
    position (not identity)
  - `math.js`: `export function linearProject(W: number[][], x: number[]): number[]`
  - `math.js`: `export function dot(a: number[], b: number[]): number`
  - `math.js`: `export function projectAll(tokens: string[], embeddings: object, W: number[][]): object`
  - `math.js`: `export function scoreMatrix(tokens: string[], Q: object, K: object): number[][]`
  - `math.js`: `export function scaleMatrix(scores: number[][], d: number): number[][]`
  - `math.js`: `export const NEG_INF = -1e9`
  - `math.js`: `export function applyCausalMask(scaled: number[][]): number[][]`
  - `math.js`: `export function softmaxRow(row: number[]): number[]`
  - `math.js`: `export function softmaxMatrix(matrix: number[][]): number[][]`
  - `math.js`: `export function weightedSum(tokens: string[], weights: number[][], V: object, d: number): number[][]`
  - `math.js`: `export function computePipeline(tokens: string[], embeddings: object, weights: {WQ,WK,WV}, options?: {causal?: boolean}): PipelineResult`
    where `PipelineResult = { tokens, embeddings, Q, K, V, scores, scaled, masked, weights, output, d, causal }`
    (`Q`/`K`/`V` are `{[token]: number[]}`; `scores`/`scaled`/`masked`/`weights`/`output` are
    `number[][]` indexed `[queryTokenIndex][keyTokenIndex]`, except `output` which is
    `number[][]` indexed `[tokenIndex][dimIndex]`)

- [ ] **Step 1: Write `js/attention/presets.js`**

```js
// js/attention/presets.js
// Worked-example data for the attention page. Two curated presets (no free-form editing in
// this phase — see docs/superpowers/specs/2026-07-19-attention-visualization-design.md).
// Weight matrices are shared across presets: hand-picked, not learned, chosen so the resulting
// attention pattern is clearly peaked (not near-uniform) after softmax.

export const PRESETS = [
  {
    id: 'cat-sat',
    label: '"the cat sat"',
    tokens: ['the', 'cat', 'sat'],
    embeddings: {
      the: [0.2, 0.8, 0.1, 0.4],
      cat: [0.9, 0.1, 0.6, 0.3],
      sat: [0.3, 0.5, 0.8, 0.2],
    },
  },
  {
    id: 'dog-ran-fast',
    label: '"dog ran fast"',
    tokens: ['dog', 'ran', 'fast'],
    embeddings: {
      dog: [0.7, 0.2, 0.3, 0.6],
      ran: [0.4, 0.6, 0.7, 0.1],
      fast: [0.2, 0.3, 0.5, 0.8],
    },
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

// Indexed by token POSITION (0/1/2), not by token identity, so recoloring stays stable when
// the preset changes which words are used.
export const TOKEN_COLORS = ['#7c8fff', '#e0b341', '#4fd1a5'];
```

- [ ] **Step 2: Write `js/attention/math.js`**

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

export function projectAll(tokens, embeddings, W) {
  const out = {};
  for (const t of tokens) out[t] = linearProject(W, embeddings[t]);
  return out;
}

export function scoreMatrix(tokens, Q, K) {
  return tokens.map((ti) => tokens.map((tj) => dot(Q[ti], K[tj])));
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

export function weightedSum(tokens, weights, V, d) {
  return tokens.map((ti, i) => {
    const out = new Array(d).fill(0);
    tokens.forEach((tj, j) => {
      const w = weights[i][j];
      V[tj].forEach((vk, k) => {
        out[k] += w * vk;
      });
    });
    return out;
  });
}

export function computePipeline(tokens, embeddings, weights, options = {}) {
  const causal = !!options.causal;
  const d = embeddings[tokens[0]].length;
  const Q = projectAll(tokens, embeddings, weights.WQ);
  const K = projectAll(tokens, embeddings, weights.WK);
  const V = projectAll(tokens, embeddings, weights.WV);
  const scores = scoreMatrix(tokens, Q, K);
  const scaled = scaleMatrix(scores, d);
  const masked = causal ? applyCausalMask(scaled) : scaled;
  const weightsOut = softmaxMatrix(masked);
  const output = weightedSum(tokens, weightsOut, V, d);
  return { tokens, embeddings, Q, K, V, scores, scaled, masked, weights: weightsOut, output, d, causal };
}
```

- [ ] **Step 3: Write the verification script (not committed)**

Create `/home/audie/Visualization/.scratch/test-math.mjs`:

```js
import assert from 'node:assert/strict';
import { computePipeline, softmaxRow, applyCausalMask, NEG_INF } from '../js/attention/math.js';
import { PRESETS, WEIGHTS } from '../js/attention/presets.js';

function close(a, b, eps = 1e-3) { return Math.abs(a - b) < eps; }

const preset = PRESETS.find((p) => p.id === 'cat-sat');
const result = computePipeline(preset.tokens, preset.embeddings, WEIGHTS);

// Hand-verified against a standalone derivation (see plan Task 1): "cat" (row index 1)
// attends most to itself, a clearly peaked (not near-uniform) softmax row.
assert.ok(close(result.weights[1][1], 0.503, 5e-3), `expected cat-self weight ~0.503, got ${result.weights[1][1]}`);
assert.ok(result.weights[1][1] > result.weights[1][0] && result.weights[1][1] > result.weights[1][2],
  'cat should attend most to itself');

// Every softmax row sums to 1.
for (const row of result.weights) {
  const sum = row.reduce((a, b) => a + b, 0);
  assert.ok(close(sum, 1, 1e-9), `row should sum to 1, got ${sum}`);
}

// Scale is dividing by sqrt(d) = 2 for d = 4.
assert.equal(result.d, 4);
assert.ok(close(result.scaled[0][0], result.scores[0][0] / 2), 'scale should divide by sqrt(d)');

// Output has one row per token, each of dimension d, and is finite.
assert.equal(result.output.length, preset.tokens.length);
for (const row of result.output) {
  assert.equal(row.length, result.d);
  for (const v of row) assert.ok(Number.isFinite(v), 'output must be finite');
}

// Causal masking: token 0 can only attend to itself.
const causal = computePipeline(preset.tokens, preset.embeddings, WEIGHTS, { causal: true });
assert.ok(close(causal.weights[0][0], 1), 'first token must attend entirely to itself under causal mask');
assert.ok(close(causal.weights[0][1], 0) && close(causal.weights[0][2], 0));
assert.equal(causal.masked[0][1], NEG_INF);

// softmaxRow on an all-equal row is uniform.
const uniform = softmaxRow([2, 2, 2]);
for (const v of uniform) assert.ok(close(v, 1 / 3));

// Property test across every preset: computePipeline never produces NaN/Infinity, and every
// softmax row is a valid probability distribution, for both masking modes.
for (const p of PRESETS) {
  for (const causalOpt of [false, true]) {
    const r = computePipeline(p.tokens, p.embeddings, WEIGHTS, { causal: causalOpt });
    for (const row of r.weights) {
      const sum = row.reduce((a, b) => a + b, 0);
      assert.ok(close(sum, 1, 1e-9), `preset ${p.id} causal=${causalOpt}: row must sum to 1`);
      for (const v of row) assert.ok(Number.isFinite(v) && v >= 0 && v <= 1);
    }
    for (const row of r.output) for (const v of row) assert.ok(Number.isFinite(v));
  }
}

console.log('ALL PASS');
```

- [ ] **Step 4: Run it and confirm every assertion passes**

```bash
mkdir -p /home/audie/Visualization/.scratch
node /home/audie/Visualization/.scratch/test-math.mjs
```

Expected output: `ALL PASS` with no thrown `AssertionError`. If the self-attention assertion
fails, do not change the expected value to match the code — recheck the weight matrices in
`presets.js` against this plan's Step 1 verbatim, the discrepancy is in the data, not the test.

- [ ] **Step 5: Delete the scratch test and commit**

```bash
rm -rf /home/audie/Visualization/.scratch
git add js/attention/presets.js js/attention/math.js
git commit -m "feat(attention): add worked-example presets and pure attention math layer"
```

---

### Task 2: SVG glyph and connector builders

**Files:**
- Create: `js/attention/glyphs.js`
- Test: `/home/audie/Visualization/.scratch/test-glyphs.mjs` (throwaway)

**Interfaces:**
- Consumes: nothing from Task 1 (pure, standalone; takes token colors as a parameter rather than
  importing `presets.js`, so it stays reusable if presets change independently)
- Produces (consumed by Task 4 and Task 5):
  - `export const STEP_IDS = ['input', 'qkv', 'scores', 'scale', 'mask', 'softmax', 'wsum', 'output']`
  - `export function glyphSVG(stepId: string, opts?: { size?: number }): string` — returns a
    complete `<svg>...</svg>` string for the given step id; throws if `stepId` is not in
    `STEP_IDS`
  - `export function connectorsSVG(count: number, opts?: { width?: number, height?: number }): string`
    — returns the inner SVG markup (defs + paths) for `count - 1` curved connectors between
    `count` evenly spaced nodes

- [ ] **Step 1: Write `js/attention/glyphs.js`**

```js
// js/attention/glyphs.js
// Pure SVG-string builders for the pipeline's node icons and connector arrows. No DOM access —
// every function here takes plain data in and returns a markup string, so it is testable with
// plain Node asserts (string/shape checks) without a browser.

export const STEP_IDS = ['input', 'qkv', 'scores', 'scale', 'mask', 'softmax', 'wsum', 'output'];

const DEFAULT_TOK = ['#7c8fff', '#e0b341', '#4fd1a5'];

function svgInput(tok) {
  return `<svg viewBox="0 0 40 40">
    <rect x="2" y="5" width="26" height="7" rx="2" fill="${tok[0]}" opacity="0.85"/>
    <rect x="2" y="17" width="20" height="7" rx="2" fill="${tok[1]}" opacity="0.85"/>
    <rect x="2" y="29" width="24" height="7" rx="2" fill="${tok[2]}" opacity="0.85"/>
  </svg>`;
}

function svgQkv(tok) {
  return `<svg viewBox="0 0 40 40">
    <line x1="0" y1="20" x2="10" y2="20" stroke="var(--hairline-strong)" stroke-width="2"/>
    <rect class="node-box" x="10" y="10" width="16" height="20" rx="3" fill="none" stroke="var(--text-muted)" stroke-width="1.6"/>
    <line x1="26" y1="14" x2="37" y2="7" stroke="${tok[0]}" stroke-width="1.6"/>
    <line x1="26" y1="20" x2="37" y2="20" stroke="${tok[1]}" stroke-width="1.6"/>
    <line x1="26" y1="26" x2="37" y2="33" stroke="${tok[2]}" stroke-width="1.6"/>
    <circle cx="38" cy="7" r="2.4" fill="${tok[0]}"/><circle cx="38" cy="20" r="2.4" fill="${tok[1]}"/><circle cx="38" cy="33" r="2.4" fill="${tok[2]}"/>
  </svg>`;
}

function gridCells(vals, cell, gap, offsetX, offsetY, opacityOf) {
  let out = '';
  vals.forEach((v, i) => {
    const x = (i % 3) * (cell + gap) + offsetX;
    const y = Math.floor(i / 3) * (cell + gap) + offsetY;
    out += `<rect x="${x}" y="${y}" width="${cell}" height="${cell}" rx="1.5" fill="var(--accent)" opacity="${opacityOf(v)}"/>`;
  });
  return out;
}

const SCORE_VALS = [0.9, 0.3, 0.15, 0.25, 0.85, 0.2, 0.15, 0.3, 0.9];

function svgScores() {
  const cells = gridCells(SCORE_VALS, 10, 2, 4, 4, (v) => (0.15 + v * 0.65).toFixed(2));
  return `<svg viewBox="0 0 40 40"><g class="node-box">${cells}
    <rect x="2" y="2" width="36" height="36" rx="4" fill="none" stroke="var(--text-muted)" stroke-width="1.4"/></g></svg>`;
}

function svgScale() {
  const cells = gridCells(SCORE_VALS, 7, 2, 7, 2, (v) => (0.1 + v * 0.4).toFixed(2));
  return `<svg viewBox="0 0 40 44">
    <g class="node-box">${cells}<rect x="6" y="0" width="28" height="28" rx="3" fill="none" stroke="var(--text-muted)" stroke-width="1.3"/></g>
    <text x="20" y="40" font-size="9" font-family="var(--font-mono)" fill="var(--text-muted)" text-anchor="middle">&#247;&#8730;d</text>
  </svg>`;
}

function svgMask() {
  // Lower-triangle + diagonal cells stay tinted (visible); upper-triangle (future positions,
  // i < j) are drawn as flat dark cells with a diagonal strike, previewing what softmax is
  // about to zero out.
  let cells = '';
  SCORE_VALS.forEach((v, i) => {
    const row = Math.floor(i / 3);
    const col = i % 3;
    const x = col * 12 + 4;
    const y = row * 12 + 4;
    if (col > row) {
      cells += `<rect x="${x}" y="${y}" width="10" height="10" rx="1.5" fill="var(--surface)" stroke="var(--hairline-strong)" stroke-width="1"/>`;
      cells += `<line x1="${x + 2}" y1="${y + 2}" x2="${x + 8}" y2="${y + 8}" stroke="var(--text-muted)" stroke-width="1"/>`;
    } else {
      cells += `<rect x="${x}" y="${y}" width="10" height="10" rx="1.5" fill="var(--accent)" opacity="${(0.15 + v * 0.65).toFixed(2)}"/>`;
    }
  });
  return `<svg viewBox="0 0 40 40"><g class="node-box">${cells}
    <rect x="2" y="2" width="36" height="36" rx="4" fill="none" stroke="var(--text-muted)" stroke-width="1.4"/></g></svg>`;
}

function svgSoftmax() {
  return `<svg viewBox="0 0 40 40">
    <rect x="4" y="4" width="10" height="10" rx="1.5" fill="var(--accent)" opacity=".85"/>
    <rect x="16" y="4" width="10" height="10" rx="1.5" fill="var(--accent)" opacity=".2"/>
    <rect x="28" y="4" width="8" height="10" rx="1.5" fill="var(--accent)" opacity=".15"/>
    <rect x="4" y="16" width="8" height="6" rx="1.5" fill="var(--accent)" opacity=".2"/>
    <rect x="14" y="16" width="12" height="6" rx="1.5" fill="var(--accent)" opacity=".9"/>
    <rect x="28" y="16" width="8" height="6" rx="1.5" fill="var(--accent)" opacity=".2"/>
    <path d="M4 30 Q12 22 20 30 T36 30" fill="none" stroke="var(--text-muted)" stroke-width="1.4"/>
  </svg>`;
}

function svgWsum(tok) {
  return `<svg viewBox="0 0 40 40">
    <defs><linearGradient id="wg-grad" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="${tok[0]}"/><stop offset=".5" stop-color="${tok[1]}"/><stop offset="1" stop-color="${tok[2]}"/></linearGradient></defs>
    <rect x="1" y="4" width="8" height="6" rx="1.5" fill="${tok[0]}" opacity=".85"/>
    <rect x="1" y="13" width="8" height="6" rx="1.5" fill="${tok[1]}" opacity=".35"/>
    <rect x="1" y="22" width="8" height="6" rx="1.5" fill="${tok[2]}" opacity=".2"/>
    <line x1="11" y1="7" x2="17" y2="18" stroke="${tok[0]}" stroke-width="1.4"/>
    <line x1="11" y1="16" x2="17" y2="18" stroke="${tok[1]}" stroke-width="1.4"/>
    <line x1="11" y1="25" x2="17" y2="18" stroke="${tok[2]}" stroke-width="1.4"/>
    <circle class="node-box" cx="19" cy="18" r="5" fill="none" stroke="var(--text-muted)" stroke-width="1.4"/>
    <line x1="16" y1="18" x2="22" y2="18" stroke="var(--text-muted)" stroke-width="1.2"/>
    <line x1="19" y1="15" x2="19" y2="21" stroke="var(--text-muted)" stroke-width="1.2"/>
    <line x1="24" y1="18" x2="34" y2="18" stroke="var(--hairline-strong)" stroke-width="1.4"/>
    <rect x="34" y="13" width="5" height="10" rx="1.5" fill="url(#wg-grad)" opacity=".9"/>
  </svg>`;
}

function svgOutput(tok) {
  return `<svg viewBox="0 0 40 40">
    <defs><linearGradient id="out-grad" x1="0" x2="1"><stop offset="0" stop-color="${tok[0]}"/><stop offset=".5" stop-color="${tok[1]}"/><stop offset="1" stop-color="${tok[2]}"/></linearGradient></defs>
    <rect x="2" y="5" width="26" height="7" rx="2" fill="url(#out-grad)" opacity=".9"/>
    <rect x="2" y="17" width="22" height="7" rx="2" fill="url(#out-grad)" opacity=".9"/>
    <rect x="2" y="29" width="25" height="7" rx="2" fill="url(#out-grad)" opacity=".9"/>
  </svg>`;
}

const BUILDERS = {
  input: svgInput,
  qkv: svgQkv,
  scores: svgScores,
  scale: svgScale,
  mask: svgMask,
  softmax: svgSoftmax,
  wsum: svgWsum,
  output: svgOutput,
};

export function glyphSVG(stepId, opts = {}) {
  const builder = BUILDERS[stepId];
  if (!builder) throw new Error(`glyphSVG: unknown step id "${stepId}"`);
  const tok = opts.tokenColors || DEFAULT_TOK;
  return builder(tok);
}

export function connectorsSVG(count, opts = {}) {
  const width = opts.width || (count - 1) * 90 + 60;
  const height = opts.height || 60;
  const xs = Array.from({ length: count }, (_, i) => 30 + (i * (width - 60)) / (count - 1));
  let out =
    '<defs><marker id="attn-arrowhead" markerWidth="6" markerHeight="6" refX="4" refY="3" orient="auto">' +
    '<path d="M0,0 L6,3 L0,6 Z" fill="var(--hairline-strong)"/></marker></defs>';
  for (let i = 0; i < xs.length - 1; i++) {
    const x1 = xs[i] + 18;
    const x2 = xs[i + 1] - 20;
    const midY = height * 0.3;
    const dipY = height * 0.67;
    out += `<path d="M${x1} ${midY} C ${x1 + 40} ${dipY}, ${x2 - 40} ${dipY}, ${x2} ${midY}" fill="none" stroke="var(--hairline-strong)" stroke-width="1.3" marker-end="url(#attn-arrowhead)"/>`;
  }
  return out;
}
```

- [ ] **Step 2: Write and run the verification script**

Create `/home/audie/Visualization/.scratch/test-glyphs.mjs`:

```js
import assert from 'node:assert/strict';
import { STEP_IDS, glyphSVG, connectorsSVG } from '../js/attention/glyphs.js';

assert.equal(STEP_IDS.length, 8);

for (const id of STEP_IDS) {
  const svg = glyphSVG(id);
  assert.ok(svg.startsWith('<svg'), `${id} glyph should start with <svg`);
  assert.ok(svg.trim().endsWith('</svg>'), `${id} glyph should end with </svg>`);
  assert.ok(!svg.includes('NaN'), `${id} glyph must not contain NaN`);
  assert.ok(!svg.includes('undefined'), `${id} glyph must not contain undefined`);
}

assert.throws(() => glyphSVG('not-a-real-step'), /unknown step id/);

const custom = glyphSVG('input', { tokenColors: ['#111111', '#222222', '#333333'] });
assert.ok(custom.includes('#111111'), 'custom token colors should be used');

const connectors = connectorsSVG(8);
assert.ok(connectors.includes('<defs>'));
// Count connector arrows specifically (each carries marker-end), not the arrowhead marker's
// own <path> inside <defs> -- a plain /<path /g count would include that defs path too (8, not 7).
const arrowCount = (connectors.match(/marker-end="url\(#attn-arrowhead\)"/g) || []).length;
assert.equal(arrowCount, 7, 'connectorsSVG(8) should draw 7 arrows between 8 nodes');
assert.ok(!connectors.includes('NaN'));

console.log('ALL PASS');
```

```bash
mkdir -p /home/audie/Visualization/.scratch
node /home/audie/Visualization/.scratch/test-glyphs.mjs
```

Expected output: `ALL PASS`.

- [ ] **Step 3: Delete the scratch test and commit**

```bash
rm -rf /home/audie/Visualization/.scratch
git add js/attention/glyphs.js
git commit -m "feat(attention): add SVG glyph and connector builders for the 8 pipeline steps"
```

---

### Task 3: Page shell, base styles, and the home page entry

**Files:**
- Create: `pages/attention.html`
- Create: `styles/attention.css`
- Modify: `index.html` (add entry 10)

**Interfaces:**
- Consumes: nothing (static markup); later tasks (4-10) populate the empty containers this task
  creates: `#pipeBarOuter` / `#pipeBar` (Task 4), each `.scene-anim` div inside the eight
  `.panel` sections (Tasks 6-9), `#presetPicker` (Task 10)
- Produces: the DOM ids/classes every later task queries —
  `#pipeBarOuter`, `#pipeBar`, `#presetPicker`, and per step `section#step-<id>`,
  `#step-<id> .scene-anim`, `#step-<id> .scene-hero-glyph`, `#step-<id> .formula`

- [ ] **Step 1: Write `pages/attention.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Attention</title>
  <script>
    window.MathJax = {
      tex: { inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['$$', '$$']] },
      svg: { fontCache: 'global' }
    };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <link rel="stylesheet" href="../styles/tokens.css">
  <link rel="stylesheet" href="../styles/system.css">
  <link rel="stylesheet" href="../styles/components.css">
  <link rel="stylesheet" href="../styles/article-ui.css">
  <link rel="stylesheet" href="../styles/attention.css">
  <link rel="stylesheet" href="../styles/section-outline.css">
  <script src="../js/theme.js"></script>
  <script type="module" src="../js/attention/main.js"></script>
  <script defer src="../js/favicon.js"></script>
  <script type="module" src="../js/section-outline.js"></script>
</head>
<body class="ui attention has-section-outline">
  <div class="container">
    <header class="page-head">
      <div class="eyebrow">// Machine learning</div>
      <h1>Attention</h1>
      <p class="lede">Single-head scaled dot-product attention, worked end to end: click any step in the pipeline below, or just scroll, to see exactly what happens to the numbers at that stage.</p>
      <div class="preset-picker" id="presetPicker"></div>
    </header>

    <div class="pipe-bar-outer" id="pipeBarOuter">
      <div class="pipe-bar" id="pipeBar"></div>
    </div>

    <main class="article-body">

      <section class="panel" id="step-input">
        <h2>Input embeddings</h2>
        <div class="scene-hero-glyph"></div>
        <p>Before any attention math happens, each token needs to become a vector a matrix can act on. This step computes nothing about relationships between tokens yet: it's just the raw material every later step consumes. Each token gets a small vector; real models learn these from an embedding table (plus positional information, set aside here to keep focus on attention itself), so here they are hand-picked. Every token keeps the same color everywhere it appears below.</p>
        <div class="callout">
          <div class="callout-label">Note</div>
          <div class="callout-body">Real transformer embeddings run hundreds or thousands of dimensions; this page uses d = 4 so every number stays visible on screen. Nothing about the mechanism changes at higher dimension, only the width of every vector shown below.</div>
        </div>
        <div class="scene-anim"></div>
      </section>

      <section class="panel" id="step-qkv">
        <h2>Q / K / V projections</h2>
        <div class="scene-hero-glyph"></div>
        <p>A raw embedding conflates everything about a token into one vector. Attention needs three different views of each token: a query ("what am I looking for"), a key ("what do I offer"), and a value ("what I actually contribute if chosen"). Splitting one vector into three roles, via three learned matrices, is what makes the comparison in the next step meaningful instead of trivial.</p>
        <div class="callout">
          <div class="callout-label">Why three matrices, not one?</div>
          <div class="callout-body">If Q and K shared a matrix, every token's query would equal its own key, so every token would trivially attend most to itself. Separate projections let a token's query and key diverge, which is what lets it end up attending to a different token when that's more useful.</div>
        </div>
        <div class="scene-anim"></div>
        <div class="formula">$$ q_i = x_i W_Q, \quad k_i = x_i W_K, \quad v_i = x_i W_V $$</div>
      </section>

      <section class="panel" id="step-scores">
        <h2>QKᵀ scores</h2>
        <div class="scene-hero-glyph"></div>
        <p>With every token holding a query and a key, comparing a query against a key is a similarity measure. This step performs every such comparison at once: how relevant each token's content is to what every other token is looking for, before any normalization.</p>
        <div class="callout">
          <div class="callout-label">Note</div>
          <div class="callout-body">The raw score is unbounded, and grows with the query and key vectors' magnitude, which is exactly what the next step exists to control.</div>
        </div>
        <div class="scene-anim"></div>
        <div class="formula">$$ \text{score}_{ij} = q_i \cdot k_j $$</div>
      </section>

      <section class="panel" id="step-scale">
        <h2>Scale</h2>
        <div class="scene-hero-glyph"></div>
        <p>Every score divides by √d. Dot products grow with dimension, and large scores push softmax into a near one-hot regime with vanishing gradients almost everywhere else; dividing by √d keeps scores in a range where softmax stays sensitive to differences between them.</p>
        <div class="callout">
          <div class="callout-label">Why √d specifically?</div>
          <div class="callout-body">If Q and K entries have roughly unit variance, their dot product's variance grows proportional to d, so its standard deviation grows with √d. Dividing by √d is what keeps the score's scale roughly constant regardless of how large d is.</div>
        </div>
        <div class="scene-anim"></div>
        <div class="formula">$$ \text{scaled}_{ij} = \frac{\text{score}_{ij}}{\sqrt{d}} $$</div>
      </section>

      <section class="panel" id="step-mask">
        <h2>Mask</h2>
        <div class="scene-hero-glyph"></div>
        <p>Every step so far treats all tokens symmetrically: any token can see any other, past or future. That's fine for encoding a sentence you already have in full, but wrong for predicting the next token, since letting a model see the answer it's supposed to predict makes training meaningless. An optional causal mask enforces the one asymmetry a decoder needs: only allow looking backward, by setting every future-position score to negative infinity before softmax.</p>
        <div class="callout">
          <div class="callout-label">Why &minus;&infin; and not just 0?</div>
          <div class="callout-body">e<sup>0</sup> = 1, so a masked score of literal 0 would still receive real, nonzero attention weight after softmax, same as any other position with a score of 0. e<sup>&minus;&infin;</sup> = 0 exactly, the only value guaranteed to zero out that position's contribution regardless of the other scores in the row.</div>
        </div>
        <div class="scene-anim"></div>
      </section>

      <section class="panel" id="step-softmax">
        <h2>Softmax</h2>
        <div class="scene-hero-glyph"></div>
        <p>Each row exponentiates and normalizes into a probability distribution over keys: the actual attention weights, always summing to one per query token.</p>
        <div class="callout">
          <div class="callout-label">Note</div>
          <div class="callout-body">The exponential is what makes softmax amplify differences: a score only slightly larger than its neighbors can end up with a much larger share of attention weight after exponentiating, part of why the Scale step matters, since unscaled scores would make softmax nearly one-hot almost everywhere.</div>
        </div>
        <div class="scene-anim"></div>
        <div class="formula">$$ \text{weight}_{ij} = \frac{e^{\text{scaled}_{ij}}}{\sum_k e^{\text{scaled}_{ik}}} $$</div>
      </section>

      <section class="panel" id="step-wsum">
        <h2>Weighted sum</h2>
        <div class="scene-hero-glyph"></div>
        <p>Now that there is a legitimate probability distribution over how much to listen to each token, the actual listening happens by blending: each token's output is the attention-weighted combination of every value vector, mostly its highest-weighted neighbors, a little of everything else.</p>
        <div class="callout">
          <div class="callout-label">Why value vectors, not the original embeddings?</div>
          <div class="callout-body">Like Q and K, V is its own learned projection, so the model can choose what a token actually contributes to others independent of what makes that token a good match (its key) or what it's searching for (its query).</div>
        </div>
        <div class="scene-anim"></div>
        <div class="formula">$$ o_i = \sum_j \text{weight}_{ij} \, v_j $$</div>
      </section>

      <section class="panel" id="step-output">
        <h2>Output</h2>
        <div class="scene-hero-glyph"></div>
        <p>Three vectors out, the same shape as the three vectors in: each one now a context-aware blend of the whole sequence rather than the token in isolation.</p>
        <div class="callout">
          <div class="callout-label">Note</div>
          <div class="callout-body">This output is usually not the end of a transformer block on its own; in a real model it typically continues through a residual connection and a feed-forward layer, both outside the scope of this page, which focuses specifically on the attention operation itself.</div>
        </div>
        <div class="scene-anim"></div>
      </section>

    </main>
    <footer class="site-footer"><span class="credit">Created by <a href="https://linkedin.com/in/aughdon/">Aughdon Breslin</a></span></footer>
  </div>
</body>
</html>
```

- [ ] **Step 2: Write `styles/attention.css` (base layout only — animation-specific rules are added in Tasks 4-9)**

```css
/* styles/attention.css
 * Attention page: sticky pipeline bar (primary nav), per-step scenes, worked-example visuals.
 * Loaded after tokens/system/components/article-ui, same order as every other page's stylesheet.
 */

.ui.attention .preset-picker { margin-top: 14px; }

/* --- sticky pipeline bar --- */
.ui.attention .pipe-bar-outer {
  position: sticky;
  top: 0;
  z-index: 20;
  background: rgba(12, 12, 13, 0.92);
  backdrop-filter: blur(6px);
  border-bottom: 1px solid var(--hairline);
  margin: 24px 0 0;
}
.ui.attention .pipe-bar {
  position: relative;
  max-width: var(--container-default);
  margin: 0 auto;
  padding: 12px 24px 8px;
}
.ui.attention .pipe-svg {
  position: absolute;
  inset: 0;
  display: block;
  width: 100%;
  height: 100%;
  overflow: hidden;
  pointer-events: none;
}
.ui.attention .pipe-row { position: relative; display: flex; justify-content: space-between; }
.ui.attention .pipe-node {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  width: 50px;
  padding: 2px 0;
  cursor: pointer;
  background: none;
  border: none;
}
.ui.attention .pipe-node svg { display: block; overflow: visible; transition: transform 260ms var(--ease-out); }
.ui.attention .pipe-node-label {
  font: 600 7.5px/1.3 var(--font-mono);
  letter-spacing: .02em;
  text-transform: uppercase;
  color: var(--text-muted);
  text-align: center;
  white-space: nowrap;
  transition: color 160ms var(--ease-out);
}
.ui.attention .pipe-node .node-box { transition: filter 160ms var(--ease-out), stroke 160ms var(--ease-out); }
.ui.attention .pipe-node:hover .node-box { stroke: var(--accent-link); }
.ui.attention .pipe-node.pulse svg { transform: scale(1.3); }
.ui.attention .pipe-node.pulse .node-box { stroke: var(--accent); filter: drop-shadow(0 0 8px rgba(var(--accent-rgb), .8)); }
.ui.attention .pipe-node.current .node-box { stroke: var(--accent); }
.ui.attention .pipe-node.current .pipe-node-label { color: var(--accent-link); }

/* --- per-step scenes --- */
.ui.attention .scene-hero-glyph {
  width: 76px;
  height: 76px;
  padding: 12px;
  border: 1px solid var(--surface-strong-border);
  background: var(--surface-strong);
  border-radius: var(--radius-lg);
  margin-bottom: 14px;
  transition: box-shadow 200ms var(--ease-out);
}
.ui.attention .scene-hero-glyph svg { width: 100%; height: 100%; overflow: visible; }
@keyframes attn-arrive-glow {
  0% { box-shadow: 0 0 0 0 rgba(var(--accent-rgb), 0); transform: scale(1); }
  30% { box-shadow: 0 0 0 8px rgba(var(--accent-rgb), .35); transform: scale(1.05); }
  100% { box-shadow: 0 0 0 0 rgba(var(--accent-rgb), 0); transform: scale(1); }
}
.ui.attention .scene-hero-glyph.just-arrived { animation: attn-arrive-glow 900ms var(--ease-out); border-color: rgba(var(--accent-rgb), .5); }

.ui.attention .scene-anim {
  background: #08080a;
  border: 1px solid var(--hairline);
  border-radius: var(--radius-md);
  padding: 18px 20px;
  margin: 16px 0;
}

/* --- heat-grid (shared by scores / scale / mask / softmax scenes) --- */
.ui.attention .heat-grid { display: grid; grid-template-columns: repeat(3, 44px); grid-template-rows: repeat(3, 44px); gap: 4px; }
.ui.attention .heat-cell {
  display: flex; align-items: center; justify-content: center;
  font: 600 12px/1.3 var(--font-mono); border-radius: 4px; color: #fff;
  transition: background 300ms var(--ease-out), opacity 300ms var(--ease-out);
}
.ui.attention .heat-cell.masked { background: var(--surface) !important; color: var(--text-muted); }
.ui.attention .anim-controls { display: flex; gap: 8px; margin-top: 14px; align-items: center; }
.ui.attention .anim-btn {
  font: 600 11px/1 var(--font-mono); color: var(--text-2);
  border: 1px solid var(--hairline); background: transparent;
  border-radius: var(--radius-sm); padding: 6px 10px; cursor: pointer;
}
.ui.attention .anim-btn:hover { border-color: var(--accent-link); color: var(--accent-link); }
.ui.attention .anim-scrub { flex: 1; }

@media (max-width: 640px) {
  .ui.attention .heat-grid { grid-template-columns: repeat(3, 30px); grid-template-rows: repeat(3, 30px); }
  .ui.attention .heat-cell { font-size: 10px; }
}
```

- [ ] **Step 3: Add the home page entry 10 to `index.html`**

Insert as the last `<li>` inside `<ul class="project-list">`, immediately after the Manifold
Learning entry (`pr-n">09`):

```html
        <li>
          <a class="project-row" href="pages/attention.html">
            <span class="pr-n">10</span>
            <span class="pr-main">
              <span class="pr-title">Attention</span>
              <span class="pr-desc">Single-head scaled dot-product attention worked end to end: a clickable pipeline diagram where every step expands into the real numbers, from Q/K/V projections through masking, softmax, and the weighted sum.</span>
            </span>
            <span class="pr-cat">Machine learning</span>
          </a>
        </li>
```

- [ ] **Step 4: Create a minimal `js/attention/main.js` stub so the page doesn't 404 on its own script**

```js
// js/attention/main.js
// Entry point stub; wired up fully in Task 4 onward.
console.log('attention page loaded');
```

- [ ] **Step 5: Manually verify the page loads with no console errors**

```bash
cd /home/audie/Visualization
python3 -m http.server 8842 &
sleep 1
curl -sf http://localhost:8842/pages/attention.html > /dev/null && echo "SERVER OK"
```

Using the headless Firefox + Playwright harness (see gradient-descent QA precedent for the exact
`LD_LIBRARY_PATH` / `firefoxUserPrefs` setup already proven in this environment):

```js
const { firefox } = require('playwright');
(async () => {
  const browser = await firefox.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto('http://localhost:8842/pages/attention.html', { waitUntil: 'networkidle' });
  console.log('title:', await page.title());
  console.log('h2 count:', await page.$$eval('main.article-body > section.panel', (els) => els.length));
  console.log('errors:', JSON.stringify(errors));
  await browser.close();
})();
```

Expected: `title: Attention`, `h2 count: 8`, `errors: []`.

```bash
kill %1
```

- [ ] **Step 6: Commit**

```bash
git add pages/attention.html styles/attention.css js/attention/main.js index.html
git commit -m "feat(attention): add page shell, base styles, and home page entry"
```

---

### Task 4: Sticky pipeline bar and unified open-node behavior

**Files:**
- Create: `js/attention/pipeline.js`
- Modify: `js/attention/main.js`

**Interfaces:**
- Consumes: `glyphSVG`, `connectorsSVG`, `STEP_IDS` from `js/attention/glyphs.js` (Task 2)
- Produces (consumed by Task 5 and Task 10):
  - `export const STEPS` — `Array<{ id: string, label: string }>`, the 8 steps in order, e.g.
    `{ id: 'qkv', label: 'Q / K / V' }`
  - `export function initPipelineBar(barEl: HTMLElement): void` — builds the bar into `barEl`,
    wires clicks (bar nodes + delegated rail clicks), sets up the current-step
    `IntersectionObserver`
  - `export function openNode(id: string): void` — pulses the matching bar node, smooth-scrolls
    to `#step-<id>`, and glows that section's `.scene-hero-glyph` on arrival

- [ ] **Step 1: Write `js/attention/pipeline.js`**

```js
// js/attention/pipeline.js
// The sticky pipeline bar: the page's primary navigation. Also wires the slim secondary rail
// (built by the shared js/section-outline.js) to trigger the identical open-node behavior,
// so both entry points feel like the same action, per
// docs/superpowers/specs/2026-07-19-attention-visualization-design.md.

import { STEP_IDS, glyphSVG, connectorsSVG } from './glyphs.js';

export const STEPS = [
  { id: 'input', label: 'Input' },
  { id: 'qkv', label: 'Q / K / V' },
  { id: 'scores', label: 'QKᵀ' },
  { id: 'scale', label: 'Scale' },
  { id: 'mask', label: 'Mask' },
  { id: 'softmax', label: 'Softmax' },
  { id: 'wsum', label: 'W. sum' },
  { id: 'output', label: 'Output' },
];

let barNodesById = new Map();

export function openNode(id) {
  const barNode = barNodesById.get(id);
  if (barNode) {
    barNode.classList.add('pulse');
    setTimeout(() => barNode.classList.remove('pulse'), 500);
  }
  const scene = document.getElementById(`step-${id}`);
  if (!scene) return;
  scene.scrollIntoView({ behavior: 'smooth', block: 'start' });
  setTimeout(() => {
    const glyph = scene.querySelector('.scene-hero-glyph');
    if (!glyph) return;
    glyph.classList.add('just-arrived');
    setTimeout(() => glyph.classList.remove('just-arrived'), 950);
  }, 550);
}

function buildBar(barEl) {
  barEl.innerHTML = `
    <svg class="pipe-svg" viewBox="0 0 700 60" preserveAspectRatio="none">${connectorsSVG(STEPS.length, { width: 700, height: 60 })}</svg>
    <div class="pipe-row"></div>
  `;
  const row = barEl.querySelector('.pipe-row');
  barNodesById = new Map();
  for (const step of STEPS) {
    const node = document.createElement('button');
    node.type = 'button';
    node.className = 'pipe-node';
    node.dataset.target = step.id;
    node.innerHTML = `${glyphSVG(step.id)}<div class="pipe-node-label">${step.label}</div>`;
    node.addEventListener('click', () => openNode(step.id));
    row.appendChild(node);
    barNodesById.set(step.id, node);
  }
}

function wireCurrentStepHighlighting() {
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const id = entry.target.id.replace('step-', '');
        for (const [nodeId, node] of barNodesById) {
          node.classList.toggle('current', nodeId === id);
        }
      }
    },
    { rootMargin: '-100px 0px -70% 0px' }
  );
  for (const id of STEP_IDS) {
    const scene = document.getElementById(`step-${id}`);
    if (scene) observer.observe(scene);
  }
}

// The rail is built by js/section-outline.js from the same <section class="panel"><h2> blocks
// used here, generating <a data-target="step-xxx"> links. Delegate on document.body (stable at
// load time) rather than querying the rail directly, since section-outline.js may build its DOM
// after this module runs.
function wireRailDelegation() {
  document.body.addEventListener('click', (e) => {
    const a = e.target.closest('.section-outline-list a[data-target]');
    if (!a) return;
    const id = a.dataset.target.replace('step-', '');
    if (!STEP_IDS.includes(id)) return; // not one of our panels (defensive; shouldn't happen)
    // Let section-outline.js's own handler run too (its e.preventDefault + navigateTo already
    // scrolls); this listener only adds the pulse/glow flourish on top of it.
    setTimeout(() => openNode(id), 0);
  });
}

export function initPipelineBar(barEl) {
  buildBar(barEl);
  wireCurrentStepHighlighting();
  wireRailDelegation();
}
```

- [ ] **Step 2: Wire it from `js/attention/main.js`**

```js
// js/attention/main.js
import { initPipelineBar } from './pipeline.js';

function init() {
  const barEl = document.getElementById('pipeBar');
  if (barEl) initPipelineBar(barEl);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
```

- [ ] **Step 3: Manually verify the bar renders and both nav paths work**

```bash
cd /home/audie/Visualization && python3 -m http.server 8842 &
sleep 1
```

```js
const { firefox } = require('playwright');
(async () => {
  const browser = await firefox.launch();
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('http://localhost:8842/pages/attention.html', { waitUntil: 'networkidle' });

  const nodeCount = await page.$$eval('#pipeBar .pipe-node', (els) => els.length);
  console.log('pipe node count:', nodeCount); // expect 8

  // Click the "scores" pipeline node; the matching scene's hero glyph should get the arrival glow.
  await page.click('#pipeBar .pipe-node[data-target="scores"]');
  await page.waitForTimeout(1200);
  const glowed = await page.$eval('#step-scores .scene-hero-glyph', (el) => el.classList.contains('just-arrived') || true);
  const scoresInView = await page.$eval('#step-scores', (el) => {
    const r = el.getBoundingClientRect();
    return r.top < window.innerHeight && r.bottom > 0;
  });
  console.log('scores section scrolled into view:', scoresInView);

  console.log('errors:', JSON.stringify(errors));
  await browser.close();
})();
```

Expected: `pipe node count: 8`, `scores section scrolled into view: true`, `errors: []`. Also
manually confirm (same script pattern) that clicking a rail entry (`.section-outline-list a`)
scrolls to the matching section — the rail only appears at >=1240px viewport width, so use that
viewport size for this check.

```bash
kill %1
```

- [ ] **Step 4: Commit**

```bash
git add js/attention/pipeline.js js/attention/main.js
git commit -m "feat(attention): add sticky pipeline bar and unified open-node navigation"
```

---

### Task 5: Scene rendering skeleton for all 8 steps

**Files:**
- Create: `js/attention/scenes.js`
- Modify: `js/attention/main.js`

**Interfaces:**
- Consumes: `computePipeline` from `math.js` (Task 1), `PRESETS`, `WEIGHTS`, `TOKEN_COLORS` from
  `presets.js` (Task 1), `STEPS` from `pipeline.js` (Task 4), `glyphSVG` from `glyphs.js` (Task 2)
- Produces (consumed by Tasks 6-9, which extend `renderScene`'s per-step branches, and Task 10,
  which calls `renderAllScenes` on preset change):
  - `export function renderAllScenes(result: PipelineResult): void` — fills every
    `.scene-hero-glyph` and, for now, a placeholder into every `.scene-anim`
  - `export function renderScene(stepId: string, result: PipelineResult): void` — renders one
    step's hero glyph + animation content; Tasks 6-9 replace this function's per-step branches
    with real animation code one step at a time

- [ ] **Step 1: Write `js/attention/scenes.js`**

```js
// js/attention/scenes.js
// Renders each step's hero glyph and worked-example animation. renderScene's per-step branches
// start as placeholders here and are filled in one step at a time by Tasks 6-9; the dispatch
// structure itself does not change after this task.

import { glyphSVG } from './glyphs.js';
import { STEPS } from './pipeline.js';

function renderPlaceholder(container, stepId) {
  container.innerHTML = `<p style="font:500 12px/1.6 var(--font-mono); color:var(--text-muted); margin:0;">step "${stepId}" animation not yet implemented</p>`;
}

const STEP_RENDERERS = {
  input: renderPlaceholder,
  qkv: renderPlaceholder,
  scores: renderPlaceholder,
  scale: renderPlaceholder,
  mask: renderPlaceholder,
  softmax: renderPlaceholder,
  wsum: renderPlaceholder,
  output: renderPlaceholder,
};

export function renderScene(stepId, result) {
  const scene = document.getElementById(`step-${stepId}`);
  if (!scene) return;
  const heroEl = scene.querySelector('.scene-hero-glyph');
  if (heroEl) heroEl.innerHTML = glyphSVG(stepId, { tokenColors: result.tokenColors });
  const animEl = scene.querySelector('.scene-anim');
  if (animEl) {
    const renderer = STEP_RENDERERS[stepId] || renderPlaceholder;
    renderer(animEl, stepId, result);
  }
}

export function renderAllScenes(result) {
  for (const step of STEPS) renderScene(step.id, result);
}
```

- [ ] **Step 2: Wire it from `js/attention/main.js`**

```js
// js/attention/main.js
import { initPipelineBar } from './pipeline.js';
import { renderAllScenes } from './scenes.js';
import { computePipeline } from './math.js';
import { PRESETS, WEIGHTS, TOKEN_COLORS } from './presets.js';

function buildResult(preset, options) {
  const result = computePipeline(preset.tokens, preset.embeddings, WEIGHTS, options);
  result.tokenColors = TOKEN_COLORS;
  return result;
}

function init() {
  const barEl = document.getElementById('pipeBar');
  if (barEl) initPipelineBar(barEl);

  const defaultPreset = PRESETS[0];
  const result = buildResult(defaultPreset, { causal: false });
  renderAllScenes(result);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
```

- [ ] **Step 3: Manually verify all 8 scenes render a hero glyph and placeholder content**

```bash
cd /home/audie/Visualization && python3 -m http.server 8842 &
sleep 1
```

```js
const { firefox } = require('playwright');
(async () => {
  const browser = await firefox.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('http://localhost:8842/pages/attention.html', { waitUntil: 'networkidle' });
  const ids = ['input', 'qkv', 'scores', 'scale', 'mask', 'softmax', 'wsum', 'output'];
  for (const id of ids) {
    const hasSvg = await page.$eval(`#step-${id} .scene-hero-glyph svg`, () => true).catch(() => false);
    const hasAnim = await page.$eval(`#step-${id} .scene-anim`, (el) => el.textContent.length > 0);
    console.log(id, 'hero svg:', hasSvg, 'anim content:', hasAnim);
  }
  console.log('errors:', JSON.stringify(errors));
  await browser.close();
})();
```

Expected: every step logs `hero svg: true anim content: true`, `errors: []`.

```bash
kill %1
```

- [ ] **Step 4: Commit**

```bash
git add js/attention/scenes.js js/attention/main.js
git commit -m "feat(attention): add scene rendering skeleton wired to the pipeline and math layer"
```

---

### Task 6: Input embeddings and Q/K/V projection animations

**Files:**
- Modify: `js/attention/scenes.js`
- Modify: `styles/attention.css`

**Interfaces:**
- Consumes: `result.embeddings`, `result.tokens`, `result.tokenColors`, `result.Q/K/V` from
  `PipelineResult` (Task 1's shape)
- Produces: no new exports; replaces the `input` and `qkv` entries in `STEP_RENDERERS`

- [ ] **Step 1: Replace the `input` renderer in `js/attention/scenes.js`**

```js
function renderInput(container, stepId, result) {
  const rows = result.tokens.map((t, i) => {
    const vec = result.embeddings[t].map((v) => v.toFixed(2)).join(', ');
    return `<div class="vec-row"><span class="vec-token" style="color:${result.tokenColors[i]}">"${t}"</span><span class="vec-values">[${vec}]</span></div>`;
  }).join('');
  container.innerHTML = `<div class="vec-list">${rows}</div>`;
}
```

Update `STEP_RENDERERS.input = renderInput;`.

- [ ] **Step 2: Replace the `qkv` renderer**

```js
function renderQkv(container, stepId, result) {
  const rows = result.tokens.map((t, i) => {
    const x = result.embeddings[t].map((v) => v.toFixed(2)).join(', ');
    const q = result.Q[t].map((v) => v.toFixed(2)).join(', ');
    const k = result.K[t].map((v) => v.toFixed(2)).join(', ');
    const v = result.V[t].map((v) => v.toFixed(2)).join(', ');
    return `
      <div class="qkv-row">
        <span class="vec-token" style="color:${result.tokenColors[i]}">"${t}"</span>
        <span class="qkv-eq">x=[${x}]</span>
        <span class="qkv-eq">q=[${q}]</span>
        <span class="qkv-eq">k=[${k}]</span>
        <span class="qkv-eq">v=[${v}]</span>
      </div>`;
  }).join('');
  container.innerHTML = `<div class="qkv-list">${rows}</div>
    <div class="anim-controls"><button class="anim-btn" type="button" data-role="step">step through per-token multiply-sum &#9654;</button></div>
    <div class="formula-worked" data-role="worked"></div>`;

  // Step control cycles which token's row is highlighted, to focus attention on one
  // multiply-then-sum at a time rather than showing all three at once. The formula-worked
  // line always mirrors whichever token is currently highlighted, so the abstract formula in
  // this step's .formula block has a live, real-number companion to trace against.
  const btn = container.querySelector('[data-role="step"]');
  const worked = container.querySelector('[data-role="worked"]');
  const rowsEls = Array.from(container.querySelectorAll('.qkv-row'));
  const fmt = (arr) => `[${arr.map((v) => v.toFixed(2)).join(', ')}]`;
  const showWorked = (i) => {
    const t = result.tokens[i];
    worked.textContent = `q_"${t}" = x_"${t}" · W_Q = ${fmt(result.embeddings[t])} · W_Q = ${fmt(result.Q[t])}`;
  };
  let idx = 0;
  showWorked(idx);
  btn.addEventListener('click', () => {
    rowsEls.forEach((r) => r.classList.remove('is-active'));
    idx = (idx + 1) % rowsEls.length;
    rowsEls[idx].classList.add('is-active');
    showWorked(idx);
  });
}
```

Update `STEP_RENDERERS.qkv = renderQkv;`.

- [ ] **Step 3: Add supporting styles to `styles/attention.css`**

```css
.ui.attention .vec-list, .ui.attention .qkv-list { display: flex; flex-direction: column; gap: 8px; }
.ui.attention .vec-row, .ui.attention .qkv-row {
  display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
  font: 500 12.5px/1.6 var(--font-mono); color: var(--text-body);
  padding: 4px 6px; border-radius: var(--radius-sm); transition: background 200ms var(--ease-out);
}
.ui.attention .qkv-row.is-active { background: var(--accent-muted); }
.ui.attention .vec-token { font-weight: 700; min-width: 44px; }
.ui.attention .vec-values, .ui.attention .qkv-eq { color: var(--text-2); }
.ui.attention .formula-worked {
  font: 500 12px/1.6 var(--font-mono); color: var(--accent-link);
  background: var(--accent-muted); border-radius: var(--radius-sm);
  padding: 8px 12px; margin-top: 12px; word-break: break-word;
}
```

- [ ] **Step 4: Manually verify**

```bash
cd /home/audie/Visualization && python3 -m http.server 8842 &
sleep 1
```

```js
const { firefox } = require('playwright');
(async () => {
  const browser = await firefox.launch();
  const page = await browser.newPage();
  await page.goto('http://localhost:8842/pages/attention.html', { waitUntil: 'networkidle' });
  const inputText = await page.$eval('#step-input .scene-anim', (el) => el.textContent);
  console.log('input scene mentions "the":', inputText.includes('the'));
  console.log('input scene mentions "0.20":', inputText.includes('0.20'));

  const workedInitial = await page.$eval('#step-qkv [data-role="worked"]', (el) => el.textContent);
  console.log('qkv formula-worked mentions default token "the":', workedInitial.includes('"the"'));
  console.log('qkv formula-worked shows real Q value 0.38:', workedInitial.includes('0.38'));

  await page.click('#step-qkv .anim-btn[data-role="step"]');
  const activeCount = await page.$$eval('#step-qkv .qkv-row.is-active', (els) => els.length);
  const workedAfterClick = await page.$eval('#step-qkv [data-role="worked"]', (el) => el.textContent);
  console.log('qkv active rows after one click:', activeCount); // expect 1
  console.log('qkv formula-worked updated after click:', workedAfterClick !== workedInitial);
  await browser.close();
})();
```

Expected: `input scene mentions "the": true`, `input scene mentions "0.20": true`,
`qkv formula-worked mentions default token "the": true`, `qkv formula-worked shows real Q value
0.38: true`, `qkv active rows after one click: 1`, `qkv formula-worked updated after click: true`.

```bash
kill %1
```

- [ ] **Step 5: Commit**

```bash
git add js/attention/scenes.js styles/attention.css
git commit -m "feat(attention): render input embeddings and Q/K/V projection scenes"
```

---

### Task 7: QKᵀ scores and Scale animations

**Files:**
- Modify: `js/attention/scenes.js`
- Modify: `styles/attention.css`

**Interfaces:**
- Consumes: `result.scores`, `result.scaled`, `result.d`, `result.tokens`, `result.tokenColors`

- [ ] **Step 1: Add a shared heat-grid builder and the `scores` renderer**

```js
function buildHeatGrid(matrix, tokens, opts = {}) {
  const { minOpacity = 0.15, maxOpacity = 0.8, formatter = (v) => v.toFixed(2), maskedCells = () => false } = opts;
  const flat = matrix.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const range = max - min || 1;
  let cells = '';
  matrix.forEach((row, i) => {
    row.forEach((v, j) => {
      if (maskedCells(i, j)) {
        cells += `<div class="heat-cell masked" data-row="${i}" data-col="${j}">&minus;&infin;</div>`;
        return;
      }
      const t = (v - min) / range;
      const opacity = (minOpacity + t * (maxOpacity - minOpacity)).toFixed(2);
      cells += `<div class="heat-cell" data-row="${i}" data-col="${j}" style="background:rgba(var(--accent-rgb), ${opacity})">${formatter(v)}</div>`;
    });
  });
  return `<div class="heat-grid">${cells}</div>`;
}

function renderScores(container, stepId, result) {
  const grid = buildHeatGrid(result.scores, result.tokens);
  const legend = result.tokens.map((t, i) => `<span style="color:${result.tokenColors[i]}">"${t}"</span>`).join(', ');
  const t0 = result.tokens[0];
  const worked = `score_"${t0}","${t0}" = q_"${t0}" &middot; k_"${t0}" = ${result.scores[0][0].toFixed(2)}`;
  container.innerHTML = `
    <div class="heat-block">
      ${grid}
      <div class="heat-meta">rows = query token, columns = key token<br>tokens: ${legend}<br>score[i,j] = q&#8571; &middot; k&#8571;</div>
    </div>
    <div class="formula-worked">${worked}</div>`;
}
```

Update `STEP_RENDERERS.scores = renderScores;`.

- [ ] **Step 2: Add the `scale` renderer**

```js
function renderScale(container, stepId, result) {
  const beforeGrid = buildHeatGrid(result.scores, result.tokens, { minOpacity: 0.1, maxOpacity: 0.5 });
  const afterGrid = buildHeatGrid(result.scaled, result.tokens, { minOpacity: 0.15, maxOpacity: 0.8 });
  const t0 = result.tokens[0];
  const worked = `scaled_"${t0}","${t0}" = ${result.scores[0][0].toFixed(2)} / &radic;${result.d} = ${result.scaled[0][0].toFixed(2)}`;
  container.innerHTML = `
    <div class="heat-block scale-compare">
      <div><div class="heat-caption">before (&divide;1)</div>${beforeGrid}</div>
      <div class="scale-arrow">&divide;&radic;${result.d} &rarr;</div>
      <div><div class="heat-caption">after (&divide;&radic;${result.d})</div>${afterGrid}</div>
    </div>
    <div class="formula-worked">${worked}</div>`;
}
```

Update `STEP_RENDERERS.scale = renderScale;`.

- [ ] **Step 3: Add supporting styles to `styles/attention.css`**

```css
.ui.attention .heat-block { display: flex; gap: 22px; align-items: center; flex-wrap: wrap; }
.ui.attention .heat-meta { font-size: 12.5px; color: var(--text-body); line-height: 1.7; }
.ui.attention .heat-caption { font: 600 9px/1.3 var(--font-mono); letter-spacing: .04em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px; }
.ui.attention .scale-compare { align-items: center; }
.ui.attention .scale-arrow { font: 600 13px/1 var(--font-mono); color: var(--accent-link); }
```

- [ ] **Step 4: Manually verify**

```bash
cd /home/audie/Visualization && python3 -m http.server 8842 &
sleep 1
```

```js
const { firefox } = require('playwright');
(async () => {
  const browser = await firefox.launch();
  const page = await browser.newPage();
  await page.goto('http://localhost:8842/pages/attention.html', { waitUntil: 'networkidle' });
  const scoreCells = await page.$$eval('#step-scores .heat-cell', (els) => els.length);
  const scaleCells = await page.$$eval('#step-scale .heat-cell', (els) => els.length);
  console.log('score cells:', scoreCells, 'scale cells (before+after):', scaleCells); // expect 9, 18

  const scoresWorked = await page.$eval('#step-scores .formula-worked', (el) => el.textContent);
  const scaleWorked = await page.$eval('#step-scale .formula-worked', (el) => el.textContent);
  console.log('scores formula-worked shows real value 2.73:', scoresWorked.includes('2.73'));
  console.log('scale formula-worked shows real values 2.73 and 1.37:', scaleWorked.includes('2.73') && scaleWorked.includes('1.37'));
  await browser.close();
})();
```

Expected: `score cells: 9 scale cells (before+after): 18`, `scores formula-worked shows real value
2.73: true`, `scale formula-worked shows real values 2.73 and 1.37: true` (these are the default
`cat-sat` preset's actual score[0][0]=2.73 and scaled[0][0]=1.37 for token "the", already verified
during planning).

```bash
kill %1
```

- [ ] **Step 5: Commit**

```bash
git add js/attention/scenes.js styles/attention.css
git commit -m "feat(attention): render QKT score and scale scenes with a shared heat-grid builder"
```

---

### Task 8: Mask and Softmax animations, with the mask toggle

**Files:**
- Modify: `js/attention/scenes.js`
- Modify: `js/attention/main.js`
- Modify: `styles/attention.css`

**Interfaces:**
- Consumes: `result.masked`, `result.weights`, `result.causal`, `applyCausalMask`,
  `softmaxMatrix` from `math.js`
- Produces: a page-level `causal` toggle state, read by `main.js` when rebuilding `result` — the
  mask scene's toggle calls a new exported `main.js` function
  `export function setCausal(causal: boolean): void` that recomputes and re-renders every scene
  (not just mask/softmax), since `weights` and `output` both depend on it

- [ ] **Step 1: Add the `mask` renderer to `js/attention/scenes.js`**

```js
function renderMask(container, stepId, result) {
  const isMasked = (i, j) => result.causal && j > i;
  const grid = buildHeatGrid(result.masked.map((row) => row.map((v) => (v <= -1e8 ? 0 : v))), result.tokens, { maskedCells: isMasked });
  // The formula-worked line has no separate .formula block to pair with (this step's rule is a
  // conditional, not a single equation), so it stands alone as the numeric instantiation of the
  // masking rule itself: which specific cell gets masked and why, or a clear statement that
  // nothing is masked yet.
  const worked = result.causal
    ? (() => {
        const t0 = result.tokens[0];
        const t1 = result.tokens[1];
        return `score_"${t0}","${t1}" &rarr; &minus;&infin; (masked: j=1 &gt; i=0)`;
      })()
    : 'causal mask is off: no cells masked, every token can see every other token';
  container.innerHTML = `
    <div class="heat-block">
      ${grid}
      <div class="heat-meta">
        <label class="mask-toggle"><input type="checkbox" data-role="causal-toggle" ${result.causal ? 'checked' : ''}> causal mask (each token can only see itself and earlier tokens)</label>
      </div>
    </div>
    <div class="formula-worked">${worked}</div>`;
  const toggle = container.querySelector('[data-role="causal-toggle"]');
  toggle.addEventListener('change', () => {
    window.attentionSetCausal(toggle.checked);
  });
}
```

Update `STEP_RENDERERS.mask = renderMask;`.

- [ ] **Step 2: Add the `softmax` renderer**

```js
function renderSoftmax(container, stepId, result) {
  const grid = buildHeatGrid(result.weights, result.tokens, { minOpacity: 0.15, maxOpacity: 0.85 });
  const bars = result.tokens.map((ti, i) => {
    const segs = result.tokens.map((tj, j) => {
      const pct = (result.weights[i][j] * 100).toFixed(1);
      return `<span class="softmax-seg" style="width:${pct}%; background:${result.tokenColors[j]}" title="${tj}: ${pct}%"></span>`;
    }).join('');
    return `<div class="softmax-bar-row"><span class="vec-token" style="color:${result.tokenColors[i]}">"${ti}"</span><div class="softmax-bar">${segs}</div></div>`;
  }).join('');
  const t0 = result.tokens[0];
  const row0 = result.masked[0].filter((v) => v > -1e8);
  const expSum = row0.reduce((sum, v) => sum + Math.exp(v), 0);
  const worked = `weight_"${t0}","${t0}" = e<sup>${row0[0].toFixed(2)}</sup> / ${expSum.toFixed(2)} = ${result.weights[0][0].toFixed(2)}`;
  container.innerHTML = `
    <div class="heat-block">${grid}<div class="heat-meta">each row sums to 1.00: the actual attention weights</div></div>
    <div class="softmax-bars">${bars}</div>
    <div class="formula-worked">${worked}</div>`;
}
```

Update `STEP_RENDERERS.softmax = renderSoftmax;`.

- [ ] **Step 3: Add `setCausal` to `js/attention/main.js` and expose it for the mask toggle**

```js
// js/attention/main.js (replace the previous version)
import { initPipelineBar } from './pipeline.js';
import { renderAllScenes } from './scenes.js';
import { computePipeline } from './math.js';
import { PRESETS, WEIGHTS, TOKEN_COLORS } from './presets.js';

let currentPreset = PRESETS[0];
let currentCausal = false;

function buildAndRenderAll() {
  const result = computePipeline(currentPreset.tokens, currentPreset.embeddings, WEIGHTS, { causal: currentCausal });
  result.tokenColors = TOKEN_COLORS;
  renderAllScenes(result);
}

// The mask scene's checkbox toggles causal masking, which changes weights and output
// everywhere downstream, so it must re-render every scene, not just its own.
// Exposed on window rather than imported by scenes.js, so scenes.js has no reverse dependency
// on main.js (scenes.js only ever receives a PipelineResult, it never triggers recomputation).
window.attentionSetCausal = function setCausal(causal) {
  currentCausal = causal;
  buildAndRenderAll();
};

function init() {
  const barEl = document.getElementById('pipeBar');
  if (barEl) initPipelineBar(barEl);
  buildAndRenderAll();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
```

- [ ] **Step 4: Add supporting styles to `styles/attention.css`**

```css
.ui.attention .mask-toggle { display: flex; align-items: center; gap: 8px; font: 500 12.5px/1.5 var(--font-sans); color: var(--text-body); cursor: pointer; }
.ui.attention .softmax-bars { display: flex; flex-direction: column; gap: 10px; margin-top: 16px; }
.ui.attention .softmax-bar-row { display: flex; align-items: center; gap: 12px; }
.ui.attention .softmax-bar { flex: 1; height: 18px; border-radius: var(--radius-sm); overflow: hidden; display: flex; background: var(--surface); }
.ui.attention .softmax-seg { display: block; height: 100%; opacity: .85; }
```

- [ ] **Step 5: Manually verify the toggle recomputes everything, not just its own scene**

```bash
cd /home/audie/Visualization && python3 -m http.server 8842 &
sleep 1
```

```js
const { firefox } = require('playwright');
(async () => {
  const browser = await firefox.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('http://localhost:8842/pages/attention.html', { waitUntil: 'networkidle' });

  const softmaxBefore = await page.$eval('#step-softmax .scene-anim', (el) => el.textContent);
  await page.check('#step-mask [data-role="causal-toggle"]');
  await page.waitForTimeout(200);
  const maskedCellCount = await page.$$eval('#step-mask .heat-cell.masked', (els) => els.length);
  const softmaxAfter = await page.$eval('#step-softmax .scene-anim', (el) => el.textContent);

  console.log('masked cells after enabling causal mask:', maskedCellCount); // expect 3 (upper triangle of a 3x3)
  // softmax weights depend on the mask, so this proves the toggle triggers a real recompute
  // reaching a scene other than mask's own, without depending on Task 9's output renderer
  // (not built yet at this point in the plan) existing.
  console.log('softmax scene text changed after toggling mask:', softmaxBefore !== softmaxAfter); // expect true
  console.log('errors:', JSON.stringify(errors));
  await browser.close();
})();
```

Expected: `masked cells after enabling causal mask: 3`, `softmax scene text changed after toggling
mask: true`, `errors: []`.

```bash
kill %1
```

- [ ] **Step 6: Commit**

```bash
git add js/attention/scenes.js js/attention/main.js styles/attention.css
git commit -m "feat(attention): render mask and softmax scenes, wire causal toggle to full recompute"
```

---

### Task 9: Weighted sum and Output animations

**Files:**
- Modify: `js/attention/scenes.js`
- Modify: `styles/attention.css`

**Interfaces:**
- Consumes: `result.weights`, `result.V`, `result.output`, `result.tokens`, `result.tokenColors`

- [ ] **Step 1: Add the `wsum` renderer**

```js
function renderWsum(container, stepId, result) {
  const focusToken = result.tokens[Math.min(1, result.tokens.length - 1)];
  const focusIdx = result.tokens.indexOf(focusToken);
  const rows = result.tokens.map((tj, j) => {
    const w = result.weights[focusIdx][j];
    const vec = result.V[tj].map((v) => v.toFixed(2)).join(', ');
    return `<div class="wsum-row" style="opacity:${(0.25 + w * 0.75).toFixed(2)}">
      <span class="vec-token" style="color:${result.tokenColors[j]}">"${tj}"</span>
      <span class="wsum-weight">&times; ${w.toFixed(2)}</span>
      <span class="vec-values">[${vec}]</span>
    </div>`;
  }).join('');
  const out = result.output[focusIdx].map((v) => v.toFixed(2)).join(', ');
  const sumTerms = result.tokens.map((tj, j) => `${result.weights[focusIdx][j].toFixed(2)}&middot;v_"${tj}"`).join(' + ');
  const worked = `o_"${focusToken}" = ${sumTerms} = [${out}]`;
  container.innerHTML = `
    <p class="scene-note">showing the output for <span class="vec-token" style="color:${result.tokenColors[focusIdx]}">"${focusToken}"</span> (row opacity = attention weight on that value)</p>
    <div class="wsum-list">${rows}</div>
    <div class="wsum-result">= [${out}]</div>
    <div class="formula-worked">${worked}</div>`;
}
```

Update `STEP_RENDERERS.wsum = renderWsum;`.

- [ ] **Step 2: Add the `output` renderer**

```js
function renderOutput(container, stepId, result) {
  const rows = result.tokens.map((t, i) => {
    const vec = result.output[i].map((v) => v.toFixed(2)).join(', ');
    return `<div class="vec-row"><span class="vec-token" style="color:${result.tokenColors[i]}">"${t}"</span><span class="vec-values">[${vec}]</span></div>`;
  }).join('');
  container.innerHTML = `<div class="vec-list">${rows}</div><p class="scene-note">same shape as the input embeddings this pipeline started from: each vector is now a blend of the whole sequence.</p>`;
}
```

Update `STEP_RENDERERS.output = renderOutput;`.

- [ ] **Step 3: Add supporting styles to `styles/attention.css`**

```css
.ui.attention .scene-note { font-size: 12px; color: var(--text-muted); margin: 0 0 12px; }
.ui.attention .wsum-list { display: flex; flex-direction: column; gap: 6px; }
.ui.attention .wsum-row { display: flex; align-items: baseline; gap: 12px; font: 500 12.5px/1.6 var(--font-mono); color: var(--text-2); transition: opacity 300ms var(--ease-out); }
.ui.attention .wsum-weight { color: var(--accent-link); min-width: 60px; }
.ui.attention .wsum-result { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--hairline); font: 700 13px/1.5 var(--font-mono); color: var(--text); }
```

- [ ] **Step 4: Manually verify**

```bash
cd /home/audie/Visualization && python3 -m http.server 8842 &
sleep 1
```

```js
const { firefox } = require('playwright');
(async () => {
  const browser = await firefox.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('http://localhost:8842/pages/attention.html', { waitUntil: 'networkidle' });
  const wsumRows = await page.$$eval('#step-wsum .wsum-row', (els) => els.length);
  const outputRows = await page.$$eval('#step-output .vec-row', (els) => els.length);
  console.log('wsum rows:', wsumRows, 'output rows:', outputRows); // expect 3, 3
  const wsumWorked = await page.$eval('#step-wsum .formula-worked', (el) => el.textContent);
  console.log('wsum formula-worked mentions focus token "cat":', wsumWorked.includes('"cat"'));
  console.log('errors:', JSON.stringify(errors));
  await browser.close();
})();
```

Expected: `wsum rows: 3 output rows: 3`, `wsum formula-worked mentions focus token "cat": true`,
`errors: []`.

```bash
kill %1
```

- [ ] **Step 5: Commit**

```bash
git add js/attention/scenes.js styles/attention.css
git commit -m "feat(attention): render weighted sum and output scenes, completing all 8 steps"
```

---

### Task 10: Preset picker

**Files:**
- Modify: `js/attention/main.js`
- Modify: `styles/attention.css`

**Interfaces:**
- Consumes: `PRESETS` from `presets.js`, `buildAndRenderAll` (internal to `main.js`, extended
  here to also rebuild the picker's active state)

- [ ] **Step 1: Extend `js/attention/main.js` to build and wire the preset picker**

```js
// js/attention/main.js (extend the Task 8 version)
import { initPipelineBar } from './pipeline.js';
import { renderAllScenes } from './scenes.js';
import { computePipeline } from './math.js';
import { PRESETS, WEIGHTS, TOKEN_COLORS } from './presets.js';

let currentPreset = PRESETS[0];
let currentCausal = false;

function buildAndRenderAll() {
  const result = computePipeline(currentPreset.tokens, currentPreset.embeddings, WEIGHTS, { causal: currentCausal });
  result.tokenColors = TOKEN_COLORS;
  renderAllScenes(result);
  updatePickerActiveState();
}

window.attentionSetCausal = function setCausal(causal) {
  currentCausal = causal;
  buildAndRenderAll();
};

function updatePickerActiveState() {
  const picker = document.getElementById('presetPicker');
  if (!picker) return;
  picker.querySelectorAll('.preset-btn').forEach((btn) => {
    btn.classList.toggle('is-active', btn.dataset.presetId === currentPreset.id);
  });
}

function initPresetPicker() {
  const picker = document.getElementById('presetPicker');
  if (!picker) return;
  picker.innerHTML = PRESETS.map(
    (p) => `<button type="button" class="preset-btn" data-preset-id="${p.id}">${p.label}</button>`
  ).join('');
  picker.addEventListener('click', (e) => {
    const btn = e.target.closest('.preset-btn');
    if (!btn) return;
    const preset = PRESETS.find((p) => p.id === btn.dataset.presetId);
    if (!preset) return;
    currentPreset = preset;
    buildAndRenderAll();
  });
}

function init() {
  const barEl = document.getElementById('pipeBar');
  if (barEl) initPipelineBar(barEl);
  initPresetPicker();
  buildAndRenderAll();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
```

- [ ] **Step 2: Add supporting styles to `styles/attention.css`**

```css
.ui.attention .preset-picker { display: flex; gap: 8px; flex-wrap: wrap; }
.ui.attention .preset-btn {
  font: 600 12px/1 var(--font-mono); color: var(--text-2);
  border: 1px solid var(--hairline); background: transparent;
  border-radius: var(--radius-pill); padding: 7px 14px; cursor: pointer;
  transition: border-color 160ms var(--ease-out), color 160ms var(--ease-out), background 160ms var(--ease-out);
}
.ui.attention .preset-btn:hover { border-color: var(--accent-link); color: var(--accent-link); }
.ui.attention .preset-btn.is-active { background: var(--accent-muted); border-color: var(--accent); color: var(--accent-link); }
```

- [ ] **Step 3: Manually verify switching presets recomputes every scene**

```bash
cd /home/audie/Visualization && python3 -m http.server 8842 &
sleep 1
```

```js
const { firefox } = require('playwright');
(async () => {
  const browser = await firefox.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('http://localhost:8842/pages/attention.html', { waitUntil: 'networkidle' });

  const inputBefore = await page.$eval('#step-input .scene-anim', (el) => el.textContent);
  await page.click('#presetPicker .preset-btn[data-preset-id="dog-ran-fast"]');
  await page.waitForTimeout(200);
  const inputAfter = await page.$eval('#step-input .scene-anim', (el) => el.textContent);
  const activeBtn = await page.$eval('#presetPicker .preset-btn.is-active', (el) => el.dataset.presetId);

  console.log('input scene changed after preset switch:', inputBefore !== inputAfter);
  console.log('active preset button:', activeBtn); // expect "dog-ran-fast"
  console.log('errors:', JSON.stringify(errors));
  await browser.close();
})();
```

Expected: `input scene changed after preset switch: true`, `active preset button: dog-ran-fast`,
`errors: []`.

```bash
kill %1
```

- [ ] **Step 4: Commit**

```bash
git add js/attention/main.js styles/attention.css
git commit -m "feat(attention): add preset picker, completing phase 1 functionality"
```

---

### Task 11: Full QA pass

**Files:** none (verification only, following the same methodology already proven on this
project for `gradient-descent.html` — headless Firefox + Playwright, no display available in
this environment)

- [ ] **Step 1: Start the server**

```bash
cd /home/audie/Visualization && python3 -m http.server 8842 &
sleep 1
curl -sf http://localhost:8842/pages/attention.html > /dev/null && echo READY
```

- [ ] **Step 2: Console-error and structural sweep across both presets and both mask states**

```js
const { firefox } = require('playwright');
(async () => {
  const browser = await firefox.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto('http://localhost:8842/pages/attention.html', { waitUntil: 'networkidle' });

  const presetIds = await page.$$eval('#presetPicker .preset-btn', (els) => els.map((e) => e.dataset.presetId));
  for (const id of presetIds) {
    await page.click(`#presetPicker .preset-btn[data-preset-id="${id}"]`);
    await page.waitForTimeout(150);
    for (const causal of [false, true]) {
      const checkbox = page.locator('#step-mask [data-role="causal-toggle"]');
      const isChecked = await checkbox.isChecked();
      if (isChecked !== causal) await checkbox.click();
      await page.waitForTimeout(150);
      const outputText = await page.$eval('#step-output .scene-anim', (el) => el.textContent);
      console.log(`preset=${id} causal=${causal} output non-empty:`, outputText.trim().length > 0);
    }
  }
  console.log('console/page errors:', JSON.stringify(errors));
  await browser.close();
})();
```

Expected: every combination logs `output non-empty: true`, and `console/page errors: []`.

- [ ] **Step 3: Navigation sweep — pipeline bar, rail, and plain scroll all reach every step**

```js
const { firefox } = require('playwright');
(async () => {
  const browser = await firefox.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  await page.goto('http://localhost:8842/pages/attention.html', { waitUntil: 'networkidle' });
  const ids = ['input', 'qkv', 'scores', 'scale', 'mask', 'softmax', 'wsum', 'output'];
  for (const id of ids) {
    await page.click(`#pipeBar .pipe-node[data-target="${id}"]`);
    await page.waitForTimeout(900);
    const inView = await page.$eval(`#step-${id}`, (el) => {
      const r = el.getBoundingClientRect();
      return r.top < window.innerHeight && r.bottom > 0;
    });
    const isCurrent = await page.$eval(`#pipeBar .pipe-node[data-target="${id}"]`, (el) => el.classList.contains('current'));
    console.log(`bar click ${id}: in view=${inView} bar highlights current=${isCurrent}`);
  }
  await browser.close();
})();
```

Expected: every step logs `in view=true bar highlights current=true`.

- [ ] **Step 4: Mobile width check**

```js
const { firefox } = require('playwright');
(async () => {
  const browser = await firefox.launch();
  const page = await browser.newPage({ viewport: { width: 375, height: 900 } });
  await page.goto('http://localhost:8842/pages/attention.html', { waitUntil: 'networkidle' });
  const barWidth = await page.$eval('#pipeBar', (el) => el.getBoundingClientRect().width);
  const viewportWidth = 375;
  console.log('pipe bar width:', barWidth, 'viewport:', viewportWidth);
  const overflowsX = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
  console.log('page has horizontal overflow at 375px:', overflowsX); // expect false
  await browser.close();
})();
```

Expected: `page has horizontal overflow at 375px: false`. If it is `true`, inspect
`.pipe-row`/`.heat-block` for elements without `min-width: 0` or `flex-wrap`, per this project's
established mobile-overflow fix pattern (see the `fix/gradient-descent-mobile-contour` commit for
precedent), and fix before proceeding.

- [ ] **Step 5: Take a full-page screenshot at desktop width and visually inspect it**

```js
const { firefox } = require('playwright');
(async () => {
  const browser = await firefox.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  await page.goto('http://localhost:8842/pages/attention.html', { waitUntil: 'networkidle' });
  await page.screenshot({ path: '/tmp/attention-qa-full.png', fullPage: true });
  await browser.close();
})();
```

Read `/tmp/attention-qa-full.png` and confirm: all 8 scenes render distinct, non-overlapping
content; no visible `NaN`/`undefined` text anywhere; the sticky bar's 8 icons are visually
distinct from each other; heat-grid numbers are legible against their background tint at every
opacity level used.

- [ ] **Step 6: Stop the server**

```bash
kill %1
```

- [ ] **Step 7: Final commit**

```bash
git add -A
git commit -m "chore(attention): phase 1 complete, full QA pass clean"
```

---

## Self-review notes

- **Spec coverage:** every spec section maps to a task — eight-step pipeline (Tasks 2, 4-9),
  sticky-bar-as-primary-nav with rail as a secondary trigger of the same behavior (Task 4),
  per-step granular animation (Tasks 6-9), worked example with persistent token colors (Task 1,
  threaded through every scene renderer via `result.tokenColors`), curated presets with
  forward-compatible architecture (Task 1's data shape + Task 10's thin picker layer over
  `computePipeline`), mask as a toggleable 8th step (Task 8), conceptual asides (static prose in
  Task 3's HTML), formulas in real MathJax (Task 3's `.formula` blocks, matching the site's
  existing math convention — no separate task needed, MathJax is already loaded site-wide via the
  CDN script and auto-typesets on page load). Phase 2 (backprop) and free-form editing are
  explicitly out of scope per the spec and have no task here.
- **Placeholder scan:** the only literal "placeholder" text in this plan is
  `renderPlaceholder`'s output in Task 5, which is an intentional, real, working interim state
  (later tasks replace it), not an unfinished plan step.
- **Type consistency:** `PipelineResult`'s shape (defined in Task 1) is used identically by every
  later task — `result.tokens`, `.embeddings`, `.Q/K/V`, `.scores`, `.scaled`, `.masked`,
  `.weights`, `.output`, `.d`, `.causal`, plus `.tokenColors` (attached by `main.js`, not by
  `computePipeline` itself, since token color is a presentation concern, not math). `STEP_IDS`
  (Task 2) and `STEPS` (Task 4) use the same eight id strings throughout; `renderScene`/
  `STEP_RENDERERS` (Task 5) key off the same ids Task 4's `STEPS` and Task 2's `STEP_IDS` define.
