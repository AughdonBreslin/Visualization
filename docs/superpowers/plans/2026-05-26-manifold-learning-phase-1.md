# Manifold Learning Phase 1 Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: Ship a working `pages/manifold.html` page that lets the user pick a dataset (Swiss roll, S-curve, or an uploaded CSV via the dataset dropdown) and two algorithms (PCA, Isomap), then walk both algorithms through a synced canonical step sequence with side-by-side 3D viz, branching step indicator, per-side IFW tabs, and collapsible pseudocode.

Architecture: Vanilla ES modules under `js/manifold/`, loaded as `<script type="module">`. The page reuses the existing `base.css` and `article.css` shells. A central state module drives recomputation. Pure logic modules (canonical steps, RNG, CSV parsing, kNN, Dijkstra, double-centering) are unit-tested via the built-in `node --test` runner; DOM- and eigendecomposition-dependent modules are verified manually in the browser.

Tech Stack: HTML5, CSS3, vanilla ES modules, d3 v7 (global, loaded from CDN), numeric.js (global, loaded from CDN), MathJax v3, Node 18+ `node --test` for unit tests.

Notes that override defaults: the user has banned em-dashes, `<em>`/`<strong>` tags, and markdown emphasis (`*`/`**`/`_`/`__`) in any generated content. Use plain prose with headings and bullets.

Scope: Phase 1 only. Phases 2 and 3 (additional algorithms and datasets) ship as separate plans following the same module contracts.

---

## File structure

Files this plan creates:

- `pages/manifold.html` page shell
- `styles/manifold.css` page styling
- `js/manifold/canonical_steps.js` canonical step IDs and sub-step helpers
- `js/manifold/rng.js` seeded random number generator
- `js/manifold/datasets.js` synthetic generators (Swiss roll, S-curve) plus CSV parsing and 3D projection
- `js/manifold/linalg.js` linear algebra helpers (kNN graph, Dijkstra all-pairs, double-centering, eigendecomposition wrappers)
- `js/manifold/algorithms/pca.js` PCA pipeline keyed by canonical sub-step
- `js/manifold/algorithms/isomap.js` Isomap pipeline keyed by canonical sub-step
- `js/manifold/viz3d.js` d3 SVG 3D orbit viewer
- `js/manifold/viz2d.js` 2D scatter renderer for the final embedding
- `js/manifold/step_indicator.js` two-rail branching step indicator
- `js/manifold/ifw.js` Intuition / Formula / Worked example tab panel
- `js/manifold/pseudocode.js` collapsible pseudocode renderer
- `js/manifold/state.js` central state with subscribe / recompute
- `js/manifold/main.js` entry point: wires controls, state, and views

Files this plan modifies:

- `index.html` adds the manifold page link to the project list

Test files this plan creates (run with `node --test test/manifold`):

- `test/manifold/canonical_steps.test.js`
- `test/manifold/rng.test.js`
- `test/manifold/datasets_csv.test.js`
- `test/manifold/datasets_synthetic.test.js`
- `test/manifold/linalg_knn.test.js`
- `test/manifold/linalg_dijkstra.test.js`
- `test/manifold/linalg_doublecenter.test.js`

Modules that read `window.d3` or `window.numeric` are verified manually in the browser at the end of the relevant task; pure logic is unit-tested in node.

---

## Canonical step contract

Both algorithms map their sub-steps onto a fixed canonical sequence. Sub-steps are strings like `0`, `1`, `2`, `2a`, `2b`, etc. Sort order: canonical index ascending; sub-step suffix ascending.

| ID | Canonical step |
|----|----------------|
| 0  | Raw data |
| 1  | Preprocess (center / normalize) |
| 2  | Neighborhood graph |
| 3  | Pairwise affinity / distances |
| 4  | Matrix transform |
| 5  | Spectral decomposition |
| 6  | Embed / project |

Phase 1 mappings:

- PCA: `['0','1','3','5','6']`. Canonical step 3 holds the 3x3 covariance matrix; canonical step 5 holds the eigendecomposition of that matrix.
- Isomap: `['0','2','3','4','5','6']`. Canonical step 3 holds the geodesic distance matrix; canonical step 4 holds the double-centered Gram matrix.

A sub-step `2a`, `2b`, etc. is reserved for cases where one algorithm has more than one operation inside the same canonical step. Phase 1 does not use sub-steps yet but the rendering code must support them.

---

## Algorithm module contract

Each algorithm file exports an object shaped like:

```javascript
{
  id: 'pca',
  label: 'PCA',
  params: [ { name: 'k', type: 'int', default: 10, min: 2, max: 50 } ],
  pseudocode: [
    { id: 'pca-center', title: '1. Center the data', steps: ['1'], lines: ['mean ← (1/N) · Σ_i x_i', '...'] },
    // ...
  ],
  run(dataset, params) {
    // returns { steps: Map<subStepId, StepState>, presentSubSteps: string[] }
  }
}
```

`StepState` shape:

```javascript
{
  points: Float64Array,   // 3D coordinates rendered by viz3d, length = 3N
  t: Float64Array | null, // optional intrinsic parameter for rainbow coloring, length = N
  colors: string[] | null,// optional per-point override colors, length = N
  edges: [number, number][] | null, // optional edge list for kNN overlay
  embed2d: Float64Array | null,     // optional 2D coords for step 6, length = 2N
  label: string,          // short label shown in the step indicator description
  ifw: {
    intuition: string | null, // HTML allowed for inline MathJax delimiters
    formula: string | null,
    worked: string | null
  }
}
```

---

## Dataset module contract

Each dataset exports:

```javascript
{
  id: 'swiss_roll',
  label: 'Swiss roll',
  generate({ samples, noise, seed, csvRows }) {
    return { X: Float64Array, t: Float64Array, N: number }
  }
}
```

`X` is row-major 3-vectors (length `3*N`). `t` is the intrinsic parameter used for rainbow coloring. CSV upload returns the same shape, projecting through PCA when the source has more than 3 numeric columns.

CSV upload in the dropdown: the CSV dataset has id `csv` and appears as the last option in the dataset dropdown. The wiring in `main.js` triggers a hidden file picker when the user selects this option. No separate file picker row exists outside the dropdown.

---

## State module contract

```javascript
createState({ algorithmsById, defaults }) returns {
  state,         // { datasetId, datasetParams, csvRows, leftAlgoId, leftAlgoParams,
                 //   rightAlgoId, rightAlgoParams, currentSubStep, cache }
  subscribe(fn), // fn(state) called after each recompute or step change
  set(updates),  // shallow merge, triggers recompute + emit
  setStep(sub),  // updates currentSubStep, emits without recompute
  recompute()    // explicit recompute (idempotent under unchanged inputs)
}
```

`cache` holds `{ dataset, left, right, key }` where `key` is a JSON fingerprint of the inputs. `recompute()` is a no-op when the key has not changed.

---

# Tasks

### Task 1: Repository scaffolding and page link

Files:
- Create: `pages/manifold.html`
- Create: `styles/manifold.css`
- Modify: `index.html` (add the manifold link before the closing `</ul>`)
- Create: `test/manifold/.gitkeep`

- [ ] Step 1: Create the empty test directory

Run:
```
mkdir -p test/manifold
touch test/manifold/.gitkeep
```

- [ ] Step 2: Create the minimal page shell

Write `pages/manifold.html` exactly:
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Manifold Learning: Step-by-Step Comparison</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/numeric/1.2.6/numeric.min.js"></script>
  <script>
    window.MathJax = {
      tex: { inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['$$', '$$']] },
      svg: { fontCache: 'global' }
    };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <link rel="stylesheet" href="../styles/base.css">
  <link rel="stylesheet" href="../styles/article.css">
  <link rel="stylesheet" href="../styles/manifold.css">
  <script type="module" src="../js/manifold/main.js"></script>
</head>
<body>
  <div class="container article manifold">
    <header>
      <h1>Manifold Learning</h1>
      <div class="subtitle">Step-by-step comparison of two algorithms on a shared dataset</div>
      <div class="home-link"><a href="../index.html">← Home</a></div>
    </header>
    <main class="article-body">
      <section class="panel">
        <h2>Dataset</h2>
        <div class="mf-controls-row">
          <div class="mf-control">
            <label for="mfDataset">Dataset</label>
            <select id="mfDataset"></select>
          </div>
          <div class="mf-control" id="mfSamplesControl">
            <label for="mfSamples">Samples</label>
            <input id="mfSamples" type="number" min="20" max="1000" step="20" value="300" />
          </div>
          <div class="mf-control" id="mfNoiseControl">
            <label for="mfNoise">Noise σ</label>
            <input id="mfNoise" type="number" min="0" step="0.05" value="0" />
          </div>
          <div class="mf-control" id="mfSeedControl">
            <label for="mfSeed">Seed</label>
            <input id="mfSeed" type="number" step="1" value="7" />
          </div>
          <div class="mf-control mf-control-action">
            <span>&nbsp;</span>
            <button id="mfReseed" type="button">↻ Reseed</button>
          </div>
        </div>
        <input id="mfCsvInput" type="file" accept=".csv,text/csv" style="display:none" />
        <div id="mfCsvLabel" class="mf-csv-name"></div>
      </section>
      <section class="panel">
        <h2>Algorithms</h2>
        <div class="mf-algos-row">
          <div class="mf-algo-card">
            <div class="mf-algo-header">
              <span class="mf-algo-side mf-algo-side-a">A</span>
              <select id="mfAlgoLeft" class="mf-algo-select"></select>
            </div>
            <div id="mfAlgoLeftParams" class="mf-algo-params"></div>
          </div>
          <div class="mf-algo-card">
            <div class="mf-algo-header">
              <span class="mf-algo-side mf-algo-side-b">B</span>
              <select id="mfAlgoRight" class="mf-algo-select"></select>
            </div>
            <div id="mfAlgoRightParams" class="mf-algo-params"></div>
          </div>
        </div>
      </section>
      <section class="panel">
        <h2>Step</h2>
        <div id="mfStep"></div>
      </section>
      <section class="panel">
        <h2>Visualization</h2>
        <div class="mf-viz-row">
          <div class="mf-viz-card">
            <h3 class="h3-tight">Left: <span id="mfLeftTitle">A</span></h3>
            <div id="mfLeftViz" class="mf-viz-host"></div>
          </div>
          <div class="mf-viz-card">
            <h3 class="h3-tight">Right: <span id="mfRightTitle">B</span></h3>
            <div id="mfRightViz" class="mf-viz-host"></div>
          </div>
        </div>
      </section>
      <section class="panel">
        <h2>Step notes</h2>
        <div class="mf-ifw-row">
          <div class="mf-ifw-card" id="mfLeftIfw"></div>
          <div class="mf-ifw-card" id="mfRightIfw"></div>
        </div>
      </section>
      <section class="panel">
        <h2>Full pseudocode</h2>
        <div class="mf-pseudo-row">
          <div class="mf-pseudo-card" id="mfLeftPseudo"></div>
          <div class="mf-pseudo-card" id="mfRightPseudo"></div>
        </div>
      </section>
    </main>
    <footer class="footer">
      <p class="footer-tag">Created by <a href="https://linkedin.com/in/aughdon/">Aughdon Breslin</a></p>
    </footer>
  </div>
</body>
</html>
```

- [ ] Step 3: Create the page CSS skeleton

Write `styles/manifold.css`:
```css
/* manifold.css */
.manifold .mf-controls-row { display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-end; margin-top: 8px; }
.manifold .mf-control { display: flex; flex-direction: column; gap: 4px; min-width: 140px; }
.manifold .mf-control input, .manifold .mf-control select { min-width: 140px; }
.manifold .mf-control-action { align-self: flex-end; }
.manifold .mf-csv-name { font-size: 0.9rem; opacity: 0.7; margin-top: 6px; min-height: 1.2em; }
.manifold .mf-algos-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }
.manifold .mf-algo-card { background: var(--surface-inset); border: 1px solid var(--border-light); border-radius: var(--radius-md); padding: 10px 12px; }
.manifold .mf-algo-header { display: flex; gap: 10px; align-items: center; margin-bottom: 8px; }
.manifold .mf-algo-header select { flex: 1; min-width: 0; }
.manifold .mf-algo-side { display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; flex: 0 0 24px; border-radius: 50%; font-weight: 700; background: rgba(255,255,255,0.12); font-size: 0.9rem; }
.manifold .mf-algo-side-a { background: #ff9f43; color: #2a1700; }
.manifold .mf-algo-side-b { background: #54a0ff; color: #06182f; }
.manifold .mf-algo-params { display: flex; flex-wrap: wrap; gap: 10px; font-size: 0.9rem; }
.manifold .mf-param { display: inline-flex; align-items: center; gap: 4px; }
.manifold .mf-param input { width: 5.5em; }
.manifold .mf-noparams { opacity: 0.6; font-style: italic; }
.manifold .mf-viz-row, .manifold .mf-ifw-row, .manifold .mf-pseudo-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.manifold .mf-viz-card, .manifold .mf-ifw-card, .manifold .mf-pseudo-card { background: var(--surface-inset); border: 1px solid var(--border-light); border-radius: var(--radius-md); padding: 10px 12px; min-width: 0; }
.manifold .mf-viz-host { position: relative; width: 100%; aspect-ratio: 4 / 3; background: rgba(0,0,0,0.25); border-radius: var(--radius-sm); overflow: hidden; }
@media (max-width: 820px) {
  .manifold .mf-algos-row, .manifold .mf-viz-row, .manifold .mf-ifw-row, .manifold .mf-pseudo-row { grid-template-columns: 1fr; }
}
```

- [ ] Step 4: Add the page link to the index

Edit `index.html`. Add this list item directly after the existing Fourier list item and before the closing `</ul>`:
```html
        <li class="project-item">
          <a href="pages/manifold.html">Manifold Learning</a>
          <div class="project-desc">Step-by-step side-by-side comparison of manifold learning algorithms on synthetic datasets or your own CSV</div>
        </li>
```

- [ ] Step 5: Manual verification

Run a static server: `python3 -m http.server 8765 --bind 127.0.0.1`

Open `http://127.0.0.1:8765/index.html`. Confirm the Manifold Learning link appears and clicking it loads the manifold page. The page should render the header and labelled section panels with empty controls (no JS wired yet). No console errors except the absent `main.js` (we have not created it yet, so the module load will 404 with a visible error in the console; that is expected).

- [ ] Step 6: Commit

```
git add pages/manifold.html styles/manifold.css index.html test/manifold/.gitkeep
git commit -m "manifold: add page shell and link from index"
```

---

### Task 2: Canonical step contract module

Files:
- Create: `js/manifold/canonical_steps.js`
- Create: `test/manifold/canonical_steps.test.js`

- [ ] Step 1: Write the failing tests

Write `test/manifold/canonical_steps.test.js`:
```javascript
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
```

- [ ] Step 2: Run the tests to verify they fail

Run: `node --test test/manifold/canonical_steps.test.js`
Expected: failures (module not found).

- [ ] Step 3: Implement the module

Write `js/manifold/canonical_steps.js`:
```javascript
export const CANONICAL_STEPS = [
  { id: '0', label: 'Raw data' },
  { id: '1', label: 'Preprocess (center/normalize)' },
  { id: '2', label: 'Neighborhood graph' },
  { id: '3', label: 'Pairwise affinity / distances' },
  { id: '4', label: 'Matrix transform' },
  { id: '5', label: 'Spectral decomposition' },
  { id: '6', label: 'Embed / project' },
];

export const CANONICAL_INDEX = new Map(CANONICAL_STEPS.map((s, i) => [s.id, i]));

export function canonicalOf(subStepId) {
  return subStepId.replace(/[a-z]$/, '');
}

export function compareSubSteps(a, b) {
  const ia = CANONICAL_INDEX.get(canonicalOf(a));
  const ib = CANONICAL_INDEX.get(canonicalOf(b));
  if (ia !== ib) return ia - ib;
  const sa = a.length > 1 ? a.charCodeAt(a.length - 1) : 0;
  const sb = b.length > 1 ? b.charCodeAt(b.length - 1) : 0;
  return sa - sb;
}

export function unionSubSteps(...lists) {
  const seen = new Set();
  for (const list of lists) for (const id of list) seen.add(id);
  return [...seen].sort(compareSubSteps);
}
```

- [ ] Step 4: Run the tests to verify they pass

Run: `node --test test/manifold/canonical_steps.test.js`
Expected: 4 passing.

- [ ] Step 5: Commit

```
git add js/manifold/canonical_steps.js test/manifold/canonical_steps.test.js
git commit -m "manifold: canonical step contract with sub-step helpers"
```

---

### Task 3: Seeded RNG module

Files:
- Create: `js/manifold/rng.js`
- Create: `test/manifold/rng.test.js`

- [ ] Step 1: Write the failing tests

Write `test/manifold/rng.test.js`:
```javascript
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
```

- [ ] Step 2: Run tests to verify they fail

Run: `node --test test/manifold/rng.test.js`
Expected: failures (module not found).

- [ ] Step 3: Implement

Write `js/manifold/rng.js`:
```javascript
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand) {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
```

- [ ] Step 4: Run tests to verify they pass

Run: `node --test test/manifold/rng.test.js`
Expected: 4 passing.

- [ ] Step 5: Commit

```
git add js/manifold/rng.js test/manifold/rng.test.js
git commit -m "manifold: seeded RNG and gaussian helper"
```

---

### Task 4: CSV parser

Files:
- Create: stub portion of `js/manifold/datasets.js` containing only `parseCSV`
- Create: `test/manifold/datasets_csv.test.js`

- [ ] Step 1: Write the failing tests

Write `test/manifold/datasets_csv.test.js`:
```javascript
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
```

- [ ] Step 2: Run tests to verify they fail

Run: `node --test test/manifold/datasets_csv.test.js`
Expected: module not found.

- [ ] Step 3: Create `js/manifold/datasets.js` with only the parser exported for now

Write `js/manifold/datasets.js`:
```javascript
export function parseCSV(text) {
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
  if (lines.length === 0) return [];
  const first = lines[0].split(',').map(s => s.trim());
  const headerIsNumeric = first.every(c => c.length > 0 && Number.isFinite(Number(c)));
  const dataLines = headerIsNumeric ? lines : lines.slice(1);
  const rows = [];
  for (const line of dataLines) {
    const parts = line.split(',').map(s => Number(s.trim()));
    if (parts.length < 2) continue;
    if (!parts.every(v => Number.isFinite(v))) continue;
    rows.push(parts);
  }
  if (rows.length === 0) return [];
  const widths = rows.map(r => r.length);
  const mode = widths.sort((a, b) => widths.filter(x => x === a).length - widths.filter(x => x === b).length).pop();
  return rows.filter(r => r.length === mode);
}
```

- [ ] Step 4: Run tests to verify they pass

Run: `node --test test/manifold/datasets_csv.test.js`
Expected: 5 passing.

- [ ] Step 5: Commit

```
git add js/manifold/datasets.js test/manifold/datasets_csv.test.js
git commit -m "manifold: CSV parser with header detection and width filtering"
```

---

### Task 5: Synthetic dataset generators

Files:
- Modify: `js/manifold/datasets.js`
- Create: `test/manifold/datasets_synthetic.test.js`

- [ ] Step 1: Write the failing tests

Write `test/manifold/datasets_synthetic.test.js`:
```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SWISS_ROLL, S_CURVE, DATASETS, DATASETS_BY_ID } from '../../js/manifold/datasets.js';

test('Swiss roll yields N points with length 3N flat array', () => {
  const out = SWISS_ROLL.generate({ samples: 50, noise: 0, seed: 1 });
  assert.equal(out.N, 50);
  assert.equal(out.X.length, 150);
  assert.equal(out.t.length, 50);
});

test('Swiss roll is reproducible for the same seed', () => {
  const a = SWISS_ROLL.generate({ samples: 30, noise: 0, seed: 7 });
  const b = SWISS_ROLL.generate({ samples: 30, noise: 0, seed: 7 });
  for (let i = 0; i < a.X.length; i++) assert.equal(a.X[i], b.X[i]);
});

test('Swiss roll changes with the seed', () => {
  const a = SWISS_ROLL.generate({ samples: 30, noise: 0, seed: 1 });
  const b = SWISS_ROLL.generate({ samples: 30, noise: 0, seed: 2 });
  let diff = 0;
  for (let i = 0; i < a.X.length; i++) if (a.X[i] !== b.X[i]) diff++;
  assert.ok(diff > 0);
});

test('S-curve also yields the right shape', () => {
  const out = S_CURVE.generate({ samples: 40, noise: 0, seed: 1 });
  assert.equal(out.N, 40);
  assert.equal(out.X.length, 120);
  assert.equal(out.t.length, 40);
});

test('DATASETS exposes swiss_roll, s_curve, and csv ids in order', () => {
  assert.deepEqual(DATASETS.map(d => d.id), ['swiss_roll', 's_curve', 'csv']);
  assert.equal(DATASETS_BY_ID.swiss_roll.label, 'Swiss roll');
});
```

- [ ] Step 2: Run tests to verify they fail

Run: `node --test test/manifold/datasets_synthetic.test.js`
Expected: missing exports.

- [ ] Step 3: Extend `js/manifold/datasets.js`

Replace the file contents with the full datasets module:
```javascript
import { mulberry32, gaussian } from './rng.js';

function allocate(N) {
  return { X: new Float64Array(N * 3), t: new Float64Array(N), N };
}

function addNoise(X, noise, rand) {
  if (noise <= 0) return;
  for (let i = 0; i < X.length; i++) X[i] += noise * gaussian(rand);
}

export const SWISS_ROLL = {
  id: 'swiss_roll',
  label: 'Swiss roll',
  generate({ samples, noise, seed }) {
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const u = 1.5 * Math.PI * (1 + 2 * rand());
      const v = 21 * rand();
      out.X[i * 3 + 0] = u * Math.cos(u);
      out.X[i * 3 + 1] = v;
      out.X[i * 3 + 2] = u * Math.sin(u);
      out.t[i] = u;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const S_CURVE = {
  id: 's_curve',
  label: 'S-curve',
  generate({ samples, noise, seed }) {
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const t = (rand() - 0.5) * 3 * Math.PI;
      const sgn = t >= 0 ? 1 : -1;
      out.X[i * 3 + 0] = Math.sin(t);
      out.X[i * 3 + 1] = 4 * (rand() - 0.5);
      out.X[i * 3 + 2] = sgn * (Math.cos(t) - 1);
      out.t[i] = t;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

function projectToThreeViaPCA(rows) {
  const N = rows.length;
  if (N === 0) return { X: new Float64Array(0), t: new Float64Array(0), N: 0, empty: true };
  const d = rows[0].length;
  const mean = new Float64Array(d);
  for (const r of rows) for (let j = 0; j < d; j++) mean[j] += r[j];
  for (let j = 0; j < d; j++) mean[j] /= N;
  const centered = rows.map(r => r.map((x, j) => x - mean[j]));
  if (d <= 3) {
    const out = allocate(N);
    for (let i = 0; i < N; i++) {
      out.X[i * 3 + 0] = centered[i][0] || 0;
      out.X[i * 3 + 1] = centered[i][1] || 0;
      out.X[i * 3 + 2] = centered[i][2] || 0;
      out.t[i] = i / Math.max(1, N - 1);
    }
    return out;
  }
  if (typeof window === 'undefined' || !window.numeric) {
    throw new Error('CSV projection requires numeric.js (browser only)');
  }
  const C = [];
  for (let i = 0; i < d; i++) C.push(new Float64Array(d));
  for (const r of centered) {
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += r[i] * r[j];
  }
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] /= Math.max(1, N - 1);
  const Cm = C.map(row => Array.from(row));
  const eig = window.numeric.eig(Cm);
  const lam = eig.lambda.x.map((v, i) => ({ v: Math.abs(v), i }));
  lam.sort((a, b) => b.v - a.v);
  const V = eig.E.x;
  const out = allocate(N);
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < 3; k++) {
      let s = 0;
      const col = lam[k].i;
      for (let j = 0; j < d; j++) s += centered[i][j] * V[j][col];
      out.X[i * 3 + k] = s;
    }
    out.t[i] = i / Math.max(1, N - 1);
  }
  return out;
}

export const CSV_UPLOAD = {
  id: 'csv',
  label: 'Upload CSV...',
  generate({ csvRows }) {
    if (!csvRows || csvRows.length === 0) {
      return { X: new Float64Array(0), t: new Float64Array(0), N: 0, empty: true };
    }
    return projectToThreeViaPCA(csvRows);
  },
};

export function parseCSV(text) {
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
  if (lines.length === 0) return [];
  const first = lines[0].split(',').map(s => s.trim());
  const headerIsNumeric = first.every(c => c.length > 0 && Number.isFinite(Number(c)));
  const dataLines = headerIsNumeric ? lines : lines.slice(1);
  const rows = [];
  for (const line of dataLines) {
    const parts = line.split(',').map(s => Number(s.trim()));
    if (parts.length < 2) continue;
    if (!parts.every(v => Number.isFinite(v))) continue;
    rows.push(parts);
  }
  if (rows.length === 0) return [];
  const widths = rows.map(r => r.length);
  const mode = widths.sort((a, b) => widths.filter(x => x === a).length - widths.filter(x => x === b).length).pop();
  return rows.filter(r => r.length === mode);
}

export const DATASETS = [SWISS_ROLL, S_CURVE, CSV_UPLOAD];
export const DATASETS_BY_ID = Object.fromEntries(DATASETS.map(d => [d.id, d]));
```

- [ ] Step 4: Run tests to verify they pass

Run: `node --test test/manifold/`
Expected: all previously passing tests continue to pass; the new synthetic tests pass.

- [ ] Step 5: Commit

```
git add js/manifold/datasets.js test/manifold/datasets_synthetic.test.js
git commit -m "manifold: swiss roll, s-curve, and CSV upload via PCA projection"
```

---

### Task 6: Linear algebra helpers (pure)

Files:
- Create: `js/manifold/linalg.js`
- Create: `test/manifold/linalg_knn.test.js`
- Create: `test/manifold/linalg_dijkstra.test.js`
- Create: `test/manifold/linalg_doublecenter.test.js`

- [ ] Step 1: Write the kNN test

Write `test/manifold/linalg_knn.test.js`:
```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { knnGraph } from '../../js/manifold/linalg.js';

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
```

- [ ] Step 2: Write the Dijkstra test

Write `test/manifold/linalg_dijkstra.test.js`:
```javascript
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
```

- [ ] Step 3: Write the double-centering test

Write `test/manifold/linalg_doublecenter.test.js`:
```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { doubleCenterSquared } from '../../js/manifold/linalg.js';

test('Double-centered Gram matrix has zero row/column sums', () => {
  const D = new Float64Array([
    0, 1, 2,
    1, 0, 1,
    2, 1, 0,
  ]);
  const B = doubleCenterSquared(D, 3);
  for (let i = 0; i < 3; i++) {
    let row = 0, col = 0;
    for (let j = 0; j < 3; j++) {
      row += B[i*3 + j];
      col += B[j*3 + i];
    }
    assert.ok(Math.abs(row) < 1e-9, `row ${i} not zero: ${row}`);
    assert.ok(Math.abs(col) < 1e-9, `col ${i} not zero: ${col}`);
  }
});
```

- [ ] Step 4: Run tests to verify they fail

Run: `node --test test/manifold/linalg_knn.test.js test/manifold/linalg_dijkstra.test.js test/manifold/linalg_doublecenter.test.js`
Expected: module not found.

- [ ] Step 5: Implement `js/manifold/linalg.js`

Write `js/manifold/linalg.js`:
```javascript
export function mean3(X) {
  const N = X.length / 3;
  const m = [0, 0, 0];
  for (let i = 0; i < N; i++) {
    m[0] += X[i * 3]; m[1] += X[i * 3 + 1]; m[2] += X[i * 3 + 2];
  }
  m[0] /= N; m[1] /= N; m[2] /= N;
  return m;
}

export function center3(X) {
  const m = mean3(X);
  const N = X.length / 3;
  const out = new Float64Array(X.length);
  for (let i = 0; i < N; i++) {
    out[i * 3] = X[i * 3] - m[0];
    out[i * 3 + 1] = X[i * 3 + 1] - m[1];
    out[i * 3 + 2] = X[i * 3 + 2] - m[2];
  }
  return { Xc: out, mean: m };
}

export function covariance3(Xc) {
  const N = Xc.length / 3;
  const C = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < N; i++) {
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) C[a][b] += Xc[i * 3 + a] * Xc[i * 3 + b];
    }
  }
  const denom = Math.max(1, N - 1);
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) C[a][b] /= denom;
  return C;
}

export function squaredDist3(X, i, j) {
  const dx = X[i*3] - X[j*3];
  const dy = X[i*3+1] - X[j*3+1];
  const dz = X[i*3+2] - X[j*3+2];
  return dx*dx + dy*dy + dz*dz;
}

export function knnGraph(X, k) {
  const N = X.length / 3;
  const adj = Array.from({ length: N }, () => []);
  const edges = [];
  const seen = new Set();
  const dist = new Float64Array(N);
  const idx = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      dist[j] = i === j ? Infinity : squaredDist3(X, i, j);
      idx[j] = j;
    }
    idx.sort((a, b) => dist[a] - dist[b]);
    for (let m = 0; m < k; m++) {
      const j = idx[m];
      const w = Math.sqrt(dist[j]);
      adj[i].push([j, w]);
      const key = i < j ? `${i},${j}` : `${j},${i}`;
      if (!seen.has(key)) {
        seen.add(key);
        edges.push([Math.min(i,j), Math.max(i,j)]);
      }
    }
  }
  for (let i = 0; i < N; i++) {
    for (const [j, w] of adj[i]) {
      if (!adj[j].some(e => e[0] === i)) adj[j].push([i, w]);
    }
  }
  return { adj, edges };
}

class MinHeap {
  constructor() { this.data = []; }
  size() { return this.data.length; }
  push(item) {
    this.data.push(item);
    let i = this.data.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.data[p][0] <= this.data[i][0]) break;
      [this.data[p], this.data[i]] = [this.data[i], this.data[p]];
      i = p;
    }
  }
  pop() {
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length > 0) {
      this.data[0] = last;
      let i = 0;
      const n = this.data.length;
      while (true) {
        const l = 2*i + 1, r = 2*i + 2;
        let m = i;
        if (l < n && this.data[l][0] < this.data[m][0]) m = l;
        if (r < n && this.data[r][0] < this.data[m][0]) m = r;
        if (m === i) break;
        [this.data[m], this.data[i]] = [this.data[i], this.data[m]];
        i = m;
      }
    }
    return top;
  }
}

export function dijkstraAllPairs(adj, N) {
  const D = new Float64Array(N * N);
  for (let s = 0; s < N; s++) {
    for (let j = 0; j < N; j++) D[s * N + j] = Infinity;
    D[s * N + s] = 0;
    const heap = new MinHeap();
    heap.push([0, s]);
    while (heap.size() > 0) {
      const [d, u] = heap.pop();
      if (d > D[s * N + u]) continue;
      for (const [v, w] of adj[u]) {
        const nd = d + w;
        if (nd < D[s * N + v]) {
          D[s * N + v] = nd;
          heap.push([nd, v]);
        }
      }
    }
  }
  return D;
}

export function doubleCenterSquared(D, N) {
  const D2 = new Float64Array(N * N);
  for (let i = 0; i < N * N; i++) {
    const v = D[i];
    D2[i] = Number.isFinite(v) ? v * v : 0;
  }
  const rowMean = new Float64Array(N);
  const colMean = new Float64Array(N);
  let total = 0;
  for (let i = 0; i < N; i++) {
    let r = 0;
    for (let j = 0; j < N; j++) r += D2[i * N + j];
    rowMean[i] = r / N;
    total += r;
  }
  for (let j = 0; j < N; j++) {
    let c = 0;
    for (let i = 0; i < N; i++) c += D2[i * N + j];
    colMean[j] = c / N;
  }
  const grand = total / (N * N);
  const B = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      B[i * N + j] = -0.5 * (D2[i * N + j] - rowMean[i] - colMean[j] + grand);
    }
  }
  return B;
}

export function eigSymSorted3(M) {
  if (typeof window === 'undefined' || !window.numeric) {
    throw new Error('eigSymSorted3 requires numeric.js (browser only)');
  }
  const Mm = M.map(row => Array.from(row));
  const eig = window.numeric.eig(Mm);
  const lambda = eig.lambda.x;
  const V = eig.E.x;
  const n = lambda.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => lambda[b] - lambda[a]);
  const sortedLambda = order.map(i => lambda[i]);
  const sortedV = [];
  for (let i = 0; i < n; i++) {
    const col = [];
    for (let r = 0; r < n; r++) col.push(V[r][order[i]]);
    sortedV.push(col);
  }
  return { lambda: sortedLambda, vectors: sortedV };
}

export function topKSymmetricEig(M, N, k) {
  if (typeof window === 'undefined' || !window.numeric) {
    throw new Error('topKSymmetricEig requires numeric.js (browser only)');
  }
  const A = [];
  for (let i = 0; i < N; i++) {
    const row = new Array(N);
    for (let j = 0; j < N; j++) row[j] = M[i * N + j];
    A.push(row);
  }
  const eig = window.numeric.eig(A);
  const lambda = eig.lambda.x;
  const V = eig.E.x;
  const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => lambda[b] - lambda[a]);
  const outLambda = new Float64Array(k);
  const outVectors = [];
  for (let m = 0; m < k; m++) {
    outLambda[m] = lambda[order[m]];
    const col = new Float64Array(N);
    for (let r = 0; r < N; r++) col[r] = V[r][order[m]];
    outVectors.push(col);
  }
  return { lambda: outLambda, vectors: outVectors };
}
```

- [ ] Step 6: Run tests to verify they pass

Run: `node --test test/manifold/`
Expected: all tests passing including the three new linalg files.

- [ ] Step 7: Commit

```
git add js/manifold/linalg.js test/manifold/linalg_*.test.js
git commit -m "manifold: pure linalg helpers (knn, dijkstra, double-centering) plus eigen wrappers"
```

---

### Task 7: PCA algorithm module

Files:
- Create: `js/manifold/algorithms/pca.js`

This module relies on numeric.js (browser only), so it is verified in the browser at the end of Task 17.

- [ ] Step 1: Implement

Write `js/manifold/algorithms/pca.js`:
```javascript
import { center3, covariance3, eigSymSorted3 } from '../linalg.js';

function formatMatrix(C) {
  return C.map(row => row.map(v => v.toFixed(3).padStart(8)).join(' ')).join('\n');
}

export const PCA = {
  id: 'pca',
  label: 'PCA',
  params: [],
  pseudocode: [
    { id: 'pca-center', title: '1. Center the data', steps: ['1'],
      lines: ['mean ← (1/N) · Σ_i x_i', 'for i = 1..N: x_i ← x_i − mean'] },
    { id: 'pca-cov', title: '2. Form the covariance matrix', steps: ['3'],
      lines: ['C ← (1/(N−1)) · X_cᵀ X_c'] },
    { id: 'pca-eig', title: '3. Eigendecompose C', steps: ['5'],
      lines: ['C = V Λ Vᵀ (eigvecs in columns of V)', 'sort columns of V by decreasing eigenvalue'] },
    { id: 'pca-project', title: '4. Project to 2D', steps: ['6'],
      lines: ['Y ← X_c · V[:, 0:2]'] },
  ],
  run(dataset, _params) {
    const X = dataset.X;
    const t = dataset.t;
    const N = X.length / 3;
    const steps = new Map();

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: null,
      label: 'Raw data',
      ifw: {
        intuition: '<p>The raw 3D point cloud as the algorithm receives it. Points are coloured by an intrinsic parameter along the data manifold so that we can later see whether the embedding preserves that ordering.</p>',
        formula: null,
        worked: null,
      },
    });

    const { Xc, mean } = center3(X);
    steps.set('1', {
      points: Xc, t, edges: null, colors: null,
      label: 'Centered data',
      ifw: {
        intuition: '<p>PCA looks for directions of maximum variance, which is only meaningful around a fixed origin. We subtract the sample mean so the cloud is centred at the origin.</p>',
        formula: '$$\\bar{x} = \\frac{1}{N}\\sum_i x_i, \\qquad x_i \\leftarrow x_i - \\bar{x}$$',
        worked: `<p>The sample mean is approximately (${mean.map(v => v.toFixed(2)).join(', ')}).</p>`,
      },
    });

    const C = covariance3(Xc);
    steps.set('3', {
      points: Xc, t, edges: null, colors: null,
      label: 'Covariance matrix',
      ifw: {
        intuition: '<p>The 3x3 covariance matrix summarises how the centred coordinates co-vary. Its eigenvectors are the directions of maximal variance.</p>',
        formula: '$$C = \\frac{1}{N-1} X_c^{\\top} X_c$$',
        worked: `<pre>${formatMatrix(C)}</pre>`,
      },
    });

    const { lambda, vectors } = eigSymSorted3(C);
    const v1 = vectors[0], v2 = vectors[1];
    steps.set('5', {
      points: Xc, t, edges: null, colors: null,
      pcAxes: { v1, v2, lambda },
      label: 'Principal directions',
      ifw: {
        intuition: '<p>Decomposing C produces orthogonal directions ordered by how much variance the data exhibits along each. The first two axes form the 2D embedding basis.</p>',
        formula: '$$C = V \\Lambda V^{\\top}$$',
        worked: `<p>Eigenvalues: (${lambda.map(v => v.toFixed(2)).join(', ')}).</p>`,
      },
    });

    const embed2d = new Float64Array(N * 2);
    for (let i = 0; i < N; i++) {
      let a = 0, b = 0;
      for (let d = 0; d < 3; d++) {
        a += Xc[i * 3 + d] * v1[d];
        b += Xc[i * 3 + d] * v2[d];
      }
      embed2d[i * 2] = a;
      embed2d[i * 2 + 1] = b;
    }
    steps.set('6', {
      points: Xc, t, edges: null, colors: null, embed2d,
      label: 'Projected to 2D',
      ifw: {
        intuition: '<p>Each centred point is projected onto the plane spanned by the top two principal directions.</p>',
        formula: '$$y_i = (v_1^{\\top} x_i,\\; v_2^{\\top} x_i)$$',
        worked: '<p>The 3D thumbnail in the corner stays orbitable so you can compare the original cloud to the projection.</p>',
      },
    });

    return { steps, presentSubSteps: ['0', '1', '3', '5', '6'] };
  },
};
```

- [ ] Step 2: Commit

```
git add js/manifold/algorithms/pca.js
git commit -m "manifold: PCA algorithm pipeline keyed by canonical sub-step"
```

---

### Task 8: Isomap algorithm module

Files:
- Create: `js/manifold/algorithms/isomap.js`

- [ ] Step 1: Implement

Write `js/manifold/algorithms/isomap.js`:
```javascript
import { knnGraph, dijkstraAllPairs, doubleCenterSquared, topKSymmetricEig } from '../linalg.js';

export const ISOMAP = {
  id: 'isomap',
  label: 'Isomap',
  params: [{ name: 'k', type: 'int', default: 10, min: 2, max: 50 }],
  pseudocode: [
    { id: 'isomap-knn', title: '1. Build kNN graph', steps: ['2'],
      lines: ['for each i: neighbours_i ← k points with smallest ||x_j − x_i||',
              'edge weight w_{ij} ← ||x_j − x_i||'] },
    { id: 'isomap-geo', title: '2. Compute geodesic distances', steps: ['3'],
      lines: ['for each i: run Dijkstra from i on the kNN graph',
              'D_{ij} ← shortest-path distance in the graph'] },
    { id: 'isomap-dc', title: '3. Double-center the squared distance matrix', steps: ['4'],
      lines: ['B ← −(1/2) H D^2 H,  with H = I − (1/N) 1 1^T'] },
    { id: 'isomap-eig', title: '4. Eigendecompose B', steps: ['5'],
      lines: ['B = V Λ V^T  (take top-2 eigvals/vecs)'] },
    { id: 'isomap-embed', title: '5. Form 2D embedding', steps: ['6'],
      lines: ['Y ← V[:, 0:2] · diag(sqrt(λ_1), sqrt(λ_2))'] },
  ],
  run(dataset, params) {
    const X = dataset.X;
    const t = dataset.t;
    const N = X.length / 3;
    const k = Math.max(2, Math.min(params.k || 10, N - 1));
    const steps = new Map();

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: null,
      label: 'Raw data',
      ifw: { intuition: '<p>Isomap starts from the raw point cloud and recovers manifold geometry from local neighborhoods.</p>', formula: null, worked: null },
    });

    const { adj, edges } = knnGraph(X, k);
    steps.set('2', {
      points: X.slice(), t, edges, colors: null,
      label: 'kNN graph (k = ' + k + ')',
      ifw: {
        intuition: '<p>Connecting each point to its k nearest Euclidean neighbours approximates the manifold by a graph whose edges follow the local surface.</p>',
        formula: '$$\\mathcal{N}_i = \\{ j : \\|x_j - x_i\\| \\text{ among the } k \\text{ smallest}\\}$$',
        worked: `<p>The graph has ${edges.length} undirected edges over ${N} points.</p>`,
      },
    });

    const D = dijkstraAllPairs(adj, N);
    let connected = true;
    for (let i = 0; i < N * N; i++) if (!Number.isFinite(D[i])) { connected = false; break; }
    steps.set('3', {
      points: X.slice(), t, edges, colors: null,
      label: 'Geodesic distances',
      ifw: {
        intuition: '<p>Distances along the graph approximate true geodesic distances on the manifold.</p>',
        formula: '$$D_{ij} = \\min_{\\text{path } i \\to j \\text{ on } G} \\sum_e w_e$$',
        worked: connected ? '<p>Shortest paths computed with Dijkstra from every node. The graph is connected for this k.</p>' : '<p>The kNN graph is disconnected at this k. Increase k.</p>',
      },
    });

    const B = doubleCenterSquared(D, N);
    steps.set('4', {
      points: X.slice(), t, edges, colors: null,
      label: 'Double-centered Gram matrix',
      ifw: {
        intuition: '<p>Classical MDS converts pairwise squared distances into an inner-product matrix B via double centering.</p>',
        formula: '$$B = -\\tfrac{1}{2} H D^{(2)} H, \\quad H = I - \\tfrac{1}{N}\\mathbf{1}\\mathbf{1}^{\\top}$$',
        worked: '<p>Row, column, and grand means are subtracted from D^2 and the result is scaled by -1/2.</p>',
      },
    });

    const { lambda, vectors } = topKSymmetricEig(B, N, 2);
    steps.set('5', {
      points: X.slice(), t, edges, colors: null,
      label: 'Top-2 eigendecomposition',
      ifw: {
        intuition: '<p>The top eigenvectors of B reveal the dominant geometry of the geodesic distances.</p>',
        formula: '$$B = V \\Lambda V^{\\top}$$',
        worked: `<p>Top two eigenvalues: (${lambda[0].toFixed(2)}, ${lambda[1].toFixed(2)}).</p>`,
      },
    });

    const embed2d = new Float64Array(N * 2);
    const s1 = Math.sqrt(Math.max(0, lambda[0]));
    const s2 = Math.sqrt(Math.max(0, lambda[1]));
    for (let i = 0; i < N; i++) {
      embed2d[i * 2] = vectors[0][i] * s1;
      embed2d[i * 2 + 1] = vectors[1][i] * s2;
    }
    steps.set('6', {
      points: X.slice(), t, edges, colors: null, embed2d,
      label: 'Isomap embedding',
      ifw: {
        intuition: '<p>The 2D coordinates flatten the manifold while preserving geodesic distances as well as possible.</p>',
        formula: '$$y_i = \\big(\\sqrt{\\lambda_1}\\, v_{1,i},\\; \\sqrt{\\lambda_2}\\, v_{2,i}\\big)$$',
        worked: '<p>Each point\'s 2D coordinates are read off rows of the top-2 eigenvectors, scaled by the square roots of the eigenvalues.</p>',
      },
    });

    return { steps, presentSubSteps: ['0', '2', '3', '4', '5', '6'] };
  },
};
```

- [ ] Step 2: Commit

```
git add js/manifold/algorithms/isomap.js
git commit -m "manifold: Isomap pipeline keyed by canonical sub-step"
```

---

### Task 9: 3D orbit viewer

Files:
- Create: `js/manifold/viz3d.js`
- Modify: `styles/manifold.css` (append)

This module is verified manually in the browser at Task 17.

- [ ] Step 1: Implement

Write `js/manifold/viz3d.js`:
```javascript
function matmul(A, B) {
  const C = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
  return C;
}

function rotX(a) { const c = Math.cos(a), s = Math.sin(a); return [[1,0,0],[0,c,-s],[0,s,c]]; }
function rotY(a) { const c = Math.cos(a), s = Math.sin(a); return [[c,0,s],[0,1,0],[-s,0,c]]; }

function rainbow(t, tMin, tMax) {
  const u = (t - tMin) / Math.max(1e-9, tMax - tMin);
  const h = (1 - u) * 240;
  return `hsl(${h.toFixed(1)}, 80%, 60%)`;
}

export function createViz3d(container, { width = 480, height = 360, isThumbnail = false } = {}) {
  const d3 = window.d3;
  const wrap = d3.select(container).append('div').attr('class', isThumbnail ? 'viz3d-thumb' : 'viz3d');
  const svg = wrap.append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%').style('height', '100%')
    .style('touch-action', 'none').style('cursor', 'grab');
  const gEdges = svg.append('g').attr('class', 'edges');
  const gPoints = svg.append('g').attr('class', 'points');
  const gAxes = svg.append('g').attr('class', 'axes');

  let R = matmul(rotX(-0.35), rotY(0.6));
  let state = null;

  function computeBounds(points) {
    if (!points || points.length === 0) return { center: [0,0,0], radius: 1 };
    let xmn = Infinity, ymn = Infinity, zmn = Infinity;
    let xmx = -Infinity, ymx = -Infinity, zmx = -Infinity;
    const N = points.length / 3;
    for (let i = 0; i < N; i++) {
      const x = points[i*3], y = points[i*3+1], z = points[i*3+2];
      if (x < xmn) xmn = x; if (x > xmx) xmx = x;
      if (y < ymn) ymn = y; if (y > ymx) ymx = y;
      if (z < zmn) zmn = z; if (z > zmx) zmx = z;
    }
    const cx = (xmn+xmx)/2, cy = (ymn+ymx)/2, cz = (zmn+zmx)/2;
    const radius = Math.max(xmx-xmn, ymx-ymn, zmx-zmn, 1e-6) / 2;
    return { center: [cx, cy, cz], radius };
  }

  function project(R, X, scale, cx, cy) {
    const N = X.length / 3;
    const out = new Array(N);
    for (let i = 0; i < N; i++) {
      const x = X[i*3], y = X[i*3+1], z = X[i*3+2];
      const px = R[0][0]*x + R[0][1]*y + R[0][2]*z;
      const py = R[1][0]*x + R[1][1]*y + R[1][2]*z;
      const pz = R[2][0]*x + R[2][1]*y + R[2][2]*z;
      out[i] = { i, sx: cx + scale*px, sy: cy - scale*py, depth: pz };
    }
    return out;
  }

  function render() {
    if (!state) return;
    const { points, t, edges, colors } = state;
    const { center, radius } = state.bounds;
    const N = points.length / 3;
    const recentered = new Float64Array(points.length);
    for (let i = 0; i < N; i++) {
      recentered[i*3] = points[i*3] - center[0];
      recentered[i*3+1] = points[i*3+1] - center[1];
      recentered[i*3+2] = points[i*3+2] - center[2];
    }
    const margin = isThumbnail ? 6 : 18;
    const scale = (Math.min(width, height) / 2 - margin) / radius;
    const cx = width / 2, cy = height / 2;
    const proj = project(R, recentered, scale, cx, cy);

    if (edges && edges.length > 0) {
      const lines = edges.map(([a,b]) => {
        const pa = proj[a], pb = proj[b];
        return { x1: pa.sx, y1: pa.sy, x2: pb.sx, y2: pb.sy, depth: (pa.depth + pb.depth) / 2 };
      });
      lines.sort((a,b) => a.depth - b.depth);
      const sel = gEdges.selectAll('line').data(lines);
      sel.enter().append('line').merge(sel)
        .attr('x1', d => d.x1).attr('y1', d => d.y1)
        .attr('x2', d => d.x2).attr('y2', d => d.y2)
        .attr('stroke', 'rgba(255,255,255,0.18)')
        .attr('stroke-width', isThumbnail ? 0.4 : 0.6);
      sel.exit().remove();
    } else {
      gEdges.selectAll('line').remove();
    }

    let tMin = Infinity, tMax = -Infinity;
    if (t && !colors) for (let i = 0; i < t.length; i++) { if (t[i] < tMin) tMin = t[i]; if (t[i] > tMax) tMax = t[i]; }
    proj.sort((a,b) => a.depth - b.depth);
    const sel = gPoints.selectAll('circle').data(proj, d => d.i);
    sel.enter().append('circle').merge(sel)
      .attr('cx', d => d.sx).attr('cy', d => d.sy)
      .attr('r', isThumbnail ? 1.6 : 2.8)
      .attr('fill', d => colors ? colors[d.i] : (t ? rainbow(t[d.i], tMin, tMax) : '#7ec8ff'))
      .attr('opacity', d => {
        const z = (d.depth + radius) / (2 * radius);
        return 0.45 + 0.55 * Math.max(0, Math.min(1, z));
      });
    sel.exit().remove();
  }

  let dragging = false, lastX = 0, lastY = 0;
  svg.on('pointerdown', (event) => {
    dragging = true; lastX = event.clientX; lastY = event.clientY;
    svg.style('cursor', 'grabbing');
    svg.node().setPointerCapture(event.pointerId);
  });
  svg.on('pointermove', (event) => {
    if (!dragging) return;
    const dx = (event.clientX - lastX) * 0.008;
    const dy = (event.clientY - lastY) * 0.008;
    lastX = event.clientX; lastY = event.clientY;
    R = matmul(matmul(rotX(dy), rotY(dx)), R);
    render();
  });
  function endDrag(event) {
    dragging = false; svg.style('cursor', 'grab');
    try { svg.node().releasePointerCapture(event.pointerId); } catch (e) {}
  }
  svg.on('pointerup', endDrag);
  svg.on('pointercancel', endDrag);
  svg.on('pointerleave', endDrag);

  function setState(next) {
    state = {
      points: next.points,
      colors: next.colors || null,
      edges: next.edges || null,
      t: next.t || null,
      bounds: computeBounds(next.points),
    };
    render();
  }

  return { setState, render };
}
```

- [ ] Step 2: Append CSS for viz3d / viz3d-thumb

Append to `styles/manifold.css`:
```css
.manifold .viz3d { position: absolute; inset: 0; }
.manifold .viz3d-thumb { position: absolute; right: 8px; bottom: 8px; width: 32%; max-width: 160px; aspect-ratio: 4 / 3; background: rgba(0,0,0,0.55); border: 1px solid rgba(255,255,255,0.25); border-radius: var(--radius-sm); overflow: hidden; }
```

- [ ] Step 3: Commit

```
git add js/manifold/viz3d.js styles/manifold.css
git commit -m "manifold: d3 SVG 3D orbit viewer"
```

---

### Task 10: 2D scatter renderer

Files:
- Create: `js/manifold/viz2d.js`
- Modify: `styles/manifold.css` (append)

- [ ] Step 1: Implement

Write `js/manifold/viz2d.js`:
```javascript
function rainbow(t, tMin, tMax) {
  const u = (t - tMin) / Math.max(1e-9, tMax - tMin);
  const h = (1 - u) * 240;
  return `hsl(${h.toFixed(1)}, 80%, 60%)`;
}

export function createViz2d(container, { width = 480, height = 360 } = {}) {
  const d3 = window.d3;
  const wrap = d3.select(container).append('div').attr('class', 'viz2d');
  const svg = wrap.append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%').style('height', '100%');
  const gAxes = svg.append('g').attr('class', 'axes2d');
  const gPoints = svg.append('g').attr('class', 'points2d');

  function setState({ embed2d, colors, t }) {
    if (!embed2d || embed2d.length === 0) return;
    const N = embed2d.length / 2;
    let xmn = Infinity, xmx = -Infinity, ymn = Infinity, ymx = -Infinity;
    for (let i = 0; i < N; i++) {
      const x = embed2d[i*2], y = embed2d[i*2+1];
      if (x < xmn) xmn = x; if (x > xmx) xmx = x;
      if (y < ymn) ymn = y; if (y > ymx) ymx = y;
    }
    const padX = (xmx - xmn) * 0.08 || 1;
    const padY = (ymx - ymn) * 0.08 || 1;
    xmn -= padX; xmx += padX; ymn -= padY; ymx += padY;
    const margin = 28;
    const sx = (width - 2*margin) / Math.max(1e-9, xmx - xmn);
    const sy = (height - 2*margin) / Math.max(1e-9, ymx - ymn);
    const s = Math.min(sx, sy);
    const ox = margin + ((width - 2*margin) - s * (xmx - xmn)) / 2;
    const oy = margin + ((height - 2*margin) - s * (ymx - ymn)) / 2;
    let tMin = Infinity, tMax = -Infinity;
    if (t && !colors) for (let i = 0; i < t.length; i++) { if (t[i] < tMin) tMin = t[i]; if (t[i] > tMax) tMax = t[i]; }
    const data = new Array(N);
    for (let i = 0; i < N; i++) {
      data[i] = {
        i,
        sx: ox + s * (embed2d[i*2] - xmn),
        sy: height - (oy + s * (embed2d[i*2+1] - ymn)),
        col: colors ? colors[i] : (t ? rainbow(t[i], tMin, tMax) : '#7ec8ff'),
      };
    }
    const sel = gPoints.selectAll('circle').data(data, d => d.i);
    sel.enter().append('circle').merge(sel)
      .attr('cx', d => d.sx).attr('cy', d => d.sy)
      .attr('r', 2.8).attr('fill', d => d.col).attr('opacity', 0.9);
    sel.exit().remove();

    const axes = [
      { x1: margin, y1: height - margin, x2: width - margin, y2: height - margin },
      { x1: margin, y1: margin, x2: margin, y2: height - margin },
    ];
    const aSel = gAxes.selectAll('line').data(axes);
    aSel.enter().append('line').merge(aSel)
      .attr('x1', d => d.x1).attr('y1', d => d.y1)
      .attr('x2', d => d.x2).attr('y2', d => d.y2)
      .attr('stroke', 'rgba(255,255,255,0.25)').attr('stroke-width', 1);
    aSel.exit().remove();
  }

  return { setState };
}
```

- [ ] Step 2: Append CSS

Append to `styles/manifold.css`:
```css
.manifold .viz2d { position: absolute; inset: 0; }
```

- [ ] Step 3: Commit

```
git add js/manifold/viz2d.js styles/manifold.css
git commit -m "manifold: 2D scatter renderer for the final embedding"
```

---

### Task 11: Branching step indicator

Files:
- Create: `js/manifold/step_indicator.js`
- Modify: `styles/manifold.css` (append)

- [ ] Step 1: Implement

Write `js/manifold/step_indicator.js`:
```javascript
import { CANONICAL_STEPS, canonicalOf, compareSubSteps } from './canonical_steps.js';

export function createStepIndicator(container, { onJump }) {
  const d3 = window.d3;
  const root = d3.select(container).append('div').attr('class', 'step-indicator');
  const labelRow = root.append('div').attr('class', 'step-label-row');
  const labelA = labelRow.append('div').attr('class', 'step-label-a');
  const labelB = labelRow.append('div').attr('class', 'step-label-b');
  const svgWrap = root.append('div').attr('class', 'step-rails-wrap');
  const svg = svgWrap.append('svg').attr('class', 'step-rails').attr('preserveAspectRatio', 'xMidYMid meet');
  const navRow = root.append('div').attr('class', 'step-nav');
  const prevBtn = navRow.append('button').attr('class', 'step-prev').attr('type', 'button').text('◀ Prev');
  const desc = navRow.append('div').attr('class', 'step-desc');
  const nextBtn = navRow.append('button').attr('class', 'step-next').attr('type', 'button').text('Next ▶');
  prevBtn.on('click', () => onJump('prev'));
  nextBtn.on('click', () => onJump('next'));

  function groupByCanonical(subSteps) {
    const out = {};
    for (const id of subSteps) { const cid = canonicalOf(id); (out[cid] ||= []).push(id); }
    for (const cid in out) out[cid].sort(compareSubSteps);
    return out;
  }

  function drawColumnSide(svg, x, railY, dir, ids, current, side, gap, canonicalLabel) {
    if (ids.length === 0) {
      svg.append('circle').attr('cx', x).attr('cy', railY).attr('r', 4)
        .attr('fill', 'transparent').attr('stroke', 'rgba(255,255,255,0.18)')
        .attr('stroke-width', 1).attr('stroke-dasharray', '2,2');
      return;
    }
    let from = railY;
    for (let k = 0; k < ids.length; k++) {
      const y = ids.length === 1 ? railY : railY + dir * gap * (k + 1);
      if (ids.length > 1) {
        svg.append('line').attr('x1', x).attr('y1', from).attr('x2', x).attr('y2', y)
          .attr('stroke', 'rgba(255,255,255,0.4)').attr('stroke-width', 1.5);
      }
      drawNode(svg, x, y, ids[k], current === ids[k], side, canonicalLabel);
      from = y;
    }
  }

  function drawNode(svg, x, y, id, isCurrent, side, canonicalLabel) {
    const g = svg.append('g').attr('class', `step-node side-${side}${isCurrent ? ' is-current' : ''}`)
      .attr('cursor', 'pointer').on('click', () => onJump(id));
    g.append('circle').attr('cx', x).attr('cy', y).attr('r', isCurrent ? 8 : 6)
      .attr('fill', isCurrent ? (side === 'a' ? '#ff9f43' : '#54a0ff') : 'rgba(255,255,255,0.85)')
      .attr('stroke', isCurrent ? 'rgba(255,255,255,0.95)' : 'rgba(0,0,0,0.4)')
      .attr('stroke-width', isCurrent ? 2 : 1);
    g.append('text').attr('x', x + (side === 'a' ? -10 : 10)).attr('y', y - 10)
      .attr('text-anchor', side === 'a' ? 'end' : 'start')
      .attr('fill', isCurrent ? '#fff' : 'rgba(255,255,255,0.7)')
      .attr('font-size', 11).text(id);
    g.append('title').text(`${canonicalLabel} (${id})`);
  }

  function describe(sub, aLabel, bLabel, leftByC, rightByC) {
    const cid = canonicalOf(sub);
    const stepDef = CANONICAL_STEPS.find(s => s.id === cid);
    const inA = (leftByC[cid] || []).includes(sub);
    const inB = (rightByC[cid] || []).includes(sub);
    let who = '';
    if (inA && inB) who = `${aLabel} and ${bLabel}`;
    else if (inA) who = aLabel;
    else if (inB) who = bLabel;
    return `Step ${sub}: ${stepDef ? stepDef.label : ''} - ${who}`;
  }

  function render({ leftLabel, rightLabel, leftSubSteps, rightSubSteps, currentSubStep }) {
    labelA.text(`A: ${leftLabel}`);
    labelB.text(`B: ${rightLabel}`);
    const leftByC = groupByCanonical(leftSubSteps);
    const rightByC = groupByCanonical(rightSubSteps);
    const W = 720;
    const PAD = 36;
    const colW = (W - 2 * PAD) / Math.max(1, CANONICAL_STEPS.length - 1);
    const railAY = 36;
    const railBY = 108;
    const gap = 18;
    let maxA = 1, maxB = 1;
    for (let i = 0; i < CANONICAL_STEPS.length; i++) {
      const cid = CANONICAL_STEPS[i].id;
      maxA = Math.max(maxA, (leftByC[cid] || []).length);
      maxB = Math.max(maxB, (rightByC[cid] || []).length);
    }
    const top = Math.max(0, (maxA - 1) * gap);
    const bot = Math.max(0, (maxB - 1) * gap);
    const H = railBY + bot + 24;
    svg.attr('viewBox', `0 -${top} ${W} ${H + top}`);
    svg.selectAll('*').remove();

    svg.append('line').attr('x1', PAD).attr('y1', railAY).attr('x2', W - PAD).attr('y2', railAY)
      .attr('stroke', 'rgba(255,255,255,0.22)').attr('stroke-width', 2);
    svg.append('line').attr('x1', PAD).attr('y1', railBY).attr('x2', W - PAD).attr('y2', railBY)
      .attr('stroke', 'rgba(255,255,255,0.22)').attr('stroke-width', 2);

    for (let i = 0; i < CANONICAL_STEPS.length; i++) {
      const cid = CANONICAL_STEPS[i].id;
      const x = PAD + colW * i;
      drawColumnSide(svg, x, railAY, -1, leftByC[cid] || [], currentSubStep, 'a', gap, CANONICAL_STEPS[i].label);
      drawColumnSide(svg, x, railBY, +1, rightByC[cid] || [], currentSubStep, 'b', gap, CANONICAL_STEPS[i].label);
      svg.append('text').attr('x', x).attr('y', railBY + Math.max(1, (rightByC[cid] || []).length) * gap + 18)
        .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.55)')
        .attr('font-size', 11).text(cid);
    }

    desc.text(describe(currentSubStep, leftLabel, rightLabel, leftByC, rightByC));
    const all = [...new Set([...leftSubSteps, ...rightSubSteps])].sort(compareSubSteps);
    const idx = all.indexOf(currentSubStep);
    prevBtn.attr('disabled', idx <= 0 ? '' : null);
    nextBtn.attr('disabled', idx >= all.length - 1 ? '' : null);
  }

  return { render };
}
```

- [ ] Step 2: Append CSS

Append to `styles/manifold.css`:
```css
.manifold .step-indicator { display: flex; flex-direction: column; gap: 8px; }
.manifold .step-label-row { display: flex; justify-content: space-between; font-size: 0.95rem; }
.manifold .step-label-a { color: #ffb877; }
.manifold .step-label-b { color: #8ec2ff; }
.manifold .step-rails-wrap { width: 100%; overflow-x: auto; }
.manifold svg.step-rails { width: 100%; min-width: 600px; height: auto; display: block; }
.manifold .step-node text { pointer-events: none; user-select: none; }
.manifold .step-nav { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 4px; }
.manifold .step-desc { font-size: 0.95rem; opacity: 0.9; text-align: center; flex: 1; }
```

- [ ] Step 3: Commit

```
git add js/manifold/step_indicator.js styles/manifold.css
git commit -m "manifold: branching step indicator with vertical sub-step stacks"
```

---

### Task 12: IFW tab panel

Files:
- Create: `js/manifold/ifw.js`
- Modify: `styles/manifold.css` (append)

- [ ] Step 1: Implement

Write `js/manifold/ifw.js`:
```javascript
const TABS = [
  { key: 'intuition', label: 'Intuition' },
  { key: 'formula', label: 'Formula' },
  { key: 'worked', label: 'Worked example' },
];

export function createIFW(container, side) {
  const d3 = window.d3;
  const root = d3.select(container).append('div').attr('class', `ifw side-${side}`);
  const tabRow = root.append('div').attr('class', 'ifw-tabs');
  const content = root.append('div').attr('class', 'ifw-content');

  let active = 'intuition';
  let current = { intuition: null, formula: null, worked: null };
  const buttons = {};
  for (const { key, label } of TABS) {
    const btn = tabRow.append('button').attr('type', 'button').attr('class', `ifw-tab tab-${key}`).text(label);
    btn.on('click', () => {
      if (!current[key]) return;
      active = key; render();
    });
    buttons[key] = btn;
  }

  function render() {
    for (const { key } of TABS) {
      const has = !!current[key];
      buttons[key].attr('disabled', has ? null : '').classed('is-active', active === key && has).classed('is-disabled', !has);
    }
    const html = current[active] || '<div class="ifw-empty">No content for this step.</div>';
    content.html(html);
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([content.node()]).catch(() => {});
    }
  }

  function setStep(ifw) {
    current = {
      intuition: (ifw && ifw.intuition) || null,
      formula: (ifw && ifw.formula) || null,
      worked: (ifw && ifw.worked) || null,
    };
    if (!current[active]) {
      const next = TABS.find(t => current[t.key]);
      if (next) active = next.key;
    }
    render();
  }

  return { setStep };
}
```

- [ ] Step 2: Append CSS

Append to `styles/manifold.css`:
```css
.manifold .ifw { display: flex; flex-direction: column; gap: 8px; }
.manifold .ifw-tabs { display: flex; gap: 4px; border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 4px; }
.manifold .ifw-tab { padding: 6px 10px; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); border-radius: var(--radius-sm) var(--radius-sm) 0 0; cursor: pointer; font-size: 0.9rem; }
.manifold .ifw-tab.is-active { background: rgba(255,255,255,0.16); border-color: rgba(255,255,255,0.3); }
.manifold .ifw-tab.is-disabled, .manifold .ifw-tab[disabled] { opacity: 0.35; cursor: not-allowed; }
.manifold .ifw-content { min-height: 80px; font-size: 0.95rem; line-height: 1.45; }
.manifold .ifw-content pre { background: rgba(0,0,0,0.35); padding: 8px; border-radius: var(--radius-sm); overflow-x: auto; font-size: 0.85rem; }
.manifold .ifw-empty { opacity: 0.55; font-style: italic; }
```

- [ ] Step 3: Commit

```
git add js/manifold/ifw.js styles/manifold.css
git commit -m "manifold: IFW tab panel with disabled tabs when content is null"
```

---

### Task 13: Collapsible pseudocode block

Files:
- Create: `js/manifold/pseudocode.js`
- Modify: `styles/manifold.css` (append)

- [ ] Step 1: Implement

Write `js/manifold/pseudocode.js`:
```javascript
export function createPseudocode(container, side) {
  const d3 = window.d3;
  const root = d3.select(container).append('div').attr('class', `pseudocode side-${side}`);
  const title = root.append('div').attr('class', 'pseudocode-title');
  const list = root.append('div').attr('class', 'pseudocode-sections');
  const expanded = new Map();

  function matchesStep(section, sub) {
    if (!section.steps) return false;
    return section.steps.includes(sub);
  }

  function render({ algoLabel, sections, currentSubStep }) {
    title.text(`${algoLabel} pseudocode`);
    list.selectAll('*').remove();
    sections.forEach((section, idx) => {
      const key = section.id || `${idx}`;
      const isCurrent = matchesStep(section, currentSubStep);
      if (!expanded.has(key)) expanded.set(key, isCurrent);
      if (isCurrent) expanded.set(key, true);
      const open = expanded.get(key);
      const sec = list.append('div').attr('class', `pc-section${isCurrent ? ' is-current' : ''}`);
      const header = sec.append('div').attr('class', 'pc-section-header').attr('role', 'button').attr('tabindex', '0');
      header.append('span').attr('class', 'pc-chevron').text(open ? '▾' : '▸');
      header.append('span').attr('class', 'pc-section-title').text(section.title);
      if (section.steps && section.steps.length) {
        header.append('span').attr('class', 'pc-section-steps').text(`step ${section.steps.join(', ')}`);
      }
      const toggle = () => { expanded.set(key, !expanded.get(key)); render({ algoLabel, sections, currentSubStep }); };
      header.on('click', toggle);
      header.on('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggle(); }
      });
      const body = sec.append('pre').attr('class', 'pc-section-body');
      if (open) body.text(section.lines.join('\n'));
      else body.style('display', 'none');
    });
  }

  return { render };
}
```

- [ ] Step 2: Append CSS

Append to `styles/manifold.css`:
```css
.manifold .pseudocode { display: flex; flex-direction: column; gap: 6px; }
.manifold .pseudocode-title { font-weight: 700; font-size: 0.95rem; opacity: 0.85; }
.manifold .pc-section { border: 1px solid rgba(255,255,255,0.08); border-radius: var(--radius-sm); overflow: hidden; }
.manifold .pc-section.is-current { border-color: rgba(255,255,255,0.3); background: rgba(255,255,255,0.03); }
.manifold .pc-section-header { display: flex; align-items: center; gap: 8px; padding: 6px 10px; background: rgba(0,0,0,0.25); cursor: pointer; font-size: 0.92rem; }
.manifold .pc-section.is-current .pc-section-header { background: rgba(255,255,255,0.08); }
.manifold .pc-chevron { width: 1ch; text-align: center; opacity: 0.7; }
.manifold .pc-section-title { font-weight: 600; flex: 1; }
.manifold .pc-section-steps { font-size: 0.8rem; opacity: 0.6; }
.manifold .pc-section-body { margin: 0; padding: 8px 12px; background: rgba(0,0,0,0.35); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace; font-size: 0.85rem; white-space: pre-wrap; }
```

- [ ] Step 3: Commit

```
git add js/manifold/pseudocode.js styles/manifold.css
git commit -m "manifold: collapsible pseudocode renderer with current-step highlight"
```

---

### Task 14: State module

Files:
- Create: `js/manifold/state.js`

- [ ] Step 1: Implement

Write `js/manifold/state.js`:
```javascript
import { DATASETS_BY_ID } from './datasets.js';

export function createState({ algorithmsById, defaults }) {
  const listeners = new Set();
  const state = {
    datasetId: defaults.datasetId,
    datasetParams: { ...defaults.datasetParams },
    csvRows: null,
    csvFileName: '',
    leftAlgoId: defaults.leftAlgoId,
    rightAlgoId: defaults.rightAlgoId,
    leftAlgoParams: { ...defaults.algoParams[defaults.leftAlgoId] },
    rightAlgoParams: { ...defaults.algoParams[defaults.rightAlgoId] },
    currentSubStep: '0',
    cache: { dataset: null, left: null, right: null, key: null },
  };

  function key() {
    return JSON.stringify({
      d: state.datasetId, dp: state.datasetParams,
      csv: state.csvRows ? state.csvRows.length : 0,
      la: state.leftAlgoId, lp: state.leftAlgoParams,
      ra: state.rightAlgoId, rp: state.rightAlgoParams,
    });
  }

  function recompute() {
    const k = key();
    if (state.cache.key === k) return;
    const ds = DATASETS_BY_ID[state.datasetId];
    const data = ds.generate({ ...state.datasetParams, csvRows: state.csvRows });
    if (data.empty) {
      state.cache = { dataset: data, left: null, right: null, key: k };
      return;
    }
    const left = algorithmsById[state.leftAlgoId].run(data, state.leftAlgoParams);
    const right = algorithmsById[state.rightAlgoId].run(data, state.rightAlgoParams);
    state.cache = { dataset: data, left, right, key: k };
  }

  function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }
  function emit() { for (const fn of listeners) fn(state); }
  function set(updates) { Object.assign(state, updates); recompute(); emit(); }
  function setStep(sub) {
    if (state.currentSubStep === sub) return;
    state.currentSubStep = sub;
    emit();
  }

  recompute();
  return { state, subscribe, set, setStep, recompute };
}
```

- [ ] Step 2: Commit

```
git add js/manifold/state.js
git commit -m "manifold: central state with subscribe/recompute and cache fingerprint"
```

---

### Task 15: Main entry wiring (with CSV-in-dropdown UX)

Files:
- Create: `js/manifold/main.js`

This is the largest task. The CSV-in-dropdown UX is critical: selecting "Upload CSV..." in the dataset dropdown triggers the hidden file input; once a file is parsed it stays the active dataset and the dropdown stays on "Upload CSV (filename.csv)".

- [ ] Step 1: Implement

Write `js/manifold/main.js`:
```javascript
import { DATASETS, parseCSV } from './datasets.js';
import { PCA } from './algorithms/pca.js';
import { ISOMAP } from './algorithms/isomap.js';
import { createState } from './state.js';
import { createViz3d } from './viz3d.js';
import { createViz2d } from './viz2d.js';
import { createStepIndicator } from './step_indicator.js';
import { createIFW } from './ifw.js';
import { createPseudocode } from './pseudocode.js';
import { compareSubSteps, unionSubSteps } from './canonical_steps.js';

const ALGORITHMS = [PCA, ISOMAP];
const ALGORITHMS_BY_ID = Object.fromEntries(ALGORITHMS.map(a => [a.id, a]));

const defaults = {
  datasetId: 'swiss_roll',
  datasetParams: { samples: 300, noise: 0.0, seed: 7 },
  leftAlgoId: 'pca',
  rightAlgoId: 'isomap',
  algoParams: { pca: {}, isomap: { k: 10 } },
};

function init() {
  const $ = (id) => document.getElementById(id);
  const datasetSelect = $('mfDataset');
  const samplesInput = $('mfSamples');
  const noiseInput = $('mfNoise');
  const seedInput = $('mfSeed');
  const reseedBtn = $('mfReseed');
  const csvInput = $('mfCsvInput');
  const csvLabel = $('mfCsvLabel');
  const leftSelect = $('mfAlgoLeft');
  const rightSelect = $('mfAlgoRight');
  const leftParamsHost = $('mfAlgoLeftParams');
  const rightParamsHost = $('mfAlgoRightParams');
  const stepHost = $('mfStep');
  const leftVizHost = $('mfLeftViz');
  const rightVizHost = $('mfRightViz');
  const leftIfwHost = $('mfLeftIfw');
  const rightIfwHost = $('mfRightIfw');
  const leftPseudoHost = $('mfLeftPseudo');
  const rightPseudoHost = $('mfRightPseudo');
  const leftTitle = $('mfLeftTitle');
  const rightTitle = $('mfRightTitle');
  const samplesControl = $('mfSamplesControl');
  const noiseControl = $('mfNoiseControl');
  const seedControl = $('mfSeedControl');

  for (const ds of DATASETS) {
    const opt = document.createElement('option');
    opt.value = ds.id; opt.textContent = ds.label;
    if (ds.id === defaults.datasetId) opt.selected = true;
    datasetSelect.appendChild(opt);
  }
  for (const algo of ALGORITHMS) {
    const a = document.createElement('option');
    a.value = algo.id; a.textContent = algo.label;
    if (algo.id === defaults.leftAlgoId) a.selected = true;
    leftSelect.appendChild(a);
    const b = document.createElement('option');
    b.value = algo.id; b.textContent = algo.label;
    if (algo.id === defaults.rightAlgoId) b.selected = true;
    rightSelect.appendChild(b);
  }
  samplesInput.value = defaults.datasetParams.samples;
  noiseInput.value = defaults.datasetParams.noise;
  seedInput.value = defaults.datasetParams.seed;

  const store = createState({ algorithmsById: ALGORITHMS_BY_ID, defaults });

  const leftViz = createViz3d(leftVizHost, {});
  const rightViz = createViz3d(rightVizHost, {});
  const leftViz2d = createViz2d(leftVizHost, {});
  const rightViz2d = createViz2d(rightVizHost, {});
  const leftThumb = createViz3d(leftVizHost, { width: 140, height: 110, isThumbnail: true });
  const rightThumb = createViz3d(rightVizHost, { width: 140, height: 110, isThumbnail: true });
  hideEl(leftVizHost, '.viz2d'); hideEl(rightVizHost, '.viz2d');
  hideEl(leftVizHost, '.viz3d-thumb'); hideEl(rightVizHost, '.viz3d-thumb');

  const stepIndicator = createStepIndicator(stepHost, {
    onJump: (target) => {
      const left = store.state.cache.left;
      const right = store.state.cache.right;
      if (!left || !right) return;
      const all = unionSubSteps(left.presentSubSteps, right.presentSubSteps);
      if (target === 'prev' || target === 'next') {
        const idx = all.indexOf(store.state.currentSubStep);
        const next = target === 'prev' ? Math.max(0, idx - 1) : Math.min(all.length - 1, idx + 1);
        store.setStep(all[next]);
      } else if (all.includes(target)) {
        store.setStep(target);
      }
    },
  });

  const leftIfw = createIFW(leftIfwHost, 'a');
  const rightIfw = createIFW(rightIfwHost, 'b');
  const leftPseudo = createPseudocode(leftPseudoHost, 'a');
  const rightPseudo = createPseudocode(rightPseudoHost, 'b');

  function renderParamHost(host, algo, current, onChange) {
    host.innerHTML = '';
    for (const p of algo.params) {
      const wrap = document.createElement('label');
      wrap.className = 'mf-param';
      wrap.textContent = `${p.name} = `;
      const input = document.createElement('input');
      input.type = p.type === 'int' || p.type === 'float' ? 'number' : 'text';
      if (p.min !== undefined) input.min = p.min;
      if (p.max !== undefined) input.max = p.max;
      input.step = p.type === 'int' ? 1 : 'any';
      input.value = current[p.name] !== undefined ? current[p.name] : p.default;
      input.addEventListener('change', () => {
        const v = p.type === 'int' ? parseInt(input.value, 10) : parseFloat(input.value);
        onChange({ ...current, [p.name]: v });
      });
      wrap.appendChild(input);
      host.appendChild(wrap);
    }
    if (algo.params.length === 0) host.innerHTML = '<span class="mf-noparams">No parameters</span>';
  }

  function rebindParamHosts() {
    renderParamHost(leftParamsHost, ALGORITHMS_BY_ID[store.state.leftAlgoId], store.state.leftAlgoParams,
      (next) => store.set({ leftAlgoParams: next }));
    renderParamHost(rightParamsHost, ALGORITHMS_BY_ID[store.state.rightAlgoId], store.state.rightAlgoParams,
      (next) => store.set({ rightAlgoParams: next }));
  }
  rebindParamHosts();

  function updateSyntheticVisibility() {
    const isCsv = store.state.datasetId === 'csv';
    samplesControl.style.display = isCsv ? 'none' : '';
    noiseControl.style.display = isCsv ? 'none' : '';
    seedControl.style.display = isCsv ? 'none' : '';
    csvLabel.textContent = isCsv ? (store.state.csvFileName ? `Loaded: ${store.state.csvFileName} (${store.state.csvRows ? store.state.csvRows.length : 0} rows)` : '') : '';
  }

  datasetSelect.addEventListener('change', () => {
    const id = datasetSelect.value;
    if (id === 'csv') {
      csvInput.value = '';
      csvInput.click();
      return;
    }
    store.set({ datasetId: id, csvRows: null, csvFileName: '' });
    updateSyntheticVisibility();
  });
  csvInput.addEventListener('change', () => {
    const file = csvInput.files && csvInput.files[0];
    if (!file) {
      datasetSelect.value = store.state.datasetId === 'csv' ? 'csv' : store.state.datasetId;
      return;
    }
    file.text().then(text => {
      const rows = parseCSV(text);
      if (rows.length === 0) {
        csvLabel.textContent = `Could not parse "${file.name}". Need at least 2 numeric columns.`;
        datasetSelect.value = store.state.datasetId;
        return;
      }
      store.set({ datasetId: 'csv', csvRows: rows, csvFileName: file.name });
      datasetSelect.value = 'csv';
      updateSyntheticVisibility();
    });
  });
  samplesInput.addEventListener('change', () => {
    const v = Math.max(20, Math.min(1000, parseInt(samplesInput.value, 10) || 300));
    samplesInput.value = v;
    store.set({ datasetParams: { ...store.state.datasetParams, samples: v } });
  });
  noiseInput.addEventListener('change', () => {
    const v = Math.max(0, parseFloat(noiseInput.value) || 0);
    noiseInput.value = v;
    store.set({ datasetParams: { ...store.state.datasetParams, noise: v } });
  });
  seedInput.addEventListener('change', () => {
    const v = parseInt(seedInput.value, 10) || 0;
    store.set({ datasetParams: { ...store.state.datasetParams, seed: v } });
  });
  reseedBtn.addEventListener('click', () => {
    const v = Math.floor(Math.random() * 100000);
    seedInput.value = v;
    store.set({ datasetParams: { ...store.state.datasetParams, seed: v } });
  });
  leftSelect.addEventListener('change', () => {
    const id = leftSelect.value;
    const algo = ALGORITHMS_BY_ID[id];
    const params = {};
    for (const p of algo.params) params[p.name] = p.default;
    store.set({ leftAlgoId: id, leftAlgoParams: params });
    rebindParamHosts();
  });
  rightSelect.addEventListener('change', () => {
    const id = rightSelect.value;
    const algo = ALGORITHMS_BY_ID[id];
    const params = {};
    for (const p of algo.params) params[p.name] = p.default;
    store.set({ rightAlgoId: id, rightAlgoParams: params });
    rebindParamHosts();
  });

  store.subscribe((s) => {
    leftTitle.textContent = ALGORITHMS_BY_ID[s.leftAlgoId].label;
    rightTitle.textContent = ALGORITHMS_BY_ID[s.rightAlgoId].label;
    const left = s.cache.left, right = s.cache.right;
    if (!left || !right) {
      stepIndicator.render({
        leftLabel: ALGORITHMS_BY_ID[s.leftAlgoId].label,
        rightLabel: ALGORITHMS_BY_ID[s.rightAlgoId].label,
        leftSubSteps: ['0'], rightSubSteps: ['0'], currentSubStep: '0',
      });
      return;
    }
    const present = unionSubSteps(left.presentSubSteps, right.presentSubSteps);
    if (!present.includes(s.currentSubStep)) s.currentSubStep = present[0];
    const leftSub = nearestSub(s.currentSubStep, left.presentSubSteps);
    const rightSub = nearestSub(s.currentSubStep, right.presentSubSteps);
    const leftState = left.steps.get(leftSub);
    const rightState = right.steps.get(rightSub);

    const isFinal = s.currentSubStep === '6';
    setStep6Mode(leftVizHost, isFinal && leftState && leftState.embed2d);
    setStep6Mode(rightVizHost, isFinal && rightState && rightState.embed2d);

    if (isFinal && leftState && leftState.embed2d) {
      leftViz2d.setState({ embed2d: leftState.embed2d, colors: leftState.colors, t: leftState.t });
      leftThumb.setState({ points: leftState.points, t: leftState.t, edges: null, colors: null });
    } else if (leftState) {
      leftViz.setState({ points: leftState.points, t: leftState.t, edges: leftState.edges, colors: leftState.colors });
    }
    if (isFinal && rightState && rightState.embed2d) {
      rightViz2d.setState({ embed2d: rightState.embed2d, colors: rightState.colors, t: rightState.t });
      rightThumb.setState({ points: rightState.points, t: rightState.t, edges: null, colors: null });
    } else if (rightState) {
      rightViz.setState({ points: rightState.points, t: rightState.t, edges: rightState.edges, colors: rightState.colors });
    }

    stepIndicator.render({
      leftLabel: ALGORITHMS_BY_ID[s.leftAlgoId].label,
      rightLabel: ALGORITHMS_BY_ID[s.rightAlgoId].label,
      leftSubSteps: left.presentSubSteps,
      rightSubSteps: right.presentSubSteps,
      currentSubStep: s.currentSubStep,
    });
    leftIfw.setStep(leftState && leftSub === s.currentSubStep ? leftState.ifw : null);
    rightIfw.setStep(rightState && rightSub === s.currentSubStep ? rightState.ifw : null);
    leftPseudo.render({ algoLabel: ALGORITHMS_BY_ID[s.leftAlgoId].label, sections: ALGORITHMS_BY_ID[s.leftAlgoId].pseudocode, currentSubStep: leftSub });
    rightPseudo.render({ algoLabel: ALGORITHMS_BY_ID[s.rightAlgoId].label, sections: ALGORITHMS_BY_ID[s.rightAlgoId].pseudocode, currentSubStep: rightSub });
  });

  updateSyntheticVisibility();
  store.set({});
}

function nearestSub(target, present) {
  if (present.includes(target)) return target;
  const sorted = [...present].sort(compareSubSteps);
  let best = sorted[0];
  for (const id of sorted) if (compareSubSteps(id, target) <= 0) best = id;
  return best;
}

function setStep6Mode(host, isFinal) {
  const v3d = host.querySelector('.viz3d');
  const v2d = host.querySelector('.viz2d');
  const thumb = host.querySelector('.viz3d-thumb');
  if (!v3d || !v2d) return;
  if (isFinal) {
    v3d.style.display = 'none';
    v2d.style.display = '';
    if (thumb) thumb.style.display = '';
  } else {
    v3d.style.display = '';
    v2d.style.display = 'none';
    if (thumb) thumb.style.display = 'none';
  }
}

function hideEl(host, sel) {
  const el = host.querySelector(sel);
  if (el) el.style.display = 'none';
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
```

- [ ] Step 2: Commit

```
git add js/manifold/main.js
git commit -m "manifold: main entry, CSV in dropdown, all controls wired to state"
```

---

### Task 16: Browser smoke verification

Files: none created.

- [ ] Step 1: Start a static server

Run: `python3 -m http.server 8765 --bind 127.0.0.1`

- [ ] Step 2: Open the page

Open: `http://127.0.0.1:8765/pages/manifold.html`

- [ ] Step 3: Verify the dataset dropdown

Open the Dataset dropdown. Confirm it shows three entries in order: Swiss roll, S-curve, Upload CSV...

- [ ] Step 4: Verify both algorithm dropdowns

Open the Algorithm A dropdown. Confirm it shows two entries: PCA, Isomap. Same for Algorithm B. Confirm PCA is selected on the left, Isomap on the right. If either dropdown is empty, open the browser console: a module load error or import failure should be visible. Fix the underlying cause before continuing.

- [ ] Step 5: Verify the 3D viz renders both sides

Confirm the left panel shows a rainbow Swiss roll cloud, the right panel shows the same Swiss roll cloud with thin gray kNN edges overlaid (Isomap step 2 cannot be at the current sub-step yet, so the left side is at step 0 and the right side at step 0 too). Both viewports should be draggable.

- [ ] Step 6: Verify step navigation

Click Next ▶. The active sub-step should advance to the next ID in the union sequence (`1` for PCA on the left, `2` for Isomap on the right after a couple of clicks). Each side should snap to its nearest applicable sub-step.

- [ ] Step 7: Verify the IFW tabs

At step 0, the Intuition tab should have content; Formula and Worked example should be disabled (greyed out) for at least one of the algorithms. At step 1 the PCA side should have all three tabs enabled; at step 0 the Isomap side should have only Intuition.

- [ ] Step 8: Verify the pseudocode block

The pseudocode section that matches the current sub-step should be expanded and highlighted on each side. Clicking another section's chevron toggles it.

- [ ] Step 9: Verify the final embedding swap at step 6

Click through to the final sub-step. Each viewport should swap to a 2D scatter with axes labelled at the edges, and a small orbitable 3D thumbnail anchored bottom-right should show the original cloud.

- [ ] Step 10: Verify the CSV upload flow

Open the dataset dropdown and select "Upload CSV...". The hidden file picker should open. Choose a CSV with at least two numeric columns. The dropdown should remain on the CSV option, the label below should read `Loaded: <filename> (N rows)`, and the synthetic param controls (Samples/Noise/Seed/Reseed) should be hidden. The viz should redraw using the uploaded points.

- [ ] Step 11: Commit nothing (verification task)

No code changes here. If issues are found, file a follow-up task to fix and commit that.

---

### Task 17: Final test sweep and PR-ready commit

Files: none created.

- [ ] Step 1: Run all unit tests

Run: `node --test test/manifold/`
Expected: all tests passing.

- [ ] Step 2: Confirm git log shows the task progression

Run: `git log --oneline -- js/manifold pages/manifold.html styles/manifold.css index.html test/manifold`
Expected: one commit per task in order.

- [ ] Step 3: Confirm no straggling files

Run: `git status`
Expected: clean working tree apart from any pre-existing modified files outside this feature (`pages/fourier.html` was modified before this work and stays as is).

- [ ] Step 4: Stop the static server

Find the server PID with `ps -ef | grep http.server` and stop it with `kill <pid>`.

---

## Self-review

Spec coverage check against the design plan items:

- Page chrome reused from base.css/article.css: covered in Task 1.
- Top-to-bottom layout (dataset row, algorithms row, step indicator, side-by-side 3D viz, IFW, pseudocode): covered by the HTML in Task 1 plus the modules in Tasks 9 to 13.
- CSV upload inside the dataset dropdown (no separate row): covered by Task 5 (CSV_UPLOAD label "Upload CSV...") and Task 15 (datasetSelect change handler triggers `csvInput.click()`).
- Branching step indicator with vertical sub-step stacks: Task 11.
- Step 6 swaps to 2D scatter with mini orbitable 3D bottom-right: Task 10 plus the `setStep6Mode` helper in Task 15.
- IFW tabs disabled when content is null: Task 12.
- Collapsible pseudocode with current-step auto-expand and highlight: Task 13.
- PCA and Isomap pipelines keyed by canonical sub-step IDs: Tasks 7 and 8.
- Datasets: Swiss roll, S-curve, CSV via dropdown: Tasks 4, 5, 15.
- Canonical step contract with sub-step union/jumping logic: Tasks 2, 14, 15.

Placeholder scan: no TBDs, no "implement later", no "similar to". Each step contains the actual code or command needed.

Type consistency check: `points` is `Float64Array` length `3N` everywhere; `embed2d` is `Float64Array` length `2N`; `edges` is `[number, number][]`; `presentSubSteps` is `string[]`; the dataset id `csv` is consistent across `datasets.js`, `state.js`, `main.js`, and the CSV dropdown logic.

Known browser-only modules without node tests: `linalg.js` eigen wrappers, `pca.js`, `isomap.js`, `viz3d.js`, `viz2d.js`, `step_indicator.js`, `ifw.js`, `pseudocode.js`, `main.js`. These are exercised in Task 16's smoke checks.

Deferred: per-step coloring overrides driven by algorithm state (rainbow can be overridden by `colors`). The infrastructure is in place (StepState `colors` field is honored by both viz modules); only PCA and Isomap leave it null. Reconvene before Phase 2 to decide what each algorithm should color.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-26-manifold-learning-phase-1.md`. Two execution options:

1. Subagent-Driven (recommended): I dispatch a fresh subagent per task, review between tasks, fast iteration. Required sub-skill: superpowers:subagent-driven-development.

2. Inline Execution: Execute tasks in this session using superpowers:executing-plans, batch execution with checkpoints for review.

Which approach?
