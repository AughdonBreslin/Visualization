# Manifold Step Experience Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: Replace the step indicator panel, the per-step viewport content, and the worked-example tab format on the manifold learning page, following the locked design at `docs/superpowers/specs/2026-05-28-manifold-step-experience-design.md`.

Architecture: Three coupled surfaces share a new `vizKind` field on each StepState. A new step-viz dispatcher mounts one of five renderer modules (point cloud, centering animation, kNN with hover, matrix strip, spectral overlay) based on `vizKind`. Worked-example content is computed inside each algorithm's per-step task using a new `format.js` helper module, then rendered as three labelled sections by the existing IFW component.

Tech Stack: Vanilla ES modules, d3 v7 (global), no build step. Node 22 `node --test` for unit-testable pure logic. Manual browser smoke for DOM-heavy modules.

User constraints (apply to every code, comment, commit message, and markdown file produced):
- No em-dashes (`—`, `&mdash;`) anywhere.
- No `<em>`, `<strong>`, `<b>`, `<i>`, `<mark>` HTML tags.
- No markdown emphasis (`*`, `**`, `_`, `__`).

---

## File structure

Files modified:

- `js/manifold/step_indicator.js` (full rewrite to match locked design).
- `js/manifold/main.js` (replace per-step viewport switching with dispatcher; remove existing viz3d/viz2d host wiring inside the subscriber).
- `js/manifold/ifw.js` (no behavior change; styled HTML emitted by algorithm modules carries the new section classes).
- `js/manifold/algorithms/pca.js` (add `vizKind` per step and rewrite `worked` strings using format helpers).
- `js/manifold/algorithms/isomap.js` (same as PCA).
- `js/manifold/viz3d.js` (no contract change; still used for step 0 cloud and for the step-5 / step-6 mini-thumbnail).
- `styles/manifold.css` (remove the old `.step-indicator` / `.step-rails-*` / `.step-node` rules; add new rules for `.sp-*` panel, bar, edge, detail; add rules for `.viz-matrix-strip`, `.viz-spectral`, `.viz-centering`, `.viz-knn`; add IFW worked-example section rules).

Files created:

- `js/manifold/format.js` (number formatting helpers: `formatVec3`, `formatMatrix`, `formatTable`).
- `js/manifold/step_viz.js` (renderer dispatcher).
- `js/manifold/viz/viz_centering.js` (Pattern 1 renderer).
- `js/manifold/viz/viz_knn.js` (Pattern 2 renderer; absorbs the kNN edge logic currently inline in `viz3d.js`).
- `js/manifold/viz/viz_matrix_strip.js` (Pattern 3 renderer).
- `js/manifold/viz/viz_spectral.js` (Pattern 4 renderer).
- `test/manifold/format.test.js` (unit tests for the format helpers).

Files left alone: `js/manifold/canonical_steps.js`, `js/manifold/rng.js`, `js/manifold/datasets.js`, `js/manifold/linalg.js`, `js/manifold/state.js`, `js/manifold/viz2d.js`, `js/manifold/pseudocode.js`, `js/manifold/worker.js`, `pages/manifold.html`.

The preview HTML files (`pages/manifold-preview-steps.html`, `pages/manifold-preview-viz.html`, `pages/manifold-preview-worked.html`) stay in the repo as visual references during implementation. They are not loaded by the main page.

---

## StepState contract addition

Each algorithm step state gains an optional `vizKind: string`. Valid values:

- `'point_cloud'`: rainbow-coloured 3D scatter rendered by `viz3d.js` (steps 0).
- `'centering'`: Pattern 1 (PCA step 1).
- `'knn_graph'`: Pattern 2 (Isomap step 2).
- `'matrix_strip'`: Pattern 3 (PCA step 3, Isomap step 3, Isomap step 4).
- `'spectral'`: Pattern 4 (steps 5).
- `'embedding'`: 2D scatter + 3D mini (step 6, existing rendering).

For `'matrix_strip'`, the StepState also carries `panes`: a 3-element array of pane descriptors. Each pane descriptor is `{ kind, label, data }`:
- `kind: 'cloud_thumb'`, `data: Float64Array` length 3N: a 3D mini cloud rendered into a small SVG.
- `kind: 'graph_thumb'`, `data: { points: Float64Array length 3N, edges: [number,number][] }`: cloud plus all kNN edges.
- `kind: 'graph_thumb_with_path'`, `data: { points, edges, pathEdges: [number,number][] }`: cloud plus all edges plus bright highlighted path edges.
- `kind: 'matrix_numbers'`, `data: number[][]`: an n x n grid of numeric cells (used for 3x3 PCA matrices).
- `kind: 'heatmap'`, `data: { matrix: Float64Array length N*N, N: number, highlightRow?: number }`: a colored heatmap.

The StepState also carries `paneOpLabels: [string, string]` for the two arrow labels between panes.

For `'spectral'`, the StepState carries:
- `algoId: 'pca' | 'isomap'` (already implied by the algorithm running it).
- For PCA: `pcAxes: { v1, v2, v3 } (each Float64Array length 3)`, `lambda: Float64Array length 3` (already partly there).
- For Isomap: `v1Values: Float64Array length N` (top-1 eigenvector entries per point), `topEigvals: Float64Array length 8` (top 8 eigenvalues for the mini bar chart).

---

# Tasks

### Task 1: Rewrite step indicator (JS)

Files:
- Modify: `js/manifold/step_indicator.js` (full replacement)

- [ ] Step 1: Replace the contents of `js/manifold/step_indicator.js` with this exact content:

```javascript
import { CANONICAL_STEPS, canonicalOf, compareSubSteps } from './canonical_steps.js';

function nearestSub(target, present) {
  if (present.includes(target)) return target;
  const sorted = [...present].sort(compareSubSteps);
  let best = sorted[0];
  for (const id of sorted) if (compareSubSteps(id, target) <= 0) best = id;
  return best;
}

export function createStepIndicator(container, { onJump }) {
  const d3 = window.d3;
  const root = d3.select(container).append('div').attr('class', 'sp-frame');

  const panels = root.append('div').attr('class', 'sp-panels');
  const panelA = panels.append('div').attr('class', 'sp-panel');
  const headerA = panelA.append('div').attr('class', 'sp-panel-header');
  const barA = panelA.append('div').attr('class', 'sp-bar');
  const detailA = panelA.append('div').attr('class', 'sp-detail');

  const panelB = panels.append('div').attr('class', 'sp-panel');
  const headerB = panelB.append('div').attr('class', 'sp-panel-header');
  const barB = panelB.append('div').attr('class', 'sp-bar');
  const detailB = panelB.append('div').attr('class', 'sp-detail');

  const navRow = root.append('div').attr('class', 'sp-nav');
  const prevBtn = navRow.append('button').attr('class', 'step-prev').attr('type', 'button').text('◀ Prev');
  const descEl = navRow.append('div').attr('class', 'step-desc');
  const nextBtn = navRow.append('button').attr('class', 'step-next').attr('type', 'button').text('Next ▶');
  prevBtn.on('click', () => onJump('prev'));
  nextBtn.on('click', () => onJump('next'));

  function toggleExpanded() {
    const open = root.classed('is-expanded');
    root.classed('is-expanded', !open);
  }

  function classifyDot(stepId, presentSet, nearest) {
    if (!presentSet.has(stepId)) return 'na';
    return compareSubSteps(stepId, nearest) <= 0 ? 'filled' : 'hollow';
  }

  function renderBar(barSel, presentSubSteps, nearest) {
    barSel.html('');
    const presentSet = new Set(presentSubSteps);
    for (let i = 0; i < CANONICAL_STEPS.length; i++) {
      const cid = CANONICAL_STEPS[i].id;
      const state = classifyDot(cid, presentSet, nearest);
      const cell = barSel.append('div').attr('class', 'sp-cell' + (state === 'na' ? ' na' : ''));
      cell.append('span').attr('class', 'sp-dot ' + state);
      cell.append('span').attr('class', 'sp-num' + (state === 'na' ? ' na' : '')).text(cid);
      if (state !== 'na') cell.on('click', (event) => { event.stopPropagation(); onJump(cid); });
      if (i < CANONICAL_STEPS.length - 1) {
        barSel.append('div').attr('class', 'sp-edge').on('click', toggleExpanded);
      }
    }
  }

  function renderDetail(detailSel, presentSubSteps, nearest, globalCurrent) {
    detailSel.html('');
    const presentSet = new Set(presentSubSteps);
    for (let i = 0; i < CANONICAL_STEPS.length; i++) {
      const step = CANONICAL_STEPS[i];
      const cid = step.id;
      let rowClass = 'sp-step';
      let label;
      if (!presentSet.has(cid)) {
        rowClass += ' na';
        label = cid + ' · not used';
      } else {
        const cmp = compareSubSteps(cid, nearest);
        if (cid === nearest && cid === globalCurrent) rowClass += ' current';
        else if (cmp < 0) rowClass += ' past';
        else if (cmp === 0) rowClass += ' current';
        else rowClass += ' future';
        label = cid + ' · ' + step.label;
      }
      const row = detailSel.append('div').attr('class', rowClass);
      row.append('span').attr('class', 'sp-step-dot');
      row.append('span').text(label);
      if (presentSet.has(cid)) row.on('click', () => onJump(cid));
    }
  }

  function render({ leftLabel, rightLabel, leftSubSteps, rightSubSteps, currentSubStep }) {
    headerA.text('A · ' + leftLabel);
    headerB.text('B · ' + rightLabel);

    const leftNearest = nearestSub(currentSubStep, leftSubSteps);
    const rightNearest = nearestSub(currentSubStep, rightSubSteps);

    renderBar(barA, leftSubSteps, leftNearest);
    renderBar(barB, rightSubSteps, rightNearest);
    renderDetail(detailA, leftSubSteps, leftNearest, currentSubStep);
    renderDetail(detailB, rightSubSteps, rightNearest, currentSubStep);

    const cid = canonicalOf(currentSubStep);
    const stepDef = CANONICAL_STEPS.find(s => s.id === cid);
    const inA = leftSubSteps.includes(currentSubStep);
    const inB = rightSubSteps.includes(currentSubStep);
    let who = '';
    if (inA && inB) who = leftLabel + ' and ' + rightLabel;
    else if (inA) who = leftLabel;
    else if (inB) who = rightLabel;
    descEl.text('Step ' + currentSubStep + ': ' + (stepDef ? stepDef.label : '') + ' - ' + who);

    const all = [...new Set([...leftSubSteps, ...rightSubSteps])].sort(compareSubSteps);
    const idx = all.indexOf(currentSubStep);
    prevBtn.attr('disabled', idx <= 0 ? '' : null);
    nextBtn.attr('disabled', idx >= all.length - 1 ? '' : null);
  }

  return { render };
}
```

- [ ] Step 2: Run syntax check

Run: `node --check js/manifold/step_indicator.js`
Expected: no output.

- [ ] Step 3: Commit

```bash
git add js/manifold/step_indicator.js
git commit -m "manifold: rewrite step indicator with compressed/expanded panels"
```

---

### Task 2: Step indicator CSS

Files:
- Modify: `styles/manifold.css` (remove the old `.step-indicator` block; append new `.sp-*` rules)

- [ ] Step 1: Remove the old step indicator rules

Open `styles/manifold.css`. Find and delete this block exactly (it was added in the original phase-1 Task 11):

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

- [ ] Step 2: Append the new step indicator rules

Append exactly this block to the end of `styles/manifold.css`:

```css
.manifold .sp-frame { background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 14px; }
.manifold .sp-panels { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.manifold .sp-panel { background: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 12px; }
.manifold .sp-panel-header { font-weight: 700; font-size: 0.95rem; margin-bottom: 4px; color: rgba(255,255,255,0.95); }
.manifold .sp-bar { display: flex; align-items: center; padding: 6px 6px 22px; }
.manifold .sp-cell { position: relative; width: 18px; height: 18px; display: flex; align-items: center; justify-content: center; cursor: pointer; flex-shrink: 0; }
.manifold .sp-cell:hover .sp-dot { transform: scale(1.18); }
.manifold .sp-cell.na { cursor: default; }
.manifold .sp-cell.na:hover .sp-dot { transform: none; }
.manifold .sp-dot { width: 11px; height: 11px; border-radius: 50%; transition: transform 120ms; box-sizing: border-box; }
.manifold .sp-dot.filled { background: rgba(255,255,255,0.92); }
.manifold .sp-dot.hollow { border: 1.5px solid rgba(255,255,255,0.7); background: transparent; }
.manifold .sp-dot.na { background: rgba(255,255,255,0.22); width: 8px; height: 8px; }
.manifold .sp-num { position: absolute; top: 100%; left: 50%; transform: translateX(-50%); margin-top: 4px; font-size: 0.72rem; opacity: 0.55; line-height: 1; }
.manifold .sp-num.na { opacity: 0.3; }
.manifold .sp-edge { flex: 1; min-width: 18px; height: 18px; display: flex; align-items: center; cursor: zoom-in; padding: 0 2px; }
.manifold .sp-frame.is-expanded .sp-edge { cursor: zoom-out; }
.manifold .sp-edge::before { content: ''; flex: 1; height: 2px; background: rgba(255,255,255,0.18); transition: background 120ms; border-radius: 1px; }
.manifold .sp-edge:hover::before { background: rgba(255,255,255,0.5); }
.manifold .sp-detail { display: none; border-top: 1px solid rgba(255,255,255,0.1); margin-top: 4px; padding-top: 8px; }
.manifold .sp-frame.is-expanded .sp-detail { display: block; }
.manifold .sp-step { display: flex; align-items: center; gap: 10px; padding: 6px 6px; border-radius: 6px; font-size: 0.93rem; cursor: pointer; transition: background 120ms; }
.manifold .sp-step:hover { background: rgba(255,255,255,0.06); }
.manifold .sp-step .sp-step-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; box-sizing: border-box; }
.manifold .sp-step.past .sp-step-dot, .manifold .sp-step.current .sp-step-dot { background: rgba(255,255,255,0.92); }
.manifold .sp-step.future .sp-step-dot { border: 1.5px solid rgba(255,255,255,0.7); background: transparent; }
.manifold .sp-step.na .sp-step-dot { background: rgba(255,255,255,0.22); width: 7px; height: 7px; }
.manifold .sp-step.future { opacity: 0.75; }
.manifold .sp-step.na { opacity: 0.4; cursor: default; font-style: italic; }
.manifold .sp-step.na:hover { background: transparent; }
.manifold .sp-step.current { font-weight: 600; background: rgba(255,255,255,0.06); }
.manifold .sp-nav { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.08); }
.manifold .sp-nav button { padding: 6px 12px; font-size: 0.95rem; }
.manifold .step-desc { flex: 1; text-align: center; opacity: 0.9; font-size: 0.95rem; }
```

- [ ] Step 3: Commit

```bash
git add styles/manifold.css
git commit -m "manifold: replace step indicator CSS with compressed panel design"
```

---

### Task 3: Format helpers (with TDD)

Files:
- Create: `js/manifold/format.js`
- Create: `test/manifold/format.test.js`

- [ ] Step 1: Write the failing tests

Write `test/manifold/format.test.js`:

```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { formatVec3, formatMatrix, formatTable } from '../../js/manifold/format.js';

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
```

- [ ] Step 2: Run tests to verify they fail

Run: `node --test test/manifold/format.test.js`
Expected: failures (module not found).

- [ ] Step 3: Implement `js/manifold/format.js`

Write `js/manifold/format.js`:

```javascript
function pad(s, width) {
  return s.length >= width ? s : ' '.repeat(width - s.length) + s;
}

export function formatVec3(v, { digits = 3, width = 6 } = {}) {
  const parts = Array.from(v).map(x => pad(Number(x).toFixed(digits), width));
  return '(' + parts.join(', ') + ')';
}

export function formatMatrix(M, { digits = 3, maxRows = null, maxCols = null, width = 7 } = {}) {
  const rows = maxRows !== null ? M.slice(0, maxRows) : M;
  const truncatedRows = maxRows !== null && M.length > maxRows;
  const lines = rows.map(row => {
    const arr = Array.from(row);
    const truncatedCols = maxCols !== null && arr.length > maxCols;
    const shown = maxCols !== null ? arr.slice(0, maxCols) : arr;
    const cells = shown.map(x => pad(Number(x).toFixed(digits), width));
    return '[ ' + cells.join(', ') + (truncatedCols ? ', ...' : '') + ' ]';
  });
  if (truncatedRows) lines.push('  ...');
  return lines.join('\n');
}

export function formatTable(headers, rows) {
  const widths = headers.map((h, i) => {
    let w = String(h).length;
    for (const row of rows) w = Math.max(w, String(row[i]).length);
    return w;
  });
  const fmtRow = (cells) => cells.map((c, i) => pad(String(c), widths[i])).join(' | ');
  const headerLine = fmtRow(headers);
  return [headerLine, ...rows.map(fmtRow)].join('\n');
}
```

- [ ] Step 4: Run tests to verify they pass

Run: `node --test test/manifold/format.test.js`
Expected: 4 passing.

- [ ] Step 5: Run the full test suite to confirm no regression

Run: `node --test 'test/manifold/*.test.js' 2>&1 | tail -5`
Expected: all tests passing (27 total: 23 previous plus 4 new).

- [ ] Step 6: Commit

```bash
git add js/manifold/format.js test/manifold/format.test.js
git commit -m "manifold: number formatting helpers (formatVec3, formatMatrix, formatTable)"
```

---

### Task 4: IFW worked-example CSS

Files:
- Modify: `styles/manifold.css` (append)

- [ ] Step 1: Append the worked-example section rules

Append exactly this block to the end of `styles/manifold.css`:

```css
.manifold .ifw-worked-section { margin: 10px 0; }
.manifold .ifw-worked-label { font-size: 0.78rem; opacity: 0.55; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
.manifold .ifw-worked-body { background: rgba(0,0,0,0.4); border-left: 2px solid rgba(255,255,255,0.25); padding: 8px 12px; border-radius: 0 4px 4px 0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace; font-size: 0.85rem; line-height: 1.5; white-space: pre-wrap; }
.manifold .ifw-worked-body.math { font-family: inherit; font-size: 0.95rem; padding-top: 6px; padding-bottom: 6px; }
```

- [ ] Step 2: Commit

```bash
git add styles/manifold.css
git commit -m "manifold: IFW worked-example section styling"
```

---

### Task 5: Update PCA module with vizKind and Format-1 worked examples

Files:
- Modify: `js/manifold/algorithms/pca.js`

- [ ] Step 1: Replace the contents of `js/manifold/algorithms/pca.js` with this exact content:

```javascript
import { center3, covariance3, eigSymSorted3 } from '../linalg.js';
import { formatVec3, formatMatrix, formatTable } from '../format.js';

function sampleIndices(N) {
  return [Math.floor(N * 0.2), Math.floor(N * 0.5), Math.floor(N * 0.8)];
}

function workedSections(input, formula, output) {
  return (
    '<div class="ifw-worked-section">' +
      '<div class="ifw-worked-label">Input (from previous step)</div>' +
      '<div class="ifw-worked-body">' + input + '</div>' +
    '</div>' +
    '<div class="ifw-worked-section">' +
      '<div class="ifw-worked-label">Formula</div>' +
      '<div class="ifw-worked-body math">' + formula + '</div>' +
    '</div>' +
    '<div class="ifw-worked-section">' +
      '<div class="ifw-worked-label">Output (after this step)</div>' +
      '<div class="ifw-worked-body">' + output + '</div>' +
    '</div>'
  );
}

function rowOf(X, i) {
  return [X[i * 3], X[i * 3 + 1], X[i * 3 + 2]];
}

export const PCA = {
  id: 'pca',
  label: 'PCA',
  params: [],
  presentSubSteps: ['0', '1', '3', '5', '6'],
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
    const presentSubSteps = ['0', '1', '3', '5', '6'];
    const pending = new Set(['1', '3', '5', '6']);
    let cancelled = false;
    const samples = sampleIndices(N);

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: null,
      vizKind: 'point_cloud',
      label: 'Raw data',
      ifw: {
        intuition: '<p>The raw 3D point cloud as the algorithm receives it. Points are coloured by an intrinsic parameter along the data manifold so that we can later see whether the embedding preserves that ordering.</p>',
        formula: null,
        worked: null,
      },
    });

    function start(onProgress) {
      const mem = {};
      const tasks = [
        () => {
          const { Xc, mean } = center3(X);
          mem.Xc = Xc;
          mem.mean = mean;
          const inputRows = samples.map(i => [i, formatVec3(rowOf(X, i))]);
          const inputTable = formatTable(['i', 'x_i'], inputRows);
          const inputBlock = inputTable + '\n\nmean μ = ' + formatVec3(mean);
          const outputRows = samples.map(i => {
            const xi = rowOf(X, i);
            const xc = rowOf(Xc, i);
            return [i, formatVec3(xi) + ' − μ', formatVec3(xc)];
          });
          const outputBlock = formatTable(['i', 'x_i − μ', 'x_i (centered)'], outputRows);
          steps.set('1', {
            points: Xc, t, edges: null, colors: null,
            vizKind: 'centering',
            rawPoints: X.slice(),
            label: 'Centered data',
            ifw: {
              intuition: '<p>PCA looks for directions of maximum variance, which is only meaningful around a fixed origin. We subtract the sample mean so the cloud is centred at the origin.</p>',
              formula: '$$\\bar{x} = \\frac{1}{N}\\sum_i x_i, \\qquad x_i \\leftarrow x_i - \\bar{x}$$',
              worked: workedSections(inputBlock, '$$x_i \\leftarrow x_i - \\mu$$', outputBlock),
            },
          });
          pending.delete('1');
          if (onProgress) onProgress('1');
        },
        () => {
          const Xc = mem.Xc;
          const C = covariance3(Xc);
          mem.C = C;
          const inputRows = samples.map(i => [i, formatVec3(rowOf(Xc, i))]);
          const inputBlock = formatTable(['i', 'x_i (centered)'], inputRows) + '\n\nN = ' + N;
          let sum00 = 0;
          for (let i = 0; i < N; i++) sum00 += Xc[i * 3] * Xc[i * 3];
          const computeLine = 'sum_i x_{i,1} x_{i,1} = ' + sum00.toFixed(3) +
            '\nC[0][0] = ' + sum00.toFixed(3) + ' / (' + N + ' − 1) = ' + C[0][0].toFixed(4);
          const outputBlock = computeLine + '\n\nC =\n' + formatMatrix(C, { digits: 4 });
          steps.set('3', {
            points: Xc, t, edges: null, colors: null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'cloud_thumb', label: 'X_c (centered)', data: Xc },
              { kind: 'matrix_numbers', label: 'X_cᵀ X_c (raw sum)', data: C.map(row => row.map(v => v * (N - 1))) },
              { kind: 'matrix_numbers', label: 'C = ÷ (N − 1)', data: C },
            ],
            paneOpLabels: ['X_cᵀ X_c', '÷ (N − 1)'],
            label: 'Covariance matrix',
            ifw: {
              intuition: '<p>The 3x3 covariance matrix summarises how the centred coordinates co-vary. Its eigenvectors are the directions of maximal variance.</p>',
              formula: '$$C = \\frac{1}{N-1} X_c^{\\top} X_c$$',
              worked: workedSections(inputBlock, '$$C_{ab} = \\frac{1}{N-1}\\sum_i x_{i,a}\\, x_{i,b}$$', outputBlock),
            },
          });
          pending.delete('3');
          if (onProgress) onProgress('3');
        },
        () => {
          const { lambda, vectors } = eigSymSorted3(mem.C);
          mem.lambda = lambda;
          mem.vectors = vectors;
          const inputBlock = 'C =\n' + formatMatrix(mem.C, { digits: 4 });
          const eigvalsBlock = 'λ = (' + lambda.map(v => Number(v).toFixed(4)).join(', ') + ')';
          const eigvecsBlock = 'V =\n' + formatMatrix(vectors.map(v => Array.from(v)), { digits: 4 });
          const outputBlock = eigvalsBlock + '\n\n' + eigvecsBlock;
          steps.set('5', {
            points: mem.Xc, t, edges: null, colors: null,
            pcAxes: { v1: vectors[0], v2: vectors[1], v3: vectors[2], lambda },
            vizKind: 'spectral',
            algoId: 'pca',
            label: 'Principal directions',
            ifw: {
              intuition: '<p>Decomposing C produces orthogonal directions ordered by how much variance the data exhibits along each. The first two axes form the 2D embedding basis.</p>',
              formula: '$$C = V \\Lambda V^{\\top}$$',
              worked: workedSections(inputBlock, '$$C\\, v_k = \\lambda_k\\, v_k$$', outputBlock),
            },
          });
          pending.delete('5');
          if (onProgress) onProgress('5');
        },
        () => {
          const Xc = mem.Xc;
          const v1 = mem.vectors[0], v2 = mem.vectors[1];
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
          const inputBlock = 'v_1 = ' + formatVec3(Array.from(v1)) + '\nv_2 = ' + formatVec3(Array.from(v2));
          const outputRows = samples.map(i => [i, formatVec3(rowOf(Xc, i)),
            '(' + embed2d[i * 2].toFixed(3) + ', ' + embed2d[i * 2 + 1].toFixed(3) + ')']);
          const outputBlock = formatTable(['i', 'x_i', 'y_i'], outputRows);
          steps.set('6', {
            points: Xc, t, edges: null, colors: null, embed2d,
            vizKind: 'embedding',
            label: 'Projected to 2D',
            ifw: {
              intuition: '<p>Each centred point is projected onto the plane spanned by the top two principal directions.</p>',
              formula: '$$y_i = (v_1^{\\top} x_i,\\; v_2^{\\top} x_i)$$',
              worked: workedSections(inputBlock, '$$y_{i,k} = v_k^{\\top} x_i$$', outputBlock),
            },
          });
          pending.delete('6');
          if (onProgress) onProgress('6');
        },
      ];

      let i = 0;
      const tick = () => {
        if (cancelled || i >= tasks.length) return;
        try { tasks[i++](); } catch (e) { console.error('PCA pipeline error:', e); return; }
        if (i < tasks.length) setTimeout(tick, 0);
      };
      setTimeout(tick, 0);
    }

    function cancel() { cancelled = true; }

    return { steps, presentSubSteps, pending, start, cancel };
  },
};
```

- [ ] Step 2: Syntax check

Run: `node --check js/manifold/algorithms/pca.js`
Expected: no output.

- [ ] Step 3: Commit

```bash
git add js/manifold/algorithms/pca.js
git commit -m "manifold: PCA vizKind per step and Format-1 worked examples"
```

---

### Task 6: Update Isomap module with vizKind and Format-1 worked examples

Files:
- Modify: `js/manifold/algorithms/isomap.js`

- [ ] Step 1: Replace the contents of `js/manifold/algorithms/isomap.js` with this exact content:

```javascript
import { knnGraph, dijkstraAllPairs, doubleCenterSquared, topKSymmetricEig } from '../linalg.js';
import { formatVec3, formatMatrix, formatTable } from '../format.js';

function sampleIndices(N) {
  return [Math.floor(N * 0.2), Math.floor(N * 0.5), Math.floor(N * 0.8)];
}

function workedSections(input, formula, output) {
  return (
    '<div class="ifw-worked-section">' +
      '<div class="ifw-worked-label">Input (from previous step)</div>' +
      '<div class="ifw-worked-body">' + input + '</div>' +
    '</div>' +
    '<div class="ifw-worked-section">' +
      '<div class="ifw-worked-label">Formula</div>' +
      '<div class="ifw-worked-body math">' + formula + '</div>' +
    '</div>' +
    '<div class="ifw-worked-section">' +
      '<div class="ifw-worked-label">Output (after this step)</div>' +
      '<div class="ifw-worked-body">' + output + '</div>' +
    '</div>'
  );
}

function rowOf(X, i) {
  return [X[i * 3], X[i * 3 + 1], X[i * 3 + 2]];
}

function topKEigvals(M, N, k) {
  const Mwork = new Float64Array(M);
  const out = new Float64Array(k);
  const Mv = new Float64Array(N);
  for (let kk = 0; kk < k; kk++) {
    const v = new Float64Array(N);
    for (let i = 0; i < N; i++) v[i] = Math.sin((i + 1) * 1.3 + kk * 0.7);
    let norm = 0;
    for (let i = 0; i < N; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < N; i++) v[i] /= norm;
    for (let it = 0; it < 60; it++) {
      for (let i = 0; i < N; i++) {
        let s = 0;
        for (let j = 0; j < N; j++) s += Mwork[i * N + j] * v[j];
        Mv[i] = s;
      }
      let mag = 0;
      for (let i = 0; i < N; i++) mag += Mv[i] * Mv[i];
      mag = Math.sqrt(mag);
      if (mag < 1e-12) break;
      for (let i = 0; i < N; i++) v[i] = Mv[i] / mag;
    }
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let j = 0; j < N; j++) s += Mwork[i * N + j] * v[j];
      Mv[i] = s;
    }
    let lam = 0;
    for (let i = 0; i < N; i++) lam += v[i] * Mv[i];
    out[kk] = lam;
    for (let i = 0; i < N; i++) {
      const lv = lam * v[i];
      for (let j = 0; j < N; j++) Mwork[i * N + j] -= lv * v[j];
    }
  }
  return out;
}

export const ISOMAP = {
  id: 'isomap',
  label: 'Isomap',
  params: [{ name: 'k', type: 'int', default: 10, min: 2, max: 50 }],
  presentSubSteps: ['0', '2', '3', '4', '5', '6'],
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
    const presentSubSteps = ['0', '2', '3', '4', '5', '6'];
    const pending = new Set(['2', '3', '4', '5', '6']);
    let cancelled = false;
    const samples = sampleIndices(N);

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: null,
      vizKind: 'point_cloud',
      label: 'Raw data',
      ifw: { intuition: '<p>Isomap starts from the raw point cloud and recovers manifold geometry from local neighborhoods.</p>', formula: null, worked: null },
    });

    function start(onProgress) {
      const mem = {};
      const tasks = [
        () => {
          const { adj, edges } = knnGraph(X, k);
          mem.adj = adj;
          mem.edges = edges;
          const sampleI = samples[0];
          const neighbours = adj[sampleI].slice(0, k).map(([j, w]) => [j, w.toFixed(3)]);
          const inputBlock = 'sample point i = ' + sampleI + ', x_i = ' + formatVec3(rowOf(X, sampleI)) + '\nN = ' + N + ', k = ' + k;
          const outputBlock = 'neighbours of point ' + sampleI + ':\n' +
            formatTable(['j', '|| x_j − x_i ||'], neighbours) + '\n\ntotal undirected edges = ' + edges.length;
          steps.set('2', {
            points: X.slice(), t, edges, colors: null,
            vizKind: 'knn_graph',
            label: 'kNN graph (k = ' + k + ')',
            ifw: {
              intuition: '<p>Connecting each point to its k nearest Euclidean neighbours approximates the manifold by a graph whose edges follow the local surface.</p>',
              formula: '$$\\mathcal{N}_i = \\{ j : \\|x_j - x_i\\| \\text{ among the } k \\text{ smallest}\\}$$',
              worked: workedSections(inputBlock, '$$w_{ij} = \\| x_j - x_i \\|$$', outputBlock),
            },
          });
          pending.delete('2');
          if (onProgress) onProgress('2');
        },
        () => {
          const D = dijkstraAllPairs(mem.adj, N);
          mem.D = D;
          let connected = true;
          for (let i = 0; i < N * N; i++) if (!Number.isFinite(D[i])) { connected = false; break; }
          const i0 = samples[0], j0 = samples[2];
          const exampleD = D[i0 * N + j0];
          const inputBlock = 'kNN graph with ' + mem.edges.length + ' undirected edges (input from step 2).';
          const excerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(D[r * N + c]);
            excerpt.push(row);
          }
          const outputBlock = 'example shortest path: i = ' + i0 + ', j = ' + j0 +
            '\nD[' + i0 + '][' + j0 + '] = ' + (Number.isFinite(exampleD) ? exampleD.toFixed(3) : '∞') +
            '\n\nD (4 of N=' + N + ' rows):\n' + formatMatrix(excerpt, { digits: 3 }) +
            '\n\ngraph connected: ' + (connected ? 'yes' : 'no');
          steps.set('3', {
            points: X.slice(), t, edges: mem.edges, colors: null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'graph_thumb', label: 'kNN graph', data: { points: X.slice(), edges: mem.edges } },
              { kind: 'graph_thumb_with_path', label: 'one path traced', data: { points: X.slice(), edges: mem.edges, pathEdges: [[i0, samples[1]], [samples[1], j0]] } },
              { kind: 'heatmap', label: 'D (N x N)', data: { matrix: D, N, highlightRow: i0 } },
            ],
            paneOpLabels: ['all-pairs Dijkstra', 'fill matrix'],
            label: 'Geodesic distances',
            ifw: {
              intuition: '<p>Distances along the graph approximate true geodesic distances on the manifold.</p>',
              formula: '$$D_{ij} = \\min_{\\text{path } i \\to j \\text{ on } G} \\sum_e w_e$$',
              worked: workedSections(inputBlock, '$$D_{ij} = \\text{shortest-path}_G(i, j)$$', outputBlock),
            },
          });
          pending.delete('3');
          if (onProgress) onProgress('3');
        },
        () => {
          const B = doubleCenterSquared(mem.D, N);
          mem.B = B;
          const D = mem.D;
          const D2 = new Float64Array(N * N);
          for (let i = 0; i < N * N; i++) D2[i] = Number.isFinite(D[i]) ? D[i] * D[i] : 0;
          const rowMean = new Float64Array(N);
          const colMean = new Float64Array(N);
          let grand = 0;
          for (let i = 0; i < N; i++) {
            let r = 0;
            for (let j = 0; j < N; j++) r += D2[i * N + j];
            rowMean[i] = r / N;
            grand += r;
          }
          for (let j = 0; j < N; j++) {
            let c = 0;
            for (let i = 0; i < N; i++) c += D2[i * N + j];
            colMean[j] = c / N;
          }
          grand /= (N * N);
          const inputExcerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(D2[r * N + c]);
            inputExcerpt.push(row);
          }
          const outExcerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(B[r * N + c]);
            outExcerpt.push(row);
          }
          const inputBlock = 'D² (4 of N=' + N + ' rows):\n' + formatMatrix(inputExcerpt, { digits: 3 }) +
            '\n\nrow means (first 4): ' + Array.from(rowMean.slice(0, 4)).map(v => v.toFixed(3)).join(', ') +
            '\ngrand mean g = ' + grand.toFixed(3);
          const sample = 'example B[1][2] = −1/2 · (' + D2[N + 2].toFixed(3) + ' − ' + rowMean[1].toFixed(3) + ' − ' + colMean[2].toFixed(3) + ' + ' + grand.toFixed(3) + ') = ' + B[N + 2].toFixed(3);
          const outputBlock = sample + '\n\nB (4 of N=' + N + ' rows):\n' + formatMatrix(outExcerpt, { digits: 3 });
          steps.set('4', {
            points: X.slice(), t, edges: mem.edges, colors: null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'heatmap', label: 'D²', data: { matrix: D2, N } },
              { kind: 'heatmap', label: 'D² − μ', data: { matrix: B, N } },
              { kind: 'heatmap', label: 'B', data: { matrix: B, N } },
            ],
            paneOpLabels: ['subtract row/col means', '× (−1/2) + grand mean'],
            label: 'Double-centered Gram matrix',
            ifw: {
              intuition: '<p>Classical MDS converts pairwise squared distances into an inner-product matrix B via double centering.</p>',
              formula: '$$B = -\\tfrac{1}{2} H D^{(2)} H, \\quad H = I - \\tfrac{1}{N}\\mathbf{1}\\mathbf{1}^{\\top}$$',
              worked: workedSections(inputBlock, '$$B_{ij} = -\\tfrac{1}{2}\\bigl(D^2_{ij} - r_i - c_j + g\\bigr)$$', outputBlock),
            },
          });
          pending.delete('4');
          if (onProgress) onProgress('4');
        },
        () => {
          const { lambda, vectors } = topKSymmetricEig(mem.B, N, 2);
          mem.lambda = lambda;
          mem.vectors = vectors;
          const topEig = topKEigvals(mem.B, N, 8);
          const excerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(mem.B[r * N + c]);
            excerpt.push(row);
          }
          const inputBlock = 'B (4 of N=' + N + ' rows):\n' + formatMatrix(excerpt, { digits: 3 });
          const outputBlock = 'top eigenvalues: λ_1 = ' + lambda[0].toFixed(3) + ', λ_2 = ' + lambda[1].toFixed(3) +
            '\n\nv_1 (first 6 entries): [' + Array.from(vectors[0].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']' +
            '\nv_2 (first 6 entries): [' + Array.from(vectors[1].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']';
          steps.set('5', {
            points: X.slice(), t, edges: mem.edges, colors: null,
            vizKind: 'spectral',
            algoId: 'isomap',
            v1Values: vectors[0],
            topEigvals: topEig,
            label: 'Top-2 eigendecomposition',
            ifw: {
              intuition: '<p>The top eigenvectors of B reveal the dominant geometry of the geodesic distances.</p>',
              formula: '$$B = V \\Lambda V^{\\top}$$',
              worked: workedSections(inputBlock, '$$B\\, v_k = \\lambda_k\\, v_k$$', outputBlock),
            },
          });
          pending.delete('5');
          if (onProgress) onProgress('5');
        },
        () => {
          const lambda = mem.lambda, vectors = mem.vectors;
          const embed2d = new Float64Array(N * 2);
          const s1 = Math.sqrt(Math.max(0, lambda[0]));
          const s2 = Math.sqrt(Math.max(0, lambda[1]));
          for (let i = 0; i < N; i++) {
            embed2d[i * 2] = vectors[0][i] * s1;
            embed2d[i * 2 + 1] = vectors[1][i] * s2;
          }
          const inputBlock = 'λ_1 = ' + lambda[0].toFixed(3) + ', λ_2 = ' + lambda[1].toFixed(3) +
            '\nv_1 (first 3): [' + Array.from(vectors[0].slice(0, 3)).map(v => v.toFixed(3)).join(', ') + ']' +
            '\nv_2 (first 3): [' + Array.from(vectors[1].slice(0, 3)).map(v => v.toFixed(3)).join(', ') + ']';
          const outputRows = samples.map(i => [i,
            '(' + embed2d[i * 2].toFixed(3) + ', ' + embed2d[i * 2 + 1].toFixed(3) + ')']);
          const outputBlock = formatTable(['i', 'y_i'], outputRows);
          steps.set('6', {
            points: X.slice(), t, edges: mem.edges, colors: null, embed2d,
            vizKind: 'embedding',
            label: 'Isomap embedding',
            ifw: {
              intuition: '<p>The 2D coordinates flatten the manifold while preserving geodesic distances as well as possible.</p>',
              formula: '$$y_i = \\big(\\sqrt{\\lambda_1}\\, v_{1,i},\\; \\sqrt{\\lambda_2}\\, v_{2,i}\\big)$$',
              worked: workedSections(inputBlock, '$$y_{i,k} = \\sqrt{\\lambda_k}\\, v_{k,i}$$', outputBlock),
            },
          });
          pending.delete('6');
          if (onProgress) onProgress('6');
        },
      ];

      let i = 0;
      const tick = () => {
        if (cancelled || i >= tasks.length) return;
        try { tasks[i++](); } catch (e) { console.error('Isomap pipeline error:', e); return; }
        if (i < tasks.length) setTimeout(tick, 0);
      };
      setTimeout(tick, 0);
    }

    function cancel() { cancelled = true; }

    return { steps, presentSubSteps, pending, start, cancel };
  },
};
```

- [ ] Step 2: Syntax check

Run: `node --check js/manifold/algorithms/isomap.js`
Expected: no output.

- [ ] Step 3: Commit

```bash
git add js/manifold/algorithms/isomap.js
git commit -m "manifold: Isomap vizKind per step and Format-1 worked examples"
```

---

### Task 7: viz_centering renderer (Pattern 1)

Files:
- Create: `js/manifold/viz/viz_centering.js`
- Modify: `styles/manifold.css` (append)

- [ ] Step 1: Create `js/manifold/viz/viz_centering.js`

```javascript
function project(R, X, scale, cx, cy) {
  const N = X.length / 3;
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    const x = X[i * 3], y = X[i * 3 + 1], z = X[i * 3 + 2];
    const px = R[0][0]*x + R[0][1]*y + R[0][2]*z;
    const py = R[1][0]*x + R[1][1]*y + R[1][2]*z;
    const pz = R[2][0]*x + R[2][1]*y + R[2][2]*z;
    out[i] = { i, sx: cx + scale*px, sy: cy - scale*py, depth: pz };
  }
  return out;
}

function rainbow(t, tMin, tMax) {
  const u = (t - tMin) / Math.max(1e-9, tMax - tMin);
  const h = (1 - u) * 240;
  return `hsl(${h.toFixed(1)}, 80%, 60%)`;
}

export function mountCentering(container, state, { width = 480, height = 360 } = {}) {
  const d3 = window.d3;
  const wrap = d3.select(container).append('div').attr('class', 'viz-centering');
  const svg = wrap.append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%').style('height', '100%');

  const raw = state.rawPoints || state.points;
  const centered = state.points;
  const t = state.t || null;
  const N = raw.length / 3;

  let xmn = Infinity, xmx = -Infinity, ymn = Infinity, ymx = -Infinity, zmn = Infinity, zmx = -Infinity;
  for (let i = 0; i < N; i++) {
    const x = raw[i * 3], y = raw[i * 3 + 1], z = raw[i * 3 + 2];
    if (x < xmn) xmn = x; if (x > xmx) xmx = x;
    if (y < ymn) ymn = y; if (y > ymx) ymx = y;
    if (z < zmn) zmn = z; if (z > zmx) zmx = z;
  }
  const radius = Math.max(xmx - xmn, ymx - ymn, zmx - zmn, 1e-6) / 2;
  const R = [[0.886, 0.0, 0.464], [-0.163, 0.937, 0.311], [-0.434, -0.348, 0.831]];
  const scale = (Math.min(width, height) / 2 - 18) / radius;
  const cx = width / 2, cy = height / 2;

  const rawProj = project(R, raw, scale, cx, cy);
  const centeredProj = project(R, centered, scale, cx, cy);
  let tMin = Infinity, tMax = -Infinity;
  if (t) for (let i = 0; i < t.length; i++) { if (t[i] < tMin) tMin = t[i]; if (t[i] > tMax) tMax = t[i]; }
  const colorOf = (i) => t ? rainbow(t[i], tMin, tMax) : '#7ec8ff';

  const gGhost = svg.append('g');
  rawProj.forEach(p => {
    gGhost.append('circle').attr('cx', p.sx).attr('cy', p.sy).attr('r', 2.6)
      .attr('fill', colorOf(p.i)).attr('opacity', 0.28);
  });

  const gPoints = svg.append('g');
  const circles = centeredProj.map(p => gPoints.append('circle')
    .attr('cx', rawProj[p.i].sx).attr('cy', rawProj[p.i].sy)
    .attr('r', 2.8).attr('fill', colorOf(p.i)).attr('opacity', 0.95));

  setTimeout(() => {
    circles.forEach((c, i) => {
      c.transition().duration(1000).ease(d3.easeCubicInOut)
        .attr('cx', centeredProj[i].sx).attr('cy', centeredProj[i].sy);
    });
  }, 60);

  return {
    unmount() { wrap.remove(); }
  };
}
```

- [ ] Step 2: Append CSS for `.viz-centering`

Append to `styles/manifold.css`:

```css
.manifold .viz-centering { position: absolute; inset: 0; }
```

- [ ] Step 3: Syntax check

Run: `node --check js/manifold/viz/viz_centering.js`
Expected: no output.

- [ ] Step 4: Commit

```bash
git add js/manifold/viz/viz_centering.js styles/manifold.css
git commit -m "manifold: Pattern 1 centering animation renderer"
```

---

### Task 8: viz_knn renderer with hover spotlight (Pattern 2)

Files:
- Create: `js/manifold/viz/viz_knn.js`
- Modify: `styles/manifold.css` (append)

- [ ] Step 1: Create `js/manifold/viz/viz_knn.js`

```javascript
function project(R, X, scale, cx, cy) {
  const N = X.length / 3;
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    const x = X[i * 3], y = X[i * 3 + 1], z = X[i * 3 + 2];
    const px = R[0][0]*x + R[0][1]*y + R[0][2]*z;
    const py = R[1][0]*x + R[1][1]*y + R[1][2]*z;
    const pz = R[2][0]*x + R[2][1]*y + R[2][2]*z;
    out[i] = { i, sx: cx + scale*px, sy: cy - scale*py, depth: pz };
  }
  return out;
}

function rainbow(t, tMin, tMax) {
  const u = (t - tMin) / Math.max(1e-9, tMax - tMin);
  const h = (1 - u) * 240;
  return `hsl(${h.toFixed(1)}, 80%, 60%)`;
}

export function mountKnn(container, state, { width = 480, height = 360 } = {}) {
  const d3 = window.d3;
  const wrap = d3.select(container).append('div').attr('class', 'viz-knn');
  const svg = wrap.append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%').style('height', '100%');

  const points = state.points;
  const edges = state.edges || [];
  const t = state.t || null;
  const N = points.length / 3;

  let xmn = Infinity, xmx = -Infinity, ymn = Infinity, ymx = -Infinity, zmn = Infinity, zmx = -Infinity;
  for (let i = 0; i < N; i++) {
    const x = points[i * 3], y = points[i * 3 + 1], z = points[i * 3 + 2];
    if (x < xmn) xmn = x; if (x > xmx) xmx = x;
    if (y < ymn) ymn = y; if (y > ymx) ymx = y;
    if (z < zmn) zmn = z; if (z > zmx) zmx = z;
  }
  const radius = Math.max(xmx - xmn, ymx - ymn, zmx - zmn, 1e-6) / 2;
  const R = [[0.886, 0.0, 0.464], [-0.163, 0.937, 0.311], [-0.434, -0.348, 0.831]];
  const scale = (Math.min(width, height) / 2 - 18) / radius;
  const cx = width / 2, cy = height / 2;
  const recentered = new Float64Array(points.length);
  const ax = (xmn + xmx) / 2, ay = (ymn + ymx) / 2, az = (zmn + zmx) / 2;
  for (let i = 0; i < N; i++) {
    recentered[i * 3] = points[i * 3] - ax;
    recentered[i * 3 + 1] = points[i * 3 + 1] - ay;
    recentered[i * 3 + 2] = points[i * 3 + 2] - az;
  }
  const proj = project(R, recentered, scale, cx, cy);

  let tMin = Infinity, tMax = -Infinity;
  if (t) for (let i = 0; i < t.length; i++) { if (t[i] < tMin) tMin = t[i]; if (t[i] > tMax) tMax = t[i]; }
  const colorOf = (i) => t ? rainbow(t[i], tMin, tMax) : '#7ec8ff';

  const gEdges = svg.append('g').attr('class', 'knn-edges');
  const edgeEls = edges.map(([a, b]) => {
    const pa = proj[a], pb = proj[b];
    return gEdges.append('line')
      .attr('x1', pa.sx).attr('y1', pa.sy)
      .attr('x2', pb.sx).attr('y2', pb.sy)
      .attr('class', 'knn-edge')
      .attr('data-from', a).attr('data-to', b)
      .attr('stroke', 'rgba(255,255,255,0.18)')
      .attr('stroke-width', 0.7)
      .attr('opacity', 0);
  });

  const gPoints = svg.append('g');
  const nodeEls = proj.map(p => gPoints.append('circle')
    .attr('cx', p.sx).attr('cy', p.sy).attr('r', 2.8)
    .attr('fill', colorOf(p.i))
    .attr('class', 'knn-node')
    .attr('data-i', p.i)
    .style('cursor', 'pointer'));

  edgeEls.forEach((e, i) => {
    e.transition().delay(60 + i * 4).duration(280).attr('opacity', 1);
  });

  nodeEls.forEach((node, i) => {
    node.on('mouseenter', () => {
      edgeEls.forEach(e => {
        const from = +e.attr('data-from');
        const to = +e.attr('data-to');
        const hit = from === i || to === i;
        e.attr('stroke', hit ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.05)')
         .attr('stroke-width', hit ? 1.6 : 0.6);
      });
    });
    node.on('mouseleave', () => {
      edgeEls.forEach(e => {
        e.attr('stroke', 'rgba(255,255,255,0.18)').attr('stroke-width', 0.7);
      });
    });
  });

  return {
    unmount() { wrap.remove(); }
  };
}
```

- [ ] Step 2: Append CSS for `.viz-knn`

Append to `styles/manifold.css`:

```css
.manifold .viz-knn { position: absolute; inset: 0; }
.manifold .knn-node:hover { stroke: rgba(255,255,255,0.95); stroke-width: 1.5; }
```

- [ ] Step 3: Syntax check

Run: `node --check js/manifold/viz/viz_knn.js`
Expected: no output.

- [ ] Step 4: Commit

```bash
git add js/manifold/viz/viz_knn.js styles/manifold.css
git commit -m "manifold: Pattern 2 kNN viz with hover spotlight"
```

---

### Task 9: viz_matrix_strip renderer (Pattern 3)

Files:
- Create: `js/manifold/viz/viz_matrix_strip.js`
- Modify: `styles/manifold.css` (append)

- [ ] Step 1: Create `js/manifold/viz/viz_matrix_strip.js`

```javascript
const VIRIDIS = ['#000000', '#3a1a6e', '#5b3a8c', '#8b5fbf', '#c179d3', '#e8a37f', '#f5cf6e', '#f9eb6b'];

function colorScale(min, max, v) {
  if (max - min < 1e-12) return VIRIDIS[0];
  const u = Math.max(0, Math.min(1, (v - min) / (max - min)));
  const idx = u * (VIRIDIS.length - 1);
  const lo = Math.floor(idx), hi = Math.min(VIRIDIS.length - 1, lo + 1);
  return idx - lo < 0.5 ? VIRIDIS[lo] : VIRIDIS[hi];
}

function project3D(points, width, height) {
  const N = points.length / 3;
  let xmn = Infinity, xmx = -Infinity, ymn = Infinity, ymx = -Infinity, zmn = Infinity, zmx = -Infinity;
  for (let i = 0; i < N; i++) {
    if (points[i * 3] < xmn) xmn = points[i * 3]; if (points[i * 3] > xmx) xmx = points[i * 3];
    if (points[i * 3 + 1] < ymn) ymn = points[i * 3 + 1]; if (points[i * 3 + 1] > ymx) ymx = points[i * 3 + 1];
    if (points[i * 3 + 2] < zmn) zmn = points[i * 3 + 2]; if (points[i * 3 + 2] > zmx) zmx = points[i * 3 + 2];
  }
  const r = Math.max(xmx - xmn, ymx - ymn, zmx - zmn, 1e-6) / 2;
  const R = [[0.886, 0.0, 0.464], [-0.163, 0.937, 0.311], [-0.434, -0.348, 0.831]];
  const s = (Math.min(width, height) / 2 - 8) / r;
  const cx = width / 2, cy = height / 2;
  const ax = (xmn + xmx) / 2, ay = (ymn + ymx) / 2, az = (zmn + zmx) / 2;
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    const x = points[i * 3] - ax, y = points[i * 3 + 1] - ay, z = points[i * 3 + 2] - az;
    const px = R[0][0]*x + R[0][1]*y + R[0][2]*z;
    const py = R[1][0]*x + R[1][1]*y + R[1][2]*z;
    out[i] = { i, sx: cx + s * px, sy: cy - s * py };
  }
  return out;
}

function renderPane(svg, pane, x, y, w, h) {
  const d3 = window.d3;
  const g = svg.append('g').attr('transform', `translate(${x}, ${y})`);
  g.append('text').attr('x', w / 2).attr('y', -6).attr('text-anchor', 'middle')
   .attr('fill', 'rgba(255,255,255,0.55)').attr('font-size', '10').text(pane.label || '');

  if (pane.kind === 'cloud_thumb') {
    const proj = project3D(pane.data, w, h);
    proj.forEach(p => g.append('circle').attr('cx', p.sx).attr('cy', p.sy).attr('r', 1.6)
      .attr('fill', 'rgba(255,255,255,0.85)'));
  } else if (pane.kind === 'graph_thumb' || pane.kind === 'graph_thumb_with_path') {
    const proj = project3D(pane.data.points, w, h);
    pane.data.edges.forEach(([a, b]) => {
      g.append('line').attr('x1', proj[a].sx).attr('y1', proj[a].sy)
        .attr('x2', proj[b].sx).attr('y2', proj[b].sy)
        .attr('stroke', 'rgba(255,255,255,0.18)').attr('stroke-width', 0.6);
    });
    if (pane.kind === 'graph_thumb_with_path') {
      pane.data.pathEdges.forEach(([a, b]) => {
        if (proj[a] && proj[b]) {
          g.append('line').attr('x1', proj[a].sx).attr('y1', proj[a].sy)
            .attr('x2', proj[b].sx).attr('y2', proj[b].sy)
            .attr('stroke', 'rgba(255,255,255,0.95)').attr('stroke-width', 1.6);
        }
      });
    }
    proj.forEach(p => g.append('circle').attr('cx', p.sx).attr('cy', p.sy).attr('r', 1.4)
      .attr('fill', 'rgba(255,255,255,0.85)'));
  } else if (pane.kind === 'matrix_numbers') {
    const M = pane.data;
    const cw = w / M[0].length, ch = h / M.length;
    g.append('rect').attr('width', w).attr('height', h).attr('fill', 'rgba(255,255,255,0.08)')
     .attr('stroke', 'rgba(255,255,255,0.3)');
    M.forEach((row, r) => {
      row.forEach((v, c) => {
        g.append('text').attr('x', c * cw + cw / 2).attr('y', r * ch + ch / 2 + 3)
         .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.92)')
         .attr('font-size', '9').text(Number(v).toFixed(2));
      });
    });
  } else if (pane.kind === 'heatmap') {
    const { matrix, N, highlightRow } = pane.data;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < matrix.length; i++) {
      const v = matrix[i];
      if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    const cellSize = Math.max(1, Math.floor(Math.min(w, h) / N));
    const total = cellSize * N;
    const ox = (w - total) / 2;
    const oy = (h - total) / 2;
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const v = matrix[r * N + c];
        g.append('rect').attr('x', ox + c * cellSize).attr('y', oy + r * cellSize)
          .attr('width', cellSize).attr('height', cellSize)
          .attr('fill', colorScale(lo, hi, v));
      }
    }
    if (highlightRow !== undefined && highlightRow < N) {
      g.append('rect').attr('x', ox).attr('y', oy + highlightRow * cellSize)
        .attr('width', total).attr('height', cellSize)
        .attr('fill', 'none').attr('stroke', '#fff').attr('stroke-width', 1.5);
    }
  }
}

export function mountMatrixStrip(container, state, { width = 480, height = 280 } = {}) {
  const d3 = window.d3;
  const wrap = d3.select(container).append('div').attr('class', 'viz-matrix-strip');
  const svg = wrap.append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%').style('height', '100%');

  const panes = state.panes || [];
  const opLabels = state.paneOpLabels || ['', ''];

  const paneW = (width - 80) / 3;
  const paneH = height - 70;
  const paneY = 30;

  panes.forEach((pane, i) => {
    const x = 10 + i * (paneW + 30);
    renderPane(svg, pane, x, paneY, paneW, paneH);
    svg.append('text').attr('x', x + paneW / 2).attr('y', paneY + paneH + 18)
      .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.5)').attr('font-size', '9')
      .text(pane.label || '');
  });

  for (let k = 0; k < 2; k++) {
    const ax = 10 + (k + 1) * paneW + (k * 30) + 8;
    const ay = paneY + paneH / 2;
    svg.append('line').attr('x1', ax).attr('y1', ay).attr('x2', ax + 14).attr('y2', ay)
      .attr('stroke', 'rgba(255,255,255,0.7)').attr('stroke-width', 1.4)
      .attr('marker-end', 'url(#strip-arrow)');
    svg.append('text').attr('x', ax + 7).attr('y', ay - 4).attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.6)').attr('font-size', '9').text(opLabels[k] || '');
  }

  const defs = svg.append('defs');
  defs.append('marker').attr('id', 'strip-arrow').attr('viewBox', '0 0 10 10').attr('refX', 9)
    .attr('refY', 5).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto-start-reverse')
    .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z').attr('fill', 'rgba(255,255,255,0.7)');

  return {
    unmount() { wrap.remove(); }
  };
}
```

- [ ] Step 2: Append CSS for `.viz-matrix-strip`

Append to `styles/manifold.css`:

```css
.manifold .viz-matrix-strip { position: absolute; inset: 0; background: rgba(0,0,0,0.25); }
```

- [ ] Step 3: Syntax check

Run: `node --check js/manifold/viz/viz_matrix_strip.js`
Expected: no output.

- [ ] Step 4: Commit

```bash
git add js/manifold/viz/viz_matrix_strip.js styles/manifold.css
git commit -m "manifold: Pattern 3 matrix-strip renderer for derivation steps"
```

---

### Task 10: viz_spectral renderer (Pattern 4)

Files:
- Create: `js/manifold/viz/viz_spectral.js`
- Modify: `styles/manifold.css` (append)

- [ ] Step 1: Create `js/manifold/viz/viz_spectral.js`

```javascript
const VIRIDIS = ['#000000', '#3a1a6e', '#5b3a8c', '#8b5fbf', '#c179d3', '#e8a37f', '#f5cf6e', '#f9eb6b'];

function projectStandard(points, width, height) {
  const N = points.length / 3;
  let xmn = Infinity, xmx = -Infinity, ymn = Infinity, ymx = -Infinity, zmn = Infinity, zmx = -Infinity;
  for (let i = 0; i < N; i++) {
    const x = points[i * 3], y = points[i * 3 + 1], z = points[i * 3 + 2];
    if (x < xmn) xmn = x; if (x > xmx) xmx = x;
    if (y < ymn) ymn = y; if (y > ymx) ymx = y;
    if (z < zmn) zmn = z; if (z > zmx) zmx = z;
  }
  const r = Math.max(xmx - xmn, ymx - ymn, zmx - zmn, 1e-6) / 2;
  const R = [[0.886, 0.0, 0.464], [-0.163, 0.937, 0.311], [-0.434, -0.348, 0.831]];
  const s = (Math.min(width, height) / 2 - 24) / r;
  const cx = width / 2, cy = height / 2;
  const ax = (xmn + xmx) / 2, ay = (ymn + ymx) / 2, az = (zmn + zmx) / 2;
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    const x = points[i * 3] - ax, y = points[i * 3 + 1] - ay, z = points[i * 3 + 2] - az;
    const px = R[0][0]*x + R[0][1]*y + R[0][2]*z;
    const py = R[1][0]*x + R[1][1]*y + R[1][2]*z;
    const pz = R[2][0]*x + R[2][1]*y + R[2][2]*z;
    out[i] = { i, sx: cx + s * px, sy: cy - s * py, depth: pz };
  }
  return { proj: out, scale: s, center: [cx, cy], R };
}

function colorFromValue(v, lo, hi) {
  const u = Math.max(0, Math.min(1, (v - lo) / Math.max(1e-12, hi - lo)));
  const idx = u * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 1, Math.floor(idx));
  return VIRIDIS[i];
}

function mountPcaSpectral(svg, state, width, height) {
  const points = state.points;
  const { proj, scale, center, R } = projectStandard(points, width, height);
  proj.forEach(p => svg.append('circle').attr('cx', p.sx).attr('cy', p.sy).attr('r', 2.4)
    .attr('fill', 'rgba(255,255,255,0.28)'));

  if (state.pcAxes) {
    const { v1, v2 } = state.pcAxes;
    const ex = v1[0], ey = v1[1], ez = v1[2];
    const fx = v2[0], fy = v2[1], fz = v2[2];
    const L = scale * 0.7;
    const cornerSign = [[1, 1], [1, -1], [-1, -1], [-1, 1]];
    const pts = cornerSign.map(([a, b]) => {
      const wx = a * ex * L + b * fx * L;
      const wy = a * ey * L + b * fy * L;
      const wz = a * ez * L + b * fz * L;
      const px = R[0][0] * wx + R[0][1] * wy + R[0][2] * wz;
      const py = R[1][0] * wx + R[1][1] * wy + R[1][2] * wz;
      return [center[0] + px, center[1] - py];
    });
    svg.append('polygon')
      .attr('points', pts.map(p => p.join(',')).join(' '))
      .attr('fill', 'rgba(255,255,255,0.06)')
      .attr('stroke', 'rgba(255,255,255,0.35)').attr('stroke-width', 1);
  }
  const mini = svg.append('g').attr('transform', `translate(${width - 130}, ${height - 90})`);
  mini.append('rect').attr('width', 122).attr('height', 80).attr('fill', 'rgba(0,0,0,0.7)')
    .attr('stroke', 'rgba(255,255,255,0.25)').attr('rx', 4);
  mini.append('text').attr('x', 61).attr('y', 12).attr('text-anchor', 'middle')
    .attr('fill', 'rgba(255,255,255,0.85)').attr('font-size', '9').text('principal axes');
  const cx = 61, cy = 50;
  const ax = state.pcAxes || { v1: [1, 0, 0], v2: [0, 1, 0], v3: [0, 0, 1], lambda: [1, 1, 1] };
  const colors = ['#ff9f43', '#54a0ff', '#6bd47b'];
  const labels = ['PC1', 'PC2', 'PC3'];
  ['v1', 'v2', 'v3'].forEach((key, k) => {
    const v = ax[key];
    if (!v) return;
    const L = 24 * Math.sqrt(Math.max(0, Math.abs(ax.lambda[k] || 1)) / Math.max(1e-9, Math.abs(ax.lambda[0])));
    const px = (R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2]) * L;
    const py = (R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2]) * L;
    mini.append('line').attr('x1', cx).attr('y1', cy).attr('x2', cx + px).attr('y2', cy - py)
      .attr('stroke', colors[k]).attr('stroke-width', 1.5);
    mini.append('text').attr('x', cx + px + 4).attr('y', cy - py)
      .attr('fill', colors[k]).attr('font-size', '8').text(labels[k]);
  });
}

function mountIsomapSpectral(svg, state, width, height) {
  const points = state.points;
  const { proj } = projectStandard(points, width, height);
  const v1 = state.v1Values || new Float64Array(points.length / 3);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < v1.length; i++) { if (v1[i] < lo) lo = v1[i]; if (v1[i] > hi) hi = v1[i]; }
  proj.forEach(p => svg.append('circle').attr('cx', p.sx).attr('cy', p.sy).attr('r', 2.6)
    .attr('fill', colorFromValue(v1[p.i], lo, hi)));

  const mini = svg.append('g').attr('transform', `translate(${width - 150}, ${height - 90})`);
  mini.append('rect').attr('width', 142).attr('height', 80).attr('fill', 'rgba(0,0,0,0.7)')
    .attr('stroke', 'rgba(255,255,255,0.25)').attr('rx', 4);
  mini.append('text').attr('x', 71).attr('y', 12).attr('text-anchor', 'middle')
    .attr('fill', 'rgba(255,255,255,0.85)').attr('font-size', '9').text('eigenvalues');
  const eig = state.topEigvals || new Float64Array(8);
  let lam0 = 1;
  for (let i = 0; i < eig.length; i++) lam0 = Math.max(lam0, Math.abs(eig[i]));
  const barW = 14;
  const maxH = 56;
  for (let i = 0; i < 8; i++) {
    const v = Math.abs(eig[i] || 0);
    const h = maxH * (v / Math.max(1e-9, lam0));
    mini.append('rect').attr('x', 10 + i * barW).attr('y', 18 + (maxH - h))
      .attr('width', barW - 2).attr('height', h)
      .attr('fill', i < 2 ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.5)');
  }
}

export function mountSpectral(container, state, { width = 480, height = 360 } = {}) {
  const d3 = window.d3;
  const wrap = d3.select(container).append('div').attr('class', 'viz-spectral');
  const svg = wrap.append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%').style('height', '100%');

  if (state.algoId === 'pca') mountPcaSpectral(svg, state, width, height);
  else if (state.algoId === 'isomap') mountIsomapSpectral(svg, state, width, height);

  return {
    unmount() { wrap.remove(); }
  };
}
```

- [ ] Step 2: Append CSS for `.viz-spectral`

Append to `styles/manifold.css`:

```css
.manifold .viz-spectral { position: absolute; inset: 0; }
```

- [ ] Step 3: Syntax check

Run: `node --check js/manifold/viz/viz_spectral.js`
Expected: no output.

- [ ] Step 4: Commit

```bash
git add js/manifold/viz/viz_spectral.js styles/manifold.css
git commit -m "manifold: Pattern 4 spectral renderer with algorithm-specific mini"
```

---

### Task 11: Step-viz dispatcher

Files:
- Create: `js/manifold/step_viz.js`

- [ ] Step 1: Create `js/manifold/step_viz.js`

```javascript
import { createViz3d } from './viz3d.js';
import { createViz2d } from './viz2d.js';
import { mountCentering } from './viz/viz_centering.js';
import { mountKnn } from './viz/viz_knn.js';
import { mountMatrixStrip } from './viz/viz_matrix_strip.js';
import { mountSpectral } from './viz/viz_spectral.js';

export function createStepViz(host) {
  let activeKind = null;
  let active = null;
  const host3dThumb = host;
  let viz3d = null;
  let viz2d = null;
  let thumb = null;

  function ensure3d() {
    if (!viz3d) viz3d = createViz3d(host, {});
    return viz3d;
  }
  function ensure2d() {
    if (!viz2d) viz2d = createViz2d(host, {});
    return viz2d;
  }
  function ensureThumb() {
    if (!thumb) thumb = createViz3d(host3dThumb, { width: 140, height: 110, isThumbnail: true });
    return thumb;
  }

  function setVisible(sel, visible) {
    const el = host.querySelector(sel);
    if (el) el.style.display = visible ? '' : 'none';
  }

  function update(state) {
    if (!state) return;
    const kind = state.vizKind || 'point_cloud';

    if (active && activeKind !== kind && active.unmount) {
      active.unmount();
      active = null;
    }
    activeKind = kind;

    if (kind === 'point_cloud') {
      ensure3d();
      setVisible('.viz3d', true);
      setVisible('.viz2d', false);
      setVisible('.viz3d-thumb', false);
      viz3d.setState({ points: state.points, t: state.t, edges: null, colors: state.colors });
    } else if (kind === 'centering') {
      setVisible('.viz3d', false);
      setVisible('.viz2d', false);
      setVisible('.viz3d-thumb', false);
      active = mountCentering(host, state);
    } else if (kind === 'knn_graph') {
      setVisible('.viz3d', false);
      setVisible('.viz2d', false);
      setVisible('.viz3d-thumb', false);
      active = mountKnn(host, state);
    } else if (kind === 'matrix_strip') {
      setVisible('.viz3d', false);
      setVisible('.viz2d', false);
      setVisible('.viz3d-thumb', false);
      active = mountMatrixStrip(host, state);
    } else if (kind === 'spectral') {
      setVisible('.viz3d', false);
      setVisible('.viz2d', false);
      setVisible('.viz3d-thumb', false);
      active = mountSpectral(host, state);
    } else if (kind === 'embedding') {
      ensure2d();
      ensureThumb();
      setVisible('.viz3d', false);
      setVisible('.viz2d', true);
      setVisible('.viz3d-thumb', true);
      viz2d.setState({ embed2d: state.embed2d, colors: state.colors, t: state.t });
      thumb.setState({ points: state.points, t: state.t, edges: null, colors: null });
    }
  }

  return { update };
}
```

- [ ] Step 2: Syntax check

Run: `node --check js/manifold/step_viz.js`
Expected: no output.

- [ ] Step 3: Commit

```bash
git add js/manifold/step_viz.js
git commit -m "manifold: step-viz dispatcher selects renderer per vizKind"
```

---

### Task 12: Wire dispatcher into main.js

Files:
- Modify: `js/manifold/main.js`

- [ ] Step 1: Open `js/manifold/main.js` and replace the imports block (lines starting with `import`) and the per-side viz creation and subscriber rendering as follows.

Replace this import block:

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
```

With:

```javascript
import { DATASETS, parseCSV } from './datasets.js';
import { PCA } from './algorithms/pca.js';
import { ISOMAP } from './algorithms/isomap.js';
import { createState } from './state.js';
import { createStepViz } from './step_viz.js';
import { createStepIndicator } from './step_indicator.js';
import { createIFW } from './ifw.js';
import { createPseudocode } from './pseudocode.js';
import { compareSubSteps, unionSubSteps } from './canonical_steps.js';
```

- [ ] Step 2: Replace the viz instance creation block

Find this block in `init()`:

```javascript
  const leftViz = createViz3d(leftVizHost, {});
  const rightViz = createViz3d(rightVizHost, {});
  const leftViz2d = createViz2d(leftVizHost, {});
  const rightViz2d = createViz2d(rightVizHost, {});
  const leftThumb = createViz3d(leftVizHost, { width: 140, height: 110, isThumbnail: true });
  const rightThumb = createViz3d(rightVizHost, { width: 140, height: 110, isThumbnail: true });
  hideEl(leftVizHost, '.viz2d'); hideEl(rightVizHost, '.viz2d');
  hideEl(leftVizHost, '.viz3d-thumb'); hideEl(rightVizHost, '.viz3d-thumb');
  appendLoading(leftVizHost); appendLoading(rightVizHost);
```

Replace with:

```javascript
  const leftStepViz = createStepViz(leftVizHost);
  const rightStepViz = createStepViz(rightVizHost);
  appendLoading(leftVizHost); appendLoading(rightVizHost);
```

- [ ] Step 3: Replace the subscriber's per-side rendering

Find this block inside the `store.subscribe((s) => { ... })` callback:

```javascript
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
```

Replace with:

```javascript
    if (leftState) leftStepViz.update(leftState);
    if (rightState) rightStepViz.update(rightState);
```

- [ ] Step 4: Remove the now-unused `setStep6Mode` helper

Find and delete the entire `function setStep6Mode(host, isFinal) { ... }` definition at the bottom of the file.

- [ ] Step 5: Syntax check

Run: `node --check js/manifold/main.js`
Expected: no output.

- [ ] Step 6: Run full test suite (no regressions in node-tested logic)

Run: `node --test 'test/manifold/*.test.js' 2>&1 | tail -5`
Expected: all tests passing.

- [ ] Step 7: Commit

```bash
git add js/manifold/main.js
git commit -m "manifold: route per-step rendering through step-viz dispatcher"
```

---

### Task 13: Browser smoke verification

Files: none changed.

- [ ] Step 1: Start the static server (if not already running)

```bash
python3 -m http.server 8765 --bind 127.0.0.1 > /tmp/manifold-server.log 2>&1 &
echo "pid=$!"
sleep 1
```

- [ ] Step 2: Confirm HTTP 200 on the page and new module files

```bash
for f in pages/manifold.html js/manifold/main.js js/manifold/format.js js/manifold/step_viz.js js/manifold/viz/viz_centering.js js/manifold/viz/viz_knn.js js/manifold/viz/viz_matrix_strip.js js/manifold/viz/viz_spectral.js; do
  curl -s -o /dev/null -w "%{http_code} $f\n" "http://127.0.0.1:8765/$f"
done
```

Expected: 200 for every line.

- [ ] Step 3: Open the page in a browser and verify each surface

Open: `http://127.0.0.1:8765/pages/manifold.html`

Manually verify, in order:

1. The step indicator panel renders as two side-by-side compressed bars at the top of the Step section, with seven dots per bar. PCA has filled dots at positions 0 and 1 (current snapped) and hollow at 3, 5, 6, with gray dots at 2 and 4. Isomap has filled at 0 and 2 (current) and hollow at 3, 4, 5, 6, with a gray dot at 1.
2. Clicking the line between two dots on either bar toggles the expanded view; the bars stay visible above a vertical detail list of all 7 canonical steps per side, with N/A rows italic and faded.
3. Clicking Next ▶ advances through the union of both sides' sub-steps and the bars update.
4. Step 1 plays a one-shot slide animation in the left (PCA) panel.
5. Step 2 plays a one-shot edge wave in the right (Isomap) panel; hovering a node lights its k-neighbour edges and dims the rest; moving off restores the full graph.
6. Steps 3 and 4 show a three-pane matrix strip with arrow labels between panes.
7. Step 5 shows the spectral viz: PCA gets a faded 3D cloud with a parallelogram projection plane and a mini in the bottom-right with three colored axis arrows. Isomap gets a 3D cloud where points are colored along a viridis gradient by their top-1 eigenvector value, with an eigenvalue bar chart mini.
8. Step 6 shows the 2D scatter with a 3D mini-thumbnail (unchanged behaviour).
9. For each step, the Worked example tab shows three labelled sections: Input (from previous step), Formula (MathJax), and Output (after this step). Numbers reflect the actual dataset.

If any item fails, file the specific defect, fix it, and re-run this checklist.

- [ ] Step 4: Stop the server

```bash
ps -ef | grep -v grep | grep "http.server 8765" | awk '{print $2}' | xargs -r kill
```

---

### Task 14: Final test sweep

Files: none changed.

- [ ] Step 1: Run all unit tests

```bash
node --test 'test/manifold/*.test.js' 2>&1 | tail -5
```

Expected: tests N, pass N, fail 0 (N is the prior count plus 4 from `format.test.js`).

- [ ] Step 2: Confirm git log shows one commit per task

```bash
git log --oneline d9da50f..HEAD
```

Expected: 12 commits, one per code-producing task (Tasks 13 and 14 are verification only).

- [ ] Step 3: Confirm working tree is clean

```bash
git status --short
```

Expected: no output (clean).

---

## Self-review

Spec coverage:

- Surface 1 (step indicator panel): Tasks 1 and 2 cover the JS and CSS exactly per the spec.
- Surface 2 (per-step viz): Tasks 7 through 11 create the four pattern renderers plus the dispatcher. Task 12 wires them into main.js. Tasks 5 and 6 add the `vizKind` field per step in each algorithm.
- Surface 3 (worked example format): Task 3 adds the formatting helpers with TDD. Task 4 adds the CSS. Tasks 5 and 6 rewrite the algorithm modules to emit the sectioned HTML.

Placeholder scan: no "TBD", no "TODO", no "implement later". Every step contains the actual code or command needed.

Type consistency: `mountCentering`, `mountKnn`, `mountMatrixStrip`, `mountSpectral` all use the same `(container, state, options) -> { unmount }` shape, matching how `step_viz.js` dispatches them. `createViz3d` and `createViz2d` keep their existing signatures and are used unchanged by the dispatcher for `point_cloud` and `embedding` kinds. `pcAxes` shape is consistent between `algorithms/pca.js` and `viz/viz_spectral.js` (`{ v1, v2, v3, lambda }`). `panes` array shape is consistent between `algorithms/{pca,isomap}.js` and `viz/viz_matrix_strip.js`.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-28-manifold-step-experience.md`. Two execution options:

1. Subagent-Driven (recommended): I dispatch a fresh subagent per task with two-stage review. Uses superpowers:subagent-driven-development.
2. Inline Execution: I execute the tasks in this session with checkpoint reviews. Uses superpowers:executing-plans.

Which approach?
