# Manifold Phase 2a Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: Ship four new algorithms (MDS, LLE, Laplacian Eigenmaps, Kernel PCA) on the existing manifold learning page, along with one new linalg helper, one new viz renderer, and the UI plumbing they need. Per the design spec at `docs/superpowers/specs/2026-05-29-manifold-phase-2-design.md`, sub-phase 2a stops short of adding new datasets; it tests the new algorithms against the phase 1 datasets (Swiss roll, S-curve, CSV upload).

Architecture: Each algorithm follows the existing contract (`run(dataset, params) -> { steps, presentSubSteps, pending, start, cancel }`). Step states carry `vizKind` and any algorithm-specific fields. A new `bottomKSymmetricEig` linalg helper supports LLE and Laplacian Eigenmaps. A new `viz_weighted_knn` renderer visualises LLE's reconstruction weights. The step-viz dispatcher gains a `'weighted_knn'` branch. Main entry wiring registers the four new algorithms in both `main.js` (ALGORITHMS array) and `worker.js` (module worker), and extends the parameter renderer to support an `'enum'` parameter type for Kernel PCA's kernel selector.

Tech Stack: Vanilla ES modules, d3 v7 global, no build step. Node 22 `node --test` runner for pure-logic unit tests. Manual browser smoke for DOM and DOM-adjacent code.

User constraints (apply to every code change, comment, commit message, and markdown file produced):
- No em-dashes anywhere.
- No `<em>`, `<strong>`, `<b>`, `<i>`, `<mark>` HTML tags.
- No markdown emphasis (`*`, `**`, `_`, `__`).

---

## File structure

Files created:

- `js/manifold/algorithms/mds.js`
- `js/manifold/algorithms/lle.js`
- `js/manifold/algorithms/laplacian.js`
- `js/manifold/algorithms/kpca.js`
- `js/manifold/viz/viz_weighted_knn.js`
- `test/manifold/linalg_bottomeig.test.js`
- `test/manifold/linalg_solve.test.js`

Files modified:

- `js/manifold/linalg.js` (add `bottomKSymmetricEig` and `solveLinearSystem`).
- `js/manifold/step_viz.js` (add `'weighted_knn'` branch).
- `js/manifold/main.js` (register four new algorithms in `ALGORITHMS`, extend `renderParamHost` for `'enum'` type, default `algoParams` for each new algorithm).
- `js/manifold/worker.js` (register four new algorithms in `ALGORITHMS`).
- `styles/manifold.css` (append `.viz-weighted-knn` rule).

Phase 2a leaves the dataset code untouched. The four new algorithms run on the existing phase 1 datasets (Swiss roll, S-curve, CSV upload).

---

# Tasks

### Task 1: bottomKSymmetricEig helper (TDD)

Files:
- Modify: `js/manifold/linalg.js` (add new export)
- Create: `test/manifold/linalg_bottomeig.test.js`

- [ ] Step 1: Write the failing tests

Write `test/manifold/linalg_bottomeig.test.js`:

```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bottomKSymmetricEig } from '../../js/manifold/linalg.js';

test('bottomKSymmetricEig on a small diagonal matrix returns the smallest eigenvalues', () => {
  const N = 5;
  const M = new Float64Array(N * N);
  const expected = [0, 1, 3, 5, 7];
  for (let i = 0; i < N; i++) M[i * N + i] = expected[i];
  const { lambda, vectors } = bottomKSymmetricEig(M, N, 2);
  assert.ok(Math.abs(lambda[0] - 0) < 1e-6, 'first eigenvalue should be 0');
  assert.ok(Math.abs(lambda[1] - 1) < 1e-6, 'second eigenvalue should be 1');
  assert.equal(vectors.length, 2);
  assert.equal(vectors[0].length, N);
});

test('bottomKSymmetricEig with skipFirst skips trivial smallest', () => {
  const N = 5;
  const M = new Float64Array(N * N);
  const expected = [0, 1, 3, 5, 7];
  for (let i = 0; i < N; i++) M[i * N + i] = expected[i];
  const { lambda } = bottomKSymmetricEig(M, N, 2, { skipFirst: 1 });
  assert.ok(Math.abs(lambda[0] - 1) < 1e-6, 'first non-trivial eigenvalue should be 1');
  assert.ok(Math.abs(lambda[1] - 3) < 1e-6, 'second non-trivial eigenvalue should be 3');
});

test('bottomKSymmetricEig on a larger diagonal matrix uses shift-deflate path', () => {
  const N = 20;
  const M = new Float64Array(N * N);
  for (let i = 0; i < N; i++) M[i * N + i] = i + 1;
  const { lambda, vectors } = bottomKSymmetricEig(M, N, 3);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(lambda[i] - (i + 1)) < 1e-3, 'eigenvalue ' + i + ' approximately ' + (i + 1) + ', got ' + lambda[i]);
  }
  assert.equal(vectors[0].length, N);
});

test('bottomKSymmetricEig returns eigenvectors satisfying M v = lambda v', () => {
  const N = 6;
  const M = new Float64Array(N * N);
  for (let i = 0; i < N; i++) M[i * N + i] = (i + 1) * 0.5;
  M[0 * N + 1] = 0.1; M[1 * N + 0] = 0.1;
  M[2 * N + 4] = 0.2; M[4 * N + 2] = 0.2;
  const { lambda, vectors } = bottomKSymmetricEig(M, N, 2);
  for (let k = 0; k < 2; k++) {
    const v = vectors[k];
    const Mv = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let j = 0; j < N; j++) s += M[i * N + j] * v[j];
      Mv[i] = s;
    }
    for (let i = 0; i < N; i++) {
      assert.ok(Math.abs(Mv[i] - lambda[k] * v[i]) < 1e-4,
        'M v[' + i + '] should equal lambda v[' + i + '] for k=' + k);
    }
  }
});
```

- [ ] Step 2: Run tests to confirm failure

Run: `node --test test/manifold/linalg_bottomeig.test.js`
Expected: import error (function not exported).

- [ ] Step 3: Implement `bottomKSymmetricEig` in `js/manifold/linalg.js`

Open `js/manifold/linalg.js` and append after the existing `topKSymmetricEig` export:

```javascript
export function bottomKSymmetricEig(M, N, k, { skipFirst = 0 } = {}) {
  if (N <= 16) {
    const matrix = [];
    for (let i = 0; i < N; i++) {
      const row = new Array(N);
      for (let j = 0; j < N; j++) row[j] = M[i * N + j];
      matrix.push(row);
    }
    const eig = jacobiEigSym(matrix);
    const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => eig.lambda[a] - eig.lambda[b]);
    const sliced = order.slice(skipFirst, skipFirst + k);
    const outLambda = new Float64Array(k);
    const outVecs = [];
    for (let i = 0; i < k; i++) {
      outLambda[i] = eig.lambda[sliced[i]];
      const v = new Float64Array(N);
      for (let r = 0; r < N; r++) v[r] = eig.vectors[sliced[i]][r];
      outVecs.push(v);
    }
    return { lambda: outLambda, vectors: outVecs };
  }
  const probe = new Float64Array(N);
  for (let i = 0; i < N; i++) probe[i] = Math.sin(i * 1.3 + 0.7);
  let nrm = 0;
  for (let i = 0; i < N; i++) nrm += probe[i] * probe[i];
  nrm = Math.sqrt(nrm);
  if (nrm > 0) for (let i = 0; i < N; i++) probe[i] /= nrm;
  const Mv = new Float64Array(N);
  for (let iter = 0; iter < 40; iter++) {
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let j = 0; j < N; j++) s += M[i * N + j] * probe[j];
      Mv[i] = s;
    }
    let mag = 0;
    for (let i = 0; i < N; i++) mag += Mv[i] * Mv[i];
    mag = Math.sqrt(mag);
    if (mag < 1e-12) break;
    for (let i = 0; i < N; i++) probe[i] = Mv[i] / mag;
  }
  let muEstimate = 0;
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let j = 0; j < N; j++) s += M[i * N + j] * probe[j];
    muEstimate += probe[i] * s;
  }
  const mu = Math.abs(muEstimate) * 1.1 + 1.0;
  const S = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      S[i * N + j] = (i === j ? mu : 0) - M[i * N + j];
    }
  }
  const { lambda: shifted, vectors } = topKSymmetricEig(S, N, k + skipFirst);
  const outLambda = new Float64Array(k);
  const outVecs = [];
  for (let i = 0; i < k; i++) {
    outLambda[i] = mu - shifted[skipFirst + i];
    outVecs.push(vectors[skipFirst + i]);
  }
  return { lambda: outLambda, vectors: outVecs };
}
```

- [ ] Step 4: Run tests to confirm pass

Run: `node --test test/manifold/linalg_bottomeig.test.js`
Expected: 4 passing.

- [ ] Step 5: Run the full manifold test suite

Run: `node --test 'test/manifold/*.test.js' 2>&1 | tail -5`
Expected: tests 31, pass 31, fail 0.

- [ ] Step 6: Commit

```bash
git add js/manifold/linalg.js test/manifold/linalg_bottomeig.test.js
git commit -m "manifold: bottomKSymmetricEig helper for LLE and Laplacian Eigenmaps"
```

---

### Task 2: solveLinearSystem helper (TDD)

Files:
- Modify: `js/manifold/linalg.js` (add new export)
- Create: `test/manifold/linalg_solve.test.js`

LLE needs to solve a small kxk linear system per point. A 10x10 system is easy to solve by partial-pivot Gaussian elimination.

- [ ] Step 1: Write the failing tests

Write `test/manifold/linalg_solve.test.js`:

```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { solveLinearSystem } from '../../js/manifold/linalg.js';

test('solveLinearSystem on a 3x3 identity returns the right-hand side', () => {
  const A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const b = [2, 3, 4];
  const x = solveLinearSystem(A, b);
  assert.deepEqual(x, [2, 3, 4]);
});

test('solveLinearSystem on a 3x3 with pivots returns the unique solution', () => {
  const A = [[2, 1, 1], [4, -6, 0], [-2, 7, 2]];
  const b = [5, -2, 9];
  const x = solveLinearSystem(A, b);
  assert.ok(Math.abs(x[0] - 1) < 1e-9);
  assert.ok(Math.abs(x[1] - 1) < 1e-9);
  assert.ok(Math.abs(x[2] - 2) < 1e-9);
});

test('solveLinearSystem returns null on a singular matrix', () => {
  const A = [[1, 2], [2, 4]];
  const b = [3, 6];
  const x = solveLinearSystem(A, b);
  assert.equal(x, null);
});
```

- [ ] Step 2: Run tests to confirm failure

Run: `node --test test/manifold/linalg_solve.test.js`
Expected: import error.

- [ ] Step 3: Implement `solveLinearSystem` in `js/manifold/linalg.js`

Append to `js/manifold/linalg.js`:

```javascript
export function solveLinearSystem(A, b) {
  const n = A.length;
  const M = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(n + 1);
    for (let j = 0; j < n; j++) row[j] = A[i][j];
    row[n] = b[i];
    M.push(row);
  }
  for (let p = 0; p < n; p++) {
    let max = Math.abs(M[p][p]);
    let idx = p;
    for (let i = p + 1; i < n; i++) {
      if (Math.abs(M[i][p]) > max) { max = Math.abs(M[i][p]); idx = i; }
    }
    if (idx !== p) { const tmp = M[p]; M[p] = M[idx]; M[idx] = tmp; }
    if (Math.abs(M[p][p]) < 1e-12) return null;
    for (let i = p + 1; i < n; i++) {
      const f = M[i][p] / M[p][p];
      for (let j = p; j <= n; j++) M[i][j] -= f * M[p][j];
    }
  }
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}
```

- [ ] Step 4: Run tests to confirm pass

Run: `node --test test/manifold/linalg_solve.test.js`
Expected: 3 passing.

- [ ] Step 5: Commit

```bash
git add js/manifold/linalg.js test/manifold/linalg_solve.test.js
git commit -m "manifold: solveLinearSystem helper (Gauss elimination with partial pivot)"
```

---

### Task 3: viz_weighted_knn renderer

Files:
- Create: `js/manifold/viz/viz_weighted_knn.js`
- Modify: `styles/manifold.css` (append rule)
- Modify: `js/manifold/step_viz.js` (add dispatch branch)

- [ ] Step 1: Create `js/manifold/viz/viz_weighted_knn.js`

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

function project(R, X, scale, cx, cy) {
  const N = X.length / 3;
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    const x = X[i * 3], y = X[i * 3 + 1], z = X[i * 3 + 2];
    const px = R[0][0]*x + R[0][1]*y + R[0][2]*z;
    const py = R[1][0]*x + R[1][1]*y + R[1][2]*z;
    out[i] = { i, sx: cx + scale*px, sy: cy - scale*py };
  }
  return out;
}

function rainbow(t, tMin, tMax) {
  const u = (t - tMin) / Math.max(1e-9, tMax - tMin);
  const h = (1 - u) * 240;
  return `hsl(${h.toFixed(1)}, 80%, 60%)`;
}

function strokeWidthForWeight(w) {
  const absW = Math.abs(w);
  return 0.6 + Math.min(2.9, absW * 4);
}

export function mountWeightedKnn(container, state, { width = 480, height = 360 } = {}) {
  const d3 = window.d3;
  const wrap = d3.select(container).append('div').attr('class', 'viz-weighted-knn');
  const svg = wrap.append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%').style('height', '100%')
    .style('touch-action', 'none').style('cursor', 'grab');

  const points = state.points;
  const edges = state.edges || [];
  const W = state.W;
  const N = points.length / 3;
  const t = state.t || null;
  let selectedPoint = (state.selectedPoint !== undefined ? state.selectedPoint : Math.floor(N * 0.2));

  let xmn = Infinity, xmx = -Infinity, ymn = Infinity, ymx = -Infinity, zmn = Infinity, zmx = -Infinity;
  for (let i = 0; i < N; i++) {
    const x = points[i * 3], y = points[i * 3 + 1], z = points[i * 3 + 2];
    if (x < xmn) xmn = x; if (x > xmx) xmx = x;
    if (y < ymn) ymn = y; if (y > ymx) ymx = y;
    if (z < zmn) zmn = z; if (z > zmx) zmx = z;
  }
  const radius = Math.max(xmx - xmn, ymx - ymn, zmx - zmn, 1e-6) / 2;
  let R = [[0.886, 0.0, 0.464], [-0.163, 0.937, 0.311], [-0.434, -0.348, 0.831]];
  const scale = (Math.min(width, height) / 2 - 18) / radius;
  const cx = width / 2, cy = height / 2;
  const recentered = new Float64Array(points.length);
  const ax = (xmn + xmx) / 2, ay = (ymn + ymx) / 2, az = (zmn + zmx) / 2;
  for (let i = 0; i < N; i++) {
    recentered[i * 3] = points[i * 3] - ax;
    recentered[i * 3 + 1] = points[i * 3 + 1] - ay;
    recentered[i * 3 + 2] = points[i * 3 + 2] - az;
  }

  let tMin = Infinity, tMax = -Infinity;
  if (t) for (let i = 0; i < t.length; i++) { if (t[i] < tMin) tMin = t[i]; if (t[i] > tMax) tMax = t[i]; }
  const colorOf = (i) => t ? rainbow(t[i], tMin, tMax) : '#7ec8ff';

  const gEdges = svg.append('g');
  const gPoints = svg.append('g');

  let proj = project(R, recentered, scale, cx, cy);
  const edgeEls = edges.map(([a, b]) =>
    gEdges.append('line').attr('data-from', a).attr('data-to', b));
  const nodeEls = proj.map(p =>
    gPoints.append('circle').attr('data-i', p.i).style('cursor', 'pointer'));

  function redraw() {
    proj = project(R, recentered, scale, cx, cy);
    edgeEls.forEach((e, idx) => {
      const a = edges[idx][0];
      const b = edges[idx][1];
      const isSelectedEdge = (a === selectedPoint || b === selectedPoint);
      let wij = 0;
      if (isSelectedEdge) {
        const other = a === selectedPoint ? b : a;
        wij = W ? W[selectedPoint * N + other] : 0;
      }
      e.attr('x1', proj[a].sx).attr('y1', proj[a].sy)
       .attr('x2', proj[b].sx).attr('y2', proj[b].sy)
       .attr('stroke', isSelectedEdge ? 'rgba(255,255,255,0.92)' : 'rgba(255,255,255,0.10)')
       .attr('stroke-width', isSelectedEdge ? strokeWidthForWeight(wij) : 0.5);
    });
    nodeEls.forEach((node, i) => {
      const isSelected = (i === selectedPoint);
      node.attr('cx', proj[i].sx).attr('cy', proj[i].sy)
        .attr('r', isSelected ? 4.5 : 2.5)
        .attr('fill', isSelected ? '#ff9f43' : colorOf(i));
    });
  }
  redraw();

  nodeEls.forEach((node, i) => {
    node.on('mouseenter', () => {
      if (i === selectedPoint) return;
      selectedPoint = i;
      redraw();
    });
  });

  let dragging = false, lastX = 0, lastY = 0;
  svg.on('pointerdown', (event) => {
    if (event.target && event.target.tagName === 'circle') return;
    dragging = true; lastX = event.clientX; lastY = event.clientY;
    svg.style('cursor', 'grabbing');
    try { svg.node().setPointerCapture(event.pointerId); } catch (e) {}
  });
  svg.on('pointermove', (event) => {
    if (!dragging) return;
    const dx = (event.clientX - lastX) * 0.008;
    const dy = (event.clientY - lastY) * 0.008;
    lastX = event.clientX; lastY = event.clientY;
    R = matmul(matmul(rotX(dy), rotY(dx)), R);
    redraw();
  });
  function endDrag(event) {
    dragging = false; svg.style('cursor', 'grab');
    try { svg.node().releasePointerCapture(event.pointerId); } catch (e) {}
  }
  svg.on('pointerup', endDrag);
  svg.on('pointercancel', endDrag);
  svg.on('pointerleave', endDrag);

  return {
    unmount() { wrap.remove(); }
  };
}
```

- [ ] Step 2: Append CSS rule

Append to `styles/manifold.css`:

```css
.manifold .viz-weighted-knn { position: absolute; inset: 0; }
```

- [ ] Step 3: Add the dispatch branch in `js/manifold/step_viz.js`

Open `js/manifold/step_viz.js`. Add this import next to the other viz imports:

```javascript
import { mountWeightedKnn } from './viz/viz_weighted_knn.js';
```

Add this branch in `update(state)` after the existing `knn_graph` branch and before `matrix_strip`:

```javascript
    } else if (kind === 'weighted_knn') {
      setVisible('.viz3d', false);
      setVisible('.viz2d', false);
      setVisible('.viz3d-thumb', false);
      active = mountWeightedKnn(host, state);
```

- [ ] Step 4: Syntax check

Run: `node --check js/manifold/viz/viz_weighted_knn.js && node --check js/manifold/step_viz.js`
Expected: no output.

- [ ] Step 5: Commit

```bash
git add js/manifold/viz/viz_weighted_knn.js styles/manifold.css js/manifold/step_viz.js
git commit -m "manifold: viz_weighted_knn renderer for LLE reconstruction weights"
```

---

### Task 4: Enum param type support in main.js

Files:
- Modify: `js/manifold/main.js` (`renderParamHost` function)

- [ ] Step 1: Locate the `renderParamHost` function

Find this block inside `init()`:

```javascript
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
```

- [ ] Step 2: Replace it with the enum-aware version

```javascript
  function renderParamHost(host, algo, current, onChange) {
    host.innerHTML = '';
    for (const p of algo.params) {
      const wrap = document.createElement('label');
      wrap.className = 'mf-param';
      wrap.textContent = `${p.name} = `;
      if (p.type === 'enum') {
        const sel = document.createElement('select');
        for (const opt of p.options) {
          const o = document.createElement('option');
          o.value = opt; o.textContent = opt;
          if ((current[p.name] || p.default) === opt) o.selected = true;
          sel.appendChild(o);
        }
        sel.addEventListener('change', () => {
          onChange({ ...current, [p.name]: sel.value });
        });
        wrap.appendChild(sel);
        host.appendChild(wrap);
        continue;
      }
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
```

- [ ] Step 3: Syntax check

Run: `node --check js/manifold/main.js`
Expected: no output.

- [ ] Step 4: Commit

```bash
git add js/manifold/main.js
git commit -m "manifold: enum param type in renderParamHost for kernel selector"
```

---

### Task 5: MDS algorithm module

Files:
- Create: `js/manifold/algorithms/mds.js`

- [ ] Step 1: Create `js/manifold/algorithms/mds.js`

```javascript
import { doubleCenterSquared, topKSymmetricEig, squaredDist3 } from '../linalg.js';
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

export const MDS = {
  id: 'mds',
  label: 'MDS',
  params: [],
  presentSubSteps: ['0', '3', '4', '5', '6'],
  pseudocode: [
    { id: 'mds-distances', title: '1. Compute pairwise distances', steps: ['3'],
      lines: ['D_{ij} = || x_i - x_j ||'] },
    { id: 'mds-dc', title: '2. Double-center the squared distance matrix', steps: ['4'],
      lines: ['B = -1/2 H D^2 H, with H = I - (1/N) 1 1^T'] },
    { id: 'mds-eig', title: '3. Eigendecompose B', steps: ['5'],
      lines: ['B = V Lambda V^T (take top-2 eigvals/vecs)'] },
    { id: 'mds-embed', title: '4. Form 2D embedding', steps: ['6'],
      lines: ['Y = V[:, 0:2] * diag(sqrt(lambda_1), sqrt(lambda_2))'] },
  ],
  run(dataset, _params) {
    const X = dataset.X;
    const t = dataset.t;
    const N = X.length / 3;
    const steps = new Map();
    const presentSubSteps = ['0', '3', '4', '5', '6'];
    const pending = new Set(['3', '4', '5', '6']);
    let cancelled = false;
    const samples = sampleIndices(N);

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: dataset.colors || null,
      vizKind: 'point_cloud',
      label: 'Raw data',
      ifw: {
        intuition: '<p>MDS preserves pairwise Euclidean distances. It starts from the raw 3D cloud and computes the full N by N distance matrix in the next step.</p>',
        formula: null,
        worked: null,
      },
    });

    function start(onProgress) {
      const mem = {};
      const tasks = [
        () => {
          const D = new Float64Array(N * N);
          for (let i = 0; i < N; i++) {
            for (let j = i + 1; j < N; j++) {
              const d = Math.sqrt(squaredDist3(X, i, j));
              D[i * N + j] = d;
              D[j * N + i] = d;
            }
          }
          mem.D = D;
          const i0 = samples[0], j0 = samples[2];
          const exampleD = D[i0 * N + j0];
          const inputBlock = 'sample points (3 of N=' + N + '):\n' +
            samples.map(i => 'x_' + i + ' = ' + formatVec3(rowOf(X, i))).join('\n');
          const excerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(D[r * N + c]);
            excerpt.push(row);
          }
          const outputBlock = 'example: D[' + i0 + '][' + j0 + '] = || x_' + i0 + ' - x_' + j0 + ' || = ' + exampleD.toFixed(3) +
            '\n\nD (4 of N=' + N + ' rows):\n' + formatMatrix(excerpt, { digits: 3 });
          steps.set('3', {
            points: X.slice(), t, edges: null, colors: dataset.colors || null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'cloud_thumb', label: 'X', data: X.slice() },
              { kind: 'heatmap', label: 'D (N x N)', data: { matrix: D, N } },
            ],
            paneOpLabels: ['D_{ij} = ||x_i - x_j||'],
            label: 'Pairwise distances',
            ifw: {
              intuition: '<p>Every pair of points contributes one distance to the N by N matrix D. The distance matrix encodes all the geometric information MDS uses.</p>',
              formula: '$$D_{ij} = \\| x_i - x_j \\|$$',
              worked: workedSections(inputBlock, '$$D_{ij} = \\| x_i - x_j \\|$$', outputBlock),
            },
          });
          pending.delete('3');
          if (onProgress) onProgress('3');
        },
        () => {
          const D = mem.D;
          const B = doubleCenterSquared(D, N);
          mem.B = B;
          const D2 = new Float64Array(N * N);
          for (let i = 0; i < N * N; i++) D2[i] = D[i] * D[i];
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
          const D2c = new Float64Array(N * N);
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
              D2c[i * N + j] = D2[i * N + j] - rowMean[i] - colMean[j];
            }
          }
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
          const inputBlock = 'D^2 (4 of N=' + N + ' rows):\n' + formatMatrix(inputExcerpt, { digits: 3 }) +
            '\n\nrow means (first 4): ' + Array.from(rowMean.slice(0, 4)).map(v => v.toFixed(3)).join(', ') +
            '\ngrand mean g = ' + grand.toFixed(3);
          const outputBlock = 'B (4 of N=' + N + ' rows):\n' + formatMatrix(outExcerpt, { digits: 3 });
          steps.set('4', {
            points: X.slice(), t, edges: null, colors: dataset.colors || null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'heatmap', label: 'D^2', data: { matrix: D2, N } },
              { kind: 'heatmap', label: 'D^2 - mu', data: { matrix: D2c, N } },
              { kind: 'heatmap', label: 'B', data: { matrix: B, N } },
            ],
            paneOpLabels: ['subtract row/col means', 'x (-1/2) + grand mean'],
            label: 'Double-centered Gram matrix',
            ifw: {
              intuition: '<p>Double-centering converts the pairwise squared distance matrix D squared into the Gram matrix B, which contains inner products relative to the centre of the cloud. The next step decomposes B to obtain the embedding.</p>',
              formula: '$$B = -\\tfrac{1}{2} H D^{(2)} H, \\quad H = I - \\tfrac{1}{N}\\mathbf{1}\\mathbf{1}^{\\top}$$',
              worked: workedSections(inputBlock, '$$B_{ij} = -\\tfrac{1}{2}\\bigl(D^2_{ij} - r_i - c_j + g\\bigr)$$', outputBlock),
            },
          });
          pending.delete('4');
          if (onProgress) onProgress('4');
        },
        () => {
          const { lambda, vectors } = topKSymmetricEig(mem.B, N, 8);
          mem.lambda = lambda;
          mem.vectors = vectors;
          const excerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(mem.B[r * N + c]);
            excerpt.push(row);
          }
          const inputBlock = 'B (4 of N=' + N + ' rows):\n' + formatMatrix(excerpt, { digits: 3 });
          const outputBlock = 'top eigenvalues: ' + Array.from(lambda).map((v, i) => 'lambda_' + (i + 1) + ' = ' + v.toFixed(3)).join(', ') +
            '\n\nv_1 (first 6 entries): [' + Array.from(vectors[0].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']' +
            '\nv_2 (first 6 entries): [' + Array.from(vectors[1].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']';
          steps.set('5', {
            points: X.slice(), t, edges: null, colors: dataset.colors || null,
            vizKind: 'spectral',
            algoId: 'mds',
            v1Values: vectors[0],
            topEigvals: lambda,
            topEigvecs: vectors,
            label: 'Top-2 eigendecomposition',
            ifw: {
              intuition: '<p>The top eigenvectors of B give a Euclidean embedding that best preserves the original pairwise distances in the least-squares sense.</p>',
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
          const inputBlock = 'lambda_1 = ' + lambda[0].toFixed(3) + ', lambda_2 = ' + lambda[1].toFixed(3) +
            '\nv_1 (first 3): [' + Array.from(vectors[0].slice(0, 3)).map(v => v.toFixed(3)).join(', ') + ']' +
            '\nv_2 (first 3): [' + Array.from(vectors[1].slice(0, 3)).map(v => v.toFixed(3)).join(', ') + ']';
          const outputRows = samples.map(i => [i,
            '(' + embed2d[i * 2].toFixed(3) + ', ' + embed2d[i * 2 + 1].toFixed(3) + ')']);
          const outputBlock = formatTable(['i', 'y_i'], outputRows);
          steps.set('6', {
            points: X.slice(), t, edges: null, colors: dataset.colors || null, embed2d,
            vizKind: 'embedding',
            label: 'MDS embedding',
            ifw: {
              intuition: '<p>The 2D coordinates approximately preserve the original Euclidean distances. The result is identical to PCA when the data is centred, but MDS reaches it through the distance matrix instead of the covariance matrix.</p>',
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
        try { tasks[i++](); } catch (e) { console.error('MDS pipeline error:', e); return; }
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

Run: `node --check js/manifold/algorithms/mds.js`
Expected: no output.

- [ ] Step 3: Commit

```bash
git add js/manifold/algorithms/mds.js
git commit -m "manifold: MDS algorithm module"
```

---

### Task 6: Laplacian Eigenmaps module

Files:
- Create: `js/manifold/algorithms/laplacian.js`

- [ ] Step 1: Create `js/manifold/algorithms/laplacian.js`

```javascript
import { knnGraph, bottomKSymmetricEig } from '../linalg.js';
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

export const LAPLACIAN = {
  id: 'laplacian',
  label: 'Laplacian Eigenmaps',
  params: [
    { name: 'k', type: 'int', default: 10, min: 2, max: 50 },
    { name: 'sigma', type: 'float', default: 1.0, min: 0.1, max: 10 },
  ],
  presentSubSteps: ['0', '2', '3', '4', '5', '6'],
  pseudocode: [
    { id: 'lap-knn', title: '1. Build kNN graph', steps: ['2'],
      lines: ['neighbours_i = k nearest by Euclidean distance'] },
    { id: 'lap-W', title: '2. Heat-kernel affinity W', steps: ['3'],
      lines: ['W_{ij} = exp(-||x_i - x_j||^2 / (2 sigma^2)) for kNN edges, else 0'] },
    { id: 'lap-L', title: '3. Graph Laplacian L = D - W', steps: ['4'],
      lines: ['D_{ii} = sum_j W_{ij}', 'L = D - W'] },
    { id: 'lap-eig', title: '4. Smallest non-trivial eigenvectors', steps: ['5'],
      lines: ['L v_k = lambda_k v_k (skip lambda_0 = 0)'] },
    { id: 'lap-embed', title: '5. Form 2D embedding', steps: ['6'],
      lines: ['Y = [v_1, v_2]'] },
  ],
  run(dataset, params) {
    const X = dataset.X;
    const t = dataset.t;
    const N = X.length / 3;
    const k = Math.max(2, Math.min(params.k || 10, N - 1));
    const sigma = params.sigma || 1.0;
    const steps = new Map();
    const presentSubSteps = ['0', '2', '3', '4', '5', '6'];
    const pending = new Set(['2', '3', '4', '5', '6']);
    let cancelled = false;
    const samples = sampleIndices(N);

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: dataset.colors || null,
      vizKind: 'point_cloud',
      label: 'Raw data',
      ifw: { intuition: '<p>Laplacian Eigenmaps preserves local proximity. It uses the kNN graph plus a heat kernel to weight nearby points more strongly.</p>', formula: null, worked: null },
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
            formatTable(['j', '||x_j - x_i||'], neighbours) + '\n\ntotal undirected edges = ' + edges.length;
          steps.set('2', {
            points: X.slice(), t, edges, colors: dataset.colors || null,
            vizKind: 'knn_graph',
            label: 'kNN graph (k = ' + k + ')',
            ifw: {
              intuition: '<p>The kNN graph captures local neighbourhood structure that the heat kernel will weight in the next step.</p>',
              formula: '$$\\mathcal{N}_i = \\{ j : \\|x_j - x_i\\| \\text{ among the } k \\text{ smallest}\\}$$',
              worked: workedSections(inputBlock, '$$w^{\\text{edge}}_{ij} = \\| x_j - x_i \\|$$', outputBlock),
            },
          });
          pending.delete('2');
          if (onProgress) onProgress('2');
        },
        () => {
          const adj = mem.adj;
          const W = new Float64Array(N * N);
          const sig2 = 2 * sigma * sigma;
          for (let i = 0; i < N; i++) {
            for (const [j, dist] of adj[i]) {
              const w = Math.exp(-dist * dist / sig2);
              W[i * N + j] = w;
              W[j * N + i] = w;
            }
          }
          mem.W = W;
          const sampleI = samples[0];
          const wRow = adj[sampleI].slice(0, Math.min(5, k)).map(([j, dist]) => [j, dist.toFixed(3), Math.exp(-dist * dist / sig2).toFixed(4)]);
          const inputBlock = 'sample point i = ' + sampleI + ', sigma = ' + sigma + '\nkNN distances visible above.';
          const outputBlock = 'W_ij for first ' + Math.min(5, k) + ' neighbours of ' + sampleI + ':\n' +
            formatTable(['j', '||x_j - x_i||', 'W_ij'], wRow);
          steps.set('3', {
            points: X.slice(), t, edges: mem.edges, colors: dataset.colors || null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'graph_thumb', label: 'kNN graph', data: { points: X.slice(), edges: mem.edges } },
              { kind: 'heatmap', label: 'W', data: { matrix: W, N } },
            ],
            paneOpLabels: ['W_{ij} = exp(-||x_i - x_j||^2 / (2 sigma^2))'],
            label: 'Heat-kernel affinity W',
            ifw: {
              intuition: '<p>Closer points get larger affinity weights via the Gaussian heat kernel; non-kNN edges are exactly zero so W stays sparse.</p>',
              formula: '$$W_{ij} = \\exp\\!\\left(-\\frac{\\| x_i - x_j \\|^2}{2 \\sigma^2}\\right) \\text{ for kNN edges, else } 0$$',
              worked: workedSections(inputBlock, '$$W_{ij} = \\exp\\!\\left(-\\frac{\\| x_i - x_j \\|^2}{2 \\sigma^2}\\right)$$', outputBlock),
            },
          });
          pending.delete('3');
          if (onProgress) onProgress('3');
        },
        () => {
          const W = mem.W;
          const D = new Float64Array(N);
          for (let i = 0; i < N; i++) {
            let s = 0;
            for (let j = 0; j < N; j++) s += W[i * N + j];
            D[i] = s;
          }
          const Dmat = new Float64Array(N * N);
          for (let i = 0; i < N; i++) Dmat[i * N + i] = D[i];
          const L = new Float64Array(N * N);
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
              L[i * N + j] = (i === j ? D[i] : 0) - W[i * N + j];
            }
          }
          mem.L = L;
          mem.D = D;
          const inputBlock = 'W (4 of N=' + N + ' rows):\n' + (function () {
            const ex = [];
            for (let r = 0; r < 4 && r < N; r++) {
              const row = [];
              for (let c = 0; c < 4 && c < N; c++) row.push(W[r * N + c]);
              ex.push(row);
            }
            return formatMatrix(ex, { digits: 3 });
          })();
          const outputBlock = 'D_ii (first 4): ' + Array.from(D.slice(0, 4)).map(v => v.toFixed(3)).join(', ') +
            '\n\nL (4 of N=' + N + ' rows):\n' + (function () {
            const ex = [];
            for (let r = 0; r < 4 && r < N; r++) {
              const row = [];
              for (let c = 0; c < 4 && c < N; c++) row.push(L[r * N + c]);
              ex.push(row);
            }
            return formatMatrix(ex, { digits: 3 });
          })();
          steps.set('4', {
            points: X.slice(), t, edges: mem.edges, colors: dataset.colors || null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'heatmap', label: 'W', data: { matrix: W, N } },
              { kind: 'heatmap', label: 'D', data: { matrix: Dmat, N } },
              { kind: 'heatmap', label: 'L = D - W', data: { matrix: L, N } },
            ],
            paneOpLabels: ['row sums = D', 'D - W = L'],
            label: 'Graph Laplacian',
            ifw: {
              intuition: '<p>L combines the degree of each node with its outgoing affinities. The smallest non-trivial eigenvectors of L are smooth on the graph and give the embedding coordinates.</p>',
              formula: '$$L = D - W,\\quad D_{ii} = \\sum_j W_{ij}$$',
              worked: workedSections(inputBlock, '$$L_{ij} = D_{ii} \\delta_{ij} - W_{ij}$$', outputBlock),
            },
          });
          pending.delete('4');
          if (onProgress) onProgress('4');
        },
        () => {
          const { lambda, vectors } = bottomKSymmetricEig(mem.L, N, 8, { skipFirst: 1 });
          mem.lambda = lambda;
          mem.vectors = vectors;
          const inputBlock = 'L (4 of N=' + N + ' rows): see step 4 output.';
          const outputBlock = 'bottom non-trivial eigenvalues: ' + Array.from(lambda).map((v, i) => 'lambda_' + (i + 1) + ' = ' + v.toFixed(3)).join(', ') +
            '\n\nv_1 (first 6 entries): [' + Array.from(vectors[0].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']' +
            '\nv_2 (first 6 entries): [' + Array.from(vectors[1].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']';
          steps.set('5', {
            points: X.slice(), t, edges: mem.edges, colors: dataset.colors || null,
            vizKind: 'spectral',
            algoId: 'laplacian',
            v1Values: vectors[0],
            topEigvals: lambda,
            topEigvecs: vectors,
            label: 'Smallest non-trivial eigenvectors',
            ifw: {
              intuition: '<p>Skipping the trivial zero eigenvalue, the next smallest eigenvectors are smooth functions on the graph. Each one becomes one coordinate of the embedding.</p>',
              formula: '$$L\\, v_k = \\lambda_k\\, v_k$$',
              worked: workedSections(inputBlock, '$$L\\, v_k = \\lambda_k\\, v_k,\\quad k = 1, 2$$', outputBlock),
            },
          });
          pending.delete('5');
          if (onProgress) onProgress('5');
        },
        () => {
          const vectors = mem.vectors;
          const embed2d = new Float64Array(N * 2);
          for (let i = 0; i < N; i++) {
            embed2d[i * 2] = vectors[0][i];
            embed2d[i * 2 + 1] = vectors[1][i];
          }
          const outputRows = samples.map(i => [i,
            '(' + embed2d[i * 2].toFixed(3) + ', ' + embed2d[i * 2 + 1].toFixed(3) + ')']);
          const outputBlock = formatTable(['i', 'y_i'], outputRows);
          steps.set('6', {
            points: X.slice(), t, edges: mem.edges, colors: dataset.colors || null, embed2d,
            vizKind: 'embedding',
            label: 'Laplacian Eigenmaps embedding',
            ifw: {
              intuition: '<p>The 2D coordinates are the values of v_1 and v_2 at each point. Nearby points on the manifold end up nearby in the embedding because L penalises differences across heavy edges.</p>',
              formula: '$$y_i = (v_{1,i}, v_{2,i})$$',
              worked: workedSections('v_1 and v_2 from step 5.', '$$y_{i,k} = v_{k,i}$$', outputBlock),
            },
          });
          pending.delete('6');
          if (onProgress) onProgress('6');
        },
      ];

      let i = 0;
      const tick = () => {
        if (cancelled || i >= tasks.length) return;
        try { tasks[i++](); } catch (e) { console.error('Laplacian pipeline error:', e); return; }
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

Run: `node --check js/manifold/algorithms/laplacian.js`
Expected: no output.

- [ ] Step 3: Commit

```bash
git add js/manifold/algorithms/laplacian.js
git commit -m "manifold: Laplacian Eigenmaps algorithm module"
```

---

### Task 7: Kernel PCA module

Files:
- Create: `js/manifold/algorithms/kpca.js`

- [ ] Step 1: Create `js/manifold/algorithms/kpca.js`

```javascript
import { topKSymmetricEig, squaredDist3 } from '../linalg.js';
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

function kernelLabel(kernel, gamma, degree, constant) {
  if (kernel === 'rbf') return 'K_{ij} = exp(-' + gamma + ' ||x_i - x_j||^2)';
  if (kernel === 'polynomial') return 'K_{ij} = (x_i . x_j + ' + constant + ')^' + degree;
  return 'K_{ij} = x_i . x_j';
}

function kernelFormula(kernel) {
  if (kernel === 'rbf') return '$$K_{ij} = \\exp(-\\gamma \\| x_i - x_j \\|^2)$$';
  if (kernel === 'polynomial') return '$$K_{ij} = (x_i \\cdot x_j + c)^d$$';
  return '$$K_{ij} = x_i \\cdot x_j$$';
}

export const KPCA = {
  id: 'kpca',
  label: 'Kernel PCA',
  params: [
    { name: 'kernel', type: 'enum', options: ['rbf', 'polynomial', 'linear'], default: 'rbf' },
    { name: 'gamma', type: 'float', default: 0.5, min: 0.01, max: 20 },
    { name: 'degree', type: 'int', default: 3, min: 1, max: 10 },
    { name: 'constant', type: 'float', default: 1, min: 0, max: 10 },
  ],
  presentSubSteps: ['0', '3', '4', '5', '6'],
  pseudocode: [
    { id: 'kpca-K', title: '1. Compute kernel matrix K', steps: ['3'],
      lines: ['rbf: K_{ij} = exp(-gamma ||x_i - x_j||^2)',
              'polynomial: K_{ij} = (x_i . x_j + c)^d',
              'linear: K_{ij} = x_i . x_j'] },
    { id: 'kpca-center', title: '2. Center K', steps: ['4'],
      lines: ['K_c = K - 1_N K - K 1_N + 1_N K 1_N'] },
    { id: 'kpca-eig', title: '3. Eigendecompose K_c', steps: ['5'],
      lines: ['K_c = V Lambda V^T (take top-2)'] },
    { id: 'kpca-embed', title: '4. Form 2D embedding', steps: ['6'],
      lines: ['Y = V[:, 0:2] * diag(sqrt(lambda_1), sqrt(lambda_2))'] },
  ],
  run(dataset, params) {
    const X = dataset.X;
    const t = dataset.t;
    const N = X.length / 3;
    const kernel = params.kernel || 'rbf';
    const gamma = params.gamma || 0.5;
    const degree = params.degree || 3;
    const constant = params.constant || 1;
    const steps = new Map();
    const presentSubSteps = ['0', '3', '4', '5', '6'];
    const pending = new Set(['3', '4', '5', '6']);
    let cancelled = false;
    const samples = sampleIndices(N);

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: dataset.colors || null,
      vizKind: 'point_cloud',
      label: 'Raw data',
      ifw: {
        intuition: '<p>Kernel PCA replaces the inner product with a non-linear kernel, then performs ordinary PCA in the feature space implicitly defined by the kernel.</p>',
        formula: null, worked: null,
      },
    });

    function computeKernel(kernel, gamma, degree, constant) {
      const K = new Float64Array(N * N);
      for (let i = 0; i < N; i++) {
        for (let j = i; j < N; j++) {
          let kij;
          if (kernel === 'rbf') {
            kij = Math.exp(-gamma * squaredDist3(X, i, j));
          } else if (kernel === 'polynomial') {
            const dot = X[i*3]*X[j*3] + X[i*3+1]*X[j*3+1] + X[i*3+2]*X[j*3+2];
            kij = Math.pow(dot + constant, degree);
          } else {
            kij = X[i*3]*X[j*3] + X[i*3+1]*X[j*3+1] + X[i*3+2]*X[j*3+2];
          }
          K[i * N + j] = kij;
          K[j * N + i] = kij;
        }
      }
      return K;
    }

    function start(onProgress) {
      const mem = {};
      const tasks = [
        () => {
          const K = computeKernel(kernel, gamma, degree, constant);
          mem.K = K;
          const i0 = samples[0], j0 = samples[1];
          const exampleK = K[i0 * N + j0];
          const inputBlock = 'sample points (3 of N=' + N + '):\n' +
            samples.map(i => 'x_' + i + ' = ' + formatVec3(rowOf(X, i))).join('\n') +
            '\n\nkernel = ' + kernel + ', gamma = ' + gamma + ', degree = ' + degree + ', constant = ' + constant;
          const excerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(K[r * N + c]);
            excerpt.push(row);
          }
          const outputBlock = 'example: K[' + i0 + '][' + j0 + '] = ' + exampleK.toFixed(4) +
            '\n\nK (4 of N=' + N + ' rows):\n' + formatMatrix(excerpt, { digits: 4 });
          steps.set('3', {
            points: X.slice(), t, edges: null, colors: dataset.colors || null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'cloud_thumb', label: 'X', data: X.slice() },
              { kind: 'heatmap', label: 'K (N x N)', data: { matrix: K, N } },
            ],
            paneOpLabels: [kernelLabel(kernel, gamma, degree, constant)],
            label: 'Kernel matrix K',
            ifw: {
              intuition: '<p>The kernel function K(x, y) measures similarity in an implicit feature space. The full kernel matrix replaces the data matrix in the rest of the pipeline.</p>',
              formula: kernelFormula(kernel),
              worked: workedSections(inputBlock, kernelFormula(kernel), outputBlock),
            },
          });
          pending.delete('3');
          if (onProgress) onProgress('3');
        },
        () => {
          const K = mem.K;
          const Krow = new Float64Array(N);
          for (let i = 0; i < N; i++) {
            let s = 0;
            for (let j = 0; j < N; j++) s += K[i * N + j];
            Krow[i] = s / N;
          }
          let grand = 0;
          for (let i = 0; i < N; i++) grand += Krow[i];
          grand /= N;
          const Kc = new Float64Array(N * N);
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
              Kc[i * N + j] = K[i * N + j] - Krow[i] - Krow[j] + grand;
            }
          }
          mem.Kc = Kc;
          const inputExcerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(K[r * N + c]);
            inputExcerpt.push(row);
          }
          const outExcerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(Kc[r * N + c]);
            outExcerpt.push(row);
          }
          const inputBlock = 'K (4 of N=' + N + ' rows):\n' + formatMatrix(inputExcerpt, { digits: 4 }) +
            '\n\nrow means (first 4): ' + Array.from(Krow.slice(0, 4)).map(v => v.toFixed(3)).join(', ') +
            '\ngrand mean = ' + grand.toFixed(4);
          const outputBlock = 'K_c (4 of N=' + N + ' rows):\n' + formatMatrix(outExcerpt, { digits: 4 });
          steps.set('4', {
            points: X.slice(), t, edges: null, colors: dataset.colors || null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'heatmap', label: 'K', data: { matrix: K, N } },
              { kind: 'heatmap', label: 'K_c (centered)', data: { matrix: Kc, N } },
            ],
            paneOpLabels: ['K - 1_N K - K 1_N + 1_N K 1_N'],
            label: 'Centered kernel matrix',
            ifw: {
              intuition: '<p>Centering K is the kernel-space analogue of centering the data before ordinary PCA. The double subtraction and grand mean addition account for the row and column shifts implied by the centering operator.</p>',
              formula: '$$K_c = K - \\mathbf{1}_N K - K \\mathbf{1}_N + \\mathbf{1}_N K \\mathbf{1}_N$$',
              worked: workedSections(inputBlock, '$$K_{c,ij} = K_{ij} - r_i - r_j + g$$', outputBlock),
            },
          });
          pending.delete('4');
          if (onProgress) onProgress('4');
        },
        () => {
          const { lambda, vectors } = topKSymmetricEig(mem.Kc, N, 8);
          mem.lambda = lambda;
          mem.vectors = vectors;
          const inputBlock = 'K_c from step 4.';
          const outputBlock = 'top eigenvalues: ' + Array.from(lambda).map((v, i) => 'lambda_' + (i + 1) + ' = ' + v.toFixed(3)).join(', ') +
            '\n\nv_1 (first 6 entries): [' + Array.from(vectors[0].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']' +
            '\nv_2 (first 6 entries): [' + Array.from(vectors[1].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']';
          steps.set('5', {
            points: X.slice(), t, edges: null, colors: dataset.colors || null,
            vizKind: 'spectral',
            algoId: 'kpca',
            v1Values: vectors[0],
            topEigvals: lambda,
            topEigvecs: vectors,
            label: 'Top-2 eigendecomposition',
            ifw: {
              intuition: '<p>The principal components in feature space are eigenvectors of the centered kernel matrix. Each carries one coordinate of the non-linear embedding.</p>',
              formula: '$$K_c\\, v_k = \\lambda_k\\, v_k$$',
              worked: workedSections(inputBlock, '$$K_c\\, v_k = \\lambda_k\\, v_k$$', outputBlock),
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
          const inputBlock = 'lambda_1 = ' + lambda[0].toFixed(3) + ', lambda_2 = ' + lambda[1].toFixed(3);
          const outputRows = samples.map(i => [i,
            '(' + embed2d[i * 2].toFixed(3) + ', ' + embed2d[i * 2 + 1].toFixed(3) + ')']);
          const outputBlock = formatTable(['i', 'y_i'], outputRows);
          steps.set('6', {
            points: X.slice(), t, edges: null, colors: dataset.colors || null, embed2d,
            vizKind: 'embedding',
            label: 'Kernel PCA embedding',
            ifw: {
              intuition: '<p>Each point\'s 2D coordinate is its projection onto the top-2 principal components in the kernel-induced feature space.</p>',
              formula: '$$y_{i,k} = \\sqrt{\\lambda_k}\\, v_{k,i}$$',
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
        try { tasks[i++](); } catch (e) { console.error('KPCA pipeline error:', e); return; }
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

Run: `node --check js/manifold/algorithms/kpca.js`
Expected: no output.

- [ ] Step 3: Commit

```bash
git add js/manifold/algorithms/kpca.js
git commit -m "manifold: Kernel PCA algorithm module (RBF, polynomial, linear)"
```

---

### Task 8: LLE module

Files:
- Create: `js/manifold/algorithms/lle.js`

- [ ] Step 1: Create `js/manifold/algorithms/lle.js`

```javascript
import { knnGraph, bottomKSymmetricEig, solveLinearSystem } from '../linalg.js';
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

export const LLE = {
  id: 'lle',
  label: 'LLE',
  params: [
    { name: 'k', type: 'int', default: 10, min: 2, max: 50 },
    { name: 'reg', type: 'float', default: 1e-3, min: 0, max: 0.1 },
  ],
  presentSubSteps: ['0', '2', '3', '5', '6'],
  pseudocode: [
    { id: 'lle-knn', title: '1. Build kNN graph', steps: ['2'],
      lines: ['neighbours_i = k points with smallest ||x_j - x_i||'] },
    { id: 'lle-W', title: '2. Reconstruction weights W', steps: ['3'],
      lines: ['for each i: solve min_w || x_i - sum_j w_j x_{n_j} ||^2',
              'subject to sum w_j = 1',
              'store W[i][n_j] = w_j'] },
    { id: 'lle-eig', title: '3. Smallest non-trivial eigenvectors of M', steps: ['5'],
      lines: ['M = (I - W)^T (I - W)', 'M v_k = lambda_k v_k (skip lambda_0 = 0)'] },
    { id: 'lle-embed', title: '4. Form 2D embedding', steps: ['6'],
      lines: ['Y = [v_1, v_2]'] },
  ],
  run(dataset, params) {
    const X = dataset.X;
    const t = dataset.t;
    const N = X.length / 3;
    const k = Math.max(2, Math.min(params.k || 10, N - 1));
    const reg = params.reg !== undefined ? params.reg : 1e-3;
    const steps = new Map();
    const presentSubSteps = ['0', '2', '3', '5', '6'];
    const pending = new Set(['2', '3', '5', '6']);
    let cancelled = false;
    const samples = sampleIndices(N);

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: dataset.colors || null,
      vizKind: 'point_cloud',
      label: 'Raw data',
      ifw: {
        intuition: '<p>LLE reconstructs each point as a weighted sum of its k nearest neighbours, then finds a low-dimensional embedding that preserves those same weights.</p>',
        formula: null, worked: null,
      },
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
            formatTable(['j', '||x_j - x_i||'], neighbours);
          steps.set('2', {
            points: X.slice(), t, edges, colors: dataset.colors || null,
            vizKind: 'knn_graph',
            label: 'kNN graph (k = ' + k + ')',
            ifw: {
              intuition: '<p>LLE assumes each point lies on a locally linear patch defined by its k nearest neighbours. The kNN graph identifies those neighbourhoods.</p>',
              formula: '$$\\mathcal{N}_i = \\{ j : \\|x_j - x_i\\| \\text{ among the } k \\text{ smallest}\\}$$',
              worked: workedSections(inputBlock, '$$w^{\\text{edge}}_{ij} = \\| x_j - x_i \\|$$', outputBlock),
            },
          });
          pending.delete('2');
          if (onProgress) onProgress('2');
        },
        () => {
          const adj = mem.adj;
          const W = new Float64Array(N * N);
          for (let i = 0; i < N; i++) {
            const neighbours = adj[i].slice(0, k).map(([j]) => j);
            const G = [];
            for (let a = 0; a < neighbours.length; a++) {
              const row = new Array(neighbours.length);
              const na = neighbours[a];
              const ax = X[na * 3] - X[i * 3], ay = X[na * 3 + 1] - X[i * 3 + 1], az = X[na * 3 + 2] - X[i * 3 + 2];
              for (let b = 0; b < neighbours.length; b++) {
                const nb = neighbours[b];
                const bx = X[nb * 3] - X[i * 3], by = X[nb * 3 + 1] - X[i * 3 + 1], bz = X[nb * 3 + 2] - X[i * 3 + 2];
                row[b] = ax * bx + ay * by + az * bz;
              }
              G.push(row);
            }
            let trace = 0;
            for (let a = 0; a < G.length; a++) trace += G[a][a];
            const lambda = reg * Math.max(trace, 1e-12) / G.length;
            for (let a = 0; a < G.length; a++) G[a][a] += lambda;
            const b = new Array(neighbours.length).fill(1);
            const w = solveLinearSystem(G, b);
            if (!w) continue;
            let sum = 0;
            for (let a = 0; a < w.length; a++) sum += w[a];
            if (Math.abs(sum) < 1e-12) continue;
            for (let a = 0; a < w.length; a++) w[a] /= sum;
            for (let a = 0; a < neighbours.length; a++) {
              W[i * N + neighbours[a]] = w[a];
            }
          }
          mem.W = W;
          const sampleI = samples[0];
          const wRow = [];
          for (let j = 0; j < N; j++) {
            const v = W[sampleI * N + j];
            if (v !== 0) wRow.push([j, v.toFixed(4)]);
          }
          const inputBlock = 'sample point i = ' + sampleI + ', k = ' + k + ', reg = ' + reg + '\nlocal Gram matrix G has size k x k.';
          const outputBlock = 'W_ij for the k = ' + wRow.length + ' neighbours of point ' + sampleI + ':\n' +
            formatTable(['j', 'W_ij'], wRow.slice(0, 8)) +
            (wRow.length > 8 ? '\n... (' + (wRow.length - 8) + ' more)' : '');
          steps.set('3', {
            points: X.slice(), t, edges: mem.edges, colors: dataset.colors || null,
            vizKind: 'weighted_knn',
            W,
            k,
            selectedPoint: sampleI,
            label: 'Reconstruction weights',
            ifw: {
              intuition: '<p>Each point is described as a weighted combination of its k neighbours. The weights are solved by a small linear system per point so that the linear combination best reconstructs the point, with the weights normalised to sum to 1.</p>',
              formula: '$$\\min_{w_i}\\ \\bigl\\| x_i - \\sum_{j \\in \\mathcal{N}_i} w_{ij}\\, x_j \\bigr\\|^2,\\ \\sum_j w_{ij} = 1$$',
              worked: workedSections(inputBlock,
                '$$G_{ab} = (x_{n_a} - x_i) \\cdot (x_{n_b} - x_i),\\ G\\,w = \\mathbf{1},\\ w \\leftarrow w / \\sum w$$',
                outputBlock),
            },
          });
          pending.delete('3');
          if (onProgress) onProgress('3');
        },
        () => {
          const W = mem.W;
          const IW = new Float64Array(N * N);
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
              IW[i * N + j] = (i === j ? 1 : 0) - W[i * N + j];
            }
          }
          const M = new Float64Array(N * N);
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
              let s = 0;
              for (let p = 0; p < N; p++) s += IW[p * N + i] * IW[p * N + j];
              M[i * N + j] = s;
            }
          }
          mem.M = M;
          const { lambda, vectors } = bottomKSymmetricEig(M, N, 8, { skipFirst: 1 });
          mem.lambda = lambda;
          mem.vectors = vectors;
          const excerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(M[r * N + c]);
            excerpt.push(row);
          }
          const inputBlock = 'M (4 of N=' + N + ' rows):\n' + formatMatrix(excerpt, { digits: 3 });
          const outputBlock = 'bottom non-trivial eigenvalues: ' + Array.from(lambda).map((v, i) => 'lambda_' + (i + 1) + ' = ' + v.toFixed(4)).join(', ') +
            '\n\nv_1 (first 6 entries): [' + Array.from(vectors[0].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']' +
            '\nv_2 (first 6 entries): [' + Array.from(vectors[1].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']';
          steps.set('5', {
            points: X.slice(), t, edges: mem.edges, colors: dataset.colors || null,
            vizKind: 'spectral',
            algoId: 'lle',
            v1Values: vectors[0],
            topEigvals: lambda,
            topEigvecs: vectors,
            label: 'Smallest non-trivial eigenvectors of M',
            ifw: {
              intuition: '<p>The smallest non-trivial eigenvectors of M produce coordinates that minimise the embedding cost while keeping the same local reconstruction weights. The trivial zero eigenvalue (constant eigenvector) is dropped.</p>',
              formula: '$$M\\, v_k = \\lambda_k\\, v_k,\\quad M = (I - W)^{\\top}(I - W)$$',
              worked: workedSections(inputBlock, '$$M\\, v_k = \\lambda_k\\, v_k$$', outputBlock),
            },
          });
          pending.delete('5');
          if (onProgress) onProgress('5');
        },
        () => {
          const vectors = mem.vectors;
          const embed2d = new Float64Array(N * 2);
          for (let i = 0; i < N; i++) {
            embed2d[i * 2] = vectors[0][i];
            embed2d[i * 2 + 1] = vectors[1][i];
          }
          const outputRows = samples.map(i => [i,
            '(' + embed2d[i * 2].toFixed(3) + ', ' + embed2d[i * 2 + 1].toFixed(3) + ')']);
          const outputBlock = formatTable(['i', 'y_i'], outputRows);
          steps.set('6', {
            points: X.slice(), t, edges: mem.edges, colors: dataset.colors || null, embed2d,
            vizKind: 'embedding',
            label: 'LLE embedding',
            ifw: {
              intuition: '<p>The 2D coordinates are the values of v_1 and v_2 at each point. LLE does not scale by eigenvalues; the absolute scale is fixed by the cost-function normalisation.</p>',
              formula: '$$y_i = (v_{1,i}, v_{2,i})$$',
              worked: workedSections('v_1 and v_2 from step 5.', '$$y_{i,k} = v_{k,i}$$', outputBlock),
            },
          });
          pending.delete('6');
          if (onProgress) onProgress('6');
        },
      ];

      let i = 0;
      const tick = () => {
        if (cancelled || i >= tasks.length) return;
        try { tasks[i++](); } catch (e) { console.error('LLE pipeline error:', e); return; }
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

Run: `node --check js/manifold/algorithms/lle.js`
Expected: no output.

- [ ] Step 3: Commit

```bash
git add js/manifold/algorithms/lle.js
git commit -m "manifold: LLE algorithm module with reconstruction weights"
```

---

### Task 9: Register algorithms in main.js and worker.js

Files:
- Modify: `js/manifold/main.js`
- Modify: `js/manifold/worker.js`

- [ ] Step 1: Update imports and ALGORITHMS in `js/manifold/main.js`

Open `js/manifold/main.js`. Replace the existing imports block top with:

```javascript
import { DATASETS, parseCSV } from './datasets.js';
import { PCA } from './algorithms/pca.js';
import { ISOMAP } from './algorithms/isomap.js';
import { MDS } from './algorithms/mds.js';
import { LLE } from './algorithms/lle.js';
import { LAPLACIAN } from './algorithms/laplacian.js';
import { KPCA } from './algorithms/kpca.js';
import { createState } from './state.js';
import { createStepViz } from './step_viz.js';
import { createStepIndicator } from './step_indicator.js';
import { createIFW } from './ifw.js';
import { createPseudocode } from './pseudocode.js';
import { compareSubSteps, unionSubSteps } from './canonical_steps.js';
```

Find the existing `ALGORITHMS` constant:

```javascript
const ALGORITHMS = [PCA, ISOMAP];
```

Replace with:

```javascript
const ALGORITHMS = [PCA, ISOMAP, MDS, LLE, LAPLACIAN, KPCA];
```

Find the existing `defaults.algoParams`:

```javascript
  algoParams: { pca: {}, isomap: { k: 10 } },
```

Replace with:

```javascript
  algoParams: {
    pca: {},
    isomap: { k: 10 },
    mds: {},
    lle: { k: 10, reg: 1e-3 },
    laplacian: { k: 10, sigma: 1.0 },
    kpca: { kernel: 'rbf', gamma: 0.5, degree: 3, constant: 1 },
  },
```

- [ ] Step 2: Update worker.js to register the new algorithms

Open `js/manifold/worker.js`. Replace its imports and ALGORITHMS:

```javascript
import { PCA } from './algorithms/pca.js';
import { ISOMAP } from './algorithms/isomap.js';
import { MDS } from './algorithms/mds.js';
import { LLE } from './algorithms/lle.js';
import { LAPLACIAN } from './algorithms/laplacian.js';
import { KPCA } from './algorithms/kpca.js';

const ALGORITHMS = { pca: PCA, isomap: ISOMAP, mds: MDS, lle: LLE, laplacian: LAPLACIAN, kpca: KPCA };
```

Leave the rest of `worker.js` unchanged.

- [ ] Step 3: Syntax check

Run: `node --check js/manifold/main.js && node --check js/manifold/worker.js`
Expected: no output.

- [ ] Step 4: Run the full unit test suite

Run: `node --test 'test/manifold/*.test.js' 2>&1 | tail -5`
Expected: tests 34, pass 34, fail 0.

- [ ] Step 5: Commit

```bash
git add js/manifold/main.js js/manifold/worker.js
git commit -m "manifold: register MDS, LLE, Laplacian Eigenmaps, Kernel PCA in main and worker"
```

---

### Task 10: Browser smoke verification

Files: none changed.

- [ ] Step 1: Start a static server

```bash
python3 -m http.server 8765 --bind 127.0.0.1 > /tmp/manifold-server.log 2>&1 &
echo "pid=$!"
sleep 1
```

- [ ] Step 2: HTTP 200 check on every new file plus the page itself

```bash
for f in pages/manifold.html \
         js/manifold/algorithms/mds.js \
         js/manifold/algorithms/lle.js \
         js/manifold/algorithms/laplacian.js \
         js/manifold/algorithms/kpca.js \
         js/manifold/viz/viz_weighted_knn.js \
         js/manifold/linalg.js \
         js/manifold/main.js \
         js/manifold/worker.js; do
  curl -s -o /dev/null -w "%{http_code} $f\n" "http://127.0.0.1:8765/$f"
done
```

Expected: 200 on every line.

- [ ] Step 3: Open the page and walk through each new algorithm

Open: `http://127.0.0.1:8765/pages/manifold.html`

For each of MDS, LLE, Laplacian Eigenmaps, Kernel PCA:

1. Select the algorithm as A or B (or both).
2. Confirm the algorithm parameter row appears below the dropdown. For Kernel PCA confirm the kernel selector is a `<select>` showing `rbf`, `polynomial`, `linear`.
3. Walk Prev / Next through each canonical sub-step. Confirm the viewport shows the expected vizKind (point cloud at step 0; matrix strip at steps 3 / 4; weighted_knn at LLE step 3; spectral at step 5; 2D scatter at step 6).
4. Open the Worked example tab at each step and confirm Input / Formula / Output blocks render with real numbers.
5. For LLE step 3: hover other nodes in the 3D viewport and confirm the highlighted point changes and the edges around it light up.
6. For Kernel PCA: switch the kernel selector and confirm the K heatmap and embedding change accordingly.

If any item fails, file the specific defect, fix it, and re-run this checklist.

- [ ] Step 4: Stop the server

```bash
ps -ef | grep -v grep | grep "http.server 8765" | awk '{print $2}' | xargs -r kill
```

---

### Task 11: Final test sweep

Files: none changed.

- [ ] Step 1: Run all unit tests

```bash
node --test 'test/manifold/*.test.js' 2>&1 | tail -5
```

Expected: tests 34, pass 34, fail 0.

- [ ] Step 2: Confirm git log shows one commit per code-producing task

```bash
git log --oneline 79c6d50..HEAD
```

Expected: 9 commits (Tasks 1 through 9 produced commits; Tasks 10 and 11 are verification only).

- [ ] Step 3: Confirm clean working tree

```bash
git status --short
```

Expected: no output.

---

## Self-review

Spec coverage:
- Linalg additions (`bottomKSymmetricEig`, plus the `solveLinearSystem` helper that LLE needs): Tasks 1 and 2.
- New viz `viz_weighted_knn` plus dispatcher branch and CSS: Task 3.
- Enum param type for KPCA's kernel selector: Task 4.
- Algorithm modules MDS, Laplacian, KPCA, LLE: Tasks 5, 6, 7, 8.
- Registration in main.js (ALGORITHMS, default params) and worker.js: Task 9.
- Verification: Tasks 10 and 11.

Placeholder scan: no TBDs or "implement later" markers. Every step contains either complete code, a complete shell command, or a deterministic UI check. The browser smoke task in Task 10 lists explicit per-algorithm verifications.

Type consistency: all four algorithm modules return `{ steps: Map, presentSubSteps: string[], pending: Set, start, cancel }`. Step states carry the same `vizKind` strings the dispatcher understands. `bottomKSymmetricEig` is consumed only by LLE step 5 and Laplacian step 5; both call it the same way (`(M, N, 8, { skipFirst: 1 })`). `solveLinearSystem` is consumed only by LLE step 3.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-29-manifold-phase-2a.md`. Two execution options:

1. Subagent-Driven (recommended): I dispatch a fresh subagent per task with two-stage review. Uses superpowers:subagent-driven-development.
2. Inline Execution: I execute the tasks in this session with checkpoint reviews. Uses superpowers:executing-plans.

Which approach?
