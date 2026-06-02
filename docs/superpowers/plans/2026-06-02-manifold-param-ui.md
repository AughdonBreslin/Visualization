# Manifold Parameter UI Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bare `name = [input]` parameter chips with a left-aligned two-column layout, readable parameter labels, and a hover tooltip that defines each parameter and its effect; also fix the LLE default-k mismatch to 12.

**Architecture:** Parameter descriptors in each algorithm and dataset module gain optional `label` and `desc` fields. A new tiny `param_tooltip.js` module owns a single floating tooltip element. `renderParamHost` in `main.js` is rewritten to emit a two-column grid (info icon + label, then control) and wire tooltips. New CSS styles the grid, info icon, inputs, and tooltip.

**Tech Stack:** Vanilla ES modules, d3 v7 (global), no build step. Node 22 `node --test` for unit tests. DOM and CSS verified by manual browser smoke.

---

## File structure

- Modify: `js/manifold/algorithms/isomap.js`, `lle.js`, `laplacian.js`, `kpca.js` (add `label`/`desc`; LLE also `k` default 10 -> 12)
- Modify: `js/manifold/datasets/synthetic_curves.js`, `synthetic_surfaces.js`, `synthetic_clusters.js` (add `label`/`desc`)
- Create: `js/manifold/param_tooltip.js` (floating tooltip helper)
- Modify: `js/manifold/main.js` (`renderParamHost` rewrite + import)
- Modify: `styles/manifold.css` (grid, info icon, inputs, tooltip)
- Create test: `test/manifold/param_metadata.test.js` (algorithms)
- Create test: `test/manifold/param_metadata_datasets.test.js` (datasets)

Style constraints (apply to every task): no em-dashes anywhere; no `<em>`/`<strong>`/`<b>`/`<i>`/`<mark>` tags in generated content (the info icon is a `<span>` containing the letter "i", not an `<i>` element). The unicode letters σ and γ are allowed and already used elsewhere in the codebase.

---

### Task 1: Algorithm parameter metadata + LLE k fix

**Files:**
- Create test: `test/manifold/param_metadata.test.js`
- Modify: `js/manifold/algorithms/isomap.js`
- Modify: `js/manifold/algorithms/lle.js`
- Modify: `js/manifold/algorithms/laplacian.js`
- Modify: `js/manifold/algorithms/kpca.js`

- [ ] **Step 1: Write the failing test**

Create `test/manifold/param_metadata.test.js`:

```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PCA } from '../../js/manifold/algorithms/pca.js';
import { ISOMAP } from '../../js/manifold/algorithms/isomap.js';
import { MDS } from '../../js/manifold/algorithms/mds.js';
import { LLE } from '../../js/manifold/algorithms/lle.js';
import { LAPLACIAN } from '../../js/manifold/algorithms/laplacian.js';
import { KPCA } from '../../js/manifold/algorithms/kpca.js';

const ALGOS = [PCA, ISOMAP, MDS, LLE, LAPLACIAN, KPCA];

test('every algorithm parameter has a non-empty label and description', () => {
  for (const a of ALGOS) {
    for (const p of a.params) {
      assert.equal(typeof p.label, 'string', `${a.id}.${p.name} label type`);
      assert.ok(p.label.length > 0, `${a.id}.${p.name} label non-empty`);
      assert.equal(typeof p.desc, 'string', `${a.id}.${p.name} desc type`);
      assert.ok(p.desc.length > 10, `${a.id}.${p.name} desc non-trivial`);
    }
  }
});

test('LLE default k is 12', () => {
  const k = LLE.params.find(p => p.name === 'k');
  assert.equal(k.default, 12);
});
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `node --test test/manifold/param_metadata.test.js 2>&1 | tail -6`
Expected: FAIL (params have no `label`/`desc`; LLE k default is 10).

- [ ] **Step 3: Edit `js/manifold/algorithms/isomap.js`**

Find:
```javascript
  params: [{ name: 'k', type: 'int', default: 10, min: 2, max: 50 }],
```
Replace with:
```javascript
  params: [{ name: 'k', type: 'int', default: 10, min: 2, max: 50,
    label: 'Neighbors (k)',
    desc: 'How many nearest neighbors define each point\'s local neighborhood. Smaller k captures finer local detail but can fragment the manifold; larger k is smoother but may link points across separate folds.' }],
```

- [ ] **Step 4: Edit `js/manifold/algorithms/lle.js` (also fix k default to 12)**

Find:
```javascript
  params: [
    { name: 'k', type: 'int', default: 10, min: 2, max: 50 },
    { name: 'reg', type: 'float', default: 1e-3, min: 0, max: 0.1 },
  ],
```
Replace with:
```javascript
  params: [
    { name: 'k', type: 'int', default: 12, min: 2, max: 50,
      label: 'Neighbors (k)',
      desc: 'How many nearest neighbors define each point\'s local neighborhood. Smaller k captures finer local detail but can fragment the manifold; larger k is smoother but may link points across separate folds.' },
    { name: 'reg', type: 'float', default: 1e-3, min: 0, max: 0.1,
      label: 'Regularization',
      desc: 'Stabilizes the per-point least-squares weight solve when a point\'s neighbors are nearly coplanar. Larger values make the reconstruction weights smoother and more uniform; too large washes out the local geometry.' },
  ],
```

- [ ] **Step 5: Edit `js/manifold/algorithms/laplacian.js`**

Find:
```javascript
    { name: 'k', type: 'int', default: 10, min: 2, max: 50 },
    { name: 'sigma', type: 'float', default: 3.0, min: 0.1, max: 10 },
```
Replace with:
```javascript
    { name: 'k', type: 'int', default: 10, min: 2, max: 50,
      label: 'Neighbors (k)',
      desc: 'How many nearest neighbors define each point\'s local neighborhood. Smaller k captures finer local detail but can fragment the manifold; larger k is smoother but may link points across separate folds.' },
    { name: 'sigma', type: 'float', default: 3.0, min: 0.1, max: 10,
      label: 'Bandwidth (σ)',
      desc: 'Width of the heat kernel that turns neighbor distances into edge weights, as a multiple of the median neighbor distance. Larger σ makes neighbor weights more uniform and the embedding smoother; smaller σ sharpens the falloff.' },
```

- [ ] **Step 6: Edit `js/manifold/algorithms/kpca.js`**

Find:
```javascript
    { name: 'kernel', type: 'enum', options: ['rbf', 'polynomial', 'linear'], default: 'rbf' },
    { name: 'gamma', type: 'float', default: 1.0, min: 0.05, max: 10, dependsOn: { kernel: 'rbf' } },
    { name: 'degree', type: 'int', default: 3, min: 1, max: 10, dependsOn: { kernel: 'polynomial' } },
```
Replace with:
```javascript
    { name: 'kernel', type: 'enum', options: ['rbf', 'polynomial', 'linear'], default: 'rbf',
      label: 'Kernel',
      desc: 'The similarity function. Kernel PCA runs ordinary PCA in the implicit feature space this kernel defines, so the choice sets what kind of nonlinear structure it can unfold.' },
    { name: 'gamma', type: 'float', default: 1.0, min: 0.05, max: 10, dependsOn: { kernel: 'rbf' },
      label: 'RBF width (γ)',
      desc: 'Width of the RBF (Gaussian) kernel, auto-scaled to the data\'s spread. Larger γ makes similarity drop off faster (very local, fine detail); smaller γ is broad and smooth.' },
    { name: 'degree', type: 'int', default: 3, min: 1, max: 10, dependsOn: { kernel: 'polynomial' },
      label: 'Polynomial degree (d)',
      desc: 'Degree of the polynomial kernel. Degree 1 is linear; higher degrees capture higher-order interactions among coordinates, bending the feature space more.' },
```

- [ ] **Step 7: Syntax check all four files**

Run: `for f in isomap lle laplacian kpca; do node --check js/manifold/algorithms/$f.js || echo FAIL $f; done`
Expected: no output.

- [ ] **Step 8: Run the test, confirm it passes**

Run: `node --test test/manifold/param_metadata.test.js 2>&1 | tail -6`
Expected: 2 tests, 2 pass, 0 fail.

- [ ] **Step 9: Confirm no banned styling**

Run: `grep -nE "—|<em>|<strong>|<b>|<i>|<mark>" js/manifold/algorithms/isomap.js js/manifold/algorithms/lle.js js/manifold/algorithms/laplacian.js js/manifold/algorithms/kpca.js`
Expected: no output.

- [ ] **Step 10: Commit**

```bash
git add test/manifold/param_metadata.test.js js/manifold/algorithms/isomap.js js/manifold/algorithms/lle.js js/manifold/algorithms/laplacian.js js/manifold/algorithms/kpca.js
git commit -m "manifold: add label/desc to algorithm params and fix LLE default k to 12"
```

---

### Task 2: Dataset parameter metadata

**Files:**
- Create test: `test/manifold/param_metadata_datasets.test.js`
- Modify: `js/manifold/datasets/synthetic_curves.js`
- Modify: `js/manifold/datasets/synthetic_surfaces.js`
- Modify: `js/manifold/datasets/synthetic_clusters.js`

- [ ] **Step 1: Write the failing test**

Create `test/manifold/param_metadata_datasets.test.js`:

```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DATASETS } from '../../js/manifold/datasets/index.js';

test('every dataset parameter has a non-empty label and description', () => {
  for (const d of DATASETS) {
    for (const p of (d.params || [])) {
      assert.equal(typeof p.label, 'string', `${d.id}.${p.name} label type`);
      assert.ok(p.label.length > 0, `${d.id}.${p.name} label non-empty`);
      assert.equal(typeof p.desc, 'string', `${d.id}.${p.name} desc type`);
      assert.ok(p.desc.length > 10, `${d.id}.${p.name} desc non-trivial`);
    }
  }
});
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `node --test test/manifold/param_metadata_datasets.test.js 2>&1 | tail -6`
Expected: FAIL (dataset params have no `label`/`desc`).

- [ ] **Step 3: Edit `js/manifold/datasets/synthetic_curves.js` (helix)**

Find:
```javascript
  params: [{ name: 'turns', type: 'int', default: 3, min: 1, max: 8 }],
```
Replace with:
```javascript
  params: [{ name: 'turns', type: 'int', default: 3, min: 1, max: 8,
    label: 'Turns',
    desc: 'Number of full turns of the helix. More turns make a longer, tighter coil.' }],
```

- [ ] **Step 4: Edit `js/manifold/datasets/synthetic_curves.js` (toroidal helix)**

Find:
```javascript
  params: [{ name: 'q', type: 'int', default: 7, min: 2, max: 15 }],
```
Replace with:
```javascript
  params: [{ name: 'q', type: 'int', default: 7, min: 2, max: 15,
    label: 'Winding (q)',
    desc: 'How many times the helix winds around the torus tube per loop around the ring. Higher q coils more tightly.' }],
```

- [ ] **Step 5: Edit `js/manifold/datasets/synthetic_curves.js` (spiral disk)**

Find:
```javascript
  params: [{ name: 'turns', type: 'int', default: 3, min: 1, max: 6 }],
```
Replace with:
```javascript
  params: [{ name: 'turns', type: 'int', default: 3, min: 1, max: 6,
    label: 'Turns',
    desc: 'Number of turns of the spiral arm. More turns wind it tighter toward the center.' }],
```

- [ ] **Step 6: Edit `js/manifold/datasets/synthetic_surfaces.js` (cylinder)**

Find:
```javascript
  params: [{ name: 'height', type: 'float', default: 2, min: 0.5, max: 5 }],
```
Replace with:
```javascript
  params: [{ name: 'height', type: 'float', default: 2, min: 0.5, max: 5,
    label: 'Height',
    desc: 'Height of the open cylinder. Taller values stretch the tube along its axis.' }],
```

- [ ] **Step 7: Edit `js/manifold/datasets/synthetic_surfaces.js` (severed sphere)**

Find:
```javascript
  params: [{ name: 'cap', type: 'float', default: 0.35, min: 0, max: 0.9 }],
```
Replace with:
```javascript
  params: [{ name: 'cap', type: 'float', default: 0.35, min: 0, max: 0.9,
    label: 'Cap size',
    desc: 'Fraction of the sphere removed at the north pole, measured along the polar angle. Larger values cut away more of the top.' }],
```

- [ ] **Step 8: Edit `js/manifold/datasets/synthetic_surfaces.js` (hilbert)**

Find:
```javascript
  params: [{ name: 'order', type: 'int', default: 4, min: 2, max: 5 }],
```
Replace with:
```javascript
  params: [{ name: 'order', type: 'int', default: 4, min: 2, max: 5,
    label: 'Order',
    desc: 'Recursion depth of the Hilbert curve. Each step subdivides the cube into a finer grid (2^order cells per side), so higher order packs the curve more densely.' }],
```

- [ ] **Step 9: Edit `js/manifold/datasets/synthetic_clusters.js`**

Find:
```javascript
  params: [
    { name: 'clusters', type: 'int', default: 5, min: 2, max: 8 },
    { name: 'sep', type: 'float', default: 2, min: 0.5, max: 5 },
  ],
```
Replace with:
```javascript
  params: [
    { name: 'clusters', type: 'int', default: 5, min: 2, max: 8,
      label: 'Clusters',
      desc: 'Number of Gaussian blobs, with centers spread evenly on a sphere.' },
    { name: 'sep', type: 'float', default: 2, min: 0.5, max: 5,
      label: 'Separation',
      desc: 'Radius of the sphere the cluster centers sit on. Larger values push the blobs farther apart.' },
  ],
```

- [ ] **Step 10: Syntax check + run test + style scan**

Run:
```bash
for f in synthetic_curves synthetic_surfaces synthetic_clusters; do node --check js/manifold/datasets/$f.js || echo FAIL $f; done
node --test test/manifold/param_metadata_datasets.test.js 2>&1 | tail -6
grep -nE "—|<em>|<strong>|<b>|<i>|<mark>" js/manifold/datasets/synthetic_curves.js js/manifold/datasets/synthetic_surfaces.js js/manifold/datasets/synthetic_clusters.js
```
Expected: no syntax output; 1 test passes; no grep output.

- [ ] **Step 11: Commit**

```bash
git add test/manifold/param_metadata_datasets.test.js js/manifold/datasets/synthetic_curves.js js/manifold/datasets/synthetic_surfaces.js js/manifold/datasets/synthetic_clusters.js
git commit -m "manifold: add label/desc to dataset params"
```

---

### Task 3: Tooltip helper module

**Files:**
- Create: `js/manifold/param_tooltip.js`

- [ ] **Step 1: Create `js/manifold/param_tooltip.js`**

```javascript
// Single shared floating tooltip for parameter info icons.
let tipEl = null;

function ensureTip() {
  if (tipEl) return tipEl;
  tipEl = document.createElement('div');
  tipEl.className = 'mf-tooltip';
  document.body.appendChild(tipEl);
  return tipEl;
}

export function attachTooltip(iconEl, { label, desc, rangeText }) {
  iconEl.addEventListener('mousemove', (e) => {
    const tip = ensureTip();
    let html = '';
    if (label) html += '<span class="mf-tooltip-title">' + label + '</span>';
    html += desc || '';
    if (rangeText) html += '<div class="mf-tooltip-range">' + rangeText + '</div>';
    tip.innerHTML = html;
    tip.style.opacity = '1';
    let x = e.clientX + 14;
    let y = e.clientY + 14;
    if (x + 300 > window.innerWidth) x = e.clientX - 304;
    if (y + 130 > window.innerHeight) y = e.clientY - 130;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  });
  iconEl.addEventListener('mouseleave', () => {
    if (tipEl) tipEl.style.opacity = '0';
  });
}
```

- [ ] **Step 2: Syntax check**

Run: `node --check js/manifold/param_tooltip.js`
Expected: no output.

- [ ] **Step 3: Confirm no banned styling**

Run: `grep -nE "—|<em>|<strong>|<b>|<i>|<mark>" js/manifold/param_tooltip.js`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add js/manifold/param_tooltip.js
git commit -m "manifold: shared floating tooltip helper for parameter info icons"
```

---

### Task 4: Rewrite renderParamHost

**Files:**
- Modify: `js/manifold/main.js`

- [ ] **Step 1: Add the import**

In `js/manifold/main.js`, find:
```javascript
import { compareSubSteps, unionSubSteps } from './canonical_steps.js';
```
Add directly after it:
```javascript
import { attachTooltip } from './param_tooltip.js';
```

- [ ] **Step 2: Replace the `renderParamHost` function**

Find the entire current function:
```javascript
  function renderParamHost(host, algo, getCurrent, onChange) {
    host.innerHTML = '';
    const current = getCurrent();
    const visible = algo.params.filter(p => {
      if (!p.dependsOn) return true;
      for (const k of Object.keys(p.dependsOn)) {
        if (current[k] !== p.dependsOn[k]) return false;
      }
      return true;
    });
    for (const p of visible) {
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
          onChange({ ...getCurrent(), [p.name]: sel.value });
          renderParamHost(host, algo, getCurrent, onChange);
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
        onChange({ ...getCurrent(), [p.name]: v });
      });
      wrap.appendChild(input);
      host.appendChild(wrap);
    }
    if (visible.length === 0) host.innerHTML = '<span class="mf-noparams">No parameters</span>';
  }
```

Replace with:
```javascript
  function renderParamHost(host, algo, getCurrent, onChange) {
    host.innerHTML = '';
    const current = getCurrent();
    const visible = algo.params.filter(p => {
      if (!p.dependsOn) return true;
      for (const k of Object.keys(p.dependsOn)) {
        if (current[k] !== p.dependsOn[k]) return false;
      }
      return true;
    });
    if (visible.length === 0) {
      host.innerHTML = '<span class="mf-noparams">No parameters</span>';
      return;
    }
    const grid = document.createElement('div');
    grid.className = 'mf-param-grid';
    for (const p of visible) {
      const lbl = document.createElement('div');
      lbl.className = 'mf-param-label';
      if (p.desc) {
        const info = document.createElement('span');
        info.className = 'mf-param-info';
        info.textContent = 'i';
        const rangeText = p.type === 'enum'
          ? p.options.join(' / ')
          : (p.min !== undefined && p.max !== undefined ? `range ${p.min} to ${p.max}` : '');
        attachTooltip(info, { label: p.label || p.name, desc: p.desc, rangeText });
        lbl.appendChild(info);
      }
      const name = document.createElement('span');
      name.className = 'mf-param-name';
      name.textContent = p.label || p.name;
      lbl.appendChild(name);
      grid.appendChild(lbl);

      const cell = document.createElement('div');
      cell.className = 'mf-param-control';
      if (p.type === 'enum') {
        const sel = document.createElement('select');
        for (const opt of p.options) {
          const o = document.createElement('option');
          o.value = opt; o.textContent = opt;
          if ((current[p.name] || p.default) === opt) o.selected = true;
          sel.appendChild(o);
        }
        sel.addEventListener('change', () => {
          onChange({ ...getCurrent(), [p.name]: sel.value });
          renderParamHost(host, algo, getCurrent, onChange);
        });
        cell.appendChild(sel);
      } else {
        const input = document.createElement('input');
        input.type = p.type === 'int' || p.type === 'float' ? 'number' : 'text';
        if (p.min !== undefined) input.min = p.min;
        if (p.max !== undefined) input.max = p.max;
        input.step = p.type === 'int' ? 1 : 'any';
        input.value = current[p.name] !== undefined ? current[p.name] : p.default;
        input.addEventListener('change', () => {
          const v = p.type === 'int' ? parseInt(input.value, 10) : parseFloat(input.value);
          onChange({ ...getCurrent(), [p.name]: v });
        });
        cell.appendChild(input);
      }
      grid.appendChild(cell);
    }
    host.appendChild(grid);
  }
```

- [ ] **Step 3: Syntax check**

Run: `node --check js/manifold/main.js`
Expected: no output.

- [ ] **Step 4: Run the full unit suite**

Run: `node --test 'test/manifold/*.test.js' 2>&1 | tail -6`
Expected: 0 fail.

- [ ] **Step 5: Confirm no banned styling**

Run: `grep -nE "—|<em>|<strong>|<b>|<i>|<mark>" js/manifold/main.js`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add js/manifold/main.js
git commit -m "manifold: two-column param layout with labels and info-icon tooltips"
```

---

### Task 5: Parameter and tooltip CSS

**Files:**
- Modify: `styles/manifold.css`

- [ ] **Step 1: Replace the old parameter styles**

In `styles/manifold.css`, find:
```css
.manifold .mf-algo-params { display: flex; flex-wrap: wrap; gap: 10px; font-size: 0.9rem; }
.manifold .mf-param { display: inline-flex; align-items: center; gap: 4px; }
.manifold .mf-param input { width: 5.5em; }
.manifold .mf-noparams { opacity: 0.6; font-style: italic; }
```

Replace with:
```css
.manifold .mf-algo-params { font-size: 0.9rem; }
.manifold .mf-param-grid { display: grid; grid-template-columns: auto 1fr; gap: 8px 12px; align-items: center; }
.manifold .mf-param-label { display: inline-flex; align-items: center; gap: 6px; }
.manifold .mf-param-name { color: rgba(255,255,255,0.85); }
.manifold .mf-param-info { display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px; border-radius: 50%; border: 1px solid rgba(255,255,255,0.45); color: rgba(255,255,255,0.6); font-size: 10px; line-height: 1; font-style: italic; cursor: help; flex: 0 0 auto; }
.manifold .mf-param-info:hover { border-color: #ff9f43; color: #ff9f43; }
.manifold .mf-param-control input, .manifold .mf-param-control select { background: rgba(0,0,0,0.3); color: #fff; border: 1px solid rgba(255,255,255,0.18); border-radius: 6px; padding: 3px 7px; font-size: 0.88rem; }
.manifold .mf-param-control input { width: 6em; }
.manifold .mf-param-control input:focus, .manifold .mf-param-control select:focus { outline: none; border-color: rgba(255,255,255,0.5); }
.manifold .mf-noparams { opacity: 0.6; font-style: italic; }
.mf-tooltip { position: fixed; max-width: 280px; background: #0a0c10; border: 1px solid rgba(255,255,255,0.3); color: #e8e8ec; font-size: 0.8rem; line-height: 1.4; padding: 9px 11px; border-radius: 8px; pointer-events: none; opacity: 0; transition: opacity 0.1s; z-index: 1000; box-shadow: 0 6px 24px rgba(0,0,0,0.5); }
.mf-tooltip-title { display: block; font-weight: 600; margin-bottom: 3px; }
.mf-tooltip-range { color: rgba(255,255,255,0.55); margin-top: 5px; }
```

Note: `.mf-param-info` uses `font-style: italic` on a `<span>` to style the letter "i"; this is CSS only, not an `<i>` tag.

- [ ] **Step 2: Confirm no em-dashes were introduced**

Run: `grep -n "—" styles/manifold.css`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add styles/manifold.css
git commit -m "manifold: style two-column params, info icon, and tooltip"
```

---

### Task 6: Browser smoke and final sweep

**Files:** none changed.

- [ ] **Step 1: Start a static server (if not already running)**

```bash
python3 -m http.server 8765 --bind 127.0.0.1 > /tmp/manifold-server.log 2>&1 &
sleep 1
```

- [ ] **Step 2: HTTP 200 check on changed/new files**

```bash
for f in pages/manifold.html js/manifold/main.js js/manifold/param_tooltip.js \
         js/manifold/algorithms/lle.js js/manifold/algorithms/kpca.js \
         js/manifold/datasets/synthetic_clusters.js styles/manifold.css; do
  curl -s -o /dev/null -w "%{http_code} $f\n" "http://127.0.0.1:8765/$f"
done
```
Expected: 200 on every line.

- [ ] **Step 3: Manual checklist**

Open `http://127.0.0.1:8765/pages/manifold.html` and confirm:
1. Algorithm params render as a two-column grid with readable labels (e.g. "Neighbors (k)", "Regularization") and an info icon per row.
2. Hovering an info icon shows a tooltip with the label, the definition, and the range line; it repositions near the right and bottom edges to stay visible.
3. Kernel PCA: switching kernel shows "RBF width (γ)" for rbf and "Polynomial degree (d)" for polynomial, never both; tooltips match.
4. Changing a value still recomputes the embedding.
5. Dataset params (select Helix, Toroidal helix, Spiral disk, Cylinder, Severed sphere, Hilbert curve, 3D Gaussian clusters) render the same way with their labels and tooltips.
6. PCA / MDS show the muted "No parameters"; CSV hides the dataset param row.
7. LLE default shows k = 12 on first load and after switching to LLE from another algorithm.

If any item fails, file the defect, fix it, and re-run this checklist.

- [ ] **Step 4: Stop the server**

```bash
ps -ef | grep -v grep | grep "http.server 8765" | awk '{print $2}' | xargs -r kill
```

- [ ] **Step 5: Final unit sweep + clean tree**

```bash
node --test 'test/manifold/*.test.js' 2>&1 | tail -6
git status --short
```
Expected: 0 fail; clean tree.

---

## Self-review

**Spec coverage:**
- Two-column layout B: Task 4 (renderParamHost grid) + Task 5 (CSS). Covered.
- Readable labels for all params: Tasks 1 and 2. Covered.
- Tooltips with definition + effect + range: Task 3 (helper) + Task 4 (wiring, range derived from min/max or options) + Task 5 (styles). Covered.
- Descriptor `label`/`desc` fields: Tasks 1 and 2. Covered.
- `dependsOn` filtering preserved: Task 4 keeps the filter and the enum re-render. Covered.
- Empty state (No parameters / hidden dataset row): Task 4 early return; CSV hiding is existing behavior in `updateSyntheticVisibility`, unchanged. Covered.
- LLE k default consistency to 12: Task 1 Step 4 (descriptor) plus existing run fallback and main.js already at 12; metadata test asserts 12. Covered.
- Out of scope (Samples/Noise/Seed, sliders, recompute behavior): untouched. Confirmed.

**Placeholder scan:** No TBD/TODO; every code step has complete code and exact commands.

**Type consistency:** `renderParamHost(host, algo, getCurrent, onChange)` signature unchanged. New descriptor fields `label`/`desc` read in Task 4 match those written in Tasks 1-2. `attachTooltip(iconEl, { label, desc, rangeText })` defined in Task 3 and called with exactly those keys in Task 4. CSS class names (`mf-param-grid`, `mf-param-label`, `mf-param-name`, `mf-param-info`, `mf-param-control`, `mf-tooltip`, `mf-tooltip-title`, `mf-tooltip-range`) match between Task 4 markup and Task 5 styles.
