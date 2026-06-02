# Manifold Phase 3 (Datasets) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 11 new synthetic datasets (curves, surfaces, spheres, clusters) to the manifold page, split the dataset code into focused files, thread per-dataset color through the worker, and render per-dataset extra parameters in the UI.

**Architecture:** The single `js/manifold/datasets.js` file is split into a `js/manifold/datasets/` directory with one file per dataset family plus a shared-helpers file and an `index.js` aggregator. The old `datasets.js` becomes a thin re-export shim so no importer changes. Datasets gain an optional `params` array (same shape as algorithm params) that the existing `renderParamHost` renders into a new dataset-param host. A `colors` array emitted by `clusters_3d` is threaded through the Web Worker boundary so algorithms can color points by cluster.

**Tech Stack:** Vanilla ES modules, d3 v7 (global), no build step. Node 22 `node --test` for unit tests. Deterministic RNG via `mulberry32` (in `js/manifold/rng.js`).

---

## Background the implementer needs

### Dataset contract

Each dataset is an object:

```javascript
{
  id: 'swiss_roll',
  label: 'Swiss roll',
  params: [ /* optional; same shape as algorithm params */ ],
  generate({ samples, noise, seed, /* ...extra params */, csvRows }) {
    // returns { X: Float64Array(3 * N), t: Float64Array(N), N }
    // may also return colors: string[] of length N
  }
}
```

- `X` is a flat array of N points, 3 floats each (x, y, z interleaved).
- `t` is the intrinsic coordinate used to rainbow-color points when no explicit `colors` array is present.
- `N` equals `samples` for synthetic datasets.
- Determinism: same `seed` must produce byte-identical `X` and `t`.

### Existing helpers

`js/manifold/rng.js` exports:
- `mulberry32(seed)` returns a deterministic `rand()` in [0, 1).
- `gaussian(rand)` returns a standard normal sample (consumes two `rand()` calls).

The current `js/manifold/datasets.js` defines two private helpers we will move into a shared file:
- `allocate(N)` returns `{ X: new Float64Array(N * 3), t: new Float64Array(N), N }`.
- `addNoise(X, noise, rand)` adds `noise * gaussian(rand)` to every entry of `X` when `noise > 0`.

### Param spec shape (for `renderParamHost`)

`renderParamHost` in `js/manifold/main.js` already handles these param descriptor types:
- `{ name, type: 'int', default, min, max }`
- `{ name, type: 'float', default, min, max }`
- `{ name, type: 'enum', options: [...], default }`
- optional `dependsOn: { otherParam: value }` to hide a param unless a sibling param matches.

It is called as `renderParamHost(host, objWithParamsArray, getCurrent, onChange)`.

### Worker boundary (why colors needs threading)

`js/manifold/state.js` `recompute()` runs `dataset.generate(...)` on the main thread, then `startWorkerRun` posts `X` and `t` buffers to the worker. `js/manifold/worker.js` rebuilds `dataset = { X, t }`. The `colors` array is currently dropped, so cluster colors never reach the algorithm step states. Task 7 threads it through.

---

## File structure

Create:
- `js/manifold/datasets/shared.js` - `allocate`, `addNoise`, `CLUSTER_PALETTE`, `fibonacciSpherePoints`.
- `js/manifold/datasets/synthetic_curves.js` - SWISS_ROLL, S_CURVE, HELIX, TREFOIL_KNOT, TOROIDAL_HELIX, SPIRAL_DISK.
- `js/manifold/datasets/synthetic_surfaces.js` - TWIN_PEAKS, SADDLE, CYLINDER, SEVERED_SPHERE, PUNCTURED_SPHERE, FULL_SPHERE.
- `js/manifold/datasets/synthetic_clusters.js` - CLUSTERS_3D.
- `js/manifold/datasets/csv_upload.js` - `parseCSV`, `projectToThreeViaPCA`, CSV_UPLOAD.
- `js/manifold/datasets/index.js` - aggregates and exports `DATASETS`, `DATASETS_BY_ID`, `parseCSV`.

Modify:
- `js/manifold/datasets.js` - becomes a re-export shim of `datasets/index.js`.
- `js/manifold/state.js` - thread `colors` into the worker postMessage.
- `js/manifold/worker.js` - rebuild `dataset.colors`.
- `js/manifold/main.js` - dataset-param host wiring + per-dataset default merge.
- `pages/manifold.html` - add the dataset-param host element.

Test:
- `test/manifold/datasets_curves.test.js`
- `test/manifold/datasets_surfaces.test.js`
- `test/manifold/datasets_clusters.test.js`
- `test/manifold/datasets_index.test.js`
- Update `test/manifold/datasets_synthetic.test.js` (the DATASETS-order assertion changes).

---

### Task 1: Shared dataset helpers

**Files:**
- Create: `js/manifold/datasets/shared.js`
- Test: `test/manifold/datasets_shared.test.js`

- [ ] **Step 1: Write the failing test**

```javascript
// test/manifold/datasets_shared.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { allocate, addNoise, CLUSTER_PALETTE, fibonacciSpherePoints } from '../../js/manifold/datasets/shared.js';
import { mulberry32 } from '../../js/manifold/rng.js';

test('allocate returns typed arrays of the right size', () => {
  const out = allocate(10);
  assert.equal(out.N, 10);
  assert.ok(out.X instanceof Float64Array);
  assert.ok(out.t instanceof Float64Array);
  assert.equal(out.X.length, 30);
  assert.equal(out.t.length, 10);
});

test('addNoise is a no-op when noise is zero', () => {
  const out = allocate(4);
  const before = out.X.slice();
  addNoise(out.X, 0, mulberry32(1));
  for (let i = 0; i < before.length; i++) assert.equal(out.X[i], before[i]);
});

test('addNoise perturbs entries when noise is positive', () => {
  const out = allocate(4);
  addNoise(out.X, 0.5, mulberry32(1));
  let nonzero = 0;
  for (let i = 0; i < out.X.length; i++) if (out.X[i] !== 0) nonzero++;
  assert.ok(nonzero > 0);
});

test('CLUSTER_PALETTE has 8 distinct color strings', () => {
  assert.equal(CLUSTER_PALETTE.length, 8);
  assert.equal(new Set(CLUSTER_PALETTE).size, 8);
  for (const c of CLUSTER_PALETTE) assert.match(c, /^#[0-9a-fA-F]{6}$/);
});

test('fibonacciSpherePoints returns count points on the requested radius', () => {
  const pts = fibonacciSpherePoints(5, 2);
  assert.equal(pts.length, 5);
  for (const p of pts) {
    const r = Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    assert.ok(Math.abs(r - 2) < 1e-9);
  }
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test test/manifold/datasets_shared.test.js`
Expected: FAIL with a module-not-found error for `datasets/shared.js`.

- [ ] **Step 3: Write the implementation**

```javascript
// js/manifold/datasets/shared.js
import { gaussian } from '../rng.js';

export function allocate(N) {
  return { X: new Float64Array(N * 3), t: new Float64Array(N), N };
}

export function addNoise(X, noise, rand) {
  if (noise <= 0) return;
  for (let i = 0; i < X.length; i++) X[i] += noise * gaussian(rand);
}

export const CLUSTER_PALETTE = [
  '#ff6b6b', '#4ecdc4', '#ffd93d', '#6a8cff',
  '#c77dff', '#ff9f43', '#54e36b', '#ff7eb6',
];

export function fibonacciSpherePoints(count, radius) {
  const pts = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const y = count === 1 ? 0 : 1 - (i / (count - 1)) * 2;
    const ring = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = golden * i;
    pts.push([radius * Math.cos(theta) * ring, radius * y, radius * Math.sin(theta) * ring]);
  }
  return pts;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test test/manifold/datasets_shared.test.js`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add js/manifold/datasets/shared.js test/manifold/datasets_shared.test.js
git commit -m "manifold: shared dataset helpers (allocate, addNoise, palette, fibonacci sphere)"
```

---

### Task 2: Synthetic curves module

**Files:**
- Create: `js/manifold/datasets/synthetic_curves.js`
- Test: `test/manifold/datasets_curves.test.js`

SWISS_ROLL and S_CURVE are moved here verbatim from `js/manifold/datasets.js` (they keep their exact current generate bodies). HELIX, TREFOIL_KNOT, TOROIDAL_HELIX, SPIRAL_DISK are new.

- [ ] **Step 1: Write the failing test**

```javascript
// test/manifold/datasets_curves.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  SWISS_ROLL, S_CURVE, HELIX, TREFOIL_KNOT, TOROIDAL_HELIX, SPIRAL_DISK,
} from '../../js/manifold/datasets/synthetic_curves.js';

const CURVES = [SWISS_ROLL, S_CURVE, HELIX, TREFOIL_KNOT, TOROIDAL_HELIX, SPIRAL_DISK];

test('every curve yields 3N flat X and length-N t as Float64Array', () => {
  for (const ds of CURVES) {
    const out = ds.generate({ samples: 20, noise: 0, seed: 1 });
    assert.equal(out.X.length, 60, `${ds.id} X length`);
    assert.equal(out.t.length, 20, `${ds.id} t length`);
    assert.ok(out.X instanceof Float64Array, `${ds.id} X type`);
    assert.ok(out.t instanceof Float64Array, `${ds.id} t type`);
  }
});

test('every curve is deterministic for a fixed seed', () => {
  for (const ds of CURVES) {
    const a = ds.generate({ samples: 15, noise: 0, seed: 3 });
    const b = ds.generate({ samples: 15, noise: 0, seed: 3 });
    for (let i = 0; i < a.X.length; i++) assert.equal(a.X[i], b.X[i], `${ds.id} det`);
  }
});

test('helix turns param changes the trace', () => {
  const a = HELIX.generate({ samples: 30, noise: 0, seed: 5, turns: 2 });
  const b = HELIX.generate({ samples: 30, noise: 0, seed: 5, turns: 6 });
  let diff = 0;
  for (let i = 0; i < a.X.length; i++) if (a.X[i] !== b.X[i]) diff++;
  assert.ok(diff > 0);
});

test('curves expose expected ids and labels', () => {
  assert.deepEqual(CURVES.map(d => d.id),
    ['swiss_roll', 's_curve', 'helix', 'trefoil_knot', 'toroidal_helix', 'spiral_disk']);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test test/manifold/datasets_curves.test.js`
Expected: FAIL with a module-not-found error.

- [ ] **Step 3: Write the implementation**

```javascript
// js/manifold/datasets/synthetic_curves.js
import { mulberry32, gaussian } from '../rng.js';
import { allocate, addNoise } from './shared.js';

export const SWISS_ROLL = {
  id: 'swiss_roll',
  label: 'Swiss roll',
  params: [],
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
  params: [],
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

export const HELIX = {
  id: 'helix',
  label: 'Helix',
  params: [{ name: 'turns', type: 'int', default: 3, min: 1, max: 8 }],
  generate({ samples, noise, seed, turns }) {
    const T = turns || 3;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const u = 2 * Math.PI * T * rand();
      out.X[i * 3 + 0] = Math.cos(u);
      out.X[i * 3 + 1] = Math.sin(u);
      out.X[i * 3 + 2] = u / (2 * Math.PI);
      out.t[i] = u;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const TREFOIL_KNOT = {
  id: 'trefoil_knot',
  label: 'Trefoil knot',
  params: [],
  generate({ samples, noise, seed }) {
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const u = 2 * Math.PI * rand();
      out.X[i * 3 + 0] = Math.sin(u) + 2 * Math.sin(2 * u);
      out.X[i * 3 + 1] = Math.cos(u) - 2 * Math.cos(2 * u);
      out.X[i * 3 + 2] = -Math.sin(3 * u);
      out.t[i] = u;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const TOROIDAL_HELIX = {
  id: 'toroidal_helix',
  label: 'Toroidal helix',
  params: [{ name: 'q', type: 'int', default: 7, min: 2, max: 15 }],
  generate({ samples, noise, seed, q }) {
    const Q = q || 7;
    const R = 2, r = 0.7;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const u = 2 * Math.PI * rand();
      const ring = R + r * Math.cos(Q * u);
      out.X[i * 3 + 0] = ring * Math.cos(u);
      out.X[i * 3 + 1] = ring * Math.sin(u);
      out.X[i * 3 + 2] = r * Math.sin(Q * u);
      out.t[i] = u;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const SPIRAL_DISK = {
  id: 'spiral_disk',
  label: 'Spiral disk',
  params: [{ name: 'turns', type: 'int', default: 3, min: 1, max: 6 }],
  generate({ samples, noise, seed, turns }) {
    const T = turns || 3;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    const thetaMax = 2 * Math.PI * T;
    const b = Math.log(40) / thetaMax;
    const r0 = 0.12;
    for (let i = 0; i < samples; i++) {
      const theta = thetaMax * rand();
      const r = r0 * Math.exp(b * theta);
      out.X[i * 3 + 0] = r * Math.cos(theta);
      out.X[i * 3 + 1] = r * Math.sin(theta);
      out.X[i * 3 + 2] = 0.02 * gaussian(rand);
      out.t[i] = theta;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test test/manifold/datasets_curves.test.js`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add js/manifold/datasets/synthetic_curves.js test/manifold/datasets_curves.test.js
git commit -m "manifold: synthetic curve datasets (helix, trefoil, toroidal helix, spiral disk)"
```

---

### Task 3: Synthetic surfaces module

**Files:**
- Create: `js/manifold/datasets/synthetic_surfaces.js`
- Test: `test/manifold/datasets_surfaces.test.js`

- [ ] **Step 1: Write the failing test**

```javascript
// test/manifold/datasets_surfaces.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  TWIN_PEAKS, SADDLE, CYLINDER, SEVERED_SPHERE, PUNCTURED_SPHERE, FULL_SPHERE,
} from '../../js/manifold/datasets/synthetic_surfaces.js';

const SURFACES = [TWIN_PEAKS, SADDLE, CYLINDER, SEVERED_SPHERE, PUNCTURED_SPHERE, FULL_SPHERE];

test('every surface yields 3N flat X and length-N t as Float64Array', () => {
  for (const ds of SURFACES) {
    const out = ds.generate({ samples: 20, noise: 0, seed: 1 });
    assert.equal(out.X.length, 60, `${ds.id} X length`);
    assert.equal(out.t.length, 20, `${ds.id} t length`);
    assert.ok(out.X instanceof Float64Array, `${ds.id} X type`);
    assert.ok(out.t instanceof Float64Array, `${ds.id} t type`);
  }
});

test('every surface is deterministic for a fixed seed', () => {
  for (const ds of SURFACES) {
    const a = ds.generate({ samples: 15, noise: 0, seed: 3 });
    const b = ds.generate({ samples: 15, noise: 0, seed: 3 });
    for (let i = 0; i < a.X.length; i++) assert.equal(a.X[i], b.X[i], `${ds.id} det`);
  }
});

test('full sphere points lie on the unit sphere', () => {
  const out = FULL_SPHERE.generate({ samples: 50, noise: 0, seed: 2 });
  for (let i = 0; i < out.N; i++) {
    const x = out.X[i * 3], y = out.X[i * 3 + 1], z = out.X[i * 3 + 2];
    assert.ok(Math.abs(Math.sqrt(x * x + y * y + z * z) - 1) < 1e-9);
  }
});

test('severed sphere removes the north polar cap', () => {
  const out = SEVERED_SPHERE.generate({ samples: 200, noise: 0, seed: 4, cap: 0.35 });
  const minZ = Math.cos(0.35 * Math.PI);
  for (let i = 0; i < out.N; i++) {
    assert.ok(out.X[i * 3 + 2] <= minZ + 1e-9, 'no point inside the removed cap');
  }
});

test('surfaces expose expected ids', () => {
  assert.deepEqual(SURFACES.map(d => d.id),
    ['twin_peaks', 'saddle', 'cylinder', 'severed_sphere', 'punctured_sphere', 'full_sphere']);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test test/manifold/datasets_surfaces.test.js`
Expected: FAIL with a module-not-found error.

- [ ] **Step 3: Write the implementation**

```javascript
// js/manifold/datasets/synthetic_surfaces.js
import { mulberry32 } from '../rng.js';
import { allocate, addNoise } from './shared.js';

function sampleSphere(rand) {
  const u = rand();
  const v = rand();
  const theta = 2 * Math.PI * v;
  const cosPhi = 1 - 2 * u;
  const phi = Math.acos(Math.max(-1, Math.min(1, cosPhi)));
  return { theta, phi };
}

function writeSpherePoint(out, i, theta, phi) {
  const s = Math.sin(phi);
  out.X[i * 3 + 0] = s * Math.cos(theta);
  out.X[i * 3 + 1] = s * Math.sin(theta);
  out.X[i * 3 + 2] = Math.cos(phi);
}

export const TWIN_PEAKS = {
  id: 'twin_peaks',
  label: 'Twin peaks',
  params: [],
  generate({ samples, noise, seed }) {
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const x = 2 * rand() - 1;
      const y = 2 * rand() - 1;
      const z = Math.exp(-(((x - 0.5) ** 2) + ((y - 0.5) ** 2)) / 0.3)
              + Math.exp(-(((x + 0.5) ** 2) + ((y + 0.5) ** 2)) / 0.3);
      out.X[i * 3 + 0] = x;
      out.X[i * 3 + 1] = y;
      out.X[i * 3 + 2] = z;
      out.t[i] = x;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const SADDLE = {
  id: 'saddle',
  label: 'Saddle',
  params: [],
  generate({ samples, noise, seed }) {
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const x = 2 * rand() - 1;
      const y = 2 * rand() - 1;
      out.X[i * 3 + 0] = x;
      out.X[i * 3 + 1] = y;
      out.X[i * 3 + 2] = x * x - y * y;
      out.t[i] = x;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const CYLINDER = {
  id: 'cylinder',
  label: 'Cylinder',
  params: [{ name: 'height', type: 'float', default: 2, min: 0.5, max: 5 }],
  generate({ samples, noise, seed, height }) {
    const H = height || 2;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const theta = 2 * Math.PI * rand();
      const h = H * rand();
      out.X[i * 3 + 0] = Math.cos(theta);
      out.X[i * 3 + 1] = Math.sin(theta);
      out.X[i * 3 + 2] = h;
      out.t[i] = theta;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const SEVERED_SPHERE = {
  id: 'severed_sphere',
  label: 'Severed sphere',
  params: [{ name: 'cap', type: 'float', default: 0.35, min: 0, max: 0.9 }],
  generate({ samples, noise, seed, cap }) {
    const C = cap === undefined ? 0.35 : cap;
    const minPhi = C * Math.PI;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    let i = 0;
    while (i < samples) {
      const { theta, phi } = sampleSphere(rand);
      if (phi < minPhi) continue;
      writeSpherePoint(out, i, theta, phi);
      out.t[i] = theta;
      i++;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const PUNCTURED_SPHERE = {
  id: 'punctured_sphere',
  label: 'Punctured sphere',
  params: [{ name: 'holeRadius', type: 'float', default: 0.4, min: 0, max: 1.5 }],
  generate({ samples, noise, seed, holeRadius }) {
    const HR = holeRadius === undefined ? 0.4 : holeRadius;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    let i = 0;
    while (i < samples) {
      const { theta, phi } = sampleSphere(rand);
      if (phi < HR) continue;
      writeSpherePoint(out, i, theta, phi);
      out.t[i] = theta;
      i++;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};

export const FULL_SPHERE = {
  id: 'full_sphere',
  label: 'Full sphere',
  params: [],
  generate({ samples, noise, seed }) {
    const rand = mulberry32(seed);
    const out = allocate(samples);
    for (let i = 0; i < samples; i++) {
      const { theta, phi } = sampleSphere(rand);
      writeSpherePoint(out, i, theta, phi);
      out.t[i] = theta;
    }
    addNoise(out.X, noise, rand);
    return out;
  },
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test test/manifold/datasets_surfaces.test.js`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add js/manifold/datasets/synthetic_surfaces.js test/manifold/datasets_surfaces.test.js
git commit -m "manifold: synthetic surface datasets (twin peaks, saddle, cylinder, spheres)"
```

---

### Task 4: Clusters module

**Files:**
- Create: `js/manifold/datasets/synthetic_clusters.js`
- Test: `test/manifold/datasets_clusters.test.js`

- [ ] **Step 1: Write the failing test**

```javascript
// test/manifold/datasets_clusters.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CLUSTERS_3D } from '../../js/manifold/datasets/synthetic_clusters.js';
import { CLUSTER_PALETTE } from '../../js/manifold/datasets/shared.js';

test('clusters_3d yields 3N flat X and length-N t', () => {
  const out = CLUSTERS_3D.generate({ samples: 20, noise: 0, seed: 1, clusters: 5, sep: 2 });
  assert.equal(out.X.length, 60);
  assert.equal(out.t.length, 20);
  assert.ok(out.X instanceof Float64Array);
});

test('clusters_3d is deterministic for a fixed seed', () => {
  const a = CLUSTERS_3D.generate({ samples: 30, noise: 0, seed: 9, clusters: 4, sep: 2 });
  const b = CLUSTERS_3D.generate({ samples: 30, noise: 0, seed: 9, clusters: 4, sep: 2 });
  for (let i = 0; i < a.X.length; i++) assert.equal(a.X[i], b.X[i]);
});

test('clusters_3d emits a colors array drawn from the palette', () => {
  const out = CLUSTERS_3D.generate({ samples: 30, noise: 0, seed: 1, clusters: 3, sep: 2 });
  assert.ok(Array.isArray(out.colors));
  assert.equal(out.colors.length, 30);
  for (const c of out.colors) assert.ok(CLUSTER_PALETTE.includes(c));
});

test('clusters_3d partitions points roughly evenly across clusters', () => {
  const out = CLUSTERS_3D.generate({ samples: 30, noise: 0, seed: 1, clusters: 3, sep: 2 });
  const counts = [0, 0, 0];
  for (let i = 0; i < out.N; i++) counts[out.t[i]]++;
  for (const c of counts) assert.equal(c, 10);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test test/manifold/datasets_clusters.test.js`
Expected: FAIL with a module-not-found error.

- [ ] **Step 3: Write the implementation**

```javascript
// js/manifold/datasets/synthetic_clusters.js
import { mulberry32, gaussian } from '../rng.js';
import { allocate, CLUSTER_PALETTE, fibonacciSpherePoints } from './shared.js';

export const CLUSTERS_3D = {
  id: 'clusters_3d',
  label: '3D Gaussian clusters',
  params: [
    { name: 'clusters', type: 'int', default: 5, min: 2, max: 8 },
    { name: 'sep', type: 'float', default: 2, min: 0.5, max: 5 },
  ],
  generate({ samples, noise, seed, clusters, sep }) {
    const K = clusters || 5;
    const S = sep || 2;
    const rand = mulberry32(seed);
    const out = allocate(samples);
    const centers = fibonacciSpherePoints(K, S);
    const colors = new Array(samples);
    const spread = 0.25;
    for (let i = 0; i < samples; i++) {
      const c = i % K;
      const ctr = centers[c];
      out.X[i * 3 + 0] = ctr[0] + spread * gaussian(rand);
      out.X[i * 3 + 1] = ctr[1] + spread * gaussian(rand);
      out.X[i * 3 + 2] = ctr[2] + spread * gaussian(rand);
      out.t[i] = c;
      colors[i] = CLUSTER_PALETTE[c % CLUSTER_PALETTE.length];
    }
    if (noise > 0) {
      for (let j = 0; j < out.X.length; j++) out.X[j] += noise * gaussian(rand);
    }
    out.colors = colors;
    return out;
  },
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test test/manifold/datasets_clusters.test.js`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add js/manifold/datasets/synthetic_clusters.js test/manifold/datasets_clusters.test.js
git commit -m "manifold: 3D Gaussian clusters dataset with per-cluster colors"
```

---

### Task 5: CSV upload module

**Files:**
- Create: `js/manifold/datasets/csv_upload.js`
- Test: reuse existing `test/manifold/datasets_csv.test.js` (it imports `parseCSV` from `datasets.js`; the shim in Task 6 keeps that working). No new test file.

This task moves `parseCSV`, `projectToThreeViaPCA`, and `CSV_UPLOAD` out of `datasets.js` into a dedicated module verbatim.

- [ ] **Step 1: Write the implementation**

```javascript
// js/manifold/datasets/csv_upload.js
import { jacobiEigSym } from '../linalg.js';
import { allocate } from './shared.js';

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
  const C = [];
  for (let i = 0; i < d; i++) C.push(new Array(d).fill(0));
  for (const r of centered) {
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += r[i] * r[j];
  }
  const denom = Math.max(1, N - 1);
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] /= denom;
  const { vectors } = jacobiEigSym(C);
  const out = allocate(N);
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < 3; k++) {
      let s = 0;
      const vk = vectors[k];
      for (let j = 0; j < d; j++) s += centered[i][j] * vk[j];
      out.X[i * 3 + k] = s;
    }
    out.t[i] = i / Math.max(1, N - 1);
  }
  return out;
}

export const CSV_UPLOAD = {
  id: 'csv',
  label: 'Upload CSV...',
  params: [],
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
```

- [ ] **Step 2: Syntax check**

Run: `node --check js/manifold/datasets/csv_upload.js`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add js/manifold/datasets/csv_upload.js
git commit -m "manifold: move CSV upload and PCA projection into datasets/csv_upload.js"
```

---

### Task 6: Aggregator index and datasets.js shim

**Files:**
- Create: `js/manifold/datasets/index.js`
- Modify: `js/manifold/datasets.js` (replace whole file with shim)
- Modify: `test/manifold/datasets_synthetic.test.js` (update the DATASETS-order assertion)
- Test: `test/manifold/datasets_index.test.js`

- [ ] **Step 1: Write the failing test**

```javascript
// test/manifold/datasets_index.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DATASETS, DATASETS_BY_ID, parseCSV } from '../../js/manifold/datasets/index.js';

const EXPECTED_ORDER = [
  'swiss_roll', 's_curve', 'helix', 'trefoil_knot', 'toroidal_helix', 'spiral_disk',
  'twin_peaks', 'saddle', 'cylinder', 'severed_sphere', 'punctured_sphere', 'full_sphere',
  'clusters_3d', 'csv',
];

test('DATASETS lists all 14 datasets in order', () => {
  assert.deepEqual(DATASETS.map(d => d.id), EXPECTED_ORDER);
});

test('DATASETS_BY_ID maps every id', () => {
  for (const id of EXPECTED_ORDER) assert.ok(DATASETS_BY_ID[id], `missing ${id}`);
});

test('parseCSV is re-exported from the aggregator', () => {
  assert.equal(typeof parseCSV, 'function');
  assert.deepEqual(parseCSV('1,2,3\n4,5,6\n'), [[1, 2, 3], [4, 5, 6]]);
});

test('the datasets.js shim re-exports the same DATASETS', async () => {
  const shim = await import('../../js/manifold/datasets.js');
  assert.deepEqual(shim.DATASETS.map(d => d.id), EXPECTED_ORDER);
  assert.equal(typeof shim.parseCSV, 'function');
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test test/manifold/datasets_index.test.js`
Expected: FAIL with a module-not-found error for `datasets/index.js`.

- [ ] **Step 3: Write the aggregator**

```javascript
// js/manifold/datasets/index.js
import { SWISS_ROLL, S_CURVE, HELIX, TREFOIL_KNOT, TOROIDAL_HELIX, SPIRAL_DISK } from './synthetic_curves.js';
import { TWIN_PEAKS, SADDLE, CYLINDER, SEVERED_SPHERE, PUNCTURED_SPHERE, FULL_SPHERE } from './synthetic_surfaces.js';
import { CLUSTERS_3D } from './synthetic_clusters.js';
import { CSV_UPLOAD, parseCSV } from './csv_upload.js';

export const DATASETS = [
  SWISS_ROLL, S_CURVE, HELIX, TREFOIL_KNOT, TOROIDAL_HELIX, SPIRAL_DISK,
  TWIN_PEAKS, SADDLE, CYLINDER, SEVERED_SPHERE, PUNCTURED_SPHERE, FULL_SPHERE,
  CLUSTERS_3D, CSV_UPLOAD,
];
export const DATASETS_BY_ID = Object.fromEntries(DATASETS.map(d => [d.id, d]));
export { parseCSV };
```

- [ ] **Step 4: Replace `js/manifold/datasets.js` with a shim**

Replace the entire contents of `js/manifold/datasets.js` with:

```javascript
// Re-export shim. The dataset implementations now live in js/manifold/datasets/.
export { DATASETS, DATASETS_BY_ID, parseCSV } from './datasets/index.js';
export { SWISS_ROLL, S_CURVE, HELIX, TREFOIL_KNOT, TOROIDAL_HELIX, SPIRAL_DISK } from './datasets/synthetic_curves.js';
export { TWIN_PEAKS, SADDLE, CYLINDER, SEVERED_SPHERE, PUNCTURED_SPHERE, FULL_SPHERE } from './datasets/synthetic_surfaces.js';
export { CLUSTERS_3D } from './datasets/synthetic_clusters.js';
export { CSV_UPLOAD } from './datasets/csv_upload.js';
```

- [ ] **Step 5: Update the stale DATASETS-order assertion in the existing synthetic test**

In `test/manifold/datasets_synthetic.test.js`, find:

```javascript
test('DATASETS exposes swiss_roll, s_curve, and csv ids in order', () => {
  assert.deepEqual(DATASETS.map(d => d.id), ['swiss_roll', 's_curve', 'csv']);
  assert.equal(DATASETS_BY_ID.swiss_roll.label, 'Swiss roll');
});
```

Replace with:

```javascript
test('DATASETS starts with swiss_roll and s_curve and ends with csv', () => {
  const ids = DATASETS.map(d => d.id);
  assert.equal(ids[0], 'swiss_roll');
  assert.equal(ids[1], 's_curve');
  assert.equal(ids[ids.length - 1], 'csv');
  assert.equal(DATASETS_BY_ID.swiss_roll.label, 'Swiss roll');
});
```

- [ ] **Step 6: Run the full dataset test suite**

Run: `node --test 'test/manifold/datasets*.test.js' 2>&1 | tail -5`
Expected: all dataset tests pass, 0 fail.

- [ ] **Step 7: Commit**

```bash
git add js/manifold/datasets/index.js js/manifold/datasets.js test/manifold/datasets_index.test.js test/manifold/datasets_synthetic.test.js
git commit -m "manifold: aggregate datasets via index.js and convert datasets.js to a shim"
```

---

### Task 7: Thread colors through the worker

**Files:**
- Modify: `js/manifold/state.js`
- Modify: `js/manifold/worker.js`

The `colors` array produced by `clusters_3d` must survive the worker boundary so algorithm step states can color points by cluster.

- [ ] **Step 1: Add colors to the worker postMessage in `state.js`**

In `js/manifold/state.js`, find the `w.postMessage({ type: 'run', ... })` call inside `startWorkerRun`:

```javascript
    runHandlers.set(runId, { onStep });
    w.postMessage({
      type: 'run', runId, algoId,
      X: data.X.buffer.slice(0),
      t: data.t.buffer.slice(0),
      params,
    });
```

Replace with:

```javascript
    runHandlers.set(runId, { onStep });
    w.postMessage({
      type: 'run', runId, algoId,
      X: data.X.buffer.slice(0),
      t: data.t.buffer.slice(0),
      colors: data.colors || null,
      params,
    });
```

- [ ] **Step 2: Rebuild dataset.colors in `worker.js`**

In `js/manifold/worker.js`, find:

```javascript
    const dataset = {
      X: new Float64Array(msg.X),
      t: new Float64Array(msg.t),
    };
```

Replace with:

```javascript
    const dataset = {
      X: new Float64Array(msg.X),
      t: new Float64Array(msg.t),
      colors: msg.colors || null,
    };
```

- [ ] **Step 3: Syntax check**

Run: `node --check js/manifold/state.js && node --check js/manifold/worker.js`
Expected: no output.

- [ ] **Step 4: Run the full unit suite to confirm nothing regressed**

Run: `node --test 'test/manifold/*.test.js' 2>&1 | tail -5`
Expected: 0 fail.

- [ ] **Step 5: Commit**

```bash
git add js/manifold/state.js js/manifold/worker.js
git commit -m "manifold: thread dataset colors through the worker boundary"
```

---

### Task 8: Dataset parameter UI

**Files:**
- Modify: `pages/manifold.html`
- Modify: `js/manifold/main.js`

Datasets with a non-empty `params` array (helix, toroidal_helix, spiral_disk, cylinder, severed_sphere, punctured_sphere, clusters_3d) render their extra knobs in a dedicated host using the existing `renderParamHost`. Switching datasets seeds the new params with their defaults.

- [ ] **Step 1: Add the dataset-param host to the HTML**

In `pages/manifold.html`, find:

```html
        <input id="mfCsvInput" type="file" accept=".csv,text/csv" style="display:none" />
        <div id="mfCsvLabel" class="mf-csv-name"></div>
```

Replace with:

```html
        <div id="mfDatasetParams" class="mf-algo-params"></div>
        <input id="mfCsvInput" type="file" accept=".csv,text/csv" style="display:none" />
        <div id="mfCsvLabel" class="mf-csv-name"></div>
```

- [ ] **Step 2: Grab the host element in `main.js`**

In `js/manifold/main.js`, find:

```javascript
  const csvInput = $('mfCsvInput');
  const csvLabel = $('mfCsvLabel');
```

Replace with:

```javascript
  const csvInput = $('mfCsvInput');
  const csvLabel = $('mfCsvLabel');
  const datasetParamsHost = $('mfDatasetParams');
```

- [ ] **Step 3: Add a renderDatasetParams function**

In `js/manifold/main.js`, find the `rebindParamHosts` function:

```javascript
  function rebindParamHosts() {
    renderParamHost(leftParamsHost, ALGORITHMS_BY_ID[store.state.leftAlgoId], () => store.state.leftAlgoParams,
      (next) => store.set({ leftAlgoParams: next }));
    renderParamHost(rightParamsHost, ALGORITHMS_BY_ID[store.state.rightAlgoId], () => store.state.rightAlgoParams,
      (next) => store.set({ rightAlgoParams: next }));
  }
  rebindParamHosts();
```

Replace with:

```javascript
  function rebindParamHosts() {
    renderParamHost(leftParamsHost, ALGORITHMS_BY_ID[store.state.leftAlgoId], () => store.state.leftAlgoParams,
      (next) => store.set({ leftAlgoParams: next }));
    renderParamHost(rightParamsHost, ALGORITHMS_BY_ID[store.state.rightAlgoId], () => store.state.rightAlgoParams,
      (next) => store.set({ rightAlgoParams: next }));
  }
  rebindParamHosts();

  function renderDatasetParams() {
    const ds = DATASETS_BY_ID[store.state.datasetId];
    if (!ds || !ds.params || ds.params.length === 0) {
      datasetParamsHost.innerHTML = '';
      return;
    }
    renderParamHost(datasetParamsHost, ds, () => store.state.datasetParams,
      (next) => store.set({ datasetParams: next }));
  }
  renderDatasetParams();
```

This requires `DATASETS_BY_ID`. Confirm the import at the top of `main.js` includes it. Find:

```javascript
import { DATASETS, parseCSV } from './datasets.js';
```

Replace with:

```javascript
import { DATASETS, DATASETS_BY_ID, parseCSV } from './datasets.js';
```

- [ ] **Step 4: Seed per-dataset defaults on dataset change**

In `js/manifold/main.js`, find the dataset select change handler:

```javascript
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
```

Replace with:

```javascript
  datasetSelect.addEventListener('change', () => {
    const id = datasetSelect.value;
    if (id === 'csv') {
      csvInput.value = '';
      csvInput.click();
      return;
    }
    const ds = DATASETS_BY_ID[id];
    const dp = {
      samples: store.state.datasetParams.samples,
      noise: store.state.datasetParams.noise,
      seed: store.state.datasetParams.seed,
    };
    if (ds && ds.params) for (const p of ds.params) dp[p.name] = p.default;
    store.set({ datasetId: id, csvRows: null, csvFileName: '', datasetParams: dp });
    updateSyntheticVisibility();
    renderDatasetParams();
  });
```

- [ ] **Step 5: Hide dataset params for CSV**

In `js/manifold/main.js`, find the `updateSyntheticVisibility` function:

```javascript
  function updateSyntheticVisibility() {
    const isCsv = store.state.datasetId === 'csv';
    samplesControl.style.display = isCsv ? 'none' : '';
    noiseControl.style.display = isCsv ? 'none' : '';
    seedControl.style.display = isCsv ? 'none' : '';
    csvLabel.textContent = isCsv ? (store.state.csvFileName ? `Loaded: ${store.state.csvFileName} (${store.state.csvRows ? store.state.csvRows.length : 0} rows)` : '') : '';
  }
```

Replace with:

```javascript
  function updateSyntheticVisibility() {
    const isCsv = store.state.datasetId === 'csv';
    samplesControl.style.display = isCsv ? 'none' : '';
    noiseControl.style.display = isCsv ? 'none' : '';
    seedControl.style.display = isCsv ? 'none' : '';
    datasetParamsHost.style.display = isCsv ? 'none' : '';
    csvLabel.textContent = isCsv ? (store.state.csvFileName ? `Loaded: ${store.state.csvFileName} (${store.state.csvRows ? store.state.csvRows.length : 0} rows)` : '') : '';
  }
```

- [ ] **Step 6: Syntax check**

Run: `node --check js/manifold/main.js`
Expected: no output.

- [ ] **Step 7: Run the full unit suite**

Run: `node --test 'test/manifold/*.test.js' 2>&1 | tail -5`
Expected: 0 fail.

- [ ] **Step 8: Commit**

```bash
git add pages/manifold.html js/manifold/main.js
git commit -m "manifold: render per-dataset parameters and seed defaults on dataset switch"
```

---

### Task 9: Browser smoke verification

**Files:** none changed.

- [ ] **Step 1: Start a static server**

```bash
python3 -m http.server 8765 --bind 127.0.0.1 > /tmp/manifold-server.log 2>&1 &
echo "pid=$!"
sleep 1
```

- [ ] **Step 2: HTTP 200 check on every new file plus the page**

```bash
for f in pages/manifold.html \
         js/manifold/datasets.js \
         js/manifold/datasets/index.js \
         js/manifold/datasets/shared.js \
         js/manifold/datasets/synthetic_curves.js \
         js/manifold/datasets/synthetic_surfaces.js \
         js/manifold/datasets/synthetic_clusters.js \
         js/manifold/datasets/csv_upload.js \
         js/manifold/main.js \
         js/manifold/worker.js; do
  curl -s -o /dev/null -w "%{http_code} $f\n" "http://127.0.0.1:8765/$f"
done
```

Expected: 200 on every line.

- [ ] **Step 3: Manual checklist (open the page)**

Open: `http://127.0.0.1:8765/pages/manifold.html`

1. Confirm the dataset dropdown lists all 14 datasets in order: Swiss roll, S-curve, Helix, Trefoil knot, Toroidal helix, Spiral disk, Twin peaks, Saddle, Cylinder, Severed sphere, Punctured sphere, Full sphere, 3D Gaussian clusters, Upload CSV.
2. Select Helix: confirm a `turns` integer input appears in the dataset-param row. Change it and confirm step 0 redraws.
3. Select Toroidal helix (`q`), Spiral disk (`turns`), Cylinder (`height`), Severed sphere (`cap`), Punctured sphere (`holeRadius`): confirm each shows only its own params.
4. Select 3D Gaussian clusters: confirm `clusters` and `sep` inputs appear, and the step-0 cloud is colored by cluster (discrete colors, not the rainbow), and that the embedding at step 6 preserves those cluster colors.
5. Select Upload CSV: confirm the dataset-param row, samples, noise, and seed controls all hide and the file picker opens.
6. Switch back to a synthetic dataset: confirm samples/noise/seed reappear and the dataset-param row returns.

If any item fails, file the specific defect, fix it, and re-run this checklist.

- [ ] **Step 4: Stop the server**

```bash
ps -ef | grep -v grep | grep "http.server 8765" | awk '{print $2}' | xargs -r kill
```

---

### Task 10: Final test sweep

**Files:** none changed.

- [ ] **Step 1: Run all unit tests**

```bash
node --test 'test/manifold/*.test.js' 2>&1 | tail -5
```

Expected: 0 fail. The suite count grows by the new dataset tests.

- [ ] **Step 2: Confirm clean working tree**

```bash
git status --short
```

Expected: no output.

---

## Self-review

**Spec coverage (against phase 2 design Surface 4):**
- All 11 new datasets present: helix, trefoil_knot, toroidal_helix, spiral_disk (Task 2); twin_peaks, saddle, cylinder, severed_sphere, punctured_sphere, full_sphere (Task 3); clusters_3d (Task 4). Covered.
- File split into synthetic_curves / synthetic_surfaces / synthetic_clusters / csv_upload / index, with datasets.js as a shim. Covered (Tasks 2-6).
- clusters_3d emits a colors array from an 8-color palette, recycled if more clusters requested (`c % CLUSTER_PALETTE.length`). Covered (Task 4). Colors threaded to the worker so renderers can use them. Covered (Task 7).
- Dataset extra params surface in the UI. Covered (Task 8). The spec deferred dataset-param UI as an open decision; this plan reuses `renderParamHost`.
- Tests per the spec test plan (X.length === 60, t.length === 20 at samples=20, determinism, clusters colors + even partition). Covered (Tasks 2-4).

**Open decisions resolved:**
- datasets.js stays a shim (decision 1).
- clusters_3d centers placed via a deterministic Fibonacci sphere (decision 4).

**Placeholder scan:** No TBD/TODO; every code step shows complete code.

**Type consistency:** Dataset objects use `{ id, label, params, generate }`. `params` descriptors match the `renderParamHost` contract (`type` int/float/enum, min/max, default). `DATASETS_BY_ID` keys equal each dataset `id`. `colors` is `string[]` of length N, read in `worker.js` as `msg.colors`. Consistent across tasks.

**Note on scope:** This plan does not add t-SNE/UMAP, Mobius/Klein datasets, or a coloring-source UI - all explicitly out of scope in the phase 2 design.
