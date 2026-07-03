# Gradient Descent Batch Size and Optimizer Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the gradient descent sandbox so it can compare GD/SGD/MBGD batch sizes (pinning one optimizer) and SGD/Momentum/AdaGrad/RMSProp/Adam optimizers (pinning one batch size), all still rendered as dots descending the same analytic loss surface.

**Architecture:** Generalize the sandbox's hardcoded 3-optimizer `OPTS` constant into two config arrays (`OPTIMIZERS`, `BATCH_MODES`) and a computed "active lines" list driven by a mode toggle plus per-mode pinned selector/checkboxes. Every rendering loop (Three.js trajectories, D3 contour, legend) already iterates over "the current set of lines" and gets repointed at that computed list instead of the old constant. The mode toggle reuses the site's shared `tabs.js` controller instead of a bespoke segmented control.

**Tech Stack:** Vanilla JS ES module, Three.js r0.169 (via import map), D3 v7, MathJax v3. No build step or test framework — static site.

## Global Constraints

- No em-dashes anywhere: prose, code, comments, docs.
- No emphasis markup (bold/italic) in generated page copy.
- Long-form prose in `<p>` tags: one full sentence per source line (matches this file's existing convention).
- Reuse shared components (`.tabs`/`.tab` + `js/tabs.js`, `.check`, `.control-row`) from `styles/components.css` / `styles/article-ui.css` rather than inventing new segmented-control or checkbox styles. `components.css:249-250` states tabs are "the single 'pick one of N' control ... there is no separate segmented control."
- `js/tabs.js` toggles a plain `active` class (not `is-active`) and shows/hides elements via `data-panel` attributes matching a button's `data-tab`; `styles/article-ui.css:162` is the alias that makes `.tab.active` render correctly, and `article-ui.css` is already linked from `pages/gradient-descent.html`.
- No new dependencies, build tooling, or test framework. Verification is manual, in-browser, matching this repo's established pattern (e.g. `docs/superpowers/plans/2026-06-17-fourier-migration.md`).
- Design reference: `docs/superpowers/specs/2026-07-02-gradient-descent-batch-optimizer-design.md`.

---

### Task 1: Generalize the optimizer/rendering core in `js/gradient-descent.js`

**Files:**
- Modify: `js/gradient-descent.js` (entire file replaced; see below)

**Interfaces:**
- Consumes: nothing new (self-contained rewrite of existing module).
- Produces (for Task 2 to wire up): `OPTIMIZERS` (array of `{key, label, color}`, keys `sgd|momentum|adagrad|rmsprop|adam`), `BATCH_MODES` (array of `{key, label, n, color}`, keys `full|mini|stochastic`), module-level mutable state `compareMode` (`'batch'|'optimizer'`), `pinnedOptimizerKey`, `pinnedBatchKey`, `checkedOptimizerKeys` (a `Set`), and `applyLineChange()` (recomputes `lines`, rebuilds the contour SVG, calls `resetAll()`).

This task lands as one atomic replacement because the config, step functions, noise injection, and every render loop that iterates over "the optimizers" are tightly coupled — a partial change leaves the module in a broken state where `doStep()` calls a step function with the old function-based gradient argument that the new step functions no longer accept.

- [ ] **Step 1: Sanity-check the new math in isolation before it goes in the real file.**

Run this with plain `node` (no imports needed — it's a standalone reimplementation of just the new formulas, to catch sign/formula errors before they're buried in browser-only code):

```bash
node -e '
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function noisyGrad([gx, gy], batchMode) {
  if (batchMode.n === Infinity) return [gx, gy];
  const sigma = 0.35 / Math.sqrt(batchMode.n);
  const norm = Math.hypot(gx, gy);
  return [gx + sigma * norm * randn(), gy + sigma * norm * randn()];
}
function stepAdaGrad(s, [gx, gy], lr, eps = 1e-8) {
  const gxSq = (s.gxSq||0) + gx*gx, gySq = (s.gySq||0) + gy*gy;
  return { x: s.x - lr*gx/(Math.sqrt(gxSq)+eps), y: s.y - lr*gy/(Math.sqrt(gySq)+eps), gxSq, gySq };
}
function stepRMSProp(s, [gx, gy], lr, beta = 0.9, eps = 1e-8) {
  const ex = beta*(s.ex||0) + (1-beta)*gx*gx, ey = beta*(s.ey||0) + (1-beta)*gy*gy;
  return { x: s.x - lr*gx/(Math.sqrt(ex)+eps), y: s.y - lr*gy/(Math.sqrt(ey)+eps), ex, ey };
}

// Full-batch must be noise-free.
const full = noisyGrad([3, -2], { n: Infinity });
console.assert(full[0] === 3 && full[1] === -2, "FAIL: full-batch must not add noise");

// Stochastic noise should visibly perturb the gradient over many trials.
let maxDelta = 0;
for (let i = 0; i < 200; i++) {
  const [nx] = noisyGrad([3, -2], { n: 1 });
  maxDelta = Math.max(maxDelta, Math.abs(nx - 3));
}
console.assert(maxDelta > 0.1, "FAIL: stochastic noise looks too small: " + maxDelta);

// AdaGrad: with a constant nonzero gradient, the accumulator only grows, so step
// magnitude on repeated identical gradients must shrink over time.
let s = { x: 0, y: 0 };
const steps = [];
for (let i = 0; i < 5; i++) { const prev = s.x; s = stepAdaGrad(s, [1, 1], 0.1); steps.push(Math.abs(s.x - prev)); }
console.assert(steps[4] < steps[0], "FAIL: AdaGrad step size should shrink under a constant gradient: " + steps);

// RMSProp: after the accumulator warms up, step size on a constant gradient stabilizes
// (does not keep shrinking every step the way AdaGrad does).
s = { x: 0, y: 0 };
const rsteps = [];
for (let i = 0; i < 20; i++) { const prev = s.x; s = stepRMSProp(s, [1, 1], 0.1); rsteps.push(Math.abs(s.x - prev)); }
console.assert(Math.abs(rsteps[19] - rsteps[10]) < 0.01, "FAIL: RMSProp step size should stabilize: " + rsteps.slice(10));

console.log("ALL PASS");
'
```

Expected output: `ALL PASS` (no `console.assert` failure lines above it). If any assertion fails, fix the formula before continuing to Step 2.

- [ ] **Step 2: Replace the entire contents of `js/gradient-descent.js`**

```js
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---- Loss functions --------------------------------------------------------

const SURF_H = 2.2;  // world height scale for the 3D surface

const FUNCTIONS = {
  elongated: {
    label: 'Elongated bowl',
    domain: 3,
    f:    (x, y) => x*x + 8*y*y,
    grad: (x, y) => [2*x, 16*y],
    zNorm: v => Math.min(v / 80, 1),
    start: { x: -2.4, y: 1.0 },
  },
  rosenbrock: {
    label: 'Rosenbrock',
    domain: 2,
    f:    (x, y) => (1 - x)**2 + 100*(y - x*x)**2,
    grad: (x, y) => [-2*(1-x) - 400*x*(y - x*x), 200*(y - x*x)],
    zNorm: v => Math.min(v / 3600, 1),
    start: { x: -1.4, y: 1.6 },
  },
  bowl: {
    label: 'Quadratic bowl',
    domain: 3,
    f:    (x, y) => x*x + y*y,
    grad: (x, y) => [2*x, 2*y],
    zNorm: v => Math.min(v / 18, 1),
    start: { x: -2.4, y: 2.0 },
  },
  saddle: {
    label: 'Saddle + valleys',
    domain: 2.5,
    f:    (x, y) => x*x - y*y + 0.12*(x**4 + y**4),
    grad: (x, y) => [2*x + 0.48*x**3, -2*y + 0.48*y**3],
    zNorm: v => (v + 10) / 20,
    start: { x: 0.4, y: 0.4 },
  },
};

// ---- Optimizers --------------------------------------------------------

const OPTIMIZERS = [
  { key: 'sgd',      label: 'SGD',      color: '#74b9ff' },
  { key: 'momentum', label: 'Momentum', color: '#fd79a8' },
  { key: 'adagrad',  label: 'AdaGrad',  color: '#a29bfe' },
  { key: 'rmsprop',  label: 'RMSProp',  color: '#55efc4' },
  { key: 'adam',     label: 'Adam',     color: '#00cec9' },
];

// ---- Batch modes -------------------------------------------------------
// n is the simulated number of examples averaged per gradient estimate. These
// surfaces are analytic functions with no real dataset to subsample, so batch
// size is simulated: full-batch uses the exact analytic gradient (no noise),
// mini-batch and stochastic add Gaussian noise scaled by 1/sqrt(n).

const BATCH_MODES = [
  { key: 'full',       label: 'Full-batch',  n: Infinity, color: '#74b9ff' },
  { key: 'mini',       label: 'Mini-batch',  n: 16,       color: '#ffeaa7' },
  { key: 'stochastic', label: 'Stochastic',  n: 1,        color: '#ff7675' },
];

const BASE_NOISE = 0.35;

function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function noisyGrad([gx, gy], batchMode) {
  if (batchMode.n === Infinity) return [gx, gy];
  const sigma = BASE_NOISE / Math.sqrt(batchMode.n);
  const norm = Math.hypot(gx, gy);
  return [gx + sigma * norm * randn(), gy + sigma * norm * randn()];
}

function stepSGD(s, gradVec, lr) {
  const [gx, gy] = gradVec;
  return { x: s.x - lr*gx, y: s.y - lr*gy };
}

function stepMomentum(s, gradVec, lr, beta = 0.9) {
  const [gx, gy] = gradVec;
  const vx = beta*(s.vx||0) - lr*gx;
  const vy = beta*(s.vy||0) - lr*gy;
  return { x: s.x + vx, y: s.y + vy, vx, vy };
}

function stepAdaGrad(s, gradVec, lr, eps = 1e-8) {
  const [gx, gy] = gradVec;
  const gxSq = (s.gxSq||0) + gx*gx;
  const gySq = (s.gySq||0) + gy*gy;
  return {
    x: s.x - lr*gx/(Math.sqrt(gxSq)+eps),
    y: s.y - lr*gy/(Math.sqrt(gySq)+eps),
    gxSq, gySq,
  };
}

function stepRMSProp(s, gradVec, lr, beta = 0.9, eps = 1e-8) {
  const [gx, gy] = gradVec;
  const ex = beta*(s.ex||0) + (1-beta)*gx*gx;
  const ey = beta*(s.ey||0) + (1-beta)*gy*gy;
  return {
    x: s.x - lr*gx/(Math.sqrt(ex)+eps),
    y: s.y - lr*gy/(Math.sqrt(ey)+eps),
    ex, ey,
  };
}

function stepAdam(s, gradVec, lr, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
  const t = (s.t||0) + 1;
  const [gx, gy] = gradVec;
  const mx = b1*(s.mx||0) + (1-b1)*gx;
  const my = b1*(s.my||0) + (1-b1)*gy;
  const vx = b2*(s.vx||0) + (1-b2)*gx*gx;
  const vy = b2*(s.vy||0) + (1-b2)*gy*gy;
  const mxh = mx / (1 - b1**t);
  const myh = my / (1 - b1**t);
  const vxh = vx / (1 - b2**t);
  const vyh = vy / (1 - b2**t);
  return { x: s.x - lr*mxh/(Math.sqrt(vxh)+eps), y: s.y - lr*myh/(Math.sqrt(vyh)+eps), mx, my, vx, vy, t };
}

const STEP_FNS = { sgd: stepSGD, momentum: stepMomentum, adagrad: stepAdaGrad, rmsprop: stepRMSProp, adam: stepAdam };

// ---- Color map (magma-like) ------------------------------------------------

function surfaceColor(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [[0.005,0.005,0.018],[0.22,0.04,0.36],[0.72,0.18,0.38],[0.97,0.50,0.07],[0.99,0.96,0.60]];
  const s = t * (stops.length - 1);
  const lo = Math.floor(s), hi = Math.min(stops.length-1, lo+1), f = s - lo;
  return stops[lo].map((v,i) => v*(1-f)+stops[hi][i]*f);
}

// ---- State -----------------------------------------------------------------

let activeFnKey = 'elongated';
let fn = FUNCTIONS[activeFnKey];
let lr = 0.01;
let startPt = { ...fn.start };

let compareMode = 'optimizer'; // 'batch' | 'optimizer'
let pinnedOptimizerKey = 'sgd';
let pinnedBatchKey = 'full';
let checkedOptimizerKeys = new Set(OPTIMIZERS.map(o => o.key));

function activeLines() {
  if (compareMode === 'batch') {
    return BATCH_MODES.map(bm => ({
      key: bm.key, label: bm.label, color: bm.color,
      optimizerKey: pinnedOptimizerKey, batchMode: bm,
    }));
  }
  const batchMode = BATCH_MODES.find(b => b.key === pinnedBatchKey);
  return OPTIMIZERS.filter(o => checkedOptimizerKeys.has(o.key)).map(o => ({
    key: o.key, label: o.label, color: o.color,
    optimizerKey: o.key, batchMode,
  }));
}

let lines = activeLines();
let states = {};
let histories = {};
let iteration = 0;
let isRunning = false;
let rafId = null;
let stepsPerFrame = 1;

// ---- Three.js --------------------------------------------------------------

let scene, camera, renderer, controls;
let surfaceMesh = null;
let trajLines = {};
let startMarker = null;
const SURF_N = 70;

function initThree(container) {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x080810);

  camera = new THREE.PerspectiveCamera(42, 1, 0.01, 200);
  camera.position.set(5.5, 4.5, 5.5);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const w0 = container.clientWidth || 600, h0 = container.clientHeight || 420;
  renderer.setSize(w0, h0);
  camera.aspect = w0 / h0;
  camera.updateProjectionMatrix();
  container.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(3, 8, 4);
  scene.add(dir);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0.5, 0);
  controls.update();

  new ResizeObserver(() => {
    const w2 = container.clientWidth, h2 = container.clientHeight;
    if (!w2 || !h2) return;
    camera.aspect = w2 / h2;
    camera.updateProjectionMatrix();
    renderer.setSize(w2, h2);
  }).observe(container);
}

function buildSurface() {
  if (surfaceMesh) { scene.remove(surfaceMesh); surfaceMesh.geometry.dispose(); }

  const N = SURF_N, D = fn.domain;
  const positions = new Float32Array(N * N * 3);
  const colors    = new Float32Array(N * N * 3);
  const indices   = [];

  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const wx = -D + (i/(N-1)) * 2*D;
      const wy = -D + (j/(N-1)) * 2*D;
      const z = fn.zNorm(fn.f(wx, wy)) * SURF_H;
      const idx = (j*N+i)*3;
      positions[idx] = wx; positions[idx+1] = z; positions[idx+2] = -wy;
      const [r,g,b] = surfaceColor(fn.zNorm(fn.f(wx, wy)));
      colors[idx] = r; colors[idx+1] = g; colors[idx+2] = b;
    }
  }
  for (let j = 0; j < N-1; j++) {
    for (let i = 0; i < N-1; i++) {
      const a=j*N+i, b=j*N+i+1, c=(j+1)*N+i, d=(j+1)*N+i+1;
      indices.push(a,c,b, b,c,d);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  surfaceMesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
  scene.add(surfaceMesh);

  // Wireframe overlay
  const wfGeo = new THREE.WireframeGeometry(geo);
  const wf = new THREE.LineSegments(wfGeo, new THREE.LineBasicMaterial({ color: 0x000000, opacity: 0.18, transparent: true }));
  surfaceMesh.add(wf);
}

function pt3(x, y) {
  const z = fn.zNorm(fn.f(x, y)) * SURF_H + 0.012;
  return new THREE.Vector3(x, z, -y);
}

function rebuildTrajLines() {
  for (const key of Object.keys(trajLines)) { scene.remove(trajLines[key]); trajLines[key].geometry.dispose(); }
  trajLines = {};
  if (startMarker) { scene.remove(startMarker); startMarker = null; }

  for (const line of lines) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3000), 3));
    geo.setDrawRange(0, 0);
    const threeLine = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: line.color, linewidth: 2 }));
    scene.add(threeLine);
    trajLines[line.key] = threeLine;
  }

  const sg = new THREE.SphereGeometry(0.055, 12, 8);
  startMarker = new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ color: 0xffffff }));
  startMarker.position.copy(pt3(startPt.x, startPt.y));
  scene.add(startMarker);
}

function updateTrajLine(key) {
  const hist = histories[key];
  const line = trajLines[key];
  if (!line) return;
  const arr = line.geometry.attributes.position.array;
  const maxPts = arr.length / 3;
  const n = Math.min(hist.length, maxPts);
  for (let i = 0; i < n; i++) {
    const p = pt3(hist[i].x, hist[i].y);
    arr[i*3] = p.x; arr[i*3+1] = p.y; arr[i*3+2] = p.z;
  }
  line.geometry.setDrawRange(0, n);
  line.geometry.attributes.position.needsUpdate = true;
}

// ---- D3 contour ------------------------------------------------------------

let svg, xSc, ySc, contourPath, dotEls = {};

function initContour(container) {
  const d3 = window.d3;
  const rect = container.getBoundingClientRect();
  const W = Math.max(rect.width, 200), H = Math.max(rect.height, 200);
  container.innerHTML = '';
  svg = d3.select(container).append('svg').attr('viewBox', `0 0 ${W} ${H}`).style('width','100%').style('height','100%');
  const pad = 12;
  const D = fn.domain;
  xSc = d3.scaleLinear([-D, D], [pad, W-pad]);
  ySc = d3.scaleLinear([-D, D], [H-pad, pad]);

  // Contour grid
  const CN = 80;
  const vals = new Float64Array(CN * CN);
  for (let j = 0; j < CN; j++) {
    for (let i = 0; i < CN; i++) {
      const wx = -D + (i/(CN-1))*2*D;
      const wy = -D + (j/(CN-1))*2*D;
      vals[j*CN+i] = fn.zNorm(fn.f(wx, wy));
    }
  }

  const thresholds = d3.range(0.02, 1, 0.05);
  const contours = d3.contours().size([CN, CN]).thresholds(thresholds)(vals);
  const colorSc = d3.scaleSequential(d3.interpolateMagma).domain([0, 1]);

  const proj = d3.geoTransform({
    point(gx, gy) {
      const wx = -D + (gx/(CN-1))*2*D;
      const wy = -D + (gy/(CN-1))*2*D;
      this.stream.point(xSc(wx), ySc(wy));
    }
  });
  const path = d3.geoPath(proj);

  const cg = svg.append('g').attr('class', 'contour-group');
  cg.selectAll('path').data(contours).join('path')
    .attr('d', path)
    .attr('fill', d => colorSc(d.value))
    .attr('stroke', 'rgba(0,0,0,0.25)')
    .attr('stroke-width', 0.4);

  // Trajectory paths
  for (const line of lines) {
    svg.append('path').attr('class', `traj-path traj-${line.key}`)
      .attr('fill','none').attr('stroke', line.color).attr('stroke-width', 1.8).attr('opacity', 0.9);
  }

  // Current position dots
  dotEls = {};
  for (const line of lines) {
    dotEls[line.key] = svg.append('circle').attr('r', 4).attr('fill', line.color)
      .attr('stroke','#fff').attr('stroke-width', 1.2);
  }

  // Start marker
  svg.append('circle').attr('class','start-dot').attr('r', 5)
    .attr('fill','#fff').attr('stroke','#000').attr('stroke-width',1.5);

  // Click to move start
  svg.on('click', function(event) {
    const [px, py] = d3.pointer(event);
    startPt = { x: xSc.invert(px), y: ySc.invert(py) };
    resetAll();
  });

  contourPath = path;
  updateContourMarkers();
}

function updateContourMarkers() {
  svg.select('.start-dot').attr('cx', xSc(startPt.x)).attr('cy', ySc(startPt.y));

  for (const line of lines) {
    const hist = histories[line.key] || [];
    if (!hist.length) continue;
    const pts = hist.map(p => [xSc(p.x), ySc(p.y)]);
    svg.select(`.traj-${line.key}`).attr('d', 'M' + pts.map(p => p.join(',')).join('L'));
    const last = pts[pts.length-1];
    dotEls[line.key].attr('cx', last[0]).attr('cy', last[1]);
  }
}

// ---- Optimizer state -------------------------------------------------------

function resetAll() {
  isRunning = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  document.getElementById('gdAnimate').textContent = 'Animate';
  document.getElementById('gdAnimate').classList.remove('is-running');

  iteration = 0;
  states = {};
  histories = {};
  for (const line of lines) {
    states[line.key] = { x: startPt.x, y: startPt.y };
    histories[line.key] = [{ x: startPt.x, y: startPt.y }];
  }

  rebuildTrajLines();
  for (const line of lines) updateTrajLine(line.key);
  if (startMarker) startMarker.position.copy(pt3(startPt.x, startPt.y));
  updateContourMarkers();
  updateLegend();
}

function clamp(v, d) { return Math.max(-d, Math.min(d, v)); }

function doStep() {
  iteration++;
  const D = fn.domain;
  for (const line of lines) {
    const s = states[line.key];
    const g = noisyGrad(fn.grad(s.x, s.y), line.batchMode);
    const next = STEP_FNS[line.optimizerKey](s, g, lr);
    next.x = clamp(isFinite(next.x) ? next.x : s.x, D * 1.5);
    next.y = clamp(isFinite(next.y) ? next.y : s.y, D * 1.5);
    states[line.key] = { ...s, ...next };
    histories[line.key].push({ x: next.x, y: next.y });
  }
}

function stepAndDraw() {
  doStep();
  for (const line of lines) updateTrajLine(line.key);
  updateContourMarkers();
  updateLegend();
}

function updateLegend() {
  const el = document.getElementById('gdLegend');
  if (!el) return;
  const summary = compareMode === 'batch'
    ? `Comparing batch size — optimizer: ${OPTIMIZERS.find(o => o.key === pinnedOptimizerKey).label}`
    : `Comparing optimizers — batch size: ${BATCH_MODES.find(b => b.key === pinnedBatchKey).label}`;
  const rows = lines.map(line => {
    const s = states[line.key];
    const loss = s ? fn.f(s.x, s.y) : 0;
    return `<div class="gd-legend-row">
      <div class="gd-legend-dot" style="background:${line.color}"></div>
      <span class="gd-legend-name">${line.label}</span>
      <span class="gd-legend-stats">iter ${iteration}<br>loss ${loss.toFixed(4)}</span>
    </div>`;
  }).join('');
  el.innerHTML = `<div class="gd-legend-summary">${summary}</div>${rows}`;
}

// ---- Animation loop --------------------------------------------------------

function animate() {
  if (!isRunning) return;
  for (let i = 0; i < stepsPerFrame; i++) doStep();
  for (const line of lines) updateTrajLine(line.key);
  updateContourMarkers();
  updateLegend();
  controls.update();
  renderer.render(scene, camera);
  rafId = requestAnimationFrame(animate);
}

function threeLoop() {
  if (isRunning) return;
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(threeLoop);
}

// ---- Controls --------------------------------------------------------------

function switchFn(key) {
  activeFnKey = key;
  fn = FUNCTIONS[key];
  startPt = { ...fn.start };
  buildSurface();
  if (svg) initContour(document.getElementById('gdContour'));
  resetAll();
}

function applyLineChange() {
  lines = activeLines();
  if (svg) initContour(document.getElementById('gdContour'));
  resetAll();
}

// ---- Init ------------------------------------------------------------------
// Modules are always deferred; the DOM is ready when this code runs.

function showErr(msg) {
  ['gdSurface', 'gdContour'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.style.cssText += ';color:#f66;font:12px monospace;padding:8px;white-space:pre-wrap'; el.textContent = msg; }
  });
}

try {
  const surfaceEl  = document.getElementById('gdSurface');
  const contourEl  = document.getElementById('gdContour');
  if (!surfaceEl) throw new Error('gdSurface element not found');
  if (!window.d3)  throw new Error('D3 not loaded');

  initThree(surfaceEl);
  buildSurface();
  initContour(contourEl);
  resetAll();
  requestAnimationFrame(threeLoop);

  document.getElementById('gdFn').addEventListener('change', e => switchFn(e.target.value));

  const lrInput = document.getElementById('gdLr');
  const lrVal   = document.getElementById('gdLrVal');
  lrInput.addEventListener('input', () => {
    lr = Math.pow(10, +lrInput.value);
    lrVal.textContent = lr.toFixed(lr < 0.01 ? 4 : lr < 0.1 ? 3 : 2);
  });

  document.getElementById('gdAnimate').addEventListener('click', () => {
    isRunning = !isRunning;
    const btn = document.getElementById('gdAnimate');
    if (isRunning) {
      btn.textContent = 'Pause';
      btn.classList.add('is-running');
      animate();
    } else {
      btn.textContent = 'Animate';
      btn.classList.remove('is-running');
      if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
      requestAnimationFrame(threeLoop);
    }
  });

  document.getElementById('gdStep').addEventListener('click', () => {
    if (isRunning) return;
    stepAndDraw();
    controls.update();
    renderer.render(scene, camera);
  });

  document.getElementById('gdReset').addEventListener('click', resetAll);
} catch (e) {
  showErr('Init error: ' + e.message + '\n' + e.stack);
}
```

- [ ] **Step 3: Visual check — default view still works with the old HTML/CSS (no new controls exist yet).**

Serve the repo root (e.g. `python3 -m http.server 8000`) and open `http://localhost:8000/pages/gradient-descent.html`.
Expected: the sandbox loads with 5 lines (not the old 3) descending the elongated bowl — SGD, Momentum, AdaGrad, RMSProp, Adam, all noise-free (default `compareMode` is `'optimizer'` with `pinnedBatchKey: 'full'`).
Click Animate: all 5 converge toward the origin; AdaGrad should visibly slow down/stall before the others; RMSProp and Adam should keep making progress.
Click Reset, click Step repeatedly: legend iteration count and loss values update for all 5 rows.
Change the Surface dropdown to Rosenbrock, Quadratic bowl, and Saddle + valleys in turn: each redraws correctly with 5 lines.
Open the browser console: no errors.

- [ ] **Step 4: Commit**

```bash
git add js/gradient-descent.js
git commit -m "feat(gradient-descent): generalize optimizer core for batch-size and optimizer comparison"
```

---

### Task 2: Add mode toggle, pinned selectors, and optimizer checkboxes

**Files:**
- Modify: `pages/gradient-descent.html:1-83` (head script list and the Sandbox section)
- Modify: `styles/gradient-descent.css` (append)
- Modify: `js/gradient-descent.js` (append event listeners inside the existing `try` block from Task 1)

**Interfaces:**
- Consumes: `applyLineChange()`, `compareMode`, `pinnedOptimizerKey`, `pinnedBatchKey`, `checkedOptimizerKeys` from Task 1's `js/gradient-descent.js`.
- Consumes: the shared `js/tabs.js` controller (already in the repo, unmodified) — toggles a plain `active` class on `[data-tab]` buttons and sets `hidden` on sibling `[data-panel]` elements whose `data-panel` doesn't match the active `data-tab`.
- Produces: DOM element ids `gdPinnedOptimizer`, `gdPinnedBatch`, and the `#gdOptimizerChecks .check` checkboxes, consumed only by this task's own listeners (no later task depends on them).

- [ ] **Step 1: Add the `tabs.js` script tag to the page head.**

In `pages/gradient-descent.html`, find:

```html
  <script src="../js/theme.js"></script>
  <script type="module" src="../js/gradient-descent.js"></script>
```

Replace with:

```html
  <script src="../js/theme.js"></script>
  <script defer src="../js/tabs.js"></script>
  <script type="module" src="../js/gradient-descent.js"></script>
```

- [ ] **Step 2: Replace the Sandbox section markup.**

Find the entire `<section class="panel">...</section>` block that starts with `<h2>Sandbox</h2>` (currently lines 47-83) and replace it with:

```html
      <section class="panel">
        <h2>Sandbox</h2>
        <p>Pick a loss surface, choose whether to compare batch size or optimizers, and watch how the pinned dimension changes the descent.
        Click anywhere on the contour map to move the start.</p>

        <div class="tabs gd-mode-tabs" role="tablist">
          <button type="button" class="tab" data-tab="batch">Compare batch size</button>
          <button type="button" class="tab active" data-tab="optimizer">Compare optimizers</button>
        </div>

        <div class="gd-controls">
          <div class="gd-control-group" data-panel="batch" hidden>
            <label class="gd-label" for="gdPinnedOptimizer">Optimizer</label>
            <select id="gdPinnedOptimizer" class="gd-select">
              <option value="sgd">SGD</option>
              <option value="momentum">Momentum</option>
              <option value="adagrad">AdaGrad</option>
              <option value="rmsprop">RMSProp</option>
              <option value="adam">Adam</option>
            </select>
          </div>
          <div class="gd-control-group" data-panel="optimizer">
            <label class="gd-label" for="gdPinnedBatch">Batch size</label>
            <select id="gdPinnedBatch" class="gd-select">
              <option value="full">Full-batch</option>
              <option value="mini">Mini-batch</option>
              <option value="stochastic">Stochastic</option>
            </select>
          </div>
          <div class="gd-control-group gd-checkbox-group" id="gdOptimizerChecks" data-panel="optimizer">
            <span class="gd-label">Optimizers</span>
            <label class="control-row"><input class="check" type="checkbox" value="sgd" checked> SGD</label>
            <label class="control-row"><input class="check" type="checkbox" value="momentum" checked> Momentum</label>
            <label class="control-row"><input class="check" type="checkbox" value="adagrad" checked> AdaGrad</label>
            <label class="control-row"><input class="check" type="checkbox" value="rmsprop" checked> RMSProp</label>
            <label class="control-row"><input class="check" type="checkbox" value="adam" checked> Adam</label>
          </div>
          <div class="gd-control-group">
            <label class="gd-label" for="gdFn">Surface</label>
            <select id="gdFn" class="gd-select">
              <option value="elongated">Elongated bowl</option>
              <option value="rosenbrock">Rosenbrock</option>
              <option value="bowl">Quadratic bowl</option>
              <option value="saddle">Saddle + valleys</option>
            </select>
          </div>
          <div class="gd-control-group">
            <label class="gd-label" for="gdLr">Learning rate</label>
            <div class="gd-slider-row">
              <input type="range" id="gdLr" class="gd-slider" min="-4" max="-1" step="0.05" value="-2">
              <output id="gdLrVal" class="gd-output">0.01</output>
            </div>
          </div>
          <div class="gd-btn-group">
            <button id="gdAnimate" class="gd-btn gd-btn-primary" type="button">Animate</button>
            <button id="gdStep"    class="gd-btn" type="button">Step</button>
            <button id="gdReset"   class="gd-btn" type="button">Reset</button>
          </div>
        </div>

        <div class="gd-viz-row">
          <div class="gd-surface-wrap" id="gdSurface"></div>
          <div class="gd-right-col">
            <div class="gd-contour-wrap" id="gdContour"></div>
            <div class="gd-legend" id="gdLegend"></div>
          </div>
        </div>
      </section>
```

Note the `optimizer` tab starts with class `active` and its two panels have no `hidden` attribute, matching Task 1's default `compareMode = 'optimizer'`; the `batch` panel starts `hidden`.

- [ ] **Step 3: Append control styling to `styles/gradient-descent.css`.**

Add after the existing `.gd-btn-primary.is-running` rule (the last rule in the "Controls bar" section, right before the `/* --- Viz row --- */` comment):

```css
.ui.gradient-descent .gd-mode-tabs { margin-bottom: 16px; }
.ui.gradient-descent .gd-checkbox-group { gap: 7px; }
.ui.gradient-descent .gd-checkbox-group .control-row { gap: 7px; }
```

Add after the existing `.gd-legend` rule (before `.gd-legend-row`):

```css
.ui.gradient-descent .gd-legend-summary {
  font: 600 10px/1.3 var(--font-mono);
  letter-spacing: .04em;
  color: var(--text-muted);
  padding-bottom: 8px;
  margin-bottom: 2px;
  border-bottom: 1px solid var(--hairline);
}
```

- [ ] **Step 4: Wire the new controls in `js/gradient-descent.js`.**

Find the last listener in the init `try` block:

```js
  document.getElementById('gdReset').addEventListener('click', resetAll);
} catch (e) {
```

Replace with:

```js
  document.getElementById('gdReset').addEventListener('click', resetAll);

  document.querySelectorAll('.gd-mode-tabs .tab').forEach(btn => {
    btn.addEventListener('click', () => {
      if (btn.dataset.tab === compareMode) return;
      compareMode = btn.dataset.tab;
      applyLineChange();
    });
  });

  document.getElementById('gdPinnedOptimizer').addEventListener('change', e => {
    pinnedOptimizerKey = e.target.value;
    applyLineChange();
  });

  document.getElementById('gdPinnedBatch').addEventListener('change', e => {
    pinnedBatchKey = e.target.value;
    applyLineChange();
  });

  document.querySelectorAll('#gdOptimizerChecks .check').forEach(cb => {
    cb.addEventListener('change', () => {
      const next = new Set(checkedOptimizerKeys);
      if (cb.checked) next.add(cb.value); else next.delete(cb.value);
      if (next.size === 0) { cb.checked = true; return; }
      checkedOptimizerKeys = next;
      applyLineChange();
    });
  });
} catch (e) {
```

- [ ] **Step 5: Visual + behavior check.**

Reload `pages/gradient-descent.html`.
Default view: "Compare optimizers" tab is active, Batch size = Full-batch, all 5 optimizer checkboxes checked, 5 noise-free lines, same as Task 1's check.
Click "Compare batch size": the tab switches active state, the Optimizer selector appears (default SGD) and the Batch size selector + checkboxes disappear, and exactly 3 lines render (Full-batch, Mini-batch, Stochastic).
Click Animate: the Mini-batch and Stochastic lines visibly jitter off the smooth path the Full-batch line takes; Stochastic jitters more than Mini-batch.
Change the pinned Optimizer to Adam while in batch-size mode: reset and re-animate — the jitter should look visibly smaller/smoothed compared to SGD, since Adam's per-parameter scaling damps noisy gradients.
Switch back to "Compare optimizers": Batch size resets to Full-batch, all 5 checked, noise-free again.
Change Batch size to Stochastic: all 5 checked optimizer lines now show noisy paths.
Uncheck AdaGrad and RMSProp: only 3 lines remain (SGD, Momentum, Adam).
Uncheck every box down to one (e.g. only Adam left), then try to uncheck that last one: the checkbox stays checked and the line count stays at 1 (does not drop to zero).
Resize the browser window narrow (< 820px): controls wrap without overlapping.
Open the browser console: no errors throughout.

- [ ] **Step 6: Commit**

```bash
git add pages/gradient-descent.html styles/gradient-descent.css js/gradient-descent.js
git commit -m "feat(gradient-descent): add batch-size/optimizer mode toggle and controls"
```

---

### Task 3: Update page content and the home page card

**Files:**
- Modify: `pages/gradient-descent.html:6` (title), `:39-43` (header), and the "The algorithms" panel (previously lines 85-111, shifted by Task 2's insertions — locate by the `<h2>The algorithms</h2>` heading)
- Modify: `index.html` (the gradient-descent card, entry 09)

**Interfaces:**
- Consumes: nothing (content-only; must not remove or rename any element id used by `js/gradient-descent.js`).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Update the page title.**

Find:

```html
  <title>Gradient Descent: SGD, Momentum, and Adam</title>
```

Replace with:

```html
  <title>Gradient Descent: Batch Size and Optimizers</title>
```

- [ ] **Step 2: Update the header copy.**

Find:

```html
    <header class="page-head">
      <div class="eyebrow">// Optimization</div>
      <h1>Gradient descent</h1>
      <p class="lede">Step through how SGD, momentum, and Adam navigate a loss surface, and see where each one struggles.</p>
    </header>
```

Replace with:

```html
    <header class="page-head">
      <div class="eyebrow">// Optimization</div>
      <h1>Gradient descent</h1>
      <p class="lede">Compare how batch size and optimizer choice change the way a point descends a loss surface, from full-batch and stochastic gradients to momentum, AdaGrad, RMSProp, and Adam.</p>
    </header>
```

- [ ] **Step 3: Replace "The algorithms" panel.**

Find the entire `<section class="panel">` block starting with `<h2>The algorithms</h2>` and ending at that section's closing `</section>` (the block containing the `Stochastic gradient descent`, `Momentum`, and `Adam` subsections). Replace it with:

```html
      <section class="panel">
        <h2>The algorithms</h2>

        <h3>SGD</h3>
        <p>At each step, SGD moves the parameters directly opposite the gradient, scaled by the learning rate $\alpha$.</p>
        <div class="formula">$$\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$$</div>
        <p>On an elongated bowl, equal step sizes in every direction means SGD bounces across the narrow axis while crawling along the long axis.
        The learning rate that is small enough to avoid divergence on the steep axis is far too small for efficient progress on the shallow axis.
        This same update rule is what every batch-size line in the sandbox uses when SGD is the pinned optimizer; only how noisy the gradient it receives changes.</p>

        <h3>Momentum</h3>
        <p>Momentum accumulates a velocity vector $v$ that carries the optimizer across flat regions and dampens oscillations across curved ones, fixing the bouncing SGD shows above.
        $\beta$ (typically 0.9) controls how much of the previous velocity survives each step.</p>
        <div class="formula">$$v_{t+1} = \beta v_t - \alpha \nabla L(\theta_t)$$</div>
        <div class="formula">$$\theta_{t+1} = \theta_t + v_{t+1}$$</div>
        <p>On the elongated bowl, the accumulated velocity along the long axis lets momentum accelerate where SGD is slow.
        The overshoot into the steep walls eventually dampens, but the path oscillates more dramatically before settling.</p>

        <h3>AdaGrad</h3>
        <p>Momentum still applies the same effective step size to every parameter, so a steeply curved axis and a shallow one are still stuck sharing one learning rate.
        AdaGrad addresses this by giving each parameter its own rate, shrinking it in proportion to the cumulative sum of squared gradients that parameter has seen so far.</p>
        <div class="formula">$$G_t = G_{t-1} + (\nabla L(\theta_t))^2$$</div>
        <div class="formula">$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t} + \epsilon} \nabla L(\theta_t)$$</div>
        <p>Parameters with consistently large gradients get their rate shrunk quickly; parameters with small gradients keep a larger effective rate.
        Because $G_t$ only ever grows, the effective rate keeps shrinking for the rest of the run, and on a long descent it can decay so far that progress effectively stalls.</p>

        <h3>RMSProp</h3>
        <p>RMSProp keeps AdaGrad's idea of a per-parameter rate but fixes the stall: instead of an ever-growing sum of squared gradients, it keeps an exponential moving average, so old gradients are gradually forgotten.</p>
        <div class="formula">$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)(\nabla L(\theta_t))^2$$</div>
        <div class="formula">$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{E[g^2]_t} + \epsilon} \nabla L(\theta_t)$$</div>
        <p>Because the average can rise again if gradients grow again, RMSProp's effective rate does not monotonically decay the way AdaGrad's does, so it keeps making progress over long runs.</p>

        <h3>Adam</h3>
        <p>Adam combines momentum's directional smoothing with RMSProp's per-parameter scaling: a running estimate of both the first moment (mean gradient $m$) and the second moment (uncentered variance $v$) of the gradient.
        The parameter update is scaled by the per-dimension gradient variance, so large gradients get smaller effective steps and small gradients get larger ones.</p>
        <div class="formula">$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L$$</div>
        <div class="formula">$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L)^2$$</div>
        <div class="formula">$$\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$</div>
        <p>The $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected estimates that account for the zero-initialization of the moment vectors at $t=0$.
        Adam navigates elongated landscapes efficiently because the per-dimension scaling automatically compensates for the difference in curvature between axes.
        Its effective learning rate for each parameter is approximately $\alpha$, regardless of gradient magnitude.</p>

        <h3>Batch size</h3>
        <p>Every update rule above needs a gradient to work with, and how that gradient is computed is a separate choice from which update rule consumes it.
        Full-batch gradient descent computes the exact gradient over the entire dataset every step: the direction is precise, but one step costs a full pass over the data.
        Stochastic gradient descent estimates the gradient from a single example: each step is cheap, but the estimate is noisy, so the path jitters.
        Mini-batch gradient descent averages over a small batch of examples, trading some of SGD's noise for some of full-batch's cost, and is what most training in practice actually uses.</p>
        <p>The surfaces in this sandbox are analytic functions, not real datasets, so there is nothing to literally subsample.
        The batch-size comparison instead adds simulated Gaussian noise to the true analytic gradient, scaled down as the batch size grows ($\sigma \propto 1/\sqrt{n}$), to illustrate the effect a real batch size would have.</p>
      </section>
```

- [ ] **Step 4: Update the home page card.**

In `index.html`, find:

```html
        <li>
          <a class="project-row" href="pages/gradient-descent.html">
            <span class="pr-n">09</span>
            <span class="pr-main">
              <span class="pr-title">Gradient Descent: SGD, Momentum, and Adam</span>
              <span class="pr-desc">Interactive sandbox stepping through how SGD, momentum, and Adam navigate a loss surface, and where each one struggles.</span>
            </span>
            <span class="pr-cat">Machine learning</span>
          </a>
        </li>
```

Replace with:

```html
        <li>
          <a class="project-row" href="pages/gradient-descent.html">
            <span class="pr-n">09</span>
            <span class="pr-main">
              <span class="pr-title">Gradient Descent: Batch Size and Optimizers</span>
              <span class="pr-desc">Interactive sandbox comparing full-batch, mini-batch, and stochastic gradient descent, and SGD, momentum, AdaGrad, RMSProp, and Adam optimizers, on a shared loss surface.</span>
            </span>
            <span class="pr-cat">Machine learning</span>
          </a>
        </li>
```

- [ ] **Step 5: Visual check.**

Reload `pages/gradient-descent.html`: MathJax renders every formula in "The algorithms" panel with no literal `$...$` text visible and no console errors; the section-outline nav (right rail) still shows one entry per `.panel` (label taken from each panel's first heading — "Sandbox" and "SGD").
Reload `index.html`: the entry 09 card shows the new title/description and still links to `pages/gradient-descent.html`.

- [ ] **Step 6: Commit**

```bash
git add pages/gradient-descent.html index.html
git commit -m "docs(gradient-descent): rewrite content for batch-size and optimizer lineage"
```

---

### Task 4: Full cross-check pass

**Files:** none (verification only; fix forward in the relevant file from Tasks 1-3 if any check fails, then re-run that check).

- [ ] **Step 1: Fresh reload, default state.** "Compare optimizers" active, Batch size = Full-batch, all 5 checked, elongated bowl. Animate to convergence: AdaGrad visibly stalls before RMSProp/Adam/Momentum/SGD finish.

- [ ] **Step 2: Every surface in both modes.** For each of the 4 surfaces, toggle both tabs at least once and click Animate. No `NaN`/off-canvas trajectories, no console errors. (Rosenbrock has the largest gradients near its start point — confirm the Stochastic batch-size line there still stays visually on the surface rather than immediately flying off; if it diverges, lower `BASE_NOISE` in `js/gradient-descent.js` and re-check.)

- [ ] **Step 3: Click-to-move-start.** In both modes, click a new point on the contour view: all active lines restart from the new point with their prior history cleared.

- [ ] **Step 4: Learning-rate slider.** Drag it across its full range in both modes: the output label updates and Step/Animate reflect the new rate immediately.

- [ ] **Step 5: Checkbox floor guard.** In "Compare optimizers" mode, uncheck all but one box, confirm the last cannot be unchecked (from Task 2 Step 5) still holds after switching surfaces and modes in between.

- [ ] **Step 6: Mobile width.** Resize to under 820px: `gd-viz-row` stacks, controls (including the new tabs/selects/checkboxes) wrap without overlapping or clipping.

- [ ] **Step 7: Home page link.** From `index.html`, click through to the gradient descent card and confirm the page loads correctly.

If every check passes, no further commit is needed (Task 4 is verification-only). If a check fails, fix the issue in the file it belongs to, re-run that specific check, then make a small fix commit referencing which check it addresses.
