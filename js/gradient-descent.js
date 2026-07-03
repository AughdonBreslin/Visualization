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

// Rosenbrock's gradient magnitude near its start point (and along its walls) is an
// order of magnitude larger than any other surface here (~200+ vs ~17 at most). Left
// unclipped, a single default-learning-rate step overshoots into a region with an even
// steeper gradient, and the trajectory cascades to the domain edge within 2-3 steps
// regardless of batch size, including full-batch with no noise at all. Clipping the
// gradient norm before it is used keeps every surface's descent numerically stable.
// During normal descent from each surface's default start point, the clip does not
// engage on the other three surfaces (elongated bowl peaks around 17). Only the
// elongated bowl can engage the clip if the user clicks a start point near a domain
// corner (e.g. at (2.9, 2.9), gradient norm around 47). The quadratic bowl and
// saddle never reach the threshold even at their domain corners. This has been
// verified to only moderately cap an already-large initial step and never cause NaN
// or a stuck trajectory.
const MAX_GRAD_NORM = 25;

// A line stops taking steps once its own displacement has stayed below this
// threshold for CONVERGE_STREAK consecutive steps, independent of every other
// line's state, so a line that reaches a flat region stops animating instead
// of running (and growing its history) forever.
const CONVERGE_EPS = 1e-4;
const CONVERGE_STREAK = 5;

// AdaGrad and RMSProp normalize each step by that parameter's own recent gradient
// scale, so near a minimum their displacement can hover at a small but roughly
// constant, non-vanishing size instead of shrinking below CONVERGE_EPS the way
// SGD/Momentum/Adam's does. This cap guarantees every line eventually stops
// animating (and stops growing its history) even in that case.
const MAX_ITER_PER_LINE = 2500;

function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function clipGrad([gx, gy], maxNorm) {
  const norm = Math.hypot(gx, gy);
  if (norm <= maxNorm || norm === 0) return [gx, gy];
  const scale = maxNorm / norm;
  return [gx * scale, gy * scale];
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
  const D = fn.domain;
  for (const line of lines) {
    const s = states[line.key];
    if (s.converged) continue;
    const g = noisyGrad(clipGrad(fn.grad(s.x, s.y), MAX_GRAD_NORM), line.batchMode);
    const next = STEP_FNS[line.optimizerKey](s, g, lr);
    next.x = clamp(isFinite(next.x) ? next.x : s.x, D * 1.5);
    next.y = clamp(isFinite(next.y) ? next.y : s.y, D * 1.5);
    const displacement = Math.hypot(next.x - s.x, next.y - s.y);
    next.convergeStreak = displacement < CONVERGE_EPS ? (s.convergeStreak || 0) + 1 : 0;
    next.iter = (s.iter || 0) + 1;
    next.converged = next.convergeStreak >= CONVERGE_STREAK || next.iter >= MAX_ITER_PER_LINE;
    states[line.key] = { ...s, ...next };
    histories[line.key].push({ x: next.x, y: next.y });
  }
}

function allConverged() {
  return lines.every(line => states[line.key] && states[line.key].converged);
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
    ? `Comparing batch size, optimizer: ${OPTIMIZERS.find(o => o.key === pinnedOptimizerKey).label}`
    : `Comparing optimizers, batch size: ${BATCH_MODES.find(b => b.key === pinnedBatchKey).label}`;
  const rows = lines.map(line => {
    const s = states[line.key];
    const loss = s ? fn.f(s.x, s.y) : 0;
    const iter = s ? (s.iter || 0) : 0;
    return `<div class="gd-legend-row">
      <div class="gd-legend-dot" style="background:${line.color}"></div>
      <span class="gd-legend-name">${line.label}</span>
      <span class="gd-legend-stats">iter ${iter}<br>loss ${loss.toFixed(4)}</span>
    </div>`;
  }).join('');
  el.innerHTML = `<div class="gd-legend-summary">${summary}</div>${rows}`;
}

// ---- Animation loop --------------------------------------------------------

function stopAnimating() {
  isRunning = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  const btn = document.getElementById('gdAnimate');
  btn.textContent = 'Animate';
  btn.classList.remove('is-running');
  requestAnimationFrame(threeLoop);
}

function animate() {
  if (!isRunning) return;
  for (let i = 0; i < stepsPerFrame; i++) doStep();
  for (const line of lines) updateTrajLine(line.key);
  updateContourMarkers();
  updateLegend();
  controls.update();
  renderer.render(scene, camera);
  if (allConverged()) { stopAnimating(); return; }
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
    if (isRunning) {
      const btn = document.getElementById('gdAnimate');
      btn.textContent = 'Pause';
      btn.classList.add('is-running');
      animate();
    } else {
      stopAnimating();
    }
  });

  document.getElementById('gdStep').addEventListener('click', () => {
    if (isRunning) return;
    stepAndDraw();
    controls.update();
    renderer.render(scene, camera);
  });

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
  showErr('Init error: ' + e.message + '\n' + e.stack);
}
