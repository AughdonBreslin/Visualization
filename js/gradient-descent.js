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

// ---- Optimizers ------------------------------------------------------------

const OPTS = [
  { key: 'sgd',      label: 'SGD',      color: '#74b9ff' },
  { key: 'momentum', label: 'Momentum', color: '#fd79a8' },
  { key: 'adam',     label: 'Adam',     color: '#00cec9' },
];

function stepSGD(s, grad, lr) {
  const [gx, gy] = grad(s.x, s.y);
  return { x: s.x - lr*gx, y: s.y - lr*gy };
}

function stepMomentum(s, grad, lr, beta = 0.9) {
  const [gx, gy] = grad(s.x, s.y);
  const vx = beta*(s.vx||0) - lr*gx;
  const vy = beta*(s.vy||0) - lr*gy;
  return { x: s.x + vx, y: s.y + vy, vx, vy };
}

function stepAdam(s, grad, lr, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
  const t = (s.t||0) + 1;
  const [gx, gy] = grad(s.x, s.y);
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

const STEP_FNS = { sgd: stepSGD, momentum: stepMomentum, adam: stepAdam };

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

  for (const opt of OPTS) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3000), 3));
    geo.setDrawRange(0, 0);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: opt.color, linewidth: 2 }));
    scene.add(line);
    trajLines[opt.key] = line;
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
  for (const opt of OPTS) {
    svg.append('path').attr('class', `traj-path traj-${opt.key}`)
      .attr('fill','none').attr('stroke', opt.color).attr('stroke-width', 1.8).attr('opacity', 0.9);
  }

  // Current position dots
  dotEls = {};
  for (const opt of OPTS) {
    dotEls[opt.key] = svg.append('circle').attr('r', 4).attr('fill', opt.color)
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
  const d3 = window.d3;
  svg.select('.start-dot').attr('cx', xSc(startPt.x)).attr('cy', ySc(startPt.y));

  for (const opt of OPTS) {
    const hist = histories[opt.key] || [];
    if (!hist.length) continue;
    const pts = hist.map(p => [xSc(p.x), ySc(p.y)]);
    svg.select(`.traj-${opt.key}`).attr('d', 'M' + pts.map(p => p.join(',')).join('L'));
    const last = pts[pts.length-1];
    dotEls[opt.key].attr('cx', last[0]).attr('cy', last[1]);
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
  for (const opt of OPTS) {
    states[opt.key] = { x: startPt.x, y: startPt.y };
    histories[opt.key] = [{ x: startPt.x, y: startPt.y }];
  }

  rebuildTrajLines();
  for (const opt of OPTS) updateTrajLine(opt.key);
  if (startMarker) startMarker.position.copy(pt3(startPt.x, startPt.y));
  updateContourMarkers();
  updateLegend();
}

function clamp(v, d) { return Math.max(-d, Math.min(d, v)); }

function doStep() {
  iteration++;
  const D = fn.domain;
  for (const opt of OPTS) {
    const s = states[opt.key];
    const next = STEP_FNS[opt.key](s, fn.grad, lr);
    next.x = clamp(isFinite(next.x) ? next.x : s.x, D * 1.5);
    next.y = clamp(isFinite(next.y) ? next.y : s.y, D * 1.5);
    states[opt.key] = { ...s, ...next };
    histories[opt.key].push({ x: next.x, y: next.y });
  }
}

function stepAndDraw() {
  doStep();
  for (const opt of OPTS) updateTrajLine(opt.key);
  updateContourMarkers();
  updateLegend();
}

function updateLegend() {
  const el = document.getElementById('gdLegend');
  if (!el) return;
  el.innerHTML = OPTS.map(opt => {
    const s = states[opt.key];
    const loss = s ? fn.f(s.x, s.y) : 0;
    return `<div class="gd-legend-row">
      <div class="gd-legend-dot" style="background:${opt.color}"></div>
      <span class="gd-legend-name">${opt.label}</span>
      <span class="gd-legend-stats">iter ${iteration}<br>loss ${loss.toFixed(4)}</span>
    </div>`;
  }).join('');
}

// ---- Animation loop --------------------------------------------------------

function animate() {
  if (!isRunning) return;
  for (let i = 0; i < stepsPerFrame; i++) doStep();
  for (const opt of OPTS) updateTrajLine(opt.key);
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
