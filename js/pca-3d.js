import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

// --- geometry helpers ---

function ensurePointsGeo(mesh, N, capRef) {
  if (N > capRef.cap) {
    mesh.geometry.dispose();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    mesh.geometry = geo;
    capRef.cap = N;
  }
  return mesh.geometry;
}

function writePoints(geo, points) {
  const arr = geo.attributes.position.array;
  for (let i = 0; i < points.length; i++) {
    arr[i * 3]     = points[i][0];
    arr[i * 3 + 1] = points[i][1];
    arr[i * 3 + 2] = points[i][2];
  }
  geo.setDrawRange(0, points.length);
  geo.attributes.position.needsUpdate = true;
}

function applyMat3(m, v) {
  return [
    m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
    m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
    m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2],
  ];
}

function makeCss2DLabel(text) {
  const div = document.createElement('div');
  div.className = 'pca-label';
  div.textContent = text;
  return new CSS2DObject(div);
}

// --- wireframe sphere helpers (used by createOperatorPlot3D) ---

const LAT_DEGS = [-60, -30, 0, 30, 60];
const LON_DEGS = [0, 30, 60, 90, 120, 150];
const WIRE_STEP_DEG = 12;
const WIRE_PTS = 360 / WIRE_STEP_DEG + 1; // 31 per circle (0 through 360, closed)

function makeCircleLine(color, opacity) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(WIRE_PTS * 3), 3));
  return new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
}

function updateWireframe(latLines, lonLines, transform) {
  const DEG = Math.PI / 180;
  LAT_DEGS.forEach((latDeg, li) => {
    const lat = latDeg * DEG;
    const arr = latLines[li].geometry.attributes.position.array;
    for (let t = 0; t <= 360; t += WIRE_STEP_DEG) {
      const lon = t * DEG;
      const p = applyMat3(transform, [Math.cos(lat)*Math.cos(lon), Math.cos(lat)*Math.sin(lon), Math.sin(lat)]);
      const idx = (t / WIRE_STEP_DEG) * 3;
      arr[idx] = p[0]; arr[idx+1] = p[1]; arr[idx+2] = p[2];
    }
    latLines[li].geometry.attributes.position.needsUpdate = true;
  });
  LON_DEGS.forEach((lonDeg, li) => {
    const lon = lonDeg * DEG;
    const arr = lonLines[li].geometry.attributes.position.array;
    for (let t = 0; t <= 360; t += WIRE_STEP_DEG) {
      const theta = t * DEG;
      const p = applyMat3(transform, [Math.cos(theta)*Math.cos(lon), Math.cos(theta)*Math.sin(lon), Math.sin(theta)]);
      const idx = (t / WIRE_STEP_DEG) * 3;
      arr[idx] = p[0]; arr[idx+1] = p[1]; arr[idx+2] = p[2];
    }
    lonLines[li].geometry.attributes.position.needsUpdate = true;
  });
}

// --- shared context factory ---

function makePlot3DContext(container) {
  // Clear any existing D3 SVG or previous canvas before mounting
  container.innerHTML = '';
  container.style.position = 'relative';

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x000000, 0);
  renderer.domElement.style.display = 'block';
  container.appendChild(renderer.domElement);

  const css2d = new CSS2DRenderer();
  css2d.domElement.style.position = 'absolute';
  css2d.domElement.style.top = '0';
  css2d.domElement.style.left = '0';
  css2d.domElement.style.width = '100%';
  css2d.domElement.style.height = '100%';
  css2d.domElement.style.pointerEvents = 'none';
  container.appendChild(css2d.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
  camera.position.set(1.45, 1.35, 1.15);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  controls.target.set(0, 0, 0);
  controls.update();

  const ctx = {
    renderer, css2d, scene, camera, controls,
    syncing: false,
    cameraListeners: [],
  };

  ctx.render = function () {
    const w = container.clientWidth || 400;
    const h = container.clientHeight || 400;
    renderer.setSize(w, h, false);
    css2d.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
    css2d.render(scene, camera);
  };

  ctx.resizeObserver = new ResizeObserver(() => ctx.render());
  ctx.resizeObserver.observe(container);

  controls.addEventListener('change', () => {
    ctx.render();
    if (!ctx.syncing) {
      const s = new THREE.Spherical().setFromVector3(camera.position);
      ctx.cameraListeners.forEach(fn => fn(s.phi, s.theta));
    }
  });

  return ctx;
}

function makeContextApi(ctx, container) {
  return {
    onCameraChange(fn) { ctx.cameraListeners.push(fn); },
    applyCameraDir(phi, theta) {
      ctx.syncing = true;
      const r = ctx.camera.position.length();
      ctx.camera.position.setFromSphericalCoords(r, phi, theta);
      ctx.controls.update();
      ctx.render();
      ctx.syncing = false;
    },
  };
}

// --- stub exports (replaced in Tasks 3 and 4) ---

export function createDataPlot3D(container) {
  const ctx = makePlot3DContext(container);
  ctx.render();
  return {
    update(_s) { ctx.render(); },
    destroy() {
      ctx.resizeObserver.disconnect();
      ctx.renderer.dispose();
      if (container.contains(ctx.renderer.domElement)) container.removeChild(ctx.renderer.domElement);
      if (container.contains(ctx.css2d.domElement)) container.removeChild(ctx.css2d.domElement);
    },
    ...makeContextApi(ctx, container),
  };
}

export function createOperatorPlot3D(container) {
  const ctx = makePlot3DContext(container);
  ctx.render();
  return {
    update(_s) { ctx.render(); },
    destroy() {
      ctx.resizeObserver.disconnect();
      ctx.renderer.dispose();
      if (container.contains(ctx.renderer.domElement)) container.removeChild(ctx.renderer.domElement);
      if (container.contains(ctx.css2d.domElement)) container.removeChild(ctx.css2d.domElement);
    },
    ...makeContextApi(ctx, container),
  };
}
