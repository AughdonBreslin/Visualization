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

function makeCss2DLabel(text, extraClass = '') {
  const div = document.createElement('div');
  div.className = extraClass ? `pca-label ${extraClass}` : 'pca-label';
  div.textContent = text;
  return new CSS2DObject(div);
}

function makeCircleTexture() {
  const size = 32;
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const c = canvas.getContext('2d');
  c.beginPath();
  c.arc(size / 2, size / 2, 14, 0, Math.PI * 2);
  c.fillStyle = '#ffffff';
  c.fill();
  return new THREE.CanvasTexture(canvas);
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
    renderer.setSize(w, h);
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
      ctx.syncing = false;
    },
  };
}

// --- createDataPlot3D ---

export function createDataPlot3D(container) {
  const ctx = makePlot3DContext(container);
  const { scene, camera, controls } = ctx;

  const PC_COLORS = [0x7dffb2, 0xffc456, 0xff7a7a];
  const AXIS_OPACITIES = [0.22, 0.18, 0.14];
  const AXIS_DIRS = [[1,0,0],[0,1,0],[0,0,1]];

  // Three axis lines, each a 2-point geometry from origin to axis tip
  const axisGeos = AXIS_DIRS.map(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0,0,0, 0,0,0]), 3));
    return g;
  });
  const axisLineMeshes = axisGeos.map((g, i) => {
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: AXIS_OPACITIES[i] }));
    scene.add(line);
    return line;
  });

  // Axis labels at tip positions
  const axisLabelObjs = AXIS_DIRS.map(() => {
    const lbl = makeCss2DLabel('');
    scene.add(lbl);
    return lbl;
  });

  // Scatter points (BufferGeometry grows on demand, never shrinks)
  const circleTex = makeCircleTexture();
  const pointsCapRef = { cap: 0 };
  const pointsMesh = new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({ color: 0x4aa3ff, size: 6, sizeAttenuation: false, transparent: true, opacity: 0.9, map: circleTex, alphaTest: 0.5 }),
  );
  scene.add(pointsMesh);
  pointsMesh.renderOrder = 1;

  // Overlay points (rank reconstruction, hidden when no overlay)
  const overlayCapRef = { cap: 0 };
  const overlayMesh = new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({ color: 0xffc456, size: 5, sizeAttenuation: false, transparent: true, opacity: 0.9, map: circleTex, alphaTest: 0.5 }),
  );
  overlayMesh.visible = false;
  scene.add(overlayMesh);
  overlayMesh.renderOrder = 1;

  // PC vector arrows (ArrowHelper: from origin toward +v, arrowhead at positive tip)
  const arrows = PC_COLORS.map(color => {
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 1, color);
    arrow.visible = false;
    scene.add(arrow);
    return arrow;
  });

  // Vector tip labels
  const vecLabelObjs = PC_COLORS.map(() => {
    const lbl = makeCss2DLabel('');
    lbl.visible = false;
    scene.add(lbl);
    return lbl;
  });

  // Point number label pool -- grows to max-seen N, never shrinks
  const ptLabelPool = [];
  function getPtLabel(i) {
    if (i < ptLabelPool.length) return ptLabelPool[i];
    const lbl = makeCss2DLabel(String(i + 1));
    lbl.visible = false;
    scene.add(lbl);
    ptLabelPool.push(lbl);
    return lbl;
  }

  const GRID_DIVS = 4;
  const GRID_TICKS = GRID_DIVS + 1; // 5: at -1, -0.5, 0, 0.5, 1 (relative)
  const gridTickLabels = Array.from({ length: 3 * GRID_TICKS }, () => {
    const lbl = makeCss2DLabel('', 'pca-tick');
    lbl.visible = false;
    scene.add(lbl);
    return lbl;
  });

  let initialized = false;
  let lastBound = 0;
  let gridBound = 0;
  let gridHelpers = [];

  function updateGrid(safebound, displayScale) {
    if (Math.abs(safebound - gridBound) > 0.01) {
      gridHelpers.forEach(g => { scene.remove(g); g.geometry.dispose(); g.material.dispose(); });
      gridHelpers = [];
      const size = safebound * 2;
      const dimColor = 0x1e2840;
      const g1 = new THREE.GridHelper(size, GRID_DIVS, dimColor, dimColor);
      g1.position.y = -safebound; scene.add(g1);
      const g2 = new THREE.GridHelper(size, GRID_DIVS, dimColor, dimColor);
      g2.rotation.x = Math.PI / 2; g2.position.z = -safebound; scene.add(g2);
      const g3 = new THREE.GridHelper(size, GRID_DIVS, dimColor, dimColor);
      g3.rotation.z = Math.PI / 2; g3.position.x = -safebound; scene.add(g3);
      gridHelpers.push(g1, g2, g3);
      gridBound = safebound;
    }
    for (let ax = 0; ax < 3; ax++) {
      for (let t = 0; t < GRID_TICKS; t++) {
        const frac = t / GRID_DIVS; // 0..1
        const p = -safebound + frac * safebound * 2;
        const actualVal = (frac * 2 - 1) * displayScale[ax];
        const lbl = gridTickLabels[ax * GRID_TICKS + t];
        lbl.element.textContent = String(Number(actualVal.toFixed(2)));
        if (ax === 0) lbl.position.set(p, -safebound, -safebound);
        else if (ax === 1) lbl.position.set(-safebound, p, -safebound);
        else lbl.position.set(-safebound, -safebound, p);
        lbl.visible = true;
      }
    }
  }

  function update({ points, principalVectors, showVectors, showLabels, overlayPoints, axisLabels, basisLabels, bound, axisDisplayScale }) {
    const safebound = bound || 1;
    const displayScale = axisDisplayScale || [safebound, safebound, safebound];
    updateGrid(safebound, displayScale);
    const axisLen = safebound * 1.15;

    controls.maxDistance = safebound * 6;
    if (!initialized) {
      camera.position.setLength(Math.max(safebound * 3, 4));
      controls.update();
      initialized = true;
    } else if (safebound > lastBound * 1.5) {
      camera.position.setLength(camera.position.length() * (safebound / lastBound));
      controls.update();
    } else if (camera.position.length() > controls.maxDistance) {
      camera.position.setLength(controls.maxDistance);
      controls.update();
    }
    lastBound = safebound;

    // Axis lines and labels
    axisGeos.forEach((g, i) => {
      const pos = g.attributes.position;
      pos.setXYZ(1, AXIS_DIRS[i][0] * axisLen, AXIS_DIRS[i][1] * axisLen, AXIS_DIRS[i][2] * axisLen);
      pos.needsUpdate = true;
      axisLabelObjs[i].position.set(AXIS_DIRS[i][0] * axisLen, AXIS_DIRS[i][1] * axisLen, AXIS_DIRS[i][2] * axisLen);
      axisLabelObjs[i].element.textContent = axisLabels[i] || '';
    });

    // Scatter points
    writePoints(ensurePointsGeo(pointsMesh, points.length, pointsCapRef), points);

    // Overlay
    if (overlayPoints && overlayPoints.length) {
      writePoints(ensurePointsGeo(overlayMesh, overlayPoints.length, overlayCapRef), overlayPoints);
      overlayMesh.visible = true;
    } else {
      overlayMesh.visible = false;
    }

    // PC arrows and vector tip labels
    const dim = principalVectors.length;
    arrows.forEach((arrow, i) => {
      const lbl = vecLabelObjs[i];
      if (!showVectors || i >= dim) { arrow.visible = false; lbl.visible = false; return; }
      const v = principalVectors[i];
      const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
      if (len < 1e-6) { arrow.visible = false; lbl.visible = false; return; }
      const headLen = Math.min(len * 0.2, axisLen * 0.12);
      arrow.setDirection(new THREE.Vector3(v[0]/len, v[1]/len, v[2]/len));
      arrow.setLength(len, headLen, headLen * 0.6);
      arrow.visible = true;
      const labelText = basisLabels[i] || '';
      lbl.position.set(v[0], v[1], v[2]);
      lbl.element.textContent = labelText;
      lbl.visible = !!labelText;
    });

    // Point number labels (only when showLabels and N <= 60)
    const showPtLabels = showLabels && points.length <= 60;
    points.forEach((pt, i) => {
      const lbl = getPtLabel(i);
      if (!showPtLabels) { lbl.visible = false; return; }
      lbl.position.set(pt[0], pt[1], pt[2]);
      lbl.element.textContent = String(i + 1);
      lbl.visible = true;
    });
    for (let i = points.length; i < ptLabelPool.length; i++) ptLabelPool[i].visible = false;

    ctx.render();
  }

  function destroy() {
    ctx.controls.dispose();
    ctx.resizeObserver.disconnect();
    axisGeos.forEach(g => g.dispose());
    axisLineMeshes.forEach(l => l.material.dispose());
    pointsMesh.geometry.dispose();
    pointsMesh.material.dispose();
    overlayMesh.geometry.dispose();
    overlayMesh.material.dispose();
    arrows.forEach(a => {
      a.line.geometry.dispose();
      a.line.material.dispose();
      a.cone.geometry.dispose();
      a.cone.material.dispose();
    });
    circleTex.dispose();
    gridHelpers.forEach(g => { g.geometry.dispose(); g.material.dispose(); });
    gridTickLabels.forEach(lbl => scene.remove(lbl));
    ctx.renderer.dispose();
    if (container.contains(ctx.renderer.domElement)) container.removeChild(ctx.renderer.domElement);
    if (container.contains(ctx.css2d.domElement)) container.removeChild(ctx.css2d.domElement);
  }

  return { update, destroy, ...makeContextApi(ctx, container) };
}

// --- createOperatorPlot3D ---

export function createOperatorPlot3D(container) {
  const ctx = makePlot3DContext(container);
  const { scene, camera, controls } = ctx;

  const PC_COLORS = [0x7dffb2, 0xffc456, 0xff7a7a];

  // Latitude circles (5): blue, relatively opaque
  const latLines = LAT_DEGS.map(() => makeCircleLine(0x4aa3ff, 0.58));
  // Longitude circles (6): white, very faint
  const lonLines = LON_DEGS.map(() => makeCircleLine(0xffffff, 0.10));
  [...latLines, ...lonLines].forEach(l => scene.add(l));

  // Dynamic back-face grid frame
  const OP_GRID_DIVS = 4;
  const OP_GRID_TICKS = OP_GRID_DIVS + 1;
  let opGridBound = 0;
  let opGridMeshes = [];
  const opGridTickLabels = Array.from({ length: 3 * OP_GRID_TICKS }, () => {
    const lbl = makeCss2DLabel('', 'pca-tick');
    lbl.visible = false;
    scene.add(lbl);
    return lbl;
  });

  function updateOpGrid(safebound) {
    if (Math.abs(safebound - opGridBound) > 0.01) {
      opGridMeshes.forEach(g => { scene.remove(g); g.geometry.dispose(); g.material.dispose(); });
      opGridMeshes = [];
      const size = safebound * 2;
      const dimColor = 0x1e2840;
      const g1 = new THREE.GridHelper(size, OP_GRID_DIVS, dimColor, dimColor);
      g1.position.y = -safebound; scene.add(g1);
      const g2 = new THREE.GridHelper(size, OP_GRID_DIVS, dimColor, dimColor);
      g2.rotation.x = Math.PI / 2; g2.position.z = -safebound; scene.add(g2);
      const g3 = new THREE.GridHelper(size, OP_GRID_DIVS, dimColor, dimColor);
      g3.rotation.z = Math.PI / 2; g3.position.x = -safebound; scene.add(g3);
      opGridMeshes.push(g1, g2, g3);
      opGridBound = safebound;
    }
    for (let ax = 0; ax < 3; ax++) {
      for (let t = 0; t < OP_GRID_TICKS; t++) {
        const frac = t / OP_GRID_DIVS;
        const p = -safebound + frac * safebound * 2;
        const lbl = opGridTickLabels[ax * OP_GRID_TICKS + t];
        lbl.element.textContent = String(Number(p.toFixed(2)));
        if (ax === 0) lbl.position.set(p, -safebound, -safebound);
        else if (ax === 1) lbl.position.set(-safebound, p, -safebound);
        else lbl.position.set(-safebound, -safebound, p);
        lbl.visible = true;
      }
    }
  }

  // Basis axis lines (static, unit-sphere scale)
  const OP_AXIS_DIRS = [[1,0,0],[0,1,0],[0,0,1]];
  const OP_AXIS_OPACITIES = [0.22, 0.18, 0.14];
  const OP_AXIS_LEN = 1.3;
  const opAxisLineMeshes = OP_AXIS_DIRS.map((dir, i) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(
      new Float32Array([0, 0, 0, dir[0] * OP_AXIS_LEN, dir[1] * OP_AXIS_LEN, dir[2] * OP_AXIS_LEN]), 3
    ));
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: OP_AXIS_OPACITIES[i] }));
    scene.add(line);
    return line;
  });

  // PC vector arrows (transformed by covariance matrix)
  const arrows = PC_COLORS.map(color => {
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 1, color);
    arrow.visible = false;
    scene.add(arrow);
    return arrow;
  });

  // Eigenvalue labels at transformed arrow tips
  const eigenLabels = PC_COLORS.map(() => {
    const lbl = makeCss2DLabel('');
    lbl.visible = false;
    scene.add(lbl);
    return lbl;
  });

  // Static axis labels at unit radius along each axis
  ['x₁','x₂','x₃'].forEach((text, i) => {
    const lbl = makeCss2DLabel(text);
    lbl.position.set(i === 0 ? 1.4 : 0, i === 1 ? 1.4 : 0, i === 2 ? 1.4 : 0);
    scene.add(lbl);
  });

  let initialized = false;

  function update({ transform, principalVectors, lambda, showVectors, bound }) {
    const safeBound = Math.max(1.5, bound || 1.5);
    updateOpGrid(safeBound);
    controls.maxDistance = safeBound * 5;
    if (!initialized) {
      camera.position.setLength(Math.max(safeBound * 3, 4));
      controls.update();
      initialized = true;
    } else if (camera.position.length() > controls.maxDistance) {
      camera.position.setLength(controls.maxDistance);
      controls.update();
    }

    updateWireframe(latLines, lonLines, transform);

    const dim = principalVectors.length;
    arrows.forEach((arrow, i) => {
      const lbl = eigenLabels[i];
      if (!showVectors || i >= dim) { arrow.visible = false; lbl.visible = false; return; }
      const tv = applyMat3(transform, principalVectors[i]);
      const len = Math.sqrt(tv[0]*tv[0] + tv[1]*tv[1] + tv[2]*tv[2]);
      if (len < 1e-6) { arrow.visible = false; lbl.visible = false; return; }
      const headLen = len * 0.2;
      arrow.setDirection(new THREE.Vector3(tv[0]/len, tv[1]/len, tv[2]/len));
      arrow.setLength(len, headLen, headLen * 0.6);
      arrow.visible = true;
      lbl.position.set(tv[0], tv[1], tv[2]);
      lbl.element.textContent = 'λ' + (i+1) + '=' + (lambda[i] || 0).toFixed(2);
      lbl.visible = true;
    });

    ctx.render();
  }

  function destroy() {
    ctx.controls.dispose();
    ctx.resizeObserver.disconnect();
    [...latLines, ...lonLines].forEach(l => { l.geometry.dispose(); l.material.dispose(); });
    opGridMeshes.forEach(g => { g.geometry.dispose(); g.material.dispose(); });
    opGridTickLabels.forEach(lbl => scene.remove(lbl));
    opAxisLineMeshes.forEach(l => { l.geometry.dispose(); l.material.dispose(); });
    arrows.forEach(a => {
      a.line.geometry.dispose();
      a.line.material.dispose();
      a.cone.geometry.dispose();
      a.cone.material.dispose();
    });
    ctx.renderer.dispose();
    if (container.contains(ctx.renderer.domElement)) container.removeChild(ctx.renderer.domElement);
    if (container.contains(ctx.css2d.domElement)) container.removeChild(ctx.css2d.domElement);
  }

  return { update, destroy, ...makeContextApi(ctx, container) };
}
