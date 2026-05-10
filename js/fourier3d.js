(function () {
  "use strict";

  const N = 256;

  document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('fourier3dCanvas');
    if (!canvas) return;

    if (typeof THREE === 'undefined') {
      console.error('[fourier3d] Three.js failed to load — 3D Surface View disabled.');
      return;
    }
    if (typeof THREE.OrbitControls === 'undefined') {
      console.error('[fourier3d] OrbitControls failed to load — 3D Surface View disabled.');
      return;
    }

    // ---- Scene ----
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d0d14);
    scene.fog = new THREE.FogExp2(0x0d0d14, 0.0028);

    // ---- Renderer ----
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // ---- Camera ----
    const camera = new THREE.PerspectiveCamera(45, 1, 1, 1200);
    camera.position.set(0, 180, 260);

    // ---- Controls: orbit around X and Y axes ----
    const controls = new THREE.OrbitControls(camera, canvas);
    controls.target.set(0, 28, 0);
    controls.enablePan = false;
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 80;
    controls.maxDistance = 680;
    controls.minPolarAngle = 0.02;
    controls.maxPolarAngle = Math.PI * 0.82;
    controls.update();

    // ---- Height map geometry ----
    // PlaneGeometry lies in XY plane; rotateX puts it in XZ plane with Y = height.
    // Vertex index for image pixel (row r, col c) = r * N + c.
    const geometry = new THREE.PlaneGeometry(N, N, N - 1, N - 1);
    geometry.rotateX(-Math.PI / 2);

    const colorData = new Float32Array(N * N * 3);
    geometry.setAttribute('color', new THREE.BufferAttribute(colorData, 3));

    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      shininess: 55,
      specular: new THREE.Color(0x3a3a3a),
    });

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    // Subtle reference grid at ground level
    const grid = new THREE.GridHelper(N, 16, 0x1a2a3a, 0x0f1820);
    grid.position.y = -1;
    scene.add(grid);

    // ---- Lighting ----
    scene.add(new THREE.AmbientLight(0xffffff, 0.50));

    const sun = new THREE.DirectionalLight(0xfff8e8, 0.80);
    sun.position.set(160, 380, 200);
    scene.add(sun);

    const rim = new THREE.DirectionalLight(0x7799cc, 0.40);
    rim.position.set(-200, 90, -220);
    scene.add(rim);

    // ---- State ----
    let heightScale = 60;
    let showSource  = 'recon';
    let origPixels  = null;
    let reconPixels = null;
    let currentPixels = null;

    // ---- Apply pixel array to height map ----
    // Color grading: 0 → black, 255 → white linear grayscale in-range;
    // values below 0 ramp into blue, values above 255 ramp into red.
    // Same convention as the 2D `renderExtended` view so Gibbs overshoot
    // and undershoot are visually identifiable across both panels.
    function applyPixels(pixels) {
      currentPixels = pixels;
      const pos = geometry.attributes.position;
      const col = geometry.attributes.color;

      for (let i = 0; i < N * N; i++) {
        const v = pixels[i];

        // Height anchored to the 0–255 scale so amplitude is physically meaningful.
        pos.setY(i, (v / 255) * heightScale);

        let r, g, b;
        if (v < 0) {
          // Undershoot: blue, intensity ramps with magnitude (capped).
          r = 0;
          g = 0;
          b = (80 + Math.min(175, -v * 0.4)) / 255;
        } else if (v > 255) {
          // Overshoot: red, intensity ramps with magnitude (capped).
          r = (80 + Math.min(175, (v - 255) * 0.4)) / 255;
          g = 0;
          b = 0;
        } else {
          // In-range: linear grayscale.
          const t = v / 255;
          r = g = b = t;
        }
        col.setXYZ(i, r, g, b);
      }

      pos.needsUpdate = true;
      col.needsUpdate = true;
      geometry.computeVertexNormals();
    }

    // ---- Controls wiring ----
    const scaleSl  = document.getElementById('fourier3dScale');
    const scaleOut = document.getElementById('fourier3dScaleOut');

    if (scaleSl) {
      scaleSl.addEventListener('input', () => {
        heightScale = +scaleSl.value;
        if (scaleOut) scaleOut.textContent = heightScale;
        if (currentPixels) applyPixels(currentPixels);
      });
    }

    document.querySelectorAll('input[name="fourier3dSource"]').forEach(radio => {
      radio.addEventListener('change', () => {
        showSource = radio.value;
        const px = showSource === 'orig' ? origPixels : reconPixels;
        if (px) applyPixels(px);
      });
    });

    // ---- Periodic tiling (InstancedMesh) ----
    let tilingEnabled = false;
    let tileExtent    = 5;       // odd → symmetric grid centered on origin
    let instancedMesh = null;

    const DEFAULT_MAX_DIST = 680;

    function rebuildTiling() {
      if (instancedMesh) {
        scene.remove(instancedMesh);
        if (typeof instancedMesh.dispose === 'function') instancedMesh.dispose();
        instancedMesh = null;
      }

      if (!tilingEnabled) {
        mesh.visible        = true;
        controls.enablePan  = false;
        controls.maxDistance = DEFAULT_MAX_DIST;
        return;
      }

      // Hide the single-tile mesh and replace with a grid of instances.
      // All instances share the geometry, so height-map updates propagate to all tiles automatically.
      mesh.visible = false;

      const count = tileExtent * tileExtent;
      instancedMesh = new THREE.InstancedMesh(geometry, material, count);
      instancedMesh.frustumCulled = false; // tiles span far beyond the origin bounding sphere

      const matrix = new THREE.Matrix4();
      const half   = (tileExtent - 1) / 2;
      let i = 0;
      for (let dz = -half; dz <= half; dz++) {
        for (let dx = -half; dx <= half; dx++) {
          matrix.makeTranslation(dx * N, 0, dz * N);
          instancedMesh.setMatrixAt(i++, matrix);
        }
      }
      instancedMesh.instanceMatrix.needsUpdate = true;
      scene.add(instancedMesh);

      controls.enablePan   = true;
      controls.maxDistance = Math.max(DEFAULT_MAX_DIST, tileExtent * N * 1.2);
    }

    const tileChk = document.getElementById('fourier3dTile');
    const tileSl  = document.getElementById('fourier3dTileExtent');
    const tileOut = document.getElementById('fourier3dTileExtentOut');

    function updateTileLabel() {
      if (!tileOut) return;
      const total = tileExtent * tileExtent;
      tileOut.textContent = tilingEnabled
        ? `${tileExtent} × ${tileExtent} = ${total} tiles`
        : `${tileExtent} × ${tileExtent} = ${total} tiles (off)`;
    }

    if (tileChk) {
      tileChk.addEventListener('change', () => {
        tilingEnabled = tileChk.checked;
        if (tileSl) tileSl.disabled = !tilingEnabled;
        rebuildTiling();
        updateTileLabel();
      });
    }

    if (tileSl) {
      tileSl.addEventListener('input', () => {
        tileExtent = +tileSl.value;
        if (tilingEnabled) rebuildTiling();
        updateTileLabel();
      });
    }

    // ---- Listen for main-demo pixel updates ----
    window.addEventListener('fourier-orig-update', e => {
      origPixels = e.detail.pixels;
      if (showSource === 'orig') applyPixels(origPixels);
    });

    window.addEventListener('fourier-recon-update', e => {
      reconPixels = e.detail.pixels;
      if (showSource === 'recon') applyPixels(reconPixels);
    });

    // ---- Pick up any pixels already dispatched before we subscribed ----
    // (fourier.js init() runs first because it loads first; without this
    //  the 3D view would stay flat-black until the user moves a slider.)
    const cached = window.__fourierState;
    if (cached) {
      if (cached.recon) reconPixels = cached.recon;
      if (cached.orig)  origPixels  = cached.orig;
      const initial = (showSource === 'orig' ? origPixels : reconPixels) || origPixels || reconPixels;
      if (initial) applyPixels(initial);
    }

    // ---- Resize ----
    function syncSize() {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight || Math.round(w * 9 / 16);
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    const ro = new ResizeObserver(syncSize);
    ro.observe(canvas);
    syncSize();

    // ---- Render loop ----
    (function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    })();
  });
})();
