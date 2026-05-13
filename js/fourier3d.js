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
    const hasFlyControls = typeof THREE.PointerLockControls !== 'undefined';
    if (!hasFlyControls) {
      console.warn('[fourier3d] PointerLockControls failed to load — Fly mode disabled.');
    }

    // ---- Scene ----
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d0d14);
    scene.fog = new THREE.FogExp2(0x0d0d14, 0.0028);

    // ---- Renderer ----
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // ---- Camera ----
    // Far plane is driven by fog density (see setFogDensity below): dense fog
    // means anything past the fog cutoff is invisible anyway, so we can clip
    // aggressively and save fragment shader work.
    const FAR_MIN = 200;
    const FAR_MAX = 20000;
    const camera = new THREE.PerspectiveCamera(45, 1, 1, FAR_MAX);
    camera.position.set(0, 180, 260);

    function farForDensity(density) {
      if (density <= 0) return FAR_MAX;
      // FogExp2 attenuation = exp(-(density * d)^2). Solving for 99% opacity
      // (factor = 0.01) gives d ≈ sqrt(ln(100)) / density ≈ 2.146 / density.
      // We push the far plane 30% past that so the smooth fog ramp hides the
      // clip plane.
      const cutoff = 2.146 / density;
      return Math.max(FAR_MIN, Math.min(FAR_MAX, cutoff * 1.3));
    }

    function setFogDensity(density) {
      if (scene.fog) scene.fog.density = density;
      camera.far = farForDensity(density);
      camera.updateProjectionMatrix();
    }

    setFogDensity(scene.fog ? scene.fog.density : 0);

    // ---- Controls: orbit around X and Y axes ----
    const controls = new THREE.OrbitControls(camera, canvas);
    controls.target.set(0, 0, 0);
    controls.enablePan = false;
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 80;
    controls.maxDistance = 680;
    controls.minPolarAngle = 0.02;
    controls.maxPolarAngle = Math.PI * 0.82;
    controls.update();

    // ---- Fly mode (WASD + pointer-lock mouse-look) ----
    // PointerLockControls handles mouse-look while the pointer is locked; we
    // drive translation manually from a key-state map. OrbitControls is left
    // attached but disabled while flying, so toggling back resumes cleanly.
    let cameraMode = 'orbit';
    const flyControls = hasFlyControls
      ? new THREE.PointerLockControls(camera, canvas)
      : null;
    const keys = { w: false, a: false, s: false, d: false, q: false, e: false, shift: false };
    const clock = new THREE.Clock();
    const FLY_BASE_SPEED = 80;   // units per second
    const FLY_SPRINT_MULT = 3;

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
    let hideOutOfRange = false;

    // Save the original index so toggling can restore it. Pre-allocate the
    // filtered buffer once to avoid per-frame allocation pressure.
    const originalIndexAttr  = geometry.index;
    const originalIndexArray = originalIndexAttr.array;
    const filteredIndexBuffer = new originalIndexArray.constructor(originalIndexArray.length);

    // Drop any triangle whose vertices aren't all in [0, 255]. The
    // InstancedMesh shares this geometry, so the clip applies to every tile.
    function applyIndexFilter(pixels) {
      if (!hideOutOfRange) {
        if (geometry.index !== originalIndexAttr) geometry.setIndex(originalIndexAttr);
        return;
      }

      let writePos = 0;
      const len = originalIndexArray.length;
      for (let t = 0; t < len; t += 3) {
        const a = originalIndexArray[t];
        const b = originalIndexArray[t + 1];
        const c = originalIndexArray[t + 2];
        const va = pixels[a], vb = pixels[b], vc = pixels[c];
        if (va >= 0 && va <= 255 && vb >= 0 && vb <= 255 && vc >= 0 && vc <= 255) {
          filteredIndexBuffer[writePos++] = a;
          filteredIndexBuffer[writePos++] = b;
          filteredIndexBuffer[writePos++] = c;
        }
      }
      geometry.setIndex(new THREE.BufferAttribute(filteredIndexBuffer.subarray(0, writePos), 1));
    }

    // ---- Apply pixel array to height map ----
    // Color grading: 0 → black, 255 → white linear grayscale in-range;
    // values below 0 ramp into blue, values above 255 ramp into red.
    // Same convention as the 2D `renderExtended` view so Gibbs overshoot
    // and undershoot are visually identifiable across both panels.
    //
    // Out-of-range heights are compressed logarithmically. High-degree
    // Legendre/Chebyshev reconstructions can produce pixel values up to
    // 10^20+ on the extended domain; mapped linearly those would stretch
    // triangles across the entire screen and cripple the fragment shader.
    // The OOR_COMPRESSION factor sets how much vertical room each decade
    // of overshoot occupies, measured in units of heightScale.
    const OOR_COMPRESSION = 0.25;

    function applyPixels(pixels) {
      currentPixels = pixels;
      const pos = geometry.attributes.position;
      const col = geometry.attributes.color;

      for (let i = 0; i < N * N; i++) {
        const v = pixels[i];

        let h;
        if (v < 0) {
          const excess = Math.min(-v, 1e20);
          h = -OOR_COMPRESSION * heightScale * Math.log10(1 + excess);
        } else if (v > 255) {
          const excess = Math.min(v - 255, 1e20);
          h = heightScale + OOR_COMPRESSION * heightScale * Math.log10(1 + excess);
        } else {
          h = (v / 255) * heightScale;
        }
        pos.setY(i, h);

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
      applyIndexFilter(pixels);
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

    const fogSl  = document.getElementById('fourier3dFog');
    const fogOut = document.getElementById('fourier3dFogOut');
    const FOG_DENSITY_PER_UNIT = 0.0001;

    if (fogSl) {
      fogSl.addEventListener('input', () => {
        const v = +fogSl.value;
        if (fogOut) fogOut.textContent = v;
        setFogDensity(v * FOG_DENSITY_PER_UNIT);
      });
    }

    document.querySelectorAll('input[name="fourier3dSource"]').forEach(radio => {
      radio.addEventListener('change', () => {
        showSource = radio.value;
        const px = showSource === 'orig' ? origPixels : reconPixels;
        if (px) applyPixels(px);
      });
    });

    // ---- Camera mode (Orbit vs Fly) ----
    const flyHint = document.getElementById('fourier3dFlyHint');
    const flyRadio = document.querySelector('input[name="fourier3dCamera"][value="fly"]');

    if (!hasFlyControls && flyRadio) flyRadio.disabled = true;

    function setCameraMode(mode) {
      if (mode === cameraMode) return;
      if (mode === 'fly' && !hasFlyControls) return;

      if (mode === 'fly') {
        controls.enabled = false;
        if (flyHint) flyHint.style.display = '';
      } else {
        if (flyControls && flyControls.isLocked) flyControls.unlock();
        // Orbit always pivots around the origin so the heightmap stays
        // centered in view regardless of where fly mode left the camera.
        controls.target.set(0, 0, 0);
        controls.enabled = true;
        if (flyHint) flyHint.style.display = 'none';
        for (const k in keys) keys[k] = false;
      }
      cameraMode = mode;
    }

    document.querySelectorAll('input[name="fourier3dCamera"]').forEach(radio => {
      radio.addEventListener('change', () => setCameraMode(radio.value));
    });

    canvas.addEventListener('click', () => {
      if (cameraMode !== 'fly' || !flyControls) return;
      if (flyControls.isLocked) flyControls.unlock();
      else flyControls.lock();
    });

    window.addEventListener('keydown', e => {
      if (cameraMode !== 'fly') return;
      if (e.key === 'Shift') { keys.shift = true; return; }
      const k = e.key.toLowerCase();
      if (k in keys) { keys[k] = true; e.preventDefault(); }
    });

    window.addEventListener('keyup', e => {
      if (e.key === 'Shift') { keys.shift = false; return; }
      const k = e.key.toLowerCase();
      if (k in keys) keys[k] = false;
    });

    // Clear stuck keys if the window loses focus mid-press.
    window.addEventListener('blur', () => {
      for (const k in keys) keys[k] = false;
    });

    const hideChk = document.getElementById('fourier3dHideOOR');
    if (hideChk) {
      hideChk.addEventListener('change', () => {
        hideOutOfRange = hideChk.checked;
        if (currentPixels) applyPixels(currentPixels);
      });
    }

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
    const tileRow = document.getElementById('fourier3dTileExtentRow');

    function updateTileLabel() {
      if (!tileOut) return;
      const total = tileExtent * tileExtent;
      tileOut.textContent = `${tileExtent} × ${tileExtent} = ${total} tiles`;
    }

    if (tileChk) {
      tileChk.addEventListener('change', () => {
        tilingEnabled = tileChk.checked;
        if (tileSl) tileSl.disabled = !tilingEnabled;
        if (tileRow) tileRow.style.display = tilingEnabled ? '' : 'none';
        rebuildTiling();
        updateTileLabel();
        syncBasisWarning();
      });
    }

    if (tileSl) {
      tileSl.addEventListener('input', () => {
        tileExtent = +tileSl.value;
        if (tilingEnabled) rebuildTiling();
        updateTileLabel();
      });
    }

    // ---- Basis-aware tiling warning ----
    // The tiling control stays usable for every basis (mirroring the 2D demo,
    // which also lets you toggle its extension/extrapolation view freely).
    // For Legendre, Chebyshev, and Haar we surface a warning so the user
    // knows the replication is visual only, not implied by the basis.
    const basisSel = document.getElementById('fourierBasis');
    const tileNote = document.getElementById('fourier3dTileBasisNote');

    function syncBasisWarning() {
      const isFourier = !basisSel || basisSel.value === 'fourier';
      const show = !isFourier && tilingEnabled;
      if (tileNote) tileNote.style.display = show ? '' : 'none';
    }

    if (basisSel) basisSel.addEventListener('change', syncBasisWarning);
    syncBasisWarning();

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
      const dt = clock.getDelta();
      if (cameraMode === 'fly' && flyControls) {
        const speed = FLY_BASE_SPEED * (keys.shift ? FLY_SPRINT_MULT : 1) * dt;
        if (keys.w) flyControls.moveForward(speed);
        if (keys.s) flyControls.moveForward(-speed);
        if (keys.a) flyControls.moveRight(-speed);
        if (keys.d) flyControls.moveRight(speed);
        if (keys.e) camera.position.y += speed;
        if (keys.q) camera.position.y -= speed;
      } else {
        controls.update();
      }
      renderer.render(scene, camera);
    })();
  });
})();
