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
    function applyPixels(pixels) {
      currentPixels = pixels;
      const pos = geometry.attributes.position;
      const col = geometry.attributes.color;

      // Normalize color to the actual data range for maximum contrast.
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < N * N; i++) {
        if (pixels[i] < lo) lo = pixels[i];
        if (pixels[i] > hi) hi = pixels[i];
      }
      const span = hi - lo || 1;

      for (let i = 0; i < N * N; i++) {
        const v = pixels[i];
        // Height anchored to the 0–255 scale so amplitude is physically meaningful.
        pos.setY(i, (v / 255) * heightScale);
        // Grayscale color stretched over actual range.
        const t = Math.max(0, Math.min(1, (v - lo) / span));
        col.setXYZ(i, t, t, t);
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
