const VIRIDIS = ['#000000', '#3a1a6e', '#5b3a8c', '#8b5fbf', '#c179d3', '#e8a37f', '#f5cf6e', '#f9eb6b'];

function matmul(A, B) {
  const C = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
  return C;
}
function rotX(a) { const c = Math.cos(a), s = Math.sin(a); return [[1,0,0],[0,c,-s],[0,s,c]]; }
function rotY(a) { const c = Math.cos(a), s = Math.sin(a); return [[c,0,s],[0,1,0],[-s,0,c]]; }

function projectVec(R, v, scale, cx, cy) {
  const px = R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2];
  const py = R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2];
  return [cx + scale * px, cy - scale * py];
}

function computeBounds(points) {
  const N = points.length / 3;
  let xmn = Infinity, xmx = -Infinity, ymn = Infinity, ymx = -Infinity, zmn = Infinity, zmx = -Infinity;
  for (let i = 0; i < N; i++) {
    const x = points[i * 3], y = points[i * 3 + 1], z = points[i * 3 + 2];
    if (x < xmn) xmn = x; if (x > xmx) xmx = x;
    if (y < ymn) ymn = y; if (y > ymx) ymx = y;
    if (z < zmn) zmn = z; if (z > zmx) zmx = z;
  }
  const r = Math.max(xmx - xmn, ymx - ymn, zmx - zmn, 1e-6) / 2;
  return { ax: (xmn + xmx) / 2, ay: (ymn + ymx) / 2, az: (zmn + zmx) / 2, r };
}

function colorFromValue(v, lo, hi) {
  const u = Math.max(0, Math.min(1, (v - lo) / Math.max(1e-12, hi - lo)));
  const idx = u * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 1, Math.floor(idx));
  return VIRIDIS[i];
}

function attachOrbit(svg, getR, setR, redraw) {
  let dragging = false, lastX = 0, lastY = 0;
  svg.style('cursor', 'grab').style('touch-action', 'none');
  svg.on('pointerdown', (event) => {
    dragging = true; lastX = event.clientX; lastY = event.clientY;
    svg.style('cursor', 'grabbing');
    try { svg.node().setPointerCapture(event.pointerId); } catch (e) {}
  });
  svg.on('pointermove', (event) => {
    if (!dragging) return;
    const dx = (event.clientX - lastX) * 0.008;
    const dy = (event.clientY - lastY) * 0.008;
    lastX = event.clientX; lastY = event.clientY;
    setR(matmul(matmul(rotX(dy), rotY(dx)), getR()));
    redraw();
  });
  function endDrag(event) {
    dragging = false; svg.style('cursor', 'grab');
    try { svg.node().releasePointerCapture(event.pointerId); } catch (e) {}
  }
  svg.on('pointerup', endDrag);
  svg.on('pointercancel', endDrag);
  svg.on('pointerleave', endDrag);
}

function mountPcaSpectral(svg, state, width, height) {
  const points = state.points;
  const { ax, ay, az, r } = computeBounds(points);
  let R = [[0.886, 0.0, 0.464], [-0.163, 0.937, 0.311], [-0.434, -0.348, 0.831]];
  const scale = (Math.min(width, height) / 2 - 28) / r;
  const cx = width / 2, cy = height / 2;

  const gCloud = svg.append('g');
  const gAxes = svg.append('g');

  const colors = ['#ff9f43', '#54a0ff', '#6bd47b'];
  const labels = ['PC1', 'PC2', 'PC3'];

  function redraw() {
    gCloud.html('');
    gAxes.html('');
    const N = points.length / 3;
    for (let i = 0; i < N; i++) {
      const x = points[i * 3] - ax, y = points[i * 3 + 1] - ay, z = points[i * 3 + 2] - az;
      const px = R[0][0]*x + R[0][1]*y + R[0][2]*z;
      const py = R[1][0]*x + R[1][1]*y + R[1][2]*z;
      gCloud.append('circle').attr('cx', cx + scale*px).attr('cy', cy - scale*py)
        .attr('r', 2.4).attr('fill', 'rgba(255,255,255,0.55)');
    }
    if (state.pcAxes) {
      const lam = state.pcAxes.lambda || [1, 1, 1];
      const top = Math.max(Math.abs(lam[0]) || 1, 1e-9);
      ['v1', 'v2', 'v3'].forEach((key, k) => {
        const v = state.pcAxes[key];
        if (!v) return;
        const len = r * 0.95 * Math.sqrt(Math.max(0, Math.abs(lam[k] || 0)) / top);
        const end = projectVec(R, [v[0] * len, v[1] * len, v[2] * len], scale, cx, cy);
        gAxes.append('line').attr('x1', cx).attr('y1', cy).attr('x2', end[0]).attr('y2', end[1])
          .attr('stroke', colors[k]).attr('stroke-width', 2.2)
          .attr('marker-end', `url(#pc-arrow-${k})`);
        gAxes.append('text').attr('x', end[0] + 6).attr('y', end[1] - 4)
          .attr('fill', colors[k]).attr('font-size', '11').text(labels[k]);
      });
    }
  }

  const defs = svg.append('defs');
  for (let k = 0; k < 3; k++) {
    defs.append('marker').attr('id', `pc-arrow-${k}`).attr('viewBox', '0 0 10 10').attr('refX', 9)
      .attr('refY', 5).attr('markerWidth', 5).attr('markerHeight', 5).attr('orient', 'auto-start-reverse')
      .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z').attr('fill', colors[k]);
  }
  redraw();
  attachOrbit(svg, () => R, (newR) => { R = newR; }, redraw);
}

function mountIsomapSpectral(svg, state, width, height) {
  const points = state.points;
  const { ax, ay, az, r } = computeBounds(points);
  let R = [[0.886, 0.0, 0.464], [-0.163, 0.937, 0.311], [-0.434, -0.348, 0.831]];
  const scale = (Math.min(width, height) / 2 - 24) / r;
  const cx = width / 2, cy = height / 2;

  const v1 = state.v1Values || new Float64Array(points.length / 3);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < v1.length; i++) { if (v1[i] < lo) lo = v1[i]; if (v1[i] > hi) hi = v1[i]; }

  const gCloud = svg.append('g');

  function redraw() {
    gCloud.html('');
    const N = points.length / 3;
    for (let i = 0; i < N; i++) {
      const x = points[i * 3] - ax, y = points[i * 3 + 1] - ay, z = points[i * 3 + 2] - az;
      const px = R[0][0]*x + R[0][1]*y + R[0][2]*z;
      const py = R[1][0]*x + R[1][1]*y + R[1][2]*z;
      gCloud.append('circle').attr('cx', cx + scale*px).attr('cy', cy - scale*py)
        .attr('r', 2.6).attr('fill', colorFromValue(v1[i], lo, hi));
    }
  }
  redraw();
  attachOrbit(svg, () => R, (newR) => { R = newR; }, redraw);

  const mini = svg.append('g').attr('transform', `translate(${width - 150}, ${height - 90})`);
  mini.append('rect').attr('width', 142).attr('height', 80).attr('fill', 'rgba(0,0,0,0.7)')
    .attr('stroke', 'rgba(255,255,255,0.25)').attr('rx', 4);
  mini.append('text').attr('x', 71).attr('y', 12).attr('text-anchor', 'middle')
    .attr('fill', 'rgba(255,255,255,0.85)').attr('font-size', '9').text('eigenvalues');
  const eig = state.topEigvals || new Float64Array(8);
  let lam0 = 1;
  for (let i = 0; i < eig.length; i++) lam0 = Math.max(lam0, Math.abs(eig[i]));
  const barW = 14;
  const maxH = 56;
  for (let i = 0; i < 8; i++) {
    const v = Math.abs(eig[i] || 0);
    const h = maxH * (v / Math.max(1e-9, lam0));
    mini.append('rect').attr('x', 10 + i * barW).attr('y', 18 + (maxH - h))
      .attr('width', barW - 2).attr('height', h)
      .attr('fill', i < 2 ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.5)');
  }
}

export function mountSpectral(container, state, { width = 480, height = 360 } = {}) {
  const d3 = window.d3;
  const wrap = d3.select(container).append('div').attr('class', 'viz-spectral');
  const svg = wrap.append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%').style('height', '100%');

  if (state.algoId === 'pca') mountPcaSpectral(svg, state, width, height);
  else if (state.algoId === 'isomap') mountIsomapSpectral(svg, state, width, height);

  return {
    unmount() { wrap.remove(); }
  };
}
