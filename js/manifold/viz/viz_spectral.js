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

  const N = points.length / 3;
  const axesV = state.pcAxes ? [state.pcAxes.v1, state.pcAxes.v2, state.pcAxes.v3] : [];
  const lam = (state.pcAxes && state.pcAxes.lambda) || [1, 1, 1];
  // Per-point projection onto each principal component (the per-point form of
  // the covariance eigenvector), used to color the cloud when a bar is hovered.
  let mx = 0, my = 0, mz = 0;
  for (let i = 0; i < N; i++) { mx += points[i * 3]; my += points[i * 3 + 1]; mz += points[i * 3 + 2]; }
  mx /= N; my /= N; mz /= N;
  const proj = axesV.map(v => {
    const arr = new Float64Array(N);
    if (v) for (let i = 0; i < N; i++) {
      arr[i] = (points[i * 3] - mx) * v[0] + (points[i * 3 + 1] - my) * v[1] + (points[i * 3 + 2] - mz) * v[2];
    }
    return arr;
  });
  let highlightedK = null;

  function redraw() {
    gCloud.html('');
    gAxes.html('');
    const vec = highlightedK !== null ? proj[highlightedK] : null;
    let lo = Infinity, hi = -Infinity;
    if (vec) for (let i = 0; i < N; i++) { if (vec[i] < lo) lo = vec[i]; if (vec[i] > hi) hi = vec[i]; }
    for (let i = 0; i < N; i++) {
      const x = points[i * 3] - ax, y = points[i * 3 + 1] - ay, z = points[i * 3 + 2] - az;
      const px = R[0][0]*x + R[0][1]*y + R[0][2]*z;
      const py = R[1][0]*x + R[1][1]*y + R[1][2]*z;
      gCloud.append('circle').attr('cx', cx + scale*px).attr('cy', cy - scale*py)
        .attr('r', 2.4).attr('fill', vec ? colorFromValue(vec[i], lo, hi) : 'rgba(255,255,255,0.55)');
    }
    if (state.pcAxes) {
      const top = Math.max(Math.abs(lam[0]) || 1, 1e-9);
      axesV.forEach((v, k) => {
        if (!v) return;
        const len = r * 0.95 * Math.sqrt(Math.max(0, Math.abs(lam[k] || 0)) / top);
        const end = projectVec(R, [v[0] * len, v[1] * len, v[2] * len], scale, cx, cy);
        const emph = highlightedK === k;
        const dim = highlightedK !== null && !emph;
        gAxes.append('line').attr('x1', cx).attr('y1', cy).attr('x2', end[0]).attr('y2', end[1])
          .attr('stroke', colors[k]).attr('stroke-width', emph ? 3.4 : 2.2).attr('opacity', dim ? 0.25 : 1)
          .attr('marker-end', `url(#pc-arrow-${k})`);
        gAxes.append('text').attr('x', end[0] + 6).attr('y', end[1] - 4)
          .attr('fill', colors[k]).attr('font-size', '11').attr('opacity', dim ? 0.25 : 1).text(labels[k]);
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

  const nb = Math.min(3, axesV.length || 3);
  const mini = svg.append('g').attr('transform', `translate(${width - 168}, ${height - 102})`);
  mini.append('rect').attr('width', 160).attr('height', 96).attr('fill', 'rgba(0,0,0,0.72)')
    .attr('stroke', 'rgba(255,255,255,0.25)').attr('rx', 4);
  mini.append('text').attr('x', 80).attr('y', 12).attr('text-anchor', 'middle')
    .attr('fill', 'rgba(255,255,255,0.85)').attr('font-size', '9').text('eigenvalues');
  const valueLabel = mini.append('text').attr('x', 80).attr('y', 90).attr('text-anchor', 'middle')
    .attr('fill', 'rgba(255,255,255,0.95)').attr('font-size', '10').text('');

  let lam0 = 1;
  for (let i = 0; i < nb; i++) lam0 = Math.max(lam0, Math.abs(lam[i] || 0));
  const barW = 30;
  const maxH = 52;
  const baseFill = (i) => i < 2 ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.5)';
  const bars = [];
  for (let i = 0; i < nb; i++) {
    const v = Math.abs(lam[i] || 0);
    const h = maxH * (v / Math.max(1e-9, lam0));
    const bar = mini.append('rect').attr('x', 16 + i * (barW + 6)).attr('y', 18 + (maxH - h))
      .attr('width', barW).attr('height', h)
      .attr('fill', baseFill(i)).style('cursor', 'pointer');
    const hitBar = mini.append('rect').attr('x', 16 + i * (barW + 6)).attr('y', 18)
      .attr('width', barW).attr('height', maxH)
      .attr('fill', 'transparent').style('cursor', 'pointer');
    bars.push(bar);
    const k = i;
    function onEnter() {
      highlightedK = k;
      bars.forEach((b, j) => b.attr('fill', j === k ? colors[k] : baseFill(j)));
      valueLabel.text('λ_' + (k + 1) + ' = ' + (lam[k] || 0).toFixed(3));
      redraw();
    }
    function onLeave() {
      highlightedK = null;
      bars.forEach((b, j) => b.attr('fill', baseFill(j)));
      valueLabel.text('');
      redraw();
    }
    bar.on('mouseenter', onEnter).on('mouseleave', onLeave);
    hitBar.on('mouseenter', onEnter).on('mouseleave', onLeave);
  }
}

function mountIsomapSpectral(svg, state, width, height) {
  const points = state.points;
  const { ax, ay, az, r } = computeBounds(points);
  let R = [[0.886, 0.0, 0.464], [-0.163, 0.937, 0.311], [-0.434, -0.348, 0.831]];
  const scale = (Math.min(width, height) / 2 - 24) / r;
  const cx = width / 2, cy = height / 2;

  const v1 = state.v1Values || new Float64Array(points.length / 3);
  const allVecs = state.topEigvecs || [v1];

  const gCloud = svg.append('g');
  const gArrow = svg.append('g');
  const N = points.length / 3;
  let mx = 0, my = 0, mz = 0;
  for (let i = 0; i < N; i++) { mx += points[i * 3]; my += points[i * 3 + 1]; mz += points[i * 3 + 2]; }
  mx /= N; my /= N; mz /= N;
  const defs = svg.append('defs');
  defs.append('marker').attr('id', 'spec-arrow').attr('viewBox', '0 0 10 10').attr('refX', 9)
    .attr('refY', 5).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto-start-reverse')
    .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z').attr('fill', '#ff9f43');
  let highlightedK = null;

  function activeVec() {
    if (highlightedK !== null && allVecs[highlightedK]) return allVecs[highlightedK];
    return v1;
  }

  function redraw() {
    gCloud.html('');
    const vec = activeVec();
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < vec.length; i++) { if (vec[i] < lo) lo = vec[i]; if (vec[i] > hi) hi = vec[i]; }
    for (let i = 0; i < N; i++) {
      const x = points[i * 3] - ax, y = points[i * 3 + 1] - ay, z = points[i * 3 + 2] - az;
      const px = R[0][0]*x + R[0][1]*y + R[0][2]*z;
      const py = R[1][0]*x + R[1][1]*y + R[1][2]*z;
      gCloud.append('circle').attr('cx', cx + scale*px).attr('cy', cy - scale*py)
        .attr('r', 2.6).attr('fill', colorFromValue(vec[i], lo, hi));
    }
    // Arrow along the 3D direction in which the active eigenvector varies most:
    // d = sum_i vec_i (x_i - mean). The eigenvector itself lives in R^N (one
    // value per point, shown by the coloring); this is its dominant axis in space.
    gArrow.html('');
    let dx = 0, dy = 0, dz = 0;
    for (let i = 0; i < N; i++) {
      const wgt = vec[i];
      dx += wgt * (points[i * 3] - mx);
      dy += wgt * (points[i * 3 + 1] - my);
      dz += wgt * (points[i * 3 + 2] - mz);
    }
    const dn = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dn > 1e-9) {
      const len = r * 0.95;
      const end = projectVec(R, [dx / dn * len, dy / dn * len, dz / dn * len], scale, cx, cy);
      gArrow.append('line').attr('x1', cx).attr('y1', cy).attr('x2', end[0]).attr('y2', end[1])
        .attr('stroke', '#ff9f43').attr('stroke-width', 2.6).attr('marker-end', 'url(#spec-arrow)');
      gArrow.append('text').attr('x', end[0] + 6).attr('y', end[1] - 4)
        .attr('fill', '#ff9f43').attr('font-size', '11')
        .text('v' + ((highlightedK !== null ? highlightedK : 0) + 1));
    }
  }
  redraw();
  attachOrbit(svg, () => R, (newR) => { R = newR; }, redraw);

  const mini = svg.append('g').attr('transform', `translate(${width - 168}, ${height - 102})`);
  mini.append('rect').attr('width', 160).attr('height', 96).attr('fill', 'rgba(0,0,0,0.72)')
    .attr('stroke', 'rgba(255,255,255,0.25)').attr('rx', 4);
  mini.append('text').attr('x', 80).attr('y', 12).attr('text-anchor', 'middle')
    .attr('fill', 'rgba(255,255,255,0.85)').attr('font-size', '9').text('eigenvalues');
  const valueLabel = mini.append('text').attr('x', 80).attr('y', 90).attr('text-anchor', 'middle')
    .attr('fill', 'rgba(255,255,255,0.95)').attr('font-size', '10').text('');

  const eig = state.topEigvals || new Float64Array(8);
  let lam0 = 1;
  for (let i = 0; i < eig.length; i++) lam0 = Math.max(lam0, Math.abs(eig[i]));
  const barW = 17;
  const maxH = 52;
  const baseFill = (i) => i < 2 ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.5)';
  const bars = [];
  for (let i = 0; i < 8; i++) {
    const v = Math.abs(eig[i] || 0);
    const h = maxH * (v / Math.max(1e-9, lam0));
    const bar = mini.append('rect').attr('x', 10 + i * barW).attr('y', 18 + (maxH - h))
      .attr('width', barW - 2).attr('height', h)
      .attr('fill', baseFill(i))
      .style('cursor', 'pointer');
    const hitBar = mini.append('rect').attr('x', 10 + i * barW).attr('y', 18)
      .attr('width', barW - 2).attr('height', maxH)
      .attr('fill', 'transparent').style('cursor', 'pointer');
    bars.push(bar);
    const k = i;
    function onEnter() {
      highlightedK = k;
      bars.forEach((b, j) => b.attr('fill', j === k ? '#ff9f43' : baseFill(j)));
      valueLabel.text('λ_' + (k + 1) + ' = ' + (eig[k] || 0).toFixed(3));
      redraw();
    }
    function onLeave() {
      highlightedK = null;
      bars.forEach((b, j) => b.attr('fill', baseFill(j)));
      valueLabel.text('');
      redraw();
    }
    bar.on('mouseenter', onEnter).on('mouseleave', onLeave);
    hitBar.on('mouseenter', onEnter).on('mouseleave', onLeave);
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
  else mountIsomapSpectral(svg, state, width, height);

  return {
    unmount() { wrap.remove(); }
  };
}
