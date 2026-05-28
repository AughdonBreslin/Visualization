const VIRIDIS = ['#000000', '#3a1a6e', '#5b3a8c', '#8b5fbf', '#c179d3', '#e8a37f', '#f5cf6e', '#f9eb6b'];

function projectStandard(points, width, height) {
  const N = points.length / 3;
  let xmn = Infinity, xmx = -Infinity, ymn = Infinity, ymx = -Infinity, zmn = Infinity, zmx = -Infinity;
  for (let i = 0; i < N; i++) {
    const x = points[i * 3], y = points[i * 3 + 1], z = points[i * 3 + 2];
    if (x < xmn) xmn = x; if (x > xmx) xmx = x;
    if (y < ymn) ymn = y; if (y > ymx) ymx = y;
    if (z < zmn) zmn = z; if (z > zmx) zmx = z;
  }
  const r = Math.max(xmx - xmn, ymx - ymn, zmx - zmn, 1e-6) / 2;
  const R = [[0.886, 0.0, 0.464], [-0.163, 0.937, 0.311], [-0.434, -0.348, 0.831]];
  const s = (Math.min(width, height) / 2 - 24) / r;
  const cx = width / 2, cy = height / 2;
  const ax = (xmn + xmx) / 2, ay = (ymn + ymx) / 2, az = (zmn + zmx) / 2;
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    const x = points[i * 3] - ax, y = points[i * 3 + 1] - ay, z = points[i * 3 + 2] - az;
    const px = R[0][0]*x + R[0][1]*y + R[0][2]*z;
    const py = R[1][0]*x + R[1][1]*y + R[1][2]*z;
    const pz = R[2][0]*x + R[2][1]*y + R[2][2]*z;
    out[i] = { i, sx: cx + s * px, sy: cy - s * py, depth: pz };
  }
  return { proj: out, scale: s, center: [cx, cy], R };
}

function colorFromValue(v, lo, hi) {
  const u = Math.max(0, Math.min(1, (v - lo) / Math.max(1e-12, hi - lo)));
  const idx = u * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 1, Math.floor(idx));
  return VIRIDIS[i];
}

function mountPcaSpectral(svg, state, width, height) {
  const points = state.points;
  const { proj, scale, center, R } = projectStandard(points, width, height);
  proj.forEach(p => svg.append('circle').attr('cx', p.sx).attr('cy', p.sy).attr('r', 2.4)
    .attr('fill', 'rgba(255,255,255,0.28)'));

  if (state.pcAxes) {
    const { v1, v2 } = state.pcAxes;
    const ex = v1[0], ey = v1[1], ez = v1[2];
    const fx = v2[0], fy = v2[1], fz = v2[2];
    const L = scale * 0.7;
    const cornerSign = [[1, 1], [1, -1], [-1, -1], [-1, 1]];
    const pts = cornerSign.map(([a, b]) => {
      const wx = a * ex * L + b * fx * L;
      const wy = a * ey * L + b * fy * L;
      const wz = a * ez * L + b * fz * L;
      const px = R[0][0] * wx + R[0][1] * wy + R[0][2] * wz;
      const py = R[1][0] * wx + R[1][1] * wy + R[1][2] * wz;
      return [center[0] + px, center[1] - py];
    });
    svg.append('polygon')
      .attr('points', pts.map(p => p.join(',')).join(' '))
      .attr('fill', 'rgba(255,255,255,0.06)')
      .attr('stroke', 'rgba(255,255,255,0.35)').attr('stroke-width', 1);
  }
  const mini = svg.append('g').attr('transform', `translate(${width - 130}, ${height - 90})`);
  mini.append('rect').attr('width', 122).attr('height', 80).attr('fill', 'rgba(0,0,0,0.7)')
    .attr('stroke', 'rgba(255,255,255,0.25)').attr('rx', 4);
  mini.append('text').attr('x', 61).attr('y', 12).attr('text-anchor', 'middle')
    .attr('fill', 'rgba(255,255,255,0.85)').attr('font-size', '9').text('principal axes');
  const cx = 61, cy = 50;
  const ax = state.pcAxes || { v1: [1, 0, 0], v2: [0, 1, 0], v3: [0, 0, 1], lambda: [1, 1, 1] };
  const colors = ['#ff9f43', '#54a0ff', '#6bd47b'];
  const labels = ['PC1', 'PC2', 'PC3'];
  ['v1', 'v2', 'v3'].forEach((key, k) => {
    const v = ax[key];
    if (!v) return;
    const L = 24 * Math.sqrt(Math.max(0, Math.abs(ax.lambda[k] || 1)) / Math.max(1e-9, Math.abs(ax.lambda[0])));
    const px = (R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2]) * L;
    const py = (R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2]) * L;
    mini.append('line').attr('x1', cx).attr('y1', cy).attr('x2', cx + px).attr('y2', cy - py)
      .attr('stroke', colors[k]).attr('stroke-width', 1.5);
    mini.append('text').attr('x', cx + px + 4).attr('y', cy - py)
      .attr('fill', colors[k]).attr('font-size', '8').text(labels[k]);
  });
}

function mountIsomapSpectral(svg, state, width, height) {
  const points = state.points;
  const { proj } = projectStandard(points, width, height);
  const v1 = state.v1Values || new Float64Array(points.length / 3);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < v1.length; i++) { if (v1[i] < lo) lo = v1[i]; if (v1[i] > hi) hi = v1[i]; }
  proj.forEach(p => svg.append('circle').attr('cx', p.sx).attr('cy', p.sy).attr('r', 2.6)
    .attr('fill', colorFromValue(v1[p.i], lo, hi)));

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
