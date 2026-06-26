function matmul(A, B) {
  const C = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
  return C;
}

function rotX(a) { const c = Math.cos(a), s = Math.sin(a); return [[1,0,0],[0,c,-s],[0,s,c]]; }
function rotY(a) { const c = Math.cos(a), s = Math.sin(a); return [[c,0,s],[0,1,0],[-s,0,c]]; }

function rainbow(t, tMin, tMax) {
  const u = (t - tMin) / Math.max(1e-9, tMax - tMin);
  const h = (1 - u) * 240;
  return `hsl(${h.toFixed(1)}, 80%, 60%)`;
}

export function createViz3d(container, { width = 480, height = 360, isThumbnail = false } = {}) {
  const d3 = window.d3;
  const wrap = d3.select(container).append('div').attr('class', isThumbnail ? 'viz3d-thumb' : 'viz3d');
  const svg = wrap.append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%').style('height', '100%')
    .style('touch-action', 'none').style('cursor', 'grab');
  const gEdges = svg.append('g').attr('class', 'edges');
  const gPoints = svg.append('g').attr('class', 'points');
  const gAxes = svg.append('g').attr('class', 'axes');

  let R = matmul(rotX(-0.35), rotY(0.6));
  let state = null;

  function computeBounds(points) {
    if (!points || points.length === 0) return { center: [0,0,0], radius: 1 };
    let xmn = Infinity, ymn = Infinity, zmn = Infinity;
    let xmx = -Infinity, ymx = -Infinity, zmx = -Infinity;
    const N = points.length / 3;
    for (let i = 0; i < N; i++) {
      const x = points[i*3], y = points[i*3+1], z = points[i*3+2];
      if (x < xmn) xmn = x; if (x > xmx) xmx = x;
      if (y < ymn) ymn = y; if (y > ymx) ymx = y;
      if (z < zmn) zmn = z; if (z > zmx) zmx = z;
    }
    const cx = (xmn+xmx)/2, cy = (ymn+ymx)/2, cz = (zmn+zmx)/2;
    const radius = Math.max(xmx-xmn, ymx-ymn, zmx-zmn, 1e-6) / 2;
    return { center: [cx, cy, cz], radius };
  }

  function project(R, X, scale, cx, cy) {
    const N = X.length / 3;
    const out = new Array(N);
    for (let i = 0; i < N; i++) {
      const x = X[i*3], y = X[i*3+1], z = X[i*3+2];
      const px = R[0][0]*x + R[0][1]*y + R[0][2]*z;
      const py = R[1][0]*x + R[1][1]*y + R[1][2]*z;
      const pz = R[2][0]*x + R[2][1]*y + R[2][2]*z;
      out[i] = { i, sx: cx + scale*px, sy: cy - scale*py, depth: pz };
    }
    return out;
  }

  function render() {
    if (!state) return;
    const { recentered, t, edges, colors, bounds: { radius } } = state;
    const N = recentered.length / 3;
    const margin = isThumbnail ? 6 : 18;
    const scale = (Math.min(width, height) / 2 - margin) / radius;
    const cx = width / 2, cy = height / 2;
    const proj = project(R, recentered, scale, cx, cy);

    if (edges && edges.length > 0) {
      const lines = edges.map(([a,b]) => {
        const pa = proj[a], pb = proj[b];
        return { x1: pa.sx, y1: pa.sy, x2: pb.sx, y2: pb.sy, depth: (pa.depth + pb.depth) / 2 };
      });
      lines.sort((a,b) => a.depth - b.depth);
      const sel = gEdges.selectAll('line').data(lines);
      sel.enter().append('line').merge(sel)
        .attr('x1', d => d.x1).attr('y1', d => d.y1)
        .attr('x2', d => d.x2).attr('y2', d => d.y2)
        .attr('stroke', 'rgba(255,255,255,0.18)')
        .attr('stroke-width', isThumbnail ? 0.4 : 0.6);
      sel.exit().remove();
    } else {
      gEdges.selectAll('line').remove();
    }

    let tMin = Infinity, tMax = -Infinity;
    if (t && !colors) for (let i = 0; i < t.length; i++) { if (t[i] < tMin) tMin = t[i]; if (t[i] > tMax) tMax = t[i]; }
    proj.sort((a,b) => a.depth - b.depth);
    const sel = gPoints.selectAll('circle').data(proj, d => d.i);
    sel.enter().append('circle').merge(sel)
      .attr('cx', d => d.sx).attr('cy', d => d.sy)
      .attr('r', isThumbnail ? 1.6 : 2.8)
      .attr('fill', d => colors ? colors[d.i] : (t ? rainbow(t[d.i], tMin, tMax) : '#7ec8ff'))
      .attr('opacity', d => {
        const z = (d.depth + radius) / (2 * radius);
        return 0.45 + 0.55 * Math.max(0, Math.min(1, z));
      });
    sel.exit().remove();
  }

  let dragging = false, lastX = 0, lastY = 0, rafId = null;
  svg.on('pointerdown', (event) => {
    dragging = true; lastX = event.clientX; lastY = event.clientY;
    svg.style('cursor', 'grabbing');
    svg.node().setPointerCapture(event.pointerId);
  });
  svg.on('pointermove', (event) => {
    if (!dragging) return;
    const dx = (event.clientX - lastX) * 0.008;
    const dy = (event.clientY - lastY) * 0.008;
    lastX = event.clientX; lastY = event.clientY;
    R = matmul(matmul(rotX(dy), rotY(dx)), R);
    if (!rafId) rafId = requestAnimationFrame(() => { rafId = null; render(); });
  });
  function endDrag(event) {
    dragging = false; svg.style('cursor', 'grab');
    try { svg.node().releasePointerCapture(event.pointerId); } catch (e) {}
  }
  svg.on('pointerup', endDrag);
  svg.on('pointercancel', endDrag);
  svg.on('pointerleave', endDrag);

  function setState(next) {
    const bounds = computeBounds(next.points);
    const N = next.points.length / 3;
    const recentered = new Float64Array(next.points.length);
    for (let i = 0; i < N; i++) {
      recentered[i*3] = next.points[i*3] - bounds.center[0];
      recentered[i*3+1] = next.points[i*3+1] - bounds.center[1];
      recentered[i*3+2] = next.points[i*3+2] - bounds.center[2];
    }
    state = {
      recentered,
      colors: next.colors || null,
      edges: next.edges || null,
      t: next.t || null,
      bounds,
    };
    render();
  }

  return { setState, render };
}
