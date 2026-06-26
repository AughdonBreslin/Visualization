function matmul(A, B) {
  const C = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
  return C;
}
function rotX(a) { const c = Math.cos(a), s = Math.sin(a); return [[1,0,0],[0,c,-s],[0,s,c]]; }
function rotY(a) { const c = Math.cos(a), s = Math.sin(a); return [[c,0,s],[0,1,0],[-s,0,c]]; }

function project(R, X, scale, cx, cy) {
  const N = X.length / 3;
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    const x = X[i * 3], y = X[i * 3 + 1], z = X[i * 3 + 2];
    const px = R[0][0]*x + R[0][1]*y + R[0][2]*z;
    const py = R[1][0]*x + R[1][1]*y + R[1][2]*z;
    out[i] = { i, sx: cx + scale*px, sy: cy - scale*py };
  }
  return out;
}

function rainbow(t, tMin, tMax) {
  const u = (t - tMin) / Math.max(1e-9, tMax - tMin);
  const h = (1 - u) * 240;
  return `hsl(${h.toFixed(1)}, 80%, 60%)`;
}

function strokeWidthForWeight(w) {
  const absW = Math.abs(w);
  return 0.6 + Math.min(2.9, absW * 4);
}

export function mountWeightedKnn(container, state, { width = 480, height = 360 } = {}) {
  const d3 = window.d3;
  const wrap = d3.select(container).append('div').attr('class', 'viz-weighted-knn');
  const svg = wrap.append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%').style('height', '100%')
    .style('touch-action', 'none').style('cursor', 'grab');

  const points = state.points;
  const edges = state.edges || [];
  const W = state.W;
  const N = points.length / 3;
  const t = state.t || null;
  let selectedPoint = (state.selectedPoint !== undefined ? state.selectedPoint : Math.floor(N * 0.2));

  let xmn = Infinity, xmx = -Infinity, ymn = Infinity, ymx = -Infinity, zmn = Infinity, zmx = -Infinity;
  for (let i = 0; i < N; i++) {
    const x = points[i * 3], y = points[i * 3 + 1], z = points[i * 3 + 2];
    if (x < xmn) xmn = x; if (x > xmx) xmx = x;
    if (y < ymn) ymn = y; if (y > ymx) ymx = y;
    if (z < zmn) zmn = z; if (z > zmx) zmx = z;
  }
  const radius = Math.max(xmx - xmn, ymx - ymn, zmx - zmn, 1e-6) / 2;
  let R = [[0.886, 0.0, 0.464], [-0.163, 0.937, 0.311], [-0.434, -0.348, 0.831]];
  const scale = (Math.min(width, height) / 2 - 18) / radius;
  const cx = width / 2, cy = height / 2;
  const recentered = new Float64Array(points.length);
  const ax = (xmn + xmx) / 2, ay = (ymn + ymx) / 2, az = (zmn + zmx) / 2;
  for (let i = 0; i < N; i++) {
    recentered[i * 3] = points[i * 3] - ax;
    recentered[i * 3 + 1] = points[i * 3 + 1] - ay;
    recentered[i * 3 + 2] = points[i * 3 + 2] - az;
  }

  let tMin = Infinity, tMax = -Infinity;
  if (t) for (let i = 0; i < t.length; i++) { if (t[i] < tMin) tMin = t[i]; if (t[i] > tMax) tMax = t[i]; }
  const colorOf = (i) => t ? rainbow(t[i], tMin, tMax) : '#7ec8ff';

  const gEdges = svg.append('g');
  const gPoints = svg.append('g');

  let proj = project(R, recentered, scale, cx, cy);
  const edgeEls = edges.map(([a, b]) =>
    gEdges.append('line').attr('data-from', a).attr('data-to', b));
  const nodeEls = proj.map(p =>
    gPoints.append('circle').attr('data-i', p.i).style('cursor', 'pointer'));

  function redraw() {
    proj = project(R, recentered, scale, cx, cy);
    edgeEls.forEach((e, idx) => {
      const a = edges[idx][0];
      const b = edges[idx][1];
      const isSelectedEdge = (a === selectedPoint || b === selectedPoint);
      let wij = 0;
      if (isSelectedEdge) {
        const other = a === selectedPoint ? b : a;
        wij = W ? W[selectedPoint * N + other] : 0;
      }
      e.attr('x1', proj[a].sx).attr('y1', proj[a].sy)
       .attr('x2', proj[b].sx).attr('y2', proj[b].sy)
       .attr('stroke', isSelectedEdge ? 'rgba(255,255,255,0.92)' : 'rgba(255,255,255,0.10)')
       .attr('stroke-width', isSelectedEdge ? strokeWidthForWeight(wij) : 0.5);
    });
    nodeEls.forEach((node, i) => {
      const isSelected = (i === selectedPoint);
      node.attr('cx', proj[i].sx).attr('cy', proj[i].sy)
        .attr('r', isSelected ? 4.5 : 2.5)
        .attr('fill', isSelected ? '#ff9f43' : colorOf(i));
    });
  }
  redraw();

  nodeEls.forEach((node, i) => {
    node.on('mouseenter', () => {
      if (i === selectedPoint) return;
      selectedPoint = i;
      redraw();
    });
  });

  let dragging = false, lastX = 0, lastY = 0, rafId = null;
  svg.on('pointerdown', (event) => {
    if (event.target && event.target.tagName === 'circle') return;
    dragging = true; lastX = event.clientX; lastY = event.clientY;
    svg.style('cursor', 'grabbing');
    try { svg.node().setPointerCapture(event.pointerId); } catch (e) {}
  });
  svg.on('pointermove', (event) => {
    if (!dragging) return;
    const dx = (event.clientX - lastX) * 0.008;
    const dy = (event.clientY - lastY) * 0.008;
    lastX = event.clientX; lastY = event.clientY;
    R = matmul(matmul(rotX(dy), rotY(dx)), R);
    if (!rafId) rafId = requestAnimationFrame(() => { rafId = null; redraw(); });
  });
  function endDrag(event) {
    dragging = false; svg.style('cursor', 'grab');
    try { svg.node().releasePointerCapture(event.pointerId); } catch (e) {}
  }
  svg.on('pointerup', endDrag);
  svg.on('pointercancel', endDrag);
  svg.on('pointerleave', endDrag);

  return {
    unmount() { wrap.remove(); }
  };
}
