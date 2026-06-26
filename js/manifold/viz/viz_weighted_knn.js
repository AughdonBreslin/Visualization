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

  // Pre-build per-node incident edge index for O(k) hover updates.
  const incidentByNode = Array.from({length: N}, () => []);
  edges.forEach(([a, b], idx) => {
    incidentByNode[a].push(idx);
    incidentByNode[b].push(idx);
  });
  const maxK = incidentByNode.reduce((m, e) => Math.max(m, e.length), 0);

  // Single path for all edges. Replaces N*k individual <line> elements so
  // drag redraws cost one attribute write instead of 6×N*k DOM mutations.
  const gEdges = svg.append('g');
  const bgPath = gEdges.append('path').attr('fill', 'none')
    .attr('stroke', 'rgba(255,255,255,0.22)').attr('stroke-width', 0.5);

  function bgPathD() {
    let d = '';
    for (let k = 0; k < edges.length; k++) {
      const a = proj[edges[k][0]], b = proj[edges[k][1]];
      d += 'M' + a.sx.toFixed(1) + ' ' + a.sy.toFixed(1) + 'L' + b.sx.toFixed(1) + ' ' + b.sy.toFixed(1);
    }
    return d;
  }

  // Pre-created incident lines: at most maxK, reused across hover changes.
  const gIncident = svg.append('g');
  const incidentLineEls = Array.from({length: maxK}, () =>
    gIncident.append('line').attr('fill', 'none')
      .attr('stroke', 'rgba(255,255,255,0.92)').attr('stroke-linecap', 'round')
      .attr('display', 'none'));

  let proj = project(R, recentered, scale, cx, cy);

  const gPoints = svg.append('g');
  const nodeEls = proj.map(p =>
    gPoints.append('circle').attr('cx', p.sx).attr('cy', p.sy)
      .attr('r', 2.5).attr('fill', colorOf(p.i)).style('cursor', 'pointer'));

  function updateIncidentLines() {
    const incs = incidentByNode[selectedPoint];
    incs.forEach((edgeIdx, k) => {
      const [a, b] = edges[edgeIdx];
      const other = a === selectedPoint ? b : a;
      const wij = W ? W[selectedPoint * N + other] : 0;
      const pa = proj[a], pb = proj[b];
      incidentLineEls[k]
        .attr('x1', pa.sx).attr('y1', pa.sy)
        .attr('x2', pb.sx).attr('y2', pb.sy)
        .attr('stroke-width', strokeWidthForWeight(wij))
        .attr('display', null);
    });
    for (let k = incs.length; k < maxK; k++) incidentLineEls[k].attr('display', 'none');
  }

  function applyNodeSelection(prev, next) {
    if (prev >= 0) nodeEls[prev].attr('r', 2.5).attr('fill', colorOf(prev));
    nodeEls[next].attr('r', 4.5).attr('fill', '#ff9f43');
  }

  // Initial render.
  bgPath.attr('d', bgPathD());
  applyNodeSelection(-1, selectedPoint);
  updateIncidentLines();
  gEdges.attr('opacity', 0.35);

  nodeEls.forEach((node, i) => {
    node.on('mouseenter', () => {
      if (i === selectedPoint) return;
      const prev = selectedPoint;
      selectedPoint = i;
      applyNodeSelection(prev, i);
      updateIncidentLines();
    });
  });

  let dragging = false, lastX = 0, lastY = 0, rafId = null;
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
    R = matmul(matmul(rotX(dy), rotY(dx)), R);
    if (!rafId) rafId = requestAnimationFrame(() => {
      rafId = null;
      proj = project(R, recentered, scale, cx, cy);
      bgPath.attr('d', bgPathD());
      nodeEls.forEach((node, i) => node.attr('cx', proj[i].sx).attr('cy', proj[i].sy));
      updateIncidentLines();
    });
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
