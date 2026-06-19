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
    const pz = R[2][0]*x + R[2][1]*y + R[2][2]*z;
    out[i] = { i, sx: cx + scale*px, sy: cy - scale*py, depth: pz };
  }
  return out;
}

function rainbow(t, tMin, tMax) {
  const u = (t - tMin) / Math.max(1e-9, tMax - tMin);
  const h = (1 - u) * 240;
  return `hsl(${h.toFixed(1)}, 80%, 60%)`;
}

export function mountKnn(container, state, { width = 480, height = 360 } = {}) {
  const d3 = window.d3;
  const wrap = d3.select(container).append('div').attr('class', 'viz-knn');
  const svg = wrap.append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%').style('height', '100%')
    .style('touch-action', 'none').style('cursor', 'grab');

  const points = state.points;
  const edges = state.edges || [];
  const t = state.t || null;
  const N = points.length / 3;

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
  let proj = project(R, recentered, scale, cx, cy);

  let tMin = Infinity, tMax = -Infinity;
  if (t) for (let i = 0; i < t.length; i++) { if (t[i] < tMin) tMin = t[i]; if (t[i] > tMax) tMax = t[i]; }
  const colorOf = (i) => t ? rainbow(t[i], tMin, tMax) : '#7ec8ff';

  // All edges share one stroke, so draw them as a single <path> rather than one <line> per
  // edge. A dense kNN graph can have several thousand edges; the per-element approach was slow
  // to build and re-setting every line's endpoints each orbit frame dropped the frame rate on
  // phones. One path means one attribute write per frame.
  const gEdges = svg.append('g').attr('class', 'knn-edges');
  function edgePathD(list) {
    const e = list || edges;
    let d = '';
    for (let k = 0; k < e.length; k++) {
      const a = proj[e[k][0]], c = proj[e[k][1]];
      d += 'M' + a.sx.toFixed(1) + ' ' + a.sy.toFixed(1) + 'L' + c.sx.toFixed(1) + ' ' + c.sy.toFixed(1);
    }
    return d;
  }
  const edgePath = gEdges.append('path').attr('class', 'knn-edge-path')
    .attr('d', edgePathD()).attr('fill', 'none')
    .attr('stroke', 'rgba(255,255,255,0.18)').attr('stroke-width', 0.7)
    .attr('opacity', 0);
  // Separate path for the edges incident to a hovered node (built on demand).
  const highlightPath = gEdges.append('path').attr('class', 'knn-edge-highlight')
    .attr('fill', 'none').attr('stroke', 'rgba(255,255,255,0.95)').attr('stroke-width', 1.6)
    .attr('opacity', 0);

  function incidentEdges(i) {
    const out = [];
    for (let k = 0; k < edges.length; k++) { if (edges[k][0] === i || edges[k][1] === i) out.push(edges[k]); }
    return out;
  }

  const gPoints = svg.append('g');
  const nodeEls = proj.map(p => gPoints.append('circle')
    .attr('cx', p.sx).attr('cy', p.sy).attr('r', 2.8)
    .attr('fill', colorOf(p.i))
    .attr('class', 'knn-node')
    .attr('data-i', p.i)
    .style('cursor', 'pointer'));

  let hoverNode = -1;
  function rerender() {
    proj = project(R, recentered, scale, cx, cy);
    edgePath.attr('d', edgePathD());
    if (hoverNode >= 0) highlightPath.attr('d', edgePathD(incidentEdges(hoverNode)));
    nodeEls.forEach((node, i) => {
      node.attr('cx', proj[i].sx).attr('cy', proj[i].sy);
    });
  }

  // One short fade-in for the whole edge layer, regardless of edge count. The old per-edge
  // stagger (delay = i * 4ms) meant a several-thousand-edge graph kept fading in for ~20s.
  edgePath.transition().delay(60).duration(420).attr('opacity', 1);

  nodeEls.forEach((node, i) => {
    node.on('mouseenter', () => {
      hoverNode = i;
      edgePath.attr('opacity', 0.06);
      highlightPath.attr('d', edgePathD(incidentEdges(i))).attr('opacity', 1);
    });
    node.on('mouseleave', () => {
      hoverNode = -1;
      edgePath.attr('opacity', 1);
      highlightPath.attr('opacity', 0);
    });
  });

  let dragging = false, lastX = 0, lastY = 0;
  svg.on('pointerdown', (event) => {
    if (event.target && event.target.classList && event.target.classList.contains('knn-node')) return;
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
    rerender();
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
