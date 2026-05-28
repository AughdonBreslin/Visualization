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
    .style('width', '100%').style('height', '100%');

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
  const R = [[0.886, 0.0, 0.464], [-0.163, 0.937, 0.311], [-0.434, -0.348, 0.831]];
  const scale = (Math.min(width, height) / 2 - 18) / radius;
  const cx = width / 2, cy = height / 2;
  const recentered = new Float64Array(points.length);
  const ax = (xmn + xmx) / 2, ay = (ymn + ymx) / 2, az = (zmn + zmx) / 2;
  for (let i = 0; i < N; i++) {
    recentered[i * 3] = points[i * 3] - ax;
    recentered[i * 3 + 1] = points[i * 3 + 1] - ay;
    recentered[i * 3 + 2] = points[i * 3 + 2] - az;
  }
  const proj = project(R, recentered, scale, cx, cy);

  let tMin = Infinity, tMax = -Infinity;
  if (t) for (let i = 0; i < t.length; i++) { if (t[i] < tMin) tMin = t[i]; if (t[i] > tMax) tMax = t[i]; }
  const colorOf = (i) => t ? rainbow(t[i], tMin, tMax) : '#7ec8ff';

  const gEdges = svg.append('g').attr('class', 'knn-edges');
  const edgeEls = edges.map(([a, b]) => {
    const pa = proj[a], pb = proj[b];
    return gEdges.append('line')
      .attr('x1', pa.sx).attr('y1', pa.sy)
      .attr('x2', pb.sx).attr('y2', pb.sy)
      .attr('class', 'knn-edge')
      .attr('data-from', a).attr('data-to', b)
      .attr('stroke', 'rgba(255,255,255,0.18)')
      .attr('stroke-width', 0.7)
      .attr('opacity', 0);
  });

  const gPoints = svg.append('g');
  const nodeEls = proj.map(p => gPoints.append('circle')
    .attr('cx', p.sx).attr('cy', p.sy).attr('r', 2.8)
    .attr('fill', colorOf(p.i))
    .attr('class', 'knn-node')
    .attr('data-i', p.i)
    .style('cursor', 'pointer'));

  edgeEls.forEach((e, i) => {
    e.transition().delay(60 + i * 4).duration(280).attr('opacity', 1);
  });

  nodeEls.forEach((node, i) => {
    node.on('mouseenter', () => {
      edgeEls.forEach(e => {
        const from = +e.attr('data-from');
        const to = +e.attr('data-to');
        const hit = from === i || to === i;
        e.attr('stroke', hit ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.05)')
         .attr('stroke-width', hit ? 1.6 : 0.6);
      });
    });
    node.on('mouseleave', () => {
      edgeEls.forEach(e => {
        e.attr('stroke', 'rgba(255,255,255,0.18)').attr('stroke-width', 0.7);
      });
    });
  });

  return {
    unmount() { wrap.remove(); }
  };
}
