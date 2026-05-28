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

export function mountCentering(container, state, { width = 480, height = 360 } = {}) {
  const d3 = window.d3;
  const wrap = d3.select(container).append('div').attr('class', 'viz-centering');
  const svg = wrap.append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%').style('height', '100%');

  const raw = state.rawPoints || state.points;
  const centered = state.points;
  const t = state.t || null;
  const N = raw.length / 3;

  let xmn = Infinity, xmx = -Infinity, ymn = Infinity, ymx = -Infinity, zmn = Infinity, zmx = -Infinity;
  for (let i = 0; i < N; i++) {
    const x = raw[i * 3], y = raw[i * 3 + 1], z = raw[i * 3 + 2];
    if (x < xmn) xmn = x; if (x > xmx) xmx = x;
    if (y < ymn) ymn = y; if (y > ymx) ymx = y;
    if (z < zmn) zmn = z; if (z > zmx) zmx = z;
  }
  const radius = Math.max(xmx - xmn, ymx - ymn, zmx - zmn, 1e-6) / 2;
  const R = [[0.886, 0.0, 0.464], [-0.163, 0.937, 0.311], [-0.434, -0.348, 0.831]];
  const scale = (Math.min(width, height) / 2 - 18) / radius;
  const cx = width / 2, cy = height / 2;

  const rawProj = project(R, raw, scale, cx, cy);
  const centeredProj = project(R, centered, scale, cx, cy);
  let tMin = Infinity, tMax = -Infinity;
  if (t) for (let i = 0; i < t.length; i++) { if (t[i] < tMin) tMin = t[i]; if (t[i] > tMax) tMax = t[i]; }
  const colorOf = (i) => t ? rainbow(t[i], tMin, tMax) : '#7ec8ff';

  const gGhost = svg.append('g');
  rawProj.forEach(p => {
    gGhost.append('circle').attr('cx', p.sx).attr('cy', p.sy).attr('r', 2.6)
      .attr('fill', colorOf(p.i)).attr('opacity', 0.28);
  });

  const gPoints = svg.append('g');
  const circles = centeredProj.map(p => gPoints.append('circle')
    .attr('cx', rawProj[p.i].sx).attr('cy', rawProj[p.i].sy)
    .attr('r', 2.8).attr('fill', colorOf(p.i)).attr('opacity', 0.95));

  setTimeout(() => {
    circles.forEach((c, i) => {
      c.transition().duration(1000).ease(d3.easeCubicInOut)
        .attr('cx', centeredProj[i].sx).attr('cy', centeredProj[i].sy);
    });
  }, 60);

  return {
    unmount() { wrap.remove(); }
  };
}
