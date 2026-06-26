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

function projectVec(R, v, scale, cx, cy) {
  const px = R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2];
  const py = R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2];
  return [cx + scale*px, cy - scale*py];
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
    .style('width', '100%').style('height', '100%')
    .style('touch-action', 'none').style('cursor', 'grab');

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
  let R = [[0.886, 0.0, 0.464], [-0.163, 0.937, 0.311], [-0.434, -0.348, 0.831]];
  const scale = (Math.min(width, height) / 2 - 18) / radius;
  const cx = width / 2, cy = height / 2;
  const axLen = radius * 1.25;
  const axes = [
    { v: [1, 0, 0], color: 'rgba(255,107,107,0.45)', label: 'x' },
    { v: [0, 1, 0], color: 'rgba(107,212,123,0.45)', label: 'y' },
    { v: [0, 0, 1], color: 'rgba(107,182,255,0.45)', label: 'z' },
  ];

  const gGrid = svg.append('g');
  const gAxes = svg.append('g');
  const gGhost = svg.append('g');
  const gPoints = svg.append('g');

  let tMin = Infinity, tMax = -Infinity;
  if (t) for (let i = 0; i < t.length; i++) { if (t[i] < tMin) tMin = t[i]; if (t[i] > tMax) tMax = t[i]; }
  const colorOf = (i) => t ? rainbow(t[i], tMin, tMax) : '#7ec8ff';

  // Pre-create axis lines + labels so orbit redraws only update attributes, not recreate elements.
  const axisLineEls = axes.map(({ v, color, label }) => {
    const line = gAxes.append('line').attr('stroke', color).attr('stroke-width', 1);
    const text = gAxes.append('text').attr('fill', color).attr('font-size', '10').text(label);
    return { v, line, text };
  });

  // Pre-create grid lines: 2 s-values × 2 directions = 4 lines.
  const gridLineEls = [];
  const gridDefs = [
    { s: -1, varAxis: 'x' }, { s: -1, varAxis: 'z' },
    { s:  1, varAxis: 'x' }, { s:  1, varAxis: 'z' },
  ];
  gridDefs.forEach(() => {
    gridLineEls.push(gGrid.append('line')
      .attr('stroke', 'rgba(255,255,255,0.06)').attr('stroke-width', 1));
  });

  function drawAxesAndGrid() {
    axisLineEls.forEach(({ v, line, text }) => {
      const a = projectVec(R, [-axLen * v[0], -axLen * v[1], -axLen * v[2]], scale, cx, cy);
      const b = projectVec(R, [axLen * v[0], axLen * v[1], axLen * v[2]], scale, cx, cy);
      line.attr('x1', a[0]).attr('y1', a[1]).attr('x2', b[0]).attr('y2', b[1]);
      text.attr('x', b[0] + 4).attr('y', b[1] + 3);
    });
    const gridStep = radius / 2;
    gridDefs.forEach(({ s, varAxis }, idx) => {
      const offset = s * gridStep;
      const a = projectVec(R,
        varAxis === 'x' ? [-axLen, 0, offset] : [offset, 0, -axLen],
        scale, cx, cy);
      const b = projectVec(R,
        varAxis === 'x' ? [axLen, 0, offset] : [offset, 0, axLen],
        scale, cx, cy);
      gridLineEls[idx].attr('x1', a[0]).attr('y1', a[1]).attr('x2', b[0]).attr('y2', b[1]);
    });
  }

  drawAxesAndGrid();

  const rawProj = project(R, raw, scale, cx, cy);
  const centeredProj = project(R, centered, scale, cx, cy);

  const ghostCircles = rawProj.map(p =>
    gGhost.append('circle').attr('cx', p.sx).attr('cy', p.sy).attr('r', 2.6)
      .attr('fill', colorOf(p.i)).attr('opacity', 0.32));

  const circles = centeredProj.map(p => gPoints.append('circle')
    .attr('cx', rawProj[p.i].sx).attr('cy', rawProj[p.i].sy)
    .attr('r', 2.8).attr('fill', colorOf(p.i)).attr('opacity', 0.95));

  const playAnimation = !state.centeringAnimated;
  if (playAnimation) state.centeringAnimated = true;

  let animationDone = !playAnimation;

  if (playAnimation) {
    setTimeout(() => {
      circles.forEach((c, i) => {
        c.transition().duration(1000).ease(d3.easeCubicInOut)
          .attr('cx', centeredProj[i].sx).attr('cy', centeredProj[i].sy)
          .on('end', i === circles.length - 1 ? () => { animationDone = true; } : null);
      });
      ghostCircles.forEach(c => {
        c.transition().duration(900).ease(d3.easeCubicInOut).attr('opacity', 0);
      });
    }, 60);
  } else {
    circles.forEach((c, i) => c.attr('cx', centeredProj[i].sx).attr('cy', centeredProj[i].sy));
    ghostCircles.forEach(c => c.attr('opacity', 0));
  }

  function redrawAfterOrbit() {
    drawAxesAndGrid();
    const proj = project(R, centered, scale, cx, cy);
    circles.forEach((c, i) => {
      c.attr('cx', proj[i].sx).attr('cy', proj[i].sy);
    });
  }

  let dragging = false, lastX = 0, lastY = 0, rafId = null;
  svg.on('pointerdown', (event) => {
    if (!animationDone) return;
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
    if (!rafId) rafId = requestAnimationFrame(() => { rafId = null; redrawAfterOrbit(); });
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
