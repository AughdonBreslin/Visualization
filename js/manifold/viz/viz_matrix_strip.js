const VIRIDIS = ['#000000', '#3a1a6e', '#5b3a8c', '#8b5fbf', '#c179d3', '#e8a37f', '#f5cf6e', '#f9eb6b'];
const VIRIDIS_RGB = VIRIDIS.map(hex => {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
});

function colorIdx(min, max, v) {
  if (!Number.isFinite(v) || max - min < 1e-12) return 0;
  const u = Math.max(0, Math.min(1, (v - min) / (max - min)));
  const idx = u * (VIRIDIS.length - 1);
  const lo = Math.floor(idx), hi = Math.min(VIRIDIS.length - 1, lo + 1);
  return idx - lo < 0.5 ? lo : hi;
}

function matmul(A, B) {
  const C = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
  return C;
}
function rotX(a) { const c = Math.cos(a), s = Math.sin(a); return [[1,0,0],[0,c,-s],[0,s,c]]; }
function rotY(a) { const c = Math.cos(a), s = Math.sin(a); return [[c,0,s],[0,1,0],[-s,0,c]]; }

function project3DFrom(R, points, ax, ay, az, scale, cx, cy) {
  const N = points.length / 3;
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    const x = points[i * 3] - ax, y = points[i * 3 + 1] - ay, z = points[i * 3 + 2] - az;
    const px = R[0][0]*x + R[0][1]*y + R[0][2]*z;
    const py = R[1][0]*x + R[1][1]*y + R[1][2]*z;
    out[i] = { i, sx: cx + scale * px, sy: cy - scale * py };
  }
  return out;
}

function paneBounds(points) {
  const N = points.length / 3;
  let xmn = Infinity, xmx = -Infinity, ymn = Infinity, ymx = -Infinity, zmn = Infinity, zmx = -Infinity;
  for (let i = 0; i < N; i++) {
    if (points[i * 3] < xmn) xmn = points[i * 3]; if (points[i * 3] > xmx) xmx = points[i * 3];
    if (points[i * 3 + 1] < ymn) ymn = points[i * 3 + 1]; if (points[i * 3 + 1] > ymx) ymx = points[i * 3 + 1];
    if (points[i * 3 + 2] < zmn) zmn = points[i * 3 + 2]; if (points[i * 3 + 2] > zmx) zmx = points[i * 3 + 2];
  }
  const r = Math.max(xmx - xmn, ymx - ymn, zmx - zmn, 1e-6) / 2;
  return { ax: (xmn + xmx) / 2, ay: (ymn + ymx) / 2, az: (zmn + zmx) / 2, r };
}

function mountOrbitablePane(svg, pane, x, y, w, h) {
  const d3 = window.d3;
  const paneG = svg.append('g').attr('transform', `translate(${x}, ${y})`);
  paneG.append('text').attr('x', w / 2).attr('y', -6).attr('text-anchor', 'middle')
    .attr('fill', 'rgba(255,255,255,0.55)').attr('font-size', '10').text(pane.label || '');

  const fo = paneG.append('foreignObject').attr('x', 0).attr('y', 0).attr('width', w).attr('height', h);
  const canvas = document.createElement('canvas');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  canvas.style.cursor = 'grab';
  canvas.style.touchAction = 'none';
  fo.node().appendChild(canvas);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const points = pane.kind === 'cloud_thumb' ? pane.data : pane.data.points;
  const { ax, ay, az, r } = paneBounds(points);
  const scale = (Math.min(w, h) / 2 - 6) / r;
  const cxp = w / 2, cyp = h / 2;
  let R = [[0.886, 0.0, 0.464], [-0.163, 0.937, 0.311], [-0.434, -0.348, 0.831]];

  function redraw() {
    ctx.clearRect(0, 0, w, h);
    const proj = project3DFrom(R, points, ax, ay, az, scale, cxp, cyp);
    if (pane.kind === 'graph_thumb' || pane.kind === 'graph_thumb_with_path') {
      ctx.strokeStyle = 'rgba(255,255,255,0.18)';
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      for (const [a, b] of pane.data.edges) {
        ctx.moveTo(proj[a].sx, proj[a].sy);
        ctx.lineTo(proj[b].sx, proj[b].sy);
      }
      ctx.stroke();
      if (pane.kind === 'graph_thumb_with_path') {
        ctx.strokeStyle = 'rgba(255,255,255,0.95)';
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        for (const [a, b] of pane.data.pathEdges) {
          if (proj[a] && proj[b]) {
            ctx.moveTo(proj[a].sx, proj[a].sy);
            ctx.lineTo(proj[b].sx, proj[b].sy);
          }
        }
        ctx.stroke();
      }
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      for (const p of proj) {
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, 1.4, 0, 6.283185307179586);
        ctx.fill();
      }
    } else {
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      for (const p of proj) {
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, 1.6, 0, 6.283185307179586);
        ctx.fill();
      }
    }
  }
  redraw();

  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener('pointerdown', (event) => {
    dragging = true; lastX = event.clientX; lastY = event.clientY;
    canvas.style.cursor = 'grabbing';
    try { canvas.setPointerCapture(event.pointerId); } catch (e) {}
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    const dx = (event.clientX - lastX) * 0.01;
    const dy = (event.clientY - lastY) * 0.01;
    lastX = event.clientX; lastY = event.clientY;
    R = matmul(matmul(rotX(dy), rotY(dx)), R);
    redraw();
  });
  function endDrag(event) {
    dragging = false; canvas.style.cursor = 'grab';
    try { canvas.releasePointerCapture(event.pointerId); } catch (e) {}
  }
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('pointerleave', endDrag);
}

function mountStaticPane(svg, pane, x, y, w, h) {
  const g = svg.append('g').attr('transform', `translate(${x}, ${y})`);
  g.append('text').attr('x', w / 2).attr('y', -6).attr('text-anchor', 'middle')
   .attr('fill', 'rgba(255,255,255,0.55)').attr('font-size', '10').text(pane.label || '');

  if (pane.kind === 'matrix_numbers') {
    const M = pane.data;
    const cw = w / M[0].length, ch = h / M.length;
    g.append('rect').attr('width', w).attr('height', h).attr('fill', 'rgba(255,255,255,0.08)')
     .attr('stroke', 'rgba(255,255,255,0.3)');
    M.forEach((row, r) => {
      row.forEach((v, c) => {
        g.append('text').attr('x', c * cw + cw / 2).attr('y', r * ch + ch / 2 + 3)
         .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.92)')
         .attr('font-size', '9').text(Number(v).toFixed(2));
      });
    });
  } else if (pane.kind === 'heatmap') {
    const { matrix, N, highlightRow } = pane.data;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < matrix.length; i++) {
      const v = matrix[i];
      if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    const total = Math.min(w, h);
    const ox = (w - total) / 2;
    const oy = (h - total) / 2;

    const canvas = document.createElement('canvas');
    canvas.width = N; canvas.height = N;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(N, N);
    const buf = img.data;
    for (let i = 0; i < N * N; i++) {
      const rgb = VIRIDIS_RGB[colorIdx(lo, hi, matrix[i])];
      const k = i * 4;
      buf[k] = rgb[0]; buf[k + 1] = rgb[1]; buf[k + 2] = rgb[2]; buf[k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    g.append('image').attr('x', ox).attr('y', oy)
      .attr('width', total).attr('height', total)
      .attr('preserveAspectRatio', 'none')
      .attr('image-rendering', 'pixelated')
      .style('image-rendering', 'pixelated')
      .attr('href', canvas.toDataURL());

    if (highlightRow !== undefined && highlightRow < N) {
      const cellSize = total / N;
      g.append('rect').attr('x', ox).attr('y', oy + highlightRow * cellSize)
        .attr('width', total).attr('height', cellSize)
        .attr('fill', 'none').attr('stroke', '#fff').attr('stroke-width', 1.5);
    }
  }
}

function renderPane(svg, pane, x, y, w, h) {
  if (pane.kind === 'cloud_thumb' || pane.kind === 'graph_thumb' || pane.kind === 'graph_thumb_with_path') {
    mountOrbitablePane(svg, pane, x, y, w, h);
  } else {
    mountStaticPane(svg, pane, x, y, w, h);
  }
}

export function mountMatrixStrip(container, state, { width = 480, height = 280 } = {}) {
  const d3 = window.d3;
  const wrap = d3.select(container).append('div').attr('class', 'viz-matrix-strip');
  const svg = wrap.append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%').style('height', '100%');

  const panes = state.panes || [];
  const opLabels = state.paneOpLabels || [];
  const paneCount = Math.max(1, panes.length);
  const gapW = 36;
  const sideMargin = 10;
  const paneW = (width - 2 * sideMargin - gapW * (paneCount - 1)) / paneCount;
  const paneH = height - 70;
  const paneY = 30;

  panes.forEach((pane, i) => {
    const x = sideMargin + i * (paneW + gapW);
    renderPane(svg, pane, x, paneY, paneW, paneH);
  });

  for (let k = 0; k < panes.length - 1; k++) {
    const paneRightEdge = sideMargin + (k + 1) * paneW + k * gapW;
    const arrowMid = paneRightEdge + gapW / 2;
    const ay = paneY + paneH / 2;
    svg.append('line').attr('x1', arrowMid - 7).attr('y1', ay).attr('x2', arrowMid + 7).attr('y2', ay)
      .attr('stroke', 'rgba(255,255,255,0.7)').attr('stroke-width', 1.4)
      .attr('marker-end', 'url(#strip-arrow)');
    svg.append('text').attr('x', arrowMid).attr('y', paneY + paneH + 14).attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.65)').attr('font-size', '10').text(opLabels[k] || '');
  }

  const defs = svg.append('defs');
  defs.append('marker').attr('id', 'strip-arrow').attr('viewBox', '0 0 10 10').attr('refX', 9)
    .attr('refY', 5).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto-start-reverse')
    .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z').attr('fill', 'rgba(255,255,255,0.7)');

  return {
    unmount() { wrap.remove(); }
  };
}
