function rainbow(t, tMin, tMax) {
  const u = (t - tMin) / Math.max(1e-9, tMax - tMin);
  const h = (1 - u) * 240;
  return `hsl(${h.toFixed(1)}, 80%, 60%)`;
}

export function createViz2d(container, { width = 480, height = 360 } = {}) {
  const d3 = window.d3;
  const wrap = d3.select(container).append('div').attr('class', 'viz2d');
  const svg = wrap.append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%').style('height', '100%');
  const gAxes = svg.append('g').attr('class', 'axes2d');
  const gPoints = svg.append('g').attr('class', 'points2d');

  function setState({ embed2d, colors, t }) {
    if (!embed2d || embed2d.length === 0) return;
    const N = embed2d.length / 2;
    let xmn = Infinity, xmx = -Infinity, ymn = Infinity, ymx = -Infinity;
    for (let i = 0; i < N; i++) {
      const x = embed2d[i*2], y = embed2d[i*2+1];
      if (x < xmn) xmn = x; if (x > xmx) xmx = x;
      if (y < ymn) ymn = y; if (y > ymx) ymx = y;
    }
    const padX = (xmx - xmn) * 0.08 || 1;
    const padY = (ymx - ymn) * 0.08 || 1;
    xmn -= padX; xmx += padX; ymn -= padY; ymx += padY;
    const margin = 28;
    const sx = (width - 2*margin) / Math.max(1e-9, xmx - xmn);
    const sy = (height - 2*margin) / Math.max(1e-9, ymx - ymn);
    const s = Math.min(sx, sy);
    const ox = margin + ((width - 2*margin) - s * (xmx - xmn)) / 2;
    const oy = margin + ((height - 2*margin) - s * (ymx - ymn)) / 2;
    let tMin = Infinity, tMax = -Infinity;
    if (t && !colors) for (let i = 0; i < t.length; i++) { if (t[i] < tMin) tMin = t[i]; if (t[i] > tMax) tMax = t[i]; }
    const data = new Array(N);
    for (let i = 0; i < N; i++) {
      data[i] = {
        i,
        sx: ox + s * (embed2d[i*2] - xmn),
        sy: height - (oy + s * (embed2d[i*2+1] - ymn)),
        col: colors ? colors[i] : (t ? rainbow(t[i], tMin, tMax) : '#7ec8ff'),
      };
    }
    const sel = gPoints.selectAll('circle').data(data, d => d.i);
    sel.enter().append('circle').merge(sel)
      .attr('cx', d => d.sx).attr('cy', d => d.sy)
      .attr('r', 2.8).attr('fill', d => d.col).attr('opacity', 0.9);
    sel.exit().remove();

    const axes = [
      { x1: margin, y1: height - margin, x2: width - margin, y2: height - margin },
      { x1: margin, y1: margin, x2: margin, y2: height - margin },
    ];
    const aSel = gAxes.selectAll('line').data(axes);
    aSel.enter().append('line').merge(aSel)
      .attr('x1', d => d.x1).attr('y1', d => d.y1)
      .attr('x2', d => d.x2).attr('y2', d => d.y2)
      .attr('stroke', 'rgba(255,255,255,0.25)').attr('stroke-width', 1);
    aSel.exit().remove();
  }

  return { setState };
}
