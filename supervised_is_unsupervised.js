// Renamed from article2.js — KDE per-class + posterior p(y=1|x) visualization
document.addEventListener('DOMContentLoaded', () => {
  const csvInput = document.getElementById('csvInput');
  const loadBtn = document.getElementById('loadCsv');
  const fileInput = document.getElementById('fileInput');
  const bandwidthInput = document.getElementById('bandwidth');
  const viz = d3.select('#viz');
  const queryInfo = document.getElementById('queryInfo');
  const classPriorsEl = document.getElementById('classPriors');

  const width = 520, height = 420, margin = { top: 10, right: 10, bottom: 20, left: 40 };
  const innerW = width - margin.left - margin.right; const innerH = height - margin.top - margin.bottom;

  const svg = viz.append('svg').attr('width', width).attr('height', height);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  // GDA UI elements
  const fitGDAButton = document.getElementById('fitGDA');
  const example1Btn = document.getElementById('example1');
  const example2Btn = document.getElementById('example2');
  const gdaParamsEl = document.getElementById('gdaParams');
  const gdaExamplesEl = document.getElementById('gdaExamples');
  const showKDEChk = document.getElementById('showKDE');
  const showGDAChk = document.getElementById('showGDA');

  const xScale = d3.scaleLinear().range([0, innerW]);
  const yScale = d3.scaleLinear().range([innerH, 0]);

  let data = [];

  function parseCSV(text) {
    const rows = d3.csvParse(text);
    const parsed = rows.map(r => ({ x: [ +r.x1, +r.x2 ], y: +r.y }));
    return parsed;
  }

  function loadAndRender(text) {
    try {
      const parsed = parseCSV(text);
      data = parsed;
      // any change to data invalidates previous GDA fit
      gda.fitted = false;
      document.getElementById('priors').textContent = `P(y=0)=–  P(y=1)=–`;
      document.getElementById('mu0').textContent = `μ0 = –`;
      document.getElementById('mu1').textContent = `μ1 = –`;
      document.getElementById('sigma0').textContent = `Σ0 = –`;
      document.getElementById('sigma1').textContent = `Σ1 = –`;
      document.getElementById('exampleOut').textContent = `No examples computed yet.`;
      // clear detailed calculation panel
      const calcP = document.getElementById('calcPriors'); if (calcP) calcP.textContent = 'Priors: —';
      const cm0 = document.getElementById('calcMuSigma0'); if (cm0) cm0.textContent = 'Class 0: μ0 = — ; Σ0 = —';
      const cm1 = document.getElementById('calcMuSigma1'); if (cm1) cm1.textContent = 'Class 1: μ1 = — ; Σ1 = —';
      const cp = document.getElementById('calcPoint'); if (cp) cp.textContent = 'No point computed yet.';
      render();
    } catch (e) {
      alert('Failed to parse CSV: ' + e);
    }
  }

  loadBtn.addEventListener('click', () => loadAndRender(csvInput.value));
  fileInput.addEventListener('change', (ev) => {
    const f = ev.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = e => loadAndRender(e.target.result);
    reader.readAsText(f);
  });

  // re-render when overlay options/bandwidth change
  bandwidthInput && bandwidthInput.addEventListener('input', ()=>{ render(); });
  showKDEChk && showKDEChk.addEventListener('change', ()=>{ render(); });
  showGDAChk && showGDAChk.addEventListener('change', ()=>{ render(); });

  function kdeEstimate(points, bandwidth) {
    const invTwoSigma2 = 1 / (2 * bandwidth * bandwidth);
    const norm = 1 / (2 * Math.PI * bandwidth * bandwidth);
    return function(x) {
      let sum = 0;
      for (const p of points) {
        const dx = x[0] - p[0]; const dy = x[1] - p[1];
        const k = Math.exp(-(dx*dx + dy*dy) * invTwoSigma2);
        sum += k;
      }
      return norm * (sum / points.length);
    };
  }

  function computeGrid(xmin,xmax,ymin,ymax,nx=120,ny=100) {
    const xs = d3.range(nx).map(i => xmin + (xmax - xmin) * i/(nx-1));
    const ys = d3.range(ny).map(j => ymin + (ymax - ymin) * j/(ny-1));
    const grid = [];
    for (let j = 0; j < ys.length; j++) for (let i = 0; i < xs.length; i++) grid.push([xs[i], ys[j]]);
    return { xs, ys, grid, nx, ny };
  }

  function render() {
    if (!data.length) return;
    const xs = data.map(d => d.x[0]); const ys = data.map(d => d.x[1]);
    const xmin = d3.min(xs) - 1, xmax = d3.max(xs) + 1, ymin = d3.min(ys) - 1, ymax = d3.max(ys) + 1;
    xScale.domain([xmin,xmax]); yScale.domain([ymin,ymax]);

    g.selectAll('*').remove();

    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale));
    g.append('g').call(d3.axisLeft(yScale));

    const class0 = data.filter(d => d.y === 0).map(d => d.x);
    const class1 = data.filter(d => d.y === 1).map(d => d.x);
    const prior0 = class0.length / data.length; const prior1 = class1.length / data.length;
    classPriorsEl.textContent = `P(y=0)=${prior0.toFixed(2)}  P(y=1)=${prior1.toFixed(2)}`;

    const bw = Math.max(0.01, +bandwidthInput.value || 0.6);
    const kde0 = class0.length ? kdeEstimate(class0, bw) : (() => 0);
    const kde1 = class1.length ? kdeEstimate(class1, bw) : (() => 0);

    const { xs: gx, ys: gy, grid, nx, ny } = computeGrid(xmin, xmax, ymin, ymax, 150, 120);

    // ensure we have up-to-date overlay checkbox elements (in case DOM changed)
    const showKDEEl = document.getElementById('showKDE');
    const showGDAEl = document.getElementById('showGDA');
    // debug: show overlay states
    console.debug('render overlays', { showKDE: !!showKDEEl && showKDEEl.checked, showGDA: !!showGDAEl && showGDAEl.checked, gdaFitted: gda.fitted });

    // KDE posterior heatmap (only draw if KDE overlay enabled)
    if (showKDEEl && showKDEEl.checked) {
      const post = grid.map(pt => {
        const p0 = kde0(pt) * prior0; const p1 = kde1(pt) * prior1;
        const denom = p0 + p1;
        return denom === 0 ? 0.5 : p1 / denom;
      });

      const rectW = innerW / (nx-1); const rectH = innerH / (ny-1);
      const color = d3.scaleSequential(t => d3.interpolateRdYlBu(1 - t)).domain([0,1]);

      g.append('g').attr('class','heatmap')
        .selectAll('rect').data(grid.map((pt,i)=>({x:pt[0],y:pt[1],v:post[i]}))).enter()
        .append('rect')
        .attr('x', d => xScale(d.x) - rectW/2)
        .attr('y', d => yScale(d.y) - rectH/2)
        .attr('width', Math.max(1, rectW))
        .attr('height', Math.max(1, rectH))
        .attr('fill', d => color(d.v))
        .attr('opacity', 0.7);

      g.append('g').selectAll('circle.contour').data(grid.map((pt,i)=>({x:pt[0],y:pt[1],v:post[i]})).filter(d=>Math.abs(d.v-0.5)<0.03))
        .enter().append('circle').attr('class','contour').attr('cx', d=>xScale(d.x)).attr('cy', d=>yScale(d.y)).attr('r',1.2).attr('fill','#000');
    }

    g.append('g').selectAll('circle.point').data(data).enter().append('circle')
      .attr('class','point')
      .attr('cx', d => xScale(d.x[0]))
      .attr('cy', d => yScale(d.x[1]))
      .attr('r', 4)
      .attr('fill', d => d.y === 1 ? '#ff4d4d' : '#4d9bff')
      .attr('stroke', '#000')
      .on('mouseover', (event,d)=>{
        const p0 = kde0(d.x) * prior0; const p1 = kde1(d.x) * prior1; const denom = p0+p1; const postx = denom===0 ? 0.5 : p1/denom;
        let msg = `Point (${d.x[0]},${d.x[1]}) class=${d.y} → KDE p(y=1|x)=${postx.toFixed(3)}`;
        if (gda.fitted) {
          const gdaRes = computeGDAForPoint(d.x);
          msg += `  | GDA p(y=1|x)=${gdaRes.posterior.toFixed(3)}`;
        }
        queryInfo.textContent = msg;
      });

    svg.on('click', function(event) {
      const [mx,my] = d3.pointer(event, g.node());
      const x = xScale.invert(mx); const y = yScale.invert(my);
      const kdeP0 = kde0([x,y]) * prior0; const kdeP1 = kde1([x,y]) * prior1; const kdeDen = kdeP0 + kdeP1; const kdePost = kdeDen===0 ? 0.5 : kdeP1/kdeDen;
      let msg = `Query (${x.toFixed(2)},${y.toFixed(2)}) → KDE p(y=1|x)=${kdePost.toFixed(3)}  p(x|y=0)=${(kde0([x,y])).toExponential(2)} p(x|y=1)=${(kde1([x,y])).toExponential(2)}`;
      if (gda.fitted) {
        const g = computeGDAForPoint([x,y]);
        msg += `  | GDA p(y=1|x)=${g.posterior.toFixed(3)}  p(x|y=0)=${g.p0.toExponential(2)} p(x|y=1)=${g.p1.toExponential(2)}`;
      }
      queryInfo.textContent = msg;
      // show detailed arithmetic for clicked point
      showDetailedCalculationForPoint([x,y]);
    });

    // GDA overlay: decision boundary and covariance ellipses
    // If the user asked to show GDA but we haven't fitted yet, fit automatically
    if (showGDAEl && showGDAEl.checked && !gda.fitted) {
      fitGDA();
    }

    if (showGDAEl && showGDAEl.checked && gda.fitted) {
      // decision boundary approximation
      const gdaPost = grid.map(pt => {
        const g = computeGDAForPoint(pt);
        return { x: pt[0], y: pt[1], v: g.posterior };
      });
      g.append('g').selectAll('circle.gda-contour').data(gdaPost.filter(d=>Math.abs(d.v-0.5)<0.03))
        .enter().append('circle').attr('class','gda-contour').attr('cx', d=>xScale(d.x)).attr('cy', d=>yScale(d.y)).attr('r',1.4).attr('fill','#fff');

      // draw 1-sigma ellipses for each class
      drawEllipse(gda.mu0, gda.sigma0, '#4d9bff');
      drawEllipse(gda.mu1, gda.sigma1, '#ff4d4d');
    }
  }

  // small linear algebra helpers for 2x2 matrices
  function meanOf(points) {
    const n = points.length;
    const s = points.reduce((acc,p)=>[acc[0]+p[0], acc[1]+p[1]],[0,0]);
    return [s[0]/n, s[1]/n];
  }

  function covOf(points, mu) {
    const n = points.length;
    let a=0,b=0,c=0; // cov = [[a,c],[c,b]]
    for (const p of points) {
      const dx = p[0]-mu[0]; const dy = p[1]-mu[1];
      a += dx*dx; b += dy*dy; c += dx*dy;
    }
    return [[a/n, c/n],[c/n, b/n]];
  }

  function det2(m) { return m[0][0]*m[1][1] - m[0][1]*m[1][0]; }
  function inv2(m) {
    const d = det2(m);
    if (Math.abs(d) < 1e-12) return null;
    return [[ m[1][1]/d, -m[0][1]/d ], [ -m[1][0]/d, m[0][0]/d ]];
  }

  function mvnPdf(x, mu, sigma) {
    // 2D multivariate normal pdf
    const eps = 1e-6;
    const s = [[sigma[0][0]+eps, sigma[0][1]],[sigma[1][0], sigma[1][1]+eps]];
    const inv = inv2(s);
    const d = det2(s);
    if (!inv || d <= 0) return 0;
    const dx = [x[0]-mu[0], x[1]-mu[1]];
    const q = dx[0]*(inv[0][0]*dx[0] + inv[0][1]*dx[1]) + dx[1]*(inv[1][0]*dx[0] + inv[1][1]*dx[1]);
    const norm = 1/(2*Math.PI*Math.sqrt(d));
    return norm * Math.exp(-0.5*q);
  }

  const gda = { fitted: false };

  function fitGDA() {
    const class0 = data.filter(d=>d.y===0).map(d=>d.x);
    const class1 = data.filter(d=>d.y===1).map(d=>d.x);
    if (class0.length===0 || class1.length===0) {
      alert('Need at least one point in each class to fit GDA');
      return;
    }
    gda.mu0 = meanOf(class0);
    gda.mu1 = meanOf(class1);
    gda.sigma0 = covOf(class0, gda.mu0);
    gda.sigma1 = covOf(class1, gda.mu1);
    gda.prior0 = class0.length / data.length; gda.prior1 = class1.length / data.length;
    gda.fitted = true;
    // update UI
    document.getElementById('priors').textContent = `P(y=0)=${gda.prior0.toFixed(3)}  P(y=1)=${gda.prior1.toFixed(3)}`;
    document.getElementById('mu0').textContent = `μ0 = [${gda.mu0.map(v=>v.toFixed(3)).join(', ')}]`;
    document.getElementById('mu1').textContent = `μ1 = [${gda.mu1.map(v=>v.toFixed(3)).join(', ')}]`;
    document.getElementById('sigma0').textContent = `Σ0 = [${gda.sigma0[0][0].toFixed(3)} ${gda.sigma0[0][1].toFixed(3)}; ${gda.sigma0[1][0].toFixed(3)} ${gda.sigma0[1][1].toFixed(3)}]`;
    document.getElementById('sigma1').textContent = `Σ1 = [${gda.sigma1[0][0].toFixed(3)} ${gda.sigma1[0][1].toFixed(3)}; ${gda.sigma1[1][0].toFixed(3)} ${gda.sigma1[1][1].toFixed(3)}]`;
    // update detailed numeric steps area
    document.getElementById('calcPriors').textContent = `Priors: P(y=0)=${gda.prior0.toFixed(3)}; P(y=1)=${gda.prior1.toFixed(3)}`;
    document.getElementById('calcMuSigma0').textContent = `Class 0: μ0 = [${gda.mu0.map(v=>v.toFixed(4)).join(', ')}]; Σ0 = [${gda.sigma0[0][0].toFixed(4)} ${gda.sigma0[0][1].toFixed(4)}; ${gda.sigma0[1][0].toFixed(4)} ${gda.sigma0[1][1].toFixed(4)}]`;
    document.getElementById('calcMuSigma1').textContent = `Class 1: μ1 = [${gda.mu1.map(v=>v.toFixed(4)).join(', ')}]; Σ1 = [${gda.sigma1[0][0].toFixed(4)} ${gda.sigma1[0][1].toFixed(4)}; ${gda.sigma1[1][0].toFixed(4)} ${gda.sigma1[1][1].toFixed(4)}]`;
    render();
  }

  function computeGDAForPoint(x) {
    if (!gda.fitted) return { p0:0, p1:0, posterior:0.5 };
    const p0 = mvnPdf(x, gda.mu0, gda.sigma0) * gda.prior0;
    const p1 = mvnPdf(x, gda.mu1, gda.sigma1) * gda.prior1;
    const denom = p0 + p1;
    return { p0, p1, posterior: denom===0 ? 0.5 : p1/denom };
  }

  function showDetailedCalculationForPoint(x) {
    // compute both KDE and GDA contributions and show the arithmetic
    const class0 = data.filter(d=>d.y===0).map(d=>d.x);
    const class1 = data.filter(d=>d.y===1).map(d=>d.x);
    const prior0 = class0.length / data.length; const prior1 = class1.length / data.length;
    const bw = Math.max(0.01, +bandwidthInput.value || 0.6);
    const kde0 = class0.length ? kdeEstimate(class0, bw) : (()=>0);
    const kde1 = class1.length ? kdeEstimate(class1, bw) : (()=>0);
    const k0 = kde0(x); const k1 = kde1(x);
    const num0 = k0 * prior0; const num1 = k1 * prior1; const den = num0 + num1; const kdePosterior = den===0 ? 0.5 : num1/den;
    const g = computeGDAForPoint(x);

    const s = [];
    s.push(`Query point x* = (${x[0].toFixed(4)}, ${x[1].toFixed(4)})`);
    s.push(`Bandwidth h = ${bw}`);
    s.push(`KDE likelihoods: p(x*|y=0) ≈ ${k0.toExponential(4)}, p(x*|y=1) ≈ ${k1.toExponential(4)}`);
    s.push(`KDE numerators: num_0 = p(y=0)*p(x*|y=0) = ${prior0.toFixed(3)} * ${k0.toExponential(4)} = ${num0.toExponential(4)}`);
    s.push(`num_1 = p(y=1)*p(x*|y=1) = ${prior1.toFixed(3)} * ${k1.toExponential(4)} = ${num1.toExponential(4)}`);
    s.push(`KDE posterior p(y=1|x*) = num_1 / (num_0 + num_1) = ${kdePosterior.toFixed(6)}`);
    s.push('---');
    s.push('GDA (Gaussian) calculations:');
    s.push(`p(x*|y=0) (Gaussian) ≈ ${g.p0.toExponential(4)}`);
    s.push(`p(x*|y=1) (Gaussian) ≈ ${g.p1.toExponential(4)}`);
    s.push(`GDA numerators: ${ (gda.prior0).toFixed(3) } * ${g.p0.toExponential(4)} = ${(g.p0 * gda.prior0).toExponential(4)} ; ${ (gda.prior1).toFixed(3)} * ${g.p1.toExponential(4)} = ${(g.p1 * gda.prior1).toExponential(4)}`);
    s.push(`GDA posterior p(y=1|x*) = ${g.posterior.toFixed(6)}`);

    document.getElementById('calcPoint').innerHTML = s.map(l=>`<div>${l}</div>`).join('');
  }

  function drawEllipse(mu, sigma, color) {
    // compute eigen decomposition of sigma (2x2)
    const a = sigma[0][0], b = sigma[0][1], c = sigma[1][1];
    const tr = a + c; const det = a*c - b*b;
    const term = Math.sqrt(Math.max(0, tr*tr/4 - det));
    const l1 = tr/2 + term; const l2 = tr/2 - term; // eigenvalues
    // eigenvector for l1
    let v1;
    if (Math.abs(b) > 1e-8) v1 = [l1 - c, b]; else v1 = [1,0];
    const len = Math.hypot(v1[0], v1[1]); v1 = [v1[0]/len, v1[1]/len];
    const angle = Math.atan2(v1[1], v1[0]);
    const sx = Math.sqrt(Math.abs(l1)); const sy = Math.sqrt(Math.abs(l2));
    // create ellipse path points
    const n = 80; const pts = d3.range(n).map(i => {
      const t = (i/n) * 2*Math.PI; const ex = Math.cos(t); const ey = Math.sin(t);
      // scale by sigma stddev
      const rx = ex * sx; const ry = ey * sy;
      // rotate by angle
      const rxr = rx * Math.cos(angle) - ry * Math.sin(angle);
      const ryr = rx * Math.sin(angle) + ry * Math.cos(angle);
      return [mu[0] + rxr, mu[1] + ryr];
    });
    g.append('path').attr('d', d3.line().x(d=>xScale(d[0])).y(d=>yScale(d[1]))(pts))
      .attr('stroke', color).attr('stroke-width', 1.2).attr('fill','none').attr('opacity',0.9);
  }

  fitGDAButton && fitGDAButton.addEventListener('click', fitGDA);
  example1Btn && example1Btn.addEventListener('click', ()=>{
    if (!gda.fitted) fitGDA();
    const pt = [2,2]; const r = computeGDAForPoint(pt);
    document.getElementById('exampleOut').textContent = `x*=(2,2): p(x|y=0)=${r.p0.toExponential(3)} p(x|y=1)=${r.p1.toExponential(3)} → p(y=1|x)=${r.posterior.toFixed(3)}`;
    showDetailedCalculationForPoint(pt);
  });
  example2Btn && example2Btn.addEventListener('click', ()=>{
    if (!gda.fitted) fitGDA();
    const pt = [4.8,4.7]; const r = computeGDAForPoint(pt);
    document.getElementById('exampleOut').textContent = `x*=(4.8,4.7): p(x|y=0)=${r.p0.toExponential(3)} p(x|y=1)=${r.p1.toExponential(3)} → p(y=1|x)=${r.posterior.toFixed(3)}`;
    showDetailedCalculationForPoint(pt);
  });

  showKDEChk && showKDEChk.addEventListener('change', render);
  showGDAChk && showGDAChk.addEventListener('change', render);

  loadAndRender(csvInput.value);
});
