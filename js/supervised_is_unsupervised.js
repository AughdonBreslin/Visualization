document.addEventListener('DOMContentLoaded', () => {
  const csvInput = document.getElementById('csvInput');
  const loadBtn = document.getElementById('loadCsv');
  const fileInput = document.getElementById('fileInput');
  const bandwidthInput = document.getElementById('bandwidth');
  const viz = d3.select('#viz');
  const queryInfo = document.getElementById('queryInfo');
  const classPriorsEl = document.getElementById('classPriors');
  const gdaParamsEl = document.getElementById('gdaParams');

  const width = 520, height = 420, margin = { top: 10, right: 10, bottom: 20, left: 40 };
  const innerW = width - margin.left - margin.right; const innerH = height - margin.top - margin.bottom;

  const svg = viz.append('svg').attr('width', width).attr('height', height);
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  // GDA UI elements
  const fitGDAButton = document.getElementById('fitGDA');
  const example1Btn = document.getElementById('example1');
  const example2Btn = document.getElementById('example2');
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
      // clear previous GDA placeholders and dynamic class UI
      if (gdaParamsEl) gdaParamsEl.innerHTML = '<div>Fitted parameters will appear here</div>';
      const calcP = document.getElementById('calcPriors'); if (calcP) calcP.textContent = 'Priors: —';
      const cm0 = document.getElementById('calcMuSigma0'); if (cm0) cm0.textContent = 'Class k: μ_k = — ; Σ_k = —';
      const cp = document.getElementById('calcPoint'); if (cp) cp.textContent = 'No point computed yet.';
      // reset GDA fit state
      gda.fitted = false; gda.classes = []; gda.params = {};
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
    reader.onload = e => {
      // Place the file contents into the csvInput textarea
      if (csvInput) csvInput.value = e.target.result;
      loadAndRender(e.target.result);
    };
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

  function computeKDEForPoint(x, classes, kdes, priors) {
    // Mirrors computeGDAForPoint(): likelihoods -> joint numerators -> normalized posteriors
    const likelihoods = classes.map((_, i) => kdes[i](x));
    const numerators = likelihoods.map((l, i) => l * priors[i]);
    const denom = numerators.reduce((a, b) => a + b, 0);
    const posts = denom === 0 ? numerators.map(() => 1 / Math.max(1, numerators.length)) : numerators.map(v => v / denom);
    const predicted = classes.length ? classes[posts.indexOf(Math.max(...posts))] : null;
    return {
      classLikelihoods: likelihoods,
      weightedNumerators: numerators,
      denominator: denom,
      classPosteriors: posts,
      predicted
    };
  }

  function render() {
    if (!data.length) return;
    const xs = data.map(d => d.x[0]); const ys = data.map(d => d.x[1]);
    const xmin = d3.min(xs) - 1, xmax = d3.max(xs) + 1, ymin = d3.min(ys) - 1, ymax = d3.max(ys) + 1;
    xScale.domain([xmin,xmax]); yScale.domain([ymin,ymax]);

    g.selectAll('*').remove();

    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale));
    g.append('g').call(d3.axisLeft(yScale));

    // support arbitrary classes by discovering unique labels
    const classes = Array.from(new Set(data.map(d => d.y))).sort((a,b)=> (a < b ? -1 : a > b ? 1 : 0));
    const classPoints = classes.map(c => data.filter(d => d.y === c).map(d => d.x));
    const priors = classPoints.map(arr => arr.length / data.length);

    const bw = Math.max(0.01, +bandwidthInput.value || 0.6);
    const kdes = classPoints.map(arr => arr.length ? kdeEstimate(arr, bw) : (() => 0));

    const { xs: gx, ys: gy, grid, nx, ny } = computeGrid(xmin, xmax, ymin, ymax, 150, 120);

    // categorical color mapping for classes
    const classColor = d3.scaleOrdinal(d3.schemeCategory10).domain(classes);

    // Populate external DOM legend (#classLegend) with class swatches and counts
    const classLegendEl = document.getElementById('classLegend');
    if (classLegendEl) {
      classLegendEl.innerHTML = '';
      if (classes.length > 0) {
        for (let i = 0; i < classes.length; i++) {
          const c = classes[i];
          const color = classColor(c);
          const count = classPoints[i].length;
          const prior = priors[i];

          const item = document.createElement('div');
          item.className = 'legend-item';
          item.style.display = 'flex'; item.style.alignItems = 'center'; item.style.marginBottom = '6px';

          const sw = document.createElement('span');
          sw.style.width = '14px'; sw.style.height = '14px'; sw.style.display = 'inline-block';
          sw.style.marginRight = '8px'; sw.style.border = '1px solid #000';
          sw.style.background = color;

          const label = document.createElement('span');
          label.textContent = `Class ${c} (n=${count}, p=${prior.toFixed(3)})`;
          label.style.fontSize = '13px';

          item.appendChild(sw); item.appendChild(label);
          classLegendEl.appendChild(item);
        }
      } else {
        classLegendEl.textContent = 'No classes loaded';
      }
    }


    // ensure we have up-to-date overlay checkbox elements (in case DOM changed)
    const showKDEEl = document.getElementById('showKDE');
    const showGDAEl = document.getElementById('showGDA');
    // debug: show overlay states
    console.debug('render overlays', { showKDE: !!showKDEEl && showKDEEl.checked, showGDA: !!showGDAEl && showGDAEl.checked, gdaFitted: gda.fitted });

    // KDE posterior heatmap (only draw if KDE overlay enabled)
    if (showKDEEl && showKDEEl.checked) {
      const postInfo = grid.map(pt => {
        const kde = computeKDEForPoint(pt, classes, kdes, priors);
        const maxIdx = kde.classPosteriors.indexOf(Math.max(...kde.classPosteriors));
        return { x: pt[0], y: pt[1], posts: kde.classPosteriors, maxIdx, maxVal: kde.classPosteriors[maxIdx] };
      });

      const rectW = innerW / (nx-1); const rectH = innerH / (ny-1);

      g.append('g').attr('class','heatmap')
        .selectAll('rect').data(postInfo).enter()
        .append('rect')
        .attr('x', d => xScale(d.x) - rectW/2)
        .attr('y', d => yScale(d.y) - rectH/2)
        .attr('width', Math.max(1, rectW))
        .attr('height', Math.max(1, rectH))
        .attr('fill', d => {
          // blend the class color toward white for a pastel (less bright) heatmap fill
          const base = classColor(classes[d.maxIdx]);
          return d3.interpolateRgb(base, '#000000')(0.3);
        })
        .attr('opacity', d => Math.max(0.12, d.maxVal * 0.7));

      // draw soft decision boundary where top two classes are close
      g.append('g').selectAll('circle.contour').data(postInfo.filter(d=>{
        const ps = d.posts.slice().sort((a,b)=>b-a); return (ps[0]-ps[1]) < 0.06;
      }))
        .enter().append('circle').attr('class','contour').attr('cx', d=>xScale(d.x)).attr('cy', d=>yScale(d.y)).attr('r',1.2).attr('fill','#000');
    }

    g.append('g').selectAll('circle.point').data(data).enter().append('circle')
      .attr('class','point')
      .attr('cx', d => xScale(d.x[0]))
      .attr('cy', d => yScale(d.x[1]))
      .attr('r', 4)
      .attr('fill', d => classColor(d.y))
      .attr('stroke', '#000')
      .on('mouseover', (event,d)=>{
        // compute KDE-based posteriors and likelihoods for this point
        const kde = computeKDEForPoint(d.x, classes, kdes, priors);

        // build header: posterior columns then likelihood columns
        const headerCols = classes.map(c=>`p(y=${c}|x)`).concat(classes.map(c=>`p(x|y=${c})`));
        const colCount = headerCols.length + 1; // +1 for row header

        // KDE row values
        const kdeVals = [];
        for (let i=0;i<classes.length;i++) kdeVals.push(kde.classPosteriors[i].toFixed(3));
        for (let i=0;i<classes.length;i++) kdeVals.push((kde.classLikelihoods[i]).toExponential(2));

        // assemble table
        let msg = `<table>`;
        msg += `<tr><th colspan="${colCount}">Point (${d.x[0].toFixed(2)},${d.x[1].toFixed(2)}) class=${d.y}</th></tr>`;
        msg += `<tr><th></th>${headerCols.map(h=>`<th>${h}</th>`).join('')}</tr>`;
        msg += `<tr><td style="font-weight:bold">KDE</td>${kdeVals.map(v=>`<td>${v}</td>`).join('')}</tr>`;

        // optional GDA row
        if (gda.fitted) {
          const gdaRes = computeGDAForPoint(d.x);
          const gdaVals = [];
          for (let i=0;i<gda.classes.length;i++) gdaVals.push(gdaRes.classPosteriors[i].toFixed(3));
          for (let i=0;i<gda.classes.length;i++) gdaVals.push((gdaRes.classLikelihoods[i]).toExponential(2));
          msg += `<tr><td style="font-weight:bold">GDA</td>${gdaVals.map(v=>`<td>${v}</td>`).join('')}</tr>`;
        }

        msg += `</table>`;
        queryInfo.innerHTML = msg;
      });

    svg.on('click', function(event) {
      const [mx,my] = d3.pointer(event, g.node());
      const x = xScale.invert(mx); const y = yScale.invert(my);

      // KDE posteriors and likelihoods
      const kde = computeKDEForPoint([x,y], classes, kdes, priors);
      const kdeVals = [];
      for (let i=0;i<classes.length;i++) kdeVals.push(kde.classPosteriors[i].toFixed(3));
      for (let i=0;i<classes.length;i++) kdeVals.push((kde.classLikelihoods[i]).toExponential(2));

      // build header columns
      const headerCols = classes.map(c=>`p(y=${c}|x)`).concat(classes.map(c=>`p(x|y=${c})`));
      const colCount = headerCols.length + 1;

      let msg = `<table>`;
      msg += `<tr><th colspan="${colCount}">Query (${x.toFixed(2)},${y.toFixed(2)})</th></tr>`;
      msg += `<tr><th></th>${headerCols.map(h=>`<th>${h}</th>`).join('')}</tr>`;
      msg += `<tr><td style="font-weight:bold">KDE</td>${kdeVals.map(v=>`<td>${v}</td>`).join('')}</tr>`;

      // ensure GDA parameters exist
      if (!gda.fitted) fitGDA_new();
      if (gda.fitted) {
        const g = computeGDAForPoint([x,y]);
        const gdaVals = [];
        for (let i=0;i<gda.classes.length;i++) gdaVals.push(g.classPosteriors[i].toFixed(3));
        for (let i=0;i<gda.classes.length;i++) gdaVals.push((g.classLikelihoods[i]).toExponential(2));
        msg += `<tr><td style="font-weight:bold">GDA</td>${gdaVals.map(v=>`<td>${v}</td>`).join('')}</tr>`;
      }

      msg += `</table>`;
      queryInfo.innerHTML = msg;
      // show detailed arithmetic for clicked point
      showDetailedCalculationForPoint([x,y]);
    });

    // GDA overlay: decision boundary and covariance ellipses
    // If the user asked to show GDA but we haven't fitted yet, fit automatically
    if (showGDAEl && showGDAEl.checked && !gda.fitted) {
      fitGDA_new();
    }

    if (showGDAEl && showGDAEl.checked && gda.fitted) {
      // compute posteriors for grid points and mark soft boundaries where top classes are close
      const gdaPost = grid.map(pt => {
        const g = computeGDAForPoint(pt);
        // determine top-two gap
        const sorted = g.classPosteriors.slice().sort((a,b)=>b-a);
        const gap = sorted[0] - (sorted[1] || 0);
        return { x: pt[0], y: pt[1], gap, topIdx: g.classPosteriors.indexOf(Math.max(...g.classPosteriors)), posts: g.classPosteriors };
      });

      // draw crisp decision boundaries: connect grid neighboring cells where the top predicted class changes
      const boundaryLines = [];
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const idx = j * nx + i;
          const cur = gdaPost[idx];
          if (!cur) continue;
          // right neighbor
          if (i < nx - 1) {
            const right = gdaPost[idx + 1];
            if (right && cur.topIdx !== right.topIdx) {
              boundaryLines.push({ x1: xScale(cur.x), y1: yScale(cur.y), x2: xScale(right.x), y2: yScale(right.y) });
            }
          }
          // down neighbor
          if (j < ny - 1) {
            const down = gdaPost[idx + nx];
            if (down && cur.topIdx !== down.topIdx) {
              boundaryLines.push({ x1: xScale(cur.x), y1: yScale(cur.y), x2: xScale(down.x), y2: yScale(down.y) });
            }
          }
        }
      }

      // main thin dark boundary line
      g.append('g').attr('class','gda-boundary')
        .selectAll('line').data(boundaryLines).enter().append('line')
        .attr('x1', d=>d.x1).attr('y1', d=>d.y1).attr('x2', d=>d.x2).attr('y2', d=>d.y2)
        .attr('stroke', '#fff').attr('stroke-width', 1.2).attr('opacity', 1).attr('stroke-linecap', 'round');
    
      // draw 1-sigma ellipses for each class using their color
      for (let i=0;i<gda.classes.length;i++) {
        const c = gda.classes[i];
        drawEllipse(gda.params[c].mu, gda.params[c].sigma, classColor(c));
      }
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

  const gda = { fitted: false, classes: [], params: {} };

  function fitGDA_legacy() { /* legacy binary GDA (kept for fallback) */
    // discover classes and compute mean/cov/prior per class
    const classes = Array.from(new Set(data.map(d=>d.y))).sort((a,b)=> (a < b ? -1 : a > b ? 1 : 0));
    const params = {};
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
    const prior0El = document.getElementById('prior0');
    const prior1El = document.getElementById('prior1');
    if (prior0El) prior0El.innerHTML = `$P(y=0)=${gda.prior0.toFixed(3)}$`;
    if (prior1El) prior1El.innerHTML = `$P(y=1)=${gda.prior1.toFixed(3)}$`;

    // set GDA parameter displays using TeX and MathJax
    const mu0El = document.getElementById('mu0');
    const mu1El = document.getElementById('mu1');
    const sigma0El = document.getElementById('sigma0');
    const sigma1El = document.getElementById('sigma1');

    if (mu0El) mu0El.innerHTML = `$\\mu_0 = [${gda.mu0.map(v=>v.toFixed(3)).join(', ')}]$`;
    if (mu1El) mu1El.innerHTML = `$\\mu_1 = [${gda.mu1.map(v=>v.toFixed(3)).join(', ')}]$`;
    if (sigma0El) sigma0El.innerHTML = `$\\Sigma_0 = \\begin{bmatrix}${gda.sigma0[0][0].toFixed(3)} & ${gda.sigma0[0][1].toFixed(3)} \\\\ ${gda.sigma0[1][0].toFixed(3)} & ${gda.sigma0[1][1].toFixed(3)}\\end{bmatrix}$`;
    if (sigma1El) sigma1El.innerHTML = `$\\Sigma_1 = \\begin{bmatrix}${gda.sigma1[0][0].toFixed(3)} & ${gda.sigma1[0][1].toFixed(3)} \\\\ ${gda.sigma1[1][0].toFixed(3)} & ${gda.sigma1[1][1].toFixed(3)}\\end{bmatrix}$`;

    // update detailed numeric steps area and set as TeX
    const calcP = document.getElementById('calcPriors');
    if (calcP) calcP.innerHTML = `$P(y=0)=${gda.prior0.toFixed(3)};\\; P(y=1)=${gda.prior1.toFixed(3)}$`;

    // typeset only the updated elements (if MathJax is available)
    const typesetEls = [prior0El, prior1El, mu0El, mu1El, sigma0El, sigma1El, calcP].filter(Boolean);
    if (typesetEls.length > 0 && window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetPromise(typesetEls).catch(err => console.warn('MathJax typeset failed:', err));
    } else if (typesetEls.length > 0 && window.MathJax && MathJax.typeset) {
      try { MathJax.typeset(typesetEls); } catch (e) { console.warn('MathJax typeset failed:', e); }
    }

    // update detailed numeric steps area
    // calcPriors set as TeX and typeset above
    document.getElementById('calcMuSigma0').textContent = `Class 0: μ0 = [${gda.mu0.map(v=>v.toFixed(3)).join(', ')}]; Σ0 = [${gda.sigma0[0][0].toFixed(3)} ${gda.sigma0[0][1].toFixed(3)}; ${gda.sigma0[1][0].toFixed(3)} ${gda.sigma0[1][1].toFixed(3)}]`;
    render();
  }

  function fitGDA_new() {
    // discover classes and compute mean/cov/prior per class
    const classes = Array.from(new Set(data.map(d=>d.y))).sort((a,b)=> (a < b ? -1 : a > b ? 1 : 0));
    const params = {};
    for (const c of classes) {
      const pts = data.filter(d=>d.y===c).map(d=>d.x);
      if (pts.length === 0) {
        alert(`Need at least one point in class ${c} to fit GDA`);
        return;
      }
      const mu = meanOf(pts);
      const sigma = covOf(pts, mu);
      const prior = pts.length / data.length;
      params[c] = { mu, sigma, prior };
    }
    gda.classes = classes; gda.params = params; gda.fitted = true;

    // render GDA params into the #gdaParams container per class (prior, μ, Σ)
    if (gdaParamsEl) {
      const blocks = classes.map(c => {
        const p = params[c];
        return `<div class="gda-class" style="margin-bottom:8px;padding:6px;border-bottom:1px solid #eee;"><div style="font-weight:bold">Class ${c}</div><div>P(y=${c}) = ${p.prior.toFixed(3)}</div><div>$$\\mu = [${p.mu.map(v=>v.toFixed(3)).join(', ')}]$$</div><div>$$\\Sigma = \\begin{bmatrix}${p.sigma[0][0].toFixed(3)} & ${p.sigma[0][1].toFixed(3)} \\\\ ${p.sigma[1][0].toFixed(3)} & ${p.sigma[1][1].toFixed(3)}\\end{bmatrix}$$</div></div>`;
      }).join('');
      gdaParamsEl.innerHTML = blocks;
    }

    // detailed numeric steps
    const calcP = document.getElementById('calcPriors');
    if (calcP) calcP.innerHTML = `<h3>Priors</h3>` + classes.map(c => `P(y=${c})=${params[c].prior.toFixed(3)}`).join('\; ');

    // typeset TeX blocks if MathJax is available
    const typesetEls = [gdaParamsEl, calcP].filter(Boolean);
    if (typesetEls.length > 0 && window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetPromise(typesetEls).catch(err => console.warn('MathJax typeset failed:', err));
    } else if (typesetEls.length > 0 && window.MathJax && MathJax.typeset) {
      try { MathJax.typeset(typesetEls); } catch (e) { console.warn('MathJax typeset failed:', e); }
    }

    const lines = [`<h3>Gaussian Parameters</h3>`];
    for (const c of classes) {
      const p = params[c];
      lines.push(`<p>Class ${c}: μ = [${p.mu.map(v=>v.toFixed(3)).join(', ')}]; Σ = [${p.sigma[0][0].toFixed(3)} ${p.sigma[0][1].toFixed(3)}; ${p.sigma[1][0].toFixed(3)} ${p.sigma[1][1].toFixed(3)}]</p>`);
    }
    document.getElementById('calcMuSigma0').innerHTML = lines.join('');

    render();
  }

  function computeGDAForPoint(x) {
    if (!gda.fitted) {
      return {
        classLikelihoods: [],
        weightedNumerators: [],
        denominator: 0,
        classPosteriors: [],
        predicted: null
      };
    }

    // Raw Gaussian likelihoods p(x|y=c)
    const likelihoods = gda.classes.map(c => mvnPdf(x, gda.params[c].mu, gda.params[c].sigma));
    // Weighted numerators p(x|y=c) P(y=c)
    const numerators = likelihoods.map((l, i) => l * gda.params[gda.classes[i]].prior);
    const denom = numerators.reduce((a,b)=>a+b,0);
    const posts = denom === 0 ? numerators.map(()=>1/numerators.length) : numerators.map(v=>v/denom);
    const predicted = gda.classes[posts.indexOf(Math.max(...posts))];
    return {
      classLikelihoods: likelihoods,
      weightedNumerators: numerators,
      denominator: denom,
      classPosteriors: posts,
      predicted
    };
  }

  function showDetailedCalculationForPoint(x) {
    // compute per-class KDE and GDA contributions and show the arithmetic
    const classes = Array.from(new Set(data.map(d=>d.y))).sort((a,b)=> (a < b ? -1 : a > b ? 1 : 0));
    const classPoints = classes.map(c => data.filter(d=>d.y===c).map(d=>d.x));
    const priors = classPoints.map(arr => arr.length / data.length);
    const bw = Math.max(0.01, +bandwidthInput.value || 0.6);
    const kdesLocal = classPoints.map(arr => arr.length ? kdeEstimate(arr, bw) : (()=>0));
    const kde = computeKDEForPoint(x, classes, kdesLocal, priors);
    const ks = kde.classLikelihoods;
    const nums = kde.weightedNumerators;
    const den = kde.denominator;
    const kdePosts = kde.classPosteriors;
    const g = computeGDAForPoint(x);

    const s = [];
    s.push(`Query point $x^* = (${x[0].toFixed(3)}, ${x[1].toFixed(3)})$`);
    s.push(`Bandwidth $h = ${bw}$`);
    s.push('');
    s.push('KDE:');
    s.push(`Step 1: Estimate class-conditional densities — Use kernel density estimation with ${classPoints.map(a=>a.length).join(', ')} samples per class.`);
    s.push(`Formula: $p(x^*|y=k) = \\dfrac{1}{n_k} \\sum_{i=1}^{n_k} K_h(x^* - x_i^{(k)})$ where $K_h$ is a Gaussian kernel with bandwidth $h$.`);
    s.push(`Gaussian kernel (2D): $K_h(u)=\\dfrac{1}{2\\pi h^2}\\exp\\left(-\\dfrac{\\|u\\|^2}{2h^2}\\right)$.`);
    for (let i=0;i<classes.length;i++) {
      const pts = classPoints[i];
      const n = pts.length;
      const norm = 1 / (2 * Math.PI * bw * bw);
      const invTwoH2 = 1 / (2 * bw * bw);

      s.push(`KDE Class ${classes[i]}`);

      s.push(`Samples: $n_k = ${n}$`);
      s.push(`Constant: $\\dfrac{1}{2\\pi h^2} = ${norm.toExponential(2)}$`);

      // Show a few representative terms so the UI doesn't explode on big datasets.
      const maxTerms = 6;
      let sumK = 0;
      for (let j=0;j<n;j++) {
        const dx0 = x[0] - pts[j][0];
        const dx1 = x[1] - pts[j][1];
        const r2 = dx0*dx0 + dx1*dx1;
        const kTerm = Math.exp(-r2 * invTwoH2);
        sumK += kTerm;

        if (j < maxTerms) {
          s.push(`Point ${j+1}: $x_i=(${pts[j][0]},${pts[j][1]})$, $\\|x^*-x_i\\|^2=${r2.toFixed(3)}$, $\\exp\\left(-\\frac{\\|x^*-x_i\\|^2}{2h^2}\\right)=${kTerm.toExponential(2)}$`);
        }
      }

      if (n > maxTerms) s.push(`(… ${n - maxTerms} more points omitted …)`);

      const density = norm * (sumK / n);
      s.push(`Sum: $\\sum_i \\exp\\left(-\\frac{\\|x^*-x_i\\|^2}{2h^2}\\right) = ${sumK.toExponential(2)}$`);
      s.push(`Plug in: $p(x^*|y=${classes[i]}) = ${norm.toExponential(2)} \\times \\frac{${sumK.toExponential(2)}}{${n}} = ${density.toExponential(2)}$`);
    }
    s.push('');
    s.push('Step 2: Compute joint probabilities — $p(x^*,y=k)=p(y=k)p(x^*|y=k)$.');
    for (let i=0;i<classes.length;i++) {
      s.push(`$p(x^*,y=${classes[i]}) = ${priors[i].toFixed(3)} \\times ${ks[i].toExponential(2)} = ${nums[i].toExponential(2)}$`);
    }
    s.push('');
    s.push('Step 3: Compute posteriors — $p(y=k|x^*) = \\dfrac{p(x^*,y=k)}{p(x^*)}$ where $p(x^*) = \\sum_j p(x^*,y=j)$.');
    s.push(`$p(x^*) = ${nums.map(n=>n.toExponential(2)).join(' + ')} = ${den.toExponential(2)}$`);
    for (let i=0;i<classes.length;i++) {
      s.push(`$p(y=${classes[i]}|x^*) = \\frac{p(x^*,y=${classes[i]})}{p(x^*)} = \\frac{${nums[i].toExponential(2)}}{${den.toExponential(2)}} = ${kdePosts[i].toFixed(3)}$`);
    }
    s.push('');
    s.push('GDA:');
    s.push('Step 1: Estimate class-conditional densities — Using the fitted Gaussian parameters.');
    s.push(`Formula: $p(x^*|y=k) = \\frac{1}{2\\pi\\sqrt{|\\Sigma_k|}} \\exp\\left(-\\frac{1}{2}(x^* - \\mu_k)^T \\Sigma_k^{-1} (x^* - \\mu_k)\\right)$`);
    for (let i=0;i<gda.classes.length;i++) {
      const c = gda.classes[i];
      const p = gda.params[c];
      const mu = p.mu;
      const sigma = p.sigma;

      s.push(`GDA Class ${c}`);

      // Match mvnPdf()'s numerical stabilization so the displayed arithmetic matches the computed value.
      const eps = 1e-6;
      const sSigma = [[sigma[0][0] + eps, sigma[0][1]], [sigma[1][0], sigma[1][1] + eps]];
      const det = det2(sSigma);
      const inv = inv2(sSigma);
      const dx = [x[0] - mu[0], x[1] - mu[1]];
      const q = (!inv || det <= 0)
        ? NaN
        : (dx[0] * (inv[0][0] * dx[0] + inv[0][1] * dx[1]) + dx[1] * (inv[1][0] * dx[0] + inv[1][1] * dx[1]));
      const norm = (!inv || det <= 0) ? 0 : (1 / (2 * Math.PI * Math.sqrt(det)));
      const plugged = (!inv || det <= 0) ? 0 : (norm * Math.exp(-0.5 * q));

      s.push(`$\\mu = [${mu.map(v=>v.toFixed(3)).join(', ')}]$, $\\Sigma = \\begin{bmatrix} ${sigma[0][0].toFixed(3)} & ${sigma[0][1].toFixed(3)} \\\\ ${sigma[1][0].toFixed(3)} & ${sigma[1][1].toFixed(3)} \\end{bmatrix}$`);
      s.push(`Compute: $x^* - \\mu = [${dx.map(v=>v.toFixed(3)).join(', ')}]$`);
      if (!inv || det <= 0) {
        s.push(`Compute: $|\\Sigma| \\le 0$ or not invertible (numerically); using density $0$.`);
      } else {
        s.push(`Compute: $|\\Sigma| = ${det.toExponential(2)}$, $\\Sigma^{-1} = \\begin{bmatrix} ${inv[0][0].toFixed(3)} & ${inv[0][1].toFixed(3)} \\\\ ${inv[1][0].toFixed(3)} & ${inv[1][1].toFixed(3)} \\end{bmatrix}$`);
        s.push(`Compute: $(x^* - \\mu)^T\\Sigma^{-1}(x^* - \\mu) = ${q.toFixed(3)}$`);
        s.push(`Plug in: $p(x^*|y=${c}) = \\frac{1}{2\\pi\\sqrt{${det.toExponential(2)}}}\\exp\\left(-\\frac{1}{2}\\cdot ${q.toFixed(3)}\\right) = ${plugged.toExponential(2)}$`);
      }
    }
    s.push('');
    s.push('Step 2: Compute joint probabilities — $p(x^*,y=k)=p(y=k)p(x^*|y=k)$.');
    for (let i=0;i<gda.classes.length;i++) {
      const c = gda.classes[i];
      s.push(`$p(x^*,y=${c}) = ${gda.params[c].prior.toFixed(3)} \\times ${g.classLikelihoods[i].toExponential(2)} = ${g.weightedNumerators[i].toExponential(2)}$`);
    }
    s.push('');
    s.push('Step 3: Compute posteriors — $p(y=k|x^*) = \\dfrac{p(x^*,y=k)}{p(x^*)}$ where $p(x^*) = \\sum_j p(x^*,y=j)$.');
    s.push(`$p(x^*) = ${g.weightedNumerators.map(n=>n.toExponential(2)).join(' + ')} = ${g.denominator.toExponential(2)}$`);
    for (let i=0;i<gda.classes.length;i++) {
      const c = gda.classes[i];
      s.push(`$p(y=${c}|x^*) = \\frac{p(x^*,y=${c})}{p(x^*)} = \\frac{${g.weightedNumerators[i].toExponential(2)}}{${g.denominator.toExponential(2)}} = ${(g.classPosteriors[i]).toFixed(3)}$`);
    }

    document.getElementById('calcPoint').innerHTML = s.map(l=>{
      if (l.startsWith('Step') || l.startsWith('KDE:') || l.startsWith('GDA:') || l.startsWith('GDA Class ') || l.startsWith('KDE Class ')) {
        let label = l;
        if (l.startsWith('GDA Class ')) label = l.replace(/^GDA Class /, 'Class ');
        if (l.startsWith('KDE Class ')) label = l.replace(/^KDE Class /, 'Class ');
        return `<div style="font-weight: bold; margin-top: 8px;">${label.replace(/\*\*/g, '')}</div>`;
      }
      return `<div>${l}</div>`;
    }).join('');
    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetPromise([document.getElementById('calcPoint')]).catch(err => console.warn('MathJax typeset failed:', err));
    } else if (window.MathJax && MathJax.typeset) {
      try { MathJax.typeset([document.getElementById('calcPoint')]); } catch (e) { console.warn('MathJax typeset failed:', e); }
    }
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

  fitGDAButton && fitGDAButton.addEventListener('click', fitGDA_new);
  example1Btn && example1Btn.addEventListener('click', ()=>{
    if (!gda.fitted) fitGDA_new();
    const pt = [75,89.5]; const r = computeGDAForPoint(pt);
    showDetailedCalculationForPoint(pt);
  });
  example2Btn && example2Btn.addEventListener('click', ()=>{
    if (!gda.fitted) fitGDA_new();
    const pt = [83,83]; const r = computeGDAForPoint(pt);
    showDetailedCalculationForPoint(pt);
  });

  showKDEChk && showKDEChk.addEventListener('change', render);
  showGDAChk && showGDAChk.addEventListener('change', render);

  loadAndRender(csvInput.value);
});