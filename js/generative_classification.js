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
  const example2x1Input = document.getElementById('example2x1');
  const example2x2Input = document.getElementById('example2x2');
  const showKDEChk = document.getElementById('showKDE');
  const showGDAChk = document.getElementById('showGDA');
  const useLogSpaceChk = document.getElementById('useLogSpace');

  const xScale = d3.scaleLinear().range([0, innerW]);
  const yScale = d3.scaleLinear().range([innerH, 0]);

  let data = [];
  let kdeState = null;

  let useLogSpace = !!useLogSpaceChk && useLogSpaceChk.checked;
  useLogSpaceChk && useLogSpaceChk.addEventListener('change', () => {
    useLogSpace = useLogSpaceChk.checked;
    render();
  });

  function safeLog(x) {
    return x > 0 ? Math.log(x) : -Infinity;
  }

  function logAddExp(a, b) {
    if (a === -Infinity) return b;
    if (b === -Infinity) return a;
    const m = Math.max(a, b);
    return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
  }

  function logSumExp(logValues) {
    let acc = -Infinity;
    for (const v of logValues) acc = logAddExp(acc, v);
    return acc;
  }

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
      const cq = document.getElementById('calcQueryPoint');
      const calcP = document.getElementById('calcPriors'); if (calcP) calcP.textContent = 'Priors: —';
      const cm0 = document.getElementById('calcMuSigma0'); if (cm0) cm0.textContent = 'Class k: μ_k = — ; Σ_k = —';
      const cpKDE = document.getElementById('calcPointKDE');
      const cpGDA = document.getElementById('calcPointGDA');
      const cpLegacy = document.getElementById('calcPoint');
      if (cq) cq.textContent = 'No point computed yet.';
      if (cpKDE) cpKDE.textContent = 'No point computed yet.';
      if (cpGDA) cpGDA.textContent = 'No point computed yet.';
      if (cpLegacy && !cpKDE && !cpGDA) cpLegacy.textContent = 'No point computed yet.';
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

  function kdeLogEstimate(points, bandwidth) {
    const n = points.length;
    if (!n) return () => -Infinity;
    const invTwoSigma2 = 1 / (2 * bandwidth * bandwidth);
    const logNorm = -Math.log(2 * Math.PI) - 2 * Math.log(bandwidth);
    const logInvN = -Math.log(n);
    return function(x) {
      let logSum = -Infinity;
      for (const p of points) {
        const dx = x[0] - p[0];
        const dy = x[1] - p[1];
        const r2 = dx*dx + dy*dy;
        const logK = -r2 * invTwoSigma2;
        logSum = logAddExp(logSum, logK);
      }
      return logNorm + logInvN + logSum;
    };
  }

  function computeGrid(xmin,xmax,ymin,ymax,nx=120,ny=100) {
    const xs = d3.range(nx).map(i => xmin + (xmax - xmin) * i/(nx-1));
    const ys = d3.range(ny).map(j => ymin + (ymax - ymin) * j/(ny-1));
    const grid = [];
    for (let j = 0; j < ys.length; j++) for (let i = 0; i < xs.length; i++) grid.push([xs[i], ys[j]]);
    return { xs, ys, grid, nx, ny };
  }

  function computeKDEForPoint(x, classes, kdes, kdesLog, priors, useLog) {
    // Mirrors computeGDAForPoint(): likelihoods -> joint numerators -> normalized posteriors
    const doLog = !!useLog;
    if (doLog) {
      const classLogLikelihoods = classes.map((_, i) => (kdesLog && kdesLog[i] ? kdesLog[i](x) : safeLog(kdes[i](x))));
      const logPriors = priors.map(p => safeLog(p));
      const logNumerators = classLogLikelihoods.map((ll, i) => ll + logPriors[i]);
      const logDenom = logSumExp(logNumerators);
      const posts = (logDenom === -Infinity)
        ? logNumerators.map(() => 1 / Math.max(1, logNumerators.length))
        : logNumerators.map(ln => Math.exp(ln - logDenom));

      const likelihoods = classLogLikelihoods.map(ll => Math.exp(ll));
      const numerators = logNumerators.map(ln => Math.exp(ln));
      const denom = Math.exp(logDenom);
      const predicted = classes.length ? classes[posts.indexOf(Math.max(...posts))] : null;
      return {
        classLikelihoods: likelihoods,
        classLogLikelihoods,
        weightedNumerators: numerators,
        logWeightedNumerators: logNumerators,
        denominator: denom,
        logDenominator: logDenom,
        classPosteriors: posts,
        predicted
      };
    }

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

  function updateQueryInfoTable(point, opts = {}) {
    if (!queryInfo) return;
    if (!kdeState || !kdeState.classes || kdeState.classes.length === 0) {
      queryInfo.textContent = 'No query yet';
      return;
    }

    const x = point[0];
    const y = point[1];
    const header = opts.header ?? `Query (${x.toFixed(2)},${y.toFixed(2)})`;
    const autoFitGDA = !!opts.autoFitGDA;

    const classes = kdeState.classes;
    const kde = computeKDEForPoint([x, y], classes, kdeState.kdes, kdeState.kdesLog, kdeState.priors, useLogSpace);

    const headerCols = classes.map(c => `p(y=${c}|x)`).concat(classes.map(c => `p(x|y=${c})`));
    const colCount = headerCols.length + 1; // +1 for row header

    const kdeVals = [];
    for (let i = 0; i < classes.length; i++) kdeVals.push(kde.classPosteriors[i].toFixed(3));
    for (let i = 0; i < classes.length; i++) kdeVals.push((kde.classLikelihoods[i]).toExponential(2));

    let msg = `<table>`;
    msg += `<tr><th colspan="${colCount}">${header}</th></tr>`;
    msg += `<tr><th></th>${headerCols.map(h => `<th>${h}</th>`).join('')}</tr>`;
    msg += `<tr><td style="font-weight:bold">KDE</td>${kdeVals.map(v => `<td>${v}</td>`).join('')}</tr>`;

    if (autoFitGDA && !gda.fitted) fitGDA_new();
    if (gda.fitted) {
      const gRes = computeGDAForPoint([x, y]);
      const gdaVals = [];
      for (let i = 0; i < gda.classes.length; i++) gdaVals.push(gRes.classPosteriors[i].toFixed(3));
      for (let i = 0; i < gda.classes.length; i++) gdaVals.push((gRes.classLikelihoods[i]).toExponential(2));
      msg += `<tr><td style="font-weight:bold">GDA</td>${gdaVals.map(v => `<td>${v}</td>`).join('')}</tr>`;
    }

    msg += `</table>`;
    queryInfo.innerHTML = msg;
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
    const kdesLog = classPoints.map(arr => arr.length ? kdeLogEstimate(arr, bw) : (() => -Infinity));
    kdeState = { classes, priors, kdes, kdesLog };

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
        const kde = computeKDEForPoint(pt, classes, kdes, kdesLog, priors, useLogSpace);
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
        updateQueryInfoTable(d.x, { header: `Point (${d.x[0].toFixed(2)},${d.x[1].toFixed(2)}) class=${d.y}` });
      });

    svg.on('click', function(event) {
      const [mx,my] = d3.pointer(event, g.node());
      const x = xScale.invert(mx); const y = yScale.invert(my);

      updateQueryInfoTable([x, y], { header: `Query (${x.toFixed(2)},${y.toFixed(2)})`, autoFitGDA: true });
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

  function mvnLogPdf(x, mu, sigma) {
    // 2D multivariate normal log pdf
    const eps = 1e-6;
    const s = [[sigma[0][0] + eps, sigma[0][1]], [sigma[1][0], sigma[1][1] + eps]];
    const inv = inv2(s);
    const d = det2(s);
    if (!inv || d <= 0) return -Infinity;
    const dx = [x[0] - mu[0], x[1] - mu[1]];
    const q = dx[0] * (inv[0][0] * dx[0] + inv[0][1] * dx[1]) + dx[1] * (inv[1][0] * dx[0] + inv[1][1] * dx[1]);
    return -Math.log(2 * Math.PI) - 0.5 * Math.log(d) - 0.5 * q;
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
    const doLog = !!useLogSpace;
    if (!gda.fitted) {
      return {
        classLikelihoods: [],
        weightedNumerators: [],
        denominator: 0,
        classPosteriors: [],
        predicted: null
      };
    }

    if (doLog) {
      const classLogLikelihoods = gda.classes.map(c => mvnLogPdf(x, gda.params[c].mu, gda.params[c].sigma));
      const logNumerators = classLogLikelihoods.map((ll, i) => ll + safeLog(gda.params[gda.classes[i]].prior));
      const logDenom = logSumExp(logNumerators);
      const posts = (logDenom === -Infinity)
        ? logNumerators.map(() => 1 / Math.max(1, logNumerators.length))
        : logNumerators.map(ln => Math.exp(ln - logDenom));

      const likelihoods = classLogLikelihoods.map(ll => Math.exp(ll));
      const numerators = logNumerators.map(ln => Math.exp(ln));
      const denom = Math.exp(logDenom);
      const predicted = gda.classes.length ? gda.classes[posts.indexOf(Math.max(...posts))] : null;
      return {
        classLikelihoods: likelihoods,
        classLogLikelihoods,
        weightedNumerators: numerators,
        logWeightedNumerators: logNumerators,
        denominator: denom,
        logDenominator: logDenom,
        classPosteriors: posts,
        predicted
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
    const kdesLocalLog = classPoints.map(arr => arr.length ? kdeLogEstimate(arr, bw) : (() => -Infinity));
    const kde = computeKDEForPoint(x, classes, kdesLocal, kdesLocalLog, priors, useLogSpace);
    const ks = kde.classLikelihoods;
    const nums = kde.weightedNumerators;
    const den = kde.denominator;
    const kdePosts = kde.classPosteriors;
    const g = computeGDAForPoint(x);

    const queryLines = [];
    queryLines.push(`Query point $x^* = (${x[0].toFixed(3)}, ${x[1].toFixed(3)})$`);
    queryLines.push(`Bandwidth $h = ${bw}$`);
    if (useLogSpace) queryLines.push(`Using log-space normalization (stable)`);

    const fmtLog = (v) => (v === -Infinity ? '-\\infty' : Number.isFinite(v) ? v.toFixed(3) : String(v));

    const sKDE = [];
    sKDE.push('KDE:');
    if (useLogSpace) {
      sKDE.push(`Step 1: Estimate class-conditional log densities $\\log p(x^*|y=k)$.`);
      sKDE.push(`Formula: For class $k$ with $n_k$ points:`);
      sKDE.push(`$\\log p(x^*|y=k) = -\\log(2\\pi) - 2\\log h - \\log n_k + \\log\\sum_{i=1}^{n_k} \\exp(\\log K_i)$. Reminder, $K_i = \\exp\\left(-\\frac{\\|x^*-x_i\\|^2}{2h^2}\\right)$. Using log-sum-exp (LSE) to compute $\\log\\sum_i K_i$ stably.`);

      for (let i=0;i<classes.length;i++) {
        const pts = classPoints[i];
        const n = pts.length;
        const invTwoH2 = 1 / (2 * bw * bw);
        const logNorm = -Math.log(2 * Math.PI) - 2 * Math.log(bw);
        const logInvN = -Math.log(Math.max(1, n));

        sKDE.push(`Class ${classes[i]}`);
        sKDE.push(`Samples: $n_${classes[i]} = ${n}$`);
        sKDE.push(`Constants: $-\\log(2\\pi) - 2\\log h = ${fmtLog(logNorm)}$, $-\\log n_${classes[i]} = ${fmtLog(logInvN)}$`);

        const maxTerms = 5;
        let logSum = -Infinity;
        for (let j=0;j<n;j++) {
          const dx0 = x[0] - pts[j][0];
          const dx1 = x[1] - pts[j][1];
          const r2 = dx0*dx0 + dx1*dx1;
          const logK = -r2 * invTwoH2;
          logSum = logAddExp(logSum, logK);
          if (j < maxTerms) {
            sKDE.push(`Point (${pts[j][0]}, ${pts[j][1]}): $\\|x^*-x_i\\|^2=${r2.toFixed(3)}$, $\\log K_i = -\\frac{\\|x^*-x_i\\|^2}{2h^2} = ${fmtLog(logK)}$`);
          }
        }
        if (n > maxTerms) sKDE.push(`(… ${n - maxTerms} more points omitted)`);

        const ll = kde.classLogLikelihoods ? kde.classLogLikelihoods[i] : (logNorm + logInvN + logSum);
        sKDE.push(`Sum: $\\mathrm{LSE}(\\log K_i) = ${fmtLog(logSum)}$`);
        sKDE.push(`Plug in: $\\log p(x^*|y=${classes[i]}) = ${fmtLog(logInvN)} + ${fmtLog(logSum)} = ${fmtLog(ll)}$`);
      }

      sKDE.push('');
      sKDE.push('Step 2: Compute log-joint densities $\\log p(x^*,y=k)$.');
      sKDE.push(`Formula: $\\log p(x^*,y=k) = \\log p(y=k) + \\log p(x^*|y=k)$`);
      const logJoints = [];
      for (let i=0;i<classes.length;i++) {
        const lp = safeLog(priors[i]);
        const ll = kde.classLogLikelihoods ? kde.classLogLikelihoods[i] : safeLog(ks[i]);
        const a = (kde.logWeightedNumerators && kde.logWeightedNumerators[i] !== undefined) ? kde.logWeightedNumerators[i] : (lp + ll);
        logJoints.push(a);
        sKDE.push(`$\\log p(x^*,y=${classes[i]}) = \\log(${priors[i].toFixed(3)}) + ${fmtLog(ll)} = ${fmtLog(a)}$`);
      }

      sKDE.push('');
      sKDE.push('Step 3: Compute posteriors $p(y=k|x^*)$.');
      sKDE.push(`Formula: $p(y=k|x^*) = \\frac{\\exp(\\log p(x^*,y=k))}{\\exp(\\log p(x^*))} = \\exp(\\log p(x^*,y=k) - \\log p(x^*))$.`);
      const logPx = (kde.logDenominator !== undefined) ? kde.logDenominator : logSumExp(logJoints);
      const lseArgs = logJoints.map(v => fmtLog(v)).join(', ');
      sKDE.push(`Log-marginal probability formula: $\\log p(x^*) = \\mathrm{LSE}(\\log p(x^*,y=0), \\dots, \\log p(x^*,y=k))$.`);
      if (logJoints.length > 6) {
        sKDE.push(`$\\log p(x^*) = \\mathrm{LSE}(${lseArgs.split(', ').slice(0, 3).join(', ')}, \\dots, ${lseArgs.split(', ').slice(-3).join(', ')}) = ${fmtLog(logPx)}$`);
      } else {
        sKDE.push(`$\\log p(x^*) = \\mathrm{LSE}(${lseArgs}) = ${fmtLog(logPx)}$`);
      }
      for (let i=0;i<classes.length;i++) {
        const post = kdePosts[i];
        sKDE.push(`$p(y=${classes[i]}|x^*) = \\exp(${fmtLog(logJoints[i])} - ${fmtLog(logPx)}) = ${post.toFixed(3)}$`);
      }
    } else {
      sKDE.push(`Step 1: Estimate class-conditional densities $p(x^*|y=k)$.`);
      sKDE.push(`Formula: $p(x^*|y=k) = \\dfrac{1}{n_k} \\sum_{i=1}^{n_k} K_h(x^* - x_i^{(k)})$ where $K_h$ is a Gaussian kernel with bandwidth $h$.`);
      sKDE.push(`Gaussian kernel (2D): $K_h(u)=\\dfrac{1}{2\\pi h^2}\\exp\\left(-\\dfrac{\\|u\\|^2}{2h^2}\\right)$.`);
      for (let i=0;i<classes.length;i++) {
        const pts = classPoints[i];
        const n = pts.length;
        const norm = 1 / (2 * Math.PI * bw * bw);
        const invTwoH2 = 1 / (2 * bw * bw);

        sKDE.push(`Class ${classes[i]}`);

        sKDE.push(`Samples: $n_k = ${n}$`);
        sKDE.push(`Constant: $\\dfrac{1}{2\\pi h^2} = ${norm.toExponential(2)}$`);

        // Show a few representative terms so the UI doesn't explode on big datasets.
        const maxTerms = 5;
        let sumK = 0;
        for (let j=0;j<n;j++) {
          const dx0 = x[0] - pts[j][0];
          const dx1 = x[1] - pts[j][1];
          const r2 = dx0*dx0 + dx1*dx1;
          const kTerm = Math.exp(-r2 * invTwoH2);
          sumK += kTerm;

          if (j < maxTerms) {
            sKDE.push(`Point (${pts[j][0]}, ${pts[j][1]}): $x_i=(${pts[j][0]},${pts[j][1]})$, $\\|x^*-x_i\\|^2=${r2.toFixed(3)}$, $\\exp\\left(-\\frac{\\|x^*-x_i\\|^2}{2h^2}\\right)=${kTerm.toExponential(2)}$`);
          }
        }

        if (n > maxTerms) sKDE.push(`(… ${n - maxTerms} more points omitted)`);

        const density = norm * (sumK / n);
        sKDE.push(`Sum: $\\sum_i \\exp\\left(-\\frac{\\|x^*-x_i\\|^2}{2h^2}\\right) = ${sumK.toExponential(2)}$`);
        sKDE.push(`Plug in: $p(x^*|y=${classes[i]}) = ${norm.toExponential(2)} \\times \\frac{${sumK.toExponential(2)}}{${n}} = ${density.toExponential(2)}$`);
      }
      sKDE.push('');
      sKDE.push('Step 2: Compute joint probabilities $p(x^*,y=k)$.');
      sKDE.push(`Formula: $p(x^*,y=k) = p(y=k) \\times p(x^*|y=k)$`);
      for (let i=0;i<classes.length;i++) {
        sKDE.push(`$p(x^*,y=${classes[i]}) = ${priors[i].toFixed(3)} \\times ${ks[i].toExponential(2)} = ${nums[i].toExponential(2)}$`);
      }
      sKDE.push('');
      sKDE.push('Step 3: Compute posteriors $p(y=k|x^*)$.');
      sKDE.push(`Formula: $p(y=k|x^*) = \\frac{p(x^*,y=k)}{p(x^*)}$ where $p(x^*) = \\sum_k p(x^*,y=k)$`);
      sKDE.push(`$p(x^*) = ${nums.map(n=>n.toExponential(2)).join(' + ')} = ${den.toExponential(2)}$`);
      for (let i=0;i<classes.length;i++) {
        sKDE.push(`$p(y=${classes[i]}|x^*) = \\frac{p(x^*,y=${classes[i]})}{p(x^*)} = \\frac{${nums[i].toExponential(2)}}{${den.toExponential(2)}} = ${kdePosts[i].toFixed(3)}$`);
      }
    }

    const sGDA = [];
    sGDA.push('GDA:');
    if (useLogSpace) {
      sGDA.push('Step 1: Estimate class-conditional log densities $\\log p(x^*|y=k)$.');
      sGDA.push(`Formula: $\\log p(x^*|y=k) = -\\log(2\\pi) - \\tfrac12\\log|\\Sigma_k| - \\tfrac12 (x^* - \\mu_k)^T\\Sigma_k^{-1}(x^* - \\mu_k)$.`);
      for (let i=0;i<gda.classes.length;i++) {
        const c = gda.classes[i];
        const p = gda.params[c];
        const mu = p.mu;
        const sigma = p.sigma;

        sGDA.push(`Class ${c}`);
        const eps = 1e-6;
        const sSigma = [[sigma[0][0] + eps, sigma[0][1]], [sigma[1][0], sigma[1][1] + eps]];
        const det = det2(sSigma);
        const inv = inv2(sSigma);
        const dx = [x[0] - mu[0], x[1] - mu[1]];
        const q = (!inv || det <= 0)
          ? NaN
          : (dx[0] * (inv[0][0] * dx[0] + inv[0][1] * dx[1]) + dx[1] * (inv[1][0] * dx[0] + inv[1][1] * dx[1]));
        const ll = (g.classLogLikelihoods && g.classLogLikelihoods[i] !== undefined)
          ? g.classLogLikelihoods[i]
          : (!inv || det <= 0 ? -Infinity : (-Math.log(2 * Math.PI) - 0.5 * Math.log(det) - 0.5 * q));

        sGDA.push(`$\\mu_{${c}} = [${mu.map(v=>v.toFixed(3)).join(', ')}]$, $\\Sigma_{${c}} = \\begin{bmatrix} ${sigma[0][0].toFixed(3)} & ${sigma[0][1].toFixed(3)} \\\\ ${sigma[1][0].toFixed(3)} & ${sigma[1][1].toFixed(3)} \\end{bmatrix}$`);
        sGDA.push(`Constants: $-\\log(2\\pi) = ${(-Math.log(2 * Math.PI)).toFixed(3)}$, $\\frac{1}{2}\\log|\\Sigma_{${c}}| = ${fmtLog(0.5 * Math.log(det))}$`);
        sGDA.push(`$x^* - \\mu_{${c}} = [${dx.map(v=>v.toFixed(3)).join(', ')}]$`);
        if (!inv || det <= 0) {
          sGDA.push(`$|\\Sigma_{${c}}| \\le 0$ or not invertible (numerically); $\\log p(x^*|y=${c}) =-\\infty$.`);
        } else {
          sGDA.push(`$|\\Sigma_{${c}}| = ${det.toExponential(2)}$, $\\log|\\Sigma_{${c}}| = ${fmtLog(Math.log(det))}$`);
          sGDA.push(`$\\Sigma_{${c}}^{-1} = \\begin{bmatrix} ${inv[0][0].toFixed(3)} & ${inv[0][1].toFixed(3)} \\\\ ${inv[1][0].toFixed(3)} & ${inv[1][1].toFixed(3)} \\end{bmatrix}$`);
          sGDA.push(`$(x^* - \\mu_{${c}})^T\\Sigma_{${c}}^{-1}(x^* - \\mu_{${c}}) = \\begin{bmatrix} ${dx[0].toFixed(3)} & ${dx[1].toFixed(3)} \\end{bmatrix} \\begin{bmatrix} ${inv[0][0].toFixed(3)} & ${inv[0][1].toFixed(3)} \\\\ ${inv[1][0].toFixed(3)} & ${inv[1][1].toFixed(3)} \\end{bmatrix} \\begin{bmatrix} ${dx[0].toFixed(3)} \\\\ ${dx[1].toFixed(3)} \\end{bmatrix} = ${q.toFixed(3)}$`);
          sGDA.push(`Plug in: $\\log p(x^*|y=${c}) = ${(-Math.log(2 * Math.PI)).toFixed(3)} - ${fmtLog(0.5 * Math.log(det))} - \\frac{1}{2}(${q.toFixed(3)}) = ${fmtLog(ll)}$`);
        }
      }

      sGDA.push('');
      sGDA.push('Step 2: Compute log-joint densities $\\log p(x^*,y=k)$.');
      sGDA.push(`Formula: $\\log p(x^*,y=k) = \\log p(y=k) + \\log p(x^*|y=k)$`);
      const logJoints = [];

      for (let i=0;i<gda.classes.length;i++) {
        const c = gda.classes[i];
        const lp = safeLog(gda.params[c].prior);
        const ll = g.classLogLikelihoods ? g.classLogLikelihoods[i] : safeLog(g.classLikelihoods[i]);
        const a = (g.logWeightedNumerators && g.logWeightedNumerators[i] !== undefined) ? g.logWeightedNumerators[i] : (lp + ll);
        logJoints.push(a);
        sGDA.push(`$\\log p(x^*,y=${c}) = \\log(${priors[i].toFixed(3)}) + ${fmtLog(ll)} = ${fmtLog(a)}$`);
      }

      sGDA.push('');
      sGDA.push('Step 3: Compute posteriors $p(y=k|x^*)$.');
      sGDA.push(`Formula: $p(y=k|x^*) = \\frac{\\exp(\\log p(x^*,y=k))}{\\sum_j \\exp(\\log p(x^*,y=j))} = \\exp(\\log p(x^*,y=k) - \\log p(x^*))$.`);

      const logJointsGDA = (g.logWeightedNumerators && g.logWeightedNumerators.length)
        ? g.logWeightedNumerators
        : gda.classes.map((c, i) => safeLog(gda.params[c].prior) + (g.classLogLikelihoods ? g.classLogLikelihoods[i] : safeLog(g.classLikelihoods[i])));
      const logPxGDA = (g.logDenominator !== undefined) ? g.logDenominator : logSumExp(logJointsGDA);
      const lseArgsGDA = logJointsGDA.map(v => fmtLog(v)).join(', ');
      sGDA.push(`Log-marginal probability formula: $\\log p(x^*) = \\mathrm{LSE}(\\log p(x^*,y=0),\\dots,\\log p(x^*,y=k))$.`);
      if (logJointsGDA.length > 6) {
        sGDA.push(`$\\log p(x^*) = \\mathrm{LSE}(${lseArgsGDA.split(', ').slice(0, 3).join(', ')}, \\dots, ${lseArgsGDA.split(', ').slice(-3).join(', ')}) = ${fmtLog(logPxGDA)}$`);
      } else {
        sGDA.push(`$\\log p(x^*) = \\mathrm{LSE}(${lseArgsGDA}) = ${fmtLog(logPxGDA)}$`);
      }
      for (let i=0;i<gda.classes.length;i++) {
        const c = gda.classes[i];
        sGDA.push(`$p(y=${c}|x^*) = \\exp(${fmtLog(logJoints[i])} - ${fmtLog(logPxGDA)}) = ${(g.classPosteriors[i]).toFixed(3)}$`);
      }
    } else {
      sGDA.push('Step 1: Estimate class-conditional densities $p(x^*|y=k)$.');
      sGDA.push(`Formula: $p(x^*|y=k) = \\frac{1}{2\\pi\\sqrt{|\\Sigma_k|}} \\exp\\left(-\\frac{1}{2}(x^* - \\mu_k)^T \\Sigma_k^{-1} (x^* - \\mu_k)\\right)$`);
      for (let i=0;i<gda.classes.length;i++) {
        const c = gda.classes[i];
        const p = gda.params[c];
        const mu = p.mu;
        const sigma = p.sigma;

        sGDA.push(`Class ${c}`);

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

        sGDA.push(`$\\mu = [${mu.map(v=>v.toFixed(3)).join(', ')}]$, $\\Sigma = \\begin{bmatrix} ${sigma[0][0].toFixed(3)} & ${sigma[0][1].toFixed(3)} \\\\ ${sigma[1][0].toFixed(3)} & ${sigma[1][1].toFixed(3)} \\end{bmatrix}$`);
        sGDA.push(`$x^* - \\mu = [${dx.map(v=>v.toFixed(3)).join(', ')}]$`);
        if (!inv || det <= 0) {
          sGDA.push(`$|\\Sigma| \\le 0$ or not invertible (numerically); using density $0$.`);
        } else {
          sGDA.push(`$|\\Sigma| = ${det.toExponential(2)}$, $\\Sigma^{-1} = \\begin{bmatrix} ${inv[0][0].toFixed(3)} & ${inv[0][1].toFixed(3)} \\\\ ${inv[1][0].toFixed(3)} & ${inv[1][1].toFixed(3)} \\end{bmatrix}$`);
          sGDA.push(`$(x^* - \\mu)^T\\Sigma^{-1}(x^* - \\mu) = ${q.toFixed(3)}$`);
          sGDA.push(`Plug in: $p(x^*|y=${c}) = \\frac{1}{2\\pi\\sqrt{${det.toExponential(2)}}}\\exp\\left(-\\frac{1}{2}\\cdot ${q.toFixed(3)}\\right) = ${plugged.toExponential(2)}$`);
        }
      }
      sGDA.push('');
      sGDA.push('Step 2: Compute joint probabilities $p(x^*,y=k)$.');
      sGDA.push(`Formula: $p(x^*,y=k) = p(y=k) p(x^*|y=k)$`);
      for (let i=0;i<gda.classes.length;i++) {
        const c = gda.classes[i];
        sGDA.push(`$p(x^*,y=${c}) = ${gda.params[c].prior.toFixed(3)} \\times ${g.classLikelihoods[i].toExponential(2)} = ${g.weightedNumerators[i].toExponential(2)}$`);
      }
      sGDA.push('');
      sGDA.push('Step 3: Compute posteriors $p(y=k|x^*)$.');
      sGDA.push(`Formula: $p(y=k|x^*) = \\frac{p(x^*,y=k)}{p(x^*)}$ where $p(x^*) = \\sum_k p(x^*,y=k)$`);
      sGDA.push(`$p(x^*) = ${g.weightedNumerators.map(n=>n.toExponential(2)).join(' + ')} = ${g.denominator.toExponential(2)}$`);
      for (let i=0;i<gda.classes.length;i++) {
        const c = gda.classes[i];
        sGDA.push(`$p(y=${c}|x^*) = \\frac{p(x^*,y=${c})}{p(x^*)} = \\frac{${g.weightedNumerators[i].toExponential(2)}}{${g.denominator.toExponential(2)}} = ${(g.classPosteriors[i]).toFixed(3)}$`);
      }
    }

    const renderLines = (lines) => lines.map(l=>{
      if (l.startsWith('KDE:') || l.startsWith('GDA:')) {
        return `<h3 class="calc-step-header">${l}</h3>`;
      }
      if (l.startsWith('Step')) {
        return `<h4 class="calc-step-subheader">${l}</h4>`;
      }
      if (l.startsWith('Class ')) {
        return `<h5 class="calc-step-subheader">${l}</h5>`;
      }
      if (l.startsWith('Plug')) {
        return `<div class="calc-step-end">${l}</div>`;
      }
      return `<div class="calc-step-content">${l}</div>`;
    }).join('');

    const calcPointKDEEl = document.getElementById('calcPointKDE');
    const calcPointGDAEl = document.getElementById('calcPointGDA');
    const calcPointLegacyEl = document.getElementById('calcPoint');
    const calcQueryPointEl = document.getElementById('calcQueryPoint');

    if (calcQueryPointEl) {
      calcQueryPointEl.innerHTML = renderLines(queryLines);
    }

    if (calcPointKDEEl) calcPointKDEEl.innerHTML = renderLines(sKDE);
    if (calcPointGDAEl) calcPointGDAEl.innerHTML = renderLines(sGDA);
    if (!calcPointKDEEl && !calcPointGDAEl && calcPointLegacyEl) {
      // Back-compat for older HTML: include query point at the top if there's no dedicated container.
      calcPointLegacyEl.innerHTML = renderLines(queryLines.concat([''], sKDE, [''], sGDA));
    }

    const typesetTargets = [calcQueryPointEl, calcPointKDEEl, calcPointGDAEl, calcPointLegacyEl].filter(Boolean);
    if (typesetTargets.length > 0 && window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetPromise(typesetTargets).catch(err => console.warn('MathJax typeset failed:', err));
    } else if (typesetTargets.length > 0 && window.MathJax && MathJax.typeset) {
      try { MathJax.typeset(typesetTargets); } catch (e) { console.warn('MathJax typeset failed:', e); }
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
    const pt = [75,89.5];
    updateQueryInfoTable(pt, { header: `Query (${pt[0].toFixed(2)},${pt[1].toFixed(2)})`, autoFitGDA: true });
    computeGDAForPoint(pt);
    showDetailedCalculationForPoint(pt);
  });
  example2Btn && example2Btn.addEventListener('click', ()=>{
    if (!gda.fitted) fitGDA_new();
    const x1 = example2x1Input ? parseFloat(example2x1Input.value) : NaN;
    const x2 = example2x2Input ? parseFloat(example2x2Input.value) : NaN;
    if (!Number.isFinite(x1) || !Number.isFinite(x2)) {
      alert('Please enter valid numeric values for x1 and x2.');
      return;
    }
    const pt = [x1, x2];
    updateQueryInfoTable(pt, { header: `Query (${pt[0].toFixed(2)},${pt[1].toFixed(2)})`, autoFitGDA: true });
    computeGDAForPoint(pt);
    showDetailedCalculationForPoint(pt);
  });

  showKDEChk && showKDEChk.addEventListener('change', render);
  showGDAChk && showGDAChk.addEventListener('change', render);

  loadAndRender(csvInput.value);
});