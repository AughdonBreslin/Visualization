document.addEventListener('DOMContentLoaded', () => {
  // Guard: only run on bayesian page
  const paramContainer = document.getElementById('bayesParamViz');
  const predContainer = document.getElementById('bayesPredViz');
  if (!paramContainer || !predContainer) return;

  const modelSelect = document.getElementById('bayesModel');
  const dataText = document.getElementById('bayesData');
  const summaryEl = document.getElementById('bayesSummary');
  const paramContextEl = document.getElementById('bayesParamContext');
  const predTitleEl = document.getElementById('bayesPredTitle');
  const predContextEl = document.getElementById('bayesPredContext');

  const betaControls = document.getElementById('bayesControlsBeta');
  const normalControls = document.getElementById('bayesControlsNormal');

  const alphaInput = document.getElementById('alpha');
  const betaInput = document.getElementById('beta');
  const probHeadsInput = document.getElementById('bayesProbHeads');
  const flipCoinBtn = document.getElementById('bayesFlipCoin');

  const mu0Input = document.getElementById('mu0');
  const tauInput = document.getElementById('tau');
  const sigmaInput = document.getElementById('sigma');
  const trueMeanInput = document.getElementById('bayesTrueMean');
  const generateSampleBtn = document.getElementById('bayesGenerateSample');

  // MAP demo elements (optional; only present on bayesian.html)
  const mapObjContainer = document.getElementById('mapObjViz');
  const mapSummaryEl = document.getElementById('mapSummary');
  const mapDataText = document.getElementById('mapData');
  const mapMu0Input = document.getElementById('mapMu0');
  const mapTauInput = document.getElementById('mapTau');
  const mapSigmaInput = document.getElementById('mapSigma');
  const mapTrueMeanInput = document.getElementById('mapTrueMean');
  const mapGenerateSampleBtn = document.getElementById('mapGenerateSample');

  // VI demo elements (optional)
  const viPostContainer = document.getElementById('viPostViz');
  const viElboContainer = document.getElementById('viElboViz');
  const viSummaryEl = document.getElementById('viSummary');
  const viDataText = document.getElementById('viData');
  const viProbHeadsInput = document.getElementById('viProbHeads');
  const viFlipCoinBtn = document.getElementById('viFlipCoin');
  const viAlphaInput = document.getElementById('viAlpha');
  const viBetaInput = document.getElementById('viBeta');
  const viMInput = document.getElementById('viM');
  const viLogSInput = document.getElementById('viLogS');
  const viStepsInput = document.getElementById('viSteps');
  const viLRInput = document.getElementById('viLR');
  const viMCInput = document.getElementById('viMC');

  function clampPositive(x, minValue = 1e-6) {
    if (!Number.isFinite(x)) return minValue;
    return Math.max(minValue, x);
  }

  // Lanczos approximation for log-gamma
  function logGamma(z) {
    const p = [
      676.5203681218851,
      -1259.1392167224028,
      771.32342877765313,
      -176.61502916214059,
      12.507343278686905,
      -0.13857109526572012,
      9.9843695780195716e-6,
      1.5056327351493116e-7,
    ];

    if (z < 0.5) {
      // Reflection formula
      return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
    }

    z -= 1;
    let x = 0.99999999999980993;
    for (let i = 0; i < p.length; i++) x += p[i] / (z + i + 1);
    const t = z + p.length - 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }

  function betaFn(a, b) {
    return Math.exp(logGamma(a) + logGamma(b) - logGamma(a + b));
  }

  function betaPdf(x, a, b) {
    if (x <= 0 || x >= 1) return 0;
    const logNum = (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x);
    const logDen = Math.log(betaFn(a, b));
    return Math.exp(logNum - logDen);
  }

  function normalPdf(x, mean, sd) {
    const s = clampPositive(sd);
    const z = (x - mean) / s;
    return (1 / (s * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * z * z);
  }

  function sigmoid(z) {
    // Stable-ish sigmoid
    if (z >= 0) {
      const ez = Math.exp(-z);
      return 1 / (1 + ez);
    }
    const ez = Math.exp(z);
    return ez / (1 + ez);
  }

  function clamp01(x) {
    const eps = 1e-9;
    return Math.min(1 - eps, Math.max(eps, x));
  }

  function logNormalPdf(x, mean, sd) {
    const s = clampPositive(sd);
    const z = (x - mean) / s;
    return -0.5 * z * z - Math.log(s) - 0.5 * Math.log(2 * Math.PI);
  }

  function plotLineSeries({ container, xs, ys, xLabel, yLabel, name, color }) {
    const points = xs.map((x, i) => ({ x, y: ys[i] }));
    const yMin = d3.min(ys);
    const yMax = d3.max(ys);
    const ySpan = Math.max(1e-9, yMax - yMin);
    const yPad = ySpan * 0.08;
    plotContinuous({
      container,
      xDomain: [d3.min(xs), d3.max(xs)],
      yDomain: [yMin - yPad, yMax + yPad],
      xLabel,
      yLabel,
      series: [{ name: name || 'ELBO', color: color || 'rgba(255, 255, 255, 0.85)', points }],
    });
  }

  function parseTokens(text) {
    return String(text)
      .replace(/\r/g, '')
      .split(/[\s,;]+/)
      .map((t) => t.trim())
      .filter(Boolean);
  }

  function parseCoinFlips(text) {
    // Accept: 1/0, H/T, heads/tails (case-insensitive)
    const tokens = parseTokens(text);
    const flips = [];
    for (const tRaw of tokens) {
      const t = tRaw.toLowerCase();
      if (t === '1' || t === 'h' || t === 'head' || t === 'heads') flips.push(1);
      else if (t === '0' || t === 't' || t === 'tail' || t === 'tails') flips.push(0);
      else if (/^\d+\/\d+$/.test(t)) {
        // Allow a shorthand like k/n (interpreted as k ones then n-k zeros)
        const [kStr, nStr] = t.split('/');
        const k = Number(kStr);
        const n = Number(nStr);
        if (Number.isFinite(k) && Number.isFinite(n) && n >= 0 && k >= 0 && k <= n) {
          for (let i = 0; i < k; i++) flips.push(1);
          for (let i = 0; i < n - k; i++) flips.push(0);
        }
      }
    }
    return flips;
  }

  function parseNumbers(text) {
    const tokens = parseTokens(text);
    const xs = [];
    for (const t of tokens) {
      const v = Number(t);
      if (Number.isFinite(v)) xs.push(v);
    }
    return xs;
  }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function hashStringToSeed(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  // Marsaglia-Tsang Gamma sampler
  function sampleGamma(shape, rng) {
    const k = clampPositive(shape);
    if (k < 1) {
      // Use Johnk's method via boosting
      const u = clampPositive(rng(), 1e-12);
      return sampleGamma(k + 1, rng) * Math.pow(u, 1 / k);
    }

    const d = k - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
      // Box-Muller for standard normal
      let u1 = clampPositive(rng(), 1e-12);
      let u2 = clampPositive(rng(), 1e-12);
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const v = Math.pow(1 + c * z, 3);
      if (v <= 0) continue;
      const u = rng();
      if (u < 1 - 0.0331 * Math.pow(z, 4)) return d * v;
      if (Math.log(u) < 0.5 * z * z + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  function sampleBeta(a, b, rng) {
    const x = sampleGamma(a, rng);
    const y = sampleGamma(b, rng);
    return x / (x + y);
  }

  function sampleStandardNormal(rng = Math.random) {
    const u1 = clampPositive(rng(), 1e-12);
    const u2 = clampPositive(rng(), 1e-12);
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  function quantileFromSamples(samples, q) {
    const sorted = samples.slice().sort((a, b) => a - b);
    const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1))));
    return sorted[idx];
  }

  function clear(el) {
    d3.select(el).selectAll('*').remove();
  }

  function appendDatasetValue(textarea, value) {
    if (!textarea) return;
    const current = String(textarea.value || '').trim();
    textarea.value = current ? `${current}, ${value}` : String(value);
  }

  function debounce(fn, delay = 220) {
    let timer = null;
    return (...args) => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => fn(...args), delay);
    };
  }

  function syncConjugatePanelText() {
    if (paramContextEl) {
      paramContextEl.textContent = 'The prior shows what parameter values were plausible before seeing the dataset, the likelihood shows which parameter values best explain the observed data, and the posterior shows the updated belief after combining both. The shaded interval on the posterior plot is a 95% credible interval, which contains parameter values that together have 95% of the posterior probability mass.';
    }

    if (!predTitleEl || !predContextEl) return;

    if (modelSelect.value === 'beta-binomial') {
      predTitleEl.textContent = 'Prediction for the next flip';
      predContextEl.textContent = 'This bar chart shows the posterior predictive probability of the next observation being 0 or 1 after averaging over uncertainty in the unknown coin-flip probability p.';
    } else {
      predTitleEl.textContent = 'Predictive distribution for a new value';
      predContextEl.textContent = 'This density curve shows where a new numeric observation is likely to fall after combining uncertainty about the unknown mean with the observation noise in the data model.';
    }
  }

  function getPlotSize(container) {
    const rect = container.getBoundingClientRect();
    const w = Math.max(320, rect.width);
    const h = 320;
    return { w, h };
  }

  function plotContinuous({
    container,
    xDomain,
    yDomain,
    series,
    xLabel,
    yLabel,
    shadeInterval,
    shadeSeriesIndex,
  }) {
    clear(container);

    const { w, h } = getPlotSize(container);
    const margin = { top: 14, right: 14, bottom: 34, left: 46 };
    const innerW = w - margin.left - margin.right;
    const innerH = h - margin.top - margin.bottom;

    const svg = d3
      .select(container)
      .append('svg')
      .attr('width', '100%')
      .attr('height', h);

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain(xDomain).range([0, innerW]);

    const allYs = series.flatMap((s) => s.points.map((p) => p.y));
    const defaultYMax = Math.max(1e-9, d3.max(allYs));
    const computedYDomain = yDomain ?? [0, defaultYMax * 1.08];
    const y = d3.scaleLinear().domain(computedYDomain).range([innerH, 0]);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(7).tickSizeOuter(0));

    g.append('g').call(d3.axisLeft(y).ticks(6).tickSizeOuter(0));

    if (xLabel) {
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 30)
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.75)')
        .attr('font-size', 12)
        .text(xLabel);
    }

    if (yLabel) {
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerH / 2)
        .attr('y', -36)
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.75)')
        .attr('font-size', 12)
        .text(yLabel);
    }

    if (shadeInterval) {
      const [a, b] = shadeInterval;
      const shadeBase = series[Math.max(0, Math.min(series.length - 1, shadeSeriesIndex ?? 0))];
      const shadePoints = shadeBase.points.filter((p) => p.x >= a && p.x <= b);
      const area = d3
        .area()
        .x((d) => x(d.x))
        .y0(y(0))
        .y1((d) => y(d.y));

      g.append('path')
        .datum(shadePoints)
        .attr('fill', 'rgba(125, 255, 178, 0.12)')
        .attr('d', area);
    }

    const line = d3
      .line()
      .x((d) => x(d.x))
      .y((d) => y(d.y))
      .curve(d3.curveMonotoneX);

    for (const s of series) {
      g.append('path')
        .datum(s.points)
        .attr('fill', 'none')
        .attr('stroke', s.color)
        .attr('stroke-width', 2)
        .attr('opacity', s.opacity ?? 1)
        .attr('d', line);
    }

    // Simple legend
    const legend = svg.append('g').attr('transform', `translate(${margin.left + 6},${margin.top + 6})`);
    let lx = 0;
    series.forEach((s) => {
      const entry = legend.append('g').attr('transform', `translate(${lx},0)`);
      entry
        .append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', 10)
        .attr('height', 10)
        .attr('rx', 2)
        .attr('fill', s.color)
        .attr('opacity', s.opacity ?? 1);
      entry
        .append('text')
        .attr('x', 14)
        .attr('y', 9)
        .attr('fill', 'rgba(255,255,255,0.82)')
        .attr('font-size', 12)
        .text(s.name);
      lx += 14 + (s.name.length * 7.2) + 18;
    });
  }

  function plotBars({ container, labels, values, colors, title }) {
    clear(container);

    const { w, h } = getPlotSize(container);
    const margin = { top: 16, right: 14, bottom: 34, left: 46 };
    const innerW = w - margin.left - margin.right;
    const innerH = h - margin.top - margin.bottom;

    const svg = d3.select(container).append('svg').attr('width', '100%').attr('height', h);
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand().domain(labels).range([0, innerW]).padding(0.3);
    const y = d3.scaleLinear().domain([0, Math.max(1e-9, d3.max(values)) * 1.1]).range([innerH, 0]);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).tickSizeOuter(0));
    g.append('g').call(d3.axisLeft(y).ticks(6).tickSizeOuter(0));

    g.selectAll('rect')
      .data(labels.map((lab, i) => ({ lab, v: values[i], c: colors[i] })))
      .enter()
      .append('rect')
      .attr('x', (d) => x(d.lab))
      .attr('y', (d) => y(d.v))
      .attr('width', x.bandwidth())
      .attr('height', (d) => innerH - y(d.v))
      .attr('rx', 6)
      .attr('fill', (d) => d.c)
      .attr('opacity', 0.8);

    g.selectAll('text.bar-label')
      .data(labels.map((lab, i) => ({ lab, v: values[i] })))
      .enter()
      .append('text')
      .attr('class', 'bar-label')
      .attr('x', (d) => x(d.lab) + x.bandwidth() / 2)
      .attr('y', (d) => y(d.v) - 6)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.85)')
      .attr('font-size', 12)
      .text((d) => d.v.toFixed(3));

    if (title) {
      svg
        .append('text')
        .attr('x', margin.left)
        .attr('y', 14)
        .attr('fill', 'rgba(255,255,255,0.85)')
        .attr('font-size', 12)
        .attr('font-weight', 650)
        .text(title);
    }
  }

  function plotObjectives({ container, xDomain, curves, markers }) {
    clear(container);

    const { w, h } = getPlotSize(container);
    const margin = { top: 14, right: 14, bottom: 34, left: 46 };
    const innerW = w - margin.left - margin.right;
    const innerH = h - margin.top - margin.bottom;

    const svg = d3.select(container).append('svg').attr('width', '100%').attr('height', h);
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain(xDomain).range([0, innerW]);

    const allYs = curves.flatMap((c) => c.points.map((p) => p.y));
    const yMax = Math.max(1e-9, d3.max(allYs));
    const y = d3.scaleLinear().domain([0, yMax * 1.08]).range([innerH, 0]);

    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(7).tickSizeOuter(0));
    g.append('g').call(d3.axisLeft(y).ticks(6).tickSizeOuter(0));

    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 30)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.75)')
      .attr('font-size', 12)
      .text('\u03bc');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2)
      .attr('y', -36)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.75)')
      .attr('font-size', 12)
      .text('normalized objective');

    const line = d3
      .line()
      .x((d) => x(d.x))
      .y((d) => y(d.y))
      .curve(d3.curveMonotoneX);

    curves.forEach((c) => {
      g.append('path')
        .datum(c.points)
        .attr('fill', 'none')
        .attr('stroke', c.color)
        .attr('stroke-width', 2)
        .attr('opacity', c.opacity ?? 1)
        .attr('d', line);
    });

    // Markers (vertical lines)
    if (markers && markers.length) {
      const mg = g.append('g').attr('class', 'markers');
      markers.forEach((m) => {
        mg.append('line')
          .attr('x1', x(m.x))
          .attr('x2', x(m.x))
          .attr('y1', 0)
          .attr('y2', innerH)
          .attr('stroke', m.color)
          .attr('stroke-width', 2)
          .attr('opacity', m.opacity ?? 0.9)
          .attr('stroke-dasharray', m.dash ?? '5 4');

        mg.append('text')
          .attr('x', x(m.x) + 4)
          .attr('y', 12)
          .attr('fill', m.color)
          .attr('font-size', 12)
          .attr('opacity', m.opacity ?? 0.95)
          .text(m.label);
      });
    }

    // Legend
    const legendPadding = 10;
    const legend = svg.append('g').attr('transform', `translate(${w - margin.right - legendPadding},${margin.top + 6})`);
    let ly = 0;
    curves.forEach((c) => {
      const entry = legend.append('g').attr('transform', `translate(0,${ly})`);
      entry
        .append('rect')
        .attr('x', -10)
        .attr('y', 0)
        .attr('width', 10)
        .attr('height', 10)
        .attr('rx', 2)
        .attr('fill', c.color)
        .attr('opacity', c.opacity ?? 1);
      entry
        .append('text')
        .attr('x', -14)
        .attr('y', 9)
        .attr('text-anchor', 'end')
        .attr('fill', 'rgba(255,255,255,0.82)')
        .attr('font-size', 12)
        .text(c.name);
      ly += 18;
    });
  }

  function renderMAPDemo() {
    if (!mapObjContainer || !mapSummaryEl || !mapDataText) return;

    const mu0Raw = Number(mapMu0Input?.value);
    const mu0 = Number.isFinite(mu0Raw) ? mu0Raw : 0;
    const tau = clampPositive(Number(mapTauInput?.value), 0.05);
    const sigma = clampPositive(Number(mapSigmaInput?.value), 0.05);

    const xs = parseNumbers(mapDataText.value);
    const n = xs.length;
    if (n === 0) {
      clear(mapObjContainer);
      mapSummaryEl.textContent = 'No numbers parsed yet. Paste values like: 0.2, -0.1, 0.4, 0.0.';
      return;
    }

    const xbar = d3.mean(xs);

    const tau2 = tau * tau;
    const sigma2 = sigma * sigma;

    const muMAP = (mu0 / tau2 + (n * xbar) / sigma2) / (1 / tau2 + n / sigma2);
    const muMLE = xbar;

    const center = muMAP;
    const spread = 4 * Math.max(tau, sigma / Math.sqrt(n), Math.abs(muMLE - mu0) + 1e-6);
    const xMin = center - spread;
    const xMax = center + spread;

    const grid = d3.range(0, 401).map((i) => xMin + (xMax - xMin) * (i / 400));

    // Objectives up to additive constants
    const nll = grid.map((m) => (n / (2 * sigma2)) * (m - muMLE) * (m - muMLE));
    const priorPenalty = grid.map((m) => (1 / (2 * tau2)) * (m - mu0) * (m - mu0));
    const post = grid.map((_, i) => nll[i] + priorPenalty[i]);

    function normalize(vals) {
      const vMin = d3.min(vals);
      const vMax = d3.max(vals);
      const denom = (vMax - vMin) || 1;
      return vals.map((v) => (v - vMin) / denom);
    }

    const nllN = normalize(nll);
    const priorN = normalize(priorPenalty);
    const postN = normalize(post);

    const nllPts = grid.map((m, i) => ({ x: m, y: nllN[i] }));
    const priorPts = grid.map((m, i) => ({ x: m, y: priorN[i] }));
    const postPts = grid.map((m, i) => ({ x: m, y: postN[i] }));

    plotObjectives({
      container: mapObjContainer,
      xDomain: [xMin, xMax],
      curves: [
        { name: '-log likelihood', color: 'rgba(74, 163, 255, 0.9)', points: nllPts, opacity: 0.9 },
        { name: '-log prior', color: 'rgba(255, 255, 255, 0.75)', points: priorPts, opacity: 0.85 },
        { name: '-log posterior', color: 'rgba(125, 255, 178, 0.95)', points: postPts, opacity: 1 },
      ],
      markers: [
        { label: 'MLE', x: muMLE, color: 'rgba(74, 163, 255, 0.9)', dash: '4 3' },
        { label: 'MAP', x: muMAP, color: 'rgba(125, 255, 178, 0.95)', dash: '6 3' },
      ],
    });

    mapSummaryEl.textContent = `Parsed n=${n} values, mean=${muMLE.toFixed(3)}. MLE = ${muMLE.toFixed(3)}. MAP = ${muMAP.toFixed(3)} (prior mean \u03bc0=${mu0.toFixed(3)}, \u03c4=${tau.toFixed(3)}, \u03c3=${sigma.toFixed(3)}).`;
  }

  function renderVIDemoOnce() {
    if (!viPostContainer || !viElboContainer || !viSummaryEl || !viDataText) return;

    const flips = parseCoinFlips(viDataText.value);
    const n = flips.length;
    if (n === 0) {
      clear(viPostContainer);
      clear(viElboContainer);
      viSummaryEl.textContent = 'No coin flips parsed yet. Paste values like: 1, 0, 1, 1, 0 (or H/T).';
      return;
    }

    const k = flips.reduce((s, v) => s + v, 0);
    const alpha0 = clampPositive(Number(viAlphaInput?.value), 0.1);
    const beta0 = clampPositive(Number(viBetaInput?.value), 0.1);

    const aPost = alpha0 + k;
    const bPost = beta0 + (n - k);

    const m = Number(viMInput?.value);
    const logS = Number(viLogSInput?.value);
    const s = Math.exp(Math.max(-6, Math.min(3, Number.isFinite(logS) ? logS : -0.5)));

    // Prior and true posterior curves (Beta)
    const grid = d3.range(0, 1.0001, 1 / 400).map((x) => clamp01(x));
    const priorPts = grid.map((p) => ({ x: p, y: betaPdf(p, alpha0, beta0) }));
    const truePts = grid.map((p) => ({ x: p, y: betaPdf(p, aPost, bPost) }));

    // VI approximation curve via histogram (logistic-normal samples)
    const seed = hashStringToSeed(`${viDataText.value}|${alpha0}|${beta0}|${m}|${s}`);
    const rng = mulberry32(seed);

    // Box-Muller eps
    function randn() {
      const u1 = clampPositive(rng(), 1e-12);
      const u2 = clampPositive(rng(), 1e-12);
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    const nSamples = 8000;
    const ps = Array.from({ length: nSamples }, () => clamp01(sigmoid(m + s * randn())));
    const bins = 60;
    const counts = new Array(bins).fill(0);
    ps.forEach((p) => {
      const idx = Math.min(bins - 1, Math.floor(p * bins));
      counts[idx] += 1;
    });
    const binWidth = 1 / bins;
    const approxPts = counts.map((c, i) => {
      const x = (i + 0.5) * binWidth;
      const density = c / (nSamples * binWidth);
      return { x, y: density };
    });

    plotContinuous({
      container: viPostContainer,
      xDomain: [0, 1],
      xLabel: 'p',
      yLabel: 'density',
      series: [
        { name: 'prior', color: 'rgba(255, 255, 255, 0.72)', points: priorPts, opacity: 0.92 },
        { name: 'true posterior', color: 'rgba(125, 255, 178, 0.95)', points: truePts, opacity: 1 },
        { name: 'VI (logistic-normal)', color: 'rgba(74, 163, 255, 0.9)', points: approxPts, opacity: 0.95 },
      ],
    });

    plotLineSeries({
      container: viElboContainer,
      xs: [0, 1],
      ys: [0, 0],
      xLabel: 'step',
      yLabel: 'ELBO',
      name: 'ELBO',
      color: 'rgba(255, 255, 255, 0.85)',
    });

    viSummaryEl.textContent = `Parsed n=${n} flips, k=${k}. True posterior is Beta(${aPost.toFixed(2)}, ${bPost.toFixed(2)}). Set variational params to m=${(Number.isFinite(m) ? m : 0).toFixed(2)}, log s=${(Number.isFinite(logS) ? logS : -0.5).toFixed(2)}.`;
  }

  function buildVariationalApproximationPoints({ m, s, seed, nSamples = 8000, bins = 60 }) {
    const rng = mulberry32(seed);

    function randn() {
      const u1 = clampPositive(rng(), 1e-12);
      const u2 = clampPositive(rng(), 1e-12);
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    const ps = Array.from({ length: nSamples }, () => clamp01(sigmoid(m + s * randn())));
    const counts = new Array(bins).fill(0);
    ps.forEach((p) => {
      const idx = Math.min(bins - 1, Math.floor(p * bins));
      counts[idx] += 1;
    });
    const binWidth = 1 / bins;
    return counts.map((count, i) => {
      const x = (i + 0.5) * binWidth;
      const density = count / (nSamples * binWidth);
      return { x, y: density };
    });
  }

  function clampMagnitude(x, limit) {
    if (!Number.isFinite(x)) return 0;
    return Math.max(-limit, Math.min(limit, x));
  }

  function runVIDemo() {
    if (!viPostContainer || !viElboContainer || !viSummaryEl || !viDataText) return;

    const flips = parseCoinFlips(viDataText.value);
    const n = flips.length;
    if (n === 0) {
      clear(viPostContainer);
      clear(viElboContainer);
      viSummaryEl.textContent = 'No coin flips parsed yet. Paste values like: 1, 0, 1, 1, 0 (or H/T).';
      return;
    }

    const k = flips.reduce((s, v) => s + v, 0);
    const alpha0 = clampPositive(Number(viAlphaInput?.value), 0.1);
    const beta0 = clampPositive(Number(viBetaInput?.value), 0.1);
    const aPost = alpha0 + k;
    const bPost = beta0 + (n - k);

    let m = Number(viMInput?.value);
    if (!Number.isFinite(m)) m = 0;
    let logS = Number(viLogSInput?.value);
    if (!Number.isFinite(logS)) logS = -0.5;
    const initialM = m;
    const initialLogS = logS;

    const steps = Math.max(1, Math.min(2000, Math.floor(Number(viStepsInput?.value) || 200)));
    const lr = Math.max(1e-5, Math.min(1, Number(viLRInput?.value) || 0.05));
    const mc = Math.max(5, Math.min(500, Math.floor(Number(viMCInput?.value) || 60)));

    const logB = Math.log(betaFn(alpha0, beta0));
    const snapshotTargets = Array.from(new Set([
      0,
      Math.max(0, Math.floor((steps - 1) / 3)),
      Math.max(0, Math.floor((2 * (steps - 1)) / 3)),
      Math.max(0, steps - 1),
    ])).sort((a, b) => a - b);
    const snapshotStates = [];

    // Deterministic RNG for reproducibility
    const seed = hashStringToSeed(`${viDataText.value}|${alpha0}|${beta0}|${m}|${logS}|${steps}|${lr}|${mc}`);
    const rng = mulberry32(seed);

    function randn() {
      const u1 = clampPositive(rng(), 1e-12);
      const u2 = clampPositive(rng(), 1e-12);
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    const elbos = [];
    const epsConst = 1e-9;
    const adamBeta1 = 0.9;
    const adamBeta2 = 0.999;
    const adamEps = 1e-8;
    const gradClip = 25;
    let mFirstMoment = 0;
    let mSecondMoment = 0;
    let logSFirstMoment = 0;
    let logSSecondMoment = 0;

    for (let t = 0; t < steps; t++) {
      if (snapshotTargets.includes(t)) {
        snapshotStates.push({ step: t + 1, m, logS });
      }

      const s = Math.exp(Math.max(-6, Math.min(3, logS)));

      let gradM = 0;
      let gradLogS = 0;
      let elboSum = 0;

      for (let j = 0; j < mc; j++) {
        const eps = randn();
        const z = m + s * eps;
        const p = clamp01(sigmoid(z));

        // ELBO sample = loglik + logprior + log|dp/dz| - log q(z)
        const logp = Math.log(p + epsConst);
        const log1mp = Math.log(1 - p + epsConst);

        const logLik = k * logp + (n - k) * log1mp;
        const logPrior = (alpha0 - 1) * logp + (beta0 - 1) * log1mp - logB;
        const logJac = logp + log1mp;
        const logQ = logNormalPdf(z, m, s);

        const elbo = logLik + logPrior + logJac - logQ;
        elboSum += elbo;

        // d/dz of (loglik + logprior + logjac)
        // Coefficients: A = k + alpha0, B = (n-k) + beta0
        const A = k + alpha0;
        const B = (n - k) + beta0;
        const dTdz = A - (A + B) * p;

        gradM += dTdz;
        gradLogS += dTdz * (s * eps) + 1; // +1 from derivative of -log q term's log(s)
      }

      gradM /= mc;
      gradLogS /= mc;
      const elboAvg = elboSum / mc;
      elbos.push(elboAvg);

      const clippedGradM = clampMagnitude(gradM, gradClip);
      const clippedGradLogS = clampMagnitude(gradLogS, gradClip);

      mFirstMoment = adamBeta1 * mFirstMoment + (1 - adamBeta1) * clippedGradM;
      mSecondMoment = adamBeta2 * mSecondMoment + (1 - adamBeta2) * clippedGradM * clippedGradM;
      logSFirstMoment = adamBeta1 * logSFirstMoment + (1 - adamBeta1) * clippedGradLogS;
      logSSecondMoment = adamBeta2 * logSSecondMoment + (1 - adamBeta2) * clippedGradLogS * clippedGradLogS;

      const stepIndex = t + 1;
      const mFirstMomentHat = mFirstMoment / (1 - Math.pow(adamBeta1, stepIndex));
      const mSecondMomentHat = mSecondMoment / (1 - Math.pow(adamBeta2, stepIndex));
      const logSFirstMomentHat = logSFirstMoment / (1 - Math.pow(adamBeta1, stepIndex));
      const logSSecondMomentHat = logSSecondMoment / (1 - Math.pow(adamBeta2, stepIndex));

      m += lr * mFirstMomentHat / (Math.sqrt(mSecondMomentHat) + adamEps);
      logS += lr * logSFirstMomentHat / (Math.sqrt(logSSecondMomentHat) + adamEps);
      logS = Math.max(-6, Math.min(3, logS));
    }

    if (!snapshotStates.length || snapshotStates[snapshotStates.length - 1].step !== steps) {
      snapshotStates.push({ step: steps, m, logS });
    }

    // Plot prior, posterior, and variational trajectory
    const grid = d3.range(0, 1.0001, 1 / 400).map((x) => clamp01(x));
    const priorPts = grid.map((p) => ({ x: p, y: betaPdf(p, alpha0, beta0) }));
    const truePts = grid.map((p) => ({ x: p, y: betaPdf(p, aPost, bPost) }));
    const trajectoryPalette = [
      'rgba(255, 255, 255, 0.75)',
      'rgba(189, 223, 255, 0.8)',
      'rgba(117, 191, 255, 0.88)',
      'rgba(74, 163, 255, 0.95)',
    ];
    const trajectorySeries = snapshotStates.map((snapshot, index) => {
      const isInitial = index === 0;
      const isFinal = index === snapshotStates.length - 1;
      const label = isInitial
        ? 'q start'
        : isFinal
          ? 'q finish'
          : `q step ${snapshot.step}`;
      return {
        name: label,
        color: trajectoryPalette[Math.min(index, trajectoryPalette.length - 1)],
        points: buildVariationalApproximationPoints({
          m: snapshot.m,
          s: Math.exp(snapshot.logS),
          seed: (seed ^ 0x517cc1b7) + index,
          nSamples: isFinal ? 12000 : 7000,
          bins: 60,
        }),
        opacity: isFinal ? 0.98 : 0.72,
      };
    });

    plotContinuous({
      container: viPostContainer,
      xDomain: [0, 1],
      xLabel: 'p',
      yLabel: 'density',
      series: [
        { name: 'prior', color: 'rgba(255, 255, 255, 0.72)', points: priorPts, opacity: 0.92 },
        { name: 'posterior', color: 'rgba(125, 255, 178, 0.95)', points: truePts, opacity: 1 },
        ...trajectorySeries,
      ],
    });

    // ELBO plot
    plotLineSeries({
      container: viElboContainer,
      xs: elbos.map((_, i) => i + 1),
      ys: elbos,
      xLabel: 'step',
      yLabel: 'ELBO',
      name: 'ELBO',
      color: 'rgba(255, 255, 255, 0.85)',
    });

    const elbo0 = elbos[0];
    const elboN = elbos[elbos.length - 1];
    viSummaryEl.textContent = `Parsed n=${n} flips, k=${k}. True posterior: Beta(${aPost.toFixed(2)}, ${bPost.toFixed(2)}). q starts at m=${initialM.toFixed(3)}, log s=${initialLogS.toFixed(3)} and ends at m=${m.toFixed(3)}, log s=${logS.toFixed(3)}. ELBO: ${elbo0.toFixed(3)} → ${elboN.toFixed(3)}.`;
  }

  function renderBetaBinomial() {
    const alpha0 = clampPositive(Number(alphaInput.value), 0.1);
    const beta0 = clampPositive(Number(betaInput.value), 0.1);

    const flips = parseCoinFlips(dataText.value);
    const n = flips.length;
    const k = flips.reduce((s, v) => s + v, 0);

    if (n === 0) {
      clear(paramContainer);
      clear(predContainer);
      summaryEl.textContent = 'No coin flips parsed yet. Paste values like: 1, 0, 1, 1, 0 (or H/T).';
      return;
    }

    const aPost = alpha0 + k;
    const bPost = beta0 + (n - k);

    const grid = d3.range(0, 1.0001, 1 / 400).map((x) => Math.min(0.999999, Math.max(0.000001, x)));

    const priorPts = grid.map((x) => ({ x, y: betaPdf(x, alpha0, beta0) }));

    // Likelihood as a normalized curve over p: p^k (1-p)^(n-k), normalized numerically.
    const likeRaw = grid.map((x) => Math.pow(x, k) * Math.pow(1 - x, n - k));
    const likeArea = d3.sum(likeRaw) * (1 / 400);
    const likePts = grid.map((x, i) => ({ x, y: likeArea > 0 ? likeRaw[i] / likeArea : 0 }));

    const postPts = grid.map((x) => ({ x, y: betaPdf(x, aPost, bPost) }));

    // 95% credible interval via deterministic Monte Carlo (seeded by text+priors)
    const seed = hashStringToSeed(`${dataText.value}|${alpha0}|${beta0}`);
    const rng = mulberry32(seed);
    const samples = Array.from({ length: 5000 }, () => sampleBeta(aPost, bPost, rng));
    const q025 = quantileFromSamples(samples, 0.025);
    const q975 = quantileFromSamples(samples, 0.975);

    plotContinuous({
      container: paramContainer,
      xDomain: [0, 1],
      xLabel: 'p',
      yLabel: 'density',
      series: [
        { name: 'prior', color: 'rgba(255, 255, 255, 0.75)', points: priorPts, opacity: 0.9 },
        { name: 'likelihood', color: 'rgba(74, 163, 255, 0.85)', points: likePts, opacity: 0.85 },
        { name: 'posterior', color: 'rgba(125, 255, 178, 0.95)', points: postPts, opacity: 1 },
      ],
      shadeInterval: [q025, q975],
      shadeSeriesIndex: 2,
    });

    const pNext = aPost / (aPost + bPost);
    plotBars({
      container: predContainer,
      labels: ['0', '1'],
      values: [1 - pNext, pNext],
      colors: ['rgba(255, 255, 255, 0.45)', 'rgba(125, 255, 178, 0.85)'],
      title: 'P(X_new | D) for next flip',
    });

    summaryEl.textContent = `Parsed n=${n} flips, k=${k} ones. Posterior is Beta(${aPost.toFixed(2)}, ${bPost.toFixed(2)}). Next-flip probabilities: P(0 | D)=${(1 - pNext).toFixed(3)}, P(1 | D)=${pNext.toFixed(3)}. 95% credible interval for p ≈ [${q025.toFixed(3)}, ${q975.toFixed(3)}].`;
  }

  function renderNormalNormal() {
    const mu0 = Number(mu0Input.value);
    const tau = clampPositive(Number(tauInput.value), 0.05);
    const sigma = clampPositive(Number(sigmaInput.value), 0.05);

    const xs = parseNumbers(dataText.value);
    const n = xs.length;
    if (n === 0) {
      clear(paramContainer);
      clear(predContainer);
      summaryEl.textContent = 'No numbers parsed yet. Paste values like: 0.2, -0.1, 0.4, 0.0.';
      return;
    }

    const xbar = d3.mean(xs);

    const tau2 = tau * tau;
    const sigma2 = sigma * sigma;

    const tauN2 = 1 / (1 / tau2 + n / sigma2);
    const muN = tauN2 * (mu0 / tau2 + (n * xbar) / sigma2);

    // Choose a plotting window based on prior/posterior scale
    const center = Number.isFinite(muN) ? muN : mu0;
    const spread = 4 * Math.max(tau, Math.sqrt(tauN2), sigma / Math.sqrt(Math.max(1, n)));
    const xMin = center - spread;
    const xMax = center + spread;

    const grid = d3.range(0, 401).map((i) => xMin + (xMax - xMin) * (i / 400));

    const priorPts = grid.map((x) => ({ x, y: normalPdf(x, mu0, tau) }));

    // Likelihood for mu given data: Normal(xbar, sigma/sqrt(n)), normalized
    const likeSd = n > 0 ? sigma / Math.sqrt(n) : sigma;
    const likePts = grid.map((x) => ({ x, y: normalPdf(x, xbar, likeSd) }));

    const postSd = Math.sqrt(tauN2);
    const postPts = grid.map((x) => ({ x, y: normalPdf(x, muN, postSd) }));

    const ciLow = muN - 1.96 * postSd;
    const ciHigh = muN + 1.96 * postSd;

    plotContinuous({
      container: paramContainer,
      xDomain: [xMin, xMax],
      xLabel: '\u03bc',
      yLabel: 'density',
      series: [
        { name: 'prior', color: 'rgba(255, 255, 255, 0.75)', points: priorPts, opacity: 0.9 },
        { name: 'likelihood', color: 'rgba(74, 163, 255, 0.85)', points: likePts, opacity: 0.85 },
        { name: 'posterior', color: 'rgba(125, 255, 178, 0.95)', points: postPts, opacity: 1 },
      ],
      shadeInterval: [ciLow, ciHigh],
      shadeSeriesIndex: 2,
    });

    // Posterior predictive for x_new: Normal(muN, sqrt(sigma^2 + tauN2))
    const predSd = Math.sqrt(sigma2 + tauN2);
    const predSpread = 4 * predSd;
    const pxMin = muN - predSpread;
    const pxMax = muN + predSpread;
    const pGrid = d3.range(0, 401).map((i) => pxMin + (pxMax - pxMin) * (i / 400));
    const predPts = pGrid.map((x) => ({ x, y: normalPdf(x, muN, predSd) }));

    plotContinuous({
      container: predContainer,
      xDomain: [pxMin, pxMax],
      xLabel: 'x_new',
      yLabel: 'density',
      series: [{ name: 'predictive', color: 'rgba(255, 255, 255, 0.85)', points: predPts, opacity: 1 }],
    });

    summaryEl.textContent = `Parsed n=${n} values, mean=${xbar.toFixed(3)}. Posterior mean for μ is ${muN.toFixed(3)} with posterior std ${postSd.toFixed(3)}. A new observation is predicted to vary with predictive std ${predSd.toFixed(3)}. 95% credible interval for μ ≈ [${ciLow.toFixed(3)}, ${ciHigh.toFixed(3)}].`;
  }

  function setModel(model) {
    const isBeta = model === 'beta-binomial';
    betaControls.hidden = !isBeta;
    normalControls.hidden = isBeta;

    if (isBeta) {
      document.getElementById('bayesDataHint').textContent = 'For coin flips: use 1/0 or H/T. You can also use k/n as a shorthand (e.g., 7/10).';
    } else {
      document.getElementById('bayesDataHint').textContent = 'For Normal–Normal: paste numbers separated by commas/spaces/newlines.';
    }
  }

  function loadDefaults(model = modelSelect.value) {
    if (model === 'beta-binomial') {
      dataText.value = '1, 1, 0, 1, 0, 1, 1, 1, 0, 1';
      alphaInput.value = '2';
      betaInput.value = '2';
      if (probHeadsInput) probHeadsInput.value = '0.6';
    } else {
      dataText.value = '0.2, -0.1, 0.4, 0.3, 0.0, 0.1, 0.6, -0.2, 0.2';
      mu0Input.value = '0';
      tauInput.value = '1';
      sigmaInput.value = '0.3';
      if (trueMeanInput) trueMeanInput.value = '0';
    }
  }

  function update() {
    syncConjugatePanelText();
    const model = modelSelect.value;
    if (model === 'beta-binomial') renderBetaBinomial();
    else renderNormalNormal();

    // Re-typeset any formulas that might have reflowed.
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise().catch(() => {});
    }
  }

  modelSelect.addEventListener('change', () => {
    setModel(modelSelect.value);
    loadDefaults(modelSelect.value);
    update();
  });

  const debouncedUpdate = debounce(update, 240);
  [dataText, alphaInput, betaInput, mu0Input, tauInput, sigmaInput, probHeadsInput, trueMeanInput].forEach((el) => {
    if (!el) return;
    el.addEventListener('input', debouncedUpdate);
    el.addEventListener('change', update);
  });

  flipCoinBtn?.addEventListener('click', () => {
    const probHeads = Math.min(1, Math.max(0, Number(probHeadsInput?.value)));
    const flip = Math.random() < probHeads ? 1 : 0;
    appendDatasetValue(dataText, flip);
    update();
  });

  generateSampleBtn?.addEventListener('click', () => {
    const trueMean = Number(trueMeanInput?.value);
    const sigma = clampPositive(Number(sigmaInput?.value), 0.05);
    const sample = (Number.isFinite(trueMean) ? trueMean : 0) + sigma * sampleStandardNormal();
    appendDatasetValue(dataText, sample.toFixed(3));
    update();
  });

  // Update on resize
  window.addEventListener('resize', () => {
    // Avoid doing too much work while resizing
    if (update._t) clearTimeout(update._t);
    update._t = setTimeout(update, 120);
  });

  // Initial render
  setModel(modelSelect.value);
  loadDefaults(modelSelect.value);
  update();

  // MAP demo wiring (if present)
  if (mapObjContainer) {
    const debouncedMapUpdate = debounce(renderMAPDemo, 240);

    [mapDataText, mapMu0Input, mapTauInput, mapSigmaInput, mapTrueMeanInput].forEach((el) => {
      if (!el) return;
      el.addEventListener('input', debouncedMapUpdate);
      el.addEventListener('change', renderMAPDemo);
    });

    mapGenerateSampleBtn?.addEventListener('click', () => {
      const trueMean = Number(mapTrueMeanInput?.value);
      const sigma = clampPositive(Number(mapSigmaInput?.value), 0.05);
      const sample = (Number.isFinite(trueMean) ? trueMean : 0) + sigma * sampleStandardNormal();
      appendDatasetValue(mapDataText, sample.toFixed(3));
      renderMAPDemo();
    });

    // Render once on load (use whatever defaults are in the HTML)
    renderMAPDemo();

    // Keep responsive
    window.addEventListener('resize', () => {
      if (renderMAPDemo._t) clearTimeout(renderMAPDemo._t);
      renderMAPDemo._t = setTimeout(renderMAPDemo, 120);
    });
  }

  // VI demo wiring (if present)
  if (viPostContainer) {
    const debouncedVIRun = debounce(runVIDemo, 280);

    [
      viDataText,
      viProbHeadsInput,
      viAlphaInput,
      viBetaInput,
      viMInput,
      viLogSInput,
      viStepsInput,
      viLRInput,
      viMCInput,
    ].forEach((el) => {
      if (!el) return;
      el.addEventListener('input', debouncedVIRun);
      el.addEventListener('change', runVIDemo);
    });

    viFlipCoinBtn?.addEventListener('click', () => {
      const probHeads = Math.min(1, Math.max(0, Number(viProbHeadsInput?.value)));
      const flip = Math.random() < probHeads ? 1 : 0;
      appendDatasetValue(viDataText, flip);
      runVIDemo();
    });

    // Render once on load using the current optimizer settings
    runVIDemo();

    window.addEventListener('resize', () => {
      if (runVIDemo._t) clearTimeout(runVIDemo._t);
      runVIDemo._t = setTimeout(runVIDemo, 120);
    });
  }
});
