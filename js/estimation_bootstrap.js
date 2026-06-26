(function () {
  "use strict";

  function mulberry32(seed) {
    let t = seed >>> 0;
    return function () {
      t += 0x6d2b79f5;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }

  function randn(rng) {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function clampInt(x, lo, hi) {
    const v = Math.trunc(Number(x));
    if (!Number.isFinite(v)) return lo;
    return Math.max(lo, Math.min(hi, v));
  }

  function clampFloat(x, lo, hi) {
    const v = Number(x);
    if (!Number.isFinite(v)) return lo;
    return Math.max(lo, Math.min(hi, v));
  }

  function fmt(x, digits = 4) {
    if (!Number.isFinite(x)) return "-";
    const abs = Math.abs(x);
    if (abs >= 1000 || (abs > 0 && abs < 1e-3)) return x.toExponential(2);
    return x.toFixed(digits);
  }

  function init() {
    const container = document.querySelector("#bootstrapCI");
    if (!container) return;
    if (typeof d3 === "undefined") {
      console.error("D3 is required for bootstrap CI widget.");
      return;
    }

    const elPlot = container.querySelector("#bootstrapPlot");
    const elStats = container.querySelector("#bootstrapStats");

    const elN = container.querySelector("#bootN");
    const elB = container.querySelector("#bootB");
    const elLevel = container.querySelector("#bootLevel");
    const elSeed = container.querySelector("#bootSeed");
    const elMu = container.querySelector("#bootMu");
    const elSigma = container.querySelector("#bootSigma");
    const elStat = container.querySelector("#bootStat");

    const state = {
      sample: [],
      last: null,
    };

    // Web Worker for the bootstrap loop -- keeps the main thread free during computation.
    const worker = new Worker('../js/estimation-bootstrap-worker.js');
    let generation = 0;

    function getConfig() {
      const n = clampInt(elN.value, 5, 500);
      const B = clampInt(elB.value, 100, 20000);
      const level = clampFloat(elLevel.value, 0.5, 0.999);
      const seed = clampInt(elSeed.value, 0, 1_000_000_000);
      const mu = clampFloat(elMu.value, -10, 10);
      const sigma = clampFloat(elSigma.value, 0.05, 10);
      const statKey = elStat.value;

      elN.value = String(n);
      elB.value = String(B);
      elLevel.value = String(level);
      elSeed.value = String(seed);
      elMu.value = String(mu);
      elSigma.value = String(sigma);

      return { n, B, level, seed, mu, sigma, statKey };
    }

    function drawSample() {
      const { n, seed, mu, sigma } = getConfig();
      const rng = mulberry32(seed);
      const sample = new Array(n);
      for (let i = 0; i < n; i++) sample[i] = mu + sigma * randn(rng);
      state.sample = sample;
      state.last = null;
    }

    const margin = { top: 16, right: 18, bottom: 34, left: 46 };
    const height = 320;

    function drawChart({ stats, thetaHat, se, ci: { lo, hi } }, { mu, sigma, level, statKey }) {
      const binsCount = Math.max(12, Math.min(46, Math.round(Math.sqrt(stats.length))));
      const xMin = Math.min(stats[0], thetaHat, lo);
      const xMax = Math.max(stats[stats.length - 1], thetaHat, hi);
      const pad = (xMax - xMin) * 0.06 || 1;

      const width = Math.max(320, elPlot.getBoundingClientRect().width);
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;

      d3.select(elPlot).selectAll("*").remove();
      const svg = d3
        .select(elPlot)
        .append("svg")
        .attr("class", "boot-svg")
        .attr("width", "100%")
        .attr("height", height);

      const g = svg
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      const x = d3
        .scaleLinear()
        .domain([xMin - pad, xMax + pad])
        .range([0, innerW]);

      const bin = d3
        .bin()
        .domain(x.domain())
        .thresholds(binsCount);

      const bins = bin(stats);

      const y = d3
        .scaleLinear()
        .domain([0, d3.max(bins, (b) => b.length) || 1])
        .nice()
        .range([innerH, 0]);

      g.append("g")
        .attr("class", "x axis")
        .attr("transform", `translate(0,${innerH})`)
        .call(d3.axisBottom(x).ticks(6));

      g.append("g").attr("class", "y axis").call(d3.axisLeft(y).ticks(5));

      g.append("text")
        .attr("x", innerW / 2)
        .attr("y", innerH + 30)
        .attr("text-anchor", "middle")
        .attr("fill", "rgba(255,255,255,0.75)")
        .attr("font-size", 12)
        .text(`Bootstrap distribution of ${statKey}`);

      // CI band
      g.append("rect")
        .attr("class", "ci-band")
        .attr("x", x(lo))
        .attr("y", 0)
        .attr("width", Math.max(0, x(hi) - x(lo)))
        .attr("height", innerH);

      // bars
      g.append("g")
        .attr("class", "bars")
        .selectAll("rect")
        .data(bins)
        .enter()
        .append("rect")
        .attr("x", (d) => x(d.x0) + 1)
        .attr("width", (d) => Math.max(0, x(d.x1) - x(d.x0) - 2))
        .attr("y", (d) => y(d.length))
        .attr("height", (d) => innerH - y(d.length));

      function vline(xVal, cls) {
        g.append("line")
          .attr("class", cls)
          .attr("x1", x(xVal))
          .attr("x2", x(xVal))
          .attr("y1", 0)
          .attr("y2", innerH);
      }

      vline(lo, "ci-line");
      vline(hi, "ci-line");
      vline(thetaHat, "hat-line");

      // True parameter marker for the simulated population.
      // For mean it's μ; for median (Normal) it's also μ.
      vline(mu, "truth-line");

      // Internal legend (inside the plot area)
      (function drawLegend() {
        const legendPadding = 10;
        const lineH = 16;
        const sampleW = 24;
        const legendW = 160;
        const legendH = 4 * lineH + 2 * legendPadding;
        const x0 = Math.max(6, innerW - legendW - 6);
        const y0 = 6;

        const lg = g.append("g").attr("class", "boot-legend").attr("transform", `translate(${x0},${y0})`);
        lg.append("rect").attr("class", "legend-box").attr("width", legendW).attr("height", legendH).attr("rx", 8);

        function item(y, label, drawSample) {
          const row = lg.append("g").attr("transform", `translate(${legendPadding},${legendPadding + y})`);
          drawSample(row);
          row.append("text")
            .attr("class", "legend-text")
            .attr("x", sampleW + 10)
            .attr("y", 4)
            .attr("dominant-baseline", "middle")
            .text(label);
        }

        item(0 * lineH, "truth", (row) => {
          row.append("line")
            .attr("class", "legend-truth")
            .attr("x1", 0)
            .attr("x2", sampleW)
            .attr("y1", 4)
            .attr("y2", 4);
        });

        item(1 * lineH, "estimate", (row) => {
          row.append("line")
            .attr("class", "legend-hat")
            .attr("x1", 0)
            .attr("x2", sampleW)
            .attr("y1", 4)
            .attr("y2", 4);
        });

        item(2 * lineH, "CI bounds", (row) => {
          row.append("line")
            .attr("class", "legend-ci")
            .attr("x1", 0)
            .attr("x2", sampleW)
            .attr("y1", 4)
            .attr("y2", 4);
        });

        item(3 * lineH, "CI band", (row) => {
          row.append("rect")
            .attr("class", "legend-ci-band")
            .attr("x", 0)
            .attr("y", -1)
            .attr("width", sampleW)
            .attr("height", 10)
            .attr("rx", 2);
        });
      })();

      elStats.innerHTML = `
        <div class="boot-metrics">
          <div><b>Population</b>: \\(\\mathcal{N}(\\mu=${fmt(mu, 3)},\\,\\sigma=${fmt(sigma, 3)})\\)</div>
          <div><b>Point estimate</b>: \\(\\hat{\\theta} = ${fmt(thetaHat, 4)}\\)</div>
          <div><b>Bootstrap SE</b>: \\(${fmt(se, 4)}\\)</div>
          <div><b>${Math.round(level * 100)}% percentile CI</b>: \\([${fmt(lo, 4)}, ${fmt(hi, 4)}]\\)</div>
        </div>
      `;

      if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
        if (typeof window.MathJax.typesetClear === "function") {
          window.MathJax.typesetClear([elStats]);
        }
        window.MathJax.typesetPromise([elStats]).catch((err) => console.error(err));
      }

      state.last = { stats, thetaHat, se, ci: { lo, hi } };
    }

    worker.onmessage = function ({ data: result }) {
      if (result.token !== generation) return;
      const { mu, sigma, level, statKey } = getConfig();
      try {
        drawChart(result, { mu, sigma, level, statKey });
      } catch (e) {
        console.error(e);
        elStats.textContent = "Error rendering bootstrap widget. See console.";
      }
    };

    function dispatchWorker({ redrawSample } = {}) {
      if (!state.sample.length || redrawSample) drawSample();
      const { B, level, seed, statKey } = getConfig();
      const token = ++generation;
      worker.postMessage({ token, sample: state.sample, statKey, B, seed, level });
    }

    // Auto-update: debounce so rapid keystrokes coalesce before dispatching to the worker.
    let renderTimer = null;
    function scheduleRender({ redrawSample } = { redrawSample: false }) {
      if (renderTimer) window.clearTimeout(renderTimer);
      renderTimer = window.setTimeout(() => dispatchWorker({ redrawSample }), 120);
    }

    function onControlEvent(el) {
      const redrawSample = (el === elN || el === elMu || el === elSigma || el === elSeed);
      scheduleRender({ redrawSample });
    }

    [elN, elB, elLevel, elSeed, elMu, elSigma, elStat].forEach((el) => {
      el.addEventListener("input", () => onControlEvent(el));
      el.addEventListener("change", () => onControlEvent(el));
    });

    window.addEventListener("resize", () => {
      if (state.last) {
        const { mu, sigma, level, statKey } = getConfig();
        drawChart(state.last, { mu, sigma, level, statKey });
      }
    });

    drawSample();
    dispatchWorker();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
