(function () {
    "use strict";

    const COLORS = {
        ols: "#e0e0e0",
        ridge: "#4aa3ff",
        lasso: "#ff7a59",
        l2gd: "#7dffb2",
        wd: "#c77dff",
        truth: "rgba(255,255,255,0.35)",
        points: "rgba(255,255,255,0.8)",
    };

    const METHODS = [
        { key: "ols", label: "Control (OLS)" },
        { key: "ridge", label: "Ridge" },
        { key: "lasso", label: "Lasso" },
        { key: "l2gd", label: "L2 penalty (GD)" },
        { key: "wd", label: "Weight decay (GD)" },
    ];

    // Display guardrails (keeps charts readable even if GD diverges).
    // Only apply tight clipping to the GD-based methods.
    const COEF_ABS_DISPLAY_CAP_TIGHT = 20;
    const PLOT_Y_ABS_DISPLAY_CAP_TIGHT = 20;

    function mulberry32(seed) {
        // https://github.com/cprosche/mulberry32
        let t = seed >>> 0;
        return function () {
            t += 0x6D2B79F5;
            let r = Math.imul(t ^ (t >>> 15), 1 | t);
            r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
            return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
        };
    }

    function randn(rng) {
        // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        let u = 0, v = 0;
        while (u === 0) u = rng();
        while (v === 0) v = rng();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    function softThreshold(z, lambda) {
        if (z > lambda) return z - lambda;
        if (z < -lambda) return z + lambda;
        return 0;
    }

    function mse(yTrue, yPred) {
        let s = 0;
        for (let i = 0; i < yTrue.length; i++) {
            const d = yTrue[i] - yPred[i];
            s += d * d;
        }
        return s / yTrue.length;
    }

    function mseGradient(X, y, w) {
        // Gradient of (1/N)||Xw - y||^2 w.r.t. w
        const N = X.length;
        const P = X[0].length;
        const yhat = predict(X, w);
        const grad = new Array(P).fill(0);

        for (let i = 0; i < N; i++) {
            const e = yhat[i] - y[i];
            for (let j = 0; j < P; j++) grad[j] += X[i][j] * e;
        }

        const scale = 2 / N;
        for (let j = 0; j < P; j++) grad[j] *= scale;
        return grad;
    }

    function l2Norm(vec, startIndex = 0) {
        let s = 0;
        for (let i = startIndex; i < vec.length; i++) s += vec[i] * vec[i];
        return Math.sqrt(s);
    }

    function countNonzero(vec, eps = 1e-8, startIndex = 0) {
        let c = 0;
        for (let i = startIndex; i < vec.length; i++) if (Math.abs(vec[i]) > eps) c++;
        return c;
    }

    function buildPolyFeatures(xs, degree) {
        // returns matrix N x (degree+1): [1, x, x^2, ...]
        const N = xs.length;
        const P = degree + 1;
        const X = new Array(N);
        for (let i = 0; i < N; i++) {
            const row = new Array(P);
            row[0] = 1;
            let v = 1;
            for (let j = 1; j < P; j++) {
                v *= xs[i];
                row[j] = v;
            }
            X[i] = row;
        }
        return X;
    }

    function standardizeColumns(X) {
        // Standardize columns j>=1, leave intercept column 0 alone.
        const N = X.length;
        const P = X[0].length;
        const means = new Array(P).fill(0);
        const stds = new Array(P).fill(1);

        for (let j = 1; j < P; j++) {
            let m = 0;
            for (let i = 0; i < N; i++) m += X[i][j];
            m /= N;
            means[j] = m;

            let v = 0;
            for (let i = 0; i < N; i++) {
                const d = X[i][j] - m;
                v += d * d;
            }
            v /= N;
            const s = Math.sqrt(v) || 1;
            stds[j] = s;
        }

        const Xs = new Array(N);
        for (let i = 0; i < N; i++) {
            const row = new Array(P);
            row[0] = 1;
            for (let j = 1; j < P; j++) row[j] = (X[i][j] - means[j]) / stds[j];
            Xs[i] = row;
        }

        return { Xs, means, stds };
    }

    function applyStandardization(X, means, stds) {
        const N = X.length;
        const P = X[0].length;
        const Xs = new Array(N);
        for (let i = 0; i < N; i++) {
            const row = new Array(P);
            row[0] = 1;
            for (let j = 1; j < P; j++) row[j] = (X[i][j] - means[j]) / stds[j];
            Xs[i] = row;
        }
        return Xs;
    }

    function predict(X, w) {
        const N = X.length;
        const yhat = new Array(N);
        for (let i = 0; i < N; i++) {
            let s = 0;
            for (let j = 0; j < w.length; j++) s += X[i][j] * w[j];
            yhat[i] = s;
        }
        return yhat;
    }

    function transposeMul(X) {
        // XtX, scaled by 1/N
        const N = X.length;
        const P = X[0].length;
        const XtX = Array.from({ length: P }, () => new Array(P).fill(0));
        for (let i = 0; i < N; i++) {
            const row = X[i];
            for (let a = 0; a < P; a++) {
                const va = row[a];
                for (let b = 0; b <= a; b++) {
                    XtX[a][b] += va * row[b];
                }
            }
        }
        const invN = 1 / N;
        for (let a = 0; a < P; a++) {
            for (let b = 0; b <= a; b++) {
                const v = XtX[a][b] * invN;
                XtX[a][b] = v;
                XtX[b][a] = v;
            }
        }
        return XtX;
    }

    function transposeMulVec(X, y) {
        // X^T y, scaled by 1/N
        const N = X.length;
        const P = X[0].length;
        const Xty = new Array(P).fill(0);
        for (let i = 0; i < N; i++) {
            const row = X[i];
            for (let j = 0; j < P; j++) Xty[j] += row[j] * y[i];
        }
        const invN = 1 / N;
        for (let j = 0; j < P; j++) Xty[j] *= invN;
        return Xty;
    }

    function addDiag(A, diag) {
        const P = A.length;
        const B = A.map(row => row.slice());
        for (let i = 0; i < P; i++) B[i][i] += diag[i];
        return B;
    }

    function solveLinear(A, b) {
        // Gaussian elimination with partial pivoting.
        const n = A.length;
        const M = A.map(row => row.slice());
        const x = b.slice();

        for (let k = 0; k < n; k++) {
            // pivot
            let pivotRow = k;
            let pivotVal = Math.abs(M[k][k]);
            for (let i = k + 1; i < n; i++) {
                const v = Math.abs(M[i][k]);
                if (v > pivotVal) {
                    pivotVal = v;
                    pivotRow = i;
                }
            }
            if (pivotVal < 1e-12) {
                // singular-ish: nudge diagonal
                M[k][k] += 1e-8;
            } else if (pivotRow !== k) {
                const tmp = M[k];
                M[k] = M[pivotRow];
                M[pivotRow] = tmp;

                const tb = x[k];
                x[k] = x[pivotRow];
                x[pivotRow] = tb;
            }

            const diag = M[k][k];
            for (let j = k; j < n; j++) M[k][j] /= diag;
            x[k] /= diag;

            for (let i = 0; i < n; i++) {
                if (i === k) continue;
                const factor = M[i][k];
                if (factor === 0) continue;
                for (let j = k; j < n; j++) M[i][j] -= factor * M[k][j];
                x[i] -= factor * x[k];
            }
        }

        return x;
    }

    function fitOLS(X, y) {
        const XtX = transposeMul(X);
        const Xty = transposeMulVec(X, y);
        // tiny diagonal for stability
        const diag = new Array(X[0].length).fill(0);
        for (let i = 0; i < diag.length; i++) diag[i] = (i === 0 ? 0 : 1e-10);
        const A = addDiag(XtX, diag);
        return solveLinear(A, Xty);
    }

    function fitRidge(X, y, lambda) {
        const XtX = transposeMul(X);
        const Xty = transposeMulVec(X, y);
        const P = X[0].length;
        const diag = new Array(P).fill(0);
        for (let i = 1; i < P; i++) diag[i] = lambda;
        const A = addDiag(XtX, diag);
        return solveLinear(A, Xty);
    }

    function fitLasso(X, y, lambda, iters) {
        return fitLassoWarm(X, y, lambda, iters, null);
    }

    function fitLassoWarm(X, y, lambda, iters, wInit) {
        // Same objective as fitLasso, but warm-start from wInit.
        const N = X.length;
        const P = X[0].length;

        const w = (wInit && wInit.length === P) ? wInit.slice() : new Array(P).fill(0);
        const yhat = new Array(N).fill(0);

        // initialize yhat = Xw
        for (let i = 0; i < N; i++) {
            let s = 0;
            for (let j = 0; j < P; j++) s += X[i][j] * w[j];
            yhat[i] = s;
        }

        const colNorm = new Array(P).fill(0);
        for (let j = 0; j < P; j++) {
            let s = 0;
            for (let i = 0; i < N; i++) s += X[i][j] * X[i][j];
            colNorm[j] = s / N;
        }

        for (let iter = 0; iter < iters; iter++) {
            // intercept
            {
                let rMean = 0;
                for (let i = 0; i < N; i++) rMean += (y[i] - yhat[i]);
                rMean /= N;
                const delta = rMean;
                w[0] += delta;
                for (let i = 0; i < N; i++) yhat[i] += delta;
            }

            for (let j = 1; j < P; j++) {
                const wOld = w[j];
                let rho = 0;
                for (let i = 0; i < N; i++) {
                    const xij = X[i][j];
                    const r = y[i] - (yhat[i] - xij * wOld);
                    rho += xij * r;
                }
                rho /= N;

                const wNew = softThreshold(rho, 0.5 * lambda) / (colNorm[j] || 1);
                const delta = wNew - wOld;
                if (delta !== 0) {
                    w[j] = wNew;
                    for (let i = 0; i < N; i++) yhat[i] += X[i][j] * delta;
                }
            }
        }

        return w;
    }

    function fitL2Gd(X, y, lambda, lr, iters) {
        return fitL2GdWarm(X, y, lambda, lr, iters, null);
    }

    function fitL2GdWarm(X, y, lambda, lr, iters, wInit) {
        const N = X.length;
        const P = X[0].length;
        const w = (wInit && wInit.length === P) ? wInit.slice() : new Array(P).fill(0);

        for (let t = 0; t < iters; t++) {
            const grad = mseGradient(X, y, w);

            for (let j = 1; j < P; j++) grad[j] += 2 * lambda * w[j];

            for (let j = 0; j < P; j++) w[j] -= lr * grad[j];
        }

        return w;
    }

    function fitWd(X, y, lambda, lr, iters) {
        return fitWdWarm(X, y, lambda, lr, iters, null);
    }

    function fitWdWarm(X, y, lambda, lr, iters, wInit) {
        const N = X.length;
        const P = X[0].length;
        const w = (wInit && wInit.length === P) ? wInit.slice() : new Array(P).fill(0);

        for (let t = 0; t < iters; t++) {
            const grad = mseGradient(X, y, w);

            const decay = 1 - 2 * lr * lambda;
            for (let j = 1; j < P; j++) w[j] *= decay;

            for (let j = 0; j < P; j++) w[j] -= lr * grad[j];
        }

        return w;
    }

    function trueFunction(name, x) {
        if (name === "linear") return 0.9 * x + 0.3;
        if (name === "cubic") return 0.15 * x * x * x - 0.2 * x * x + 0.4 * x;
        // sine
        return Math.sin(1.25 * x);
    }

    function makeDataset({ trueFn, nTrain, noiseVar, seed }) {
        const rng = mulberry32(seed);
        const sigma = Math.sqrt(Math.max(0, noiseVar));
        const xs = [];
        const ys = [];
        for (let i = 0; i < nTrain; i++) {
            const x = -3 + 6 * rng();
            const y = trueFunction(trueFn, x) + sigma * randn(rng);
            xs.push(x);
            ys.push(y);
        }

        const nTest = 200;
        const xt = [];
        const yt = [];
        for (let i = 0; i < nTest; i++) {
            const x = -3 + (6 * i) / (nTest - 1);
            xt.push(x);
            yt.push(trueFunction(trueFn, x) + sigma * randn(rng));
        }

        return { xs, ys, xt, yt };
    }

    function fmt(v) {
        if (!Number.isFinite(v)) return "-";
        const av = Math.abs(v);
        if (av >= 1e5) return v.toExponential(2);
        if (av >= 1000) return v.toFixed(0);
        if (av >= 10) return v.toFixed(2);
        return v.toFixed(3);
    }

    function fmt3(v) {
        if (!Number.isFinite(v)) return "-";
        const av = Math.abs(v);
        if (av >= 1e5) return v.toExponential(3);
        // Round to at most 3 decimals and avoid trailing zeros.
        const rounded = Math.round(v * 1000) / 1000;
        return rounded.toLocaleString(undefined, { maximumFractionDigits: 3 });
    }

    function fmtLambda(v) {
        if (!Number.isFinite(v)) return "-";
        if (v === 0) return "0";
        const av = Math.abs(v);
        // λ can be extremely small; prefer scientific notation there.
        if (av < 1e-3 || av >= 1e4) return v.toExponential(4);
        // Otherwise show more precision than other metrics.
        return v.toLocaleString(undefined, { maximumSignificantDigits: 8 });
    }

    function init() {
        const root = document;
        const el = {
            regViz: root.getElementById("regViz"),
            coefViz: root.getElementById("coefViz"),
            metricsTable: root.getElementById("metricsTable"),
            coeffTitle: root.getElementById("coeffTitle"),
            coeffInfo: root.getElementById("coeffInfo"),
            trueFn: root.getElementById("trueFn"),
            nTrain: root.getElementById("nTrain"),
            noise: root.getElementById("noise"),
            seed: root.getElementById("seed"),
            degree: root.getElementById("degree"),
            lambda: root.getElementById("lambda"),
            lambdaValue: root.getElementById("lambdaValue"),
            selectedMethod: root.getElementById("selectedMethod"),
            showOLS: root.getElementById("showOLS"),
            showRidge: root.getElementById("showRidge"),
            showLasso: root.getElementById("showLasso"),
            showL2GD: root.getElementById("showL2GD"),
            showWD: root.getElementById("showWD"),
            lr: root.getElementById("lr"),
            iters: root.getElementById("iters"),
            lassoIters: root.getElementById("lassoIters"),

            bvMseViz: root.getElementById("bvMseViz"),
            bvBvnViz: root.getElementById("bvBvnViz"),
            bvSummaryTable: root.getElementById("bvSummaryTable"),
            bvRecompute: root.getElementById("bvRecompute"),
            bvStatus: root.getElementById("bvStatus"),
        };

        if (!el.regViz) return;

        const state = {
            dataset: null,
            fits: {},
            degree: Number(el.degree.value),
            trueFn: el.trueFn.value,
            nTrain: Number(el.nTrain.value),
            // Noise control is variance (σ²)
            noise: Number(el.noise.value),
            seed: Number(el.seed.value),
            log10lambda: Number(el.lambda.value),
            lambda: Math.pow(10, Number(el.lambda.value)),
            lr: Number(el.lr.value),
            iters: Number(el.iters.value),
            lassoIters: Number(el.lassoIters.value),
        };

        // --- D3 setup
        const container = d3.select(el.regViz);
        const svg = container.append("svg").attr("class", "reg-svg");
        const gMain = svg.append("g");

        const gAxes = gMain.append("g").attr("class", "axes");
        const gTruth = gMain.append("path").attr("class", "truth");
        const gLines = gMain.append("g").attr("class", "lines");
        const gPoints = gMain.append("g").attr("class", "points");

        // coefficient svg
        const coefContainer = d3.select(el.coefViz);
        const coefSvg = coefContainer.append("svg").attr("class", "coef-svg");
        const coefG = coefSvg.append("g");

        // bias-variance svgs (optional: only if section exists)
        const bv = {
            enabled: Boolean(el.bvMseViz && el.bvBvnViz && el.bvSummaryTable),
            mseSvg: null,
            mseG: null,
            bvnSvg: null,
            bvnG: null,
            mseTooltip: null,
            bvnTooltip: null,
            cacheKey: null,
            cached: null,
            dirty: true,
            computing: false,
        };

        if (bv.enabled) {
            const mseContainer = d3.select(el.bvMseViz);
            bv.mseSvg = mseContainer.append("svg").attr("class", "bv-svg");
            bv.mseG = bv.mseSvg.append("g");

            bv.mseTooltip = mseContainer.append("div")
                .attr("class", "bv-tooltip");

            const bvnContainer = d3.select(el.bvBvnViz);
            bv.bvnSvg = bvnContainer.append("svg").attr("class", "bv-svg");
            bv.bvnG = bv.bvnSvg.append("g");

            bv.bvnTooltip = bvnContainer.append("div")
                .attr("class", "bv-tooltip");
        }

        function updateLambda() {
            state.log10lambda = Number(el.lambda.value);
            state.lambda = Math.pow(10, state.log10lambda);
            el.lambdaValue.textContent = `λ = ${fmt(state.lambda)} (10^${state.log10lambda.toFixed(2)})`;
        }

        function setSelection({ methodKey, log10lambda = null }) {
            if (methodKey && el.selectedMethod.value !== methodKey) {
                el.selectedMethod.value = methodKey;
            }
            if (log10lambda !== null && Number.isFinite(log10lambda) && el.lambda) {
                el.lambda.value = String(log10lambda);
            }
            // Keep λ display in sync immediately; rerender re-reads controls.
            updateLambda();
            rerender();
        }

        function setBvStatus(msg) {
            if (!bv.enabled || !el.bvStatus) return;
            el.bvStatus.textContent = msg;
        }

        function setBvBusy(isComputing) {
            bv.computing = Boolean(isComputing);
            if (el.bvRecompute) {
                el.bvRecompute.disabled = bv.computing;
                el.bvRecompute.textContent = bv.computing ? "Recomputing…" : "Recompute sweep";
            }
        }

        function palette(methodKey) {
            const base = COLORS[methodKey] || "#e0e0e0";
            const c = d3.color(base) || d3.color("#e0e0e0");
            const rgba = (alpha) => `rgba(${c.r},${c.g},${c.b},${alpha})`;
            return {
                base,
                strong: rgba(0.95),
                mid: rgba(0.65),
                soft: rgba(0.38),
                faint: rgba(0.22),
            };
        }

        const BV_GD_DISPLAY_CAP = 20;

        function isGd(methodKey) {
            return methodKey === "l2gd" || methodKey === "wd";
        }

        function finiteCap(cap) {
            return (v) => (Number.isFinite(v) && v <= cap ? v : NaN);
        }

        function capMse(methodKey) {
            const cap = isGd(methodKey) ? BV_GD_DISPLAY_CAP : Infinity;
            if (!Number.isFinite(cap)) return (v) => v;
            return finiteCap(cap);
        }

        function bestTestMse(points, methodKey) {
            const cap = capMse(methodKey);
            let best = null;
            let bestVal = Infinity;
            for (const p of points) {
                const v = cap(p.testMSE);
                if (!Number.isFinite(v)) continue;
                if (v < bestVal) {
                    bestVal = v;
                    best = p;
                }
            }
            return best;
        }

        function readControls() {
            state.trueFn = el.trueFn.value;
            state.nTrain = Number(el.nTrain.value);
            state.noise = Number(el.noise.value);
            state.seed = Number(el.seed.value);
            state.degree = Math.max(1, Math.min(18, Number(el.degree.value)));
            state.lr = Number(el.lr.value);
            state.iters = Number(el.iters.value);
            state.lassoIters = Number(el.lassoIters.value);
            updateLambda();
        }

        function regenData() {
            readControls();
            state.dataset = makeDataset({
                trueFn: state.trueFn,
                nTrain: state.nTrain,
                noiseVar: state.noise,
                seed: state.seed,
            });
        }

        function fitAll() {
            const { xs, ys } = state.dataset;
            const degree = state.degree;
            const lambda = state.lambda;

            const X = buildPolyFeatures(xs, degree);
            const { Xs, means, stds } = standardizeColumns(X);

            const fits = {};
            fits._standardization = { means, stds, degree };

            fits.ols = fitOLS(Xs, ys);
            fits.ridge = fitRidge(Xs, ys, lambda);
            fits.lasso = fitLasso(Xs, ys, lambda, state.lassoIters);
            fits.l2gd = fitL2Gd(Xs, ys, lambda, state.lr, state.iters);
            fits.wd = fitWd(Xs, ys, lambda, state.lr, state.iters);

            state.fits = fits;
        }

        function methodVisible(key) {
            if (key === "ols") return el.showOLS.checked;
            if (key === "ridge") return el.showRidge.checked;
            if (key === "lasso") return el.showLasso.checked;
            if (key === "l2gd") return el.showL2GD.checked;
            if (key === "wd") return el.showWD.checked;
            return false;
        }

        function layout() {
            const width = Math.max(320, el.regViz.clientWidth || 720);
            const height = 420;
            const margin = { top: 16, right: 30, bottom: 42, left: 52 };

            svg.attr("width", width).attr("height", height);
            gMain.attr("transform", `translate(${margin.left},${margin.top})`);

            const innerW = width - margin.left - margin.right;
            const innerH = height - margin.top - margin.bottom;

            return { width, height, margin, innerW, innerH };
        }

        function coefLayout() {
            const width = Math.max(320, el.coefViz.clientWidth || 520);
            const height = 220;
            const margin = { top: 10, right: 10, bottom: 28, left: 44 };
            coefSvg.attr("width", width).attr("height", height);
            coefG.attr("transform", `translate(${margin.left},${margin.top})`);
            return { width, height, margin, innerW: width - margin.left - margin.right, innerH: height - margin.top - margin.bottom };
        }

        function computeCurves() {
            const gridN = 240;
            const xGrid = d3.range(gridN).map(i => -3 + (6 * i) / (gridN - 1));
            const Xgrid = buildPolyFeatures(xGrid, state.degree);
            const std = state.fits._standardization;
            const XgridS = applyStandardization(Xgrid, std.means, std.stds);

            const yTrue = xGrid.map(x => trueFunction(state.trueFn, x));

            const curves = { xGrid, yTrue, methods: {} };
            for (const m of METHODS) {
                const w = state.fits[m.key];
                curves.methods[m.key] = predict(XgridS, w);
            }
            return curves;
        }

        function computeMetrics() {
            const { xs, ys, xt, yt } = state.dataset;
            const std = state.fits._standardization;

            const Xtr = applyStandardization(buildPolyFeatures(xs, std.degree), std.means, std.stds);
            const Xte = applyStandardization(buildPolyFeatures(xt, std.degree), std.means, std.stds);

            const out = {};
            for (const m of METHODS) {
                const w = state.fits[m.key];
                const ytr = predict(Xtr, w);
                const yte = predict(Xte, w);
                out[m.key] = {
                    trainMSE: mse(ys, ytr),
                    testMSE: mse(yt, yte),
                    l2: l2Norm(w, 1),
                    nnz: countNonzero(w, 1e-6, 1),
                };
            }
            return out;
        }

        function bvLayout(containerEl, svgSel, gSel) {
            const width = Math.max(320, containerEl.clientWidth || 520);
            const height = 280;
            const margin = { top: 14, right: 30, bottom: 42, left: 52 };
            svgSel.attr("width", width).attr("height", height);
            gSel.attr("transform", `translate(${margin.left},${margin.top})`);
            return { width, height, margin, innerW: width - margin.left - margin.right, innerH: height - margin.top - margin.bottom };
        }

        function lambdaSweep() {
            // Coarse sweep for performance (esp. Lasso + GD), but aligned to the slider's step
            // so the table shows values that are actually reachable with the UI.
            const minAttr = Number(el.lambda?.min);
            const maxAttr = Number(el.lambda?.max);
            const stepAttr = Number(el.lambda?.step);

            const lo = Number.isFinite(minAttr) ? minAttr : -6;
            const hi = Number.isFinite(maxAttr) ? maxAttr : 2;
            const step = (Number.isFinite(stepAttr) && stepAttr > 0) ? stepAttr : 0.05;

            // Take every Nth slider tick to keep it cheap, but still valid.
            const stride = 5; // 5 * 0.05 = 0.25 in log10 space
            const tickCount = Math.max(1, Math.floor((hi - lo) / step) + 1);

            const out = [];
            for (let i = 0; i < tickCount; i += stride) {
                const log10lambda = lo + i * step;
                out.push({ log10lambda, lambda: Math.pow(10, log10lambda) });
            }
            // Ensure we include the max end of the slider.
            if (out.length === 0 || out[out.length - 1].log10lambda < hi - 1e-12) {
                out.push({ log10lambda: hi, lambda: Math.pow(10, hi) });
            }
            return out;
        }

        function fitSweep(methodKey, Xs, y, lambda, wInit) {
            if (methodKey === "ols") return fitOLS(Xs, y);
            if (methodKey === "ridge") return fitRidge(Xs, y, lambda);
            if (methodKey === "lasso") {
                const iters = Math.max(10, Math.min(120, state.lassoIters));
                return fitLassoWarm(Xs, y, lambda, iters, wInit);
            }
            if (methodKey === "l2gd") {
                const iters = Math.max(50, Math.min(1200, state.iters));
                return fitL2GdWarm(Xs, y, lambda, state.lr, iters, wInit);
            }
            if (methodKey === "wd") {
                const iters = Math.max(50, Math.min(1200, state.iters));
                return fitWdWarm(Xs, y, lambda, state.lr, iters, wInit);
            }
            return fitRidge(Xs, y, lambda);
        }

        function computeBvAll() {
            if (!bv.enabled) return null;

            const sweep = lambdaSweep();
            const degree = state.degree;
            const sigma2 = Math.max(0, state.noise);
            const sigma = Math.sqrt(sigma2);

            // Resamples: tuned for interactivity.
            const B = 20;
            const evalN = 200;
            const xEval = d3.range(evalN).map(i => -3 + (6 * i) / (evalN - 1));
            const fStar = xEval.map(x => trueFunction(state.trueFn, x));
            const XevalRaw = buildPolyFeatures(xEval, degree);

            const methodKeys = METHODS.map(m => m.key);
            const methodAgg = {};
            for (const key of methodKeys) {
                methodAgg[key] = {
                    sums: sweep.map(() => ({
                        valid: 0,
                        trainMSESum: 0,
                        predSum: new Array(evalN).fill(0),
                        pred2Sum: new Array(evalN).fill(0),
                    })),
                    testMSESum: new Array(sweep.length).fill(0),
                    testValid: new Array(sweep.length).fill(0),
                };
            }

            for (let b = 0; b < B; b++) {
                const repSeed = state.seed + 1009 * (b + 1);
                const rng = mulberry32(repSeed);

                const xs = [];
                const ys = [];
                for (let i = 0; i < state.nTrain; i++) {
                    const x = -3 + 6 * rng();
                    const y = trueFunction(state.trueFn, x) + sigma * randn(rng);
                    xs.push(x);
                    ys.push(y);
                }

                const yTestNoisy = new Array(evalN);
                for (let i = 0; i < evalN; i++) yTestNoisy[i] = fStar[i] + sigma * randn(rng);

                const XtrRaw = buildPolyFeatures(xs, degree);
                const { Xs, means, stds } = standardizeColumns(XtrRaw);
                const Xeval = applyStandardization(XevalRaw, means, stds);

                // warm starts per method across the sweep
                const warm = {};
                for (const key of methodKeys) warm[key] = null;

                // OLS is lambda-independent: fit once and reuse across sweep points.
                let wOls = null;
                try {
                    wOls = fitOLS(Xs, ys);
                    if (!wOls || wOls.some(v => !Number.isFinite(v))) wOls = null;
                } catch (_) {
                    wOls = null;
                }

                for (let k = 0; k < sweep.length; k++) {
                    const { lambda } = sweep[k];

                    for (const methodKey of methodKeys) {
                        let w;
                        if (methodKey === "ols") {
                            w = wOls;
                        } else {
                            try {
                                    w = fitSweep(methodKey, Xs, ys, lambda, warm[methodKey]);
                            } catch (_) {
                                w = null;
                            }
                        }

                        if (!w || w.some(v => !Number.isFinite(v))) {
                            warm[methodKey] = null;
                            continue;
                        }

                        warm[methodKey] = w;

                        const agg = methodAgg[methodKey];
                        const s = agg.sums[k];

                        const yhatTr = predict(Xs, w);
                        const trMSE = mse(ys, yhatTr);
                        if (Number.isFinite(trMSE)) s.trainMSESum += trMSE;

                        const yhatEval = predict(Xeval, w);
                        let ok = true;
                        for (let i = 0; i < evalN; i++) {
                            const v = yhatEval[i];
                            if (!Number.isFinite(v)) { ok = false; break; }
                        }
                        if (!ok) {
                            warm[methodKey] = null;
                            continue;
                        }

                        s.valid++;
                        for (let i = 0; i < evalN; i++) {
                            const v = yhatEval[i];
                            s.predSum[i] += v;
                            s.pred2Sum[i] += v * v;
                        }

                        const teMSE = mse(yTestNoisy, yhatEval);
                        if (Number.isFinite(teMSE)) {
                            agg.testMSESum[k] += teMSE;
                            agg.testValid[k] += 1;
                        }
                    }
                }
            }

            const byMethod = {};
            for (const methodKey of methodKeys) {
                const agg = methodAgg[methodKey];
                const points = [];

                for (let k = 0; k < sweep.length; k++) {
                    const s = agg.sums[k];
                    const valid = s.valid;
                    if (valid <= 0) {
                        points.push({
                            log10lambda: sweep[k].log10lambda,
                            lambda: sweep[k].lambda,
                            trainMSE: NaN,
                            testMSE: NaN,
                            bias2: NaN,
                            variance: NaN,
                            noise: sigma2,
                            valid,
                        });
                        continue;
                    }

                    let bias2 = 0;
                    let variance = 0;
                    for (let i = 0; i < evalN; i++) {
                        const mu = s.predSum[i] / valid;
                        const ex2 = s.pred2Sum[i] / valid;
                        const v = Math.max(0, ex2 - mu * mu);
                        const b2 = (mu - fStar[i]) * (mu - fStar[i]);
                        bias2 += b2;
                        variance += v;
                    }
                    bias2 /= evalN;
                    variance /= evalN;

                    const trainMSE = s.trainMSESum / valid;
                    const testMSE = (agg.testValid[k] > 0) ? (agg.testMSESum[k] / agg.testValid[k]) : (bias2 + variance + sigma2);

                    points.push({
                        log10lambda: sweep[k].log10lambda,
                        lambda: sweep[k].lambda,
                        trainMSE,
                        testMSE,
                        bias2,
                        variance,
                        noise: sigma2,
                        valid,
                    });
                }

                byMethod[methodKey] = {
                    methodKey,
                    methodLabel: (METHODS.find(m => m.key === methodKey)?.label) || methodKey,
                    points,
                    sigma2,
                    B,
                };
            }

            return { byMethod, sweep, sigma2, B };
        }

        function renderBvTable(bvAll) {
            if (!bv.enabled) return;
            if (!bvAll || !bvAll.byMethod) {
                el.bvSummaryTable.textContent = "-";
                return;
            }

            const rows = [];
            for (const m of METHODS) {
                const data = bvAll.byMethod[m.key];
                if (!data || !data.points || !data.points.length) continue;
                const best = bestTestMse(data.points, m.key);
                if (!best) continue;
                rows.push({
                    key: m.key,
                    label: data.methodLabel,
                    best,
                });
            }

            if (!rows.length) {
                el.bvSummaryTable.textContent = "-";
                return;
            }

            const selected = el.selectedMethod.value;

            const table = d3.create("table").attr("class", "metric-table");
            const thead = table.append("thead");
            thead.append("tr")
                .selectAll("th")
                .data(["Method", "$\\log_{10}\\lambda$", "$\\lambda$", "Train MSE", "Test MSE", "$\\text{bias}^2$", "variance", "noise ($\\sigma^2$)"])
                .enter()
                .append("th")
                .text(d => d);

            const tbody = table.append("tbody");

            const tr = tbody.selectAll("tr")
                .data(rows)
                .enter()
                .append("tr")
                .attr("class", d => (d.key === selected ? "selected" : null))
                .attr("tabindex", 0)
                .attr("role", "button")
                .attr("aria-label", d => `Select ${d.label} at its lowest Test MSE λ`)
                .on("click", (_, d) => {
                    const log10lambda = (d.key === "ols") ? null : (d.best?.log10lambda ?? null);
                    setSelection({ methodKey: d.key, log10lambda });
                })
                .on("keydown", (event, d) => {
                    if (event.key !== "Enter" && event.key !== " ") return;
                    event.preventDefault();
                    const log10lambda = (d.key === "ols") ? null : (d.best?.log10lambda ?? null);
                    setSelection({ methodKey: d.key, log10lambda });
                });

            tr.append("td").text(d => d.label);
            tr.append("td").text(d => (d.key === "ols" ? "-" : d.best.log10lambda.toFixed(2)));
            tr.append("td").text(d => (d.key === "ols" ? "-" : fmtLambda(d.best.lambda)));
            tr.append("td").text(d => fmt3(d.best.trainMSE));
            tr.append("td").text(d => fmt3(d.best.testMSE));
            tr.append("td").text(d => fmt3(d.best.bias2));
            tr.append("td").text(d => fmt3(d.best.variance));
            tr.append("td").text(d => fmt3(d.best.noise));

            el.bvSummaryTable.innerHTML = "";
            el.bvSummaryTable.appendChild(table.node());
        }

        function renderBvMseChart(data) {
            // Specialized rendering for the Train vs Test MSE chart:
            // - hover tooltip (log10λ, Train MSE, Test MSE)
            // - dashed vertical marker at min Test MSE, with label
            const points = data.points;
            const cap = capMse(data.methodKey);
            const pal = palette(data.methodKey);
            const series = [
                { key: "trainMSE", label: "Train MSE", color: pal.mid, dash: "6 4" },
                { key: "testMSE", label: "Test MSE", color: pal.strong, dash: null },
            ];

            const containerEl = el.bvMseViz;
            const svgSel = bv.mseSvg;
            const gSel = bv.mseG;

            const L = bvLayout(containerEl, svgSel, gSel);
            const innerW = L.innerW;
            const innerH = L.innerH;

            gSel.selectAll("*").remove();

            const x = d3.scaleLinear()
                .domain(d3.extent(points, d => d.log10lambda))
                .range([0, innerW]);

            const allY = [];
            for (const s of series) {
                for (const p of points) {
                    const v = cap(p[s.key]);
                    if (Number.isFinite(v)) allY.push(v);
                }
            }

            const y = d3.scaleLinear()
                .domain(d3.extent(allY.length ? allY : [0, 1]))
                .nice()
                .range([innerH, 0]);

            gSel.append("g")
                .attr("transform", `translate(0,${innerH})`)
                .call(d3.axisBottom(x).ticks(6))
                .selectAll("text");

            gSel.append("g")
                .call(d3.axisLeft(y).ticks(6))
                .selectAll("text");

            const line = d3.line()
                .defined(d => Number.isFinite(d.y))
                .x(d => x(d.x))
                .y(d => y(d.y));

            for (const s of series) {
                const dataLine = points.map(p => ({ x: p.log10lambda, y: cap(p[s.key]) }));
                gSel.append("path")
                    .attr("fill", "none")
                    .attr("stroke", s.color)
                    .attr("stroke-width", 2)
                    .attr("opacity", 0.92)
                    .attr("stroke-dasharray", s.dash || null)
                    .attr("d", line(dataLine));
            }

            // Best (min) Test MSE marker.
            const best = bestTestMse(points, data.methodKey);
            if (best) {
                const bx = x(best.log10lambda);
                gSel.append("line")
                    .attr("x1", bx)
                    .attr("x2", bx)
                    .attr("y1", 0)
                    .attr("y2", innerH)
                    .attr("stroke", "rgba(255,255,255,0.92)")
                    .attr("stroke-width", 1.5)
                    .attr("stroke-dasharray", "6 4");
            }

            // Current lambda marker (subtle), useful when comparing to best.
            gSel.append("line")
                .attr("x1", x(state.log10lambda))
                .attr("x2", x(state.log10lambda))
                .attr("y1", 0)
                .attr("y2", innerH)
                .attr("stroke", pal.mid)
                .attr("stroke-width", 1)
                .attr("stroke-dasharray", "2 6");

            // Legend
            const leg = gSel.append("g").attr("class", "bv-legend").attr("transform", `translate(10,10)`);
            const items = leg.selectAll("g.item").data(series.map(s => ({ label: s.label, color: s.color }))).enter().append("g").attr("class", "item")
                .attr("transform", (_, i) => `translate(0,${i * 16})`);
            items.append("line").attr("x1", 0).attr("x2", 18).attr("y1", 0).attr("y2", 0).attr("stroke-width", 3)
                .attr("stroke", d => d.color)
                .attr("stroke-dasharray", (_, i) => (series[i]?.dash || null));
            items.append("text")
                .attr("class", "legend-text")
                .attr("x", 24)
                .attr("y", 4)
                .text(d => d.label);

            gSel.append("text")
                .attr("x", innerW / 2)
                .attr("y", innerH + 30)
                .attr("text-anchor", "middle")
                .attr("class", "axis-label")
                .text("log10(λ)");

            // Hover interaction
            const tooltip = bv.mseTooltip;
            if (tooltip) tooltip.classed("visible", false);

            const focus = gSel.append("g").style("display", "none");
            focus.append("line")
                .attr("class", "focus-x")
                .attr("y1", 0)
                .attr("y2", innerH)
                .attr("stroke", "rgba(255,255,255,0.35)")
                .attr("stroke-dasharray", "4 4");

            const focusDots = focus.selectAll("circle")
                .data(series)
                .enter()
                .append("circle")
                .attr("r", 4)
                .attr("fill", d => d.color)
                .attr("stroke", "rgba(0,0,0,0.35)")
                .attr("stroke-width", 1);

            const overlay = gSel.append("rect")
                .attr("class", "bv-overlay")
                .attr("x", 0)
                .attr("y", 0)
                .attr("width", innerW)
                .attr("height", innerH)
                .attr("fill", "transparent");

            function nearestPoint(xVal) {
                let bestP = null;
                let bestDist = Infinity;
                for (const p of points) {
                    // Only consider points we actually display (both lines capped & finite)
                    if (!Number.isFinite(cap(p.trainMSE)) || !Number.isFinite(cap(p.testMSE))) continue;
                    const d = Math.abs(p.log10lambda - xVal);
                    if (d < bestDist) {
                        bestDist = d;
                        bestP = p;
                    }
                }
                return bestP;
            }

            function showTooltip(event, p) {
                if (!tooltip) return;
                const rect = containerEl.getBoundingClientRect();
                const pad = 10;
                const tr = cap(p.trainMSE);
                const te = cap(p.testMSE);
                if (!Number.isFinite(tr) || !Number.isFinite(te)) {
                    tooltip.classed("visible", false);
                    return;
                }
                const html = `log10(λ) = <b>${p.log10lambda.toFixed(2)}</b><br>`
                    + `Train MSE = <b>${fmt3(tr)}</b><br>`
                    + `Test MSE = <b>${fmt3(te)}</b>`;
                tooltip.html(html);

                const tw = tooltip.node().offsetWidth || 180;
                const th = tooltip.node().offsetHeight || 64;
                let left = (event.clientX - rect.left) + 12;
                let top = (event.clientY - rect.top) - th - 10;
                if (left + tw + pad > rect.width) left = rect.width - tw - pad;
                if (left < pad) left = pad;
                if (top < pad) top = (event.clientY - rect.top) + 12;
                if (top + th + pad > rect.height) top = rect.height - th - pad;
                tooltip.style("left", `${left}px`).style("top", `${top}px`).classed("visible", true);
            }

            overlay
                .on("mouseenter", () => {
                    focus.style("display", null);
                })
                .on("mouseleave", () => {
                    focus.style("display", "none");
                    if (tooltip) tooltip.classed("visible", false);
                })
                .on("mousemove", (event) => {
                    const [mx] = d3.pointer(event);
                    const xVal = x.invert(mx);
                    const p = nearestPoint(xVal);
                    if (!p) return;
                    const fx = x(p.log10lambda);
                    focus.select("line.focus-x").attr("x1", fx).attr("x2", fx);
                    focusDots
                        .attr("cx", () => fx)
                        .attr("cy", d => y(cap(p[d.key])));
                    showTooltip(event, p);
                });
        }

        function renderBvDecompChart(data) {
            // Bias/variance/noise decomposition chart with hover tooltip.
            const points = data.points;
            const cap = capMse(data.methodKey);

            const pal = palette(data.methodKey);
            const series = [
                { key: "bias2", label: "bias²", color: pal.strong, dash: null },
                { key: "variance", label: "variance", color: pal.mid, dash: "6 4" },
                { key: "noise", label: "noise (σ²)", color: pal.soft, dash: "2 4" },
            ];

            const containerEl = el.bvBvnViz;
            const svgSel = bv.bvnSvg;
            const gSel = bv.bvnG;

            const L = bvLayout(containerEl, svgSel, gSel);
            const innerW = L.innerW;
            const innerH = L.innerH;

            gSel.selectAll("*").remove();

            const x = d3.scaleLinear()
                .domain(d3.extent(points, d => d.log10lambda))
                .range([0, innerW]);

            const allY = [];
            for (const s of series) {
                for (const p of points) {
                    const v = cap(p[s.key]);
                    if (Number.isFinite(v)) allY.push(v);
                }
            }
            const y = d3.scaleLinear()
                .domain(d3.extent(allY.length ? allY : [0, 1]))
                .nice()
                .range([innerH, 0]);

            gSel.append("g")
                .attr("transform", `translate(0,${innerH})`)
                .call(d3.axisBottom(x).ticks(6))
                .selectAll("text");

            gSel.append("g")
                .call(d3.axisLeft(y).ticks(6))
                .selectAll("text");

            const line = d3.line()
                .defined(d => Number.isFinite(d.y))
                .x(d => x(d.x))
                .y(d => y(d.y));

            for (const s of series) {
                const dataLine = points.map(p => ({ x: p.log10lambda, y: cap(p[s.key]) }));
                gSel.append("path")
                    .attr("fill", "none")
                    .attr("stroke", s.color)
                    .attr("stroke-width", 2)
                    .attr("opacity", 0.92)
                    .attr("stroke-dasharray", s.dash || null)
                    .attr("d", line(dataLine));
            }

            // Best (min) Test MSE marker (same λ as in the Train/Test MSE chart).
            const best = bestTestMse(points, data.methodKey);
            if (best) {
                const bx = x(best.log10lambda);
                gSel.append("line")
                    .attr("x1", bx)
                    .attr("x2", bx)
                    .attr("y1", 0)
                    .attr("y2", innerH)
                    .attr("stroke", "rgba(255,255,255,0.75)")
                    .attr("stroke-width", 1.5)
                    .attr("stroke-dasharray", "6 4");
            }

            // Current lambda marker
            gSel.append("line")
                .attr("x1", x(state.log10lambda))
                .attr("x2", x(state.log10lambda))
                .attr("y1", 0)
                .attr("y2", innerH)
                .attr("stroke", pal.mid)
                .attr("stroke-dasharray", "4 4");

            // Legend
            const leg = gSel.append("g").attr("class", "bv-legend").attr("transform", `translate(10,10)`);
            const items = leg.selectAll("g.item")
                .data(series.map(s => ({ label: s.label, color: s.color })))
                .enter()
                .append("g")
                .attr("class", "item")
                .attr("transform", (_, i) => `translate(0,${i * 16})`);
            items.append("line").attr("x1", 0).attr("x2", 18).attr("y1", 0).attr("y2", 0).attr("stroke-width", 3)
                .attr("stroke", d => d.color)
                .attr("stroke-dasharray", (_, i) => (series[i]?.dash || null));
            items.append("text")
                .attr("class", "legend-text")
                .attr("x", 24)
                .attr("y", 4)
                .text(d => d.label);

            gSel.append("text")
                .attr("x", innerW)
                .attr("y", innerH + 34)
                .attr("text-anchor", "end")
                .attr("class", "axis-label")
                .text("log10(λ)");

            // Hover interaction
            const tooltip = bv.bvnTooltip;
            if (tooltip) tooltip.classed("visible", false);

            const focus = gSel.append("g").style("display", "none");
            focus.append("line")
                .attr("class", "focus-x")
                .attr("y1", 0)
                .attr("y2", innerH)
                .attr("stroke", "rgba(255,255,255,0.35)")
                .attr("stroke-dasharray", "4 4");

            const focusDots = focus.selectAll("circle")
                .data(series)
                .enter()
                .append("circle")
                .attr("r", 4)
                .attr("fill", d => d.color)
                .attr("stroke", "rgba(0,0,0,0.35)")
                .attr("stroke-width", 1);

            const overlay = gSel.append("rect")
                .attr("class", "bv-overlay")
                .attr("x", 0)
                .attr("y", 0)
                .attr("width", innerW)
                .attr("height", innerH)
                .attr("fill", "transparent");

            function nearestPoint(xVal) {
                let bestP = null;
                let bestDist = Infinity;
                for (const p of points) {
                    // require at least one visible value at this x
                    const anyVisible = series.some(s => Number.isFinite(cap(p[s.key])));
                    if (!anyVisible) continue;
                    const d = Math.abs(p.log10lambda - xVal);
                    if (d < bestDist) {
                        bestDist = d;
                        bestP = p;
                    }
                }
                return bestP;
            }

            function showTooltip(event, p) {
                if (!tooltip) return;
                const rect = containerEl.getBoundingClientRect();
                const pad = 10;

                const b2 = cap(p.bias2);
                const vv = cap(p.variance);
                const nn = cap(p.noise);

                const f = (v) => (Number.isFinite(v) ? fmt3(v) : "-");
                const html = `log10(λ) = <b>${p.log10lambda.toFixed(2)}</b><br>`
                    + `bias² = <b>${f(b2)}</b><br>`
                    + `variance = <b>${f(vv)}</b><br>`
                    + `noise (σ²) = <b>${f(nn)}</b>`;
                tooltip.html(html);

                const tw = tooltip.node().offsetWidth || 180;
                const th = tooltip.node().offsetHeight || 80;
                let left = (event.clientX - rect.left) + 12;
                let top = (event.clientY - rect.top) - th - 10;
                if (left + tw + pad > rect.width) left = rect.width - tw - pad;
                if (left < pad) left = pad;
                if (top < pad) top = (event.clientY - rect.top) + 12;
                if (top + th + pad > rect.height) top = rect.height - th - pad;
                tooltip.style("left", `${left}px`).style("top", `${top}px`).classed("visible", true);
            }

            overlay
                .on("mouseenter", () => {
                    focus.style("display", null);
                })
                .on("mouseleave", () => {
                    focus.style("display", "none");
                    if (tooltip) tooltip.classed("visible", false);
                })
                .on("mousemove", (event) => {
                    const [mx] = d3.pointer(event);
                    const xVal = x.invert(mx);
                    const p = nearestPoint(xVal);
                    if (!p) return;
                    const fx = x(p.log10lambda);
                    focus.select("line.focus-x").attr("x1", fx).attr("x2", fx);

                    focusDots
                        .attr("cx", () => fx)
                        .attr("cy", d => y(cap(p[d.key])))
                        .attr("display", d => (Number.isFinite(cap(p[d.key])) ? null : "none"));

                    showTooltip(event, p);
                });
        }

        function renderBvEmpty(message) {
            if (!bv.enabled) return;

            const L1 = bvLayout(el.bvMseViz, bv.mseSvg, bv.mseG);
            bv.mseG.selectAll("*").remove();
            bv.mseG.append("text")
                .attr("class", "bv-placeholder")
                .attr("x", L1.innerW / 2)
                .attr("y", L1.innerH / 2)
                .attr("text-anchor", "middle")
                .text(message);

            const L2 = bvLayout(el.bvBvnViz, bv.bvnSvg, bv.bvnG);
            bv.bvnG.selectAll("*").remove();
            bv.bvnG.append("text")
                .attr("class", "bv-placeholder")
                .attr("x", L2.innerW / 2)
                .attr("y", L2.innerH / 2)
                .attr("text-anchor", "middle")
                .text(message);

            if (el.bvSummaryTable) el.bvSummaryTable.textContent = "-";
        }

        function renderBvSection() {
            if (!bv.enabled) return;

            // The sweep is expensive; only compute it when the user presses the button.
            // Here we only invalidate cached results when inputs change.
            const cacheKey = JSON.stringify({
                trueFn: state.trueFn,
                nTrain: state.nTrain,
                noise: state.noise,
                seed: state.seed,
                degree: state.degree,
                lr: state.lr,
                iters: state.iters,
                lassoIters: state.lassoIters,
            });

            if (bv.cacheKey !== cacheKey) {
                bv.cacheKey = cacheKey;
                bv.cached = null;
                bv.dirty = true;
                setBvStatus("Settings changed - press “Recompute sweep” to update.");
            }

            if (!bv.cached) {
                if (bv.dirty) {
                    renderBvEmpty("Press “Recompute sweep”");
                }
                return;
            }

            const all = bv.cached;
            if (!all || !all.byMethod) return;

            const data = all.byMethod[el.selectedMethod.value] || all.byMethod.ridge || all.byMethod.ols;
            if (!data) return;

            // MSE sweep plot (interactive + best marker)
            renderBvMseChart(data);

            // Bias/variance/noise decomposition plot
            renderBvDecompChart(data);

            renderBvTable(all);
            setBvStatus(`Computed sweep (B=${all.B}, steps=${all.sweep?.length || "?"}).`);

            if (window.MathJax) {
                try {
                    if (typeof window.MathJax.typesetPromise === "function") {
                        window.MathJax.typesetPromise([el.bvSummaryTable]);
                    } else if (typeof window.MathJax.typeset === "function") {
                        window.MathJax.typeset([el.bvSummaryTable]);
                    }
                } catch (_) {
                    /* ignore */
                }
            }
        }

        function renderMetrics(metrics) {
            const rows = METHODS.map(m => {
                const r = metrics[m.key];
                return {
                    key: m.key,
                    label: m.label,
                    train: r.trainMSE,
                    test: r.testMSE,
                    l2: r.l2,
                    nnz: r.nnz,
                };
            });

            const selected = el.selectedMethod.value;

            const table = d3.create("table").attr("class", "metric-table");
            const thead = table.append("thead");
            thead.append("tr")
                .selectAll("th")
                .data(["Method", "Train MSE", "Test MSE", "$\\lVert w\\rVert_2$", "# of nonzero $w$"])
                .enter()
                .append("th")
                .text(d => d);

            const tbody = table.append("tbody");
            const tr = tbody.selectAll("tr")
                .data(rows)
                .enter()
                .append("tr")
                .attr("class", d => (d.key === selected ? "selected" : null))
                .attr("tabindex", 0)
                .attr("role", "button")
                .attr("aria-label", d => `Select ${d.label}`)
                .on("click", (_, d) => {
                    if (el.selectedMethod.value === d.key) return;
                    setSelection({ methodKey: d.key });
                })
                .on("keydown", (event, d) => {
                    if (event.key !== "Enter" && event.key !== " ") return;
                    event.preventDefault();
                    if (el.selectedMethod.value === d.key) return;
                    setSelection({ methodKey: d.key });
                });

            tr.append("td").text(d => d.label);
            tr.append("td").text(d => fmt3(d.train));
            tr.append("td").text(d => fmt3(d.test));
            tr.append("td").text(d => fmt3(d.l2));
            tr.append("td").text(d => String(d.nnz));

            el.metricsTable.innerHTML = "";
            el.metricsTable.appendChild(table.node());
        }

        function renderCoefs() {
            const key = el.selectedMethod.value;
            const w = state.fits[key];
            const degree = state.degree;

            const showCoeffInfo = (msg) => {
                if (!el.coeffInfo) return;
                el.coeffInfo.style.display = "block";
                el.coeffInfo.textContent = msg;
            };

            const hideCoeffInfo = () => {
                if (!el.coeffInfo) return;
                el.coeffInfo.style.display = "none";
                el.coeffInfo.textContent = "";
            };

            if (el.coeffTitle) {
                const label = (METHODS.find(m => m.key === key)?.label) || key;
                el.coeffTitle.textContent = label;
            }

            if (!w || !Array.isArray(w) || w.length === 0) {
                showCoeffInfo("coefficients unavailable");
                coefG.selectAll("*").remove();
                return;
            }

            const hasNonFinite = w.some(v => !Number.isFinite(v));
            if (hasNonFinite) {
                showCoeffInfo("coefficients diverged (non-finite values)");
                coefG.selectAll("*").remove();
                return;
            }

            hideCoeffInfo();

            const items = [];
            for (let j = 0; j < w.length; j++) {
                items.push({
                    name: j === 0 ? "bias" : `x^${j}`,
                    idx: j,
                    value: w[j],
                    abs: Math.abs(w[j]),
                });
            }

            const rawMaxAbs = d3.max(items, d => d.abs) || 0;
            const coefCap = (key === "l2gd" || key === "wd") ? COEF_ABS_DISPLAY_CAP_TIGHT : Infinity;
            const clipped = Number.isFinite(coefCap) && rawMaxAbs > coefCap;
            for (const it of items) {
                it.cappedAbs = Number.isFinite(coefCap) ? Math.min(it.abs, coefCap) : it.abs;
                it.isClipped = Number.isFinite(coefCap) && it.abs > coefCap;
            }

            const L = coefLayout();
            const innerW = L.innerW;
            const innerH = L.innerH;

            const x = d3.scaleBand()
                .domain(items.map(d => d.name))
                .range([0, innerW])
                .padding(0.2);

            const maxAbs = d3.max(items, d => d.cappedAbs) || 1;
            const y = d3.scaleLinear()
                .domain([0, maxAbs])
                .nice()
                .range([innerH, 0]);

            coefG.selectAll(".coef-axis").remove();
            coefG.append("g").attr("class", "coef-axis")
                .attr("transform", `translate(0,${innerH})`)
                .call(d3.axisBottom(x).tickValues(items.length > 10 ? items.filter((_, i) => (i % 2 === 0)).map(d => d.name) : items.map(d => d.name)))
                .selectAll("text");

            coefG.append("g").attr("class", "coef-axis")
                .call((() => {
                    const yAxis = d3.axisLeft(y).ticks(5);
                    if (maxAbs >= 1e5) yAxis.tickFormat(d3.format(".2e"));
                    return yAxis;
                })())
                .selectAll("text");
            const bars = coefG.selectAll("rect.bar").data(items, d => d.name);
            bars.enter()
                .append("rect")
                .attr("class", "bar")
                .attr("x", d => x(d.name))
                .attr("width", x.bandwidth())
                .attr("y", innerH)
                .attr("height", 0)
                .attr("fill", COLORS[key] || "rgba(255,255,255,0.6)")
                .merge(bars)
                .transition()
                .duration(180)
                .attr("x", d => x(d.name))
                .attr("width", x.bandwidth())
                .attr("y", d => y(d.cappedAbs))
                .attr("height", d => innerH - y(d.cappedAbs))
                .attr("fill", d => (d.isClipped ? "rgba(255,255,255,0.35)" : (COLORS[key] || "rgba(255,255,255,0.6)")));

            bars.exit().remove();
        }

        function renderMainPlot(curves) {
            const L = layout();
            const innerW = L.innerW;
            const innerH = L.innerH;

            const { xs, ys } = state.dataset;

            const visibleMethods = METHODS.filter(m => methodVisible(m.key));
            const visibleKeys = new Set(visibleMethods.map(m => m.key));

            const clamp = (v, cap) => {
                if (!Number.isFinite(cap)) return v;
                return (v > cap ? cap : (v < -cap ? -cap : v));
            };

            // Sanitize method curves for display: clip extreme values and break lines on non-finite values.
            const displayMethods = {};
            const methodFlags = {};
            for (const m of METHODS) {
                const arr = curves.methods[m.key] || [];
                const cap = (m.key === "l2gd" || m.key === "wd") ? PLOT_Y_ABS_DISPLAY_CAP_TIGHT : Infinity;
                let anyNonFinite = false;
                let anyClipped = false;
                const out = new Array(arr.length);
                for (let i = 0; i < arr.length; i++) {
                    const v = arr[i];
                    if (!Number.isFinite(v)) {
                        anyNonFinite = true;
                        out[i] = NaN;
                        continue;
                    }
                    const c = clamp(v, cap);
                    if (c !== v) anyClipped = true;
                    out[i] = c;
                }
                displayMethods[m.key] = out;
                methodFlags[m.key] = { anyNonFinite, anyClipped, cap };
            }

            const displayTruth = curves.yTrue.map(v => (Number.isFinite(v) ? v : NaN));

            const anyVisibleNonFinite = METHODS.some(m => visibleKeys.has(m.key) && methodFlags[m.key]?.anyNonFinite);
            const anyVisibleClipped = METHODS.some(m => visibleKeys.has(m.key) && methodFlags[m.key]?.anyClipped);

            const allY = [
                ...ys,
                ...displayTruth,
                ...visibleMethods.flatMap(m => displayMethods[m.key] || []),
            ].filter(Number.isFinite);

            const xScale = d3.scaleLinear().domain([-3, 3]).range([0, innerW]);
            const yScale = d3.scaleLinear()
                .domain(d3.extent(allY))
                .nice()
                .range([innerH, 0]);

            const yMaxAbs = d3.max(allY, y => Math.abs(y)) || 0;

            gAxes.selectAll("*").remove();
            gAxes.append("g")
                .attr("transform", `translate(0,${innerH})`)
                .call(d3.axisBottom(xScale).ticks(6))
                .selectAll("text");
            gAxes.append("g")
                .call((() => {
                    const yAxis = d3.axisLeft(yScale).ticks(6);
                    if (yMaxAbs >= 1e5) yAxis.tickFormat(d3.format(".2e"));
                    return yAxis;
                })())
                .selectAll("text");

            const line = d3.line()
                .defined(d => Number.isFinite(d))
                .x((d, i) => xScale(curves.xGrid[i]))
                .y(d => yScale(d));

            // truth
            gTruth
                .attr("fill", "none")
                .attr("stroke", COLORS.truth)
                .attr("stroke-width", 2)
                .attr("d", line(displayTruth));

            // method lines
            const lineSel = gLines.selectAll("path.method").data(visibleMethods, d => d.key);

            lineSel.enter()
                .append("path")
                .attr("class", "method")
                .attr("fill", "none")
                .attr("stroke", d => COLORS[d.key])
                .attr("stroke-width", 2)
                .attr("opacity", 0.9)
                .merge(lineSel)
                .attr("stroke", d => COLORS[d.key])
                .attr("d", d => line(displayMethods[d.key]));

            lineSel.exit().remove();

            // points
            const pts = xs.map((x, i) => ({ x, y: ys[i] }));
            const pSel = gPoints.selectAll("circle.pt").data(pts);
            pSel.enter()
                .append("circle")
                .attr("class", "pt")
                .attr("r", 3)
                .attr("fill", COLORS.points)
                .attr("opacity", 0.9)
                .merge(pSel)
                .attr("cx", d => xScale(d.x))
                .attr("cy", d => yScale(d.y));
            pSel.exit().remove();

            // legend
            const legendData = [{ key: "truth", label: "True function", color: COLORS.truth }]
                .concat(visibleMethods.map(m => ({ key: m.key, label: m.label, color: COLORS[m.key] })));

            const legend = gMain.selectAll("g.legend").data([null]);
            const legendEnter = legend.enter().append("g").attr("class", "legend");
            legendEnter.merge(legend)
                .attr("transform", `translate(${L.margin.left + 8},${L.margin.top + 8})`);

            const leg = gMain.select("g.legend");
            const items = leg.selectAll("g.item").data(legendData, d => d.key);
            const itemsEnter = items.enter().append("g").attr("class", "item");
            itemsEnter.append("line").attr("x1", 0).attr("x2", 18).attr("y1", 0).attr("y2", 0).attr("stroke-width", 3);
            itemsEnter.append("text").attr("class", "legend-text").attr("x", 24).attr("y", 4);

            itemsEnter.merge(items)
                .attr("transform", (_, i) => `translate(0,${i * 16})`);

            itemsEnter.merge(items).select("line")
                .attr("stroke", d => d.color)
                .attr("opacity", d => (d.key === "truth" ? 1 : 0.9));

            itemsEnter.merge(items).select("text")
                .text(d => d.label);

            items.exit().remove();

            // warning label (keeps UX clear when GD goes unstable)
            const warnMsg = (() => {
                if (!(anyVisibleNonFinite || anyVisibleClipped)) return null;
                const clippedKeys = visibleMethods.filter(m => methodFlags[m.key]?.anyClipped).map(m => m.label);
                const nonFiniteKeys = visibleMethods.filter(m => methodFlags[m.key]?.anyNonFinite).map(m => m.label);
                const parts = [];
                if (nonFiniteKeys.length) parts.push(`non-finite: ${nonFiniteKeys.join(", ")}`);
                if (clippedKeys.length) parts.push(`clipped: ${clippedKeys.join(", ")}`);
                return `Display note: ${parts.join(" · ")}`;
            })();

            const warnSel = gMain.selectAll("text.plot-warning").data(warnMsg ? [warnMsg] : []);
            warnSel.enter()
                .append("text")
                .attr("class", "plot-warning")
                .merge(warnSel)
                .attr("x", L.margin.left + 8)
                .attr("y", L.height - 10)
                .text(d => d);
            warnSel.exit().remove();
        }

        function rerender() {
            readControls();
            fitAll();
            const curves = computeCurves();
            const metrics = computeMetrics();
            renderMainPlot(curves);
            renderMetrics(metrics);
            renderCoefs();
            renderBvSection();

            if (window.MathJax) {
                try {
                    if (typeof window.MathJax.typesetPromise === "function") {
                        window.MathJax.typesetPromise([el.metricsTable]);
                    } else if (typeof window.MathJax.typeset === "function") {
                        window.MathJax.typeset([el.metricsTable]);
                    }
                } catch (_) {
                    /* ignore */
                }
            }
        }

        // event wiring
        const rerenderOnInput = [
            el.trueFn,
            el.nTrain,
            el.noise,
            el.seed,
            el.degree,
            el.lambda,
            el.lr,
            el.iters,
            el.lassoIters,
            el.showOLS,
            el.showRidge,
            el.showLasso,
            el.showL2GD,
            el.showWD,
            el.selectedMethod,
        ];

        for (const c of rerenderOnInput) {
            c.addEventListener("input", () => {
                // Data controls: regenerate only when these change.
                if (c === el.trueFn || c === el.nTrain || c === el.noise || c === el.seed) {
                    regenData();
                }
                rerender();
            });
        }

        if (bv.enabled && el.bvRecompute) {
            el.bvRecompute.addEventListener("click", () => {
                if (bv.computing) return;
                // Ensure state is current before computing.
                readControls();

                // Invalidate and compute.
                bv.cached = null;
                bv.dirty = true;
                setBvBusy(true);
                setBvStatus("Computing sweep… (this can take a few seconds)");

                // Yield a frame so the status/button update paints before heavy work.
                setTimeout(() => {
                    try {
                        bv.cached = computeBvAll();
                        bv.dirty = false;
                    } catch (e) {
                        bv.cached = null;
                        bv.dirty = true;
                        setBvStatus("Sweep failed to compute (see console)." );
                        try { console.error(e); } catch (_) { /* ignore */ }
                    } finally {
                        setBvBusy(false);
                        rerender();
                    }
                }, 0);
            });
        }

        window.addEventListener("resize", () => rerender());

        // initial
        updateLambda();
        regenData();
        if (bv.enabled) {
            setBvBusy(false);
            setBvStatus("Press “Recompute sweep” to calculate.");
            renderBvEmpty("Press “Recompute sweep”");
        }
        rerender();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
