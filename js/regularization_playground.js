/* regularization_playground.js
   Linear regression regularization playground: OLS vs Ridge vs Lasso vs L2 penalty (GD) vs Weight decay (GD)
*/

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

    function mean(arr) {
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }

    function mse(yTrue, yPred) {
        let s = 0;
        for (let i = 0; i < yTrue.length; i++) {
            const d = yTrue[i] - yPred[i];
            s += d * d;
        }
        return s / yTrue.length;
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

    function matVec(A, x) {
        const n = A.length;
        const y = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            let s = 0;
            for (let j = 0; j < x.length; j++) s += A[i][j] * x[j];
            y[i] = s;
        }
        return y;
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

    function fitLassoCoordinateDescent(X, y, lambda, iters) {
        // Objective: (1/N)||Xw - y||^2 + lambda * ||w||_1, do not penalize intercept.
        const N = X.length;
        const P = X[0].length;

        const w = new Array(P).fill(0);
        const yhat = new Array(N).fill(0);

        // precompute column norms (1/N)*sum x_ij^2
        const colNorm = new Array(P).fill(0);
        for (let j = 0; j < P; j++) {
            let s = 0;
            for (let i = 0; i < N; i++) s += X[i][j] * X[i][j];
            colNorm[j] = s / N;
        }

        for (let iter = 0; iter < iters; iter++) {
            // update intercept separately (no penalty)
            {
                let rMean = 0;
                for (let i = 0; i < N; i++) rMean += (y[i] - yhat[i]);
                rMean /= N;
                const delta = rMean; // since x0=1
                w[0] += delta;
                for (let i = 0; i < N; i++) yhat[i] += delta;
            }

            for (let j = 1; j < P; j++) {
                const wOld = w[j];
                // compute rho = (1/N) * sum x_ij * (y_i - (yhat_i - x_ij*w_j))
                let rho = 0;
                for (let i = 0; i < N; i++) {
                    const xij = X[i][j];
                    const r = y[i] - (yhat[i] - xij * wOld);
                    rho += xij * r;
                }
                rho /= N;

                const wNew = softThreshold(rho, lambda) / (colNorm[j] || 1);
                const delta = wNew - wOld;
                if (delta !== 0) {
                    w[j] = wNew;
                    for (let i = 0; i < N; i++) yhat[i] += X[i][j] * delta;
                }
            }
        }

        return w;
    }

    function fitL2PenaltyGD(X, y, lambda, lr, iters) {
        // Minimize (1/N)||Xw-y||^2 + lambda||w||^2 (exclude intercept)
        const N = X.length;
        const P = X[0].length;
        const w = new Array(P).fill(0);

        for (let t = 0; t < iters; t++) {
            const yhat = predict(X, w);
            const grad = new Array(P).fill(0);

            for (let i = 0; i < N; i++) {
                const e = yhat[i] - y[i];
                for (let j = 0; j < P; j++) grad[j] += X[i][j] * e;
            }
            const scale = 2 / N;
            for (let j = 0; j < P; j++) grad[j] *= scale;

            // add L2 term (exclude intercept)
            for (let j = 1; j < P; j++) grad[j] += 2 * lambda * w[j];

            for (let j = 0; j < P; j++) w[j] -= lr * grad[j];
        }

        return w;
    }

    function fitWeightDecayGD(X, y, lambda, lr, iters) {
        // Decoupled weight decay update: w <- (1-lr*lambda)w - lr * grad(MSE)
        const N = X.length;
        const P = X[0].length;
        const w = new Array(P).fill(0);

        for (let t = 0; t < iters; t++) {
            const yhat = predict(X, w);
            const grad = new Array(P).fill(0);

            for (let i = 0; i < N; i++) {
                const e = yhat[i] - y[i];
                for (let j = 0; j < P; j++) grad[j] += X[i][j] * e;
            }
            const scale = 2 / N;
            for (let j = 0; j < P; j++) grad[j] *= scale;

            // gradient step on MSE
            for (let j = 0; j < P; j++) w[j] -= lr * grad[j];

            // decoupled shrink (exclude intercept)
            const decay = 1 - lr * lambda;
            for (let j = 1; j < P; j++) w[j] *= decay;
        }

        return w;
    }

    function trueFunction(name, x) {
        if (name === "linear") return 0.9 * x + 0.3;
        if (name === "cubic") return 0.15 * x * x * x - 0.2 * x * x + 0.4 * x;
        // sine
        return Math.sin(1.25 * x);
    }

    function generateDataset({ trueFn, nTrain, noise, seed }) {
        const rng = mulberry32(seed);
        const xs = [];
        const ys = [];
        for (let i = 0; i < nTrain; i++) {
            const x = -3 + 6 * rng();
            const y = trueFunction(trueFn, x) + noise * randn(rng);
            xs.push(x);
            ys.push(y);
        }

        // fixed test grid
        const nTest = 200;
        const xt = [];
        const yt = [];
        for (let i = 0; i < nTest; i++) {
            const x = -3 + (6 * i) / (nTest - 1);
            xt.push(x);
            yt.push(trueFunction(trueFn, x));
        }

        return { xs, ys, xt, yt };
    }

    function formatNum(v) {
        if (!Number.isFinite(v)) return "—";
        const av = Math.abs(v);
        if (av >= 1e5) return v.toExponential(2);
        if (av >= 1000) return v.toFixed(0);
        if (av >= 10) return v.toFixed(2);
        return v.toFixed(3);
    }

    function formatNumMax3(v) {
        if (!Number.isFinite(v)) return "—";
        const av = Math.abs(v);
        if (av >= 1e5) return v.toExponential(3);
        // Round to at most 3 decimals and avoid trailing zeros.
        const rounded = Math.round(v * 1000) / 1000;
        return rounded.toLocaleString(undefined, { maximumFractionDigits: 3 });
    }

    function init() {
        const root = document;
        const el = {
            regViz: root.getElementById("regViz"),
            coefViz: root.getElementById("coefViz"),
            metricsTable: root.getElementById("metricsTable"),
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
        };

        if (!el.regViz) return;

        const state = {
            dataset: null,
            fits: {},
            degree: Number(el.degree.value),
            trueFn: el.trueFn.value,
            nTrain: Number(el.nTrain.value),
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

        function currentLambda() {
            state.log10lambda = Number(el.lambda.value);
            state.lambda = Math.pow(10, state.log10lambda);
            el.lambdaValue.textContent = `λ = ${formatNum(state.lambda)} (10^${state.log10lambda.toFixed(2)})`;
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
            currentLambda();
        }

        function regenerateDataset() {
            readControls();
            state.dataset = generateDataset({
                trueFn: state.trueFn,
                nTrain: state.nTrain,
                noise: state.noise,
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
            fits.lasso = fitLassoCoordinateDescent(Xs, ys, lambda, state.lassoIters);
            fits.l2gd = fitL2PenaltyGD(Xs, ys, lambda, state.lr, state.iters);
            fits.wd = fitWeightDecayGD(Xs, ys, lambda, state.lr, state.iters);

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
            const margin = { top: 16, right: 16, bottom: 42, left: 52 };

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
                .data(["Method", "Train MSE", "Test MSE", "||w||2", "#nonzero"])
                .enter()
                .append("th")
                .text(d => d);

            const tbody = table.append("tbody");
            const tr = tbody.selectAll("tr")
                .data(rows)
                .enter()
                .append("tr")
                .attr("class", d => (d.key === selected ? "selected" : null));

            tr.append("td").text(d => d.label);
            tr.append("td").text(d => formatNumMax3(d.train));
            tr.append("td").text(d => formatNumMax3(d.test));
            tr.append("td").text(d => formatNumMax3(d.l2));
            tr.append("td").text(d => String(d.nnz));

            el.metricsTable.innerHTML = "";
            el.metricsTable.appendChild(table.node());
        }

        function renderCoefficients() {
            const key = el.selectedMethod.value;
            const w = state.fits[key];
            const degree = state.degree;

            if (!w || !Array.isArray(w) || w.length === 0) {
                el.coeffInfo.textContent = `degree=${degree}, coefficients unavailable`;
                coefG.selectAll("*").remove();
                return;
            }

            const hasNonFinite = w.some(v => !Number.isFinite(v));
            if (hasNonFinite) {
                el.coeffInfo.textContent = `degree=${degree}, coefficients diverged (non-finite values)`;
                coefG.selectAll("*").remove();
                return;
            }

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

            // info line
            el.coeffInfo.textContent = `degree=${degree}, ||w||2=${formatNumMax3(l2Norm(w, 1))}, nonzero=${countNonzero(w, 1e-6, 1)}${clipped ? ", (coef bars clipped for display)" : ""}`;

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
                .selectAll("text")
                .style("font-size", "10px")
                .style("fill", "rgba(255,255,255,0.75)");

            coefG.append("g").attr("class", "coef-axis")
                .call((() => {
                    const yAxis = d3.axisLeft(y).ticks(5);
                    if (maxAbs >= 1e5) yAxis.tickFormat(d3.format(".2e"));
                    return yAxis;
                })())
                .selectAll("text")
                .style("font-size", "10px")
                .style("fill", "rgba(255,255,255,0.75)");

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

        function renderPlot(curves) {
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
                .selectAll("text")
                .style("fill", "rgba(255,255,255,0.75)");
            gAxes.append("g")
                .call((() => {
                    const yAxis = d3.axisLeft(yScale).ticks(6);
                    if (yMaxAbs >= 1e5) yAxis.tickFormat(d3.format(".2e"));
                    return yAxis;
                })())
                .selectAll("text")
                .style("fill", "rgba(255,255,255,0.75)");

            gAxes.selectAll(".domain, .tick line")
                .attr("stroke", "rgba(255,255,255,0.22)");

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
            itemsEnter.append("text").attr("x", 24).attr("y", 4).style("font-size", "12px");

            itemsEnter.merge(items)
                .attr("transform", (_, i) => `translate(0,${i * 16})`);

            itemsEnter.merge(items).select("line")
                .attr("stroke", d => d.color)
                .attr("opacity", d => (d.key === "truth" ? 1 : 0.9));

            itemsEnter.merge(items).select("text")
                .text(d => d.label)
                .style("fill", "rgba(255,255,255,0.85)");

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
                .style("font-size", "12px")
                .style("fill", "rgba(255,255,255,0.75)")
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
            renderPlot(curves);
            renderMetrics(metrics);
            renderCoefficients();

            if (window.MathJax && window.MathJax.typeset) {
                // Keep it cheap: only typeset on first load / major rerenders.
                try { window.MathJax.typeset(); } catch (_) { /* ignore */ }
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
                    regenerateDataset();
                }
                rerender();
            });
        }

        window.addEventListener("resize", () => rerender());

        // initial
        currentLambda();
        regenerateDataset();
        rerender();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
