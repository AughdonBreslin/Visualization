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

    function mulberry32(seed) {
        let t = seed >>> 0;
        return function () {
            t += 0x6D2B79F5;
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
        if (av >= 1000) return v.toFixed(0);
        if (av >= 10) return v.toFixed(2);
        return v.toFixed(3);
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
            regen: root.getElementById("regen"),
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
            tr.append("td").text(d => formatNum(d.train));
            tr.append("td").text(d => formatNum(d.test));
            tr.append("td").text(d => formatNum(d.l2));
            tr.append("td").text(d => String(d.nnz));

            el.metricsTable.innerHTML = "";
            el.metricsTable.appendChild(table.node());
        }

        function renderCoefficients() {
            const key = el.selectedMethod.value;
            const w = state.fits[key];
            const degree = state.degree;

            const items = [];
            for (let j = 0; j < w.length; j++) {
                items.push({
                    name: j === 0 ? "bias" : `x^${j}`,
                    idx: j,
                    value: w[j],
                    abs: Math.abs(w[j]),
                });
            }

            // info line
            el.coeffInfo.textContent = `degree=${degree}, ||w||2=${formatNum(l2Norm(w, 1))}, nonzero=${countNonzero(w, 1e-6, 1)}`;

            const L = coefLayout();
            const innerW = L.innerW;
            const innerH = L.innerH;

            const x = d3.scaleBand()
                .domain(items.map(d => d.name))
                .range([0, innerW])
                .padding(0.2);

            const maxAbs = d3.max(items, d => d.abs) || 1;
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
                .call(d3.axisLeft(y).ticks(5))
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
                .attr("y", d => y(d.abs))
                .attr("height", d => innerH - y(d.abs))
                .attr("fill", COLORS[key] || "rgba(255,255,255,0.6)");

            bars.exit().remove();
        }

        function renderPlot(curves) {
            const L = layout();
            const innerW = L.innerW;
            const innerH = L.innerH;

            const { xs, ys } = state.dataset;

            const allY = [
                ...ys,
                ...curves.yTrue,
                ...Object.values(curves.methods).flat(),
            ].filter(Number.isFinite);

            const xScale = d3.scaleLinear().domain([-3, 3]).range([0, innerW]);
            const yScale = d3.scaleLinear()
                .domain(d3.extent(allY))
                .nice()
                .range([innerH, 0]);

            gAxes.selectAll("*").remove();
            gAxes.append("g")
                .attr("transform", `translate(0,${innerH})`)
                .call(d3.axisBottom(xScale).ticks(6))
                .selectAll("text")
                .style("fill", "rgba(255,255,255,0.75)");
            gAxes.append("g")
                .call(d3.axisLeft(yScale).ticks(6))
                .selectAll("text")
                .style("fill", "rgba(255,255,255,0.75)");

            gAxes.selectAll(".domain, .tick line")
                .attr("stroke", "rgba(255,255,255,0.22)");

            const line = d3.line()
                .x((d, i) => xScale(curves.xGrid[i]))
                .y(d => yScale(d));

            // truth
            gTruth
                .attr("fill", "none")
                .attr("stroke", COLORS.truth)
                .attr("stroke-width", 2)
                .attr("d", line(curves.yTrue));

            // method lines
            const visibleMethods = METHODS.filter(m => methodVisible(m.key));
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
                .attr("d", d => line(curves.methods[d.key]));

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
        el.regen.addEventListener("click", () => {
            regenerateDataset();
            rerender();
        });

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
