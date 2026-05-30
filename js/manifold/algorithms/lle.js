import { knnGraph, bottomKSymmetricEig, solveLinearSystem } from '../linalg.js';
import { formatVec3, formatMatrix, formatTable } from '../format.js';

function sampleIndices(N) {
  return [Math.floor(N * 0.2), Math.floor(N * 0.5), Math.floor(N * 0.8)];
}

function workedSections(input, formula, output) {
  return (
    '<div class="ifw-worked-section">' +
      '<div class="ifw-worked-label">Input (from previous step)</div>' +
      '<div class="ifw-worked-body">' + input + '</div>' +
    '</div>' +
    '<div class="ifw-worked-section">' +
      '<div class="ifw-worked-label">Formula</div>' +
      '<div class="ifw-worked-body math">' + formula + '</div>' +
    '</div>' +
    '<div class="ifw-worked-section">' +
      '<div class="ifw-worked-label">Output (after this step)</div>' +
      '<div class="ifw-worked-body">' + output + '</div>' +
    '</div>'
  );
}

function rowOf(X, i) {
  return [X[i * 3], X[i * 3 + 1], X[i * 3 + 2]];
}

export const LLE = {
  id: 'lle',
  label: 'LLE',
  params: [
    { name: 'k', type: 'int', default: 10, min: 2, max: 50 },
    { name: 'reg', type: 'float', default: 1e-3, min: 0, max: 0.1 },
  ],
  presentSubSteps: ['0', '2', '3', '5', '6'],
  pseudocode: [
    { id: 'lle-knn', title: '1. Build kNN graph', steps: ['2'],
      lines: ['$\\mathcal{N}_i = k$ points with smallest $\\| x_j - x_i \\|$'] },
    { id: 'lle-W', title: '2. Reconstruction weights W', steps: ['3'],
      lines: ['for each $i$: minimise $\\bigl\\| x_i - \\sum_j w_j x_{n_j} \\bigr\\|^2$',
              'subject to $\\sum_j w_j = 1$',
              'store $W_{i, n_j} = w_j$'] },
    { id: 'lle-eig', title: '3. Smallest non-trivial eigenvectors of M', steps: ['5'],
      lines: ['$M = (I - W)^{\\top} (I - W)$', '$M v_k = \\lambda_k v_k$ (skip $\\lambda_0 = 0$)'] },
    { id: 'lle-embed', title: '4. Form 2D embedding', steps: ['6'],
      lines: ['$Y = [\\, v_1 \\;\\; v_2 \\,]$'] },
  ],
  run(dataset, params) {
    const X = dataset.X;
    const t = dataset.t;
    const N = X.length / 3;
    const k = Math.max(2, Math.min(params.k || 10, N - 1));
    const reg = params.reg !== undefined ? params.reg : 1e-3;
    const steps = new Map();
    const presentSubSteps = ['0', '2', '3', '5', '6'];
    const pending = new Set(['2', '3', '5', '6']);
    let cancelled = false;
    const samples = sampleIndices(N);

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: dataset.colors || null,
      vizKind: 'point_cloud',
      label: 'Raw data',
      ifw: {
        intuition: '<p>LLE reconstructs each point as a weighted sum of its k nearest neighbours, then finds a low-dimensional embedding that preserves those same weights.</p>',
        formula: null, worked: null,
      },
    });

    function start(onProgress) {
      const mem = {};
      const tasks = [
        () => {
          const t0 = Date.now();
          const { adj, edges } = knnGraph(X, k);
          mem.adj = adj;
          mem.edges = edges;
          const sampleI = samples[0];
          const neighbours = adj[sampleI].slice(0, k).map(([j, w]) => [j, w.toFixed(3)]);
          const inputBlock = 'sample point i = ' + sampleI + ', x_i = ' + formatVec3(rowOf(X, sampleI)) + '\nN = ' + N + ', k = ' + k;
          const outputBlock = 'neighbours of point ' + sampleI + ':\n' +
            formatTable(['j', '||x_j - x_i||'], neighbours);
          steps.set('2', {
            points: X.slice(), t, edges, colors: dataset.colors || null,
            vizKind: 'knn_graph',
            label: 'kNN graph (k = ' + k + ')',
            ifw: {
              intuition: '<p>LLE assumes each point lies on a locally linear patch defined by its k nearest neighbours. The kNN graph identifies those neighbourhoods.</p>',
              formula: '$$\\mathcal{N}_i = \\{ j : \\|x_j - x_i\\| \\text{ among the } k \\text{ smallest}\\}$$',
              worked: workedSections(inputBlock, '$$w^{\\text{edge}}_{ij} = \\| x_j - x_i \\|$$', outputBlock),
            },
          });
          setTimeout(() => {
            if (cancelled) return;
            pending.delete('2');
            if (onProgress) onProgress('2');
          }, Math.max(0, 5000 - (Date.now() - t0)));
        },
        () => {
          const adj = mem.adj;
          const W = new Float64Array(N * N);
          for (let i = 0; i < N; i++) {
            const neighbours = adj[i].slice(0, k).map(([j]) => j);
            const G = [];
            for (let a = 0; a < neighbours.length; a++) {
              const row = new Array(neighbours.length);
              const na = neighbours[a];
              const ax = X[na * 3] - X[i * 3], ay = X[na * 3 + 1] - X[i * 3 + 1], az = X[na * 3 + 2] - X[i * 3 + 2];
              for (let b = 0; b < neighbours.length; b++) {
                const nb = neighbours[b];
                const bx = X[nb * 3] - X[i * 3], by = X[nb * 3 + 1] - X[i * 3 + 1], bz = X[nb * 3 + 2] - X[i * 3 + 2];
                row[b] = ax * bx + ay * by + az * bz;
              }
              G.push(row);
            }
            let trace = 0;
            for (let a = 0; a < G.length; a++) trace += G[a][a];
            const lambda = reg * Math.max(trace, 1e-12) / G.length;
            for (let a = 0; a < G.length; a++) G[a][a] += lambda;
            const b = new Array(neighbours.length).fill(1);
            const w = solveLinearSystem(G, b);
            if (!w) continue;
            let sum = 0;
            for (let a = 0; a < w.length; a++) sum += w[a];
            if (Math.abs(sum) < 1e-12) continue;
            for (let a = 0; a < w.length; a++) w[a] /= sum;
            for (let a = 0; a < neighbours.length; a++) {
              W[i * N + neighbours[a]] = w[a];
            }
          }
          mem.W = W;
          const sampleI = samples[0];
          const wRow = [];
          for (let j = 0; j < N; j++) {
            const v = W[sampleI * N + j];
            if (v !== 0) wRow.push([j, v.toFixed(4)]);
          }
          const inputBlock = 'sample point i = ' + sampleI + ', k = ' + k + ', reg = ' + reg + '\nlocal Gram matrix G has size k x k.';
          const outputBlock = 'W_ij for the k = ' + wRow.length + ' neighbours of point ' + sampleI + ':\n' +
            formatTable(['j', 'W_ij'], wRow.slice(0, 8)) +
            (wRow.length > 8 ? '\n... (' + (wRow.length - 8) + ' more)' : '');
          steps.set('3', {
            points: X.slice(), t, edges: mem.edges, colors: dataset.colors || null,
            vizKind: 'weighted_knn',
            W,
            k,
            selectedPoint: sampleI,
            label: 'Reconstruction weights',
            ifw: {
              intuition: '<p>Each point is described as a weighted combination of its k neighbours. The weights are solved by a small linear system per point so that the linear combination best reconstructs the point, with the weights normalised to sum to 1.</p>',
              formula: '$$\\min_{w_i}\\ \\bigl\\| x_i - \\sum_{j \\in \\mathcal{N}_i} w_{ij}\\, x_j \\bigr\\|^2,\\ \\sum_j w_{ij} = 1$$',
              worked: workedSections(inputBlock,
                '$$G_{ab} = (x_{n_a} - x_i) \\cdot (x_{n_b} - x_i),\\ G\\,w = \\mathbf{1},\\ w \\leftarrow w / \\sum w$$',
                outputBlock),
            },
          });
          pending.delete('3');
          if (onProgress) onProgress('3');
        },
        () => {
          const W = mem.W;
          const IW = new Float64Array(N * N);
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
              IW[i * N + j] = (i === j ? 1 : 0) - W[i * N + j];
            }
          }
          const M = new Float64Array(N * N);
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
              let s = 0;
              for (let p = 0; p < N; p++) s += IW[p * N + i] * IW[p * N + j];
              M[i * N + j] = s;
            }
          }
          mem.M = M;
          const { lambda, vectors } = bottomKSymmetricEig(M, N, 8, { skipFirst: 1 });
          mem.lambda = lambda;
          mem.vectors = vectors;
          const excerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(M[r * N + c]);
            excerpt.push(row);
          }
          const inputBlock = 'M (4 of N=' + N + ' rows):\n' + formatMatrix(excerpt, { digits: 3 });
          const outputBlock = 'bottom non-trivial eigenvalues: ' + Array.from(lambda).map((v, i) => 'lambda_' + (i + 1) + ' = ' + v.toFixed(4)).join(', ') +
            '\n\nv_1 (first 6 entries): [' + Array.from(vectors[0].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']' +
            '\nv_2 (first 6 entries): [' + Array.from(vectors[1].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']';
          steps.set('5', {
            points: X.slice(), t, edges: mem.edges, colors: dataset.colors || null,
            vizKind: 'spectral',
            algoId: 'lle',
            v1Values: vectors[0],
            topEigvals: lambda,
            topEigvecs: vectors,
            label: 'Smallest non-trivial eigenvectors of M',
            ifw: {
              intuition: '<p>The smallest non-trivial eigenvectors of M produce coordinates that minimise the embedding cost while keeping the same local reconstruction weights. The trivial zero eigenvalue (constant eigenvector) is dropped.</p>',
              formula: '$$M\\, v_k = \\lambda_k\\, v_k,\\quad M = (I - W)^{\\top}(I - W)$$',
              worked: workedSections(inputBlock, '$$M\\, v_k = \\lambda_k\\, v_k$$', outputBlock),
            },
          });
          pending.delete('5');
          if (onProgress) onProgress('5');
        },
        () => {
          const vectors = mem.vectors;
          const embed2d = new Float64Array(N * 2);
          for (let i = 0; i < N; i++) {
            embed2d[i * 2] = vectors[0][i];
            embed2d[i * 2 + 1] = vectors[1][i];
          }
          const outputRows = samples.map(i => [i,
            '(' + embed2d[i * 2].toFixed(3) + ', ' + embed2d[i * 2 + 1].toFixed(3) + ')']);
          const outputBlock = formatTable(['i', 'y_i'], outputRows);
          steps.set('6', {
            points: X.slice(), t, edges: mem.edges, colors: dataset.colors || null, embed2d,
            vizKind: 'embedding',
            label: 'LLE embedding',
            ifw: {
              intuition: '<p>The 2D coordinates are the values of v_1 and v_2 at each point. LLE does not scale by eigenvalues; the absolute scale is fixed by the cost-function normalisation.</p>',
              formula: '$$y_i = (v_{1,i}, v_{2,i})$$',
              worked: workedSections('v_1 and v_2 from step 5.', '$$y_{i,k} = v_{k,i}$$', outputBlock),
            },
          });
          pending.delete('6');
          if (onProgress) onProgress('6');
        },
      ];

      let i = 0;
      const tick = () => {
        if (cancelled || i >= tasks.length) return;
        try { tasks[i++](); } catch (e) { console.error('LLE pipeline error:', e); return; }
        if (i < tasks.length) setTimeout(tick, 0);
      };
      setTimeout(tick, 0);
    }

    function cancel() { cancelled = true; }

    return { steps, presentSubSteps, pending, start, cancel };
  },
};
