import { knnGraph, bottomKSymmetricEig } from '../linalg.js';
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

export const LAPLACIAN = {
  id: 'laplacian',
  label: 'Laplacian Eigenmaps',
  params: [
    { name: 'k', type: 'int', default: 10, min: 2, max: 50 },
    { name: 'sigma', type: 'float', default: 1.0, min: 0.1, max: 10 },
  ],
  presentSubSteps: ['0', '2', '3', '4', '5', '6'],
  pseudocode: [
    { id: 'lap-knn', title: '1. Build kNN graph', steps: ['2'],
      lines: ['neighbours_i = k nearest by Euclidean distance'] },
    { id: 'lap-W', title: '2. Heat-kernel affinity W', steps: ['3'],
      lines: ['W_{ij} = exp(-||x_i - x_j||^2 / (2 sigma^2)) for kNN edges, else 0'] },
    { id: 'lap-L', title: '3. Graph Laplacian L = D - W', steps: ['4'],
      lines: ['D_{ii} = sum_j W_{ij}', 'L = D - W'] },
    { id: 'lap-eig', title: '4. Smallest non-trivial eigenvectors', steps: ['5'],
      lines: ['L v_k = lambda_k v_k (skip lambda_0 = 0)'] },
    { id: 'lap-embed', title: '5. Form 2D embedding', steps: ['6'],
      lines: ['Y = [v_1, v_2]'] },
  ],
  run(dataset, params) {
    const X = dataset.X;
    const t = dataset.t;
    const N = X.length / 3;
    const k = Math.max(2, Math.min(params.k || 10, N - 1));
    const sigma = params.sigma || 1.0;
    const steps = new Map();
    const presentSubSteps = ['0', '2', '3', '4', '5', '6'];
    const pending = new Set(['2', '3', '4', '5', '6']);
    let cancelled = false;
    const samples = sampleIndices(N);

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: dataset.colors || null,
      vizKind: 'point_cloud',
      label: 'Raw data',
      ifw: { intuition: '<p>Laplacian Eigenmaps preserves local proximity. It uses the kNN graph plus a heat kernel to weight nearby points more strongly.</p>', formula: null, worked: null },
    });

    function start(onProgress) {
      const mem = {};
      const tasks = [
        () => {
          const { adj, edges } = knnGraph(X, k);
          mem.adj = adj;
          mem.edges = edges;
          const sampleI = samples[0];
          const neighbours = adj[sampleI].slice(0, k).map(([j, w]) => [j, w.toFixed(3)]);
          const inputBlock = 'sample point i = ' + sampleI + ', x_i = ' + formatVec3(rowOf(X, sampleI)) + '\nN = ' + N + ', k = ' + k;
          const outputBlock = 'neighbours of point ' + sampleI + ':\n' +
            formatTable(['j', '||x_j - x_i||'], neighbours) + '\n\ntotal undirected edges = ' + edges.length;
          steps.set('2', {
            points: X.slice(), t, edges, colors: dataset.colors || null,
            vizKind: 'knn_graph',
            label: 'kNN graph (k = ' + k + ')',
            ifw: {
              intuition: '<p>The kNN graph captures local neighbourhood structure that the heat kernel will weight in the next step.</p>',
              formula: '$$\\mathcal{N}_i = \\{ j : \\|x_j - x_i\\| \\text{ among the } k \\text{ smallest}\\}$$',
              worked: workedSections(inputBlock, '$$w^{\\text{edge}}_{ij} = \\| x_j - x_i \\|$$', outputBlock),
            },
          });
          pending.delete('2');
          if (onProgress) onProgress('2');
        },
        () => {
          const adj = mem.adj;
          const W = new Float64Array(N * N);
          const sig2 = 2 * sigma * sigma;
          for (let i = 0; i < N; i++) {
            for (const [j, dist] of adj[i]) {
              const w = Math.exp(-dist * dist / sig2);
              W[i * N + j] = w;
              W[j * N + i] = w;
            }
          }
          mem.W = W;
          const sampleI = samples[0];
          const wRow = adj[sampleI].slice(0, Math.min(5, k)).map(([j, dist]) => [j, dist.toFixed(3), Math.exp(-dist * dist / sig2).toFixed(4)]);
          const inputBlock = 'sample point i = ' + sampleI + ', sigma = ' + sigma + '\nkNN distances visible above.';
          const outputBlock = 'W_ij for first ' + Math.min(5, k) + ' neighbours of ' + sampleI + ':\n' +
            formatTable(['j', '||x_j - x_i||', 'W_ij'], wRow);
          steps.set('3', {
            points: X.slice(), t, edges: mem.edges, colors: dataset.colors || null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'graph_thumb', label: 'kNN graph', data: { points: X.slice(), edges: mem.edges } },
              { kind: 'heatmap', label: 'W', data: { matrix: W, N } },
            ],
            paneOpLabels: ['W_{ij} = exp(-||x_i - x_j||^2 / (2 sigma^2))'],
            label: 'Heat-kernel affinity W',
            ifw: {
              intuition: '<p>Closer points get larger affinity weights via the Gaussian heat kernel; non-kNN edges are exactly zero so W stays sparse.</p>',
              formula: '$$W_{ij} = \\exp\\!\\left(-\\frac{\\| x_i - x_j \\|^2}{2 \\sigma^2}\\right) \\text{ for kNN edges, else } 0$$',
              worked: workedSections(inputBlock, '$$W_{ij} = \\exp\\!\\left(-\\frac{\\| x_i - x_j \\|^2}{2 \\sigma^2}\\right)$$', outputBlock),
            },
          });
          pending.delete('3');
          if (onProgress) onProgress('3');
        },
        () => {
          const W = mem.W;
          const D = new Float64Array(N);
          for (let i = 0; i < N; i++) {
            let s = 0;
            for (let j = 0; j < N; j++) s += W[i * N + j];
            D[i] = s;
          }
          const Dmat = new Float64Array(N * N);
          for (let i = 0; i < N; i++) Dmat[i * N + i] = D[i];
          const L = new Float64Array(N * N);
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
              L[i * N + j] = (i === j ? D[i] : 0) - W[i * N + j];
            }
          }
          mem.L = L;
          mem.D = D;
          const inputBlock = 'W (4 of N=' + N + ' rows):\n' + (function () {
            const ex = [];
            for (let r = 0; r < 4 && r < N; r++) {
              const row = [];
              for (let c = 0; c < 4 && c < N; c++) row.push(W[r * N + c]);
              ex.push(row);
            }
            return formatMatrix(ex, { digits: 3 });
          })();
          const outputBlock = 'D_ii (first 4): ' + Array.from(D.slice(0, 4)).map(v => v.toFixed(3)).join(', ') +
            '\n\nL (4 of N=' + N + ' rows):\n' + (function () {
            const ex = [];
            for (let r = 0; r < 4 && r < N; r++) {
              const row = [];
              for (let c = 0; c < 4 && c < N; c++) row.push(L[r * N + c]);
              ex.push(row);
            }
            return formatMatrix(ex, { digits: 3 });
          })();
          steps.set('4', {
            points: X.slice(), t, edges: mem.edges, colors: dataset.colors || null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'heatmap', label: 'W', data: { matrix: W, N } },
              { kind: 'heatmap', label: 'D', data: { matrix: Dmat, N } },
              { kind: 'heatmap', label: 'L = D - W', data: { matrix: L, N } },
            ],
            paneOpLabels: ['row sums = D', 'D - W = L'],
            label: 'Graph Laplacian',
            ifw: {
              intuition: '<p>L combines the degree of each node with its outgoing affinities. The smallest non-trivial eigenvectors of L are smooth on the graph and give the embedding coordinates.</p>',
              formula: '$$L = D - W,\\quad D_{ii} = \\sum_j W_{ij}$$',
              worked: workedSections(inputBlock, '$$L_{ij} = D_{ii} \\delta_{ij} - W_{ij}$$', outputBlock),
            },
          });
          pending.delete('4');
          if (onProgress) onProgress('4');
        },
        () => {
          const { lambda, vectors } = bottomKSymmetricEig(mem.L, N, 8, { skipFirst: 1 });
          mem.lambda = lambda;
          mem.vectors = vectors;
          const inputBlock = 'L (4 of N=' + N + ' rows): see step 4 output.';
          const outputBlock = 'bottom non-trivial eigenvalues: ' + Array.from(lambda).map((v, i) => 'lambda_' + (i + 1) + ' = ' + v.toFixed(3)).join(', ') +
            '\n\nv_1 (first 6 entries): [' + Array.from(vectors[0].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']' +
            '\nv_2 (first 6 entries): [' + Array.from(vectors[1].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']';
          steps.set('5', {
            points: X.slice(), t, edges: mem.edges, colors: dataset.colors || null,
            vizKind: 'spectral',
            algoId: 'laplacian',
            v1Values: vectors[0],
            topEigvals: lambda,
            topEigvecs: vectors,
            label: 'Smallest non-trivial eigenvectors',
            ifw: {
              intuition: '<p>Skipping the trivial zero eigenvalue, the next smallest eigenvectors are smooth functions on the graph. Each one becomes one coordinate of the embedding.</p>',
              formula: '$$L\\, v_k = \\lambda_k\\, v_k$$',
              worked: workedSections(inputBlock, '$$L\\, v_k = \\lambda_k\\, v_k,\\quad k = 1, 2$$', outputBlock),
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
            label: 'Laplacian Eigenmaps embedding',
            ifw: {
              intuition: '<p>The 2D coordinates are the values of v_1 and v_2 at each point. Nearby points on the manifold end up nearby in the embedding because L penalises differences across heavy edges.</p>',
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
        try { tasks[i++](); } catch (e) { console.error('Laplacian pipeline error:', e); return; }
        if (i < tasks.length) setTimeout(tick, 0);
      };
      setTimeout(tick, 0);
    }

    function cancel() { cancelled = true; }

    return { steps, presentSubSteps, pending, start, cancel };
  },
};
