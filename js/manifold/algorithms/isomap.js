import { knnGraph, dijkstraAllPairs, doubleCenterSquared, topKSymmetricEig } from '../linalg.js';
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

export const ISOMAP = {
  id: 'isomap',
  label: 'Isomap',
  params: [{ name: 'k', type: 'int', default: 10, min: 2, max: 50 }],
  presentSubSteps: ['0', '2', '3', '4', '5', '6'],
  pseudocode: [
    { id: 'isomap-knn', title: '1. Build kNN graph', steps: ['2'],
      lines: ['for each i: neighbours_i ← k points with smallest ||x_j − x_i||',
              'edge weight w_{ij} ← ||x_j − x_i||'] },
    { id: 'isomap-geo', title: '2. Compute geodesic distances', steps: ['3'],
      lines: ['for each i: run Dijkstra from i on the kNN graph',
              'D_{ij} ← shortest-path distance in the graph'] },
    { id: 'isomap-dc', title: '3. Double-center the squared distance matrix', steps: ['4'],
      lines: ['B ← −(1/2) H D^2 H,  with H = I − (1/N) 1 1^T'] },
    { id: 'isomap-eig', title: '4. Eigendecompose B', steps: ['5'],
      lines: ['B = V Λ V^T  (take top-2 eigvals/vecs)'] },
    { id: 'isomap-embed', title: '5. Form 2D embedding', steps: ['6'],
      lines: ['Y ← V[:, 0:2] · diag(sqrt(λ_1), sqrt(λ_2))'] },
  ],
  run(dataset, params) {
    const X = dataset.X;
    const t = dataset.t;
    const N = X.length / 3;
    const k = Math.max(2, Math.min(params.k || 10, N - 1));
    const steps = new Map();
    const presentSubSteps = ['0', '2', '3', '4', '5', '6'];
    const pending = new Set(['2', '3', '4', '5', '6']);
    let cancelled = false;
    const samples = sampleIndices(N);

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: null,
      vizKind: 'point_cloud',
      label: 'Raw data',
      ifw: { intuition: '<p>Isomap starts from the raw point cloud and recovers manifold geometry from local neighborhoods.</p>', formula: null, worked: null },
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
            formatTable(['j', '|| x_j − x_i ||'], neighbours) + '\n\ntotal undirected edges = ' + edges.length;
          steps.set('2', {
            points: X.slice(), t, edges, colors: null,
            vizKind: 'knn_graph',
            label: 'kNN graph (k = ' + k + ')',
            ifw: {
              intuition: '<p>Connecting each point to its k nearest Euclidean neighbours approximates the manifold by a graph whose edges follow the local surface.</p>',
              formula: '$$\\mathcal{N}_i = \\{ j : \\|x_j - x_i\\| \\text{ among the } k \\text{ smallest}\\}$$',
              worked: workedSections(inputBlock, '$$w_{ij} = \\| x_j - x_i \\|$$', outputBlock),
            },
          });
          pending.delete('2');
          if (onProgress) onProgress('2');
        },
        () => {
          const D = dijkstraAllPairs(mem.adj, N);
          mem.D = D;
          let connected = true;
          for (let i = 0; i < N * N; i++) if (!Number.isFinite(D[i])) { connected = false; break; }
          const i0 = samples[0], j0 = samples[2];
          const exampleD = D[i0 * N + j0];
          const inputBlock = 'kNN graph with ' + mem.edges.length + ' undirected edges (input from step 2).';
          const excerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(D[r * N + c]);
            excerpt.push(row);
          }
          const outputBlock = 'example shortest path: i = ' + i0 + ', j = ' + j0 +
            '\nD[' + i0 + '][' + j0 + '] = ' + (Number.isFinite(exampleD) ? exampleD.toFixed(3) : '∞') +
            '\n\nD (4 of N=' + N + ' rows):\n' + formatMatrix(excerpt, { digits: 3 }) +
            '\n\ngraph connected: ' + (connected ? 'yes' : 'no');
          steps.set('3', {
            points: X.slice(), t, edges: mem.edges, colors: null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'graph_thumb', label: 'kNN graph', data: { points: X.slice(), edges: mem.edges } },
              { kind: 'graph_thumb_with_path', label: 'one path traced', data: { points: X.slice(), edges: mem.edges, pathEdges: [[i0, samples[1]], [samples[1], j0]] } },
              { kind: 'heatmap', label: 'D (N x N)', data: { matrix: D, N, highlightRow: i0 } },
            ],
            paneOpLabels: ['all-pairs Dijkstra', 'fill matrix'],
            label: 'Geodesic distances',
            ifw: {
              intuition: '<p>Distances along the graph approximate true geodesic distances on the manifold.</p>',
              formula: '$$D_{ij} = \\min_{\\text{path } i \\to j \\text{ on } G} \\sum_e w_e$$',
              worked: workedSections(inputBlock, '$$D_{ij} = \\text{shortest-path}_G(i, j)$$', outputBlock),
            },
          });
          pending.delete('3');
          if (onProgress) onProgress('3');
        },
        () => {
          const B = doubleCenterSquared(mem.D, N);
          mem.B = B;
          const D = mem.D;
          const D2 = new Float64Array(N * N);
          for (let i = 0; i < N * N; i++) D2[i] = Number.isFinite(D[i]) ? D[i] * D[i] : 0;
          const rowMean = new Float64Array(N);
          const colMean = new Float64Array(N);
          let grand = 0;
          for (let i = 0; i < N; i++) {
            let r = 0;
            for (let j = 0; j < N; j++) r += D2[i * N + j];
            rowMean[i] = r / N;
            grand += r;
          }
          for (let j = 0; j < N; j++) {
            let c = 0;
            for (let i = 0; i < N; i++) c += D2[i * N + j];
            colMean[j] = c / N;
          }
          grand /= (N * N);
          const inputExcerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(D2[r * N + c]);
            inputExcerpt.push(row);
          }
          const outExcerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(B[r * N + c]);
            outExcerpt.push(row);
          }
          const inputBlock = 'D² (4 of N=' + N + ' rows):\n' + formatMatrix(inputExcerpt, { digits: 3 }) +
            '\n\nrow means (first 4): ' + Array.from(rowMean.slice(0, 4)).map(v => v.toFixed(3)).join(', ') +
            '\ngrand mean g = ' + grand.toFixed(3);
          const sample = 'example B[1][2] = −1/2 · (' + D2[N + 2].toFixed(3) + ' − ' + rowMean[1].toFixed(3) + ' − ' + colMean[2].toFixed(3) + ' + ' + grand.toFixed(3) + ') = ' + B[N + 2].toFixed(3);
          const outputBlock = sample + '\n\nB (4 of N=' + N + ' rows):\n' + formatMatrix(outExcerpt, { digits: 3 });
          const D2c = new Float64Array(N * N);
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
              D2c[i * N + j] = D2[i * N + j] - rowMean[i] - colMean[j];
            }
          }
          steps.set('4', {
            points: X.slice(), t, edges: mem.edges, colors: null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'heatmap', label: 'D²', data: { matrix: D2, N } },
              { kind: 'heatmap', label: 'D² − μ', data: { matrix: D2c, N } },
              { kind: 'heatmap', label: 'B', data: { matrix: B, N } },
            ],
            paneOpLabels: ['subtract row/col means', '× (−1/2) + grand mean'],
            label: 'Double-centered Gram matrix',
            ifw: {
              intuition: '<p>The geodesic distance matrix tells us how far apart every pair of points is along the manifold, but eigendecomposition needs a different form. Double-centering subtracts the row mean, the column mean, and re-adds the grand mean from every squared distance, then scales by minus one half. The result is the matrix B, whose entries look like inner products between points relative to the centre of the cloud. This matrix is what the next step decomposes to recover embedding coordinates.</p>',
              formula: '$$B = -\\tfrac{1}{2} H D^{(2)} H, \\quad H = I - \\tfrac{1}{N}\\mathbf{1}\\mathbf{1}^{\\top}$$',
              worked: workedSections(inputBlock, '$$B_{ij} = -\\tfrac{1}{2}\\bigl(D^2_{ij} - r_i - c_j + g\\bigr)$$', outputBlock),
            },
          });
          pending.delete('4');
          if (onProgress) onProgress('4');
        },
        () => {
          const { lambda, vectors } = topKSymmetricEig(mem.B, N, 8);
          mem.lambda = lambda;
          mem.vectors = vectors;
          const topEig = lambda;
          const excerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(mem.B[r * N + c]);
            excerpt.push(row);
          }
          const inputBlock = 'B (4 of N=' + N + ' rows):\n' + formatMatrix(excerpt, { digits: 3 });
          const outputBlock = 'top eigenvalues: λ_1 = ' + lambda[0].toFixed(3) + ', λ_2 = ' + lambda[1].toFixed(3) +
            '\n\nv_1 (first 6 entries): [' + Array.from(vectors[0].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']' +
            '\nv_2 (first 6 entries): [' + Array.from(vectors[1].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']';
          steps.set('5', {
            points: X.slice(), t, edges: mem.edges, colors: null,
            vizKind: 'spectral',
            algoId: 'isomap',
            v1Values: vectors[0],
            topEigvals: topEig,
            topEigvecs: vectors,
            label: 'Top-2 eigendecomposition',
            ifw: {
              intuition: '<p>The top eigenvectors of B reveal the dominant geometry of the geodesic distances.</p>',
              formula: '$$B = V \\Lambda V^{\\top}$$',
              worked: workedSections(inputBlock, '$$B\\, v_k = \\lambda_k\\, v_k$$', outputBlock),
            },
          });
          pending.delete('5');
          if (onProgress) onProgress('5');
        },
        () => {
          const lambda = mem.lambda, vectors = mem.vectors;
          const embed2d = new Float64Array(N * 2);
          const s1 = Math.sqrt(Math.max(0, lambda[0]));
          const s2 = Math.sqrt(Math.max(0, lambda[1]));
          for (let i = 0; i < N; i++) {
            embed2d[i * 2] = vectors[0][i] * s1;
            embed2d[i * 2 + 1] = vectors[1][i] * s2;
          }
          const inputBlock = 'λ_1 = ' + lambda[0].toFixed(3) + ', λ_2 = ' + lambda[1].toFixed(3) +
            '\nv_1 (first 3): [' + Array.from(vectors[0].slice(0, 3)).map(v => v.toFixed(3)).join(', ') + ']' +
            '\nv_2 (first 3): [' + Array.from(vectors[1].slice(0, 3)).map(v => v.toFixed(3)).join(', ') + ']';
          const outputRows = samples.map(i => [i,
            '(' + embed2d[i * 2].toFixed(3) + ', ' + embed2d[i * 2 + 1].toFixed(3) + ')']);
          const outputBlock = formatTable(['i', 'y_i'], outputRows);
          steps.set('6', {
            points: X.slice(), t, edges: mem.edges, colors: null, embed2d,
            vizKind: 'embedding',
            label: 'Isomap embedding',
            ifw: {
              intuition: '<p>The 2D coordinates flatten the manifold while preserving geodesic distances as well as possible.</p>',
              formula: '$$y_i = \\big(\\sqrt{\\lambda_1}\\, v_{1,i},\\; \\sqrt{\\lambda_2}\\, v_{2,i}\\big)$$',
              worked: workedSections(inputBlock, '$$y_{i,k} = \\sqrt{\\lambda_k}\\, v_{k,i}$$', outputBlock),
            },
          });
          pending.delete('6');
          if (onProgress) onProgress('6');
        },
      ];

      let i = 0;
      const tick = () => {
        if (cancelled || i >= tasks.length) return;
        try { tasks[i++](); } catch (e) { console.error('Isomap pipeline error:', e); return; }
        if (i < tasks.length) setTimeout(tick, 0);
      };
      setTimeout(tick, 0);
    }

    function cancel() { cancelled = true; }

    return { steps, presentSubSteps, pending, start, cancel };
  },
};
