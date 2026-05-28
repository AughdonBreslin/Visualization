import { knnGraph, dijkstraAllPairs, doubleCenterSquared, topKSymmetricEig } from '../linalg.js';

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

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: null,
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
          steps.set('2', {
            points: X.slice(), t, edges, colors: null,
            label: 'kNN graph (k = ' + k + ')',
            ifw: {
              intuition: '<p>Connecting each point to its k nearest Euclidean neighbours approximates the manifold by a graph whose edges follow the local surface.</p>',
              formula: '$$\\mathcal{N}_i = \\{ j : \\|x_j - x_i\\| \\text{ among the } k \\text{ smallest}\\}$$',
              worked: `<p>The graph has ${edges.length} undirected edges over ${N} points.</p>`,
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
          steps.set('3', {
            points: X.slice(), t, edges: mem.edges, colors: null,
            label: 'Geodesic distances',
            ifw: {
              intuition: '<p>Distances along the graph approximate true geodesic distances on the manifold.</p>',
              formula: '$$D_{ij} = \\min_{\\text{path } i \\to j \\text{ on } G} \\sum_e w_e$$',
              worked: connected ? '<p>Shortest paths computed with Dijkstra from every node. The graph is connected for this k.</p>' : '<p>The kNN graph is disconnected at this k. Increase k.</p>',
            },
          });
          pending.delete('3');
          if (onProgress) onProgress('3');
        },
        () => {
          const B = doubleCenterSquared(mem.D, N);
          mem.B = B;
          steps.set('4', {
            points: X.slice(), t, edges: mem.edges, colors: null,
            label: 'Double-centered Gram matrix',
            ifw: {
              intuition: '<p>Classical MDS converts pairwise squared distances into an inner-product matrix B via double centering.</p>',
              formula: '$$B = -\\tfrac{1}{2} H D^{(2)} H, \\quad H = I - \\tfrac{1}{N}\\mathbf{1}\\mathbf{1}^{\\top}$$',
              worked: '<p>Row, column, and grand means are subtracted from D^2 and the result is scaled by -1/2.</p>',
            },
          });
          pending.delete('4');
          if (onProgress) onProgress('4');
        },
        () => {
          const { lambda, vectors } = topKSymmetricEig(mem.B, N, 2);
          mem.lambda = lambda;
          mem.vectors = vectors;
          steps.set('5', {
            points: X.slice(), t, edges: mem.edges, colors: null,
            label: 'Top-2 eigendecomposition',
            ifw: {
              intuition: '<p>The top eigenvectors of B reveal the dominant geometry of the geodesic distances.</p>',
              formula: '$$B = V \\Lambda V^{\\top}$$',
              worked: `<p>Top two eigenvalues: (${lambda[0].toFixed(2)}, ${lambda[1].toFixed(2)}).</p>`,
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
          steps.set('6', {
            points: X.slice(), t, edges: mem.edges, colors: null, embed2d,
            label: 'Isomap embedding',
            ifw: {
              intuition: '<p>The 2D coordinates flatten the manifold while preserving geodesic distances as well as possible.</p>',
              formula: '$$y_i = \\big(\\sqrt{\\lambda_1}\\, v_{1,i},\\; \\sqrt{\\lambda_2}\\, v_{2,i}\\big)$$',
              worked: '<p>Each point\'s 2D coordinates are read off rows of the top-2 eigenvectors, scaled by the square roots of the eigenvalues.</p>',
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
