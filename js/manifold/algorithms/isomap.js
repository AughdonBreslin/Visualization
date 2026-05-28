import { knnGraph, dijkstraAllPairs, doubleCenterSquared, topKSymmetricEig } from '../linalg.js';

export const ISOMAP = {
  id: 'isomap',
  label: 'Isomap',
  params: [{ name: 'k', type: 'int', default: 10, min: 2, max: 50 }],
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

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: null,
      label: 'Raw data',
      ifw: { intuition: '<p>Isomap starts from the raw point cloud and recovers manifold geometry from local neighborhoods.</p>', formula: null, worked: null },
    });

    const { adj, edges } = knnGraph(X, k);
    steps.set('2', {
      points: X.slice(), t, edges, colors: null,
      label: 'kNN graph (k = ' + k + ')',
      ifw: {
        intuition: '<p>Connecting each point to its k nearest Euclidean neighbours approximates the manifold by a graph whose edges follow the local surface.</p>',
        formula: '$$\\mathcal{N}_i = \\{ j : \\|x_j - x_i\\| \\text{ among the } k \\text{ smallest}\\}$$',
        worked: `<p>The graph has ${edges.length} undirected edges over ${N} points.</p>`,
      },
    });

    const D = dijkstraAllPairs(adj, N);
    let connected = true;
    for (let i = 0; i < N * N; i++) if (!Number.isFinite(D[i])) { connected = false; break; }
    steps.set('3', {
      points: X.slice(), t, edges, colors: null,
      label: 'Geodesic distances',
      ifw: {
        intuition: '<p>Distances along the graph approximate true geodesic distances on the manifold.</p>',
        formula: '$$D_{ij} = \\min_{\\text{path } i \\to j \\text{ on } G} \\sum_e w_e$$',
        worked: connected ? '<p>Shortest paths computed with Dijkstra from every node. The graph is connected for this k.</p>' : '<p>The kNN graph is disconnected at this k. Increase k.</p>',
      },
    });

    const B = doubleCenterSquared(D, N);
    steps.set('4', {
      points: X.slice(), t, edges, colors: null,
      label: 'Double-centered Gram matrix',
      ifw: {
        intuition: '<p>Classical MDS converts pairwise squared distances into an inner-product matrix B via double centering.</p>',
        formula: '$$B = -\\tfrac{1}{2} H D^{(2)} H, \\quad H = I - \\tfrac{1}{N}\\mathbf{1}\\mathbf{1}^{\\top}$$',
        worked: '<p>Row, column, and grand means are subtracted from D^2 and the result is scaled by -1/2.</p>',
      },
    });

    const { lambda, vectors } = topKSymmetricEig(B, N, 2);
    steps.set('5', {
      points: X.slice(), t, edges, colors: null,
      label: 'Top-2 eigendecomposition',
      ifw: {
        intuition: '<p>The top eigenvectors of B reveal the dominant geometry of the geodesic distances.</p>',
        formula: '$$B = V \\Lambda V^{\\top}$$',
        worked: `<p>Top two eigenvalues: (${lambda[0].toFixed(2)}, ${lambda[1].toFixed(2)}).</p>`,
      },
    });

    const embed2d = new Float64Array(N * 2);
    const s1 = Math.sqrt(Math.max(0, lambda[0]));
    const s2 = Math.sqrt(Math.max(0, lambda[1]));
    for (let i = 0; i < N; i++) {
      embed2d[i * 2] = vectors[0][i] * s1;
      embed2d[i * 2 + 1] = vectors[1][i] * s2;
    }
    steps.set('6', {
      points: X.slice(), t, edges, colors: null, embed2d,
      label: 'Isomap embedding',
      ifw: {
        intuition: '<p>The 2D coordinates flatten the manifold while preserving geodesic distances as well as possible.</p>',
        formula: '$$y_i = \\big(\\sqrt{\\lambda_1}\\, v_{1,i},\\; \\sqrt{\\lambda_2}\\, v_{2,i}\\big)$$',
        worked: '<p>Each point\'s 2D coordinates are read off rows of the top-2 eigenvectors, scaled by the square roots of the eigenvalues.</p>',
      },
    });

    return { steps, presentSubSteps: ['0', '2', '3', '4', '5', '6'] };
  },
};
