import { center3, covariance3, eigSymSorted3 } from '../linalg.js';

function formatMatrix(C) {
  return C.map(row => row.map(v => v.toFixed(3).padStart(8)).join(' ')).join('\n');
}

export const PCA = {
  id: 'pca',
  label: 'PCA',
  params: [],
  pseudocode: [
    { id: 'pca-center', title: '1. Center the data', steps: ['1'],
      lines: ['mean ← (1/N) · Σ_i x_i', 'for i = 1..N: x_i ← x_i − mean'] },
    { id: 'pca-cov', title: '2. Form the covariance matrix', steps: ['3'],
      lines: ['C ← (1/(N−1)) · X_cᵀ X_c'] },
    { id: 'pca-eig', title: '3. Eigendecompose C', steps: ['5'],
      lines: ['C = V Λ Vᵀ (eigvecs in columns of V)', 'sort columns of V by decreasing eigenvalue'] },
    { id: 'pca-project', title: '4. Project to 2D', steps: ['6'],
      lines: ['Y ← X_c · V[:, 0:2]'] },
  ],
  run(dataset, _params) {
    const X = dataset.X;
    const t = dataset.t;
    const N = X.length / 3;
    const steps = new Map();

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: null,
      label: 'Raw data',
      ifw: {
        intuition: '<p>The raw 3D point cloud as the algorithm receives it. Points are coloured by an intrinsic parameter along the data manifold so that we can later see whether the embedding preserves that ordering.</p>',
        formula: null,
        worked: null,
      },
    });

    const { Xc, mean } = center3(X);
    steps.set('1', {
      points: Xc, t, edges: null, colors: null,
      label: 'Centered data',
      ifw: {
        intuition: '<p>PCA looks for directions of maximum variance, which is only meaningful around a fixed origin. We subtract the sample mean so the cloud is centred at the origin.</p>',
        formula: '$$\\bar{x} = \\frac{1}{N}\\sum_i x_i, \\qquad x_i \\leftarrow x_i - \\bar{x}$$',
        worked: `<p>The sample mean is approximately (${mean.map(v => v.toFixed(2)).join(', ')}).</p>`,
      },
    });

    const C = covariance3(Xc);
    steps.set('3', {
      points: Xc, t, edges: null, colors: null,
      label: 'Covariance matrix',
      ifw: {
        intuition: '<p>The 3x3 covariance matrix summarises how the centred coordinates co-vary. Its eigenvectors are the directions of maximal variance.</p>',
        formula: '$$C = \\frac{1}{N-1} X_c^{\\top} X_c$$',
        worked: `<pre>${formatMatrix(C)}</pre>`,
      },
    });

    const { lambda, vectors } = eigSymSorted3(C);
    const v1 = vectors[0], v2 = vectors[1];
    steps.set('5', {
      points: Xc, t, edges: null, colors: null,
      pcAxes: { v1, v2, lambda },
      label: 'Principal directions',
      ifw: {
        intuition: '<p>Decomposing C produces orthogonal directions ordered by how much variance the data exhibits along each. The first two axes form the 2D embedding basis.</p>',
        formula: '$$C = V \\Lambda V^{\\top}$$',
        worked: `<p>Eigenvalues: (${lambda.map(v => v.toFixed(2)).join(', ')}).</p>`,
      },
    });

    const embed2d = new Float64Array(N * 2);
    for (let i = 0; i < N; i++) {
      let a = 0, b = 0;
      for (let d = 0; d < 3; d++) {
        a += Xc[i * 3 + d] * v1[d];
        b += Xc[i * 3 + d] * v2[d];
      }
      embed2d[i * 2] = a;
      embed2d[i * 2 + 1] = b;
    }
    steps.set('6', {
      points: Xc, t, edges: null, colors: null, embed2d,
      label: 'Projected to 2D',
      ifw: {
        intuition: '<p>Each centred point is projected onto the plane spanned by the top two principal directions.</p>',
        formula: '$$y_i = (v_1^{\\top} x_i,\\; v_2^{\\top} x_i)$$',
        worked: '<p>The 3D thumbnail in the corner stays orbitable so you can compare the original cloud to the projection.</p>',
      },
    });

    return { steps, presentSubSteps: ['0', '1', '3', '5', '6'] };
  },
};
