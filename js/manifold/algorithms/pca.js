import { center3, covariance3, eigSymSorted3 } from '../linalg.js';
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

export const PCA = {
  id: 'pca',
  label: 'PCA',
  params: [],
  presentSubSteps: ['0', '1', '3', '5', '6'],
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
    const presentSubSteps = ['0', '1', '3', '5', '6'];
    const pending = new Set(['1', '3', '5', '6']);
    let cancelled = false;
    const samples = sampleIndices(N);

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: null,
      vizKind: 'point_cloud',
      label: 'Raw data',
      ifw: {
        intuition: '<p>The raw 3D point cloud as the algorithm receives it. Points are coloured by an intrinsic parameter along the data manifold so that we can later see whether the embedding preserves that ordering.</p>',
        formula: null,
        worked: null,
      },
    });

    function start(onProgress) {
      const mem = {};
      const tasks = [
        () => {
          const { Xc, mean } = center3(X);
          mem.Xc = Xc;
          mem.mean = mean;
          const inputRows = samples.map(i => [i, formatVec3(rowOf(X, i))]);
          const inputTable = formatTable(['i', 'x_i'], inputRows);
          const inputBlock = inputTable + '\n\nmean μ = ' + formatVec3(mean);
          const outputRows = samples.map(i => {
            const xi = rowOf(X, i);
            const xc = rowOf(Xc, i);
            return [i, formatVec3(xi) + ' − μ', formatVec3(xc)];
          });
          const outputBlock = formatTable(['i', 'x_i − μ', 'x_i (centered)'], outputRows);
          steps.set('1', {
            points: Xc, t, edges: null, colors: null,
            vizKind: 'centering',
            rawPoints: X.slice(),
            label: 'Centered data',
            ifw: {
              intuition: '<p>PCA looks for directions of maximum variance, which is only meaningful around a fixed origin. We subtract the sample mean so the cloud is centred at the origin.</p>',
              formula: '$$\\bar{x} = \\frac{1}{N}\\sum_i x_i, \\qquad x_i \\leftarrow x_i - \\bar{x}$$',
              worked: workedSections(inputBlock, '$$x_i \\leftarrow x_i - \\mu$$', outputBlock),
            },
          });
          pending.delete('1');
          if (onProgress) onProgress('1');
        },
        () => {
          const Xc = mem.Xc;
          const C = covariance3(Xc);
          mem.C = C;
          const inputRows = samples.map(i => [i, formatVec3(rowOf(Xc, i))]);
          const inputBlock = formatTable(['i', 'x_i (centered)'], inputRows) + '\n\nN = ' + N;
          let sum00 = 0;
          for (let i = 0; i < N; i++) sum00 += Xc[i * 3] * Xc[i * 3];
          const computeLine = 'sum_i x_{i,1} x_{i,1} = ' + sum00.toFixed(3) +
            '\nC[0][0] = ' + sum00.toFixed(3) + ' / (' + N + ' − 1) = ' + C[0][0].toFixed(4);
          const outputBlock = computeLine + '\n\nC =\n' + formatMatrix(C, { digits: 4 });
          steps.set('3', {
            points: Xc, t, edges: null, colors: null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'cloud_thumb', label: 'X_c (centered)', data: Xc },
              { kind: 'matrix_numbers', label: 'C', data: C },
            ],
            paneOpLabels: ['(1 / (N − 1)) · X_cᵀ X_c'],
            label: 'Covariance matrix',
            ifw: {
              intuition: '<p>The 3x3 covariance matrix summarises how the centred coordinates co-vary. Its eigenvectors are the directions of maximal variance.</p>',
              formula: '$$C = \\frac{1}{N-1} X_c^{\\top} X_c$$',
              worked: workedSections(inputBlock, '$$C_{ab} = \\frac{1}{N-1}\\sum_i x_{i,a}\\, x_{i,b}$$', outputBlock),
            },
          });
          pending.delete('3');
          if (onProgress) onProgress('3');
        },
        () => {
          const { lambda, vectors } = eigSymSorted3(mem.C);
          mem.lambda = lambda;
          mem.vectors = vectors;
          const inputBlock = 'C =\n' + formatMatrix(mem.C, { digits: 4 });
          const eigvalsBlock = 'λ = (' + lambda.map(v => Number(v).toFixed(4)).join(', ') + ')';
          const eigvecsBlock = 'V =\n' + formatMatrix(vectors.map(v => Array.from(v)), { digits: 4 });
          const outputBlock = eigvalsBlock + '\n\n' + eigvecsBlock;
          steps.set('5', {
            points: mem.Xc, t, edges: null, colors: null,
            pcAxes: { v1: vectors[0], v2: vectors[1], v3: vectors[2], lambda },
            vizKind: 'spectral',
            algoId: 'pca',
            label: 'Principal directions',
            ifw: {
              intuition: '<p>Decomposing C produces orthogonal directions ordered by how much variance the data exhibits along each. The first two axes form the 2D embedding basis.</p>',
              formula: '$$C = V \\Lambda V^{\\top}$$',
              worked: workedSections(inputBlock, '$$C\\, v_k = \\lambda_k\\, v_k$$', outputBlock),
            },
          });
          pending.delete('5');
          if (onProgress) onProgress('5');
        },
        () => {
          const Xc = mem.Xc;
          const v1 = mem.vectors[0], v2 = mem.vectors[1];
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
          const inputBlock = 'v_1 = ' + formatVec3(Array.from(v1)) + '\nv_2 = ' + formatVec3(Array.from(v2));
          const outputRows = samples.map(i => [i, formatVec3(rowOf(Xc, i)),
            '(' + embed2d[i * 2].toFixed(3) + ', ' + embed2d[i * 2 + 1].toFixed(3) + ')']);
          const outputBlock = formatTable(['i', 'x_i', 'y_i'], outputRows);
          steps.set('6', {
            points: Xc, t, edges: null, colors: null, embed2d,
            vizKind: 'embedding',
            label: 'Projected to 2D',
            ifw: {
              intuition: '<p>Each centred point is projected onto the plane spanned by the top two principal directions.</p>',
              formula: '$$y_i = (v_1^{\\top} x_i,\\; v_2^{\\top} x_i)$$',
              worked: workedSections(inputBlock, '$$y_{i,k} = v_k^{\\top} x_i$$', outputBlock),
            },
          });
          pending.delete('6');
          if (onProgress) onProgress('6');
        },
      ];

      let i = 0;
      const tick = () => {
        if (cancelled || i >= tasks.length) return;
        try { tasks[i++](); } catch (e) { console.error('PCA pipeline error:', e); return; }
        if (i < tasks.length) setTimeout(tick, 0);
      };
      setTimeout(tick, 0);
    }

    function cancel() { cancelled = true; }

    return { steps, presentSubSteps, pending, start, cancel };
  },
};
