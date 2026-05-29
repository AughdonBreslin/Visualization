import { topKSymmetricEig, squaredDist3 } from '../linalg.js';
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

function kernelLabel(kernel, gamma, degree, constant) {
  if (kernel === 'rbf') return 'K_{ij} = exp(-' + gamma + ' ||x_i - x_j||^2)';
  if (kernel === 'polynomial') return 'K_{ij} = (x_i . x_j + ' + constant + ')^' + degree;
  return 'K_{ij} = x_i . x_j';
}

function kernelFormula(kernel) {
  if (kernel === 'rbf') return '$$K_{ij} = \\exp(-\\gamma \\| x_i - x_j \\|^2)$$';
  if (kernel === 'polynomial') return '$$K_{ij} = (x_i \\cdot x_j + c)^d$$';
  return '$$K_{ij} = x_i \\cdot x_j$$';
}

export const KPCA = {
  id: 'kpca',
  label: 'Kernel PCA',
  params: [
    { name: 'kernel', type: 'enum', options: ['rbf', 'polynomial', 'linear'], default: 'rbf' },
    { name: 'gamma', type: 'float', default: 0.5, min: 0.01, max: 20 },
    { name: 'degree', type: 'int', default: 3, min: 1, max: 10 },
    { name: 'constant', type: 'float', default: 1, min: 0, max: 10 },
  ],
  presentSubSteps: ['0', '3', '4', '5', '6'],
  pseudocode: [
    { id: 'kpca-K', title: '1. Compute kernel matrix K', steps: ['3'],
      lines: ['rbf: K_{ij} = exp(-gamma ||x_i - x_j||^2)',
              'polynomial: K_{ij} = (x_i . x_j + c)^d',
              'linear: K_{ij} = x_i . x_j'] },
    { id: 'kpca-center', title: '2. Center K', steps: ['4'],
      lines: ['K_c = K - 1_N K - K 1_N + 1_N K 1_N'] },
    { id: 'kpca-eig', title: '3. Eigendecompose K_c', steps: ['5'],
      lines: ['K_c = V Lambda V^T (take top-2)'] },
    { id: 'kpca-embed', title: '4. Form 2D embedding', steps: ['6'],
      lines: ['Y = V[:, 0:2] * diag(sqrt(lambda_1), sqrt(lambda_2))'] },
  ],
  run(dataset, params) {
    const X = dataset.X;
    const t = dataset.t;
    const N = X.length / 3;
    const kernel = params.kernel || 'rbf';
    const gamma = params.gamma || 0.5;
    const degree = params.degree || 3;
    const constant = params.constant || 1;
    const steps = new Map();
    const presentSubSteps = ['0', '3', '4', '5', '6'];
    const pending = new Set(['3', '4', '5', '6']);
    let cancelled = false;
    const samples = sampleIndices(N);

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: dataset.colors || null,
      vizKind: 'point_cloud',
      label: 'Raw data',
      ifw: {
        intuition: '<p>Kernel PCA replaces the inner product with a non-linear kernel, then performs ordinary PCA in the feature space implicitly defined by the kernel.</p>',
        formula: null, worked: null,
      },
    });

    function computeKernel(kernel, gamma, degree, constant) {
      const K = new Float64Array(N * N);
      for (let i = 0; i < N; i++) {
        for (let j = i; j < N; j++) {
          let kij;
          if (kernel === 'rbf') {
            kij = Math.exp(-gamma * squaredDist3(X, i, j));
          } else if (kernel === 'polynomial') {
            const dot = X[i*3]*X[j*3] + X[i*3+1]*X[j*3+1] + X[i*3+2]*X[j*3+2];
            kij = Math.pow(dot + constant, degree);
          } else {
            kij = X[i*3]*X[j*3] + X[i*3+1]*X[j*3+1] + X[i*3+2]*X[j*3+2];
          }
          K[i * N + j] = kij;
          K[j * N + i] = kij;
        }
      }
      return K;
    }

    function start(onProgress) {
      const mem = {};
      const tasks = [
        () => {
          const K = computeKernel(kernel, gamma, degree, constant);
          mem.K = K;
          const i0 = samples[0], j0 = samples[1];
          const exampleK = K[i0 * N + j0];
          const inputBlock = 'sample points (3 of N=' + N + '):\n' +
            samples.map(i => 'x_' + i + ' = ' + formatVec3(rowOf(X, i))).join('\n') +
            '\n\nkernel = ' + kernel + ', gamma = ' + gamma + ', degree = ' + degree + ', constant = ' + constant;
          const excerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(K[r * N + c]);
            excerpt.push(row);
          }
          const outputBlock = 'example: K[' + i0 + '][' + j0 + '] = ' + exampleK.toFixed(4) +
            '\n\nK (4 of N=' + N + ' rows):\n' + formatMatrix(excerpt, { digits: 4 });
          steps.set('3', {
            points: X.slice(), t, edges: null, colors: dataset.colors || null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'cloud_thumb', label: 'X', data: X.slice() },
              { kind: 'heatmap', label: 'K (N x N)', data: { matrix: K, N } },
            ],
            paneOpLabels: [kernelLabel(kernel, gamma, degree, constant)],
            label: 'Kernel matrix K',
            ifw: {
              intuition: '<p>The kernel function K(x, y) measures similarity in an implicit feature space. The full kernel matrix replaces the data matrix in the rest of the pipeline.</p>',
              formula: kernelFormula(kernel),
              worked: workedSections(inputBlock, kernelFormula(kernel), outputBlock),
            },
          });
          pending.delete('3');
          if (onProgress) onProgress('3');
        },
        () => {
          const K = mem.K;
          const Krow = new Float64Array(N);
          for (let i = 0; i < N; i++) {
            let s = 0;
            for (let j = 0; j < N; j++) s += K[i * N + j];
            Krow[i] = s / N;
          }
          let grand = 0;
          for (let i = 0; i < N; i++) grand += Krow[i];
          grand /= N;
          const Kc = new Float64Array(N * N);
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
              Kc[i * N + j] = K[i * N + j] - Krow[i] - Krow[j] + grand;
            }
          }
          mem.Kc = Kc;
          const inputExcerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(K[r * N + c]);
            inputExcerpt.push(row);
          }
          const outExcerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(Kc[r * N + c]);
            outExcerpt.push(row);
          }
          const inputBlock = 'K (4 of N=' + N + ' rows):\n' + formatMatrix(inputExcerpt, { digits: 4 }) +
            '\n\nrow means (first 4): ' + Array.from(Krow.slice(0, 4)).map(v => v.toFixed(3)).join(', ') +
            '\ngrand mean = ' + grand.toFixed(4);
          const outputBlock = 'K_c (4 of N=' + N + ' rows):\n' + formatMatrix(outExcerpt, { digits: 4 });
          steps.set('4', {
            points: X.slice(), t, edges: null, colors: dataset.colors || null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'heatmap', label: 'K', data: { matrix: K, N } },
              { kind: 'heatmap', label: 'K_c (centered)', data: { matrix: Kc, N } },
            ],
            paneOpLabels: ['K - 1_N K - K 1_N + 1_N K 1_N'],
            label: 'Centered kernel matrix',
            ifw: {
              intuition: '<p>Centering K is the kernel-space analogue of centering the data before ordinary PCA. The double subtraction and grand mean addition account for the row and column shifts implied by the centering operator.</p>',
              formula: '$$K_c = K - \\mathbf{1}_N K - K \\mathbf{1}_N + \\mathbf{1}_N K \\mathbf{1}_N$$',
              worked: workedSections(inputBlock, '$$K_{c,ij} = K_{ij} - r_i - r_j + g$$', outputBlock),
            },
          });
          pending.delete('4');
          if (onProgress) onProgress('4');
        },
        () => {
          const { lambda, vectors } = topKSymmetricEig(mem.Kc, N, 8);
          mem.lambda = lambda;
          mem.vectors = vectors;
          const inputBlock = 'K_c from step 4.';
          const outputBlock = 'top eigenvalues: ' + Array.from(lambda).map((v, i) => 'lambda_' + (i + 1) + ' = ' + v.toFixed(3)).join(', ') +
            '\n\nv_1 (first 6 entries): [' + Array.from(vectors[0].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']' +
            '\nv_2 (first 6 entries): [' + Array.from(vectors[1].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']';
          steps.set('5', {
            points: X.slice(), t, edges: null, colors: dataset.colors || null,
            vizKind: 'spectral',
            algoId: 'kpca',
            v1Values: vectors[0],
            topEigvals: lambda,
            topEigvecs: vectors,
            label: 'Top-2 eigendecomposition',
            ifw: {
              intuition: '<p>The principal components in feature space are eigenvectors of the centered kernel matrix. Each carries one coordinate of the non-linear embedding.</p>',
              formula: '$$K_c\\, v_k = \\lambda_k\\, v_k$$',
              worked: workedSections(inputBlock, '$$K_c\\, v_k = \\lambda_k\\, v_k$$', outputBlock),
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
          const inputBlock = 'lambda_1 = ' + lambda[0].toFixed(3) + ', lambda_2 = ' + lambda[1].toFixed(3);
          const outputRows = samples.map(i => [i,
            '(' + embed2d[i * 2].toFixed(3) + ', ' + embed2d[i * 2 + 1].toFixed(3) + ')']);
          const outputBlock = formatTable(['i', 'y_i'], outputRows);
          steps.set('6', {
            points: X.slice(), t, edges: null, colors: dataset.colors || null, embed2d,
            vizKind: 'embedding',
            label: 'Kernel PCA embedding',
            ifw: {
              intuition: '<p>Each point\'s 2D coordinate is its projection onto the top-2 principal components in the kernel-induced feature space.</p>',
              formula: '$$y_{i,k} = \\sqrt{\\lambda_k}\\, v_{k,i}$$',
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
        try { tasks[i++](); } catch (e) { console.error('KPCA pipeline error:', e); return; }
        if (i < tasks.length) setTimeout(tick, 0);
      };
      setTimeout(tick, 0);
    }

    function cancel() { cancelled = true; }

    return { steps, presentSubSteps, pending, start, cancel };
  },
};
