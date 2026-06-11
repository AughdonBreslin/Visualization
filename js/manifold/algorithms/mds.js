import { doubleCenterSquared, topKSymmetricEig, squaredDist3 } from '../linalg.js';
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

export const MDS = {
  id: 'mds',
  label: 'MDS',
  params: [],
  presentSubSteps: ['0', '3', '4', '5', '6'],
  pseudocode: [
    { id: 'mds-distances', title: '1. Compute pairwise distances', steps: ['3'],
      lines: ['$D_{ij} = \\| x_i - x_j \\|$'] },
    { id: 'mds-dc', title: '2. Double-center the squared distance matrix', steps: ['4'],
      lines: ['$B = -\\tfrac{1}{2} H D^2 H$, with $H = I - \\tfrac{1}{N}\\mathbf{1}\\mathbf{1}^{\\top}$'] },
    { id: 'mds-eig', title: '3. Eigendecompose B', steps: ['5'],
      lines: ['$B = V \\Lambda V^{\\top}$ (take top 2 eigenpairs)'] },
    { id: 'mds-embed', title: '4. Form 2D embedding', steps: ['6'],
      lines: ['$Y = [\\, v_1 \\;\\; v_2 \\,]\\, \\mathrm{diag}(\\sqrt{\\lambda_1}, \\sqrt{\\lambda_2})$'] },
  ],
  run(dataset, _params) {
    const X = dataset.X;
    const t = dataset.t;
    const N = X.length / 3;
    const steps = new Map();
    const presentSubSteps = ['0', '3', '4', '5', '6'];
    const pending = new Set(['3', '4', '5', '6']);
    let canceled = false;
    const samples = sampleIndices(N);

    steps.set('0', {
      points: X.slice(), t, edges: null, colors: dataset.colors || null,
      vizKind: 'point_cloud',
      label: 'Raw data',
      ifw: {
        intuition: '<p>MDS preserves pairwise Euclidean distances. It starts from the raw 3D cloud and computes the full N by N distance matrix in the next step.</p>',
        formula: null,
        worked: null,
      },
    });

    function start(onProgress) {
      const mem = {};
      const tasks = [
        () => {
          const D = new Float64Array(N * N);
          for (let i = 0; i < N; i++) {
            for (let j = i + 1; j < N; j++) {
              const d = Math.sqrt(squaredDist3(X, i, j));
              D[i * N + j] = d;
              D[j * N + i] = d;
            }
          }
          mem.D = D;
          const i0 = samples[0], j0 = samples[2];
          const exampleD = D[i0 * N + j0];
          const inputBlock = 'sample points (3 of N=' + N + '):\n' +
            samples.map(i => 'x_' + i + ' = ' + formatVec3(rowOf(X, i))).join('\n');
          const excerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(D[r * N + c]);
            excerpt.push(row);
          }
          const outputBlock = 'example: D[' + i0 + '][' + j0 + '] = || x_' + i0 + ' - x_' + j0 + ' || = ' + exampleD.toFixed(3) +
            '\n\nD (4 of N=' + N + ' rows):\n' + formatMatrix(excerpt, { digits: 3 });
          steps.set('3', {
            points: X.slice(), t, edges: null, colors: dataset.colors || null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'cloud_thumb', label: 'X', data: X.slice() },
              { kind: 'heatmap', label: 'D (N x N)', data: { matrix: D, N } },
            ],
            paneOpLabels: ['D_{ij} = ||x_i - x_j||'],
            label: 'Pairwise distances',
            ifw: {
              intuition: '<p>Every pair of points contributes one distance to the N by N matrix D. The distance matrix encodes all the geometric information MDS uses.</p>',
              formula: '$$D_{ij} = \\| x_i - x_j \\|$$',
              worked: workedSections(inputBlock, '$$D_{ij} = \\| x_i - x_j \\|$$', outputBlock),
            },
          });
          pending.delete('3');
          if (onProgress) onProgress('3');
        },
        () => {
          const D = mem.D;
          const B = doubleCenterSquared(D, N);
          mem.B = B;
          const D2 = new Float64Array(N * N);
          for (let i = 0; i < N * N; i++) D2[i] = D[i] * D[i];
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
          const D2c = new Float64Array(N * N);
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
              D2c[i * N + j] = D2[i * N + j] - rowMean[i] - colMean[j];
            }
          }
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
          const inputBlock = 'D^2 (4 of N=' + N + ' rows):\n' + formatMatrix(inputExcerpt, { digits: 3 }) +
            '\n\nrow means (first 4): ' + Array.from(rowMean.slice(0, 4)).map(v => v.toFixed(3)).join(', ') +
            '\ngrand mean g = ' + grand.toFixed(3);
          const outputBlock = 'B (4 of N=' + N + ' rows):\n' + formatMatrix(outExcerpt, { digits: 3 });
          steps.set('4', {
            points: X.slice(), t, edges: null, colors: dataset.colors || null,
            vizKind: 'matrix_strip',
            panes: [
              { kind: 'heatmap', label: 'D^2', data: { matrix: D2, N } },
              { kind: 'heatmap', label: 'D^2 - mu', data: { matrix: D2c, N } },
              { kind: 'heatmap', label: 'B', data: { matrix: B, N } },
            ],
            paneOpLabels: ['subtract row/col means', 'x (-1/2) + grand mean'],
            label: 'Double-centered Gram matrix',
            ifw: {
              intuition: '<p>Double-centering converts the pairwise squared distance matrix D squared into the Gram matrix B, which contains inner products relative to the center of the cloud. The next step decomposes B to obtain the embedding.</p>',
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
          const excerpt = [];
          for (let r = 0; r < 4 && r < N; r++) {
            const row = [];
            for (let c = 0; c < 4 && c < N; c++) row.push(mem.B[r * N + c]);
            excerpt.push(row);
          }
          const inputBlock = 'B (4 of N=' + N + ' rows):\n' + formatMatrix(excerpt, { digits: 3 });
          const outputBlock = 'top eigenvalues: ' + Array.from(lambda).map((v, i) => 'lambda_' + (i + 1) + ' = ' + v.toFixed(3)).join(', ') +
            '\n\nv_1 (first 6 entries): [' + Array.from(vectors[0].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']' +
            '\nv_2 (first 6 entries): [' + Array.from(vectors[1].slice(0, 6)).map(v => v.toFixed(3)).join(', ') + ']';
          steps.set('5', {
            points: X.slice(), t, edges: null, colors: dataset.colors || null,
            vizKind: 'spectral',
            algoId: 'mds',
            v1Values: vectors[0],
            topEigvals: lambda,
            topEigvecs: vectors,
            label: 'Top-2 eigendecomposition',
            ifw: {
              intuition: '<p>The top eigenvectors of B give a Euclidean embedding that best preserves the original pairwise distances in the least-squares sense.</p>',
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
          const inputBlock = 'lambda_1 = ' + lambda[0].toFixed(3) + ', lambda_2 = ' + lambda[1].toFixed(3) +
            '\nv_1 (first 3): [' + Array.from(vectors[0].slice(0, 3)).map(v => v.toFixed(3)).join(', ') + ']' +
            '\nv_2 (first 3): [' + Array.from(vectors[1].slice(0, 3)).map(v => v.toFixed(3)).join(', ') + ']';
          const outputRows = samples.map(i => [i,
            '(' + embed2d[i * 2].toFixed(3) + ', ' + embed2d[i * 2 + 1].toFixed(3) + ')']);
          const outputBlock = formatTable(['i', 'y_i'], outputRows);
          steps.set('6', {
            points: X.slice(), t, edges: null, colors: dataset.colors || null, embed2d,
            vizKind: 'embedding',
            label: 'MDS embedding',
            ifw: {
              intuition: '<p>The 2D coordinates approximately preserve the original Euclidean distances. The result is identical to PCA when the data is centered, but MDS reaches it through the distance matrix instead of the covariance matrix.</p>',
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
        if (canceled || i >= tasks.length) return;
        try { tasks[i++](); } catch (e) { console.error('MDS pipeline error:', e); return; }
        if (i < tasks.length) setTimeout(tick, 0);
      };
      setTimeout(tick, 0);
    }

    function cancel() { canceled = true; }

    return { steps, presentSubSteps, pending, start, cancel };
  },
};
