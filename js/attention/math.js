// js/attention/math.js
// Pure computation for single-head scaled dot-product attention. No DOM access anywhere in
// this file — it is the only part of this page's code unit-testable with plain Node asserts.

export function linearProject(W, x) {
  return W.map((row) => row.reduce((sum, w, j) => sum + w * x[j], 0));
}

export function dot(a, b) {
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

export function projectAll(tokens, embeddings, W) {
  const out = {};
  for (const t of tokens) out[t] = linearProject(W, embeddings[t]);
  return out;
}

export function scoreMatrix(tokens, Q, K) {
  return tokens.map((ti) => tokens.map((tj) => dot(Q[ti], K[tj])));
}

export function scaleMatrix(scores, d) {
  const s = Math.sqrt(d);
  return scores.map((row) => row.map((v) => v / s));
}

export const NEG_INF = -1e9;

export function applyCausalMask(scaled) {
  return scaled.map((row, i) => row.map((v, j) => (j > i ? NEG_INF : v)));
}

export function softmaxRow(row) {
  const m = Math.max(...row);
  const exps = row.map((v) => Math.exp(v - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export function softmaxMatrix(matrix) {
  return matrix.map(softmaxRow);
}

export function weightedSum(tokens, weights, V, d) {
  return tokens.map((ti, i) => {
    const out = new Array(d).fill(0);
    tokens.forEach((tj, j) => {
      const w = weights[i][j];
      V[tj].forEach((vk, k) => {
        out[k] += w * vk;
      });
    });
    return out;
  });
}

export function computePipeline(tokens, embeddings, weights, options = {}) {
  const causal = !!options.causal;
  const d = embeddings[tokens[0]].length;
  const Q = projectAll(tokens, embeddings, weights.WQ);
  const K = projectAll(tokens, embeddings, weights.WK);
  const V = projectAll(tokens, embeddings, weights.WV);
  const scores = scoreMatrix(tokens, Q, K);
  const scaled = scaleMatrix(scores, d);
  const masked = causal ? applyCausalMask(scaled) : scaled;
  const weightsOut = softmaxMatrix(masked);
  const output = weightedSum(tokens, weightsOut, V, d);
  return { tokens, embeddings, Q, K, V, scores, scaled, masked, weights: weightsOut, output, d, causal };
}
