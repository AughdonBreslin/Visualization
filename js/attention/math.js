// js/attention/math.js
// Pure computation for single-head scaled dot-product attention. No DOM access anywhere in
// this file: it is the only part of this page's code unit-testable with plain Node asserts.

export function linearProject(W, x) {
  return W.map((row) => row.reduce((sum, w, j) => sum + w * x[j], 0));
}

export function dot(a, b) {
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

// Standard sinusoidal positional encoding (Vaswani et al. 2017, "Attention Is All You Need",
// section 3.5). Computed exactly -- unlike the weight matrices, nothing here is hand-picked.
export function positionalEncoding(pos, d) {
  const out = new Array(d);
  for (let i = 0; i < d / 2; i++) {
    const divisor = Math.pow(10000, (2 * i) / d);
    out[2 * i] = Math.sin(pos / divisor);
    out[2 * i + 1] = Math.cos(pos / divisor);
  }
  return out;
}

// Embeddings are scaled by sqrt(d) before adding positional encoding, the same detail the
// original paper's embedding layer uses (section 3.4): without it, the positional signal
// (bounded to [-1, 1]) can overwhelm the hand-picked embedding values once added, flattening
// the resulting attention pattern toward uniform.
export function buildInputMatrix(embeddings, d) {
  const scale = Math.sqrt(d);
  return embeddings.map((embedding, pos) => {
    const pe = positionalEncoding(pos, d);
    return embedding.map((v, k) => v * scale + pe[k]);
  });
}

export function projectAll(embeddings, W) {
  return embeddings.map((x) => linearProject(W, x));
}

export function scoreMatrix(Q, K) {
  return Q.map((qi) => K.map((kj) => dot(qi, kj)));
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

export function weightedSum(weights, V, d) {
  return weights.map((row) => {
    const out = new Array(d).fill(0);
    row.forEach((w, j) => {
      V[j].forEach((vk, k) => {
        out[k] += w * vk;
      });
    });
    return out;
  });
}

export function computePipeline(tokens, embeddings, weights, options = {}) {
  const causal = !!options.causal;
  const d = embeddings[0].length;
  const X = buildInputMatrix(embeddings, d);
  const Q = projectAll(X, weights.WQ);
  const K = projectAll(X, weights.WK);
  const V = projectAll(X, weights.WV);
  const scores = scoreMatrix(Q, K);
  const scaled = scaleMatrix(scores, d);
  const masked = causal ? applyCausalMask(scaled) : scaled;
  const weightsOut = softmaxMatrix(masked);
  const output = weightedSum(weightsOut, V, d);
  return { tokens, embeddings, X, Q, K, V, scores, scaled, masked, weights: weightsOut, output, d, causal };
}
