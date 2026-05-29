export function mean3(X) {
  const N = X.length / 3;
  const m = [0, 0, 0];
  for (let i = 0; i < N; i++) {
    m[0] += X[i * 3]; m[1] += X[i * 3 + 1]; m[2] += X[i * 3 + 2];
  }
  m[0] /= N; m[1] /= N; m[2] /= N;
  return m;
}

export function center3(X) {
  const m = mean3(X);
  const N = X.length / 3;
  const out = new Float64Array(X.length);
  for (let i = 0; i < N; i++) {
    out[i * 3] = X[i * 3] - m[0];
    out[i * 3 + 1] = X[i * 3 + 1] - m[1];
    out[i * 3 + 2] = X[i * 3 + 2] - m[2];
  }
  return { Xc: out, mean: m };
}

export function covariance3(Xc) {
  const N = Xc.length / 3;
  const C = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < N; i++) {
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) C[a][b] += Xc[i * 3 + a] * Xc[i * 3 + b];
    }
  }
  const denom = Math.max(1, N - 1);
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) C[a][b] /= denom;
  return C;
}

export function squaredDist3(X, i, j) {
  const dx = X[i*3] - X[j*3];
  const dy = X[i*3+1] - X[j*3+1];
  const dz = X[i*3+2] - X[j*3+2];
  return dx*dx + dy*dy + dz*dz;
}

export function knnGraph(X, k) {
  const N = X.length / 3;
  const adj = Array.from({ length: N }, () => []);
  const edges = [];
  const seen = new Set();
  const dist = new Float64Array(N);
  const idx = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      dist[j] = i === j ? Infinity : squaredDist3(X, i, j);
      idx[j] = j;
    }
    idx.sort((a, b) => dist[a] - dist[b]);
    for (let m = 0; m < k; m++) {
      const j = idx[m];
      const w = Math.sqrt(dist[j]);
      adj[i].push([j, w]);
      const key = i < j ? `${i},${j}` : `${j},${i}`;
      if (!seen.has(key)) {
        seen.add(key);
        edges.push([Math.min(i,j), Math.max(i,j)]);
      }
    }
  }
  for (let i = 0; i < N; i++) {
    for (const [j, w] of adj[i]) {
      if (!adj[j].some(e => e[0] === i)) adj[j].push([i, w]);
    }
  }
  return { adj, edges };
}

class MinHeap {
  constructor() { this.data = []; }
  size() { return this.data.length; }
  push(item) {
    this.data.push(item);
    let i = this.data.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.data[p][0] <= this.data[i][0]) break;
      [this.data[p], this.data[i]] = [this.data[i], this.data[p]];
      i = p;
    }
  }
  pop() {
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length > 0) {
      this.data[0] = last;
      let i = 0;
      const n = this.data.length;
      while (true) {
        const l = 2*i + 1, r = 2*i + 2;
        let m = i;
        if (l < n && this.data[l][0] < this.data[m][0]) m = l;
        if (r < n && this.data[r][0] < this.data[m][0]) m = r;
        if (m === i) break;
        [this.data[m], this.data[i]] = [this.data[i], this.data[m]];
        i = m;
      }
    }
    return top;
  }
}

export function dijkstraAllPairs(adj, N) {
  const D = new Float64Array(N * N);
  for (let s = 0; s < N; s++) {
    for (let j = 0; j < N; j++) D[s * N + j] = Infinity;
    D[s * N + s] = 0;
    const heap = new MinHeap();
    heap.push([0, s]);
    while (heap.size() > 0) {
      const [d, u] = heap.pop();
      if (d > D[s * N + u]) continue;
      for (const [v, w] of adj[u]) {
        const nd = d + w;
        if (nd < D[s * N + v]) {
          D[s * N + v] = nd;
          heap.push([nd, v]);
        }
      }
    }
  }
  return D;
}

export function doubleCenterSquared(D, N) {
  const D2 = new Float64Array(N * N);
  for (let i = 0; i < N * N; i++) {
    const v = D[i];
    D2[i] = Number.isFinite(v) ? v * v : 0;
  }
  const rowMean = new Float64Array(N);
  const colMean = new Float64Array(N);
  let total = 0;
  for (let i = 0; i < N; i++) {
    let r = 0;
    for (let j = 0; j < N; j++) r += D2[i * N + j];
    rowMean[i] = r / N;
    total += r;
  }
  for (let j = 0; j < N; j++) {
    let c = 0;
    for (let i = 0; i < N; i++) c += D2[i * N + j];
    colMean[j] = c / N;
  }
  const grand = total / (N * N);
  const B = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      B[i * N + j] = -0.5 * (D2[i * N + j] - rowMean[i] - colMean[j] + grand);
    }
  }
  return B;
}

export function jacobiEigSym(Ain) {
  const N = Ain.length;
  const A = [];
  for (let i = 0; i < N; i++) A.push(Array.from(Ain[i]));
  const V = [];
  for (let i = 0; i < N; i++) {
    const row = new Array(N).fill(0);
    row[i] = 1;
    V.push(row);
  }
  const tol = 1e-12;
  for (let iter = 0; iter < 200; iter++) {
    let p = 0, q = 1, maxv = 0;
    for (let i = 0; i < N - 1; i++) {
      for (let j = i + 1; j < N; j++) {
        const a = Math.abs(A[i][j]);
        if (a > maxv) { maxv = a; p = i; q = j; }
      }
    }
    if (maxv < tol) break;
    const app = A[p][p], aqq = A[q][q], apq = A[p][q];
    let c, s;
    if (apq === 0) { c = 1; s = 0; }
    else {
      const phi = (aqq - app) / (2 * apq);
      const t = phi >= 0
        ? 1 / (phi + Math.sqrt(phi * phi + 1))
        : 1 / (phi - Math.sqrt(phi * phi + 1));
      c = 1 / Math.sqrt(1 + t * t);
      s = t * c;
    }
    A[p][p] = c * c * app - 2 * c * s * apq + s * s * aqq;
    A[q][q] = s * s * app + 2 * c * s * apq + c * c * aqq;
    A[p][q] = 0; A[q][p] = 0;
    for (let i = 0; i < N; i++) {
      if (i !== p && i !== q) {
        const aip = A[i][p];
        const aiq = A[i][q];
        const nip = c * aip - s * aiq;
        const niq = s * aip + c * aiq;
        A[i][p] = nip; A[p][i] = nip;
        A[i][q] = niq; A[q][i] = niq;
      }
    }
    for (let i = 0; i < N; i++) {
      const vip = V[i][p];
      const viq = V[i][q];
      V[i][p] = c * vip - s * viq;
      V[i][q] = s * vip + c * viq;
    }
  }
  const lambda = new Array(N);
  for (let i = 0; i < N; i++) lambda[i] = A[i][i];
  const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => lambda[b] - lambda[a]);
  const sortedLambda = order.map(i => lambda[i]);
  const sortedV = [];
  for (let m = 0; m < N; m++) {
    const col = new Array(N);
    for (let r = 0; r < N; r++) col[r] = V[r][order[m]];
    sortedV.push(col);
  }
  return { lambda: sortedLambda, vectors: sortedV };
}

export function eigSymSorted3(M) {
  return jacobiEigSym(M);
}

export function topKSymmetricEig(M, N, k, iters = 120) {
  const Mwork = new Float64Array(M);
  const lambdaOut = new Float64Array(k);
  const vectorsOut = [];
  const Mv = new Float64Array(N);
  for (let kk = 0; kk < k; kk++) {
    const v = new Float64Array(N);
    for (let i = 0; i < N; i++) v[i] = Math.sin((i + 1) * 1.3 + kk * 0.7);
    let norm = 0;
    for (let i = 0; i < N; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < N; i++) v[i] /= norm;
    for (let iter = 0; iter < iters; iter++) {
      for (let i = 0; i < N; i++) {
        let s = 0;
        for (let j = 0; j < N; j++) s += Mwork[i * N + j] * v[j];
        Mv[i] = s;
      }
      norm = 0;
      for (let i = 0; i < N; i++) norm += Mv[i] * Mv[i];
      norm = Math.sqrt(norm);
      if (norm < 1e-12) break;
      for (let i = 0; i < N; i++) v[i] = Mv[i] / norm;
    }
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let j = 0; j < N; j++) s += Mwork[i * N + j] * v[j];
      Mv[i] = s;
    }
    let lambdaK = 0;
    for (let i = 0; i < N; i++) lambdaK += v[i] * Mv[i];
    lambdaOut[kk] = lambdaK;
    vectorsOut.push(v);
    for (let i = 0; i < N; i++) {
      const lv = lambdaK * v[i];
      for (let j = 0; j < N; j++) Mwork[i * N + j] -= lv * v[j];
    }
  }
  return { lambda: lambdaOut, vectors: vectorsOut };
}
