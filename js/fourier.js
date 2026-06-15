(function () {
  "use strict";

  const N = 256;
  const HALF = N / 2;
  const N2 = N * N;
  const MAX_POLY_DEG = 128;
  const MAX_WAVELET_LEVELS = 8; // log2(256)

  // ---- FFT (Cooley-Tukey radix-2 DIT, in-place) ----

  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len;
      const wRe = Math.cos(ang), wIm = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let tRe = 1, tIm = 0;
        const h = len >> 1;
        for (let j = 0; j < h; j++) {
          const uRe = re[i + j], uIm = im[i + j];
          const vRe = re[i + j + h] * tRe - im[i + j + h] * tIm;
          const vIm = re[i + j + h] * tIm + im[i + j + h] * tRe;
          re[i + j] = uRe + vRe;
          im[i + j] = uIm + vIm;
          re[i + j + h] = uRe - vRe;
          im[i + j + h] = uIm - vIm;
          const nr = tRe * wRe - tIm * wIm;
          tIm = tRe * wIm + tIm * wRe;
          tRe = nr;
        }
      }
    }
  }

  function fft2d(re, im) {
    const buf = new Float64Array(N), bufI = new Float64Array(N);
    for (let r = 0; r < N; r++) {
      const o = r * N;
      buf.set(re.subarray(o, o + N)); bufI.set(im.subarray(o, o + N));
      fft(buf, bufI);
      re.set(buf, o); im.set(bufI, o);
    }
    for (let c = 0; c < N; c++) {
      for (let r = 0; r < N; r++) { buf[r] = re[r * N + c]; bufI[r] = im[r * N + c]; }
      fft(buf, bufI);
      for (let r = 0; r < N; r++) { re[r * N + c] = buf[r]; im[r * N + c] = bufI[r]; }
    }
  }

  function ifft2d(re, im) {
    for (let i = 0; i < N2; i++) im[i] = -im[i];
    fft2d(re, im);
    for (let i = 0; i < N2; i++) { re[i] /= N2; im[i] = -im[i] / N2; }
  }

  // ---- Legendre polynomials ----

  function buildLegendreAt(tValues, maxDeg) {
    const n = tValues.length;
    const lv = [new Float64Array(n).fill(1)];
    if (maxDeg < 1) return lv;
    lv.push(Float64Array.from(tValues));
    for (let k = 1; k < maxDeg; k++) {
      const pk = lv[k], pkm1 = lv[k - 1];
      const pk1 = new Float64Array(n);
      for (let i = 0; i < n; i++)
        pk1[i] = ((2 * k + 1) * tValues[i] * pk[i] - k * pkm1[i]) / (k + 1);
      lv.push(pk1);
    }
    return lv;
  }

  const tStd = new Float64Array(N);
  for (let i = 0; i < N; i++) tStd[i] = 2 * i / (N - 1) - 1;
  const legStd = buildLegendreAt(tStd, MAX_POLY_DEG);

  const tExt = new Float64Array(N);
  for (let i = 0; i < N; i++) tExt[i] = 4 * i / (N - 1) - 2;
  const legExt = buildLegendreAt(tExt, MAX_POLY_DEG);

  // ---- Polynomial 2D separable decomposition / reconstruction ----

  function polyDecomp2D(pixels) {
    const D = MAX_POLY_DEG, D1 = D + 1;
    const row = new Float64Array(N);
    const B = new Float64Array(N * D1);
    for (let y = 0; y < N; y++) {
      const off = y * N;
      for (let k = 0; k < D1; k++) {
        const pk = legStd[k];
        let sum = 0;
        for (let x = 0; x < N; x++) sum += pixels[off + x] * pk[x];
        B[y * D1 + k] = (2 * k + 1) / N * sum;
      }
    }
    const C = new Float64Array(D1 * D1);
    for (let k = 0; k < D1; k++) {
      for (let y = 0; y < N; y++) row[y] = B[y * D1 + k];
      for (let j = 0; j < D1; j++) {
        const pj = legStd[j];
        let sum = 0;
        for (let y = 0; y < N; y++) sum += row[y] * pj[y];
        C[j * D1 + k] = (2 * j + 1) / N * sum;
      }
    }
    return C;
  }

  function polyRecon2D(C, cutoff, lv) {
    const D1 = MAX_POLY_DEG + 1;
    const c1 = cutoff + 1;
    const V = new Float64Array(N * c1);
    for (let y = 0; y < N; y++) {
      for (let k = 0; k < c1; k++) {
        let sum = 0;
        for (let j = 0; j < c1; j++) sum += C[j * D1 + k] * lv[j][y];
        V[y * c1 + k] = sum;
      }
    }
    const out = new Float64Array(N2);
    for (let y = 0; y < N; y++) {
      const vy = V.subarray(y * c1, y * c1 + c1);
      const off = y * N;
      for (let x = 0; x < N; x++) {
        let val = 0;
        for (let k = 0; k < c1; k++) val += vy[k] * lv[k][x];
        out[off + x] = val;
      }
    }
    return out;
  }

  // ---- Chebyshev polynomials of the first kind (T_k) ----
  // Orthogonal on [-1,1] with weight w(t) = 1/sqrt(1-t^2).
  // T_0=1, T_1=t, T_{k+1}=2t*T_k - T_{k-1}.
  // Norm: ||T_0||^2 = pi, ||T_k||^2 = pi/2 (k>=1).
  // Use midpoint nodes t_i = (2i+1)/N - 1 to avoid singularity at +-1.

  const MAX_CHEB_DEG = 128;

  function buildChebyshevAt(tValues, maxDeg) {
    const n = tValues.length;
    const lv = [new Float64Array(n).fill(1)];
    if (maxDeg < 1) return lv;
    lv.push(Float64Array.from(tValues));
    for (let k = 1; k < maxDeg; k++) {
      const pk = lv[k], pkm1 = lv[k - 1];
      const pk1 = new Float64Array(n);
      for (let i = 0; i < n; i++) pk1[i] = 2 * tValues[i] * pk[i] - pkm1[i];
      lv.push(pk1);
    }
    return lv;
  }

  const tChebStd = new Float64Array(N);
  for (let i = 0; i < N; i++) tChebStd[i] = (2 * i + 1) / N - 1;

  const chebWeights = new Float64Array(N);
  for (let i = 0; i < N; i++) chebWeights[i] = 1 / Math.sqrt(1 - tChebStd[i] ** 2);

  const chebStd = buildChebyshevAt(tChebStd, MAX_CHEB_DEG);

  const tChebExt = new Float64Array(N);
  for (let i = 0; i < N; i++) tChebExt[i] = 4 * (i + 0.5) / N - 2;
  const chebExt = buildChebyshevAt(tChebExt, MAX_CHEB_DEG);

  function chebDecomp2D(pixels) {
    const D1 = MAX_CHEB_DEG + 1;
    const B = new Float64Array(N * D1);
    for (let y = 0; y < N; y++) {
      const off = y * N;
      for (let k = 0; k < D1; k++) {
        const Tk = chebStd[k];
        const factor = (k === 0 ? 1 : 2) * 2 / (Math.PI * N);
        let sum = 0;
        for (let x = 0; x < N; x++) sum += pixels[off + x] * Tk[x] * chebWeights[x];
        B[y * D1 + k] = factor * sum;
      }
    }
    const C = new Float64Array(D1 * D1);
    const row = new Float64Array(N);
    for (let k = 0; k < D1; k++) {
      for (let y = 0; y < N; y++) row[y] = B[y * D1 + k];
      for (let j = 0; j < D1; j++) {
        const Tj = chebStd[j];
        const factor = (j === 0 ? 1 : 2) * 2 / (Math.PI * N);
        let sum = 0;
        for (let y = 0; y < N; y++) sum += row[y] * Tj[y] * chebWeights[y];
        C[j * D1 + k] = factor * sum;
      }
    }
    return C;
  }

  function chebRecon2D(C, cutoff, lv) {
    const D1 = MAX_CHEB_DEG + 1;
    const c1 = cutoff + 1;
    const V = new Float64Array(N * c1);
    for (let y = 0; y < N; y++)
      for (let k = 0; k < c1; k++) {
        let sum = 0;
        for (let j = 0; j < c1; j++) sum += C[j * D1 + k] * lv[j][y];
        V[y * c1 + k] = sum;
      }
    const out = new Float64Array(N2);
    for (let y = 0; y < N; y++) {
      const vy = V.subarray(y * c1, y * c1 + c1);
      const off = y * N;
      for (let x = 0; x < N; x++) {
        let val = 0;
        for (let k = 0; k < c1; k++) val += vy[k] * lv[k][x];
        out[off + x] = val;
      }
    }
    return out;
  }

  // ---- Haar wavelet 2D transform ----
  // Forward layout after full MAX_WAVELET_LEVELS-level DWT:
  //   Level L detail subbands (L=1 finest, L=8 coarsest), sh = N >> L:
  //     LH_L: rows [0,sh),  cols [sh, 2sh)
  //     HL_L: rows [sh,2sh), cols [0, sh)
  //     HH_L: rows [sh,2sh), cols [sh,2sh)
  //   DC: single pixel at (0,0)

  const wBuf = new Float64Array(N); // scratch for wavelet row/col ops

  function haarForward2D(data) {
    let size = N;
    while (size > 1) {
      const h = size >> 1;
      // transform rows
      for (let r = 0; r < size; r++) {
        const off = r * N;
        for (let i = 0; i < h; i++) {
          wBuf[i]     = (data[off + 2*i] + data[off + 2*i+1]) * 0.5;
          wBuf[h + i] = (data[off + 2*i] - data[off + 2*i+1]) * 0.5;
        }
        for (let i = 0; i < size; i++) data[off + i] = wBuf[i];
      }
      // transform columns
      for (let c = 0; c < size; c++) {
        for (let r = 0; r < size; r++) wBuf[r] = data[r * N + c];
        for (let i = 0; i < h; i++) {
          data[i * N + c]       = (wBuf[2*i] + wBuf[2*i+1]) * 0.5;
          data[(h + i) * N + c] = (wBuf[2*i] - wBuf[2*i+1]) * 0.5;
        }
      }
      size >>= 1;
    }
  }

  function haarInverse2D(data) {
    // Reverse order: coarsest first (size=2), finest last (size=N)
    for (let size = 2; size <= N; size <<= 1) {
      const h = size >> 1;
      // inverse columns first (columns were applied last in forward)
      for (let c = 0; c < size; c++) {
        for (let r = 0; r < size; r++) wBuf[r] = data[r * N + c];
        for (let i = 0; i < h; i++) {
          data[2*i * N + c]       = wBuf[i] + wBuf[h + i];
          data[(2*i+1) * N + c]   = wBuf[i] - wBuf[h + i];
        }
      }
      // then inverse rows
      for (let r = 0; r < size; r++) {
        const off = r * N;
        for (let c = 0; c < size; c++) wBuf[c] = data[off + c];
        for (let i = 0; i < h; i++) {
          data[off + 2*i]   = wBuf[i] + wBuf[h + i];
          data[off + 2*i+1] = wBuf[i] - wBuf[h + i];
        }
      }
    }
  }

  // ---- Image presets ----

  function makePreset(name) {
    const px = new Float64Array(N2);
    const cx = HALF, cy = HALF;
    switch (name) {
      case 'quadrant':
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
          px[y * N + x] = ((x < HALF) === (y < HALF)) ? 220 : 15;
        break;
      case 'circle':
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
          px[y * N + x] = (x - cx) ** 2 + (y - cy) ** 2 < 72 * 72 ? 220 : 12;
        break;
      case 'gaussian': {
        const s2 = 2 * 52 * 52;
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
          px[y * N + x] = 220 * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / s2);
        break;
      }
      case 'step':
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
          px[y * N + x] = x < HALF ? 215 : 12;
        break;
      case 'rect':
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
          px[y * N + x] = Math.abs(x - cx) < 70 && Math.abs(y - cy) < 52 ? 220 : 12;
        break;
    }
    return px;
  }

  // ---- Colormap (hot: black → red → yellow → white) ----

  function writeHot(d, idx, t) {
    d[idx]     = Math.min(255, Math.round(t * 3 * 255));
    d[idx + 1] = Math.max(0,   Math.min(255, Math.round((t * 3 - 1) * 255)));
    d[idx + 2] = Math.max(0,   Math.min(255, Math.round((t * 3 - 2) * 255)));
    d[idx + 3] = 255;
  }

  // ---- Canvas rendering ----

  function renderGrayscale(canvas, px) {
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(N, N);
    const d = img.data;
    for (let i = 0; i < N2; i++) {
      const v = Math.max(0, Math.min(255, Math.round(px[i])));
      d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v;
      d[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  function renderSpectrum(canvas, re, im, radius) {
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(N, N);
    const d = img.data;
    const mags = new Float64Array(N2);
    let maxM = 0;
    for (let i = 0; i < N2; i++) {
      mags[i] = Math.log1p(Math.sqrt(re[i] * re[i] + im[i] * im[i]));
      if (mags[i] > maxM) maxM = mags[i];
    }
    if (maxM === 0) maxM = 1;
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const sr = (r + HALF) % N, sc = (c + HALF) % N;
        writeHot(d, (r * N + c) * 4, mags[sr * N + sc] / maxM);
      }
    }
    ctx.putImageData(img, 0, 0);
    ctx.beginPath();
    ctx.arc(HALF, HALF, Math.min(radius, HALF - 1), 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(74, 163, 255, 0.9)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  function renderCoeffMatrix(canvas, C, cutoff) {
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(N, N);
    const d = img.data;
    const D = MAX_POLY_DEG, D1 = D + 1;
    const cellSize = N / D1;
    let maxC = 0;
    for (let i = 0; i < D1 * D1; i++) {
      const m = Math.abs(C[i]);
      if (m > maxC) maxC = m;
    }
    if (maxC === 0) maxC = 1;
    for (let r = 0; r < N; r++) {
      const j = Math.min(D, Math.floor(r / cellSize));
      for (let c = 0; c < N; c++) {
        const k = Math.min(D, Math.floor(c / cellSize));
        const active = j <= cutoff && k <= cutoff;
        const t = Math.abs(C[j * D1 + k]) / maxC;
        writeHot(d, (r * N + c) * 4, active ? t : t * 0.18);
      }
    }
    ctx.putImageData(img, 0, 0);
    const box = Math.round((cutoff + 1) * cellSize);
    ctx.strokeStyle = 'rgba(74, 163, 255, 0.9)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(0.75, 0.75, box - 1.5, box - 1.5);
  }

  // Wavelet pyramid display with per-subband normalization.
  // Subbands at level L: sh = N >> L.
  //   LH: rows [0,sh), cols [sh,2sh)
  //   HL: rows [sh,2sh), cols [0,sh)
  //   HH: rows [sh,2sh), cols [sh,2sh)
  // Active if L >= MAX_WAVELET_LEVELS - keepLevels + 1  (i.e. 2*sh <= keepSize).
  function renderWaveletCoeffs(canvas, coeffs, keepLevels) {
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(N, N);
    const d = img.data;
    for (let i = 3; i < N2 * 4; i += 4) d[i] = 255;

    function fillSubband(r0, c0, sh, isActive) {
      let maxAbs = 0;
      for (let r = r0; r < r0 + sh; r++)
        for (let c = c0; c < c0 + sh; c++) {
          const a = Math.abs(coeffs[r * N + c]);
          if (a > maxAbs) maxAbs = a;
        }
      if (maxAbs === 0) maxAbs = 1;
      for (let r = r0; r < r0 + sh; r++)
        for (let c = c0; c < c0 + sh; c++) {
          const t = Math.abs(coeffs[r * N + c]) / maxAbs;
          writeHot(d, (r * N + c) * 4, isActive ? t : t * 0.18);
        }
    }

    const threshold = MAX_WAVELET_LEVELS - keepLevels + 1;
    for (let L = MAX_WAVELET_LEVELS; L >= 1; L--) {
      const sh = N >> L;
      const isActive = L >= threshold;
      fillSubband(0,  sh, sh, isActive); // LH
      fillSubband(sh, 0,  sh, isActive); // HL
      fillSubband(sh, sh, sh, isActive); // HH
    }
    // DC always max brightness
    writeHot(d, 0, 1.0);

    ctx.putImageData(img, 0, 0);

    const keepSize = Math.max(2, 1 << keepLevels);
    ctx.strokeStyle = 'rgba(74, 163, 255, 0.9)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(0.75, 0.75, keepSize - 1.5, keepSize - 1.5);
  }

  function renderExtended(canvas, px) {
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(N, N);
    const d = img.data;
    for (let i = 0; i < N2; i++) {
      const v = px[i];
      const idx = i * 4;
      d[idx + 3] = 255;
      if (v < 0) {
        d[idx] = 0; d[idx + 1] = 0;
        d[idx + 2] = Math.min(255, Math.round(80 + Math.min(175, -v * 0.4)));
      } else if (v > 255) {
        d[idx] = Math.min(255, Math.round(80 + Math.min(175, (v - 255) * 0.4)));
        d[idx + 1] = 0; d[idx + 2] = 0;
      } else {
        const g = Math.round(v);
        d[idx] = d[idx + 1] = d[idx + 2] = g;
      }
    }
    ctx.putImageData(img, 0, 0);
    const lo = Math.round((N - 1) / 4);
    const hi = Math.round(3 * (N - 1) / 4);
    ctx.strokeStyle = 'rgba(255, 80, 80, 0.85)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.strokeRect(lo, lo, hi - lo, hi - lo);
    ctx.setLineDash([]);
  }

  function renderTiled(destCanvas, srcCanvas) {
    const ctx = destCanvas.getContext('2d');
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    for (let ty = 0; ty < 2; ty++)
      for (let tx = 0; tx < 2; tx++)
        ctx.drawImage(srcCanvas, tx * HALF, ty * HALF, HALF, HALF);
    ctx.strokeStyle = 'rgba(255, 80, 80, 0.85)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(HALF, 0); ctx.lineTo(HALF, N);
    ctx.moveTo(0, HALF); ctx.lineTo(N, HALF);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  // ---- Fourier reconstruction ----

  const workRe = new Float64Array(N2);
  const workIm = new Float64Array(N2);

  function computeRecon(fRe, fIm, radius) {
    workRe.set(fRe);
    workIm.set(fIm);
    const r2 = radius * radius;
    for (let r = 0; r < N; r++) {
      const fr = r < HALF ? r : r - N;
      const fr2 = fr * fr;
      for (let c = 0; c < N; c++) {
        const fc = c < HALF ? c : c - N;
        if (fr2 + fc * fc > r2) { workRe[r * N + c] = 0; workIm[r * N + c] = 0; }
      }
    }
    ifft2d(workRe, workIm);
  }

  function countCoeffs(radius) {
    let count = 0;
    const r2 = radius * radius;
    for (let r = 0; r < N; r++) {
      const fr = r < HALF ? r : r - N;
      const fr2 = fr * fr;
      for (let c = 0; c < N; c++) {
        const fc = c < HALF ? c : c - N;
        if (fr2 + fc * fc <= r2) count++;
      }
    }
    return count;
  }

  // ---- Initialization ----

  function init() {
    const basisSel       = document.getElementById('fourierBasis');
    const presetSel      = document.getElementById('fourierPreset');
    const radiusSl       = document.getElementById('fourierRadius');
    const radiusOut      = document.getElementById('fourierRadiusValue');
    const radiusLabel    = document.getElementById('fourierRadiusLabel');
    const formulaBox     = document.getElementById('fourierFormulaBox');
    const tilingChk      = document.getElementById('fourierTiling');

    // Definitions for the formula-panel info icons, keyed by basis then by which
    // formula block (recon / basis / worked). Shown in a floating tooltip.
    const FORMULA_TERMS = {
      fourier: {
        recon: 'f[x,y]: the reconstructed pixel value.\nF[u,v]: the Fourier coefficient (complex) of the wave with u cycles across x and v across y.\n1 / N^2: normalization (N = 256).\nThe sum keeps every frequency inside the filter radius r.',
        basis: 'phi_{u,v}[x,y] = e^{i 2 pi (u x + v y) / N}: a 2D wave. Analysis uses its conjugate (e^{-i...}) to measure how much of each wave is present; synthesis adds the waves back, each scaled by F[u,v].',
        worked: 'F[1,1]: the coefficient of the lowest diagonal wave (one cycle in each direction). It is complex, and |F[1,1]| is its magnitude.'
      },
      poly: {
        recon: 'f(x,y): the reconstructed pixel value.\nC[j,k]: the weight of the basis product of Legendre degree j (in y) and k (in x), found by projecting the image onto that product.\nThe sum keeps all degrees up to the cutoff d.',
        basis: 'P_j: the Legendre polynomial of degree j (P_0 = 1, P_1 = x, P_2 = (3x^2 - 1)/2, ...).\nx-tilde: the pixel coordinate rescaled from [0, N) to [-1, 1].',
        worked: 'C[1,1]: the weight of P_1(x-tilde) P_1(y-tilde) = x-tilde * y-tilde, the diagonal tilt (saddle) term. It is large for the Quadrant image.'
      },
      cheb: {
        recon: 'f(x,y): the reconstructed pixel value.\nC[j,k]: the weight of the Chebyshev product of degree j (in y) and k (in x), from projecting the image onto it.\nThe sum keeps all degrees up to the cutoff d.',
        basis: 'T_j: the Chebyshev polynomial, T_j(x) = cos(j arccos x) (T_0 = 1, T_1 = x, T_2 = 2x^2 - 1, ...).\nx-tilde: the pixel coordinate rescaled to [-1, 1] using midpoint nodes.',
        worked: 'C[1,1]: the weight of T_1(x-tilde) T_1(y-tilde) = x-tilde * y-tilde, the diagonal tilt term.'
      },
      haar: {
        recon: 'f(x,y): the reconstructed pixel value.\nA_L: the coarse approximation kept at level L, the repeatedly averaged low-pass band (a low-resolution version of the image).\nLH_l, HL_l, HH_l: the detail subbands at level l, capturing vertical, horizontal, and diagonal edges. The sum adds detail from level 1 up to L.',
        basis: 'psi: the Haar wavelet, +1 on the first half of its interval and -1 on the second.\nphi: the scaling function (the averaging box) behind the A_L term.\npsi_{j,m}(t) = 2^{j/2} psi(2^j t - m): the wavelet at scale j and position m. Each 2D subband multiplies one of psi or phi per axis.',
        worked: 'd^{HH}_{1,1}: the coarsest-scale HH (diagonal) detail coefficient, the projection of the image onto psi(x) psi(y) over the whole image (the diagonal contrast between opposite quadrants).'
      }
    };
    let formulaTip = null;
    function ensureFormulaTip() {
      if (formulaTip) return formulaTip;
      formulaTip = document.createElement('div');
      formulaTip.className = 'fourier-tip';
      document.body.appendChild(formulaTip);
      return formulaTip;
    }
    function showFormulaTip(icon, clientX, clientY) {
      const defs = FORMULA_TERMS[basisSel.value];
      const tip = ensureFormulaTip();
      tip.textContent = (defs && defs[icon.getAttribute('data-term')]) || '';
      tip.dataset.term = icon.getAttribute('data-term');
      const margin = 8;
      tip.style.maxWidth = Math.min(320, window.innerWidth - 2 * margin) + 'px';
      tip.style.opacity = '1';
      // Measure after content is set, then keep the whole box on screen (the old
      // fixed offsets pushed it off the side on narrow / mobile viewports).
      const r = tip.getBoundingClientRect();
      let x = clientX + 14;
      let y = clientY + 14;
      if (x + r.width + margin > window.innerWidth) x = window.innerWidth - r.width - margin;
      if (x < margin) x = margin;
      if (y + r.height + margin > window.innerHeight) y = clientY - r.height - 14;
      if (y < margin) y = margin;
      tip.style.left = Math.round(x) + 'px';
      tip.style.top = Math.round(y) + 'px';
    }
    function hideFormulaTip() { if (formulaTip) formulaTip.style.opacity = '0'; }
    const canHover = window.matchMedia('(hover: hover)').matches;
    if (formulaBox) {
      // Hover devices: the tip follows the cursor while hovering an icon. On
      // touch this is skipped so the synthesized mousemove cannot dismiss a
      // tip opened by tapping.
      formulaBox.addEventListener('mousemove', function (e) {
        if (!canHover) return;
        const icon = e.target.closest ? e.target.closest('.fourier-info') : null;
        if (!icon) { hideFormulaTip(); return; }
        showFormulaTip(icon, e.clientX, e.clientY);
      });
      formulaBox.addEventListener('mouseleave', function () { if (canHover) hideFormulaTip(); });
      // Touch: tap an icon to toggle its tip (mousemove never fires on touch).
      formulaBox.addEventListener('click', function (e) {
        const icon = e.target.closest ? e.target.closest('.fourier-info') : null;
        if (!icon) return;
        e.stopPropagation();
        const term = icon.getAttribute('data-term');
        if (formulaTip && formulaTip.style.opacity === '1' && formulaTip.dataset.term === term) {
          hideFormulaTip();
          return;
        }
        const ir = icon.getBoundingClientRect();
        showFormulaTip(icon, ir.left + ir.width / 2, ir.bottom);
      });
      // Tap / click anywhere else dismisses the tip.
      document.addEventListener('click', function (e) {
        if (!e.target.closest || !e.target.closest('.fourier-info')) hideFormulaTip();
      });
    }
    const tilingText     = document.getElementById('fourierTilingText');
    const origLabel      = document.getElementById('fourierOrigLabel');
    const specLabel      = document.getElementById('fourierSpecLabel');
    const reconLabel     = document.getElementById('fourierReconLabel');
    const canvasOrig     = document.getElementById('fourierOrigCanvas');
    const canvasSpec     = document.getElementById('fourierSpecCanvas');
    const canvasRecon    = document.getElementById('fourierReconCanvas');
    const uploadInput    = document.getElementById('fourierUpload');

    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = N; srcCanvas.height = N;
    const reconSrcCanvas = document.createElement('canvas');
    reconSrcCanvas.width = N; reconSrcCanvas.height = N;

    let fftRe = null, fftIm = null;
    let polyC = null;
    let haarC = null;
    let chebC = null;

    function loadPixels(px) {
      renderGrayscale(srcCanvas, px);

      fftRe = new Float64Array(px);
      fftIm = new Float64Array(N2);
      fft2d(fftRe, fftIm);

      polyC = polyDecomp2D(px);

      haarC = new Float64Array(px);
      haarForward2D(haarC);

      chebC = chebDecomp2D(px);

      window.__fourierState = window.__fourierState || {};
      window.__fourierState.orig = px;
      window.dispatchEvent(new CustomEvent('fourier-orig-update', { detail: { pixels: px } }));

      render();
    }

    function onBasisChange() {
      const basis = basisSel.value;
      if (basis === 'fourier') {
        radiusSl.min = 1; radiusSl.max = 128; radiusSl.step = 1;
        if (+radiusSl.value > 128) radiusSl.value = 20;
        radiusLabel.textContent = 'Low-pass filter radius';
        tilingText.textContent  = 'Show periodic extension (2×2 tile)';
        specLabel.textContent   = 'Frequency Spectrum (log magnitude)';
      } else if (basis === 'poly') {
        radiusSl.min = 0; radiusSl.max = MAX_POLY_DEG; radiusSl.step = 1;
        if (+radiusSl.value > MAX_POLY_DEG) radiusSl.value = MAX_POLY_DEG;
        if (+radiusSl.value < 5) radiusSl.value = 5;
        radiusLabel.textContent = 'Max polynomial degree';
        tilingText.textContent  = 'Show polynomial extrapolation (2× domain)';
        specLabel.textContent   = 'Coefficient Matrix |C[j,k]|';
      } else if (basis === 'haar') {
        radiusSl.min = 0; radiusSl.max = MAX_WAVELET_LEVELS; radiusSl.step = 1;
        if (+radiusSl.value > MAX_WAVELET_LEVELS) radiusSl.value = MAX_WAVELET_LEVELS;
        if (+radiusSl.value < 3) radiusSl.value = 3;
        radiusLabel.textContent = 'Levels to keep';
        tilingText.textContent  = 'Show periodic extension (2×2 tile)';
        specLabel.textContent   = 'Wavelet Pyramid (per-subband normalized)';
      } else { // cheb
        radiusSl.min = 0; radiusSl.max = MAX_CHEB_DEG; radiusSl.step = 1;
        if (+radiusSl.value > MAX_CHEB_DEG) radiusSl.value = MAX_CHEB_DEG;
        if (+radiusSl.value < 5) radiusSl.value = 5;
        radiusLabel.textContent = 'Max Chebyshev degree';
        tilingText.textContent  = 'Show Chebyshev extrapolation (2× domain)';
        specLabel.textContent   = 'Chebyshev Coefficient Matrix |C[j,k]|';
      }
      render();
    }

    // Live formula box: reconstruction sum (with the current cutoff substituted)
    // plus the basis function for the selected basis. Debounced so dragging the
    // cutoff slider does not re-typeset MathJax on every tick.
    let formulaTimer = null;
    function scheduleFormula(basis, val) {
      if (!formulaBox) return;
      if (formulaTimer) clearTimeout(formulaTimer);
      formulaTimer = setTimeout(function () { updateFormula(basis, val); }, 80);
    }
    function typesetFormula() {
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([formulaBox]).catch(function () {});
      } else {
        setTimeout(typesetFormula, 150);
      }
    }
    function fmtCoeff(x) {
      if (!isFinite(x)) return '0';
      // Group large integers with LaTeX thin spaces (\,) instead of commas, which
      // MathJax would render as punctuation with a trailing space.
      if (Math.abs(x) >= 1000) {
        return String(Math.round(x)).replace(/\B(?=(\d{3})+(?!\d))/g, '\\,');
      }
      return (+x).toFixed(2);
    }
    function updateFormula(basis, val) {
      if (!formulaBox) return;
      let recon, basisFn, worked;
      if (basis === 'fourier') {
        recon = 'f[x,y] \\approx \\frac{1}{N^2} \\sum_{u^2+v^2 \\le ' + val + '^2} F[u,v]\\, e^{\\,i 2\\pi(ux+vy)/N}';
        basisFn = '\\varphi_{u,v}[x,y] = e^{\\,i 2\\pi(ux+vy)/N}';
        const re = fftRe ? fftRe[N + 1] : 0;
        const im = fftIm ? fftIm[N + 1] : 0;
        const mag = Math.sqrt(re * re + im * im);
        worked = 'F[1,1] = ' + fmtCoeff(re) + (im >= 0 ? ' + ' : ' - ') + fmtCoeff(Math.abs(im)) + 'i, \\qquad |F[1,1]| = ' + fmtCoeff(mag) + '\\ \\ (\\text{lowest diagonal wave})';
      } else if (basis === 'poly') {
        recon = 'f(x,y) \\approx \\sum_{j=0}^{' + val + '} \\sum_{k=0}^{' + val + '} C[j,k]\\, P_j(\\tilde x)\\, P_k(\\tilde y)';
        basisFn = 'P_j(\\tilde x) = \\frac{1}{2^j\\, j!}\\,\\frac{d^j}{d\\tilde x^{\\,j}}\\big(\\tilde x^2 - 1\\big)^j, \\quad \\tilde x = \\tfrac{2x}{N-1} - 1';
        const c = polyC ? polyC[(MAX_POLY_DEG + 1) + 1] : 0;
        worked = 'C[1,1] = ' + fmtCoeff(c) + ', \\qquad C[1,1]\\, P_1(\\tilde x)\\, P_1(\\tilde y) = ' + fmtCoeff(c) + '\\,\\tilde x\\,\\tilde y \\ \\ (P_1(\\tilde x) = \\tilde x)';
      } else if (basis === 'cheb') {
        recon = 'f(x,y) \\approx \\sum_{j=0}^{' + val + '} \\sum_{k=0}^{' + val + '} C[j,k]\\, T_j(\\tilde x)\\, T_k(\\tilde y)';
        basisFn = 'T_j(\\tilde x) = \\cos\\!\\big(j \\arccos \\tilde x\\big), \\quad \\tilde x = \\tfrac{2x+1}{N} - 1';
        const c = chebC ? chebC[(MAX_CHEB_DEG + 1) + 1] : 0;
        worked = 'C[1,1] = ' + fmtCoeff(c) + ', \\qquad C[1,1]\\, T_1(\\tilde x)\\, T_1(\\tilde y) = ' + fmtCoeff(c) + '\\,\\tilde x\\,\\tilde y \\ \\ (T_1(\\tilde x) = \\tilde x)';
      } else {
        recon = 'f(x,y) \\approx A_{' + val + '} + \\sum_{\\ell=1}^{' + val + '} \\big( LH_\\ell + HL_\\ell + HH_\\ell \\big)';
        basisFn = '\\psi(t) = \\begin{cases} +1 & 0 \\le t < \\tfrac12 \\\\ -1 & \\tfrac12 \\le t < 1 \\\\ 0 & \\text{otherwise} \\end{cases}, \\quad \\psi_{j,m}(t) = 2^{j/2}\\psi(2^j t - m)';
        const a = haarC ? haarC[N + 1] : 0;
        worked = 'd^{\\,HH}_{1,1} = ' + fmtCoeff(a) + '\\ \\ (\\text{coarsest diagonal detail coefficient})';
      }
      formulaBox.innerHTML =
        '<div class="formulas">' +
          '<div class="formula"><div class="fourier-formula-label">Reconstruction <span class="fourier-info" data-term="recon">i</span></div>$$' + recon + '$$</div>' +
          '<div class="formula"><div class="fourier-formula-label">Basis function <span class="fourier-info" data-term="basis">i</span></div>$$' + basisFn + '$$</div>' +
        '</div>' +
        '<div class="formula fourier-formula-example"><div class="fourier-formula-label">Worked example (current image) <span class="fourier-info" data-term="worked">i</span></div>$$' + worked + '$$</div>';
      typesetFormula();
    }

    function render() {
      const basis      = basisSel.value;
      const val        = parseInt(radiusSl.value, 10);
      const showExt    = tilingChk.checked;

      // --- coefficient count label ---
      if (basis === 'fourier') {
        const n = countCoeffs(val);
        radiusOut.textContent = `r = ${val}  (${n.toLocaleString()} coefficients, ${Math.round(n/N2*100)}% of ${N}×${N})`;
      } else if (basis === 'poly') {
        const n = (val + 1) ** 2;
        radiusOut.textContent = `degree = ${val}  (${n} coefficients, ${Math.round(n/N2*100)}% of ${N}×${N})`;
      } else if (basis === 'haar') {
        const n = (1 << val) ** 2;
        radiusOut.textContent = `${val} level${val !== 1 ? 's' : ''}  (${n.toLocaleString()} coefficients, ${Math.round(n/N2*100)}% of ${N}×${N})`;
      } else {
        const n = (val + 1) ** 2;
        radiusOut.textContent = `degree = ${val}  (${n} coefficients, ${Math.round(n/N2*100)}% of ${N}×${N})`;
      }

      // --- live reconstruction formula box ---
      scheduleFormula(basis, val);

      // --- left panel: original or extension/extrapolation ---
      if (showExt) {
        if (basis === 'poly') {
          origLabel.textContent = 'Polynomial Extrapolation (2× domain)';
          if (polyC) renderExtended(canvasOrig, polyRecon2D(polyC, MAX_POLY_DEG, legExt));
        } else if (basis === 'cheb') {
          origLabel.textContent = 'Chebyshev Extrapolation (2× domain)';
          if (chebC) renderExtended(canvasOrig, chebRecon2D(chebC, MAX_CHEB_DEG, chebExt));
        } else {
          origLabel.textContent = 'Periodic Extension (2×2 tile)';
          renderTiled(canvasOrig, srcCanvas);
        }
      } else {
        origLabel.textContent = 'Original Image';
        const ctx = canvasOrig.getContext('2d');
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(srcCanvas, 0, 0);
      }

      // --- center panel: spectrum / coeff matrix / wavelet pyramid ---
      if (basis === 'fourier' && fftRe) {
        renderSpectrum(canvasSpec, fftRe, fftIm, val);
      } else if (basis === 'poly' && polyC) {
        renderCoeffMatrix(canvasSpec, polyC, val);
      } else if (basis === 'haar' && haarC) {
        renderWaveletCoeffs(canvasSpec, haarC, val);
      } else if (basis === 'cheb' && chebC) {
        renderCoeffMatrix(canvasSpec, chebC, val);
      }

      // --- right panel: reconstruction ---
      function drawRecon(px) {
        renderGrayscale(reconSrcCanvas, px);
        if (showExt) {
          renderTiled(canvasRecon, reconSrcCanvas);
        } else {
          const ctx = canvasRecon.getContext('2d');
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(reconSrcCanvas, 0, 0);
        }
      }

      let reconPx = null;

      if (basis === 'fourier' && fftRe) {
        reconLabel.textContent = val >= 127 ? 'Reconstruction (full)' : `Reconstruction (r = ${val})`;
        computeRecon(fftRe, fftIm, val);
        drawRecon(workRe);
        reconPx = workRe;
      } else if (basis === 'poly' && polyC) {
        reconLabel.textContent = `Reconstruction (degree = ${val})`;
        reconPx = polyRecon2D(polyC, val, showExt ? legExt : legStd);
        if (showExt) renderExtended(canvasRecon, reconPx);
        else drawRecon(reconPx);
      } else if (basis === 'haar' && haarC) {
        reconLabel.textContent = val >= MAX_WAVELET_LEVELS ? 'Reconstruction (full)' : `Reconstruction (${val} levels)`;
        const truncated = new Float64Array(haarC);
        const keepSize = 1 << val;
        for (let r = 0; r < N; r++)
          for (let c = 0; c < N; c++)
            if (r >= keepSize || c >= keepSize) truncated[r * N + c] = 0;
        haarInverse2D(truncated);
        drawRecon(truncated);
        reconPx = truncated;
      } else if (basis === 'cheb' && chebC) {
        reconLabel.textContent = val >= MAX_CHEB_DEG ? 'Reconstruction (full)' : `Reconstruction (degree = ${val})`;
        reconPx = chebRecon2D(chebC, val, showExt ? chebExt : chebStd);
        if (showExt) renderExtended(canvasRecon, reconPx);
        else drawRecon(reconPx);
      }

      if (reconPx) {
        window.__fourierState = window.__fourierState || {};
        window.__fourierState.recon = reconPx;
        window.dispatchEvent(new CustomEvent('fourier-recon-update', { detail: { pixels: reconPx } }));
      }
    }

    basisSel.addEventListener('change', onBasisChange);
    radiusSl.addEventListener('input', render);
    tilingChk.addEventListener('change', render);

    // The Image dropdown combines presets with an "upload" action. The action
    // option triggers the hidden file input; the upload handler then inserts
    // a filename option above "Choose file…" so the user can switch back to
    // the uploaded image later.
    const UPLOAD_SLOT   = '__uploaded__';
    let uploadedPixels = null;

    // The file dialog is opened by tapping the "Choose file…" label that wraps
    // the hidden input (see fourier.html). iOS Safari only opens a file picker
    // from a direct user gesture on the input/label, so we must not trigger it
    // programmatically from this select's change handler.
    presetSel.addEventListener('change', () => {
      const v = presetSel.value;
      if (v === UPLOAD_SLOT && uploadedPixels) {
        loadPixels(uploadedPixels);
      } else {
        loadPixels(makePreset(v));
      }
    });

    uploadInput.addEventListener('change', function () {
      const file = this.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function (e) {
        const img = new Image();
        img.onload = function () {
          const tmp = document.createElement('canvas');
          tmp.width = N; tmp.height = N;
          const tctx = tmp.getContext('2d');
          const side = Math.min(img.width, img.height);
          tctx.drawImage(img, (img.width-side)/2, (img.height-side)/2, side, side, 0, 0, N, N);
          const data = tctx.getImageData(0, 0, N, N).data;
          const px = new Float64Array(N2);
          for (let i = 0; i < N2; i++)
            px[i] = 0.299*data[i*4] + 0.587*data[i*4+1] + 0.114*data[i*4+2];
          uploadedPixels = px;

          let slot = presetSel.querySelector(`option[value="${UPLOAD_SLOT}"]`);
          if (!slot) {
            slot = document.createElement('option');
            slot.value = UPLOAD_SLOT;
            presetSel.appendChild(slot);
          }
          slot.textContent = file.name;
          presetSel.value = UPLOAD_SLOT;
          uploadInput.value = '';
          loadPixels(px);
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    });

    loadPixels(makePreset(presetSel.value));
  }

  document.addEventListener('DOMContentLoaded', init);
})();
