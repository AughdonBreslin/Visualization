// js/attention/scenes.js
// Renders each step's hero glyph and worked-example animation. renderScene's per-step branches
// start as placeholders here and are filled in one step at a time by Tasks 6-9; the dispatch
// structure itself does not change after this task.

import { glyphSVG } from './glyphs.js';
import { STEPS } from './pipeline.js';

function renderPlaceholder(container, stepId) {
  container.innerHTML = `<p style="font:500 12px/1.6 var(--font-mono); color:var(--text-muted); margin:0;">step "${stepId}" animation not yet implemented</p>`;
}

function renderInput(container, stepId, result) {
  const rows = result.tokens.map((t, i) => {
    const vec = result.embeddings[t].map((v) => v.toFixed(2)).join(', ');
    return `<div class="vec-row"><span class="vec-token" style="color:${result.tokenColors[i]}">"${t}"</span><span class="vec-values">[${vec}]</span></div>`;
  }).join('');
  container.innerHTML = `<div class="vec-list">${rows}</div>`;
}

function renderQkv(container, stepId, result) {
  const rows = result.tokens.map((t, i) => {
    const x = result.embeddings[t].map((v) => v.toFixed(2)).join(', ');
    const q = result.Q[t].map((v) => v.toFixed(2)).join(', ');
    const k = result.K[t].map((v) => v.toFixed(2)).join(', ');
    const v = result.V[t].map((v) => v.toFixed(2)).join(', ');
    return `
      <div class="qkv-row">
        <span class="vec-token" style="color:${result.tokenColors[i]}">"${t}"</span>
        <span class="qkv-eq">x=[${x}]</span>
        <span class="qkv-eq">q=[${q}]</span>
        <span class="qkv-eq">k=[${k}]</span>
        <span class="qkv-eq">v=[${v}]</span>
      </div>`;
  }).join('');
  container.innerHTML = `<div class="qkv-list">${rows}</div>
    <div class="anim-controls"><button class="anim-btn" type="button" data-role="step">step through per-token multiply-sum &#9654;</button></div>
    <div class="formula-worked" data-role="worked"></div>`;

  // Step control cycles which token's row is highlighted, to focus attention on one
  // multiply-then-sum at a time rather than showing all three at once. The formula-worked
  // line always mirrors whichever token is currently highlighted, so the abstract formula in
  // this step's .formula block has a live, real-number companion to trace against.
  const btn = container.querySelector('[data-role="step"]');
  const worked = container.querySelector('[data-role="worked"]');
  const rowsEls = Array.from(container.querySelectorAll('.qkv-row'));
  const fmt = (arr) => `[${arr.map((v) => v.toFixed(2)).join(', ')}]`;
  const showWorked = (i) => {
    const t = result.tokens[i];
    worked.textContent = `q_"${t}" = x_"${t}" · W_Q = ${fmt(result.embeddings[t])} · W_Q = ${fmt(result.Q[t])}`;
  };
  let idx = 0;
  showWorked(idx);
  btn.addEventListener('click', () => {
    rowsEls.forEach((r) => r.classList.remove('is-active'));
    idx = (idx + 1) % rowsEls.length;
    rowsEls[idx].classList.add('is-active');
    showWorked(idx);
  });
}

function buildHeatGrid(matrix, tokens, opts = {}) {
  const { minOpacity = 0.15, maxOpacity = 0.8, formatter = (v) => v.toFixed(2), maskedCells = () => false } = opts;
  const flat = matrix.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const range = max - min || 1;
  let cells = '';
  matrix.forEach((row, i) => {
    row.forEach((v, j) => {
      if (maskedCells(i, j)) {
        cells += `<div class="heat-cell masked" data-row="${i}" data-col="${j}">&minus;&infin;</div>`;
        return;
      }
      const t = (v - min) / range;
      const opacity = (minOpacity + t * (maxOpacity - minOpacity)).toFixed(2);
      cells += `<div class="heat-cell" data-row="${i}" data-col="${j}" style="background:rgba(var(--accent-rgb), ${opacity})">${formatter(v)}</div>`;
    });
  });
  return `<div class="heat-grid">${cells}</div>`;
}

function renderScores(container, stepId, result) {
  const grid = buildHeatGrid(result.scores, result.tokens);
  const legend = result.tokens.map((t, i) => `<span style="color:${result.tokenColors[i]}">"${t}"</span>`).join(', ');
  const t0 = result.tokens[0];
  const worked = `score_"${t0}","${t0}" = q_"${t0}" &middot; k_"${t0}" = ${result.scores[0][0].toFixed(2)}`;
  container.innerHTML = `
    <div class="heat-block">
      ${grid}
      <div class="heat-meta">rows = query token, columns = key token<br>tokens: ${legend}<br>score[i,j] = q&#8571; &middot; k&#8571;</div>
    </div>
    <div class="formula-worked">${worked}</div>`;
}

function renderScale(container, stepId, result) {
  const beforeGrid = buildHeatGrid(result.scores, result.tokens, { minOpacity: 0.1, maxOpacity: 0.5 });
  const afterGrid = buildHeatGrid(result.scaled, result.tokens, { minOpacity: 0.15, maxOpacity: 0.8 });
  const t0 = result.tokens[0];
  const worked = `scaled_"${t0}","${t0}" = ${result.scores[0][0].toFixed(2)} / &radic;${result.d} = ${result.scaled[0][0].toFixed(2)}`;
  container.innerHTML = `
    <div class="heat-block scale-compare">
      <div><div class="heat-caption">before (&divide;1)</div>${beforeGrid}</div>
      <div class="scale-arrow">&divide;&radic;${result.d} &rarr;</div>
      <div><div class="heat-caption">after (&divide;&radic;${result.d})</div>${afterGrid}</div>
    </div>
    <div class="formula-worked">${worked}</div>`;
}

function renderMask(container, stepId, result) {
  const isMasked = (i, j) => result.causal && j > i;
  const grid = buildHeatGrid(result.masked.map((row) => row.map((v) => (v <= -1e8 ? 0 : v))), result.tokens, { maskedCells: isMasked });
  // The formula-worked line has no separate .formula block to pair with (this step's rule is a
  // conditional, not a single equation), so it stands alone as the numeric instantiation of the
  // masking rule itself: which specific cell gets masked and why, or a clear statement that
  // nothing is masked yet.
  const worked = result.causal
    ? (() => {
        const t0 = result.tokens[0];
        const t1 = result.tokens[1];
        return `score_"${t0}","${t1}" &rarr; &minus;&infin; (masked: j=1 &gt; i=0)`;
      })()
    : 'causal mask is off: no cells masked, every token can see every other token';
  container.innerHTML = `
    <div class="heat-block">
      ${grid}
      <div class="heat-meta">
        <label class="mask-toggle"><input type="checkbox" data-role="causal-toggle" ${result.causal ? 'checked' : ''}> causal mask (each token can only see itself and earlier tokens)</label>
      </div>
    </div>
    <div class="formula-worked">${worked}</div>`;
  const toggle = container.querySelector('[data-role="causal-toggle"]');
  toggle.addEventListener('change', () => {
    window.attentionSetCausal(toggle.checked);
  });
}

function renderSoftmax(container, stepId, result) {
  const grid = buildHeatGrid(result.weights, result.tokens, { minOpacity: 0.15, maxOpacity: 0.85 });
  const bars = result.tokens.map((ti, i) => {
    const segs = result.tokens.map((tj, j) => {
      const pct = (result.weights[i][j] * 100).toFixed(1);
      return `<span class="softmax-seg" style="width:${pct}%; background:${result.tokenColors[j]}" title="${tj}: ${pct}%"></span>`;
    }).join('');
    return `<div class="softmax-bar-row"><span class="vec-token" style="color:${result.tokenColors[i]}">"${ti}"</span><div class="softmax-bar">${segs}</div></div>`;
  }).join('');
  const t0 = result.tokens[0];
  const row0 = result.masked[0].filter((v) => v > -1e8);
  const expSum = row0.reduce((sum, v) => sum + Math.exp(v), 0);
  const worked = `weight_"${t0}","${t0}" = e<sup>${row0[0].toFixed(2)}</sup> / ${expSum.toFixed(2)} = ${result.weights[0][0].toFixed(2)}`;
  container.innerHTML = `
    <div class="heat-block">${grid}<div class="heat-meta">each row sums to 1.00: the actual attention weights</div></div>
    <div class="softmax-bars">${bars}</div>
    <div class="formula-worked">${worked}</div>`;
}

const STEP_RENDERERS = {
  input: renderInput,
  qkv: renderQkv,
  scores: renderScores,
  scale: renderScale,
  mask: renderMask,
  softmax: renderSoftmax,
  wsum: renderPlaceholder,
  output: renderPlaceholder,
};

export function renderScene(stepId, result) {
  const scene = document.getElementById(`step-${stepId}`);
  if (!scene) return;
  const heroEl = scene.querySelector('.scene-hero-glyph');
  if (heroEl) heroEl.innerHTML = glyphSVG(stepId, { tokenColors: result.tokenColors });
  const animEl = scene.querySelector('.scene-anim');
  if (animEl) {
    const renderer = STEP_RENDERERS[stepId] || renderPlaceholder;
    renderer(animEl, stepId, result);
  }
}

export function renderAllScenes(result) {
  for (const step of STEPS) renderScene(step.id, result);
}
