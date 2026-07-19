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

const STEP_RENDERERS = {
  input: renderInput,
  qkv: renderQkv,
  scores: renderPlaceholder,
  scale: renderPlaceholder,
  mask: renderPlaceholder,
  softmax: renderPlaceholder,
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
