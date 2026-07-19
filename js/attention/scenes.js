// js/attention/scenes.js
// Renders each step's hero glyph and worked-example animation. renderScene's per-step branches
// start as placeholders here and are filled in one step at a time by Tasks 6-9; the dispatch
// structure itself does not change after this task.

import { glyphSVG } from './glyphs.js';
import { STEPS } from './pipeline.js';

function renderPlaceholder(container, stepId) {
  container.innerHTML = `<p style="font:500 12px/1.6 var(--font-mono); color:var(--text-muted); margin:0;">step "${stepId}" animation not yet implemented</p>`;
}

const STEP_RENDERERS = {
  input: renderPlaceholder,
  qkv: renderPlaceholder,
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
