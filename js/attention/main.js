// js/attention/main.js
import { initPipelineBar } from './pipeline.js';
import { renderAllScenes } from './scenes.js';
import { computePipeline } from './math.js';
import { PRESETS, WEIGHTS, TOKEN_COLORS } from './presets.js';

let currentPreset = PRESETS[0];
let currentCausal = false;

function buildAndRenderAll() {
  const result = computePipeline(currentPreset.tokens, currentPreset.embeddings, WEIGHTS, { causal: currentCausal });
  result.tokenColors = TOKEN_COLORS;
  renderAllScenes(result);
}

// The mask scene's checkbox toggles causal masking, which changes weights and output
// everywhere downstream, so it must re-render every scene, not just its own.
// Exposed on window rather than imported by scenes.js, so scenes.js has no reverse dependency
// on main.js (scenes.js only ever receives a PipelineResult, it never triggers recomputation).
window.attentionSetCausal = function setCausal(causal) {
  currentCausal = causal;
  buildAndRenderAll();
};

function init() {
  const barEl = document.getElementById('pipeBar');
  if (barEl) initPipelineBar(barEl);
  buildAndRenderAll();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
