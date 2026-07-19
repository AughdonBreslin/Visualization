// js/attention/main.js
import { initPipelineBar } from './pipeline.js';
import { renderAllScenes } from './scenes.js';
import { computePipeline } from './math.js';
import { PRESETS, WEIGHTS, TOKEN_COLORS } from './presets.js';

function buildResult(preset, options) {
  const result = computePipeline(preset.tokens, preset.embeddings, WEIGHTS, options);
  result.tokenColors = TOKEN_COLORS;
  return result;
}

function init() {
  const barEl = document.getElementById('pipeBar');
  if (barEl) initPipelineBar(barEl);

  const defaultPreset = PRESETS[0];
  const result = buildResult(defaultPreset, { causal: false });
  renderAllScenes(result);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
