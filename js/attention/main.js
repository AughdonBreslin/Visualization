// js/attention/main.js
import { initPipelineBar } from './pipeline.js';
import { renderAllScenes } from './scenes.js';
import { computePipeline } from './math.js';
import { PRESETS, WEIGHTS, TOKEN_COLORS } from './presets.js';
import { typesetMath } from './mathjax.js';
import { initFilmstrips } from './filmstrip.js';

let currentPreset = PRESETS[0];
let currentCausal = false;

function buildAndRenderAll() {
  const result = computePipeline(currentPreset.tokens, currentPreset.embeddings, WEIGHTS, { causal: currentCausal });
  result.tokenColors = TOKEN_COLORS;
  renderAllScenes(result);
  updatePickerActiveState();
  // Some concept-stage prose is real MathJax notation ($...$), injected after MathJax's own
  // initial pass over the static page, so it needs an explicit re-typeset every render.
  typesetMath(document.querySelector('.article-body'));
  // renderAllScenes rebuilds every filmstrip's DOM from scratch, so its arrows/dots/drag
  // handling needs rewiring every time too.
  initFilmstrips();
}

// The mask scene's checkbox toggles causal masking, which changes weights and output
// everywhere downstream, so it must re-render every scene, not just its own.
// Exposed on window rather than imported by scenes.js, so scenes.js has no reverse dependency
// on main.js (scenes.js only ever receives a PipelineResult, it never triggers recomputation).
window.attentionSetCausal = function setCausal(causal) {
  currentCausal = causal;
  buildAndRenderAll();
};

function updatePickerActiveState() {
  const picker = document.getElementById('presetPicker');
  if (!picker) return;
  picker.querySelectorAll('.preset-btn').forEach((btn) => {
    btn.classList.toggle('is-active', btn.dataset.presetId === currentPreset.id);
  });
}

function initPresetPicker() {
  const picker = document.getElementById('presetPicker');
  if (!picker) return;
  picker.innerHTML = PRESETS.map(
    (p) => `<button type="button" class="preset-btn" data-preset-id="${p.id}">${p.label}</button>`
  ).join('');
  picker.addEventListener('click', (e) => {
    const btn = e.target.closest('.preset-btn');
    if (!btn) return;
    const preset = PRESETS.find((p) => p.id === btn.dataset.presetId);
    if (!preset) return;
    currentPreset = preset;
    buildAndRenderAll();
  });
}

function init() {
  const barEl = document.getElementById('pipeBar');
  if (barEl) initPipelineBar(barEl);
  initPresetPicker();
  buildAndRenderAll();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
