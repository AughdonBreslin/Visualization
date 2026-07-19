// js/attention/main.js
import { initPipelineBar } from './pipeline.js';

function init() {
  const barEl = document.getElementById('pipeBar');
  if (barEl) initPipelineBar(barEl);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
