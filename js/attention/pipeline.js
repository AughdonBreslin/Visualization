// js/attention/pipeline.js
// The sticky pipeline bar: the page's primary navigation. Also wires the slim secondary rail
// (built by the shared js/section-outline.js) to trigger the identical open-node behavior,
// so both entry points feel like the same action, per
// docs/superpowers/specs/2026-07-19-attention-visualization-design.md.

import { STEP_IDS, glyphSVG, connectorsSVG } from './glyphs.js';

export const STEPS = [
  { id: 'input', label: 'Input' },
  { id: 'qkv', label: 'Q / K / V' },
  { id: 'scores', label: 'QKᵀ' },
  { id: 'mask', label: 'Mask' },
  { id: 'softmax', label: 'Softmax' },
  { id: 'wsum', label: 'W. sum' },
  { id: 'output', label: 'Output' },
];

let barNodesById = new Map();

export function openNode(id) {
  const barNode = barNodesById.get(id);
  if (barNode) {
    barNode.classList.add('pulse');
    setTimeout(() => barNode.classList.remove('pulse'), 500);
  }
  const scene = document.getElementById(`step-${id}`);
  if (!scene) return;
  scene.scrollIntoView({ behavior: 'smooth', block: 'start' });
  setTimeout(() => {
    const glyph = scene.querySelector('.scene-hero-glyph');
    if (!glyph) return;
    glyph.classList.add('just-arrived');
    setTimeout(() => glyph.classList.remove('just-arrived'), 950);
  }, 550);
}

function buildBar(barEl) {
  barEl.innerHTML = `
    <div class="pipe-scroll">
      <svg class="pipe-svg" viewBox="0 0 700 60" preserveAspectRatio="none">${connectorsSVG(STEPS.length, { width: 700, height: 60 })}</svg>
      <div class="pipe-row"></div>
    </div>
  `;
  const row = barEl.querySelector('.pipe-row');
  barNodesById = new Map();
  for (const step of STEPS) {
    const node = document.createElement('button');
    node.type = 'button';
    node.className = 'pipe-node';
    node.dataset.target = step.id;
    node.innerHTML = `${glyphSVG(step.id)}<div class="pipe-node-label">${step.label}</div>`;
    node.addEventListener('click', () => openNode(step.id));
    row.appendChild(node);
    barNodesById.set(step.id, node);
  }
}

function wireCurrentStepHighlighting() {
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const id = entry.target.id.replace('step-', '');
        for (const [nodeId, node] of barNodesById) {
          const isCurrent = nodeId === id;
          node.classList.toggle('current', isCurrent);
          // On mobile .pipe-bar is a horizontal scroller (see attention.css); follow the active
          // step so it's never left scrolled out of view as the page scrolls. inline: 'center'
          // scrolls that horizontal axis only -- block: 'nearest' stops this from also nudging
          // the page's own vertical scroll, which is already what triggered this in the first
          // place. A no-op on desktop, where the bar never overflows.
          if (isCurrent) node.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
        }
      }
    },
    { rootMargin: '-100px 0px -70% 0px' }
  );
  for (const id of STEP_IDS) {
    const scene = document.getElementById(`step-${id}`);
    if (scene) observer.observe(scene);
  }
}

// The rail is built by js/section-outline.js from the same <section class="panel"><h2> blocks
// used here, generating <a data-target="step-xxx"> links. Delegate on document.body (stable at
// load time) rather than querying the rail directly, since section-outline.js may build its DOM
// after this module runs.
function wireRailDelegation() {
  document.body.addEventListener('click', (e) => {
    const a = e.target.closest('.section-outline-list a[data-target]');
    if (!a) return;
    const id = a.dataset.target.replace('step-', '');
    if (!STEP_IDS.includes(id)) return; // not one of our panels (defensive; shouldn't happen)
    // Let section-outline.js's own handler run too (its e.preventDefault + navigateTo already
    // scrolls); this listener only adds the pulse/glow flourish on top of it.
    setTimeout(() => openNode(id), 0);
  });
}

export function initPipelineBar(barEl) {
  buildBar(barEl);
  wireCurrentStepHighlighting();
  wireRailDelegation();
}
