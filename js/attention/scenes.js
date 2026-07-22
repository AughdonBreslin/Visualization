// js/attention/scenes.js
// Renders each step's hero glyph and worked-example animation. The step's motivation and concept
// live in the static intro paragraph in pages/attention.html, not in here; this file only builds
// the filmstrip of concrete, numbered stages ("storage / slice / transform", not every step uses
// all three) that walk through the worked example. Every value shown comes from the live
// PipelineResult, never a hardcoded number.

import { glyphSVG } from './glyphs.js';
import { STEPS } from './pipeline.js';
import { WEIGHTS } from './presets.js';

function renderPlaceholder(container, stepId) {
  container.innerHTML = `<p style="font:500 12px/1.6 var(--font-mono); color:var(--text-muted); margin:0;">step "${stepId}" animation not yet implemented</p>`;
}

// ---- shared visual primitives --------------------------------------------------------------

function maxAbs(arr) {
  return Math.max(...arr.map(Math.abs), 0.01);
}

// A single hue (the site's own accent blue-violet), climbing only in saturation and lightness
// with magnitude. A blue-green-red sweep reads as noise once many cells render at once (e.g.
// QKV's four full matrices); one hue keeps every step's heat bars and grids calm and consistent
// while still making low vs. high magnitude easy to tell apart at a glance.
function heatColor(t) {
  t = Math.max(0, Math.min(1, t));
  const h = 233;
  const s = 22 + t * 50;
  const l = 14 + t * 40;
  return `hsl(${h}, ${s.toFixed(0)}%, ${l.toFixed(0)}%)`;
}

// A vector rendered as one box per dimension, side by side in a single bracket-framed row: each
// box's fill color is its "temperature" (magnitude relative to this vector's own largest entry),
// with the raw signed value printed inside it. No label above a box by default -- a dimension
// index (d0, d1, ...) is noise for a plain embedding/Q/K/V display where no particular dimension
// is being singled out; pass opts.labels only when a box genuinely identifies something (a
// token, e.g. Softmax's slice row, which is indexed by key token rather than by dimension).
function heatBarList(values, opts = {}) {
  const m = maxAbs(values);
  const labels = opts.labels;
  return values.map((v, i) => {
    const t = Math.abs(v) / m;
    const labelHtml = labels ? `<div class="heatbox-label">${labels[i]}</div>` : '';
    return `<div class="heatbox-wrap">
      ${labelHtml}
      <div class="heatbox-cell" style="background:${heatColor(t)}">${v.toFixed(2)}</div>
    </div>`;
  }).join('');
}

// A matrix rendered as a heat-colored grid, same heat scale as heatBarList, with optional row
// labels (token names or "row N") and highlighting. maskedCells(i, j) marks a cell as masked
// (rendered as a flat "-inf" cell instead of a heat-colored number).
function heatMatrixGrid(matrix, opts = {}) {
  const flat = matrix.flat();
  const m = maxAbs(flat);
  const hiRow = opts.hiRow;
  const hiCell = opts.hiCell;
  const maskedCells = opts.maskedCells || (() => false);
  const cols = matrix[0].length;
  const cellsHtml = matrix.map((row, i) => row.map((v, j) => {
    if (maskedCells(i, j)) {
      return `<div class="mcell masked" data-row="${i}" data-col="${j}">&minus;&infin;</div>`;
    }
    const t = Math.abs(v) / m;
    const isHi = hiCell ? (hiCell[0] === i && hiCell[1] === j) : hiRow === i;
    const isDim = (hiRow !== undefined && hiRow !== i) || (hiCell && hiCell[0] !== i);
    return `<div class="mcell ${isHi ? 'hi' : ''} ${isDim && !isHi ? 'dim' : ''}" data-row="${i}" data-col="${j}" style="background:${heatColor(t)}">${v.toFixed(1)}</div>`;
  }).join('')).join('');
  const activeRow = hiCell ? hiCell[0] : hiRow;
  const rowLabelsHtml = opts.rowLabels
    ? `<div class="mgrid-rowlabels">${opts.rowLabels.map((l, i) => `<div class="mgrid-rowlabel" style="color:${i === activeRow ? 'var(--accent-link)' : 'var(--text-muted)'}">${l}</div>`).join('')}</div>`
    : '';
  return `<div class="mgrid-wrap">${rowLabelsHtml}<div class="mgrid g${matrix.length}x${cols}">${cellsHtml}</div></div>`;
}

// The "multiply position by position, then sum" breakdown shared by every dot-product-shaped
// transform (Q/K/V projection, QKT scores). Not reused by Scale/Mask/Softmax, whose transforms
// are a single elementwise operation rather than a multiply-and-reduce.
function multBreakdown(a, b, resultLabel, resultValue) {
  const rows = a.map((av, i) => {
    const bv = b[i];
    const prod = (av * bv).toFixed(3);
    return `<div class="mult-row">
      <span class="mult-dimlabel">d${i}</span>
      <span class="mult-chip" style="color:var(--accent-link)">${av.toFixed(2)}</span>
      <span class="mult-eq">&times;</span>
      <span class="mult-chip" style="color:#ffd97a">${bv.toFixed(2)}</span>
      <span class="mult-eq">=</span>
      <span class="mult-prod">${prod}</span>
    </div>`;
  }).join('');
  return `<div class="mult-list">${rows}</div><div class="sum-arrow">&darr; add the ${a.length} products</div><div class="sum-result">${resultLabel} = ${resultValue}</div>`;
}

function stageCard(n, title, proseHtml, bodyHtml, noteHtml, extraClass = '') {
  return `<div class="stage${extraClass ? ` ${extraClass}` : ''}">
    <div class="stage-connector"></div>
    <div class="stage-n">${n}</div>
    <div class="stage-title">${title}</div>
    ${proseHtml ? `<p class="stage-prose">${proseHtml}</p>` : ''}
    <div class="stage-body">${bodyHtml}</div>
    ${noteHtml ? `<div class="stage-note">${noteHtml}</div>` : ''}
  </div>`;
}

function filmstrip(stages) {
  const dots = stages
    .map((_, i) => `<button type="button" class="filmstrip-dot" data-role="fs-dot" data-index="${i}" aria-label="Go to stage ${i + 1} of ${stages.length}"></button>`)
    .join('');
  return `<div class="filmstrip-wrap" data-role="filmstrip-wrap">
    <div class="filmstrip" data-role="filmstrip">${stages.join('')}</div>
    <button type="button" class="filmstrip-arrow filmstrip-arrow-prev" data-role="fs-prev" aria-label="Previous stage">&#8249;</button>
    <button type="button" class="filmstrip-arrow filmstrip-arrow-next" data-role="fs-next" aria-label="Next stage">&#8250;</button>
    <div class="filmstrip-dots" data-role="fs-dots">${dots}</div>
  </div>`;
}

// ---- per-step renderers ----------------------------------------------------------------------

function renderInput(container, stepId, result) {
  const storageBody = `<div class="heatbar-block-row">${result.tokens
    .map((t) => labeledVecBlock(`&quot;${t}&quot;`, result.embeddings[t]))
    .join('')}</div>`;
  const stage1 = stageCard(
    '01: STORAGE',
    'The three embeddings',
    `Every token in this worked example starts as a ${result.d}-number vector called an <b>embedding</b>. Stacked together, the embeddings below form $X$, the matrix the rest of this pipeline operates on.`,
    storageBody,
    `${result.tokens.length} tokens, ${result.d} numbers each, stored in full`
  );
  container.innerHTML = filmstrip([stage1]);
}

// X fans out into three independent multiplications at once: one line in, three lines out,
// landing on W_Q/W_K/W_V stacked to the right. Scales up the same one-box-three-lines motif
// already used for this step's own hero glyph (js/attention/glyphs.js's svgQkv), just with real
// matrices instead of abstract dots. Lines only, no text or circles: preserveAspectRatio="none"
// (needed so this stretches to match the W-column's real, dynamic height) distorts any glyph or
// curve into a visibly skewed shape, but a straight line under non-uniform scaling is still a
// straight line, just at a different angle - so it's the one shape that tolerates this cleanly.
function qkvFanoutSVG() {
  return `<svg class="qkv-fanout" viewBox="0 0 70 220" preserveAspectRatio="none">
    <line x1="0" y1="110" x2="16" y2="110" stroke="var(--hairline-strong)" stroke-width="2"/>
    <line x1="16" y1="110" x2="54" y2="24" stroke="var(--hairline-strong)" stroke-width="1.6"/>
    <line x1="16" y1="110" x2="54" y2="110" stroke="var(--hairline-strong)" stroke-width="1.6"/>
    <line x1="16" y1="110" x2="54" y2="196" stroke="var(--hairline-strong)" stroke-width="1.6"/>
  </svg>`;
}

function renderQkv(container, stepId, result) {
  const t0 = result.tokens[0];
  const xMatrix = result.tokens.map((t) => result.embeddings[t]);
  const qMatrix = result.tokens.map((t) => result.Q[t]);
  const kMatrix = result.tokens.map((t) => result.K[t]);
  const vMatrix = result.tokens.map((t) => result.V[t]);
  const stage1 = stageCard(
    '01: STORAGE',
    'The full data at rest',
    `$X$ below stacks all the embeddings, one row per token. The three matrices on the right are the actual $W_Q$, $W_K$, and $W_V$ this model learned: $X$ gets multiplied by each of them independently, producing a query, a key, and a value.`,
    `<div class="qkv-storage-row">
       <div class="qkv-storage-block"><div class="heatbar-block-title">$X$: one row per token</div>${heatMatrixGrid(xMatrix, { rowLabels: result.tokens })}</div>
       ${qkvFanoutSVG()}
       <div class="qkv-storage-outputs">
         <div class="qkv-storage-block"><div class="heatbar-block-title">$W_Q$</div>${heatMatrixGrid(WEIGHTS.WQ)}</div>
         <div class="qkv-storage-block"><div class="heatbar-block-title">$W_K$</div>${heatMatrixGrid(WEIGHTS.WK)}</div>
         <div class="qkv-storage-block"><div class="heatbar-block-title">$W_V$</div>${heatMatrixGrid(WEIGHTS.WV)}</div>
       </div>
     </div>`,
    `one shared $X$, three independent multiplications`
  );
  const stage2 = stageCard(
    '02: SLICE',
    'One token, three projections',
    `Take one token's embedding, $x_{\\text{${t0}}}$, and multiply it by all three matrices at once: $x \\cdot W_Q$ gives its query, $x \\cdot W_K$ gives its key, $x \\cdot W_V$ gives its value, all independently and in parallel off the same input vector.`,
    `<div class="formula">$$ q_i = x_i W_Q, \\quad k_i = x_i W_K, \\quad v_i = x_i W_V $$</div>
     <div class="heatbar-block-row">
       ${labeledVecBlock(`$q_{\\text{${t0}}}$`, result.Q[t0])}
       ${labeledVecBlock(`$k_{\\text{${t0}}}$`, result.K[t0])}
       ${labeledVecBlock(`$v_{\\text{${t0}}}$`, result.V[t0])}
     </div>`,
    `one input vector, three separate matrix multiplies, three separate outputs`
  );
  const stage3 = stageCard(
    '03: TRANSFORM',
    'The same multiply, every token at once',
    `Each token in $X$ goes through $W_Q$, $W_K$, and $W_V$, producing the full $Q$, $K$, $V$ matrices below.`,
    `<div class="heatbar-block-row">
       <div class="qkv-storage-block"><div class="heatbar-block-title">$Q$</div>${heatMatrixGrid(qMatrix, { rowLabels: result.tokens, hiRow: 0 })}</div>
       <div class="qkv-storage-block"><div class="heatbar-block-title">$K$</div>${heatMatrixGrid(kMatrix, { rowLabels: result.tokens, hiRow: 0 })}</div>
       <div class="qkv-storage-block"><div class="heatbar-block-title">$V$</div>${heatMatrixGrid(vMatrix, { rowLabels: result.tokens, hiRow: 0 })}</div>
     </div>`,
    `${result.tokens.length} tokens &times; 3 matrices, all produced by the same multiply shown in stage 2`
  );
  const stage4 = stageCard(
    '04: RELATED RESEARCH',
    'Why three matrices, not one?',
    null,
    `<p class="concept-box">If $Q$ and $K$ shared a matrix, every token's query would equal its own key, forcing every attention pattern to be symmetric (token A attends to B exactly as much as B attends to A) and collapsing the asking and answering roles into one. A 2026 study, <a href="https://arxiv.org/abs/2606.04032" target="_blank" rel="noopener">Do Transformers Need Three Projections? Systematic Study of QKV Variants</a>, tested this directly: tying $Q$ and $K$ broke attention's directionality and hurt quality, while tying $K$ and $V$ instead held up, cutting the KV cache in half for only a &tilde;3% perplexity increase. The asymmetry makes sense: a key (&quot;how to be found&quot;) and a value (&quot;what to contribute&quot;) can share a representational space without conflict, but a token's query and key must stay free to diverge, or it could only ever attend to itself.</p>`
  );
  container.innerHTML = filmstrip([stage1, stage2, stage3, stage4]);
}

function renderScores(container, stepId, result) {
  const t0 = result.tokens[0];
  const stage1 = stageCard(
    '01: STORAGE',
    'Every query, every key',
    `The previous step already produced a query vector and a key vector for every token. $Q$ below stacks all the query vectors, one row per token; $K$ stacks all the key vectors the same way.`,
    `<div><div class="heatbar-block-title">$Q$: one row per token</div>${heatMatrixGrid(result.tokens.map((t) => result.Q[t]), { hiRow: 0, rowLabels: result.tokens })}</div>
     <div><div class="heatbar-block-title">$K$: one row per token</div>${heatMatrixGrid(result.tokens.map((t) => result.K[t]), { hiRow: 0, rowLabels: result.tokens })}</div>`,
    `every query paired with every key, comparisons still to come`
  );
  const stage2 = stageCard(
    '02: SLICE',
    'One query, one key',
    `To fill in exactly one cell of the score grid, where the first token's query meets the first token's key, row 0 column 0, we only need one row from $Q$ and one row from $K$, both highlighted above. Every other row belongs to a different cell and isn't used here.`,
    `<div><div class="heatbar-block-title">$q_{\\text{${t0}}}$</div><div class="heatbar-list">${heatBarList(result.Q[t0])}</div></div>
     <div><div class="heatbar-block-title">$k_{\\text{${t0}}}$</div><div class="heatbar-list">${heatBarList(result.K[t0])}</div></div>
     <div class="calc-line">${result.Q[t0].map((qv, j) => `${qv.toFixed(2)}&times;${result.K[t0][j].toFixed(2)}`).join(' + ')} = <b>${result.scores[0][0].toFixed(2)}</b></div>`,
    `this pair lands in score grid cell [0,0]: that's a dot product, the standard way to measure how aligned two vectors are`
  );
  const n = result.tokens.length;
  const blankGrid = `<div class="mgrid-wrap"><div class="mgrid-rowlabels">${result.tokens.map((t) => `<div class="mgrid-rowlabel" style="color:var(--text-muted)">${t}</div>`).join('')}</div><div class="mgrid g${n}x${n}">${result.tokens.map(() => result.tokens.map(() => '<div class="mcell pending">?</div>').join('')).join('')}</div></div>`;
  const stage3 = stageCard(
    '03: TRANSFORM',
    'Fill in the grid, one comparison at a time',
    `Every cell repeats the same operation: pair up one query row and one key row by position, multiply each pair, add the results. Click below to watch all cells compute at once.`,
    `<div class="formula">$$ \\text{score}_{ij} = q_i \\cdot k_j $$</div>
     <div class="scale-shrink-wrap"><div data-role="sweep-grid">${blankGrid}</div></div>
     <div class="anim-controls"><button class="anim-btn" type="button" data-role="sweep-btn">&#9654; compute all cells</button></div>`,
    `repeat for all combinations of tokens to fill out the grid`
  );
  const sqrtD = Math.sqrt(result.d);
  const stage4 = stageCard(
    '04: SCALE',
    'Shrink every cell by √d',
    `The raw score grows with $d$, the number of dimensions summed, so every score divides by $\\sqrt{d}$ to keep its scale roughly constant; $d = ${result.d}$ here, so $\\sqrt{d} = ${sqrtD.toFixed(2)}$.`,
    `<div class="formula">$$ \\text{scaled}_{ij} = \\frac{\\text{score}_{ij}}{\\sqrt{d}} $$</div>
     <div class="scale-shrink-wrap"><div data-role="scale-grid">${blankGrid}</div></div>`
  );
  container.innerHTML = filmstrip([stage1, stage2, stage3, stage4]);

  // The grid starts blank ("?" in every cell) and fills in all at once on click, each cell
  // fading in with a staggered delay -- a single grid materializing, not nine separate reveals
  // to click through, and visually distinct from Scale's shrink and QKV's static breakdown.
  // The same click also fills in stage4's grid with the scaled values, since scaling is just
  // that same score grid divided by sqrt(d) -- one click computes both.
  const sweepGrid = container.querySelector('[data-role="sweep-grid"]');
  const sweepBtn = container.querySelector('[data-role="sweep-btn"]');
  const scaleGrid = container.querySelector('[data-role="scale-grid"]');
  sweepBtn.addEventListener('click', () => {
    sweepGrid.innerHTML = heatMatrixGrid(result.scores, { rowLabels: result.tokens });
    sweepGrid.querySelectorAll('.mcell').forEach((cell, idx) => {
      cell.style.animationDelay = `${idx * 55}ms`;
      cell.classList.add('cell-fade-in');
    });
    scaleGrid.innerHTML = heatMatrixGrid(result.scaled, { rowLabels: result.tokens });
    scaleGrid.querySelectorAll('.mcell').forEach((cell, idx) => {
      cell.style.animationDelay = `${idx * 55}ms`;
      cell.classList.add('cell-fade-in');
    });
    sweepBtn.disabled = true;
    sweepBtn.textContent = 'all cells computed';
  });
}

function renderMask(container, stepId, result) {
  const toggleHtml = `<label class="mask-toggle"><input type="checkbox" data-role="causal-toggle" ${result.causal ? 'checked' : ''}> causal mask on (each token can only see itself and earlier tokens)</label>`;
  const stage1 = stageCard(
    '01: TRANSFORM',
    'The full scaled score matrix',
    `With the causal toggle set to <b>${result.causal ? 'on' : 'off'}</b>, this is the current state of every score after scaling.`,
    `<div class="mask-row">${heatMatrixGrid(result.masked.map((row) => row.map((v) => (v <= -1e8 ? 0 : v))), {
      rowLabels: result.tokens,
      maskedCells: (i, j) => result.causal && j > i,
    })}${toggleHtml}</div>`,
    `the toggle changes this grid live`
  );
  container.innerHTML = filmstrip([stage1]);
  const toggle = container.querySelector('[data-role="causal-toggle"]');
  toggle.addEventListener('change', () => {
    window.attentionSetCausal(toggle.checked);
  });
}

function renderSoftmax(container, stepId, result) {
  const stage1 = stageCard(
    '01: TRANSFORM',
    'Exponentiate, then normalize',
    `We get the actual attention weights by softmaxing the scaled (and possibly masked) scores: exponentiate every value in a row, then divide each by that row's sum, so the row becomes a probability distribution that adds up to exactly 1.00.`,
    `<div class="formula">$$ \\text{weight}_{ij} = \\frac{e^{\\text{scaled}_{ij}}}{\\sum_k e^{\\text{scaled}_{ik}}} $$</div>
     <div><div class="heatbar-block-title">before softmax</div>${heatMatrixGrid(result.masked.map((row) => row.map((v) => (v <= -1e8 ? 0 : v))), {
       rowLabels: result.tokens,
       maskedCells: (i, j) => result.causal && j > i,
     })}</div>
     <div class="sum-arrow">&darr; softmax</div>
     <div><div class="heatbar-block-title">after softmax: the attention weights</div>${heatMatrixGrid(result.weights, { rowLabels: result.tokens })}</div>`,
    `every row sums to exactly 1.00`
  );
  container.innerHTML = filmstrip([stage1]);
}

// A vector with its label (a token, usually) placed beside it rather than on its own line above
// it, so a stack of several vectors (e.g. every token's embedding) doesn't spend a whole extra
// line of vertical space per vector just to name it.
function labeledVecBlock(label, values, opts = {}) {
  const style = opts.style ? ` style="${opts.style}"` : '';
  return `<div class="heatbar-block"${style}>
    <div class="heatbar-block-label">${label}</div>
    <div class="heatbar-list">${heatBarList(values, opts)}</div>
  </div>`;
}

// A value vector scaled by its attention weight: labeledVecBlock with the label naming the
// weight, and the whole block's opacity driven by that weight, so a barely-attended token
// visibly fades instead of just being one more identical-looking row in the list.
function weightedVecBlock(token, weight, vec) {
  return labeledVecBlock(`&quot;${token}&quot; &times; ${weight.toFixed(2)}`, vec, {
    style: `opacity:${(0.3 + weight * 0.7).toFixed(2)}`,
  });
}

function renderWsum(container, stepId, result) {
  const focusIdx = Math.min(1, result.tokens.length - 1);
  const t0 = result.tokens[focusIdx];
  const rowWeights = result.weights[focusIdx];
  const stage1 = stageCard(
    '01: STORAGE',
    'Every attention weight, every value',
    `Softmax already produced a full row of weights for every query token; the Q/K/V projection step already produced a value vector for every token. Both are just collected here, nothing new computed yet.`,
    `<div><div class="heatbar-block-title">attention weights</div>${heatMatrixGrid(result.weights, { rowLabels: result.tokens, hiRow: focusIdx })}</div>
     <div><div class="heatbar-block-title">$V$: one row per token</div>${heatMatrixGrid(result.tokens.map((t) => result.V[t]), { rowLabels: result.tokens })}</div>`,
    `every row of weights will blend the same ${result.tokens.length} value vectors, just with different weights`
  );
  const stage2 = stageCard(
    '02: SLICE',
    'One query token&#39;s weights',
    `Focus on the attention weights for &quot;${t0}&quot;, highlighted above: ${rowWeights.map((w, j) => `${(w * 100).toFixed(0)}% on &quot;${result.tokens[j]}&quot;`).join(', ')}. Each value vector below is shown at an opacity matching its weight, so the one &quot;${t0}&quot; is attending to most is the most visible.`,
    result.tokens.map((t, j) => weightedVecBlock(t, rowWeights[j], result.V[t])).join(''),
    `these weights are the same ones computed in the Softmax step, always summing to 1.00`
  );
  const stage3 = stageCard(
    '03: TRANSFORM',
    'Scale each value vector, then add them',
    `Multiply each value vector by its own weight (a scalar times a vector, not a dot product), then add the ${result.tokens.length} resulting vectors together, position by position. That sum is the output for &quot;${t0}&quot;.`,
    `<div class="sum-arrow">&darr; add ${result.tokens.length} weighted vectors</div><div><div class="heatbar-block-title">output for &quot;${t0}&quot;</div><div class="heatbar-list">${heatBarList(result.output[focusIdx])}</div></div>
     <div class="formula">$$ o_i = \\sum_j \\text{weight}_{ij} \\, v_j $$</div>`
  );
  container.innerHTML = filmstrip([stage1, stage2, stage3]);
}

function renderOutput(container, stepId, result) {
  const storageBody = result.tokens.map((t, i) => labeledVecBlock(`&quot;${t}&quot;`, result.output[i])).join('');
  const stage1 = stageCard(
    '01: STORAGE',
    'Three vectors out, same shape as three vectors in',
    `Every output vector here is exactly ${result.d} numbers, the same width as the embeddings this pipeline started from. Nothing about the shape changed; what changed is that each vector is now a blend of the whole sequence rather than the token in isolation.`,
    storageBody,
    `compare this to the Input embeddings step: same shape, different content`
  );
  container.innerHTML = filmstrip([stage1]);
}

const STEP_RENDERERS = {
  input: renderInput,
  qkv: renderQkv,
  scores: renderScores,
  mask: renderMask,
  softmax: renderSoftmax,
  wsum: renderWsum,
  output: renderOutput,
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
