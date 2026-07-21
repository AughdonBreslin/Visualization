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

// A single bar split into proportional, token-colored segments -- the distribution primitive,
// used only by Softmax, where "these numbers become proportions of a whole that sum to 1" is
// the entire point of the step and is best shown as a bar splitting into parts, not more bars
// or grids.
function probBar(tokens, weights, tokenColors, rowLabel) {
  const segs = tokens.map((t, j) => {
    const pct = weights[j] * 100;
    return `<div class="propbar-seg" style="width:${pct.toFixed(2)}%; background:${tokenColors[j]}" title="${t}: ${pct.toFixed(1)}%">${pct >= 12 ? `<span class="propbar-label">${pct.toFixed(0)}%</span>` : ''}</div>`;
  }).join('');
  const rowLabelHtml = rowLabel ? `<div class="propbar-rowlabel">${rowLabel}</div>` : '';
  return `<div class="propbar-row">${rowLabelHtml}<div class="propbar">${segs}</div></div>`;
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
     <div><div class="heatbar-block-title">$k_{\\text{${t0}}}$</div><div class="heatbar-list">${heatBarList(result.K[t0])}</div></div>`,
    `this pair lands in score grid cell [0,0]`
  );
  const n = result.tokens.length;
  const blankGrid = `<div class="mgrid-wrap"><div class="mgrid-rowlabels">${result.tokens.map((t) => `<div class="mgrid-rowlabel" style="color:var(--text-muted)">${t}</div>`).join('')}</div><div class="mgrid g${n}x${n}">${result.tokens.map(() => result.tokens.map(() => '<div class="mcell pending">?</div>').join('')).join('')}</div></div>`;
  const cellWorked = `score[&quot;${t0}&quot;,&quot;${t0}&quot;] = q &middot; k = ${result.Q[t0].map((qv, j) => `${qv.toFixed(2)}&times;${result.K[t0][j].toFixed(2)}`).join(' + ')} = ${result.scores[0][0].toFixed(2)}`;
  const stage3 = stageCard(
    '03: TRANSFORM',
    'Fill in the grid, one comparison at a time',
    `Every cell repeats the same operation: pair up one query row and one key row by position, multiply each pair, add the results. That's a dot product, the standard way to measure how aligned two vectors are. Click below to watch all cells compute at once.`,
    `<div class="scale-shrink-wrap"><div data-role="sweep-grid">${blankGrid}</div></div>
     <div class="anim-controls"><button class="anim-btn" type="button" data-role="sweep-btn">&#9654; compute all cells</button></div>
     <div class="stage-note" data-role="sweep-worked" style="display:none">${cellWorked}</div>
     <div class="formula">$$ \\text{score}_{ij} = q_i \\cdot k_j $$</div>`
  );
  container.innerHTML = filmstrip([stage1, stage2, stage3]);

  // The grid starts blank ("?" in every cell) and fills in all at once on click, each cell
  // fading in with a staggered delay -- a single grid materializing, not nine separate reveals
  // to click through, and visually distinct from Scale's shrink and QKV's static breakdown.
  const sweepGrid = container.querySelector('[data-role="sweep-grid"]');
  const sweepBtn = container.querySelector('[data-role="sweep-btn"]');
  const sweepWorked = container.querySelector('[data-role="sweep-worked"]');
  sweepBtn.addEventListener('click', () => {
    sweepGrid.innerHTML = heatMatrixGrid(result.scores, { rowLabels: result.tokens });
    sweepGrid.querySelectorAll('.mcell').forEach((cell, idx) => {
      cell.style.animationDelay = `${idx * 55}ms`;
      cell.classList.add('cell-fade-in');
    });
    sweepWorked.style.display = '';
    sweepBtn.disabled = true;
    sweepBtn.textContent = 'all cells computed';
  });
}

function renderScale(container, stepId, result) {
  const t0 = result.tokens[0];
  const before = result.scores[0][0];
  const sqrtD = Math.sqrt(result.d);
  const stage1 = stageCard(
    '01: STORAGE',
    'The full score matrix',
    `Every score computed in the previous step is about to divide by &radic;${result.d}. Here's the whole grid before that happens.`,
    heatMatrixGrid(result.scores, { rowLabels: result.tokens }),
    `${result.tokens.length * result.tokens.length} cells, all about to shrink by the same factor`
  );
  const stage2 = stageCard(
    '02: SLICE',
    'One cell, before scaling',
    `Focus on the cell where query &quot;${t0}&quot; meets key &quot;${t0}&quot;. Before scaling it's ${before.toFixed(2)}; every other cell scales the exact same way, independently.`,
    `<div class="sum-result" style="margin-bottom:10px">before: ${before.toFixed(3)}</div>`,
    `this is score grid cell [0,0], the same cell used in the previous step's worked example`
  );
  const stage3 = stageCard(
    '03: TRANSFORM',
    'Watch the whole grid shrink',
    `d = ${result.d} here, so &radic;d = ${sqrtD.toFixed(2)}. Every cell in the grid divides by that same number at once, not one at a time. Click below to watch it happen.`,
    `<div class="scale-shrink-wrap"><div class="scale-shrink-grid" data-role="shrink-grid">${heatMatrixGrid(result.scores, { rowLabels: result.tokens })}</div></div>
     <div class="anim-controls"><button class="anim-btn" type="button" data-role="shrink-btn">&#9654; divide every cell by &radic;${result.d}</button></div>
     <div class="formula">$$ \\text{scaled}_{ij} = \\frac{\\text{score}_{ij}}{\\sqrt{d}} $$</div>`
  );
  container.innerHTML = filmstrip([stage1, stage2, stage3]);

  // The shrink button toggles between the pre- and post-scale grid: the CSS transform on
  // .scale-shrink-grid animates the visual size change, while the innerHTML swap (heat color +
  // printed value) happens instantly underneath it, together reading as "the grid shrinks."
  const shrinkGrid = container.querySelector('[data-role="shrink-grid"]');
  const shrinkBtn = container.querySelector('[data-role="shrink-btn"]');
  let shrunk = false;
  shrinkBtn.addEventListener('click', () => {
    shrunk = !shrunk;
    shrinkGrid.classList.toggle('shrunk', shrunk);
    shrinkGrid.innerHTML = heatMatrixGrid(shrunk ? result.scaled : result.scores, { rowLabels: result.tokens });
    shrinkBtn.innerHTML = shrunk
      ? '&#9664; show before scaling'
      : `&#9654; divide every cell by &radic;${result.d}`;
  });
}

function renderMask(container, stepId, result) {
  const t0 = result.tokens[0];
  const t1 = result.tokens.length > 1 ? result.tokens[1] : t0;
  const cellValue = result.scaled[0][Math.min(1, result.tokens.length - 1)];
  const toggleHtml = `<label class="mask-toggle"><input type="checkbox" data-role="causal-toggle" ${result.causal ? 'checked' : ''}> causal mask on (each token can only see itself and earlier tokens)</label>`;
  const stage1 = stageCard(
    '01: STORAGE',
    'The full scaled score matrix',
    `With the toggle above set to <b>${result.causal ? 'on' : 'off'}</b>, this is the current state of every score after scaling.`,
    heatMatrixGrid(result.masked.map((row) => row.map((v) => (v <= -1e8 ? 0 : v))), {
      rowLabels: result.tokens,
      maskedCells: (i, j) => result.causal && j > i,
    }),
    `toggle above changes this grid live`
  );
  const stage2 = stageCard(
    '02: SLICE',
    'One future position',
    `Focus on the cell where query &quot;${t0}&quot; meets key &quot;${t1}&quot;, where column index 1 is greater than row index 0, so under a causal mask this key comes <b>after</b> this query in the sequence.`,
    `<div class="sum-result" style="margin-bottom:10px">scaled value here: ${cellValue.toFixed(2)}</div>`,
    `j=1 &gt; i=0, so this cell is a future position relative to the query`
  );
  const stage3 = stageCard(
    '03: TRANSFORM',
    'If masked, force it to &minus;&infin;',
    `The rule is a simple condition, not a formula: for every cell where the key's position (j) is later than the query's position (i), replace the scaled value with negative infinity before softmax runs. Every other cell is left untouched.`,
    `<div class="mult-row"><span class="mult-dimlabel">rule</span><span class="mult-chip" style="color:var(--accent-link)">j &gt; i ?</span><span class="mult-eq">&rarr;</span><span class="mult-prod">&minus;&infin;</span></div>
     <div class="mult-row"><span class="mult-dimlabel">else</span><span class="mult-chip" style="color:var(--accent-link)">j &le; i ?</span><span class="mult-eq">&rarr;</span><span class="mult-prod">unchanged</span></div>`
  );
  container.innerHTML = toggleHtml + filmstrip([stage1, stage2, stage3]);
  const toggle = container.querySelector('[data-role="causal-toggle"]');
  toggle.addEventListener('change', () => {
    window.attentionSetCausal(toggle.checked);
  });
}

function renderSoftmax(container, stepId, result) {
  const t0 = result.tokens[0];
  const row0 = result.masked[0].filter((v) => v > -1e8);
  const rowTokens = result.tokens.filter((_, j) => result.masked[0][j] > -1e8);
  const exps = row0.map((v) => Math.exp(v));
  const expSum = exps.reduce((a, b) => a + b, 0);
  const stage1 = stageCard(
    '01: STORAGE',
    'The full scaled (and possibly masked) matrix',
    `Softmax runs on whatever this matrix looks like after scaling and masking, the same numbers the previous two steps produced.`,
    heatMatrixGrid(result.masked.map((row) => row.map((v) => (v <= -1e8 ? 0 : v))), {
      rowLabels: result.tokens,
      hiRow: 0,
      maskedCells: (i, j) => result.causal && j > i,
    }),
    `softmax processes one full row at a time, never a single cell`
  );
  const stage2 = stageCard(
    '02: SLICE',
    'One full row',
    `Unlike the previous steps, softmax can't be sliced down to a single cell: normalizing means every value in a row depends on every other value in that same row. So the smallest meaningful slice is the whole row for query &quot;${t0}&quot;.`,
    `<div class="heatbar-block-title">scaled row for &quot;${t0}&quot;</div><div class="heatbar-list">${heatBarList(row0, { labels: rowTokens })}</div>`,
    `${row0.length} value${row0.length === 1 ? '' : 's'} in this row (masked cells are excluded, they're already headed to 0)`
  );
  const expRows = row0.map((v, i) => `<div class="mult-row"><span class="mult-dimlabel">${rowTokens[i]}</span><span class="mult-chip" style="color:var(--accent-link)">${v.toFixed(2)}</span><span class="mult-eq">&rarr; e^</span><span class="mult-prod">${exps[i].toFixed(2)}</span></div>`).join('');
  const rowColors = rowTokens.map((t) => result.tokenColors[result.tokens.indexOf(t)]);
  const allBars = result.tokens.map((ti, i) => probBar(result.tokens, result.weights[i], result.tokenColors, `&quot;${ti}&quot;`)).join('');
  const stage3 = stageCard(
    '03: TRANSFORM',
    'Exponentiate, then normalize',
    `Two steps, not one: first exponentiate every value in the row, which makes everything positive and stretches the gaps between them. Then divide each exponentiated value by their sum, so the row splits into the proportions shown below, adding up to exactly 1.00.`,
    `${expRows}<div class="sum-arrow">&darr; divide each by the sum (${expSum.toFixed(2)})</div>${probBar(rowTokens, exps.map((e) => e / expSum), rowColors)}
     <div class="formula">$$ \\text{weight}_{ij} = \\frac{e^{\\text{scaled}_{ij}}}{\\sum_k e^{\\text{scaled}_{ik}}} $$</div>
     <div class="heatbar-block-title" style="margin-top:8px">every row becomes its own distribution</div>${allBars}`
  );
  container.innerHTML = filmstrip([stage1, stage2, stage3]);
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
  scale: renderScale,
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
