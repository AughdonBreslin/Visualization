// js/attention/scenes.js
// Renders each step's hero glyph and worked-example animation, using a shared "storage / slice /
// transform / concept" filmstrip pattern (not every step uses all four stages: Input has nothing
// to transform, so it only uses two). Every value shown comes from the live PipelineResult, never
// a hardcoded number.

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

// 0 = dim cool blue, 0.5 = green, 1 = hot red-orange. Interpolating through a midpoint keeps
// mid-range magnitudes visually distinct instead of muddying into a single blue<->red blend.
function heatColor(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [
    { p: 0, h: 224, s: 42, l: 28 },
    { p: 0.5, h: 130, s: 45, l: 40 },
    { p: 1, h: 8, s: 82, l: 54 },
  ];
  const [a, b] = t > 0.5 ? [stops[1], stops[2]] : [stops[0], stops[1]];
  const localT = a.p === 0.5 ? (t - 0.5) / 0.5 : t / 0.5;
  const h = a.h + (b.h - a.h) * localT;
  const s = a.s + (b.s - a.s) * localT;
  const l = a.l + (b.l - a.l) * localT;
  return `hsl(${h.toFixed(0)}, ${s.toFixed(0)}%, ${l.toFixed(0)}%)`;
}

// A vector rendered as one row per dimension: a fixed-width, bracket-framed 0-1 heat-filled
// track (magnitude relative to this vector's own largest entry) plus the raw signed value
// printed alongside it. Heat color and fill length redundantly encode the same magnitude on
// purpose. No row label by default -- a dimension index (d0, d1, ...) is noise for a plain
// embedding/Q/K/V display where no particular dimension is being singled out; pass opts.labels
// only when a row genuinely identifies something (a token, a specific highlighted dimension).
// dimExcept, if given, keeps only that one dimension at full opacity (used in "slice" stages to
// show one component still belongs to the full vector without erasing the rest).
function heatBarList(values, opts = {}) {
  const m = maxAbs(values);
  const dimExcept = opts.dimExcept;
  const labels = opts.labels;
  return values.map((v, i) => {
    const t = Math.abs(v) / m;
    const pct = (t * 100).toFixed(0);
    const isDim = dimExcept !== undefined && i !== dimExcept;
    const labelHtml = labels ? `<div class="heatbar-label">${labels[i]}</div>` : '';
    return `<div class="heatbar-row ${isDim ? 'dim' : ''}">
      ${labelHtml}
      <div class="heatbar-track"><div class="heatbar-fill" style="width:${pct}%; background:${heatColor(t)}"></div></div>
      <div class="heatbar-val">${v.toFixed(2)}</div>
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

function stageCard(n, title, proseHtml, bodyHtml, noteHtml) {
  return `<div class="stage">
    <div class="stage-connector"></div>
    <div class="stage-n">${n}</div>
    <div class="stage-title">${title}</div>
    ${proseHtml ? `<p class="stage-prose">${proseHtml}</p>` : ''}
    <div class="stage-body">${bodyHtml}</div>
    ${noteHtml ? `<div class="stage-note">${noteHtml}</div>` : ''}
  </div>`;
}

function filmstrip(stages) {
  return `<div class="filmstrip">${stages.join('')}</div>`;
}

// ---- per-step renderers ----------------------------------------------------------------------

function renderInput(container, stepId, result) {
  const storageBody = result.tokens.map((t) => labeledVecBlock(`&quot;${t}&quot;`, result.embeddings[t])).join('');
  const stage1 = stageCard(
    '01: STORAGE',
    'The three embeddings',
    `Every token in this worked example starts as a ${result.d}-number vector called an <b>embedding</b>. These numbers were picked by hand for this toy example; a real trained model would have learned them from data.`,
    storageBody,
    `${result.tokens.length} tokens, ${result.d} numbers each, stored in full`
  );
  const stage2 = stageCard(
    '02: CONCEPT',
    'Where these numbers come from',
    null,
    `<p class="concept-box">In a real transformer, embeddings are one of the things training learns: every token in the vocabulary gets its own vector, nudged during training until tokens used in similar ways end up with similar vectors. This page uses d = ${result.d} so every number stays visible on screen; a real model typically uses hundreds or thousands of dimensions, but nothing about the mechanism changes at that scale, only the width of every vector shown on this page. The rest of this page shows what happens to these numbers once a model already has them; see the planned Phase 3 for how training would actually produce them.</p>`
  );
  container.innerHTML = filmstrip([stage1, stage2]);
}

function renderQkv(container, stepId, result) {
  const t0 = result.tokens[0];
  const wqRow0 = WEIGHTS.WQ[0];
  const stage1 = stageCard(
    '01: STORAGE',
    'The full data at rest',
    `Every token's embedding gets multiplied by three learned weight matrices, producing a query, a key, and a value vector. Below is the embedding for &quot;${t0}&quot; and the query vector it produces: the full output of this step, for one token.`,
    `<div><div class="heatbar-block-title">x &quot;${t0}&quot; (input)</div><div class="heatbar-list">${heatBarList(result.embeddings[t0])}</div></div>
     <div><div class="heatbar-block-title">q &quot;${t0}&quot; (output)</div><div class="heatbar-list">${heatBarList(result.Q[t0])}</div></div>`,
    `1 embedding in, 1 query vector out, per token`
  );
  const stage2 = stageCard(
    '02: SLICE',
    'Focus on one output number',
    `We're computing just one number: the first entry of the query vector, <code>q&#8320;</code> = ${result.Q[t0][0].toFixed(2)}. Producing it only needs the embedding on the left and one row of the weight matrix W_Q on the right, specifically the row responsible for output dimension 0.`,
    `<div><div class="heatbar-block-title">x &quot;${t0}&quot;</div><div class="heatbar-list">${heatBarList(result.embeddings[t0])}</div></div>
     <div><div class="heatbar-block-title">W_Q, row 0 only</div><div class="heatbar-list">${heatBarList(wqRow0)}</div></div>`,
    `every other row of W_Q would produce a different output dimension (q&#8321;, q&#8322;, q&#8323;) and plays no part in this one`
  );
  const stage3 = stageCard(
    '03: TRANSFORM',
    'Multiply position by position, then sum',
    `Pair up <code>x</code> and the row of W_Q from stage 2 by position, multiply each pair, then add all ${result.d} products together. That sum <b>is</b> q&#8320;, and nothing more happens to it.`,
    multBreakdown(result.embeddings[t0], wqRow0, 'q₀', result.Q[t0][0].toFixed(2))
  );
  const stage4 = stageCard(
    '04: CONCEPT',
    'Three separate projections',
    null,
    `<p class="concept-box">A raw embedding conflates everything about a token into one vector. Attention needs three different views of it: a <b>query</b> (&quot;what am I looking for&quot;), a <b>key</b> (&quot;what do I offer&quot;), and a <b>value</b> (&quot;what I actually contribute if chosen&quot;). If Q and K shared a weight matrix, every token's query would equal its own key, and every token would trivially attend most to itself. Separate, independently learned projections let a token's query and key diverge, which is what lets it end up attending to a <i style="font-style:normal">different</i> token when that's more useful.</p>`
  );
  container.innerHTML = filmstrip([stage1, stage2, stage3, stage4]);
}

function renderScores(container, stepId, result) {
  const t0 = result.tokens[0];
  const stage1 = stageCard(
    '01: STORAGE',
    'Every query, every key',
    `The previous step already produced a query vector and a key vector for <b>every</b> token, not just &quot;${t0}&quot;. <code>Q</code> below stacks all ${result.tokens.length} query vectors, one row per token; <code>K</code> stacks all ${result.tokens.length} key vectors the same way.`,
    `<div><div class="heatbar-block-title">Q: one row per token</div>${heatMatrixGrid(result.tokens.map((t) => result.Q[t]), { hiRow: 0, rowLabels: result.tokens })}</div>
     <div><div class="heatbar-block-title">K: one row per token</div>${heatMatrixGrid(result.tokens.map((t) => result.K[t]), { hiRow: 0, rowLabels: result.tokens })}</div>`,
    `${result.tokens.length} queries &times; ${result.tokens.length} keys = ${result.tokens.length * result.tokens.length} comparisons still to come`
  );
  const stage2 = stageCard(
    '02: SLICE',
    'One query, one key',
    `To fill in exactly one cell of the score grid, where query &quot;${t0}&quot; meets key &quot;${t0}&quot;, row 0 column 0, we only need <b>one</b> row from Q and <b>one</b> row from K, both highlighted above. Every other row belongs to a different cell and isn't used here.`,
    `<div><div class="heatbar-block-title">q &quot;${t0}&quot;</div><div class="heatbar-list">${heatBarList(result.Q[t0])}</div></div>
     <div><div class="heatbar-block-title">k &quot;${t0}&quot;</div><div class="heatbar-list">${heatBarList(result.K[t0])}</div></div>`,
    `this pair lands in score grid cell [0,0]`
  );
  const n = result.tokens.length;
  const blankGrid = `<div class="mgrid-wrap"><div class="mgrid-rowlabels">${result.tokens.map((t) => `<div class="mgrid-rowlabel" style="color:var(--text-muted)">${t}</div>`).join('')}</div><div class="mgrid g${n}x${n}">${result.tokens.map(() => result.tokens.map(() => '<div class="mcell pending">?</div>').join('')).join('')}</div></div>`;
  const cellWorked = `score[&quot;${t0}&quot;,&quot;${t0}&quot;] = q &middot; k = ${result.Q[t0].map((qv, j) => `${qv.toFixed(2)}&times;${result.K[t0][j].toFixed(2)}`).join(' + ')} = ${result.scores[0][0].toFixed(2)}`;
  const stage3 = stageCard(
    '03: TRANSFORM',
    'Fill in the grid, one comparison at a time',
    `Every cell repeats the same operation: pair up one query row and one key row by position, multiply each pair, add the results. That's a <b>dot product</b>, the standard way to measure how aligned two vectors are. Click below to watch all ${n * n} cells compute at once.`,
    `<div class="scale-shrink-wrap"><div data-role="sweep-grid">${blankGrid}</div></div>
     <div class="anim-controls"><button class="anim-btn" type="button" data-role="sweep-btn">&#9654; compute all ${n * n} cells</button></div>
     <div class="stage-note" data-role="sweep-worked" style="display:none">${cellWorked}</div>`
  );
  const stage4 = stageCard(
    '04: CONCEPT',
    'A dot product measures match',
    null,
    `<p class="concept-box">A large dot product means q and k point in a similar direction: loosely, &quot;what this token is looking for&quot; closely matches &quot;what that token offers.&quot; Every cell of the grid is the result of the exact same multiply-then-sum you just watched fill in, just for a different query/key pair each time.</p>`
  );
  container.innerHTML = filmstrip([stage1, stage2, stage3, stage4]);

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
  const after = result.scaled[0][0];
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
     <div class="anim-controls"><button class="anim-btn" type="button" data-role="shrink-btn">&#9654; divide every cell by &radic;${result.d}</button></div>`
  );
  const stage4 = stageCard(
    '04: CONCEPT',
    'Keeps softmax from saturating',
    null,
    `<p class="concept-box">Dot-product magnitude grows with the number of dimensions being summed, d. If Q and K entries have roughly unit variance, the dot product's own variance grows proportional to d, so its standard deviation grows with &radic;d. Dividing by &radic;d is exactly what keeps a score's scale roughly constant no matter how large d is chosen to be; without it, larger d would make every softmax in the next steps saturate toward a near one-hot output, with vanishing gradients almost everywhere else.</p>`
  );
  container.innerHTML = filmstrip([stage1, stage2, stage3, stage4]);

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
  const stage4 = stageCard(
    '04: CONCEPT',
    '&minus;&infin;, not 0',
    null,
    `<p class="concept-box">Every step so far treats tokens symmetrically: any token can see any other, past or future. That's fine for encoding a complete sentence, wrong for predicting the next token, since letting a model see the answer it's predicting makes training meaningless. e<sup>0</sup> = 1, so a masked score of literal 0 would still receive real, nonzero attention weight after softmax, the same as any other position scoring 0. e<sup>&minus;&infin;</sup> = 0 exactly: the only value guaranteed to zero out a position's contribution regardless of what the other scores in its row happen to be.</p>`
  );
  container.innerHTML = toggleHtml + filmstrip([stage1, stage2, stage3, stage4]);
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
  const stage3 = stageCard(
    '03: TRANSFORM',
    'Exponentiate, then normalize',
    `Two steps, not one: first exponentiate every value in the row, which makes everything positive and stretches the gaps between them. Then divide each exponentiated value by their sum, so the row splits into the proportions shown below, adding up to exactly 1.00.`,
    `${expRows}<div class="sum-arrow">&darr; divide each by the sum (${expSum.toFixed(2)})</div>${probBar(rowTokens, exps.map((e) => e / expSum), rowColors)}`
  );
  const allBars = result.tokens.map((ti, i) => probBar(result.tokens, result.weights[i], result.tokenColors, `&quot;${ti}&quot;`)).join('');
  const stage4 = stageCard(
    '04: CONCEPT',
    'Exponentiate before normalizing',
    `The exponential is what makes softmax amplify differences: a score only slightly larger than its neighbors can end up with a much larger share of the final weight, once the gap has been stretched by exponentiating. This is part of why the Scale step mattered earlier: unscaled scores would make softmax nearly one-hot almost everywhere, leaving no useful gradient for training to work with.`,
    `<div class="heatbar-block-title">every row becomes its own distribution</div>${allBars}`
  );
  container.innerHTML = filmstrip([stage1, stage2, stage3, stage4]);
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
     <div><div class="heatbar-block-title">V: one row per token</div>${heatMatrixGrid(result.tokens.map((t) => result.V[t]), { rowLabels: result.tokens })}</div>`,
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
    `<div class="sum-arrow">&darr; add ${result.tokens.length} weighted vectors</div><div><div class="heatbar-block-title">output for &quot;${t0}&quot;</div><div class="heatbar-list">${heatBarList(result.output[focusIdx])}</div></div>`
  );
  const stage4 = stageCard(
    '04: CONCEPT',
    'Value vectors carry the content',
    null,
    `<p class="concept-box">Like Q and K, V is its own learned projection, so the model can choose what a token actually contributes to others independent of what makes it a good match (its key) or what it's searching for (its query). A token can be highly relevant (a large attention weight) while contributing very little of any one particular feature, or the reverse, because relevance and content are computed by two entirely separate weight matrices.</p>`
  );
  container.innerHTML = filmstrip([stage1, stage2, stage3, stage4]);
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
  const stage2 = stageCard(
    '02: CONCEPT',
    'Not usually the end of the line',
    null,
    `<p class="concept-box">This output isn't usually the end of a transformer block on its own; in a real model it typically continues through a residual connection and a feed-forward layer, both outside the scope of this page, which focuses specifically on the attention operation itself. Everything shown on this page so far has been inference: one forward pass through a single attention head, using weight matrices that are already fixed. The planned Phase 2 would compute how those weights should change for this example; a further Phase 3 would show them actually being learned.</p>`
  );
  container.innerHTML = filmstrip([stage1, stage2]);
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
