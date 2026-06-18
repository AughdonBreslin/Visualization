# Generative Classification Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate `pages/generative_classification.html` onto the `.ui` article-page archetype,
keeping `js/generative_classification.js` behavior intact (one tiny Σ-as-matrix edit), with the
viz demo as plot-left / controls-right and editable x* compute coordinates.

**Architecture:** Opt-in `<body class="ui generative-classification">`. Visual rules scoped under
`.ui.generative-classification` in a rewritten `styles/generative_classification.css`. All element
ids, the `#viz` D3 mount, and the JS-emitted hook classes are preserved. The only JS change is
rendering Σ as a MathJax matrix in the `#calcMuSigma0` readout.

**Tech Stack:** Static HTML/CSS/JS, d3 v7, MathJax tex-svg, the archetype CSS (tokens, system,
components, article-ui), shared section-outline.js.

Spec: `docs/superpowers/specs/2026-06-18-genclass-migration-design.md`. Read it before starting.

**Process note:** Verify visually with the Playwright harness at `/tmp/pwverify` and grep guards,
not unit tests. Serve the repo root on `http://localhost:8000`. Run the Task 0 guards before each
commit.

---

### Task 0: Guards (reference, used by every task)

**Dash + emphasis guard** (must print nothing):

```bash
cd /mnt/c/Users/aughb/PersonalProjects/Visualization
grep -rlP "[\x{2014}\x{2013}]" pages/generative_classification.html styles/generative_classification.css 2>/dev/null
grep -nE "<(em|strong)[ >]" pages/generative_classification.html 2>/dev/null
```

**JS hook id guard** (every id present exactly once after the HTML rewrite):

```bash
cd /mnt/c/Users/aughb/PersonalProjects/Visualization
for id in csvInput datasetWarning fileInput bandwidth queryInfo gdaParams classLegend \
  calcQueryPoint calcPriors calcMuSigma0 calcPointKDE calcPointGDA fitGDA example1 example2 \
  example2x1 example2x2 showKDE showGDA useLogSpace prior0 prior1 mu0 mu1 sigma0 sigma1 viz; do
  c=$(grep -c "id=\"$id\"" pages/generative_classification.html); [ "$c" = "1" ] || echo "BAD id=$id count=$c";
done; echo "ids checked"
```

---

### Task 1: Rewrite `styles/generative_classification.css` to the `.ui.generative-classification` archetype

**Files:**
- Rewrite: `styles/generative_classification.css`

- [ ] **Step 1: Replace the entire file with:**

```css
/* generative_classification.css - migrated (.ui) page. Typography, sections, links, math, the
 * demo system, buttons, callouts, metrics, and the footer come from the archetype. This file adds
 * the CSV input, the GDA parameter readout, the KDE/GDA viz demo (plot left, controls right), the
 * legend, the click-to-query readout, and the posterior-walkthrough readouts. */

/* --- Zone 1: CSV dataset input --- */
.ui.generative-classification .gc-csv { width: 100%; min-height: 150px; resize: vertical; background: #060607; border: 1px solid var(--hairline); border-radius: var(--radius-md); color: #cfd1d8; font: 500 12px/1.55 var(--font-mono); padding: 11px 13px; }
.ui.generative-classification .gc-load-row { display: flex; align-items: center; gap: 12px; margin-top: 12px; }
.ui.generative-classification .gc-warning { margin-top: 14px; }

/* --- Zone 2: fitted parameters readout --- */
.ui.generative-classification .gc-params { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px 26px; }
.ui.generative-classification .gc-params .gda-class { min-width: 0; }
.ui.generative-classification .gc-params .gda-class-title { font: 600 10px/1.2 var(--font-mono); letter-spacing: .1em; text-transform: uppercase; color: #6c6e77; margin-bottom: 8px; }
.ui.generative-classification .gc-params .gda-class > div { margin: 4px 0; color: #cfd1d8; font-size: 13px; }
@container (max-width: 560px) { .ui.generative-classification .gc-params { grid-template-columns: 1fr; } }

/* --- Zone 3: viz demo grid --- */
.ui.generative-classification .gc-viz-grid { display: grid; grid-template-columns: 1fr 270px; gap: 26px; align-items: start; }
@container (max-width: 840px) { .ui.generative-classification .gc-viz-grid { grid-template-columns: 1fr; gap: 22px; } }
.ui.generative-classification #viz { width: 520px; max-width: 100%; }
.ui.generative-classification .gc-viz-controls { min-width: 0; }
.ui.generative-classification .gc-ctrl-group + .gc-ctrl-group { margin-top: 20px; }
.ui.generative-classification .gc-opt { display: flex; align-items: center; gap: 9px; font: 500 12.5px/1.35 var(--font-sans); color: #cfd1d8; cursor: pointer; }
.ui.generative-classification .gc-opt + .gc-opt { margin-top: 9px; }
.ui.generative-classification .gc-opt input { accent-color: var(--accent); width: 14px; height: 14px; flex: none; }
.ui.generative-classification .gc-num { width: 100%; background: none; border: 0; border-bottom: 1px solid var(--hairline-strong); color: #dadbe0; font: 500 13px/1 var(--font-sans); padding: 5px 2px; }

/* legend (JS builds .legend-item > .legend-swatch[--swatch-color] + .legend-label) */
.ui.generative-classification .class-legend { display: flex; flex-direction: column; gap: 7px; font: 400 12px/1.3 var(--font-sans); color: var(--text-muted); }
.ui.generative-classification .legend-item { display: flex; align-items: center; gap: 9px; color: #cfd1d8; }
.ui.generative-classification .legend-swatch { width: 11px; height: 11px; border-radius: 3px; flex: none; background: var(--swatch-color, #888); }

/* click-to-query readout (JS builds a <table> with .query-row-label cells) */
.ui.generative-classification .gc-query { margin-top: 16px; overflow-x: auto; }
.ui.generative-classification .gc-query table { border-collapse: collapse; width: 100%; font: 500 12.5px/1.4 var(--font-mono); color: #e7e8ec; font-variant-numeric: tabular-nums; }
.ui.generative-classification .gc-query th, .ui.generative-classification .gc-query td { padding: 5px 10px; text-align: right; border-top: 1px solid var(--hairline); }
.ui.generative-classification .gc-query tr:first-child th, .ui.generative-classification .gc-query tr:first-child td { border-top: 0; }
.ui.generative-classification .gc-query .query-row-label { text-align: left; color: var(--text-body); font-family: var(--font-sans); }

/* subsidiary "Interpreting the results" step text */
.ui.generative-classification .gc-substep { margin-top: 24px; padding-left: 15px; border-left: 2px solid var(--hairline); }
.ui.generative-classification .gc-substep h4 { font: 600 11px/1 var(--font-mono); letter-spacing: .1em; text-transform: uppercase; color: var(--accent); margin: 0 0 9px; }
.ui.generative-classification .gc-substep p { font: 400 13px/1.6 var(--font-sans); color: var(--text-body); margin: 0 0 10px; max-width: 78ch; }

/* --- Zone 4: posterior walkthrough --- */
.ui.generative-classification .gc-actions { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin: 6px 0 8px; }
.ui.generative-classification .gc-compute { display: inline-flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.ui.generative-classification .gc-compute-text { font: 500 13px/1 var(--font-sans); color: var(--text-body); }
.ui.generative-classification .gc-inline-num { width: 56px; background: none; border: 0; border-bottom: 1px solid var(--hairline-strong); color: #dadbe0; font: 500 13px/1 var(--font-mono); padding: 4px 2px; text-align: center; }
.ui.generative-classification .examples-hint { font: 400 12px/1.5 var(--font-sans); color: var(--text-muted); margin: 4px 0 14px; }
.ui.generative-classification .gc-calc { margin-top: 12px; font: 500 12.5px/1.5 var(--font-mono); color: #cfd1d8; }
.ui.generative-classification .gc-calc h3, .ui.generative-classification .gc-calc .calc-step-header { font: 600 11px/1.2 var(--font-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); margin: 4px 0 8px; }
.ui.generative-classification .gc-calc h4, .ui.generative-classification .gc-calc h5, .ui.generative-classification .gc-calc .calc-step-subheader { font: 600 12px/1.3 var(--font-sans); color: #cfd1d8; margin: 8px 0 5px; }
.ui.generative-classification .gc-calc .calc-step-content { margin: 3px 0; }
.ui.generative-classification .gc-calc .calc-step-end { margin: 5px 0; color: #e7e8ec; }
```

- [ ] **Step 2: Confirm no legacy artifacts remain** (these were in the old file or markup):
Run: `grep -nE "1400px|viz-row|viz-panel|theory-block|options-group|bandwidth-row" styles/generative_classification.css`
Expected: no output.

- [ ] **Step 3: Dash guard over the CSS.** Expected: no output.

- [ ] **Step 4: Commit.**

```bash
git add styles/generative_classification.css
git commit -m "redesign: rewrite generative_classification.css onto the .ui archetype"
```

---

### Task 2: HTML head, shell, prose sections, footer

This does the head swap, body shell, and the two prose-only sections (Overview, Generative vs
Discriminative), and de-`collapsible`s ALL sections. The interactive zones (Input dataset,
Calculations, Visuals, Posterior) keep their legacy inner markup for now; later tasks replace them.

**Files:**
- Modify: `pages/generative_classification.html`

- [ ] **Step 1: Replace the head asset block** (keep d3 + MathJax config + tex-svg.js):

```html
  <link rel="stylesheet" href="../styles/tokens.css">
  <link rel="stylesheet" href="../styles/system.css">
  <link rel="stylesheet" href="../styles/components.css">
  <link rel="stylesheet" href="../styles/article-ui.css">
  <link rel="stylesheet" href="../styles/generative_classification.css">
  <link rel="stylesheet" href="../styles/section-outline.css">
  <script src="../js/theme.js"></script>
  <script defer src="../js/generative_classification.js"></script>
  <script defer src="../js/favicon.js"></script>
  <script type="module" src="../js/section-outline.js"></script>
```

(Removed: base.css, article.css, responsive.css, formulas_layout.js, collapsible.js.)

- [ ] **Step 2: Replace the body open + header:**

```html
<body class="ui generative-classification">
  <div class="container">
    <header class="page-head">
      <div class="eyebrow">// Machine learning</div>
      <h1>Unsupervised Supervised Learning</h1>
      <p class="lede">Using unsupervised density estimation to solve supervised classification.</p>
    </header>
    <main class="article-body">
```

Remove the old `subtitle` and `home-link`.

- [ ] **Step 3: De-`collapsible` every section.** Change each `<section class="... panel
  collapsible" ...>` to `<section class="panel" ...>` keeping any id (`calculations-panel`,
  `posterior-panel`, `difference-panel`) and dropping `data-open-mobile`. Keep all headings and
  prose verbatim.

- [ ] **Step 4: Convert math blocks.** Each standalone `<div class="formula">$$...$$</div>` stays
  as is inside the archetype (article-ui.css styles `.formula`/`.formulas`); ensure single displays
  are wrapped `<div class="formulas"><div class="formula">$$...$$</div></div>` to match estimation.
  Keep `.calc-explain` inner structure and its `.formulas` blocks. Keep LaTeX exactly.

- [ ] **Step 5: Convert "Note:" paragraphs to callouts.** The Visuals "Note: To see the log-space
  ..." paragraph (when it moves to the substep in Task 5) stays as prose; any standalone `<p>` that
  begins "Note:" elsewhere becomes a `.callout`. (Overview/Difference have none; this mainly
  applies later. No action needed in the prose-only sections.)

- [ ] **Step 6: Replace the footer:**

```html
    <footer class="site-footer">
      <span class="credit">Created by <a href="https://linkedin.com/in/aughdon/">Aughdon Breslin</a></span>
    </footer>
```

- [ ] **Step 7: Guards + visual.** Dash guard: no output. Load the page; confirm the rail builds
  with Home at 0, sections render in the dark theme, math renders. The interactive zones may look
  legacy-unstyled until later tasks; that is expected. The D3 viz should still draw (the JS runs).

- [ ] **Step 8: Commit.**

```bash
git add pages/generative_classification.html
git commit -m "redesign: migrate generative_classification head, shell, and prose sections"
```

---

### Task 3: Zone 1 (Input dataset)

**Files:**
- Modify: `pages/generative_classification.html` (the "Input Dataset" section)

- [ ] **Step 1: Replace the section body** (keep `<section class="panel">` and `<h2>`):

```html
      <section class="panel">
        <h2>Input dataset</h2>
        <p>Paste a CSV of points $x_1, x_2, y$ where $x_1, x_2 \in \mathbb{R}$ and the class $y$ is any integer. Edits below apply automatically.</p>
        <label class="demo-label" for="csvInput">CSV data</label>
        <textarea id="csvInput" rows="15" class="gc-csv">x1,x2,y
81.0,85.0,1
... KEEP THE EXACT EXISTING DEFAULT CSV ROWS, UNCHANGED ...
74.0,98.0,2
</textarea>
        <div class="gc-load-row">
          <label class="btn-upload">Upload CSV file<input id="fileInput" name="fileInput" type="file" accept="text/csv" /></label>
        </div>
        <div id="datasetWarning" class="callout warning gc-warning" role="status" hidden></div>
      </section>
```

Preserve the exact existing CSV default content inside `#csvInput` (do not retype the numbers from
memory; keep the existing lines). `#fileInput` keeps its id/name/accept and is now wrapped in a
`.btn-upload` label. `#datasetWarning` keeps id + `hidden` + `role="status"`.

- [ ] **Step 2: Guards.** id guard (csvInput, fileInput, datasetWarning present once). Dash: none.

- [ ] **Step 3: Visual.** Textarea renders dark/mono; editing it updates the plot; the upload label
  is styled; the warning is hidden by default.

- [ ] **Step 4: Commit.**

```bash
git add pages/generative_classification.html
git commit -m "redesign: rebuild generative_classification dataset input on the archetype"
```

---

### Task 4: Zone 2 (Calculations + fitted params) and the Σ-as-matrix JS fix

**Files:**
- Modify: `pages/generative_classification.html` (the "Calculations" section)
- Modify: `js/generative_classification.js` (one readout line + one typeset list)

- [ ] **Step 1: Keep the prose + math of the Calculations section** (de-collapsibled in Task 2).
  Replace only the button + params block at the end of the section with:

```html
        <p>After loading data, click "Fit GDA" to compute class priors and Gaussian parameters (mean $\mu$ and covariance $\Sigma$) for each class.</p>
        <button id="fitGDA" class="btn">Fit GDA (compute &mu;, &Sigma;)</button>

        <h3>Fitted parameters</h3>
        <div id="gdaParams" class="gda-params metrics gc-params">
          <div id="prior0"></div>
          <div id="prior1"></div>
          <div id="mu0"></div>
          <div id="mu1"></div>
          <div id="sigma0"></div>
          <div id="sigma1"></div>
        </div>
```

- [ ] **Step 2: Σ-as-matrix JS fix.** In `js/generative_classification.js`, find the `fitGDA`
  function's `lines.push(...)` that builds the `#calcMuSigma0` text (currently:
  `<p>Class ${c}: μ = [...]; Σ = [a b; c d]</p>`). Replace that template literal with inline
  MathJax so Σ is a matrix, consistent with `#gdaParams`:

```js
      lines.push(`<p>Class ${c}: $\\mu = [${p.mu.map(v=>v.toFixed(3)).join(', ')}]$, $\\Sigma = \\begin{bmatrix} ${p.sigma[0][0].toFixed(3)} & ${p.sigma[0][1].toFixed(3)} \\\\ ${p.sigma[1][0].toFixed(3)} & ${p.sigma[1][1].toFixed(3)} \\end{bmatrix}$</p>`);
```

- [ ] **Step 3: Typeset `#calcMuSigma0`.** In the same `fitGDA`, the MathJax typeset call currently
  targets `[gdaParamsEl, calcP]`. Add `calcMuSigmaEl` so the new inline math renders. Change the
  filter line to include it:

```js
    const typesetEls = [gdaParamsEl, calcP, calcMuSigmaEl].filter(Boolean);
```

Make no other JS change.

- [ ] **Step 4: Syntax + guards.** Run `node --check js/generative_classification.js` (passes).
  Dash guard over the JS file too:
  `grep -rlP "[\x{2014}\x{2013}]" js/generative_classification.js` must print nothing. (The Greek
  μ/Σ and the existing `μ`/`Σ` literals are fine; only em/en dashes are banned.)

- [ ] **Step 5: Visual.** Click "Fit GDA": `#gdaParams` fills with per-class blocks showing prior,
  μ vector, and Σ as a MathJax matrix, laid out two-up in a boxed panel. Later (Task 6) confirm the
  `#calcMuSigma0` readout also shows Σ as a matrix.

- [ ] **Step 6: Commit.**

```bash
git add pages/generative_classification.html js/generative_classification.js
git commit -m "redesign: GDA fit params as a boxed metrics block; render Sigma as a matrix in the posterior readout"
```

---

### Task 5: Zone 3 (Visuals demo)

**Files:**
- Modify: `pages/generative_classification.html` (the "Visuals: KDE & GDA" section)

- [ ] **Step 1: Replace the section body** (keep `<section class="panel">` and `<h2>`):

```html
      <section class="panel">
        <h2>Visuals: KDE and GDA</h2>
        <div class="demo">
          <div class="gc-viz-grid">
            <div class="demo-fig">
              <div class="figcap">Class-conditional densities and decision regions</div>
              <div id="viz" class="viz"></div>
              <div class="help-text">Click on the plot to query $p(y=k \mid x)$ and $p(x \mid y=k)$ at that point.</div>
              <div class="info-row">
                <div id="queryInfo" class="metrics gc-query">No query yet</div>
              </div>
            </div>

            <aside class="gc-viz-controls">
              <div class="gc-ctrl-group">
                <span class="demo-label">Overlays</span>
                <label class="gc-opt"><input type="checkbox" id="showKDE" checked /> Show KDE heatmap</label>
                <label class="gc-opt"><input type="checkbox" id="showGDA" /> Show GDA ellipses &amp; decision</label>
                <label class="gc-opt"><input type="checkbox" id="useLogSpace" /> Use log-space (stabilizes for extreme values)</label>
              </div>
              <div class="gc-ctrl-group">
                <label class="demo-label" for="bandwidth">Bandwidth</label>
                <input id="bandwidth" type="number" step="0.1" min="0.01" value="0.6" class="gc-num" />
              </div>
              <div class="gc-ctrl-group">
                <span class="demo-label">Legend</span>
                <div id="classLegend" class="class-legend">No classes loaded</div>
              </div>
            </aside>
          </div>

          <div class="gc-substep">
            <h4>Interpreting the results</h4>
            <p>... existing paragraph 1 verbatim ...</p>
            <p>... existing paragraph 2 verbatim ...</p>
            <p>... existing paragraph 3 (the "Note: To see the log-space ..." paragraph) verbatim ...</p>
          </div>
        </div>
      </section>
```

Keep the three "Interpreting the results" paragraphs verbatim. Preserve every id
(`viz`, `queryInfo`, `showKDE`, `showGDA`, `useLogSpace`, `bandwidth`, `classLegend`).

- [ ] **Step 2: Guards.** id guard for the Zone-3 ids (each once). Dash: none.

- [ ] **Step 3: Visual.** Plot renders left, controls right. Toggling Show KDE / Show GDA /
  log-space updates the plot; bandwidth changes KDE; clicking the plot fills `#queryInfo` (styled
  as a boxed table). The legend lists classes with color swatches. At <=840px container the
  controls drop below the plot.

- [ ] **Step 4: Commit.**

```bash
git add pages/generative_classification.html
git commit -m "redesign: rebuild the KDE/GDA viz demo as plot-left / controls-right"
```

---

### Task 6: Zone 4 (Posterior computation)

**Files:**
- Modify: `pages/generative_classification.html` (the "Posterior Computation" section)

- [ ] **Step 1: Replace the section body** (keep `<section id="posterior-panel" class="panel">` and
  `<h2>`):

```html
      <section id="posterior-panel" class="panel">
        <h2>Posterior computation</h2>
        <p>Walking through the calculations for both KDE and GDA to compute the posterior probabilities $p(y|x^*)$ at a query point $x^*$.</p>
        <div class="gc-actions">
          <button id="example1" type="button" class="btn">Example (75, 89.5)</button>
          <div class="gc-compute">
            <span class="gc-compute-text">Compute for x* = (</span>
            <input id="example2x1" type="number" step="0.1" value="83" aria-label="x1 coordinate" class="gc-inline-num" />
            <span class="gc-compute-text">,</span>
            <input id="example2x2" type="number" step="0.1" value="83" aria-label="x2 coordinate" class="gc-inline-num" />
            <span class="gc-compute-text">)</span>
            <button id="example2" type="button" class="btn">Compute</button>
          </div>
        </div>
        <p class="examples-hint">Or click anywhere on the plot above to query a point.</p>
        <div id="calcQueryPoint" class="metrics gc-calc">No point computed yet.</div>
        <div id="calcPriors" class="metrics gc-calc">Priors: -</div>
        <div id="calcMuSigma0" class="metrics gc-calc">Class 0: &mu;0 = - ; &Sigma;0 = -</div>
        <div id="calcPointKDE" class="metrics gc-calc">No point computed yet.</div>
        <div id="calcPointGDA" class="metrics gc-calc">No point computed yet.</div>
      </section>
```

The coordinate inputs are now OUTSIDE the `#example2` button (a sibling `.gc-compute` group), so
they are freely editable; `#example2` is a separate "Compute" button. All ids preserved.

- [ ] **Step 2: Guards.** id guard (example1, example2, example2x1, example2x2, calcQueryPoint,
  calcPriors, calcMuSigma0, calcPointKDE, calcPointGDA present once). Dash: none.

- [ ] **Step 3: Behavior check.** Type a non-default coordinate into the x1/x2 fields (e.g.
  `-2`, `120`); focusing/typing does NOT trigger a compute. Click "Compute": the `#calc*` readouts
  fill for exactly the entered point, and `#calcMuSigma0` shows Σ as a matrix. Click "Example
  (75, 89.5)" and confirm it computes for that point. Clicking the plot still updates these.

- [ ] **Step 4: Commit.**

```bash
git add pages/generative_classification.html
git commit -m "redesign: posterior walkthrough with editable x* coordinates and boxed readouts"
```

---

### Task 7: Verification, isolation, backlog + memory

**Files:**
- Modify: `docs/design/redesign-backlog.md`
- Modify: the `project_redesign_backlog` memory file.

- [ ] **Step 1: Responsive screenshots.** With Playwright, capture the page at ~1280px (rail, plot
  + controls side by side), ~760px (controls below plot, params still readable), ~440px (single
  column). Review: no clipping, no horizontal scroll except the intentional `gc-query`/plot
  overflow on very narrow widths.

- [ ] **Step 2: Full behavior pass** (desktop): edit CSV -> plot updates; Fit GDA -> params with Σ
  matrices; toggle overlays + log-space; change bandwidth; click plot -> query table; Example +
  custom Compute -> calc readouts with Σ matrix; legend swatches. No console errors.

- [ ] **Step 3: Isolation.** Load an un-migrated page (`pages/manifold.html` or
  `pages/distributions.html`) and confirm no visual change (rules are `.ui.generative-classification`
  scoped).

- [ ] **Step 4: Final guards.** Dash guard + id guard over the final files; `node --check` the JS.

- [ ] **Step 5: Update backlog.** In `docs/design/redesign-backlog.md`, move
  generative_classification to Done: `Done: estimation (pilot), bayesian, regularization, pca,
  fourier, generative_classification. Next: manifold, distributions.`

- [ ] **Step 6: Update memory.** Update `project_redesign_backlog` memory migration status to add
  generative_classification as done.

- [ ] **Step 7: Commit.**

```bash
git add docs/design/redesign-backlog.md
git commit -m "docs: mark generative_classification migration done in the backlog"
```

---

## Self-review notes

- Every JS-read id and `#viz` are reproduced across Tasks 3-6 and guarded in Task 0. The only JS
  edit is the Σ-as-matrix readout fix + adding `calcMuSigmaEl` to the typeset list (Task 4).
- JS-emitted hook classes (`.gda-class`, `.legend-item`/`.legend-swatch`/`.legend-label`,
  `#queryInfo` table + `.query-row-label`, `renderLines` step classes) are styled in Task 1.
- The `#example2` coordinate inputs are pulled out of the button (Task 6) so any numeric coordinate
  can be entered without a premature compute; the handler reads them by id, unchanged.
- Footer (`site-footer`/`credit`), buttons (`.btn`, `.btn-upload`), callout (`.callout.warning`),
  and metrics (`.metrics`) reuse the verified archetype primitives.
- The D3 viz is a fixed 520x420 SVG; `#viz` is sized to it (`max-width: 100%`). Fluid plot scaling
  is deferred to the robustness/mobile pass.
- Deferred per backlog: content edits, title shortening, input-range robustness, full mobile pass.
