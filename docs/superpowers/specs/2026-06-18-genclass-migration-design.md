# Generative classification page migration design

Migrate `pages/generative_classification.html` onto the `.ui` article-page archetype (the system
used by estimation, bayesian, regularization, pca, fourier). Phase 4 page migration: structural and
visual only. The prose, the math, and the four interactive zones keep their behavior;
`js/generative_classification.js` is NOT edited.

## Goal

The page renders in the dark archetype with the section-outline rail (Home at position 0). The four
interactive zones map onto archetype primitives: the CSV dataset input, the Fit-GDA step with a
boxed parameter readout, the main KDE/GDA visualization demo (plot left, controls right), and the
posterior-computation walkthrough. The D3 viz, the click-to-query, and the GDA fit keep working.

## Constraints (carry through every task)

- No em-dashes or en-dashes (U+2014 / U+2013) anywhere. Grep `grep -rlP "[\x{2014}\x{2013}]"` over
  touched files before each commit.
- No `<em>` / `<strong>` and no markdown emphasis in page content.
- Preserve every element id `js/generative_classification.js` reads, plus the D3 mount `#viz` and
  the hook classes the JS emits. The JS is edited in exactly ONE place (the Σ-as-matrix fix in
  "Zone 2 / Zone 4 notes" below); nothing else in the JS changes. Verified id list:
  `csvInput`, `datasetWarning`, `fileInput`, `bandwidth`, `queryInfo`, `gdaParams`, `classLegend`,
  `calcQueryPoint`, `calcPriors`, `calcMuSigma0`, `calcPointKDE`, `calcPointGDA`, `calcPoint`,
  `fitGDA`, `example1`, `example2`, `example2x1`, `example2x2`, `showKDE`, `showGDA`, `useLogSpace`,
  and the prior/mu/sigma slots `prior0`, `prior1`, `mu0`, `mu1`, `sigma0`, `sigma1`, plus `viz`.
- Hook classes the JS writes (must stay valid as style targets, do not rename): `.gda-class`,
  `.gda-class-title` (in `#gdaParams`); `.legend-item`, `.legend-swatch` (uses a `--swatch-color`
  custom property), `.legend-label` (in `#classLegend`); the `<table>` with `.query-row-label`
  cells (in `#queryInfo`); the line markup from `renderLines` (in `#calcQueryPoint`,
  `#calcPointKDE`, `#calcPointGDA`).
- Opt-in via `<body class="ui generative-classification">`. All visual rules scoped under
  `.ui.generative-classification` (or `.ui`); never bare element selectors. Must not affect any
  un-migrated page.

## Files

- Modify: `pages/generative_classification.html` (head swap, body class, restructure).
- Rewrite: `styles/generative_classification.css` to `.ui.generative-classification` archetype
  styling (replace the legacy styles; style every JS-emitted readout container).
- Not edited: `js/generative_classification.js`.
- Update: `docs/design/redesign-backlog.md` and the `project_redesign_backlog` memory (move
  generative_classification from Next to Done).

## Head and body

Swap the legacy asset block for the archetype set (keep d3 and the MathJax inline config +
`tex-svg.js`):

```html
<script src="https://d3js.org/d3.v7.min.js"></script>
... MathJax config + tex-svg.js ...
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

Drop `base.css`, `article.css`, `responsive.css`, `formulas_layout.js`, `collapsible.js`.

Body:

```html
<body class="ui generative-classification">
  <div class="container">
    <header class="page-head">
      <div class="eyebrow">// Machine learning</div>
      <h1>Unsupervised Supervised Learning</h1>
      <p class="lede">Using unsupervised density estimation to solve supervised classification.</p>
    </header>
    <main class="article-body">
      ... sections ...
    </main>
  </div>
</body>
```

Drop the in-body `subtitle` and `home-link` (Home is injected into the rail). Replace the legacy
footer with the archetype `site-footer` / `credit` markup used on the migrated pages.

## Sections

Each `<section class="... panel collapsible">` becomes `<section class="panel">` (drop
`collapsible`; keep ids like `calculations-panel`, `posterior-panel`, `difference-panel`). Keep
`<h2>`/`<h3>`/`<h4>` and prose verbatim. `<div class="formula">` and `.formulas` blocks become the
archetype `.formulas` / `.formula` wrappers (centered, no equation numbers, body-size math).
Bare-`<p>` notes that start with "Note:" become `.callout` blocks.

Section order unchanged: Overview, Input Dataset, Calculations, Visuals, Posterior Computation,
Generative vs Discriminative. Long-title shortening deferred to the content pass.

## Zone 1: Input dataset

```html
<section class="panel">
  <h2>Input dataset</h2>
  <p>Paste a CSV of points ... Edits below apply automatically.</p>
  <label class="demo-label" for="csvInput">CSV data</label>
  <textarea id="csvInput" rows="15" class="gc-csv">... unchanged default CSV ...</textarea>
  <div class="gc-load-row">
    <label class="btn-upload">Upload CSV file<input id="fileInput" name="fileInput" type="file" accept="text/csv" /></label>
  </div>
  <div id="datasetWarning" class="callout warning gc-warning" role="status" hidden></div>
</section>
```

`#csvInput` keeps its id and default content; styled as a dark mono textarea (`.gc-csv`). The file
input keeps id `fileInput` and is wrapped in the archetype `.btn-upload` label (iOS-safe direct
tap). `#datasetWarning` keeps its id and `hidden`; the JS sets its innerHTML and toggles `hidden`;
style its container as a warning callout (only visible when the JS unhides it).

## Zone 2: Calculations and fitted parameters

Prose + math as above. The action button and the params readout:

```html
<button id="fitGDA" class="btn">Fit GDA (compute &mu;, &Sigma;)</button>
<h3>Fitted parameters</h3>
<div id="gdaParams" class="gda-params metrics gc-params">
  <div id="prior0"></div><div id="prior1"></div>
  <div id="mu0"></div><div id="mu1"></div>
  <div id="sigma0"></div><div id="sigma1"></div>
</div>
```

Keep the six empty slot divs (`prior0/1`, `mu0/1`, `sigma0/1`) so any JS that targets them stays
valid. After a fit, the JS replaces `#gdaParams` innerHTML with per-class `.gda-class` blocks
(title + MathJax μ and Σ). The Σ here is already a MathJax `\begin{bmatrix}` matrix. Style
`#gdaParams` as a boxed `.metrics` surface and lay the `.gda-class` blocks out as columns (a 2-up
grid that collapses to 1 column when narrow); `.gda-class-title` reads as a small label.

### Σ-as-matrix JS fix (the one allowed JS edit)

Σ already renders as a MathJax matrix everywhere except one spot: the posterior-walkthrough line
that fills `#calcMuSigma0` (currently `Σ = [a b; c d]`, a bracketed 2d list). Change that line so Σ
is a MathJax matrix consistent with the rest of the page, and ensure `#calcMuSigma0` is typeset:

- In `fitGDA`, change the `lines.push(...)` that builds `Class k: μ = [...]; Σ = [a b; c d]` to emit
  inline MathJax: `Class ${c}: $\mu = [...]$, $\Sigma = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$`
  (escape backslashes for the JS string, as the existing `#gdaParams` line at 552 does).
- Add `calcMuSigmaEl` to the MathJax typeset call in `fitGDA` (it currently typesets only
  `[gdaParamsEl, calcP]`) so the new inline math renders.

This is the only JS change. Do not alter any other readout, computation, or handler.

## Zone 3: Visuals demo (plot left, controls right)

Approved layout: the plot in the left column with the help text and the query readout beneath it;
the overlay/bandwidth/legend controls in a right column that drops below the plot when the demo
narrows. "Interpreting the results" becomes subsidiary step text under the demo.

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
          <label class="gc-opt"><input type="checkbox" id="useLogSpace" /> Use log-space (stabilizes extreme values)</label>
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
      ... the three existing paragraphs verbatim ...
    </div>
  </div>
</section>
```

`.gc-viz-grid` is `grid-template-columns: 1fr 270px` collapsing to `1fr` at the archetype demo
breakpoint (container width <= 840px, matching `.demo-controls`). `#viz` keeps its id and `.viz`
border/background; the D3 plot mounts into it. `#queryInfo` is styled as a boxed `.metrics` block;
its JS-built `<table>` (with `.query-row-label`) is styled to read as label/value rows.
`#classLegend` keeps its id; style `.legend-item` / `.legend-swatch` (background from
`--swatch-color`) / `.legend-label`. The "Interpreting the results" `theory-block` becomes
`.gc-substep` (left-bordered subsidiary step text, accent mono heading).

## Zone 4: Posterior computation

```html
<section id="posterior-panel" class="panel">
  <h2>Posterior computation</h2>
  <p>Walking through ... $p(y \mid x^*)$ ...</p>
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
  <div id="calcMuSigma0" class="metrics gc-calc">Class 0: ...</div>
  <div id="calcPointKDE" class="metrics gc-calc">No point computed yet.</div>
  <div id="calcPointGDA" class="metrics gc-calc">No point computed yet.</div>
</section>
```

Keep all ids. The coordinate inputs `example2x1` / `example2x2` are pulled OUT of the `#example2`
button into standalone editable fields, with `#example2` now a separate "Compute" button. The
legacy markup nested the inputs inside the button, so focusing a field fired the button's compute
prematurely and the coordinate could not be freely edited. The handler (`example2` click reads
`example2x1.value` / `example2x2.value` by id, parsed with `parseFloat`, no min/max) is unchanged,
so the user can now enter any numeric coordinate and click Compute. Style `.gc-compute` as an
inline field group (label text + two compact underline number fields + the Compute button) that
wraps on narrow widths; `.gc-inline-num` is a compact underline number input. The five `#calc*`
outputs become boxed `.metrics` readouts; their `renderLines` line markup is styled as mono
tabular rows.

## styles/generative_classification.css rewrite

Replace the file with `.ui.generative-classification`-scoped rules:

- `.gc-csv`: dark mono textarea (token border/background, vertical resize).
- `.gc-load-row`: upload row; `.btn-upload` comes from components.css.
- `.gc-warning`: spacing for the warning callout when shown.
- `.gc-params`: boxed; `.gda-class` as a 2-up grid (1-up when narrow); `.gda-class-title` label.
- `.gc-viz-grid`: `1fr 270px`, collapse to `1fr` at `<=840px` container width.
- `.gc-viz-controls`, `.gc-ctrl-group` (spacing), `.gc-opt` (checkbox row), `.gc-num` (underline
  number field).
- `.class-legend` / `.legend-item` / `.legend-swatch` (`background: var(--swatch-color)`) /
  `.legend-label`.
- `.gc-query` + `#queryInfo table` + `.query-row-label`: boxed readout with tabular rows.
- `.gc-substep`: subsidiary step text (left hairline border, accent mono `h4`, muted body).
- `.gc-actions` (button row, wraps), `.gc-compute` (inline field group: text + two number inputs +
  Compute button, baseline-aligned, wraps), `.gc-compute-text` (muted inline label spans),
  `.gc-inline-num` (compact underline number input), `.examples-hint` (muted helper line).
- `.gc-calc`: boxed mono readout for the posterior walkthrough lines.
- Remove all legacy non-archetype styling, including any fixed `max-width` that fights the
  archetype content cap.

## Responsive

- Viz demo grid: `1fr 270px` to `1fr` at container `<=840px` (matching `.demo-controls`).
- Fitted-params grid: 2-up to 1-up at container `<=560px`.
- Use container queries on `.demo` (archetype sets `container-type: inline-size` on `.demo`).

## Update (2026-06-18): query table transpose, sort, and collapsibles

Added after review. The original query table put methods as rows and `2K+1` columns, which does not
scale past a few classes. Changes:

- Transpose the query table (always, regardless of K): one row per class, fixed columns
  `Class | KDE p(y|x) | KDE p(x|y)` plus `GDA p(y|x) | GDA p(x|y)` when GDA is fitted. The table
  grows downward, so any number of classes is fine. This is a focused rewrite of `updateQueryTable`
  in `generative_classification.js` (a second allowed JS change, beyond the Σ-matrix fix).
- Sortable columns: each header cell (`.gc-sort-th`, `data-col`) is clickable. Clicking sorts the
  class rows by that column; clicking the active column toggles ascending/descending; a small arrow
  (▲ / ▼, unicode, not a dash) marks the active column. Sort state is held in module-level
  `querySort` and the last query args are cached so a header click re-renders the same point with
  the new sort. Default sort: Class ascending.
- Collapsible blocks via native `<details class="gc-collapse">` + `<summary>` (accessible, minimal
  JS): wrap the fitted-parameters readout (`#gdaParams`), the legend (`#classLegend`), and the
  query readout (`#queryInfo`). The `<summary>` becomes the block label (replacing the standalone
  "Fitted parameters" `<h3>`, the "Legend" demo-label, and a "Query result" label). The JS keeps
  writing into the inner ids, which stay inside the details body. All open by default.

CSS for these lives in `styles/generative_classification.css` (`.gc-query` transposed table,
`.gc-sort-th`, `.gc-collapse` disclosure). No change to the other readouts.

## Out of scope (deferred per backlog)

- Content edits, title shortening, in-sequence note reorg: content pass.
- Input-range / width robustness sweep: stability pass.
- Full mobile pass: mobile pass.
- Further `generative_classification.js` changes beyond the single Σ-as-matrix fix: the JS already
  emits styleable hook classes, so no other JS edit is needed for the migration.

## Verification

- Load the page: rail builds with Home at 0; the default CSV renders the D3 viz; toggling Show KDE
  / Show GDA / log-space updates the plot; changing bandwidth updates KDE; clicking the plot fills
  `#queryInfo`; "Fit GDA" fills `#gdaParams` with per-class μ/Σ (MathJax); the Example buttons fill
  the `#calc*` readouts; the legend lists classes with color swatches.
- Σ renders as a matrix (not a bracketed 2d list) in both `#gdaParams` and the `#calcMuSigma0`
  posterior readout.
- The coordinate fields are editable: typing arbitrary values into `example2x1` / `example2x2`
  (e.g. negative or large numbers) does not trigger a premature compute, and clicking the separate
  "Compute" button computes the posterior at exactly the entered point.
- Playwright screenshots at desktop (rail, plot+controls side by side), tablet (~760px, controls
  below plot), narrow (~440px). No clipping or horizontal scroll.
- Isolation: load an un-migrated page (manifold or distributions) and confirm no visual change.
- Dash and `<em>`/`<strong>` scans over the touched files before commit. Id-presence check for the
  full id list plus `#viz`.
