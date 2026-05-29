# Manifold Page: Step Experience Redesign

Date: 2026-05-28

## Goal

Replace three surfaces on the manifold learning page with redesigned versions:

1. The step indicator panel that shows where each algorithm is in its pipeline.
2. The per-step visualization shown inside each algorithm's viewport.
3. The worked-example content rendered inside the Intuition / Formula / Worked example tabs.

These three changes ship as a single redesign because they are tightly coupled: the step indicator drives which step is active, the viewport shows what that step is doing, and the worked example shows the numbers that flow through it.

The existing dataset, algorithm, state, Web Worker, and pseudocode subsystems all stay as-is. This redesign only affects the step indicator, the viewport content per step, and the IFW worked-example content.

## User constraints

The following constraints apply to all generated content (HTML, JS, CSS, docs):

- No em-dashes anywhere.
- No `<em>`, `<strong>`, `<b>`, `<i>`, `<mark>` HTML tags.
- No markdown emphasis (`*`, `**`, `_`, `__`).

Plain prose, headings, and lists only.

## Surface 1: Step indicator panel

### Locked design

The step indicator replaces `js/manifold/step_indicator.js` and the matching CSS rules in `styles/manifold.css`.

The new indicator has two states: compressed (default) and expanded. In both states, two side-by-side panels (one per algorithm) sit horizontally inside a shared frame. The bottom row of the frame holds the Prev / step-description / Next nav.

Compressed state shows a short horizontal lineage of seven dot positions per panel, one for each canonical step 0..6. Each dot reflects the algorithm's relationship to that canonical step:

- Filled white: a step the algorithm executes that is at or before the algorithm's current sub-step.
- Hollow white outline: a step the algorithm executes that is after the algorithm's current sub-step.
- Dim gray smaller dot: a canonical step the algorithm does not execute (N/A for this algorithm).

The current sub-step has no glow, no halo, no special marker. It is simply the last filled dot in the lineage. The progression is read like a discrete progress bar.

A small numeric label (the canonical step id 0..6) floats below each dot via absolute positioning. The label is also dim gray for N/A positions and the same opacity as the dim N/A dot.

The line connecting consecutive dots is rendered through the dots' vertical center. Achieved by giving the bar `display: flex; align-items: center` and giving every cell and edge the same fixed height. The number label is absolutely positioned so it does not push the line off the center.

Clicking a dot would jump the global sub-step to that step (only valid for non-N/A dots). Clicking an edge (the line segment between two dots) toggles the expanded state of the whole frame. N/A dots are not clickable.

Expanded state preserves the side-by-side bars and adds a vertical detail list inside each panel. The detail list shows seven rows, one per canonical step. Each row matches the bar's classification (past / current / future / N/A) with the same dot styling at a smaller size, plus the step's title. The current row is bold with a subtle background tint and no glow. N/A rows are italic, dimmed, and read "not used".

Color palette: white at varying opacity for everything. No orange, blue, or any algorithm-specific accent. Per-algorithm distinction is conveyed by the panel headers ("A · PCA", "B · Isomap") and by layout position only.

The visual reference implementation lives in `pages/manifold-preview-steps.html`.

### Algorithm contract changes

The current contract already exposes `algorithm.presentSubSteps` (since the worker-driven refactor). The step indicator only needs the union of A's and B's `presentSubSteps`, plus the current sub-step. No further contract changes required.

### Interaction with state

The existing `state.setStep(subStep)` API stays. The indicator calls it from dot clicks. Edge clicks call a new `setExpanded(open)` API on the indicator module, kept locally within the indicator component (not part of central state, since expansion is purely view-side).

## Surface 2: Per-step visualization

### Locked design

The viewport currently renders the same 3D point cloud at every step, with kNN edges overlaid from step 2 onward. The new design shows what the step is actually doing. When the operation is not on the data points themselves, the viewport renders the operation (matrix transformation, eigenvalue spectrum, etc.) instead of the data.

The four locked patterns are:

#### Pattern 1: Data transform (PCA step 1, centering)

When the user enters step 1, the viewport plays a one-shot slide animation. Each point starts at its original raw-data position (ghosted at low opacity), and over roughly 800 to 1200 ms the points slide to their centered positions. The ghost cloud stays visible at low opacity after the animation; the centered cloud sits at the origin with full opacity. No additional overlay (no arrow, no annotation) is needed because the worked example tab carries the numerical detail.

The animation is implemented as a CSS transform or SVG `<animate>` triggered when the step is first rendered, and is debounced so re-rendering the same step (for example, due to a parameter tweak that does not change the cache key) does not replay it. Subsequent visits within the same dataset key show the static end state.

#### Pattern 2: Graph construction (Isomap step 2, kNN)

When the user enters step 2, the viewport plays a one-shot wave animation in which kNN edges fade in around the cloud over roughly 1200 to 1800 ms. After the wave, the full graph is shown statically.

Hovering a node lights up the edges connected to that node and dims all other edges. Moving the cursor off restores the full graph. Hover is event-driven on the SVG, attached after the wave completes. The brightening uses opacity and stroke width changes only, no color shift.

#### Pattern 3: Matrix derivation (PCA step 3, Isomap step 3, Isomap step 4)

The viewport switches from a 3D viewer to a horizontal three-pane strip layout: input on the left, intermediate in the middle, output on the right, with operation labels and arrows between panes.

PCA step 3 (covariance) panes:

- Input pane: a small thumbnail of the centered 3D cloud.
- Middle pane: the unscaled outer-product sum matrix shown as a 3x3 grid of numbers.
- Output pane: the covariance matrix C shown as a 3x3 grid of numbers.

Inter-pane labels: "X_cᵀ X_c" between input and middle; "÷ (N − 1)" between middle and output.

Isomap step 3 (geodesic distances) panes:

- Input pane: a small thumbnail of the kNN graph.
- Middle pane: a copy of the graph with one example shortest-path highlighted in bright white.
- Output pane: the N x N distance matrix as a heatmap with one row highlighted.

Inter-pane labels: "all-pairs Dijkstra" between input and middle; "fill matrix" between middle and output.

Isomap step 4 (double-center) panes:

- Input pane: D² as a heatmap.
- Middle pane: D² with row means and column means subtracted.
- Output pane: the Gram matrix B as a heatmap.

Inter-pane labels: "subtract row/col means" between input and middle; "× (−1/2) + grand mean" between middle and output.

Heatmaps use a single sequential colormap (the viridis-like purple-to-yellow gradient already shown in the preview).

#### Pattern 4: Spectral decomposition (PCA step 5, Isomap step 5)

The viewport shows the main 2D projection result in 3D context, with a small overlay in the bottom-right corner showing the algorithm-specific spectral information.

PCA step 5 main view: the 3D cloud with the PC1-PC2 plane drawn through it as a thin translucent parallelogram, plus the projected points lying on the plane. A few dashed perpendicular connector lines from the cloud to its projection on the plane communicate the projection operation.

PCA step 5 mini (bottom-right corner overlay): three colored arrows from origin representing PC1, PC2, PC3. Arrow length proportional to the square root of the corresponding eigenvalue. Colors: orange for PC1, blue for PC2, green for PC3. These are the only allowed accent colors and they only appear inside this mini.

Isomap step 5 main view: the 3D cloud with each point colored along a viridis-like purple-to-yellow gradient by its top-1 eigenvector value (a preview of where it will land on the first embedding axis at step 6).

Isomap step 5 mini (bottom-right corner overlay): a bar chart of the top eight eigenvalues, with the top two bars filled white at full opacity and the rest at lower opacity.

#### Pattern 0 and 6 (unchanged)

Step 0 keeps the existing static 3D point cloud with rainbow coloring by intrinsic parameter. Step 6 keeps the existing 2D scatter with a small 3D mini-thumbnail in the bottom-right corner.

### Implementation notes

The current `viz3d.js` renders only the point cloud and kNN edges. To support the new patterns, the viewport switching logic in `main.js` needs to dispatch to different renderers per step. The proposed structure is:

- A new viz dispatcher module (`js/manifold/step_viz.js`) reads the step state and chooses which renderer to mount.
- New small renderer modules under `js/manifold/viz/`:
  - `viz_centering.js` (Pattern 1)
  - `viz_knn.js` (Pattern 2; replaces the current edge overlay in viz3d.js with a richer hover-aware version)
  - `viz_matrix_strip.js` (Pattern 3, parameterized by pane contents)
  - `viz_spectral.js` (Pattern 4, branches by algorithm)
  - The existing viz3d.js and viz2d.js handle steps 0 and 6.
- Each renderer exposes `mount(container, state)` and `unmount()`.
- `step_viz.js` chooses the renderer based on a new `vizKind` field added to each StepState.

The algorithm contract adds an optional `vizKind` field on each StepState. Valid values:

- `'point_cloud'` (default; used for step 0)
- `'centering'` (PCA step 1)
- `'knn_graph'` (Isomap step 2)
- `'matrix_strip'` (PCA step 3, Isomap step 3, Isomap step 4)
- `'spectral'` (steps 5; the renderer branches on algorithm id)
- `'embedding'` (step 6)

For `'matrix_strip'`, the StepState also carries `panes`: an array of three pane descriptors, each `{ kind: 'cloud_thumb' | 'graph_thumb' | 'graph_thumb_with_path' | 'matrix_numbers' | 'heatmap' | 'heatmap_with_row', data, label }`, plus `labels` for the two operation arrows between panes.

For `'spectral'`, the StepState carries:

- PCA: `pcAxes` (already present), the principal plane parametrized as origin + two vectors.
- Isomap: top-1 eigenvector values per point (for cloud coloring) and the top eight eigenvalues (for the mini bar chart).

The visual reference implementation lives in `pages/manifold-preview-viz.html`.

## Surface 3: Worked-example format

### Locked design

Format 1: three sectioned blocks, used uniformly across every step that has a worked example.

The three sections are labeled "Input (from previous step)", "Formula", and "Output (after this step)". Each section has a small uppercase label above a left-bordered code area. The Formula section renders the existing MathJax equation. The Input and Output sections render monospace text with concrete numbers from the actual dataset.

The numbers shown are samples from the real dataset, not hardcoded toy values. For point-wise operations (centering, kNN neighbor list), three sample point indices are pulled deterministically (e.g., indices floor(N/4), floor(N/2), floor(3N/4)). For matrix operations, a small excerpt (4x4 or so) of the actual matrix is shown plus one or two concrete element-wise computations.

### Implementation notes

The current IFW `worked` field is an HTML string set per step by the algorithm module. To produce the sectioned-blocks format, the algorithm module computes the worked example content at runtime from real data and emits HTML that uses three labeled `<div class="ifw-worked-section">` blocks: `ifw-worked-input`, `ifw-worked-formula`, `ifw-worked-output`.

The IFW component (`js/manifold/ifw.js`) stays mostly as-is. It receives the worked-example HTML and renders it as before. The CSS gets new rules for the three section labels and left-bordered code areas, matching the preview styling.

Algorithm modules need a small helper for formatting numbers: a `formatVec3(v)` for 3-tuples, a `formatMatrix(M, rows, cols)` for matrix excerpts, and a `formatTable(rows)` for the input/output row pairs in the centering case. These can live in a new `js/manifold/format.js`.

Per-step worked example outline:

- PCA step 0 (raw): no worked example.
- PCA step 1 (center): input shows three sample raw points and the computed mean; output shows the same three points after subtraction.
- PCA step 3 (covariance): input shows three sample centered points; output shows the 3x3 covariance matrix with one element computation spelled out.
- PCA step 5 (eigen): input shows the 3x3 covariance; output shows the three eigenvalues and the three eigenvector columns.
- PCA step 6 (project): input shows the top-2 eigenvectors; output shows three sample 2D embedding coordinates.
- Isomap step 0 (raw): no worked example.
- Isomap step 2 (kNN): input shows three sample point indices; output shows their k nearest neighbours and edge weights.
- Isomap step 3 (geodesic): input shows the kNN graph for one example pair (i, j) with the shortest path traced; output shows a 4x4 excerpt of D and one element computation.
- Isomap step 4 (double-center): as in the preview (Format 1, Example B).
- Isomap step 5 (eigen): input shows the 4x4 excerpt of B; output shows the top two eigenvalues and the first six entries of v1 and v2.
- Isomap step 6 (embed): input shows the top eigenvalues and first three entries of v1, v2; output shows the three resulting 2D embedding coordinates.

The visual reference implementation lives in `pages/manifold-preview-worked.html`.

## File structure changes

Files modified:

- `js/manifold/step_indicator.js` (full rewrite to match locked design).
- `js/manifold/main.js` (mount the new step viz dispatcher; pass the IFW worked-example content through unchanged but with new CSS classes).
- `js/manifold/ifw.js` (style updates; the existing API stays).
- `js/manifold/algorithms/pca.js` (add `vizKind` and richer worked-example content per step; expose `presentSubSteps` if not already).
- `js/manifold/algorithms/isomap.js` (same as PCA).
- `styles/manifold.css` (new rules for step indicator, matrix strip, spectral mini, worked-example sections).

Files created:

- `js/manifold/step_viz.js` (renderer dispatcher per step).
- `js/manifold/viz/viz_centering.js`.
- `js/manifold/viz/viz_knn.js` (replaces the kNN edge logic currently inline in viz3d.js).
- `js/manifold/viz/viz_matrix_strip.js`.
- `js/manifold/viz/viz_spectral.js`.
- `js/manifold/format.js` (number formatting helpers).

Files left alone:

- `js/manifold/canonical_steps.js`.
- `js/manifold/rng.js`.
- `js/manifold/datasets.js`.
- `js/manifold/linalg.js`.
- `js/manifold/state.js` (no contract changes needed; `vizKind` rides on the existing StepState pass-through).
- `js/manifold/viz3d.js` (still handles steps 0 and 6's 3D cloud rendering plus the bottom-right mini for step 6 and step 5).
- `js/manifold/viz2d.js` (still handles step 6's 2D scatter).
- `js/manifold/pseudocode.js`.
- `js/manifold/worker.js` (the worker continues to import algorithm modules and stream step states; the worked-example HTML is just one more string field).
- `pages/manifold.html` (no structural changes; viz hosts already exist).

Files deleted:

- None.

Test files:

- The existing unit tests stay. Add unit tests for `format.js` (number formatting helpers) under `test/manifold/format.test.js`.
- The new viz modules are browser-only and are exercised in the final browser smoke pass; they do not get node tests.

## Open decisions for the implementation plan

These are deliberately deferred to the implementation plan rather than the spec:

1. Whether to extract the existing rainbow coloring helper (currently duplicated between `viz3d.js` and `viz2d.js`) into the new `viz/` directory at the same time, or leave it for a later refactor.
2. Whether to introduce a small viz lifecycle interface (`mount` / `update` / `unmount`) or keep each renderer as a function. Probably the former, but the implementation plan will pin it down.
3. Whether the spectral mini in pattern 4 is itself an SVG drawn inline or its own small viewport. Implementation plan picks one.
4. Specific animation timings (the spec gives ranges); the plan picks exact values.
5. The exact viridis-like gradient stops. The previews used `#000`, `#5b3a8c`, `#8b5fbf`, `#c179d3`, `#e8a37f`, `#f5cf6e`, `#f9eb6b`. Confirming these in the plan or replacing with a small generated palette.

## Out of scope

- Phase 2 algorithms (MDS, LLE, Laplacian Eigenmaps, t-SNE, etc.). The redesign assumes only PCA and Isomap for now; adding more algorithms in phase 2 follows the same vizKind contract.
- Phase 2 datasets beyond Swiss roll, S-curve, and CSV upload.
- Adding new IFW tabs beyond Intuition / Formula / Worked example.
- Reworking the pseudocode block at the bottom of the page.
- Touch and mobile-specific gestures for the step indicator (the design supports clicks; we accept that touch users tap-to-toggle without hover).
