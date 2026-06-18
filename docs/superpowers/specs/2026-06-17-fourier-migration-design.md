# Fourier page migration design

Migrate `pages/fourier.html` onto the `.ui` article-page archetype (the system used by
estimation, bayesian, regularization, pca). This is Phase 4 page migration, not a content
rewrite: the prose, the math, and the two interactive demos keep their current behavior. The
work is structural and visual only.

## Goal

`pages/fourier.html` renders in the dark archetype: tokenized type, sectioned `.article-body`,
the section-outline rail with Home at position 0, and the archetype demo control system. The two
interactives (2D image decomposition, 3D height-map) keep working with `js/fourier.js` and
`js/fourier3d.js` untouched. The Image selector becomes the iOS-safe custom dropdown (approved
mock) backed by a hidden `<select id="fourierPreset">` so the file upload opens on mobile.

## Constraints (carry through every task)

- No em-dashes or en-dashes (U+2014 / U+2013) anywhere: prose, code, comments, HTML. Grep
  `grep -rlP "[\x{2014}\x{2013}]"` over touched files before each commit.
- No `<em>` / `<strong>` and no markdown `*` / `**` emphasis in page content.
- Preserve every element `id` and every `name=` group that `js/fourier.js` and `js/fourier3d.js`
  read. The JS files are NOT edited. The verified hook inventory:
  - fourier.js getElementById: `fourierBasis`, `fourierPreset`, `fourierRadius`,
    `fourierRadiusValue`, `fourierRadiusLabel`, `fourierFormulaBox`, `fourierTiling`,
    `fourierTilingText`, `fourierOrigLabel`, `fourierSpecLabel`, `fourierReconLabel`,
    `fourierOrigCanvas`, `fourierSpecCanvas`, `fourierReconCanvas`, `fourierUpload`.
  - fourier.js also does `presetSel.querySelector('option[value="__uploaded__"]')` and appends an
    `<option>` to `#fourierPreset`, so `#fourierPreset` must stay a real `<select>` element.
  - fourier3d.js getElementById: `fourier3dCanvas`, `fourier3dScale`, `fourier3dScaleOut`,
    `fourier3dFlyHint`, `fourier3dFlyLabel`, `fourier3dHideOOR`, `fourier3dTile`,
    `fourier3dTileExtent`, `fourier3dTileExtentOut`, `fourier3dTileExtentRow`,
    `fourier3dTileBasisNote`, plus reads `fourierBasis`.
  - fourier3d.js radio groups by name: `fourier3dSource`, `fourier3dCamera`.
- The migration is opt-in via `<body class="ui fourier">`. It must not affect any un-migrated
  page. Visual rules are scoped under `.ui.fourier` (or `.ui`), never bare element selectors.

## Files

- Modify: `pages/fourier.html` (head asset swap, body class, full body restructure).
- Rewrite: `styles/fourier.css` to `.ui.fourier` archetype styling (replace the old blue-accent
  boxed design; keep the formula tooltip, info icon, warning, and legend-color helpers restyled
  to tokens).
- Create: `js/fourier-image-dropdown.js` (the custom Image dropdown UI shim; syncs to the hidden
  `#fourierPreset` select, never edits fourier.js).
- Not edited: `js/fourier.js`, `js/fourier3d.js`.
- Update: `docs/design/redesign-backlog.md` and the `project_redesign_backlog` memory (move
  fourier from Next to Done).

## Head and body

Replace the legacy stylesheet/script block with the archetype set, matching estimation/pca:

```html
<link rel="stylesheet" href="../styles/tokens.css">
<link rel="stylesheet" href="../styles/system.css">
<link rel="stylesheet" href="../styles/components.css">
<link rel="stylesheet" href="../styles/article-ui.css">
<link rel="stylesheet" href="../styles/fourier.css">
<link rel="stylesheet" href="../styles/section-outline.css">
<script src="../js/theme.js"></script>
```

MathJax config and `tex-svg.js` stay. Keep the deferred scripts the page needs: `fourier.js`,
the three `three@0.147.0` scripts, `fourier3d.js`, `favicon.js`, and the new
`fourier-image-dropdown.js`. Add `tabs.js` (for the archetype interpretation tabs) and keep
`section-outline.js` as a module. Drop `formulas_layout.js` and `collapsible.js` (the archetype
has no collapsible panels; sections are always open). Drop `article.css`, `base.css`,
`responsive.css`.

Body becomes:

```html
<body class="ui fourier">
  <div class="container">
    <header class="page-head">
      <div class="eyebrow">// Signals</div>
      <h1>Fourier Image Decomposition</h1>
      <p class="lede">Periodicity, smoothness assumptions, and why the DFT tiles an image into an infinite checkerboard.</p>
    </header>
    <main class="article-body">
      ... sections ...
    </main>
  </div>
</body>
```

No in-body `home-link` div: Home is injected into the rail at position 0 by section-outline.js.
The legacy `<footer class="footer">` is replaced by the archetype footer treatment already used
on migrated pages (right-aligned "Created by" tag); match estimation's footer markup exactly.

## Sections (archetype `<section class="panel">`)

Each existing `<section class="panel collapsible">` becomes `<section class="panel">` (drop
`collapsible`). `<h2>` stays. `<div class="formula">` blocks become the archetype `.formulas` /
`.formula` wrapper used on estimation (math centered, no equation numbers, math at body size).
`<p class="note">` keeps its meaning; render it with the archetype note/callout style used on the
other migrated pages (verify the class estimation/bayesian use for inline notes and reuse it).

Section order is unchanged: Assumption of Smoothness, Representing Periodicity, The Discrete
Fourier Transform, Other Image Reconstruction Techniques, Interactive Demo (the two demos),
Reconstruction Formula, Interpreting the visualization, Summary.

The prose is copied verbatim except for any em/en-dash or `<strong>`/`<em>` cleanup. Long
section-title shortening is deferred to the content pass; titles are not changed here.

## Demo 1: Image decomposition

Markup follows the approved `fourier-demo.html` mock and the PCA full-width control pattern
(`.demo-controls` overridden to `display:block`, control band below the figures).

```html
<section class="panel">
  <h2>Interactive Demo</h2>

  <div class="demo">
    <div class="opt-h"><span class="eyebrow">Demo 1</span><h3>Image decomposition</h3></div>

    <div class="fourier-canvases">
      <div class="fourier-canvas-card">
        <div class="figcap"><span id="fourierOrigLabel">Original image</span></div>
        <canvas id="fourierOrigCanvas" width="256" height="256" class="fourier-canvas" aria-label="Original image or periodic extension"></canvas>
      </div>
      <div class="fourier-canvas-card">
        <div class="figcap"><span id="fourierSpecLabel">Frequency spectrum (log magnitude)</span></div>
        <canvas id="fourierSpecCanvas" width="256" height="256" class="fourier-canvas" aria-label="Frequency magnitude spectrum"></canvas>
      </div>
      <div class="fourier-canvas-card">
        <div class="figcap"><span id="fourierReconLabel">Reconstruction</span></div>
        <canvas id="fourierReconCanvas" width="256" height="256" class="fourier-canvas" aria-label="Low-pass reconstructed image"></canvas>
      </div>
    </div>

    <div class="demo-controls fourier-controls">
      <div class="demo-band fourier-band">
        <div class="demo-group">
          <label class="demo-label" for="fourierBasis">Basis</label>
          <select id="fourierBasis">... four options unchanged ...</select>
        </div>

        <div class="demo-group fourier-image-group">
          <span class="demo-label">Image</span>
          <!-- custom dropdown trigger + menu injected/wired by fourier-image-dropdown.js -->
          <select id="fourierPreset" class="fourier-preset-native">... five preset options unchanged ...</select>
          <!-- the iOS-safe upload input keeps its id; the dropdown moves its label into the menu -->
        </div>

        <div class="demo-group">
          <div class="fourier-slabel">
            <label class="demo-label" id="fourierRadiusLabel" for="fourierRadius">Low-pass filter radius</label>
            <output id="fourierRadiusValue" for="fourierRadius" class="fourier-out">r = 20</output>
          </div>
          <input id="fourierRadius" type="range" min="1" max="128" step="1" value="20" class="fourier-slider" />
        </div>

        <div class="demo-group">
          <span class="demo-label">Options</span>
          <label class="fourier-check">
            <input id="fourierTiling" type="checkbox" />
            <span id="fourierTilingText">Show periodic extension (2x2 tile)</span>
          </label>
        </div>
      </div>
    </div>
  </div>
</section>
```

`.fourier-band` is the 4-up control band: `grid-template-columns: repeat(4, minmax(0,1fr))`,
collapsing to 2 columns at `<=780px` (container width) and 1 column at `<=460px`. This is the
exact responsive behavior approved in the mock.

### Custom Image dropdown (`js/fourier-image-dropdown.js`)

Why custom: a native `<select>` cannot open the file picker on iOS (programmatic
`fileInput.click()` from a change handler is blocked); only a direct tap on a label-wrapped file
input works. The dropdown gives the "Upload image..." item a real label-wrapped
`<input type="file" id="fourierUpload">`.

The hidden real `<select id="fourierPreset">` stays in the DOM (fourier.js reads its value,
listens for its `change`, and appends the `__uploaded__` option). The shim never edits fourier.js
and never removes the select. Behavior:

- On `DOMContentLoaded`, the shim hides `#fourierPreset` visually (off-screen, still in DOM,
  focusable-by-script) and builds a trigger button reading the selected option's text plus a
  chevron, styled like the archetype underline select (the `.dd-trigger` look from the
  `fourier-upload-dropdown.html` mock).
- Clicking the trigger opens a `.fourier-dd-menu` listing one `.fourier-dd-item` per preset
  option (text from the select), the current value ticked, then a `.fourier-dd-sep`, then a
  `<label class="fourier-dd-item fourier-dd-upload">` that wraps the real
  `<input type="file" id="fourierUpload" accept="image/*">`. The upload item is a label, so
  tapping it opens the picker directly (iOS-safe); no programmatic click.
- Selecting a preset item sets `presetSel.value` and dispatches a synthetic
  `new Event('change', { bubbles: true })` on `#fourierPreset` so fourier.js's existing change
  handler runs. Then the menu closes and the trigger text updates.
- After an upload, fourier.js sets `presetSel.value = '__uploaded__'` and appends an option
  without firing `change`. The shim keeps the trigger and menu in sync with a `MutationObserver`
  on `#fourierPreset` (childList for the new option) plus its own `change` listener, rebuilding
  the trigger text and menu from the current options. This is why the shim observes rather than
  hard-codes the option list.
- Menu closes on outside click, on Escape, and on selecting a preset. Trigger has
  `aria-haspopup="listbox"` and `aria-expanded`; items are buttons. Keyboard: Enter/Space on the
  trigger toggles; Escape closes and returns focus to the trigger.

The `#fourierUpload` input is created/owned inside the menu markup the shim builds, but keeps the
exact id and `accept="image/*"` so fourier.js's `uploadInput.addEventListener('change', ...)`
binds to it. fourier.js calls `getElementById('fourierUpload')` after DOM is built; the shim must
have inserted the input into the DOM before fourier.js's `init` runs on `DOMContentLoaded`. Both
run on `DOMContentLoaded`; load `fourier-image-dropdown.js` BEFORE `fourier.js` in the head so its
listener (and the inserted input) is in place first. If ordering proves fragile, the shim instead
pre-renders the upload input in the static HTML and only moves it into the menu, guaranteeing the
id exists regardless of script order. Prefer the static-HTML upload input for robustness.

Decision: put the real `<input type="file" id="fourierUpload">` in the static HTML inside the
image group (visually hidden by default), and have the shim relocate it into the menu's upload
label at build time. This guarantees the id exists for fourier.js no matter the script order.

## Demo 2: 3D height-map

Same band model, second control band. The 3D canvas spans full width above its controls (mock
layout). All ids/names preserved.

```html
<div class="demo">
  <div class="opt-h"><span class="eyebrow">Demo 2</span><h3>3D height-map</h3></div>
  <canvas id="fourier3dCanvas" class="fourier-3d-canvas" aria-label="3D height-map surface of the current image"></canvas>

  <div class="demo-controls fourier-controls">
    <div class="demo-band fourier-band">
      <div class="demo-group">
        <span class="demo-label">Source</span>
        <div class="fourier-radios">
          <label class="fourier-radio"><input type="radio" name="fourier3dSource" value="recon" checked />Reconstruction</label>
          <label class="fourier-radio"><input type="radio" name="fourier3dSource" value="orig" />Original image</label>
        </div>
      </div>
      <div class="demo-group">
        <span class="demo-label">Camera</span>
        <div class="fourier-radios">
          <label class="fourier-radio"><input type="radio" name="fourier3dCamera" value="orbit" checked />Orbit</label>
          <label class="fourier-radio"><input type="radio" name="fourier3dCamera" value="fly" /><span id="fourier3dFlyLabel">Fly</span></label>
        </div>
        <p id="fourier3dFlyHint" class="fourier-hint" style="display:none">Click the canvas to lock the cursor; click again or press ESC to release. WASD to move, Q/E for down/up, Shift to sprint.</p>
      </div>
      <div class="demo-group">
        <div class="fourier-slabel">
          <label class="demo-label" for="fourier3dScale">Height scale</label>
          <output id="fourier3dScaleOut" class="fourier-out">60</output>
        </div>
        <input id="fourier3dScale" type="range" min="10" max="200" step="5" value="60" class="fourier-slider" />

        <div id="fourier3dTileExtentRow" class="fourier-subitem" style="display:none">
          <div class="fourier-slabel">
            <label class="demo-label" for="fourier3dTileExtent">Tile extent</label>
            <output id="fourier3dTileExtentOut" class="fourier-out">5 x 5 = 25 tiles</output>
          </div>
          <input id="fourier3dTileExtent" type="range" min="3" max="21" step="2" value="5" disabled class="fourier-slider" />
        </div>
      </div>
      <div class="demo-group">
        <span class="demo-label">Display</span>
        <label class="fourier-check"><input id="fourier3dHideOOR" type="checkbox" />Hide overshoot regions</label>
        <label class="fourier-check"><input id="fourier3dTile" type="checkbox" />Periodic tiling</label>
        <p id="fourier3dTileBasisNote" class="fourier-warning" style="display:none">Basis isn't periodic, replicating tiles instead</p>
      </div>
    </div>
  </div>
</div>
```

Note the inline `style="display:none"` on `fourier3dFlyHint`, `fourier3dTileExtentRow`, and
`fourier3dTileBasisNote` is preserved exactly: fourier3d.js toggles these via
`style.display`. The `disabled` attribute on `fourier3dTileExtent` is preserved.

The two demos live in one `<section class="panel">` (the existing "Interactive Demo" section), as
two `.demo` blocks. Keep the section `<h2>Interactive Demo</h2>`.

## Reconstruction Formula section

`<div id="fourierFormulaBox" class="fourier-formula-box" aria-live="polite"></div>` stays (id is
read by fourier.js to write the live formula). fourier.js injects `.fourier-info` tooltip icons
into this box; restyle `.fourier-info` and the fixed `.fourier-tip` to archetype tokens (accent
periwinkle on hover instead of the old blue). Keep `.fourier-tip` as a fixed-position element
since fourier.js positions it.

## Interpreting the visualization (tabs)

Convert the bespoke `.fourier-tabs` / `.fourier-tab` / `.fourier-tabpanel` block and its inline
`<script>` to the archetype tab system used on estimation/bayesian (`.tabs` / `.tab` /
`.tab-panel` driven by `js/tabs.js`). Verify tabs.js's expected attributes
(`data-tab` on `.tab`, `data-panel` on `.tab-panel`, `.active`/`hidden` toggling) and match them;
remove the inline IIFE. Five tabs unchanged: Gibbs ringing, Extrapolation, Overshoot colors,
Center panel, Polynomial bases. The two inline color words ("Red shading", "blue shading") keep
the `.fourier-legend-red` / `.fourier-legend-blue` helper spans, restyled to tokens; these are
color legends, not emphasis, so they are allowed.

## styles/fourier.css rewrite

Replace the file with `.ui.fourier`-scoped rules:

- `.ui.fourier .fourier-canvases`: 3-up grid, `repeat(3, minmax(0,1fr))`, gap 16px; 2-up at
  `<=1100px`; 1-up at `<=680px` (container-aware via the same approach as the mock). Canvas cards:
  figcap label (mono, uppercase, muted) above a 1:1 `<canvas>` with token border/background.
- `.ui.fourier .fourier-band`: the 4 -> 2 -> 1 control band described above.
- `.ui.fourier .fourier-controls.demo-controls { display:block; }` (full-width, like pca).
- `.fourier-slabel` (label + output on one baseline), `.fourier-slider` (archetype slider:
  3px track, 14px accent thumb), `.fourier-out` (mono tabular readout).
- `.fourier-check` (checkbox row), `.fourier-radios` / `.fourier-radio` (stacked radio options).
- Custom dropdown: `.fourier-dd-trigger`, `.fourier-dd-menu`, `.fourier-dd-item`,
  `.fourier-dd-item.sel`, `.fourier-dd-sep`, `.fourier-dd-upload` (from the approved mock,
  tokenized; surface-strong menu, periwinkle tick/selected state).
- `.fourier-preset-native`: visually hidden but present (the real select).
- `.fourier-3d-canvas`: full-width 16:9, token border, grab cursor.
- `.fourier-hint` (the fly hint), `.fourier-warning` (basis-not-periodic note, amber kept but
  toned to the dark palette), `.fourier-info` + `.fourier-tip` (restyled), `.fourier-legend-red`
  / `.fourier-legend-blue`.
- `.opt-h` (Demo N eyebrow + h3 header) matching the mock.
- Remove all legacy `#4aa3ff` blue accents, the `.fourier-controls-block` boxed container, the
  `.fourier-upload-btn`, `.fourier-control-section/-heading/-grid/-column`, and the old
  `.fourier.container { max-width:1400px }` (the archetype caps content at 950px via
  `.ui.has-section-outline .container`).

## Responsive

- Control band: 4 -> 2 at 780px, 2 -> 1 at 460px (container width, matching the mock and the
  approved answer).
- Canvas grid: 3 -> 2 at 1100px, 2 -> 1 at 680px.
- Use container queries on `.demo` (the archetype sets `container-type: inline-size` on `.demo`)
  where the breakpoint should track the demo width, consistent with the other migrated demos.

## Out of scope (deferred, per backlog)

- Content edits, in-sequence note reorg, section-title shortening: content pass.
- Input-range / width robustness sweep: stability pass.
- Full mobile pass (touch targets, drawer, real device): mobile pass.

## Verification

- Serve the site and load `pages/fourier.html` with `?ui` not needed (body has `ui` class).
  Confirm: rail builds with Home at position 0; both demos render (3 canvases draw, spectrum and
  reconstruction update on slider/preset/basis change); the custom Image dropdown opens, selecting
  a preset updates all three canvases, "Upload image..." opens a file picker and a chosen file
  becomes the selected entry; the 3D canvas renders and orbit/fly/scale/toggles work; the
  reconstruction formula box fills and its info tooltips show; the interpretation tabs switch.
- Playwright screenshots at desktop (rail visible, 4-up band), tablet (~740px, 2x2 band, 2-up
  canvases), and narrow (~440px, 1-col) widths.
- Isolation: load an un-migrated page (manifold or generative_classification) and confirm no
  visual change from this work.
- Dash scan and `<em>`/`<strong>` scan over `pages/fourier.html`, `styles/fourier.css`,
  `js/fourier-image-dropdown.js` before commit.
