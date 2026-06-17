# Fourier Page Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate `pages/fourier.html` onto the `.ui` article-page archetype (matching estimation,
bayesian, regularization, pca), keeping `js/fourier.js` and `js/fourier3d.js` untouched and the
two interactive demos working, with the approved iOS-safe custom Image dropdown.

**Architecture:** Opt-in `<body class="ui fourier">`. Visual rules scoped under `.ui.fourier` in a
rewritten `styles/fourier.css`. All element ids and radio `name=` groups the JS reads are
preserved. The Image selector keeps a hidden real `<select id="fourierPreset">`; a new
`js/fourier-image-dropdown.js` shim builds a custom dropdown UI on top of it and relocates the
static `<input type="file" id="fourierUpload">` into its menu so the picker opens on iOS.

**Tech Stack:** Static HTML/CSS/JS. MathJax (tex-svg), Three.js 0.147, the archetype CSS
(tokens.css, system.css, components.css, article-ui.css), shared tabs.js and section-outline.js.

Spec: `docs/superpowers/specs/2026-06-17-fourier-migration-design.md`. Read it before starting.

**Process note:** These migrations verify visually with the Playwright harness at `/tmp/pwverify`
and with grep guards, not unit tests (the JS is untouched). Serve the repo root on
`http://localhost:8000`. Before each commit run the dash and emphasis guards from Task 0.

---

### Task 0: Guards and serving (reference, used by every task)

**Dash + emphasis guard** (must print nothing):

```bash
cd /mnt/c/Users/aughb/PersonalProjects/Visualization
grep -rlP "[\x{2014}\x{2013}]" pages/fourier.html styles/fourier.css js/fourier-image-dropdown.js 2>/dev/null
grep -nE "<(em|strong)[ >]|\*\*|(^|[^*])\*[^*]" pages/fourier.html 2>/dev/null
```

(The second grep may match math/JS incidentally; inspect any hit and confirm it is not prose
emphasis. The hard rule is no `<em>`/`<strong>` and no markdown `*`/`**` emphasis in page text.)

**Serve** (if not already running): `python3 -m http.server 8000` from the repo root.

**JS hook guard** (must print the same count before and after; ids must all survive). After the
HTML rewrite, confirm every id/name the JS reads still exists exactly once:

```bash
cd /mnt/c/Users/aughb/PersonalProjects/Visualization
for id in fourierBasis fourierPreset fourierRadius fourierRadiusValue fourierRadiusLabel \
  fourierFormulaBox fourierTiling fourierTilingText fourierOrigLabel fourierSpecLabel \
  fourierReconLabel fourierOrigCanvas fourierSpecCanvas fourierReconCanvas fourierUpload \
  fourier3dCanvas fourier3dScale fourier3dScaleOut fourier3dFlyHint fourier3dFlyLabel \
  fourier3dHideOOR fourier3dTile fourier3dTileExtent fourier3dTileExtentOut \
  fourier3dTileExtentRow fourier3dTileBasisNote; do
  c=$(grep -c "id=\"$id\"" pages/fourier.html); [ "$c" = "1" ] || echo "BAD id=$id count=$c";
done
grep -c 'name="fourier3dSource"' pages/fourier.html   # expect 2
grep -c 'name="fourier3dCamera"' pages/fourier.html   # expect 2
```

---

### Task 1: Rewrite `styles/fourier.css` to the `.ui.fourier` archetype

**Files:**
- Rewrite: `styles/fourier.css`

- [ ] **Step 1: Replace the entire file with the archetype-scoped stylesheet below.**

```css
/* fourier.css - migrated (.ui) page. Typography, sections, links, math, the demo control
 * system, tabs, callouts, and the footer come from the archetype. This file keeps the three
 * image canvases, the 3D canvas, the two 4-up control bands, the custom Image dropdown, and the
 * reconstruction-formula tooltip. */

/* --- Demo sub-headers (Demo 1 / Demo 2) --- */
.ui.fourier .fourier-demo-head { display: flex; align-items: baseline; gap: 12px; margin: 0 0 14px; }
.ui.fourier .fourier-demo-head .eyebrow { margin: 0; }
.ui.fourier .fourier-demo-head h3 { margin: 0; }

/* --- Three image canvases --- */
.ui.fourier .fourier-canvases { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; align-items: start; }
.ui.fourier .fourier-canvas-card { min-width: 0; display: flex; flex-direction: column; gap: 8px; }
.ui.fourier .fourier-canvas-card .figcap { margin: 0; }
.ui.fourier .fourier-canvas { width: 100%; aspect-ratio: 1 / 1; display: block; border-radius: var(--radius-md); background: #060607; border: 1px solid var(--hairline); }

/* --- Full-width control block (override the demo-controls grid, like pca) --- */
.ui.fourier .fourier-controls.demo-controls { display: block; margin-top: 20px; }

/* 4-up control band collapsing 4 -> 2 -> 1 by container width */
.ui.fourier .fourier-band { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 18px 30px; align-items: start; }
@container (max-width: 780px) { .ui.fourier .fourier-band { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@container (max-width: 460px) { .ui.fourier .fourier-band { grid-template-columns: 1fr; } }

/* --- Slider with inline readout --- */
.ui.fourier .fourier-slabel { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
.ui.fourier .fourier-slabel .demo-label { margin: 0; }
.ui.fourier .fourier-out { font: 500 12px/1 var(--font-mono); color: #cfd1d8; font-variant-numeric: tabular-nums; }
.ui.fourier .fourier-slider { -webkit-appearance: none; appearance: none; width: 100%; height: 3px; border-radius: 2px; background: var(--hairline-strong); margin: 12px 0 2px; cursor: pointer; }
.ui.fourier .fourier-slider::-webkit-slider-thumb { -webkit-appearance: none; width: 15px; height: 15px; border-radius: 50%; background: var(--accent); cursor: pointer; }
.ui.fourier .fourier-slider::-moz-range-thumb { width: 15px; height: 15px; border: 0; border-radius: 50%; background: var(--accent); cursor: pointer; }
.ui.fourier .fourier-subitem { margin-top: 16px; }

/* --- Checkboxes and radios --- */
.ui.fourier .fourier-check { display: flex; align-items: center; gap: 10px; font: 500 13px/1.35 var(--font-sans); color: #cfd1d8; cursor: pointer; }
.ui.fourier .fourier-check + .fourier-check { margin-top: 8px; }
.ui.fourier .fourier-check input { accent-color: var(--accent); width: 15px; height: 15px; flex: none; cursor: pointer; }
.ui.fourier .fourier-radios { display: flex; flex-direction: column; gap: 8px; }
.ui.fourier .fourier-radio { display: flex; align-items: center; gap: 9px; font: 500 13px/1.3 var(--font-sans); color: #cfd1d8; cursor: pointer; }
.ui.fourier .fourier-radio input { accent-color: var(--accent); width: 14px; height: 14px; flex: none; cursor: pointer; }
.ui.fourier .fourier-hint { font: 400 12px/1.45 var(--font-sans); color: var(--text-muted); margin: 7px 0 0; max-width: 44ch; }

/* --- Custom Image dropdown (real <select> hidden, custom UI on top) --- */
.ui.fourier .fourier-preset-native { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
.ui.fourier .fourier-dd-trigger { display: flex; align-items: center; justify-content: space-between; gap: 10px; width: 100%; font: 500 13px/1.3 var(--font-sans); color: #dadbe0; background: none; border: 0; border-bottom: 1px solid var(--hairline-strong); padding: 6px 2px; cursor: pointer; }
.ui.fourier .fourier-dd-trigger .chev { width: 9px; height: 6px; flex: none; }
.ui.fourier .fourier-dd-wrap { position: relative; }
.ui.fourier .fourier-dd-menu { position: absolute; left: 0; top: calc(100% + 6px); z-index: 50; min-width: 100%; background: var(--surface-strong); border: 1px solid rgba(255,255,255,.1); border-radius: var(--radius-md); padding: 5px; box-shadow: 0 12px 30px rgba(0,0,0,.5); }
.ui.fourier .fourier-dd-menu[hidden] { display: none; }
.ui.fourier .fourier-dd-item { display: flex; align-items: center; gap: 10px; width: 100%; text-align: left; font: 500 13px/1.2 var(--font-sans); color: #cfd1d8; background: none; border: 0; border-radius: 6px; padding: 9px 10px; cursor: pointer; }
.ui.fourier .fourier-dd-item:hover { background: rgba(255,255,255,.05); }
.ui.fourier .fourier-dd-item.sel { background: var(--accent-muted); color: #fff; }
.ui.fourier .fourier-dd-item.sel::after { content: ""; margin-left: auto; width: 4px; height: 8px; border: solid var(--accent); border-width: 0 2px 2px 0; transform: rotate(45deg) translateY(-1px); }
.ui.fourier .fourier-dd-sep { height: 1px; background: var(--hairline); margin: 5px 6px; }
.ui.fourier .fourier-dd-upload { color: #cfd1d8; }
.ui.fourier .fourier-dd-upload input[type="file"] { position: absolute; width: 1px; height: 1px; opacity: 0; pointer-events: none; }

/* --- 3D surface canvas --- */
.ui.fourier .fourier-3d-canvas { display: block; width: 100%; aspect-ratio: 16 / 9; border-radius: var(--radius-md); border: 1px solid var(--hairline); background: #060607; cursor: grab; }
.ui.fourier .fourier-3d-canvas:active { cursor: grabbing; }

/* --- Basis-not-periodic warning --- */
.ui.fourier .fourier-warning { margin: 7px 0 0; padding: 6px 9px; border-left: 3px solid #c79a3a; background: rgba(199,154,58,.10); color: #e0bd6e; font: 400 12px/1.4 var(--font-sans); border-radius: 2px; max-width: 44ch; }

/* --- Reconstruction formula box + info tooltip --- */
.ui.fourier .fourier-formula-box { font-variant-numeric: tabular-nums; }
.ui.fourier .fourier-info { display: inline-flex; align-items: center; justify-content: center; width: 14px; height: 14px; border-radius: 50%; border: 1px solid rgba(255,255,255,.45); color: var(--text-muted); font: italic 9px/1 var(--font-sans); cursor: help; margin-left: 5px; vertical-align: middle; }
.ui.fourier .fourier-info:hover { border-color: var(--accent); color: var(--accent); }
.fourier-tip { position: fixed; max-width: 320px; background: var(--surface-strong); border: 1px solid rgba(255,255,255,.22); color: #e8e8ec; font: 400 12px/1.45 var(--font-sans); padding: 9px 11px; border-radius: 8px; pointer-events: none; opacity: 0; transition: opacity .1s; z-index: 1000; white-space: pre-line; box-shadow: 0 6px 24px rgba(0,0,0,.5); }

/* --- Inline color legends used in the interpretation text (legends, not emphasis) --- */
.ui.fourier .fourier-legend-red { color: #ff6f6f; font-weight: 600; }
.ui.fourier .fourier-legend-blue { color: #6f9bff; font-weight: 600; }

/* --- Canvas grid responsive --- */
@container (max-width: 1100px) { .ui.fourier .fourier-canvases { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@container (max-width: 680px) { .ui.fourier .fourier-canvases { grid-template-columns: 1fr; } }
```

- [ ] **Step 2: Confirm no legacy blue (`#4aa3ff` / `#4090ff`) or `max-width: 1400px` remains.**

Run: `grep -nE "4aa3ff|1400px|fourier-controls-block|fourier-upload-btn" styles/fourier.css`
Expected: no output.

- [ ] **Step 3: Run the dash guard (Task 0) over styles/fourier.css.** Expected: no output.

- [ ] **Step 4: Commit.**

```bash
git add styles/fourier.css
git commit -m "redesign: rewrite fourier.css onto the .ui archetype control system"
```

---

### Task 2: Rewrite `pages/fourier.html` head, shell, and prose sections

This task does everything EXCEPT the two demo blocks and the interpretation-tab conversion (those
are Tasks 3-6). Leave the existing "Interactive Demo", "Reconstruction Formula", and
"Interpreting the visualization" sections in place for now so the page is never half-built; later
tasks replace their internals.

**Files:**
- Modify: `pages/fourier.html`

- [ ] **Step 1: Replace the `<head>` asset block.** Keep the MathJax inline config and
  `tex-svg.js`. Replace the stylesheet links and scripts with:

```html
  <link rel="stylesheet" href="../styles/tokens.css">
  <link rel="stylesheet" href="../styles/system.css">
  <link rel="stylesheet" href="../styles/components.css">
  <link rel="stylesheet" href="../styles/article-ui.css">
  <link rel="stylesheet" href="../styles/fourier.css">
  <link rel="stylesheet" href="../styles/section-outline.css">
  <script src="../js/theme.js"></script>

  <script defer src="../js/fourier-image-dropdown.js"></script>
  <script defer src="../js/fourier.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/three@0.147.0/build/three.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/three@0.147.0/examples/js/controls/OrbitControls.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/three@0.147.0/examples/js/controls/PointerLockControls.js"></script>
  <script defer src="../js/fourier3d.js"></script>
  <script defer src="../js/tabs.js"></script>
  <script defer src="../js/favicon.js"></script>
  <script type="module" src="../js/section-outline.js"></script>
```

(Removed: `article.css`, `base.css`, `responsive.css`, `formulas_layout.js`, `collapsible.js`.
Added: `theme.js`, `fourier-image-dropdown.js`, `tabs.js`. `fourier-image-dropdown.js` loads
before `fourier.js`.)

- [ ] **Step 2: Replace the body open + header.** Change `<body>` and the `<div class="container
  article fourier">` / `<header>` to:

```html
<body class="ui fourier">
  <div class="container">
    <header class="page-head">
      <div class="eyebrow">// Signals</div>
      <h1>Fourier Image Decomposition</h1>
      <p class="lede">Periodicity, smoothness assumptions, and why the DFT tiles an image into an infinite checkerboard.</p>
    </header>
    <main class="article-body">
```

Remove the old `<div class="subtitle">` and `<div class="home-link">` (Home is injected into the
rail by section-outline.js).

- [ ] **Step 3: Convert every prose `<section class="panel collapsible">` to
  `<section class="panel">`** (drop the `collapsible` class on all of them, including the demo and
  interpretation sections). Keep each `<h2>` and all prose text verbatim.

- [ ] **Step 4: Convert math display blocks.** Each `<div class="formula">$$...$$</div>` becomes
  wrapped so it matches estimation's pattern: a single display stays as
  `<div class="formulas"><div class="formula">$$...$$</div></div>`. Keep the LaTeX exactly.

- [ ] **Step 5: Convert the `<p class="note">` in "The Discrete Fourier Transform" to a callout**
  (components.css `.callout`):

```html
        <div class="callout">
          <div class="callout-label">Note</div>
          <div class="callout-body">
            The summations span $0$ to $N-1$ instead of $0$ to $N/2$. Indices $0$ to $N/2$ are those natural frequencies; indices $N/2$ to $N-1$ are their complex conjugates. For a real image, the conjugate half is redundant, carrying the same magnitude information as the positive half, but it is still required in the reconstruction sum so the imaginary parts cancel out and the rebuilt image comes out real-valued.
          </div>
        </div>
```

- [ ] **Step 6: Replace the footer.** Change `<footer class="footer">...</footer>` to:

```html
    <footer class="site-footer">
      <span class="credit">Created by <a href="https://linkedin.com/in/aughdon/">Aughdon Breslin</a></span>
    </footer>
```

- [ ] **Step 7: Run guards.** Dash guard (Task 0) over `pages/fourier.html`: no output. Then load
  `http://localhost:8000/pages/fourier.html` in the Playwright harness and confirm: the rail
  builds with Home at position 0, the eight section headers render in the dark theme, math renders.
  The demos may look unstyled-legacy at this point (their internals are replaced in later tasks);
  that is expected. Verify no console errors from missing `fourier-image-dropdown.js` (it does not
  exist yet, but `defer` + its own guard in Task 4 means it must no-op if the dropdown markup is
  absent; until Task 3/4 the page still has the OLD demo markup, so the shim must tolerate that).
  If console shows the 404 for `fourier-image-dropdown.js`, that is acceptable until Task 4 creates
  it; note it and proceed.

- [ ] **Step 8: Commit.**

```bash
git add pages/fourier.html
git commit -m "redesign: migrate fourier head, shell, and prose sections to the .ui archetype"
```

---

### Task 3: Demo 1 markup (image decomposition)

**Files:**
- Modify: `pages/fourier.html` (the "Interactive Demo" section)

- [ ] **Step 1: Replace the `<h2>Interactive Demo</h2>` section's body.** Replace everything from
  `<div class="fourier-canvas-grid">` through the end of the Demo-1 controls (the first
  `fourier-control-section`) with the Demo 1 block. Keep the section element and its `<h2>`. Demo 2
  is handled in Task 5; for now leave the old 3D markup (`fourier-control-section` "3D View" plus
  `#fourier3dCanvas`) in place below the new Demo 1 block.

```html
  <div class="demo">
    <div class="fourier-demo-head"><span class="eyebrow">Demo 1</span><h3>Image decomposition</h3></div>

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
          <select id="fourierBasis">
            <option value="fourier" selected>Fourier series</option>
            <option value="poly">Legendre polynomials</option>
            <option value="cheb">Chebyshev polynomials</option>
            <option value="haar">Haar wavelets</option>
          </select>
        </div>

        <div class="demo-group fourier-image-group">
          <span class="demo-label">Image</span>
          <select id="fourierPreset" class="fourier-preset-native">
            <option value="quadrant" selected>Quadrant</option>
            <option value="circle">Hard circle</option>
            <option value="gaussian">Gaussian blob</option>
            <option value="step">Step function</option>
            <option value="rect">Rectangle</option>
          </select>
          <input id="fourierUpload" type="file" accept="image/*" class="fourier-upload-input" hidden />
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
```

Note: `#fourierUpload` is in the static HTML (with `hidden`) so its id always exists for
fourier.js regardless of script order; the dropdown shim (Task 4) relocates it into the menu.

- [ ] **Step 2: Run the JS hook guard (Task 0).** All Demo-1 ids present exactly once. Dash guard:
  no output.

- [ ] **Step 3: Visual check.** Reload the page. The three canvases render in a row and draw
  content (Quadrant by default), the slider moves and updates the reconstruction, the radius
  readout updates, the basis select changes the spectrum/reconstruction. The Image select is
  visually hidden (a custom trigger appears only after Task 4; until then the group shows just the
  "Image" label, which is expected). Tiling checkbox toggles the periodic extension.

- [ ] **Step 4: Commit.**

```bash
git add pages/fourier.html
git commit -m "redesign: rebuild fourier Demo 1 (image decomposition) on the demo control band"
```

---

### Task 4: Custom Image dropdown shim (`js/fourier-image-dropdown.js`)

**Files:**
- Create: `js/fourier-image-dropdown.js`

- [ ] **Step 1: Create the shim.** It must: tolerate the markup being absent (no-op), hide the
  real select, build a trigger + menu, relocate `#fourierUpload` into the menu's upload label,
  sync via `change` + `MutationObserver`, and dispatch a synthetic `change` on preset selection so
  fourier.js runs untouched.

```js
/* fourier-image-dropdown.js - custom Image dropdown for the Fourier demo.
 *
 * A native <select> cannot open the file picker on iOS (a programmatic click from a change
 * handler is blocked); only a direct tap on a label-wrapped file input works. This shim keeps the
 * real <select id="fourierPreset"> in the DOM (fourier.js reads its value, listens for its
 * change, and appends an "__uploaded__" option), hides it, and renders a custom dropdown whose
 * "Upload image..." item is a <label> wrapping the real <input type="file" id="fourierUpload">.
 * fourier.js is never modified. */
(function () {
  function init() {
    const select = document.getElementById("fourierPreset");
    const group = select && select.closest(".fourier-image-group");
    const upload = document.getElementById("fourierUpload");
    if (!select || !group || !upload) return; // markup not present: no-op

    const UPLOAD_VALUE = "__uploaded__";

    const wrap = document.createElement("div");
    wrap.className = "fourier-dd-wrap";

    const trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "fourier-dd-trigger";
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    trigger.innerHTML =
      '<span class="fourier-dd-value"></span>' +
      '<svg class="chev" viewBox="0 0 9 6" aria-hidden="true"><path d="M1 1l3.5 3.5L8 1" fill="none" stroke="#7a7c84" stroke-width="1.3"/></svg>';

    const menu = document.createElement("div");
    menu.className = "fourier-dd-menu";
    menu.setAttribute("role", "listbox");
    menu.hidden = true;

    // The upload label wraps the real file input (moved out of static HTML).
    const uploadLabel = document.createElement("label");
    uploadLabel.className = "fourier-dd-item fourier-dd-upload";
    uploadLabel.textContent = "Upload image…"; // "Upload image..." with an ellipsis char
    upload.removeAttribute("hidden");
    uploadLabel.appendChild(upload);

    wrap.appendChild(trigger);
    wrap.appendChild(menu);
    // Place the custom UI right after the hidden select.
    select.insertAdjacentElement("afterend", wrap);

    const valueEl = trigger.querySelector(".fourier-dd-value");

    function currentText() {
      const opt = select.options[select.selectedIndex];
      return opt ? opt.textContent : "";
    }

    function renderMenu() {
      // Rebuild preset items from the select's current options, then the separator + upload.
      Array.from(menu.querySelectorAll(".fourier-dd-item:not(.fourier-dd-upload), .fourier-dd-sep"))
        .forEach((n) => n.remove());
      const frag = document.createDocumentFragment();
      Array.from(select.options).forEach((opt) => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "fourier-dd-item" + (opt.selected ? " sel" : "");
        item.setAttribute("role", "option");
        item.textContent = opt.textContent;
        item.addEventListener("click", () => {
          select.value = opt.value;
          select.dispatchEvent(new Event("change", { bubbles: true }));
          close();
        });
        frag.appendChild(item);
      });
      const sep = document.createElement("div");
      sep.className = "fourier-dd-sep";
      frag.appendChild(sep);
      menu.insertBefore(frag, uploadLabel);
    }

    function syncTrigger() {
      valueEl.textContent = currentText();
    }

    function open() {
      renderMenu();
      menu.hidden = false;
      trigger.setAttribute("aria-expanded", "true");
      document.addEventListener("mousedown", onOutside);
      document.addEventListener("keydown", onKey);
    }
    function close() {
      menu.hidden = true;
      trigger.setAttribute("aria-expanded", "false");
      document.removeEventListener("mousedown", onOutside);
      document.removeEventListener("keydown", onKey);
    }
    function onOutside(e) { if (!wrap.contains(e.target)) close(); }
    function onKey(e) { if (e.key === "Escape") { close(); trigger.focus(); } }

    trigger.addEventListener("click", () => { menu.hidden ? open() : close(); });

    // Keep the trigger (and an open menu) in sync when fourier.js mutates the select after an
    // upload: it sets select.value = "__uploaded__" and appends an option without firing change.
    select.addEventListener("change", () => { syncTrigger(); if (!menu.hidden) renderMenu(); });
    new MutationObserver(() => { syncTrigger(); if (!menu.hidden) renderMenu(); })
      .observe(select, { childList: true, attributes: true, attributeFilter: ["value"] });
    // The upload sets select.value programmatically (no change event, no attribute mutation),
    // so also resync after the file input finishes.
    upload.addEventListener("change", () => { setTimeout(syncTrigger, 0); });

    if (!select.options[UPLOAD_VALUE]) { /* no-op marker so UPLOAD_VALUE is referenced */ }
    syncTrigger();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
```

Important: this script and `fourier.js` both run on `DOMContentLoaded`. `fourier.js` is loaded
AFTER this file in the head (Task 2 Step 1), so this shim's `DOMContentLoaded` listener registers
first and runs first, relocating `#fourierUpload` into the menu before fourier.js binds to it.
fourier.js binds by id, so the relocation does not break the binding.

- [ ] **Step 2: Remove the leftover ellipsis check.** The line
  `if (!select.options[UPLOAD_VALUE]) { ... }` is a no-op kept only to reference `UPLOAD_VALUE`;
  delete both that line and the `const UPLOAD_VALUE` line if a linter flags unused vars, OR keep
  them. Either is fine; do not leave a dangling reference.

- [ ] **Step 3: Dash guard over the new file.** Note `"Upload image…"` uses the escaped
  ellipsis; ensure no literal em/en dash. Expected: no output.

- [ ] **Step 4: Visual + behavior check.** Reload. The Image group shows the custom trigger reading
  "Quadrant" with a chevron. Click it: the menu lists the five presets (Quadrant ticked), a
  divider, then "Upload image...". Click "Hard circle": all three canvases update and the trigger
  reads "Hard circle". Click the trigger, click "Upload image...", choose a local image: the
  picker opens, the canvases update, and the trigger/menu show the uploaded filename as the
  selected entry. Escape and outside-click close the menu.

- [ ] **Step 5: Commit.**

```bash
git add js/fourier-image-dropdown.js
git commit -m "redesign: iOS-safe custom Image dropdown for the Fourier demo"
```

---

### Task 5: Demo 2 markup (3D height-map)

**Files:**
- Modify: `pages/fourier.html` (replace the old "3D View" control section and the
  `#fourier3dCanvas` placement)

- [ ] **Step 1: Replace the old Demo-2 markup.** Remove the second `fourier-control-section`
  ("3D View") and the standalone `<canvas id="fourier3dCanvas">` that followed the controls block,
  and the now-empty `.fourier-controls-block` wrapper. Insert this Demo 2 block immediately after
  the Demo 1 `.demo` block, still inside the "Interactive Demo" section:

```html
  <div class="demo">
    <div class="fourier-demo-head"><span class="eyebrow">Demo 2</span><h3>3D height-map</h3></div>
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

Preserve the inline `style="display:none"` on `fourier3dFlyHint`, `fourier3dTileExtentRow`,
`fourier3dTileBasisNote` and the `disabled` on `fourier3dTileExtent` exactly (fourier3d.js toggles
them).

- [ ] **Step 2: JS hook guard (Task 0).** All `fourier3d*` ids present once; both radio name
  counts equal 2. Dash guard: no output.

- [ ] **Step 3: Visual + behavior check.** Reload. The 3D canvas renders the surface. Orbit drag
  rotates; switching to Fly shows the fly hint and enables pointer-lock movement; the Height scale
  slider changes the relief and the readout; Hide overshoot and Periodic tiling toggle; selecting
  a non-Fourier basis shows the "Basis isn't periodic" note when Periodic tiling is on.

- [ ] **Step 4: Commit.**

```bash
git add pages/fourier.html
git commit -m "redesign: rebuild fourier Demo 2 (3D height-map) on the demo control band"
```

---

### Task 6: Reconstruction Formula restyle + Interpreting tabs conversion

**Files:**
- Modify: `pages/fourier.html`

- [ ] **Step 1: Reconstruction Formula section.** Keep
  `<div id="fourierFormulaBox" class="fourier-formula-box" aria-live="polite"></div>` inside its
  `<section class="panel">`. No markup change beyond the section already being de-`collapsible`d in
  Task 2. The `.fourier-info` / `.fourier-tip` restyle is already in fourier.css (Task 1).

- [ ] **Step 2: Convert the interpretation tabs to the shared `.tabs` system.** tabs.js expects a
  `.tabs` bar of `[data-tab]` buttons and sibling `[data-panel]` elements in the bar's
  `parentElement`. Replace the `<div class="fourier-tabs">` + five `.fourier-tabpanel` blocks with:

```html
        <div class="tabs" role="tablist">
          <button type="button" class="tab active" data-tab="gibbs">Gibbs ringing</button>
          <button type="button" class="tab" data-tab="extrap">Extrapolation</button>
          <button type="button" class="tab" data-tab="overshoot">Overshoot colors</button>
          <button type="button" class="tab" data-tab="center">Center panel</button>
          <button type="button" class="tab" data-tab="poly">Polynomial bases</button>
        </div>

        <div class="tab-panel" data-panel="gibbs" role="tabpanel">
          ... existing Gibbs paragraphs verbatim ...
        </div>
        <div class="tab-panel" data-panel="extrap" role="tabpanel" hidden>
          ... existing Extrapolation paragraphs verbatim ...
        </div>
        <div class="tab-panel" data-panel="overshoot" role="tabpanel" hidden>
          ... existing Overshoot paragraphs verbatim ...
        </div>
        <div class="tab-panel" data-panel="center" role="tabpanel" hidden>
          ... existing Center-panel paragraphs verbatim ...
        </div>
        <div class="tab-panel" data-panel="poly" role="tabpanel" hidden>
          ... existing Polynomial-bases paragraphs verbatim ...
        </div>
```

Keep all paragraph text verbatim. In the Overshoot panel, keep the two color-legend spans but
update their class to the page helpers: the words describing red/blue shading use
`<span class="fourier-legend-red">Red shading</span>` and
`<span class="fourier-legend-blue">blue shading</span>` (these are color legends, not emphasis).
If the current markup uses inline-styled spans, replace them with these classes.

- [ ] **Step 3: Remove the inline tab `<script>` IIFE** at the bottom of the body (the
  `var tabs = document.querySelectorAll('.fourier-tab')` block). tabs.js now drives the tabs.

- [ ] **Step 4: Guards.** Dash guard: no output. Confirm the bar and panels are siblings (same
  parent) so tabs.js scopes correctly: the `.tabs` div and the five `.tab-panel` divs share the
  parent `<section>`.

- [ ] **Step 5: Visual + behavior check.** Reload. The five interpretation tabs render in the
  archetype tab style; clicking each shows its panel and hides the others; the initial tab is
  Gibbs ringing. The reconstruction formula box fills with the live formula and its info icons show
  tooltips on hover, styled in the dark palette.

- [ ] **Step 6: Commit.**

```bash
git add pages/fourier.html
git commit -m "redesign: convert fourier interpretation tabs to the shared tab system and restyle the formula tooltip"
```

---

### Task 7: Full verification, isolation, and backlog update

**Files:**
- Modify: `docs/design/redesign-backlog.md`
- Modify: the `project_redesign_backlog` memory file and `MEMORY.md` pointer if needed.

- [ ] **Step 1: Responsive screenshots.** With the Playwright harness, capture
  `pages/fourier.html` at three widths and review each:
  - Desktop ~1280px: rail visible with Home at 0; both control bands 4-up; canvases 3-up.
  - Tablet ~740px: control bands fold to 2x2; canvases 2-up; 3D canvas full width.
  - Narrow ~440px: control bands 1-col; canvases 1-col.
  Confirm no clipping, no horizontal scroll, and the custom dropdown menu is not cut off.

- [ ] **Step 2: Full behavior pass** (desktop width): cycle all four bases; the center canvas
  label/content updates per basis (spectrum vs coefficient matrix vs wavelet pyramid); the radius
  slider and tiling toggle work; the custom Image dropdown selects presets and uploads; the 3D view
  responds to source/camera/scale/toggles; the interpretation tabs switch. No console errors.

- [ ] **Step 3: Isolation.** Load an un-migrated page (`pages/manifold.html` or
  `pages/generative_classification.html`) and confirm it looks identical to before this work (no
  `.ui` leakage). The `fourier.css` rules are all scoped under `.ui.fourier`, so this should hold;
  verify visually.

- [ ] **Step 4: Final guards.** Run the dash guard and the JS hook guard (Task 0) once more over
  the final `pages/fourier.html`, `styles/fourier.css`, `js/fourier-image-dropdown.js`. All clean;
  all ids present once; both radio name counts equal 2.

- [ ] **Step 5: Update the backlog.** In `docs/design/redesign-backlog.md`, move fourier from the
  "Next" list to "Done": change the Phase 4 line to read
  `Done: estimation (pilot), bayesian, regularization, pca, fourier. Next: manifold,
  generative_classification, distributions.`

- [ ] **Step 6: Update memory.** Update the `project_redesign_backlog` memory file's migration
  status to include fourier as done, and refresh the `MEMORY.md` pointer hook if it lists migrated
  pages.

- [ ] **Step 7: Commit.**

```bash
git add docs/design/redesign-backlog.md
git commit -m "docs: mark fourier migration done in the redesign backlog"
```

---

## Self-review notes

- Every JS hook id and both radio `name=` groups are reproduced in Tasks 3 and 5 and guarded in
  Task 0. fourier.js and fourier3d.js are never edited.
- The `#fourierUpload` input exists in static HTML (Task 3) so its id is present regardless of
  script ordering; the shim relocates it (Task 4). `#fourierPreset` stays a real `<select>` so
  fourier.js's `querySelector('option[value="__uploaded__"]')` and option append keep working.
- tabs.js contract (sibling `.tabs` bar + `[data-panel]` panels, `hidden` toggling, `.active`
  initial) is matched in Task 6; the inline IIFE is removed.
- Footer (`site-footer` / `credit`) and callout (`callout` / `callout-label` / `callout-body`)
  markup is copied from the verified estimation/bayesian/components.css patterns.
- Control band 4 -> 2 (780px) -> 1 (460px) and canvas grid 3 -> 2 (1100px) -> 1 (680px) use
  container queries on `.demo` (archetype sets `container-type: inline-size`), matching the
  approved mock and the user's confirmed breakpoints.
- Deferred per backlog: content edits, title shortening, input-range robustness, full mobile pass.
