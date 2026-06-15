# Section Outline Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auto-generated on-page outline to all eight content pages: a sticky rail in a reserved left column on desktop, a hamburger drawer on narrow/phone widths, with scrollspy and click-to-jump (opening collapsed panels first).

**Architecture:** One ES module `js/section-outline.js` exports pure string helpers (unit-tested with `node:test`) and, when a DOM is present, discovers each page's top-level `.panel` blocks, builds the rail and drawer, wires click/scroll/deep-link/scrollspy, then self-initializes. A companion `styles/section-outline.css` handles the reserved-column rail, the hamburger drawer, the 1100px breakpoint, and reduced-motion. The eight pages gain two include lines; manifold.html additionally has its panels made collapsible.

**Tech Stack:** Vanilla ES modules + CSS, no build step. Tests: `node:test` + `node:assert/strict` (existing convention in `test/manifold/`). Behavioral verification: Playwright MCP against `python3 -m http.server`.

**Spec:** `docs/superpowers/specs/2026-06-14-section-outline-nav-design.md`

**Dependency order:** Task 1 (pure helpers + tests) first. Task 2 builds discovery/markup on those helpers. Tasks 3 and 4 extend behavior. Task 5 (CSS) can be done after Task 2. Task 6 (page includes) and Task 7 (manifold collapsible) enable the Task 8 verification pass, which is last.

---

## Task 1: Pure helpers with unit tests

**Files:**
- Create: `js/section-outline.js`
- Test: `test/section-outline.test.js`

- [ ] **Step 1: Write the failing test**

```js
// test/section-outline.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { slugify, uniqueId, normalizeLabel } from '../js/section-outline.js';

test('slugify lowercases and hyphenates non-word runs', () => {
  assert.equal(slugify('PCA and SVD'), 'pca-and-svd');
});

test('slugify strips punctuation and trims edge hyphens', () => {
  assert.equal(slugify('  Mean squared error (MSE) '), 'mean-squared-error-mse');
});

test('slugify falls back to "section" for empty input', () => {
  assert.equal(slugify(''), 'section');
  assert.equal(slugify('   '), 'section');
});

test('uniqueId returns base when free, then numeric suffixes', () => {
  const used = new Set(['overview']);
  assert.equal(uniqueId('overview', used), 'overview-2');
  assert.equal(uniqueId('overview', used), 'overview-3');
  assert.equal(uniqueId('dataset', used), 'dataset');
});

test('normalizeLabel collapses whitespace and newlines', () => {
  assert.equal(normalizeLabel('  Active\n  Distributions '), 'Active Distributions');
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test test/section-outline.test.js`
Expected: FAIL with `Cannot find module` or `slugify is not a function` (file does not exist yet).

- [ ] **Step 3: Create the module with the pure helpers**

```js
// js/section-outline.js
// On-page section outline: a sticky rail in a reserved left column on desktop,
// a hamburger drawer on narrow/phone widths. Auto-generated from each page's
// top-level .panel blocks. Pure helpers are exported for unit testing; the DOM
// build self-initializes only when a document is present (so node:test can
// import the helpers without a DOM).

export function slugify(text) {
  const s = String(text)
    .toLowerCase()
    .trim()
    .replace(/[^\w]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return s || 'section';
}

export function uniqueId(base, used) {
  let id = base;
  let n = 2;
  while (used.has(id)) {
    id = `${base}-${n}`;
    n += 1;
  }
  used.add(id);
  return id;
}

export function normalizeLabel(text) {
  return String(text).replace(/\s+/g, ' ').trim();
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test test/section-outline.test.js`
Expected: PASS with all 5 tests passing.

- [ ] **Step 5: Commit**

```bash
git add js/section-outline.js test/section-outline.test.js
git commit -m "feat: section-outline pure helpers (slugify, uniqueId, normalizeLabel)"
```

---

## Task 2: Panel discovery and rail/drawer construction

**Files:**
- Modify: `js/section-outline.js`
- Create: `styles/section-outline.css` (minimal stub here; full styles in Task 5)
- Verify with: Playwright MCP

- [ ] **Step 1: Add discovery + build + init to the module**

Append to `js/section-outline.js` (after the pure helpers):

```js
// ---- DOM build (skipped under node:test where document is undefined) ----

function collectPanels(root) {
  const panels = Array.from(root.querySelectorAll('.panel')).filter(
    (el) => !el.parentElement || !el.parentElement.closest('.panel')
  );
  const used = new Set(Array.from(root.querySelectorAll('[id]')).map((el) => el.id));
  const entries = [];
  for (const panel of panels) {
    const heading = panel.querySelector(':scope > h2, :scope > h3');
    if (!heading) continue;
    const label = normalizeLabel(heading.textContent);
    if (!label) continue;
    if (panel.id) used.add(panel.id);
    else panel.id = uniqueId(slugify(label), used);
    panel.style.scrollMarginTop = 'var(--outline-scroll-offset, 16px)';
    entries.push({ id: panel.id, label, panel });
  }
  return entries;
}

function buildNav(entries) {
  const nav = document.createElement('nav');
  nav.className = 'section-outline';
  nav.setAttribute('aria-label', 'On this page');

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'section-outline-toggle';
  btn.setAttribute('aria-label', 'Open section outline');
  btn.setAttribute('aria-expanded', 'false');
  btn.setAttribute('aria-controls', 'section-outline-panel');
  btn.innerHTML = '<span class="section-outline-bars" aria-hidden="true"></span>';

  const backdrop = document.createElement('div');
  backdrop.className = 'section-outline-backdrop';
  backdrop.hidden = true;

  const panel = document.createElement('div');
  panel.className = 'section-outline-panel';
  panel.id = 'section-outline-panel';

  const heading = document.createElement('div');
  heading.className = 'section-outline-heading';
  heading.textContent = 'On this page';
  panel.appendChild(heading);

  const list = document.createElement('ul');
  list.className = 'section-outline-list';
  const linkById = new Map();
  for (const entry of entries) {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = `#${entry.id}`;
    a.textContent = entry.label;
    a.dataset.target = entry.id;
    li.appendChild(a);
    list.appendChild(li);
    linkById.set(entry.id, a);
  }
  panel.appendChild(list);

  nav.append(btn, backdrop, panel);
  return { nav, btn, backdrop, panel, list, linkById };
}

function initSectionOutline() {
  const entries = collectPanels(document);
  if (entries.length < 2) return; // not worth an outline
  const ui = buildNav(entries);
  document.body.appendChild(ui.nav);
  document.body.classList.add('has-section-outline');
  return { entries, ui };
}

function ready(fn) {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', fn);
  } else {
    fn();
  }
}

if (typeof document !== 'undefined') {
  ready(initSectionOutline);
}
```

- [ ] **Step 2: Create a minimal CSS stub so the nav is visible during verification**

```css
/* styles/section-outline.css (full styling lands in Task 5). */
.section-outline-list { list-style: none; margin: 0; padding: 0; }
.section-outline-list a { display: block; padding: 4px 8px; text-decoration: none; }
```

- [ ] **Step 3: Add the includes to one page for verification (pca.html)**

In `pages/pca.html`, add inside `<head>` after the existing stylesheet links (near line 32):

```html
  <link rel="stylesheet" href="../styles/section-outline.css">
```

and after the existing `<script defer>` includes (near line 37):

```html
  <script type="module" src="../js/section-outline.js"></script>
```

- [ ] **Step 4: Verify discovery renders the right entries (Playwright MCP)**

Start a server: `python3 -m http.server 8000` (run from repo root, in the background).
With the Playwright MCP tools:
1. `browser_navigate` to `http://localhost:8000/pages/pca.html`.
2. `browser_resize` to 1500x900.
3. `browser_evaluate` returning the outline labels:

```js
() => Array.from(document.querySelectorAll('.section-outline-list a')).map(a => a.textContent)
```

Expected (order matters): `["Overview","PCA and SVD","Covariance eigendecomposition and PCA as a change of basis","Visual Step Through","Dimensionality reduction and reconstruction","Conclusion"]`.

4. `browser_evaluate` to confirm every panel got an id:

```js
() => Array.from(document.querySelectorAll('.panel')).filter(p => !p.closest('.panel:not(:scope)') && !p.id).length
```

Expected: a number; confirm each top-level panel referenced by the list has a matching element via `document.getElementById`. Specifically check `["overview","pca-and-svd"].every(id => !!document.getElementById(id))` returns `true`.

- [ ] **Step 5: Commit**

```bash
git add js/section-outline.js styles/section-outline.css pages/pca.html
git commit -m "feat: section-outline panel discovery and nav construction"
```

---

## Task 3: Click-to-jump, collapsed-panel open, and deep links

**Files:**
- Modify: `js/section-outline.js`
- Verify with: Playwright MCP

- [ ] **Step 1: Add navigation behavior to the module**

Add these helpers above `initSectionOutline` in `js/section-outline.js`:

```js
const PHONE_MAX = 640; // matches collapsible.js breakpoint

function isCollapsedPanel(panel) {
  return panel.classList.contains('collapsible') && !panel.classList.contains('open');
}

function openIfCollapsed(panel) {
  // Reuse collapsible.js: clicking its head toggles open (and fires resize +
  // MathJax retypeset). Only click when actually collapsed to avoid closing it.
  if (!isCollapsedPanel(panel)) return;
  const head = panel.querySelector(':scope > .collapsible-head');
  if (head) head.click();
}

function prefersReducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function scrollToPanel(panel) {
  panel.scrollIntoView({
    behavior: prefersReducedMotion() ? 'auto' : 'smooth',
    block: 'start',
  });
}

function navigateTo(id, byId) {
  const panel = document.getElementById(id);
  if (!panel) return;
  openIfCollapsed(panel);
  // Defer scroll one frame so a just-opened panel has its final height.
  requestAnimationFrame(() => scrollToPanel(panel));
  history.replaceState(null, '', `#${id}`);
}
```

Then, inside `initSectionOutline`, after `document.body.appendChild(ui.nav)`, wire clicks and deep links:

```js
  ui.list.addEventListener('click', (e) => {
    const a = e.target.closest('a[data-target]');
    if (!a) return;
    e.preventDefault();
    navigateTo(a.dataset.target, ui.linkById);
    closeDrawer(ui); // defined in Task 4; safe no-op if drawer is closed
  });

  // Deep link on load: open + scroll to the hashed panel after layout settles.
  if (location.hash.length > 1) {
    const id = decodeURIComponent(location.hash.slice(1));
    requestAnimationFrame(() => navigateTo(id, ui.linkById));
  }
```

Note: `closeDrawer` is added in Task 4. For this task, add a temporary stub near the top of the module so the click handler runs:

```js
function closeDrawer() {} // replaced in Task 4
```

- [ ] **Step 2: Verify click-to-jump on desktop (Playwright MCP)**

With the server running and `pages/pca.html` open at 1500x900:
1. `browser_evaluate`: click the "Conclusion" entry and report scroll change.

```js
() => {
  const a = [...document.querySelectorAll('.section-outline-list a')].find(x => x.textContent === 'Conclusion');
  const before = window.scrollY;
  a.click();
  return new Promise(res => setTimeout(() => res({ before, after: window.scrollY, hash: location.hash }), 600));
}
```

Expected: `after` is greater than `before`, and `hash` is `#conclusion`.

- [ ] **Step 3: Verify collapsed-panel open on phone width**

Requires collapsible.js on the page; pca.html already includes it.
1. `browser_resize` to 390x844.
2. `browser_evaluate`: confirm a non-default panel starts collapsed, click its entry, confirm it opens.

```js
() => {
  const panel = document.getElementById('pca-and-svd');
  const wasOpen = panel.classList.contains('open');
  const a = document.querySelector('.section-outline-list a[data-target="pca-and-svd"]');
  a.click();
  return new Promise(res => setTimeout(() => res({ wasOpen, nowOpen: panel.classList.contains('open') }), 400));
}
```

Expected: `wasOpen` is `false`, `nowOpen` is `true`.

- [ ] **Step 4: Verify deep link on load**

1. `browser_navigate` to `http://localhost:8000/pages/pca.html#conclusion`.
2. `browser_resize` to 1500x900.
3. `browser_evaluate`:

```js
() => new Promise(res => setTimeout(() => res({ scrolled: window.scrollY > 100 }), 800))
```

Expected: `scrolled` is `true`.

- [ ] **Step 5: Commit**

```bash
git add js/section-outline.js
git commit -m "feat: section-outline click-to-jump, collapsed-panel open, deep links"
```

---

## Task 4: Hamburger drawer toggle and scrollspy

**Files:**
- Modify: `js/section-outline.js`
- Verify with: Playwright MCP

- [ ] **Step 1: Replace the `closeDrawer` stub with real drawer control**

Remove the `function closeDrawer() {}` stub and add real drawer functions above `initSectionOutline`:

```js
function openDrawer(ui) {
  ui.nav.classList.add('open');
  ui.btn.setAttribute('aria-expanded', 'true');
  ui.btn.setAttribute('aria-label', 'Close section outline');
  ui.backdrop.hidden = false;
  const first = ui.list.querySelector('a');
  if (first) first.focus();
}

function closeDrawer(ui) {
  if (!ui || !ui.nav.classList.contains('open')) return;
  ui.nav.classList.remove('open');
  ui.btn.setAttribute('aria-expanded', 'false');
  ui.btn.setAttribute('aria-label', 'Open section outline');
  ui.backdrop.hidden = true;
  ui.btn.focus();
}
```

- [ ] **Step 2: Wire the toggle, backdrop, and Esc inside `initSectionOutline`**

Add after the click-list wiring from Task 3:

```js
  ui.btn.addEventListener('click', () => {
    if (ui.nav.classList.contains('open')) closeDrawer(ui);
    else openDrawer(ui);
  });
  ui.backdrop.addEventListener('click', () => closeDrawer(ui));
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeDrawer(ui);
  });
```

- [ ] **Step 3: Add scrollspy with IntersectionObserver**

Add a function above `initSectionOutline`:

```js
function wireScrollspy(entries, linkById) {
  let activeId = null;
  const setActive = (id) => {
    if (id === activeId) return;
    if (activeId && linkById.get(activeId)) {
      linkById.get(activeId).classList.remove('active');
      linkById.get(activeId).removeAttribute('aria-current');
    }
    activeId = id;
    const a = linkById.get(id);
    if (a) {
      a.classList.add('active');
      a.setAttribute('aria-current', 'true');
    }
  };
  const visible = new Set();
  const obs = new IntersectionObserver(
    (records) => {
      for (const r of records) {
        if (r.isIntersecting) visible.add(r.target.id);
        else visible.delete(r.target.id);
      }
      // Pick the visible panel that appears earliest in document order.
      const order = entries.map((e) => e.id);
      const top = order.find((id) => visible.has(id));
      if (top) setActive(top);
    },
    { rootMargin: '-10% 0px -70% 0px', threshold: 0 }
  );
  for (const e of entries) obs.observe(e.panel);
}
```

Call it at the end of `initSectionOutline`:

```js
  wireScrollspy(entries, ui.linkById);
```

- [ ] **Step 4: Verify drawer + scrollspy (Playwright MCP)**

Drawer (phone width 390x844 on pca.html):

```js
() => {
  const btn = document.querySelector('.section-outline-toggle');
  btn.click();
  const openState = document.querySelector('.section-outline').classList.contains('open');
  document.querySelector('.section-outline-backdrop').click();
  const closedState = document.querySelector('.section-outline').classList.contains('open');
  return { openState, closedState };
}
```

Expected: `{ openState: true, closedState: false }`.

Scrollspy (desktop width 1500x900):

```js
() => {
  document.getElementById('conclusion').scrollIntoView();
  return new Promise(res => setTimeout(() => {
    const active = document.querySelector('.section-outline-list a.active');
    res({ active: active ? active.textContent : null });
  }, 500));
}
```

Expected: `active` is `"Conclusion"` (or an adjacent late entry, depending on panel heights; confirm it is not an early entry like "Overview").

- [ ] **Step 5: Commit**

```bash
git add js/section-outline.js
git commit -m "feat: section-outline hamburger drawer and scrollspy"
```

---

## Task 5: Full stylesheet (reserved rail, drawer, breakpoint, reduced-motion)

**Files:**
- Modify: `styles/section-outline.css`
- Verify with: Playwright MCP

- [ ] **Step 1: Replace the stub with the full stylesheet**

Overwrite `styles/section-outline.css`:

```css
/* styles/section-outline.css
 * On-page section outline. Desktop (>=1100px): a fixed rail in a reserved left
 * column. Narrow/phone (<1100px): a hamburger button opening a slide-in drawer.
 */
:root {
  --outline-rail-w: 160px;
  --outline-rail-gap: 16px;
  --outline-scroll-offset: 16px;
}

.section-outline-list { list-style: none; margin: 0; padding: 0; }
.section-outline-list a {
  display: block;
  padding: 6px 10px;
  border-left: 2px solid transparent;
  color: #5a6472;
  text-decoration: none;
  font-size: 0.9rem;
  line-height: 1.3;
  border-radius: 0 4px 4px 0;
}
.section-outline-list a:hover { background: rgba(0, 0, 0, 0.05); color: #1f2933; }
.section-outline-list a.active {
  color: #1f2933;
  border-left-color: #3b82f6;
  background: rgba(59, 130, 246, 0.08);
  font-weight: 600;
}
.section-outline-heading {
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #8a94a6;
  padding: 0 10px 8px;
}

/* ---- Desktop: reserved column + fixed rail (>=1100px) ---- */
@media (min-width: 1100px) {
  body.has-section-outline {
    padding-left: calc(var(--outline-rail-w) + var(--outline-rail-gap) * 2);
  }
  .section-outline-toggle,
  .section-outline-backdrop { display: none; }
  .section-outline-panel {
    position: fixed;
    top: 0;
    left: 0;
    width: var(--outline-rail-w);
    max-height: 100vh;
    overflow-y: auto;
    padding: 24px var(--outline-rail-gap);
    box-sizing: border-box;
  }
}

/* ---- Narrow/phone: hamburger + drawer (<1100px) ---- */
@media (max-width: 1099px) {
  body.has-section-outline { padding-left: 0; }
  .section-outline-toggle {
    position: fixed;
    top: 12px;
    left: 12px;
    z-index: 1001;
    width: 40px;
    height: 40px;
    display: grid;
    place-items: center;
    background: #ffffff;
    border: 1px solid #d4d9e0;
    border-radius: 8px;
    cursor: pointer;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.12);
  }
  .section-outline-bars,
  .section-outline-bars::before,
  .section-outline-bars::after {
    content: "";
    display: block;
    width: 18px;
    height: 2px;
    background: #1f2933;
    position: relative;
  }
  .section-outline-bars::before { position: absolute; top: -6px; }
  .section-outline-bars::after { position: absolute; top: 6px; }

  .section-outline-backdrop {
    position: fixed;
    inset: 0;
    z-index: 1000;
    background: rgba(0, 0, 0, 0.4);
  }
  .section-outline-panel {
    position: fixed;
    top: 0;
    left: 0;
    z-index: 1002;
    width: 78%;
    max-width: 320px;
    height: 100vh;
    overflow-y: auto;
    padding: 56px 16px 24px;
    box-sizing: border-box;
    background: #ffffff;
    box-shadow: 2px 0 12px rgba(0, 0, 0, 0.18);
    transform: translateX(-100%);
    transition: transform 0.22s ease;
  }
  .section-outline.open .section-outline-panel { transform: translateX(0); }
}

@media (prefers-reduced-motion: reduce) {
  .section-outline-panel { transition: none; }
  html { scroll-behavior: auto; }
}
```

- [ ] **Step 2: Verify the reserved column shifts content at desktop width**

On `pages/pca.html` at 1500x900:

```js
() => {
  const pad = getComputedStyle(document.body).paddingLeft;
  const railVisible = !!document.querySelector('.section-outline-panel').offsetWidth;
  const toggleHidden = getComputedStyle(document.querySelector('.section-outline-toggle')).display === 'none';
  return { pad, railVisible, toggleHidden };
}
```

Expected: `pad` is roughly `192px`, `railVisible` is `true`, `toggleHidden` is `true`.

- [ ] **Step 3: Verify the hamburger replaces the rail below 1100px**

Resize to 900x800, then:

```js
() => {
  const pad = getComputedStyle(document.body).paddingLeft;
  const toggleShown = getComputedStyle(document.querySelector('.section-outline-toggle')).display !== 'none';
  return { pad, toggleShown };
}
```

Expected: `pad` is `0px`, `toggleShown` is `true`.

- [ ] **Step 4: Screenshot for a visual sanity check**

`browser_take_screenshot` at 1500x900 and at 390x844 (drawer open). Confirm the rail does not overlap the page content and the drawer sits above the page.

- [ ] **Step 5: Commit**

```bash
git add styles/section-outline.css
git commit -m "feat: section-outline stylesheet (reserved rail, drawer, breakpoint)"
```

---

## Task 6: Add includes to the remaining content pages

**Files:**
- Modify: `pages/distributions.html`, `pages/generative_classification.html`, `pages/regularization.html`, `pages/estimation.html`, `pages/bayesian.html`, `pages/fourier.html` (pca.html done in Task 2; manifold.html in Task 7)

- [ ] **Step 1: Add the two include lines to each page**

For each file listed above, in `<head>`:
- Add after the last `<link rel="stylesheet" ...>`:

```html
  <link rel="stylesheet" href="../styles/section-outline.css">
```

- Add after the last `<script defer src="../js/...">` include:

```html
  <script type="module" src="../js/section-outline.js"></script>
```

Match each file's existing indentation (two spaces for most; `distributions.html` uses two spaces; `regularization.html` and `generative_classification.html` use the same head pattern, so verify by reading the head block first).

- [ ] **Step 2: Verify each page builds an outline (Playwright MCP)**

With the server running, for each of the six URLs, navigate at 1500x900 and run:

```js
() => Array.from(document.querySelectorAll('.section-outline-list a')).map(a => a.textContent)
```

Expected non-empty lists, e.g.:
- `distributions.html`: `["Controls","Probability Density Function (PDF)","Cumulative Distribution Function (CDF)","Active Distributions"]`
- `regularization.html`: `["Overview","Visualization","Bias-Variance tradeoff","Observations","Learning Rate/Regularization Boundaries"]`
- `estimation.html`: starts with `["Overview","Bias and Variance",...]` (11 entries).
- `bayesian.html`, `fourier.html`, `generative_classification.html`: non-empty, matching their `<h2>` panel headings.

- [ ] **Step 3: Commit**

```bash
git add pages/distributions.html pages/generative_classification.html pages/regularization.html pages/estimation.html pages/bayesian.html pages/fourier.html
git commit -m "feat: include section-outline on remaining content pages"
```

---

## Task 7: Make manifold panels collapsible and add the outline

**Files:**
- Modify: `pages/manifold.html`

- [ ] **Step 1: Add `collapsible` to manifold's panels**

In `pages/manifold.html`, change each top-level panel opening tag:
- `<section class="panel mf-isomap">` (line ~37) becomes `<section class="panel mf-isomap collapsible">`
- `<section class="panel">` for Dataset (line ~91) becomes `<section class="panel collapsible">`
- `<section class="panel sp-frame" id="mfAlgoStepPanel">` (line ~119) becomes `<section class="panel sp-frame collapsible" id="mfAlgoStepPanel">`
- `<section class="panel">` for Visualization (line ~145) becomes `<section class="panel collapsible" data-open-mobile>`
- `<section class="panel">` for Step notes (line ~158) becomes `<section class="panel collapsible">`
- `<section class="panel">` for Full pseudocode (line ~165) becomes `<section class="panel collapsible">`

(Read the file first to confirm exact current attributes before editing.)

- [ ] **Step 2: Add the collapsible, outline CSS, and outline script includes**

In `<head>` of `pages/manifold.html`:
- Add after the last stylesheet `<link>` (after `responsive.css`, line ~13):

```html
  <link rel="stylesheet" href="../styles/section-outline.css">
```

- Add after the existing `<script defer src="../js/favicon.js"></script>` (line ~27):

```html
  <script defer src="../js/collapsible.js"></script>
  <script type="module" src="../js/section-outline.js"></script>
```

- [ ] **Step 3: Verify manifold outline and collapsible behavior (Playwright MCP)**

At 1500x900 on `pages/manifold.html`:

```js
() => Array.from(document.querySelectorAll('.section-outline-list a')).map(a => a.textContent)
```

Expected: `["Algorithm walkthrough","Dataset","Algorithms","Visualization","Step notes","Full pseudocode"]`.

At 390x844, confirm panels collapse and the Visualization panel is open by default:

```js
() => ({
  vizOpen: document.getElementById('visualization')?.classList.contains('open'),
  datasetOpen: document.getElementById('dataset')?.classList.contains('open')
})
```

Expected: `vizOpen` is `true`, `datasetOpen` is `false`. Then click the "Dataset" entry and confirm it opens (same pattern as Task 3 Step 3). Confirm the isomap player in the first panel still renders (no console errors via `browser_console_messages`).

- [ ] **Step 4: Commit**

```bash
git add pages/manifold.html
git commit -m "feat: make manifold panels collapsible and add section outline"
```

---

## Task 8: Cross-page verification pass

**Files:**
- No code changes unless a defect is found (then fix in the relevant module/page and re-commit).
- Verify with: Playwright MCP

- [ ] **Step 1: Three-width sweep over all eight pages**

For each page (`distributions`, `generative_classification`, `regularization`, `estimation`, `bayesian`, `pca`, `fourier`, `manifold`):
- At 1500x900: rail visible, body has left padding ~192px, hamburger hidden, clicking the last entry scrolls down and sets the hash.
- At 900x800: hamburger visible, no left padding; opening the drawer then clicking an entry scrolls and closes the drawer.
- At 390x844: hamburger visible; clicking an entry for a collapsed panel opens it then scrolls.

Use `browser_console_messages` after each load and assert there are no errors.

- [ ] **Step 2: Accessibility check**

On one page (bayesian.html) at 900x800:
- `browser_press_key` Tab until the toggle is focused; Enter opens the drawer; confirm focus moved into the list (`browser_evaluate` reading `document.activeElement.className`); Esc closes it and focus returns to the toggle (`document.activeElement.className` contains `section-outline-toggle`).

- [ ] **Step 3: Reduced-motion check**

`browser_evaluate` to emulate reduced motion is limited; instead verify the CSS rule exists and that `navigateTo` uses `behavior: 'auto'` when the media query matches by temporarily forcing it:

```js
() => {
  const mql = window.matchMedia('(prefers-reduced-motion: reduce)');
  return { hasRule: !!mql }; // smoke check; full check is manual in a reduced-motion OS setting
}
```

Document the result; flag for manual confirmation in a reduced-motion environment.

- [ ] **Step 4: Final commit (only if fixes were made)**

```bash
git add -A
git commit -m "fix: section-outline cross-page verification adjustments"
```

- [ ] **Step 5: Stop the server**

Stop the background `python3 -m http.server 8000` process.

---

## Notes for the implementer

- All page scripts for the outline use `type="module"`, which defers by default and runs after the DOM is parsed. `collapsible.js` stays a plain `<script defer>`; the outline triggers it by clicking the panel head, so no import coupling exists.
- Do not restructure existing page layouts. The reserved column is achieved purely by `body` left-padding plus a fixed rail, so individual pages are untouched except for the include lines (and manifold's `collapsible` class additions).
- If a page has interactive charts that read width on load (pca, fourier, regularization, generative_classification, manifold), the reserved-column padding changes the available width. Confirm those charts still size correctly at 1500px after the padding is applied (they already redraw on resize; the outline does not suppress that).
