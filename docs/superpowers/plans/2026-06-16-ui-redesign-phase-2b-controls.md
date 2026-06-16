# UI Redesign Phase 2b: Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the minimal/underline control set (buttons, select, inputs, tabs, segmented, slider, toggle, checkbox, upload), the three control-management patterns (grouped grid / primary+more / tabbed), and the mobile hit-area treatment to `styles/components.css` under `.ui`, verified on the preview page.

**Architecture:** Continues the opt-in `.ui` layer. Real native controls are styled with `appearance: none` plus pseudo-elements; `.ui .*` class specificity overrides `base.css` on migrated pages. Tab/segmented active state uses an `is-active` class (CSS only; JS wiring happens at page migration). The mobile hit-area uses `@media (pointer: coarse)` to enlarge tap targets without changing the desktop look.

**Tech Stack:** Plain CSS. Verification: `python3 -m http.server 8000` + Playwright (`/tmp/pwverify/node_modules/playwright`, cached chromium), Node scripts asserting `getComputedStyle`. Import form: `import pkg from '...'; const { chromium } = pkg;`. For pseudo-element rules that `getComputedStyle` cannot read (slider thumb, switch knob), assert the rule text is present by fetching the stylesheet (as Phase 1 did for `::selection`).

**Spec:** sections 5 (controls), 6 (control-management). Mockups: `docs/design/mockups/controls.html` (A), `manage-controls.html` (C tabbed default).

**Depends on:** Phase 1 + Phase 2a (committed on this branch).

**Scope:** Controls + control-management + mobile hit-area. NOT here: composites (Phase 2c), settings (Phase 3), page migration (Phase 4).

---

## Harness

Server (start once): `python3 -m http.server 8000 >/tmp/redesign_srv.log 2>&1 &` from repo root.
Each test `/tmp/pwverify/<name>.mjs` starts:
```js
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
```
Reused helpers in each test:
```js
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
const sheet = () => p.evaluate(async () => (await fetch('../styles/components.css')).text());
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
const wantCss = (txt, needle) => { if (!txt.includes(needle)) { console.log(`FAIL css missing ${needle}`); fail++; } };
```

Color reference: `#dadbe0`->`rgb(218, 219, 224)`, `#74767f`->`rgb(116, 118, 127)`, `#ececf0`->`rgb(236, 236, 240)`, `#a9abb3`->`rgb(169, 171, 179)`, accent `#6b7cff`->`rgb(107, 124, 255)`.

All preview additions go inside a new `<section id="controls">` (Task 1 creates it).

---

## Task 1: Buttons + upload + focus ring

**Files:** Modify `styles/components.css`, `pages/_redesign-preview.html`. Test: `/tmp/pwverify/p2b_buttons.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p2b_buttons.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:2600} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
const sheet = async () => p.evaluate(async () => (await fetch('../styles/components.css')).text());
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await want('.ui .btn', 'color', 'rgb(107, 124, 255)');
await want('.ui .btn', 'borderBottomColor', 'rgb(107, 124, 255)');
await want('.ui .btn', 'backgroundColor', 'rgba(0, 0, 0, 0)');
await want('.ui .btn.secondary', 'color', 'rgb(169, 171, 179)');
await want('.ui .btn-upload', 'color', 'rgb(107, 124, 255)');
const css = await sheet();
if (!/:focus-visible\s*\{\s*outline:\s*var\(--focus-ring\)/.test(css)) { console.log('FAIL focus-visible rule missing'); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Append to `styles/components.css`**

```css
/* ===== Controls (minimal / underline) ===== */

/* Buttons */
.ui .btn {
  font: 500 13px/1 var(--font-sans);
  color: var(--accent);
  background: none;
  border: 0;
  border-bottom: 1px solid var(--accent);
  padding: 6px 1px;
  cursor: pointer;
}
.ui .btn.secondary { color: #a9abb3; border-bottom-color: var(--hairline-strong); }
.ui .btn:hover { color: #aeb8ff; }
.ui .btn.secondary:hover { color: #cfd1d8; }

/* Upload: a <label> wrapping a visually-hidden file input (iOS-safe). */
.ui .btn-upload {
  display: inline-block;
  font: 500 13px/1 var(--font-sans);
  color: var(--accent);
  border-bottom: 1px solid var(--accent);
  padding: 6px 1px;
  cursor: pointer;
}
.ui .btn-upload input[type="file"] {
  position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0;
}

/* Focus ring on every interactive control. */
.ui .btn:focus-visible,
.ui .btn-upload:focus-within,
.ui .field select:focus-visible,
.ui .field input:focus-visible,
.ui .tab:focus-visible,
.ui .seg-option:focus-visible,
.ui input[type="range"]:focus-visible,
.ui .toggle:focus-visible,
.ui .check:focus-visible {
  outline: var(--focus-ring);
  outline-offset: 3px;
}
```

- [ ] **Step 4: Add the controls specimen section to the preview** (before `</main>`, after the `#components` section):

```html
      <hr class="rule">
      <section id="controls">
        <div style="display:flex;gap:24px;align-items:center;flex-wrap:wrap">
          <button class="btn" type="button">Recompute</button>
          <button class="btn secondary" type="button">Reset</button>
          <label class="btn-upload">Choose file<input type="file"></label>
        </div>
      </section>
```

- [ ] **Step 5: Run -> `ALL PASS`. Commit.**

```bash
git add styles/components.css pages/_redesign-preview.html
git commit -m "feat(redesign): buttons, upload, control focus ring"
```

---

## Task 2: Select + text/number input (field wrapper)

**Files:** Modify `styles/components.css`, `pages/_redesign-preview.html`. Test: `/tmp/pwverify/p2b_fields.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p2b_fields.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:2600} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await want('.ui .field select', 'appearance', 'none');
await want('.ui .field select', 'color', 'rgb(218, 219, 224)');
await want('.ui .field select', 'borderBottomColor', 'rgba(255, 255, 255, 0.22)');
await want('.ui .field select', 'backgroundColor', 'rgba(0, 0, 0, 0)');
await want('.ui .field input', 'borderBottomColor', 'rgba(255, 255, 255, 0.22)');
await want('.ui .field .field-label', 'textTransform', 'uppercase');
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Append to `styles/components.css`**

```css
/* Field wrapper: a label above a control. */
.ui .field { display: inline-flex; flex-direction: column; gap: 8px; }
.ui .field .field-label {
  font: 600 10.5px/1 var(--font-sans);
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #6c6e77;
}

/* Select + text/number input (underline). */
.ui .field select,
.ui .field input[type="text"],
.ui .field input[type="number"] {
  appearance: none;
  -webkit-appearance: none;
  font: 500 13px/1.2 var(--font-sans);
  color: #dadbe0;
  background: transparent;
  border: 0;
  border-bottom: 1px solid var(--hairline-strong);
  padding: 6px 2px;
  cursor: pointer;
}
.ui .field select {
  padding-right: 20px;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='9' height='6' viewBox='0 0 9 6'%3E%3Cpath d='M1 1l3.5 3.5L8 1' fill='none' stroke='%237a7c84' stroke-width='1.3'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 2px center;
}
.ui .field input { cursor: text; }
.ui .field select:hover, .ui .field input:hover { border-bottom-color: rgba(255, 255, 255, 0.4); }
```

- [ ] **Step 4: Append the specimen to `#controls`**

```html
        <div style="display:flex;gap:26px;align-items:flex-end;flex-wrap:wrap;margin-top:22px">
          <label class="field"><span class="field-label">Dataset</span>
            <select><option>Swiss roll</option><option>S-curve</option></select></label>
          <label class="field"><span class="field-label">Neighbors</span>
            <input type="number" value="10" style="width:60px"></label>
        </div>
```

- [ ] **Step 5: Run -> `ALL PASS`. Re-run p2b_buttons -> `ALL PASS`. Commit.**

```bash
git add styles/components.css pages/_redesign-preview.html
git commit -m "feat(redesign): select + input fields (underline, custom chevron)"
```

---

## Task 3: Tabs + segmented control

**Files:** Modify `styles/components.css`, `pages/_redesign-preview.html`. Test: `/tmp/pwverify/p2b_tabs.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p2b_tabs.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:2600} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await want('.ui .tab', 'color', 'rgb(116, 118, 127)');
await want('.ui .tab', 'borderBottomColor', 'rgba(0, 0, 0, 0)');
await want('.ui .tab.is-active', 'color', 'rgb(236, 236, 240)');
await want('.ui .tab.is-active', 'borderBottomColor', 'rgb(107, 124, 255)');
await want('.ui .seg-option.is-active', 'borderBottomColor', 'rgb(107, 124, 255)');
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Append to `styles/components.css`**

```css
/* Tabs (one shared pattern; active = .is-active). */
.ui .tabs { display: inline-flex; gap: 18px; }
.ui .tab {
  font: 500 13px/1 var(--font-sans);
  color: #74767f;
  background: none;
  border: 0;
  padding: 0 0 7px;
  border-bottom: 1.5px solid transparent;
  cursor: pointer;
}
.ui .tab:hover { color: #cfd1d8; }
.ui .tab.is-active { color: #ececf0; border-bottom-color: var(--accent); }

/* Segmented control (same visual language as tabs). */
.ui .segmented { display: inline-flex; gap: 6px; }
.ui .seg-option {
  font: 500 12.5px/1 var(--font-sans);
  color: #74767f;
  background: none;
  border: 0;
  padding: 6px 10px 7px;
  border-bottom: 1.5px solid transparent;
  cursor: pointer;
}
.ui .seg-option.is-active { color: #ececf0; border-bottom-color: var(--accent); }
```

- [ ] **Step 4: Append the specimen to `#controls`**

```html
        <div style="display:flex;gap:40px;align-items:center;flex-wrap:wrap;margin-top:24px">
          <div class="tabs"><button class="tab is-active">Fourier</button><button class="tab">Haar</button><button class="tab">DCT</button></div>
          <div class="segmented"><button class="seg-option is-active">PDF</button><button class="seg-option">CDF</button></div>
        </div>
```

- [ ] **Step 5: Run -> `ALL PASS`. Commit.**

```bash
git add styles/components.css pages/_redesign-preview.html
git commit -m "feat(redesign): tabs + segmented control"
```

---

## Task 4: Slider, toggle, checkbox

**Files:** Modify `styles/components.css`, `pages/_redesign-preview.html`. Test: `/tmp/pwverify/p2b_inputs.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p2b_inputs.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:2800} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
const sheet = async () => p.evaluate(async () => (await fetch('../styles/components.css')).text());
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await want('.ui input[type="range"]', 'appearance', 'none');
await want('.ui input[type="range"]', 'backgroundColor', 'rgba(255, 255, 255, 0.16)');
await want('.ui .toggle', 'appearance', 'none');
await want('.ui .toggle', 'width', '30px');
await want('.ui .toggle:checked', 'backgroundColor', 'rgba(107, 124, 255, 0.55)');
await want('.ui .check', 'borderTopColor', 'rgb(107, 124, 255)');
const css = await sheet();
for (const n of ['::-webkit-slider-thumb', '::-moz-range-thumb', '.toggle::after', '.check:checked::after']) {
  if (!css.includes(n)) { console.log(`FAIL css missing ${n}`); fail++; }
}
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Append to `styles/components.css`**

```css
/* Range slider. (Progress fill is added per-page via JS if needed.) */
.ui input[type="range"] {
  appearance: none;
  -webkit-appearance: none;
  width: 140px;
  height: 2px;
  background: rgba(255, 255, 255, 0.16);
  border-radius: 2px;
  cursor: pointer;
}
.ui input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 13px; height: 13px; border-radius: 50%;
  background: var(--accent); border: none; cursor: pointer;
}
.ui input[type="range"]::-moz-range-thumb {
  width: 13px; height: 13px; border-radius: 50%;
  background: var(--accent); border: none; cursor: pointer;
}

/* Toggle switch (a checkbox styled as a pill). */
.ui .toggle {
  appearance: none;
  -webkit-appearance: none;
  width: 30px; height: 17px;
  border-radius: var(--radius-pill);
  background: rgba(255, 255, 255, 0.12);
  position: relative;
  cursor: pointer;
}
.ui .toggle::after {
  content: '';
  position: absolute; top: 2px; left: 2px;
  width: 13px; height: 13px; border-radius: 50%;
  background: #e9e9ee;
  transition: left var(--dur-hover) var(--ease-out);
}
.ui .toggle:checked { background: rgba(107, 124, 255, 0.55); }
.ui .toggle:checked::after { left: 15px; }

/* Checkbox. */
.ui .check {
  appearance: none;
  -webkit-appearance: none;
  width: 15px; height: 15px;
  border: 1.5px solid var(--accent);
  border-radius: var(--radius-xs);
  position: relative;
  cursor: pointer;
}
.ui .check:checked::after {
  content: '';
  position: absolute; inset: 3px;
  background: var(--accent);
  border-radius: 1px;
}
.ui .control-row { display: inline-flex; align-items: center; gap: 9px; font: 500 13px/1 var(--font-sans); color: #b7b9c1; }
```

- [ ] **Step 4: Append the specimen to `#controls`**

```html
        <div style="display:flex;gap:34px;align-items:center;flex-wrap:wrap;margin-top:24px">
          <input type="range" min="0" max="100" value="38">
          <label class="control-row"><input class="toggle" type="checkbox" checked>Tiling</label>
          <label class="control-row"><input class="check" type="checkbox" checked>3D spectrum</label>
        </div>
```

- [ ] **Step 5: Run -> `ALL PASS`. Commit.**

```bash
git add styles/components.css pages/_redesign-preview.html
git commit -m "feat(redesign): slider, toggle switch, checkbox"
```

---

## Task 5: Control-management patterns

**Files:** Modify `styles/components.css`, `pages/_redesign-preview.html`. Test: `/tmp/pwverify/p2b_manage.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p2b_manage.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:3000} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await want('.ui .control-groups', 'display', 'grid');
await want('.ui .control-more', 'display', 'block');
await want('.ui .control-more-toggle', 'color', 'rgb(154, 166, 255)');
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Append to `styles/components.css`**

```css
/* ===== Control-management patterns (chosen per page; tabbed is the default for dense sets) ===== */

/* A. Grouped grid: labelled groups in an auto-fit grid. */
.ui .control-groups { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 24px 30px; }
.ui .control-group { display: flex; flex-direction: column; gap: 11px; }
.ui .control-group > .group-label {
  font: 600 9.5px/1 var(--font-sans); letter-spacing: 0.14em; text-transform: uppercase; color: #6c6e77;
  padding-bottom: 8px; border-bottom: 1px solid var(--surface-border);
}

/* B. Primary + "more" disclosure. */
.ui .control-primary { display: flex; gap: 30px; flex-wrap: wrap; align-items: flex-end; }
.ui .control-more { display: block; margin-top: 16px; }
.ui .control-more-toggle { font: 500 12px/1 var(--font-sans); color: var(--accent-link); background: none; border: 0; cursor: pointer; padding: 0; }
.ui .control-advanced { margin-top: 14px; padding-top: 14px; border-top: 1px dashed rgba(255, 255, 255, 0.1); display: flex; gap: 30px; flex-wrap: wrap; align-items: center; }
.ui .control-more[hidden] .control-advanced { display: none; }

/* C. Tabbed groups (default for dense control sets). */
.ui .control-tabs { display: inline-flex; gap: 18px; margin-bottom: 16px; }
```

- [ ] **Step 4: Append the specimen to `#controls`**

```html
        <div class="control-groups" style="margin-top:28px;max-width:680px">
          <div class="control-group"><span class="group-label">Image</span>
            <label class="field"><select><option>Quadrant</option></select></label></div>
          <div class="control-group"><span class="group-label">Filter</span>
            <input type="range" value="38"></div>
          <div class="control-group"><span class="group-label">View</span>
            <label class="control-row"><input class="toggle" type="checkbox" checked>Tiling</label></div>
        </div>
        <div class="control-more" style="margin-top:6px">
          <button class="control-more-toggle">Fewer options</button>
        </div>
```

- [ ] **Step 5: Run -> `ALL PASS`. Commit.**

```bash
git add styles/components.css pages/_redesign-preview.html
git commit -m "feat(redesign): control-management patterns (grouped, primary+more, tabbed)"
```

---

## Task 6: Mobile hit-area (pointer: coarse)

**Files:** Modify `styles/components.css`. Test: `/tmp/pwverify/p2b_touch.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p2b_touch.mjs
// pointer:coarse cannot be emulated via Playwright, so verify the rule by stylesheet text.
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext()).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const css = await p.evaluate(async () => (await fetch('../styles/components.css')).text());
let fail = 0;
const wantCss = (needle) => { if (!css.includes(needle)) { console.log(`FAIL css missing ${needle}`); fail++; } };
wantCss('@media (pointer: coarse)');
wantCss('min-width: 44px');
wantCss('min-height: 44px');
// the coarse block must position the controls and add a ::before overlay
const block = css.slice(css.indexOf('@media (pointer: coarse)'));
if (!/position:\s*relative/.test(block)) { console.log('FAIL coarse block missing position: relative'); fail++; }
if (!/::before/.test(block)) { console.log('FAIL coarse block missing ::before overlay'); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Append to `styles/components.css`**

```css
/* ===== Mobile / touch: enlarge tap targets to >= 44px without changing the look ===== */
@media (pointer: coarse) {
  .ui .btn,
  .ui .btn-upload,
  .ui .field select,
  .ui .field input,
  .ui .tab,
  .ui .seg-option,
  .ui .toggle,
  .ui .check {
    position: relative;
  }
  .ui .btn::before,
  .ui .btn-upload::before,
  .ui .field select::before,
  .ui .tab::before,
  .ui .seg-option::before,
  .ui .toggle::before,
  .ui .check::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    min-width: 44px;
    min-height: 44px;
    width: 100%;
    height: 100%;
  }
}
```

- [ ] **Step 4: Run -> `ALL PASS`. Commit.**

```bash
git add styles/components.css
git commit -m "feat(redesign): enlarge control tap targets on touch (>= 44px)"
```

Note: `.field select` already has a `background-image` chevron, and a `::before` overlay does not conflict with it. The overlay is transparent and only enlarges the tap region; `select`'s native picker still opens. (Inputs use the `:focus-visible` outline; a `::before` overlay on a text input would block text selection, so inputs are intentionally omitted from the `::before` set and instead get touch padding by the surrounding `.field` spacing.)

---

## Task 7: Verification + visual check

- [ ] **Step 1: Run all Phase 2b tests**

```bash
for t in buttons fields tabs inputs manage touch; do echo "== $t =="; node /tmp/pwverify/p2b_$t.mjs | tail -1; done
```
Expected: every line `ALL PASS`.

- [ ] **Step 2: Re-run Phase 1 + Phase 2a tests (no regression)**

```bash
for t in tokens type inline polish isolation; do node /tmp/pwverify/p1_$t.mjs | tail -1; done
for t in callout code table tooltip math figure; do node /tmp/pwverify/p2_$t.mjs | tail -1; done
```
Expected: all `ALL PASS`.

- [ ] **Step 3: JS suite**

Run: `node --test 'test/**/*.test.js'` -> `# pass 65`, `# fail 0`.

- [ ] **Step 4: Screenshot the controls specimen**

```bash
cat > /tmp/pwverify/p2b_shot.mjs <<'EOF'
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1100,height:1000} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const el = await p.$('#controls');
await el.screenshot({ path: '/tmp/pwverify/p2b_controls.png' });
await b.close();
EOF
node /tmp/pwverify/p2b_shot.mjs
```
Confirm against `docs/design/mockups/controls.html` (A) and `manage-controls.html` (C).

- [ ] **Step 5: Em-dash sweep**

```bash
grep -lP "\x{2014}" styles/components.css && echo "FIX em-dash" || echo "clean"
```

---

## Notes for the implementer

- Append to `styles/components.css` in order; do not modify `tokens.css`, `system.css`, or pages other than `pages/_redesign-preview.html`.
- Strict: no em-dash characters in any file.
- Native controls use `appearance: none`; pseudo-element rules (`::-webkit-slider-thumb`, `.toggle::after`, `.check:checked::after`) cannot be read by `getComputedStyle`, so their tests assert the rule text is present in the fetched stylesheet (as Phase 1 did for `::selection`).
- Tab/segmented active state is the `.is-active` class (one ARIA-native pattern site-wide). JS wiring of tabs/disclosure happens at page migration (Phase 4).
- Later: Phase 2c composites (home index, outline-rail re-skin, walkthrough player, paired-viz layout).
