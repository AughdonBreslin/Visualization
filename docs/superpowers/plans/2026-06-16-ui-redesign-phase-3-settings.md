# UI Redesign Phase 3: Settings + Theming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users customize accent color (curated swatches + custom hex) and density (Compact / Balanced / Generous), persisted in localStorage and applied before first paint, via a hidden settings page.

**Architecture:** Density is driven by tokenizing the type scale: the type sizes move into `:root` tokens (default = Balanced), and `html[data-density="compact"|"generous"]` override them. Accent is overridden by setting `--accent` (and its derived `--accent-muted`/`--accent-link`/`--focus-ring`) on `document.documentElement`. A new `js/theme.js`, loaded synchronously in the head, reads localStorage and applies both before the body paints, and exposes `window.UITheme` for the settings UI. The settings page (`pages/settings.html`) is a `.ui` page, hidden (noindex, not linked in nav).

**Tech Stack:** Plain CSS + vanilla JS. Verification: `python3 -m http.server 8000` + Playwright (`/tmp/pwverify/node_modules/playwright`, cached chromium). Import: `import pkg from '...'; const { chromium } = pkg;`.

**Spec:** sections 3 (type scale), 12 (settings/theming). Mockup: `docs/design/mockups/settings.html` (approved), and the Compact/Balanced/Generous values from `docs/design/mockups/type-scale.html`.

**Depends on:** Phases 1, 2a, 2b, 2c (committed on this branch).

**Scope:** Type tokenization, density, accent theming, theme.js, settings page. NOT here: page migration (Phase 4) wires theme.js into all eight pages; this phase wires it into the settings page and the preview only.

---

## Harness

Server: `python3 -m http.server 8000 >/tmp/redesign_srv.log 2>&1 &`. Tests `/tmp/pwverify/<name>.mjs` start with the CJS import and:
```js
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
const root = (n) => p.evaluate((n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim(), n);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
```

---

## Task 1: Tokenize the type scale (appearance-neutral at default)

**Files:** Modify `styles/tokens.css`, `styles/system.css`. Test: `/tmp/pwverify/p3_typetokens.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p3_typetokens.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:900} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
const root = (n) => p.evaluate((n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim(), n);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
// new tokens exist with Balanced defaults
const eq = async (n, exp) => { const g = await root(n); if (g !== exp) { console.log(`FAIL ${n}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await eq('--h1-size', '33px'); await eq('--h2-size', '19px'); await eq('--body-size', '14.5px'); await eq('--lede-size', '15px');
// appearance unchanged at default density
await want('.ui h1', 'fontSize', '33px');
await want('.ui h2', 'fontSize', '19px');
await want('.ui h3', 'fontSize', '15px');
await want('.ui section p', 'fontSize', '14.5px');
await want('.ui .lede', 'fontSize', '15px');
await want('.ui h1', 'letterSpacing', '-0.825px'); // -0.025em of 33px
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Add type tokens to `styles/tokens.css`** (inside `:root`, after the existing tokens, before the closing brace):

```css
  /* type scale (default = Balanced; density overrides below switch these) */
  --h1-size: 33px;   --h1-lh: 1.1;   --h1-track: -0.025em;
  --h2-size: 19px;   --h2-lh: 1.25;  --h2-track: -0.014em;
  --h3-size: 15px;   --h3-lh: 1.35;
  --body-size: 14.5px; --body-lh: 1.7;
  --lede-size: 15px; --lede-lh: 1.6;
```

- [ ] **Step 4: Refactor the type rules in `styles/system.css`** to use the tokens (longhand, to avoid `font:` shorthand + var() pitfalls). Replace the existing `.ui h1`, `.ui h2`, `.ui h3`, `.ui p`, `.ui .lede` blocks with:

```css
.ui h1 {
  font-family: var(--font-sans);
  font-weight: 600;
  font-size: var(--h1-size);
  line-height: var(--h1-lh);
  letter-spacing: var(--h1-track);
  color: var(--text);
  text-wrap: balance;
  margin: 0 0 var(--sp-3);
}
.ui h2 {
  font-family: var(--font-sans);
  font-weight: 600;
  font-size: var(--h2-size);
  line-height: var(--h2-lh);
  letter-spacing: var(--h2-track);
  color: #ededf0;
  text-wrap: balance;
  margin: 0 0 var(--sp-2);
}
.ui h3 {
  font-family: var(--font-sans);
  font-weight: 600;
  font-size: var(--h3-size);
  line-height: var(--h3-lh);
  color: #cfd1d8;
  margin: 0 0 var(--sp-2);
}
.ui p {
  font-family: var(--font-sans);
  font-weight: 400;
  font-size: var(--body-size);
  line-height: var(--body-lh);
  color: var(--text-body);
  max-width: 62ch;
  text-wrap: pretty;
  margin: 0 0 var(--sp-4);
}
.ui .lede {
  font-family: var(--font-sans);
  font-weight: 400;
  font-size: var(--lede-size);
  line-height: var(--lede-lh);
  color: var(--text-body);
  max-width: 56ch;
}
```

- [ ] **Step 5: Run the test** -> `ALL PASS`. Re-run `node /tmp/pwverify/p1_type.mjs` -> still `ALL PASS` (appearance unchanged at default density).

- [ ] **Step 6: Commit**

```bash
git add styles/tokens.css styles/system.css
git commit -m "refactor(redesign): tokenize the type scale (Balanced defaults)"
```

---

## Task 2: Density override blocks

**Files:** Modify `styles/tokens.css`. Test: `/tmp/pwverify/p3_density.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p3_density.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:900} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
const setD = (d) => p.evaluate((d) => { if (d) document.documentElement.setAttribute('data-density', d); else document.documentElement.removeAttribute('data-density'); }, d);
const want = async (s, pr, exp, msg) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${msg} ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await setD('compact');
await want('.ui h1', 'fontSize', '27px', 'compact');
await want('.ui section p', 'fontSize', '13.5px', 'compact');
await setD('generous');
await want('.ui h1', 'fontSize', '41px', 'generous');
await want('.ui section p', 'fontSize', '15.5px', 'generous');
await setD(null);
await want('.ui h1', 'fontSize', '33px', 'default');
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Append the density blocks to `styles/tokens.css`** (after the closing `}` of `:root`):

```css
/* Density overrides (user setting). Default (no attribute) = Balanced. Values from
   docs/design/mockups/type-scale.html. */
html[data-density="compact"] {
  --h1-size: 27px;   --h1-lh: 1.12;  --h1-track: -0.02em;
  --h2-size: 16px;   --h2-lh: 1.25;  --h2-track: -0.01em;
  --h3-size: 13.5px; --h3-lh: 1.3;
  --body-size: 13.5px; --body-lh: 1.6;
  --lede-size: 14px; --lede-lh: 1.55;
}
html[data-density="generous"] {
  --h1-size: 41px;   --h1-lh: 1.06;  --h1-track: -0.03em;
  --h2-size: 23px;   --h2-lh: 1.25;  --h2-track: -0.018em;
  --h3-size: 17px;   --h3-lh: 1.4;
  --body-size: 15.5px; --body-lh: 1.75;
  --lede-size: 16px; --lede-lh: 1.6;
}
```

- [ ] **Step 4: Run the test** -> `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
git add styles/tokens.css
git commit -m "feat(redesign): density overrides (compact / generous)"
```

---

## Task 3: theme.js (accent + density + boot)

**Files:** Create `js/theme.js`, modify `pages/_redesign-preview.html` (load it). Test: `/tmp/pwverify/p3_theme.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p3_theme.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const ctx = await b.newContext({ viewport:{width:1200,height:900} });
// seed localStorage before the page loads, then assert the boot applied it before paint
await ctx.addInitScript(() => { try { localStorage.setItem('ui-accent', '#d9a441'); localStorage.setItem('ui-density', 'compact'); } catch(e){} });
const p = await ctx.newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const root = (n) => p.evaluate((n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim(), n);
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
// accent applied + derived
if (await root('--accent') !== '#d9a441') { console.log('FAIL accent not applied'); fail++; }
if (await root('--accent-muted') !== 'rgba(217,164,65,0.14)') { console.log(`FAIL accent-muted: ${await root('--accent-muted')}`); fail++; }
// density applied (compact h1)
if (await cs('.ui h1', 'fontSize') !== '27px') { console.log('FAIL density not applied'); fail++; }
// UITheme API present and reverts
await p.evaluate(() => window.UITheme.applyAccent('#6b7cff'));
const accentBack = await p.evaluate(() => document.documentElement.style.getPropertyValue('--accent'));
if (accentBack !== '') { console.log(`FAIL default accent should clear inline override: ${accentBack}`); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Create `js/theme.js`**

```js
/* js/theme.js - user theming (accent + density), persisted in localStorage.
 * Loaded synchronously in the <head> so saved prefs apply before first paint.
 * Exposes window.UITheme for the settings page. */
(function () {
  var DEFAULT_ACCENT = '#6b7cff';
  var KEY_ACCENT = 'ui-accent';
  var KEY_DENSITY = 'ui-density';
  var root = document.documentElement;

  function hexToRgb(hex) {
    hex = String(hex).replace('#', '');
    if (hex.length === 3) hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    var n = parseInt(hex, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function lighten(c, t) {
    return [Math.round(c[0] + (255 - c[0]) * t), Math.round(c[1] + (255 - c[1]) * t), Math.round(c[2] + (255 - c[2]) * t)];
  }
  function isHex(s) { return /^#?[0-9a-fA-F]{3}$|^#?[0-9a-fA-F]{6}$/.test(String(s || '').trim()); }

  function applyAccent(hex) {
    if (!hex || !isHex(hex) || hex.toLowerCase() === DEFAULT_ACCENT) {
      root.style.removeProperty('--accent');
      root.style.removeProperty('--accent-muted');
      root.style.removeProperty('--accent-link');
      root.style.removeProperty('--focus-ring');
      return;
    }
    if (hex[0] !== '#') hex = '#' + hex;
    var c = hexToRgb(hex);
    var lk = lighten(c, 0.28);
    root.style.setProperty('--accent', hex);
    root.style.setProperty('--accent-muted', 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',0.14)');
    root.style.setProperty('--accent-link', 'rgb(' + lk[0] + ',' + lk[1] + ',' + lk[2] + ')');
    root.style.setProperty('--focus-ring', '2px solid rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',0.6)');
  }
  function applyDensity(d) {
    if (!d || d === 'balanced') root.removeAttribute('data-density');
    else root.setAttribute('data-density', d);
  }
  function get(key, def) { try { return localStorage.getItem(key) || def; } catch (e) { return def; } }
  function set(key, val) { try { localStorage.setItem(key, val); } catch (e) {} }

  function setAccent(hex) { applyAccent(hex); set(KEY_ACCENT, hex || DEFAULT_ACCENT); }
  function setDensity(d) { applyDensity(d); set(KEY_DENSITY, d || 'balanced'); }
  function reset() { setAccent(DEFAULT_ACCENT); setDensity('balanced'); }

  // boot: apply saved prefs immediately (before paint)
  applyAccent(get(KEY_ACCENT, DEFAULT_ACCENT));
  applyDensity(get(KEY_DENSITY, 'balanced'));

  window.UITheme = {
    DEFAULT_ACCENT: DEFAULT_ACCENT,
    applyAccent: applyAccent, applyDensity: applyDensity,
    setAccent: setAccent, setDensity: setDensity, reset: reset,
    current: function () { return { accent: get(KEY_ACCENT, DEFAULT_ACCENT), density: get(KEY_DENSITY, 'balanced') }; }
  };
})();
```

- [ ] **Step 4: Load it synchronously in the preview head.** In `pages/_redesign-preview.html`, immediately after the `components.css` link (and before the MathJax block):

```html
  <script src="../js/theme.js"></script>
```

(Synchronous, no `defer`, so it runs before the body paints.)

- [ ] **Step 5: Run the test** -> `ALL PASS`. Re-run `node /tmp/pwverify/p1_type.mjs` and `node /tmp/pwverify/p1_isolation.mjs` -> `ALL PASS`.

- [ ] **Step 6: Commit**

```bash
git add js/theme.js pages/_redesign-preview.html
git commit -m "feat(redesign): theme.js (accent + density boot, UITheme API)"
```

---

## Task 4: The settings page

**Files:** Create `pages/settings.html`. Test: `/tmp/pwverify/p3_settings.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p3_settings.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const ctx = await b.newContext({ viewport:{width:900,height:900} });
const p = await ctx.newPage();
await p.goto('http://localhost:8000/pages/settings.html', { waitUntil:'networkidle' });
let fail = 0;
// page renders under .ui
const isUi = await p.evaluate(() => document.body.classList.contains('ui'));
if (!isUi) { console.log('FAIL settings body not .ui'); fail++; }
// clicking the gold swatch applies + persists
await p.click('.swatch[data-accent="#d9a441"]');
const acc = await p.evaluate(() => document.documentElement.style.getPropertyValue('--accent'));
if (acc !== '#d9a441') { console.log(`FAIL swatch did not set accent: ${acc}`); fail++; }
const saved = await p.evaluate(() => localStorage.getItem('ui-accent'));
if (saved !== '#d9a441') { console.log(`FAIL accent not persisted: ${saved}`); fail++; }
// density tab applies
await p.click('.density-tab[data-density="compact"]');
const dens = await p.evaluate(() => document.documentElement.getAttribute('data-density'));
if (dens !== 'compact') { console.log(`FAIL density not applied: ${dens}`); fail++; }
// custom hex applies on input
await p.fill('#accent-hex', '#22cc88');
await p.dispatchEvent('#accent-hex', 'change');
const acc2 = await p.evaluate(() => document.documentElement.style.getPropertyValue('--accent'));
if (acc2 !== '#22cc88') { console.log(`FAIL custom hex not applied: ${acc2}`); fail++; }
// reset
await p.click('#reset');
const accReset = await p.evaluate(() => document.documentElement.style.getPropertyValue('--accent'));
if (accReset !== '') { console.log(`FAIL reset did not clear accent: ${accReset}`); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run -> FAIL (404).**

- [ ] **Step 3: Create `pages/settings.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Settings</title>
  <meta name="robots" content="noindex" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="../styles/tokens.css">
  <link rel="stylesheet" href="../styles/system.css">
  <link rel="stylesheet" href="../styles/components.css">
  <link rel="stylesheet" href="../styles/settings.css">
  <script src="../js/theme.js"></script>
</head>
<body class="ui">
  <div class="container settings">
    <header class="page-head">
      <div class="eyebrow">Preferences</div>
      <h1>Settings</h1>
      <p class="lede">Personalize the look. Changes apply instantly and are saved on this device.</p>
    </header>
    <hr class="rule">

    <section class="setting">
      <div class="setting-label">Accent color</div>
      <p class="setting-help">Used for links, active states, highlights, and focus rings.</p>
      <div class="accent-row">
        <button class="swatch" data-accent="#6b7cff" style="--c:#6b7cff" title="Periwinkle"></button>
        <button class="swatch" data-accent="#4aa3ff" style="--c:#4aa3ff" title="Azure"></button>
        <button class="swatch" data-accent="#5ec99a" style="--c:#5ec99a" title="Mint"></button>
        <button class="swatch" data-accent="#d9a441" style="--c:#d9a441" title="Gold"></button>
        <button class="swatch" data-accent="#8a9bd6" style="--c:#8a9bd6" title="Steel"></button>
        <span class="custom-accent"><span class="custom-label">Custom</span>
          <span class="hex-field"><span class="hex-well"></span><input id="accent-hex" value="#6b7cff" spellcheck="false"></span></span>
      </div>
    </section>

    <section class="setting">
      <div class="setting-label">Density</div>
      <p class="setting-help">How compact the type and spacing are.</p>
      <div class="tabs density-tabs">
        <button class="tab density-tab" data-density="compact">Compact</button>
        <button class="tab density-tab is-active" data-density="balanced">Balanced</button>
        <button class="tab density-tab" data-density="generous">Generous</button>
      </div>
      <div class="setting-preview">
        <div class="eyebrow">Visualization</div>
        <h2>Principal Component Analysis</h2>
        <p>The top components give the best low-rank <a href="#">reconstruction</a> of the data.</p>
        <button class="btn" type="button">Recompute</button>
      </div>
    </section>

    <hr class="rule">
    <div class="settings-foot">
      <span class="settings-note">Saved automatically on this device.</span>
      <button class="btn secondary" id="reset" type="button">Reset to defaults</button>
    </div>
  </div>

  <script>
    (function () {
      var T = window.UITheme;
      var hex = document.getElementById('accent-hex');
      var swatches = Array.prototype.slice.call(document.querySelectorAll('.swatch'));
      var dtabs = Array.prototype.slice.call(document.querySelectorAll('.density-tab'));
      var well = document.querySelector('.hex-well');

      function markSwatch(accent) {
        swatches.forEach(function (s) { s.classList.toggle('on', s.dataset.accent.toLowerCase() === String(accent).toLowerCase()); });
        if (well) well.style.background = accent;
        if (hex && document.activeElement !== hex) hex.value = accent;
      }
      function markDensity(d) { dtabs.forEach(function (t) { t.classList.toggle('is-active', t.dataset.density === d); }); }

      swatches.forEach(function (s) {
        s.addEventListener('click', function () { T.setAccent(s.dataset.accent); markSwatch(s.dataset.accent); });
      });
      hex.addEventListener('change', function () {
        var v = hex.value.trim(); if (v[0] !== '#') v = '#' + v;
        T.setAccent(v); markSwatch(v);
      });
      dtabs.forEach(function (t) {
        t.addEventListener('click', function () { T.setDensity(t.dataset.density); markDensity(t.dataset.density); });
      });
      document.getElementById('reset').addEventListener('click', function () {
        T.reset(); markSwatch(T.DEFAULT_ACCENT); markDensity('balanced');
      });

      // reflect current saved state in the UI
      var cur = T.current();
      markSwatch(cur.accent); markDensity(cur.density);
    })();
  </script>
</body>
</html>
```

- [ ] **Step 4: Create `styles/settings.css`** (settings-page-only layout, scoped under `.ui .settings`):

```css
/* styles/settings.css - settings page layout (under .ui .settings). */
.ui .settings { max-width: 680px; }
.ui .settings .setting { margin-bottom: 34px; }
.ui .settings .setting-label { font: 600 13px/1.3 var(--font-sans); color: #ededf0; margin: 0 0 4px; }
.ui .settings .setting-help { font: 400 13px/1.5 var(--font-sans); color: #74767f; margin: 0 0 16px; max-width: 56ch; }

.ui .settings .accent-row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.ui .settings .swatch { width: 30px; height: 30px; border-radius: 50%; border: 0; padding: 0; cursor: pointer; background: var(--c); }
.ui .settings .swatch.on { box-shadow: 0 0 0 2px var(--bg), 0 0 0 4px var(--c); }
.ui .settings .custom-accent { display: inline-flex; align-items: center; gap: 10px; margin-left: 8px; padding-left: 18px; border-left: 1px solid var(--hairline); }
.ui .settings .custom-label { font: 600 9.5px/1 var(--font-sans); letter-spacing: 0.14em; text-transform: uppercase; color: #6c6e77; }
.ui .settings .hex-field { display: inline-flex; align-items: center; gap: 8px; }
.ui .settings .hex-well { width: 22px; height: 22px; border-radius: var(--radius-sm); background: var(--accent); border: 1px solid rgba(255,255,255,0.16); }
.ui .settings .hex-field input { width: 84px; font: 500 13px/1 var(--font-mono); color: #dadbe0; background: transparent; border: 0; border-bottom: 1px solid var(--hairline-strong); padding: 6px 2px; }

.ui .settings .setting-preview { margin-top: 14px; border: 1px solid var(--hairline); border-radius: var(--radius-lg); padding: 20px 22px; background: #0a0a0b; }
.ui .settings .setting-preview h2 { margin-top: 6px; }
.ui .settings .settings-foot { display: flex; justify-content: space-between; align-items: center; }
.ui .settings .settings-note { font: 400 12px/1.5 var(--font-sans); color: var(--text-muted); }
```

- [ ] **Step 5: Run the test** -> `ALL PASS`.

- [ ] **Step 6: Commit**

```bash
git add pages/settings.html styles/settings.css
git commit -m "feat(redesign): hidden settings page (accent + density)"
```

---

## Task 5: Verification + visual check

- [ ] **Step 1: Run all Phase 3 tests**

```bash
for t in typetokens density theme settings; do echo "== $t =="; node /tmp/pwverify/p3_$t.mjs | tail -1; done
```
Expected: every line `ALL PASS`.

- [ ] **Step 2: Re-run prior suites (no regression)**

```bash
for t in tokens type inline polish isolation; do node /tmp/pwverify/p1_$t.mjs | tail -1; done
for t in callout code table tooltip math figure; do node /tmp/pwverify/p2_$t.mjs | tail -1; done
for t in buttons fields tabs inputs manage touch; do node /tmp/pwverify/p2b_$t.mjs | tail -1; done
for t in home pair player rail; do node /tmp/pwverify/p2c_$t.mjs | tail -1; done
```
Expected: all `ALL PASS`.

- [ ] **Step 3: JS suite** -> `node --test 'test/**/*.test.js'` -> `# pass 65`, `# fail 0`.

- [ ] **Step 4: Screenshots**

```bash
cat > /tmp/pwverify/p3_shot.mjs <<'EOF'
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:760,height:900} })).newPage();
await p.goto('http://localhost:8000/pages/settings.html', { waitUntil:'networkidle' });
await p.screenshot({ path: '/tmp/pwverify/p3_settings.png' });
// gold + compact applied
await p.click('.swatch[data-accent="#d9a441"]');
await p.click('.density-tab[data-density="compact"]');
await p.screenshot({ path: '/tmp/pwverify/p3_settings_gold.png' });
await b.close();
EOF
node /tmp/pwverify/p3_shot.mjs
```
Confirm the settings page matches `docs/design/mockups/settings.html`, and that picking gold + compact live-updates the eyebrow/title/link/button in the preview card.

- [ ] **Step 5: Em-dash sweep** -> `grep -lP "\x{2014}" js/theme.js styles/settings.css pages/settings.html styles/tokens.css styles/system.css && echo FIX || echo clean`.

---

## Notes for the implementer

- Strict: no em-dash characters anywhere.
- `theme.js` is loaded synchronously (no `defer`) in the head so saved prefs apply before paint.
- The default accent (`#6b7cff`) is applied by clearing the inline overrides (CSS defaults take over), so re-selecting periwinkle exactly restores the tuned default `--accent-link` (`#9aa6ff`) etc. Non-default accents derive `--accent-muted`/`--accent-link`/`--focus-ring` in JS.
- Do NOT add any other setting to this page without checking with the user first.
- Page migration (Phase 4) adds the `js/theme.js` head include to all eight pages so prefs apply site-wide. This phase only wires the preview and the settings page.
