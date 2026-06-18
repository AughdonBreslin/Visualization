# Article Page Archetype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the article/content page archetype from `docs/superpowers/specs/2026-06-16-article-page-design.md` in the shared `.ui` system and apply it to the estimation pilot page.

**Architecture:** All styling lives in the existing opt-in `.ui` layer (`styles/tokens.css`, `styles/system.css`, `styles/article-ui.css`), with theming in `js/theme.js` and the on-page outline in `js/section-outline.js`. Changes are scoped to `.ui` so un-migrated pages stay untouched. The estimation page (`pages/estimation.html`) and the hidden settings page (`pages/settings.html`) are the concrete surfaces. Term-to-definition-page wiring is deferred (see spec Scope); this plan delivers only the definition-link treatment.

**Tech Stack:** Static HTML/CSS/JS. MathJax (tex-svg) for math. Verification via Playwright scripts under `/tmp/pwverify` (cached chromium), asserting `getComputedStyle` values against pages served from `http://localhost:8000`.

---

## Preconditions

- A static server is running: `http://localhost:8000` serving the repo root. If not, start one:
  `python3 -m http.server 8000` from the repo root (background it).
- Playwright is available at `/tmp/pwverify/node_modules/playwright` (CJS import form:
  `import pkg from '/tmp/pwverify/node_modules/playwright/index.js'; const { chromium } = pkg;`).
- Every committed file must contain no em-dash (U+2014) or en-dash (U+2013). Before each commit run:
  `grep -rlP "[\x{2014}\x{2013}]" <files>` and expect no output.

## File Structure

- `styles/tokens.css` (modify): add the `--link-ul` token (link-underline width, default 0).
- `styles/system.css` (modify): rework `.ui a` (underline off by default, hover reveal), add `.ui a.def` and `.ui a.ref`, make `.ui .eyebrow` mono, widen `.ui p` to 82ch.
- `styles/article-ui.css` (modify): section-header mono number style, the embedded-demo layout (figure caption, plot, controls below, results readout), and the responsive rail rules (hide on narrow desktop, mobile hamburger bottom-left).
- `js/theme.js` (modify): persist and apply the link-underline preference; expose on `window.UITheme`.
- `js/section-outline.js` (modify): inject a mono section number into each section heading on `.ui` pages, matching the rail number.
- `pages/estimation.html` (modify): add the mono category eyebrow, de-bold formula labels, restructure the bootstrap demo to the new layout.
- `pages/settings.html` (modify): add the "Link underlines" setting (default off) and wire it.

---

### Task 1: Link-underline token and the .def / .ref link treatments

**Files:**
- Modify: `styles/tokens.css` (`:root` block)
- Modify: `styles/system.css:89-95` (the `/* links */` block)
- Test: `/tmp/pwverify/ap_links.mjs`

- [ ] **Step 1: Write the failing test**

Create `/tmp/pwverify/ap_links.mjs`:

```js
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1300,height:900} })).newPage();
await p.goto('http://localhost:8000/pages/estimation.html', { waitUntil:'domcontentloaded' });
let fail = 0;
const tok = await p.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--link-ul').trim());
if (tok !== '0px') { console.log(`FAIL --link-ul default: "${tok}"`); fail++; }
// Build sample links and read computed styles.
const r = await p.evaluate(() => {
  const mk = (cls) => { const a = document.createElement('a'); if (cls) a.className = cls; a.href='#'; a.textContent='x';
    document.querySelector('.article-body p').appendChild(a); return getComputedStyle(a); };
  const base = mk('');            // plain .ui a
  const def = mk('def');
  const ref = mk('ref');
  return {
    baseW: base.borderBottomWidth, baseColor: base.color,
    defW: def.borderBottomWidth, defStyle: def.borderBottomStyle, defColor: def.color,
    refW: ref.borderBottomWidth, refStyle: ref.borderBottomStyle, refColor: ref.color,
  };
});
if (r.baseW !== '0px') { console.log(`FAIL base underline should be off: ${r.baseW}`); fail++; }
if (r.defW !== '0px') { console.log(`FAIL def underline should be off by default: ${r.defW}`); fail++; }
if (r.defColor !== r.baseColor && r.defColor !== 'rgb(139, 141, 150)') { /* def blends into body */ }
if (r.refColor !== 'rgb(154, 166, 255)') { console.log(`FAIL ref color (accent-link): ${r.refColor}`); fail++; }
// When the setting turns underlines on, both reveal.
const on = await p.evaluate(() => {
  document.documentElement.style.setProperty('--link-ul', '1px');
  const def = document.querySelector('.article-body p a.def');
  const ref = document.querySelector('.article-body p a.ref');
  return { defW: getComputedStyle(def).borderBottomWidth, defStyle: getComputedStyle(def).borderBottomStyle,
           refW: getComputedStyle(ref).borderBottomWidth, refStyle: getComputedStyle(ref).borderBottomStyle };
});
if (on.defW !== '1px' || on.defStyle !== 'dotted') { console.log(`FAIL def underline-on: ${on.defW}/${on.defStyle}`); fail++; }
if (on.refW !== '1px' || on.refStyle !== 'solid') { console.log(`FAIL ref underline-on: ${on.refW}/${on.refStyle}`); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node /tmp/pwverify/ap_links.mjs`
Expected: FAIL (the `--link-ul` token does not exist yet, and `.ui a` still has a 1px underline).

- [ ] **Step 3: Add the token**

In `styles/tokens.css`, inside the `:root { ... }` block, add this line next to the other accent/link tokens:

```css
  --link-ul: 0px; /* link underline width; the settings toggle sets it to 1px */
```

- [ ] **Step 4: Rework the link rules**

In `styles/system.css`, replace the entire `/* links */` block (currently lines 89-95):

```css
/* links */
.ui a {
  color: var(--accent-link);
  text-decoration: none;
  border-bottom: 1px solid rgba(154, 166, 255, 0.40);
}
.ui a:hover { color: #c2caff; }
```

with:

```css
/* links: underline is off by default (var(--link-ul) == 0); the "Show link underlines"
   setting raises it to 1px. Hover always reveals the underline. Two roles:
   .ref = ordinary cross-reference (accent color, solid underline);
   .def = definition link (blends into prose at body color, dotted underline). */
.ui a {
  color: var(--accent-link);
  text-decoration: none;
  border-bottom: var(--link-ul) solid rgba(154, 166, 255, 0.45);
}
.ui a:hover { color: #c2caff; border-bottom: 1px solid rgba(154, 166, 255, 0.55); }

.ui a.ref { color: var(--accent-link); border-bottom-style: solid; }
.ui a.ref:hover { border-bottom: 1px solid rgba(154, 166, 255, 0.55); }

.ui a.def { color: inherit; border-bottom: var(--link-ul) dotted rgba(255, 255, 255, 0.30); }
.ui a.def:hover { color: var(--accent); border-bottom: 1px dotted var(--accent); }
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `node /tmp/pwverify/ap_links.mjs`
Expected: `ALL PASS` (exit 0).

- [ ] **Step 6: Commit**

```bash
grep -rlP "[\x{2014}\x{2013}]" styles/tokens.css styles/system.css || echo clean
git add styles/tokens.css styles/system.css
git commit -m "feat(ui): link-underline token + .def/.ref link treatments (underline off by default)"
```

---

### Task 2: Persist and apply the link-underline preference in theme.js

**Files:**
- Modify: `js/theme.js`
- Test: `/tmp/pwverify/ap_theme_ul.mjs`

- [ ] **Step 1: Write the failing test**

Create `/tmp/pwverify/ap_theme_ul.mjs`:

```js
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const ctx = await b.newContext({ viewport:{width:1300,height:900} });
// Seed the saved pref ON; the boot should apply it before paint.
await ctx.addInitScript(() => { try { localStorage.setItem('ui-link-underline', '1'); } catch(e){} });
const p = await ctx.newPage();
await p.goto('http://localhost:8000/pages/estimation.html', { waitUntil:'domcontentloaded' });
let fail = 0;
const tok = () => p.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--link-ul').trim());
if (await tok() !== '1px') { console.log(`FAIL boot did not apply saved underline pref: ${await tok()}`); fail++; }
// API present; turning it off clears the token.
await p.evaluate(() => window.UITheme.setLinkUnderline(false));
if (await tok() !== '0px') { console.log(`FAIL setLinkUnderline(false): ${await tok()}`); fail++; }
await p.evaluate(() => window.UITheme.setLinkUnderline(true));
if (await tok() !== '1px') { console.log(`FAIL setLinkUnderline(true): ${await tok()}`); fail++; }
const cur = await p.evaluate(() => window.UITheme.current().linkUnderline);
if (cur !== true) { console.log(`FAIL current().linkUnderline: ${cur}`); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node /tmp/pwverify/ap_theme_ul.mjs`
Expected: FAIL (`setLinkUnderline` is undefined; boot does not read the key).

- [ ] **Step 3: Add the preference to theme.js**

In `js/theme.js`, add the storage key near the other keys (after `var KEY_DENSITY = 'ui-density';`):

```js
  var KEY_LINKUL = 'ui-link-underline';
```

Add this function after `applyDensity` (after its closing brace, before `function get`):

```js
  function applyLinkUnderline(on) {
    var enabled = on === true || on === '1' || on === 'true';
    if (enabled) root.style.setProperty('--link-ul', '1px');
    else root.style.removeProperty('--link-ul');
  }
```

Add this setter after `function setDensity(...)`:

```js
  function setLinkUnderline(on) { applyLinkUnderline(on); set(KEY_LINKUL, on ? '1' : '0'); }
```

In `reset()`, add the underline reset so "Reset to defaults" also clears it. Replace:

```js
  function reset() { setAccent(DEFAULT_ACCENT); setDensity('balanced'); }
```

with:

```js
  function reset() { setAccent(DEFAULT_ACCENT); setDensity('balanced'); setLinkUnderline(false); }
```

In the boot section, after `applyDensity(get(KEY_DENSITY, 'balanced'));`, add:

```js
  applyLinkUnderline(get(KEY_LINKUL, '0'));
```

In the `window.UITheme = { ... }` object, add `applyLinkUnderline` and `setLinkUnderline` to the method list, and extend `current` to report the flag. Replace the whole assignment:

```js
  window.UITheme = {
    DEFAULT_ACCENT: DEFAULT_ACCENT,
    applyAccent: applyAccent, applyDensity: applyDensity,
    setAccent: setAccent, setDensity: setDensity, reset: reset,
    current: function () { return { accent: get(KEY_ACCENT, DEFAULT_ACCENT), density: get(KEY_DENSITY, 'balanced') }; }
  };
```

with:

```js
  window.UITheme = {
    DEFAULT_ACCENT: DEFAULT_ACCENT,
    applyAccent: applyAccent, applyDensity: applyDensity, applyLinkUnderline: applyLinkUnderline,
    setAccent: setAccent, setDensity: setDensity, setLinkUnderline: setLinkUnderline, reset: reset,
    current: function () {
      return {
        accent: get(KEY_ACCENT, DEFAULT_ACCENT),
        density: get(KEY_DENSITY, 'balanced'),
        linkUnderline: get(KEY_LINKUL, '0') === '1'
      };
    }
  };
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `node /tmp/pwverify/ap_theme_ul.mjs`
Expected: `ALL PASS` (exit 0).

- [ ] **Step 5: Commit**

```bash
grep -rlP "[\x{2014}\x{2013}]" js/theme.js || echo clean
git add js/theme.js
git commit -m "feat(theme): persist + apply link-underline preference"
```

---

### Task 3: "Link underlines" control on the settings page

**Files:**
- Modify: `pages/settings.html`
- Test: `/tmp/pwverify/ap_settings_ul.mjs`

- [ ] **Step 1: Write the failing test**

Create `/tmp/pwverify/ap_settings_ul.mjs`:

```js
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:900,height:900} })).newPage();
await p.goto('http://localhost:8000/pages/settings.html', { waitUntil:'domcontentloaded' });
let fail = 0;
const tabs = await p.$$('.underline-tab');
if (tabs.length !== 2) { console.log(`FAIL expected 2 underline tabs, got ${tabs.length}`); fail++; }
// Default state: the "Hidden" tab is active and the token is 0.
const tok = () => p.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--link-ul').trim());
if (await tok() !== '0px') { console.log(`FAIL default token: ${await tok()}`); fail++; }
// Click "Shown" -> token becomes 1px and persists.
await p.click('.underline-tab[data-underline="1"]');
if (await tok() !== '1px') { console.log(`FAIL after Shown: ${await tok()}`); fail++; }
const saved = await p.evaluate(() => localStorage.getItem('ui-link-underline'));
if (saved !== '1') { console.log(`FAIL persisted value: ${saved}`); fail++; }
// Reset returns to hidden.
await p.click('#reset');
if (await tok() !== '0px') { console.log(`FAIL after reset: ${await tok()}`); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node /tmp/pwverify/ap_settings_ul.mjs`
Expected: FAIL (no `.underline-tab` elements exist).

- [ ] **Step 3: Add the setting markup**

In `pages/settings.html`, after the Density `</section>` (the one ending at line 53) and before `<hr class="rule">`, insert:

```html
    <section class="setting">
      <div class="setting-label">Link underlines</div>
      <p class="setting-help">Underline links all the time, or only on hover. Hidden is the default.</p>
      <div class="tabs underline-tabs">
        <button class="tab underline-tab is-active" data-underline="0">Hidden</button>
        <button class="tab underline-tab" data-underline="1">Shown</button>
      </div>
    </section>
```

- [ ] **Step 4: Wire it in the page script**

In the `<script>` at the bottom of `pages/settings.html`, add references and a marker next to the existing ones. After:

```js
      var dtabs = Array.prototype.slice.call(document.querySelectorAll('.density-tab'));
```

add:

```js
      var utabs = Array.prototype.slice.call(document.querySelectorAll('.underline-tab'));
```

After the `markDensity` function definition, add:

```js
      function markUnderline(on) { utabs.forEach(function (t) { t.classList.toggle('is-active', (t.dataset.underline === '1') === on); }); }
```

After the `dtabs.forEach(...)` click-wiring block, add:

```js
      utabs.forEach(function (t) {
        t.addEventListener('click', function () { var on = t.dataset.underline === '1'; T.setLinkUnderline(on); markUnderline(on); });
      });
```

In the reset handler, add the underline marker. Replace:

```js
      document.getElementById('reset').addEventListener('click', function () {
        T.reset(); markSwatch(T.DEFAULT_ACCENT); markDensity('balanced');
      });
```

with:

```js
      document.getElementById('reset').addEventListener('click', function () {
        T.reset(); markSwatch(T.DEFAULT_ACCENT); markDensity('balanced'); markUnderline(false);
      });
```

At the bottom, reflect saved state. Replace:

```js
      var cur = T.current();
      markSwatch(cur.accent); markDensity(cur.density);
```

with:

```js
      var cur = T.current();
      markSwatch(cur.accent); markDensity(cur.density); markUnderline(cur.linkUnderline);
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `node /tmp/pwverify/ap_settings_ul.mjs`
Expected: `ALL PASS` (exit 0).

- [ ] **Step 6: Commit**

```bash
grep -rlP "[\x{2014}\x{2013}]" pages/settings.html || echo clean
git add pages/settings.html
git commit -m "feat(settings): Link underlines control (default Hidden)"
```

---

### Task 4: Mono category eyebrow, wider prose, estimation header eyebrow

**Files:**
- Modify: `styles/system.css:46-55` (`.ui p`) and `styles/system.css:68-75` (`.ui .eyebrow`)
- Modify: `pages/estimation.html:34-38` (header)
- Test: `/tmp/pwverify/ap_header.mjs`

- [ ] **Step 1: Write the failing test**

Create `/tmp/pwverify/ap_header.mjs`:

```js
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1300,height:900} })).newPage();
await p.goto('http://localhost:8000/pages/estimation.html', { waitUntil:'domcontentloaded' });
let fail = 0;
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
// Eyebrow present in the header, mono font, accent color, expected text.
const eb = await p.evaluate(() => { const e = document.querySelector('.page-head .eyebrow'); return e ? e.textContent.trim() : null; });
if (eb !== '// Statistics') { console.log(`FAIL eyebrow text: ${eb}`); fail++; }
const ff = await cs('.page-head .eyebrow', 'fontFamily');
if (!/mono/i.test(ff)) { console.log(`FAIL eyebrow not mono: ${ff}`); fail++; }
// No meta row.
const meta = await p.$('.page-head .meta');
if (meta) { console.log('FAIL meta row should not exist'); fail++; }
// Prose widened to 82ch. Measure the page's actual ch (advance of '0' at the paragraph font)
// with a probe, then assert the paragraph max-width exceeds 70ch (old value was 62ch).
const probe = await p.evaluate(() => {
  const para = document.querySelector('.article-body p');
  const e = document.createElement('span');
  const cs = getComputedStyle(para);
  e.style.cssText = `position:absolute;visibility:hidden;white-space:pre;font-family:${cs.fontFamily};font-size:${cs.fontSize};font-weight:${cs.fontWeight}`;
  e.textContent = '0'.repeat(100);
  document.body.appendChild(e);
  const ch = e.getBoundingClientRect().width / 100;
  e.remove();
  return { ch, mw: parseFloat(cs.maxWidth) };
});
if (!(probe.mw > probe.ch * 70)) { console.log(`FAIL prose max-width too narrow: ${probe.mw}px (= ${(probe.mw/probe.ch).toFixed(0)}ch)`); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node /tmp/pwverify/ap_header.mjs`
Expected: FAIL (no eyebrow in the header; prose still 62ch).

- [ ] **Step 3: Widen prose**

In `styles/system.css`, in the `.ui p` rule, change `max-width: 62ch;` to:

```css
  max-width: 82ch;
```

- [ ] **Step 4: Make the eyebrow mono**

In `styles/system.css`, in the `.ui .eyebrow` rule, change the `font:` shorthand from `var(--font-sans)` to `var(--font-mono)`:

```css
  font: 600 11px/1 var(--font-mono);
```

- [ ] **Step 5: Add the eyebrow to the estimation header**

In `pages/estimation.html`, replace the header (lines 34-38):

```html
    <header class="page-head">
      <a class="back-home" href="../index.html">Home</a>
      <h1>Estimation</h1>
      <p class="lede">Point estimation, function estimation, bias/variance, standard error, MSE, consistency, and likelihood objectives.</p>
    </header>
```

with:

```html
    <header class="page-head">
      <a class="back-home" href="../index.html">Home</a>
      <div class="eyebrow">// Statistics</div>
      <h1>Estimation</h1>
      <p class="lede">Point estimation, function estimation, bias/variance, standard error, MSE, consistency, and likelihood objectives.</p>
    </header>
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `node /tmp/pwverify/ap_header.mjs`
Expected: `ALL PASS` (exit 0).

- [ ] **Step 7: Commit**

```bash
grep -rlP "[\x{2014}\x{2013}]" styles/system.css pages/estimation.html || echo clean
git add styles/system.css pages/estimation.html
git commit -m "feat(ui): mono category eyebrow + wider prose; add eyebrow to estimation header"
```

---

### Task 5: De-bold the display-math labels on the estimation page

**Files:**
- Modify: `pages/estimation.html` (every `\textbf{...}` inside a `.formula`)
- Test: `/tmp/pwverify/ap_labels.mjs`

- [ ] **Step 1: Write the failing test**

Create `/tmp/pwverify/ap_labels.mjs`:

```js
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1300,height:900} })).newPage();
// Read the RAW HTML from the navigation response (before MathJax rewrites the $$...$$ into SVG).
const res = await p.goto('http://localhost:8000/pages/estimation.html', { waitUntil:'commit' });
const html = await res.text();
let fail = 0;
if (/\\textbf\{/.test(html)) { console.log('FAIL \\textbf{ still present in source'); fail++; }
// The labels must still be there as plain text.
if (!/\\text\{Point:\}/.test(html)) { console.log('FAIL Point label missing as \\text{}'); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node /tmp/pwverify/ap_labels.mjs`
Expected: FAIL (`\textbf{` is present).

- [ ] **Step 3: Replace every `\textbf{` with `\text{` inside the estimation formulas**

In `pages/estimation.html`, change each `\textbf{` to `\text{`. The occurrences are the labels:
`\textbf{Point:}`, `\textbf{Function:}` (line 53-54); `\textbf{Point estimation:}`, `\textbf{Function estimation:}` (line 60-61); `\textbf{Bias:}`, `\textbf{Variance:}` (line 75-76); `\textbf{Standard error:}` (line 119); `\textbf{MSE:}`, `\textbf{Decomposition:}` (line 224-225); `\textbf{MSE regression:}` (line 316); `\textbf{Conditional log-likelihood:}` (line 319). Replace the command name only; keep the brace content. For example line 53:

```html
          <div class="formula">$$\text{Point:}\quad X_1,\dots,X_n\sim P_\theta\;\Rightarrow\;\hat\theta\approx\theta$$</div>
```

Apply the same change to all listed lines. (There is no `\textbf` anywhere else in the file, so a global replace of `\textbf{` with `\text{` in this file is correct.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `node /tmp/pwverify/ap_labels.mjs`
Expected: `ALL PASS` (exit 0).

- [ ] **Step 5: Commit**

```bash
grep -rlP "[\x{2014}\x{2013}]" pages/estimation.html || echo clean
git add pages/estimation.html
git commit -m "tweak(estimation): de-bold display-math labels (\\text not \\textbf)"
```

---

### Task 6: Mono section numbers in section headers, tied to the rail

**Files:**
- Modify: `js/section-outline.js` (`buildNav`)
- Modify: `styles/article-ui.css` (new heading-number rule)
- Test: `/tmp/pwverify/ap_secnum.mjs`

- [ ] **Step 1: Write the failing test**

Create `/tmp/pwverify/ap_secnum.mjs`:

```js
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1300,height:900} })).newPage();
await p.goto('http://localhost:8000/pages/estimation.html', { waitUntil:'networkidle' });
let fail = 0;
// First section heading carries a mono "01" that matches the first rail number.
const r = await p.evaluate(() => {
  const h = document.querySelector('.article-body section h2 .sec-n');
  const railN = document.querySelector('.section-outline-list a .rail-n');
  return { sec: h ? h.textContent.trim() : null, secFont: h ? getComputedStyle(h).fontFamily : null,
           rail: railN ? railN.textContent.trim() : null };
});
if (r.sec !== '01') { console.log(`FAIL section number: ${r.sec}`); fail++; }
if (r.rail !== '01') { console.log(`FAIL rail number: ${r.rail}`); fail++; }
if (r.sec !== r.rail) { console.log('FAIL section and rail numbers differ'); fail++; }
if (!/mono/i.test(r.secFont || '')) { console.log(`FAIL section number not mono: ${r.secFont}`); fail++; }
// Rail label must NOT have absorbed the section number (no double numbering).
const label = await p.evaluate(() => document.querySelector('.section-outline-list a').textContent.replace(/\s+/g,' ').trim());
if (/^01\s*01/.test(label)) { console.log(`FAIL doubled number in rail label: ${label}`); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node /tmp/pwverify/ap_secnum.mjs`
Expected: FAIL (no `.sec-n` in headings).

- [ ] **Step 3: Inject the heading number in buildNav**

In `js/section-outline.js`, inside `buildNav`, in the `entries.forEach((entry, i) => { ... })` loop, after the existing rail-number block that appends to the `a` element and before `a.dataset.target = entry.id;`, add a block that also numbers the page heading (only when `numbered`). Insert:

```js
    if (numbered) {
      const heading = entry.panel.querySelector(':scope > h2, :scope > h3');
      if (heading && !heading.querySelector('.sec-n')) {
        const sn = document.createElement('span');
        sn.className = 'sec-n';
        sn.textContent = String(i + 1).padStart(2, '0');
        heading.insertBefore(sn, heading.firstChild);
      }
    }
```

(The rail label was captured in `collectPanels` before this injection, so the rail text is unaffected.)

- [ ] **Step 4: Style the heading number**

In `styles/article-ui.css`, after the `.ui .h3-section` rule (around line 76), add:

```css
/* Mono section number prefixing each section heading; matches the rail numbering. */
.ui .article-body h2 .sec-n,
.ui .article-body h3 .sec-n {
  font: 500 13px/1 var(--font-mono);
  color: var(--text-muted);
  margin-right: 14px;
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `node /tmp/pwverify/ap_secnum.mjs`
Expected: `ALL PASS` (exit 0).

- [ ] **Step 6: Commit**

```bash
grep -rlP "[\x{2014}\x{2013}]" js/section-outline.js styles/article-ui.css || echo clean
git add js/section-outline.js styles/article-ui.css
git commit -m "feat(outline): mono section numbers in headings, synced to rail"
```

---

### Task 7: Restructure the bootstrap demo to the figure + controls-below + results pattern

**Files:**
- Modify: `pages/estimation.html:163-214` (the `.viz-row` block inside `#bootstrapCI`)
- Modify: `styles/article-ui.css` (replace the `.viz-row`/`.viz-panel`/`.viz-controls` rules)
- Test: `/tmp/pwverify/ap_demo.mjs`

- [ ] **Step 1: Write the failing test**

Create `/tmp/pwverify/ap_demo.mjs`:

```js
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1300,height:900} })).newPage();
await p.goto('http://localhost:8000/pages/estimation.html', { waitUntil:'networkidle' });
let fail = 0;
// New structure exists and preserves the JS hooks.
const has = (s) => p.evaluate((s) => !!document.querySelector(s), s);
for (const sel of ['.demo .figcap', '.demo #bootstrapPlot', '.demo .demo-controls', '.demo .demo-results #bootstrapStats', '.demo .demo-inputs .tabs']) {
  if (!(await has(sel))) { console.log(`FAIL missing ${sel}`); fail++; }
}
// figcap contains no em/en dash.
const cap = await p.evaluate(() => document.querySelector('.demo .figcap').textContent);
if (/[\u2014\u2013]/.test(cap)) { console.log('FAIL figcap contains a dash'); fail++; }
// Controls sit BELOW the plot (plot bottom <= controls top), i.e., stacked not side-by-side.
const geo = await p.evaluate(() => {
  const plot = document.querySelector('#bootstrapPlot').getBoundingClientRect();
  const ctrl = document.querySelector('.demo-controls').getBoundingClientRect();
  return { plotBottom: plot.bottom, ctrlTop: ctrl.top };
});
if (!(geo.ctrlTop >= geo.plotBottom - 1)) { console.log(`FAIL controls not below plot: plotBottom=${geo.plotBottom} ctrlTop=${geo.ctrlTop}`); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node /tmp/pwverify/ap_demo.mjs`
Expected: FAIL (no `.demo`/`.figcap`/`.demo-controls` structure).

- [ ] **Step 3: Restructure the demo markup**

In `pages/estimation.html`, replace the entire `<div class="viz-row"> ... </div>` block (lines 163-214) with:

```html
        <div class="demo">
          <div class="figcap">Fig. 1 &middot; bootstrap distribution of the statistic</div>
          <div id="bootstrapPlot" class="viz"></div>
          <div class="help-text">Note the standard error and confidence interval tend to tighten as $n$ grows.</div>

          <div class="demo-controls">
            <div class="demo-inputs">
              <div class="tabs" role="tablist" aria-label="Bootstrap controls">
                <button class="tab active" type="button" data-tab="sample">Sample</button>
                <button class="tab" type="button" data-tab="population">Population</button>
              </div>

              <div class="tab-panel" data-panel="sample" role="tabpanel">
                <div class="control-grid">
                  <label for="bootStat">Statistic</label>
                  <select id="bootStat">
                    <option value="mean" selected>Mean</option>
                    <option value="median">Median</option>
                  </select>

                  <label for="bootN">Sample size (n)</label>
                  <input id="bootN" type="number" min="5" max="500" step="1" value="40" />

                  <label for="bootB">Bootstrap reps (B)</label>
                  <input id="bootB" type="number" min="100" max="20000" step="100" value="2000" />

                  <label for="bootLevel">Confidence level</label>
                  <select id="bootLevel">
                    <option value="0.90">90%</option>
                    <option value="0.95" selected>95%</option>
                    <option value="0.99">99%</option>
                  </select>
                </div>
              </div>

              <div class="tab-panel" data-panel="population" role="tabpanel" hidden>
                <div class="control-grid">
                  <label for="bootMu">Population &mu;</label>
                  <input id="bootMu" type="number" min="-10" max="10" step="0.1" value="0" />

                  <label for="bootSigma">Population &sigma;</label>
                  <input id="bootSigma" type="number" min="0.05" max="10" step="0.05" value="1" />

                  <label for="bootSeed">Seed</label>
                  <input id="bootSeed" type="number" step="1" value="7" />
                </div>
              </div>
            </div>

            <div class="demo-results">
              <div class="label">Results</div>
              <div id="bootstrapStats">-</div>
            </div>
          </div>
        </div>
```

(IDs `bootstrapPlot`, `bootstrapStats`, `bootStat`, `bootN`, `bootB`, `bootLevel`, `bootMu`, `bootSigma`, `bootSeed`, the `data-tab`/`data-panel` hooks, and the `tab active` markup are all preserved, so `estimation_bootstrap.js` and `tabs.js` keep working.)

- [ ] **Step 4: Replace the demo layout CSS**

In `styles/article-ui.css`, replace the visualization-row rules (the block currently at lines 70-76, from the `/* Visualization row */` comment through the `.ui .h3-section` rule) with:

```css
/* Embedded demo: a figure caption, the framed plot full width, controls below it
   (separated by a hairline), and a Results readout column beside the inputs. */
.ui .demo { margin-top: 18px; }
.ui .figcap {
  font: 500 10.5px/1.3 var(--font-mono);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #62646c;
  margin-bottom: 10px;
}
.ui .viz { border: 1px solid var(--figure-edge); border-radius: var(--radius-md); overflow: hidden; background: #060607; }
.ui .help-text { font: 400 12px/1.5 var(--font-sans); color: var(--text-muted); margin-top: 8px; max-width: 60ch; }
.ui .demo-controls {
  margin-top: 18px;
  padding-top: 18px;
  border-top: 1px solid var(--hairline);
  display: flex;
  gap: 40px;
  flex-wrap: wrap;
  align-items: flex-start;
}
.ui .demo-inputs { flex: 1; min-width: 260px; }
.ui .demo-results { min-width: 200px; }
.ui .demo-results .label { margin-bottom: 10px; }
.ui .h3-section { font: 600 10.5px/1 var(--font-sans); letter-spacing: 0.14em; text-transform: uppercase; color: #6c6e77; margin: 18px 0 10px; }
```

Then, in the responsive block at the bottom of `styles/article-ui.css`, replace:

```css
@media (max-width: 700px) {
  .ui .viz-row { flex-direction: column; }
  .ui .viz-controls { width: 100%; }
}
```

with:

```css
@media (max-width: 700px) {
  .ui .demo-controls { gap: 24px; }
  .ui .demo-inputs, .ui .demo-results { min-width: 0; width: 100%; }
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `node /tmp/pwverify/ap_demo.mjs`
Expected: `ALL PASS` (exit 0).

- [ ] **Step 6: Commit**

```bash
grep -rlP "[\x{2014}\x{2013}]" pages/estimation.html styles/article-ui.css || echo clean
git add pages/estimation.html styles/article-ui.css
git commit -m "feat(estimation): demo as figure + controls-below + results readout"
```

---

### Task 8: Responsive rail: hide on narrow desktop, mobile hamburger at bottom-left

**Files:**
- Modify: `styles/article-ui.css` (the responsive block, lines 123-131)
- Test: `/tmp/pwverify/ap_rail.mjs`

- [ ] **Step 1: Write the failing test**

Create `/tmp/pwverify/ap_rail.mjs`:

```js
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const ctx = await b.newContext();
const p = await ctx.newPage();
const url = 'http://localhost:8000/pages/estimation.html';
let fail = 0;
const visible = (s) => p.evaluate((s) => { const e = document.querySelector(s); if (!e) return false; const r = e.getBoundingClientRect(); const st = getComputedStyle(e); return st.display !== 'none' && r.width > 0 && r.height > 0; }, s);

// Wide desktop (1400): rail visible, no hamburger.
await p.setViewportSize({ width: 1400, height: 900 });
await p.goto(url, { waitUntil: 'networkidle' });
if (!(await visible('.section-outline-panel'))) { console.log('FAIL wide: rail should be visible'); fail++; }
if (await visible('.section-outline-toggle')) { console.log('FAIL wide: hamburger should be hidden'); fail++; }

// Narrow desktop / tablet (1000): rail hidden AND no hamburger.
await p.setViewportSize({ width: 1000, height: 900 });
await p.waitForTimeout(60);
if (await visible('.section-outline-panel')) { console.log('FAIL mid: rail should be hidden'); fail++; }
if (await visible('.section-outline-toggle')) { console.log('FAIL mid: hamburger should be hidden'); fail++; }

// Mobile (560): hamburger visible and anchored to the bottom-left.
await p.setViewportSize({ width: 560, height: 900 });
await p.waitForTimeout(60);
if (!(await visible('.section-outline-toggle'))) { console.log('FAIL mobile: hamburger should be visible'); fail++; }
const pos = await p.evaluate(() => { const e = document.querySelector('.section-outline-toggle'); const r = e.getBoundingClientRect(); return { left: r.left, bottomGap: window.innerHeight - r.bottom }; });
if (!(pos.left < 80 && pos.bottomGap < 80)) { console.log(`FAIL mobile: hamburger not bottom-left: ${JSON.stringify(pos)}`); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node /tmp/pwverify/ap_rail.mjs`
Expected: FAIL (at 1000px the global rail still shows from `>=1100`? no, but the hamburger shows from `<=1099`; at 1000px the global hamburger is visible top-left, so "mid: hamburger should be hidden" fails; mobile hamburger is top-left, so the bottom-left check fails).

- [ ] **Step 3: Replace the responsive block**

In `styles/article-ui.css`, replace the current responsive rules (lines 123-131):

```css
/* Below the rail breakpoint the hamburger is fixed top-left; clear the back-home past it. */
@media (max-width: 1099px) {
  .ui .back-home { margin-left: 44px; }
}

@media (max-width: 700px) {
  .ui .demo-controls { gap: 24px; }
  .ui .demo-inputs, .ui .demo-results { min-width: 0; width: 100%; }
}
```

with:

```css
/* Rail behavior on migrated (.ui) pages. The global section-outline.css shows the rail at
   >=1100 and a top-left hamburger at <=1099. Override so the rail appears only when there is
   real room for it (>=1240), is hidden entirely in the middle band with no hamburger, and the
   mobile hamburger sits at the bottom-left (which also frees the top-left for the back-home). */
@media (min-width: 721px) and (max-width: 1239px) {
  body.ui.has-section-outline { padding-left: 0; }
  .ui .section-outline-toggle,
  .ui .section-outline-backdrop,
  .ui .section-outline-panel { display: none !important; }
}
@media (max-width: 720px) {
  .ui .section-outline-toggle { top: auto; bottom: 16px; left: 16px; }
}

@media (max-width: 700px) {
  .ui .demo-controls { gap: 24px; }
  .ui .demo-inputs, .ui .demo-results { min-width: 0; width: 100%; }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `node /tmp/pwverify/ap_rail.mjs`
Expected: `ALL PASS` (exit 0).

- [ ] **Step 5: Confirm un-migrated pages are unaffected**

Run a quick isolation check that `pca.html` (not `.ui`) still behaves with its original rail/hamburger at 1000px (hamburger visible, since the override is `.ui`-scoped):

```bash
cat > /tmp/pwverify/ap_isolation.mjs <<'EOF'
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1000,height:900} })).newPage();
await p.goto('http://localhost:8000/pages/pca.html', { waitUntil:'networkidle' });
let fail = 0;
const tog = await p.evaluate(() => { const e = document.querySelector('.section-outline-toggle'); if (!e) return false; const r = e.getBoundingClientRect(); return getComputedStyle(e).display !== 'none' && r.width > 0; });
if (!tog) { console.log('FAIL pca hamburger should still show at 1000px'); fail++; }
const ui = await p.evaluate(() => document.body.classList.contains('ui'));
if (ui) { console.log('FAIL pca should not be a .ui page'); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
EOF
node /tmp/pwverify/ap_isolation.mjs
```

Expected: `ALL PASS` (exit 0). If it fails, the rail override leaked outside `.ui`; re-check that every selector in the new block is `.ui`-scoped (or `body.ui`).

- [ ] **Step 6: Commit**

```bash
grep -rlP "[\x{2014}\x{2013}]" styles/article-ui.css || echo clean
git add styles/article-ui.css
git commit -m "feat(ui): rail hidden on narrow desktop; mobile hamburger at bottom-left"
```

---

## Final verification

After all tasks, re-run the full suite against the running server and confirm every script passes:

```bash
for f in ap_links ap_theme_ul ap_settings_ul ap_header ap_labels ap_secnum ap_demo ap_rail ap_isolation; do
  printf '%s: ' "$f"; node /tmp/pwverify/$f.mjs | tail -1;
done
```

Expected: every line ends in `ALL PASS`.

Also load `http://localhost:8000/pages/estimation.html` at 1400px, 1000px, and 560px and eyeball: the rail with synced numbers, the mono eyebrow, de-bolded formula labels, wider prose, the demo (figure over controls-below over results), links quiet until hover, and the bottom-left mobile hamburger. Confirm `http://localhost:8000/pages/settings.html` shows the Link underlines control and that toggling it changes link rendering on the estimation page.

## Notes / deferred

- Converting estimation's terms (estimator, bias, variance, etc.) into `.def` links is deferred with the definition-page / glossary content system (spec Scope). This plan ships the `.def`/`.ref` treatment only; the estimation page keeps plain prose for now.
- Shortening long section titles happens per page as migration continues; not in this plan.