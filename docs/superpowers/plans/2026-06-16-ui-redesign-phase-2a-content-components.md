# UI Redesign Phase 2a: Content Components Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable content components (callout, code block, data table, tooltip, display-math wrapper + worked numeric block, figure frame + caption) in a new `styles/components.css`, scoped under `.ui`, verified on the preview page. No existing page is touched.

**Architecture:** Continues Phase 1's opt-in `.ui` layer. A new `styles/components.css` (loaded after `system.css`) holds component classes. The hidden `pages/_redesign-preview.html` gains a specimen of each component for computed-style verification. Values are exact from spec section 11 and the cited mockups.

**Tech Stack:** Plain CSS. Verification: `python3 -m http.server 8000` + Playwright (lib at `/tmp/pwverify/node_modules/playwright`, cached chromium) running Node scripts asserting `getComputedStyle`. Use the CJS import form `import pkg from '...'; const { chromium } = pkg;`.

**Spec:** `docs/superpowers/specs/2026-06-15-ui-redesign-design.md` section 11. Mockups: `docs/design/mockups/{callouts,code-blocks,tables,tooltips,math-ruled-numbered,paired-viz-layout}.html`.

**Depends on:** Phase 1 (tokens.css, system.css, preview page) which is committed on this branch.

**Scope:** Content components only. NOT here: controls + control-management (Phase 2b), composites (player, outline-rail re-skin, home index, paired-viz layout: Phase 2c), page migration (Phase 4).

---

## Harness (used by every task)

Server (start once; restart the line if it stopped):
```bash
cd /mnt/c/Users/aughb/PersonalProjects/Visualization
python3 -m http.server 8000 >/tmp/redesign_srv.log 2>&1 &
```
Each test is `/tmp/pwverify/<name>.mjs`, run with `node /tmp/pwverify/<name>.mjs`, starting:
```js
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
```
Helper used below:
```js
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
```

Color reference: `#6b7cff`->`rgb(107, 124, 255)`, `#9fb0ff`->`rgb(159, 176, 255)`, `#f0c460`->`rgb(240, 196, 96)`, `#060607`->`rgb(6, 6, 7)`, `#141418`->`rgb(20, 20, 24)`, `#b6b8c0`->`rgb(182, 184, 192)`, `#e4e6ea`->`rgb(228, 230, 234)`.

---

## Task 1: Create components.css + callout, wire into preview

**Files:**
- Create: `styles/components.css`
- Modify: `pages/_redesign-preview.html` (link components.css; add a components specimen section)
- Test: `/tmp/pwverify/p2_callout.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p2_callout.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:1400} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await want('.ui .callout', 'borderLeftWidth', '2px');
await want('.ui .callout', 'borderLeftColor', 'rgb(107, 124, 255)');
await want('.ui .callout', 'backgroundColor', 'rgba(0, 0, 0, 0)');     // no fill
await want('.ui .callout.warning', 'borderLeftColor', 'rgb(224, 179, 65)');
await want('.ui .callout .callout-label', 'textTransform', 'uppercase');
await want('.ui .callout.warning .callout-label', 'color', 'rgb(240, 196, 96)');
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run to verify it fails**

Run: `node /tmp/pwverify/p2_callout.mjs`
Expected: FAIL (components.css + specimens absent).

- [ ] **Step 3: Create `styles/components.css`**

```css
/* styles/components.css - redesign shared components (scoped under .ui). Loaded after system.css. */

/* Callout / alert (minimal left rule, no fill). Variant color via --co. */
.ui .callout {
  border-left: 2px solid var(--co, var(--accent));
  padding: 2px 0 2px 16px;
  margin: 0 0 var(--sp-4);
}
.ui .callout .callout-label {
  font: 600 10.5px/1 var(--font-sans);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--co-label, #9fb0ff);
}
.ui .callout .callout-body {
  font: 400 13.5px/1.6 var(--font-sans);
  color: #9a9ca4;
  max-width: 60ch;
  margin-top: 6px;
}
.ui .callout.warning { --co: var(--warning); --co-label: var(--warning-text); }
.ui .callout.error   { --co: #e06a6a;        --co-label: #f0a0a0; }
```

- [ ] **Step 4: Wire components.css + a specimen into the preview page**

In `pages/_redesign-preview.html`, add after the `system.css` link:
```html
  <link rel="stylesheet" href="../styles/components.css">
```
And before `</main>`, add a components specimen wrapper (subsequent tasks append into it):
```html
      <hr class="rule">
      <section id="components">
        <div class="callout"><span class="callout-label">Note</span>
          <div class="callout-body">The DFT assumes the image tiles periodically.</div></div>
        <div class="callout warning"><span class="callout-label">Warning</span>
          <div class="callout-body">Non-periodic images produce checkerboard artifacts.</div></div>
      </section>
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `node /tmp/pwverify/p2_callout.mjs`
Expected: `ALL PASS`.

- [ ] **Step 6: Commit**

```bash
git add styles/components.css pages/_redesign-preview.html
git commit -m "feat(redesign): components.css + callout component"
```

---

## Task 2: Code / pseudocode block (dark inset)

**Files:**
- Modify: `styles/components.css`
- Modify: `pages/_redesign-preview.html`
- Test: `/tmp/pwverify/p2_code.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p2_code.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:1600} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await want('.ui .codeblock', 'backgroundColor', 'rgb(6, 6, 7)');
await want('.ui .codeblock', 'borderTopWidth', '1px');
await want('.ui .codeblock', 'borderTopColor', 'rgba(255, 255, 255, 0.08)');
const ff = await cs('.ui .codeblock', 'fontFamily');
if (!/JetBrains Mono/.test(ff)) { console.log(`FAIL code font: ${ff}`); fail++; }
await want('.ui .codeblock .kw', 'color', 'rgb(159, 176, 255)');
await want('.ui .codeblock .cm', 'color', 'rgb(93, 95, 104)');
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run to verify it fails**

Run: `node /tmp/pwverify/p2_code.mjs` -> FAIL.

- [ ] **Step 3: Append to `styles/components.css`**

```css
/* Code / pseudocode block (dark inset). */
.ui .codeblock {
  margin: 0;
  font: 500 12.5px/1.85 var(--font-mono);
  color: #9a9ca4;
  background: #060607;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: var(--radius-lg);
  padding: 16px 18px;
  overflow-x: auto;
}
.ui .codeblock .ln { display: inline-block; width: 20px; color: #4d4f57; user-select: none; }
.ui .codeblock .kw { color: #9fb0ff; }
.ui .codeblock .fn { color: #cfd1d8; }
.ui .codeblock .cm { color: #5d5f68; }
```

- [ ] **Step 4: Append the specimen to the preview `#components` section**

```html
        <pre class="codeblock"><code><span class="ln">1</span><span class="kw">function</span> <span class="fn">isomap</span>(X, k, d):
<span class="ln">2</span>    G = <span class="fn">knn_graph</span>(X, k)   <span class="cm"># neighbors</span>
<span class="ln">3</span>    <span class="kw">return</span> <span class="fn">embed</span>(G, d)</code></pre>
```

- [ ] **Step 5: Run the test** -> `ALL PASS`. Re-run `node /tmp/pwverify/p2_callout.mjs` -> still `ALL PASS`.

- [ ] **Step 6: Commit**

```bash
git add styles/components.css pages/_redesign-preview.html
git commit -m "feat(redesign): code/pseudocode block component"
```

---

## Task 3: Data table (row hairlines)

**Files:**
- Modify: `styles/components.css`
- Modify: `pages/_redesign-preview.html`
- Test: `/tmp/pwverify/p2_table.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p2_table.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:1800} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await want('.ui .data-table th', 'textTransform', 'uppercase');
await want('.ui .data-table th', 'borderBottomColor', 'rgba(255, 255, 255, 0.22)');
await want('.ui .data-table td', 'borderBottomColor', 'rgba(255, 255, 255, 0.08)');
await want('.ui .data-table td.num', 'fontVariantNumeric', 'tabular-nums');
await want('.ui .data-table td.num', 'textAlign', 'right');
const hf = await cs('.ui .data-table th', 'fontFamily');
if (!/JetBrains Mono/.test(hf)) { console.log(`FAIL th font: ${hf}`); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run to verify it fails** -> FAIL.

- [ ] **Step 3: Append to `styles/components.css`**

```css
/* Data table (row hairlines). */
.ui .data-table { border-collapse: collapse; width: 100%; max-width: 620px; }
.ui .data-table th {
  font: 600 10px/1 var(--font-mono);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #6c6e77;
  text-align: right;
  padding: 0 0 11px;
  border-bottom: 1px solid var(--hairline-strong);
}
.ui .data-table th:first-child { text-align: left; }
.ui .data-table td {
  font: 400 13px/1 var(--font-sans);
  color: #b6b8c0;
  text-align: right;
  padding: 11px 0;
  border-bottom: 1px solid var(--hairline);
}
.ui .data-table td:first-child { text-align: left; color: #dadbe0; font-weight: 500; }
.ui .data-table td.num { font-family: var(--font-mono); font-weight: 500; font-variant-numeric: tabular-nums; }
.ui .data-table th + th, .ui .data-table td + td { padding-left: 26px; }
.ui .data-table tr:last-child td { border-bottom: 0; }
.ui .data-table .best td { color: #cfd1d8; }
.ui .data-table .best .hl { color: var(--accent); }
```

- [ ] **Step 4: Append the specimen to `#components`**

```html
        <table class="data-table"><thead><tr><th>Model</th><th>Test MSE</th><th>Var</th></tr></thead>
        <tbody>
          <tr><td>OLS</td><td class="num">0.471</td><td class="num">0.410</td></tr>
          <tr class="best"><td>Ridge</td><td class="num hl">0.388</td><td class="num">0.246</td></tr>
        </tbody></table>
```

- [ ] **Step 5: Run the test** -> `ALL PASS`.

- [ ] **Step 6: Commit**

```bash
git add styles/components.css pages/_redesign-preview.html
git commit -m "feat(redesign): data table component"
```

---

## Task 4: Tooltip (card + pill)

**Files:**
- Modify: `styles/components.css`
- Modify: `pages/_redesign-preview.html`
- Test: `/tmp/pwverify/p2_tooltip.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p2_tooltip.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:2000} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await want('.ui .tooltip', 'backgroundColor', 'rgb(20, 20, 24)');
await want('.ui .tooltip', 'borderTopColor', 'rgba(255, 255, 255, 0.1)');
await want('.ui .tooltip .tt-value', 'color', 'rgb(107, 124, 255)');
await want('.ui .tooltip.pill', 'backgroundColor', 'rgb(6, 7, 10)');
await want('.ui .tooltip.pill', 'fontVariantNumeric', 'tabular-nums');
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run to verify it fails** -> FAIL.

- [ ] **Step 3: Append to `styles/components.css`**

```css
/* Tooltip (card default; .pill for a single value). JS positions it. */
.ui .tooltip {
  display: inline-block;
  background: #141418;
  border: 1px solid rgba(255, 255, 255, 0.10);
  border-radius: var(--radius-md);
  padding: 9px 11px;
  box-shadow: 0 1px 0 rgba(255, 255, 255, 0.04) inset, 0 10px 28px -10px rgba(0, 0, 0, 0.8);
  font-family: var(--font-sans);
}
.ui .tooltip .tt-title { font: 600 11px/1 var(--font-sans); color: #ededf0; margin-bottom: 7px; }
.ui .tooltip .tt-row { display: flex; justify-content: space-between; gap: 18px; font: 500 11.5px/1.5 var(--font-mono); }
.ui .tooltip .tt-row span { color: #7a7c84; }
.ui .tooltip .tt-value { color: var(--accent); font-weight: 500; }
.ui .tooltip.pill {
  padding: 6px 9px;
  background: #06070a;
  border-color: rgba(255, 255, 255, 0.06);
  border-radius: var(--radius-sm);
  box-shadow: 0 6px 18px -8px rgba(0, 0, 0, 0.8);
  font: 500 11.5px/1 var(--font-mono);
  color: #cfd1d8;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
```

- [ ] **Step 4: Append the specimen to `#components`**

```html
        <div style="display:flex;gap:20px;margin-top:8px">
          <div class="tooltip"><div class="tt-title">Sample 14</div>
            <div class="tt-row"><span>x</span><b class="tt-value">1.42</b></div>
            <div class="tt-row"><span>density</span><b class="tt-value">0.83</b></div></div>
          <div class="tooltip pill">x <b class="tt-value">1.42</b></div>
        </div>
```

- [ ] **Step 5: Run the test** -> `ALL PASS`.

- [ ] **Step 6: Commit**

```bash
git add styles/components.css pages/_redesign-preview.html
git commit -m "feat(redesign): tooltip component (card + pill)"
```

---

## Task 5: Display-math wrapper + worked numeric block

**Files:**
- Modify: `styles/components.css`
- Modify: `pages/_redesign-preview.html`
- Test: `/tmp/pwverify/p2_math.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p2_math.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:2200} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await want('.ui .math-display', 'borderLeftWidth', '2px');
await want('.ui .math-display', 'borderLeftColor', 'rgb(107, 124, 255)');
await want('.ui .math-display', 'textAlign', 'left');
await want('.ui .math-display .eq-num', 'position', 'absolute');
await want('.ui .math-display .eq-num', 'color', 'rgb(93, 95, 104)');
await want('.ui .worked', 'backgroundColor', 'rgba(255, 255, 255, 0.04)');
await want('.ui .worked .hl', 'color', 'rgb(107, 124, 255)');
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run to verify it fails** -> FAIL.

- [ ] **Step 3: Append to `styles/components.css`**

```css
/* Display math wrapper (left rule, left-aligned, numbered) around MathJax output. */
.ui .math-display {
  position: relative;
  border-left: 2px solid var(--accent);
  padding: 13px 56px 13px 22px;
  margin: 20px 0;
  text-align: left;
}
.ui .math-display .eq-num {
  position: absolute;
  right: 18px;
  top: 50%;
  transform: translateY(-50%);
  font: 500 12px/1 var(--font-mono);
  color: var(--text-muted);
}

/* Worked numeric block (matrices / step values). */
.ui .worked {
  display: inline-flex;
  align-items: center;
  gap: 18px;
  background: var(--surface);
  border: 1px solid var(--surface-border);
  border-radius: 9px;
  padding: 14px 18px;
  font: 500 13px/1.6 var(--font-mono);
  color: #d2d4db;
  font-variant-numeric: tabular-nums;
}
.ui .worked .mat { border-left: 1.5px solid rgba(255, 255, 255, 0.25); border-right: 1.5px solid rgba(255, 255, 255, 0.25); padding: 2px 12px; }
.ui .worked .hl { color: var(--accent); }
```

- [ ] **Step 4: Append the specimen to `#components`**

```html
        <div class="math-display"><span>x&#770; = &#8721; (x&#183;v) v</span><span class="eq-num">(6)</span></div>
        <div class="worked"><span class="mat">2.41&nbsp;0.30<br>0.30&nbsp;0.92</span><span>&rarr;</span><span class="mat"><span class="hl">2.48</span>&nbsp;0.00<br>0.00&nbsp;<span class="hl">0.85</span></span></div>
```

- [ ] **Step 5: Run the test** -> `ALL PASS`.

- [ ] **Step 6: Commit**

```bash
git add styles/components.css pages/_redesign-preview.html
git commit -m "feat(redesign): display-math wrapper + worked numeric block"
```

---

## Task 6: Figure frame + caption

**Files:**
- Modify: `styles/components.css`
- Modify: `pages/_redesign-preview.html`
- Test: `/tmp/pwverify/p2_figure.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p2_figure.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:2400} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await want('.ui .figure-frame', 'borderTopColor', 'rgba(255, 255, 255, 0.1)');
await want('.ui .figure-frame', 'borderTopLeftRadius', '8px');
await want('.ui .figure-frame', 'overflow', 'hidden');
await want('.ui .figcap', 'textTransform', 'none');
const cf = await cs('.ui .figcap', 'fontFamily');
if (!/JetBrains Mono/.test(cf)) { console.log(`FAIL figcap font: ${cf}`); fail++; }
await want('.ui .figcap b', 'color', 'rgb(189, 192, 200)');
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run to verify it fails** -> FAIL.

- [ ] **Step 3: Append to `styles/components.css`**

```css
/* Figure frame + caption. The figure carries its own edge (no-frames page, framed instrument). */
.ui .figure-frame {
  background: radial-gradient(120% 120% at 50% 45%, #0e0f15, #070709);
  border: 1px solid var(--figure-edge);
  border-radius: var(--radius-md);
  overflow: hidden;
}
.ui .figcap {
  font: 500 10.5px/1.3 var(--font-mono);
  letter-spacing: 0.04em;
  color: #7a7c84;
  margin-top: 8px;
}
.ui .figcap b { color: #bdc0c8; font-weight: 600; }
```

- [ ] **Step 4: Append the specimen to `#components`**

```html
        <div style="max-width:220px;margin-top:8px">
          <div class="figure-frame" style="aspect-ratio:4/3"></div>
          <div class="figcap"><b>Original</b> swiss roll, 3D</div>
        </div>
```

- [ ] **Step 5: Run the test** -> `ALL PASS`.

- [ ] **Step 6: Commit**

```bash
git add styles/components.css pages/_redesign-preview.html
git commit -m "feat(redesign): figure frame + caption component"
```

---

## Task 7: Full verification + visual check

**Files:** none (verification only)

- [ ] **Step 1: Run all Phase 2a tests**

```bash
for t in callout code table tooltip math figure; do echo "== $t =="; node /tmp/pwverify/p2_$t.mjs | tail -1; done
```
Expected: every line `ALL PASS`.

- [ ] **Step 2: Re-run Phase 1 tests (no regression)**

```bash
for t in tokens type inline polish isolation; do echo "== $t =="; node /tmp/pwverify/p1_$t.mjs | tail -1; done
```
Expected: every line `ALL PASS`.

- [ ] **Step 3: Run the JS suite**

Run: `node --test 'test/**/*.test.js'`
Expected: `# pass 65`, `# fail 0`.

- [ ] **Step 4: Screenshot the components specimen**

```bash
cat > /tmp/pwverify/p2_shot.mjs <<'EOF'
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:1300} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const el = await p.$('#components');
await el.screenshot({ path: '/tmp/pwverify/p2_components.png' });
await b.close();
EOF
node /tmp/pwverify/p2_shot.mjs
```
Confirm the rendered components match the cited mockups (callout left rule, dark-inset code, hairline table, tooltips, left-ruled numbered equation + worked block, framed figure).

- [ ] **Step 5: Em-dash sweep**

```bash
grep -lP "[^\x00-\x7f]" styles/components.css && echo "REVIEW non-ascii" || echo "ascii-clean"
```
(The components.css should be ASCII; HTML entities in the preview are fine.)

---

## Notes for the implementer

- Append to `styles/components.css` in order; do not modify `tokens.css`, `system.css`, or any existing page.
- Strict: no em-dash characters in any file.
- Tooltip and figure components are CSS only here; their JS positioning/content wiring happens during page migration (Phase 4). The classes and look are what this phase locks.
- Later phases (own plans): 2b controls + control-management patterns; 2c composites (home index, outline-rail re-skin, walkthrough player, paired-viz layout); then page migration.
