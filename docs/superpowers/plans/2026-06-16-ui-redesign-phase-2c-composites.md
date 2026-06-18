# UI Redesign Phase 2c: Composite Components Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the composite components (home index, paired-visualization layout, walkthrough player, section-outline rail re-skin) to `styles/components.css` under `.ui`, verified on the preview page.

**Architecture:** Continues the opt-in `.ui` layer. The outline rail re-skin is `.ui`-scoped overrides of the existing `section-outline.css` classes, so it only applies on pages that opt into `.ui` (the eight live pages, which do not have `.ui`, stay on the current rail styling). The player's step dropdown and tab/disclosure behavior are CSS only; JS wiring happens at page migration (Phase 4). The preview page renders static specimens (dropdown shown open) for verification.

**Tech Stack:** Plain CSS. Verification: `python3 -m http.server 8000` + Playwright (`/tmp/pwverify/node_modules/playwright`, cached chromium). Import form: `import pkg from '...'; const { chromium } = pkg;`.

**Spec:** sections 4 (chrome), 7 (figures/paired layout), 9 (player), 10 (home index), 11.5 (rail). Mockups: `docs/design/mockups/{home-directory,paired-viz-layout,video-player,step-nav-approaches,outline-rail}.html`.

**Depends on:** Phase 1 + 2a + 2b (committed on this branch).

**Scope:** The four composites. NOT here: settings page (Phase 3), page migration (Phase 4).

---

## Harness

Server (start once): `python3 -m http.server 8000 >/tmp/redesign_srv.log 2>&1 &`.
Each test `/tmp/pwverify/<name>.mjs` starts with the CJS import and these helpers:
```js
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
```
Each composite gets its own preview specimen inside a new `<section id="composites">` (Task 1 creates it).

Color reference: accent `#6b7cff`->`rgb(107, 124, 255)`, `#d6d8df`->`rgb(214, 216, 223)`, `#74767f`->`rgb(116, 118, 127)`, `#5d5f68`->`rgb(93, 95, 104)`, `#fafafa`->`rgb(250, 250, 250)`, `#ededf0`->`rgb(237, 237, 240)`, `#9a9ca4`->`rgb(154, 156, 164)`.

---

## Task 1: Home index

**Files:** Modify `styles/components.css`, `pages/_redesign-preview.html`. Test: `/tmp/pwverify/p2c_home.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p2c_home.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:3400} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await want('.ui .home-index .index-row', 'borderTopColor', 'rgba(255, 255, 255, 0.08)');
await want('.ui .home-index .index-num', 'color', 'rgb(93, 95, 104)');
const nf = await cs('.ui .home-index .index-num', 'fontFamily');
if (!/JetBrains Mono/.test(nf)) { console.log(`FAIL index-num font: ${nf}`); fail++; }
await want('.ui .home-index .index-title', 'color', 'rgb(214, 216, 223)');
await want('.ui .home-index .index-desc', 'color', 'rgb(116, 118, 127)');
await want('.ui .home-index .index-arrow', 'color', 'rgb(93, 95, 104)');
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Append to `styles/components.css`**

```css
/* ===== Composites ===== */

/* Home index (single-column editorial directory). */
.ui .home-index .index-list { margin-top: 34px; }
.ui .home-index .index-row {
  display: flex;
  align-items: baseline;
  gap: 20px;
  padding: 17px 6px 17px 0;
  border: 0;
  border-top: 1px solid var(--hairline);
  text-decoration: none;
}
.ui .home-index .index-row:last-child { border-bottom: 1px solid var(--hairline); }
.ui .home-index .index-num { font: 500 12px/1.3 var(--font-mono); color: var(--text-muted); flex-shrink: 0; width: 22px; }
.ui .home-index .index-body { flex: 1; min-width: 0; }
.ui .home-index .index-title { font: 600 16px/1.3 var(--font-sans); letter-spacing: -0.01em; color: #d6d8df; }
.ui .home-index .index-desc { font: 400 13px/1.5 var(--font-sans); color: #74767f; margin-top: 3px; max-width: 62ch; }
.ui .home-index .index-arrow { font: 500 14px/1 var(--font-mono); color: var(--text-muted); flex-shrink: 0; align-self: center; transition: transform var(--dur-hover) var(--ease-out), color var(--dur-hover); }
.ui .home-index .index-row:hover .index-title { color: var(--text); }
.ui .home-index .index-row:hover .index-num,
.ui .home-index .index-row:hover .index-arrow { color: var(--accent); }
.ui .home-index .index-row:hover .index-arrow { transform: translateX(3px); }

/* Footer */
.ui .site-footer { margin-top: 30px; padding-top: 18px; border-top: 1px solid var(--hairline); display: flex; justify-content: space-between; align-items: center; }
.ui .site-footer .credit { font: 400 12px/1 var(--font-sans); color: var(--text-muted); }
.ui .site-footer .credit a { color: var(--accent-link); }
.ui .site-footer .meta { font: 500 11px/1 var(--font-mono); letter-spacing: 0.08em; color: #4d4f57; }
```

- [ ] **Step 4: Add the composites specimen section to the preview** (before `</main>`, after `#controls`):

```html
      <hr class="rule">
      <section id="composites" class="home-index">
        <div class="index-list">
          <a class="index-row" href="#"><span class="index-num">01</span>
            <span class="index-body"><span class="index-title">Distribution Visualizer</span>
              <span class="index-desc">PDF/CDF visualizer for Normal, Uniform, and mixtures.</span></span>
            <span class="index-arrow">&rarr;</span></a>
          <a class="index-row" href="#"><span class="index-num">02</span>
            <span class="index-body"><span class="index-title">Principal Component Analysis</span>
              <span class="index-desc">PCA via SVD, principal directions, and reconstruction.</span></span>
            <span class="index-arrow">&rarr;</span></a>
        </div>
        <div class="site-footer"><span class="credit">Created by <a href="#">Aughdon Breslin</a></span><span class="meta">2 PROJECTS</span></div>
      </section>
```

- [ ] **Step 5: Run -> `ALL PASS`. Commit.**

```bash
git add styles/components.css pages/_redesign-preview.html
git commit -m "feat(redesign): home index composite"
```

---

## Task 2: Paired-visualization layout

**Files:** Modify `styles/components.css`, `pages/_redesign-preview.html`. Test: `/tmp/pwverify/p2c_pair.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p2c_pair.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:3800} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await want('.ui .viz-pair', 'display', 'grid');
// computed grid-template-columns resolves to pixel tracks; assert two tracks on desktop
const desk = await cs('.ui .viz-pair', 'gridTemplateColumns');
if (!/^\d+(\.\d+)?px \d+(\.\d+)?px$/.test(desk)) { console.log(`FAIL viz-pair not two columns: ${desk}`); fail++; }
await want('.ui .viz-controls', 'borderTopColor', 'rgba(255, 255, 255, 0.08)');
// at narrow width the pair stacks to one column
await p.setViewportSize({ width: 560, height: 3800 });
const cols = await cs('.ui .viz-pair', 'gridTemplateColumns');
if (!/^\d+(\.\d+)?px$/.test(cols)) { console.log(`FAIL viz-pair should be single column on mobile: ${cols}`); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Append to `styles/components.css`**

```css
/* Paired-visualization layout: two figures side by side, controls below. */
.ui .viz-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.ui .viz-controls { margin-top: 18px; padding-top: 18px; border-top: 1px solid var(--hairline); }
@media (max-width: 640px) {
  .ui .viz-pair { grid-template-columns: 1fr; }
}
```

- [ ] **Step 4: Append the specimen to `#composites`** (after the footer, still inside the section):

```html
        <hr class="rule">
        <div class="viz-pair">
          <div><div class="figure-frame" style="aspect-ratio:4/3"></div><div class="figcap"><b>Original</b> swiss roll, 3D</div></div>
          <div><div class="figure-frame" style="aspect-ratio:4/3"></div><div class="figcap"><b>Isomap</b> 2D embedding</div></div>
        </div>
        <div class="viz-controls">
          <label class="field"><span class="field-label">Neighbors</span><input type="range" value="38"></label>
        </div>
```

- [ ] **Step 5: Run -> `ALL PASS`. Commit.**

```bash
git add styles/components.css pages/_redesign-preview.html
git commit -m "feat(redesign): paired-visualization layout (controls below)"
```

---

## Task 3: Walkthrough player (stacked) + step dropdown

**Files:** Modify `styles/components.css`, `pages/_redesign-preview.html`. Test: `/tmp/pwverify/p2c_player.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p2c_player.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:4400} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await want('.ui .player .video', 'borderTopColor', 'rgba(255, 255, 255, 0.1)');
await want('.ui .player .video', 'overflow', 'hidden');
await want('.ui .player .scrub-track', 'backgroundColor', 'rgba(255, 255, 255, 0.12)');
await want('.ui .player .scrub-fill', 'backgroundColor', 'rgb(107, 124, 255)');
await want('.ui .player .step-k', 'color', 'rgb(107, 124, 255)');
await want('.ui .player .step-menu', 'backgroundColor', 'rgb(20, 20, 24)');
await want('.ui .player .step-menu .step-item.current', 'color', 'rgb(237, 237, 240)');
const tf = await cs('.ui .player .scrub-time', 'fontVariantNumeric');
if (!/tabular-nums/.test(tf)) { console.log(`FAIL scrub-time tabular: ${tf}`); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Append to `styles/components.css`**

```css
/* Walkthrough player (stacked). JS wires play/scrub/step nav at migration. */
.ui .player .video {
  position: relative;
  aspect-ratio: 16/9;
  background: radial-gradient(120% 120% at 50% 40%, #0f1016, #060608);
  border: 1px solid rgba(255, 255, 255, 0.10);
  border-radius: var(--radius-lg);
  overflow: hidden;
}
.ui .player .video > svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.ui .player .video-title { position: absolute; left: 16px; top: 14px; font: 500 11px/1.3 var(--font-mono); color: rgba(255, 255, 255, 0.5); }
.ui .player .video-title b { display: block; font: 600 13px/1.4 var(--font-sans); color: rgba(255, 255, 255, 0.85); letter-spacing: -0.01em; margin-top: 3px; }
.ui .player .big-play { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 58px; height: 58px; border-radius: 50%; background: rgba(8, 8, 12, 0.55); -webkit-backdrop-filter: blur(4px); backdrop-filter: blur(4px); border: 1px solid rgba(255, 255, 255, 0.18); display: flex; align-items: center; justify-content: center; }
.ui .player .big-play::after { content: ''; border-left: 15px solid var(--accent); border-top: 9px solid transparent; border-bottom: 9px solid transparent; margin-left: 4px; }

.ui .player .scrub { display: flex; align-items: center; gap: 14px; margin-top: 16px; }
.ui .player .scrub-play { width: 30px; height: 30px; border-radius: 50%; border: 1px solid rgba(255, 255, 255, 0.16); background: rgba(255, 255, 255, 0.04); flex-shrink: 0; cursor: pointer; }
.ui .player .scrub-track { position: relative; flex: 1; height: 4px; background: rgba(255, 255, 255, 0.12); border-radius: 3px; }
.ui .player .scrub-fill { position: absolute; left: 0; top: 0; bottom: 0; width: 34%; background: var(--accent); border-radius: 3px; }
.ui .player .scrub-tick { position: absolute; top: -3px; width: 2px; height: 10px; background: rgba(255, 255, 255, 0.22); border-radius: 1px; }
.ui .player .scrub-head { position: absolute; left: 34%; top: 50%; width: 12px; height: 12px; border-radius: 50%; background: var(--accent); border: 2px solid var(--bg); transform: translate(-50%, -50%); }
.ui .player .scrub-time { font: 500 12px/1 var(--font-mono); color: #9a9ca4; font-variant-numeric: tabular-nums; flex-shrink: 0; }

.ui .player .step-bar { display: flex; align-items: flex-start; justify-content: space-between; margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--hairline); }
.ui .player .step-k { font: 600 10.5px/1 var(--font-mono); letter-spacing: 0.16em; color: var(--accent); }
.ui .player .step-trigger { display: inline-flex; align-items: center; gap: 9px; cursor: pointer; margin-top: 5px; }
.ui .player .step-title { font: 600 16px/1.3 var(--font-sans); color: #ededf0; letter-spacing: -0.01em; }
.ui .player .step-chev { color: #7a7c84; font-size: 11px; }
.ui .player .step-menu-wrap { position: relative; }
.ui .player .step-menu { position: absolute; top: calc(100% + 8px); left: 0; width: 320px; background: var(--surface-strong); border: 1px solid var(--surface-strong-border); border-radius: var(--radius-lg); padding: 6px; box-shadow: 0 16px 40px -16px rgba(0, 0, 0, 0.8); z-index: 5; }
.ui .player .step-item { display: flex; align-items: center; gap: 11px; padding: 9px 10px; border-radius: var(--radius-sm); cursor: pointer; }
.ui .player .step-item .step-n { font: 500 11px/1 var(--font-mono); color: var(--text-muted); width: 16px; flex-shrink: 0; }
.ui .player .step-item .step-l { font: 500 13px/1.3 var(--font-sans); color: #74767f; }
.ui .player .step-item:hover { background: rgba(255, 255, 255, 0.04); }
.ui .player .step-item.current { background: var(--accent-muted); }
.ui .player .step-item.current .step-l { color: #ededf0; }
.ui .player .step-item.current .step-n { color: var(--accent); }
.ui .player .step-item.done .step-l { color: #9a9ca4; }
.ui .player .step-nav { display: flex; gap: 10px; }
.ui .player .step-nav .btn { font: 500 12.5px/1 var(--font-sans); color: #c4c6cd; background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: var(--radius-md); padding: 9px 13px; cursor: pointer; }
.ui .player .transcript { margin-top: 18px; font: 400 14px/1.7 var(--font-sans); color: var(--text-body); max-width: 64ch; }
```

- [ ] **Step 4: Append the specimen to `#composites`** (after the viz-controls):

```html
        <hr class="rule">
        <div class="player">
          <div class="video">
            <svg viewBox="0 0 384 216" preserveAspectRatio="xMidYMid slice">
              <defs><linearGradient id="proll" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6b7cff"/><stop offset=".5" stop-color="#3fb9b1"/><stop offset="1" stop-color="#e6a93f"/></linearGradient></defs>
              <path d="M192 108 C 214 108 228 90 228 74 C 228 50 200 40 174 40 C 138 40 118 70 118 104 C 118 150 162 174 206 174 C 262 174 296 128 296 76 C 296 22 246 -6 190 -6" fill="none" stroke="url(#proll)" stroke-width="11" stroke-linecap="round" opacity=".92"/>
            </svg>
            <div class="video-title">CHAPTER 3<b>Build the neighborhood graph</b></div>
            <div class="big-play"></div>
          </div>
          <div class="scrub">
            <span class="scrub-play"></span>
            <div class="scrub-track"><div class="scrub-fill"></div><span class="scrub-tick" style="left:14%"></span><span class="scrub-tick" style="left:52%"></span><span class="scrub-tick" style="left:80%"></span><div class="scrub-head"></div></div>
            <span class="scrub-time">01:12 / 03:40</span>
          </div>
          <div class="step-bar">
            <div class="step-menu-wrap">
              <div class="step-k">STEP 3 / 7</div>
              <div class="step-trigger"><span class="step-title">Build the neighborhood graph</span><span class="step-chev">&#9662;</span></div>
              <div class="step-menu">
                <div class="step-item done"><span class="step-n">01</span><span class="step-l">Sample the swiss roll</span></div>
                <div class="step-item current"><span class="step-n">03</span><span class="step-l">Build the neighborhood graph</span></div>
                <div class="step-item"><span class="step-n">04</span><span class="step-l">Shortest-path geodesics</span></div>
              </div>
            </div>
            <div class="step-nav"><button class="btn">&#8249; Prev</button><button class="btn">Next &#8250;</button></div>
          </div>
          <div class="transcript">Connect each point to its k nearest neighbors; the geodesic distance is the shortest path along this graph.</div>
        </div>
```

- [ ] **Step 5: Run -> `ALL PASS`. Commit.**

```bash
git add styles/components.css pages/_redesign-preview.html
git commit -m "feat(redesign): walkthrough player (stacked) + step dropdown"
```

---

## Task 4: Section-outline rail re-skin (.ui-scoped)

**Files:** Modify `styles/components.css`, `pages/_redesign-preview.html`. Test: `/tmp/pwverify/p2c_rail.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p2c_rail.mjs
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1200,height:4800} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await want('.ui .section-outline-list a', 'color', 'rgb(116, 118, 127)');
await want('.ui .section-outline-list a .rail-n', 'color', 'rgb(77, 79, 87)');
await want('.ui .section-outline-list a.active', 'color', 'rgb(250, 250, 250)');
await want('.ui .section-outline-list a.active', 'borderLeftColor', 'rgb(107, 124, 255)');
await want('.ui .section-outline-list a.active .rail-n', 'color', 'rgb(107, 124, 255)');
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Append to `styles/components.css`**

```css
/* Section-outline rail re-skin (mono-numbered). Overrides section-outline.css only under
   .ui, so the eight live pages keep their current rail until they opt in. */
.ui .section-outline-list a {
  display: flex;
  gap: 10px;
  align-items: baseline;
  padding: 6px 10px;
  border: 0;
  border-left: 2px solid transparent;
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  color: #74767f;
  font: 500 13px/1.3 var(--font-sans);
  text-decoration: none;
}
.ui .section-outline-list a .rail-n { font: 500 10.5px/1.3 var(--font-mono); color: #4d4f57; }
.ui .section-outline-list a:hover { color: #cfd1d8; background: rgba(255, 255, 255, 0.03); }
.ui .section-outline-list a.active { color: var(--text); border-left-color: var(--accent); background: var(--accent-muted); font-weight: 600; }
.ui .section-outline-list a.active .rail-n { color: var(--accent); }
```

- [ ] **Step 4: Append the specimen to `#composites`**

```html
        <hr class="rule">
        <nav class="section-outline" aria-label="On this page" style="max-width:200px">
          <ul class="section-outline-list" style="list-style:none;margin:0;padding:0">
            <li><a href="#"><span class="rail-n">01</span>Overview</a></li>
            <li><a href="#" class="active"><span class="rail-n">02</span>Covariance</a></li>
            <li><a href="#"><span class="rail-n">03</span>Reconstruction</a></li>
          </ul>
        </nav>
```

- [ ] **Step 5: Run -> `ALL PASS`. Commit.**

```bash
git add styles/components.css pages/_redesign-preview.html
git commit -m "feat(redesign): section-outline rail re-skin (.ui, mono-numbered)"
```

---

## Task 5: Verification + visual check

- [ ] **Step 1: Run all Phase 2c tests**

```bash
for t in home pair player rail; do echo "== $t =="; node /tmp/pwverify/p2c_$t.mjs | tail -1; done
```
Expected: every line `ALL PASS`.

- [ ] **Step 2: Re-run Phase 1 + 2a + 2b (no regression)**

```bash
for t in tokens type inline polish isolation; do node /tmp/pwverify/p1_$t.mjs | tail -1; done
for t in callout code table tooltip math figure; do node /tmp/pwverify/p2_$t.mjs | tail -1; done
for t in buttons fields tabs inputs manage touch; do node /tmp/pwverify/p2b_$t.mjs | tail -1; done
```
Expected: all `ALL PASS`.

- [ ] **Step 3: JS suite** -> `node --test 'test/**/*.test.js'` -> `# pass 65`, `# fail 0`.

- [ ] **Step 4: Screenshot the composites section**

```bash
cat > /tmp/pwverify/p2c_shot.mjs <<'EOF'
import pkg from '/tmp/pwverify/node_modules/playwright/index.js';
const { chromium } = pkg;
const b = await chromium.launch();
const p = await (await b.newContext({ viewport:{width:1100,height:1400} })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil:'networkidle' });
const el = await p.$('#composites');
await el.screenshot({ path: '/tmp/pwverify/p2c_composites.png' });
await b.close();
EOF
node /tmp/pwverify/p2c_shot.mjs
```
Confirm against `docs/design/mockups/{home-directory,paired-viz-layout,video-player,outline-rail}.html`.

- [ ] **Step 5: Em-dash sweep** -> `grep -lP "\x{2014}" styles/components.css && echo FIX || echo clean`.

---

## Notes for the implementer

- Append to `styles/components.css` in order; do not modify `tokens.css`, `system.css`, `section-outline.css`, or any page other than `pages/_redesign-preview.html`.
- Strict: no em-dash characters in any file.
- The outline-rail re-skin is intentionally `.ui`-scoped so the eight live pages are unaffected; do not edit `section-outline.css`.
- The player's step dropdown is shown open in the specimen for verification; open/close, scrub, and step-jump behavior are wired during page migration (Phase 4).
- After this phase, Phase 2 (components) is complete. Next: Phase 3 (settings page + theming boot script), then Phase 4 (page migration).
