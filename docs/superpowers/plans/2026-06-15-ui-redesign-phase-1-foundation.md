# UI Redesign Phase 1: Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the redesign's token + global-text foundation as an opt-in layer (`styles/tokens.css` + `styles/system.css`, gated by a `ui` body class) validated on a hidden preview page, without changing any of the eight existing pages.

**Architecture:** New CSS lives in two new files. `tokens.css` defines all design tokens on `:root` (custom properties are inert until used, so this is globally safe). `system.css` defines global element styling (body, type scale, links, inline elements, polish) scoped under a `.ui` class so it only applies where opted in; class-level specificity cleanly overrides `base.css` element rules on pages that later opt in. A hidden, unlinked `pages/_redesign-preview.html` opts in and serves as the pilot and the verification surface. No existing page is touched in this phase.

**Tech Stack:** Plain CSS, no build. Verification: `python3 -m http.server 8000` (repo root) + Playwright (the `playwright` lib in `/tmp/pwverify`, driving the cached chromium) running Node scripts that assert `getComputedStyle` values against the spec. Fonts: Inter + JetBrains Mono via Google Fonts.

**Spec:** `docs/superpowers/specs/2026-06-15-ui-redesign-design.md` (sections 2, 3, 4). Visual source: `docs/design/mockups/*.html`.

**Scope:** Tokens, global text/links/inline, global polish, font loading, pilot preview page. NOT in this phase: components (`components.css`), real-page migration, settings page (separate plans).

---

## Verification harness (read once; used by every task)

All checks run against the preview page on the local server with a Playwright Node script.

Setup (run once at the start of execution; re-run the server line if it stopped):
```bash
cd /mnt/c/Users/aughb/PersonalProjects/Visualization
python3 -m http.server 8000 >/tmp/redesign_srv.log 2>&1 &   # serves the repo
# Playwright lib + cached chromium already present at /tmp/pwverify and ~/.cache/ms-playwright
node -e "require('/tmp/pwverify/node_modules/playwright')" && echo "playwright OK"
```

Each task's test is a `/tmp/pwverify/<name>.mjs` script importing chromium from the local lib:
```js
import { chromium } from '/tmp/pwverify/node_modules/playwright/index.js';
```
Run with `node /tmp/pwverify/<name>.mjs`. A task "fails first" because the page/CSS/value does not exist yet, then passes after implementation.

---

## Task 1: Hidden preview page skeleton + tokens.css

**Files:**
- Create: `pages/_redesign-preview.html`
- Create: `styles/tokens.css`
- Test: `/tmp/pwverify/p1_tokens.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p1_tokens.mjs
import { chromium } from '/tmp/pwverify/node_modules/playwright/index.js';
const b = await chromium.launch();
const p = await (await b.newContext()).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil: 'networkidle' });
const v = (name) => p.evaluate((n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim(), name);
const checks = {
  '--bg': '#0c0c0d', '--text': '#fafafa', '--text-body': '#8b8d96', '--text-muted': '#5d5f68',
  '--accent': '#6b7cff', '--radius-md': '8px', '--sp-4': '16px', '--dur-hover': '140ms',
  '--container-default': '1200px',
};
let fail = 0;
for (const [k, want] of Object.entries(checks)) {
  const got = await v(k);
  if (got !== want) { console.log(`FAIL ${k}: ${JSON.stringify(got)} != ${want}`); fail++; }
}
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run to verify it fails**

Run: `node /tmp/pwverify/p1_tokens.mjs`
Expected: FAIL (the page returns 404 / tokens unset), script throws or reports FAIL.

- [ ] **Step 3: Create `styles/tokens.css`**

```css
/* styles/tokens.css - redesign design tokens (custom properties only; inert until used). */
:root {
  /* fonts (substitutable; single source of truth) */
  --font-sans: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, monospace;

  /* neutrals (graphite) */
  --bg: #0c0c0d;
  --text: #fafafa;
  --text-body: #8b8d96;
  --text-muted: #5d5f68;
  --hairline: rgba(255, 255, 255, 0.08);
  --hairline-strong: rgba(255, 255, 255, 0.22);
  --surface: rgba(255, 255, 255, 0.04);
  --surface-border: rgba(255, 255, 255, 0.06);
  --surface-strong: #141418;
  --surface-strong-border: rgba(255, 255, 255, 0.10);
  --figure-edge: rgba(255, 255, 255, 0.10);

  /* accent (user-overridable) */
  --accent: #6b7cff;
  --accent-muted: rgba(107, 124, 255, 0.14);
  --accent-link: #9aa6ff;
  --focus-ring: 2px solid rgba(107, 124, 255, 0.60);

  /* status */
  --warning: #e0b341;
  --warning-bg: rgba(224, 179, 65, 0.10);
  --warning-text: #f0c460;

  /* radii */
  --radius-xs: 3px; --radius-sm: 6px; --radius-md: 8px; --radius-lg: 10px;
  --radius-xl: 12px; --radius-pill: 999px;

  /* spacing */
  --sp-1: 4px; --sp-2: 8px; --sp-3: 12px; --sp-4: 16px; --sp-5: 20px; --sp-6: 24px; --sp-8: 32px;

  /* motion */
  --dur-instant: 80ms; --dur-hover: 140ms; --dur-state: 200ms;
  --ease-out: cubic-bezier(0, 0, 0.2, 1); --ease-in: cubic-bezier(0.4, 0, 1, 1);

  /* containers */
  --container-narrow: 1100px; --container-default: 1200px;
  --container-wide: 1400px; --container-xl: 1560px;
}
```

- [ ] **Step 4: Create `pages/_redesign-preview.html`** (hidden, not linked anywhere)

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Redesign preview (internal)</title>
  <meta name="robots" content="noindex" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="../styles/tokens.css">
  <link rel="stylesheet" href="../styles/system.css">
</head>
<body class="ui">
  <div class="container">
    <header class="page-head">
      <div class="eyebrow">Visualization</div>
      <h1>Principal Component Analysis</h1>
      <p class="lede">Interactive walkthrough of PCA via SVD, principal directions, and reconstruction.</p>
    </header>
    <main>
      <hr class="rule">
      <section>
        <h2>Covariance and change of basis</h2>
        <h3>Reconstruction error</h3>
        <p>PCA finds the orthogonal directions of maximum variance; the top components give the
        best low-rank fit. See <a href="#">the SVD step</a> or set <code>rank = 2</code>.</p>
        <p class="caption">Figure 1 - projection onto the first two components</p>
        <div class="label">Components</div>
        <p class="readout">variance explained <b>92.4%</b></p>
      </section>
    </main>
  </div>
</body>
</html>
```

(`system.css` does not exist yet; the page still loads. The token test only needs `tokens.css`.)

- [ ] **Step 5: Run the test to verify tokens pass**

Run: `node /tmp/pwverify/p1_tokens.mjs`
Expected: `ALL PASS`.

- [ ] **Step 6: Commit**

```bash
git add styles/tokens.css pages/_redesign-preview.html
git commit -m "feat(redesign): design tokens + hidden preview page"
```

---

## Task 2: Global base + type scale (`system.css` under `.ui`)

**Files:**
- Create: `styles/system.css`
- Test: `/tmp/pwverify/p1_type.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p1_type.mjs
import { chromium } from '/tmp/pwverify/node_modules/playwright/index.js';
const b = await chromium.launch();
const p = await (await b.newContext({ viewport: { width: 1200, height: 900 } })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil: 'networkidle' });
const cs = (sel, prop) => p.evaluate(([s, pr]) => {
  const el = document.querySelector(s); return el ? getComputedStyle(el)[pr] : null;
}, [sel, prop]);
let fail = 0;
const want = async (sel, prop, expected) => {
  const got = await cs(sel, prop);
  if (got !== expected) { console.log(`FAIL ${sel}.${prop}: ${JSON.stringify(got)} != ${expected}`); fail++; }
};
await want('body.ui', 'backgroundColor', 'rgb(12, 12, 13)');
await want('body.ui', 'color', 'rgb(139, 141, 150)');       // --text-body
await want('.ui h1', 'fontSize', '33px');
await want('.ui h1', 'fontWeight', '600');
await want('.ui h2', 'fontSize', '19px');
await want('.ui h3', 'fontSize', '15px');
await want('.ui p', 'fontSize', '14.5px');
await want('.ui h1', 'color', 'rgb(250, 250, 250)');        // --text
const ff = await cs('.ui h1', 'fontFamily');
if (!/Inter/.test(ff)) { console.log(`FAIL fontFamily missing Inter: ${ff}`); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run to verify it fails**

Run: `node /tmp/pwverify/p1_type.mjs`
Expected: FAIL (`system.css` missing, body has no redesign styles).

- [ ] **Step 3: Create `styles/system.css`** (base + type, scoped under `.ui`)

```css
/* styles/system.css - redesign global styling. Scoped under .ui so it only applies on
   opted-in pages and overrides base.css element rules via class specificity. */

.ui {
  background: var(--bg);
  color: var(--text-body);
  font-family: var(--font-sans);
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.ui .container {
  max-width: var(--container-default);
  margin: 0 auto;
  padding: var(--sp-5);
}

/* type scale (Balanced) */
.ui h1 {
  font: 600 33px/1.1 var(--font-sans);
  letter-spacing: -0.025em;
  color: var(--text);
  text-wrap: balance;
  margin: 0 0 var(--sp-3);
}
.ui h2 {
  font: 600 19px/1.25 var(--font-sans);
  letter-spacing: -0.014em;
  color: #ededf0;
  text-wrap: balance;
  margin: 0 0 var(--sp-2);
}
.ui h3 {
  font: 600 15px/1.35 var(--font-sans);
  color: #cfd1d8;
  margin: 0 0 var(--sp-2);
}
.ui p {
  font: 400 14.5px/1.7 var(--font-sans);
  color: var(--text-body);
  max-width: 62ch;
  text-wrap: pretty;
  margin: 0 0 var(--sp-4);
}
.ui .page-head { margin-bottom: var(--sp-5); }
.ui .lede {
  font: 400 15px/1.6 var(--font-sans);
  color: var(--text-body);
  max-width: 56ch;
}
.ui .rule { border: 0; height: 1px; background: var(--hairline); margin: var(--sp-6) 0; }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `node /tmp/pwverify/p1_type.mjs`
Expected: `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
git add styles/system.css
git commit -m "feat(redesign): global base + Balanced type scale (.ui)"
```

---

## Task 3: Eyebrow, label, caption, links, inline code, numeric readout

**Files:**
- Modify: `styles/system.css`
- Test: `/tmp/pwverify/p1_inline.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p1_inline.mjs
import { chromium } from '/tmp/pwverify/node_modules/playwright/index.js';
const b = await chromium.launch();
const p = await (await b.newContext({ viewport: { width: 1200, height: 900 } })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil: 'networkidle' });
const cs = (s, pr) => p.evaluate(([s, pr]) => { const e = document.querySelector(s); return e ? getComputedStyle(e)[pr] : null; }, [s, pr]);
let fail = 0;
const want = async (s, pr, exp) => { const g = await cs(s, pr); if (g !== exp) { console.log(`FAIL ${s}.${pr}: ${JSON.stringify(g)} != ${exp}`); fail++; } };
await want('.ui .eyebrow', 'color', 'rgb(107, 124, 255)');      // --accent
await want('.ui .eyebrow', 'textTransform', 'uppercase');
await want('.ui .eyebrow', 'fontSize', '11px');
await want('.ui .label', 'fontSize', '10.5px');
await want('.ui .label', 'textTransform', 'uppercase');
await want('.ui .caption', 'color', 'rgb(93, 95, 104)');       // --text-muted
await want('.ui a', 'color', 'rgb(154, 166, 255)');            // --accent-link
await want('.ui a', 'textDecorationLine', 'none');
const codeFF = await cs('.ui code', 'fontFamily');
if (!/JetBrains Mono/.test(codeFF)) { console.log(`FAIL code font: ${codeFF}`); fail++; }
const rn = await cs('.ui .readout', 'fontVariantNumeric');
if (!/tabular-nums/.test(rn)) { console.log(`FAIL readout tabular-nums: ${rn}`); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run to verify it fails**

Run: `node /tmp/pwverify/p1_inline.mjs`
Expected: FAIL (classes unstyled).

- [ ] **Step 3: Append to `styles/system.css`**

```css
/* eyebrow / label / caption */
.ui .eyebrow {
  font: 600 11px/1 var(--font-sans);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--accent);
}
.ui .label {
  font: 600 10.5px/1 var(--font-sans);
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #6c6e77;
}
.ui .caption {
  font: 400 12px/1.45 var(--font-sans);
  color: var(--text-muted);
}

/* links */
.ui a {
  color: var(--accent-link);
  text-decoration: none;
  border-bottom: 1px solid rgba(154, 166, 255, 0.40);
}
.ui a:hover { color: #c2caff; }

/* inline code */
.ui code {
  font-family: var(--font-mono);
  font-size: 0.88em;
  color: #c9cdfb;
  background: rgba(107, 124, 255, 0.10);
  padding: 1px 5px;
  border-radius: var(--radius-xs);
}

/* numeric readout */
.ui .readout {
  font: 500 13px/1 var(--font-mono);
  font-variant-numeric: tabular-nums;
  color: #cfd1d8;
}
.ui .readout b { color: var(--accent); font-weight: 600; }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `node /tmp/pwverify/p1_inline.mjs`
Expected: `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
git add styles/system.css
git commit -m "feat(redesign): eyebrow, label, caption, links, code, readout"
```

---

## Task 4: Global polish (selection, placeholder, focus, reduced-motion)

**Files:**
- Modify: `styles/system.css`
- Test: `/tmp/pwverify/p1_polish.mjs`

- [ ] **Step 1: Write the failing test**

```js
// /tmp/pwverify/p1_polish.mjs
import { chromium } from '/tmp/pwverify/node_modules/playwright/index.js';
const b = await chromium.launch();
const p = await (await b.newContext()).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil: 'networkidle' });
let fail = 0;
// the stylesheet text must contain the polish rules (computed ::selection is not readable)
const css = await p.evaluate(async () => {
  const res = await fetch('../styles/system.css'); return res.text();
});
const need = ['::selection', '::placeholder', 'prefers-reduced-motion', ':focus-visible'];
for (const n of need) { if (!css.includes(n)) { console.log(`FAIL missing ${n}`); fail++; } }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run to verify it fails**

Run: `node /tmp/pwverify/p1_polish.mjs`
Expected: FAIL (rules absent).

- [ ] **Step 3: Append to `styles/system.css`**

```css
/* global polish */
.ui ::selection { background: var(--accent-muted); color: #fff; }
.ui ::placeholder { color: var(--text-muted); opacity: 1; }
.ui :focus-visible { outline: var(--focus-ring); outline-offset: 2px; }

@media (prefers-reduced-motion: reduce) {
  .ui * { transition: none !important; animation: none !important; }
  html { scroll-behavior: auto; }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `node /tmp/pwverify/p1_polish.mjs`
Expected: `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
git add styles/system.css
git commit -m "feat(redesign): selection, placeholder, focus-visible, reduced-motion"
```

---

## Task 5: Confirm fonts load and isolation (no leakage to other pages)

**Files:**
- Test: `/tmp/pwverify/p1_isolation.mjs`

- [ ] **Step 1: Write the test**

```js
// /tmp/pwverify/p1_isolation.mjs
import { chromium } from '/tmp/pwverify/node_modules/playwright/index.js';
const b = await chromium.launch();
const ctx = await b.newContext({ viewport: { width: 1200, height: 900 } });
let fail = 0;
// (a) preview page actually renders in Inter
const p1 = await ctx.newPage();
await p1.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil: 'networkidle' });
const h1ff = await p1.evaluate(() => getComputedStyle(document.querySelector('h1')).fontFamily);
if (!/Inter/.test(h1ff)) { console.log(`FAIL preview not Inter: ${h1ff}`); fail++; }
// (b) an existing page is UNCHANGED: pca still has its old background (#131313 from base.css)
const p2 = await ctx.newPage();
await p2.goto('http://localhost:8000/pages/pca.html', { waitUntil: 'networkidle' });
const bodyBg = await p2.evaluate(() => getComputedStyle(document.body).backgroundColor);
if (bodyBg !== 'rgb(19, 19, 19)') { console.log(`FAIL pca background changed: ${bodyBg} (expected rgb(19,19,19))`); fail++; }
console.log(fail ? `${fail} FAIL` : 'ALL PASS');
await b.close();
process.exit(fail ? 1 : 0);
```

- [ ] **Step 2: Run the test**

Run: `node /tmp/pwverify/p1_isolation.mjs`
Expected: `ALL PASS` (preview is Inter; pca.html background is still `rgb(19, 19, 19)`, proving the new system did not leak because pca does not include tokens/system.css and has no `.ui` class).

- [ ] **Step 3: Screenshot the preview for a human check**

```bash
cat > /tmp/pwverify/p1_shot.mjs <<'EOF'
import { chromium } from '/tmp/pwverify/node_modules/playwright/index.js';
const b = await chromium.launch();
const p = await (await b.newContext({ viewport: { width: 1200, height: 900 } })).newPage();
await p.goto('http://localhost:8000/pages/_redesign-preview.html', { waitUntil: 'networkidle' });
await p.screenshot({ path: '/tmp/pwverify/p1_preview.png' });
await b.close();
EOF
node /tmp/pwverify/p1_shot.mjs
```
Confirm the screenshot matches the editorial type/colour of `docs/design/mockups/type-scale.html` (Balanced) and `neutrals-palette.html` (A).

- [ ] **Step 4: Commit (only if any fix was needed)**

```bash
git add -A && git commit -m "test(redesign): foundation isolation + font verification" || echo "nothing to commit"
```

---

## Task 6: Run the existing test suite (no regressions)

**Files:** none (verification only)

- [ ] **Step 1: Run the JS suite**

Run: `node --test 'test/**/*.test.js'`
Expected: `# pass 65`, `# fail 0` (Phase 1 adds only CSS + a hidden HTML page; the JS suite must be unaffected).

- [ ] **Step 2: Stop the server**

```bash
pkill -f "http.server 8000" 2>/dev/null || true
```

---

## Notes for the implementer

- Do NOT modify `styles/base.css` or any existing page in this phase. The opt-in `.ui` + new files keep the seven shipped pages and the home page visually untouched; Task 5 enforces that.
- `tokens.css` is safe to load anywhere (custom properties are inert). `system.css` only acts under `.ui`.
- Color equality in tests is on computed `rgb(...)` form: `#0c0c0d` -> `rgb(12, 12, 13)`, `#fafafa` -> `rgb(250, 250, 250)`, `#8b8d96` -> `rgb(139, 141, 150)`, `#5d5f68` -> `rgb(93, 95, 104)`, `#6b7cff` -> `rgb(107, 124, 255)`, `#9aa6ff` -> `rgb(154, 166, 255)`.
- Strict project rule: no em-dash characters anywhere in code, comments, or content.
- Later phases (own plans): `components.css` (controls, control-management patterns, figures, math wrapper, player, home index, tooltip, callout, table, outline-rail restyle), then risk-ordered page migration that adds `class="ui"` + the new stylesheets and removes per-page CSS, then the hidden settings page + theming boot script, then the final audit sweep.
