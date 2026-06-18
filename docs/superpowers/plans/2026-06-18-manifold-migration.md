# Manifold Page Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax.

**Goal:** Migrate `pages/manifold.html` onto the `.ui` archetype, keeping both interactive systems
(the Isomap video explainer and the algorithm sandbox) working with no JS changes.

**Architecture:** Opt-in `<body class="ui manifold">`. Rewrite `styles/manifold.css` and
`styles/manifold_isomap.css` scoped under `.ui.manifold`, translating legacy tokens / raw values to
the archetype system. All element ids, data-attributes, and JS-built class hooks are preserved.

**Tech Stack:** Static HTML/CSS, d3 v7, MathJax tex-svg, ES module JS under `js/manifold/` plus
`js/manifold_isomap.js`, the archetype CSS, section-outline.js.

Spec: `docs/superpowers/specs/2026-06-18-manifold-migration-design.md`. Read it (especially the id /
class-hook list) before starting.

**Process note:** Verify visually with the Playwright harness at `/tmp/pwverify`; serve the repo on
`http://localhost:8000`. Run the Task 0 guards before each commit.

---

### Task 0: Guards (reference)

**Dash + emphasis:**
```bash
cd /mnt/c/Users/aughb/PersonalProjects/Visualization
grep -rlP "[\x{2014}\x{2013}]" pages/manifold.html styles/manifold.css styles/manifold_isomap.css 2>/dev/null
grep -nE "<(em|strong)[ >]" pages/manifold.html 2>/dev/null
```

**Id-presence** (each must print 1):
```bash
for id in mfiAlgoSel mfiDatasetSel mfiFsWrap mfiStage mfiVideo mfiBigPlay mfiFsHint mfiOverlay \
  mfiProgress mfiProgressFill mfiProgressMarks mfiPlay mfiPrev mfiNext mfiTime mfiSpeed mfiFull \
  mfiSteps mfiTranscript mfDataset mfSamples mfNoise mfSeed mfReseed mfDatasetParams mfCsvInput \
  mfCsvLabel mfAlgoStepPanel mfAlgoLeft mfAlgoRight mfAlgoLeftParams mfAlgoRightParams mfLeftViz \
  mfRightViz mfLeftTitle mfRightTitle mfLeftIfw mfRightIfw mfLeftPseudo mfRightPseudo; do
  c=$(grep -c "id=\"$id\"" pages/manifold.html); [ "$c" = "1" ] || echo "BAD id=$id count=$c";
done; echo "ids checked"
```

**Token-mapping table** (apply in both CSS files):

| Legacy | Archetype |
| --- | --- |
| `var(--accent, #4aa3ff)` / `#4aa3ff` | `var(--accent)` |
| `#ff9f43` (orange info/hover) | `var(--accent)` (hover), info glyph idle `var(--text-muted)` |
| `var(--surface-inset)` / card `rgba(0,0,0,.25..)` | `var(--surface)`; deep wells -> `#060607` |
| `rgba(255,255,255,.08)`..`.18` borders / `var(--border-light)` | `var(--hairline)` (subtle) / `var(--hairline-strong)` (stronger) |
| `rgba(255,255,255,.9)`..`.85` text | `#e7e8ec` / `var(--text-body)` |
| `var(--text-muted)` / `rgba(255,255,255,.55..6)` | `var(--text-muted)` |
| active bg `rgba(74,163,255,.18)` + `rgba(74,163,255,.7)` | `var(--accent-muted)` bg + `var(--accent)` underline/border |
| filled dot `rgba(255,255,255,.92)` | `#e7e8ec`; hollow border `var(--hairline-strong)`; na `rgba(255,255,255,.22)` |

Preserve every selector, geometry value (sizes, positions, masks, keyframes), media query, and
`:fullscreen` rule. Only colors / backgrounds / borders / fonts change to archetype tokens.

---

### Task 1: HTML shell, sections, footer

**Files:** Modify `pages/manifold.html`.

- [ ] **Step 1: Replace the head asset block** with the archetype set (Spec "Head and body"),
  keeping d3, the MathJax config + tex-svg.js, `js/manifold/main.js` (module),
  `js/manifold_isomap.js` (module), favicon.js, section-outline.js. Add tokens/system/components/
  article-ui + theme.js. Drop base.css, article.css, responsive.css, collapsible.js.

- [ ] **Step 2: Replace `<body>` + header** with `<body class="ui manifold">`, `.container`,
  `.page-head` (eyebrow `// Machine learning`, h1 `Manifold Learning`, lede
  `Step-by-step comparison of two algorithms on a shared dataset.`), open `<main class="article-body">`.
  Remove the old `subtitle` and `home-link`.

- [ ] **Step 3: De-collapsible the sections.** On each `<section>`, drop `collapsible` and
  `data-open-mobile`; keep the section id (`mfAlgoStepPanel`) and the `mf-isomap` / `sp-frame`
  classes. Keep ALL inner markup, ids, data-attributes, and option lists exactly.

- [ ] **Step 4: Replace the footer** with the archetype `site-footer` / `credit` markup.

- [ ] **Step 5: Guards.** Run the Task 0 dash + id-presence guards (all ids count 1, no dashes).
  Load the page; confirm sections render and (even if unstyled-legacy) the video and selects
  appear. Console may warn about missing styles until Tasks 2-3; that is fine.

- [ ] **Step 6: Commit.**
```bash
git add pages/manifold.html
git commit -m "redesign: migrate manifold head, shell, and sections to the .ui archetype"
```

---

### Task 2: Rewrite `styles/manifold_isomap.css` (video player) to archetype tokens

**Files:** Rewrite `styles/manifold_isomap.css`.

- [ ] **Step 1: Translate the file.** Scope every rule under `.ui.manifold` (e.g.
  `.ui.manifold .mfi-stage { ... }`; the floating `.mf-tooltip` and `:fullscreen` wrapper rules
  stay effectively global/scoped as today but prefix selectors with `.ui.manifold` where they
  target page elements). Apply the Task 0 token-mapping table. Specifically:
  - `.mfi-progress-fill` background -> `var(--accent)`.
  - `.mfi-stage`, `.mfi-video` keep geometry; border -> `var(--hairline)`; bg `#000` stays (video).
  - `.mfi-bigplay`, `.mfi-overlay` scrim, `.mfi-ctrl` hover, `.mfi-ico-*` glyphs: keep geometry;
    keep white-on-video controls (they sit over the black video, so `#fff` / translucent white is
    correct) but the focus ring -> `var(--focus-ring)` / accent.
  - `.mfi-speed` select -> archetype field (subtle bg, hairline border, accent focus).
  - `.mfi-steps li` -> hairline border + `var(--text-muted)`; `.is-active` -> `var(--accent-muted)`
    bg + `var(--accent)` inset underline + `#fff`.
  - `.mfi-transcript` -> `var(--surface)` bg, `var(--radius-md)`; `.mfi-caption` -> `var(--text)`;
    `.mfi-formula` / `.mfi-explain` -> `var(--text-muted)`.
  - `.mfi-pick-sel` -> archetype field; `.mfi-pick` label -> `var(--text-muted)`.
  - Keep all `:fullscreen`, `@keyframes mfiHintBob`, and `@media (max-width: 560px)` rules,
    re-scoped.
  Preserve the structural comments (they explain non-obvious behavior).

- [ ] **Step 2: Guards.** Dash guard over the file: no output. Confirm no `#4aa3ff` or `#ff9f43`
  remain: `grep -nE "4aa3ff|ff9f43" styles/manifold_isomap.css` prints nothing.

- [ ] **Step 3: Visual check.** Load the page; the player renders in the dark archetype: dark
  stage, periwinkle progress fill, legible controls, accent-highlighted active step, archetype
  transcript card. Big-play + play/seek/prev/next/speed work.

- [ ] **Step 4: Commit.**
```bash
git add styles/manifold_isomap.css
git commit -m "redesign: rewrite the manifold Isomap player styles onto the .ui archetype"
```

---

### Task 3: Rewrite `styles/manifold.css` (sandbox) to archetype tokens

**Files:** Rewrite `styles/manifold.css`.

- [ ] **Step 1: Translate the file.** Scope every page-element rule under `.ui.manifold`; the
  floating `.mf-tooltip` rules stay as today (prefixed where they target page content). Apply the
  Task 0 mapping. Specifically:
  - `.mf-control` inputs/selects -> archetype field look; `.mf-controls-row` layout unchanged.
  - `.mf-algo-card`, `.sp-frame`, `.sp-panel`, `.mf-viz-card`, `.mf-ifw-card`, `.mf-pseudo-card`
    -> `var(--surface)` bg + `var(--hairline)` border + `var(--radius-md)`.
  - `.mf-param-info` -> idle `var(--text-muted)` hairline circle, hover `var(--accent)`
    (was orange).
  - `.mf-param-control input/select` -> archetype field; keep the spinner-hiding rules.
  - Step indicator: `.sp-dot.filled` -> `#e7e8ec`; `.hollow` border -> `var(--hairline-strong)`;
    `.na` -> `rgba(255,255,255,.22)`; `.sp-edge::before` line -> `var(--hairline-strong)`, hover
    `var(--accent)`; `.sp-step.current` -> `var(--accent-muted)` bg.
  - `.ifw-tab` -> hairline tab; `.is-active` -> `var(--accent-muted)` bg + `var(--accent)` border;
    `.ifw-content pre`, `.ifw-worked-body` -> `#060607` mono wells with hairline / accent left rule.
  - `.pseudocode` / `.pc-section` / headers / `.pc-line` -> archetype surfaces + hairlines;
    `.pc-section.is-current` -> accent-tinged border + `var(--surface)`.
  - `.viz-host`, `.viz3d-thumb`, `.viz-loading` -> `#060607` / `var(--surface-strong)` wells with
    hairline borders.
  - Keep both `@media (max-width: 820px)` single-column rules (including the trailing override).

- [ ] **Step 2: Guards.** Dash guard: none. `grep -nE "4aa3ff|ff9f43" styles/manifold.css` prints
  nothing.

- [ ] **Step 3: Visual check.** The Dataset, Algorithms (sp-panels + indicator), Visualization,
  Step notes, and Pseudocode sections render in the archetype: surface cards, hairline borders,
  periwinkle accents on active/selected states, archetype fields. The step indicator dots, the
  prev/next nav, the ifw tabs, and the pseudocode sections all read correctly.

- [ ] **Step 4: Commit.**
```bash
git add styles/manifold.css
git commit -m "redesign: rewrite the manifold sandbox styles onto the .ui archetype"
```

---

### Task 4: Verification, isolation, backlog + memory

**Files:** Modify `docs/design/redesign-backlog.md` and the `project_redesign_backlog` memory.

- [ ] **Step 1: Behavior pass** (desktop): video plays; big-play / seek / prev / next / speed /
  fullscreen work; step list highlights + transcript updates on section change; sandbox dataset /
  samples / noise / seed / reseed update the viz; algorithm selects + params work; the step
  indicator + prev/next advance the comparison; both viz hosts render; ifw tabs switch; pseudocode
  sections expand. No console errors.

- [ ] **Step 2: Responsive screenshots** at ~1280px, ~760px (panels stack), ~440px. No clipping or
  horizontal scroll. Capture the player, the sp-panels, and the viz row.

- [ ] **Step 3: Isolation.** Load `pages/distributions.html` (un-migrated) and confirm no visual
  change (rules are `.ui.manifold` scoped, except the two floating tooltips which only appear on
  manifold interactions).

- [ ] **Step 4: Final guards.** Dash + id-presence over the final files.

- [ ] **Step 5: Backlog.** In `docs/design/redesign-backlog.md`, move manifold to Done:
  `Done: estimation (pilot), bayesian, regularization, pca, fourier, generative_classification,
  manifold. Next: distributions.`

- [ ] **Step 6: Memory.** Update `project_redesign_backlog` migration status to add manifold.

- [ ] **Step 7: Commit.**
```bash
git add docs/design/redesign-backlog.md
git commit -m "docs: mark manifold migration done in the backlog"
```

---

## Self-review notes

- Every JS-read id and JS-built class hook is preserved (Task 1 keeps inner markup verbatim;
  Task 0 guards the ids). No `js/manifold*` file is edited.
- The two CSS rewrites are pure token translations: geometry, masks, keyframes, media queries, and
  `:fullscreen` behavior are unchanged; only colors / surfaces / borders / fields move to archetype
  tokens.
- The white-on-video player controls stay white (they sit over the black video); only the accent
  (progress fill), focus rings, and the off-video chrome (steps, transcript, pickers, speed select)
  adopt archetype tokens.
- Footer (`site-footer` / `credit`) and the archetype field / tab / surface treatments reuse the
  verified primitives from the prior migrations.
- Deferred per backlog: content edits, robustness sweep, full mobile/device pass, manim rebuilds.
