# Manifold page migration design

Migrate `pages/manifold.html` onto the `.ui` article-page archetype. Phase 4 page migration:
structural and visual only. The page has two interactive systems that keep their behavior:
the Isomap video explainer (`mfi*` ids, driven by `js/manifold_isomap.js`) and the side-by-side
algorithm sandbox (`mf*` ids and the `sp-*` / `ifw-*` / `pc-*` widgets, driven by the
`js/manifold/` ES modules). No JS is edited.

## Goal

The page renders in the dark archetype with the section-outline rail (Home at position 0). The
six sections map onto `.panel`s. The video player, the step indicator, the comparison cards, the
two viz hosts, the step-notes, and the pseudocode all keep working; only their styling moves to
the archetype tokens (periwinkle accent, hairlines, archetype surfaces and fields).

## Constraints (carry through every task)

- No em-dashes or en-dashes (U+2014 / U+2013) anywhere. Grep before each commit.
- No `<em>` / `<strong>` and no markdown emphasis in page content.
- Preserve every element id, the `data-side` / `data-tab` / `data-open-mobile` attributes, and the
  class hooks the JS reads or builds. The JS is NOT edited. Player ids (read by
  `js/manifold_isomap.js`): `mfiAlgoSel`, `mfiDatasetSel`, `mfiFsWrap`, `mfiStage`, `mfiVideo`,
  `mfiBigPlay`, `mfiFsHint`, `mfiOverlay`, `mfiProgress`, `mfiProgressFill`, `mfiProgressMarks`,
  `mfiControls` content, `mfiPlay`, `mfiPrev`, `mfiNext`, `mfiTime`, `mfiSpeed`, `mfiFull`,
  `mfiSteps`, `mfiTranscript`. Sandbox ids (read by the `js/manifold/` modules, often via
  `d3.select('#...')`): `mfDataset`, `mfSamples`, `mfNoise`, `mfSeed`, `mfReseed`,
  `mfDatasetParams`, `mfCsvInput`, `mfCsvLabel`, `mfAlgoLeft`, `mfAlgoRight`, `mfAlgoLeftParams`,
  `mfAlgoRightParams`, `mfAlgoStepPanel`, `mfLeftViz`, `mfRightViz`, `mfLeftTitle`, `mfRightTitle`,
  `mfLeftIfw`, `mfRightIfw`, `mfLeftPseudo`, `mfRightPseudo`. JS-built class hooks to keep as style
  targets (do not rename): `sp-frame`, `sp-panels`, `sp-panel`, `sp-panel-header`, `sp-bar`,
  `sp-detail`, `sp-cell`, `sp-dot`, `sp-num`, `sp-edge`, `sp-step`, `sp-step-dot`, `sp-nav`,
  `step-prev`, `step-next`, `step-desc`, `mf-algo-card`, `mf-algo-select`, `mf-param-grid`,
  `mf-param-label`, `mf-param-name`, `mf-param-info`, `mf-param-control`, `mf-noparams`,
  `mf-viz-host`, `viz3d` / `viz2d` / `viz3d-thumb` / `viz-knn` / `viz-centering` / `viz-spectral` /
  `viz-weighted-knn` / `viz-matrix-strip` / `viz-loading`, `ifw`, `ifw-tabs`, `ifw-tab`,
  `ifw-content`, `ifw-empty`, `ifw-worked-section` / `-label` / `-body`, `pseudocode`,
  `pseudocode-title`, `pc-section`, `pc-section-header`, `pc-chevron`, `pc-section-title`,
  `pc-section-steps`, `pc-section-body`, `pc-line`, `mfi-steps li.is-active`,
  `mfi-transcript .mfi-caption` / `.mfi-formula` / `.mfi-explain`, `mfi-stage.is-playing` /
  `.is-hidden-ui`, `mfi-progress-mark`, `mfi-ico-*`, `mfi-ico-full.is-fs`, `mf-tooltip` (+ title /
  range), `mf-csv-name`, `mfi-pickers` / `mfi-pick` / `mfi-pick-sel`, `mfi-datasetnote`.
- Opt-in via `<body class="ui manifold">`. All visual rules scoped under `.ui.manifold`
  (or `.ui`); never bare element selectors. Must not affect any un-migrated page.

## Files

- Modify: `pages/manifold.html` (head swap, body class, shell, de-collapsible, footer).
- Rewrite: `styles/manifold.css` (sandbox) and `styles/manifold_isomap.css` (video player) to
  `.ui.manifold`-scoped archetype styling.
- Not edited: any `js/manifold*` file.
- Update: `docs/design/redesign-backlog.md` + the `project_redesign_backlog` memory.

## Head and body

Swap the legacy asset block for the archetype set (keep d3, the MathJax config + tex-svg.js, and
all the manifold module scripts in the same order):

```html
<link rel="stylesheet" href="../styles/tokens.css">
<link rel="stylesheet" href="../styles/system.css">
<link rel="stylesheet" href="../styles/components.css">
<link rel="stylesheet" href="../styles/article-ui.css">
<link rel="stylesheet" href="../styles/manifold.css">
<link rel="stylesheet" href="../styles/manifold_isomap.css">
<link rel="stylesheet" href="../styles/section-outline.css">
<script src="../js/theme.js"></script>
... keep: d3 (defer), MathJax config + tex-svg.js (defer),
    <script type="module" src="../js/manifold/main.js"></script>,
    <script type="module" src="../js/manifold_isomap.js"></script>,
    favicon.js (defer), section-outline.js (module) ...
```

Drop `base.css`, `article.css`, `responsive.css`, and `collapsible.js`.

Body:

```html
<body class="ui manifold">
  <div class="container">
    <header class="page-head">
      <div class="eyebrow">// Machine learning</div>
      <h1>Manifold Learning</h1>
      <p class="lede">Step-by-step comparison of two algorithms on a shared dataset.</p>
    </header>
    <main class="article-body">
      ... six sections ...
    </main>
  </div>
</body>
```

Drop the in-body `subtitle` and `home-link`. Replace the legacy footer with the archetype
`site-footer` / `credit` markup. Each `<section class="... collapsible ...">` becomes
`<section class="...">` (drop `collapsible` and `data-open-mobile`); keep section ids
(`mfAlgoStepPanel`) and the `mf-isomap` and `sp-frame` classes (CSS/JS hooks). Keep all inner
markup and ids exactly.

## styles rewrite (the bulk of the work)

Rewrite both CSS files scoped under `.ui.manifold`, translating the legacy tokens / raw values to
the archetype system. The structure, layout, and component behavior stay identical; only the
visual tokens change. Mapping rules:

- Accent: the player already uses `var(--accent)`; keep it (now periwinkle from tokens.css). The
  orange param-info / matrix hover (`#ff9f43`) becomes the periwinkle accent
  (`var(--accent)` / `var(--accent-link)`) for consistency.
- Surfaces: `var(--surface-inset)` and the various `rgba(0,0,0,.25..)` / `rgba(255,255,255,.03..)`
  card backgrounds become `var(--surface)` / `#060607` for inset wells, matching the other
  migrated pages.
- Borders: `var(--border-light)` and the `rgba(255,255,255,.08..)` hairlines become
  `var(--hairline)` / `var(--hairline-strong)`.
- Text: `var(--text-muted)` / `rgba(255,255,255,.85)` etc. map to `var(--text)`,
  `var(--text-body)`, `var(--text-muted)`.
- Radii: keep `var(--radius-md)` / `var(--radius-sm)` (defined in tokens.css).
- Selects, number inputs, and the player speed select adopt the archetype field look (the demo
  control fields): subtle background, hairline border, accent focus ring; hide native number
  spinners (the sandbox params already do).
- The active-state highlights (`mfi-steps li.is-active`, `ifw-tab.is-active`, `sp-step.current`,
  `pc-section.is-current`) use the accent-muted background + accent underline/border, matching the
  archetype tab/selected treatment.
- The step-indicator dots (`sp-dot.filled` / `.hollow` / `.na`), the progress marks, and the
  big-play affordance keep their geometry; recolor to the archetype neutrals / accent.
- The two floating tooltips (`mf-tooltip`, and reuse of the fourier-style tip pattern) restyle to
  `var(--surface-strong)` + hairline, accent on hover, like the migrated fourier `.fourier-tip`.
- Keep all responsive rules (the `@media (max-width: 820px)` single-column stacks, the
  `@media (max-width: 560px)` control-bar tightening, and the `:fullscreen` player rules) intact,
  re-scoped under `.ui.manifold`.

The page content width follows the archetype cap (`.ui.has-section-outline .container`
max-width 950px); drop the legacy `.mf-isomap.container { max-width: 1100px }` (it never matched a
real element anyway). The dual-panel comparisons render ~440px per panel, which is fine; they
already collapse to one column at <=820px.

## Sections (unchanged structure)

1. Algorithm walkthrough (`mf-isomap`): pickers (`mfi-pickers`), the video player
   (`mfi-stage` + `mfi-overlay` controls + `mfi-bigplay`), the step list (`mfi-steps`), and the
   transcript (`mfi-transcript`). Restyle only.
2. Dataset: `mf-controls-row` controls (dataset, samples, noise, seed, reseed), `mfDatasetParams`,
   the hidden CSV input + `mfCsvLabel`.
3. Algorithms (`sp-frame` / `mfAlgoStepPanel`): two `sp-panel` cards (algorithm select, params,
   `sp-bar` step indicator, `sp-detail`), and `sp-nav` (prev / desc / next).
4. Visualization: two `mf-viz-card`s hosting `mfLeftViz` / `mfRightViz`.
5. Step notes: two `mf-ifw-card`s (`mfLeftIfw` / `mfRightIfw`).
6. Full pseudocode: two `mf-pseudo-card`s (`mfLeftPseudo` / `mfRightPseudo`).

## Out of scope (deferred per backlog)

- Content edits, title shortening: content pass.
- Input-range / width robustness sweep: stability pass.
- Full mobile pass (the player has bespoke mobile + fullscreen behavior; a real device check is
  deferred).
- Rebuilding the manim clips or the explainer logic: not part of this migration.

## Verification

- Load the page: rail builds with Home at 0; the video loads and plays; the big-play, progress
  seek, prev/next, speed, and fullscreen controls work; the step list highlights and the
  transcript updates; the sandbox dataset/algorithm selects, params, step indicator, and
  prev/next work; both viz hosts render; the step-notes tabs and pseudocode sections render. No
  console errors.
- Playwright screenshots at desktop (rail, player, dual panels), tablet (~760px, panels stack),
  narrow (~440px). No clipping or horizontal scroll.
- Isolation: load an un-migrated page (distributions) and confirm no visual change.
- Dash + `<em>`/`<strong>` scans, and an id-presence check for the full id list, before commit.
