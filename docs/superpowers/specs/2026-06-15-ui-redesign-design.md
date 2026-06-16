# UI Redesign Design System Spec

**Date:** 2026-06-15
**Branch:** `redesign/ui-system`
**Visual source of truth:** `docs/design/mockups/*.html` (exact CSS; see that README for the chosen option per screen)
**Audit (current-state evidence):** see "Audit reference" at the end.

## Goal

Make the site visually consistent and professional by replacing the ad hoc per-page styling
with one shared, token-driven design system, and refreshing the visual language to a dark,
modern, minimal, editorial direction. The dark minimal identity is kept; typography, panels,
borders, and contrast are substantially revised. Implementation must be a faithful recreation
of the mockups and the values below, not a summarized approximation.

## How to read this spec

Every value below is normative. Where a mockup file is cited, the implementation must match it.
Hex colors, font sizes (px), line-heights, and letter-spacing are exact. "Token" means a CSS
custom property defined once on `:root` in `styles/base.css` and referenced everywhere.

---

## 1. Locked decisions (with the mockup that defines each)

1. Direction: Editorial. No boxed panels; sections separated by whitespace + hairline rules.
   Interactive figures carry their own edge. (`foundation-directions.html` A)
2. Framing: No frames now. Wrapping interactive regions in a quiet hairline frame is kept as a
   future option, not used initially. (`container-framing.html`, `fourier-frame-compare.html`)
3. Typeface: Inter (sans) + JetBrains Mono (mono), defined as substitutable tokens.
   (`typeface-options.html` 1)
4. Type scale: Balanced. (`type-scale.html` 2)
5. Neutrals: Neutral graphite (true grays, no temperature). (`neutrals-palette.html` A)
6. Accent: Periwinkle `#6b7cff`, default and user-customizable via the hidden settings page.
   (`accent-color.html` 1)
7. Controls: Minimal / underline. Visible control stays slim; touch hit-areas get enlarged
   (revisit on mobile). (`controls.html` A)
8. Home: Single-column editorial index. Revisit if the project count grows a lot.
   (`home-directory.html` A)
9. Display math: periwinkle left accent rule, equation left-aligned in the block, equation
   number flush right. (`math-ruled-numbered.html` 2)
10. Walkthrough player: Stacked. (`video-player.html` A)
11. Step navigation: Dropdown from the step title (desktop). Mobile revisit (likely inline
    expander). (`step-nav-approaches.html` 1)
12. Paired-visualization layout: controls below the figures. (`paired-viz-layout.html` C)
13. Dense control sets: tabbed groups as the default for busy pages; the three patterns
    (grouped grid, primary + more, tabbed) are a reusable set chosen per page, not mutually
    exclusive. Tabbed-group organization to be refined against each real file.
    (`manage-controls.html` C)

---

## 2. Tokens (`styles/base.css :root`)

### 2.1 Fonts (substitutable; single source)

```css
--font-sans: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
--font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, monospace;
```

- Load Inter (400,500,600,700) and JetBrains Mono (400,500,600) with `<link rel="preconnect">`
  to `fonts.gstatic.com` and `display=swap`. Self-hosting is acceptable later; the token is the
  only reference point so swapping is a one-line change.
- The font is referenced only via `var(--font-sans)` / `var(--font-mono)` everywhere. No element
  hardcodes a family. This is a hard requirement (the user wants the font ubiquitous and easily
  substituted).

### 2.2 Color / neutrals (graphite)

```css
--bg: #0c0c0d;              /* page background (was #131313) */
--text: #fafafa;           /* titles / primary */
--text-body: #8b8d96;      /* body copy */
--text-muted: #5d5f68;     /* captions, secondary, axis labels */
--hairline: rgba(255,255,255,0.08);   /* section rules, dividers, control underlines (.22 for active inputs) */
--surface: rgba(255,255,255,0.04);    /* raised inset (code / readout / worked block) */
--surface-border: rgba(255,255,255,0.06);
--surface-strong: #141418;            /* menus / dropdowns / tooltips */
--surface-strong-border: rgba(255,255,255,0.10);
```

Control underline borders use `rgba(255,255,255,0.22)`. Figure edges use `rgba(255,255,255,0.10)`.

### 2.3 Accent (user-overridable)

```css
--accent: #6b7cff;                         /* default; overridable via settings page */
--accent-muted: rgba(107,124,255,0.14);    /* chip bg, active item bg (use .16 for stronger) */
--accent-link: #9aa6ff;                    /* links (lighter tint of accent) */
--focus-ring: 2px solid rgba(107,124,255,0.60);
```

When the accent is changed at runtime, only `--accent` needs to change; `--accent-muted`,
`--accent-link`, and `--focus-ring` should derive from it with `color-mix(in srgb, var(--accent) N%, ...)`
where browser support allows, with the literal values above as the static fallback.

### 2.4 Warning / status (for callouts; from audit, not yet mocked)

```css
--warning: #e0b341; --warning-bg: rgba(224,179,65,0.10); --warning-text: #f0c460;
```

### 2.5 Radii

```css
--radius-xs: 3px; --radius-sm: 6px; --radius-md: 8px; --radius-lg: 10px;
--radius-xl: 12px; --radius-pill: 999px;
```

Usage: controls/menus 7-8px (`--radius-md`), figures 8-9px, dropdowns/tooltips 10px, large
surfaces 12-13px. Replace all raw-px radii found in the audit.

### 2.6 Spacing scale

```css
--sp-1:4px; --sp-2:8px; --sp-3:12px; --sp-4:16px; --sp-5:20px; --sp-6:24px; --sp-8:32px;
```

All padding/margin/gap use the scale. Section vertical rhythm: hairline rule with ~`--sp-6`
margins; page "slice" padding ~30-40px (desktop).

### 2.7 Motion

```css
--dur-instant:80ms; --dur-hover:140ms; --dur-state:200ms;
--ease-out: cubic-bezier(0,0,0.2,1); --ease-in: cubic-bezier(0.4,0,1,1);
```

All hover/state transitions use `--dur-hover`; the current 300ms input transition is removed.
Everything wrapped by `@media (prefers-reduced-motion: reduce)` disabling transitions and
smooth scroll.

### 2.8 Container widths

```css
--container-narrow:1100px; --container-default:1200px; --container-wide:1400px; --container-xl:1560px;
```

Page picks a semantic width (pca historically 1560, fourier 1400, manifold-isomap 1100).

---

## 3. Type scale (Balanced) and global text

All sizes in px, family `var(--font-sans)` unless noted. Source: `type-scale.html` (Balanced),
`accent-color.html`, `home-directory.html`.

| Role | Weight | Size | Line-height | Letter-spacing | Color |
|------|--------|------|-------------|----------------|-------|
| eyebrow | 600 | 11 | 1 | .2em, uppercase | `--accent` |
| h1 (page title) | 600 | 33 | 1.10 | -.025em | `--text` |
| h1 (home hero) | 600 | 40 | 1.05 | -.03em | `--text` |
| h2 (section) | 600 | 19 | 1.25 | -.014em | `#ededf0` |
| h3 (sub) | 600 | 15 | 1.35 | 0 | `#cfd1d8` |
| body | 400 | 14.5 | 1.7 | 0 | `--text-body` |
| caption (prose) | 400 | 12 | 1.45 | 0 | `--text-muted` |
| figure caption | 500 | 10.5 | 1.3 | .04em | `#7a7c84` (bold part `#bdc0c8`), `var(--font-mono)` |
| label (control) | 600 | 10.5 | 1 | .14em, uppercase | `#6c6e77` |

- Body copy max line length ~60-62ch.
- Eyebrow default is sans; a `var(--font-mono)` eyebrow (e.g. `// section`) is an allowed
  variant used on player/figure contexts. (Open: pick one convention site-wide or keep both.)
- Headings get `text-wrap: balance`; body paragraphs `text-wrap: pretty` (audit).
- Apply `-webkit-font-smoothing: antialiased` on the body.

### 3.1 Inline elements

- Link: `color: var(--accent-link); text-decoration: none; border-bottom: 1px solid rgba(154,166,255,0.40)`.
- Inline code: `var(--font-mono); color:#c9cdfb; background:var(--accent-muted-ish rgba(107,124,255,0.10)); padding:1px 5px; border-radius:var(--radius-xs..sm); font-size:.88em`.
- Emphasized term ("chip"): `color:#c9cdfb` (accent-tinted), no background in prose.
- Numeric readout: `var(--font-mono); font-variant-numeric: tabular-nums; color:#cfd1d8`, the
  highlighted number `color: var(--accent)`.
- `::selection` and `::placeholder` styled to the dark theme (audit: both currently unset).

---

## 4. Global layout and chrome

- Body: `background: var(--bg); color: var(--text-body); font-family: var(--font-sans); line-height:1.55;`
  base padding ~20px; centered `.container` at the page's chosen `--container-*`.
- Page header: eyebrow + h1 + lede (body, max ~50-58ch). A back-to-home affordance sits at the
  top of article pages (default: a small `← Home` in link style; final placement to confirm).
- Footer (`home-directory.html`): hairline top rule; left `Created by <a>Aughdon Breslin</a>`
  (`--text-muted`, link `--accent-link`); right a small mono meta line (e.g. `8 PROJECTS`).
- Section structure: vertical stack; sections separated by a 1px `var(--hairline)` rule with
  ~`--sp-6` margins; section heading is h2. No boxed panels.
- `section-outline` rail (already shipped): keep its behavior; restyle to consume the new tokens
  (`--accent`, `--hairline`, radii) instead of its own `#6ea8fe` / literals. (To finalize.)

---

## 5. Controls (minimal / underline): `controls.html` A

Shared component set; one definition each, no per-page reimplementation.

- Primary button: text; `color: var(--accent)`; `border-bottom: 1px solid var(--accent)`;
  `padding: 6px 1px`.
- Secondary button: text `#a9abb3`; `border-bottom: 1px solid rgba(255,255,255,0.22)`.
- Select / number / text input: transparent bg; `color:#dadbe0`;
  `border-bottom:1px solid rgba(255,255,255,0.22)`; `padding:6px 2px`. No `outline:none` without
  a focus replacement.
- Tabs: `color:#74767f; padding-bottom:6-7px; border-bottom:1.5px solid transparent`. Active:
  `color:#ececf0; border-bottom-color: var(--accent)`. Use `aria-selected` (one ARIA-native
  pattern site-wide; retire the `is-active` clones from the audit).
- No separate segmented control. Tabs are the single "pick one of N" control, used for both
  view switches (e.g. Fourier/Haar/DCT) and compact toggles (e.g. PDF/CDF).
- Slider: track `2px rgba(255,255,255,0.16)`; filled portion `var(--accent)` at ~.5 opacity;
  handle `11px` circle `var(--accent)`.
- Toggle: pill `30x17`; off `rgba(255,255,255,0.12)`; on `rgba(107,124,255,0.55)`; knob `13px #e9e9ee`.
- Checkbox: `15px` box, `1.5px solid var(--accent)`, accent fill inset when checked.
- Upload ("Choose file"): accent text + `border-bottom:1px solid var(--accent)`. (Reuses the
  iOS-safe pattern already on main: a `<label>` wrapping a visually-hidden, not `display:none`,
  file input.)
- Focus: every interactive element gets `:focus-visible { outline: var(--focus-ring); outline-offset:2px }`.

OPEN REFINEMENT (mobile hit areas): the visible control stays minimal, but the interactive
target is enlarged to >=44px via padding or an invisible `::before` overlay so touch targets
meet the touch/Fitts guideline. Finalize in a dedicated mobile pass.

---

## 6. Control-management patterns: `manage-controls.html`

A reusable set; each page selects the pattern that fits. Default for dense pages is Tabbed.

- Grouped grid: labeled groups (uppercase label with a 1px bottom rule) in
  `repeat(auto-fit, minmax(160px,1fr))`. Use when most controls are used often and there is room.
- Primary + more: primary controls in a row; a "More / Fewer options" disclosure (accent text
  button) reveals an advanced row separated by a 1px dashed rule. Use when there are clear
  primary vs secondary controls (PCA step-through).
- Tabbed groups: a tab row (e.g. Decompose / Filter / View) switches which control group shows.
  Default for busy pages (Fourier). Smallest footprint; one extra click to reach a group.
  The grouping/labels per page are an open refinement, tuned against the real file.

Controls sit below the figures (decision 12). On mobile, control regions go full-width single
column under the (stacked) figures.

---

## 7. Figures and paired-visualization layout: `paired-viz-layout.html` C

- Figure frame: `aspect-ratio` per content (4/3 default; 1/1 for square canvases; 16/9 for 3D);
  background `radial-gradient(120% 120% at 50% 45%, #0e0f15, #070709)` or flat `#050506`;
  `border:1px solid rgba(255,255,255,0.10)`; `border-radius:8-9px`; `overflow:hidden`.
- Figure caption below the frame: mono `10.5px #7a7c84`, bold label `#bdc0c8`.
- Paired figures: `grid-template-columns:1fr 1fr; gap:18px`. Controls BELOW, separated by a 1px
  hairline top rule with `padding-top:~18px`.
- Mobile: figures stack to a single column; controls follow.
- The same "controls below" rule is the default for the other paired/multi-figure pages
  (distributions PDF/CDF, the 3-up Fourier canvases), subject to per-file tuning.

CONSTRAINT (from audit, critical): pca/fourier/manifold/generative_classification/distributions
size canvases/plots from CSS. Do not add padding/border to the inner canvas containers without
re-checking the JS that reads their dimensions. The figure edge is on the frame, not the canvas.

---

## 8. Math: `math-ruled-numbered.html` (left-aligned) + `math-formula.html`

- Inline math: variables render in the MathJax math font (italic serif, near `Newsreader`),
  `color:#e4e6ea`; numbers/operators in tabular mono. Keep MathJax for actual rendering.
- Display math wrapper: `border-left:2px solid var(--accent); padding:15px 58px 15px 26px;
  margin:22px 0`. Equation LEFT-ALIGNED within the block. Equation number flush right, mono
  `12px var(--text-muted)`, e.g. `(6)`. Numbering is opt-in per equation.
- Worked numeric block (matrices / step values): `display:inline-flex; gap:18px;
  background:var(--surface); border:1px solid var(--surface-border); border-radius:9px;
  padding:14px 18px`. Matrix body tabular mono `#d2d4db` with bracket rules
  (`border-left/right:1.5px solid rgba(255,255,255,0.25)`); highlighted values in `var(--accent)`.

---

## 9. Walkthrough player (stacked): `video-player.html` A + `step-nav-approaches.html` 1

Order top-to-bottom: video, scrubber, step bar (with dropdown nav), transcript.

- Video frame: `aspect-ratio:16/9; border:1px solid rgba(255,255,255,0.10); border-radius:10px;
  overflow:hidden`. Title overlay top-left: mono chapter label + Inter `600 13` title. Center
  play button when paused: `58px` circle, `background:rgba(8,8,12,0.55); backdrop-filter:blur(4px);
  border:1px solid rgba(255,255,255,0.18)`, accent triangle.
- Scrubber: play icon (`30px` circle, hairline border, `rgba(255,255,255,0.04)` bg); track
  `4px rgba(255,255,255,0.12)` with `var(--accent)` fill, chapter ticks (`2px rgba(255,255,255,0.22)`),
  and a `12px` accent head; time mono tabular `#9a9ca4` formatted `MM:SS / MM:SS`.
- Step bar: hairline top rule. Left: mono `STEP n / N` in `var(--accent)` + step title (h2-ish,
  `600 16 #ededf0`). The step title is a dropdown trigger (chevron). The dropdown is a
  `--surface-strong` menu (`#141418`, border `rgba(255,255,255,0.10)`, `radius:10px`, layered
  shadow) listing all steps with states: done (`#9a9ca4`), current (`bg var(--accent-muted)`,
  label `#ededf0`, number `var(--accent)`), upcoming (`#74767f`); hover `rgba(255,255,255,0.04)`;
  click to jump. Prev/Next remain available.
- Transcript: body prose below.
- Mobile: revisit step navigation (dropdown is the desktop choice; mobile likely the inline
  expander from `step-nav-approaches.html` 2).

---

## 10. Home index (single column): `home-directory.html` A

- eyebrow + hero h1 (`600 40 / -.03em`) + intro (body `15`).
- Index rows: `border-top:1px solid var(--hairline)` (last row also bottom); `padding:17px 6px 17px 0`;
  layout = mono number (`500 12 #5d5f68`, width 22) + body (title `600 16 #d6d8df`, one-line
  description `400 13 #74767f`) + trailing arrow (mono `#5d5f68`).
- Hover/focus: title -> `#fafafa`, number + arrow -> `var(--accent)`, arrow `translateX(3px)`.
- Footer as in section 4. The whole row is the link target (enlarge hit area).
- Revisit if projects grow significantly (grouping by theme or two columns were mocked as
  alternatives in `home-directory.html`).

---

## 11. Content and composite components (finalized; mockups cited)

These were mocked and chosen. Exact CSS lives in the cited `docs/design/mockups/*.html`.

### 11.1 Callout / alert (`callouts.html`, treatment C: minimal left rule, no fill)

One `.callout` component, no background fill. Structure: a `2px` colored left rule +
`padding: 2px 0 2px 16px`, an uppercase label (`600 10.5px .12em`) in the variant color, then the
body (`400 13.5px/1.6 #9a9ca4`). Variants by token: note = `var(--accent)` (label `#9fb0ff`),
warning = `var(--warning)` (label `var(--warning-text)`), error = a red token (same shape). Matches
the no-frames math rule; differentiated from code by having no surface.

### 11.2 Code / pseudocode block (`code-blocks.html` + `code-in-context.html`, treatment C: dark inset)

`pre` on a darker-than-page inset: `background:#060607; border:1px solid rgba(255,255,255,0.08);
border-radius: var(--radius-lg); padding:16px 18px; overflow-x:auto`. Text `500 12.5px/1.85
var(--font-mono)`. Gutter line numbers (`.ln`, width ~20-22px, `#4d4f57`, `user-select:none`).
Light tinting: keywords `#9fb0ff`, identifiers `#cfd1d8`, comments `#5d5f68`, default `#9a9ca4`.
The dark inset deliberately reads as a distinct artifact, set apart from the left-ruled prose and
math.

### 11.3 Data table (`tables.html`, treatment A: row hairlines)

`border-collapse: collapse`. Header `th`: `600 10px .1em uppercase var(--font-mono) #6c6e77`,
`border-bottom: 1px solid var(--hairline-strong)`. Body `td`: `400 13px var(--font-sans) #b6b8c0`,
`border-bottom: 1px solid var(--hairline)` (last row none), `padding:11px 0`. Numeric columns use
`var(--font-mono)`, `font-variant-numeric: tabular-nums`, right-aligned; the first (label) column
is left-aligned `#dadbe0 500`. A highlighted cell/row (e.g. best metric) uses `var(--accent)`.
No vertical borders, no zebra. Replaces the base `data-table` (`#ccc`, `12px`).

### 11.4 Tooltip (`tooltips.html`, content-driven: card A or pill B)

One component with two layouts chosen by content:
- Card (A) when points are labelled, carry multiple values, or hold explanatory text (e.g. the
  manifold param tooltips): `background:#141418; border:1px solid rgba(255,255,255,0.10);
  border-radius: var(--radius-md); padding:9px 11px; box-shadow: layered (inset highlight +
  `0 10px 28px -10px rgba(0,0,0,0.8)`)`. Title `600 11px #ededf0`; rows are `label`(`#7a7c84`) /
  `value`(mono `var(--accent)`); small arrow.
- Pill (B) when a single simple value: `background:#06070a; border:1px solid rgba(255,255,255,0.06);
  border-radius: var(--radius-sm); padding:6px 9px`; one line of tabular mono `#cfd1d8` with the
  key value in `var(--accent)`; no arrow.

Replaces the four tooltip implementations in the audit.

### 11.5 Section-outline rail re-skin (`outline-rail.html`, treatment B: mono-numbered)

Keep the shipped rail's behavior (sticky, scrollspy, click-to-jump, hamburger drawer on mobile,
aligned with the first panel); re-skin to the new tokens. Each item: a mono number (`500 10.5px
var(--font-mono) #4d4f57`) + label (`500 13px var(--font-sans) #74767f`), `padding:6px 10px`,
`border-left: 2px solid transparent`. Active: label `#fafafa 600`, number `var(--accent)`,
`border-left-color: var(--accent)`, `background: var(--accent-muted)` (the `0.08` tint). Hover:
label `#cfd1d8`, faint `rgba(255,255,255,0.03)` bg. The numbers echo the home index and the
walkthrough step numbers, tying the system together. Replaces the rail's own `#6ea8fe`/literals.

NOTE: section titles should be shortened for succinctness during page migration to suit the
numbered rail and the home index.

## 11b. Still to finalize

- Hidden settings page shell (see section 12).
- Mobile pass: control hit areas, step-nav on mobile, paired-figure stacking, dense-control
  patterns on small screens.

---

## 12. Settings page and theming architecture

- All themeable values are CSS custom properties on `:root` in `base.css`.
- User-overridable preferences (at minimum accent color, and font per the substitutability
  requirement) are applied by rewriting the corresponding token on `:root`. Preferences persist
  in `localStorage`; a small inline boot script applies saved values before first paint to avoid
  a flash of default theme.
- The settings page is hidden for now (not linked in nav). Default accent is periwinkle
  `#6b7cff`; default fonts Inter + JetBrains Mono.
- Process rule: do not add any specific setting to this page without checking with the user first.

---

## 13. Implementation approach (phased; each phase its own plan + sign-off)

System-first, then migrate pages, matching the user's chosen sequencing.

1. Foundation: `base.css` tokens (sections 2-3), global text/links/selection/placeholder,
   reduced-motion, type scale. Validate on one pilot page.
2. Shared components: a `styles/components.css` consolidating controls, control-management
   patterns, figures, math wrapper, tooltip, callout, table, player, home index; retire the
   per-page clones identified in the audit.
3. Finalize-and-mock the section 11 components as needed, with sign-off.
4. Migrate pages, risk-ordered: content pages first (estimation, bayesian, regularization),
   then canvas/plot pages (pca, fourier, manifold, generative_classification, distributions)
   with render verification each (respect the section 7 constraint).
5. Hidden settings page + theming boot script.
6. Final consistency + accessibility sweep (re-run the userinterface-wiki audit).

Implementation must follow the mockups and the exact values here. Where a real file forces a
deviation (e.g. a dense control set), use the section 6 patterns and flag the deviation rather
than silently summarizing.

### Phase-1 outcome and a migration constraint (recorded during execution)

Phase 1 shipped the foundation as an opt-in layer rather than editing `base.css`: new
`styles/tokens.css` (the `:root` tokens) and `styles/system.css` (global styling scoped under a
`.ui` body class), validated on the hidden `pages/_redesign-preview.html`. The eight existing
pages are untouched (an isolation test enforces this).

CONSTRAINT for Phase 4 migration: `base.css` still defines its own `:root` with overlapping token
names but old values (`--bg:#131313`, `--text-muted: rgba(255,255,255,0.75)`, some `--radius-*`).
Same-specificity `:root` rules resolve by load order, so a page that opts in MUST load
`tokens.css` after `base.css`, or the old values silently win and undo the redesign. The migration
plan should either enforce that include order on every migrated page or remove the overlapping
tokens from `base.css`. Also confirm during migration that `--bg` reaches `html`/`body` (not only
the `.ui` block) so scroll overflow does not reveal the old background.

---

## 14. Open refinements (consolidated revisit list)

1. Framing: frames kept as a future option (currently no-frames).
2. Mobile control hit areas (enlarge invisible target to >=44px).
3. Mobile step navigation (dropdown is desktop; mobile likely inline expander).
4. Tabbed control-group organization, tuned per real file.
5. Per-page control-management pattern choice (grouped / primary+more / tabbed).
6. Home index growth (grouping or two columns if many projects).
7. Finalize tooltips, tables, callouts, code blocks, section-outline rail, settings shell.
8. Eyebrow convention: sans vs a mono variant, site-wide.
9. Back-to-home affordance placement on article pages.

---

## Audit reference

The current-state audit (run with the `userinterface-wiki` skill) found: no h2/h3 type scale;
~26 distinct font sizes; accent hardcoded in ~7 places as two blues; tabs reimplemented 3x;
tooltips 4x; warning alert 2x; control grids duplicated (pca/bayesian); ~15 raw-px radii;
input transition 300ms vs button 140ms; inconsistent focus rings; magic-number container widths;
`#ccc`/`12px` in the base data-table; sub-32px touch targets; missing tabular-nums, text-wrap,
`::selection`/`::placeholder`, and reduced-motion guards. The token system and component
consolidation above are the direct response. Heavy-JS pages (pca, fourier, manifold,
generative_classification, distributions) drive canvas/plot sizing from CSS and need render
verification on migration.
