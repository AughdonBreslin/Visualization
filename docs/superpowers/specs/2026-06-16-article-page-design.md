# Article Page Archetype: Design

Date: 2026-06-16
Branch: redesign/ui-system
Status: approved (design); implementation pending

## Purpose

The estimation migration (Phase 4 pilot) revealed that the article/content page needs a
deliberate page-level design, not just migrated components. This spec defines the article
page archetype: the layout, header, section rhythm, math, in-context demos, link treatment,
and the rail behavior. It is the reference all content pages (estimation, bayesian,
regularization, and the canvas pages once migrated) will follow.

The visual reference is the mockup at `.superpowers/mocks/article-page-v3.html`. Colors, type
tokens, and components come from the existing system (`styles/tokens.css`, `styles/system.css`,
`styles/components.css`, `styles/article-ui.css`).

## Scope

In scope: the article page composition and the link/definition treatment, plus one new
settings token (link underlines). Out of scope and deferred: the definition-page / glossary
content system itself (this spec settles only how a definition link looks and behaves; the
pages it points to are a separate follow-on). The running glossary of terms lives at
`docs/design/glossary.md`.

## Layout and rail

- The page is a centered group of two columns: the section-outline rail (about 210px) and the
  content column. The container uses the wide max-width. The rail is glued to the content's
  left edge so the two read as one group (already implemented in `js/section-outline.js`).
- The rail is the on-page section outline. Each entry is a mono two-digit number plus the
  section title (`01 Overview`, `02 Bias and variance`). The active entry carries the accent
  left border and the accent-muted background; its number is in the accent color.
- Body paragraphs and lists run the full content width, out to the section spacer lines (no
  prose max-width cap on article sections). The content column itself is capped (about 950px) and
  centered with the rail as one group, so the line length stays bounded by the column.

### Rail responsive behavior

- The Home link lives at the top of the rail (position 0) with a back-arrow icon instead of a
  mono number; it is not in the page header. The rail is offset so the first section entry lines
  up with its section header at the top of the page.
- Desktop wide (>=1240px): the rail is shown.
- Every width below 1240px (the thin desktop band and mobile): the outline becomes a bottom-left
  hamburger drawer that carries Home plus the section outline, so both stay reachable in the thin
  band rather than disappearing. One mechanism across all non-wide widths.

## Header and identity

The header, top to bottom (Home is not here; it lives at the top of the rail, see above):

1. Eyebrow: the page category in mono, uppercase, letter-spaced, accent color (for example
   `// Statistics`).
2. Title: the page name as the h1 (about 40px, tight tracking).
3. Lede: one or two sentences summarizing the page, body color.

There is no meta row (no section count, interactive count, or read-time estimate). It was
considered and rejected.

## Section rhythm

- Each section opens with a header that is the mono section number next to the section title
  (`01` then `Overview`). The number uses the same numbering as the rail, so the rail and the
  page share one system.
- Sections are separated by a hairline top border with generous padding and margin above. The
  first section has no top border.
- Section titles stay succinct (the agreed revision to shorten long titles applies here).

## Math

- All math renders via real LaTeX (MathJax), inline and display.
- Inline math is standard MathJax inline. Variables render italic because that is standard math
  typography, not emphasis.
- Display math keeps the sizing from the mockup. Every `.formula` is a left-ruled block (accent
  wall). Display equations are not numbered (nothing on a notes page cites them by number); a
  `.formulas` wrapper holds one or more side-by-side results, the first keeping the accent wall
  and the rest aligned but wall-less. Numbering can be reintroduced opt-in later if a page ever
  needs cross-references.
- Formula labels such as `Point:` and `Function:` are regular upright text (`\text{}`), not
  bold.

## Demos in context

The approved pattern for an embedded interactive (Option A, decided 2026-06-17):

- Plots flow two-up. A `.demo-figs` auto-fit grid lays figures side by side; one plot fills the
  row, two sit side by side, collapsing to a single column below ~720px. Each figure carries its
  own mono caption above the framed dark-inset plot, and its help text below.
- Controls form a horizontal grouped bar below the plots, separated by a hairline: control groups
  are `.demo-col` columns (and a `.field-row` of `.control-item` cells for tabbed fields), so the
  controls flow across the width instead of stacking into a tall vertical list. Uppercase labels,
  underline-style inputs and selects.
- Results readout: a boxed readout column at the right for structured metrics (for example the
  bootstrap demo, mono tabular numbers with the primary estimate in the accent color); for prose
  summaries (the bayesian demos) it drops to a full-width band below the controls, capped for
  readability.

## Links and definitions

Two link types, both with underlines off by default:

- Definition link: marks a term that has a definition. It blends into the prose at body color
  with no underline. On hover it shifts to the accent color and shows a dotted underline. It
  points to that term's definition page (the page system is deferred; see Scope).
- Cross-reference link: an ordinary navigational link to another page or section. It shows in
  the accent-link color (so it is findable) with no underline by default, and a solid underline
  on hover.

Emphasis (bold or italic) is not used in prose to mark terms. A term that warrants attention is
either a definition link or it is left as plain prose.

### Link-underline setting

A new setting on the hidden settings page, "Show link underlines", default off. When off,
underlines appear only on hover (the default above). When on, both link types show their
underline at all times (dotted for definition links, solid for cross-references). This is
implemented as a CSS token toggled by `js/theme.js`, in the same pattern as accent and density.
The user proposed this setting, so it is approved for the settings page.

## Changes from the estimation pilot

The estimation page (`pages/estimation.html`) and `styles/article-ui.css` already implement
much of this. To reach the archetype:

1. Add the mono section number to each section header, tied to the rail numbering.
2. Add the mono category eyebrow to the header; confirm there is no meta row.
3. Widen prose max-width to about 82ch.
4. Replace any emphasized terms with definition links; add the `.def` (definition) and `.ref`
   (cross-reference) link treatments with underline-off-by-default and hover reveal.
5. De-bold formula labels (`\text{}` rather than bold).
6. Adopt the demo pattern: figure caption, framed plot, controls below, results readout column.
7. Rail responsive: hide the rail on narrow desktop (drop the desktop hamburger); move the
   mobile hamburger to the bottom-left.
8. Add the link-underline token and its settings-page toggle.

## Open follow-ons (not this spec)

- The definition-page / glossary content system (what a definition page looks like, how links
  resolve, how the glossary is authored). Tracked separately; term list at
  `docs/design/glossary.md`.
- Shortening long section titles across pages as they migrate.
