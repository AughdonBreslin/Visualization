# Section Outline Navigation Design

**Date:** 2026-06-14

**Goal:** Give every article/visualizer page an on-page outline that lets a reader jump
directly to any panel. On wide screens it is a persistent sticky rail in a reserved left
column; on narrow screens and phones it collapses behind a hamburger that opens a slide-in
drawer. The current panel is highlighted as the reader scrolls.

## Background

Each page under `pages/` is a vertical stack of `<section class="panel collapsible">`
blocks. `js/collapsible.js` makes those panels tap-to-toggle on phones (max-width 640px)
and always-open on wider screens. The pages share `styles/base.css`, `styles/article.css`
(or `styles/visualizer.css`), and `styles/responsive.css`.

The outline reuses this structure. It is one shared, auto-generating module so there is no
hand-maintained list per page and new panels appear automatically.

## Scope

In scope: all eight content pages (distributions, generative_classification, regularization,
estimation, bayesian, pca, fourier, manifold). The home `index.html` is a short directory
and is out of scope.

## Panel discovery

The module selects every `.panel` that is not nested inside another `.panel` (top-level
panels only; nested plot cards such as `.pca-plot-card` are not `.panel` and are excluded).
For each panel:

- Label = trimmed text of its first heading (`:scope > h2, :scope > h3`). Panels with no
  heading are skipped.
- If the panel has no `id`, assign a slug derived from the label (lowercased, non-word runs
  to hyphens, de-duplicated with a numeric suffix on collision).
- Add `scroll-margin-top` so a scrolled-to panel is not clipped under any sticky element.

This covers distributions (Controls / PDF / CDF / Active Distributions, whose headings are
`h3` except the last) and manifold (its seven sections, once collapsible).

## Behavior

**Click an entry**
- If the target panel is `.collapsible` and not `.open` (phone width), open it first by
  triggering the existing collapsible toggle (which already fires the resize event and the
  MathJax retypeset), then smooth-scroll to it.
- On wider screens panels are always open, so it just smooth-scrolls.
- Update `location.hash` with `history.replaceState` so any panel is directly linkable.

**Deep link on load**
- If the URL has a hash matching a panel id, open it (when collapsed) and scroll to it after
  layout settles.

**Scrollspy**
- An `IntersectionObserver` over the panels marks the topmost visible panel's entry as
  active (`aria-current="true"` plus an `.active` class). `rootMargin` is tuned so the
  active entry flips near the top of the viewport.

## Layout: rail vs drawer

The module injects a single `<nav class="section-outline" aria-label="On this page">` into
`document.body` and adds a `has-section-outline` class to `<body>`. CSS presents it two ways.

**Reserved rail (desktop, min-width 1100px)**
- `body.has-section-outline` gets extra left padding of `--rail-w + --rail-gap + base inset`,
  reserving a column. The centered `.container` recenters in the remaining space (its
  `margin: 0 auto` is unchanged; it simply has less room and stays at or below its 1200px
  cap).
- The rail is `position: fixed` at the left edge, full height, vertically scrollable if the
  list is long. It shows the entries as a simple vertical list with the active one highlighted.
- Suggested values: `--rail-w: 160px`, `--rail-gap: 16px`. Final numbers tuned in
  implementation.

**Hamburger drawer (max-width 1099px, includes phones)**
- No reserved column. A fixed hamburger button (top-left, clear of the page header) toggles a
  slide-in drawer with a dimmed backdrop.
- The drawer closes on backdrop tap, Esc, or selecting an entry.
- Accessibility: button has `aria-expanded` and `aria-controls`; the drawer has
  `aria-label`; focus moves into the drawer on open and returns to the button on close.

This is the "reserve the column until width forbids it" rule: the rail is present on real
desktops and laptops down to 1100px, and the hamburger covers everything narrower.

## Files

New:
- `js/section-outline.js` - discovery, rail/drawer construction, click + scroll + deep-link,
  scrollspy. Self-initializes on DOMContentLoaded; no per-page configuration.
- `styles/section-outline.css` - rail, reserved-column padding, hamburger, drawer, backdrop,
  active state, and the 1100px breakpoint.

Edited (all 8 content pages): add the two include lines.

manifold.html specifically:
- Add `collapsible` to each of its seven `<section class="panel">` blocks.
- Mark one panel `data-open-mobile` (the Visualization panel) so it is open by default on
  phones.
- Add `<script defer src="../js/collapsible.js"></script>`.

## Edge cases and constraints

- Pages whose panel headings are `h3` (distributions) are handled by the `h2, h3` heading
  selector; no markup change needed there beyond the includes.
- Load order: `section-outline.js` reads the DOM after `DOMContentLoaded`; it does not depend
  on `collapsible.js` having run, but it triggers the collapsible toggle by dispatching the
  same interaction collapsible.js listens for, so collapsible.js must be included on any page
  with collapsible panels (already true for all but manifold, which this change fixes).
- The rail must not overlap fixed page chrome; verify against the manifold isomap player and
  the fourier/pca interactive panels at the 1100px boundary.
- Respect `prefers-reduced-motion`: disable smooth-scroll and drawer slide animation when set.

## Testing

- Manual pass on each of the eight pages at three widths: >=1400px (rail), ~900px (hamburger,
  reserved column gone), <=640px (hamburger, panels collapsed; clicking an entry opens the
  target panel then scrolls).
- Verify deep links (`page.html#some-panel`) open and scroll correctly, including to a
  collapsed panel on a phone.
- Verify scrollspy highlights track the panel in view.
- Verify keyboard: Tab to hamburger, Enter opens, Esc closes, focus returns; rail entries are
  reachable and activate with Enter.
- Verify `prefers-reduced-motion` disables animations.
