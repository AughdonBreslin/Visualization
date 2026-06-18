# Distribution Visualizer redesign

The last Phase 4 page. Unlike the others, distributions is a standalone tool, not an article, so it
gets a dedicated tool/app layout (the approved mocks: `.superpowers/mocks/distributions-layout.html`
+ `distributions-mixture.html`), in the dark flat aesthetic, rather than the article archetype.

## Approved design

Flat, minimal (hairlines + spacing, no bordered "bubble" cards), the periwinkle accent, underline
fields, eyebrow labels. No section-outline rail (a tool is one interactive surface, not a scrollable
article). The tool may be a touch wider than article pages (container ~1080px).

Layout: **plots-led** (the output is the first thing you see), top to bottom:

1. **Header**: eyebrow `// Statistics`, h1 "Distribution Visualizer", a one-line lede.
2. **Plots**: the PDF and CDF charts side by side (full width), restyled to the dark theme. Each
   active distribution draws as a line in its own color; the curves overlay.
3. **Controls**: a toolbar row (X range = two underline fields "min to max", a right-aligned
   "Create mixture" and accent "+ Add distribution"), then the distribution list as flat rows.
4. **Active distributions**: per-distribution details (formula, moments, use cases), flat.

### Distribution rows

Each distribution is a flat row (hairline-separated, no card): a color swatch, a Distribution type
select (underline), the type's parameter fields (underline, small uppercase labels), and a Remove
control on the right. Parameter symbol labels keep their natural case (do NOT uppercase the Greek
letters; `text-transform: uppercase` turns mu/sigma/lambda into capitals that read as Latin M / E /
A, so the param labels are styled without uppercase, or the symbol is excluded from the uppercased
text).

### Mixture rows

A mixture is a distribution row that expands into a nested component table under a left hairline:

- Header row: WEIGHT · TYPE · PARAMETERS.
- One component row each: a weight field (relative weight) + a share bar/percentage, the
  component's type select, that type's parameter fields, and a remove (x).
- A "+ Add component" action.

Weight model (approved): **relative weights, auto-normalized**. The user types a non-negative
relative weight per component; the effective mixing weight is `w_i / sum(w_j)`. Editing one weight
changes only that field; every component's share bar / percentage recomputes from the new total,
but the other typed values are left alone. The PDF/CDF use the normalized weights. The mixture's
density is `f(x) = sum_i w_i f_i(x)` with normalized `w_i`.

### Active distribution details

Per active distribution, flat: a swatch + name header, the formula (mono code well or MathJax), the
moments (Expected value / Variance as small label + value pairs), and the "Common uses in deep
learning" list. For a mixture: the formula `f(x) = sum_i w_i f_i(x)` and a Components list (each
normalized weight + swatch + component type with its params).

## Implementation

### Files

- Modify: `pages/distributions.html` (head swap to the `.ui` archetype assets, body class, the
  plots-led structure, footer). Add the markup the JS expects (the mixture create/section hooks if
  they are wired; preserve `#forms`, `#pdf`, `#cdf`, `#addDist`, `#resetAll`, `#distributions-details`).
- Rewrite: `styles/visualizer.css` to `.ui` / `.ui.distributions`-scoped flat tool styling (replace
  the legacy blue-accent boxed design).
- Modify: `js/distributions.js` as needed for: the flat row structure (if the existing built
  structure cannot be styled flat via CSS alone), the mixture component table with the weight
  share bar/percentage, the relative-normalized weight display, the parameter-label case fix, and
  the dark plot styling (axis/grid/line colors; the first default color becomes periwinkle and the
  per-distribution palette stays distinct on the dark background). Preserve all d3 select hooks
  (`#forms`, `#pdf`, `#cdf`, `#addDist`, `#resetAll`, mixture hooks) and the
  `getElementById('distributions-details')`.
- Update: `docs/design/redesign-backlog.md` + the `project_redesign_backlog` memory (mark
  distributions done; Phase 4 complete).

### Plots (d3) dark restyle

The PDF/CDF d3 charts get the dark treatment: transparent/near-black plot background, hairline
axes and gridlines, muted tick labels (mono tabular), and distribution lines in the per-distribution
swatch colors (default periwinkle `#6b7cff`, then a distinct dark-friendly palette for additional
distributions). Titles become small mono eyebrow captions ("Probability density (PDF)" /
"Cumulative (CDF)"). Keep the existing scales/curve logic; only colors/axes/labels change.

## Constraints

- No em / en dashes; no `<em>` / `<strong>`. Preserve all element ids and d3 hooks. Scope CSS under
  `.ui` (and `.ui.distributions`).

## Out of scope (deferred)

- Content edits, full mobile pass, input-range robustness sweep (the standard deferred phases).

## Verification

- The plots render at top with the dark styling and per-distribution colors; adding a distribution
  adds a row and a curve; editing params updates the curves; Remove works; Clear All empties.
- A mixture row expands to its weighted components; editing one component weight leaves the other
  typed weights alone and updates the share bars and the curves; Add component / remove component
  work; the mixture detail shows the formula + components.
- Param symbol labels are not mis-uppercased. No console errors. Dash + id guards clean.
