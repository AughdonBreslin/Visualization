# Glossary / definition-page system - design

## Context

The article-page archetype shipped the definition-link treatment (`a.def` in
`styles/system.css`): a quiet, body-color link with a dotted underline on hover, used in prose
to mark a term that has a definition instead of using bold or italic emphasis. The pages those
links point to do not exist yet, and no page currently uses `.def`. A running term seed lives at
`docs/design/glossary.md` and currently covers estimation only (~21 terms).

This spec defines the glossary content system and builds the first vertical slice on the
estimation page, end to end: prose `.def` link, click, glossary entry. Other content pages
(pca, fourier, bayesian, regularization, manifold, generative classification) follow later, page
by page, reusing this system.

The site is a static site with no build step. Every page is hand-authored HTML served as-is.

## Scope

In scope:
- A new glossary page at `pages/glossary.html` (clean URL `/pages/glossary`).
- Hand-authored entries for the estimation term set.
- A slug convention and the full slug table for the estimation terms.
- Wiring the matching terms in `pages/estimation.html` prose into `.def` links.
- A glossary entry point: a book icon in the home footer beside the settings gear.

Out of scope, deferred to later passes:
- Terms for any page other than estimation.
- Live search or filtering.
- External citations / outbound source links.
- `.def` wiring on any page other than estimation.

## The page

`pages/glossary.html` follows the article archetype: the same `.ui` shell, header, hairline
rhythm, and typography as the other content pages (see
`docs/superpowers/specs/2026-06-16-article-page-design.md`). It carries the Vercel analytics
script like every other public page.

Structure:
- Page header (title "Glossary", short standfirst describing what it is).
- An A-Z jump bar: a row of letter links anchoring to the first entry under each letter. Cheap
  and on-brand. May be cut without affecting the rest of the design.
- One flat list of entries, sorted alphabetically by term. As other pages contribute terms
  later, they merge into this single A-Z list.

## An entry

Each term is a hand-authored block:

```html
<section class="term" id="<slug>">
  <h2>Term name</h2>
  <div class="term-visual">
    <figure>
      <svg ...>...</svg>
      <figcaption>Short caption.</figcaption>
    </figure>
    <!-- up to 3 figures -->
  </div>
  <p>Explanation in measured prose. May include LaTeX math via the existing setup.</p>
  <p class="see-also">
    See also: <a href="#variance">variance</a>, <a href="#mse">MSE</a>.
    Learn more: <a href="estimation#bias-variance">Estimation - Bias and variance</a>.
  </p>
</section>
```

Rules:
- Visual slot holds 1 to 3 small inline SVG sketches. A very simple CSS or SVG animation is
  allowed where it clarifies; keep it simple. Each figure has a `figcaption` for accessibility.
- Body is plain prose, no `<em>` or `<strong>`. LaTeX math is allowed.
- References line has two kinds of links: see-also (internal `#slug` anchors to related entries)
  and one "Learn more" cross-reference back to the source article section.
- One idea per entry. Measured, non-dramatic phrasing.

New CSS lives in a dedicated `styles/glossary.css` (the per-page pattern used by the other
pages). It styles `.term`, `.term-visual`, the figure/caption layout, `.see-also`, and the A-Z
jump bar. The `a.def` link style already exists and is not changed.

## Slugs

Slug = kebab-case of the term, with parentheticals dropped. Terms commonly referenced by an
abbreviation use the abbreviation as their slug. Full table for the estimation set:

| Term | Slug |
|---|---|
| estimator | `estimator` |
| parameter | `parameter` |
| point estimation | `point-estimation` |
| function estimation | `function-estimation` |
| bias | `bias` |
| variance | `variance` |
| unbiased estimator | `unbiased-estimator` |
| standard error | `standard-error` |
| sampling distribution | `sampling-distribution` |
| bootstrap | `bootstrap` |
| confidence interval | `confidence-interval` |
| consistency | `consistency` |
| convergence in probability | `convergence-in-probability` |
| mean squared error (MSE) | `mse` |
| bias-variance decomposition | `bias-variance-decomposition` |
| irreducible noise | `irreducible-noise` |
| maximum likelihood estimation (MLE) | `mle` |
| likelihood | `likelihood` |
| log-likelihood | `log-likelihood` |
| conditional log-likelihood | `conditional-log-likelihood` |
| cross-entropy | `cross-entropy` |

## Wiring estimation.html

Convert the matching terms in estimation's prose into definition links:

```html
<a class="def" href="glossary#<slug>">term</a>
```

Density: link the first occurrence of each term within each section. Do not link every instance,
headings, or terms inside math. This keeps the prose quiet while still surfacing the links where
a reader first meets a term.

The `a.def` style already ships, so this is markup only; no CSS change.

## Entry point

A book icon is added to the home footer row in `index.html`, immediately beside the existing
settings gear:

```html
<a class="footer-glossary" href="pages/glossary" aria-label="Glossary" title="Glossary">
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <!-- Feather-style book icon, matched to the 15x15 cog -->
  </svg>
</a>
```

It reuses the footer styling (`.footer-settings` is the reference; `.footer-glossary` gets the
same muted color and inline-flex treatment). The settings gear lives only in the home footer, so
the book icon appears only there. On content pages, the in-context entry to the glossary is the
`.def` links in prose.

## Out of scope reminder

Other pages' terms, search, external citations, and `.def` wiring beyond estimation are separate
follow-on passes. The seed list at `docs/design/glossary.md` grows as each page is migrated.
