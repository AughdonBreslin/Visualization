# UI redesign backlog (deferred phases)

Branch: redesign/ui-system. This tracks work intentionally deferred so the current focus
(building the article-page archetype and migrating pages, on desktop) stays unblocked. Items
here are not started; each becomes its own brainstorm -> spec -> plan when picked up.

## In-progress / next

- Phase 4 (page migration): migrate the remaining article pages onto the archetype. Done:
  estimation (pilot), bayesian, regularization, pca. Next: fourier, manifold,
  generative_classification, distributions. The archetype lives in `styles/article-ui.css`,
  `styles/system.css`, `styles/tokens.css`, `js/section-outline.js`, `js/theme.js`. The demo
  control system (bands / column-flow / integrated tables / sliders / container-query reflow) is
  in `styles/article-ui.css`.

## Deferred phases (later, in no fixed order)

### Content adjustments pass
Review and revise the prose and math content of migrated pages for clarity and accuracy
(separate from styling). Includes shortening long section titles (the agreed revision), and
tightening copy. Definition-link wiring depends on the glossary system below.

### Stability / robustness pass
Make the interactive demos and layouts hold up under stress, on desktop first:
- Exercise the demos across their full input ranges and edge values (very small / very large n,
  reps, seeds, extreme parameters) and confirm no NaNs, broken plots, or layout breakage.
- Check behavior across desktop and tablet widths (the rail show / hide / reflow bands, the
  demo controls wrap, math overflow, long equations) and fix anything that jumps or clips.
- Confirm theme settings (accent, density, link underlines) all hold across pages.

### Full mobile pass
A dedicated mobile sweep. The current work targets desktop; mobile is only roughly handled
(bottom-left hamburger, stacked demo controls). This phase does mobile properly: touch targets,
the outline drawer, demo usability on small screens, math overflow / horizontal scroll, type
scale, spacing, and a real device check.

### Definition-page / glossary system
The definition-link treatment (`.def`) ships, but the pages it points to do not exist yet.
Design the glossary / definition-page content system: what a definition page looks like, how
links resolve, how terms are authored. Term list seed at `docs/design/glossary.md`.

## Done (for reference)
- Phases 1, 2a/2b/2c, 3, and the article-page archetype (estimation pilot) are complete on this
  branch. See `docs/superpowers/specs/2026-06-16-article-page-design.md`.
