# Manifold comparison sections redesign

A redesign (not just restyle) of the manifold sandbox's comparison sections, following the
approved flat mock (`.superpowers/mocks/manifold-final.html`). Unlike the CSS migration, this
changes HTML structure and some JS. The dataset section and the Isomap video explainer are
unchanged.

## Approved design

Flat, minimal (no bordered "bubble" cards; hairlines + spacing + underline selects + eyebrow
labels). The two algorithms diverge in their steps (PCA has no kNN / geodesic), so the step
indicator is a single aligned dual-track widget: a shared canonical stage axis with one dot row per
algorithm, where a step an algorithm skips shows a small grey "na" dot and a dashed connector. No
per-side color swatches on column headers (the user asked to drop them).

Section structure (replacing the current Algorithms + Visualization + Step notes + Pseudocode):

1. **Algorithm comparison** (merged): Algorithm A / B pickers (each with its params), the aligned
   dual-track stepper, prev / step-description / next, then the Visualization (the two viz hosts
   side by side under a "Visualization" eyebrow).
2. **Step notes** (separate): two columns (left algo / right algo), each a flat IFW (eyebrow tabs
   Intuition / Formula / Worked example) with the algorithm name as a plain header (no swatch).
3. **Full pseudocode** (separate): two columns, each a flat pseudocode list (numbered step rows,
   current step marked by an accent left bar + a code well).

## Implementation

### JS

- `js/manifold/step_indicator.js`: rewrite the render to build the aligned dual-track instead of
  two `.sp-bar` rows + `.sp-detail` lists. Keep the module interface exactly:
  `createStepIndicator(container, { onJump })` -> `{ render({ leftLabel, rightLabel, leftSubSteps,
  rightSubSteps, currentSubStep }) }`, so `main.js` is unchanged. The module now:
  - selects `.mf-tracks` (new container in the HTML), `.step-prev`, `.step-next`, `.step-desc`;
  - builds a grid: a stage-label header row (the `CANONICAL_STEPS` labels) + one dot row per
    algorithm. Each dot is classified with the existing `classifyDot` (na / filled / hollow) plus
    a `current` state when the dot's canonical id equals the nearest sub-step for the current
    step; clicking an applicable dot calls `onJump(cid)`;
  - keeps the prev/next disabled logic and the description text from the current `render`.
  - Drops the expandable `.sp-detail` step list (the tracks show everything).
- `js/manifold/main.js`: one small addition only if needed: set plain text headers for the Step
  notes / Pseudocode columns (the pseudocode card already renders `.pseudocode-title` with the algo
  label; for Step notes, add `mfLeftIfwTitle` / `mfRightIfwTitle` spans updated in `subscribe`).
  Everything else (ids, viz, ifw, pseudocode wiring) is unchanged; the hosts keep their ids and are
  only relocated in the HTML.
- No other `js/manifold/*` file changes. `ifw.js` / `pseudocode.js` output is restyled by CSS.

### HTML (`pages/manifold.html`)

- Replace the "Algorithms" section (`sp-frame` / `mfAlgoStepPanel`) body: the two `.sp-panel` cards
  (which held the selects, params, `.sp-bar`, `.sp-detail`) become a flat header row of the two
  pickers + their param hosts (`mfAlgoLeft` + `mfAlgoLeftParams`, `mfAlgoRight` +
  `mfAlgoRightParams`), a `<div class="mf-tracks"></div>`, and the `.sp-nav` (`.step-prev` /
  `.step-desc` / `.step-next`). Keep `id="mfAlgoStepPanel"` (the indicator container) and the
  `step-prev/next/desc` class hooks.
- Move the Visualization markup (`mf-viz-row` with `mfLeftViz` / `mfRightViz` and the
  `mfLeftTitle` / `mfRightTitle` spans) into the Algorithm comparison section under a
  "Visualization" eyebrow. Remove the standalone Visualization section.
- Keep the Step notes and Full pseudocode sections, each with two columns; add a plain per-column
  header (algo name) above each `mfLeftIfw` / `mfRightIfw` and `mfLeftPseudo` / `mfRightPseudo`.
  Drop the colored swatches.
- All element ids preserved so `main.js` keeps working.

### CSS (`styles/manifold.css`)

- Add the flat treatments: `.mf-tracks` grid (stage labels + two dot rows, connectors, na/current
  states), the flat underline pickers, the eyebrow `Visualization` label, flat IFW tabs (eyebrow
  underline tabs), flat pseudocode rows (numbered, accent-left current + `#060607` code well),
  plain column headers.
- Remove / replace the now-unused `.sp-frame` / `.sp-panel` / `.sp-bar` / `.sp-cell` / `.sp-dot` /
  `.sp-edge` / `.sp-detail` / `.sp-step` card + dot rules (superseded by `.mf-tracks`). Keep the
  `.mf-controls-row` (Dataset) and viz-host rules.
- Stay within the archetype tokens (periwinkle accent, hairlines, `#060607` wells); no bordered
  surface cards on the comparison sections.

## Constraints

- No em / en dashes; no `<em>` / `<strong>`. Preserve all element ids. Scope CSS under
  `.ui.manifold`.

## Verification

- The dual-track stepper renders both algorithms aligned on the stage axis, shows na dots where an
  algorithm skips a step, highlights the current stage, and clicking a dot / prev / next changes
  the step (both viz, notes, and pseudocode update). The merged Visualization shows under the
  stepper. Step notes and pseudocode render as separate flat sections. No console errors. Isolation
  + dash guards clean.
