# Manifold Parameter UI Cleanup Design

**Date:** 2026-06-02
**Status:** Approved (design), pending implementation plan

## Goal

Make the algorithm and dataset parameter controls on the manifold page look polished, read clearly, and explain themselves. Replace the bare `name = [input]` chips with a left-aligned two-column layout (chosen "layout B"), readable parameter names, and a hover tooltip per parameter that defines the term and how it affects the result.

## Scope

In scope:
- The shared `renderParamHost` in `js/manifold/main.js`, which renders:
  - the left algorithm params (`#mfAlgoLeftParams`)
  - the right algorithm params (`#mfAlgoRightParams`)
  - the dataset extra-params row (`#mfDatasetParams`)
- Parameter descriptors in every algorithm and dataset module that exposes params, which gain two optional fields: `label` and `desc`.
- New CSS for the two-column layout, the info icon, and a shared floating tooltip.

Out of scope:
- The fixed Samples / Noise / Seed controls (separate markup in `.mf-controls-row`). They keep their current look. Matching tooltips for them are a possible later follow-up.
- Slider controls, live-drag apply, and any change to the commit-on-change recompute behavior.
- Any change to parameter values, ranges, or algorithm behavior.

## Layout (chosen: B)

Each visible parameter is one row of a two-column grid inside the param host:

```
[i] Neighbors (k)        [ 10 ]
[i] Regularization       [ 0.001 ]
```

- Left column: an info icon followed by the human-readable label, left-aligned.
- Right column: the control. Number input for `int` and `float`, a dropdown for `enum`.
- Columns align across rows (grid `auto 1fr`).
- `dependsOn` filtering is unchanged: a row only renders when its dependency is satisfied (Kernel PCA shows the kernel plus either gamma or degree, never both).
- Empty state (no visible params, e.g. PCA or a dataset with no extra params): render nothing for datasets; for algorithms, show the existing muted "No parameters" text.

The layout is vertical, so an algorithm with more params simply grows taller; nothing wraps horizontally.

## Descriptor changes

Param descriptors today look like:

```javascript
{ name: 'k', type: 'int', default: 10, min: 2, max: 50 }
```

They gain two optional fields:

```javascript
{ name: 'k', type: 'int', default: 10, min: 2, max: 50,
  label: 'Neighbors (k)',
  desc: 'How many nearest neighbors define each point\'s local neighborhood. Smaller k captures finer local detail but can fragment the manifold; larger k is smoother but may link points across separate folds.' }
```

`renderParamHost` uses `label` for the display name (falling back to `name`) and `desc` for the tooltip body (no tooltip / no info icon if `desc` is absent). The range shown in the tooltip is derived from the existing `min`/`max` (or the `options` list for enums), so it does not need to be repeated in `desc`.

## Tooltip mechanism

- A single floating tooltip element is created once and reused. It is positioned near the cursor on hover and hidden on mouse-out.
- Each parameter with a `desc` gets a small circular `i` info icon next to its label. Hovering the icon shows: the label (title), the `desc` text, and a derived range line ("range 2 to 50", or "rbf / polynomial / linear" for enums).
- The tooltip is plain styled HTML, consistent with the page's dark theme. It must not interfere with pointer interactions (pointer-events none) and must reposition to stay on screen near the right edge.

## Labels and definitions

### Algorithm parameters

| module | name | label | desc |
|---|---|---|---|
| isomap, lle, laplacian | k | Neighbors (k) | How many nearest neighbors define each point's local neighborhood. Smaller k captures finer local detail but can fragment the manifold; larger k is smoother but may link points across separate folds. |
| lle | reg | Regularization | Stabilizes the per-point least-squares weight solve when a point's neighbors are nearly coplanar. Larger values make the reconstruction weights smoother and more uniform; too large washes out the local geometry. |
| laplacian | sigma | Bandwidth (σ) | Width of the heat kernel that turns neighbor distances into edge weights, as a multiple of the median neighbor distance. Larger σ makes neighbor weights more uniform and the embedding smoother; smaller σ sharpens the falloff. |
| kpca | kernel | Kernel | The similarity function. Kernel PCA runs ordinary PCA in the implicit feature space this kernel defines, so the choice sets what kind of nonlinear structure it can unfold. |
| kpca | gamma | RBF width (γ) | Width of the RBF (Gaussian) kernel, auto-scaled to the data's spread. Larger γ makes similarity drop off faster (very local, fine detail); smaller γ is broad and smooth. |
| kpca | degree | Polynomial degree (d) | Degree of the polynomial kernel. Degree 1 is linear; higher degrees capture higher-order interactions among coordinates, bending the feature space more. |

### Dataset parameters

| module | name | label | desc |
|---|---|---|---|
| helix | turns | Turns | Number of full turns of the helix. More turns make a longer, tighter coil. |
| toroidal_helix | q | Winding (q) | How many times the helix winds around the torus tube per loop around the ring. Higher q coils more tightly. |
| spiral_disk | turns | Turns | Number of turns of the spiral arm. More turns wind it tighter toward the center. |
| cylinder | height | Height | Height of the open cylinder. Taller values stretch the tube along its axis. |
| severed_sphere | cap | Cap size | Fraction of the sphere removed at the north pole, measured along the polar angle. Larger values cut away more of the top. |
| hilbert | order | Order | Recursion depth of the Hilbert curve. Each step subdivides the cube into a finer grid (2^order cells per side), so higher order packs the curve more densely. |
| clusters_3d | clusters | Clusters | Number of Gaussian blobs, with centers spread evenly on a sphere. |
| clusters_3d | sep | Separation | Radius of the sphere the cluster centers sit on. Larger values push the blobs farther apart. |

Parameter ranges (for the derived tooltip range line), from current code:
- k: 2 to 50; reg: 0 to 0.1; sigma: 0.1 to 10
- kernel: rbf / polynomial / linear; gamma: 0.05 to 10; degree: 1 to 10
- turns (helix): 1 to 8; q: 2 to 15; turns (spiral_disk): 1 to 6; height: 0.5 to 5; cap: 0 to 0.9; order: 2 to 5; clusters: 2 to 8; sep: 0.5 to 5

## Components and data flow

- `renderParamHost(host, owner, getCurrent, onChange)` is unchanged in signature. Internally it builds a grid container, and for each visible param appends a label cell (info icon + label) and a control cell. The control wiring (number parsing, enum change, `dependsOn` re-render, commit via `onChange`) is unchanged.
- A new small tooltip helper module (or a function within main.js) owns the single floating tooltip element and exposes `attachTooltip(iconEl, { label, desc, rangeText })`.
- No change to `state.js`, the worker, or any algorithm computation.

## Styling

- New CSS classes for the grid (`.mf-param-grid` or reuse `.mf-algo-params` restyled), the label cell, the info icon, and the tooltip. Use existing theme variables (`--surface-inset`, `--border-light`, etc.) where available so it matches the page.
- Number inputs and selects get consistent sizing and focus styling.
- The earlier rejected "pill chip" styling is not reused.

## Testing

- This is DOM and styling work; verification is manual browser smoke, since the project has no DOM test harness.
- Manual checklist: each algorithm (PCA, Isomap, MDS, LLE, Laplacian, Kernel PCA) and each parameterized dataset (helix, toroidal helix, spiral disk, cylinder, severed sphere, hilbert, 3D clusters) shows correctly labeled rows; info icons show the right definition and range on hover; Kernel PCA toggling hides/shows gamma vs degree; changing a value still recomputes; CSV and no-param datasets show no stray param row.
- `node --test 'test/manifold/*.test.js'` must still pass (no logic changes expected, but descriptors are edited so the modules must still load).

## Edge cases

- A param with no `desc`: render the label with no info icon and no tooltip.
- A param with no `label`: fall back to `name`.
- Tooltip near the right or bottom edge of the viewport repositions to stay visible.
- Rapid hover across rows reuses the single tooltip element without leaking listeners.

## Style constraints (project)

- No em-dashes anywhere in code, comments, or copy.
- No `<em>`, `<strong>`, `<b>`, `<i>` (as emphasis), or `<mark>` tags in generated content. The info icon uses a styled `<span>`, not an `<i>` tag.
