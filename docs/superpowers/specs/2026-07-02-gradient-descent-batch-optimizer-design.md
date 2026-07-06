# Gradient Descent: Batch Size and Optimizer Comparison Design

Date: 2026-07-02
Status: approved for implementation

## Problem

The gradient descent sandbox (`pages/gradient-descent.html`, `js/gradient-descent.js`) currently shows exactly three fixed optimizers (SGD, Momentum, Adam) descending the same analytic loss surface, always all three at once.
There is no way to compare gradient descent's batch-size variants (full-batch GD, mini-batch MBGD, stochastic SGD), and no way to add or isolate other optimizers (AdaGrad, RMSProp).
Today's "SGD" line is also not actually stochastic: every step uses the exact analytic gradient, since the surfaces are hand-written functions rather than a dataset with individual examples to subsample.

## Scope

Extend the existing sandbox, in place, to support two comparisons:

1. Batch size: full-batch GD vs mini-batch MBGD vs stochastic SGD, all using one fixed, user-selected optimizer/update rule.
2. Optimizers: SGD vs Momentum vs AdaGrad vs RMSProp vs Adam, all using one fixed, user-selected batch size.

Explicitly out of scope: ensemble/boosting methods (XGBoost, AdaBoost).
These are gradient boosting algorithms operating in function space over an ensemble of weak learners, not first-order update rules over a continuous weight vector, and cannot share this sandbox's "single dot descending a fixed loss surface" visualization.

Since the surfaces are analytic functions with no underlying dataset, there is no real per-example gradient to subsample.
Batch size is instead simulated by injecting synthetic Gaussian noise into the true analytic gradient, scaled down as batch size grows.
This is a pedagogical approximation, not a literal computation from data, and the page copy says so explicitly.

## Interaction model

A segmented toggle is added to the controls: **Compare batch size** / **Compare optimizers**.

**Compare batch size** mode:
- A new "Optimizer" select (SGD / Momentum / AdaGrad / RMSProp / Adam) pins the update rule. Default: SGD.
- All 3 batch-size lines (Full-batch, Mini-batch, Stochastic) always run simultaneously, using that one pinned update rule.
- Lines are colored by batch mode.

**Compare optimizers** mode:
- A new "Batch size" select (Full-batch / Mini-batch / Stochastic) pins the noise level. Default: Full-batch (matches today's noise-free behavior).
- 5 checkboxes, one per optimizer, all checked by default. Only checked optimizers run.
- Lines are colored by optimizer.

Switching mode, the pinned selector, or a checkbox rebuilds the active line set and calls the existing `resetAll()` reset path, same as switching surfaces does today.
Surface picker, learning-rate slider, Animate/Step/Reset, and click-to-set-start-point are unchanged and apply to whichever mode is active.

A one-line summary sits above the legend describing the pinned dimension, e.g. "Comparing batch size — optimizer: Adam" or "Comparing optimizers — batch size: Mini-batch", so per-line legend labels stay short (just the varying dimension: "Full-batch" / "Mini-batch" / "Stochastic", or "SGD" / "Momentum" / "AdaGrad" / "RMSProp" / "Adam").

## Data model

Two static config arrays replace the current fixed `OPTS` constant:

```js
const OPTIMIZERS = [
  { key: 'sgd',      label: 'SGD',      color: '#74b9ff' },
  { key: 'momentum', label: 'Momentum', color: '#fd79a8' },
  { key: 'adagrad',  label: 'AdaGrad',  color: '#a29bfe' },
  { key: 'rmsprop',  label: 'RMSProp',  color: '#55efc4' },
  { key: 'adam',     label: 'Adam',     color: '#00cec9' },
];

const BATCH_MODES = [
  { key: 'full',       label: 'Full-batch',  n: Infinity, color: '#74b9ff' },
  { key: 'mini',       label: 'Mini-batch',  n: 16,       color: '#ffeaa7' },
  { key: 'stochastic', label: 'Stochastic',  n: 1,        color: '#ff7675' },
];
```

`BATCH_MODES` colors are sequential (calm blue to hot coral) so color intuitively reads as "more noise," distinct from `OPTIMIZERS`'s qualitative palette.

Step functions (`stepSGD`, `stepMomentum`, `stepAdam`, plus new `stepAdaGrad`, `stepRMSProp`) are refactored to accept an already-computed gradient vector `[gx, gy]` instead of calling `fn.grad` internally.
This is what lets noise be injected between computing the true gradient and handing it to the optimizer.

AdaGrad accumulates the sum of squared gradients and divides by its square root:

```
G_t = G_{t-1} + (∇L)^2
θ_{t+1} = θ_t - α · ∇L / (√G_t + ε)
```

RMSProp replaces AdaGrad's unbounded cumulative sum with an exponential moving average, so the effective learning rate stops decaying to zero over a long run:

```
E[g²]_t = β·E[g²]_{t-1} + (1-β)(∇L)²
θ_{t+1} = θ_t - α · ∇L / (√E[g²]_t + ε)
```

Noise injection, applied once per line per step before the gradient reaches the optimizer's step function:

```js
function noisyGrad([gx, gy], batchMode) {
  if (batchMode.n === Infinity) return [gx, gy];
  const sigma = BASE_NOISE / Math.sqrt(batchMode.n);
  const norm = Math.hypot(gx, gy);
  return [gx + sigma * norm * randn(), gy + sigma * norm * randn()];
}
```

`randn()` is a standard Box-Muller Gaussian sample.
Noise is scaled by the gradient's overall norm (not each component independently) so a component that happens to be exactly zero can still be perturbed off-axis, which matters near symmetric points (e.g. the saddle surface's origin).
`BASE_NOISE` is a single tuned constant calibrated during implementation to look reasonable across all 4 existing surfaces.

## Rendering and legend

Trajectory lines (3D scene), contour-view paths/dots, and the legend all currently loop over the fixed `OPTS` constant.
They generalize to loop over a computed "active lines" list instead:

- Batch-size mode: all 3 `BATCH_MODES` entries, each carrying the pinned optimizer key.
- Optimizer mode: the checked `OPTIMIZERS` entries, each carrying the pinned batch mode key.

No other change to the Three.js or D3 rendering paths is needed; they already key off `opt.key` / `opt.color`, which the active-lines list continues to provide.

## Content updates

Header copy (`<title>`, eyebrow, `<h1>`, lede) and the home page card description (`index.html`) are updated to reflect the fuller scope: batch size plus 5 optimizers, not just "SGD, momentum, and Adam."

"The algorithms" panel is reordered and extended to narrate the motivation lineage explicitly, not just present formulas in isolation:

1. **SGD** (existing): plain update rule; struggles with elongated/curved surfaces because one learning rate applies equally to every direction.
2. **Momentum** (existing): motivated by SGD's oscillation on the steep axis of an elongated bowl; accumulates velocity to smooth the path.
3. **AdaGrad** (new): motivated by Momentum still using the same effective step size for every parameter, regardless of how differently curved each axis is; introduces a per-parameter rate that shrinks based on the cumulative squared gradient.
4. **RMSProp** (new): motivated by AdaGrad's cumulative sum causing its per-parameter rate to decay toward zero and stall on long runs; replaces the cumulative sum with an exponential moving average.
5. **Adam** (existing, text adjusted): motivated as combining Momentum's directional smoothing (first moment) with RMSProp's per-parameter scaling (second moment) into one update rule.

A new **"Batch size"** subsection follows, explaining the GD/SGD/MBGD lineage: full-batch GD gives an exact gradient but costs a full pass over the data per step; SGD updates from a single example, making each step cheap but noisy; mini-batch MBGD is the practical middle ground.
It states explicitly that this sandbox simulates that noise (Gaussian, scaled by `1/√n`) on top of an analytic surface, since there's no real dataset to subsample from, rather than computing it from actual per-example gradients.

## Testing

This is a static, client-side page with no build or test tooling in the repo (confirmed: no `package.json`, no test/spec files).
Verification is manual, in-browser, per the project's existing standard: load the page, exercise both modes, all pinned-selector values, checkbox combinations, all 4 surfaces, Animate/Step/Reset, and click-to-set-start-point, checking that trajectories, legend, and colors update correctly and that noisy lines visibly differ from full-batch lines.
