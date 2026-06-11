# Manifold-learning manim walkthroughs: MDS, LLE, Laplacian Eigenmaps, Kernel PCA

Date: 2026-06-08
Status: approved, ready for implementation plan

## Goal

Extend the existing manim explainer system (Isomap, PCA) with step-by-step
walkthroughs for four more manifold-learning algorithms, all on the swiss roll:

- MDS (classical multidimensional scaling)
- LLE (locally linear embedding)
- Laplacian Eigenmaps
- Kernel PCA, rendered once per kernel: rbf, polynomial, linear (3 clips)

Each walkthrough reuses the Isomap explainer's shared visual system so the new
clips match the Isomap and PCA clips and can slot into the same web player.

## Established pattern (reused, not reinvented)

- One module per algorithm: `manimexp/<algo>/walkthrough.py`.
- A single `ThreeDScene` authored with `self.next_section(...)` per step so
  objects persist and transform across step boundaries; `--save_sections`
  emits one MP4 per step plus a full `walkthrough.mp4`.
- Reuse `manimexp.isomap.style` (`S`), `manimexp.isomap.builders` (`B`), and
  `manimexp.isomap.data` for colors, timing constants, mobject builders, and
  dataset loading. `manimexp/pca/walkthrough.py` is the closest reference
  template.
- Continuous ambient camera rotation runs across all sections and only flattens
  to a face-on (phi=0) view for the final 2D embedding step. No snap-back, no
  pauses between beats (continuous-camera preference).
- Default dataset is swiss_roll, kept parametrized via `MFI_DATASET` for free.
- Overlay helpers mirror PCA: `set_caption`, `set_pseudo`, per-step formulas in
  the top-right corner, pseudocode panel in the top-left.
- Captions follow house style: measured, non-dramatic prose; no em-dashes; no
  em/strong emphasis. Each step motivates its what/why/how rather than asserting
  a one-line claim.

## Shared infrastructure additions

1. `builders.py`: add `heatmap(matrix, n, ...)` that renders an N x N matrix as a
   downsampled color grid (square cells colored on a diverging or sequential
   scale, with a compact label). Used for D, D^2, B, W, D (degree), L, K, Kc.
   The matrix is downsampled to a fixed cell budget (e.g. up to ~32 x 32) so it
   stays legible and cheap at N=1000.
2. `data.py`: add helpers only where existing ones do not already reproduce the
   JS math. Existing `knn_graph`, `double_center_squared`, `top2_eig`,
   `geodesic_distances` cover most steps. New helpers as needed:
   - heat-kernel affinity `W_ij = exp(-||x_i - x_j||^2 / 2 sigma^2)` over kNN
     edges, the degree matrix `D`, and the graph Laplacian `L = D - W`
     (Laplacian Eigenmaps).
   - LLE reconstruction weights `W` (per-point constrained least squares) and
     `M = (I - W)^T (I - W)`, with smallest non-trivial eigenvectors.
   - kernel matrices (rbf/polynomial/linear), kernel centering, top-2 eig
     (Kernel PCA). Kernel and gamma scaling must match `js/.../kpca.js`
     (gamma auto-scaled to mean squared pairwise distance).
   All math must agree with the corresponding `js/manifold/algorithms/*.js`.

## Per-algorithm step timelines

Step ids mirror the `presentSubSteps` of the matching JS algorithm so the clips
line up with the web walkthrough.

### MDS (`MDSWalkthrough`) - steps 0,3,4,5,6

1. Raw cloud. Intro caption: a curved cloud in 3D; MDS preserves pairwise
   distances.
2. Pairwise distances D: sample connecting lines between a few points in 3D
   (geometry) plus a heatmap of D. Formula `D_ij = ||x_i - x_j||`.
3. Double-center: `B = -1/2 H D^2 H`, `H = I - (1/N) 11^T`. Show D^2 -> B as
   heatmaps with the formula (reuse the Isomap double-centering rationale).
4. Eigendecompose B, top-2 eigenpairs, with an eigenvalue readout.
5. Embed `Y = [v1 v2] diag(sqrt(l1), sqrt(l2))`; morph cloud to the 2D
   embedding face-on. Outro: MDS preserves straight-line distances, so the
   rolled layers still overlap. This is the contrast that motivates Isomap
   swapping in geodesic distances.

### LLE (`LLEWalkthrough`) - steps 0,2,3,5,6

1. Raw cloud.
2. kNN graph (reuse `knn_sphere` + `graph_edges`), local neighborhoods.
3. Reconstruction weights W: pick one point, show it rebuilt as a weighted
   combination of its k neighbors (3D weighted arrows), plus a sparse W heatmap.
   `minimize ||x_i - sum_j w_j x_{n_j}||^2`.
4. Smallest non-trivial eigenvectors of `M = (I - W)^T (I - W)` (skip l0 = 0).
5. Embed `Y = [v1 v2]`; morph to 2D. Outro: local linear reconstructions are
   preserved, so the sheet unrolls flat.

### Laplacian Eigenmaps (`LaplacianWalkthrough`) - steps 0,2,3,4,5,6

1. Raw cloud.
2. kNN graph.
3. Heat-kernel affinity W: kNN edges colored by `exp(-||x_i - x_j||^2 / 2 sigma^2)`
   plus a W heatmap.
4. Graph Laplacian `L = D - W` with `D_ii = sum_j W_ij`: W, D, L heatmaps and
   the formula.
5. Smallest non-trivial eigenvectors of L (skip l0 = 0).
6. Embed `Y = [v1 v2]`; morph to 2D. Outro: nearby points stay nearby, so the
   sheet flattens while preserving locality.

### Kernel PCA (`KPCAWalkthrough`, `MFI_KERNEL` in {rbf, polynomial, linear}) - steps 0,3,4,5,6

1. Raw cloud; intro names the active kernel.
2. Kernel matrix K: heatmap plus the kernel formula
   (rbf `exp(-gamma ||x_i - x_j||^2)`, polynomial `(x_i . x_j + 1)^d`,
   linear `x_i . x_j`).
3. Center K -> Kc: `Kc = K - 1_N K - K 1_N + 1_N K 1_N`; heatmaps plus formula.
4. Eigendecompose Kc, top-2.
5. Embed `y_{i,k} = sqrt(l_k) v_{k,i}`; morph to 2D. Per-kernel outro:
   - rbf: unfolds the nonlinear structure of the swiss roll.
   - linear: collapses to ordinary PCA, cannot unroll.
   - polynomial: bends the feature space but does not cleanly unroll.

## Build approach

- All four built together. Dispatch four `manim-animator` agents in parallel,
  one per algorithm, each given: this spec, the relevant
  `js/manifold/algorithms/<algo>.js` step structure, and
  `manimexp/pca/walkthrough.py` as the reference template. The KPCA agent
  produces the 3-kernel `MFI_KERNEL` parametrization.
- `manifold-math-verifier` supplies exact eigenvalues/spectra for any on-screen
  worked numbers so they match `linalg.js` / the JS algorithms.
- `motion-design-reviewer` reviews each rendered clip; the animator applies the
  fixes.
- Render scripts follow the existing `manimexp/render.sh` pattern: full-quality
  `-qh --fps 60 --save_sections`, then per-step section MP4s remuxed with
  `+faststart`, plus poster frames, copied to `assets/manim/<algo>/`. KPCA
  yields three variant clips (`walkthrough-rbf.mp4`, etc., and matching
  per-step clips if needed by the player).

## Out of scope (separate follow-up)

Wiring the new clips into the Manifold Learning page player. Done after the
clips are rendered and approved, unless requested earlier.

## Success criteria

- Four new `manimexp/<algo>/walkthrough.py` modules render without error at both
  preview (`MFI_N=120 -ql`) and full (`-qh`, N=1000) quality.
- On-screen math matches the corresponding JS algorithm and `linalg.js`.
- Clips are visually consistent with the Isomap/PCA clips (same palette, timing,
  caption style, continuous camera).
- Rendered outputs land in `assets/manim/{mds,lle,laplacian,kpca}/` with
  per-step clips, posters, and a combined walkthrough; KPCA has one set per
  kernel.
