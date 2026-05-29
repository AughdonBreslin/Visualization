# Manifold Page: Phase 2 Design

Date: 2026-05-29

## Goal

Extend the manifold learning page with four new algorithms (MDS, LLE, Laplacian Eigenmaps, Kernel PCA), eleven new synthetic datasets, one new visualization pattern for LLE's reconstruction weights, and one new linear algebra helper for smallest-eigenpair extraction. The work is split into two sub-phases that ship as independent PRs: 2a covers algorithms and infrastructure; 2b covers the datasets.

The existing phase 1 contracts stay frozen: algorithm step states still carry `vizKind`, the IFW worked example is still three labelled sections built via `workedSections`, pseudocode is still per-algorithm sections, datasets still expose `generate({samples, noise, seed}) returning {X, t, N}`, and the Web Worker still drives the algorithm pipelines.

## User constraints

- No em-dashes anywhere.
- No `<em>`, `<strong>`, `<b>`, `<i>`, `<mark>` HTML tags.
- No markdown emphasis (`*`, `**`, `_`, `__`).

Applies to all generated code, comments, commit messages, spec text.

## Scope

In:

- Linalg: `bottomKSymmetricEig(M, N, k, { skipFirst })` for LLE and Laplacian.
- Algorithm modules: MDS, LLE, Laplacian Eigenmaps, Kernel PCA. Each fits the existing algorithm contract.
- New viz: `viz_weighted_knn` for LLE step 3 (reconstruction weights).
- Datasets: severed_sphere, punctured_sphere, helix, twin_peaks, trefoil_knot, toroidal_helix, clusters_3d, full_sphere, saddle, cylinder, spiral_disk.
- UI: extend the algorithm parameter renderer to support an `enum` parameter type (rendered as a `<select>`) so Kernel PCA can switch between RBF, polynomial, and linear kernels.
- File split: `js/manifold/datasets.js` is decomposed into `js/manifold/datasets/` with one file per dataset family plus an index that re-exports the existing public surface.

Out (phase 3):

- t-SNE, UMAP, Sammon Mapping, Diffusion Maps, LTSA.
- Topologically pathological datasets: Mobius strip, Klein bottle slice, Roman surface, figure-8 immersion.
- Global UI for coloring policy. Phase 2 algorithms decide per step whether to emit `colors`; no user-facing toggle is added.

## Sub-phase split

Phase 2a (algorithms and infrastructure):

- New file: `js/manifold/linalg.js` (modified) — `bottomKSymmetricEig` helper plus a small unit test.
- New algorithm modules: `mds.js`, `lle.js`, `laplacian.js`, `kpca.js`.
- New viz: `viz_weighted_knn.js`.
- Updated dispatcher: `step_viz.js` adds the `'weighted_knn'` branch.
- Updated entry: `main.js` registers the four new algorithms and extends `renderParamHost` for the `enum` param type.
- Updated worker: `worker.js` registers the four new algorithms so the worker pipeline runs them.
- Test sweep on phase 1 datasets only (Swiss roll, S-curve, CSV). All four new algorithms must produce visually correct results on at least Swiss roll and S-curve.

Phase 2b (datasets):

- File split: `js/manifold/datasets/` directory with `index.js`, `synthetic_curves.js`, `synthetic_surfaces.js`, `synthetic_clusters.js`, `csv_upload.js`. The existing `js/manifold/datasets.js` becomes a thin shim that re-exports from `datasets/index.js` so no consumers need updating.
- Eleven new dataset generators (see Surface 4 for the table).
- New tests under `test/manifold/datasets_phase2.test.js`: each new dataset produces the right shape and is deterministic for a given seed.

Each sub-phase has its own plan document, ships as its own PR, and can be reviewed independently. Phase 2b depends only on the existing dataset contract, so it does not block on 2a.

## Coloring policy (settled from phase 1 Task 4)

Default rainbow-by-intrinsic-parameter at every step unless the algorithm or the dataset explicitly sets `colors` on the step state. Phase 2 introduces two new override sites:

- LLE step 3 (reconstruction weights): the renderer overlays per-edge thickness from W; no `colors` override is required.
- `clusters_3d` dataset: emits a `colors` array keyed by cluster index. This colour applies to every step's `colors` field on both algorithms because clusters are an intrinsic property of the dataset, not an algorithm-driven coloring.

No other phase 2 algorithm emits `colors`. Isomap's existing per-eigenvector recoloring inside the spectral viz stays as-is and is reused by MDS, LLE, and Laplacian (they pass `topEigvecs` so the eigenvalue mini hover recolors the cloud).

## Surface 1: Linalg additions

`bottomKSymmetricEig(M, N, k, { skipFirst = 0 }) → { lambda: Float64Array(k), vectors: Float64Array[] }`

Returns the k smallest eigenpairs of M after skipping the `skipFirst` smallest. Used for LLE (`skipFirst = 1` because M has a trivial zero eigenvalue) and Laplacian Eigenmaps (same trivial-eigenvalue skip).

Implementation strategy:

1. Estimate the largest eigenvalue of M by a few power iterations on the unmodified M. Call the estimate `mu`.
2. Build the shifted matrix `S = mu * I - M`. The smallest eigenvalues of M correspond to the largest eigenvalues of S.
3. Call the existing `topKSymmetricEig(S, N, k + skipFirst)` and translate back: `lambda_k(M) = mu - lambda_k(S)`. The eigenvectors are identical.
4. Drop the first `skipFirst` entries from the returned arrays.

Falls back to the existing `jacobiEigSym` for very small N (less than 16), reading the smallest entries directly.

Test (`test/manifold/linalg_bottomeig.test.js`): construct a small 5x5 symmetric matrix with known eigenvalues 0, 1, 3, 5, 7. Confirm `bottomKSymmetricEig(M, 5, 2, { skipFirst: 1 })` returns eigenvalues approximately 1 and 3 (with vectors whose `Mv = lambda v` holds to 1e-6).

## Surface 2: Algorithm modules

All four modules export an object with the same shape as PCA and Isomap. Pseudocode follows the existing format. Worked-example HTML uses `workedSections` and `format.js` helpers (`formatVec3`, `formatMatrix`, `formatTable`). The numbers in the worked example are computed from actual dataset values inside each step's task callback.

### MDS (`js/manifold/algorithms/mds.js`)

Export: `MDS = { id: 'mds', label: 'MDS', params: [], presentSubSteps: ['0','3','4','5','6'], pseudocode, run }`.

Steps:

- 0: `vizKind: 'point_cloud'`.
- 3 (pairwise Euclidean distances): `vizKind: 'matrix_strip'`, panes `[{kind:'cloud_thumb', data:X}, {kind:'heatmap', data:{matrix:D, N}}]`, paneOpLabels `['||x_i - x_j||']`. Worked example: a 4x4 excerpt of D plus one element computation.
- 4 (double-center): `vizKind: 'matrix_strip'`, panes `[{kind:'heatmap', data:{matrix:D2,N}}, {kind:'heatmap', data:{matrix:D2c,N}}, {kind:'heatmap', data:{matrix:B,N}}]`, paneOpLabels `['subtract row/col means', 'x (-1/2) + grand mean']`. Same shape as Isomap step 4.
- 5: `vizKind: 'spectral'`, `algoId: 'mds'`, `topEigvals: lambda` (length 8), `topEigvecs: vectors`. Renderer reuses Isomap's spectral path. Worked example shows top-2 eigenvalues and the first 6 entries of v_1 and v_2.
- 6: `vizKind: 'embedding'`, `embed2d = V[:, 0:2] * sqrt(diag(lambda[0:2]))`.

### LLE (`js/manifold/algorithms/lle.js`)

Export: `LLE = { id: 'lle', label: 'LLE', params: [...], presentSubSteps: ['0','2','3','5','6'], pseudocode, run }`.

Params: `[{name:'k', type:'int', default:10, min:2, max:50}, {name:'reg', type:'float', default:1e-3, min:0, max:0.1}]`.

Steps:

- 0: `vizKind: 'point_cloud'`.
- 2 (kNN graph): `vizKind: 'knn_graph'`, same payload as Isomap step 2.
- 3 (reconstruction weights): `vizKind: 'weighted_knn'`. State carries `points`, `edges` (the kNN graph), `W` (NxN sparse row-wise), `selectedPoint` (default `floor(N * 0.2)`). Worked example: for the selected point, show its k neighbours, the reconstruction weights computed, and the local error.
- 5 (eigendecompose M = (I - W)^T (I - W)): `vizKind: 'spectral'`, `algoId: 'lle'`. Use `bottomKSymmetricEig(M, N, 8, { skipFirst: 1 })`. The embedding step reads indices 0 and 1 (the two smallest non-trivial eigenpairs); the spectral mini bar chart shows all 8 returned eigenvalues with the first two highlighted. Worked example shows the two smallest non-trivial eigenvalues and the first 6 entries of v_1 and v_2. The `topEigvecs` and `topEigvals` state fields are populated with these 8 bottom eigenpairs so the existing Isomap-style mini-hover recolour works on the bottom spectrum.
- 6: `vizKind: 'embedding'`, `embed2d = V[:, 0:2]` (no eigenvalue scaling for LLE).

Weight computation (step 3): for each point i, solve `min_w ||x_i - sum_j w_j x_{n_j}||^2` subject to `sum w_j = 1` using the standard local Gram matrix approach: `G = (X_n - x_i 1^T)^T (X_n - x_i 1^T)` where `X_n` is the kxd matrix of i's neighbours. Add `reg * trace(G) / k * I` for numerical stability. Solve `G w = 1`, then normalize `w = w / sum(w)`. Store `W[i][n_j] = w_j` for each neighbour.

### Laplacian Eigenmaps (`js/manifold/algorithms/laplacian.js`)

Export: `LAPLACIAN = { id: 'laplacian', label: 'Laplacian Eigenmaps', params: [...], presentSubSteps: ['0','2','3','4','5','6'], pseudocode, run }`.

Params: `[{name:'k', type:'int', default:10, min:2, max:50}, {name:'sigma', type:'float', default:1.0, min:0.1, max:10}]`.

Steps:

- 0: `vizKind: 'point_cloud'`.
- 2 (kNN graph): `vizKind: 'knn_graph'`.
- 3 (heat-kernel affinity W): `vizKind: 'matrix_strip'`, panes `[{kind:'graph_thumb', data:{points,edges}}, {kind:'heatmap', data:{matrix:W,N}}]`, paneOpLabels `['W_ij = exp(-||x_i - x_j||^2 / (2 sigma^2))']`. W is sparse: only kNN edges get non-zero entries.
- 4 (Laplacian L = D - W): `vizKind: 'matrix_strip'`, panes `[{kind:'heatmap', data:{matrix:W,N}}, {kind:'heatmap', data:{matrix:D_diag,N}}, {kind:'heatmap', data:{matrix:L,N}}]`, paneOpLabels `['row sums = D', 'D - W = L']`. D is shown as a diagonal matrix (other entries zero).
- 5 (smallest non-trivial eigenvectors): `vizKind: 'spectral'`, `algoId: 'laplacian'`. Use `bottomKSymmetricEig(L, N, 8, { skipFirst: 1 })`; the embedding reads indices 0 and 1, the mini bar chart shows all 8, and `topEigvecs` / `topEigvals` carry them for the mini hover recolour. Worked example as for LLE.
- 6: `vizKind: 'embedding'`, `embed2d = V[:, 0:2]`.

### Kernel PCA (`js/manifold/algorithms/kpca.js`)

Export: `KPCA = { id: 'kpca', label: 'Kernel PCA', params: [...], presentSubSteps: ['0','3','4','5','6'], pseudocode, run }`.

Params: `[{name:'kernel', type:'enum', options:['rbf','polynomial','linear'], default:'rbf'}, {name:'gamma', type:'float', default:0.5, min:0.01, max:20}, {name:'degree', type:'int', default:3, min:1, max:10}, {name:'constant', type:'float', default:1, min:0, max:10}]`.

Kernel functions:

- `rbf`: `K(x,y) = exp(-gamma * ||x - y||^2)`.
- `polynomial`: `K(x,y) = (x . y + constant)^degree`.
- `linear`: `K(x,y) = x . y`.

Steps:

- 0: `vizKind: 'point_cloud'`.
- 3 (kernel matrix K): `vizKind: 'matrix_strip'`, panes `[{kind:'cloud_thumb', data:X}, {kind:'heatmap', data:{matrix:K,N}}]`. paneOpLabels carries the active kernel formula (different string per kernel choice). Worked example shows one K[i][j] computation with concrete numbers for the selected kernel.
- 4 (center K): `vizKind: 'matrix_strip'`, panes `[{kind:'heatmap', data:{matrix:K,N}}, {kind:'heatmap', data:{matrix:Kc,N}}]`, paneOpLabels `['K - 1_N K - K 1_N + 1_N K 1_N']`.
- 5: `vizKind: 'spectral'`, `algoId: 'kpca'`, top-2 via `topKSymmetricEig(Kc, N, 8)`.
- 6: `vizKind: 'embedding'`, `embed2d[i,k] = sqrt(lambda_k) * v_{k,i}` (same shape as Isomap step 6).

## Surface 3: New viz weighted_knn

`js/manifold/viz/viz_weighted_knn.js` exporting `mountWeightedKnn(container, state, { width, height }) → { unmount }`.

State shape (in addition to standard StepState fields):

- `points: Float64Array(3N)` (the cloud)
- `t: Float64Array(N)` (intrinsic for rainbow)
- `edges: [number, number][]` (full kNN graph, for the faint background)
- `W: Float64Array(N*N)` (sparse row-wise; the renderer reads `W[i*N + j]` for each kNN edge)
- `selectedPoint: number` (initial selection)
- `k: number` (the kNN parameter, for the legend)

Renderer behavior:

1. Computes the orbit camera and projects all points.
2. Draws all kNN edges in the same dim grey as `viz_knn`'s background edges.
3. Draws the selected point larger and brighter.
4. For each kNN edge `(selectedPoint, j)`, draws an additional bright stroke whose `stroke-width` is proportional to `|W[selectedPoint*N + j]|` (clamped to a sane range like 0.5 to 3.5 px).
5. Each non-selected node is hoverable; hover changes the selection (re-runs the bright-edge pass) without remounting.
6. Drag on background orbits. Drag on a node does not orbit (it lets the click select the node).

CSS rule appended: `.manifold .viz-weighted-knn { position: absolute; inset: 0; }`.

Dispatcher: `step_viz.js` gains a `'weighted_knn'` branch that mounts the renderer like `'knn_graph'`.

## Surface 4: Datasets (phase 2b)

Each dataset is one entry in the `DATASETS` array exported from the `datasets/` directory. Files:

- `synthetic_curves.js`: SWISS_ROLL, S_CURVE, HELIX, TREFOIL_KNOT, TOROIDAL_HELIX, SPIRAL_DISK.
- `synthetic_surfaces.js`: TWIN_PEAKS, SADDLE, CYLINDER, SEVERED_SPHERE, PUNCTURED_SPHERE, FULL_SPHERE.
- `synthetic_clusters.js`: CLUSTERS_3D.
- `csv_upload.js`: parseCSV, CSV_UPLOAD.
- `index.js`: imports from each and re-exports `DATASETS, DATASETS_BY_ID, parseCSV`.

The existing `js/manifold/datasets.js` becomes a one-line re-export shim of `datasets/index.js` so no consumer imports change.

Dataset table (extending phase 1):

| ID | Label | Shape | Extra params | Intrinsic t |
|----|-------|-------|---------------|-------------|
| `severed_sphere` | Severed sphere | sphere with polar cap removed | `cap: float, default 0.35` | longitude theta |
| `punctured_sphere` | Punctured sphere | uniform sphere minus a disk around the north pole | `holeRadius: float, default 0.4` | longitude theta |
| `helix` | Helix | x = cos(t), y = sin(t), z = t / (2 pi) over `turns` rotations | `turns: int, default 3` | t |
| `twin_peaks` | Twin peaks | z = sum of two Gaussian bumps on a 2D plane | none | x |
| `trefoil_knot` | Trefoil knot | x = sin(t) + 2 sin(2t), y = cos(t) - 2 cos(2t), z = - sin(3t) | none | t |
| `toroidal_helix` | Toroidal helix | helix wrapped q times around a torus of radius R and minor radius r | `q: int, default 7` | t |
| `clusters_3d` | 3D Gaussian clusters | `clusters` isotropic Gaussian blobs whose centers lie on a sphere of radius `sep` | `clusters: int, default 5; sep: float, default 2` | cluster index (returns `colors` array; rainbow disabled) |
| `full_sphere` | Full sphere | uniform on unit sphere | none | longitude theta |
| `saddle` | Saddle | z = x^2 - y^2 on x,y in [-1, 1] | none | x |
| `cylinder` | Cylinder | open cylinder of `height` | `height: float, default 2` | longitude theta |
| `spiral_disk` | Spiral disk | logarithmic spiral on z = 0 plane (tiny noise in z) | `turns: int, default 3` | t |

CLUSTERS_3D is the first dataset to emit a `colors: string[]` array. Renderers must use it when present (the existing viz3d already checks `state.colors` first). The color palette is a fixed list of 8 named colors (one per cluster), recycled if more clusters are requested.

Tests (`test/manifold/datasets_phase2.test.js`):

- For each new dataset: generate with `samples=20`, `seed=1`, confirm `X.length === 60`, `t.length === 20`, both are `Float64Array`.
- Determinism: same seed, two calls, byte-equal outputs.
- Clusters_3d only: generate with `clusters=3, samples=30`; confirm `colors` is a 30-element array of strings drawn from the palette and clusters partition the points roughly evenly.

## Surface 5: UI changes

`renderParamHost` in `main.js` extension for `enum` type:

```javascript
if (p.type === 'enum') {
  const sel = document.createElement('select');
  for (const opt of p.options) {
    const o = document.createElement('option');
    o.value = opt; o.textContent = opt;
    if ((current[p.name] || p.default) === opt) o.selected = true;
    sel.appendChild(o);
  }
  sel.addEventListener('change', () => {
    onChange({ ...current, [p.name]: sel.value });
  });
  wrap.appendChild(sel);
  return;
}
```

Algorithms with conditional parameters (KPCA's gamma is relevant only for RBF, degree and constant only for polynomial) show all parameters at all times. The user can ignore unused fields. A future enhancement could hide irrelevant params via a `dependsOn` field; this is deferred.

`ALGORITHMS` array in `main.js` and `ALGORITHMS` map in `worker.js` gain the four new algorithms.

## Open decisions for the implementation plans

Deliberately deferred to plan-writing time:

1. Whether to keep `js/manifold/datasets.js` as a shim or update all importers. Lean toward shim to minimize churn.
2. Exact `bottomKSymmetricEig` shift-deflate parameters (number of iterations for the shift estimate). Pin once the eigenvalue test is in place.
3. Whether the kernel formula labels for KPCA step 3 use LaTeX or plain ASCII. Phase 1 used plain ASCII in `paneOpLabels`; staying with that.
4. Whether `clusters_3d` distributes cluster centers via a deterministic Fibonacci-sphere or a seeded random walk. Pin in plan.

## Out of scope for phase 2

- t-SNE, UMAP, Sammon, Diffusion Maps, LTSA. These get their own brainstorm and spec.
- Mobius / Klein / Roman / figure-8 datasets.
- A global UI for selecting coloring source.
- Mobile or touch-specific gestures.
- Per-step animation toggles.
- Algorithm parameter validation UI feedback (the current uncontrolled `<input>` boxes stay).
