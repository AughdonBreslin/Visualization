# Manifold Walkthroughs (MDS / LLE / Laplacian / Kernel PCA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add manim step-by-step walkthroughs for MDS, LLE, Laplacian Eigenmaps, and Kernel PCA (rbf/polynomial/linear), all on the swiss roll, reusing the Isomap explainer's visual system.

**Architecture:** Shared infrastructure first (a `heatmap` builder and per-algorithm math helpers in the isomap `data`/`builders` modules, all unit-tested against the JS algorithms), then four `manimexp/<algo>/walkthrough.py` scenes built in parallel by manim-animator agents off the PCA template, then render scripts, math verification, and a motion review pass.

**Tech Stack:** Python 3 + manim 0.18.1 (`manimexp/.venv`), numpy, unittest; ffmpeg/ffprobe for remux/posters. Reference: `manimexp/pca/walkthrough.py`, `manimexp/isomap/*`, `js/manifold/algorithms/*.js`.

**Spec:** `docs/2026-06-08-manifold-walkthroughs-design.md`

**Dependency order:** Task 1 and Task 2 (shared infra) MUST complete before Tasks 3-6. Tasks 3-6 are independent and run in parallel. Task 7 (render) follows each scene. Task 8 (motion review) is last.

---

## Task 1: Shared `heatmap` builder

**Files:**
- Modify: `manimexp/isomap/builders.py`
- Test: `manimexp/tests/test_builders.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# manimexp/tests/test_builders.py
import unittest
import numpy as np
from manimexp.isomap import builders as B


class TestHeatmap(unittest.TestCase):
    def test_heatmap_downsamples_to_cell_budget(self):
        M = np.random.RandomState(0).rand(100, 100)
        hm = B.heatmap(M, 100, max_cells=32)
        # One square per displayed cell; downsampled grid is at most 32x32.
        rows = hm.meta["rows"]
        cols = hm.meta["cols"]
        self.assertLessEqual(rows, 32)
        self.assertLessEqual(cols, 32)
        self.assertEqual(len(hm.submobjects), rows * cols)

    def test_heatmap_small_matrix_is_exact(self):
        M = np.array([[0.0, 1.0], [1.0, 0.0]])
        hm = B.heatmap(M, 2, max_cells=32)
        self.assertEqual(hm.meta["rows"], 2)
        self.assertEqual(hm.meta["cols"], 2)
        self.assertEqual(len(hm.submobjects), 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/c/Users/aughb/PersonalProjects/Visualization && PYTHONPATH=. manimexp/.venv/bin/python -m unittest manimexp.tests.test_builders -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'heatmap'`

- [ ] **Step 3: Implement `heatmap` in `builders.py`**

Add to `manimexp/isomap/builders.py` (uses existing imports `VGroup`, `Square`/`Rectangle`; add `Square` and `interpolate_color` to the manim import if not present, plus `numpy as np`):

```python
def heatmap(matrix, n, max_cells=32, cell=0.12, cmap=None, diverging=False):
    """Render an n x n matrix as a downsampled grid of colored squares.

    The matrix is block-averaged down to at most max_cells x max_cells so it
    stays legible and cheap at large n. Colors run on a sequential ramp
    (MUTED -> ACCENT) or, when diverging=True, WARM (negative) -> BG -> GOOD
    (positive). Returns a VGroup with .meta = {rows, cols, vmin, vmax}.
    """
    import numpy as np
    M = np.asarray(matrix, dtype=float).reshape(n, n)
    step = max(1, int(np.ceil(n / max_cells)))
    rows = int(np.ceil(n / step))
    cols = rows
    # Block-mean downsample.
    ds = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            block = M[r*step:(r+1)*step, c*step:(c+1)*step]
            ds[r, c] = float(block.mean()) if block.size else 0.0
    vmin, vmax = float(ds.min()), float(ds.max())
    grid = VGroup()
    for r in range(rows):
        for c in range(cols):
            v = ds[r, c]
            if diverging:
                m = max(abs(vmin), abs(vmax)) or 1.0
                if v >= 0:
                    col = interpolate_color(S.BG, S.GOOD, min(1.0, v / m))
                else:
                    col = interpolate_color(S.BG, S.WARM, min(1.0, -v / m))
            else:
                rng = (vmax - vmin) or 1.0
                col = interpolate_color(S.MUTED, S.ACCENT, (v - vmin) / rng)
            sq = Square(side_length=cell, fill_color=col, fill_opacity=1.0,
                        stroke_width=0)
            sq.move_to([c * cell, -r * cell, 0])
            grid.add(sq)
    grid.move_to([0, 0, 0])
    grid.meta = {"rows": rows, "cols": cols, "vmin": vmin, "vmax": vmax}
    return grid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/c/Users/aughb/PersonalProjects/Visualization && PYTHONPATH=. manimexp/.venv/bin/python -m unittest manimexp.tests.test_builders -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add manimexp/isomap/builders.py manimexp/tests/test_builders.py
git commit -m "manifold: shared heatmap builder for NxN matrix steps"
```

---

## Task 2: Per-algorithm math helpers in `data.py`

These reproduce the JS math so on-screen numbers are correct. Each gets a unit test asserting agreement with the defining formula (the manifold-math-verifier agent confirms agreement with `js/manifold/algorithms/*.js`).

**Files:**
- Modify: `manimexp/isomap/data.py`
- Test: `manimexp/tests/test_algo_math.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
# manimexp/tests/test_algo_math.py
import unittest
import numpy as np
from manimexp.isomap import data


class TestAlgoMath(unittest.TestCase):
    def setUp(self):
        self.pts = data.swiss_roll(n=60, seed=0)["points"]

    def test_heat_affinity_and_laplacian(self):
        adj, edges = data.knn_graph(self.pts, k=8)
        W = data.heat_affinity(self.pts, edges, sigma=1.0)
        np.testing.assert_allclose(W, W.T)
        self.assertTrue(np.all(np.diag(W) == 0))
        L, Ddeg = data.graph_laplacian(W)
        np.testing.assert_allclose(np.diag(Ddeg), W.sum(axis=1))
        np.testing.assert_allclose(L, Ddeg - W)

    def test_lle_weights_reconstruct(self):
        adj, edges = data.knn_graph(self.pts, k=8)
        Wlle = data.lle_weights(self.pts, k=8, reg=1e-3)
        # rows sum to 1, weights only on neighbors, reconstruction is close.
        np.testing.assert_allclose(Wlle.sum(axis=1), np.ones(len(self.pts)), atol=1e-6)
        recon = Wlle @ self.pts
        self.assertLess(np.linalg.norm(recon - self.pts) / len(self.pts), 0.5)

    def test_kernel_matrix_variants(self):
        Krbf = data.kernel_matrix(self.pts, "rbf", gamma=1.0)
        np.testing.assert_allclose(np.diag(Krbf), np.ones(len(self.pts)))
        Klin = data.kernel_matrix(self.pts, "linear")
        np.testing.assert_allclose(Klin, self.pts @ self.pts.T)
        Kc = data.center_kernel(Krbf)
        np.testing.assert_allclose(Kc, Kc.T, atol=1e-9)

    def test_bottom_eigvecs_skips_trivial(self):
        adj, edges = data.knn_graph(self.pts, k=8)
        W = data.heat_affinity(self.pts, edges, sigma=1.0)
        L, _ = data.graph_laplacian(W)
        vecs, vals = data.bottom2_eig(L)
        self.assertEqual(vecs.shape, (len(self.pts), 2))
        # eigenvalues are the two smallest non-trivial (>~0), ascending.
        self.assertLessEqual(vals[0], vals[1] + 1e-9)
        self.assertGreater(vals[0], 1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/aughb/PersonalProjects/Visualization && PYTHONPATH=. manimexp/.venv/bin/python -m unittest manimexp.tests.test_algo_math -v`
Expected: FAIL with `AttributeError` for the new helpers.

- [ ] **Step 3: Implement the helpers in `data.py`**

Add these functions (match `js/manifold/algorithms/{laplacian,lle,kpca}.js`; for rbf, gamma is auto-scaled to mean squared pairwise distance exactly as the JS does):

```python
def heat_affinity(points, edges, sigma=1.0):
    """W_ij = exp(-||x_i - x_j||^2 / (2 sigma^2)) on kNN edges, symmetric, else 0."""
    n = len(points)
    W = np.zeros((n, n))
    for (i, j) in edges:
        d2 = float(np.sum((points[i] - points[j]) ** 2))
        w = np.exp(-d2 / (2.0 * sigma * sigma))
        W[i, j] = w
        W[j, i] = w
    return W


def graph_laplacian(W):
    """Return (L, D) with D = diag(row sums), L = D - W."""
    deg = W.sum(axis=1)
    D = np.diag(deg)
    return D - W, D


def lle_weights(points, k=8, reg=1e-3):
    """Per-point constrained least-squares reconstruction weights (rows sum to 1)."""
    n = len(points)
    adj, _ = knn_graph(points, k=k)
    W = np.zeros((n, n))
    for i in range(n):
        nbrs = np.where(adj[i] > 0)[0]
        if len(nbrs) == 0:
            continue
        Z = points[nbrs] - points[i]          # k x 3
        G = Z @ Z.T                            # local Gram
        G = G + reg * np.trace(G) * np.eye(len(nbrs)) if np.trace(G) > 0 else G + reg * np.eye(len(nbrs))
        w = np.linalg.solve(G, np.ones(len(nbrs)))
        w = w / w.sum()
        W[i, nbrs] = w
    return W


def lle_matrix(W):
    """M = (I - W)^T (I - W)."""
    n = W.shape[0]
    ImW = np.eye(n) - W
    return ImW.T @ ImW


def kernel_matrix(points, kernel="rbf", gamma=1.0, degree=3, constant=1.0):
    """rbf/polynomial/linear kernel matrix. rbf gamma auto-scales like the JS."""
    n = len(points)
    if kernel == "rbf":
        c = points.mean(axis=0)
        mean_sq = (2.0 * float(np.sum((points - c) ** 2)) / n) or 1.0
        g = gamma / mean_sq
        sq = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
        return np.exp(-g * sq)
    dot = points @ points.T
    if kernel == "polynomial":
        return (dot + constant) ** degree
    return dot


def center_kernel(K):
    """Kc = K - 1_N K - K 1_N + 1_N K 1_N (double-centering of the kernel)."""
    n = K.shape[0]
    row = K.mean(axis=1, keepdims=True)
    grand = float(K.mean())
    return K - row - row.T + grand


def bottom2_eig(M):
    """Two smallest non-trivial eigenpairs (skip the near-zero eigenvalue)."""
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    # skip eigenvalues numerically equal to zero (constant eigenvector)
    keep = [idx for idx in range(len(vals)) if vals[idx] > 1e-9]
    sel = keep[:2] if len(keep) >= 2 else order[1:3]
    return vecs[:, sel], vals[sel]


def top2_kernel_embed(Kc):
    """Top-2 eigenpairs of centered kernel; embedding y_{i,k} = sqrt(l_k) v_{k,i}."""
    vals, vecs = np.linalg.eigh(Kc)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order][:2], vecs[:, order][:, :2]
    Y = vecs * np.sqrt(np.maximum(vals, 0.0))
    return Y, vecs, vals
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/Users/aughb/PersonalProjects/Visualization && PYTHONPATH=. manimexp/.venv/bin/python -m unittest manimexp.tests.test_algo_math -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Verify against the JS with the math verifier**

Dispatch the `manifold-math-verifier` agent: confirm `heat_affinity`, `graph_laplacian`, `lle_weights`/`lle_matrix`, `kernel_matrix` (all three kernels, including rbf gamma auto-scaling), `center_kernel`, `bottom2_eig`, and `top2_kernel_embed` produce values matching `js/manifold/algorithms/{laplacian,lle,kpca}.js` and `js/manifold/linalg.js` on swiss_roll. Fix any mismatch, re-run tests.

- [ ] **Step 6: Commit**

```bash
git add manimexp/isomap/data.py manimexp/tests/test_algo_math.py
git commit -m "manifold: data helpers for MDS/LLE/Laplacian/KPCA walkthroughs"
```

---

## Task 3: MDS walkthrough  (parallel — manim-animator agent)

**Files:**
- Create: `manimexp/mds/__init__.py` (empty)
- Create: `manimexp/mds/walkthrough.py` — class `MDSWalkthrough(ThreeDScene)`

**Brief for the animator agent:**
Copy `manimexp/pca/walkthrough.py` as the structural template (same overlay
helpers `set_caption`/`set_pseudo`, same corner formula placement, same
continuous ambient rotation that only flattens to phi=0 for the embedding).
Sections, ids matching the JS `presentSubSteps` of `mds.js` (0,3,4,5,6):

1. `step-1-raw`: axes + `B.point_cloud`, intro caption, begin ambient rotation.
2. `step-3-distances`: draw `B.straight_line` between a handful of sample point
   pairs in 3D; add `B.heatmap(D, N)` as a fixed-in-frame panel (corner).
   Formula `D_{ij} = \| x_i - x_j \|`.
3. `step-4-double-center`: heatmap of `D^2` transform to heatmap of
   `B = data.double_center_squared(D)`; formula
   `B = -\tfrac12 H D^2 H,\ H = I - \tfrac1N \mathbf{1}\mathbf{1}^\top`.
4. `step-5-eig`: `data.top2_eig(B)`; show top-2 eigenvalue readout.
5. `step-6-embedding`: `Y = [v1 v2] diag(sqrt l1, sqrt l2)`, scale to frame,
   morph cloud into world xy-plane, move camera to phi=0 face-on (shortest-path
   theta like the PCA file). Outro caption: MDS preserves straight-line
   distances, so the rolled layers still overlap; this is what Isomap fixes by
   using geodesic distances.

Captions: measured prose, no em-dashes, no em/strong, motivate what/why/how.

- [ ] **Step 1:** Write `manimexp/mds/__init__.py` (empty) and `walkthrough.py`.
- [ ] **Step 2:** Preview render: `cd <repo> && MFI_N=120 PYTHONPATH=. manimexp/.venv/bin/manim -ql --fps 30 --save_sections manimexp/mds/walkthrough.py MDSWalkthrough`. Expected: renders without error, 5 section MP4s under `media/videos/walkthrough/...`.
- [ ] **Step 3:** Eyeball the preview MP4s for correct staging.
- [ ] **Step 4: Commit**

```bash
git add manimexp/mds/
git commit -m "manifold: MDS manim walkthrough"
```

---

## Task 4: LLE walkthrough  (parallel — manim-animator agent)

**Files:**
- Create: `manimexp/lle/__init__.py` (empty)
- Create: `manimexp/lle/walkthrough.py` — class `LLEWalkthrough(ThreeDScene)`

**Brief:** PCA template + isomap kNN visuals (`B.knn_sphere`, `B.graph_edges`).
Sections matching `lle.js` `presentSubSteps` (0,2,3,5,6):

1. `step-1-raw`: cloud + intro caption, begin ambient rotation.
2. `step-2-knn`: kNN graph via `data.knn_graph(pts, k=8)` + `B.graph_edges`;
   highlight one point's `B.knn_sphere` neighborhood. Caption: local neighbors.
3. `step-3-weights`: pick one point; draw weighted arrows from its k neighbors
   (use `data.lle_weights`), label a few weights; optional sparse
   `B.heatmap(Wlle, N)` panel. Formula
   `\min \| x_i - \sum_j w_j x_{n_j} \|^2,\ \sum_j w_j = 1`.
4. `step-5-eig`: `data.lle_matrix(Wlle)`; `data.bottom2_eig(M)`; caption about
   smallest non-trivial eigenvectors (skip lambda_0 = 0).
5. `step-6-embedding`: `Y = [v1 v2]`, morph to 2D face-on. Outro: LLE preserves
   local linear reconstructions, so the sheet unrolls flat.

- [ ] **Step 1:** Write `__init__.py` + `walkthrough.py`.
- [ ] **Step 2:** Preview: `MFI_N=120 PYTHONPATH=. manimexp/.venv/bin/manim -ql --fps 30 --save_sections manimexp/lle/walkthrough.py LLEWalkthrough`. Expected: renders, 5 sections.
- [ ] **Step 3:** Eyeball preview.
- [ ] **Step 4: Commit**

```bash
git add manimexp/lle/
git commit -m "manifold: LLE manim walkthrough"
```

---

## Task 5: Laplacian Eigenmaps walkthrough  (parallel — manim-animator agent)

**Files:**
- Create: `manimexp/laplacian/__init__.py` (empty)
- Create: `manimexp/laplacian/walkthrough.py` — class `LaplacianWalkthrough(ThreeDScene)`

**Brief:** PCA template + isomap kNN visuals. Sections matching `laplacian.js`
`presentSubSteps` (0,2,3,4,5,6):

1. `step-1-raw`: cloud + intro, begin ambient rotation.
2. `step-2-knn`: kNN graph.
3. `step-3-affinity`: recolor kNN edges by `data.heat_affinity(pts, edges, sigma)`
   weight (strong=GOOD, weak=MUTED) + `B.heatmap(W, N)` panel. Formula
   `W_{ij} = \exp(-\| x_i - x_j \|^2 / 2\sigma^2)`.
4. `step-4-laplacian`: `data.graph_laplacian(W)` -> show `B.heatmap(W)`,
   `B.heatmap(D)`, `B.heatmap(L)`; formula `L = D - W,\ D_{ii} = \sum_j W_{ij}`.
5. `step-5-eig`: `data.bottom2_eig(L)`; smallest non-trivial eigenvectors.
6. `step-6-embedding`: `Y = [v1 v2]`, morph to 2D face-on. Outro: nearby points
   stay nearby, so the sheet flattens while preserving locality.

- [ ] **Step 1:** Write `__init__.py` + `walkthrough.py`.
- [ ] **Step 2:** Preview: `MFI_N=120 PYTHONPATH=. manimexp/.venv/bin/manim -ql --fps 30 --save_sections manimexp/laplacian/walkthrough.py LaplacianWalkthrough`. Expected: renders, 6 sections.
- [ ] **Step 3:** Eyeball preview.
- [ ] **Step 4: Commit**

```bash
git add manimexp/laplacian/
git commit -m "manifold: Laplacian Eigenmaps manim walkthrough"
```

---

## Task 6: Kernel PCA walkthrough (3 kernels)  (parallel — manim-animator agent)

**Files:**
- Create: `manimexp/kpca/__init__.py` (empty)
- Create: `manimexp/kpca/walkthrough.py` — class `KPCAWalkthrough(ThreeDScene)`

**Brief:** PCA template, parametrized by `MFI_KERNEL` in {rbf, polynomial,
linear} (read at module top like `MFI_DATASET`). Sections matching `kpca.js`
`presentSubSteps` (0,3,4,5,6):

1. `step-1-raw`: cloud; intro caption names the active kernel.
2. `step-3-kernel`: `data.kernel_matrix(pts, KERNEL, ...)`; `B.heatmap(K, N)`;
   kernel formula chosen by KERNEL (rbf `\exp(-\gamma \| x_i - x_j \|^2)`,
   polynomial `(x_i \cdot x_j + 1)^d`, linear `x_i \cdot x_j`).
3. `step-4-center`: `data.center_kernel(K)`; heatmap K -> Kc; formula
   `K_c = K - \mathbf{1}_N K - K \mathbf{1}_N + \mathbf{1}_N K \mathbf{1}_N`.
4. `step-5-eig`: `data.top2_kernel_embed(Kc)` top-2; eigenvalue readout.
5. `step-6-embedding`: `y_{i,k} = \sqrt{\lambda_k}\, v_{k,i}`, morph to 2D
   face-on. Per-kernel outro from a dict: rbf unfolds the nonlinear structure;
   linear collapses to ordinary PCA and cannot unroll; polynomial bends the
   feature space but does not cleanly unroll.

- [ ] **Step 1:** Write `__init__.py` + `walkthrough.py` with the `MFI_KERNEL` switch.
- [ ] **Step 2:** Preview all three:
```bash
for k in rbf polynomial linear; do MFI_N=120 MFI_KERNEL=$k PYTHONPATH=. manimexp/.venv/bin/manim -ql --fps 30 --save_sections manimexp/kpca/walkthrough.py KPCAWalkthrough; done
```
Expected: each renders without error, 5 sections.
- [ ] **Step 3:** Eyeball each kernel's preview; confirm embeddings differ as expected.
- [ ] **Step 4: Commit**

```bash
git add manimexp/kpca/
git commit -m "manifold: Kernel PCA manim walkthrough (rbf/polynomial/linear)"
```

---

## Task 7: Render scripts (full quality + faststart + posters)

**Files:**
- Create: `manimexp/render_mds.sh`, `manimexp/render_lle.sh`, `manimexp/render_laplacian.sh`, `manimexp/render_kpca.sh`

Each mirrors `manimexp/render.sh`: `-qh --fps 60 --save_sections
--disable_caching`, locate `media/videos/walkthrough/.../sections`, remux each
section MP4 with `-movflags +faststart`, extract a mid-clip poster, copy to
`assets/manim/<algo>/step-N.mp4|png`, and also copy the combined
`walkthrough.mp4`. The KPCA script loops `MFI_KERNEL` over rbf/polynomial/linear
and writes to `assets/manim/kpca/<kernel>/`.

- [ ] **Step 1:** Write the four render scripts; `chmod +x` them.
- [ ] **Step 2:** Run each (full quality, slow). Expected: per-step + walkthrough MP4s and posters land in `assets/manim/{mds,lle,laplacian,kpca}/`.
- [ ] **Step 3:** Verify with `ffprobe` that each output has its moov atom at the front (faststart) and plays in the browser at http://localhost:8770.
- [ ] **Step 4: Commit**

```bash
git add manimexp/render_*.sh assets/manim/mds assets/manim/lle assets/manim/laplacian assets/manim/kpca
git commit -m "manifold: render scripts and clips for MDS/LLE/Laplacian/KPCA"
```

---

## Task 8: Motion review pass

- [ ] **Step 1:** For each rendered walkthrough, dispatch the `motion-design-reviewer` agent (timing, easing, staging, continuity, legibility, consistency with Isomap/PCA clips). Collect concrete fixes.
- [ ] **Step 2:** The manim-animator applies the fixes; re-render the affected clips.
- [ ] **Step 3:** Confirm consistency across all four algorithms (palette, caption style, camera behavior).
- [ ] **Step 4: Commit** any fixes.

```bash
git add -A && git commit -m "manifold: motion-review fixes for new walkthroughs"
```

---

## Self-review notes

- Spec coverage: heatmap builder (Task 1), per-algo math (Task 2), four scenes
  (Tasks 3-6), render + faststart/posters (Task 7), math verification (Task 2
  Step 5), motion review (Task 8). KPCA 3-kernel parametrization (Task 6) and
  swiss_roll default with env parametrization (all scene tasks) covered.
- Page-player integration is explicitly out of scope per the spec.
- Helper names are consistent across tasks: `heat_affinity`, `graph_laplacian`,
  `lle_weights`/`lle_matrix`, `kernel_matrix`, `center_kernel`, `bottom2_eig`,
  `top2_kernel_embed`, `top2_eig` (existing), `double_center_squared` (existing),
  `knn_graph` (existing), and `B.heatmap`.
