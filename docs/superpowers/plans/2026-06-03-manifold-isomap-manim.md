# Isomap Manim Explainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This project has dedicated agents in `.claude/agents/`: route tasks to them per the "Agent" line on each task, coordinated by `manifold-overseer`.

**Goal:** Ship a manim-first, animated, step-by-step Isomap explainer: one continuous 3Blue1Brown-style film rendered as six navigable per-step clips, played on a dedicated page, with the formula and worked numbers fused into each scene.

**Architecture:** A single manim `ThreeDScene` (`IsomapWalkthrough`) authored with `self.next_section()` per step, so objects persist and transform across step boundaries (true seamless continuity) while `--save_sections` emits one MP4 per step. Pure-Python/numpy/scipy math helpers (mirroring `js/manifold/linalg.js`) feed the scene and are unit-tested so on-screen numbers are provably correct. A static vanilla-JS page plays the per-step clips.

**Tech Stack:** Python 3.12, manim Community, numpy, scipy; ffmpeg + a LaTeX subset (MathTex); vanilla ES-module JS + MathJax for the page.

**Conventions (every task):**
- No em-dashes anywhere (prose, code, comments, captions, HTML). No emphasis tags (`<em>`, `<strong>`, `<b>`, `<i>`, `<mark>`).
- Measured, non-dramatic prose in captions and page copy.
- Spec: `docs/superpowers/specs/2026-06-03-manifold-isomap-manim-design.md`.
- Branch: `manifold-isomap-manim`. Work in the main checkout `/mnt/c/Users/aughb/PersonalProjects/Visualization`.
- Python tests: `python3 -m unittest discover -s manim/tests -v`.

---

## File Structure

- `manim/` (new Python render project, dev-time only)
  - `manim/requirements.txt` - pinned deps (manim, numpy, scipy).
  - `manim/isomap/__init__.py`
  - `manim/isomap/style.py` - palette, fonts, timing constants (one visual system).
  - `manim/isomap/data.py` - dataset + math (swiss roll, kNN, geodesics, double-center, eig, embedding) and 4x4 worked excerpts. Mirrors `js/manifold/linalg.js`.
  - `manim/isomap/builders.py` - manim mobject builders (point cloud, graph edges, matrix grid, formula, caption) using `style`.
  - `manim/isomap/walkthrough.py` - the single `IsomapWalkthrough` ThreeDScene with six sections.
  - `manim/tests/test_data.py` - unit tests for `data.py`.
  - `manim/render.sh` - render at 60fps/1080p with `--save_sections`, copy step clips and posters to `assets/manim/isomap/`.
  - `manim/README.md` - how to set up the toolchain and render.
- `assets/manim/isomap/` (new) - committed `step-1.mp4 ... step-6.mp4` and `step-1.png ... step-6.png`.
- `pages/manifold_isomap.html`, `js/manifold_isomap.js`, `styles/manifold_isomap.css` (new) - the explainer page + player.
- `index.html` (modify) - add the explainer to the project list.

---

## Task 1: Toolchain setup (gated, needs sudo)

**Agent:** manifold-overseer (coordination); the user runs the sudo command if prompted.

**Files:**
- Create: `manim/requirements.txt`
- Create: `manim/README.md`

- [ ] **Step 1: Create `manim/requirements.txt`**

```
manim==0.18.1
numpy>=1.26,<2.0
scipy>=1.11
```

- [ ] **Step 2: Create `manim/README.md`**

````markdown
# Isomap manim explainer (render project)

Dev-time only. Renders the Isomap walkthrough to MP4 clips committed under
`assets/manim/isomap/`. Nothing here runs at page-load time.

## One-time toolchain setup

System packages (needs sudo; run this yourself if apt prompts for a password):

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg texlive-latex-base texlive-latex-extra \
  texlive-fonts-recommended dvisvgm libcairo2-dev libpango1.0-dev pkg-config
```

Python deps (a venv is recommended):

```bash
python3 -m venv manim/.venv
source manim/.venv/bin/activate
pip install -r manim/requirements.txt
```

Verify:

```bash
python3 -c "import manim, numpy, scipy; print('manim', manim.__version__)"
```

## Render

```bash
bash manim/render.sh
```
````

- [ ] **Step 3: Install the toolchain**

Run (this needs sudo; if it prompts for a password, hand the apt line to the user via the `!` prefix):
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvisvgm libcairo2-dev libpango1.0-dev pkg-config
python3 -m venv manim/.venv && manim/.venv/bin/pip install -r manim/requirements.txt
```
Expected: ffmpeg, latex, and manim install without error.

- [ ] **Step 4: Verify manim renders a hello world**

Run:
```bash
manim/.venv/bin/python -c "import manim, numpy, scipy; print('manim', manim.__version__)"
printf 'from manim import *\nclass Hello(Scene):\n    def construct(self):\n        self.play(Write(Text("ok")))\n' > /tmp/hello_manim.py
manim/.venv/bin/manim -ql --fps 30 /tmp/hello_manim.py Hello -o hello.mp4 && echo RENDER_OK
```
Expected: prints the manim version and `RENDER_OK`, producing an MP4 under `media/`.

- [ ] **Step 5: Add the venv to gitignore and commit**

```bash
printf '\n# manim render venv and media\nmanim/.venv/\nmanim/media/\nmedia/\n' >> .gitignore
git add manim/requirements.txt manim/README.md .gitignore
git commit -m "manifold isomap: manim render project scaffold and toolchain docs"
```

---

## Task 2: Style module (one visual system)

**Agent:** manim-animator

**Files:**
- Create: `manim/isomap/__init__.py`
- Create: `manim/isomap/style.py`

- [ ] **Step 1: Create the package init**

```bash
mkdir -p manim/isomap manim/tests
printf '' > manim/isomap/__init__.py
printf '' > manim/tests/__init__.py
```

- [ ] **Step 2: Create `manim/isomap/style.py`**

```python
"""Shared visual system for the Isomap walkthrough: colors, fonts, timing.

Defined once and reused by every section so all six clips look identical.
"""
from manim import ManimColor

# Palette (matches the site dark theme).
BG = ManimColor("#0a0c10")
INK = ManimColor("#e0e0e0")
MUTED = ManimColor("#9aa3ad")
ACCENT = ManimColor("#4aa3ff")        # site blue rgba(74,163,255)
ACCENT_SOFT = ManimColor("#4aa3ff")   # used with low opacity for edges
WARM = ManimColor("#ff8c5a")          # negative values
GOOD = ManimColor("#79c98f")          # worked-example / output highlight

# Typography sizes.
CAPTION_SIZE = 30
FORMULA_SIZE = 40
LABEL_SIZE = 26

# Timing constants (seconds). Generous holds for a contemplative 3b1b feel.
T_HOLD = 1.2          # pause after a beat so the viewer can absorb it
T_FAST = 0.6
T_NORMAL = 1.2
T_SLOW = 2.0
T_INTRO = 1.5

# Geometry.
EDGE_OPACITY = 0.18   # thin translucent kNN edges so 1000 points stay readable
EDGE_WIDTH = 1.0
DOT_RADIUS = 0.018
```

- [ ] **Step 3: Commit**

```bash
git add manim/isomap/__init__.py manim/tests/__init__.py manim/isomap/style.py
git commit -m "manifold isomap: shared manim style module"
```

---

## Task 3: Math/data module with unit tests (TDD)

**Agent:** manifold-math-verifier

**Files:**
- Create: `manim/isomap/data.py`
- Test: `manim/tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

Create `manim/tests/test_data.py`:

```python
import unittest
import numpy as np
from manim.isomap import data


class TestData(unittest.TestCase):
    def test_swiss_roll_deterministic_and_shaped(self):
        a = data.swiss_roll(n=200, seed=7)
        b = data.swiss_roll(n=200, seed=7)
        np.testing.assert_allclose(a["points"], b["points"])
        self.assertEqual(a["points"].shape, (200, 3))
        self.assertEqual(a["t"].shape, (200,))

    def test_knn_is_symmetric_with_k_neighbors(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], float)
        adj, edges = data.knn_graph(pts, k=2)
        self.assertEqual(adj.shape, (5, 5))
        np.testing.assert_allclose(adj, adj.T)            # symmetric
        self.assertTrue(np.all(np.diag(adj) == 0))        # no self loops
        for (i, j) in edges:
            self.assertLess(i, j)                          # canonical order

    def test_double_center_matches_formula(self):
        # B = -1/2 J D^2 J on a tiny known squared-distance matrix.
        D = np.array([[0.0, 1.0, 2.0],
                      [1.0, 0.0, 1.0],
                      [2.0, 1.0, 0.0]])
        B = data.double_center_squared(D)
        n = 3
        J = np.eye(n) - np.ones((n, n)) / n
        expected = -0.5 * J @ (D ** 2) @ J
        np.testing.assert_allclose(B, expected, atol=1e-12)
        np.testing.assert_allclose(B, B.T, atol=1e-12)     # symmetric

    def test_geodesic_at_least_euclidean(self):
        d = data.build_dataset(n=120, k=8, seed=1)
        i, j = d["src"], d["tgt"]
        eucl = np.linalg.norm(d["points"][i] - d["points"][j])
        self.assertGreaterEqual(d["D"][i, j] + 1e-9, eucl)
        self.assertGreater(len(d["path"]), 1)

    def test_embedding_shape_and_excerpts(self):
        d = data.build_dataset(n=120, k=8, seed=1)
        self.assertEqual(d["embedding"].shape, (120, 2))
        self.assertEqual(np.array(d["excerpt_D"]).shape, (4, 4))
        self.assertEqual(np.array(d["excerpt_B"]).shape, (4, 4))
        self.assertEqual(len(d["eigvals"]), 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest manim.tests.test_data -v`
Expected: FAIL (module `manim.isomap.data` has no such attributes / import error).

- [ ] **Step 3: Implement `manim/isomap/data.py`**

```python
"""Isomap math for the explainer, mirroring js/manifold/linalg.js.

Pure numpy/scipy. Deterministic given a seed so the rendered numbers are stable
and unit-testable.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def swiss_roll(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n))   # angle along the roll
    height = 21.0 * rng.random(n)
    x = t * np.cos(t)
    z = t * np.sin(t)
    points = np.stack([x, height, z], axis=1)
    # Center and scale to a tidy viewing box.
    points = points - points.mean(axis=0)
    points = points / np.abs(points).max() * 3.0
    return {"points": points, "t": t}


def knn_graph(points, k=8):
    n = points.shape[0]
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    adj = np.zeros((n, n))
    for i in range(n):
        nn = np.argsort(dist[i])[:k]
        for j in nn:
            w = dist[i, j]
            adj[i, j] = w
            adj[j, i] = w  # symmetrize
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] > 0]
    return adj, edges


def geodesic_distances(adj):
    g = csr_matrix(adj)
    D, predecessors = shortest_path(g, method="D", directed=False, return_predecessors=True)
    return D, predecessors


def path_between(predecessors, src, tgt):
    path = [tgt]
    cur = tgt
    while cur != src and cur >= 0:
        cur = predecessors[src, cur]
        if cur < 0:
            break
        path.append(cur)
    path.reverse()
    return path


def double_center_squared(D):
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * J @ (D ** 2) @ J


def top2_eig(B):
    vals, vecs = np.linalg.eigh(B)          # ascending
    order = np.argsort(vals)[::-1][:2]      # top 2
    return vals[order], vecs[:, order]


def build_dataset(n=1000, k=8, seed=0):
    roll = swiss_roll(n=n, seed=seed)
    points = roll["points"]
    adj, edges = knn_graph(points, k=k)
    D, predecessors = geodesic_distances(adj)
    # Pick a far-apart source/target along the manifold for the geodesic trace.
    finite = np.where(np.isfinite(D), D, -1)
    src = int(np.unravel_index(np.argmax(finite), finite.shape)[0])
    tgt = int(np.unravel_index(np.argmax(finite), finite.shape)[1])
    path = path_between(predecessors, src, tgt)
    # Replace any inf (disconnected) with the max finite distance so the math is stable.
    Dfix = np.where(np.isfinite(D), D, finite.max())
    B = double_center_squared(Dfix)
    eigvals, eigvecs = top2_eig(B)
    embedding = eigvecs * np.sqrt(np.maximum(eigvals, 0.0))
    return {
        "points": points, "t": roll["t"], "adj": adj, "edges": edges,
        "D": Dfix, "src": src, "tgt": tgt, "path": path,
        "B": B, "eigvals": eigvals, "eigvecs": eigvecs, "embedding": embedding,
        "excerpt_D": np.round(Dfix[:4, :4], 2).tolist(),
        "excerpt_B": np.round(B[:4, :4], 2).tolist(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest manim.tests.test_data -v`
Expected: PASS (5 tests). If scipy/numpy are not yet installed, run with `manim/.venv/bin/python -m unittest ...`.

- [ ] **Step 5: Commit**

```bash
git add manim/isomap/data.py manim/tests/test_data.py
git commit -m "manifold isomap: dataset and math helpers with unit tests"
```

---

## Task 4: Mobject builders (shared visual system)

**Agent:** manim-animator

**Files:**
- Create: `manim/isomap/builders.py`

- [ ] **Step 1: Implement `manim/isomap/builders.py`**

```python
"""Reusable manim mobject builders so every section looks consistent."""
import numpy as np
from manim import (VGroup, Dot3D, Line, Line3D, Text, MathTex, Table,
                   SurroundingRectangle, Create, FadeIn, ORIGIN, DOWN, UP)
from . import style as S


def color_for_t(t, tmin, tmax):
    u = (t - tmin) / max(1e-9, (tmax - tmin))
    # Blue (low) to warm (high), matching the site rainbow feel.
    from manim import interpolate_color
    return interpolate_color(S.ACCENT, S.WARM, u)


def point_cloud(points, t):
    tmin, tmax = float(t.min()), float(t.max())
    dots = VGroup(*[
        Dot3D(point=points[i], radius=S.DOT_RADIUS, color=color_for_t(t[i], tmin, tmax))
        for i in range(points.shape[0])
    ])
    return dots


def graph_edges(points, edges, color=S.ACCENT, opacity=S.EDGE_OPACITY):
    lines = VGroup(*[
        Line3D(start=points[i], end=points[j], color=color, stroke_width=S.EDGE_WIDTH)
        for (i, j) in edges
    ])
    lines.set_opacity(opacity)
    return lines


def path_polyline(points, path, color=S.GOOD, width=4.0):
    segs = VGroup(*[
        Line3D(start=points[path[a]], end=points[path[a + 1]], color=color, stroke_width=width)
        for a in range(len(path) - 1)
    ])
    return segs


def caption(text):
    return Text(text, font_size=S.CAPTION_SIZE, color=S.INK)


def formula(tex):
    return MathTex(tex, font_size=S.FORMULA_SIZE, color=S.INK)


def matrix_grid(values, highlight_negative=True):
    rows = [[f"{v:.2f}" for v in row] for row in values]
    tbl = Table(rows, include_outer_lines=True)
    tbl.scale(0.5)
    if highlight_negative:
        for r, row in enumerate(values):
            for c, v in enumerate(row):
                if v < 0:
                    tbl.get_entries((r + 1, c + 1)).set_color(S.WARM)
    return tbl
```

- [ ] **Step 2: Commit**

```bash
git add manim/isomap/builders.py
git commit -m "manifold isomap: shared mobject builders"
```

---

## Task 5: Walkthrough scene + Section 1 (Raw data)

**Agent:** manim-animator

**Files:**
- Create: `manim/isomap/walkthrough.py`

- [ ] **Step 1: Create the scene with the dataset and Section 1**

```python
"""The single continuous Isomap walkthrough.

One ThreeDScene authored with self.next_section() per step so objects persist and
transform across step boundaries (seamless continuity) while --save_sections emits
one MP4 per step for the navigable player.
"""
from manim import (ThreeDScene, Section, DEGREES, FadeIn, FadeOut, Create,
                   Write, Transform, ReplacementTransform, UP, DOWN, LEFT, RIGHT, ORIGIN)
from . import style as S
from . import builders as B
from .data import build_dataset

N = 1000
K = 8
SEED = 0


class IsomapWalkthrough(ThreeDScene):
    def construct(self):
        self.camera.background_color = S.BG
        self.data = build_dataset(n=N, k=K, seed=SEED)
        self.cloud = None
        self.cap = None
        self.section_raw()
        # Later tasks append: section_knn, section_geodesic, section_double_center,
        # section_eigendecomp, section_embedding.

    def set_caption(self, text):
        new = B.caption(text).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(new)
        if self.cap is None:
            self.play(FadeIn(new, run_time=S.T_FAST))
        else:
            self.play(ReplacementTransform(self.cap, new, run_time=S.T_NORMAL))
        self.cap = new

    def section_raw(self):
        self.next_section("step-1-raw", type=Section.NORMAL)
        self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES, zoom=0.9)
        self.cloud = B.point_cloud(self.data["points"], self.data["t"])
        self.play(FadeIn(self.cloud, run_time=S.T_INTRO))
        self.set_caption("A 2D sheet rolled up in 3D. The goal: recover the flat sheet.")
        self.begin_ambient_camera_rotation(rate=0.08)
        self.wait(S.T_SLOW)
        self.stop_ambient_camera_rotation()
        self.wait(S.T_HOLD)
```

- [ ] **Step 2: Render Section 1 to verify it builds**

Run (low quality for a fast check):
```bash
cd /mnt/c/Users/aughb/PersonalProjects/Visualization
manim/.venv/bin/manim -ql --fps 30 --save_sections manim/isomap/walkthrough.py IsomapWalkthrough && echo RENDER_OK
```
Expected: `RENDER_OK`; a section video appears under `media/videos/walkthrough/.../sections/`. (1000 Dot3D is slow at high quality; low quality keeps the check fast.)

- [ ] **Step 3: Commit**

```bash
git add manim/isomap/walkthrough.py
git commit -m "manifold isomap: walkthrough scene with raw-data section"
```

---

## Task 6: Section 2 (kNN graph)

**Agent:** manim-animator

**Files:**
- Modify: `manim/isomap/walkthrough.py`

- [ ] **Step 1: Add the call in `construct`**

After `self.section_raw()` add:
```python
        self.section_knn()
```

- [ ] **Step 2: Add the `section_knn` method**

```python
    def section_knn(self):
        self.next_section("step-2-knn", type=Section.NORMAL)
        pts, edges = self.data["points"], self.data["edges"]
        self.edges_mob = B.graph_edges(pts, edges)
        # Thin translucent edges so 1000 points stay readable; grow them in.
        self.play(Create(self.edges_mob, run_time=S.T_SLOW))
        self.set_caption("Connect each point to its k = 8 nearest neighbors.")
        self.wait(S.T_HOLD)
```

- [ ] **Step 3: Render to verify**

Run: `manim/.venv/bin/manim -ql --fps 30 --save_sections manim/isomap/walkthrough.py IsomapWalkthrough && echo RENDER_OK`
Expected: `RENDER_OK`; two section files now exist; the cloud from step 1 is still present (continuity).

- [ ] **Step 4: Commit**

```bash
git add manim/isomap/walkthrough.py
git commit -m "manifold isomap: kNN graph section"
```

---

## Task 7: Section 3 (Geodesic distances)

**Agent:** manim-animator

**Files:**
- Modify: `manim/isomap/walkthrough.py`

- [ ] **Step 1: Add the call in `construct`**

After `self.section_knn()` add:
```python
        self.section_geodesic()
```

- [ ] **Step 2: Add the `section_geodesic` method**

```python
    def section_geodesic(self):
        self.next_section("step-3-geodesic", type=Section.NORMAL)
        pts, path = self.data["points"], self.data["path"]
        src, tgt = self.data["src"], self.data["tgt"]
        # Fade the bulk graph back so the traced path reads clearly.
        self.play(self.edges_mob.animate.set_opacity(0.06), run_time=S.T_FAST)
        straight = B.Line3D(start=pts[src], end=pts[tgt], color=S.MUTED, stroke_width=3.0)
        geo = B.path_polyline(pts, path, color=S.GOOD, width=5.0)
        self.play(Create(straight, run_time=S.T_NORMAL))
        self.set_caption("Straight-line distance cuts through space, off the sheet.")
        self.wait(S.T_HOLD)
        self.play(Create(geo, run_time=S.T_SLOW))
        self.set_caption("Geodesic distance follows the graph along the sheet.")
        self.geo, self.straight = geo, straight
        self.wait(S.T_HOLD)
```

- [ ] **Step 3: Render to verify**

Run: `manim/.venv/bin/manim -ql --fps 30 --save_sections manim/isomap/walkthrough.py IsomapWalkthrough && echo RENDER_OK`
Expected: `RENDER_OK`; three section files.

- [ ] **Step 4: Commit**

```bash
git add manim/isomap/walkthrough.py
git commit -m "manifold isomap: geodesic distance section"
```

---

## Task 8: Section 4 (Double-centering)

**Agent:** manim-animator (numbers supplied by manifold-math-verifier)

**Files:**
- Modify: `manim/isomap/walkthrough.py`

- [ ] **Step 1: Add the call in `construct`**

After `self.section_geodesic()` add:
```python
        self.section_double_center()
```

- [ ] **Step 2: Add the `section_double_center` method**

```python
    def section_double_center(self):
        self.next_section("step-4-double-center", type=Section.NORMAL)
        # Move to a flat, front-on view and clear the 3D scene for the matrix math.
        self.play(FadeOut(self.cloud), FadeOut(self.edges_mob),
                  FadeOut(self.geo), FadeOut(self.straight), run_time=S.T_FAST)
        self.move_camera(phi=0, theta=-90 * DEGREES, zoom=1.0, run_time=S.T_NORMAL)
        dmat = B.matrix_grid(self.data["excerpt_D"], highlight_negative=False).to_edge(LEFT)
        self.add_fixed_in_frame_mobjects(dmat)
        self.play(FadeIn(dmat, run_time=S.T_NORMAL))
        self.set_caption("Take the geodesic distances, square them.")
        f = B.formula(r"B = -\tfrac{1}{2}\, J\, D^2\, J").to_edge(UP)
        self.add_fixed_in_frame_mobjects(f)
        self.play(Write(f, run_time=S.T_NORMAL))
        self.set_caption("Subtract row and column means, re-add the grand mean, scale by -1/2.")
        bmat = B.matrix_grid(self.data["excerpt_B"], highlight_negative=True).to_edge(RIGHT)
        self.add_fixed_in_frame_mobjects(bmat)
        self.play(ReplacementTransform(dmat.copy(), bmat, run_time=S.T_SLOW))
        self.set_caption("The result B behaves like inner products about the center.")
        self.formula_dc, self.bmat = f, bmat
        self.wait(S.T_HOLD)
```

- [ ] **Step 3: Render to verify**

Run: `manim/.venv/bin/manim -ql --fps 30 --save_sections manim/isomap/walkthrough.py IsomapWalkthrough && echo RENDER_OK`
Expected: `RENDER_OK`; four section files; MathTex compiled (confirms LaTeX works).

- [ ] **Step 4: Commit**

```bash
git add manim/isomap/walkthrough.py
git commit -m "manifold isomap: double-centering section"
```

---

## Task 9: Section 5 (Eigendecomposition)

**Agent:** manim-animator (eigenvalues from manifold-math-verifier)

**Files:**
- Modify: `manim/isomap/walkthrough.py`

- [ ] **Step 1: Add the call in `construct`**

After `self.section_double_center()` add:
```python
        self.section_eigendecomp()
```

- [ ] **Step 2: Add the `section_eigendecomp` method**

```python
    def section_eigendecomp(self):
        self.next_section("step-5-eigendecomp", type=Section.NORMAL)
        l1, l2 = float(self.data["eigvals"][0]), float(self.data["eigvals"][1])
        self.play(FadeOut(self.bmat), run_time=S.T_FAST)
        f2 = B.formula(r"B v_i = \lambda_i v_i").to_edge(UP)
        self.add_fixed_in_frame_mobjects(f2)
        self.play(ReplacementTransform(self.formula_dc, f2, run_time=S.T_NORMAL))
        self.set_caption("Find the eigenvectors of B; the largest eigenvalues carry the shape.")
        vals = B.formula(rf"\lambda_1 = {l1:.2f}\quad \lambda_2 = {l2:.2f}").next_to(f2, DOWN)
        self.add_fixed_in_frame_mobjects(vals)
        self.play(Write(vals, run_time=S.T_NORMAL))
        self.set_caption("Keep the top two: they span the recovered plane.")
        self.formula_eig, self.vals = f2, vals
        self.wait(S.T_HOLD)
```

- [ ] **Step 3: Render to verify**

Run: `manim/.venv/bin/manim -ql --fps 30 --save_sections manim/isomap/walkthrough.py IsomapWalkthrough && echo RENDER_OK`
Expected: `RENDER_OK`; five section files.

- [ ] **Step 4: Commit**

```bash
git add manim/isomap/walkthrough.py
git commit -m "manifold isomap: eigendecomposition section"
```

---

## Task 10: Section 6 (Embedding / unroll)

**Agent:** manim-animator

**Files:**
- Modify: `manim/isomap/walkthrough.py`

- [ ] **Step 1: Add the call in `construct`**

After `self.section_eigendecomp()` add:
```python
        self.section_embedding()
```

- [ ] **Step 2: Add the `section_embedding` method**

```python
    def section_embedding(self):
        self.next_section("step-6-embedding", type=Section.NORMAL)
        self.play(FadeOut(self.formula_eig), FadeOut(self.vals), run_time=S.T_FAST)
        import numpy as np
        emb = self.data["embedding"]
        emb = emb / np.abs(emb).max() * 3.0          # scale to view box
        pts3 = np.column_stack([emb[:, 0], emb[:, 1], np.zeros(emb.shape[0])])
        flat = B.point_cloud(pts3, self.data["t"])
        f3 = B.formula(r"Y = [\sqrt{\lambda_1}\,v_1,\ \sqrt{\lambda_2}\,v_2]").to_edge(UP)
        self.add_fixed_in_frame_mobjects(f3)
        self.play(FadeIn(flat, run_time=S.T_SLOW), Write(f3, run_time=S.T_NORMAL))
        self.set_caption("The sheet unrolls into 2D, geodesic distances preserved.")
        self.wait(S.T_SLOW)
```

- [ ] **Step 3: Render the full walkthrough (low quality) to verify all six sections**

Run: `manim/.venv/bin/manim -ql --fps 30 --save_sections manim/isomap/walkthrough.py IsomapWalkthrough && echo RENDER_OK`
Expected: `RENDER_OK`; six section files under `media/videos/walkthrough/.../sections/`.

- [ ] **Step 4: Commit**

```bash
git add manim/isomap/walkthrough.py
git commit -m "manifold isomap: embedding section, full six-step walkthrough"
```

---

## Task 11: Render script (high quality clips + posters)

**Agent:** manim-animator

**Files:**
- Create: `manim/render.sh`

- [ ] **Step 1: Create `manim/render.sh`**

```bash
#!/usr/bin/env bash
# Render the Isomap walkthrough at 60fps/1080p, split into per-step section MP4s,
# and copy them plus poster frames to assets/manim/isomap/.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

PY=manim/.venv/bin/manim
OUT=assets/manim/isomap
mkdir -p "$OUT"

# -qh = 1080p60 when --fps 60 is set. --save_sections emits one MP4 per next_section().
$PY -qh --fps 60 --save_sections manim/isomap/walkthrough.py IsomapWalkthrough

SECTIONS_DIR=$(find media/videos/walkthrough -type d -name sections | head -1)
i=1
for name in step-1-raw step-2-knn step-3-geodesic step-4-double-center step-5-eigendecomp step-6-embedding; do
  src=$(find "$SECTIONS_DIR" -name "*${name}*.mp4" | head -1)
  cp "$src" "$OUT/step-${i}.mp4"
  ffmpeg -y -i "$OUT/step-${i}.mp4" -frames:v 1 -q:v 2 "$OUT/step-${i}.png"
  i=$((i + 1))
done
echo "Rendered ${OUT}/step-1..6.mp4 and posters."
```

- [ ] **Step 2: Make it executable and render**

```bash
chmod +x manim/render.sh
bash manim/render.sh && ls -1 assets/manim/isomap/
```
Expected: `step-1.mp4 ... step-6.mp4` and `step-1.png ... step-6.png` listed. (High-quality render of 1000 Dot3D is slow; this is a one-time offline render.)

- [ ] **Step 3: Commit the script and the rendered assets**

```bash
git add manim/render.sh assets/manim/isomap/step-1.mp4 assets/manim/isomap/step-2.mp4 assets/manim/isomap/step-3.mp4 assets/manim/isomap/step-4.mp4 assets/manim/isomap/step-5.mp4 assets/manim/isomap/step-6.mp4 assets/manim/isomap/step-1.png assets/manim/isomap/step-2.png assets/manim/isomap/step-3.png assets/manim/isomap/step-4.png assets/manim/isomap/step-5.png assets/manim/isomap/step-6.png
git commit -m "manifold isomap: render script and rendered step clips + posters"
```

---

## Task 12: Explainer page and player

**Agent:** web-player-engineer

**Files:**
- Create: `pages/manifold_isomap.html`
- Create: `styles/manifold_isomap.css`
- Create: `js/manifold_isomap.js`
- Modify: `index.html`

- [ ] **Step 1: Create `pages/manifold_isomap.html`**

```html
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Isomap, Step by Step</title>

  <script>
    window.MathJax = {
      tex: { inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['$$', '$$']] },
      svg: { fontCache: 'global' }
    };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

  <link rel="stylesheet" href="../styles/base.css">
  <link rel="stylesheet" href="../styles/article.css">
  <link rel="stylesheet" href="../styles/manifold_isomap.css">

  <script type="module" src="../js/manifold_isomap.js"></script>
</head>

<body>
  <div class="container article mf-isomap">
    <header>
      <h1>Isomap, Step by Step</h1>
      <div class="subtitle">An animated walk from a rolled-up sheet to its flattened map</div>
      <div class="home-link"><a href="../index.html">&larr; Home</a></div>
    </header>

    <main class="article-body">
      <section class="panel">
        <div class="mfi-player">
          <video id="mfiVideo" class="mfi-video" playsinline preload="auto"></video>
          <div class="mfi-controls">
            <button id="mfiPrev" type="button">&larr; Prev</button>
            <button id="mfiPlay" type="button">Play</button>
            <button id="mfiNext" type="button">Next &rarr;</button>
            <input id="mfiScrub" class="mfi-scrub" type="range" min="0" max="1000" value="0" />
          </div>
          <ol id="mfiSteps" class="mfi-steps"></ol>
        </div>
        <div id="mfiTranscript" class="mfi-transcript"></div>
        <p class="mfi-tryit">Want to explore it on your own data?
          <a href="manifold.html">Try the interactive sandbox</a>.</p>
      </section>
    </main>
  </div>
</body>

</html>
```

- [ ] **Step 2: Create `styles/manifold_isomap.css`**

```css
/* manifold_isomap.css - Isomap explainer player */
.mf-isomap.container { max-width: 1100px; }
.mfi-player { display: flex; flex-direction: column; gap: 12px; }
.mfi-video {
  width: 100%; aspect-ratio: 16 / 9; background: #000;
  border-radius: var(--radius-md); border: 1px solid rgba(255, 255, 255, 0.08);
}
.mfi-controls { display: flex; align-items: center; gap: 10px; }
.mfi-controls button {
  font: inherit; padding: 8px 14px; border-radius: var(--radius-sm);
  border: 1px solid var(--border-light); background: rgba(255, 255, 255, 0.05);
  color: var(--text); cursor: pointer;
}
.mfi-controls button:hover { background: rgba(255, 255, 255, 0.1); }
.mfi-scrub { flex: 1; }
.mfi-steps { display: flex; flex-wrap: wrap; gap: 6px; list-style: none; padding: 0; margin: 0; }
.mfi-steps li {
  padding: 6px 10px; border-radius: var(--radius-sm); cursor: pointer;
  border: 1px solid var(--border-light); color: var(--text-muted); font-size: 0.9rem;
}
.mfi-steps li.is-active {
  background: rgba(74, 163, 255, 0.18); color: #fff;
  box-shadow: inset 0 -2px 0 rgba(74, 163, 255, 0.7);
}
.mfi-transcript { margin-top: 16px; padding: 12px 14px;
  background: var(--surface-inset); border-radius: var(--radius-sm); line-height: 1.5; }
.mfi-transcript .mfi-formula { margin-top: 6px; color: var(--text-muted); }
.mfi-tryit { margin-top: 14px; color: var(--text-muted); }
```

- [ ] **Step 3: Create `js/manifold_isomap.js`**

```javascript
// manifold_isomap.js - per-step clip player for the Isomap explainer (ES module).
const STEPS = [
  { title: '1. Raw data', caption: 'A 2D sheet rolled up in 3D. The goal is to recover the flat sheet.', formula: '' },
  { title: '2. kNN graph', caption: 'Connect each point to its k = 8 nearest neighbors.', formula: '' },
  { title: '3. Geodesic distances', caption: 'Distance measured along the graph, not straight through space.', formula: '' },
  { title: '4. Double-centering', caption: 'Turn squared geodesic distances into the matrix B.', formula: 'B = -\\tfrac{1}{2} J D^2 J' },
  { title: '5. Eigendecomposition', caption: 'The top eigenvectors of B carry the recovered shape.', formula: 'B v_i = \\lambda_i v_i' },
  { title: '6. Embedding', caption: 'The sheet unrolls into 2D, geodesic distances preserved.', formula: 'Y = [\\sqrt{\\lambda_1} v_1,\\ \\sqrt{\\lambda_2} v_2]' },
];

const video = document.getElementById('mfiVideo');
const stepsEl = document.getElementById('mfiSteps');
const transcript = document.getElementById('mfiTranscript');
const scrub = document.getElementById('mfiScrub');
const playBtn = document.getElementById('mfiPlay');
let current = 0;
let autoChain = false;

function srcFor(i) { return `../assets/manim/isomap/step-${i + 1}.mp4`; }
function posterFor(i) { return `../assets/manim/isomap/step-${i + 1}.png`; }

function renderSteps() {
  stepsEl.innerHTML = '';
  STEPS.forEach((s, i) => {
    const li = document.createElement('li');
    li.textContent = s.title;
    if (i === current) li.classList.add('is-active');
    li.addEventListener('click', () => load(i, false));
    stepsEl.appendChild(li);
  });
}

function renderTranscript() {
  const s = STEPS[current];
  transcript.innerHTML = '';
  const cap = document.createElement('div');
  cap.textContent = s.caption;
  transcript.appendChild(cap);
  if (s.formula) {
    const f = document.createElement('div');
    f.className = 'mfi-formula';
    f.textContent = `\\[${s.formula}\\]`;
    transcript.appendChild(f);
  }
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetClear && window.MathJax.typesetClear();
    window.MathJax.typesetPromise([transcript]).catch(() => {});
  }
}

function load(i, autoplay) {
  current = Math.max(0, Math.min(STEPS.length - 1, i));
  video.poster = posterFor(current);
  video.src = srcFor(current);
  video.load();
  renderSteps();
  renderTranscript();
  if (autoplay) video.play().catch(() => {});
}

video.addEventListener('timeupdate', () => {
  if (video.duration) scrub.value = String(Math.round((video.currentTime / video.duration) * 1000));
});
scrub.addEventListener('input', () => {
  if (video.duration) video.currentTime = (scrub.value / 1000) * video.duration;
});
video.addEventListener('ended', () => {
  if (autoChain && current < STEPS.length - 1) load(current + 1, true);
});
playBtn.addEventListener('click', () => {
  autoChain = true;
  video.play().catch(() => {});
});
document.getElementById('mfiPrev').addEventListener('click', () => load(current - 1, true));
document.getElementById('mfiNext').addEventListener('click', () => load(current + 1, true));

renderSteps();
load(0, false);
```

- [ ] **Step 4: Add the homepage link in `index.html`**

Add alongside the other `pages/*.html` project items (match their structure):
```html
        <li class="project-item">
          <a href="pages/manifold_isomap.html">Isomap, Step by Step</a>
          <div class="project-desc">An animated, step-by-step manim walkthrough of the Isomap manifold-learning algorithm</div>
        </li>
```

- [ ] **Step 5: Verify the page loads and serves the clips**

Run:
```bash
cd /mnt/c/Users/aughb/PersonalProjects/Visualization
node --check js/manifold_isomap.js && echo SYNTAX_OK
python3 -m http.server 8781 >/dev/null 2>&1 & SRV=$!; sleep 1
curl -s -o /dev/null -w "page %{http_code}\n" http://127.0.0.1:8781/pages/manifold_isomap.html
curl -s -o /dev/null -w "clip %{http_code}\n" http://127.0.0.1:8781/assets/manim/isomap/step-1.mp4
kill $SRV
```
Expected: `SYNTAX_OK`, `page 200`, `clip 200`.

- [ ] **Step 6: Commit**

```bash
git add pages/manifold_isomap.html styles/manifold_isomap.css js/manifold_isomap.js index.html
git commit -m "manifold isomap: explainer page and per-step clip player"
```

---

## Task 13: Motion-design review and polish loop

**Agent:** motion-design-reviewer (critique) -> manim-animator (fix), coordinated by manifold-overseer

**Files:**
- Modify: `manim/isomap/walkthrough.py`, `manim/isomap/style.py` (as needed)
- Re-render: `assets/manim/isomap/*`

- [ ] **Step 1: Render the full quality clips (if not already)**

Run: `bash manim/render.sh`
Expected: six clips + posters in `assets/manim/isomap/`.

- [ ] **Step 2: Review each clip against the quality bar**

The motion-design-reviewer inspects every clip with ffprobe + extracted frames:
```bash
for i in 1 2 3 4 5 6; do
  ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate,width,height,duration \
    -of default=nw=1 assets/manim/isomap/step-$i.mp4
  mkdir -p /tmp/mfi-frames/$i
  ffmpeg -v error -y -i assets/manim/isomap/step-$i.mp4 -vf fps=2 /tmp/mfi-frames/$i/f-%03d.png
done
```
Expected per clip: `r_frame_rate=60/1`, `width=1920`, `height=1080`, duration 15-30s. Then Read sampled frames and check: smooth staging, captions readable and not overlapping, dense graph legible, geodesic path clearly visible, last frame of step N matches first frame of step N+1 (continuity), consistent palette/typography. Produce a PASS / NEEDS_WORK report with specific timestamps and fixes.

- [ ] **Step 3: Apply fixes and re-render until PASS**

The animator applies the reviewer's concrete fixes (timing constants in `style.py`, staging in `walkthrough.py`), re-runs `bash manim/render.sh`, and the reviewer re-checks. Repeat until PASS. Each fix round:
```bash
git add manim/ assets/manim/isomap/
git commit -m "manifold isomap: motion-design polish pass"
```

- [ ] **Step 4: Final commit of polished assets**

```bash
git add manim/ assets/manim/isomap/
git commit -m "manifold isomap: clips pass the motion-design quality bar"
```

---

## Task 14: Final integration check

**Agent:** manifold-overseer

**Files:** none (verification only; fixes routed to the owning agent)

- [ ] **Step 1: Run the Python math tests**

Run: `manim/.venv/bin/python -m unittest discover -s manim/tests -v` (or `python3 -m unittest manim.tests.test_data -v`)
Expected: all tests pass.

- [ ] **Step 2: Browser smoke of the page**

Serve and open `http://127.0.0.1:8781/pages/manifold_isomap.html`. Verify: each step clip plays; prev/next/play and the step list switch clips; pressing Play auto-advances through all six; the scrubber seeks; the transcript shows the caption and the typeset formula; "Try the interactive sandbox" links to `manifold.html`; the homepage card opens the page.

- [ ] **Step 3: Banned-pattern sweep**

Run: `grep -rnP "<(em|strong|b|i|mark)>|\x{2014}|&mdash;" pages/manifold_isomap.html js/manifold_isomap.js styles/manifold_isomap.css manim/`
Expected: no matches.

- [ ] **Step 4: Final commit (if any fixes were made)**

```bash
git add -A
git commit -m "manifold isomap: final integration fixes"
```

---

## Self-Review Notes

- Spec coverage: manim-first dedicated page (Tasks 5-12); six navigable per-step clips via one continuous scene with `--save_sections` (Tasks 5-11); seamless continuity since it is one scene (Tasks 5-10); geometry + formula + worked numbers fused in-scene (Tasks 8-10); captions only (all sections); fixed Swiss roll 1000 / k=8 (Task 3); legibility at 1000 via translucent edges + highlighted geodesic (Tasks 6-7); 3b1b smooth easing + 60fps + quality bar (Tasks 2, 11, 13); pipeline to committed assets (Task 11); page player with prev/play/next/scrub/transcript + try-it-live link + homepage link (Task 12); toolchain milestone (Task 1); math unit tests mirroring linalg.js (Task 3); five agents own their tasks (Agent lines); style constraints swept (Task 14).
- Continuity note: because the six clips are sections of one scene, the last frame of step N is by construction the first frame of step N+1 in the source render; the player loads discrete files, so the reviewer verifies the seam visually (Task 13 Step 2).
- Polish reality: animation timing and staging are refined through the reviewer loop (Task 13); Tasks 5-10 establish correct, renderable first-pass scenes.
- Type/name consistency: `build_dataset` keys (`points, t, adj, edges, D, src, tgt, path, B, eigvals, eigvecs, embedding, excerpt_D, excerpt_B`) defined in Task 3 are the exact keys read in Tasks 5-10; builder names (`point_cloud, graph_edges, path_polyline, caption, formula, matrix_grid`) defined in Task 4 match their uses; section names (`step-1-raw ... step-6-embedding`) match `render.sh` and the player's `step-N.mp4`.
