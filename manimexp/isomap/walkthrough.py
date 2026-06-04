"""The single continuous Isomap walkthrough.

One ThreeDScene authored with self.next_section() per step so objects persist and
transform across step boundaries (seamless continuity) while --save_sections emits
one MP4 per step for the navigable player.

Manim 0.18.1 API adaptations versus the design spec:
- Section.NORMAL does not exist; use DefaultSectionType.NORMAL (manim.scene.section).
- Dot3D and Line3D are 3D surface meshes (Cylinder subclasses). At N=120, Dot3D
  renders at ~1.3 s/frame (unusable at N=1000). Replaced with 2D Dot and Line
  VMobjects in builders.py -- same visual result, orders of magnitude faster.
  Line3D is kept only for the highlighted geodesic path and straight-line (small N).
- Line3D uses thickness (scene units) not stroke_width (pixel units).
- Graph edges use FadeIn rather than Create to avoid per-stroke progressive drawing
  overhead on ~480 (N=120) to ~4000 (N=1000) edge segments.
- add_fixed_in_frame_mobjects() must be called before self.play() for 2D overlays.
"""
import os
import numpy as np
from manim import (
    ThreeDScene, DEGREES, FadeIn, FadeOut, Create, Write,
    ReplacementTransform, DOWN, UP, RIGHT, LEFT,
)
from manim.scene.section import DefaultSectionType
from manimexp.isomap import style as S
from manimexp.isomap import builders as B
from manimexp.isomap.data import build_dataset

N = int(os.environ.get("MFI_N", "1000"))
K = 8
SEED = 0

_SEC = DefaultSectionType.NORMAL


class IsomapWalkthrough(ThreeDScene):

    def construct(self):
        self.camera.background_color = S.BG
        self.data = build_dataset(n=N, k=K, seed=SEED)
        self.cap = None

        self.section_raw()
        self.section_knn()
        self.section_geodesic()
        self.section_double_center()
        self.section_eigendecomp()
        self.section_embedding()

    # ------------------------------------------------------------------ #
    # Caption helper                                                       #
    # ------------------------------------------------------------------ #

    def set_caption(self, text):
        new = B.caption(text).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(new)
        new.set_opacity(0)
        if self.cap is None:
            self.play(new.animate.set_opacity(1.0), run_time=S.T_FAST)
        else:
            old = self.cap
            self.play(FadeOut(old, shift=0.0), new.animate.set_opacity(1.0), run_time=S.T_FAST)
            self.remove(old)
        self.cap = new

    # ------------------------------------------------------------------ #
    # Section 1: raw point cloud on the Swiss roll                        #
    # ------------------------------------------------------------------ #

    def section_raw(self):
        self.next_section("step-1-raw", type=_SEC)
        self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES, zoom=0.9)
        self.cloud = B.point_cloud(self.data["points"], self.data["t"])
        self.play(FadeIn(self.cloud, run_time=S.T_INTRO))
        self.set_caption("A 2D sheet rolled up in 3D. The goal: recover the flat sheet.")
        self.begin_ambient_camera_rotation(rate=0.08)
        self.wait(S.T_SLOW)
        self.stop_ambient_camera_rotation()
        self.wait(S.T_HOLD)

    # ------------------------------------------------------------------ #
    # Section 2: kNN graph edges                                          #
    # ------------------------------------------------------------------ #

    def section_knn(self):
        self.next_section("step-2-knn", type=_SEC)
        self.edges_mob = B.graph_edges(self.data["points"], self.data["edges"])
        # FadeIn instead of Create: avoids progressive stroke drawing overhead
        # on potentially thousands of edge segments.
        self.play(FadeIn(self.edges_mob, run_time=S.T_SLOW))
        self.set_caption("Connect each point to its k = 8 nearest neighbors.")
        self.wait(S.T_HOLD)

    # ------------------------------------------------------------------ #
    # Section 3: geodesic vs. Euclidean path                             #
    # ------------------------------------------------------------------ #

    def section_geodesic(self):
        self.next_section("step-3-geodesic", type=_SEC)
        pts, path = self.data["points"], self.data["path"]
        src, tgt = self.data["src"], self.data["tgt"]
        self.play(self.edges_mob.animate.set_opacity(0.06), run_time=S.T_FAST)
        straight = B.straight_line(pts[src], pts[tgt])
        geo = B.path_polyline(pts, path)
        self.play(Create(straight, run_time=S.T_NORMAL))
        self.set_caption("Straight-line distance cuts through space, off the sheet.")
        self.wait(S.T_HOLD)
        self.play(Create(geo, run_time=S.T_SLOW))
        self.set_caption("Geodesic distance follows the graph along the sheet.")
        self.geo, self.straight = geo, straight
        self.wait(S.T_HOLD)

    # ------------------------------------------------------------------ #
    # Section 4: double centering                                         #
    # ------------------------------------------------------------------ #

    def section_double_center(self):
        self.next_section("step-4-double-center", type=_SEC)
        self.play(
            FadeOut(self.cloud),
            FadeOut(self.edges_mob),
            FadeOut(self.geo),
            FadeOut(self.straight),
            run_time=S.T_FAST,
        )
        self.move_camera(phi=0, theta=-90 * DEGREES, zoom=1.0, run_time=S.T_NORMAL)

        # Scale all three objects to 0.45 and stack them with clear vertical separation
        # so they never overlap during transitions.
        SCALE = 0.45
        Y_TOP = 2.8
        Y_MID = 0.4
        Y_BOT = -1.8

        dmat = B.matrix_grid(self.data["excerpt_D"], highlight_negative=False)
        dmat.scale(SCALE).move_to([0, Y_TOP, 0])
        self.add_fixed_in_frame_mobjects(dmat)
        self.play(FadeIn(dmat, run_time=S.T_NORMAL))
        self.set_caption("Take the geodesic distances, square them.")

        f = B.formula(r"B = -\tfrac{1}{2}\, J\, D^2\, J")
        f.scale(0.85).move_to([0, Y_MID, 0])
        self.add_fixed_in_frame_mobjects(f)
        self.play(Write(f, run_time=S.T_NORMAL))
        self.set_caption("Subtract row and column means, re-add the grand mean, scale by -1/2.")

        bmat = B.matrix_grid(self.data["excerpt_B"], highlight_negative=True)
        bmat.scale(SCALE).move_to([0, Y_BOT, 0])
        self.add_fixed_in_frame_mobjects(bmat)
        self.play(ReplacementTransform(dmat.copy(), bmat, run_time=S.T_SLOW))
        self.set_caption("The result B behaves like inner products about the center.")
        self.formula_dc, self.bmat, self.dmat = f, bmat, dmat
        self.wait(S.T_HOLD)

    # ------------------------------------------------------------------ #
    # Section 5: eigendecomposition                                       #
    # ------------------------------------------------------------------ #

    def section_eigendecomp(self):
        self.next_section("step-5-eigendecomp", type=_SEC)
        l1, l2 = float(self.data["eigvals"][0]), float(self.data["eigvals"][1])

        # Fade out D and the formula; keep B on screen (eigen content derives from B).
        self.play(
            FadeOut(self.dmat),
            FadeOut(self.formula_dc),
            run_time=S.T_FAST,
        )

        # Move B to the left side so the right side is free for eigen content.
        self.play(self.bmat.animate.move_to([-3.2, 0, 0]), run_time=S.T_FAST)

        # Eigen equation and eigenvalues placed to the right of B, no overlap.
        f2 = B.formula(r"B v_i = \lambda_i v_i").move_to([1.5, 1.2, 0])
        self.add_fixed_in_frame_mobjects(f2)
        self.play(FadeIn(f2, run_time=S.T_NORMAL))
        self.set_caption("Find the eigenvectors of B; the largest eigenvalues carry the shape.")

        vals = B.formula(
            rf"\lambda_1 = {l1:.1f}\quad \lambda_2 = {l2:.1f}"
        ).move_to([1.5, -0.6, 0])
        self.add_fixed_in_frame_mobjects(vals)
        self.play(Write(vals, run_time=S.T_NORMAL))
        self.set_caption("Keep the top two: they span the recovered plane.")
        self.formula_eig, self.vals = f2, vals
        self.wait(S.T_HOLD)

    # ------------------------------------------------------------------ #
    # Section 6: 2D embedding reveal                                      #
    # ------------------------------------------------------------------ #

    def section_embedding(self):
        self.next_section("step-6-embedding", type=_SEC)
        # Clear every remaining overlay (B matrix, eigen formula, eigenvalues)
        # before the embedding so nothing overlaps the final arc.
        self.play(
            FadeOut(self.bmat),
            FadeOut(self.formula_eig),
            FadeOut(self.vals),
            run_time=S.T_FAST,
        )

        emb = self.data["embedding"].copy()
        scale = np.abs(emb).max()
        if scale > 1e-9:
            emb = emb / scale * 3.5
        pts3 = np.column_stack([emb[:, 0], emb[:, 1], np.zeros(emb.shape[0])])
        flat = B.point_cloud(pts3, self.data["t"])

        f3 = B.formula(
            r"Y = [\sqrt{\lambda_1}\,v_1,\ \sqrt{\lambda_2}\,v_2]"
        ).to_edge(UP)
        self.add_fixed_in_frame_mobjects(f3)
        self.play(
            FadeIn(flat, run_time=S.T_SLOW),
            Write(f3, run_time=S.T_NORMAL),
        )
        self.set_caption("The sheet unrolls into 2D, geodesic distances preserved.")
        self.wait(S.T_SLOW)
