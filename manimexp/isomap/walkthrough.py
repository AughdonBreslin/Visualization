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
    ThreeDScene, ThreeDAxes, DEGREES, FadeIn, FadeOut, Create, Write,
    ReplacementTransform, DOWN, UP, RIGHT, LEFT, IN, OUT,
    LaggedStart, AnimationGroup, Dot, Dot3D, VGroup, Line,
    Text, MathTex, interpolate_color,
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
        self.pseudo = None

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
    # Pseudocode panel helper (Enhancement 2)                             #
    # ------------------------------------------------------------------ #

    def set_pseudo(self, active_index):
        """Show/update the pseudocode panel in the top-left corner.

        Builds a new panel with the given line highlighted, fades out the old
        one if present, then fades in the new one.
        """
        new_panel = B.pseudocode_panel(active_index)
        # Place top-left: align top edge near +UP, left edge near screen left.
        new_panel.to_corner(LEFT + UP, buff=0.15)
        self.add_fixed_in_frame_mobjects(new_panel)
        new_panel.set_opacity(0)
        if self.pseudo is None:
            self.play(new_panel.animate.set_opacity(1.0), run_time=S.T_FAST)
        else:
            old = self.pseudo
            self.play(FadeOut(old, shift=0.0), new_panel.animate.set_opacity(1.0), run_time=S.T_FAST)
            self.remove(old)
        self.pseudo = new_panel

    # ------------------------------------------------------------------ #
    # Section 1: raw point cloud on the Swiss roll                        #
    # ------------------------------------------------------------------ #

    def section_raw(self):
        self.next_section("step-1-raw", type=_SEC)
        self.set_camera_orientation(phi=65 * DEGREES, theta=30 * DEGREES, zoom=0.9)

        # Enhancement 1: ThreeDAxes behind the cloud (muted, thin).
        self.axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-4, 4, 1],
            x_length=8,
            y_length=8,
            z_length=8,
            axis_config={"stroke_color": S.MUTED, "stroke_width": 0.8, "stroke_opacity": 0.45},
        )
        # Axis labels (cheap: small Text anchored to axis tips).
        x_lbl = Text("x", font_size=20, color=S.MUTED).move_to(self.axes.x_axis.get_end() + RIGHT * 0.3)
        y_lbl = Text("y", font_size=20, color=S.MUTED).move_to(self.axes.y_axis.get_end() + UP * 0.3)
        z_lbl = Text("z", font_size=20, color=S.MUTED).move_to(self.axes.z_axis.get_end() + OUT * 0.3)
        self.axes_labels = VGroup(x_lbl, y_lbl, z_lbl)

        self.play(FadeIn(self.axes, run_time=S.T_FAST))
        self.add(self.axes_labels)

        self.cloud = B.point_cloud(self.data["points"], self.data["t"])
        self.play(FadeIn(self.cloud, run_time=S.T_INTRO))

        # Pseudocode panel: step 0 highlighted.
        self.set_pseudo(0)

        self.set_caption("A 2D sheet rolled up in 3D. The goal: recover the flat sheet.")
        # Full slow orbit to show off the 3D structure, returning to the start
        # orientation so the next step continues seamlessly.
        orbit_time = 9.0
        self.begin_ambient_camera_rotation(rate=2 * np.pi / orbit_time, about="theta")
        self.wait(orbit_time)
        self.stop_ambient_camera_rotation()
        self.wait(S.T_HOLD)

    # ------------------------------------------------------------------ #
    # Section 2: kNN graph edges                                          #
    # ------------------------------------------------------------------ #

    def section_knn(self):
        self.next_section("step-2-knn", type=_SEC)

        # Update pseudocode panel: step 1 highlighted.
        self.set_pseudo(1)

        # ------------------------------------------------------------------ #
        # Beat 1: clear stage for the schematic illustration                  #
        # ------------------------------------------------------------------ #
        # Fade out the real dataset and axes so the schematic reads on a clean
        # stage. Move camera to a front-on view so the synthetic star is centered.
        self.play(
            FadeOut(self.cloud),
            FadeOut(self.axes),
            FadeOut(self.axes_labels),
            run_time=S.T_FAST,
        )
        self.move_camera(
            phi=65 * DEGREES,
            theta=-90 * DEGREES,
            zoom=1.0,
            frame_center=[0, 0, 0],
            run_time=S.T_NORMAL,
        )

        self.set_caption("To build the graph, link each point to its nearest neighbors.")

        # ------------------------------------------------------------------ #
        # Beat 2: synthetic center sphere at the origin                       #
        # ------------------------------------------------------------------ #
        center_pt = np.array([0.0, 0.0, 0.0])
        center_sphere = Dot3D(point=center_pt, radius=0.14, color=S.ACCENT)
        self.play(FadeIn(center_sphere, run_time=S.T_FAST))
        self.wait(0.3)

        # ------------------------------------------------------------------ #
        # Beat 3: 6 synthetic neighbor points at hand-picked offsets          #
        # Camera phi=65, theta=-90: camera along -Y axis, so screen X=X,     #
        # screen vertical ~= Z. Y is depth only.                              #
        # Offsets use large X/Z spread and deliberately varied distances      #
        # (1.0 to 2.2) so labels display distinct numeric values.             #
        # ------------------------------------------------------------------ #
        offsets = np.array([
            [ 1.55,  0.20,  0.10],   # right,         d~1.56
            [ 0.70,  0.30,  1.70],   # upper-right,   d~1.86
            [-0.95,  0.35,  1.30],   # upper-left,    d~1.62
            [-1.60, -0.25, -0.15],   # left,          d~1.62
            [-0.65, -0.30, -1.50],   # lower-left,    d~1.66
            [ 0.95,  0.40, -1.20],   # lower-right,   d~1.55
        ], dtype=float)

        neighbor_pts = [center_pt + off for off in offsets]
        # Compute Euclidean distances for weight labels.
        distances = [float(np.linalg.norm(off)) for off in offsets]

        # ------------------------------------------------------------------ #
        # Beat 4: draw edges one at a time with neighbor sphere + label       #
        # ------------------------------------------------------------------ #
        self.set_caption("Each link is weighted by the distance between the points.")

        schematic_lines = VGroup()
        schematic_spheres = VGroup()
        schematic_labels = []

        for idx, (nbr_pt, dist) in enumerate(zip(neighbor_pts, distances)):
            # Edge from center to neighbor.
            seg = Line(
                start=center_pt,
                end=nbr_pt,
                stroke_width=2.8,
                color=S.ACCENT,
            ).set_opacity(0.85)

            # Neighbor sphere (Dot3D; small count, cheap).
            nbr_sphere = Dot3D(point=nbr_pt, radius=0.11, color=S.WARM)

            # Weight label placed near the neighbor end of the edge, offset
            # slightly outward (away from center) so it does not overlap the
            # neighbor sphere. Each label ends up near its own neighbor and
            # therefore far from every other edge's label.
            edge_dir = (nbr_pt - center_pt) / dist  # unit vector center -> neighbor
            label_pos = nbr_pt + edge_dir * 0.25
            lbl = Text(f"{dist:.2f}", font_size=22, color=S.INK).move_to(label_pos)

            schematic_lines.add(seg)
            schematic_spheres.add(nbr_sphere)
            schematic_labels.append(lbl)

            # Draw edge, fade in sphere and label; fully sequential.
            self.play(
                Create(seg),
                FadeIn(nbr_sphere),
                FadeIn(lbl),
                run_time=0.5,
            )

        # ------------------------------------------------------------------ #
        # Beat 5: hold so the finished star reads                             #
        # ------------------------------------------------------------------ #
        self.wait(S.T_HOLD)

        # ------------------------------------------------------------------ #
        # Beat 6: fade out schematic, fade dataset and axes back in           #
        # ------------------------------------------------------------------ #
        schematic_all = VGroup(center_sphere, schematic_lines, schematic_spheres,
                               *schematic_labels)
        self.play(FadeOut(schematic_all), run_time=S.T_FAST)

        # Restore working camera orientation and bring the real data back.
        self.move_camera(
            phi=65 * DEGREES,
            theta=30 * DEGREES,
            zoom=0.9,
            frame_center=[0, 0, 0],
            run_time=S.T_NORMAL,
        )
        self.play(
            FadeIn(self.cloud),
            FadeIn(self.axes),
            run_time=S.T_FAST,
        )
        self.add(self.axes_labels)

        # ------------------------------------------------------------------ #
        # Beat 7: illuminate the full kNN graph with a progressive sweep      #
        # ------------------------------------------------------------------ #
        self.set_caption("Do this for every point and the graph lights up across the data.")
        self.edges_mob = B.graph_edges(self.data["points"], self.data["edges"])
        # LaggedStart with a small lag_ratio so edges appear in a sweeping wave.
        self.play(
            LaggedStart(
                *[FadeIn(edge) for edge in self.edges_mob],
                lag_ratio=0.004,
                run_time=S.T_SLOW,
            )
        )

        # ------------------------------------------------------------------ #
        # Beat 8: roll the graph upright and orbit (existing behavior kept)   #
        # ------------------------------------------------------------------ #
        orbit_time = 8.0
        self.move_camera(phi=80 * DEGREES, theta=30 * DEGREES, run_time=S.T_NORMAL)
        self.begin_ambient_camera_rotation(rate=2 * np.pi / orbit_time, about="theta")
        self.wait(orbit_time)
        self.stop_ambient_camera_rotation()
        # Return to the working orientation for continuity into step 3.
        self.move_camera(phi=65 * DEGREES, theta=30 * DEGREES, zoom=0.9, run_time=S.T_NORMAL)
        self.wait(S.T_HOLD)

    # ------------------------------------------------------------------ #
    # Section 3: geodesic vs. Euclidean path                             #
    # ------------------------------------------------------------------ #

    def section_geodesic(self):
        self.next_section("step-3-geodesic", type=_SEC)

        # Update pseudocode panel: step 2 highlighted.
        self.set_pseudo(2)

        pts, path = self.data["points"], self.data["path"]
        src, tgt = self.data["src"], self.data["tgt"]
        dijk_order = self.data["dijkstra_order"]

        self.play(self.edges_mob.animate.set_opacity(0.06), run_time=S.T_FAST)

        # Enhancement 4: geodesic definition caption, then Dijkstra wavefront.
        self.set_caption("Geodesic: shortest distance along the surface, not straight through space.")
        self.wait(S.T_HOLD)

        # Dijkstra wavefront animation.
        # Subsample: advance in ~25 chunks so animation is fast.
        n_nodes = len(dijk_order)
        n_chunks = min(25, n_nodes)
        chunk_size = max(1, n_nodes // n_chunks)

        # Color map: settled nodes recolor from ACCENT to a sweep color.
        SWEEP_COLOR = S.GOOD

        # Pre-build a dict: node index -> Dot in self.cloud (same ordering as points).
        # self.cloud[i] is the Dot for points[i] (VGroup preserves insertion order).
        cloud_dots = self.cloud.submobjects

        # Mark src node specially before wavefront starts.
        src_dot = cloud_dots[src]
        self.play(src_dot.animate.set_color(S.ACCENT).set_opacity(1.0), run_time=S.T_FAST)
        self.wait(0.2)

        # Animate wavefront chunk by chunk.
        for chunk_start in range(0, n_nodes, chunk_size):
            chunk = dijk_order[chunk_start: chunk_start + chunk_size]
            if not chunk:
                continue
            anims = [
                cloud_dots[node].animate.set_color(SWEEP_COLOR).set_opacity(0.9)
                for node in chunk
                if node != src and node != tgt
            ]
            if anims:
                self.play(AnimationGroup(*anims, lag_ratio=0.0), run_time=0.15)

        # Highlight src and tgt after wavefront.
        self.play(
            cloud_dots[tgt].animate.set_color(S.WARM).set_opacity(1.0),
            run_time=S.T_FAST,
        )
        self.wait(S.T_HOLD)

        # Refinement 4: recolor the whole cloud by geodesic distance from src.
        dist = self.data["D"][self.data["src"]]
        geo_color_anims = B.recolor_cloud_by_values(self.cloud, dist, S.ACCENT, S.WARM)
        self.set_caption("Color shows geodesic distance from the source point.")
        self.play(AnimationGroup(*geo_color_anims, lag_ratio=0.0), run_time=S.T_SLOW)
        self.wait(S.T_HOLD)

        # Keep the geodesic-distance coloring; do NOT rebuild with t colors.
        self.play(self.edges_mob.animate.set_opacity(0.06), run_time=S.T_FAST)

        # Straight line vs. geodesic path.
        straight = B.straight_line(pts[src], pts[tgt])
        # Enhancement 5: gradient geodesic path.
        geo = B.path_polyline(pts, path, gradient=True)

        self.play(Create(straight, run_time=S.T_NORMAL))
        self.set_caption("Straight-line distance cuts through space, off the sheet.")

        # Orbit visibly while showing the straight-line and geodesic beats so the
        # depth contrast between the chord and the surface path is obvious.
        # Target roughly a half revolution over the combined duration.
        # Duration: T_HOLD (wait after straight) + T_SLOW (create geo) + set_caption
        # overhead + T_HOLD + T_SLOW (wait after geo caption) ~ 7-8 s total.
        # Rate = pi / 7.5 ~ 0.42 rad/s gives ~half revolution; clearly perceptible.
        orbit_rate_paths = np.pi / 7.5
        self.begin_ambient_camera_rotation(rate=orbit_rate_paths, about="theta")

        self.wait(S.T_HOLD)
        self.play(Create(geo, run_time=S.T_SLOW))
        self.set_caption("Geodesic distance follows the graph along the sheet.")
        self.wait(S.T_HOLD + S.T_SLOW)

        self.stop_ambient_camera_rotation()
        # Snap back to a clean orientation (same phi/theta as the section start) so
        # the transition into section_double_center is smooth.
        self.move_camera(phi=65 * DEGREES, theta=30 * DEGREES, run_time=S.T_FAST)
        self.geo, self.straight = geo, straight
        self.wait(S.T_HOLD)

    # ------------------------------------------------------------------ #
    # Section 4: double centering                                         #
    # ------------------------------------------------------------------ #

    def section_double_center(self):
        self.next_section("step-4-double-center", type=_SEC)

        # Update pseudocode panel: step 3 highlighted.
        self.set_pseudo(3)

        self.play(
            FadeOut(self.cloud),
            FadeOut(self.edges_mob),
            FadeOut(self.geo),
            FadeOut(self.straight),
            run_time=S.T_FAST,
        )

        # Fade out axes when scene flattens to matrix view (Enhancement 1).
        self.play(
            FadeOut(self.axes),
            FadeOut(self.axes_labels),
            run_time=S.T_FAST,
        )

        self.move_camera(phi=0, theta=-90 * DEGREES, zoom=1.0, run_time=S.T_NORMAL)

        # Scale all three objects to 0.45 and stack them with clear vertical separation
        # so they never overlap during transitions.
        # X_SHIFT moves the column right to clear the pseudocode panel (top-left corner).
        SCALE = 0.45
        X_SHIFT = 1.8
        Y_TOP = 2.8
        Y_MID = 0.7
        Y_BOT = -1.0

        # Enhancement 6: use D_sample and B_sample (real geodesic distances among
        # the 4 sampled path points) instead of generic excerpt_D / excerpt_B.
        dmat = B.matrix_grid(self.data["D_sample"], highlight_negative=False)
        dmat.scale(SCALE).move_to([X_SHIFT, Y_TOP, 0])
        self.add_fixed_in_frame_mobjects(dmat)
        self.play(FadeIn(dmat, run_time=S.T_NORMAL))
        # Refinement 6: explain the why with sequenced captions.
        self.set_caption("Distances alone do not place points; we need inner products.")
        self.wait(S.T_HOLD)
        self.set_caption("Square the distances: |x_i - x_j|^2 expands into inner products plus norms.")
        self.wait(S.T_HOLD)

        f = B.formula(r"B = -\tfrac{1}{2}\, J\, D^2\, J")
        f.scale(0.85).move_to([X_SHIFT, Y_MID, 0])
        self.add_fixed_in_frame_mobjects(f)
        self.play(Write(f, run_time=S.T_NORMAL))
        self.set_caption(
            "Center it: subtract the row and column means to fix the origin at the centroid."
        )
        self.wait(S.T_HOLD)

        bmat = B.matrix_grid(self.data["B_sample"], highlight_negative=True)
        bmat.scale(SCALE).move_to([X_SHIFT, Y_BOT, 0])
        bmat.set_opacity(0)
        self.add_fixed_in_frame_mobjects(bmat)
        self.play(bmat.animate.set_opacity(1.0), run_time=S.T_SLOW)
        self.set_caption("What remains is B: the inner products of centered points, ready to factor.")
        self.formula_dc, self.bmat, self.dmat = f, bmat, dmat
        self.wait(S.T_HOLD)

    # ------------------------------------------------------------------ #
    # Section 5: eigendecomposition                                       #
    # ------------------------------------------------------------------ #

    def section_eigendecomp(self):
        self.next_section("step-5-eigendecomp", type=_SEC)

        # Update pseudocode panel: step 4 highlighted.
        self.set_pseudo(4)

        l1, l2 = float(self.data["eigvals"][0]), float(self.data["eigvals"][1])
        power_rayleigh = self.data["power_rayleigh"]

        # Fade out D and the formula; keep B on screen (eigen content derives from B).
        self.play(
            FadeOut(self.dmat),
            FadeOut(self.formula_dc),
            run_time=S.T_FAST,
        )

        # Move B to the left side so the right side is free for eigen content.
        self.play(self.bmat.animate.move_to([-3.2, 0, 0]), run_time=S.T_FAST)

        # Enhancement 7: show power iteration finding the top eigenvector.
        # Update rule label placed above the Rayleigh counter.
        update_rule = B.formula(r"v \leftarrow Bv\,/\,\|Bv\|").move_to([1.5, 1.8, 0])
        self.add_fixed_in_frame_mobjects(update_rule)
        self.play(FadeIn(update_rule, run_time=S.T_NORMAL))
        self.set_caption("Power iteration: multiply B by a vector, normalize, repeat.")
        self.wait(S.T_HOLD)

        # Rayleigh quotient display: show number climbing toward lambda1.
        # We'll show a selection of iterations for readability.
        # Show iterations: 0 (initial), 1, 2, then 10 (converged).
        show_iters = [0, 1, 2, len(power_rayleigh) - 1]
        rayleigh_label = None
        for it in show_iters:
            rq_val = power_rayleigh[it]
            new_rq = B.formula(
                rf"v^T B v = {rq_val:.1f}"
            ).move_to([1.5, 0.6, 0])
            self.add_fixed_in_frame_mobjects(new_rq)
            if rayleigh_label is None:
                new_rq.set_opacity(0)
                self.play(new_rq.animate.set_opacity(1.0), run_time=S.T_FAST)
            else:
                old_rq = rayleigh_label
                self.play(
                    FadeOut(old_rq, shift=0.0),
                    new_rq.animate.set_opacity(1.0) if new_rq.get_opacity() == 1.0 else new_rq.animate.set_opacity(1.0),
                    run_time=S.T_FAST,
                )
                self.remove(old_rq)
            rayleigh_label = new_rq
            self.wait(0.45)

        # Dashed line marking lambda1 (target).
        target_label = B.formula(
            rf"\lambda_1 = {l1:.1f}"
        ).move_to([1.5, -0.3, 0]).set_color(S.ACCENT)
        self.add_fixed_in_frame_mobjects(target_label)
        self.play(FadeIn(target_label, run_time=S.T_FAST))
        self.set_caption("The Rayleigh quotient converges to the dominant eigenvalue.")
        self.wait(S.T_HOLD)

        # Keep only the top two eigenvalues stated.
        self.play(
            FadeOut(update_rule),
            FadeOut(rayleigh_label),
            FadeOut(target_label),
            run_time=S.T_FAST,
        )

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

        # Update pseudocode panel: step 5 highlighted.
        self.set_pseudo(5)

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
        # Refinement 4 cont: color the embedded cloud by geodesic distance (consistent
        # with the 3D cloud coloring introduced in section_geodesic).
        dist = self.data["D"][self.data["src"]]
        dist_arr = np.asarray(dist, dtype=float)
        dmin, dmax = float(dist_arr.min()), float(dist_arr.max())
        span = max(1e-9, dmax - dmin)
        flat = VGroup(*[
            Dot(
                point=pts3[i],
                radius=S.DOT_RADIUS,
                fill_opacity=1.0,
                color=interpolate_color(S.ACCENT, S.WARM, (dist_arr[i] - dmin) / span),
            )
            for i in range(pts3.shape[0])
        ])

        # Place formula at top-right to avoid the pseudocode panel (top-left).
        f3 = B.formula(
            r"Y = [\sqrt{\lambda_1}\,v_1,\ \sqrt{\lambda_2}\,v_2]"
        ).to_corner(RIGHT + UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(f3)
        self.play(
            FadeIn(flat, run_time=S.T_SLOW),
            Write(f3, run_time=S.T_NORMAL),
        )
        self.set_caption("The sheet unrolls into 2D. Color shows geodesic distance from the source, preserved.")
        self.wait(S.T_SLOW)
