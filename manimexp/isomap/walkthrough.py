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
    Text, MathTex, interpolate_color, Indicate, Arrow, GrowArrow,
)
from manim.scene.section import DefaultSectionType
from manimexp.isomap import style as S
from manimexp.isomap import builders as B
from manimexp.isomap.data import build_dataset

N = int(os.environ.get("MFI_N", "1000"))
K = 8
SEED = 0
DATASET = os.environ.get("MFI_DATASET", "swiss_roll")

# Per-dataset opening line. The rest of the script speaks generically of "the
# surface" so it reads correctly for any developable shape.
DATASET_INTRO = {
    "swiss_roll": "A 2D sheet rolled up in 3D. The goal: recover the flat sheet.",
    "s_curve": "A 2D sheet bent into an S in 3D. The goal: recover the flat sheet.",
    "twin_peaks": "A bumpy height surface in 3D. The goal: recover its flat layout.",
    "saddle": "A curved saddle surface in 3D. The goal: recover its flat layout.",
    "cylinder": "A sheet wrapped into a cylinder. The goal: unroll it back to flat.",
    "severed_sphere": "A sphere with its cap removed, an open curved surface. The goal: flatten it.",
    "helix": "A ribbon wound into a helix in 3D. The goal: unroll it to a flat strip.",
    "trefoil_knot": "A ribbon tied into a trefoil knot. The goal: unroll it to a flat strip.",
    "toroidal_helix": "A ribbon coiled around a torus. The goal: unroll it to a flat strip.",
    "spiral_disk": "A ribbon wound into a spiral. The goal: unroll it to a flat strip.",
    # Limitation cases: Isomap cannot cleanly flatten these.
    "full_sphere": "A full sphere, a closed surface. A sphere has no flat layout, so watch Isomap struggle.",
    "hilbert": "A Hilbert curve: a 1D path folded to fill a cube. It is a curve, not a surface.",
    "clusters_3d": "Separate clusters of points, with no surface connecting them.",
}

# Per-dataset closing caption for the 2D embedding. The developable shapes unroll
# cleanly; the limitation cases below explain why the result is not a clean map.
DATASET_OUTRO = {
    "full_sphere": "A sphere cannot be flattened without distortion, so the layout squashes it. Isomap needs a developable surface.",
    "hilbert": "A folded curve has no 2D layout; nearby folds get linked, so it collapses. Isomap needs a true surface.",
    "clusters_3d": "With no surface connecting the clusters, geodesic distance between them is undefined, so Isomap cannot place them meaningfully.",
}

_SEC = DefaultSectionType.NORMAL


class IsomapWalkthrough(ThreeDScene):

    def construct(self):
        self.camera.background_color = S.BG
        self.data = build_dataset(n=N, k=K, seed=SEED, dataset=DATASET)
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
        if os.environ.get("MFI_CAPTION_LOG"):
            t = getattr(self.renderer, "time", 0.0) or 0.0
            print(f"[CAPTION] {t:7.2f} | {text}", flush=True)
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
        if os.environ.get("MFI_CAPTION_LOG"):
            t = getattr(self.renderer, "time", 0.0) or 0.0
            print(f"[SECTION] {t:7.2f} | pseudo step {active_index}", flush=True)
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

        self.set_caption(DATASET_INTRO.get(DATASET, "A curved 2D surface in 3D. The goal: recover its flat layout."))
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
        # The schematic illustration is a flat 2D diagram: look straight down the
        # z-axis (phi=0) so the star of neighbors lies in the screen plane with no
        # perspective foreshortening. Everything below lives in the xy-plane (z=0).
        self.move_camera(
            phi=0,
            theta=-90 * DEGREES,
            zoom=1.0,
            frame_center=[0, 0, 0],
            run_time=S.T_NORMAL,
        )

        self.set_caption("To build the graph, link each point to its nearest neighbors.")

        # ------------------------------------------------------------------ #
        # Beat 2: center node at the origin (flat 2D Dot)                     #
        # ------------------------------------------------------------------ #
        center_pt = np.array([0.0, 0.0, 0.0])
        center_sphere = Dot(point=center_pt, radius=0.10, color=S.ACCENT)
        self.play(FadeIn(center_sphere, run_time=S.T_FAST))

        # ------------------------------------------------------------------ #
        # Beat 3: the nearest neighbors themselves appear under this caption,  #
        # so the "link each point to its nearest neighbors" idea has screen    #
        # time before weighting is introduced. Distances are varied so the     #
        # weight labels (next beat) show distinct numbers.                     #
        # ------------------------------------------------------------------ #
        offsets = np.array([
            [ 1.60,  0.30, 0.0],   # right,        d~1.63
            [ 0.90,  1.55, 0.0],   # upper-right,  d~1.79
            [-1.00,  1.30, 0.0],   # upper-left,   d~1.64
            [-1.65, -0.30, 0.0],   # left,         d~1.68
            [-0.80, -1.40, 0.0],   # lower-left,   d~1.61
            [ 1.00, -1.30, 0.0],   # lower-right,  d~1.64
        ], dtype=float)

        neighbor_pts = [center_pt + off for off in offsets]
        # Compute Euclidean distances for weight labels.
        distances = [float(np.linalg.norm(off)) for off in offsets]

        neighbor_dots = VGroup(*[
            Dot(point=p, radius=0.08, color=S.WARM) for p in neighbor_pts
        ])
        self.play(
            LaggedStart(*[FadeIn(d) for d in neighbor_dots],
                        lag_ratio=0.18, run_time=S.T_NORMAL)
        )
        self.wait(2.4)

        # ------------------------------------------------------------------ #
        # Beat 4: draw the weighted links one at a time with midpoint labels   #
        # ------------------------------------------------------------------ #
        self.set_caption("Each link is weighted by the distance between the points.")

        schematic_lines = VGroup()
        schematic_labels = []

        for idx, (nbr_pt, dist) in enumerate(zip(neighbor_pts, distances)):
            # Edge from center to neighbor.
            seg = Line(
                start=center_pt,
                end=nbr_pt,
                stroke_width=2.8,
                color=S.ACCENT,
            ).set_opacity(0.85)

            # Weight label sits at the midpoint of the edge, nudged perpendicular
            # to the edge so the number rests beside the line rather than on it.
            edge_dir = (nbr_pt - center_pt) / dist  # unit vector center -> neighbor
            perp = np.array([-edge_dir[1], edge_dir[0], 0.0])
            midpoint = (center_pt + nbr_pt) / 2.0
            label_pos = midpoint + perp * 0.24
            lbl = Text(f"{dist:.2f}", font_size=22, color=S.INK).move_to(label_pos)

            schematic_lines.add(seg)
            schematic_labels.append(lbl)

            # Draw edge and fade in its weight label; the neighbor node is
            # already on screen from beat 3.
            self.play(
                Create(seg),
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
        schematic_all = VGroup(center_sphere, schematic_lines, neighbor_dots,
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
        # Beat 8: roll the graph upright and start a continuous orbit         #
        # The rotation is intentionally NOT stopped or reverted here: it keeps #
        # running straight into the geodesic section so the cloud spins        #
        # without a pause or a snap back to the starting orientation.          #
        # ------------------------------------------------------------------ #
        orbit_time = 16.0
        self.move_camera(phi=80 * DEGREES, theta=30 * DEGREES, run_time=S.T_NORMAL)
        self.begin_ambient_camera_rotation(rate=2 * np.pi / orbit_time, about="theta")
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

        # The spoken geodesic definition is omitted from the video (it lives in the
        # text companion); the wavefront and recolor share one caption instead.
        # Dijkstra wavefront animation.
        # Advance in exactly n_chunks steps regardless of the point count, so the
        # video timeline (and therefore the companion timestamps) is identical at
        # any N. A floor-division chunk_size would yield a varying step count.
        n_nodes = len(dijk_order)
        n_chunks = min(25, n_nodes)

        # Pre-build a dict: node index -> Dot in self.cloud (same ordering as points).
        # self.cloud[i] is the Dot for points[i] (VGroup preserves insertion order).
        cloud_dots = self.cloud.submobjects

        # Per-node saturated geodesic color: as each node settles it lights up in
        # its FINAL high-saturation rainbow color (by geodesic distance from src),
        # not a flat sweep color. The points are therefore clearly and vividly
        # colored while the caption is on screen, instead of only at the end.
        dist = self.data["D"][src]
        dmin = float(dist.min())
        dspan = max(1e-9, float(dist.max()) - dmin)
        GROW = 1.7

        def geo_color(node):
            return B.rainbow_color((dist[node] - dmin) / dspan)

        self.set_caption("Color shows geodesic distance from the source point.")

        # Mark the source point so "from the source point" is identifiable while
        # the wavefront spreads; it settles into its gradient color afterward.
        src_dot = cloud_dots[src]
        self.play(src_dot.animate.set_color(S.ACCENT).set_opacity(1.0), run_time=S.T_FAST)
        self.wait(0.2)

        # Animate wavefront in exactly n_chunks equal slices of the settle order.
        # Each settled node grows and takes its saturated geodesic color at once.
        for c in range(n_chunks):
            lo = c * n_nodes // n_chunks
            hi = (c + 1) * n_nodes // n_chunks
            chunk = dijk_order[lo:hi]
            anims = []
            for node in chunk:
                if node == src or node == tgt:
                    continue
                dot = cloud_dots[node]
                dot.set_sheen(0.0)
                anims.append(dot.animate.set_color(geo_color(node)).set_opacity(1.0).scale(GROW))
            if anims:
                self.play(AnimationGroup(*anims, lag_ratio=0.0), run_time=0.15)

        # Highlight tgt after the wavefront, then settle src and tgt into the same
        # saturated geodesic gradient so the whole cloud reads as one color scale.
        tgt_dot = cloud_dots[tgt]
        self.play(tgt_dot.animate.set_color(S.WARM).set_opacity(1.0), run_time=S.T_FAST)
        self.wait(S.T_HOLD)

        for dot, node in ((src_dot, src), (tgt_dot, tgt)):
            dot.set_sheen(0.0)
        self.play(
            src_dot.animate.set_color(geo_color(src)).set_opacity(1.0).scale(GROW),
            tgt_dot.animate.set_color(geo_color(tgt)).set_opacity(1.0).scale(GROW),
            run_time=S.T_NORMAL,
        )
        self.wait(S.T_HOLD)

        # Keep the geodesic-distance coloring; do NOT rebuild with t colors.
        self.play(self.edges_mob.animate.set_opacity(0.06), run_time=S.T_FAST)

        # Straight line vs. geodesic path.
        straight = B.straight_line(pts[src], pts[tgt])
        # Enhancement 5: gradient geodesic path.
        geo = B.path_polyline(pts, path, gradient=True)

        # The camera is already orbiting continuously from the kNN section, so the
        # straight-line and geodesic beats play while the cloud keeps spinning.
        # No new rotation is started and the orientation is never snapped back.
        self.play(Create(straight, run_time=S.T_NORMAL))
        self.set_caption("Straight-line distance cuts through space, off the surface.")

        self.wait(S.T_HOLD)
        self.play(Create(geo, run_time=S.T_SLOW))
        self.set_caption("Geodesic distance follows the graph along the surface.")
        self.wait(S.T_HOLD + S.T_SLOW)

        # Leave the orbit running: the dataset keeps rotating into the fade-out at
        # the start of the next section, rather than freezing before it disappears.
        self.geo, self.straight = geo, straight
        self.wait(S.T_HOLD)

    # ------------------------------------------------------------------ #
    # Section 4: double centering                                         #
    # ------------------------------------------------------------------ #

    def section_double_center(self):
        self.next_section("step-4-double-center", type=_SEC)

        # Update pseudocode panel: step 3 highlighted.
        self.set_pseudo(3)

        # The cloud is still orbiting from the previous section: fade it out while
        # it keeps rotating, then stop the orbit and flatten to the matrix view.
        self.play(
            FadeOut(self.cloud),
            FadeOut(self.edges_mob),
            FadeOut(self.geo),
            FadeOut(self.straight),
            run_time=S.T_SLOW,
        )
        self.stop_ambient_camera_rotation()

        # Fade out axes when scene flattens to matrix view (Enhancement 1).
        self.play(
            FadeOut(self.axes),
            FadeOut(self.axes_labels),
            run_time=S.T_FAST,
        )

        self.move_camera(phi=0, theta=-90 * DEGREES, zoom=1.0, run_time=S.T_NORMAL)

        SCALE = 0.38

        # ------------------------------------------------------------------ #
        # Phase 1: the goal, then build D cell by cell with point provenance  #
        # ------------------------------------------------------------------ #
        self.set_caption("We have geodesic distances. We want coordinates: a position for each point.")

        # Four labeled sample points act as the visual source for every D entry.
        pt_pos = [
            np.array([0.2, 2.6, 0.0]),
            np.array([1.2, 2.95, 0.0]),
            np.array([2.2, 2.5, 0.0]),
            np.array([3.2, 2.95, 0.0]),
        ]
        sample_dots = VGroup(*[Dot(point=p, radius=0.09, color=S.WARM) for p in pt_pos])
        sample_labels = VGroup(*[
            Text(str(i + 1), font_size=20, color=S.INK).next_to(sample_dots[i], UP, buff=0.08)
            for i in range(4)
        ])
        sample_grp = VGroup(sample_dots, sample_labels)
        self.add_fixed_in_frame_mobjects(sample_grp)
        self.play(FadeIn(sample_dots), FadeIn(sample_labels), run_time=S.T_FAST)

        # Empty D grid in the center; numbers are hidden until their cell fills.
        dmat = B.matrix_grid(self.data["D_sample"], highlight_negative=False)
        dmat.scale(SCALE).move_to([1.0, -0.3, 0])
        for r in range(4):
            for c in range(4):
                dmat.get_entries((r + 1, c + 1)).set_opacity(0)
        self.add_fixed_in_frame_mobjects(dmat)
        self.play(FadeIn(dmat), run_time=S.T_FAST)
        self.wait(1.8)

        self.set_caption("Each entry is the distance between two points; the diagonal is zero.")

        # Diagonal first: a point's distance to itself is zero.
        for i in range(4):
            e = dmat.get_entries((i + 1, i + 1))
            self.play(
                Indicate(sample_dots[i], color=S.GOOD, scale_factor=1.5),
                e.animate.set_opacity(1.0),
                run_time=0.3,
            )

        # Off-diagonal pairs: draw the link between the two points, reveal both
        # symmetric cells, then drop the link.
        for i in range(4):
            for j in range(i + 1, 4):
                link = Line(pt_pos[i], pt_pos[j], stroke_width=3, color=S.GOOD).set_opacity(0.9)
                self.add_fixed_in_frame_mobjects(link)
                e1 = dmat.get_entries((i + 1, j + 1))
                e2 = dmat.get_entries((j + 1, i + 1))
                self.play(
                    Create(link),
                    sample_dots[i].animate.set_color(S.GOOD),
                    sample_dots[j].animate.set_color(S.GOOD),
                    e1.animate.set_opacity(1.0),
                    e2.animate.set_opacity(1.0),
                    run_time=0.45,
                )
                self.play(
                    FadeOut(link),
                    sample_dots[i].animate.set_color(S.WARM),
                    sample_dots[j].animate.set_color(S.WARM),
                    run_time=0.18,
                )
        self.wait(S.T_HOLD)

        # ------------------------------------------------------------------ #
        # Phase 2: motivate inner products (the Gram matrix and why it helps) #
        # Each idea is its own short caption, held on screen for T_READ so it  #
        # can be read before the next one replaces it. The full derivation     #
        # lives in manimexp/isomap/double_centering_explained.md.              #
        # ------------------------------------------------------------------ #
        self.play(FadeOut(sample_grp), run_time=S.T_FAST)
        self.play(dmat.animate.move_to([-4.6, 0.0, 0]), run_time=S.T_NORMAL)

        T_READ = 3.6

        self.set_caption("Since rotating or shifting all the points together would yield identical pairwise distances,")
        self.wait(T_READ)
        self.set_caption("distances alone could not create unique coordinates, since many placements share one distance table.")
        self.wait(T_READ)

        f_ip = B.formula(r"G_{ij} = x_i \cdot x_j").scale(0.75).move_to([1.4, 1.6, 0])
        self.add_fixed_in_frame_mobjects(f_ip)
        self.play(Write(f_ip), run_time=S.T_NORMAL)
        self.set_caption("Instead, we can base it off of the inner product of x_i and x_j and the angle between them from a shared origin.")
        self.wait(T_READ)

        f_gram = B.formula(r"G = X X^{\top}").scale(0.75).move_to([1.4, 0.6, 0])
        self.add_fixed_in_frame_mobjects(f_gram)
        self.play(Write(f_gram), run_time=S.T_NORMAL)
        self.set_caption("The Gram matrix G is the result of collecting every inner product, equal to X times X-transpose.")
        self.wait(T_READ)

        self.set_caption("G captures the geometric relationships from relative geometry, a bridge between distances and coordinates.")
        self.wait(T_READ)

        f_factor = B.formula(r"G = V\,\Lambda\,V^{\top}").scale(0.75).move_to([1.4, -0.5, 0])
        self.add_fixed_in_frame_mobjects(f_factor)
        self.play(Write(f_factor), run_time=S.T_NORMAL)
        self.set_caption("By eigendecomposing the Gram matrix, we find the eigenvectors that best explain the data's geometry.")
        self.wait(T_READ)

        # ------------------------------------------------------------------ #
        # Phase 3: the transform  D -> square entries -> double-center -> B    #
        # The three tables are spread across the frame so each transform        #
        # formula has room above its arrow. New mobjects are added at opacity 0  #
        # and animated up (never add-then-FadeIn) so no table flashes at full    #
        # opacity for a frame before settling.                                   #
        # ------------------------------------------------------------------ #
        self.play(FadeOut(f_ip), FadeOut(f_gram), FadeOut(f_factor), run_time=S.T_FAST)

        # Square every entry of D.
        D_sq = (np.asarray(self.data["D_sample"], dtype=float) ** 2).tolist()
        d2mat = B.matrix_grid(D_sq, highlight_negative=False).scale(SCALE).move_to([-0.2, 0.0, 0])
        arrow1 = Arrow([-3.7, 0, 0], [-1.3, 0, 0], buff=0.05, color=S.MUTED, stroke_width=3)
        sq_lab = B.formula(r"(\cdot)^2").scale(0.85).next_to(arrow1, UP, buff=0.12)
        for m in (arrow1, sq_lab, d2mat):
            m.set_opacity(0)
        self.add_fixed_in_frame_mobjects(arrow1, sq_lab, d2mat)
        self.set_caption("We cannot form G from coordinates we do not have, so we build it from the distances instead.")
        self.wait(3.4)
        self.set_caption("First, square every entry of the distance matrix.")
        self.play(arrow1.animate.set_opacity(1.0), sq_lab.animate.set_opacity(1.0), run_time=S.T_FAST)
        self.play(d2mat.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.wait(2.8)

        # Double-center to land on the Gram matrix B.
        bmat = B.matrix_grid(self.data["B_sample"], highlight_negative=True).scale(SCALE).move_to([4.4, 0.0, 0])
        arrow2 = Arrow([0.9, 0, 0], [3.3, 0, 0], buff=0.05, color=S.MUTED, stroke_width=3)
        dc_lab = B.formula(r"-\tfrac12\, J(\cdot)J").scale(0.72).next_to(arrow2, UP, buff=0.12)
        for m in (arrow2, dc_lab, bmat):
            m.set_opacity(0)
        self.add_fixed_in_frame_mobjects(arrow2, dc_lab, bmat)
        self.set_caption("Then double-center it: subtract each row and column mean, and scale by minus one half.")
        self.play(arrow2.animate.set_opacity(1.0), dc_lab.animate.set_opacity(1.0), run_time=S.T_FAST)
        self.play(bmat.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.wait(3.0)

        self.set_caption("What remains is B, the Gram matrix of inner products, ready to factor into coordinates.")
        self.wait(2.2)

        # formula_dc bundles the pipeline middle so the next section can clear it
        # while keeping D (dmat) and the Gram matrix B (bmat).
        self.formula_dc = VGroup(arrow1, sq_lab, d2mat, arrow2, dc_lab)
        self.bmat, self.dmat = bmat, dmat
        self.wait(S.T_HOLD)

    # ------------------------------------------------------------------ #
    # Section 5: eigendecomposition                                       #
    # ------------------------------------------------------------------ #

    def section_eigendecomp(self):
        self.next_section("step-5-eigendecomp", type=_SEC)

        # Update pseudocode panel: step 4 highlighted.
        self.set_pseudo(4)

        # Power iteration is shown on the visible 4x4 Gram matrix B (the sample),
        # with real vectors and real v^T B v values at every step.
        vecs = self.data["sample_power_vectors"]
        rqs = self.data["sample_power_rayleigh"]
        samp_eig = self.data["sample_eigvals"]
        lam1, lam2 = float(samp_eig[0]), float(samp_eig[1])

        # Clear D and the pipeline middle; keep the Gram matrix B.
        self.play(FadeOut(self.dmat), FadeOut(self.formula_dc), run_time=S.T_FAST)

        # Center B for a v^T B v product layout.
        Bm = self.bmat
        self.play(Bm.animate.move_to([0.0, 0.4, 0]), run_time=S.T_FAST)

        # Scale the vectors so their entries match B's cell size.
        vscale = Bm.height / B.colvec(vecs[0]).height

        # -------- where the vector comes from: a random unit vector -------- #
        vcol = B.colvec(vecs[0]).scale(vscale).next_to(Bm, RIGHT, buff=0.35)
        origin_lbl = B.formula(r"v_0:\ \text{random},\ \lVert v\rVert = 1").scale(0.62)
        origin_lbl.next_to(vcol, UP, buff=0.35)
        for m in (vcol, origin_lbl):
            m.set_opacity(0)
        self.add_fixed_in_frame_mobjects(vcol, origin_lbl)
        self.set_caption("Power iteration starts from a random unit vector v.")
        self.play(vcol.animate.set_opacity(1.0), origin_lbl.animate.set_opacity(1.0),
                  run_time=S.T_NORMAL)
        self.wait(2.0)
        self.play(origin_lbl.animate.set_opacity(0.0), run_time=S.T_FAST)
        self.remove(origin_lbl)

        # -------- the update rule on top -------- #
        update_rule = B.formula(r"v \leftarrow Bv\,/\,\lVert Bv\rVert").scale(0.85)
        update_rule.move_to([0.0, 2.7, 0]).set_opacity(0)
        self.add_fixed_in_frame_mobjects(update_rule)
        self.play(update_rule.animate.set_opacity(1.0), run_time=S.T_FAST)
        self.set_caption("Each step multiplies v by B, then divides by its length.")
        self.wait(2.6)

        # -------- the actual v^T B v product -------- #
        vrow = B.rowvec(vecs[0]).scale(vscale).next_to(Bm, LEFT, buff=0.35)
        eqres = B.formula(rf"=\ {rqs[0]:.2f}").scale(0.85).next_to(vcol, RIGHT, buff=0.35)
        lbl_vt = B.formula(r"v^{\top}").scale(0.7).next_to(vrow, UP, buff=0.2)
        lbl_Bm = B.formula(r"B").scale(0.7).next_to(Bm, UP, buff=0.2)
        lbl_v = B.formula(r"v").scale(0.7).next_to(vcol, UP, buff=0.2)
        for m in (vrow, eqres, lbl_vt, lbl_Bm, lbl_v):
            m.set_opacity(0)
        self.add_fixed_in_frame_mobjects(vrow, eqres, lbl_vt, lbl_Bm, lbl_v)
        self.set_caption("The value is v-transpose times B times v: an actual matrix product.")
        self.play(*[m.animate.set_opacity(1.0) for m in (vrow, eqres, lbl_vt, lbl_Bm, lbl_v)],
                  run_time=S.T_NORMAL)
        self.wait(2.2)

        it_lbl = B.formula(r"\text{iteration } 0").scale(0.62).to_corner(RIGHT + UP, buff=0.4)
        it_lbl.set_opacity(0)
        self.add_fixed_in_frame_mobjects(it_lbl)
        self.play(it_lbl.animate.set_opacity(1.0), run_time=S.T_FAST)

        # Crossfade the row, column, value, and counter each iteration. Fixed-in-
        # frame mobjects are updated by animating opacity and then removing the
        # old ones (ReplacementTransform leaves stray fixed-frame copies behind).
        self.set_caption("Multiply by B and renormalize; the value climbs each iteration.")
        for k in (1, 2, 3):
            new_vrow = B.rowvec(vecs[k]).scale(vscale).next_to(Bm, LEFT, buff=0.35).set_opacity(0)
            new_vcol = B.colvec(vecs[k]).scale(vscale).next_to(Bm, RIGHT, buff=0.35).set_opacity(0)
            new_eq = B.formula(rf"=\ {rqs[k]:.2f}").scale(0.85).next_to(new_vcol, RIGHT, buff=0.35).set_opacity(0)
            new_it = B.formula(rf"\text{{iteration }} {k}").scale(0.62).to_corner(RIGHT + UP, buff=0.4).set_opacity(0)
            self.add_fixed_in_frame_mobjects(new_vrow, new_vcol, new_eq, new_it)
            self.play(
                vrow.animate.set_opacity(0), vcol.animate.set_opacity(0),
                eqres.animate.set_opacity(0), it_lbl.animate.set_opacity(0),
                new_vrow.animate.set_opacity(1), new_vcol.animate.set_opacity(1),
                new_eq.animate.set_opacity(1), new_it.animate.set_opacity(1),
                run_time=S.T_NORMAL,
            )
            self.remove(vrow, vcol, eqres, it_lbl)
            vrow, vcol, eqres, it_lbl = new_vrow, new_vcol, new_eq, new_it
            self.wait(1.4)

        # The converged value is the top eigenvalue.
        lam_lbl = B.formula(rf"\lambda_1 = {lam1:.2f}").scale(0.85).set_color(S.ACCENT)
        lam_lbl.next_to(eqres, DOWN, buff=0.5).set_opacity(0)
        self.add_fixed_in_frame_mobjects(lam_lbl)
        self.set_caption("The value settles at the top eigenvalue lambda-1.")
        self.play(lam_lbl.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.wait(2.2)

        # -------- eigen equation and the top two eigenvalues -------- #
        # Explicit opacity-to-zero then remove (not FadeOut) so no fixed-in-frame
        # remnant survives into the eigen-equation beat.
        clear = [vrow, vcol, eqres, lam_lbl, update_rule, it_lbl, lbl_vt, lbl_Bm, lbl_v]
        self.play(*[m.animate.set_opacity(0) for m in clear], run_time=S.T_FAST)
        self.remove(*clear)
        self.play(Bm.animate.move_to([-3.2, 0, 0]), run_time=S.T_FAST)

        f2 = B.formula(r"B v_i = \lambda_i v_i").move_to([1.6, 1.0, 0]).set_opacity(0)
        self.add_fixed_in_frame_mobjects(f2)
        self.play(f2.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.set_caption("Every eigenvector satisfies B v = λ v; the largest eigenvalues carry the shape.")
        self.wait(2.2)

        vals = B.formula(
            rf"\lambda_1 = {lam1:.2f}\quad \lambda_2 = {lam2:.2f}"
        ).move_to([1.6, -0.7, 0]).set_opacity(0)
        self.add_fixed_in_frame_mobjects(vals)
        self.play(vals.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.set_caption("Keep the top two eigenvalues: they span the recovered plane.")
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
        # Color the embedded cloud by geodesic distance with the same high-
        # saturation rainbow used in section_geodesic, so the preserved gradient
        # is immediately recognizable.
        dist = self.data["D"][self.data["src"]]
        dist_arr = np.asarray(dist, dtype=float)
        dmin, dmax = float(dist_arr.min()), float(dist_arr.max())
        span = max(1e-9, dmax - dmin)
        flat = VGroup(*[
            Dot(
                point=pts3[i],
                radius=S.DOT_RADIUS,
                fill_opacity=1.0,
                color=B.rainbow_color((dist_arr[i] - dmin) / span),
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
        self.set_caption(DATASET_OUTRO.get(DATASET, "The surface unrolls into 2D. Color shows geodesic distance from the source, preserved."))
        self.wait(4.0)
