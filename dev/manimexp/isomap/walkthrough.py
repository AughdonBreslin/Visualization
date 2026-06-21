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
from manimexp.isomap.data import build_dataset, double_center_squared

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
    "swiss_roll": "The surface unrolls into 2D, with the geodesic-distance coloring preserved.",
    "s_curve": "The bent sheet unrolls into a flat rectangle, the geodesic coloring preserved.",
    "twin_peaks": "The bumpy surface flattens into 2D, stretched where it curved most.",
    "saddle": "The saddle flattens into 2D, its geodesic coloring preserved.",
    "cylinder": "A closed band has no flat sheet, so the cylinder lays out as a loop.",
    "severed_sphere": "The open cap flattens into a disk, with mild stretching from its curvature.",
    "helix": "A 1D ribbon has no area to fill, so the helix lays out as a thin arc.",
    "trefoil_knot": "A closed ribbon cannot become a flat strip, so the knot lays out as a band.",
    "toroidal_helix": "A closed coil lays out as a band, not a flat strip.",
    "spiral_disk": "The wound spiral unrolls into a thin curved strip.",
    "full_sphere": "A sphere cannot flatten without distortion, so Isomap squashes it; it needs a developable surface.",
    "hilbert": "A folded curve has no 2D layout; nearby folds get linked, so it collapses.",
    "clusters_3d": "With no surface joining the clusters, geodesic distance between them is undefined, so Isomap cannot place them.",
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
        self.set_caption("Repeat for every point and the graph spans the surface.")
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
        # Bring the kNN edges back up so the graph stays visible while the
        # geodesic path (which follows those edges) and the straight line are drawn.
        self.play(self.edges_mob.animate.set_opacity(0.32), run_time=S.T_FAST)

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

        self.set_caption("Rotating or shifting all the points leaves every pairwise distance unchanged.")
        self.wait(T_READ)
        self.set_caption("So distances alone cannot fix coordinates: many layouts share one distance table.")
        self.wait(T_READ)

        f_ip = B.formula(r"G_{ij} = x_i \cdot x_j").scale(0.75).move_to([1.4, 1.6, 0])
        self.add_fixed_in_frame_mobjects(f_ip)
        self.play(Write(f_ip), run_time=S.T_NORMAL)
        self.set_caption("Instead, use the inner product of two points, which captures the angle between them from a shared origin.")
        self.wait(T_READ)

        f_gram = B.formula(r"G = X X^{\top}").scale(0.75).move_to([1.4, 0.6, 0])
        self.add_fixed_in_frame_mobjects(f_gram)
        self.play(Write(f_gram), run_time=S.T_NORMAL)
        self.set_caption("Collecting every inner product gives the Gram matrix G, equal to X times X-transpose.")
        self.wait(T_READ)

        self.set_caption("Since G equals X times X-transpose, factoring G recovers the coordinates.")
        self.wait(T_READ)

        f_factor = B.formula(r"G = V\,\Lambda\,V^{\top}").scale(0.75).move_to([1.4, -0.5, 0])
        self.add_fixed_in_frame_mobjects(f_factor)
        self.play(Write(f_factor), run_time=S.T_NORMAL)
        self.set_caption("Eigendecompose G to find the eigenvectors that best explain the geometry.")
        self.wait(T_READ)

        # ------------------------------------------------------------------ #
        # Phase 3: the transform on the FULL matrix as a lineage.             #
        # The 4x4 sample taught the idea on real numbers (zero diagonal,      #
        # squaring); now apply D -> D^2 -> B to the whole geodesic matrix as  #
        # a flowing heatmap lineage so the structure shows at scale.          #
        # ------------------------------------------------------------------ #
        self.play(
            FadeOut(f_ip), FadeOut(f_gram), FadeOut(f_factor), FadeOut(dmat),
            run_time=S.T_FAST,
        )

        Dfull = np.asarray(self.data["D"], dtype=float)
        Bfull = double_center_squared(Dfull)
        t_order = np.argsort(np.asarray(self.data["t"], dtype=float))
        Dr = Dfull[np.ix_(t_order, t_order)]
        Br = Bfull[np.ix_(t_order, t_order)]
        hm_D = B.heatmap(Dr, N, max_cells=32, diverging=False)
        hm_D2 = B.heatmap(Dr ** 2, N, max_cells=32, diverging=False)
        hm_B = B.heatmap(Br, N, max_cells=32, diverging=True)

        hm_D.scale_to_fit_height(2.3).move_to(np.array([0.0, -0.1, 0.0]))
        self.add_fixed_in_frame_mobjects(hm_D)
        hm_D.set_opacity(0)
        self.set_caption("Here's the same process applied to the full distance matrix.")
        self.play(hm_D.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.wait(0.8)

        self.lineage = B.MatrixLineage(self)
        self.lineage.start(hm_D, "D")
        self.wait(1.2)
        self.lineage.push(hm_D2, "D^2", r"(\,\cdot\,)^2",
                          caption="Square every entry of the distance matrix.")
        self.wait(2.0)
        self.lineage.push(hm_B, "B", r"-\tfrac{1}{2}\,J(\,\cdot\,)J",
                          caption="Double-center the result to reach the Gram matrix B.")
        self.wait(2.5)
        self.set_caption("B holds inner products, ready to factor into coordinates.")
        self.wait(2.5)

        # Keep the full Gram matrix and the t-ordering for the full-matrix power
        # iteration in the eigendecomposition step.
        self.Bfull, self.t_order = Bfull, t_order

    # ------------------------------------------------------------------ #
    # Section 5: eigendecomposition                                       #
    # ------------------------------------------------------------------ #

    def _pi4(self, Bnp, grid, vscale, v, n_iters, name):
        """Detailed 4x4 power iteration shown next to the matrix `grid`.

        The iterations focus on updating v: multiply by the matrix to get
        (name)v, show its length, then divide by that length to renormalize.
        Only after v has converged is the eigenvalue read off as the product
        v^T (name) v. Returns the converged unit vector, a VGroup of the
        persistent mobjects, and the value mobject.
        """
        tname = name.replace("_", "")   # caption-friendly (B_2 -> B2)
        v = v / (np.linalg.norm(v) or 1.0)
        vcol = B.colvec(np.round(v, 2)).scale(vscale).next_to(grid, RIGHT, buff=0.55)
        vlbl = B.formula("v").scale(0.66).next_to(vcol, UP, buff=0.14)
        for m in (vcol, vlbl):
            self.add_fixed_in_frame_mobjects(m)
            m.set_opacity(0)
        self.set_caption("Start from a random unit vector v.")
        self.play(vcol.animate.set_opacity(1.0), vlbl.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.wait(1.0)

        # Iterations: keep updating v until it converges to the eigenvector.
        for _ in range(n_iters):
            Bv = Bnp @ v
            nrm = float(np.linalg.norm(Bv))
            bvcol = B.colvec(np.round(Bv, 1)).scale(vscale).next_to(vcol, RIGHT, buff=1.15)
            bvlbl = B.formula(rf"{name}v").scale(0.66).next_to(bvcol, UP, buff=0.14)
            arr = Arrow(vcol.get_right(), bvcol.get_left(), buff=0.12,
                        color=S.MUTED, stroke_width=3)
            arrlbl = B.formula(rf"\times {name}").scale(0.58).next_to(arr, UP, buff=0.06)
            nrmlbl = B.formula(rf"\lVert {name}v\rVert = {nrm:.1f}").scale(0.62).next_to(bvcol, DOWN, buff=0.25)
            for m in (arr, arrlbl, bvcol, bvlbl, nrmlbl):
                self.add_fixed_in_frame_mobjects(m)
                m.set_opacity(0)
            self.set_caption(f"Multiply v by {tname} to get {tname}v.")
            self.play(arr.animate.set_opacity(1.0), arrlbl.animate.set_opacity(1.0),
                      bvcol.animate.set_opacity(1.0), bvlbl.animate.set_opacity(1.0),
                      run_time=S.T_NORMAL)
            self.play(nrmlbl.animate.set_opacity(1.0), run_time=S.T_FAST)
            self.wait(1.0)
            # renormalize: divide by the length, back to a unit vector.
            v = Bv / nrm
            new_vcol = B.colvec(np.round(v, 2)).scale(vscale).next_to(grid, RIGHT, buff=0.55)
            self.add_fixed_in_frame_mobjects(new_vcol)
            new_vcol.set_opacity(0)
            self.set_caption("Divide by the length to renormalize v back to unit length.")
            self.play(
                bvcol.animate.set_opacity(0), bvlbl.animate.set_opacity(0),
                arr.animate.set_opacity(0), arrlbl.animate.set_opacity(0),
                nrmlbl.animate.set_opacity(0), vcol.animate.set_opacity(0),
                new_vcol.animate.set_opacity(1.0),
                run_time=S.T_NORMAL,
            )
            self.remove(bvcol, bvlbl, arr, arrlbl, nrmlbl, vcol)
            vcol = new_vcol
            self.wait(0.8)

        # v has converged; read the eigenvalue off the product v^T (name) v.
        vrow = B.rowvec(np.round(v, 2)).scale(vscale).next_to(grid, LEFT, buff=0.4)
        vtlbl = B.formula(r"v^{\top}").scale(0.66).next_to(vrow, UP, buff=0.14)
        rq = float(v @ Bnp @ v)
        val = B.formula(rf"= {rq:.1f}").scale(0.8).next_to(vcol, RIGHT, buff=0.4)
        for m in (vrow, vtlbl, val):
            self.add_fixed_in_frame_mobjects(m)
            m.set_opacity(0)
        self.set_caption(f"With v converged, the product v-transpose {tname} v gives the eigenvalue.")
        self.play(*[m.animate.set_opacity(1.0) for m in (vrow, vtlbl, val)], run_time=S.T_NORMAL)
        self.wait(1.6)
        return v, VGroup(vcol, vlbl, vrow, vtlbl, val), val

    def section_eigendecomp(self):
        self.next_section("step-5-eigendecomp", type=_SEC)

        # Update pseudocode panel: step 4 highlighted.
        self.set_pseudo(4)

        # Exact 4x4 eigenpairs (for the displayed eigenvalues and the deflation).
        Bs = np.asarray(self.data["B_sample"], dtype=float)
        ev, evec = np.linalg.eigh(Bs)
        order = np.argsort(ev)[::-1]
        ev, evec = ev[order], evec[:, order]
        lam1, lam2 = float(ev[0]), float(ev[1])
        v1 = evec[:, 0]

        # Clear the full-matrix lineage; drop to the 4x4 sample for readable numbers.
        self.play(FadeOut(self.lineage.group()), run_time=S.T_FAST)
        Bm = B.matrix_grid(np.round(Bs, 2).tolist(), highlight_negative=True)
        # Center-left so the v^T row in the v^T B v product still fits on screen.
        Bm.scale(0.42).move_to([-1.5, 0.3, 0])
        lbl_Bm = B.formula("B").scale(0.72).next_to(Bm, UP, buff=0.18)
        for m in (Bm, lbl_Bm):
            self.add_fixed_in_frame_mobjects(m)
            m.set_opacity(0)
        self.set_caption("Eigendecompose B. On a 4x4 sample, power iteration finds the top eigenvector.")
        self.play(Bm.animate.set_opacity(1.0), lbl_Bm.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.wait(1.5)

        update_rule = B.formula(r"v \leftarrow Bv\,/\,\lVert Bv\rVert").scale(0.82)
        update_rule.move_to([0.0, 2.6, 0]).set_opacity(0)
        self.add_fixed_in_frame_mobjects(update_rule)
        self.play(update_rule.animate.set_opacity(1.0), run_time=S.T_FAST)

        vscale = Bm.height / B.colvec([0.5, 0.5, 0.5, 0.5]).height
        rng = np.random.default_rng(3)

        # -------- power iteration on B: iterate v, then read off lambda 1 -------- #
        _, grpB, valB = self._pi4(Bs, Bm, vscale, rng.standard_normal(4), 3, "B")
        lam1tag = B.formula(r"= \lambda_1").scale(0.74).set_color(S.ACCENT)
        lam1tag.next_to(valB, RIGHT, buff=0.2).set_opacity(0)
        self.add_fixed_in_frame_mobjects(lam1tag)
        self.set_caption("That converged value is the largest eigenvalue, lambda 1.")
        self.play(valB.animate.set_color(S.ACCENT), lam1tag.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.wait(2.2)

        # -------- deflation as a matrix operation: B - lam1 v1 v1^T -> B2 -------- #
        # Shown like the matrix lineage: B flows through the deflation operator to
        # B2, whose dominant eigenvalue is now lambda 2.
        self.set_caption("To find the next eigenvalue, deflate B by its dominant eigenvector.")
        self.play(FadeOut(grpB), lam1tag.animate.set_opacity(0),
                  VGroup(Bm, lbl_Bm).animate.shift([-2.5, 0, 0]),
                  run_time=S.T_NORMAL)
        self.remove(lam1tag)

        B2np = Bs - lam1 * np.outer(v1, v1)
        B2m = B.matrix_grid(np.round(B2np, 2).tolist(), highlight_negative=True)
        B2m.scale(0.42).move_to([1.5, 0.3, 0])
        lbl_B2 = B.formula("B_2").scale(0.72).next_to(B2m, UP, buff=0.18)
        darr = Arrow(Bm.get_right(), B2m.get_left(), buff=0.2, color=S.MUTED, stroke_width=3)
        dop = B.formula(r"-\, \lambda_1\, v_1 v_1^{\top}").scale(0.62).next_to(darr, UP, buff=0.12)
        for m in (B2m, lbl_B2, darr, dop):
            self.add_fixed_in_frame_mobjects(m)
            m.set_opacity(0)
        self.set_caption("Subtract lambda 1 times v1 v1-transpose; this zeroes the lambda 1 direction.")
        self.play(darr.animate.set_opacity(1.0), dop.animate.set_opacity(1.0), run_time=S.T_FAST)
        self.play(B2m.animate.set_opacity(1.0), lbl_B2.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.wait(2.8)

        # Bring B2 to the iteration position, clear B and the operator, then iterate.
        self.set_caption("Now lambda 2 is the largest eigenvalue of B2.")
        self.play(
            FadeOut(Bm), FadeOut(lbl_Bm), FadeOut(darr), FadeOut(dop),
            VGroup(B2m, lbl_B2).animate.shift([-3.0, 0, 0]),
            run_time=S.T_NORMAL,
        )
        self.remove(Bm, lbl_Bm, darr, dop)
        self.wait(0.8)

        # -------- power iteration on B2: iterate v, then read off lambda 2 -------- #
        _, grpB2, valB2 = self._pi4(B2np, B2m, vscale, rng.standard_normal(4), 2, "B_2")
        lam2tag = B.formula(r"= \lambda_2").scale(0.74).set_color(S.ACCENT)
        lam2tag.next_to(valB2, RIGHT, buff=0.2).set_opacity(0)
        self.add_fixed_in_frame_mobjects(lam2tag)
        self.set_caption("That converged value is the second eigenvalue, lambda 2.")
        self.play(valB2.animate.set_color(S.ACCENT), lam2tag.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.wait(2.2)

        # -------- the same iteration on the FULL Gram matrix -------- #
        # The 4x4 showed the mechanics on real numbers; now run power iteration on
        # the whole matrix, where the eigenvector is an N-vector shown as a strip.
        clear = [B2m, lbl_B2, update_rule, lam2tag, *grpB2]
        self.play(*[m.animate.set_opacity(0) for m in clear], run_time=S.T_FAST)
        self.remove(*clear)

        self.set_caption("The same iteration scales to the full matrix.")
        Br = self.Bfull[np.ix_(self.t_order, self.t_order)]
        hmB = B.heatmap(Br, N, max_cells=32, diverging=True).scale_to_fit_height(2.0)
        hmB.move_to([-3.0, 0.0, 0])
        lblB = B.formula("B").scale(0.78).next_to(hmB, UP, buff=0.16)
        rule2 = B.formula(r"v \leftarrow Bv\,/\,\lVert Bv\rVert").scale(0.78).move_to([0.5, 2.5, 0])
        for m in (hmB, lblB, rule2):
            self.add_fixed_in_frame_mobjects(m)
            m.set_opacity(0)
        self.play(*[m.animate.set_opacity(1.0) for m in (hmB, lblB, rule2)], run_time=S.T_NORMAL)

        # The eigenvector is an N-vector, shown as a strip beside the matrix, just
        # like the 4x4 layout: B, then v, then Bv with its magnitude.
        def fstrip(vec, x):
            s = B.vector_strip(vec[self.t_order], height=2.0, width=0.46, diverging=True)
            return s.move_to([x, 0.0, 0])

        rng = np.random.default_rng(1)
        vv = rng.standard_normal(N).astype(float)
        vv /= (np.linalg.norm(vv) or 1.0)
        vstrip = fstrip(vv, -1.2)
        vlbl = B.formula("v").scale(0.7).next_to(vstrip, UP, buff=0.14)
        for m in (vstrip, vlbl):
            self.add_fixed_in_frame_mobjects(m)
            m.set_opacity(0)
        self.set_caption("Start from a random vector, shown as a strip.")
        self.play(vstrip.animate.set_opacity(1.0), vlbl.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.wait(1.0)

        for _ in range(4):
            Bv = self.Bfull @ vv
            nrm = float(np.linalg.norm(Bv))
            bvstrip = fstrip(Bv, 0.9)
            bvlbl = B.formula("Bv").scale(0.7).next_to(bvstrip, UP, buff=0.14)
            arr = Arrow(vstrip.get_right(), bvstrip.get_left(), buff=0.12,
                        color=S.MUTED, stroke_width=3)
            arrlbl = B.formula(r"\times B").scale(0.6).next_to(arr, UP, buff=0.06)
            nrmlbl = B.formula(rf"\lVert Bv\rVert = {nrm:.0f}").scale(0.66).next_to(bvstrip, DOWN, buff=0.22)
            for m in (arr, arrlbl, bvstrip, bvlbl, nrmlbl):
                self.add_fixed_in_frame_mobjects(m)
                m.set_opacity(0)
            self.set_caption("Multiply by B; its magnitude is the length of Bv.")
            self.play(arr.animate.set_opacity(1.0), arrlbl.animate.set_opacity(1.0),
                      bvstrip.animate.set_opacity(1.0), bvlbl.animate.set_opacity(1.0),
                      run_time=S.T_NORMAL)
            self.play(nrmlbl.animate.set_opacity(1.0), run_time=S.T_FAST)
            self.wait(0.8)
            vv = Bv / nrm
            new_vstrip = fstrip(vv, -1.2)
            self.add_fixed_in_frame_mobjects(new_vstrip)
            new_vstrip.set_opacity(0)
            self.set_caption("Divide by the magnitude to renormalize v; the strip smooths out.")
            self.play(
                bvstrip.animate.set_opacity(0), bvlbl.animate.set_opacity(0),
                arr.animate.set_opacity(0), arrlbl.animate.set_opacity(0),
                nrmlbl.animate.set_opacity(0), vstrip.animate.set_opacity(0),
                new_vstrip.animate.set_opacity(1.0),
                run_time=S.T_NORMAL,
            )
            self.remove(bvstrip, bvlbl, arr, arrlbl, nrmlbl, vstrip)
            vstrip = new_vstrip
            self.wait(0.5)

        # v has converged; the full product v^T B v gives lambda 1.
        ev = np.sort(np.linalg.eigvalsh(self.Bfull))[::-1]
        lam1f, lam2f = float(ev[0]), float(ev[1])
        vtbv = B.formula(rf"v^{{\top}} B v = {lam1f:.0f}").scale(0.8).set_color(S.ACCENT)
        vtbv.move_to([0.6, -1.5, 0])
        lam1tag = B.formula(r"= \lambda_1").scale(0.74).set_color(S.ACCENT).next_to(vtbv, RIGHT, buff=0.2)
        for m in (vtbv, lam1tag):
            self.add_fixed_in_frame_mobjects(m)
            m.set_opacity(0)
        self.set_caption("With v converged, v-transpose B v gives the largest eigenvalue, lambda 1.")
        self.play(vtbv.animate.set_opacity(1.0), lam1tag.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.wait(2.2)

        # Pull up lambda 2 (the deflation found it) and keep both for the embedding.
        vals = B.formula(rf"\lambda_1 = {lam1f:.0f}\quad \lambda_2 = {lam2f:.0f}").scale(0.82)
        vals.move_to([0.6, -1.5, 0]).set_opacity(0)
        self.add_fixed_in_frame_mobjects(vals)
        self.set_caption("Deflate and repeat for lambda 2; these two eigenvectors form the 2D embedding.")
        self.play(vtbv.animate.set_opacity(0), lam1tag.animate.set_opacity(0),
                  vals.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.remove(vtbv, lam1tag)
        self.wait(S.T_HOLD + 1.5)

        # Bundle the overlays so the embedding step can clear them in one fade.
        self.eig_overlays = VGroup(hmB, lblB, rule2, vstrip, vlbl, vals)

    # ------------------------------------------------------------------ #
    # Section 6: 2D embedding reveal                                      #
    # ------------------------------------------------------------------ #

    def section_embedding(self):
        self.next_section("step-6-embedding", type=_SEC)

        # Update pseudocode panel: step 5 highlighted.
        self.set_pseudo(5)

        # Clear the eigendecomposition overlays (full B, the iteration, the
        # eigenvalues) before the embedding so nothing overlaps the final arc.
        self.play(FadeOut(self.eig_overlays), run_time=S.T_FAST)

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
        self.set_caption(DATASET_OUTRO.get(DATASET, "The surface unrolls into 2D, with the geodesic-distance coloring preserved."))
        self.wait(4.0)
