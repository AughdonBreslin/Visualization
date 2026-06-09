"""Laplacian Eigenmaps walkthrough: six steps animated with manim.

Reuses the Isomap explainer's shared visual system (style, builders, data) so
the Laplacian clips match the Isomap and PCA clips and slot into the same web
player. The step timeline mirrors the JS presentSubSteps (0,2,3,4,5,6) of
laplacian.js.

Env: MFI_N (point count, default 1000), MFI_DATASET (dataset id, default
swiss_roll).

Manim 0.18.1 API notes (same as isomap walkthrough):
- Section.NORMAL does not exist; use DefaultSectionType.NORMAL.
- Dot and Line (2D VMobjects) are used for the point cloud and graph edges
  because Dot3D and Line3D render extremely slowly at N=1000.
- add_fixed_in_frame_mobjects() must be called before self.play() for 2D
  overlays to appear.
- Heatmap panels are fixed-in-frame so they do not orbit with the 3D scene.
"""
import os
import numpy as np
from manim import (
    ThreeDScene, ThreeDAxes, DEGREES, FadeIn, FadeOut, Create, Write,
    AnimationGroup, LaggedStart, DOWN, UP, RIGHT, LEFT,
    VGroup, Dot, Dot3D, Line, Line3D, Text, MathTex, interpolate_color,
    Transform,
)
from manim.scene.section import DefaultSectionType
from manimexp.isomap import style as S
from manimexp.isomap import builders as B
from manimexp.isomap.data import (
    load_dataset_points, knn_graph, heat_affinity, graph_laplacian, bottom2_eig,
)

N = int(os.environ.get("MFI_N", "1000"))
K = 8
SIGMA = 3.0
SEED = 0
DATASET = os.environ.get("MFI_DATASET", "swiss_roll")

_SEC = DefaultSectionType.NORMAL

DATASET_INTRO = {
    "swiss_roll": "A 2D sheet rolled up in 3D. Laplacian Eigenmaps will recover the flat sheet by preserving local connections.",
    "s_curve": "A 2D sheet bent into an S in 3D. Laplacian Eigenmaps will flatten it by preserving local proximity.",
    "twin_peaks": "A bumpy height surface in 3D. Laplacian Eigenmaps recovers its flat layout by following the local graph.",
    "saddle": "A curved saddle surface in 3D. Laplacian Eigenmaps uses local edge weights to flatten it.",
    "cylinder": "A sheet wrapped into a cylinder. Laplacian Eigenmaps unrolls it by following the local neighborhood graph.",
    "full_sphere": "A full sphere in 3D. A closed surface cannot flatten without distortion; watch how Laplacian Eigenmaps handles this.",
    "hilbert": "A Hilbert curve filling a cube. A folded curve is not a surface; Laplacian Eigenmaps will struggle to unroll it.",
    "clusters_3d": "Separate clusters in 3D. Laplacian Eigenmaps treats each cluster as a separate local graph component.",
}

DATASET_OUTRO = {
    "swiss_roll": "Laplacian Eigenmaps keeps strongly connected nearby points close in the embedding, so the sheet flattens while preserving locality.",
    "s_curve": "Laplacian Eigenmaps keeps nearby points close, so the S-curve unrolls into a flat strip.",
    "twin_peaks": "The local neighborhood graph keeps each height region together, flattening the surface.",
    "saddle": "The saddle flattens because nearby points on the surface remain close in the embedding.",
    "cylinder": "The cylinder unrolls because the local graph does not bridge its seam.",
    "full_sphere": "A sphere cannot be flattened without distortion, so some local neighborhoods are compressed. Laplacian Eigenmaps needs a developable surface for a clean embedding.",
    "hilbert": "A folded curve has no 2D layout; nearby folds get linked by the graph, so the embedding collapses.",
    "clusters_3d": "With disconnected components the Laplacian is block-diagonal; each cluster maps independently.",
}

_OUTRO_DEFAULT = "Laplacian Eigenmaps keeps strongly connected nearby points close in the embedding, so the sheet flattens while preserving locality."

LAP_PSEUDO = [
    r"0:\ \text{input: points } X,\ \text{neighbors } k",
    r"1:\ \mathcal{N}_i = k\text{-nearest neighbors of } x_i",
    r"2:\ W_{ij} = \exp(-\|x_i-x_j\|^2 / 2\sigma^2)",
    r"3:\ D_{ii} = \sum_j W_{ij},\quad L = D - W",
    r"4:\ L\,v_k = \lambda_k\,v_k\quad(\text{skip }\lambda_0=0)",
    r"5:\ Y = [\,v_1\;\; v_2\,]",
]


def lap_panel(active_index):
    items = []
    for idx, tex in enumerate(LAP_PSEUDO):
        color = S.ACCENT if idx == active_index else S.MUTED
        t = MathTex(tex, font_size=24, color=color)
        t.set_opacity(1.0 if idx == active_index else 0.35)
        items.append(t)
    group = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    max_w = 6.2
    if group.width > max_w:
        group.scale(max_w / group.width)
    return group


class LaplacianWalkthrough(ThreeDScene):

    def construct(self):
        self.camera.background_color = S.BG

        # --- Load dataset and run all Laplacian math up front ---
        d = load_dataset_points(DATASET, N, seed=SEED)
        pts, t = d["points"], d["t"]
        adj, edges = knn_graph(pts, k=K)
        W = heat_affinity(pts, edges, sigma=SIGMA)
        L, D_deg = graph_laplacian(W)
        vecs, vals = bottom2_eig(L)
        Y = vecs  # shape (N, 2)

        self.d = {
            "pts": pts, "t": t, "adj": adj, "edges": edges,
            "W": W, "L": L, "D_deg": D_deg,
            "vecs": vecs, "vals": vals, "Y": Y,
        }
        self.cap = None
        self.pseudo = None

        self.section_raw()
        self.section_knn()
        self.section_affinity()
        self.section_laplacian()
        self.section_eig()
        self.section_embedding()

    # ------------------------------------------------------------------ #
    # Overlay helpers (mirror the Isomap and PCA walkthroughs)           #
    # ------------------------------------------------------------------ #

    def set_caption(self, text):
        new = B.caption(text).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(new)
        new.set_opacity(0)
        if self.cap is None:
            self.play(new.animate.set_opacity(1.0), run_time=S.T_FAST)
        else:
            old = self.cap
            self.play(
                FadeOut(old, shift=0.0),
                new.animate.set_opacity(1.0),
                run_time=S.T_FAST,
            )
            self.remove(old)
        self.cap = new

    def set_pseudo(self, active_index):
        panel = lap_panel(active_index).to_corner(LEFT + UP, buff=0.15)
        self.add_fixed_in_frame_mobjects(panel)
        panel.set_opacity(0)
        if self.pseudo is None:
            self.play(panel.animate.set_opacity(1.0), run_time=S.T_FAST)
        else:
            old = self.pseudo
            self.play(
                FadeOut(old, shift=0.0),
                panel.animate.set_opacity(1.0),
                run_time=S.T_FAST,
            )
            self.remove(old)
        self.pseudo = panel

    # ------------------------------------------------------------------ #
    # Section 1: raw point cloud                                          #
    # ------------------------------------------------------------------ #

    def section_raw(self):
        self.next_section("step-1-raw", type=_SEC)
        self.set_camera_orientation(phi=65 * DEGREES, theta=30 * DEGREES, zoom=0.9)

        self.axes = ThreeDAxes(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1], z_range=[-4, 4, 1],
            x_length=8, y_length=8, z_length=8,
            axis_config={"stroke_color": S.MUTED, "stroke_width": 0.8, "stroke_opacity": 0.45},
        )
        self.play(FadeIn(self.axes, run_time=S.T_FAST))

        self.cloud = B.point_cloud(self.d["pts"], self.d["t"])
        self.play(FadeIn(self.cloud, run_time=S.T_INTRO))

        self.set_pseudo(0)
        self.set_caption(DATASET_INTRO.get(
            DATASET,
            "A curved surface in 3D. Laplacian Eigenmaps recovers its flat layout by preserving local connections.",
        ))
        self.begin_ambient_camera_rotation(rate=2 * np.pi / 14.0, about="theta")
        self.wait(4.0)

    # ------------------------------------------------------------------ #
    # Section 2: kNN graph                                                #
    # ------------------------------------------------------------------ #

    def section_knn(self):
        self.next_section("step-2-knn", type=_SEC)
        self.set_pseudo(1)

        # --- Beat 1: schematic to show the idea before the full graph ---
        # Fade cloud and axes so the schematic reads on a clean stage.
        self.play(
            FadeOut(self.cloud),
            FadeOut(self.axes),
            run_time=S.T_FAST,
        )
        self.move_camera(
            phi=0, theta=-90 * DEGREES, zoom=1.0,
            frame_center=[0, 0, 0],
            run_time=S.T_NORMAL,
        )
        self.set_caption("Link each point to its k nearest neighbors to capture the local structure of the surface.")

        center_pt = np.array([0.0, 0.0, 0.0])
        center_dot = Dot(point=center_pt, radius=0.10, color=S.ACCENT)
        self.play(FadeIn(center_dot, run_time=S.T_FAST))

        offsets = np.array([
            [ 1.60,  0.30, 0.0],
            [ 0.90,  1.55, 0.0],
            [-1.00,  1.30, 0.0],
            [-1.65, -0.30, 0.0],
            [-0.80, -1.40, 0.0],
            [ 1.00, -1.30, 0.0],
        ], dtype=float)
        neighbor_pts = [center_pt + off for off in offsets]
        distances = [float(np.linalg.norm(off)) for off in offsets]

        neighbor_dots = VGroup(*[
            Dot(point=p, radius=0.08, color=S.WARM) for p in neighbor_pts
        ])
        self.play(
            LaggedStart(*[FadeIn(d) for d in neighbor_dots],
                        lag_ratio=0.18, run_time=S.T_NORMAL)
        )
        self.wait(1.8)

        self.set_caption("Each link is weighted by the distance between the two points.")
        schematic_lines = VGroup()
        schematic_labels = []
        for nbr_pt, dist in zip(neighbor_pts, distances):
            seg = Line(
                start=center_pt, end=nbr_pt,
                stroke_width=2.8, color=S.ACCENT,
            ).set_opacity(0.85)
            edge_dir = (nbr_pt - center_pt) / dist
            perp = np.array([-edge_dir[1], edge_dir[0], 0.0])
            midpoint = (center_pt + nbr_pt) / 2.0
            label_pos = midpoint + perp * 0.24
            lbl = Text(f"{dist:.2f}", font_size=22, color=S.INK).move_to(label_pos)
            schematic_lines.add(seg)
            schematic_labels.append(lbl)
            self.play(Create(seg), FadeIn(lbl), run_time=0.45)
        self.wait(S.T_HOLD)

        # --- Beat 2: fade schematic, restore cloud, draw full kNN graph ---
        schematic_all = VGroup(center_dot, schematic_lines, neighbor_dots,
                               *schematic_labels)
        self.play(FadeOut(schematic_all), run_time=S.T_FAST)
        self.move_camera(
            phi=65 * DEGREES, theta=30 * DEGREES, zoom=0.9,
            frame_center=[0, 0, 0],
            run_time=S.T_NORMAL,
        )
        self.play(FadeIn(self.cloud), FadeIn(self.axes), run_time=S.T_FAST)

        self.set_caption("Repeat for every point and the kNN graph spans the whole surface.")
        self.edges_mob = B.graph_edges(self.d["pts"], self.d["edges"])
        self.play(
            LaggedStart(
                *[FadeIn(edge) for edge in self.edges_mob],
                lag_ratio=0.004,
                run_time=S.T_SLOW,
            )
        )

        # --- Beat 3: highlight one neighborhood with a kNN sphere ---
        pts = self.d["pts"]
        adj = self.d["adj"]
        centroid = pts.mean(axis=0)
        center_idx = int(np.argmin(np.linalg.norm(pts - centroid, axis=1)))
        nbr_indices = [j for j in range(len(pts)) if j != center_idx and adj[center_idx, j] > 0]

        center_sphere = B.knn_sphere(pts[center_idx], radius=0.10, color=S.GOOD)
        neighbor_spheres = VGroup(*[
            B.knn_sphere(pts[j], radius=0.07, color=S.WARM) for j in nbr_indices
        ])
        nbr_lines = VGroup(*[
            Line3D(
                start=pts[center_idx], end=pts[j],
                thickness=0.018, color=S.GOOD,
            )
            for j in nbr_indices
        ])

        self.set_caption("The k neighbors of one point form its local neighborhood on the surface.")
        self.play(FadeIn(center_sphere), run_time=S.T_FAST)
        self.play(
            LaggedStart(*[FadeIn(s) for s in neighbor_spheres],
                        lag_ratio=0.12, run_time=S.T_NORMAL)
        )
        self.play(
            LaggedStart(*[Create(l) for l in nbr_lines],
                        lag_ratio=0.12, run_time=S.T_NORMAL)
        )
        self.wait(S.T_HOLD + 1.0)

        # Clean up the neighborhood highlight; keep edges and cloud for the next section.
        self.play(
            FadeOut(center_sphere),
            FadeOut(neighbor_spheres),
            FadeOut(nbr_lines),
            run_time=S.T_FAST,
        )

        # Keep the ambient orbit running continuously.
        orbit_time = 16.0
        self.move_camera(phi=80 * DEGREES, theta=30 * DEGREES, run_time=S.T_NORMAL)
        self.wait(S.T_HOLD)

    # ------------------------------------------------------------------ #
    # Section 3: heat-kernel affinity W                                   #
    # ------------------------------------------------------------------ #

    def section_affinity(self):
        self.next_section("step-3-affinity", type=_SEC)
        self.set_pseudo(2)

        pts = self.d["pts"]
        edges = self.d["edges"]
        W = self.d["W"]

        # Compute per-edge weight for coloring.
        edge_weights = np.array([float(W[i, j]) for (i, j) in edges])
        w_min = float(edge_weights.min())
        w_max = float(edge_weights.max())
        w_span = max(1e-9, w_max - w_min)

        self.set_caption("The heat kernel turns each edge distance into a weight: nearby pairs get values close to 1, distant pairs get values close to 0.")

        # Recolor each edge line by its weight: strong (close to 1) -> GOOD, weak -> MUTED.
        recolor_anims = []
        for line, w in zip(self.edges_mob.submobjects, edge_weights):
            u = (w - w_min) / w_span
            col = interpolate_color(S.MUTED, S.GOOD, u)
            recolor_anims.append(line.animate.set_color(col).set_opacity(min(0.85, 0.12 + 0.73 * u)))
        self.play(AnimationGroup(*recolor_anims, lag_ratio=0.0), run_time=S.T_SLOW)
        self.wait(1.0)

        # Formula in top-right corner.
        f_W = B.formula(r"W_{ij} = \exp\!\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)").scale(0.75)
        self.add_fixed_in_frame_mobjects(f_W)
        f_W.to_corner(RIGHT + UP, buff=0.35).set_opacity(0)
        self.play(f_W.animate.set_opacity(1.0), run_time=S.T_FAST)

        # Heatmap of W as a fixed-in-frame panel, bottom-right to avoid the
        # pseudocode panel (top-left) and caption (bottom-center).
        hm_W = B.heatmap(W, N, max_cells=32, cell=0.13, diverging=False)
        hm_W.scale_to_fit_height(2.6)
        lbl_W = Text("W", font_size=26, color=S.INK)
        self.add_fixed_in_frame_mobjects(hm_W, lbl_W)
        hm_W.to_corner(RIGHT + DOWN, buff=0.5)
        lbl_W.next_to(hm_W, UP, buff=0.12)
        hm_W.set_opacity(0)
        lbl_W.set_opacity(0)
        self.play(
            hm_W.animate.set_opacity(1.0),
            lbl_W.animate.set_opacity(1.0),
            run_time=S.T_NORMAL,
        )
        self.set_caption("Brighter cells in W correspond to pairs that are close neighbors. Non-neighbor entries are exactly zero, so W is sparse.")
        self.wait(S.T_HOLD + 2.0)

        self.affinity_f = f_W
        self.hm_W = hm_W
        self.lbl_W = lbl_W

    # ------------------------------------------------------------------ #
    # Section 4: graph Laplacian L = D - W                                #
    # ------------------------------------------------------------------ #

    def section_laplacian(self):
        self.next_section("step-4-laplacian", type=_SEC)
        self.set_pseudo(3)

        W = self.d["W"]
        L = self.d["L"]
        D_deg = self.d["D_deg"]

        self.set_caption("The degree matrix D collects the row sums of W. The graph Laplacian L subtracts W from D.")

        # Fade out the affinity formula and keep the W heatmap visible for continuity.
        self.play(FadeOut(self.affinity_f), run_time=S.T_FAST)

        # Heatmap of D (diagonal only, so brighter along the diagonal).
        hm_D = B.heatmap(D_deg, N, max_cells=32, cell=0.13, diverging=False)
        hm_D.scale_to_fit_height(2.6)
        lbl_D = Text("D", font_size=26, color=S.INK)
        self.add_fixed_in_frame_mobjects(hm_D, lbl_D)
        # Position D heatmap to the left of W heatmap.
        hm_D.next_to(self.hm_W, LEFT, buff=0.35)
        lbl_D.next_to(hm_D, UP, buff=0.12)
        hm_D.set_opacity(0)
        lbl_D.set_opacity(0)
        self.play(
            hm_D.animate.set_opacity(1.0),
            lbl_D.animate.set_opacity(1.0),
            run_time=S.T_NORMAL,
        )
        self.wait(1.2)

        # Heatmap of L (diverging: negative off-diagonal entries are warm-colored).
        hm_L = B.heatmap(L, N, max_cells=32, cell=0.13, diverging=True)
        hm_L.scale_to_fit_height(2.6)
        lbl_L = Text("L", font_size=26, color=S.INK)
        self.add_fixed_in_frame_mobjects(hm_L, lbl_L)
        hm_L.next_to(hm_D, LEFT, buff=0.35)
        lbl_L.next_to(hm_L, UP, buff=0.12)
        hm_L.set_opacity(0)
        lbl_L.set_opacity(0)
        self.play(
            hm_L.animate.set_opacity(1.0),
            lbl_L.animate.set_opacity(1.0),
            run_time=S.T_NORMAL,
        )

        # Formula below the three heatmaps.
        f_L = B.formula(r"L = D - W,\quad D_{ii} = \sum_j W_{ij}").scale(0.72)
        self.add_fixed_in_frame_mobjects(f_L)
        f_L.next_to(self.hm_W, DOWN, buff=0.3).set_opacity(0)
        self.play(f_L.animate.set_opacity(1.0), run_time=S.T_FAST)

        self.set_caption("L has non-negative diagonal entries (degrees) and non-positive off-diagonal entries (-W_ij). It captures how much each point differs from its neighbors.")
        self.wait(S.T_HOLD + 2.0)

        self.hm_D = hm_D
        self.lbl_D = lbl_D
        self.hm_L = hm_L
        self.lbl_L = lbl_L
        self.laplacian_f = f_L

    # ------------------------------------------------------------------ #
    # Section 5: smallest non-trivial eigenvectors of L                   #
    # ------------------------------------------------------------------ #

    def section_eig(self):
        self.next_section("step-5-eig", type=_SEC)
        self.set_pseudo(4)

        vals = self.d["vals"]

        # Clear the three heatmaps and the Laplacian formula.
        self.play(
            FadeOut(self.hm_W), FadeOut(self.lbl_W),
            FadeOut(self.hm_D), FadeOut(self.lbl_D),
            FadeOut(self.hm_L), FadeOut(self.lbl_L),
            FadeOut(self.laplacian_f),
            run_time=S.T_FAST,
        )

        self.set_caption("Find the two smallest non-trivial eigenvalues of L. The trivial zero eigenvalue corresponds to the constant eigenvector, which carries no coordinate information.")
        self.wait(2.8)

        f_eig = B.formula(r"L\,v_k = \lambda_k\,v_k").scale(0.85)
        self.add_fixed_in_frame_mobjects(f_eig)
        f_eig.to_corner(RIGHT + UP, buff=0.4).set_opacity(0)
        self.play(f_eig.animate.set_opacity(1.0), run_time=S.T_FAST)

        self.set_caption("Each eigenvector is a smooth function on the graph. The smallest non-trivial ones vary as slowly as possible across the edges.")
        self.wait(2.0)

        lam_str = rf"\lambda_1 = {vals[0]:.4f},\quad \lambda_2 = {vals[1]:.4f}"
        f_vals = B.formula(lam_str).scale(0.70)
        if f_vals.width > 5.6:
            f_vals.scale_to_fit_width(5.6)
        self.add_fixed_in_frame_mobjects(f_vals)
        f_vals.next_to(f_eig, DOWN, buff=0.3, aligned_edge=RIGHT).set_opacity(0)
        self.play(f_vals.animate.set_opacity(1.0), run_time=S.T_FAST)

        self.set_caption("These two eigenvectors become the x and y coordinates of the 2D embedding.")
        self.wait(S.T_HOLD + 2.0)

        # Recolor the cloud by the first eigenvector's values to preview the
        # embedding coordinate on the 3D shape.
        vecs = self.d["vecs"]
        v1 = vecs[:, 0]
        recolor_anims = B.recolor_cloud_by_values(
            self.cloud, v1, c_lo=S.MUTED, c_hi=S.ACCENT, grow=1.5,
        )
        self.set_caption("Coloring the cloud by the first eigenvector shows which direction the graph unfolds smoothly along.")
        self.play(AnimationGroup(*recolor_anims, lag_ratio=0.0), run_time=S.T_SLOW)
        self.wait(S.T_HOLD + 1.0)

        self.eig_f = f_eig
        self.eig_vals_f = f_vals

    # ------------------------------------------------------------------ #
    # Section 6: 2D embedding                                             #
    # ------------------------------------------------------------------ #

    def section_embedding(self):
        self.next_section("step-6-embedding", type=_SEC)
        self.set_pseudo(5)

        self.stop_ambient_camera_rotation()

        Y = self.d["Y"]
        s = 3.4 / (float(np.abs(Y).max()) or 1.0)
        target = np.column_stack([Y[:, 0] * s, Y[:, 1] * s, np.zeros(Y.shape[0])])

        self.play(
            FadeOut(self.eig_f),
            FadeOut(self.eig_vals_f),
            run_time=S.T_FAST,
        )

        f_embed = B.formula(r"Y = [\,v_1\;\; v_2\,]").scale(0.85)
        self.add_fixed_in_frame_mobjects(f_embed)
        f_embed.to_corner(RIGHT + UP, buff=0.4).set_opacity(0)
        self.play(f_embed.animate.set_opacity(1.0), run_time=S.T_FAST)

        # Flatten camera to face-on (phi=0) by the shortest angular path.
        cur_theta = float(self.camera.get_theta())
        tgt_theta = -90 * DEGREES
        tgt_theta += round((cur_theta - tgt_theta) / (2 * np.pi)) * 2 * np.pi

        self.set_caption("Rotate the view to face the embedding plane, and move each point to its 2D position.")

        # Morph cloud to the 2D embedding positions while flattening the camera.
        cloud_dots = self.cloud.submobjects
        n_pts = len(cloud_dots)
        self.move_camera(
            phi=0, theta=tgt_theta,
            run_time=S.T_SLOW,
            added_anims=[
                cloud_dots[i].animate.move_to(target[i])
                for i in range(n_pts)
            ],
        )
        self.play(FadeOut(self.axes), run_time=S.T_FAST)
        self.set_caption(DATASET_OUTRO.get(DATASET, _OUTRO_DEFAULT))
        self.wait(5.0)
