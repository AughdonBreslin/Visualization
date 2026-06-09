"""LLE walkthrough: the steps of Locally Linear Embedding animated with manim.

Reuses the Isomap explainer's shared visual system (style, builders, data) and
dataset loading so LLE clips match the Isomap and PCA clips and slot into the
same web player.

Step ids mirror the JS lle.js presentSubSteps: 0, 2, 3, 5, 6.

Manim 0.18.1 API gotchas (inherited from isomap/walkthrough.py):
- Use DefaultSectionType.NORMAL from manim.scene.section, not Section.NORMAL.
- 2D Dot and Line VMobjects for the point cloud and graph edges (Dot3D at N=120
  is ~1.3 s/frame; Line3D is kept only for the highlighted geodesic-style path
  in the kNN zoom). add_fixed_in_frame_mobjects() must precede self.play().
- Line3D uses thickness (scene units), not stroke_width.
- Camera flattens to phi=0 only for the final 2D embedding beat; continuous
  ambient rotation runs across all other sections.

Env: MFI_N (point count, default 1000), MFI_DATASET (default swiss_roll).
"""
import os
import numpy as np
from manim import (
    ThreeDScene, ThreeDAxes, DEGREES, FadeIn, FadeOut, Create, Write,
    AnimationGroup, LaggedStart, DOWN, UP, RIGHT, LEFT,
    Dot, VGroup, Line, Arrow, MathTex, Text,
)
from manim.scene.section import DefaultSectionType
from manimexp.isomap import style as S
from manimexp.isomap import builders as B
from manimexp.isomap.data import (
    load_dataset_points,
    knn_graph,
    lle_weights,
    lle_matrix,
    bottom2_eig,
)

N = int(os.environ.get("MFI_N", "1000"))
K = 8
SEED = 0
DATASET = os.environ.get("MFI_DATASET", "swiss_roll")

_SEC = DefaultSectionType.NORMAL

# Per-dataset opening caption.
DATASET_INTRO = {
    "swiss_roll": "A 2D sheet rolled up in 3D. LLE will recover the flat sheet by preserving local linear structure.",
    "s_curve": "A 2D sheet bent into an S in 3D. LLE will recover the flat sheet by preserving local linear structure.",
    "twin_peaks": "A bumpy height surface in 3D. LLE recovers its flat layout by preserving local neighborhoods.",
    "saddle": "A curved saddle surface in 3D. LLE will attempt to recover its flat layout.",
    "cylinder": "A sheet wrapped into a cylinder. LLE will attempt to unroll it by preserving local patches.",
    "full_sphere": "A full sphere in 3D. LLE cannot flatten it without tearing; watch the embedding struggle.",
    "hilbert": "A Hilbert curve filling a cube. LLE finds local patches but the result is not a clean surface.",
    "clusters_3d": "Separate clusters in 3D. LLE works within each cluster; global structure is undefined.",
}

DATASET_OUTRO = {
    "swiss_roll": "LLE preserved each point's local linear reconstruction from its neighbors, so the sheet unrolls flat.",
    "s_curve": "LLE preserved local patches, so the bent sheet unrolls into a flat layout.",
    "twin_peaks": "LLE preserved local structure; the surface flattens with some distortion at the peaks.",
    "saddle": "LLE preserved local patches; the saddle flattens into 2D with some distortion.",
    "cylinder": "LLE preserved local patches, partially unrolling the cylinder. A full unwrap needs more neighbors.",
    "full_sphere": "A sphere cannot flatten without tearing, so the LLE embedding shows distortion. LLE needs a developable surface.",
    "hilbert": "A folded curve has no clean 2D layout; LLE collapses it because folds share neighbors.",
    "clusters_3d": "LLE placed each cluster internally but cannot define inter-cluster geometry. The global layout is arbitrary.",
}

LLE_PSEUDO = [
    r"0:\ \text{input: points } X,\ \text{neighbors } k",
    r"1:\ \mathcal{N}_i = k \text{ nearest neighbors of } x_i",
    r"2:\ \min \bigl\| x_i - \textstyle\sum_j w_j x_{n_j} \bigr\|^2,\ \sum_j w_j = 1",
    r"3:\ M = (I - W)^{\top}(I - W)",
    r"4:\ Y = [\, v_1 \;\; v_2 \,]\quad(\text{bottom non-trivial eigenvectors})",
]


def lle_panel(active_index):
    """Build the LLE pseudocode panel with one line highlighted."""
    items = []
    for idx, tex in enumerate(LLE_PSEUDO):
        color = S.ACCENT if idx == active_index else S.MUTED
        t = MathTex(tex, font_size=24, color=color)
        t.set_opacity(1.0 if idx == active_index else 0.35)
        items.append(t)
    group = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    max_w = 6.2
    if group.width > max_w:
        group.scale(max_w / group.width)
    return group


class LLEWalkthrough(ThreeDScene):

    def construct(self):
        self.camera.background_color = S.BG

        # Load dataset.
        d = load_dataset_points(DATASET, N, seed=SEED)
        pts = d["points"]
        t_param = d["t"]

        # Compute all LLE math up front so rendering is deterministic.
        adj, edges = knn_graph(pts, k=K)
        W = lle_weights(pts, k=K, reg=1e-3)
        M = lle_matrix(W)
        vecs, vals = bottom2_eig(M)
        Y = vecs  # shape (N, 2): the two eigenvectors directly (no sqrt scaling)

        # Pick a representative center point (closest to the cloud centroid).
        centroid = pts.mean(axis=0)
        center_idx = int(np.argmin(np.linalg.norm(pts - centroid, axis=1)))

        # Neighbors of the center point and their LLE weights.
        nbr_indices = np.where(adj[center_idx] > 0)[0]
        nbr_weights = np.array([W[center_idx, j] for j in nbr_indices])

        self.d = dict(
            pts=pts, t=t_param, adj=adj, edges=edges,
            W=W, M=M, vecs=vecs, vals=vals, Y=Y,
            center_idx=center_idx,
            nbr_indices=nbr_indices,
            nbr_weights=nbr_weights,
        )
        self.cap = None
        self.pseudo = None

        self.section_raw()
        self.section_knn()
        self.section_weights()
        self.section_eig()
        self.section_embed()

    # ------------------------------------------------------------------ #
    # Overlay helpers                                                     #
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

    def set_pseudo(self, active_index):
        panel = lle_panel(active_index).to_corner(LEFT + UP, buff=0.15)
        self.add_fixed_in_frame_mobjects(panel)
        panel.set_opacity(0)
        if self.pseudo is None:
            self.play(panel.animate.set_opacity(1.0), run_time=S.T_FAST)
        else:
            old = self.pseudo
            self.play(FadeOut(old, shift=0.0), panel.animate.set_opacity(1.0), run_time=S.T_FAST)
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
        intro = DATASET_INTRO.get(
            DATASET,
            "A curved 2D surface in 3D. LLE recovers its flat layout by preserving local linear structure.",
        )
        self.set_caption(intro)

        self.begin_ambient_camera_rotation(rate=2 * np.pi / 14.0, about="theta")
        self.wait(4.0)

    # ------------------------------------------------------------------ #
    # Section 2: kNN graph                                                #
    # ------------------------------------------------------------------ #

    def section_knn(self):
        self.next_section("step-2-knn", type=_SEC)
        self.set_pseudo(1)

        # Beat 1: fade out cloud and axes; move camera flat for the schematic.
        self.play(
            FadeOut(self.cloud),
            FadeOut(self.axes),
            run_time=S.T_FAST,
        )
        self.move_camera(
            phi=0, theta=-90 * DEGREES, zoom=1.0, frame_center=[0, 0, 0],
            run_time=S.T_NORMAL,
        )
        self.set_caption("LLE links each point to its k nearest neighbors to define a local patch.")

        # Beat 2: schematic star of neighbors (flat, z = 0).
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
            [ 0.30,  1.80, 0.0],
            [-1.40,  0.80, 0.0],
        ], dtype=float)
        nbr_pts = [center_pt + off for off in offsets]
        distances = [float(np.linalg.norm(off)) for off in offsets]

        nbr_dots = VGroup(*[
            Dot(point=p, radius=0.08, color=S.WARM) for p in nbr_pts
        ])
        self.play(
            LaggedStart(*[FadeIn(d) for d in nbr_dots], lag_ratio=0.15, run_time=S.T_NORMAL)
        )
        self.wait(1.8)

        # Beat 3: draw edges one by one with distance labels.
        self.set_caption("Each link is weighted by the distance between the points.")

        schematic_lines = VGroup()
        schematic_labels = []
        for nbr_pt, dist in zip(nbr_pts, distances):
            seg = Line(
                start=center_pt, end=nbr_pt,
                stroke_width=2.8, color=S.ACCENT,
            ).set_opacity(0.85)
            edge_dir = (nbr_pt - center_pt) / dist
            perp = np.array([-edge_dir[1], edge_dir[0], 0.0])
            midpoint = (center_pt + nbr_pt) / 2.0
            label_pos = midpoint + perp * 0.24
            lbl = Text(f"{dist:.2f}", font_size=20, color=S.INK).move_to(label_pos)
            schematic_lines.add(seg)
            schematic_labels.append(lbl)
            self.play(Create(seg), FadeIn(lbl), run_time=0.40)

        self.wait(S.T_HOLD)

        # Beat 4: fade schematic; restore dataset and full kNN graph.
        schematic_all = VGroup(center_dot, schematic_lines, nbr_dots, *schematic_labels)
        self.play(FadeOut(schematic_all), run_time=S.T_FAST)

        self.move_camera(
            phi=65 * DEGREES, theta=30 * DEGREES, zoom=0.9, frame_center=[0, 0, 0],
            run_time=S.T_NORMAL,
        )
        self.play(FadeIn(self.cloud), FadeIn(self.axes), run_time=S.T_FAST)

        self.set_caption("Do this for every point and the kNN graph covers the whole surface.")
        self.edges_mob = B.graph_edges(self.d["pts"], self.d["edges"])
        self.play(
            LaggedStart(
                *[FadeIn(edge) for edge in self.edges_mob],
                lag_ratio=0.004,
                run_time=S.T_SLOW,
            )
        )

        # Highlight one neighborhood with knn_sphere.
        ci = self.d["center_idx"]
        ni = self.d["nbr_indices"]
        pts = self.d["pts"]
        center_sphere = B.knn_sphere(pts[ci], radius=0.06, color=S.ACCENT)
        nbr_spheres = VGroup(*[
            B.knn_sphere(pts[j], radius=0.05, color=S.WARM) for j in ni
        ])
        self.play(FadeIn(center_sphere), run_time=S.T_FAST)
        self.play(
            LaggedStart(*[FadeIn(s) for s in nbr_spheres], lag_ratio=0.10, run_time=S.T_NORMAL)
        )
        self.set_caption("Each highlighted point lies on the same local patch as its neighbors.")
        self.wait(S.T_HOLD)

        self.knn_highlight = VGroup(center_sphere, nbr_spheres)
        # Keep the ambient rotation running into the next section.
        self.move_camera(phi=80 * DEGREES, theta=30 * DEGREES, run_time=S.T_FAST)

    # ------------------------------------------------------------------ #
    # Section 3: reconstruction weights                                   #
    # ------------------------------------------------------------------ #

    def section_weights(self):
        self.next_section("step-3-weights", type=_SEC)
        self.set_pseudo(2)

        # Fade out the kNN highlight and dim the graph edges.
        self.play(
            FadeOut(self.knn_highlight),
            self.edges_mob.animate.set_opacity(0.05),
            run_time=S.T_FAST,
        )

        pts = self.d["pts"]
        ci = self.d["center_idx"]
        ni = self.d["nbr_indices"]
        nw = self.d["nbr_weights"]

        self.set_caption("For each point, find weights so it is a weighted sum of its k neighbors. Weights must sum to 1.")

        # Draw arrows from each neighbor to the center, scaled by weight magnitude.
        # Sort neighbors by descending |weight| so the most influential ones appear first.
        order = np.argsort(np.abs(nw))[::-1]
        weight_arrows = VGroup()
        weight_labels = []

        # Show only the top K neighbors to keep the view readable.
        show_count = min(K, len(ni))
        for rank, idx in enumerate(order[:show_count]):
            j = ni[idx]
            w_val = float(nw[idx])
            # Arrow direction: neighbor -> center, magnitude encodes weight.
            start_3d = pts[j].tolist()
            end_3d = pts[ci].tolist()
            col = S.GOOD if w_val >= 0 else S.WARM

            arr = Arrow(
                start=start_3d, end=end_3d,
                buff=0.0, stroke_width=max(1.0, 2.5 * abs(w_val) * 8),
                color=col, max_tip_length_to_length_ratio=0.25,
            ).set_opacity(0.75)
            weight_arrows.add(arr)

            # Label the top 4 weights so the screen stays readable.
            if rank < 4:
                mid = (np.array(start_3d) + np.array(end_3d)) / 2.0
                # Nudge slightly off the line.
                nudge = np.array([0.0, 0.12, 0.0])
                lbl_pos = (mid + nudge).tolist()
                lbl = Text(f"{w_val:.3f}", font_size=20, color=S.INK).move_to(lbl_pos)
                weight_labels.append(lbl)

        self.play(
            LaggedStart(*[Create(a) for a in weight_arrows], lag_ratio=0.12, run_time=S.T_SLOW)
        )
        if weight_labels:
            self.play(
                LaggedStart(*[FadeIn(lbl) for lbl in weight_labels], lag_ratio=0.15, run_time=S.T_NORMAL)
            )
        self.wait(1.8)

        # Show the weight formula as a fixed-in-frame overlay.
        f_w = B.formula(
            r"\min \bigl\| x_i - \textstyle\sum_j w_j x_{n_j} \bigr\|^2,\quad \sum_j w_j = 1"
        ).scale(0.72)
        self.add_fixed_in_frame_mobjects(f_w)
        f_w.to_corner(RIGHT + UP, buff=0.3).set_opacity(0)
        self.play(f_w.animate.set_opacity(1.0), run_time=S.T_FAST)
        self.set_caption("The weights minimize reconstruction error subject to summing to 1. A small per-point linear system determines them.")
        self.wait(2.2)

        # Optionally show a heatmap of W (the full weight matrix) in the corner.
        N_pts = len(pts)
        hm_W = B.heatmap(self.d["W"], N_pts, max_cells=32, cell=0.11, diverging=True)
        hm_lbl = B.formula(r"W").scale(0.65)
        hm_grp = VGroup(hm_lbl, hm_W).arrange(DOWN, buff=0.10)
        hm_grp.scale(1.0)
        # Size cap so the heatmap does not crowd the formula.
        if hm_grp.width > 2.4:
            hm_grp.scale_to_fit_width(2.4)
        self.add_fixed_in_frame_mobjects(hm_grp)
        hm_grp.to_corner(RIGHT + DOWN, buff=0.3).set_opacity(0)
        self.play(hm_grp.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.set_caption("The weight matrix W is sparse: most entries are zero. Each row has at most k non-zero entries.")
        self.wait(2.4)

        self.weight_arrows = weight_arrows
        self.weight_labels = VGroup(*weight_labels)
        self.weight_formula = f_w
        self.weight_hm = hm_grp
        self.wait(S.T_HOLD)

    # ------------------------------------------------------------------ #
    # Section 4 (step-5-eig): eigenvectors of M = (I - W)^T (I - W)     #
    # ------------------------------------------------------------------ #

    def section_eig(self):
        self.next_section("step-5-eig", type=_SEC)
        self.set_pseudo(3)

        # Clear the weight visualization.
        self.play(
            FadeOut(self.weight_arrows),
            FadeOut(self.weight_labels),
            FadeOut(self.weight_formula),
            FadeOut(self.weight_hm),
            run_time=S.T_FAST,
        )

        # Dim the graph edges further; the cloud stays to show the eigenvector coloring.
        self.play(
            self.edges_mob.animate.set_opacity(0.04),
            run_time=S.T_FAST,
        )

        self.set_caption("Form M = (I - W) transpose times (I - W). Its smallest non-trivial eigenvectors give the 2D coordinates.")

        # Show the M formula.
        f_M = B.formula(r"M = (I - W)^{\top}(I - W)").scale(0.80)
        self.add_fixed_in_frame_mobjects(f_M)
        f_M.to_corner(RIGHT + UP, buff=0.35).set_opacity(0)
        self.play(f_M.animate.set_opacity(1.0), run_time=S.T_FAST)
        self.wait(1.8)

        # Show eigenvalue readout.
        vals = self.d["vals"]
        lam1, lam2 = float(vals[0]), float(vals[1])
        f_vals = B.formula(
            rf"\lambda_1 = {lam1:.4f}\quad \lambda_2 = {lam2:.4f}"
        ).scale(0.65)
        if f_vals.width > 5.6:
            f_vals.scale_to_fit_width(5.6)
        self.add_fixed_in_frame_mobjects(f_vals)
        f_vals.next_to(f_M, DOWN, buff=0.30, aligned_edge=RIGHT).set_opacity(0)
        self.play(f_vals.animate.set_opacity(1.0), run_time=S.T_FAST)
        self.set_caption("The trivial zero eigenvalue (the constant eigenvector) is skipped. The next two carry the shape.")
        self.wait(S.T_HOLD + 1.6)

        # Color the cloud by the first eigenvector value to show it encodes position.
        v1 = self.d["vecs"][:, 0]
        recolor_anims = B.recolor_cloud_by_values(self.cloud, v1, cmap=B.rainbow_color, grow=1.0)
        self.play(AnimationGroup(*recolor_anims, lag_ratio=0.0), run_time=S.T_SLOW)
        self.set_caption("Coloring points by the first eigenvector shows it traces the unrolled sheet's main axis.")
        self.wait(S.T_HOLD + 1.0)

        self.eig_f_M = f_M
        self.eig_f_vals = f_vals

    # ------------------------------------------------------------------ #
    # Section 5 (step-6-embedding): 2D embedding                         #
    # ------------------------------------------------------------------ #

    def section_embed(self):
        self.next_section("step-6-embedding", type=_SEC)
        self.set_pseudo(4)

        # Clear eig overlays.
        self.play(
            FadeOut(self.eig_f_M),
            FadeOut(self.eig_f_vals),
            run_time=S.T_FAST,
        )

        Y = self.d["Y"]
        s = 3.4 / (float(np.abs(Y).max()) or 1.0)
        pts3 = np.column_stack([Y[:, 0] * s, Y[:, 1] * s, np.zeros(Y.shape[0])])

        # Stop the ambient rotation and move to face-on (phi=0) by the shortest
        # path, mirroring the PCA template's theta-correction logic.
        self.stop_ambient_camera_rotation()
        cur_theta = float(self.camera.get_theta())
        tgt_theta = -90 * DEGREES
        tgt_theta += round((cur_theta - tgt_theta) / (2 * np.pi)) * 2 * np.pi

        # Embedding formula.
        f_embed = B.formula(r"Y = [\, v_1 \;\; v_2 \,]").scale(0.85)
        self.add_fixed_in_frame_mobjects(f_embed)
        f_embed.to_corner(RIGHT + UP, buff=0.35).set_opacity(0)
        self.play(f_embed.animate.set_opacity(1.0), run_time=S.T_FAST)

        self.set_caption("Turn to look straight down at the embedding plane, then move each point to its 2D position.")

        # Morph the cloud into the 2D plane while flattening the camera.
        pts_list = self.cloud.submobjects
        self.move_camera(
            phi=0, theta=tgt_theta, run_time=S.T_SLOW,
            added_anims=[
                pts_list[i].animate.move_to(pts3[i]) for i in range(len(pts_list))
            ],
        )

        # Fade out the axes (no longer meaningful in 2D).
        self.play(FadeOut(self.axes), FadeOut(self.edges_mob), run_time=S.T_FAST)

        outro = DATASET_OUTRO.get(
            DATASET,
            "LLE preserved each point's local linear reconstruction from its neighbors, so the sheet unrolls flat.",
        )
        self.set_caption(outro)
        self.wait(5.0)
