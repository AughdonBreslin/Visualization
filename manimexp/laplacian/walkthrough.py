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
    "swiss_roll": "Laplacian Eigenmaps folds the sheet into an arc; the second axis repeats the first instead of adding width.",
    "s_curve": "Laplacian Eigenmaps keeps nearby points close, but the S-curve collapses toward a thin curved arc.",
    "twin_peaks": "The local graph flattens the bumpy surface into a 2D layout.",
    "saddle": "The saddle flattens into a rough blob, nearby points kept close.",
    "cylinder": "The closed band maps to a loop; the local graph does not bridge its seam.",
    "severed_sphere": "The cap flattens into a rough disk, nearby points kept close.",
    "helix": "Laplacian Eigenmaps collapses the thin helix toward a point.",
    "trefoil_knot": "The closed knot folds into an arc rather than a flat strip.",
    "toroidal_helix": "The closed coil collapses toward a 1D arc.",
    "spiral_disk": "Laplacian Eigenmaps collapses the spiral toward a single line.",
    "full_sphere": "A sphere cannot flatten without distortion, so the embedding compresses some neighborhoods.",
    "hilbert": "A folded curve has no 2D layout; linked folds scatter it across the plane.",
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

        # Bottom-6 eigenvalues of L (ascending) for the bar chart.
        all_vals_asc = np.sort(np.linalg.eigvalsh(L))
        bottom6_vals = all_vals_asc[:6].tolist()

        self.d = {
            "pts": pts, "t": t, "adj": adj, "edges": edges,
            "W": W, "L": L, "D_deg": D_deg,
            "vecs": vecs, "vals": vals, "Y": Y,
            "bottom6_vals": bottom6_vals,
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
        self.wait(5.5)

    # ------------------------------------------------------------------ #
    # Section 2: kNN graph                                                #
    # ------------------------------------------------------------------ #

    def section_knn(self):
        self.next_section("step-2-knn", type=_SEC)
        self.set_pseudo(1)

        # Stop the orbit so the flat schematic (with distance labels) stays still
        # and readable; the rotation resumes once the view returns to 3D.
        self.stop_ambient_camera_rotation()

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
        self.wait(0.6)

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
        self.wait(1.0)

        self.set_caption("Each link records the distance between its two points.")
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
        # Back in 3D: resume the continuous orbit.
        self.begin_ambient_camera_rotation(rate=2 * np.pi / 14.0, about="theta")

        self.set_caption("Repeat for every point and the kNN graph spans the whole surface.")
        self.edges_mob = B.graph_edges(self.d["pts"], self.d["edges"])
        self.play(
            LaggedStart(
                *[FadeIn(edge) for edge in self.edges_mob],
                lag_ratio=0.004,
                run_time=S.T_SLOW,
            )
        )
        self.wait(2.5)

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
        self.wait(4.0)

        # Clean up the neighborhood highlight; keep edges and cloud for the next
        # section. No camera move here: the orbit keeps advancing theta, so a
        # move_camera that reset theta would snap the rotation backwards.
        self.play(
            FadeOut(center_sphere),
            FadeOut(neighbor_spheres),
            FadeOut(nbr_lines),
            run_time=S.T_FAST,
        )
        self.wait(2.5)

    # ------------------------------------------------------------------ #
    # Section 3: heat-kernel affinity W                                   #
    # ------------------------------------------------------------------ #

    def section_affinity(self):
        self.next_section("step-3-affinity", type=_SEC)
        self.set_pseudo(2)

        # Stop the orbit here and hold the dataset at a fixed 45-degree (x-y) view,
        # which is roughly where the rotation has carried it by this point. The
        # nearest-equivalent target avoids any visible snap from the current theta.
        self.stop_ambient_camera_rotation()
        cur_theta = float(self.camera.get_theta())
        tgt_theta = 45 * DEGREES
        tgt_theta += round((cur_theta - tgt_theta) / (2 * np.pi)) * 2 * np.pi
        self.move_camera(theta=tgt_theta, run_time=S.T_NORMAL)

        pts = self.d["pts"]
        edges = self.d["edges"]
        W = self.d["W"]

        # Compute per-edge weight for coloring.
        edge_weights = np.array([float(W[i, j]) for (i, j) in edges])
        w_min = float(edge_weights.min())
        w_max = float(edge_weights.max())
        w_span = max(1e-9, w_max - w_min)

        self.set_caption("The heat kernel turns each edge distance into a weight.")

        # Recolor each edge line by its weight: strong (close to 1) -> GOOD, weak -> MUTED.
        recolor_anims = []
        for line, w in zip(self.edges_mob.submobjects, edge_weights):
            u = (w - w_min) / w_span
            col = interpolate_color(S.MUTED, S.GOOD, u)
            recolor_anims.append(line.animate.set_color(col).set_opacity(min(0.85, 0.12 + 0.73 * u)))
        self.play(AnimationGroup(*recolor_anims, lag_ratio=0.0), run_time=S.T_SLOW)
        self.wait(3.0)

        # Formula in top-right corner via fit_formula.
        f_W = B.fit_formula(
            r"W_{ij} = \exp\!\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)",
            max_width=4.6, scale=0.75,
        )
        self.add_fixed_in_frame_mobjects(f_W)
        f_W.to_corner(RIGHT + UP, buff=0.4).set_opacity(0)
        self.play(f_W.animate.set_opacity(1.0), run_time=S.T_FAST)

        # Sparse-pattern heatmap of W: reorder rows/cols by manifold parameter t
        # so neighbors become index-adjacent; the nonzeros form a near-diagonal band.
        t_order = np.argsort(self.d["t"])
        W_reord = W[np.ix_(t_order, t_order)]
        sub_size = min(60, N)
        W_sub = W_reord[:sub_size, :sub_size]
        hm_W = B.heatmap(W_sub, sub_size, max_cells=sub_size, cell=0.055,
                         diverging=True, mode="pattern")
        hm_W.scale_to_fit_height(2.2)
        lbl_W = Text(f"W (reordered, {sub_size}x{sub_size} block)", font_size=20, color=S.INK)
        self.add_fixed_in_frame_mobjects(hm_W, lbl_W)
        hm_W.to_corner(RIGHT + UP, buff=0.4)
        hm_W.shift(DOWN * (f_W.height + 0.5))
        lbl_W.next_to(hm_W, UP, buff=0.10)
        hm_W.set_opacity(0)
        lbl_W.set_opacity(0)
        self.play(
            hm_W.animate.set_opacity(1.0),
            lbl_W.animate.set_opacity(1.0),
            run_time=S.T_NORMAL,
        )
        self.set_caption("W is sparse: only the k-nearest pairs are nonzero.")
        self.wait(5.0)

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

        # Hide the dataset and the affinity overlays; the matrix algebra plays out
        # in the foreground as a lineage W -> D -> L. The orbit keeps running with
        # nothing 3D visible so the camera stays continuous for the embedding.
        self.play(
            FadeOut(self.affinity_f),
            FadeOut(self.hm_W),
            FadeOut(self.lbl_W),
            FadeOut(self.cloud),
            FadeOut(self.edges_mob),
            FadeOut(self.axes),
            run_time=S.T_NORMAL,
        )

        # Reorder rows/cols by manifold parameter t so the sparse band structure
        # of each 60x60 sub-block is visible rather than a wash.
        idx = np.argsort(self.d["t"])[:min(60, N)]
        sub = len(idx)
        W_sub = W[np.ix_(idx, idx)]
        D_sub = D_deg[np.ix_(idx, idx)]
        L_sub = L[np.ix_(idx, idx)]
        # diverging maps zeros to the dark background and nonzeros to color, so the
        # band of W, the lone diagonal of D, and the signed entries of L all read
        # cleanly (diverging=False would tint the empty cells a flat mid-tone).
        # For L, the diagonal degree dwarfs the off-diagonal -W entries, so clip it
        # to the off-diagonal scale; otherwise the -W band normalizes to near-zero
        # and L looks purely diagonal. Clipping keeps the green diagonal (D) and the
        # red off-diagonal band (-W) both visible, making L = D - W literal.
        L_off_max = float(np.abs(L_sub - np.diag(np.diag(L_sub))).max()) or 1.0
        L_disp = np.clip(L_sub, -2.0 * L_off_max, 2.0 * L_off_max)
        hm_W = B.heatmap(W_sub, sub, max_cells=sub, cell=0.055, diverging=True, mode="pattern")
        hm_D = B.heatmap(D_sub, sub, max_cells=sub, cell=0.055, diverging=True, mode="pattern")
        hm_L = B.heatmap(L_disp, sub, max_cells=sub, cell=0.055, diverging=True, mode="pattern")

        # Bring W to the center and run the lineage W -> D -> L. The affinity step
        # already introduced W, so the caption here looks ahead to the goal.
        hm_W.scale_to_fit_height(2.3).move_to(np.array([0.0, -0.1, 0.0]))
        self.add_fixed_in_frame_mobjects(hm_W)
        hm_W.set_opacity(0)
        self.set_caption("Combine W with its degrees to build the Laplacian.")
        self.play(hm_W.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.wait(0.8)

        self.lineage = B.MatrixLineage(self)
        self.lineage.start(hm_W, "W")
        self.wait(1.2)
        self.lineage.push(hm_D, "D", r"\text{row sums}",
                          caption="D sums each row of W onto its diagonal.")
        self.wait(2.0)
        self.lineage.push(hm_L, "L", r"(\,\cdot\,) - W",
                          caption="L = D - W: a green diagonal minus the red W band.")
        self.wait(3.0)

    # ------------------------------------------------------------------ #
    # Section 5: smallest non-trivial eigenvectors of L                   #
    # ------------------------------------------------------------------ #

    def section_eig(self):
        self.next_section("step-5-eig", type=_SEC)
        self.set_pseudo(4)

        # Say what L measures and why we want its smallest eigenvalues before the
        # eigendecomposition takes the foreground (dataset stays hidden).
        self.set_caption("L measures how much a coordinate varies across edges.")
        self.wait(3.5)
        self.set_caption("Smooth coordinates vary the least across the graph.")
        self.wait(3.5)

        # Foreground the eigendecomposition of L. L is positive semi-definite, so
        # its eigenvalues are non-negative; the smallest is the trivial constant
        # mode (skip it) and the next two are the smoothest, lowest-variation
        # coordinates. Strips are reordered by t so their structure reads.
        t_sort = np.argsort(self.d["t"])
        self.eig_overlays = self.lineage.eig_focus(
            r"L = V\,\Lambda\,V^{\top}",
            self.d["bottom6_vals"],
            self.d["vecs"][t_sort],
            caption="So take L's smallest eigenvalues, not its largest.",
            caption_vectors="Skip the trivial mode; keep v1 and v2.",
            highlight_idxs=(1, 2),
            trivial_idx=0,
        )
        self.wait(3.0)

        # Contrast with variance methods so the choice of smallest is explicit.
        self.set_caption("Variance methods (PCA, MDS) keep the largest; Laplacian keeps the smallest.")
        self.wait(4.0)

    # ------------------------------------------------------------------ #
    # Section 6: 2D embedding                                             #
    # ------------------------------------------------------------------ #

    def section_embedding(self):
        self.next_section("step-6-embedding", type=_SEC)
        self.set_pseudo(5)

        Y = self.d["Y"]
        s = 3.4 / (float(np.abs(Y).max()) or 1.0)
        target = np.column_stack([Y[:, 0] * s, Y[:, 1] * s, np.zeros(Y.shape[0])])

        # Stop the orbit before the cloud returns so it is presented at a fixed
        # angle while it recolors, rather than rotating, before the embedding morph.
        self.stop_ambient_camera_rotation()

        # Clear the eigendecomposition overlays and restore the cloud, recolored
        # by the first eigenvector so the embedding carries the coordinate coloring.
        self.play(FadeOut(self.eig_overlays), FadeIn(self.cloud), run_time=S.T_FAST)
        v1 = self.d["vecs"][:, 0]
        recolor_anims = B.recolor_cloud_by_values(self.cloud, v1, cmap=B.rainbow_color, grow=1.5)
        self.play(AnimationGroup(*recolor_anims, lag_ratio=0.0), run_time=S.T_NORMAL)

        f_embed = B.formula(r"Y = [\,v_1\;\; v_2\,]").scale(0.85)
        self.add_fixed_in_frame_mobjects(f_embed)
        f_embed.to_corner(RIGHT + UP, buff=0.4).set_opacity(0)
        self.play(f_embed.animate.set_opacity(1.0), run_time=S.T_FAST)

        # Flatten camera to face-on (phi=0) by the shortest angular path.
        cur_theta = float(self.camera.get_theta())
        tgt_theta = -90 * DEGREES
        tgt_theta += round((cur_theta - tgt_theta) / (2 * np.pi)) * 2 * np.pi

        self.set_caption("Face the embedding plane and place each point in 2D.")

        # Morph the cloud into the 2D embedding positions while flattening.
        cloud_dots = self.cloud.submobjects
        n_pts = len(cloud_dots)
        self.move_camera(
            phi=0, theta=tgt_theta,
            run_time=S.T_SLOW,
            added_anims=[
                cloud_dots[i].animate.move_to(target[i]) for i in range(n_pts)
            ],
        )
        self.wait(1.5)

        self.set_caption(DATASET_OUTRO.get(DATASET, _OUTRO_DEFAULT))
        self.wait(5.0)
        # Fade the caption, pseudocode panel, and formula so the final 2D
        # embedding is shown unobstructed for a beat before the clip ends.
        self.play(
            FadeOut(self.cap), FadeOut(self.pseudo), FadeOut(f_embed),
            run_time=S.T_NORMAL,
        )
        self.cap, self.pseudo = None, None
        self.wait(2.5)
