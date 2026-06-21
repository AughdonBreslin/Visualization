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
    AnimationGroup, LaggedStart, Indicate, DOWN, UP, RIGHT, LEFT,
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
    "swiss_roll": "LLE partly unrolls the sheet but warps it into an uneven fan, not a clean rectangle.",
    "s_curve": "LLE preserves local patches, but on the S-curve they pinch into a narrow wedge rather than a flat sheet.",
    "twin_peaks": "LLE flattens the surface into an uneven fan, distorting the bumps.",
    "saddle": "LLE flattens the saddle into a rough blob, with noticeable local distortion.",
    "cylinder": "LLE unrolls the cylinder into a rough sheet, though the seam stays distorted.",
    "severed_sphere": "LLE flattens the cap into a distorted fan.",
    "helix": "LLE collapses the thin helix toward a single point; a 1D coil has no patches to preserve.",
    "trefoil_knot": "LLE folds the closed knot into an arc rather than a flat strip.",
    "toroidal_helix": "LLE collapses the coil into a thin bent line.",
    "spiral_disk": "LLE collapses the spiral toward a single line.",
    "full_sphere": "A sphere cannot flatten without tearing, so LLE distorts it into a fan.",
    "hilbert": "A folded curve has no clean 2D layout, so LLE scatters it without meaningful order.",
    "clusters_3d": "LLE collapses each disconnected cluster to a point; the global layout is arbitrary.",
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

        # Bottom 6 eigenvalues of M (including the trivial one at index 0).
        all_vals_asc = np.sort(np.linalg.eigvalsh(M))
        bottom6_vals = all_vals_asc[:6].tolist()

        self.d = dict(
            pts=pts, t=t_param, adj=adj, edges=edges,
            W=W, M=M, vecs=vecs, vals=vals, Y=Y,
            center_idx=center_idx,
            nbr_indices=nbr_indices,
            nbr_weights=nbr_weights,
            bottom6_vals=bottom6_vals,
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
        self.wait(2.5)

        # Beat 2: schematic star of neighbors (flat, z = 0).
        center_pt = np.array([0.0, 0.0, 0.0])
        center_dot = Dot(point=center_pt, radius=0.10, color=S.ACCENT)
        self.play(FadeIn(center_dot, run_time=S.T_FAST))

        # Eight neighbors at evenly-spaced angles so no dots or distance
        # labels overlap, regardless of how many neighbors k has.
        # A small phase shift (pi/8) avoids placing any node exactly on
        # the cardinal axes. Radii vary slightly to give distinct distances.
        _phase = np.pi / 8
        _radii = [1.55, 1.70, 1.60, 1.45, 1.65, 1.50, 1.75, 1.58]
        offsets = np.array([
            [_radii[i] * np.cos(2 * np.pi * i / 8 + _phase),
             _radii[i] * np.sin(2 * np.pi * i / 8 + _phase),
             0.0]
            for i in range(8)
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
        # Back in 3D: resume the continuous orbit.
        self.begin_ambient_camera_rotation(rate=2 * np.pi / 14.0, about="theta")

        self.set_caption("Do this for every point and the kNN graph covers the whole surface.")
        self.edges_mob = B.graph_edges(self.d["pts"], self.d["edges"])
        self.play(
            LaggedStart(
                *[FadeIn(edge) for edge in self.edges_mob],
                lag_ratio=0.004,
                run_time=S.T_SLOW,
            )
        )
        self.wait(3.0)

    # ------------------------------------------------------------------ #
    # Section 3: reconstruction weights                                   #
    # ------------------------------------------------------------------ #

    def section_weights(self):
        self.next_section("step-3-weights", type=_SEC)
        self.set_pseudo(2)

        pts = self.d["pts"]
        ci = self.d["center_idx"]
        ni = self.d["nbr_indices"]
        nw = self.d["nbr_weights"]
        tvals = self.d["t"]

        # Hide the main dataset; the reconstruction plays out in the foreground on
        # a single sample point and its neighbors. The orbit keeps running with
        # nothing 3D visible, so the camera stays continuous for the embedding.
        self.play(
            FadeOut(self.cloud),
            FadeOut(self.edges_mob),
            FadeOut(self.axes),
            run_time=S.T_NORMAL,
        )

        # Show every neighbor that carries weight (sorted by influence) so the
        # displayed weights sum to exactly 1 and the reconstruction is exact.
        sel = np.argsort(np.abs(nw))[::-1]
        ni_s, nw_s = ni[sel], nw[sel]

        # Project neighbor offsets onto their local plane and scale up so the patch
        # fills a clear disc around the center.
        C3 = pts[ci]
        offs = pts[ni_s] - C3
        if offs.shape[0] >= 2:
            _, _, Vt = np.linalg.svd(offs, full_matrices=False)
            basis = Vt[:2]
        else:
            basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        proj = offs @ basis.T
        rmax = float(np.linalg.norm(proj, axis=1).max()) or 1.0
        s = 1.85 / rmax
        origin = np.array([-0.9, -0.6, 0.0])
        nbr_pos = [origin + np.array([proj[i, 0] * s, proj[i, 1] * s, 0.0])
                   for i in range(len(ni_s))]

        # Center node x_i and neighbor nodes (colored like the cloud).
        center_dot = Dot(point=origin, radius=0.13, color=S.INK)
        center_lbl = MathTex("x_i", font_size=30, color=S.INK)
        center_lbl.next_to(center_dot, DOWN, buff=0.12)
        nbr_dots = VGroup(*[
            Dot(point=nbr_pos[i], radius=0.10, color=B.rainbow_color(float(tvals[j])))
            for i, j in enumerate(ni_s)
        ])
        self.add_fixed_in_frame_mobjects(center_dot, center_lbl, nbr_dots)
        for m in (center_dot, center_lbl, nbr_dots):
            m.set_opacity(0)

        self.set_caption("Take one point and its k neighbors.")
        self.play(center_dot.animate.set_opacity(1.0),
                  center_lbl.animate.set_opacity(1.0), run_time=S.T_FAST)
        self.play(LaggedStart(*[d.animate.set_opacity(1.0) for d in nbr_dots],
                              lag_ratio=0.12, run_time=S.T_NORMAL))
        self.wait(1.0)

        # Place each weight label just outside its neighbor, then relax the labels
        # apart so none overlap, however the neighbors happen to project.
        lpos = []
        for i in range(len(ni_s)):
            radial = nbr_pos[i] - origin
            rn = float(np.linalg.norm(radial))
            radial = radial / rn if rn > 1e-9 else np.array([0.0, 1.0, 0.0])
            lpos.append(nbr_pos[i] + radial * 0.46)
        for _ in range(60):
            for a in range(len(lpos)):
                for b in range(a + 1, len(lpos)):
                    diff = lpos[a] - lpos[b]
                    dist = float(np.linalg.norm(diff))
                    if dist < 0.7:
                        push = (0.7 - dist) / 2.0 * (diff / (dist + 1e-9))
                        lpos[a] = lpos[a] + push
                        lpos[b] = lpos[b] - push

        # Arrows neighbor -> center, width by |weight|, colored by sign.
        wmax = float(np.abs(nw_s).max()) or 1.0
        arrows = VGroup()
        labels = VGroup()
        for i in range(len(ni_s)):
            w = float(nw_s[i])
            col = S.GOOD if w >= 0 else S.WARM
            arrows.add(Arrow(nbr_pos[i], origin, buff=0.16,
                             stroke_width=2.0 + 6.0 * abs(w) / wmax, color=col,
                             max_tip_length_to_length_ratio=0.18))
            labels.add(MathTex(f"{w:.2f}", font_size=26, color=col).move_to(lpos[i]))
        self.add_fixed_in_frame_mobjects(arrows, labels)
        arrows.set_opacity(0)
        labels.set_opacity(0)

        self.set_caption("Write it as a weighted blend of those neighbors.")
        self.play(LaggedStart(*[a.animate.set_opacity(0.9) for a in arrows],
                              lag_ratio=0.12, run_time=S.T_SLOW))
        self.play(LaggedStart(*[l.animate.set_opacity(1.0) for l in labels],
                              lag_ratio=0.10, run_time=S.T_NORMAL))
        self.wait(1.2)

        # Why these weights: least squares solves for the blend whose weighted
        # average is exactly x_i. A neighbor's weight is the share it must carry
        # so the neighbors balance around x_i; opposite neighbors take negative
        # weight to pull the average back into place.
        f_recon = B.fit_formula(r"x_i = \sum_j w_j\, x_j", max_width=4.2, scale=0.9)
        self.add_fixed_in_frame_mobjects(f_recon)
        f_recon.move_to(np.array([4.0, 1.1, 0.0])).set_opacity(0)
        self.set_caption("Each weight is solved so the blend balances on x_i.")
        self.play(f_recon.animate.set_opacity(1.0), run_time=S.T_FAST)
        self.wait(2.6)

        # Reconstruction: start at the neighbors' center and add each neighbor's
        # weighted offset tip-to-tail; the steps arrive exactly at x_i, so the
        # weighted blend rebuilds it.
        nbr_arr = np.array(nbr_pos)
        cen = nbr_arr.mean(axis=0)
        order = np.argsort([np.arctan2((nbr_pos[i] - cen)[1], (nbr_pos[i] - cen)[0])
                            for i in range(len(ni_s))])
        start_dot = Dot(point=cen, radius=0.07, color=S.MUTED)
        walker = Dot(point=cen, radius=0.11, color=S.ACCENT)
        self.add_fixed_in_frame_mobjects(start_dot, walker)
        start_dot.set_opacity(0)
        walker.set_opacity(0)
        # Dim the weight arrows so the reconstruction path is the clear focus; the
        # neighbor dots and weight values stay readable for context.
        self.set_caption("Start at the neighbors' center, then add each weighted offset.")
        self.play(
            arrows.animate.set_opacity(0.1),
            labels.animate.set_opacity(0.45),
            start_dot.animate.set_opacity(0.85),
            walker.animate.set_opacity(1.0),
            run_time=S.T_FAST,
        )
        trail = VGroup()
        acc = cen.astype(float).copy()
        n_ord = len(order)
        for rank, i in enumerate(order):
            w = float(nw_s[i])
            nxt = origin.copy() if rank == n_ord - 1 else acc + w * (nbr_pos[i] - cen)
            seg = Line(acc, nxt, stroke_width=5.0,
                       color=(S.GOOD if w >= 0 else S.WARM)).set_opacity(0.95)
            self.add_fixed_in_frame_mobjects(seg)
            trail.add(seg)
            self.play(Create(seg), walker.animate.move_to(nxt), run_time=0.5)
            acc = nxt
        self.set_caption("The steps land on x_i: the weighted blend rebuilds it.")
        self.play(Indicate(center_dot, color=S.GOOD, scale_factor=1.8), run_time=S.T_NORMAL)
        self.wait(2.4)

        # The weights sum to exactly one (an affine combination): hold on the
        # computed value so the constraint reads clearly.
        sumw = float(np.sum(nw_s))
        f_sum = B.fit_formula(rf"\sum_j w_j = {sumw:.2f}", max_width=4.2, scale=0.9)
        self.add_fixed_in_frame_mobjects(f_sum)
        f_sum.move_to(np.array([4.0, -0.6, 0.0])).set_opacity(0)
        self.set_caption("And the weights are constrained to sum to exactly 1.")
        self.play(f_sum.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.wait(4.0)

        # Clear the reconstruction; the matrices take the foreground next.
        self.recon_diagram = VGroup(center_dot, center_lbl, nbr_dots, arrows,
                                    labels, f_recon, f_sum, start_dot, walker, trail)
        self.play(FadeOut(self.recon_diagram), run_time=S.T_FAST)

    # ------------------------------------------------------------------ #
    # Section 4 (step-5-eig): eigenvectors of M = (I - W)^T (I - W)     #
    # ------------------------------------------------------------------ #

    def section_eig(self):
        self.next_section("step-5-eig", type=_SEC)
        self.set_pseudo(3)

        # Build the matrices in the foreground as a lineage W -> I-W -> M (dataset
        # stays hidden), then eigendecompose M. Heatmaps use t-reordered 60x60
        # sub-blocks at per-entry resolution so their sparse band structure shows.
        idx = np.argsort(self.d["t"])[:min(60, N)]
        sub = len(idx)
        W_sub = self.d["W"][np.ix_(idx, idx)]
        IW_sub = np.eye(sub) - W_sub
        M_sub = self.d["M"][np.ix_(idx, idx)]
        hm_W = B.heatmap(W_sub, sub, max_cells=sub, cell=0.055, diverging=True, mode="pattern")
        hm_IW = B.heatmap(IW_sub, sub, max_cells=sub, cell=0.055, diverging=True, mode="pattern")
        hm_M = B.heatmap(M_sub, sub, max_cells=sub, cell=0.055, diverging=True, mode="mean")

        # Show the whole weight matrix W in the foreground first, then run the
        # lineage W -> I-W -> M.
        hm_W.scale_to_fit_height(2.3).move_to(np.array([0.0, -0.1, 0.0]))
        self.add_fixed_in_frame_mobjects(hm_W)
        hm_W.set_opacity(0)
        self.set_caption("Stack every point's weights into the matrix W.")
        self.play(hm_W.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.wait(2.5)

        self.lineage = B.MatrixLineage(self)
        self.lineage.start(hm_W, "W", caption="W is sparse: k weights per row.")
        self.wait(1.5)
        self.lineage.push(hm_IW, "I - W", r"I - (\cdot)",
                          caption="Subtract W from the identity.")
        self.wait(1.5)
        self.lineage.push(hm_M, "M", r"(\cdot)^{\top}(\cdot)",
                          caption="Multiply by its transpose to form M.")
        self.wait(1.5)

        # Say what M is, and why we want its smallest eigenvalues, before the
        # eigendecomposition takes the foreground.
        self.set_caption("M scores how badly each point's neighbors miss it.")
        self.wait(3.5)
        self.set_caption("Coordinates with the least error are M's smallest eigenvectors.")
        self.wait(3.5)

        # Foreground the eigendecomposition. M is positive semi-definite, so its
        # eigenvalues are non-negative; the smallest is the trivial constant mode
        # (skip it) and the next two are the smoothest, lowest-error coordinates.
        # Strips are reordered by manifold parameter so their structure reads.
        t_sort = np.argsort(self.d["t"])
        self.eig_overlays = self.lineage.eig_focus(
            r"M = V\,\Lambda\,V^{\top}",
            self.d["bottom6_vals"],
            self.d["vecs"][t_sort],
            caption="So take M's smallest eigenvalues, not its largest.",
            caption_vectors="Skip the trivial mode; keep v1 and v2.",
            highlight_idxs=(1, 2),
            trivial_idx=0,
        )
        self.wait(3.0)

        # Contrast with variance methods so the choice of smallest is explicit.
        self.set_caption("Variance methods (PCA, MDS) keep the largest; LLE keeps the smallest.")
        self.wait(4.0)

    # ------------------------------------------------------------------ #
    # Section 5 (step-6-embedding): 2D embedding                         #
    # ------------------------------------------------------------------ #

    def section_embed(self):
        self.next_section("step-6-embedding", type=_SEC)
        self.set_pseudo(4)

        # Clear the eigendecomposition overlays and restore the cloud, recolored by
        # the first eigenvector so the embedding carries the coordinate's coloring.
        self.play(FadeOut(self.eig_overlays), FadeIn(self.cloud), run_time=S.T_FAST)
        v1 = self.d["vecs"][:, 0]
        recolor_anims = B.recolor_cloud_by_values(self.cloud, v1, cmap=B.rainbow_color, grow=1.0)
        self.play(AnimationGroup(*recolor_anims, lag_ratio=0.0), run_time=S.T_NORMAL)

        Y = self.d["Y"]
        s = 3.4 / (float(np.abs(Y).max()) or 1.0)
        pts3 = np.column_stack([Y[:, 0] * s, Y[:, 1] * s, np.zeros(Y.shape[0])])

        # Stop the ambient rotation and move to face-on (phi=0) by the shortest
        # path, mirroring the PCA template's theta-correction logic.
        self.stop_ambient_camera_rotation()
        cur_theta = float(self.camera.get_theta())
        tgt_theta = -90 * DEGREES
        tgt_theta += round((cur_theta - tgt_theta) / (2 * np.pi)) * 2 * np.pi

        # Embedding formula, standardized buff=0.4.
        f_embed = B.formula(r"Y = [\, v_1 \;\; v_2 \,]").scale(0.85)
        self.add_fixed_in_frame_mobjects(f_embed)
        f_embed.to_corner(RIGHT + UP, buff=0.4).set_opacity(0)
        self.play(f_embed.animate.set_opacity(1.0), run_time=S.T_FAST)

        self.set_caption("Look straight down and place each point at its 2D position.")

        # Morph the cloud into the 2D plane while flattening the camera.
        pts_list = self.cloud.submobjects
        self.move_camera(
            phi=0, theta=tgt_theta, run_time=S.T_SLOW,
            added_anims=[
                pts_list[i].animate.move_to(pts3[i]) for i in range(len(pts_list))
            ],
        )
        self.wait(1.5)

        outro = DATASET_OUTRO.get(
            DATASET,
            "LLE preserved each point's local linear reconstruction from its neighbors, so the sheet unrolls flat.",
        )
        self.set_caption(outro)
        self.wait(4.5)
        # Fade the caption, pseudocode panel, and formula so the final 2D
        # embedding is shown unobstructed for a beat before the clip ends.
        self.play(
            FadeOut(self.cap), FadeOut(self.pseudo), FadeOut(f_embed),
            run_time=S.T_NORMAL,
        )
        self.cap, self.pseudo = None, None
        self.wait(2.5)
