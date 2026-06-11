"""Kernel PCA walkthrough: three kernels animated with manim.

Reuses the Isomap explainer's shared visual system (style, builders) and
dataset loading so Kernel PCA clips match the Isomap and PCA clips.

The same scene class renders three clips, one per kernel, controlled by the
MFI_KERNEL env var (rbf | polynomial | linear, default rbf). The step
structure mirrors kpca.js presentSubSteps (0, 3, 4, 5, 6).

Env: MFI_N (point count), MFI_DATASET (dataset id, default swiss_roll),
     MFI_KERNEL (kernel name, default rbf).

Manim 0.18.1 API notes (same as isomap walkthrough header):
- Section.NORMAL does not exist; use DefaultSectionType.NORMAL.
- add_fixed_in_frame_mobjects() must be called before self.play() for 2D overlays.
- Heatmaps use B.heatmap (VGroup of Square mobjects); add_fixed_in_frame_mobjects
  is required so they stay fixed while the camera orbits.
"""
import os
import numpy as np
from manim import (
    ThreeDScene, ThreeDAxes, DEGREES, FadeIn, FadeOut, Transform,
    ReplacementTransform, DOWN, UP, RIGHT, LEFT,
    VGroup, Dot, Line3D, Text, MathTex,
)
from manim.scene.section import DefaultSectionType
from manimexp.isomap import style as S
from manimexp.isomap import builders as B
from manimexp.isomap.data import (
    load_dataset_points, kernel_matrix, center_kernel, top2_kernel_embed,
)

N = int(os.environ.get("MFI_N", "1000"))
SEED = 0
DATASET = os.environ.get("MFI_DATASET", "swiss_roll")
KERNEL = os.environ.get("MFI_KERNEL", "rbf").lower().strip()

_SEC = DefaultSectionType.NORMAL

# ------------------------------------------------------------------ #
# Per-dataset intro captions                                          #
# ------------------------------------------------------------------ #

_KERNEL_LABEL = {
    "rbf": "RBF",
    "polynomial": "polynomial",
    "linear": "linear",
}

_INTRO_SUFFIX = (
    " It runs PCA in the implicit feature space the kernel defines."
)

DATASET_INTRO = {
    "swiss_roll": "A 2D sheet rolled up in 3D.",
    "s_curve": "A 2D sheet bent into an S in 3D.",
    "twin_peaks": "A bumpy height surface in 3D.",
    "saddle": "A curved saddle surface in 3D.",
    "cylinder": "A sheet wrapped into a cylinder.",
    "severed_sphere": "A sphere with its cap removed.",
    "helix": "A ribbon wound into a helix in 3D.",
    "trefoil_knot": "A ribbon tied into a trefoil knot.",
    "toroidal_helix": "A ribbon coiled around a torus.",
    "spiral_disk": "A ribbon wound into a spiral.",
    "full_sphere": "A full sphere in 3D.",
    "hilbert": "A Hilbert curve filling a cube.",
    "clusters_3d": "Several clusters of points in 3D.",
}

# Per-kernel outro captions for the embedding step.
KERNEL_OUTRO = {
    "rbf": (
        "The RBF kernel maps points to a high-dimensional Gaussian feature space, "
        "but it does not recover the flat sheet of the swiss roll. "
        "The top two components correlate only weakly with the roll angle and height, "
        "so the rolled layers remain entangled in this 2D view. "
        "Isomap, LLE, and Laplacian Eigenmaps use the graph structure to unroll the sheet; "
        "Kernel PCA with RBF does not."
    ),
    "polynomial": (
        "The polynomial kernel bends the feature space by raising dot products "
        "to a higher power. This curves the similarity structure but does not "
        "cleanly separate the swiss roll's layers, so the embedding remains tangled."
    ),
    "linear": (
        "The linear kernel K(x, y) = x . y is equivalent to the ordinary dot "
        "product. Kernel PCA with a linear kernel collapses to standard PCA and "
        "projects by variance, so the rolled layers still overlap."
    ),
}
_OUTRO_DEFAULT = "The 2D embedding shows the kernel-induced feature space coordinates."

# ------------------------------------------------------------------ #
# Pseudocode lines                                                    #
# ------------------------------------------------------------------ #

KPCA_PSEUDO = [
    r"0:\ \text{input: points } X,\ \text{kernel } k",
    r"1:\ K_{ij} = k(x_i,\, x_j)",
    r"2:\ K_c = K - \mathbf{1}_N K - K \mathbf{1}_N + \mathbf{1}_N K \mathbf{1}_N",
    r"3:\ K_c = V\,\Lambda\,V^{\top}",
    r"4:\ y_{i,k} = \sqrt{\lambda_k}\, v_{k,i}",
]


def kpca_panel(active_index):
    items = []
    for idx, tex in enumerate(KPCA_PSEUDO):
        color = S.ACCENT if idx == active_index else S.MUTED
        t = MathTex(tex, font_size=24, color=color)
        t.set_opacity(1.0 if idx == active_index else 0.35)
        items.append(t)
    group = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    max_w = 6.2
    if group.width > max_w:
        group.scale(max_w / group.width)
    return group


# ------------------------------------------------------------------ #
# Per-kernel formula strings                                          #
# ------------------------------------------------------------------ #

KERNEL_FORMULA = {
    "rbf": r"K_{ij} = \exp\!\bigl(-\gamma \|\, x_i - x_j \,\|^2\bigr)",
    "polynomial": r"K_{ij} = (x_i \cdot x_j + 1)^d",
    "linear": r"K_{ij} = x_i \cdot x_j",
}


class KPCAWalkthrough(ThreeDScene):

    def construct(self):
        self.camera.background_color = S.BG
        d = load_dataset_points(DATASET, N, seed=SEED)
        pts, t = d["points"], d["t"]

        # Run the full KPCA pipeline up front so all steps use consistent data.
        K_mat = kernel_matrix(pts, KERNEL, gamma=1.0, degree=3, constant=1.0)
        Kc_mat = center_kernel(K_mat)
        Y, vecs, vals = top2_kernel_embed(Kc_mat)

        # Top-6 eigenvalues of Kc (descending) for the bar chart.
        all_vals_desc = np.sort(np.linalg.eigvalsh(Kc_mat))[::-1]
        top6_vals = all_vals_desc[:6].tolist()

        self.d = dict(pts=pts, t=t, K=K_mat, Kc=Kc_mat, Y=Y, vecs=vecs, vals=vals,
                      top6_vals=top6_vals)
        self.cap = None
        self.pseudo = None

        self.section_raw()
        self.section_kernel()
        self.section_center()
        self.section_eig()
        self.section_embedding()

    # ------------------------------------------------------------------ #
    # Overlay helpers (mirror the PCA / Isomap walkthroughs)             #
    # ------------------------------------------------------------------ #

    def set_caption(self, text):
        new = B.caption(text).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(new)
        new.set_opacity(0)
        if self.cap is None:
            self.play(new.animate.set_opacity(1.0), run_time=S.T_FAST)
        else:
            old = self.cap
            self.play(FadeOut(old, shift=0.0), new.animate.set_opacity(1.0),
                      run_time=S.T_FAST)
            self.remove(old)
        self.cap = new

    def set_pseudo(self, active_index):
        panel = kpca_panel(active_index).to_corner(LEFT + UP, buff=0.15)
        self.add_fixed_in_frame_mobjects(panel)
        panel.set_opacity(0)
        if self.pseudo is None:
            self.play(panel.animate.set_opacity(1.0), run_time=S.T_FAST)
        else:
            old = self.pseudo
            self.play(FadeOut(old, shift=0.0), panel.animate.set_opacity(1.0),
                      run_time=S.T_FAST)
            self.remove(old)
        self.pseudo = panel

    # ------------------------------------------------------------------ #
    # Step 0: raw point cloud                                             #
    # ------------------------------------------------------------------ #

    def section_raw(self):
        self.next_section("step-1-raw", type=_SEC)
        self.set_camera_orientation(phi=65 * DEGREES, theta=30 * DEGREES, zoom=0.9)

        self.axes = ThreeDAxes(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1], z_range=[-4, 4, 1],
            x_length=8, y_length=8, z_length=8,
            axis_config={
                "stroke_color": S.MUTED, "stroke_width": 0.8, "stroke_opacity": 0.45,
            },
        )
        self.play(FadeIn(self.axes, run_time=S.T_FAST))

        self.cloud = B.point_cloud(self.d["pts"], self.d["t"])
        self.play(FadeIn(self.cloud, run_time=S.T_INTRO))

        self.set_pseudo(0)

        ds_line = DATASET_INTRO.get(DATASET, "A curved cloud in 3D.")
        klabel = _KERNEL_LABEL.get(KERNEL, KERNEL)
        intro = (
            ds_line
            + " Kernel PCA with a " + klabel + " kernel."
            + _INTRO_SUFFIX
        )
        self.set_caption(intro)

        self.begin_ambient_camera_rotation(rate=2 * np.pi / 14.0, about="theta")
        self.wait(5.5)

    # ------------------------------------------------------------------ #
    # Step 3: kernel matrix K                                             #
    # ------------------------------------------------------------------ #

    def section_kernel(self):
        self.next_section("step-3-kernel", type=_SEC)
        self.set_pseudo(1)

        # Caption 1: what we are building.
        self.set_caption(
            "Build the kernel matrix K: each entry K_ij = k(x_i, x_j) is the "
            "similarity of two points."
        )

        # Heatmap panel top-right, grid scaled to height 2.4, label above it.
        hm_K = B.heatmap(self.d["K"], N, diverging=False)
        hm_K.scale_to_fit_height(2.4)
        lbl_K = Text("K", font_size=26, color=S.INK)
        self.add_fixed_in_frame_mobjects(hm_K, lbl_K)
        hm_K.to_corner(RIGHT + UP, buff=0.4)
        lbl_K.next_to(hm_K, UP, buff=0.10)
        hm_K.set_opacity(0)
        lbl_K.set_opacity(0)
        self.play(
            hm_K.animate.set_opacity(1.0),
            lbl_K.animate.set_opacity(1.0),
            run_time=S.T_NORMAL,
        )

        # Kernel formula below the heatmap via fit_formula so it never overflows.
        f = B.fit_formula(
            KERNEL_FORMULA.get(KERNEL, r"K_{ij} = k(x_i, x_j)"),
            max_width=4.4, scale=0.75,
        )
        self.add_fixed_in_frame_mobjects(f)
        f.next_to(hm_K, DOWN, buff=0.28)
        # Clamp so formula stays above the caption zone.
        if f.get_bottom()[1] < -2.6:
            f.shift(UP * (abs(f.get_bottom()[1]) - 2.6))
        f.set_opacity(0)
        self.play(f.animate.set_opacity(1.0), run_time=S.T_FAST)
        self.wait(3.5)

        # Caption 2: what the N x N matrix does in the pipeline.
        self.set_caption(
            "K is N x N and replaces the data matrix for the rest of the pipeline."
        )
        self.wait(4.5)

        self.hm_K = hm_K
        self.lbl_K = lbl_K
        self.kernel_f = f

    # ------------------------------------------------------------------ #
    # Step 4: center the kernel matrix                                    #
    # ------------------------------------------------------------------ #

    def section_center(self):
        self.next_section("step-4-center", type=_SEC)
        self.set_pseudo(2)

        self.set_caption(
            "Center the kernel matrix: the feature-space analogue of "
            "subtracting the mean."
        )

        # This step is pure matrix algebra, so clear the 3D dataset and axes and
        # bring the kernel matrix front and center: the K heatmap glides from its
        # corner to the middle of the frame and grows, then cross-fades into the
        # centered matrix Kc so the centering reads as a transformation of K.
        CENTER = np.array([0.0, 0.55, 0.0])
        HM_H = 3.0
        grow = HM_H / self.hm_K.height
        self.play(
            FadeOut(self.cloud),
            FadeOut(self.axes),
            FadeOut(self.kernel_f),
            self.hm_K.animate.scale(grow).move_to(CENTER),
            self.lbl_K.animate.scale(1.15).move_to(CENTER + np.array([0.0, HM_H / 2 + 0.32, 0.0])),
            run_time=S.T_NORMAL,
        )

        hm_Kc = B.heatmap(self.d["Kc"], N, diverging=True)
        hm_Kc.scale_to_fit_height(HM_H)
        hm_Kc.move_to(self.hm_K.get_center())
        lbl_Kc = Text("Kc", font_size=30, color=S.INK)
        self.add_fixed_in_frame_mobjects(hm_Kc, lbl_Kc)
        lbl_Kc.next_to(hm_Kc, UP, buff=0.12)
        hm_Kc.set_opacity(0)
        lbl_Kc.set_opacity(0)

        # Cross-fade K -> Kc in place so it reads as the centering transformation.
        self.play(
            FadeOut(self.hm_K, shift=0.0),
            FadeOut(self.lbl_K, shift=0.0),
            hm_Kc.animate.set_opacity(1.0),
            lbl_Kc.animate.set_opacity(1.0),
            run_time=S.T_NORMAL,
        )
        self.remove(self.hm_K, self.lbl_K)

        # Centering formula below the centered matrix, larger and wider now that
        # it has the whole frame width to itself.
        f_center = B.fit_formula(
            r"K_c = K - \mathbf{1}_N K - K \mathbf{1}_N + \mathbf{1}_N K \mathbf{1}_N",
            max_width=6.4, scale=0.78,
        )
        self.add_fixed_in_frame_mobjects(f_center)
        f_center.next_to(hm_Kc, DOWN, buff=0.30)
        if f_center.get_bottom()[1] < -2.0:
            f_center.shift(UP * (abs(f_center.get_bottom()[1]) - 2.0))
        f_center.set_opacity(0)
        self.play(f_center.animate.set_opacity(1.0), run_time=S.T_FAST)

        # Hold on the centered matrix; the detailed walk-through lives in the
        # page's step text beside the video.
        self.wait(4.5)
        self.hm_Kc = hm_Kc
        self.lbl_Kc = lbl_Kc
        self.center_f = f_center

    # ------------------------------------------------------------------ #
    # Step 5: eigendecompose Kc, top-2 readout                           #
    # ------------------------------------------------------------------ #

    def section_eig(self):
        self.next_section("step-5-eig", type=_SEC)
        self.set_pseudo(3)

        # Fade out the centered matrix and restore the 3D dataset and axes, which
        # the embedding step morphs into the 2D layout.
        self.play(
            FadeOut(self.hm_Kc),
            FadeOut(self.lbl_Kc),
            FadeOut(self.center_f),
            FadeIn(self.cloud),
            FadeIn(self.axes),
            run_time=S.T_FAST,
        )

        self.set_caption(
            "Eigendecompose Kc: its eigenvectors are the principal components "
            "in feature space."
        )
        self.wait(3.5)

        # Formula: eigenvalue equation.
        f_eig = B.formula(r"K_c\, v_k = \lambda_k\, v_k").scale(0.85)
        self.add_fixed_in_frame_mobjects(f_eig)
        f_eig.to_corner(RIGHT + UP, buff=0.4).set_opacity(0)
        self.play(f_eig.animate.set_opacity(1.0), run_time=S.T_FAST)

        # Top-2 eigenvalue readout via fit_formula.
        vals = self.d["vals"]
        v1, v2 = float(vals[0]), float(vals[1])
        readout = B.fit_formula(
            rf"\lambda_1 = {v1:.2f} \qquad \lambda_2 = {v2:.2f}",
            max_width=4.8, scale=0.65,
        )
        self.add_fixed_in_frame_mobjects(readout)
        readout.next_to(f_eig, DOWN, buff=0.32, aligned_edge=RIGHT).set_opacity(0)
        self.play(readout.animate.set_opacity(1.0), run_time=S.T_FAST)
        self.wait(2.5)

        # Eigenvalue bar chart: top-6 of Kc (descending), highlight top 2.
        top6 = self.d["top6_vals"]
        bar_group = B.eig_bar_chart(top6, highlight_idxs=[0, 1])

        bar_labels = VGroup()
        label_specs = [
            (r"\lambda_1", S.ACCENT),
            (r"\lambda_2", S.ACCENT),
        ]
        for i, (tex, col) in enumerate(label_specs):
            if i < len(bar_group.submobjects):
                lbl = MathTex(tex, font_size=18, color=col)
                lbl.next_to(bar_group.submobjects[i], DOWN, buff=0.06)
                bar_labels.add(lbl)

        chart_group = VGroup(bar_group, bar_labels)
        self.add_fixed_in_frame_mobjects(chart_group)
        chart_group.next_to(readout, DOWN, buff=0.35, aligned_edge=RIGHT)
        if chart_group.get_bottom()[1] < -2.5:
            chart_group.shift(UP * (abs(chart_group.get_bottom()[1]) - 2.5))
        chart_group.set_opacity(0)
        self.play(chart_group.animate.set_opacity(1.0), run_time=S.T_NORMAL)

        self.set_caption(
            "The top two eigenvalues capture the dominant variation; keep them "
            "for the 2D embedding."
        )
        self.wait(4.0)

        # Fade the chart before the embedding step.
        self.play(FadeOut(chart_group), run_time=S.T_FAST)

        self.eig_f = f_eig
        self.eig_vals = readout

    # ------------------------------------------------------------------ #
    # Step 6: 2D embedding, morph cloud face-on                          #
    # ------------------------------------------------------------------ #

    def section_embedding(self):
        self.next_section("step-6-embedding", type=_SEC)
        self.set_pseudo(4)
        self.stop_ambient_camera_rotation()

        # Scale the embedding to fill the frame (z = 0 so it lives in the xy-plane).
        Y = self.d["Y"]
        s = 3.4 / (float(np.abs(Y).max()) or 1.0)
        target = np.column_stack([Y[:, 0] * s, Y[:, 1] * s, np.zeros(Y.shape[0])])

        # Fade out eigen overlays; bring in the embedding formula.
        self.play(
            FadeOut(self.eig_f),
            FadeOut(self.eig_vals),
            run_time=S.T_FAST,
        )
        f_embed = B.fit_formula(r"y_{i,k} = \sqrt{\lambda_k}\, v_{k,i}", max_width=4.8, scale=0.85)
        self.add_fixed_in_frame_mobjects(f_embed)
        f_embed.to_corner(RIGHT + UP, buff=0.4).set_opacity(0)
        self.play(f_embed.animate.set_opacity(1.0), run_time=S.T_FAST)

        # Flatten to face-on by the shortest theta path (matches PCA walkthrough).
        cur_theta = float(self.camera.get_theta())
        tgt_theta = -90 * DEGREES
        tgt_theta += round((cur_theta - tgt_theta) / (2 * np.pi)) * 2 * np.pi

        self.set_caption(
            "Flatten to face-on and move each point to its 2D embedding coordinate."
        )
        self.move_camera(
            phi=0, theta=tgt_theta, run_time=S.T_SLOW,
            added_anims=[
                *[
                    self.cloud[i].animate.move_to(target[i])
                    for i in range(len(self.cloud.submobjects))
                ],
            ],
        )
        self.wait(1.5)

        # Fade out axes; the cloud remains as the embedding.
        self.play(FadeOut(self.axes), run_time=S.T_FAST)

        outro = KERNEL_OUTRO.get(KERNEL, _OUTRO_DEFAULT)
        self.set_caption(outro)
        self.wait(5.5)
        # Fade the caption, pseudocode panel, and formula so the final 2D
        # embedding is shown unobstructed for a beat before the clip ends.
        self.play(
            FadeOut(self.cap), FadeOut(self.pseudo), FadeOut(f_embed),
            run_time=S.T_NORMAL,
        )
        self.cap, self.pseudo = None, None
        self.wait(2.5)
