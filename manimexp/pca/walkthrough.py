"""PCA walkthrough: the steps of PCA animated with manim.

Reuses the Isomap explainer's shared visual system (style, builders) and dataset
loading so PCA clips match the Isomap clips and slot into the same web player.
The step timeline mirrors the Isomap clip lengths is NOT required here; this clip
stands on its own with the player chapter markers derived from its own sections.

Env: MFI_N (point count), MFI_DATASET (dataset id, default swiss_roll).
"""
import os
import numpy as np
from manim import (
    ThreeDScene, ThreeDAxes, DEGREES, FadeIn, FadeOut, Create, Write, Transform,
    AnimationGroup, DOWN, UP, RIGHT, LEFT, OUT, VGroup, Dot, Line3D,
    Text, MathTex,
)
from manim.scene.section import DefaultSectionType
from manimexp.isomap import style as S
from manimexp.isomap import builders as B
from manimexp.isomap.data import load_dataset_points

N = int(os.environ.get("MFI_N", "1000"))
SEED = 0
DATASET = os.environ.get("MFI_DATASET", "swiss_roll")
_SEC = DefaultSectionType.NORMAL

_INTRO_TAIL = " PCA looks for the directions of greatest variance."
DATASET_INTRO = {
    "swiss_roll": "A swiss roll in 3D." + _INTRO_TAIL,
    "s_curve": "An S-curve sheet in 3D." + _INTRO_TAIL,
    "twin_peaks": "A bumpy height surface in 3D." + _INTRO_TAIL,
    "saddle": "A saddle surface in 3D." + _INTRO_TAIL,
    "cylinder": "A cylinder in 3D." + _INTRO_TAIL,
    "severed_sphere": "A sphere with its cap removed." + _INTRO_TAIL,
    "helix": "A helical ribbon in 3D." + _INTRO_TAIL,
    "trefoil_knot": "A trefoil-knot ribbon in 3D." + _INTRO_TAIL,
    "toroidal_helix": "A ribbon coiled around a torus." + _INTRO_TAIL,
    "spiral_disk": "A spiral ribbon, nearly flat." + _INTRO_TAIL,
    "full_sphere": "A full sphere in 3D." + _INTRO_TAIL,
    "hilbert": "A Hilbert curve filling a cube." + _INTRO_TAIL,
    "clusters_3d": "Several clusters of points in 3D." + _INTRO_TAIL,
}

DATASET_OUTRO = {
    "swiss_roll": "PCA flattens by projection, so the rolled layers overlap. A linear method cannot unroll the swiss roll.",
    "s_curve": "PCA projects the bent sheet flat, folding the ends over the middle. A linear method cannot unbend it.",
    "twin_peaks": "PCA keeps the two widest directions; the bumps along the vertical collapse into the plane.",
    "saddle": "PCA keeps the two widest directions; the saddle's curvature collapses into the plane.",
    "cylinder": "PCA projects the tube flat, so its front and back overlap. A linear method cannot unwrap it.",
    "severed_sphere": "PCA projects the curved cap flat, overlapping near and far points.",
    "helix": "PCA projects the coil flat, so its turns stack over each other.",
    "trefoil_knot": "PCA projects the knot flat, so its strands cross over each other.",
    "toroidal_helix": "PCA projects the coil flat, so its layers overlap.",
    "spiral_disk": "The spiral is nearly flat already, so PCA recovers it with little distortion.",
    "full_sphere": "PCA projects the sphere to a disk, overlapping opposite sides. A sphere has no flat 2D map.",
    "hilbert": "PCA finds the widest plane, but the folded curve overlaps heavily when flattened.",
    "clusters_3d": "PCA keeps the directions that best separate the clusters, giving a clean 2D view.",
}
_OUTRO_DEFAULT = "PCA keeps the two highest-variance directions; structure along the dropped third direction collapses."

PCA_PSEUDO = [
    r"0:\ \text{input: points } X",
    r"1:\ x_i \leftarrow x_i - \bar{x}\quad(\text{center})",
    r"2:\ C = \tfrac{1}{N-1}\, X_c^{\top} X_c",
    r"3:\ C = V\,\Lambda\,V^{\top}",
    r"4:\ Y = X_c\,[\, v_1 \;\; v_2 \,]",
]

# Distinct colors for the three principal axes (most to least variance).
AXIS_COLORS = [S.ACCENT, S.GOOD, S.WARM]


def pca_panel(active_index):
    items = []
    for idx, tex in enumerate(PCA_PSEUDO):
        color = S.ACCENT if idx == active_index else S.MUTED
        t = MathTex(tex, font_size=24, color=color)
        t.set_opacity(1.0 if idx == active_index else 0.35)
        items.append(t)
    group = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    max_w = 6.2
    if group.width > max_w:
        group.scale(max_w / group.width)
    return group


class PCAWalkthrough(ThreeDScene):

    def construct(self):
        self.camera.background_color = S.BG
        d = load_dataset_points(DATASET, N, seed=SEED)
        pts, t = d["points"], d["t"]

        # --- PCA math ---
        mean = pts.mean(axis=0)
        Xc = pts - mean
        C = (Xc.T @ Xc) / max(1, (pts.shape[0] - 1))
        evals, evecs = np.linalg.eigh(C)          # ascending
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        Y = Xc @ evecs[:, :2]

        self.d = dict(pts=pts, t=t, Xc=Xc, C=C, evals=evals, evecs=evecs, Y=Y)
        self.cap = None
        self.pseudo = None

        self.section_raw()
        self.section_center()
        self.section_cov()
        self.section_eig()
        self.section_project()
        self.section_embed()

    # ------------------------------------------------------------------ #
    # Overlay helpers (mirror the Isomap walkthrough)                     #
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
        panel = pca_panel(active_index).to_corner(LEFT + UP, buff=0.15)
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
    # Step 1: raw cloud                                                   #
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
        self.set_caption(DATASET_INTRO.get(DATASET, "A curved cloud in 3D. PCA looks for the directions of greatest variance."))
        self.begin_ambient_camera_rotation(rate=2 * np.pi / 14.0, about="theta")
        self.wait(3.0)

    # ------------------------------------------------------------------ #
    # Step 2: center                                                      #
    # ------------------------------------------------------------------ #

    def section_center(self):
        self.next_section("step-2-center", type=_SEC)
        self.set_pseudo(1)
        # The loaded data is already centred; mark the centroid at the origin.
        centroid = Dot(point=[0, 0, 0], radius=0.12, color=S.INK)
        self.play(FadeIn(centroid, run_time=S.T_FAST))
        f = B.formula(r"x_i \leftarrow x_i - \bar{x}").scale(0.9)
        self.add_fixed_in_frame_mobjects(f)
        f.to_corner(RIGHT + UP, buff=0.4).set_opacity(0)
        self.play(f.animate.set_opacity(1.0), run_time=S.T_FAST)
        self.set_caption("Center the data: subtract the mean so the cloud sits at the origin.")
        self.wait(S.T_HOLD + 1.8)
        self.play(FadeOut(centroid), FadeOut(f), run_time=S.T_FAST)
        self.centroid = centroid

    # ------------------------------------------------------------------ #
    # Step 3: covariance matrix                                           #
    # ------------------------------------------------------------------ #

    def section_cov(self):
        self.next_section("step-3-cov", type=_SEC)
        self.set_pseudo(2)
        self.set_caption("Form the 3 by 3 covariance matrix of the centered coordinates.")
        cov = B.matrix_grid(np.round(self.d["C"], 2).tolist(), highlight_negative=True)
        cov.scale(0.5).to_corner(RIGHT + UP, buff=0.5)
        cov.set_opacity(0)
        self.add_fixed_in_frame_mobjects(cov)
        f = B.formula(r"C = \tfrac{1}{N-1}\, X_c^{\top} X_c").scale(0.8)
        f.next_to(cov, DOWN, buff=0.35)
        self.add_fixed_in_frame_mobjects(f)
        f.set_opacity(0)
        self.play(cov.animate.set_opacity(1.0), run_time=S.T_NORMAL)
        self.play(f.animate.set_opacity(1.0), run_time=S.T_FAST)
        self.wait(1.8)
        self.set_caption("Each entry measures how two coordinates vary together.")
        self.wait(S.T_HOLD + 2.0)
        self.cov, self.cov_formula = cov, f

    # ------------------------------------------------------------------ #
    # Step 4: eigendecomposition (principal axes)                         #
    # ------------------------------------------------------------------ #

    def section_eig(self):
        self.next_section("step-4-eig", type=_SEC)
        self.set_pseudo(3)
        self.play(FadeOut(self.cov), FadeOut(self.cov_formula), run_time=S.T_FAST)

        evals, evecs = self.d["evals"], self.d["evecs"]
        emax = float(np.max(evals)) or 1.0
        axes_arrows = VGroup()
        for k in range(3):
            length = 3.4 * np.sqrt(max(evals[k], 0.0) / emax)
            v = evecs[:, k] * length
            arrow = Line3D(start=[0, 0, 0], end=list(v), thickness=0.03, color=AXIS_COLORS[k])
            axes_arrows.add(arrow)
        self.set_caption("Eigenvectors of C are the principal axes, ordered by variance.")
        for arrow in axes_arrows:
            self.play(Create(arrow), run_time=S.T_FAST)

        f = B.formula(r"C = V\,\Lambda\,V^{\top}").scale(0.85)
        self.add_fixed_in_frame_mobjects(f)
        f.to_corner(RIGHT + UP, buff=0.4).set_opacity(0)
        self.play(f.animate.set_opacity(1.0), run_time=S.T_FAST)
        vals = B.formula(
            rf"\lambda_1={evals[0]:.2f}\quad \lambda_2={evals[1]:.2f}\quad \lambda_3={evals[2]:.2f}"
        ).scale(0.55)
        if vals.width > 5.6:
            vals.scale_to_fit_width(5.6)
        self.add_fixed_in_frame_mobjects(vals)
        vals.next_to(f, DOWN, buff=0.3, aligned_edge=RIGHT).set_opacity(0)
        self.play(vals.animate.set_opacity(1.0), run_time=S.T_FAST)
        self.wait(1.2)
        self.set_caption("The longest axis carries the most variance; the shortest, the least.")
        self.wait(S.T_HOLD + 2.0)
        self.eig_arrows, self.eig_f, self.eig_vals = axes_arrows, f, vals

    # ------------------------------------------------------------------ #
    # Step 5: project onto the top two axes (collapse the third)          #
    # ------------------------------------------------------------------ #

    def section_project(self):
        self.next_section("step-5-project", type=_SEC)
        self.set_pseudo(4)
        self.stop_ambient_camera_rotation()

        # The projection coordinates Y = X_c[v1 v2], scaled to fill the frame and
        # laid in the world xy-plane (z = 0). Reorienting the cloud into this plane
        # while flattening the camera to top-down (phi = 0, a reliable face-on
        # view) shows the two principal components head-on, not foreshortened.
        Y = self.d["Y"]
        s = 3.4 / (float(np.abs(Y).max()) or 1.0)
        target = np.column_stack([Y[:, 0] * s, Y[:, 1] * s, np.zeros(Y.shape[0])])

        self.play(
            FadeOut(self.eig_arrows[2]), FadeOut(self.eig_f), FadeOut(self.eig_vals),
            run_time=S.T_FAST,
        )
        f = B.formula(r"Y = X_c\,[\, v_1 \;\; v_2 \,]").scale(0.85)
        self.add_fixed_in_frame_mobjects(f)
        f.to_corner(RIGHT + UP, buff=0.4).set_opacity(0)
        self.play(f.animate.set_opacity(1.0), run_time=S.T_FAST)

        # New axis arrows in-plane: v1 along x, v2 along y.
        L1 = float(np.abs(Y[:, 0] * s).max())
        L2 = float(np.abs(Y[:, 1] * s).max())
        new_a1 = Line3D(start=[0, 0, 0], end=[L1, 0, 0], thickness=0.03, color=AXIS_COLORS[0])
        new_a2 = Line3D(start=[0, 0, 0], end=[0, L2, 0], thickness=0.03, color=AXIS_COLORS[1])

        # Tilt to a top-down, head-on view by the shortest path. The ambient
        # rotation has advanced theta by a couple of turns, so target the -90
        # degree orientation nearest the current theta rather than unwinding all
        # the way around (which looked like excessive spinning).
        cur_theta = float(self.camera.get_theta())
        tgt_theta = -90 * DEGREES
        tgt_theta += round((cur_theta - tgt_theta) / (2 * np.pi)) * 2 * np.pi
        self.set_caption("Turn so the two principal components face us head-on, then project onto them.")
        self.move_camera(
            phi=0, theta=tgt_theta, run_time=S.T_SLOW,
            added_anims=[
                *[self.cloud[i].animate.move_to(target[i]) for i in range(len(self.cloud.submobjects))],
                Transform(self.eig_arrows[0], new_a1),
                Transform(self.eig_arrows[1], new_a2),
            ],
        )
        self.wait(S.T_HOLD + 1.5)
        self.project_f = f

    # ------------------------------------------------------------------ #
    # Step 6: 2D embedding (PCA flattens, does not unroll)                #
    # ------------------------------------------------------------------ #

    def section_embed(self):
        self.next_section("step-6-embedding", type=_SEC)
        # Already face-on from the projection step: the cloud in the v1-v2 plane is
        # the 2D embedding. Clear the axes and the two principal-axis arrows and
        # keep the cloud; the Y formula stays on screen.
        self.play(
            FadeOut(self.axes),
            FadeOut(self.eig_arrows[0]), FadeOut(self.eig_arrows[1]),
            run_time=S.T_FAST,
        )
        self.set_caption("PCA flattens by projection, so the rolled layers overlap. A linear method cannot unroll the swiss roll.")
        self.wait(5.0)
