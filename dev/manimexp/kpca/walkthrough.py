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

# Per-kernel, per-dataset outro captions for the embedding step. Kept to one
# short line and honest about what each kernel's embedding actually does to the
# data (see manimexp/audit_embeddings.py). The detailed explanation lives in the
# page's subsidiary step text (js/manifold_isomap.js).
KERNEL_OUTRO = {
    "rbf": {
        "swiss_roll": "The RBF kernel spreads the points onto a bounded shell; the rolled sheet is not unrolled.",
        "s_curve": "The RBF kernel maps the bent sheet to a bounded blob; the coloring survives but it is not unrolled.",
        "twin_peaks": "The RBF kernel maps the surface to a bounded blob; its structure is not recovered.",
        "saddle": "The RBF kernel maps the saddle to a bounded blob; its structure is not recovered.",
        "cylinder": "The RBF kernel maps the band to a ring; the gradient survives but the tube is not unwrapped.",
        "severed_sphere": "The RBF kernel maps the cap to a bounded blob.",
        "helix": "The RBF kernel maps the coil to a pair of loops; it is not unrolled.",
        "trefoil_knot": "The RBF kernel maps the knot to a ring; the strands are not separated.",
        "toroidal_helix": "The RBF kernel maps the coil to a rosette; it is not unrolled.",
        "spiral_disk": "The RBF kernel collapses the spiral toward a thin arc.",
        "full_sphere": "The RBF kernel maps the sphere to a bounded blob; a sphere has no flat map.",
        "hilbert": "The RBF kernel scatters the folded curve across a bounded disk without meaningful order.",
        "clusters_3d": "The RBF kernel separates the clusters but spreads each onto a bounded arc.",
    },
    "polynomial": {
        "swiss_roll": "The polynomial kernel warps the sheet into a spiky fan; the layers are not separated.",
        "s_curve": "The polynomial kernel warps the bent sheet into a spiky star; its structure is distorted.",
        "twin_peaks": "The polynomial kernel warps the surface into a spiky star.",
        "saddle": "The polynomial kernel warps the saddle into spiky arms.",
        "cylinder": "The polynomial kernel warps the band into a spiked ring.",
        "severed_sphere": "The polynomial kernel maps the cap to a blob with little structure.",
        "helix": "The polynomial kernel folds the coil into overlapping arcs.",
        "trefoil_knot": "The polynomial kernel maps the knot to crossed loops.",
        "toroidal_helix": "The polynomial kernel warps the coil into a pinwheel.",
        "spiral_disk": "The polynomial kernel warps the spiral into a small spiked cluster.",
        "full_sphere": "The polynomial kernel maps the sphere to a blob; a sphere has no flat map.",
        "hilbert": "The polynomial kernel scatters the folded curve into spikes without meaningful order.",
        "clusters_3d": "The polynomial kernel splays the clusters into a spiky star.",
    },
    "linear": {
        "swiss_roll": "The linear kernel reduces to PCA, projecting the roll flat so its layers overlap.",
        "s_curve": "The linear kernel reduces to PCA, projecting the bent sheet flat and folding its ends over the middle.",
        "twin_peaks": "The linear kernel reduces to PCA, keeping the two widest directions; the bumps collapse.",
        "saddle": "The linear kernel reduces to PCA; the saddle's curvature collapses into the plane.",
        "cylinder": "The linear kernel reduces to PCA, projecting the tube flat so front and back overlap.",
        "severed_sphere": "The linear kernel reduces to PCA, projecting the curved cap flat.",
        "helix": "The linear kernel reduces to PCA, projecting the coil flat so its turns stack.",
        "trefoil_knot": "The linear kernel reduces to PCA, projecting the knot flat so its strands cross.",
        "toroidal_helix": "The linear kernel reduces to PCA, projecting the coil flat so its layers overlap.",
        "spiral_disk": "The spiral is nearly flat already, so linear kernel PCA recovers it with little distortion.",
        "full_sphere": "The linear kernel reduces to PCA, projecting the sphere to a disk with opposite sides overlapping.",
        "hilbert": "The linear kernel reduces to PCA; the folded curve overlaps heavily when flattened.",
        "clusters_3d": "The linear kernel reduces to PCA, keeping the directions that best separate the clusters.",
    },
}

# Per-kernel fallback when a dataset has no bespoke line.
KERNEL_OUTRO_DEFAULT = {
    "rbf": "The RBF kernel maps the data onto a bounded shell; the coloring survives but the shape is not recovered.",
    "polynomial": "The polynomial kernel warps the data into a spiky star; the structure is distorted.",
    "linear": "The linear kernel reduces to PCA, projecting the data to its 2D shadow.",
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

        # Order points along the manifold parameter t. The pipeline is
        # permutation-invariant, but ordering makes the kernel heatmaps show real
        # structure: without it, block-mean pooling averages random subsets of
        # points and the matrix paints a near-uniform wash.
        order = np.argsort(t)
        pts, t = pts[order], t[order]

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
        # The formula is wider than the heatmap, so right-align it to the heatmap
        # (which hugs the right margin) and let it extend left into open space,
        # rather than centering it where it would spill off the right edge.
        f.align_to(hm_K, RIGHT)
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

        # Centering is pure matrix algebra, so clear the 3D dataset and show the
        # lineage as a left-to-right chain (shared helper): K glides from its
        # corner to the focus slot, then K_c fades in to its right with an
        # H(.)H transform arrow between them. Captions stay one short line; the
        # detail lives in the page's subsidiary step text.
        self.lineage = B.MatrixLineage(self)
        self.lineage.start(
            self.hm_K, "K",
            caption="Center the kernel matrix K.",
            extra_anims=[
                FadeOut(self.cloud), FadeOut(self.axes),
                FadeOut(self.kernel_f), FadeOut(self.lbl_K),
            ],
        )
        self.remove(self.lbl_K)
        self.wait(S.T_HOLD + 0.5)

        hm_Kc = B.heatmap(self.d["Kc"], N, diverging=True)
        self.lineage.push(
            hm_Kc, "K_c", r"H(\,\cdot\,)H",
            caption="Subtract the feature-space mean to get Kc.",
        )
        self.wait(S.T_HOLD + 2.5)

        self.set_caption("Kc is the centered kernel, ready to eigendecompose.")
        self.wait(2.5)

    # ------------------------------------------------------------------ #
    # Step 5: eigendecompose Kc, top-2 readout                           #
    # ------------------------------------------------------------------ #

    def section_eig(self):
        self.next_section("step-5-eig", type=_SEC)
        self.set_pseudo(3)

        # Foreground the eigendecomposition (shared helper): Kc slides left as the
        # source, then the factorization, the eigenvalue spectrum, and the top-2
        # eigenvectors (columns of V) take the center with the dataset hidden.
        self.eig_overlays = self.lineage.eig_focus(
            r"K_c = V\,\Lambda\,V^{\top}",
            self.d["top6_vals"],
            self.d["vecs"],
            caption="Eigendecompose Kc into eigenvectors and eigenvalues.",
            caption_vectors="v1, v2 are the top eigenvectors.",
        )
        self.wait(4.0)

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

        # Fade the eigendecomposition overlays and restore the 3D dataset and
        # axes (hidden since the centering step), then bring in the embedding
        # formula. The cloud reappears at the camera's current orbit angle.
        self.play(
            FadeOut(self.eig_overlays),
            FadeIn(self.cloud),
            FadeIn(self.axes),
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

        outro = (KERNEL_OUTRO.get(KERNEL, {}).get(DATASET)
                 or KERNEL_OUTRO_DEFAULT.get(KERNEL, _OUTRO_DEFAULT))
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
