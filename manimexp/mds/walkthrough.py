"""MDS walkthrough: the steps of classical MDS animated with manim.

Mirrors the JS presentSubSteps for mds.js (0,3,4,5,6) so the clip chapter
markers line up with the interactive walkthrough on the web page.

Reuses the Isomap/PCA shared visual system: style, builders, data loading.
The reference template is manimexp/pca/walkthrough.py; the overlay helpers
(set_caption, set_pseudo), corner formula placement, and the continuous
ambient camera rotation that only flattens for the embedding step are copied
exactly from that file.

Manim 0.18.1 API notes (from isomap/walkthrough.py header):
- Section type: DefaultSectionType.NORMAL (not Section.NORMAL).
- Line3D uses thickness (scene units), not stroke_width.
- add_fixed_in_frame_mobjects() must be called before self.play().
- Point cloud and graph edges use 2D Dot/Line VMobjects for render speed.

Env: MFI_N (point count, default 1000), MFI_DATASET (default swiss_roll).
"""
import os
import numpy as np
from manim import (
    ThreeDScene, ThreeDAxes, DEGREES, FadeIn, FadeOut, Create, Transform,
    ReplacementTransform, AnimationGroup, DOWN, UP, RIGHT, LEFT,
    VGroup, Dot, Line3D, Text, MathTex,
)
from manim.scene.section import DefaultSectionType
from manimexp.isomap import style as S
from manimexp.isomap import builders as B
from manimexp.isomap.data import (
    load_dataset_points,
    knn_graph,
    double_center_squared,
    top2_eig,
)

N = int(os.environ.get("MFI_N", "1000"))
SEED = 0
DATASET = os.environ.get("MFI_DATASET", "swiss_roll")
_SEC = DefaultSectionType.NORMAL

# How many sample pairs to draw as connecting lines in the distances step.
N_SAMPLE_PAIRS = 5

_INTRO_TAIL = " MDS looks for a 2D layout that preserves those pairwise distances."
DATASET_INTRO = {
    "swiss_roll": "A 2D sheet rolled up in 3D." + _INTRO_TAIL,
    "s_curve": "An S-curve sheet bent in 3D." + _INTRO_TAIL,
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
    "swiss_roll": (
        "MDS preserves straight-line Euclidean distances, so the rolled layers"
        " still overlap in the embedding. Isomap fixes this by replacing"
        " straight-line distances with geodesic distances along the surface."
    ),
    "s_curve": (
        "The bent sheet folds back on itself because MDS measures straight-line"
        " distances through 3D space. Geodesic distances would separate the two sides."
    ),
    "twin_peaks": (
        "MDS collapses the height variation into the plane, keeping the"
        " widest spread. The bumps are lost."
    ),
    "saddle": (
        "MDS keeps the two widest directions; the saddle curvature collapses flat."
    ),
    "cylinder": (
        "MDS projects the tube flat, overlapping front and back."
        " A geodesic method could unroll it."
    ),
    "severed_sphere": (
        "MDS projects the curved cap flat, overlapping near and far points."
    ),
    "helix": (
        "MDS projects the coil flat, stacking its turns."
    ),
    "trefoil_knot": (
        "MDS projects the knot flat; its strands cross in the embedding."
    ),
    "toroidal_helix": (
        "MDS projects the coil flat, overlapping its layers."
    ),
    "spiral_disk": (
        "The spiral is nearly flat, so MDS recovers it with modest distortion."
    ),
    "full_sphere": (
        "A sphere has no isometric flat map, so MDS distorts it."
    ),
    "hilbert": (
        "MDS finds the widest plane through the folded curve, but the folds overlap."
    ),
    "clusters_3d": (
        "MDS keeps the directions that best separate the clusters,"
        " giving a clean 2D separation."
    ),
}
_OUTRO_DEFAULT = (
    "MDS preserves straight-line Euclidean distances. Isomap improves on this"
    " by measuring geodesic distances along the manifold surface."
)

MDS_PSEUDO = [
    r"0:\ \text{input: points } X \in \mathbb{R}^{N \times 3}",
    r"1:\ D_{ij} = \| x_i - x_j \|",
    r"2:\ B = -\tfrac{1}{2}\, H D^2 H,\quad H = I - \tfrac{1}{N}\mathbf{1}\mathbf{1}^\top",
    r"3:\ B = V\,\Lambda\,V^{\top}\quad (\text{top 2 eigenpairs})",
    r"4:\ Y = [\,v_1\ v_2\,]\,\mathrm{diag}(\sqrt{\lambda_1},\,\sqrt{\lambda_2})",
]


def mds_panel(active_index):
    items = []
    for idx, tex in enumerate(MDS_PSEUDO):
        color = S.ACCENT if idx == active_index else S.MUTED
        t = MathTex(tex, font_size=24, color=color)
        t.set_opacity(1.0 if idx == active_index else 0.35)
        items.append(t)
    group = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    max_w = 6.2
    if group.width > max_w:
        group.scale(max_w / group.width)
    return group


class MDSWalkthrough(ThreeDScene):

    def construct(self):
        self.camera.background_color = S.BG

        d = load_dataset_points(DATASET, N, seed=SEED)
        pts, t = d["points"], d["t"]

        # Pairwise Euclidean distance matrix.
        diff = pts[:, None, :] - pts[None, :, :]
        D = np.sqrt((diff ** 2).sum(axis=2))

        # Double-center the squared distances to get the Gram matrix B.
        Bmat = double_center_squared(D)

        # Top-2 eigenpairs of B.
        eigvals, eigvecs = top2_eig(Bmat)

        # Classical MDS embedding: Y_i = (sqrt(l1)*v1_i, sqrt(l2)*v2_i).
        Y = eigvecs * np.sqrt(np.maximum(eigvals, 0.0))

        # Spread-out sample pairs for the distance-lines beat (step 3).
        # Pick pairs spread across the roll so the lines are visible and varied.
        rng = np.random.default_rng(42)
        perm = rng.permutation(N)
        # Take pairs from thirds of the permutation so they span the cloud.
        third = N // 3
        pairs = []
        for k in range(N_SAMPLE_PAIRS):
            a = perm[k * (third // N_SAMPLE_PAIRS)]
            b = perm[third + k * (third // N_SAMPLE_PAIRS)]
            pairs.append((int(a), int(b)))

        self.d = dict(pts=pts, t=t, D=D, Bmat=Bmat, eigvals=eigvals,
                      eigvecs=eigvecs, Y=Y, pairs=pairs)
        self.cap = None
        self.pseudo = None

        self.section_raw()
        self.section_distances()
        self.section_double_center()
        self.section_eig()
        self.section_embed()

    # ------------------------------------------------------------------ #
    # Overlay helpers (mirror pca/walkthrough.py exactly)                 #
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
        panel = mds_panel(active_index).to_corner(LEFT + UP, buff=0.15)
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
    # Step 1 (id step-1-raw): raw cloud                                   #
    # ------------------------------------------------------------------ #

    def section_raw(self):
        self.next_section("step-1-raw", type=_SEC)
        self.set_camera_orientation(phi=65 * DEGREES, theta=30 * DEGREES, zoom=0.9)
        self.axes = ThreeDAxes(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1], z_range=[-4, 4, 1],
            x_length=8, y_length=8, z_length=8,
            axis_config={
                "stroke_color": S.MUTED,
                "stroke_width": 0.8,
                "stroke_opacity": 0.45,
            },
        )
        self.play(FadeIn(self.axes, run_time=S.T_FAST))
        self.cloud = B.point_cloud(self.d["pts"], self.d["t"])
        self.play(FadeIn(self.cloud, run_time=S.T_INTRO))
        self.set_pseudo(0)
        intro = DATASET_INTRO.get(
            DATASET,
            "A curved cloud in 3D. MDS looks for a 2D layout that preserves"
            " pairwise distances.",
        )
        self.set_caption(intro)
        self.begin_ambient_camera_rotation(rate=2 * np.pi / 14.0, about="theta")
        self.wait(3.0)

    # ------------------------------------------------------------------ #
    # Step 3 (id step-3-distances): pairwise distances + heatmap of D     #
    # ------------------------------------------------------------------ #

    def section_distances(self):
        self.next_section("step-3-distances", type=_SEC)
        self.set_pseudo(1)
        self.set_caption(
            "Every pair of points has a straight-line distance. Collect all"
            " N times N of them into the distance matrix D."
        )

        # Draw a handful of sample connecting lines in 3D.
        pts = self.d["pts"]
        pairs = self.d["pairs"]
        lines = VGroup()
        for (a, b) in pairs:
            seg = B.straight_line(pts[a], pts[b], color=S.ACCENT)
            lines.add(seg)
        for seg in lines:
            self.play(Create(seg), run_time=0.35)

        # Formula top-right.
        f = B.formula(r"D_{ij} = \| x_i - x_j \|").scale(0.85)
        self.add_fixed_in_frame_mobjects(f)
        f.to_corner(RIGHT + UP, buff=0.4).set_opacity(0)
        self.play(f.animate.set_opacity(1.0), run_time=S.T_FAST)

        # Heatmap of D as a fixed-in-frame corner panel.
        D = self.d["D"]
        hm_D = B.heatmap(D, N, max_cells=32, diverging=False)
        # Scale to fit a corner panel (~2.5 scene units wide) and place bottom-right.
        panel_w = 2.5
        if hm_D.width > 0:
            hm_D.scale(panel_w / hm_D.width)
        hm_D.to_corner(RIGHT + UP, buff=0.4)
        hm_D.shift(DOWN * (f.height + 0.35))
        self.add_fixed_in_frame_mobjects(hm_D)
        hm_D.set_opacity(0)
        self.play(hm_D.animate.set_opacity(1.0), run_time=S.T_NORMAL)

        self.set_caption(
            "The color in row i, column j encodes how far point i is from"
            " point j. Bright entries are large distances; dark entries are small."
        )
        self.wait(S.T_HOLD + 2.0)

        self.dist_lines = lines
        self.dist_f = f
        self.hm_D = hm_D

    # ------------------------------------------------------------------ #
    # Step 4 (id step-4-double-center): D^2 -> B via double-centering     #
    # ------------------------------------------------------------------ #

    def section_double_center(self):
        self.next_section("step-4-double-center", type=_SEC)
        self.set_pseudo(2)

        # Replace the D heatmap with a D^2 heatmap, then morph it into B.
        D = self.d["D"]
        Bmat = self.d["Bmat"]

        hm_D2 = B.heatmap(D ** 2, N, max_cells=32, diverging=False)
        panel_w = 2.5
        if hm_D2.width > 0:
            hm_D2.scale(panel_w / hm_D2.width)
        hm_D2.to_corner(RIGHT + UP, buff=0.4)
        hm_D2.shift(DOWN * (self.dist_f.height + 0.35))
        self.add_fixed_in_frame_mobjects(hm_D2)
        hm_D2.set_opacity(0)

        self.set_caption(
            "Square every entry to get D squared. Then double-center: subtract"
            " each row mean and column mean and add back the grand mean."
        )
        # Swap D heatmap for D^2 heatmap.
        self.play(
            FadeOut(self.hm_D),
            FadeOut(self.dist_lines),
            hm_D2.animate.set_opacity(1.0),
            run_time=S.T_NORMAL,
        )
        self.remove(self.hm_D)

        # Update the formula to the double-centering formula.
        f_dc = B.formula(
            r"B = -\tfrac{1}{2}\, H D^2 H,\quad H = I - \tfrac{1}{N}\mathbf{1}\mathbf{1}^\top"
        ).scale(0.7)
        self.add_fixed_in_frame_mobjects(f_dc)
        f_dc.to_corner(RIGHT + UP, buff=0.4).set_opacity(0)
        self.play(
            FadeOut(self.dist_f),
            f_dc.animate.set_opacity(1.0),
            run_time=S.T_FAST,
        )
        self.remove(self.dist_f)

        self.wait(S.T_HOLD)
        self.set_caption(
            "The centering matrix H removes the translation component."
            " The result B is a Gram matrix: its entries are inner products"
            " relative to the centroid, which is exactly what eigendecomposition needs."
        )

        # Morph D^2 heatmap into B heatmap (diverging because B has negatives).
        hm_B = B.heatmap(Bmat, N, max_cells=32, diverging=True)
        if hm_B.width > 0:
            hm_B.scale(panel_w / hm_B.width)
        hm_B.to_corner(RIGHT + UP, buff=0.4)
        hm_B.shift(DOWN * (f_dc.height + 0.35))
        self.add_fixed_in_frame_mobjects(hm_B)
        hm_B.set_opacity(0)
        self.play(
            FadeOut(hm_D2),
            hm_B.animate.set_opacity(1.0),
            run_time=S.T_NORMAL,
        )
        self.remove(hm_D2)

        self.set_caption(
            "Warm cells are negative inner products; cool cells are positive."
            " The diagonal is zero because each point has zero inner product with itself."
        )
        self.wait(S.T_HOLD + 2.0)

        self.dc_f = f_dc
        self.hm_B = hm_B

    # ------------------------------------------------------------------ #
    # Step 5 (id step-5-eig): eigendecompose B, show top-2 eigenvalues    #
    # ------------------------------------------------------------------ #

    def section_eig(self):
        self.next_section("step-5-eig", type=_SEC)
        self.set_pseudo(3)

        eigvals = self.d["eigvals"]
        self.play(
            FadeOut(self.hm_B),
            FadeOut(self.dc_f),
            run_time=S.T_FAST,
        )

        # Eigendecomposition formula top-right.
        f_eig = B.formula(r"B = V\,\Lambda\,V^{\top}").scale(0.85)
        self.add_fixed_in_frame_mobjects(f_eig)
        f_eig.to_corner(RIGHT + UP, buff=0.4).set_opacity(0)
        self.play(f_eig.animate.set_opacity(1.0), run_time=S.T_FAST)

        self.set_caption(
            "Eigendecompose the Gram matrix B. The top eigenvectors carry the"
            " directions of largest spread in inner-product space."
        )
        self.wait(S.T_HOLD)

        # Eigenvalue readout below the formula.
        lam1, lam2 = float(eigvals[0]), float(eigvals[1])
        vals_tex = (
            rf"\lambda_1 = {lam1:.2f}\quad \lambda_2 = {lam2:.2f}"
        )
        vals_mob = B.formula(vals_tex).scale(0.6)
        if vals_mob.width > 5.6:
            vals_mob.scale_to_fit_width(5.6)
        self.add_fixed_in_frame_mobjects(vals_mob)
        vals_mob.next_to(f_eig, DOWN, buff=0.3, aligned_edge=RIGHT).set_opacity(0)
        self.play(vals_mob.animate.set_opacity(1.0), run_time=S.T_FAST)

        self.set_caption(
            "The first eigenvalue is much larger than the second, reflecting the"
            " dominant direction of spread. Both are positive because B is"
            " positive semi-definite when the data comes from a Euclidean space."
        )
        self.wait(S.T_HOLD + 2.0)

        self.eig_f = f_eig
        self.eig_vals_mob = vals_mob

    # ------------------------------------------------------------------ #
    # Step 6 (id step-6-embedding): form Y, morph cloud to 2D face-on     #
    # ------------------------------------------------------------------ #

    def section_embed(self):
        self.next_section("step-6-embedding", type=_SEC)
        self.set_pseudo(4)
        self.stop_ambient_camera_rotation()

        Y = self.d["Y"]
        # Scale to fill the frame (~3.4 units half-width).
        s = 3.4 / (float(np.abs(Y).max()) or 1.0)
        target = np.column_stack([
            Y[:, 0] * s,
            Y[:, 1] * s,
            np.zeros(Y.shape[0]),
        ])

        self.play(
            FadeOut(self.eig_f),
            FadeOut(self.eig_vals_mob),
            run_time=S.T_FAST,
        )

        # Embedding formula top-right.
        f_embed = B.formula(
            r"Y = [\,v_1\ v_2\,]\,\mathrm{diag}(\sqrt{\lambda_1},\,\sqrt{\lambda_2})"
        ).scale(0.75)
        self.add_fixed_in_frame_mobjects(f_embed)
        f_embed.to_corner(RIGHT + UP, buff=0.4).set_opacity(0)
        self.play(f_embed.animate.set_opacity(1.0), run_time=S.T_FAST)

        # Find the shortest-path theta to face-on (phi=0).
        # The ambient rotation may have advanced theta by several turns, so
        # snap to the nearest equivalent of -90 degrees rather than unwinding all
        # the way around (mirrors the PCA file's theta-unwinding logic).
        cur_theta = float(self.camera.get_theta())
        tgt_theta = -90 * DEGREES
        tgt_theta += round((cur_theta - tgt_theta) / (2 * np.pi)) * 2 * np.pi

        self.set_caption(
            "Scale the eigenvectors by the square roots of their eigenvalues"
            " to get the 2D embedding. Each point's coordinates now approximate"
            " the original pairwise distances."
        )
        self.move_camera(
            phi=0,
            theta=tgt_theta,
            run_time=S.T_SLOW,
            added_anims=[
                self.cloud[i].animate.move_to(target[i])
                for i in range(len(self.cloud.submobjects))
            ],
        )
        self.wait(S.T_HOLD + 1.5)

        outro = DATASET_OUTRO.get(DATASET, _OUTRO_DEFAULT)
        self.set_caption(outro)
        self.wait(5.0)
