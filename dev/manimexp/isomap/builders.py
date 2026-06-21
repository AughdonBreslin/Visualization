"""Reusable manim mobject builders so every section looks consistent.

Manim 0.18.1 API notes (adaptations from the design spec):
- The spec calls for Dot3D (3D sphere surface) and Line3D (3D cylinder surface) for
  the point cloud and graph edges. Both are full 3D surface meshes and render
  extremely slowly at high N (1.3 s/frame for N=120 Dot3D, making N=1000 infeasible).
- Adaptation: point cloud and graph edges use 2D VMobjects (Dot and Line), which can
  be positioned anywhere in 3D space in a ThreeDScene and render at full 2D speed.
  Only the highlighted geodesic path and straight-line comparison use Line3D, where
  the 3D cylinder appearance adds visual value and the count is small (one path).
- Line3D uses `thickness` (scene units) not `stroke_width` (pixel units).
- Line stroke_width applies normally to 2D Line VMobjects.
- Table.get_entries((r, c)) uses 1-based (row, col) indices.
"""
import colorsys
import textwrap
import numpy as np
from manim import (VGroup, Dot, Dot3D, Line, Line3D, Square, Rectangle, Text,
                   MathTex, Table, Matrix, Arrow, FadeOut, ManimColor,
                   interpolate_color, config, DOWN, UP, LEFT, RIGHT, UL)
from . import style as S


def rainbow_color(u):
    """Map u in [0, 1] to a high-saturation rainbow color.

    The near end (u = 0) is blue/violet and the far end (u = 1) is red, sweeping
    through cyan, green, yellow, and orange. Full saturation and value make the
    points pop and read as a clear gradient.
    """
    u = float(min(1.0, max(0.0, u)))
    hue = 0.70 * (1.0 - u)
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return ManimColor("#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)))

# Thickness for Line3D (scene units, ~0.01-0.04 range)
PATH_THICKNESS = 0.022
STRAIGHT_THICKNESS = 0.014


def color_for_t(t, tmin, tmax):
    u = (t - tmin) / max(1e-9, (tmax - tmin))
    return rainbow_color(u)


def point_cloud(points, t):
    """VGroup of 2D Dot mobjects placed at 3D positions.

    Dot is a VMobject (fast to render) that can live anywhere in 3D space
    inside a ThreeDScene. DOT_RADIUS is in scene units.

    Each dot is colored from the start by a full-saturation rainbow (no sheen),
    so the points read as vivid, saturated samples throughout the video.
    """
    tmin, tmax = float(np.min(t)), float(np.max(t))
    dots = []
    for i in range(points.shape[0]):
        dot = Dot(
            point=points[i],
            radius=S.DOT_RADIUS,
            fill_opacity=1.0,
            color=color_for_t(t[i], tmin, tmax),
        )
        dots.append(dot)
    return VGroup(*dots)


def knn_sphere(point, radius, color):
    """A true 3D Dot3D sphere for the local kNN beat.

    Only used for the ~10 objects in the local neighborhood where the cost is
    acceptable and the spherical look is visually important at high zoom.
    """
    return Dot3D(point=point, radius=radius, color=color)


def graph_edges(points, edges, color=None, opacity=S.EDGE_OPACITY):
    """VGroup of 2D Line mobjects for the kNN graph.

    Using Line (VMobject, stroke-based) rather than Line3D (cylinder surface)
    gives the thin translucent look the spec requires and renders in milliseconds
    per frame vs seconds for Line3D.
    """
    color = color or S.ACCENT
    lines = VGroup(*[
        Line(
            start=points[i],
            end=points[j],
            stroke_width=S.EDGE_WIDTH,
            color=color,
        )
        for (i, j) in edges
    ])
    lines.set_opacity(opacity)
    return lines


def path_polyline(points, path, gradient=True):
    """VGroup of Line3D segments for the highlighted geodesic path.

    Using Line3D here (cylinder surface) because it is a single path with
    few segments and the thick 3D cylinder look makes it visually salient
    against the thin 2D edge lines.

    When gradient=True, each segment is colored by its fractional position
    along the path (from GOOD at the start to WARM at the end).
    """
    n_segs = len(path) - 1
    segs = []
    for a in range(n_segs):
        if gradient and n_segs > 1:
            frac = a / (n_segs - 1)
            seg_color = rainbow_color(frac)
        else:
            seg_color = S.GOOD
        segs.append(
            Line3D(
                start=points[path[a]],
                end=points[path[a + 1]],
                thickness=PATH_THICKNESS,
                color=seg_color,
            )
        )
    return VGroup(*segs)


def straight_line(start, end, color=None):
    """A single Line3D for the Euclidean straight-line comparison."""
    color = color or S.MUTED
    return Line3D(start=start, end=end, thickness=STRAIGHT_THICKNESS, color=color)


def recolor_cloud_by_values(cloud, values, c_lo=None, c_hi=None, cmap=None, grow=1.0):
    """Animate-recolor a VGroup of Dots by a per-point scalar array.

    Parameters
    ----------
    cloud : VGroup  -- the point cloud whose submobjects are Dot objects.
    values : array-like  -- per-point scalar values (length must match cloud).
    c_lo, c_hi : ManimColor  -- endpoints of a two-color gradient (used when no
        cmap is given). Defaults to ACCENT and WARM.
    cmap : callable(u) -> ManimColor  -- optional colormap mapping a normalized
        value u in [0, 1] to a color (for example rainbow_color). Overrides
        c_lo / c_hi when provided.
    grow : float  -- factor to scale each dot by while recoloring, so the points
        read more clearly (1.0 leaves the size unchanged).

    Returns
    -------
    list of .animate expressions suitable for self.play(*...).
    """
    vals = np.asarray(values, dtype=float)
    vmin, vmax = float(vals.min()), float(vals.max())
    span = max(1e-9, vmax - vmin)
    if cmap is None and (c_lo is None or c_hi is None):
        c_lo, c_hi = S.ACCENT, S.WARM
    anims = []
    for i, dot in enumerate(cloud.submobjects):
        u = (vals[i] - vmin) / span
        col = cmap(u) if cmap is not None else interpolate_color(c_lo, c_hi, u)
        # Drop the radial sheen highlight so the assigned hue reads at full
        # saturation rather than being washed toward white.
        dot.set_sheen(0.0)
        anim = dot.animate.set_color(col).set_opacity(1.0)
        if grow != 1.0:
            anim = anim.scale(grow)
        anims.append(anim)
    return anims


def caption(text):
    # Wrap long captions to keep them inside the frame, then hard-cap the width
    # so nothing overflows off-screen.
    wrapped = "\n".join(textwrap.wrap(text, width=42)) or text
    mob = Text(wrapped, font_size=S.CAPTION_SIZE, color=S.INK, line_spacing=0.6)
    max_w = config.frame_width - 1.2
    if mob.width > max_w:
        mob.scale_to_fit_width(max_w)
    return mob


def formula(tex):
    return MathTex(tex, font_size=S.FORMULA_SIZE, color=S.INK)


def matrix_grid(values, highlight_negative=True):
    rows = [[f"{v:.2f}" for v in row] for row in values]
    tbl = Table(rows, include_outer_lines=True).scale(0.5)
    if highlight_negative:
        for r, row in enumerate(values):
            for c, v in enumerate(row):
                if v < 0:
                    # get_entries uses 1-based (row, col) indices
                    tbl.get_entries((r + 1, c + 1)).set_color(S.WARM)
    return tbl


def heatmap(matrix, n, max_cells=32, cell=0.12, diverging=False, mode="mean"):
    """Render an n x n matrix as a downsampled grid of colored squares.

    The matrix is block-downsampled to at most max_cells x max_cells. Colors
    run on a sequential ramp (MUTED -> ACCENT) or, when diverging=True, WARM
    (negative) -> BG -> GOOD (positive). Returns a VGroup with .meta = {rows,
    cols, vmin, vmax}.

    Parameters
    ----------
    mode : "mean" (default) -- each cell shows the block mean; smooths sparse
           matrices into a wash.
           "pattern" -- each cell is fully lit (at the per-element max magnitude)
           if ANY entry in the block is non-zero, and dark (BG) if the whole
           block is zero. Use this to show sparsity structure when the block mean
           would wash out the zeros.
    """
    M = np.asarray(matrix, dtype=float).reshape(n, n)
    step = max(1, int(np.ceil(n / max_cells)))
    rows = int(np.ceil(n / step))
    cols = rows
    ds = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            block = M[r * step:(r + 1) * step, c * step:(c + 1) * step]
            if block.size == 0:
                ds[r, c] = 0.0
            elif mode == "pattern":
                # Mark nonzero if any entry in the block is nonzero; carry the
                # sign of the entry with largest absolute value so diverging
                # colors still distinguish positive from negative weights.
                abs_max_idx = int(np.argmax(np.abs(block)))
                ds[r, c] = float(block.flat[abs_max_idx]) if np.any(block != 0) else 0.0
            else:
                ds[r, c] = float(block.mean())
    # For pattern mode, rescale so nonzero cells paint at full intensity.
    if mode == "pattern":
        nz = ds[ds != 0]
        if nz.size > 0:
            scale = float(np.abs(nz).max()) or 1.0
            ds = ds / scale
        vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = float(ds.min()), float(ds.max())
    grid = VGroup()
    for r in range(rows):
        for c in range(cols):
            v = ds[r, c]
            if diverging:
                m = max(abs(vmin), abs(vmax)) or 1.0
                if v >= 0:
                    col = interpolate_color(S.BG, S.GOOD, min(1.0, v / m))
                else:
                    col = interpolate_color(S.BG, S.WARM, min(1.0, -v / m))
            else:
                rng = (vmax - vmin) or 1.0
                col = interpolate_color(S.MUTED, S.ACCENT, (v - vmin) / rng)
            sq = Square(side_length=cell, fill_color=col, fill_opacity=1.0,
                        stroke_width=0)
            sq.move_to([c * cell, -r * cell, 0])
            grid.add(sq)
    grid.move_to([0, 0, 0])
    grid.meta = {"rows": rows, "cols": cols, "vmin": vmin, "vmax": vmax}
    return grid


def vector_strip(vec, height=2.4, width=0.4, max_cells=48, diverging=True,
                 rainbow=False):
    """Render a 1-D vector as a vertical strip of colored cells (a column of V).

    Used to show an eigenvector: each cell is the block-mean of a run of entries.
    Color modes:
    - rainbow=True: map the value across the same rainbow ramp the 3D point cloud
      uses, so the strip visually ties back to the point colors.
    - diverging=True (default): WARM (negative) -> BG -> GOOD (positive), matching
      the matrix heatmaps.
    - otherwise: sequential MUTED -> ACCENT.
    When the points are ordered along the manifold, a smooth eigenvector reads as
    smooth color bands down the strip.
    """
    v = np.asarray(vec, dtype=float).ravel()
    n = v.size
    step = max(1, int(np.ceil(n / max_cells)))
    rows = int(np.ceil(n / step))
    ds = np.array([v[r * step:(r + 1) * step].mean() for r in range(rows)])
    vmin, vmax = float(ds.min()), float(ds.max())
    m = max(abs(vmin), abs(vmax)) or 1.0
    rng = (vmax - vmin) or 1.0
    cell_h = height / rows
    grid = VGroup()
    for r, val in enumerate(ds):
        if rainbow:
            col = rainbow_color((val - vmin) / rng)
        elif diverging:
            if val >= 0:
                col = interpolate_color(S.BG, S.GOOD, min(1.0, val / m))
            else:
                col = interpolate_color(S.BG, S.WARM, min(1.0, -val / m))
        else:
            col = interpolate_color(S.MUTED, S.ACCENT, (val - vmin) / rng)
        sq = Rectangle(width=width, height=cell_h, fill_color=col,
                       fill_opacity=1.0, stroke_width=0)
        sq.move_to([0, -r * cell_h, 0])
        grid.add(sq)
    grid.move_to([0, 0, 0])
    grid.meta = {"rows": rows, "vmin": vmin, "vmax": vmax}
    return grid


def pseudocode_panel(active_index):
    """Build a fixed-in-frame pseudocode panel for the top-left corner.

    Each line is rendered with MathTex so the mathematics typesets properly
    (subscripts, norms, the eigenvalue matrix Lambda, square roots). The active
    line is shown in ACCENT at full opacity; the others are dimmed to MUTED.

    Parameters
    ----------
    active_index : int (0-5), which line to highlight.

    Returns
    -------
    VGroup of MathTex mobjects, suitable for add_fixed_in_frame_mobjects.
    """
    lines = [
        r"0:\ \text{input: points } X,\ \text{neighbors } k",
        r"1:\ \text{link each } i \text{ to its } k \text{ nearest},\ "
        r"w_{ij}=\lVert x_i - x_j\rVert",
        r"2:\ D_{ij} = \text{shortest path } i \!\to\! j",
        r"3:\ B = -\tfrac{1}{2}\, J\, D^{2}\, J",
        r"4:\ B = V\,\Lambda\,V^{\top}\quad(\text{top eigenvectors})",
        r"5:\ Y = \left[\sqrt{\lambda_1}\,v_1,\ \sqrt{\lambda_2}\,v_2\right]",
    ]
    items = []
    for idx, tex in enumerate(lines):
        color = S.ACCENT if idx == active_index else S.MUTED
        t = MathTex(tex, font_size=24, color=color)
        t.set_opacity(1.0 if idx == active_index else 0.35)
        items.append(t)

    group = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    # Cap the panel width so the longest line never runs into the scene content.
    max_w = 6.2
    if group.width > max_w:
        group.scale(max_w / group.width)
    return group


def colvec(values, color=None):
    """A column-vector Matrix mobject from a 1-D array of numbers."""
    color = color or S.INK
    m = Matrix([[f"{x:.2f}"] for x in values], v_buff=0.7,
               bracket_h_buff=0.12, bracket_v_buff=0.12)
    m.set_color(color)
    return m


def rowvec(values, color=None):
    """A row-vector Matrix mobject from a 1-D array of numbers."""
    color = color or S.INK
    m = Matrix([[f"{x:.2f}" for x in values]], h_buff=1.9,
               bracket_h_buff=0.12, bracket_v_buff=0.12)
    m.set_color(color)
    return m


def eig_bar_chart(vals, highlight_idxs, trivial_idx=None, bar_w=0.28, bar_max_h=1.2):
    """Build a small vertical bar chart from eigenvalue magnitudes.

    Parameters
    ----------
    vals : sequence of floats  -- the eigenvalues to display (pass a slice,
        e.g. top-6 descending for MDS/KPCA, or bottom-6 ascending for Laplacian).
    highlight_idxs : list of int  -- bar indices (0-based) that should be
        colored ACCENT to draw attention (the informative components).
    trivial_idx : int or None  -- if given, that bar is greyed with lower
        opacity (the trivial near-zero eigenvalue for Laplacian/LLE).
    bar_w : float  -- width of each bar in scene units.
    bar_max_h : float  -- maximum bar height in scene units.

    Returns
    -------
    VGroup of Rectangle bars arranged left to right, aligned at the bottom edge.
    """
    vmax = float(max(abs(float(v)) for v in vals)) or 1.0
    group = VGroup()
    for i, v in enumerate(vals):
        h = max(0.04, bar_max_h * abs(float(v)) / vmax)
        if trivial_idx is not None and i == trivial_idx:
            color = S.MUTED
            opacity = 0.45
        elif i in highlight_idxs:
            color = S.ACCENT
            opacity = 0.92
        else:
            color = S.MUTED
            opacity = 0.92
        rect = Rectangle(
            width=bar_w, height=h,
            fill_color=color, fill_opacity=opacity,
            stroke_width=0,
        )
        group.add(rect)
    group.arrange(RIGHT, buff=0.10, aligned_edge=DOWN)
    return group


def fit_formula(tex, max_width=5.2, scale=0.8):
    """Build a formula, apply an initial scale, then clamp to max_width if needed.

    Use this for any formula that sits near a heatmap or in a tight corner, to
    prevent overflow off the right or bottom edge of the frame.

    Parameters
    ----------
    tex : str  -- LaTeX string for MathTex.
    max_width : float  -- maximum allowed width in scene units.
    scale : float  -- initial scale factor applied before the width clamp.

    Returns
    -------
    MathTex mobject, scaled and clamped.
    """
    mob = formula(tex)
    mob.scale(scale)
    if mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


class MatrixLineage:
    """A left-to-right chain of matrices for a transform pipeline.

    Used to show a derivation such as D -> D^2 -> B (MDS) or K -> K_c (Kernel
    PCA) as a flowing lineage: each new matrix is featured large at the focus
    slot on the right; when the next step begins, the current matrix shrinks and
    slides left, joined by an arrow carrying a compact transform operator. By the
    end the whole chain reads left to right with the latest step largest.

    All on-screen text stays short by design: the operator on each arrow is a
    compact symbol (for example (\\cdot)^2), and the per-step caption is a single
    short line. Detailed derivations belong in the page's subsidiary step text,
    not on the frame.

    The heatmaps are added as fixed-in-frame mobjects, so the scene's camera may
    keep orbiting underneath without disturbing the lineage.

    Parameters
    ----------
    scene : a ThreeDScene exposing add_fixed_in_frame_mobjects, play, and an
        optional set_caption(text) helper (used when a caption is passed).
    focus_h, park_h : focused and parked matrix heights in scene units.
    band_y : vertical center of the chain.
    arrow_w, buff : arrow slot width and the gap on each side of an arrow.
    """

    def __init__(self, scene, focus_h=2.3, park_h=1.15, band_y=-0.1,
                 arrow_w=1.1, buff=0.2):
        self.scene = scene
        self.focus_h = focus_h
        self.park_h = park_h
        self.band_y = band_y
        self.arrow_w = arrow_w
        self.buff = buff
        self.cards = []    # list of dict(hm=, lbl=)
        self.arrows = []   # VGroup(arrow, operator) between consecutive cards

    # -- internal helpers -------------------------------------------------- #

    def _caption(self, text):
        if text is not None and hasattr(self.scene, "set_caption"):
            self.scene.set_caption(text)

    def _make_arrow(self, op_tex):
        arr = Arrow(LEFT * 0.45, RIGHT * 0.45, buff=0.0, stroke_width=3.0,
                    color=S.MUTED, max_tip_length_to_length_ratio=0.35)
        op = MathTex(op_tex, font_size=22, color=S.MUTED)
        op.next_to(arr, UP, buff=0.10)
        return VGroup(arr, op)

    def _layout(self):
        # Left-to-right centers; last card is focus height, earlier ones parked.
        n = len(self.cards)
        heights = [self.park_h] * (n - 1) + [self.focus_h]
        total = sum(heights) + (n - 1) * (self.arrow_w + 2 * self.buff)
        x = -total / 2.0
        card_x, arrow_x = [], []
        for i in range(n):
            card_x.append(x + heights[i] / 2.0)
            x += heights[i]
            if i < n - 1:
                x += self.buff
                arrow_x.append(x + self.arrow_w / 2.0)
                x += self.arrow_w + self.buff
        return heights, card_x, arrow_x

    def _relayout_anims(self, new_index):
        # Animations to move every existing card/arrow to its target slot. The
        # card at new_index is the focus (snapped into place by the caller).
        heights, card_x, arrow_x = self._layout()
        n = len(self.cards)
        anims = []
        for i, card in enumerate(self.cards):
            if i == new_index:
                continue
            hm, lbl = card["hm"], card["lbl"]
            th = heights[i]
            cpos = np.array([card_x[i], self.band_y, 0.0])
            lpos = cpos + np.array([0.0, th / 2.0 + 0.20, 0.0])
            anims += [
                hm.animate.scale_to_fit_height(th).move_to(cpos).set_opacity(0.5),
                lbl.animate.scale_to_fit_height(0.22).move_to(lpos).set_opacity(0.5),
            ]
        for j, ar in enumerate(self.arrows):
            apos = np.array([arrow_x[j], self.band_y, 0.0])
            if j == len(self.arrows) - 1 and new_index == n - 1:
                ar.move_to(apos)                       # new arrow: snapped, faded in below
            else:
                anims.append(ar.animate.move_to(apos).set_opacity(0.6))
        return anims, heights, card_x, arrow_x

    # -- public API -------------------------------------------------------- #

    def start(self, hm, label_tex, caption=None, extra_anims=()):
        """Register the first matrix and animate it into the focus slot.

        hm is assumed already on screen (e.g. a corner heatmap from a prior
        step); it is grown to focus height and moved to center. extra_anims lets
        the caller fold in scene-specific fades (cloud, axes) into the same play.
        """
        lbl = MathTex(label_tex, font_size=30, color=S.INK)
        self.scene.add_fixed_in_frame_mobjects(lbl)
        self.cards.append({"hm": hm, "lbl": lbl})
        _, card_x, _ = self._layout()
        cpos = np.array([card_x[0], self.band_y, 0.0])
        lbl.scale_to_fit_height(0.34)
        lbl.move_to(cpos + np.array([0.0, self.focus_h / 2.0 + 0.20, 0.0]))
        lbl.set_opacity(0)
        self._caption(caption)
        self.scene.play(
            hm.animate.scale_to_fit_height(self.focus_h).move_to(cpos),
            lbl.animate.set_opacity(1.0),
            *extra_anims,
            run_time=getattr(S, "T_NORMAL", 1.2),
        )

    def push(self, hm, label_tex, op_tex, caption=None):
        """Add the next matrix: shrink/slide the chain left, fade the new one in."""
        lbl = MathTex(label_tex, font_size=30, color=S.INK)
        arrow = self._make_arrow(op_tex)
        self.scene.add_fixed_in_frame_mobjects(hm, lbl, arrow)
        hm.set_opacity(0.0)
        lbl.set_opacity(0.0)
        arrow.set_opacity(0.0)
        self.cards.append({"hm": hm, "lbl": lbl})
        self.arrows.append(arrow)

        new_index = len(self.cards) - 1
        anims, heights, card_x, _ = self._relayout_anims(new_index)
        self._caption(caption)

        # Snap the new focus card + its label into place, then fade them in.
        th = heights[new_index]
        cpos = np.array([card_x[new_index], self.band_y, 0.0])
        hm.scale_to_fit_height(th).move_to(cpos)
        lbl.scale_to_fit_height(0.34).move_to(cpos + np.array([0.0, th / 2.0 + 0.20, 0.0]))
        anims += [hm.animate.set_opacity(1.0), lbl.animate.set_opacity(1.0)]
        self.arrows[-1].set_opacity(0.0)
        anims.append(self.arrows[-1].animate.set_opacity(1.0))
        self.scene.play(*anims, run_time=getattr(S, "T_NORMAL", 1.2))

    def group(self):
        """A VGroup of every card and arrow, for a single fade-out at the end."""
        return VGroup(
            *[c["hm"] for c in self.cards],
            *[c["lbl"] for c in self.cards],
            *self.arrows,
        )

    def eig_focus(self, factor_tex, spectrum_vals, eigvecs_two,
                  caption=None, caption_vectors=None, rainbow_vectors=False,
                  highlight_idxs=(0, 1), trivial_idx=None):
        """Foreground the eigendecomposition of the last (focus) matrix.

        The focus matrix slides left as the source; the factorization, the
        eigenvalue spectrum, and the two highlighted eigenvectors (as vertical
        strips, columns of V) take the center, with the 3D dataset hidden.

        highlight_idxs picks which spectrum bars carry the kept eigenvalues
        (labeled lambda_1, lambda_2 in order); trivial_idx, when given, greys a
        skipped bar and labels it lambda_0. Variance methods keep the largest
        eigenvalues (highlight 0,1); locality methods (LLE, Laplacian) keep the
        smallest non-trivial ones (highlight 1,2 with trivial_idx 0). eigvecs_two
        is the N-by-2 array of the two eigenvectors to draw as strips.

        Captions are passed in and kept to one short line. Returns a VGroup of
        every overlay so the caller can clear it in one fade at the embedding step.
        """
        T_NORMAL = getattr(S, "T_NORMAL", 1.2)
        T_FAST = getattr(S, "T_FAST", 0.6)

        # Slide the focus matrix left as the source; drop the rest of the chain.
        src = self.cards[-1]
        drop = VGroup(
            *[c["hm"] for c in self.cards[:-1]],
            *[c["lbl"] for c in self.cards[:-1]],
            *self.arrows,
        )
        src_pos = np.array([-4.4, 0.1, 0.0])
        src_h = 1.7
        self.scene.play(
            FadeOut(drop),
            src["hm"].animate.scale_to_fit_height(src_h).move_to(src_pos),
            src["lbl"].animate.scale_to_fit_height(0.28).move_to(
                src_pos + np.array([0.0, src_h / 2.0 + 0.20, 0.0])).set_opacity(1.0),
            run_time=T_NORMAL,
        )
        self._caption(caption)

        # Factorization at the top of the focus area.
        f = fit_formula(factor_tex, max_width=4.6, scale=0.9)
        self.scene.add_fixed_in_frame_mobjects(f)
        f.move_to(np.array([0.7, 2.0, 0.0])).set_opacity(0)
        self.scene.play(f.animate.set_opacity(1.0), run_time=T_FAST)

        # Eigenvalue spectrum (Lambda): kept eigenvalues highlighted, the trivial
        # one (if any) greyed and labeled lambda_0.
        bars = eig_bar_chart(spectrum_vals, highlight_idxs=list(highlight_idxs),
                             bar_w=0.34, bar_max_h=1.5, trivial_idx=trivial_idx)
        bar_labels = VGroup()
        for rank, idx in enumerate(highlight_idxs):
            if idx < len(bars.submobjects):
                lab = MathTex(rf"\lambda_{rank + 1}", font_size=20, color=S.ACCENT)
                lab.next_to(bars.submobjects[idx], DOWN, buff=0.07)
                bar_labels.add(lab)
        if trivial_idx is not None and trivial_idx < len(bars.submobjects):
            lab0 = MathTex(r"\lambda_0\!\approx\!0", font_size=17, color=S.MUTED)
            lab0.next_to(bars.submobjects[trivial_idx], DOWN, buff=0.07)
            bar_labels.add(lab0)
        spectrum = VGroup(bars, bar_labels)
        spectrum.move_to(np.array([-0.7, -0.35, 0.0]))

        # The two highlighted eigenvectors as vertical strips: columns of V.
        strips = VGroup()
        strip_labels = VGroup()
        strip_x = [2.4, 3.4]
        for k in range(2):
            strip = vector_strip(np.asarray(eigvecs_two)[:, k], height=2.4,
                                 width=0.42, diverging=True,
                                 rainbow=rainbow_vectors)
            strip.move_to(np.array([strip_x[k], -0.1, 0.0]))
            lab = MathTex(rf"v_{k + 1}", font_size=22, color=S.INK)
            lab.next_to(strip, DOWN, buff=0.12)
            strips.add(strip)
            strip_labels.add(lab)
        vecs_group = VGroup(strips, strip_labels)

        overlays_new = VGroup(spectrum, vecs_group)
        self.scene.add_fixed_in_frame_mobjects(spectrum, vecs_group)
        overlays_new.set_opacity(0)
        self.scene.play(overlays_new.animate.set_opacity(1.0), run_time=T_NORMAL)
        # Hold the spectrum with its caption before swapping to the eigenvector
        # caption, so the line about which eigenvalues to keep has time to read.
        self.scene.wait(2.6)
        self._caption(caption_vectors)

        return VGroup(src["hm"], src["lbl"], f, spectrum, vecs_group)
