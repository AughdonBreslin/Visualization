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
import textwrap
import numpy as np
from manim import (VGroup, Dot, Dot3D, Line, Line3D, Text, MathTex, Table,
                   interpolate_color, config, DOWN, LEFT, UL)
from . import style as S

# Thickness for Line3D (scene units, ~0.01-0.04 range)
PATH_THICKNESS = 0.022
STRAIGHT_THICKNESS = 0.014


def color_for_t(t, tmin, tmax):
    u = (t - tmin) / max(1e-9, (tmax - tmin))
    return interpolate_color(S.ACCENT, S.WARM, u)


def point_cloud(points, t):
    """VGroup of 2D Dot mobjects placed at 3D positions.

    Dot is a VMobject (fast to render) that can live anywhere in 3D space
    inside a ThreeDScene. DOT_RADIUS is in scene units.

    A radial sheen highlight (set_sheen) gives each flat circle a spherical
    appearance without the cost of a true 3D surface mesh.
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
        dot.set_sheen(0.5, direction=UL)
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
            seg_color = interpolate_color(S.GOOD, S.WARM, frac)
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


def recolor_cloud_by_values(cloud, values, c_lo, c_hi):
    """Animate-recolor a VGroup of Dots by a per-point scalar array.

    Parameters
    ----------
    cloud : VGroup  -- the point cloud whose submobjects are Dot objects.
    values : array-like  -- per-point scalar values (length must match cloud).
    c_lo, c_hi : ManimColor  -- colors mapped to the minimum and maximum value.

    Returns
    -------
    list of .animate expressions suitable for self.play(*...).
    """
    vals = np.asarray(values, dtype=float)
    vmin, vmax = float(vals.min()), float(vals.max())
    span = max(1e-9, vmax - vmin)
    anims = []
    for i, dot in enumerate(cloud.submobjects):
        u = (vals[i] - vmin) / span
        col = interpolate_color(c_lo, c_hi, u)
        anims.append(dot.animate.set_color(col))
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


def pseudocode_panel(active_index):
    """Build a small fixed-in-frame pseudocode panel for the top-left corner.

    Lines are rendered as plain Text (not MathTex) for speed. The active line
    is shown in ACCENT at full opacity; the others are dimmed to MUTED at 0.35.

    Parameters
    ----------
    active_index : int (0-5), which line to highlight.

    Returns
    -------
    VGroup of Text mobjects, suitable for add_fixed_in_frame_mobjects.
    """
    lines = [
        "0: input: points X, neighbors k",
        "1: for each i: link k nearest; w(i,j)=||x_i - x_j||",
        "2: D[i,j] = shortest path (Dijkstra)",
        "3: B = -1/2 J D^2 J",
        "4: B = V L V^T  (top eigenvectors)",
        "5: Y = [sqrt(L1) v1, sqrt(L2) v2]",
    ]
    items = []
    for idx, txt in enumerate(lines):
        if idx == active_index:
            t = Text(txt, font_size=18, color=S.ACCENT)
            t.set_opacity(1.0)
        else:
            t = Text(txt, font_size=18, color=S.MUTED)
            t.set_opacity(0.35)
        items.append(t)

    group = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    return group
