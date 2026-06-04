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
import numpy as np
from manim import VGroup, Dot, Line, Line3D, Text, MathTex, Table, interpolate_color
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
    """
    tmin, tmax = float(np.min(t)), float(np.max(t))
    return VGroup(*[
        Dot(
            point=points[i],
            radius=S.DOT_RADIUS,
            fill_opacity=1.0,
            color=color_for_t(t[i], tmin, tmax),
        )
        for i in range(points.shape[0])
    ])


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


def path_polyline(points, path, color=None):
    """VGroup of Line3D segments for the highlighted geodesic path.

    Using Line3D here (cylinder surface) because it is a single path with
    few segments and the thick 3D cylinder look makes it visually salient
    against the thin 2D edge lines.
    """
    color = color or S.GOOD
    return VGroup(*[
        Line3D(
            start=points[path[a]],
            end=points[path[a + 1]],
            thickness=PATH_THICKNESS,
            color=color,
        )
        for a in range(len(path) - 1)
    ])


def straight_line(start, end, color=None):
    """A single Line3D for the Euclidean straight-line comparison."""
    color = color or S.MUTED
    return Line3D(start=start, end=end, thickness=STRAIGHT_THICKNESS, color=color)


def caption(text):
    return Text(text, font_size=S.CAPTION_SIZE, color=S.INK)


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
