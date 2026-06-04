"""Reusable manim mobject builders so every section looks consistent.

Manim 0.18.1 API notes (adaptations from the design spec):
- Line3D uses `thickness` (scene units, ~0.005-0.04) not `stroke_width` (pixel units).
  EDGE_WIDTH and path width specs are converted here to thickness values.
- Line3D inherits from Cylinder/Surface/VGroup; set_opacity works on the VGroup.
- Table.get_entries((r, c)) uses 1-based row/column indices.
"""
import numpy as np
from manim import VGroup, Dot3D, Line3D, Text, MathTex, Table, interpolate_color
from . import style as S

# Thickness values in scene units for Line3D
# The design spec uses stroke_width (pixel units): 1.0 for edges, 4-5 for paths.
# Approximate mapping: thin graph edge ~ 0.005, highlighted path ~ 0.025
EDGE_THICKNESS = 0.005
PATH_THICKNESS = 0.025
STRAIGHT_THICKNESS = 0.018


def color_for_t(t, tmin, tmax):
    u = (t - tmin) / max(1e-9, (tmax - tmin))
    return interpolate_color(S.ACCENT, S.WARM, u)


def point_cloud(points, t):
    tmin, tmax = float(np.min(t)), float(np.max(t))
    return VGroup(*[
        Dot3D(
            point=points[i],
            radius=S.DOT_RADIUS,
            color=color_for_t(t[i], tmin, tmax),
        )
        for i in range(points.shape[0])
    ])


def graph_edges(points, edges, color=None, opacity=S.EDGE_OPACITY):
    color = color or S.ACCENT
    lines = VGroup(*[
        Line3D(
            start=points[i],
            end=points[j],
            thickness=EDGE_THICKNESS,
            color=color,
        )
        for (i, j) in edges
    ])
    lines.set_opacity(opacity)
    return lines


def path_polyline(points, path, color=None, thickness=PATH_THICKNESS):
    color = color or S.GOOD
    return VGroup(*[
        Line3D(
            start=points[path[a]],
            end=points[path[a + 1]],
            thickness=thickness,
            color=color,
        )
        for a in range(len(path) - 1)
    ])


def straight_line(start, end, color=None, thickness=STRAIGHT_THICKNESS):
    color = color or S.MUTED
    return Line3D(start=start, end=end, thickness=thickness, color=color)


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
