"""Shared visual system for the Isomap walkthrough: colors, fonts, timing.
Defined once and reused by every section so all six clips look identical.
"""
from manim import ManimColor

BG = ManimColor("#0a0c10")
INK = ManimColor("#e0e0e0")
MUTED = ManimColor("#9aa3ad")
ACCENT = ManimColor("#4aa3ff")
WARM = ManimColor("#ff8c5a")
GOOD = ManimColor("#79c98f")

CAPTION_SIZE = 30
FORMULA_SIZE = 40
LABEL_SIZE = 26

T_HOLD = 1.2
T_FAST = 0.6
T_NORMAL = 1.2
T_SLOW = 2.0
T_INTRO = 1.5

EDGE_OPACITY = 0.18
EDGE_WIDTH = 1.0
DOT_RADIUS = 0.018
