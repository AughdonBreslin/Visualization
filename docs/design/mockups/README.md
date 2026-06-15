# Redesign mockups (visual source of truth)

These are the brainstorming mockups produced for the UI redesign (branch `redesign/ui-system`).
They hold the exact CSS values the implementation must match. Each file is a content fragment
that was served through the brainstorming visual companion; open any in a browser to view (they
load Inter + JetBrains Mono from Google Fonts).

Chosen option per screen (the locked decision):

| File | Decision | Chosen |
|------|----------|--------|
| `foundation-directions.html` | Overall direction | A · Editorial |
| `container-framing.html`, `fourier-frame-compare.html` | Framing | No frames (frames kept as a future option) |
| `fourier-editorial.html` | Editorial on a dense page | reference |
| `typeface-options.html` | Typeface | 1 · Inter + JetBrains Mono |
| `type-scale.html` | Type scale | 2 · Balanced |
| `neutrals-palette.html` | Neutrals | A · Neutral graphite |
| `accent-color.html` | Accent | 1 · Periwinkle `#6b7cff` (user-customizable) |
| `controls.html` | Control language | A · Minimal / underline (mobile hit-area revisit) |
| `home-directory.html` | Home layout | A · Single-column index |
| `math-formula.html`, `math-ruled-numbered.html` | Display math | Left accent rule, left-aligned in block, numbered |
| `video-player.html` | Walkthrough player | A · Stacked |
| `step-nav-approaches.html` | Step navigation | 1 · Dropdown from step (desktop; mobile revisit) |
| `paired-viz-layout.html` | Paired-figure layout | C · Controls below |
| `manage-controls.html` | Dense controls | C · Tabbed groups default; all three patterns reusable per page |

See `docs/superpowers/specs/2026-06-15-ui-redesign-design.md` for the full system spec.
