# Manifold Isomap Manim Explainer Design

**Date:** 2026-06-03
**Status:** Approved (design), pending implementation plan

## Goal

Build a manim-first, animated, step-by-step explainer for the Isomap algorithm as a pilot,
to evaluate whether pre-rendered manim clips are a better way to teach the more complex
manifold-learning processes than the current static per-step visuals. Each step is a short
clip in which the geometry, the formula, and a worked numeric example are choreographed
together in one scene, with on-screen captions carrying the intuition. If the pilot works,
the same pattern extends to the other algorithms in later efforts.

## Scope

In scope:
- A dedicated Isomap explainer page driven by six pre-rendered manim clips (one per canonical
  step), with a player to step through, replay, and auto-advance.
- A manim (Python) render pipeline that produces the clips and poster images as committed
  static assets.
- The fixed canonical example used in the clips, plus unit tests for the Python helpers that
  generate the data and the worked numbers shown on screen.

Out of scope (this pilot):
- The other five algorithms (PCA, MDS, LLE, Laplacian Eigenmaps, Kernel PCA). They keep their
  current behavior on the existing interactive page.
- Any change to the existing interactive sandbox (`pages/manifold.html`) beyond adding links
  to and from the new explainer page.
- Live, data-driven animation. The clips are pre-rendered on a fixed example and do not react
  to user-chosen datasets, params, or seeds. The live sandbox remains the interactive path.
- Audio narration. Clips are silent with on-screen captions.

## Approach summary (decisions reached during brainstorming)

- Animation technology: pre-rendered manim clips (manim-first), not web-native animation.
- Page structure: manim-first. This page is the film; the interactive sandbox is the
  secondary "try it live" companion.
- Clip structure: navigable per-step clips (six), not one continuous reel. Auto-advance so
  pressing play yields the full film, but any step can be isolated and replayed.
- Continuity: seamless. Objects persist and transform across step boundaries, so the end state
  of each clip is the start state of the next and the six clips play as one evolving scene.
  Steps remain individually navigable and replayable. This couples adjacent clips: each scene
  is authored to begin exactly where the previous one ended.
- Motion style: 3Blue1Brown-style: smooth, eased, contemplative (see Motion and polish quality
  bar).
- Composition: geometry-centric single scene per step. Formula and worked numbers live in the
  same canvas as the geometry (overlay style), not in segmented side panels.
- Narration: on-screen captions only, no audio.
- Pilot breadth: Isomap only.
- Execution: dedicated specialized agents (see Execution: dedicated agents) build and review
  the work, with fluidity and professional polish as a first-class quality bar.

## The six Isomap step clips

Each clip renders the step's geometry, builds the relevant formula in the same scene, animates
a small worked-number excerpt, and shows a caption line for the intuition.

1. Raw data: the 3D Swiss roll point cloud (full 1000 points), gently rotating. Caption frames
   the goal: recover the underlying 2D sheet.
2. kNN graph: edges appear, nearest-neighbor connections forming. Shows `k`. See "Legibility at
   1000 points" below for how the dense graph is kept readable.
3. Geodesic distances: a shortest path traces node to node across the graph from a highlighted
   source point, contrasted with the straight-line (ambient) distance.
4. Double-centering: the squared-distance grid morphs as row means and column means are
   subtracted and the grand mean re-added, then scaled by -1/2 into B. The formula
   `B = -1/2 J D^2 J` builds in; a 4x4 numeric excerpt updates live.
5. Eigendecomposition: the top eigenvectors of B sweep in; eigenvalues lambda_1, lambda_2 are
   shown.
6. Embedding: the cloud unrolls into the 2D coordinates `Y = [sqrt(lambda_1) v_1,
   sqrt(lambda_2) v_2]`.

## Fixed example

- Dataset: Swiss roll, 1000 samples.
- Neighborhood: `k = 8`.
- Random seed fixed so the clips are reproducible and the unit-tested numbers match what is on
  screen.

### Legibility at 1000 points

A full kNN graph over 1000 points (about 8000 edges) and a geodesic trace would be an
unreadable hairball if drawn naively. The clips handle this:
- Point-cloud and embedding steps (1 and 6) use the full dense 1000 points, which read well.
- Graph and geodesic steps (2 and 3) draw edges thin and translucent, highlight the structure
  being explained, and the geodesic path is drawn bright over the faded graph. The "edges grow
  one by one" beat zooms into a local neighborhood of a few points so individual edges are
  visible, then pulls back to the whole graph.
- Worked-number excerpts stay 4x4 regardless of N, so the matrix math remains legible.

## Visual style

- Dark background matching the site theme (near-black), accent blue `rgba(74,163,255)`, a warm
  accent for negative matrix values.
- Formulas rendered with manim MathTex (LaTeX); captions in a sans face.
- Roughly 15 to 30 seconds per clip.
- Output: H.264 MP4 at 1080p for broad browser support, plus one poster PNG per step.

## Motion and polish quality bar

Fluidity and a professional finish are first-class requirements, not afterthoughts. Every clip
is held to this bar:

- Smooth, eased motion in the 3Blue1Brown idiom: continuous transforms (manim `Transform` /
  `ReplacementTransform` / `MoveToTarget`) with smooth rate functions, not hard cuts or popping.
- 60fps render so motion is fluid.
- Seamless continuity across the six steps: the object that ends one step is the object that
  begins the next, transformed in place, so the whole sequence reads as one evolving scene.
- Deliberate staging: one idea introduced at a time, with brief holds so the viewer can absorb
  each beat before the next begins; nothing appears or vanishes abruptly.
- A consistent visual system across all clips: shared palette, typography, formula style,
  caption placement, margins, and timing constants defined once and reused.
- No jank: no overlapping or colliding elements, no text that appears faster than it can be
  read, no flicker, no elements that jump position between related beats.
- Legibility before spectacle: when the dense 1000-point graph or a matrix is on screen, it
  must stay readable (see Legibility at 1000 points).

The motion-design reviewer agent evaluates rendered clips against this bar and sends concrete
fixes back to the animator until it is met.

## Manim pipeline and repository structure

- `manim/` (new): the Python render project.
  - One `Scene` per step (six scenes), in one file or a small module per step.
  - Shared helpers: Swiss roll generation, kNN, geodesic (Dijkstra), double-centering, and the
    worked-number computations, plus shared styling/colors. These mirror the math already in
    `js/manifold/linalg.js` so the clips and the live sandbox tell the same story.
  - `manim/requirements.txt` pinning manim and its Python deps.
  - `manim/render.sh` renders all six scenes into `assets/manim/isomap/step-1.mp4` through
    `step-6.mp4`, each with a `step-N.png` poster.
- `assets/manim/isomap/` (new): the committed rendered clips and posters. The static site
  serves these directly; no Python runs at page-load time. The render pipeline is dev-time only.

## Page integration and UX

- New page `pages/manifold_isomap.html` plus `js/manifold_isomap.js` and
  `styles/manifold_isomap.css`, following the existing per-page pattern.
- A video player that plays the current step's clip, with prev / play / next controls, a
  scrubber, and a step list (the six step titles) for direct navigation. Auto-advance to the
  next step at clip end so pressing play once yields the whole film.
- Captions are baked into the clips. In addition, each step's caption text and formula are
  shown as a short DOM transcript beneath the player for accessibility and skimming.
- A clear "Try it live" link to the existing interactive sandbox (`pages/manifold.html`), which
  is unchanged and becomes the hands-on companion.
- Linked from `index.html` (homepage project list) and from the sandbox page.

## First milestone: toolchain (gated)

The current environment has Python 3.12 and pip but is missing ffmpeg, LaTeX, and cairo/pango.
Before any clips can render, install the toolchain:
- System packages via apt: `ffmpeg`, `texlive-latex-base`, `texlive-latex-extra`, `dvisvgm`,
  `libcairo2-dev`, `libpango1.0-dev`, `pkg-config`.
- Then `pip install manim` (pinned in `manim/requirements.txt`).
- Verify by rendering a manim hello-world scene to MP4.

This step needs sudo. If apt prompts for a password, the user runs that one install command
(for example via the `!` prefix in the session). Everything after the toolchain is in place is
handled normally.

## Execution: dedicated agents

The work is split across five dedicated agents (defined in `.claude/agents/`), each with one
clear responsibility:

- Overseer (coordinator): manages the individual subagent tasks, relays information back and
  forth between specialists, and ensures each one has the clarity and context it needs to
  produce the expected quality. It decomposes work into precise task briefs, dispatches the
  right specialist, carries findings from one agent to another (for example, the reviewer's
  fixes back to the animator, or the verifier's correct numbers into the animation), and does
  not mark a step done until the motion-design reviewer and math verifier both pass. It holds
  the spec as the source of truth and is the single point that keeps the four specialists in
  sync.
- Manim animator: implements the manim scenes and the render pipeline, owning easing, staging,
  seamless continuity, and the shared visual system. Builds to the Motion and polish quality
  bar.
- Motion-design reviewer: reviews rendered clips (by extracting frames and inspecting timing)
  for fluidity and professional polish, and returns concrete, specific fixes. Read-only on
  code; it critiques, the animator fixes, repeat until the bar is met.
- Manifold-math verifier: proves the algorithm and the on-screen worked numbers are correct and
  match the existing `js/manifold/linalg.js` results, via Python unit tests.
- Web player engineer: builds the explainer page and its video player UX (step navigation,
  scrubber, auto-advance, transcript, accessibility, links).

Quality bar for the whole effort: fluid, professional motion. The motion-design reviewer is the
gate on that bar before any step is considered done.

## Testing

- Python unit tests (`python -m unittest`) for the data and math helpers: Swiss roll
  generation determinism, kNN correctness, geodesic distances, double-centering, and the
  worked-number excerpts, so the numbers shown in the clips are provably correct and match the
  existing `linalg.js` results.
- Render verification: render the six clips and eyeball them for correctness and legibility.
- Page playback: manual browser check of the player (step nav, scrub, auto-advance, transcript,
  links).

## Style constraints

- No em-dashes anywhere (prose, code, comments, HTML).
- No emphasis tags or markdown emphasis (`<em>`, `<strong>`, `<b>`, `<i>`, `<mark>`, `*`,
  `**`) in generated page content.
- Measured, non-dramatic prose in page copy and captions.
- Reuse the shared `base.css` / `article.css` theme on the new page.
