# Attention Mechanism Visualization: Design

Date: 2026-07-19
Status: approved for implementation (phase 1: forward pass)

## Problem

DistCharts has nine pages, each a deep interactive explainer for one ML/statistics concept.
The tenth adds single-head scaled dot-product attention: `Attention(Q,K,V) = softmax(QKᵀ/√d)V`.
This page is explicitly meant to go deeper than every other page on the site: full arithmetic
animation through a real worked example, an editable input, conceptual grounding at every step,
and (in a later phase) the backward pass as well.

## Scope

In scope (phase 1, this spec): the forward pass of single-head scaled dot-product attention,
including the optional causal/padding mask. A pipeline diagram of every computational step,
serving as the page's primary navigation, where each step expands into a full worked-example
walkthrough with granular, scrubbable arithmetic animation.

Out of scope for this spec:
- Multi-head attention (splitting/concatenating heads). Confirmed out of scope entirely — the
  page stays single-head throughout.
- The backward pass / backpropagation through attention. This is a deliberate phase 2, tracked
  as a follow-on spec once phase 1 ships. Section "Phase 2" below records the architectural
  decisions already made so phase 1 doesn't paint itself into a corner.
- Free-form numeric editing of the worked example. Phase 1 ships a small set of curated presets;
  free-form editing is a possible future enhancement, not committed to here.

## The eight steps

The complete, ordered list of computational steps, confirmed against the standard formula plus
its common causal-masked variant:

1. **Input embeddings** — the starting token vectors
2. **Q / K / V projections** — three learned linear projections of the input
3. **QKᵀ scores** — every query dotted with every key
4. **Scale** — divide every score by √d
5. **Mask** (toggleable: none / causal) — set masked positions to −∞ before softmax; this is
   what makes GPT-style decoder self-attention unable to look ahead
6. **Softmax** — row-wise normalize into attention weights
7. **Weighted sum with V** — combine value vectors by their attention weight
8. **Output** — one context-aware vector per input token

## Visual reference

The interaction pattern in this spec (the pipeline bar, the open-node pulse/scroll/glow, the
sticky-nav-plus-slim-rail structure) was validated interactively during brainstorming and is not
speculative prose — it matches two working mockups, kept for implementation reference:
`.lavish/attention-interaction-mockups.html` (the four candidate expand patterns, and why
semantic zoom was chosen over master-detail, accordion, and overlay modal) and
`.lavish/attention-scroll-zoom-mockup.html` (the final validated structure: sticky pipeline bar,
slim secondary rail, eight real scrollable scenes, unified open-node behavior). Node glyph shapes,
connector curves, and the arrive-glow animation in this spec's later sections are drawn directly
from the second mockup.

## Architecture

Follows the existing article-page archetype exactly (same shell every other page uses):
`pages/attention.html`, `js/attention.js`, `styles/attention.css`, wired into `index.html` as
entry 10 and linked from the section rail. No new shared infrastructure or framework — matches
the pattern of every other page (`gradient-descent.js`, `manifold_isomap.js`, etc.): a dedicated,
self-contained JS file per page, no build step.

The departure from other pages: instead of generic prose sections, the rail's numbered entries
map one-to-one to the eight pipeline steps (`01 Overview`, `02 Input embeddings`, ... `09
Output`), and the pipeline diagram itself — not the rail — is the primary way users navigate.

## The pipeline bar (primary navigation)

A compact horizontal row of eight simplified, stroke-only SVG glyphs, one shape family per data
type so the shape communicates what kind of object lives at that stage (stacked bars for token
vectors, a fanning square for the Q/K/V projection, a tinted 3×3 grid for the score/mask/softmax
stages, converging lines into a multiply node for the weighted sum). Nodes connect with shallow,
sagging cubic-bezier arrows (not straight lines), with a slow dash-offset drift so the row reads
as "data flowing right" even at rest. The Q/K/V node's K and V arrows honestly arc further to
reach the QKᵀ and weighted-sum nodes rather than pretending the flow is a simple straight chain.

This bar starts inline below the page's hero copy, then becomes `position: sticky; top: 0` and
stays pinned for the remainder of the page — present through every one of the eight step scenes
below it. The step currently in view is highlighted in the bar via `IntersectionObserver`.

A slim, secondary rail (matching every other page's section-outline rail, narrower and more
muted here) lists the same eight steps as text. It is not a separate navigation system — clicking
a rail entry triggers the identical "open node" behavior as clicking the corresponding pipeline
node (see below), just via a second entry point.

## Interaction model: "open node"

Clicking a node — in the sticky pipeline bar, from anywhere on the page, or in the slim rail —
runs one function: the clicked pipeline-bar node briefly pulses (grows, glows), the page smooth-
scrolls to that step's section, and on arrival that section's large hero glyph catches a matching
glow. Section and bar glyph share color, shape, and accent styling, so the two moments read as
one continuous "zoom" even though the mechanism is scroll-plus-restyle, not a literal DOM morph
(a true FLIP/cross-document morph was considered and rejected as unnecessary complexity and
fragility for the visual payoff — this approach was validated interactively and felt right).

Every step is a real, permanently scrollable page section — never a modal or overlay that closes.
Reaching a step by scrolling normally, by the slim rail, or by the pipeline bar all land in
exactly the same place; only the pipeline-bar/rail route adds the pulse-and-glow flourish.

## Per-step scene content

Each of the eight scenes shares one layout: hero glyph (a larger render of that step's pipeline-
bar icon) → eyebrow ("Step N of 8") → title → a short conceptual aside (why this operation
exists, what problem it solves relative to the previous step — e.g. why scale by √d, why mask
before softmax rather than after) → the granular worked-example animation with play/step/scrub
controls → the operation's formula in real LaTeX (MathJax, matching the site's existing math
convention).

The animation device is specific to each operation's shape:
- **Input embeddings**: the token vectors simply laid out, color-coded per token
- **Q/K/V projections**: element-wise multiply-then-sum pulses through the weight matrix, staged
  per token and per output column
- **QKᵀ**: a crosshair sweeps the score grid; each cell's dot product counts up as its heatmap
  tint fades in, cell by cell
- **Scale**: every cell's number counter-rolls from raw score to scaled score
- **Mask**: masked cells visibly darken to −∞ and their heatmap tint drops to zero, previewing
  what softmax is about to do to them
- **Softmax**: numbers lift out of the grid onto a shared number line, exponentiate, normalize
  into a stacked probability bar, then drop back into the grid as the final attention weights
- **Weighted sum**: the three value vectors fade to opacity equal to their attention weight, then
  visibly slide and merge into the output vector, with a running per-element accumulation readout
- **Output**: mirrors the input embeddings visual exactly — same shape in, same shape out — the
  visual rhyme is itself part of the explanation

## Worked example & presets

One example threads through all eight scenes: 3 tokens, d=4, persistent per-token color coding
(the same hue for "cat" in every scene, in every visualization, so it stays visually findable).
A small preset picker offers 2-4 curated token sets — at minimum a plain sentence and one chosen
to show an interesting attention pattern (e.g. a clear long-range dependency or a repeated
token). Switching presets reruns the entire pipeline live, recomputing every downstream scene.
Numbers are hand-tuned per preset so every animation and every conceptual callout (e.g. "peak
attention here") stays coherent — no free-form numeric editing in this phase.

The preset system should be built as a thin UI layer over one generic recompute path: a preset
is just a set of token embeddings fed into the same pipeline math every scene already runs, not
special-cased logic per preset. A future free-form editor (or a "random"/"custom" preset slot)
should then be addable by swapping the input widget for that recompute path, without touching
the eight scenes' animation or layout code. This is the same forward-compatibility principle as
the phase 2 sticky-bar swap: don't build phase 1 in a way that forecloses the obvious next step.

## Phase 2 (follow-on, not built now)

Recorded here so phase 1's architecture doesn't foreclose it: the backward pass will flow through
the same eight steps in reverse, with its own gradient-flow visuals. The sticky pipeline bar swaps
from the forward-pipeline glyphs to a backward-pipeline rendering at the point in the scroll where
backprop content begins, while the slim rail keeps incrementing step numbers continuously across
both passes rather than resetting — one continuous document, not two separate pages glued
together. This means the phase 1 sticky-bar container should be built as a swappable region (its
content driven by which step group is active) rather than a single hardcoded diagram.

## Testing

Same standard as every other page on this site: no build or test tooling, manual in-browser
verification. Before calling phase 1 done: all eight scenes on all presets, pipeline-bar click vs.
rail click vs. plain scroll (confirming only the first two trigger the pulse/glow), the mask
toggle, mobile width, and a check for console errors — matching the QA pass already run on
gradient-descent.
