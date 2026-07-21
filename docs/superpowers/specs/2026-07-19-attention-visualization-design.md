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
- The backward pass / backpropagation through attention, and training (the weight matrices
  actually being learned). Two deliberate follow-on phases, tracked as separate future specs once
  phase 1 ships. Section "Future phases" below records the architectural decisions already made
  so phase 1 doesn't paint itself into a corner.
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
`pages/attention.html`, `styles/attention.css`, wired into `index.html` as entry 10. Given this
page's size, its JS follows the `js/manifold/` precedent (an ES-module directory) rather than
`gradient-descent.js`'s single-file pattern: `js/attention/` holds `math.js` (pure step
computations, no DOM — the only part of this codebase's frontend that is unit-testable in the
ordinary sense), `glyphs.js` (SVG glyph and connector builders), `pipeline.js` (the sticky bar and
the shared open-node behavior), `scenes.js` (per-step scene rendering and animation), `presets.js`
(preset data and the picker), and `main.js` (the page's entry point, mirroring the try/catch init
block every other page's script uses). No new shared infrastructure or build step.

The rail is the site's existing `js/section-outline.js` component, unmodified — it already
auto-discovers `<section class="panel">` blocks with an `h2`/`h3` heading and builds a numbered,
scrollspy-highlighted rail from them, which is exactly the "slim secondary list of the same
steps" the design calls for. Each of the eight step scenes is one such `.panel`; the hero/intro
above the sticky bar is plain content, not a panel, so it is not numbered (matching the validated
mockup: eight numbered rail entries, `01`-`08`, one per step — not nine). `attention/pipeline.js`
adds one delegated click listener for `.section-outline-list a[data-target]` clicks so a rail
click triggers the same pulse-and-scroll-and-glow the pipeline bar does, layered on top of
section-outline.js's own (unmodified) scroll-and-highlight behavior rather than replacing it.

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

## Per-step material (the conceptual aside content, in full)

Per later steering during implementation, each step gets real teaching content, not one line —
closer to the depth of a self-contained mini explainer than a caption. Each step's aside pairs a
motivation paragraph (why this operation exists, what problem it solves relative to the previous
step), a mechanism paragraph (what literally happens to the numbers, already covered above), and
one `.callout` (the site's existing reusable component — see `styles/components.css`, used
elsewhere e.g. `pages/fourier.html`) carrying either a "why this specific design choice" aside or
a plain note. This is the exact copy to use, validated with the user before implementation:

**1. Input embeddings** — *Motivation:* Before any attention math happens, each token needs to
become a vector a matrix can act on. This step computes nothing about relationships between
tokens yet; it's just the raw material every later step consumes. *Callout (Note):* Real
transformer embeddings run hundreds or thousands of dimensions; this page uses d=4 so every
number stays visible on screen. Nothing about the mechanism changes at higher dimension, only the
width of every vector shown below.

**2. Q/K/V projections** — *Motivation:* A raw embedding conflates everything about a token into
one vector. Attention needs three different views: a query ("what am I looking for"), a key
("what do I offer"), and a value ("what I actually contribute if chosen"). Splitting one vector
into three roles via three learned matrices is what makes the next step's comparison meaningful
instead of trivial. *Callout (Why three matrices, not one?):* If Q and K shared a matrix, every
token's query would equal its own key, so every token would trivially attend most to itself.
Separate projections let a token's query and key diverge.

**3. QKᵀ scores** — *Motivation:* With every token holding a query and a key, comparing a query
against a key is a similarity measure. This step performs every such comparison at once. *Callout
(Note):* The raw score is unbounded and grows with the query/key vectors' magnitude, exactly what
the next step exists to control.

**4. Scale** — *Motivation:* Dot-product magnitude grows with dimension d; large scores push
softmax toward near one-hot output with vanishing gradients elsewhere. *Callout (Why √d
specifically?):* If Q and K entries have roughly unit variance, their dot product's variance
grows proportional to d, so its standard deviation grows with √d. Dividing by √d keeps the
score's scale roughly constant regardless of d.

**5. Mask** — *Motivation:* Every step so far treats tokens symmetrically. Fine for encoding a
complete sentence, wrong for predicting the next token; letting a model see the answer it's
predicting makes training meaningless. *Callout (Why −∞ and not just 0?):* e⁰ = 1, so a masked
score of 0 would still receive real attention weight. e^(−∞) = 0 exactly, the only value
guaranteed to zero out that position.

**6. Softmax** — *Motivation:* Converts raw scaled scores into an actual probability distribution
over keys. *Callout (Note):* The exponential amplifies differences, part of why Scale matters,
since unscaled scores would make softmax nearly one-hot almost everywhere.

**7. Weighted sum** — *Motivation:* Now that there's a real probability distribution over "how
much to listen to each token," the actual listening happens by blending. *Callout (Why value
vectors, not the original embeddings?):* Like Q and K, V is its own learned projection, so the
model can choose what a token contributes independent of what makes it a good match or what it's
searching for.

**8. Output** — *Motivation:* Closes the loop; same shape as the input, now context-aware.
*Callout (Note):* Not usually the end of a transformer block; in a real model it continues
through a residual connection and a feed-forward layer, both outside this page's scope.

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

## Future phases (follow-on, not built now)

Confirmed with the user during implementation: phase 1, as built and as it continues to be built,
is inference only. Every one of the eight steps shows what happens to one input as it flows
through a single attention head using weight matrices (`W_Q`, `W_K`, `W_V`) that are already
fixed — hand-picked for this toy example rather than learned, but treated exactly as if the model
had already been trained. Nothing in phase 1 trains anything or shows weights changing. Two
distinct follow-on phases extend past that boundary, recorded now so phase 1's architecture
doesn't foreclose either one.

### Phase 2: the backward pass (backprop)

For the one worked example already on the page, compute how much each weight *should* change:
the gradient of a loss with respect to `W_Q`, `W_K`, `W_V` (and every intermediate value along
the way — scores, scaled scores, softmax weights). This flows through the same eight steps in
reverse, with its own gradient-flow visuals. It is a diagnostic on top of the existing fixed
weights, not training: it computes a gradient, it does not apply one.

The sticky pipeline bar swaps from the forward-pipeline glyphs to a backward-pipeline rendering
at the point in the scroll where backprop content begins, while the slim rail keeps incrementing
step numbers continuously across both passes rather than resetting — one continuous document, not
two separate pages glued together. This means the phase 1 sticky-bar container should be built as
a swappable region (its content driven by which step group is active) rather than a single
hardcoded diagram.

Positional encoding is also deferred to this phase rather than phase 1. Real transformer input
sums a positional encoding into each token embedding before attention (attention itself has no
notion of sequence order without it); phase 1's Input embeddings step sets that aside to keep its
scope to attention alone. A future addition would extend that step to show each token's raw
embedding, its positional encoding (by position), and the sum that actually feeds Q/K/V — three
vectors merging into one, mirroring the Q/K/V split later in the pipeline.

### Phase 3: training

Confirmed with the user: a genuinely further phase, larger than phase 2, showing where `W_Q`,
`W_K`, and `W_V` actually come from — not one gradient computation, but the weights visibly
converging from random initial values to something like the fixed values phase 1 uses throughout,
via repeated forward and backward passes over training data. This needs, at minimum: a loss
function, more than one training example (a small synthetic dataset, not just the single "the cat
sat" worked example), an iteration loop (the weights change every step), and a visualization of
that change over time (e.g. a loss curve, and the weight matrices' heatmaps visibly shifting
across iterations rather than sitting static).

One open design question, not resolved here: attention is not normally trained in isolation. In a
real transformer, attention sits inside a larger model trained end-to-end on a task like
next-token prediction, and it receives its gradient from that larger task's loss, not from a loss
defined on attention's output directly. Giving this page's single attention head something to
train against will need either a small synthetic pretext task with a defined loss on the
attention output, or a tiny toy downstream layer (e.g. a linear classifier) stacked on top of the
attention output, whose loss backpropagates through attention into `W_Q`/`W_K`/`W_V`. Deciding
between those (or another approach) is phase 3's own design work, to happen when phase 3 is
actually brainstormed, not assumed here.

## Testing

Same standard as every other page on this site: no build or test tooling, manual in-browser
verification. Before calling phase 1 done: all eight scenes on all presets, pipeline-bar click vs.
rail click vs. plain scroll (confirming only the first two trigger the pulse/glow), the mask
toggle, mobile width, and a check for console errors — matching the QA pass already run on
gradient-descent.
