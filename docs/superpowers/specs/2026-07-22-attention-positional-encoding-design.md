# Attention Page: Positional Encoding

Date: 2026-07-22
Status: approved for implementation

## Problem

Attention on this page is permutation-invariant: nothing in the pipeline knows which token came
first. Input's own blurb already names this as a deliberate simplification ("with positional
information set aside to keep focus on attention itself"). This spec implements it for real.

Fixing it also surfaces a real, currently dormant bug: `Q`, `K`, `V`, and `embeddings` are keyed
by token *string*, not position. If a token repeats in a sentence, both occurrences collide onto
one vector. This was flagged and accepted as non-blocking when Task 1 was first reviewed, since
neither existing preset repeats a word. It stops being safe to ignore once position matters, since
disambiguating two instances of the same word is exactly what positional encoding is for.

## Scope

In scope:
- Real (not illustrative) additive sinusoidal positional encoding: `X[i] = embeddings[i] +
  PE(i, d)`, computed exactly, added to every existing preset's input. This is a genuine pipeline
  change — every downstream value (`Q`, `K`, `V`, scores, weights, output) shifts for all three
  presets, not just a demo confined to Input.
- The underlying data-model fix: `Q`/`K`/`V`/`embeddings` become plain arrays indexed by
  position (parallel to `tokens`), matching how `scores`/`weights`/`output` already work, instead
  of dicts keyed by token string.
- A third preset, `"the dog chased the cat"` (5 tokens, repeats "the" at positions 0 and 3),
  demonstrating that the fix actually disambiguates a repeated word once position is added.
- Restructured Input filmstrip: `01: STORAGE` (unchanged) → new `02: TRANSFORM` (the addition
  itself, with an explicit before/after comparison of the two "the" positions) → `03: CONCEPT`
  (the existing scale-stats stage, renumbered) → new `04: RELATED RESEARCH` (RoPE and ALiBi as
  alternatives, citing the RoFormer paper).
- `TOKEN_COLORS` extended from 3 to 5 entries; new `.mgrid.g5x5` / `.mgrid.g5x4` CSS rules
  (desktop + mobile), matching the existing per-size rule pattern.
- Re-verification that all three presets still produce a clearly-peaked (non-uniform) attention
  pattern after positional encoding is added — the same bar the original embeddings had to clear,
  checked the same way (computed peakedness, not eyeballed).

Out of scope:
- RoPE and ALiBi themselves are not implemented, only cited/explained in prose. Implementing
  either would require different pipeline surgery (RoPE: rotate `Q`/`K` between projection and
  scoring; ALiBi: a distance-based bias on raw scores, alongside Mask) and is a candidate for a
  future spec, not this one.
- Backprop / training (Phase 2 / Phase 3 from the original spec) — untouched by this work.
- Multi-head attention — still out of scope per the original spec; unaffected by this change.
- Free-form sentence input — still out of scope per the original spec (confirmed again this
  session: generated embeddings for arbitrary words would be linguistically meaningless without a
  real trained model, so free-form input was explicitly ruled out this session).

## Data model fix

Today: `embeddings`, `Q`, `K`, `V` are all `{tokenString: vector}` dicts. `math.js`'s
`projectAll`, `scoreMatrix`, and `weightedSum` all look values up by token string
(`Q[ti]`, `V[tj]`). `presets.js`'s `embeddings` field is a `{word: vector}` object per preset.

After: all four become plain arrays, one entry per position, in the same order as `tokens`.

- `presets.js`: each preset's `embeddings` field becomes an array of vectors instead of a
  `{word: vector}` object.
- `math.js`:
  - `projectAll(embeddings, W)` drops the `tokens` parameter (no longer needed for lookup) and
    becomes `embeddings.map((x) => linearProject(W, x))`.
  - `scoreMatrix(Q, K)` drops `tokens`, becomes `Q.map((qi) => K.map((kj) => dot(qi, kj)))`.
  - `weightedSum(weights, V, d)` drops `tokens`, indexes `V` by position (`V[j]`) instead of by
    token string (`V[tokens[j]]`).
  - `computePipeline` adds the positional-encoding step (below) before projecting Q/K/V, and
    builds `Q`/`K`/`V`/`output` as arrays.
- `scenes.js`: every call site that currently does `result.Q[t]` / `result.K[t]` / `result.V[t]`
  / `result.embeddings[t]` (a token string lookup) changes to index by position instead. In
  `renderQkv`, `renderScores`, and `renderWsum`, the position index is already computed and in
  scope at every one of these call sites (`focusIdx`, `qIdx`, `kIdx`, or the surrounding loop's
  own index) — this is a mechanical swap, not new logic. `renderInput`'s two call sites
  (`result.embeddings[t]` in the storage body and the labeled-vector block) simplify to just
  `result.embeddings` directly, since it's already the full array in position order.

## Positional encoding

Standard sinusoidal encoding from the original Transformer paper, computed exactly (no
hand-picking, unlike the weight matrices):

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Added once, in `computePipeline`, before anything else runs: `X[i] = embeddings[i] + PE(i, d)`.
Everything downstream (`Q`, `K`, `V`, scores, scale, mask, softmax, weighted sum, output) already
consumes whatever `X` is, so no other step's code changes — only its *numbers* do.

## New preset: "the dog chased the cat"

5 tokens: `the`, `dog`, `chased`, `the`, `cat`. Repeats "the" (a determiner) rather than a noun,
so there's no "is this actually the same entity" ambiguity muddying the lesson — it isolates the
position-encoding point cleanly. Reuses "dog" and "cat" from the two existing presets (ties the
new example back to already-established vocabulary and per-token colors) and needs exactly one
genuinely new hand-picked embedding: "chased". Shares the existing `WEIGHTS` (all three presets
already share one weight matrix set).

## UI changes (Input step)

- `01: STORAGE`: unchanged.
- New `02: TRANSFORM`: shows the `PE(pos, 2i)` / `PE(pos, 2i+1)` formula, the addition
  `X[i] = embeddings[i] + PE(i, d)`, and an explicit side-by-side of the two "the" positions (0
  and 3) for the new preset — their embeddings are identical, their final `X` rows are not. This
  is the stage that makes the fix's payoff visible rather than implied.
- `03: CONCEPT` (renumbered): the existing scale-stats stage, content unchanged.
- New `04: RELATED RESEARCH`: matches the concept-box + citation pattern already used by QKV's
  own Related Research stage. Covers RoPE (rotates `Q`/`K` by a position-proportional angle
  instead of adding to the embedding, making every attention score depend on relative distance
  structurally; cites Su et al. 2021, arXiv:2104.09864, "RoFormer: Enhanced Transformer with
  Rotary Position Embedding"; names its adoption in LLaMA, GPT-NeoX, PaLM, and Falcon) and ALiBi
  (a distance-based penalty added directly to raw scores, the same place the causal mask already
  operates, as a smooth decay instead of a hard cutoff).
- Input's static blurb (in `pages/attention.html`, not the filmstrip) needs two corrections,
  caught in spec self-review: it currently says "with positional information set aside to keep
  focus on attention itself," which becomes false once this ships — needs rewording to describe
  what's now added instead of what's being deferred. It also says "every token keeps the same
  color everywhere it appears below," which breaks for the new preset's repeated "the": colors
  are keyed by position, not word identity (`TOKEN_COLORS`' own header comment already says this),
  so the two "the" positions get two different colors — consistent with the code, but the prose's
  current wording claims otherwise. Reword to describe color as tied to position, not token
  identity.

## Verification plan

- Computed (not eyeballed) peakedness check for all three presets after positional encoding is
  wired in, mirroring this project's own established bar and precedent (an earlier preset once
  produced near-uniform attention and had to be redone; the same check applies here since adding
  position shifts every preset's numbers, not just the new one's).
- Full headless-browser pass per this session's established process: 0 console errors, 0
  horizontal overflow at 1400px and 375px, screenshots confirming each changed/new stage, on every
  preset (not just the new one — the two existing presets' numbers change too).

## Future phases

Unchanged from the original spec: Phase 2 (backprop, computing how the weight matrices should
change for a worked example) and Phase 3 (training, applying those gradients over iterations and
watching convergence) remain deferred, tracked as future work, not affected by this change.
