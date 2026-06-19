# Redesign TODO (post-merge follow-ups)

Specific bugs, mobile issues, perf, and enhancements gathered after the dark UI redesign merged.
Roll these into the deferred phases in `redesign-backlog.md` (robustness / mobile) as picked up.

## Functional bugs (desktop)
- [x] **Regularization:** on load, the ridge coefficients chart is not rendered properly (likely
  needs a redraw / resize after layout settles). Fixed: ResizeObserver re-renders when the
  section-outline rail shrinks the column after init.
- [x] **Bayesian:** the variational inference lines are not all represented in the legend. Fixed:
  the horizontal legend now wraps to multiple rows instead of running off the right edge.
- [x] **PCA:** the "Covariance operator" title is blocked / overlapped by the principal components
  graph (occurs on mobile, where the figures stack). Fixed: definite .pca-viz height so the plot
  matches its grid row instead of overflowing onto the next caption.
- [x] **Manifold:** the pairwise affinity displays do not fit inside their boxes (overflowed on
  mobile). Fixed: the matrix strip renders at the host's real pixel size (viewBox = host, scale 1)
  so the foreignObject canvases no longer depend on SVG viewBox scaling that mobile WebKit ignores;
  re-renders on resize.

## Mobile
- [ ] Zooming in and out should be clean (pinch-zoom behavior across pages).
- [ ] Math must not bleed off the side of the page; make math blocks internally scrollable
  (horizontal scroll) rather than overflowing the viewport.
- [ ] **PCA:** the point labels are fixed on screen instead of being positionally attached to a
  location in the chart (they should track the chart space, not the viewport).
- [ ] **Fourier:** the five interpretation tabs start to bleed off the right side; wrap or make the
  tab row scrollable.
- [ ] **Manifold:** the neighborhood graph takes far too long to appear and makes the page choppy;
  the pairwise affinity displays also do not fit in their boxes.

## Performance
- [ ] On mobile the page can get frame-y when expensive computations (e.g., a render) run while not
  currently visible. Defer / pause offscreen work (e.g., IntersectionObserver so a demo only
  computes/animates when in view).
- [ ] **Manifold:** speed up the neighborhood graph step (slow to appear, choppy); overlaps with
  the manifold mobile item above.

## Features / enhancements
- [ ] **Distributions:** add the ability to hide individual components within a mixture model
  (per-component visibility toggle).
- [ ] Generally add collapsibility to more components / sections across pages.
- [ ] **Fourier:** make the worked examples better.
