// js/attention/filmstrip.js
// Wires up the horizontal scroll-snap filmstrip's navigation affordances: hover arrows,
// click-to-jump dots, and click-and-drag scrolling for mouse users (touch/trackpad already pan
// natively). Call initFilmstrips() after every render - scenes.js rebuilds each filmstrip's DOM
// from scratch on every preset switch and mask toggle, so listeners need rewiring each time.

// Drag state lives at module scope, and the window-level mousemove/mouseup listeners are wired
// exactly once, ever: attaching a fresh pair of window listeners inside initOne() below (which
// reruns on every render) would leak one extra pair per filmstrip per render.
let activeDrag = null;
let windowListenersWired = false;

function ensureWindowListeners() {
  if (windowListenersWired) return;
  windowListenersWired = true;
  window.addEventListener('mousemove', (e) => {
    if (!activeDrag) return;
    const dx = e.clientX - activeDrag.startX;
    if (Math.abs(dx) > 4) activeDrag.moved = true;
    activeDrag.track.scrollLeft = activeDrag.startScroll - dx;
  });
  window.addEventListener('mouseup', () => {
    if (!activeDrag) return;
    const { track, moved, onEnd } = activeDrag;
    track.classList.remove('is-dragging');
    track.style.scrollBehavior = '';
    activeDrag = null;
    onEnd(moved);
  });
}

function initOne(wrap) {
  const track = wrap.querySelector('[data-role="filmstrip"]');
  const prevBtn = wrap.querySelector('[data-role="fs-prev"]');
  const nextBtn = wrap.querySelector('[data-role="fs-next"]');
  const dots = [...wrap.querySelectorAll('[data-role="fs-dot"]')];
  const stages = [...track.children];

  const hasOverflow = track.scrollWidth > track.clientWidth + 1;
  wrap.classList.toggle('no-scroll', !hasOverflow);
  if (!hasOverflow) return;

  function currentIndex() {
    // The last stage's own offsetLeft is often unreachable (there's no more room to scroll
    // it flush left once its right edge already hits the track's max scroll position), so
    // nearest-offset matching would otherwise never report the last stage as current.
    if (track.scrollLeft >= track.scrollWidth - track.clientWidth - 1) return stages.length - 1;
    let idx = 0;
    let best = Infinity;
    stages.forEach((stage, i) => {
      const d = Math.abs(stage.offsetLeft - track.scrollLeft);
      if (d < best) { best = d; idx = i; }
    });
    return idx;
  }

  function scrollToStage(i) {
    const stage = stages[Math.max(0, Math.min(stages.length - 1, i))];
    if (stage) track.scrollTo({ left: stage.offsetLeft, behavior: 'smooth' });
  }

  function updateUI() {
    const atStart = track.scrollLeft <= 1;
    const atEnd = track.scrollLeft >= track.scrollWidth - track.clientWidth - 1;
    prevBtn.classList.toggle('is-disabled', atStart);
    nextBtn.classList.toggle('is-disabled', atEnd);
    const idx = currentIndex();
    dots.forEach((dot, i) => dot.classList.toggle('is-current', i === idx));
  }

  prevBtn.addEventListener('click', () => scrollToStage(currentIndex() - 1));
  nextBtn.addEventListener('click', () => scrollToStage(currentIndex() + 1));
  dots.forEach((dot, i) => dot.addEventListener('click', () => scrollToStage(i)));

  let scrollTimer = null;
  track.addEventListener('scroll', () => {
    if (scrollTimer) clearTimeout(scrollTimer);
    scrollTimer = setTimeout(updateUI, 60);
  }, { passive: true });

  // Click-and-drag scrolling: a plain overflow:auto container only pans via touch/trackpad, not
  // a mouse click-drag, so that gesture needs to be built by hand.
  let suppressNextClick = false;
  track.addEventListener('mousedown', (e) => {
    ensureWindowListeners();
    activeDrag = {
      track,
      startX: e.clientX,
      startScroll: track.scrollLeft,
      moved: false,
      onEnd: (moved) => {
        suppressNextClick = moved;
        if (moved) scrollToStage(currentIndex());
      },
    };
    track.classList.add('is-dragging');
    track.style.scrollBehavior = 'auto'; // 1:1 tracking while dragging, not the CSS smooth easing
  });
  // A drag that actually moved the strip shouldn't also fire a click on whatever's underneath
  // (Scale's shrink button, Mask's causal toggle, Scores'/Softmax's compute buttons).
  track.addEventListener('click', (e) => {
    if (!suppressNextClick) return;
    suppressNextClick = false;
    e.preventDefault();
    e.stopPropagation();
  }, true);

  updateUI();
}

export function initFilmstrips(root = document) {
  root.querySelectorAll('[data-role="filmstrip-wrap"]').forEach(initOne);
}
