// formulas_layout.js
// Makes side-by-side .formula blocks default to 50/50, but allows a single
// overflowing formula to take more width when its neighbor is shorter.

(function () {
  function isOverflowingX(el) {
    // Use a small epsilon to avoid flicker due to subpixel rounding.
    return el.scrollWidth - el.clientWidth > 1;
  }

  function updateFormulaRow(row) {
    const items = Array.from(row.querySelectorAll(':scope > .formula'));
    if (items.length < 2) return;

    items.forEach((el) => el.classList.remove('formula--wide'));

    const overflowFlags = items.map(isOverflowingX);
    const anyOverflow = overflowFlags.some(Boolean);
    if (!anyOverflow) return;

    const allOverflow = overflowFlags.every(Boolean);
    if (allOverflow) return;

    overflowFlags.forEach((isOverflowing, idx) => {
      if (isOverflowing) items[idx].classList.add('formula--wide');
    });
  }

  function updateAll() {
    document.querySelectorAll('.formulas').forEach(updateFormulaRow);
  }

  function scheduleUpdate() {
    // Two RAFs so layout (and MathJax) has time to settle.
    requestAnimationFrame(() => requestAnimationFrame(updateAll));
  }

  document.addEventListener('DOMContentLoaded', () => {
    scheduleUpdate();

    // If MathJax is present, wait for initial typesetting before measuring.
    const mj = window.MathJax;
    if (mj && mj.startup && mj.startup.promise && typeof mj.startup.promise.then === 'function') {
      mj.startup.promise.then(scheduleUpdate).catch(() => {
        // If MathJax fails, still keep the base 50/50 layout.
      });
    }

    window.addEventListener('resize', scheduleUpdate);
  });
})();
