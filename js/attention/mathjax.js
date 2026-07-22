// Typeset a DOM node with MathJax, retrying until MathJax has finished loading.
// MathJax v3 is loaded with `defer`, so window.MathJax may still be the bare
// config object (no typesetPromise) during the first few renders. We retry on a
// short interval so freshly injected math is rendered as soon as the library is ready.
// (Same approach as js/manifold/mathjax.js, duplicated rather than shared since the two
// pages' JS trees don't otherwise depend on each other.)
export function typesetMath(node) {
  if (!node) return;
  function attempt() {
    const mj = window.MathJax;
    if (mj && mj.typesetPromise) {
      mj.typesetPromise([node]).catch(() => {});
      return true;
    }
    return false;
  }
  if (attempt()) return;
  let tries = 0;
  const id = setInterval(() => {
    tries += 1;
    if (attempt() || tries > 40) clearInterval(id);
  }, 100);
}
