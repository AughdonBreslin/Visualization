/* collapsible.js - shared, mobile-only collapsible sections.
 *
 * Any element with class "collapsible" whose first child is an h2/h3 heading
 * becomes a tap-to-toggle section on phones (<=640px). On wider screens the
 * body is always shown and there is no toggle (the CSS affordance is gated to
 * phone width in responsive.css), so desktop/tablet behaviour is unchanged.
 *
 * Sections collapse by default on phones; add data-open-mobile to keep one open.
 * Expanding fires a window resize event so charts that redraw on resize (the
 * D3 visualizers) lay out correctly when revealed from a hidden container.
 */
(function () {
  const PHONE = window.matchMedia("(max-width: 640px)");

  function wire(el) {
    if (el.dataset.collapsibleReady) return;
    const head = el.querySelector(":scope > h2, :scope > h3");
    if (!head) {
      // No heading to toggle: drop the class so the CSS never hides its content
      // (otherwise a headless collapsible would render empty on mobile).
      el.classList.remove("collapsible");
      return;
    }
    el.dataset.collapsibleReady = "1";
    head.classList.add("collapsible-head");
    head.setAttribute("role", "button");
    head.setAttribute("tabindex", "0");

    const toggle = () => {
      const open = el.classList.toggle("open");
      head.setAttribute("aria-expanded", String(open));
      if (open) {
        // Revealing a previously-hidden container: refit resize-driven charts,
        // and (re)typeset any math that could not lay out while hidden.
        window.dispatchEvent(new Event("resize"));
        const MJ = window.MathJax;
        if (MJ && typeof MJ.typesetPromise === "function") {
          MJ.typesetPromise([el]).catch(function () {});
        }
        // Let page code react to a section being opened (e.g. kick off work
        // that is deferred while the section is collapsed).
        el.dispatchEvent(new CustomEvent("collapsible:open", { bubbles: true }));
      }
    };

    head.addEventListener("click", toggle);
    head.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggle();
      }
    });
  }

  function setDefaults() {
    document.querySelectorAll(".collapsible").forEach((el) => {
      wire(el);
      if (!el.classList.contains("collapsible")) return; // headless: left visible
      const open = !PHONE.matches || el.hasAttribute("data-open-mobile");
      el.classList.toggle("open", open);
      const head = el.querySelector(":scope > .collapsible-head");
      if (head) head.setAttribute("aria-expanded", String(open));
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setDefaults);
  } else {
    setDefaults();
  }
  // Re-apply the default open/closed state when crossing the phone breakpoint.
  PHONE.addEventListener("change", setDefaults);
})();
