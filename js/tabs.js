/* tabs.js - simple, accessible tab panels (shared).
 *
 * Markup: a `.tabs` bar containing buttons with data-tab="name", and sibling
 * elements with data-panel="name". Clicking a tab shows its panel and hides the
 * others. Mark one button .active (or the first is used) for the initial tab.
 */
(function () {
  function initBar(bar) {
    const buttons = Array.from(bar.querySelectorAll("[data-tab]"));
    const scope = bar.parentElement;
    const panels = Array.from(scope.querySelectorAll("[data-panel]"));
    if (!buttons.length || !panels.length) return;

    function activate(name) {
      buttons.forEach((b) => {
        const on = b.dataset.tab === name;
        b.classList.toggle("active", on);
        b.setAttribute("aria-selected", on ? "true" : "false");
        b.setAttribute("tabindex", on ? "0" : "-1");
      });
      panels.forEach((p) => {
        p.hidden = p.dataset.panel !== name;
      });
    }

    buttons.forEach((b) => {
      b.setAttribute("role", "tab");
      b.addEventListener("click", () => activate(b.dataset.tab));
    });

    const initial = buttons.find((b) => b.classList.contains("active")) || buttons[0];
    activate(initial.dataset.tab);
  }

  function init() {
    document.querySelectorAll(".tabs").forEach(initBar);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
