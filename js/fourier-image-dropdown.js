/* fourier-image-dropdown.js - custom Image dropdown for the Fourier demo.
 *
 * A native <select> cannot open the file picker on iOS (a programmatic click from a change
 * handler is blocked); only a direct tap on a label-wrapped file input works. This shim keeps the
 * real <select id="fourierPreset"> in the DOM (fourier.js reads its value, listens for its
 * change, and appends an "__uploaded__" option), hides it, and renders a custom dropdown whose
 * "Upload image..." item is a <label> wrapping the real <input type="file" id="fourierUpload">.
 * fourier.js is never modified. */
(function () {
  function init() {
    const select = document.getElementById("fourierPreset");
    const group = select && select.closest(".fourier-image-group");
    const upload = document.getElementById("fourierUpload");
    if (!select || !group || !upload) return; // markup not present: no-op

    const wrap = document.createElement("div");
    wrap.className = "fourier-dd-wrap";

    const trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "fourier-dd-trigger";
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    trigger.innerHTML =
      '<span class="fourier-dd-value"></span>' +
      '<svg class="chev" viewBox="0 0 9 6" aria-hidden="true"><path d="M1 1l3.5 3.5L8 1" fill="none" stroke="#7a7c84" stroke-width="1.3"/></svg>';

    const menu = document.createElement("div");
    menu.className = "fourier-dd-menu";
    menu.setAttribute("role", "listbox");
    menu.hidden = true;

    // The upload label wraps the real file input (moved out of static HTML).
    const uploadLabel = document.createElement("label");
    uploadLabel.className = "fourier-dd-item fourier-dd-upload";
    uploadLabel.textContent = "Upload image…"; // "Upload image..." with an ellipsis char
    upload.removeAttribute("hidden");
    uploadLabel.appendChild(upload);

    wrap.appendChild(trigger);
    wrap.appendChild(menu);
    // Place the custom UI right after the hidden select.
    select.insertAdjacentElement("afterend", wrap);

    const valueEl = trigger.querySelector(".fourier-dd-value");

    function currentText() {
      const opt = select.options[select.selectedIndex];
      return opt ? opt.textContent : "";
    }

    function renderMenu() {
      // Rebuild preset items from the select's current options, then the separator + upload.
      Array.from(menu.querySelectorAll(".fourier-dd-item:not(.fourier-dd-upload), .fourier-dd-sep"))
        .forEach((n) => n.remove());
      const frag = document.createDocumentFragment();
      Array.from(select.options).forEach((opt) => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "fourier-dd-item" + (opt.selected ? " sel" : "");
        item.setAttribute("role", "option");
        item.textContent = opt.textContent;
        item.addEventListener("click", () => {
          select.value = opt.value;
          select.dispatchEvent(new Event("change", { bubbles: true }));
          close();
        });
        frag.appendChild(item);
      });
      const sep = document.createElement("div");
      sep.className = "fourier-dd-sep";
      frag.appendChild(sep);
      menu.insertBefore(frag, uploadLabel);
    }

    function syncTrigger() {
      valueEl.textContent = currentText();
    }

    function open() {
      renderMenu();
      menu.hidden = false;
      trigger.setAttribute("aria-expanded", "true");
      document.addEventListener("mousedown", onOutside);
      document.addEventListener("keydown", onKey);
    }
    function close() {
      menu.hidden = true;
      trigger.setAttribute("aria-expanded", "false");
      document.removeEventListener("mousedown", onOutside);
      document.removeEventListener("keydown", onKey);
    }
    function onOutside(e) { if (!wrap.contains(e.target)) close(); }
    function onKey(e) { if (e.key === "Escape") { close(); trigger.focus(); } }

    trigger.addEventListener("click", () => { menu.hidden ? open() : close(); });

    // Keep the trigger (and an open menu) in sync when fourier.js mutates the select after an
    // upload: it sets select.value = "__uploaded__" and appends an option without firing change.
    select.addEventListener("change", () => { syncTrigger(); if (!menu.hidden) renderMenu(); });
    new MutationObserver(() => { syncTrigger(); if (!menu.hidden) renderMenu(); })
      .observe(select, { childList: true, attributes: true, attributeFilter: ["value"] });
    // The upload sets select.value programmatically (no change event, no attribute mutation),
    // so also resync after the file input finishes.
    upload.addEventListener("change", () => { setTimeout(syncTrigger, 0); });

    syncTrigger();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
