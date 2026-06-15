// js/section-outline.js
// On-page section outline: a sticky rail in a reserved left column on desktop,
// a hamburger drawer on narrow/phone widths. Auto-generated from each page's
// top-level .panel blocks. Pure helpers are exported for unit testing; the DOM
// build self-initializes only when a document is present (so node:test can
// import the helpers without a DOM).

export function slugify(text) {
  const s = String(text)
    .toLowerCase()
    .trim()
    .replace(/[^\w]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return s || 'section';
}

export function uniqueId(base, used) {
  let id = base;
  let n = 2;
  while (used.has(id)) {
    id = `${base}-${n}`;
    n += 1;
  }
  used.add(id);
  return id;
}

export function normalizeLabel(text) {
  return String(text).replace(/\s+/g, ' ').trim();
}

function openDrawer(ui) {
  ui.nav.classList.add('open');
  ui.btn.setAttribute('aria-expanded', 'true');
  ui.btn.setAttribute('aria-label', 'Close section outline');
  ui.backdrop.hidden = false;
  const first = ui.list.querySelector('a');
  if (first) first.focus();
}

function closeDrawer(ui) {
  if (!ui || !ui.nav.classList.contains('open')) return;
  ui.nav.classList.remove('open');
  ui.btn.setAttribute('aria-expanded', 'false');
  ui.btn.setAttribute('aria-label', 'Open section outline');
  ui.backdrop.hidden = true;
  ui.btn.focus();
}

// ---- DOM build (skipped under node:test where document is undefined) ----

function collectPanels(root) {
  const panels = Array.from(root.querySelectorAll('.panel')).filter(
    (el) => !el.parentElement || !el.parentElement.closest('.panel')
  );
  const used = new Set(Array.from(root.querySelectorAll('[id]')).map((el) => el.id));
  const entries = [];
  for (const panel of panels) {
    const heading = panel.querySelector(':scope > h2, :scope > h3');
    if (!heading) continue;
    const label = normalizeLabel(heading.textContent);
    if (!label) continue;
    if (panel.id) used.add(panel.id);
    else panel.id = uniqueId(slugify(label), used);
    panel.style.scrollMarginTop = 'var(--outline-scroll-offset, 16px)';
    entries.push({ id: panel.id, label, panel });
  }
  return entries;
}

function buildNav(entries) {
  const nav = document.createElement('nav');
  nav.className = 'section-outline';
  nav.setAttribute('aria-label', 'On this page');

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'section-outline-toggle';
  btn.setAttribute('aria-label', 'Open section outline');
  btn.setAttribute('aria-expanded', 'false');
  btn.setAttribute('aria-controls', 'section-outline-panel');
  btn.innerHTML = '<span class="section-outline-bars" aria-hidden="true"></span>';

  const backdrop = document.createElement('div');
  backdrop.className = 'section-outline-backdrop';
  backdrop.hidden = true;

  const panel = document.createElement('div');
  panel.className = 'section-outline-panel';
  panel.id = 'section-outline-panel';

  const heading = document.createElement('div');
  heading.className = 'section-outline-heading';
  heading.textContent = 'On this page';
  panel.appendChild(heading);

  const list = document.createElement('ul');
  list.className = 'section-outline-list';
  const linkById = new Map();
  for (const entry of entries) {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = `#${entry.id}`;
    a.textContent = entry.label;
    a.dataset.target = entry.id;
    li.appendChild(a);
    list.appendChild(li);
    linkById.set(entry.id, a);
  }
  panel.appendChild(list);

  nav.append(btn, backdrop, panel);
  return { nav, btn, backdrop, panel, list, linkById };
}

function isCollapsedPanel(panel) {
  return panel.classList.contains('collapsible') && !panel.classList.contains('open');
}

function openIfCollapsed(panel) {
  // Reuse collapsible.js: clicking its head toggles open (and fires resize +
  // MathJax retypeset). Only click when actually collapsed to avoid closing it.
  if (!isCollapsedPanel(panel)) return;
  const head = panel.querySelector(':scope > .collapsible-head');
  if (head) head.click();
}

function prefersReducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function scrollToPanel(panel) {
  panel.scrollIntoView({
    behavior: prefersReducedMotion() ? 'auto' : 'smooth',
    block: 'start',
  });
}

function navigateTo(id) {
  const panel = document.getElementById(id);
  if (!panel) return;
  openIfCollapsed(panel);
  // Defer scroll one frame so a just-opened panel has its final height.
  requestAnimationFrame(() => scrollToPanel(panel));
  history.replaceState(null, '', `#${id}`);
}

function wireScrollspy(entries, linkById) {
  let activeId = null;
  const setActive = (id) => {
    if (id === activeId) return;
    if (activeId && linkById.get(activeId)) {
      linkById.get(activeId).classList.remove('active');
      linkById.get(activeId).removeAttribute('aria-current');
    }
    activeId = id;
    const a = linkById.get(id);
    if (a) {
      a.classList.add('active');
      a.setAttribute('aria-current', 'location');
    }
  };
  const visible = new Set();
  const order = entries.map((e) => e.id);
  const obs = new IntersectionObserver(
    (records) => {
      for (const r of records) {
        if (r.isIntersecting) visible.add(r.target.id);
        else visible.delete(r.target.id);
      }
      // Highlight the visible panel that appears earliest in document order.
      const top = order.find((id) => visible.has(id));
      if (top) setActive(top);
    },
    { rootMargin: '-10% 0px -70% 0px', threshold: 0 }
  );
  for (const e of entries) obs.observe(e.panel);
}

function initSectionOutline() {
  const entries = collectPanels(document);
  if (entries.length < 2) return; // not worth an outline
  const ui = buildNav(entries);
  document.body.appendChild(ui.nav);
  document.body.classList.add('has-section-outline');

  ui.list.addEventListener('click', (e) => {
    const a = e.target.closest('a[data-target]');
    if (!a) return;
    e.preventDefault();
    navigateTo(a.dataset.target);
    closeDrawer(ui); // selecting an entry also closes the mobile drawer
  });

  // Deep link on load: open + scroll to the hashed panel after layout settles.
  if (location.hash.length > 1) {
    const id = decodeURIComponent(location.hash.slice(1));
    requestAnimationFrame(() => navigateTo(id));
  }

  ui.btn.addEventListener('click', () => {
    if (ui.nav.classList.contains('open')) closeDrawer(ui);
    else openDrawer(ui);
  });
  ui.backdrop.addEventListener('click', () => closeDrawer(ui));
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeDrawer(ui);
  });

  wireScrollspy(entries, ui.linkById);

  return { entries, ui };
}

function ready(fn) {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', fn);
  } else {
    fn();
  }
}

if (typeof document !== 'undefined') {
  ready(initSectionOutline);
}
