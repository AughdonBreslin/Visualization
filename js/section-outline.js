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
