---
title: Accent Swatch Management
date: 2026-06-25
status: approved
---

# Accent Swatch Management

## Overview

Add the ability to remove any accent color swatch from the settings page by hovering over it and clicking a small x button.
The swatch list -- both presets and user-added custom colors -- is unified into a single localStorage key, seeded from the five built-in defaults on first visit.

## Scope

- Remove-on-hover x button for every swatch (presets and custom)
- Unified swatch list in localStorage, rendered entirely by JS
- No export/import; localStorage is the sole persistence layer

---

## Swatch Row UX

Each swatch is wrapped in a `<div class="swatch-wrap">` with `position: relative`.
Inside the wrapper: the existing `<button class="swatch">` (select) and a new `<button class="swatch-remove" aria-label="Remove">` (delete).

The remove button is absolutely positioned at the top-right corner of the wrapper.
It is invisible by default and becomes visible on `wrapper:hover` via CSS (`opacity: 0` → `opacity: 1`, pointer-events toggled to match).
Clicking the remove button removes the swatch from the list and re-renders the row.

The five preset `<button class="swatch">` elements are removed from the HTML.
All swatches are rendered by JS on page load from the unified list.

---

## Storage

Single localStorage key: `ui-accent-swatches`

Value: JSON array of hex strings, e.g. `["#6b7cff","#4aa3ff","#5ec99a","#d9a441","#8a9bd6"]`

### Initialization (first visit)

When `ui-accent-swatches` is absent from localStorage, write the five defaults:

```
["#6b7cff", "#4aa3ff", "#5ec99a", "#d9a441", "#8a9bd6"]
```

### Migration (returning user with old custom swatches)

The old format of `ui-accent-swatches` stored only user-added custom colors (no presets).
To detect this: if the stored array is non-empty and contains none of the five default hex values, treat it as old-format and prepend the five defaults, deduplicated by lowercase hex.
This is a one-time migration; after it runs the array will contain at least one default, so the condition won't trigger again.

### Add

Adding a custom color (via hex field + Enter) appends the hex to the array if not already present, then saves.

### Remove

Removing any swatch splices it from the array by lowercase hex match, then saves.
If the removed swatch is the currently active accent, the active accent is not changed (the color stays applied; the swatch is just no longer pinned).

### Reset

`UITheme.reset()` restores `ui-accent-swatches` to the five defaults and re-renders the row.

---

## CSS

```css
.swatch-wrap {
  position: relative;
  display: inline-block;
}

.swatch-remove {
  position: absolute;
  top: -4px;
  right: -4px;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--bg-2);
  border: 1px solid var(--hairline-strong);
  color: var(--text-muted);
  font-size: 10px;
  line-height: 1;
  padding: 0;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  pointer-events: none;
  transition: opacity 120ms ease;
}

.swatch-wrap:hover .swatch-remove {
  opacity: 1;
  pointer-events: auto;
}
```

---

## JS Changes (`pages/settings.html` inline script)

- `SWATCH_KEY` stays `'ui-accent-swatches'`
- `DEFAULT_SWATCHES` constant: `['#6b7cff', '#4aa3ff', '#5ec99a', '#d9a441', '#8a9bd6']`
- `loadSwatches()`: reads the key; if absent, seeds with defaults; if present and a migration is needed (array contains no defaults), prepends defaults deduplicated
- `saveSwatches(arr)`: serializes and writes
- `renderSwatches()`: clears and rebuilds the swatch DOM inside `.accent-row` (before `.custom-accent`), creating `.swatch-wrap` + `.swatch` + `.swatch-remove` for each entry
- `removeSwatch(hex)`: splices from array, saves, re-renders
- `addCustomSwatch(hex)`: appends if absent, saves, re-renders

The existing `makeSwatch` and `addCustomSwatch` functions are replaced by the above.
`wireSwatch` and `swatchExists` are removed; logic folds into `renderSwatches` and `addCustomSwatch`.

---

## Reset behavior

`T.reset()` call in the reset handler additionally calls `saveSwatches(DEFAULT_SWATCHES)` and `renderSwatches()` to restore the preset list.

---

## Out of scope

- Export/import of preferences
- Touch-specific affordance for swatch removal (desktop hover is sufficient)
- Reordering swatches
