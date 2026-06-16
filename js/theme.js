/* js/theme.js - user theming (accent + density), persisted in localStorage.
 * Loaded synchronously in the <head> so saved prefs apply before first paint.
 * Exposes window.UITheme for the settings page. */
(function () {
  var DEFAULT_ACCENT = '#6b7cff';
  var KEY_ACCENT = 'ui-accent';
  var KEY_DENSITY = 'ui-density';
  var root = document.documentElement;

  function hexToRgb(hex) {
    hex = String(hex).replace('#', '');
    if (hex.length === 3) hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    var n = parseInt(hex, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function lighten(c, t) {
    return [Math.round(c[0] + (255 - c[0]) * t), Math.round(c[1] + (255 - c[1]) * t), Math.round(c[2] + (255 - c[2]) * t)];
  }
  function isHex(s) { return /^#?[0-9a-fA-F]{3}$|^#?[0-9a-fA-F]{6}$/.test(String(s || '').trim()); }

  function applyAccent(hex) {
    if (!hex || !isHex(hex) || hex.toLowerCase() === DEFAULT_ACCENT) {
      root.style.removeProperty('--accent');
      root.style.removeProperty('--accent-muted');
      root.style.removeProperty('--accent-link');
      root.style.removeProperty('--focus-ring');
      return;
    }
    if (hex[0] !== '#') hex = '#' + hex;
    var c = hexToRgb(hex);
    var lk = lighten(c, 0.28);
    root.style.setProperty('--accent', hex);
    root.style.setProperty('--accent-muted', 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',0.14)');
    root.style.setProperty('--accent-link', 'rgb(' + lk[0] + ',' + lk[1] + ',' + lk[2] + ')');
    root.style.setProperty('--focus-ring', '2px solid rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',0.6)');
  }
  function applyDensity(d) {
    if (!d || d === 'balanced') root.removeAttribute('data-density');
    else root.setAttribute('data-density', d);
  }
  function get(key, def) { try { return localStorage.getItem(key) || def; } catch (e) { return def; } }
  function set(key, val) { try { localStorage.setItem(key, val); } catch (e) {} }

  function setAccent(hex) { applyAccent(hex); set(KEY_ACCENT, hex || DEFAULT_ACCENT); }
  function setDensity(d) { applyDensity(d); set(KEY_DENSITY, d || 'balanced'); }
  function reset() { setAccent(DEFAULT_ACCENT); setDensity('balanced'); }

  // boot: apply saved prefs immediately (before paint)
  applyAccent(get(KEY_ACCENT, DEFAULT_ACCENT));
  applyDensity(get(KEY_DENSITY, 'balanced'));

  window.UITheme = {
    DEFAULT_ACCENT: DEFAULT_ACCENT,
    applyAccent: applyAccent, applyDensity: applyDensity,
    setAccent: setAccent, setDensity: setDensity, reset: reset,
    current: function () { return { accent: get(KEY_ACCENT, DEFAULT_ACCENT), density: get(KEY_DENSITY, 'balanced') }; }
  };
})();
