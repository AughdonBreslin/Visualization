/* js/theme.js - user theming (accent + density), persisted in localStorage.
 * Loaded synchronously in the <head> so saved prefs apply before first paint.
 * Exposes window.UITheme for the settings page. */
(function () {
  var DEFAULT_ACCENT = '#6b7cff';
  var KEY_ACCENT = 'ui-accent';
  var KEY_DENSITY = 'ui-density';
  var KEY_LINKUL = 'ui-link-underline';
  var KEY_TRIBUTE = 'ui-tribute';
  var KEY_BRIGHTNESS = 'ui-brightness';
  // Text brightness: level 0 = the base token colors, higher lifts the dim grays toward white.
  // Default is raised above 0 so body/muted text reads more easily on dimmer displays.
  var DEFAULT_BRIGHTNESS = 50;
  var TEXT_2_BASE = [207, 209, 216];    // #cfd1d8, matches tokens.css --text-2
  var TEXT_BODY_BASE = [139, 141, 150]; // #8b8d96, matches tokens.css --text-body
  var TEXT_MUTED_BASE = [93, 95, 104];  // #5d5f68, matches tokens.css --text-muted
  var MONOCRAFT = "'Monocraft', ui-monospace, SFMono-Regular, monospace";
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
      root.style.removeProperty('--accent-rgb');
      root.style.removeProperty('--accent-muted');
      root.style.removeProperty('--accent-link');
      root.style.removeProperty('--focus-ring');
      return;
    }
    if (hex[0] !== '#') hex = '#' + hex;
    var c = hexToRgb(hex);
    var lk = lighten(c, 0.28);
    var rgb = c[0] + ', ' + c[1] + ', ' + c[2];
    root.style.setProperty('--accent', hex);
    // The triplet drives every rgba(var(--accent-rgb), a) rule (link underlines/hover, inline
    // code, toggle, focus ring), so a custom accent recolors them too.
    root.style.setProperty('--accent-rgb', rgb);
    root.style.setProperty('--accent-muted', 'rgba(' + rgb + ', 0.14)');
    root.style.setProperty('--accent-link', 'rgb(' + lk[0] + ', ' + lk[1] + ', ' + lk[2] + ')');
    root.style.setProperty('--focus-ring', '2px solid rgba(' + rgb + ', 0.6)');
  }
  function applyDensity(d) {
    if (!d || d === 'balanced') root.removeAttribute('data-density');
    else root.setAttribute('data-density', d);
  }
  function applyLinkUnderline(on) {
    var enabled = on === true || on === '1' || on === 'true';
    if (enabled) root.style.setProperty('--link-ul', '1px');
    else root.style.removeProperty('--link-ul');
  }
  // Tribute mode: convert the whole site to Monocraft by overriding both font tokens.
  // data-font lets CSS make pixel-font-specific tweaks (e.g. drop negative tracking).
  function applyTribute(on) {
    var enabled = on === true || on === '1' || on === 'true';
    if (enabled) {
      root.style.setProperty('--font-sans', MONOCRAFT);
      root.style.setProperty('--font-mono', MONOCRAFT);
      root.setAttribute('data-font', 'monocraft');
    } else {
      root.style.removeProperty('--font-sans');
      root.style.removeProperty('--font-mono');
      root.removeAttribute('data-font');
    }
  }
  function clampLevel(v) { v = parseInt(v, 10); if (isNaN(v)) return DEFAULT_BRIGHTNESS; return Math.max(0, Math.min(100, v)); }
  function rgbStr(c) { return 'rgb(' + c[0] + ', ' + c[1] + ', ' + c[2] + ')'; }
  function applyBrightness(level) {
    level = clampLevel(level);
    if (level === 0) {
      root.style.removeProperty('--text-2');
      root.style.removeProperty('--text-body');
      root.style.removeProperty('--text-muted');
      return;
    }
    var t = (level / 100) * 0.45;
    root.style.setProperty('--text-2', rgbStr(lighten(TEXT_2_BASE, t)));
    root.style.setProperty('--text-body', rgbStr(lighten(TEXT_BODY_BASE, t)));
    root.style.setProperty('--text-muted', rgbStr(lighten(TEXT_MUTED_BASE, t)));
  }
  function get(key, def) { try { return localStorage.getItem(key) || def; } catch (e) { return def; } }
  function set(key, val) { try { localStorage.setItem(key, val); } catch (e) {} }

  function setAccent(hex) { applyAccent(hex); set(KEY_ACCENT, hex || DEFAULT_ACCENT); }
  function setDensity(d) { applyDensity(d); set(KEY_DENSITY, d || 'balanced'); }
  function setLinkUnderline(on) { applyLinkUnderline(on); set(KEY_LINKUL, on ? '1' : '0'); }
  function setTribute(on) { applyTribute(on); set(KEY_TRIBUTE, on ? '1' : '0'); }
  function setBrightness(level) { level = clampLevel(level); applyBrightness(level); set(KEY_BRIGHTNESS, String(level)); }
  function reset() { setAccent(DEFAULT_ACCENT); setDensity('balanced'); setLinkUnderline(false); setTribute(false); setBrightness(DEFAULT_BRIGHTNESS); }

  // boot: apply saved prefs immediately (before paint)
  applyAccent(get(KEY_ACCENT, DEFAULT_ACCENT));
  applyDensity(get(KEY_DENSITY, 'balanced'));
  applyLinkUnderline(get(KEY_LINKUL, '0'));
  applyTribute(get(KEY_TRIBUTE, '0'));
  applyBrightness(get(KEY_BRIGHTNESS, String(DEFAULT_BRIGHTNESS)));

  window.UITheme = {
    DEFAULT_ACCENT: DEFAULT_ACCENT,
    DEFAULT_BRIGHTNESS: DEFAULT_BRIGHTNESS,
    applyAccent: applyAccent, applyDensity: applyDensity, applyLinkUnderline: applyLinkUnderline,
    applyTribute: applyTribute, applyBrightness: applyBrightness,
    setAccent: setAccent, setDensity: setDensity, setLinkUnderline: setLinkUnderline, setTribute: setTribute,
    setBrightness: setBrightness,
    reset: reset,
    current: function () {
      return {
        accent: get(KEY_ACCENT, DEFAULT_ACCENT),
        density: get(KEY_DENSITY, 'balanced'),
        linkUnderline: get(KEY_LINKUL, '0') === '1',
        tribute: get(KEY_TRIBUTE, '0') === '1',
        brightness: clampLevel(get(KEY_BRIGHTNESS, String(DEFAULT_BRIGHTNESS)))
      };
    }
  };
})();
