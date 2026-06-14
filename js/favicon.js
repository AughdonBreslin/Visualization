// Sets the site favicon to a random manifold embedding thumbnail, chosen once
// per browser session (sessionStorage) so it changes when a new session starts.
// Thumbnails and their manifest live in assets/favicons/.
(function () {
  // Resolve assets/favicons/ from this script's own URL, falling back to the
  // page depth so it works from the root (index) and from /pages/ alike.
  var base = (function () {
    var s = document.currentScript;
    if (s && s.src) return s.src.replace(/js\/favicon\.js(\?.*)?$/, 'assets/favicons/');
    return (/\/pages\//.test(location.pathname) ? '../' : '') + 'assets/favicons/';
  })();

  function setFavicon(href) {
    var link = document.querySelector('link[rel~="icon"]');
    if (!link) {
      link = document.createElement('link');
      link.rel = 'icon';
      document.head.appendChild(link);
    }
    link.type = 'image/png';
    link.href = href;
  }

  var KEY = 'siteFaviconChoice';
  var stored = null;
  try { stored = sessionStorage.getItem(KEY); } catch (e) { /* storage blocked */ }

  // Reuse the session's existing pick straight away; only read the manifest when
  // a fresh choice is needed (first page of a new session).
  if (stored) { setFavicon(base + stored); return; }

  fetch(base + 'manifest.json')
    .then(function (r) { return r.json(); })
    .then(function (list) {
      if (!Array.isArray(list) || !list.length) return;
      var choice = list[Math.floor(Math.random() * list.length)];
      try { sessionStorage.setItem(KEY, choice); } catch (e) { /* storage blocked */ }
      setFavicon(base + choice);
    })
    .catch(function () { /* leave the default favicon */ });
})();
