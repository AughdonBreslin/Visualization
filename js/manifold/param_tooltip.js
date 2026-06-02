// Single shared floating tooltip for parameter info icons.
let tipEl = null;

function ensureTip() {
  if (tipEl) return tipEl;
  tipEl = document.createElement('div');
  tipEl.className = 'mf-tooltip';
  document.body.appendChild(tipEl);
  return tipEl;
}

export function attachTooltip(iconEl, { label, desc, rangeText }) {
  iconEl.addEventListener('mousemove', (e) => {
    const tip = ensureTip();
    let html = '';
    if (label) html += '<span class="mf-tooltip-title">' + label + '</span>';
    html += desc || '';
    if (rangeText) html += '<div class="mf-tooltip-range">' + rangeText + '</div>';
    tip.innerHTML = html;
    tip.style.opacity = '1';
    let x = e.clientX + 14;
    let y = e.clientY + 14;
    if (x + 300 > window.innerWidth) x = e.clientX - 304;
    if (y + 130 > window.innerHeight) y = e.clientY - 130;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  });
  iconEl.addEventListener('mouseleave', () => {
    if (tipEl) tipEl.style.opacity = '0';
  });
}
