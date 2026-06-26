const TABS = [
  { key: 'intuition', label: 'Intuition' },
  { key: 'formula', label: 'Formula' },
  { key: 'worked', label: 'Worked example' },
];

import { typesetMath } from './mathjax.js';

export function createIFW(container, side) {
  const d3 = window.d3;
  const root = d3.select(container).append('div').attr('class', `ifw side-${side}`);
  const tabRow = root.append('div').attr('class', 'ifw-tabs');
  const contentWrap = root.append('div').attr('class', 'ifw-content');

  let active = 'intuition';
  let current = { intuition: null, formula: null, worked: null };

  // One persistent div per tab. Content is only replaced (and re-typeset) when the HTML
  // changes; switching tabs just toggles display so MathJax SVGs are never thrown away.
  const tabDivs = {};
  const tabLastHtml = {};
  for (const { key } of TABS) {
    tabDivs[key] = contentWrap.append('div').attr('class', 'ifw-tab-content').style('display', 'none');
    tabLastHtml[key] = null;
  }

  const buttons = {};
  for (const { key, label } of TABS) {
    const btn = tabRow.append('button').attr('type', 'button').attr('class', `ifw-tab tab-${key}`).text(label);
    btn.on('click', () => {
      if (!current[key]) return;
      active = key;
      render();
    });
    buttons[key] = btn;
  }

  function render() {
    for (const { key } of TABS) {
      const has = !!current[key];
      buttons[key].attr('disabled', has ? null : '').classed('is-active', active === key && has).classed('is-disabled', !has);
    }
    for (const { key } of TABS) {
      const html = current[key];
      const div = tabDivs[key];

      if (html !== tabLastHtml[key]) {
        if (html) {
          div.html(html);
          typesetMath(div.node());
        } else {
          div.html('<div class="ifw-empty">No content for this step.</div>');
        }
        tabLastHtml[key] = html;
      }

      div.style('display', key === active ? null : 'none');
    }
  }

  function setStep(ifw) {
    current = {
      intuition: (ifw && ifw.intuition) || null,
      formula: (ifw && ifw.formula) || null,
      worked: (ifw && ifw.worked) || null,
    };
    if (!current[active]) {
      const next = TABS.find(t => current[t.key]);
      if (next) active = next.key;
    }
    render();
  }

  return { setStep };
}
