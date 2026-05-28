const TABS = [
  { key: 'intuition', label: 'Intuition' },
  { key: 'formula', label: 'Formula' },
  { key: 'worked', label: 'Worked example' },
];

export function createIFW(container, side) {
  const d3 = window.d3;
  const root = d3.select(container).append('div').attr('class', `ifw side-${side}`);
  const tabRow = root.append('div').attr('class', 'ifw-tabs');
  const content = root.append('div').attr('class', 'ifw-content');

  let active = 'intuition';
  let current = { intuition: null, formula: null, worked: null };
  const buttons = {};
  for (const { key, label } of TABS) {
    const btn = tabRow.append('button').attr('type', 'button').attr('class', `ifw-tab tab-${key}`).text(label);
    btn.on('click', () => {
      if (!current[key]) return;
      active = key; render();
    });
    buttons[key] = btn;
  }

  function render() {
    for (const { key } of TABS) {
      const has = !!current[key];
      buttons[key].attr('disabled', has ? null : '').classed('is-active', active === key && has).classed('is-disabled', !has);
    }
    const html = current[active] || '<div class="ifw-empty">No content for this step.</div>';
    content.html(html);
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([content.node()]).catch(() => {});
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
