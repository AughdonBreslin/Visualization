export function createPseudocode(container, side) {
  const d3 = window.d3;
  const root = d3.select(container).append('div').attr('class', `pseudocode side-${side}`);
  const title = root.append('div').attr('class', 'pseudocode-title');
  const list = root.append('div').attr('class', 'pseudocode-sections');
  const expanded = new Map();

  function matchesStep(section, sub) {
    if (!section.steps) return false;
    return section.steps.includes(sub);
  }

  function render({ algoLabel, sections, currentSubStep }) {
    title.text(`${algoLabel} pseudocode`);
    list.selectAll('*').remove();
    sections.forEach((section, idx) => {
      const key = section.id || `${idx}`;
      const isCurrent = matchesStep(section, currentSubStep);
      if (!expanded.has(key)) expanded.set(key, isCurrent);
      if (isCurrent) expanded.set(key, true);
      const open = expanded.get(key);
      const sec = list.append('div').attr('class', `pc-section${isCurrent ? ' is-current' : ''}`);
      const header = sec.append('div').attr('class', 'pc-section-header').attr('role', 'button').attr('tabindex', '0');
      header.append('span').attr('class', 'pc-chevron').text(open ? '▾' : '▸');
      header.append('span').attr('class', 'pc-section-title').text(section.title);
      if (section.steps && section.steps.length) {
        header.append('span').attr('class', 'pc-section-steps').text(`step ${section.steps.join(', ')}`);
      }
      const toggle = () => { expanded.set(key, !expanded.get(key)); render({ algoLabel, sections, currentSubStep }); };
      header.on('click', toggle);
      header.on('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggle(); }
      });
      const body = sec.append('pre').attr('class', 'pc-section-body');
      if (open) body.text(section.lines.join('\n'));
      else body.style('display', 'none');
    });
  }

  return { render };
}
