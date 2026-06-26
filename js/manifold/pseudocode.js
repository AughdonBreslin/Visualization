import { typesetMath } from './mathjax.js';

export function createPseudocode(container, side) {
  const d3 = window.d3;
  const root = d3.select(container).append('div').attr('class', `pseudocode side-${side}`);
  const title = root.append('div').attr('class', 'pseudocode-title');
  const list = root.append('div').attr('class', 'pseudocode-sections');
  const expanded = new Map();

  let lastAlgoLabel = null;
  let lastSections = null;
  let sectionRefs = [];
  let currentSub = null;

  function matchesStep(section, sub) {
    if (!section.steps) return false;
    return section.steps.includes(sub);
  }

  function applyBodyState(ref) {
    const open = expanded.get(ref.key);
    ref.chevron.text(open ? '▾' : '▸');
    if (open) {
      if (!ref.rendered) {
        for (const line of ref.section.lines) {
          ref.body.append('div').attr('class', 'pc-line').html(line);
        }
        typesetMath(ref.body.node());
        ref.rendered = true;
      }
      ref.body.style('display', null);
    } else {
      ref.body.style('display', 'none');
    }
  }

  function rebuild(algoLabel, sections, sub) {
    lastAlgoLabel = algoLabel;
    lastSections = sections;
    expanded.clear();
    list.selectAll('*').remove();
    sectionRefs = [];

    sections.forEach((section, idx) => {
      const key = section.id || `${idx}`;
      const isCurrent = matchesStep(section, sub);
      expanded.set(key, isCurrent);

      const sec = list.append('div').attr('class', `pc-section${isCurrent ? ' is-current' : ''}`);
      const header = sec.append('div').attr('class', 'pc-section-header').attr('role', 'button').attr('tabindex', '0');
      const chevron = header.append('span').attr('class', 'pc-chevron');
      header.append('span').attr('class', 'pc-section-title').text(section.title);
      if (section.steps && section.steps.length) {
        header.append('span').attr('class', 'pc-section-steps').text(`step ${section.steps.join(', ')}`);
      }
      const body = sec.append('div').attr('class', 'pc-section-body');
      const ref = { key, section, sec, chevron, body, rendered: false };
      sectionRefs.push(ref);

      const toggle = () => {
        expanded.set(key, !expanded.get(key));
        applyBodyState(ref);
      };
      header.on('click', toggle);
      header.on('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggle(); }
      });
      applyBodyState(ref);
    });
  }

  function render({ algoLabel, sections, currentSubStep }) {
    currentSub = currentSubStep;
    title.text(`${algoLabel} pseudocode`);

    if (algoLabel !== lastAlgoLabel || sections !== lastSections) {
      rebuild(algoLabel, sections, currentSubStep);
      return;
    }

    // Same algorithm: update is-current class and expand the newly active section without
    // rebuilding DOM or re-typesetting already-rendered bodies.
    sectionRefs.forEach(ref => {
      const isCurrent = matchesStep(ref.section, currentSubStep);
      ref.sec.classed('is-current', isCurrent);
      if (isCurrent && !expanded.get(ref.key)) {
        expanded.set(ref.key, true);
        applyBodyState(ref);
      }
    });
  }

  return { render };
}
