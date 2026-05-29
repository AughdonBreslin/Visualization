import { CANONICAL_STEPS, canonicalOf, compareSubSteps } from './canonical_steps.js';

function nearestSub(target, present) {
  if (present.includes(target)) return target;
  const sorted = [...present].sort(compareSubSteps);
  let best = sorted[0];
  for (const id of sorted) if (compareSubSteps(id, target) <= 0) best = id;
  return best;
}

export function createStepIndicator(container, { onJump }) {
  const d3 = window.d3;
  const root = d3.select(container);

  const barA = root.select('.sp-bar[data-side="a"]');
  const barB = root.select('.sp-bar[data-side="b"]');
  const detailA = root.select('.sp-detail[data-side="a"]');
  const detailB = root.select('.sp-detail[data-side="b"]');
  const prevBtn = root.select('.step-prev');
  const nextBtn = root.select('.step-next');
  const descEl = root.select('.step-desc');

  prevBtn.on('click', () => onJump('prev'));
  nextBtn.on('click', () => onJump('next'));

  function toggleExpanded() {
    const open = root.classed('is-expanded');
    root.classed('is-expanded', !open);
  }

  function classifyDot(stepId, presentSet, nearest) {
    if (!presentSet.has(stepId)) return 'na';
    return compareSubSteps(stepId, nearest) <= 0 ? 'filled' : 'hollow';
  }

  function renderBar(barSel, presentSubSteps, nearest) {
    barSel.html('');
    const presentSet = new Set(presentSubSteps);
    for (let i = 0; i < CANONICAL_STEPS.length; i++) {
      const cid = CANONICAL_STEPS[i].id;
      const state = classifyDot(cid, presentSet, nearest);
      const cell = barSel.append('div').attr('class', 'sp-cell' + (state === 'na' ? ' na' : ''));
      cell.append('span').attr('class', 'sp-dot ' + state);
      cell.append('span').attr('class', 'sp-num' + (state === 'na' ? ' na' : '')).text(cid);
      if (state !== 'na') cell.on('click', (event) => { event.stopPropagation(); onJump(cid); });
      if (i < CANONICAL_STEPS.length - 1) {
        barSel.append('div').attr('class', 'sp-edge').on('click', toggleExpanded);
      }
    }
  }

  function renderDetail(detailSel, presentSubSteps, nearest, globalCurrent) {
    detailSel.html('');
    const presentSet = new Set(presentSubSteps);
    for (let i = 0; i < CANONICAL_STEPS.length; i++) {
      const step = CANONICAL_STEPS[i];
      const cid = step.id;
      let rowClass = 'sp-step';
      let label;
      if (!presentSet.has(cid)) {
        rowClass += ' na';
        label = cid + ' · not used';
      } else {
        const cmp = compareSubSteps(cid, nearest);
        if (cmp < 0) rowClass += ' past';
        else if (cmp === 0) rowClass += ' current';
        else rowClass += ' future';
        label = cid + ' · ' + step.label;
      }
      const row = detailSel.append('div').attr('class', rowClass);
      row.append('span').attr('class', 'sp-step-dot');
      row.append('span').text(label);
      if (presentSet.has(cid)) row.on('click', () => onJump(cid));
    }
  }

  function render({ leftLabel, rightLabel, leftSubSteps, rightSubSteps, currentSubStep }) {
    const leftNearest = nearestSub(currentSubStep, leftSubSteps);
    const rightNearest = nearestSub(currentSubStep, rightSubSteps);

    renderBar(barA, leftSubSteps, leftNearest);
    renderBar(barB, rightSubSteps, rightNearest);
    renderDetail(detailA, leftSubSteps, leftNearest, currentSubStep);
    renderDetail(detailB, rightSubSteps, rightNearest, currentSubStep);

    const cid = canonicalOf(currentSubStep);
    const stepDef = CANONICAL_STEPS.find(s => s.id === cid);
    const inA = leftSubSteps.includes(currentSubStep);
    const inB = rightSubSteps.includes(currentSubStep);
    let who = '';
    if (inA && inB) who = leftLabel + ' and ' + rightLabel;
    else if (inA) who = leftLabel;
    else if (inB) who = rightLabel;
    descEl.text('Step ' + currentSubStep + ': ' + (stepDef ? stepDef.label : '') + ' - ' + who);

    const all = [...new Set([...leftSubSteps, ...rightSubSteps])].sort(compareSubSteps);
    const idx = all.indexOf(currentSubStep);
    prevBtn.attr('disabled', idx <= 0 ? '' : null);
    nextBtn.attr('disabled', idx >= all.length - 1 ? '' : null);
  }

  return { render };
}
