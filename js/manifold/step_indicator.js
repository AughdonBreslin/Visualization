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
  const root = d3.select(container).append('div').attr('class', 'sp-frame');

  const panels = root.append('div').attr('class', 'sp-panels');
  const panelA = panels.append('div').attr('class', 'sp-panel');
  const headerA = panelA.append('div').attr('class', 'sp-panel-header');
  const barA = panelA.append('div').attr('class', 'sp-bar');
  const detailA = panelA.append('div').attr('class', 'sp-detail');

  const panelB = panels.append('div').attr('class', 'sp-panel');
  const headerB = panelB.append('div').attr('class', 'sp-panel-header');
  const barB = panelB.append('div').attr('class', 'sp-bar');
  const detailB = panelB.append('div').attr('class', 'sp-detail');

  const navRow = root.append('div').attr('class', 'sp-nav');
  const prevBtn = navRow.append('button').attr('class', 'step-prev').attr('type', 'button').text('◀ Prev');
  const descEl = navRow.append('div').attr('class', 'step-desc');
  const nextBtn = navRow.append('button').attr('class', 'step-next').attr('type', 'button').text('Next ▶');
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
        if (cid === nearest && cid === globalCurrent) rowClass += ' current';
        else if (cmp < 0) rowClass += ' past';
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
    headerA.text('A · ' + leftLabel);
    headerB.text('B · ' + rightLabel);

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
