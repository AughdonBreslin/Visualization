import { CANONICAL_STEPS, canonicalOf, compareSubSteps } from './canonical_steps.js';

function nearestSub(target, present) {
  if (present.includes(target)) return target;
  const sorted = [...present].sort(compareSubSteps);
  let best = sorted[0];
  for (const id of sorted) if (compareSubSteps(id, target) <= 0) best = id;
  return best;
}

// Dot state for one algorithm at one canonical stage. Matches the previous bar/detail logic
// (exact canonical-id presence), with the current stage called out distinctly.
function dotState(cid, presentSet, nearest) {
  if (!presentSet.has(cid)) return 'na';
  const cmp = compareSubSteps(cid, nearest);
  if (cmp < 0) return 'done';
  if (cmp === 0) return 'cur';
  return 'future';
}

export function createStepIndicator(container, { onJump }) {
  const d3 = window.d3;
  const root = d3.select(container);

  const tracks = root.select('.mf-tracks');
  const prevBtn = root.select('.step-prev');
  const nextBtn = root.select('.step-next');
  const descEl = root.select('.step-desc');

  prevBtn.on('click', () => onJump('prev'));
  nextBtn.on('click', () => onJump('next'));

  // One algorithm's row of dots aligned to the canonical stage axis. A step the algorithm skips
  // shows a small "na" dot and a dashed connector, so divergence reads at a glance.
  function appendRow(label, subSteps, nearest) {
    const presentSet = new Set(subSteps);
    tracks.append('div').attr('class', 'mf-tg-name').text(label);
    CANONICAL_STEPS.forEach((s, i) => {
      const st = dotState(s.id, presentSet, nearest);
      const prevSt = i > 0 ? dotState(CANONICAL_STEPS[i - 1].id, presentSet, nearest) : null;
      const naEdge = st === 'na' || prevSt === 'na';
      const cell = tracks.append('div')
        .attr('class', 'mf-tg-cell' + (i === 0 ? ' first' : '') + (naEdge ? ' na-edge' : ''));
      cell.append('span').attr('class', 'mf-tg-dot ' + st);
      if (st !== 'na') {
        cell.classed('clickable', true)
          .on('click', (event) => { event.stopPropagation(); onJump(s.id); });
      }
    });
  }

  function render({ leftLabel, rightLabel, leftSubSteps, rightSubSteps, currentSubStep }) {
    const leftNearest = nearestSub(currentSubStep, leftSubSteps);
    const rightNearest = nearestSub(currentSubStep, rightSubSteps);
    const curCanon = canonicalOf(currentSubStep);

    tracks.html('');
    // header row: corner + the canonical stage labels (current stage emphasized)
    tracks.append('div').attr('class', 'mf-tg-corner');
    CANONICAL_STEPS.forEach((s) => {
      tracks.append('div')
        .attr('class', 'mf-tg-stage' + (s.id === curCanon ? ' cur' : ''))
        .text(s.label);
    });
    appendRow(leftLabel, leftSubSteps, leftNearest);
    appendRow(rightLabel, rightSubSteps, rightNearest);

    const stepDef = CANONICAL_STEPS.find(s => s.id === curCanon);
    const inA = leftSubSteps.includes(currentSubStep);
    const inB = rightSubSteps.includes(currentSubStep);
    let who = '';
    if (inA && inB) who = leftLabel + ' and ' + rightLabel;
    else if (inA) who = leftLabel;
    else if (inB) who = rightLabel;
    descEl.text('Step ' + currentSubStep + ': ' + (stepDef ? stepDef.label : '') + (who ? ' · ' + who : ''));

    const all = [...new Set([...leftSubSteps, ...rightSubSteps])].sort(compareSubSteps);
    const idx = all.indexOf(currentSubStep);
    prevBtn.attr('disabled', idx <= 0 ? '' : null);
    nextBtn.attr('disabled', idx >= all.length - 1 ? '' : null);
  }

  return { render };
}
