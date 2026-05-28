import { CANONICAL_STEPS, canonicalOf, compareSubSteps } from './canonical_steps.js';

export function createStepIndicator(container, { onJump }) {
  const d3 = window.d3;
  const root = d3.select(container).append('div').attr('class', 'step-indicator');
  const labelRow = root.append('div').attr('class', 'step-label-row');
  const labelA = labelRow.append('div').attr('class', 'step-label-a');
  const labelB = labelRow.append('div').attr('class', 'step-label-b');
  const svgWrap = root.append('div').attr('class', 'step-rails-wrap');
  const svg = svgWrap.append('svg').attr('class', 'step-rails').attr('preserveAspectRatio', 'xMidYMid meet');
  const navRow = root.append('div').attr('class', 'step-nav');
  const prevBtn = navRow.append('button').attr('class', 'step-prev').attr('type', 'button').text('◀ Prev');
  const desc = navRow.append('div').attr('class', 'step-desc');
  const nextBtn = navRow.append('button').attr('class', 'step-next').attr('type', 'button').text('Next ▶');
  prevBtn.on('click', () => onJump('prev'));
  nextBtn.on('click', () => onJump('next'));

  function groupByCanonical(subSteps) {
    const out = {};
    for (const id of subSteps) { const cid = canonicalOf(id); (out[cid] ||= []).push(id); }
    for (const cid in out) out[cid].sort(compareSubSteps);
    return out;
  }

  function drawColumnSide(svg, x, railY, dir, ids, current, side, gap, canonicalLabel) {
    if (ids.length === 0) {
      svg.append('circle').attr('cx', x).attr('cy', railY).attr('r', 4)
        .attr('fill', 'transparent').attr('stroke', 'rgba(255,255,255,0.18)')
        .attr('stroke-width', 1).attr('stroke-dasharray', '2,2');
      return;
    }
    let from = railY;
    for (let k = 0; k < ids.length; k++) {
      const y = ids.length === 1 ? railY : railY + dir * gap * (k + 1);
      if (ids.length > 1) {
        svg.append('line').attr('x1', x).attr('y1', from).attr('x2', x).attr('y2', y)
          .attr('stroke', 'rgba(255,255,255,0.4)').attr('stroke-width', 1.5);
      }
      drawNode(svg, x, y, ids[k], current === ids[k], side, canonicalLabel);
      from = y;
    }
  }

  function drawNode(svg, x, y, id, isCurrent, side, canonicalLabel) {
    const g = svg.append('g').attr('class', `step-node side-${side}${isCurrent ? ' is-current' : ''}`)
      .attr('cursor', 'pointer').on('click', () => onJump(id));
    g.append('circle').attr('cx', x).attr('cy', y).attr('r', isCurrent ? 8 : 6)
      .attr('fill', isCurrent ? (side === 'a' ? '#ff9f43' : '#54a0ff') : 'rgba(255,255,255,0.85)')
      .attr('stroke', isCurrent ? 'rgba(255,255,255,0.95)' : 'rgba(0,0,0,0.4)')
      .attr('stroke-width', isCurrent ? 2 : 1);
    g.append('text').attr('x', x + (side === 'a' ? -10 : 10)).attr('y', y - 10)
      .attr('text-anchor', side === 'a' ? 'end' : 'start')
      .attr('fill', isCurrent ? '#fff' : 'rgba(255,255,255,0.7)')
      .attr('font-size', 11).text(id);
    g.append('title').text(`${canonicalLabel} (${id})`);
  }

  function describe(sub, aLabel, bLabel, leftByC, rightByC) {
    const cid = canonicalOf(sub);
    const stepDef = CANONICAL_STEPS.find(s => s.id === cid);
    const inA = (leftByC[cid] || []).includes(sub);
    const inB = (rightByC[cid] || []).includes(sub);
    let who = '';
    if (inA && inB) who = `${aLabel} and ${bLabel}`;
    else if (inA) who = aLabel;
    else if (inB) who = bLabel;
    return `Step ${sub}: ${stepDef ? stepDef.label : ''} - ${who}`;
  }

  function render({ leftLabel, rightLabel, leftSubSteps, rightSubSteps, currentSubStep }) {
    labelA.text(`A: ${leftLabel}`);
    labelB.text(`B: ${rightLabel}`);
    const leftByC = groupByCanonical(leftSubSteps);
    const rightByC = groupByCanonical(rightSubSteps);
    const W = 720;
    const PAD = 36;
    const colW = (W - 2 * PAD) / Math.max(1, CANONICAL_STEPS.length - 1);
    const railAY = 36;
    const railBY = 108;
    const gap = 18;
    let maxA = 1, maxB = 1;
    for (let i = 0; i < CANONICAL_STEPS.length; i++) {
      const cid = CANONICAL_STEPS[i].id;
      maxA = Math.max(maxA, (leftByC[cid] || []).length);
      maxB = Math.max(maxB, (rightByC[cid] || []).length);
    }
    const top = Math.max(0, (maxA - 1) * gap);
    const bot = Math.max(0, (maxB - 1) * gap);
    const H = railBY + bot + 40;
    svg.attr('viewBox', `0 -${top} ${W} ${H + top}`);
    svg.selectAll('*').remove();

    svg.append('line').attr('x1', PAD).attr('y1', railAY).attr('x2', W - PAD).attr('y2', railAY)
      .attr('stroke', 'rgba(255,255,255,0.22)').attr('stroke-width', 2);
    svg.append('line').attr('x1', PAD).attr('y1', railBY).attr('x2', W - PAD).attr('y2', railBY)
      .attr('stroke', 'rgba(255,255,255,0.22)').attr('stroke-width', 2);

    for (let i = 0; i < CANONICAL_STEPS.length; i++) {
      const cid = CANONICAL_STEPS[i].id;
      const x = PAD + colW * i;
      drawColumnSide(svg, x, railAY, -1, leftByC[cid] || [], currentSubStep, 'a', gap, CANONICAL_STEPS[i].label);
      drawColumnSide(svg, x, railBY, +1, rightByC[cid] || [], currentSubStep, 'b', gap, CANONICAL_STEPS[i].label);
      svg.append('text').attr('x', x).attr('y', railBY + Math.max(1, (rightByC[cid] || []).length) * gap + 18)
        .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.55)')
        .attr('font-size', 11).text(cid);
    }

    desc.text(describe(currentSubStep, leftLabel, rightLabel, leftByC, rightByC));
    const all = [...new Set([...leftSubSteps, ...rightSubSteps])].sort(compareSubSteps);
    const idx = all.indexOf(currentSubStep);
    prevBtn.attr('disabled', idx <= 0 ? '' : null);
    nextBtn.attr('disabled', idx >= all.length - 1 ? '' : null);
  }

  return { render };
}
