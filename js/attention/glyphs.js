// js/attention/glyphs.js
// Pure SVG-string builders for the pipeline's node icons and connector arrows. No DOM access —
// every function here takes plain data in and returns a markup string, so it is testable with
// plain Node asserts (string/shape checks) without a browser.

export const STEP_IDS = ['input', 'qkv', 'scores', 'mask', 'softmax', 'wsum', 'output'];

const DEFAULT_TOK = ['#7c8fff', '#e0b341', '#4fd1a5'];

function svgInput(tok) {
  return `<svg viewBox="0 0 40 40">
    <rect x="2" y="5" width="26" height="7" rx="2" fill="${tok[0]}" opacity="0.85"/>
    <rect x="2" y="17" width="20" height="7" rx="2" fill="${tok[1]}" opacity="0.85"/>
    <rect x="2" y="29" width="24" height="7" rx="2" fill="${tok[2]}" opacity="0.85"/>
  </svg>`;
}

function svgQkv(tok) {
  return `<svg viewBox="0 0 40 40">
    <line x1="0" y1="20" x2="10" y2="20" stroke="var(--hairline-strong)" stroke-width="2"/>
    <rect class="node-box" x="10" y="10" width="16" height="20" rx="3" fill="none" stroke="var(--text-muted)" stroke-width="1.6"/>
    <line x1="26" y1="14" x2="37" y2="7" stroke="${tok[0]}" stroke-width="1.6"/>
    <line x1="26" y1="20" x2="37" y2="20" stroke="${tok[1]}" stroke-width="1.6"/>
    <line x1="26" y1="26" x2="37" y2="33" stroke="${tok[2]}" stroke-width="1.6"/>
    <circle cx="38" cy="7" r="2.4" fill="${tok[0]}"/><circle cx="38" cy="20" r="2.4" fill="${tok[1]}"/><circle cx="38" cy="33" r="2.4" fill="${tok[2]}"/>
  </svg>`;
}

function gridCells(vals, cell, gap, offsetX, offsetY, opacityOf) {
  let out = '';
  vals.forEach((v, i) => {
    const x = (i % 3) * (cell + gap) + offsetX;
    const y = Math.floor(i / 3) * (cell + gap) + offsetY;
    out += `<rect x="${x}" y="${y}" width="${cell}" height="${cell}" rx="1.5" fill="var(--accent)" opacity="${opacityOf(v)}"/>`;
  });
  return out;
}

const SCORE_VALS = [0.9, 0.3, 0.15, 0.25, 0.85, 0.2, 0.15, 0.3, 0.9];

function svgScores() {
  const cells = gridCells(SCORE_VALS, 10, 2, 4, 4, (v) => (0.15 + v * 0.65).toFixed(2));
  return `<svg viewBox="0 0 40 40"><g class="node-box">${cells}
    <rect x="2" y="2" width="36" height="36" rx="4" fill="none" stroke="var(--text-muted)" stroke-width="1.4"/></g></svg>`;
}

function svgMask() {
  // Lower-triangle + diagonal cells stay tinted (visible); upper-triangle (future positions,
  // i < j) are drawn as flat dark cells with a diagonal strike, previewing what softmax is
  // about to zero out.
  let cells = '';
  SCORE_VALS.forEach((v, i) => {
    const row = Math.floor(i / 3);
    const col = i % 3;
    const x = col * 12 + 4;
    const y = row * 12 + 4;
    if (col > row) {
      cells += `<rect x="${x}" y="${y}" width="10" height="10" rx="1.5" fill="var(--surface)" stroke="var(--hairline-strong)" stroke-width="1"/>`;
      cells += `<line x1="${x + 2}" y1="${y + 2}" x2="${x + 8}" y2="${y + 8}" stroke="var(--text-muted)" stroke-width="1"/>`;
    } else {
      cells += `<rect x="${x}" y="${y}" width="10" height="10" rx="1.5" fill="var(--accent)" opacity="${(0.15 + v * 0.65).toFixed(2)}"/>`;
    }
  });
  return `<svg viewBox="0 0 40 40"><g class="node-box">${cells}
    <rect x="2" y="2" width="36" height="36" rx="4" fill="none" stroke="var(--text-muted)" stroke-width="1.4"/></g></svg>`;
}

function svgSoftmax() {
  return `<svg viewBox="0 0 40 40">
    <rect x="4" y="4" width="10" height="10" rx="1.5" fill="var(--accent)" opacity=".85"/>
    <rect x="16" y="4" width="10" height="10" rx="1.5" fill="var(--accent)" opacity=".2"/>
    <rect x="28" y="4" width="8" height="10" rx="1.5" fill="var(--accent)" opacity=".15"/>
    <rect x="4" y="16" width="8" height="6" rx="1.5" fill="var(--accent)" opacity=".2"/>
    <rect x="14" y="16" width="12" height="6" rx="1.5" fill="var(--accent)" opacity=".9"/>
    <rect x="28" y="16" width="8" height="6" rx="1.5" fill="var(--accent)" opacity=".2"/>
    <path d="M4 30 Q12 22 20 30 T36 30" fill="none" stroke="var(--text-muted)" stroke-width="1.4"/>
  </svg>`;
}

function svgWsum(tok) {
  return `<svg viewBox="0 0 40 40">
    <defs><linearGradient id="wg-grad" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="${tok[0]}"/><stop offset=".5" stop-color="${tok[1]}"/><stop offset="1" stop-color="${tok[2]}"/></linearGradient></defs>
    <rect x="1" y="4" width="8" height="6" rx="1.5" fill="${tok[0]}" opacity=".85"/>
    <rect x="1" y="13" width="8" height="6" rx="1.5" fill="${tok[1]}" opacity=".35"/>
    <rect x="1" y="22" width="8" height="6" rx="1.5" fill="${tok[2]}" opacity=".2"/>
    <line x1="11" y1="7" x2="17" y2="18" stroke="${tok[0]}" stroke-width="1.4"/>
    <line x1="11" y1="16" x2="17" y2="18" stroke="${tok[1]}" stroke-width="1.4"/>
    <line x1="11" y1="25" x2="17" y2="18" stroke="${tok[2]}" stroke-width="1.4"/>
    <circle class="node-box" cx="19" cy="18" r="5" fill="none" stroke="var(--text-muted)" stroke-width="1.4"/>
    <line x1="16" y1="18" x2="22" y2="18" stroke="var(--text-muted)" stroke-width="1.2"/>
    <line x1="19" y1="15" x2="19" y2="21" stroke="var(--text-muted)" stroke-width="1.2"/>
    <line x1="24" y1="18" x2="34" y2="18" stroke="var(--hairline-strong)" stroke-width="1.4"/>
    <rect x="34" y="13" width="5" height="10" rx="1.5" fill="url(#wg-grad)" opacity=".9"/>
  </svg>`;
}

function svgOutput(tok) {
  return `<svg viewBox="0 0 40 40">
    <defs><linearGradient id="out-grad" x1="0" x2="1"><stop offset="0" stop-color="${tok[0]}"/><stop offset=".5" stop-color="${tok[1]}"/><stop offset="1" stop-color="${tok[2]}"/></linearGradient></defs>
    <rect x="2" y="5" width="26" height="7" rx="2" fill="url(#out-grad)" opacity=".9"/>
    <rect x="2" y="17" width="22" height="7" rx="2" fill="url(#out-grad)" opacity=".9"/>
    <rect x="2" y="29" width="25" height="7" rx="2" fill="url(#out-grad)" opacity=".9"/>
  </svg>`;
}

const BUILDERS = {
  input: svgInput,
  qkv: svgQkv,
  scores: svgScores,
  mask: svgMask,
  softmax: svgSoftmax,
  wsum: svgWsum,
  output: svgOutput,
};

export function glyphSVG(stepId, opts = {}) {
  const builder = BUILDERS[stepId];
  if (!builder) throw new Error(`glyphSVG: unknown step id "${stepId}"`);
  const tok = opts.tokenColors || DEFAULT_TOK;
  return builder(tok);
}

export function connectorsSVG(count, opts = {}) {
  const width = opts.width || (count - 1) * 90 + 60;
  const height = opts.height || 60;
  const xs = Array.from({ length: count }, (_, i) => 30 + (i * (width - 60)) / (count - 1));
  let out =
    '<defs><marker id="attn-arrowhead" markerWidth="6" markerHeight="6" refX="4" refY="3" orient="auto">' +
    '<path d="M0,0 L6,3 L0,6 Z" fill="var(--hairline-strong)"/></marker></defs>';
  for (let i = 0; i < xs.length - 1; i++) {
    const x1 = xs[i] + 18;
    const x2 = xs[i + 1] - 20;
    const midY = height * 0.3;
    const dipY = height * 0.67;
    out += `<path d="M${x1} ${midY} C ${x1 + 40} ${dipY}, ${x2 - 40} ${dipY}, ${x2} ${midY}" fill="none" stroke="var(--hairline-strong)" stroke-width="1.3" marker-end="url(#attn-arrowhead)"/>`;
  }
  return out;
}
