// js/attention/presets.js
// Worked-example data for the attention page. Two curated presets (no free-form editing in
// this phase — see docs/superpowers/specs/2026-07-19-attention-visualization-design.md).
// Weight matrices are shared across presets: hand-picked, not learned, chosen so the resulting
// attention pattern is clearly peaked (not near-uniform) after softmax.

export const PRESETS = [
  {
    id: 'cat-sat',
    label: '"the cat sat"',
    tokens: ['the', 'cat', 'sat'],
    embeddings: {
      the: [0.2, 0.8, 0.1, 0.4],
      cat: [0.9, 0.1, 0.6, 0.3],
      sat: [0.3, 0.5, 0.8, 0.2],
    },
  },
  {
    id: 'dog-ran-fast',
    label: '"dog ran fast"',
    tokens: ['dog', 'ran', 'fast'],
    embeddings: {
      dog: [0.7, 0.2, 0.3, 0.6],
      ran: [0.4, 0.6, 0.7, 0.1],
      fast: [0.2, 0.3, 0.5, 0.8],
    },
  },
];

export const WEIGHTS = {
  WQ: [
    [1.5, 0.0, 0.8, 0.0],
    [0.0, 1.6, 0.0, 0.6],
    [0.8, 0.0, 1.5, 0.0],
    [0.0, 0.6, 0.0, 1.6],
  ],
  WK: [
    [1.4, 0.3, 0.0, 0.0],
    [0.3, 1.4, 0.0, 0.0],
    [0.0, 0.0, 1.4, 0.3],
    [0.0, 0.0, 0.3, 1.4],
  ],
  WV: [
    [0.9, 0.0, 0.0, 0.2],
    [0.0, 0.9, 0.2, 0.0],
    [0.0, 0.2, 0.9, 0.0],
    [0.2, 0.0, 0.0, 0.9],
  ],
};

// Indexed by token POSITION (0/1/2), not by token identity, so recoloring stays stable when
// the preset changes which words are used.
export const TOKEN_COLORS = ['#7c8fff', '#e0b341', '#4fd1a5'];
