// js/attention/presets.js
// Worked-example data for the attention page. Three curated presets (no free-form editing in
// this phase - see docs/superpowers/specs/2026-07-19-attention-visualization-design.md).
// Weight matrices are shared across presets: hand-picked, not learned, chosen so the resulting
// attention pattern is clearly peaked (not near-uniform) after softmax. Embeddings are arrays
// indexed by position (parallel to `tokens`), not objects keyed by word: a repeated word (see
// "dog-chased-cat" below) gets one array entry per occurrence instead of colliding onto one.

export const PRESETS = [
  {
    // The default preset. Repeats "the" at positions 0 and 3 with an identical raw embedding,
    // so positional encoding is the only thing that can tell the two occurrences apart --
    // proving the string-keyed-collision fix actually does something. "chased" is a hand-picked
    // embedding, tuned (together with the sqrt(d) embedding scale in math.js) so every row of
    // every preset stays clearly peaked -- see dev/test/attention-presets.test.js.
    id: 'dog-chased-cat',
    label: '"the dog chased the cat"',
    tokens: ['the', 'dog', 'chased', 'the', 'cat'],
    embeddings: [
      [0.2, 0.8, 0.1, 0.4],
      [1.0, 0.0, 0.9, 0.0],
      [0.2, 0.4, 1.0, 1.0],
      [0.2, 0.8, 0.1, 0.4],
      [0.9, 0.1, 0.6, 0.3],
    ],
  },
  {
    // "quiet" and "##ly" split what's semantically one word ("quietly") into two tokens, using
    // the WordPiece-style "##" continuation-marker convention (the same convention BERT-base
    // uses, already cited elsewhere on this page) -- demonstrating that tokenization isn't
    // always one token per word.
    id: 'cat-sat',
    label: '"the cat sat quietly"',
    tokens: ['the', 'cat', 'sat', 'quiet', '##ly'],
    embeddings: [
      [0.2, 0.8, 0.1, 0.4],
      [0.9, 0.1, 0.6, 0.3],
      [0.3, 0.5, 0.8, 0.2],
      [1.2, 0.2, 1.2, 0.8],
      [0.0, 0.0, 0.0, 1.2],
    ],
  },
  {
    id: 'dog-ran-fast',
    label: '"the dog ran very fast"',
    tokens: ['the', 'dog', 'ran', 'very', 'fast'],
    embeddings: [
      [0.2, 0.8, 0.1, 0.4],
      [1.0, 0.0, 0.9, 0.0],
      [0.0, 1.0, 0.0, 0.9],
      [-0.6, -0.6, -0.2, -0.6],
      [0.95, 0.05, 0.85, 0.05],
    ],
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

// Indexed by token POSITION (0-4), not by token identity, so recoloring stays stable when the
// preset changes which words are used, and so a repeated word (e.g. "the" in dog-chased-cat)
// gets two different colors, one per occurrence -- reinforcing that position, not word
// identity, is what's being tracked from here on.
export const TOKEN_COLORS = ['#7c8fff', '#e0b341', '#4fd1a5', '#c77dff', '#ff8fa3'];
