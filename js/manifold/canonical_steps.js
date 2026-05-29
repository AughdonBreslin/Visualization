export const CANONICAL_STEPS = [
  { id: '0', label: 'Raw data' },
  { id: '1', label: 'Preprocess (center/normalize)' },
  { id: '2', label: 'Neighborhood graph' },
  { id: '3', label: 'Pairwise affinity / distances' },
  { id: '4', label: 'Matrix transform' },
  { id: '5', label: 'Spectral decomposition' },
  { id: '6', label: 'Embed / project' },
];

export const CANONICAL_INDEX = new Map(CANONICAL_STEPS.map((s, i) => [s.id, i]));

export function canonicalOf(subStepId) {
  return subStepId.replace(/[a-z]$/, '');
}

export function compareSubSteps(a, b) {
  const ia = CANONICAL_INDEX.get(canonicalOf(a));
  const ib = CANONICAL_INDEX.get(canonicalOf(b));
  if (ia !== ib) return ia - ib;
  const sa = a.length > 1 ? a.charCodeAt(a.length - 1) : 0;
  const sb = b.length > 1 ? b.charCodeAt(b.length - 1) : 0;
  return sa - sb;
}

export function unionSubSteps(...lists) {
  const seen = new Set();
  for (const list of lists) for (const id of list) seen.add(id);
  return [...seen].sort(compareSubSteps);
}
