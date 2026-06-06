// manifold_isomap.js - Isomap explainer player (ES module).
// Plays one continuous video and treats the steps as chapter markers, so playback
// is seamless across step boundaries and the active step always matches the
// pseudocode timeline in the top-left of the video. `start` is the step's start
// time in seconds (the section boundaries of the rendered walkthrough).
// Explanations mirror manimexp/isomap/walkthrough_explained.md.
const STEPS = [
  {
    start: 0,
    title: '1. Raw data',
    caption: 'A 2D sheet rolled up in 3D; the goal is to recover the flat sheet.',
    explain: 'The data is a Swiss roll: a flat two-dimensional sheet rolled up inside ' +
      'three-dimensional space, so it has only two genuine degrees of freedom. Two points ' +
      'can sit close together in 3D yet be far apart along the sheet, the way two layers of ' +
      'a rolled poster nearly touch but are far apart along the paper. Recovering the flat ' +
      'sheet means finding a 2D coordinate for every point that respects distance along the ' +
      'surface. Color runs along a full rainbow so each region is easy to follow.',
    formula: '',
  },
  {
    start: 13.5,
    title: '2. kNN graph',
    caption: 'Link each point to its k = 8 nearest neighbors, weighted by distance.',
    explain: 'Every point is connected to its $k = 8$ closest points, and each link is ' +
      'weighted by the straight-line distance between its endpoints. Over a short hop ' +
      'between near neighbors that distance stays on the surface, so the weights are good ' +
      'local estimates of true surface distance. Done for every point, the links form a ' +
      'mesh that follows the sheet and serves as the scaffold for measuring distance along it.',
    formula: 'w_{ij} = \\lVert x_i - x_j \\rVert',
  },
  {
    start: 32.9,
    title: '3. Geodesic distances',
    caption: 'Distance measured along the graph, not straight through space.',
    explain: 'A geodesic distance is the length of the shortest route that stays on the ' +
      'surface: on the neighbor graph, the shortest path between two nodes, found by adding ' +
      'up edge weights (Dijkstra). A straight chord cuts directly through the empty space ' +
      'between the rolls; it is shorter in 3D but meaningless for the sheet. The cloud is ' +
      'colored by geodesic distance from one source across a full rainbow, so two points on ' +
      'different turns of the roll receive very different colors.',
    formula: 'D_{ij} = \\text{shortest path } i \\to j',
  },
  {
    start: 55.27,
    title: '4. Double-centering',
    caption: 'Turn the geodesic distances into the Gram matrix B of inner products.',
    explain: 'A distance table cannot pin down coordinates, because rotating or shifting all ' +
      'the points together leaves every pairwise distance unchanged. Inner products can: ' +
      '$G_{ij} = x_i \\cdot x_j$ records the two points’ lengths and the angle between ' +
      'them from a shared origin, and stacking them gives $G = X X^\\top$, one matrix product ' +
      'away from the coordinates. $G$ captures the relative geometry, a bridge between ' +
      'distances and coordinates. We do not have the coordinates to form $G$ directly, so we ' +
      'build it from distances: square every entry, then double-center, ' +
      '$B = -\\tfrac{1}{2} J D^2 J$. The result $B$ is the Gram matrix of the centered points.',
    formula: 'B = -\\tfrac{1}{2} J D^2 J',
  },
  {
    start: 120.07,
    title: '5. Eigendecomposition',
    caption: 'The top eigenvectors of B carry the recovered shape.',
    explain: 'Factor $B$. Power iteration finds the dominant eigenvector: start from a random ' +
      'unit vector and repeatedly apply $v \\leftarrow Bv / \\lVert Bv \\rVert$. The Rayleigh ' +
      'quotient $v^\\top B v$, shown as an actual matrix product each step, climbs to the ' +
      'largest eigenvalue $\\lambda_1$. The eigenvectors with the largest eigenvalues are the ' +
      'directions in which the centered points vary the most, the genuine low-dimensional ' +
      'structure; the top two are kept.',
    formula: 'B v_i = \\lambda_i v_i',
  },
  {
    start: 155.27,
    title: '6. Embedding',
    caption: 'The sheet unrolls into 2D, geodesic distances preserved.',
    explain: 'The recovered coordinates are $Y = [\\sqrt{\\lambda_1}\\, v_1,\\ ' +
      '\\sqrt{\\lambda_2}\\, v_2]$: each kept eigenvector scaled by the square root of its ' +
      'eigenvalue. The cloud settles into a flat band, the sheet unrolled, and the rainbow ' +
      'geodesic coloring now varies smoothly across the flat layout, the visual confirmation ' +
      'that along-the-sheet distances were preserved while the extra dimension was removed.',
    formula: 'Y = [\\sqrt{\\lambda_1} v_1,\\ \\sqrt{\\lambda_2} v_2]',
  },
];

const ASSET_BASE = '../assets/manim/isomap/';
const POSTER_SRC = ASSET_BASE + 'step-1.png';

// Each dataset with a rendered walkthrough maps to its clip and its opening
// caption. All clips share the same step timeline, so the chapter markers above
// (STEPS[i].start) apply to every one. Datasets without a clip fall back to the
// Swiss roll. The closed-loop shapes note that they lay out as a loop, not flat.
const DATASET_INFO = {
  swiss_roll: { video: 'walkthrough.mp4', intro: 'A 2D sheet rolled up in 3D; the goal is to recover the flat sheet.' },
  s_curve: { video: 'drafts/s_curve.mp4', intro: 'A 2D sheet bent into an S; the goal is to recover the flat sheet.' },
  twin_peaks: { video: 'drafts/twin_peaks.mp4', intro: 'A bumpy height surface in 3D; the goal is to recover its flat layout.' },
  saddle: { video: 'drafts/saddle.mp4', intro: 'A curved saddle surface; the goal is to recover its flat layout.' },
  cylinder: { video: 'drafts/cylinder.mp4', intro: 'A sheet wrapped into a cylinder; a closed band lays out as a loop, not a flat sheet.' },
  severed_sphere: { video: 'drafts/severed_sphere.mp4', intro: 'A sphere with its cap removed, an open curved surface; the goal is to flatten it.' },
  helix: { video: 'drafts/helix.mp4', intro: 'A ribbon wound into a helix; the goal is to unroll it to a flat strip.' },
  trefoil_knot: { video: 'drafts/trefoil_knot.mp4', intro: 'A ribbon tied into a trefoil knot; a closed band lays out as a loop, not a flat strip.' },
  toroidal_helix: { video: 'drafts/toroidal_helix.mp4', intro: 'A ribbon coiled around a torus; a closed band lays out as a loop, not a flat strip.' },
  spiral_disk: { video: 'drafts/spiral_disk.mp4', intro: 'A ribbon wound into a spiral; the goal is to unroll it to a flat strip.' },
};

const video = document.getElementById('mfiVideo');
const stepsEl = document.getElementById('mfiSteps');
const transcript = document.getElementById('mfiTranscript');
const scrub = document.getElementById('mfiScrub');
const playBtn = document.getElementById('mfiPlay');
const speedSel = document.getElementById('mfiSpeed');
let current = -1;

function stepIndexAt(t) {
  let idx = 0;
  for (let i = 0; i < STEPS.length; i++) {
    if (t >= STEPS[i].start - 0.05) idx = i;
  }
  return idx;
}

function renderSteps() {
  stepsEl.innerHTML = '';
  STEPS.forEach((s, i) => {
    const li = document.createElement('li');
    li.textContent = s.title;
    if (i === current) li.classList.add('is-active');
    li.addEventListener('click', () => seekToStep(i));
    stepsEl.appendChild(li);
  });
}

function renderTranscript() {
  const s = STEPS[current] || STEPS[0];
  transcript.innerHTML = '';

  const cap = document.createElement('div');
  cap.className = 'mfi-caption';
  cap.textContent = s.caption;
  transcript.appendChild(cap);

  if (s.formula) {
    const f = document.createElement('div');
    f.className = 'mfi-formula';
    f.textContent = `\\[${s.formula}\\]`;
    transcript.appendChild(f);
  }

  if (s.explain) {
    const e = document.createElement('p');
    e.className = 'mfi-explain';
    e.innerHTML = s.explain;
    transcript.appendChild(e);
  }

  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetClear && window.MathJax.typesetClear();
    window.MathJax.typesetPromise([transcript]).catch(() => {});
  }
}

// Set the active step (highlight + transcript) without moving the video.
function setActive(i) {
  i = Math.max(0, Math.min(STEPS.length - 1, i));
  if (i === current) return;
  current = i;
  renderSteps();
  renderTranscript();
}

// Move the video to a step boundary (clicking a step or Prev/Next).
function seekToStep(i) {
  i = Math.max(0, Math.min(STEPS.length - 1, i));
  if (video.duration) video.currentTime = Math.min(STEPS[i].start, video.duration - 0.05);
  else video.addEventListener('loadedmetadata', () => { video.currentTime = STEPS[i].start; }, { once: true });
  setActive(i);
}

// --- wiring ---
video.poster = POSTER_SRC;
video.preload = 'auto';

// On manifold.html the player follows the dataset chosen in the sandbox; on the
// standalone page (no dataset select) it stays on the Swiss roll.
const datasetSel = document.getElementById('mfDataset');
const datasetNote = document.createElement('div');
datasetNote.className = 'mfi-datasetnote';
stepsEl.insertAdjacentElement('afterend', datasetNote);
let currentBlobUrl = null;

function currentDataset() { return (datasetSel && datasetSel.value) || 'swiss_roll'; }

// Load the clip for a dataset as a Blob so the whole file is held in memory and
// is fully seekable, even when the host does not serve HTTP range requests (a
// plain static server returns the whole file with no Accept-Ranges, which
// otherwise blocks seeking). Updates the opening caption and resets to step 1.
function loadVideo(datasetId) {
  const known = Object.prototype.hasOwnProperty.call(DATASET_INFO, datasetId);
  const info = known ? DATASET_INFO[datasetId] : DATASET_INFO.swiss_roll;
  STEPS[0].caption = info.intro;
  const url = ASSET_BASE + info.video;
  fetch(url)
    .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(String(r.status)))))
    .then((blob) => {
      if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
      currentBlobUrl = URL.createObjectURL(blob);
      video.src = currentBlobUrl;
    })
    .catch(() => { video.src = url; });
  if (datasetNote) {
    const label = (datasetSel && datasetSel.selectedOptions[0])
      ? datasetSel.selectedOptions[0].textContent : 'Swiss roll';
    datasetNote.textContent = known ? '' : `No walkthrough yet for ${label}; showing the Swiss roll.`;
  }
  current = -1;
  setActive(0);
}

video.addEventListener('timeupdate', () => {
  if (video.duration) scrub.value = String(Math.round((video.currentTime / video.duration) * 1000));
  setActive(stepIndexAt(video.currentTime));
});
scrub.addEventListener('input', () => {
  if (video.duration) video.currentTime = (scrub.value / 1000) * video.duration;
});

playBtn.addEventListener('click', () => {
  if (video.paused) video.play().catch(() => {});
  else video.pause();
});
video.addEventListener('play', () => { playBtn.textContent = 'Pause'; });
video.addEventListener('pause', () => { playBtn.textContent = 'Play'; });
video.addEventListener('ended', () => { playBtn.textContent = 'Play'; });

document.getElementById('mfiPrev').addEventListener('click', () => seekToStep(current - 1));
document.getElementById('mfiNext').addEventListener('click', () => seekToStep(current + 1));

if (speedSel) {
  speedSel.addEventListener('change', () => { video.playbackRate = parseFloat(speedSel.value) || 1; });
  video.playbackRate = parseFloat(speedSel.value) || 1;
}

if (datasetSel) datasetSel.addEventListener('change', () => loadVideo(currentDataset()));

renderSteps();
loadVideo(currentDataset());
