// manifold_isomap.js - per-step clip player for the Isomap explainer (ES module).
const STEPS = [
  { title: '1. Raw data', caption: 'A 2D sheet rolled up in 3D. The goal is to recover the flat sheet.', formula: '' },
  { title: '2. kNN graph', caption: 'Connect each point to its k = 8 nearest neighbors.', formula: '' },
  { title: '3. Geodesic distances', caption: 'Distance measured along the graph, not straight through space.', formula: '' },
  { title: '4. Double-centering', caption: 'Turn squared geodesic distances into the matrix B.', formula: 'B = -\\tfrac{1}{2} J D^2 J' },
  { title: '5. Eigendecomposition', caption: 'The top eigenvectors of B carry the recovered shape.', formula: 'B v_i = \\lambda_i v_i' },
  { title: '6. Embedding', caption: 'The sheet unrolls into 2D, geodesic distances preserved.', formula: 'Y = [\\sqrt{\\lambda_1} v_1,\\ \\sqrt{\\lambda_2} v_2]' },
];

const video = document.getElementById('mfiVideo');
const stepsEl = document.getElementById('mfiSteps');
const transcript = document.getElementById('mfiTranscript');
const scrub = document.getElementById('mfiScrub');
const playBtn = document.getElementById('mfiPlay');
let current = 0;
let autoChain = false;

function srcFor(i) { return `../assets/manim/isomap/step-${i + 1}.mp4`; }
function posterFor(i) { return `../assets/manim/isomap/step-${i + 1}.png`; }

function renderSteps() {
  stepsEl.innerHTML = '';
  STEPS.forEach((s, i) => {
    const li = document.createElement('li');
    li.textContent = s.title;
    if (i === current) li.classList.add('is-active');
    li.addEventListener('click', () => load(i, false));
    stepsEl.appendChild(li);
  });
}

function renderTranscript() {
  const s = STEPS[current];
  transcript.innerHTML = '';
  const cap = document.createElement('div');
  cap.textContent = s.caption;
  transcript.appendChild(cap);
  if (s.formula) {
    const f = document.createElement('div');
    f.className = 'mfi-formula';
    f.textContent = `\\[${s.formula}\\]`;
    transcript.appendChild(f);
  }
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetClear && window.MathJax.typesetClear();
    window.MathJax.typesetPromise([transcript]).catch(() => {});
  }
}

function load(i, autoplay) {
  current = Math.max(0, Math.min(STEPS.length - 1, i));
  video.poster = posterFor(current);
  video.src = srcFor(current);
  video.load();
  renderSteps();
  renderTranscript();
  if (autoplay) video.play().catch(() => {});
}

video.addEventListener('timeupdate', () => {
  if (video.duration) scrub.value = String(Math.round((video.currentTime / video.duration) * 1000));
});
scrub.addEventListener('input', () => {
  if (video.duration) video.currentTime = (scrub.value / 1000) * video.duration;
});
video.addEventListener('ended', () => {
  if (autoChain && current < STEPS.length - 1) load(current + 1, true);
});
playBtn.addEventListener('click', () => {
  autoChain = true;
  video.play().catch(() => {});
});
document.getElementById('mfiPrev').addEventListener('click', () => load(current - 1, true));
document.getElementById('mfiNext').addEventListener('click', () => load(current + 1, true));

renderSteps();
load(0, false);
