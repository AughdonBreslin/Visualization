import { DATASETS, parseCSV } from './datasets.js';
import { PCA } from './algorithms/pca.js';
import { ISOMAP } from './algorithms/isomap.js';
import { createState } from './state.js';
import { createViz3d } from './viz3d.js';
import { createViz2d } from './viz2d.js';
import { createStepIndicator } from './step_indicator.js';
import { createIFW } from './ifw.js';
import { createPseudocode } from './pseudocode.js';
import { compareSubSteps, unionSubSteps } from './canonical_steps.js';

const ALGORITHMS = [PCA, ISOMAP];
const ALGORITHMS_BY_ID = Object.fromEntries(ALGORITHMS.map(a => [a.id, a]));

const defaults = {
  datasetId: 'swiss_roll',
  datasetParams: { samples: 300, noise: 0.0, seed: 7 },
  leftAlgoId: 'pca',
  rightAlgoId: 'isomap',
  algoParams: { pca: {}, isomap: { k: 10 } },
};

function init() {
  const $ = (id) => document.getElementById(id);
  const datasetSelect = $('mfDataset');
  const samplesInput = $('mfSamples');
  const noiseInput = $('mfNoise');
  const seedInput = $('mfSeed');
  const reseedBtn = $('mfReseed');
  const csvInput = $('mfCsvInput');
  const csvLabel = $('mfCsvLabel');
  const leftSelect = $('mfAlgoLeft');
  const rightSelect = $('mfAlgoRight');
  const leftParamsHost = $('mfAlgoLeftParams');
  const rightParamsHost = $('mfAlgoRightParams');
  const stepHost = $('mfStep');
  const leftVizHost = $('mfLeftViz');
  const rightVizHost = $('mfRightViz');
  const leftIfwHost = $('mfLeftIfw');
  const rightIfwHost = $('mfRightIfw');
  const leftPseudoHost = $('mfLeftPseudo');
  const rightPseudoHost = $('mfRightPseudo');
  const leftTitle = $('mfLeftTitle');
  const rightTitle = $('mfRightTitle');
  const samplesControl = $('mfSamplesControl');
  const noiseControl = $('mfNoiseControl');
  const seedControl = $('mfSeedControl');

  for (const ds of DATASETS) {
    const opt = document.createElement('option');
    opt.value = ds.id; opt.textContent = ds.label;
    if (ds.id === defaults.datasetId) opt.selected = true;
    datasetSelect.appendChild(opt);
  }
  for (const algo of ALGORITHMS) {
    const a = document.createElement('option');
    a.value = algo.id; a.textContent = algo.label;
    if (algo.id === defaults.leftAlgoId) a.selected = true;
    leftSelect.appendChild(a);
    const b = document.createElement('option');
    b.value = algo.id; b.textContent = algo.label;
    if (algo.id === defaults.rightAlgoId) b.selected = true;
    rightSelect.appendChild(b);
  }
  samplesInput.value = defaults.datasetParams.samples;
  noiseInput.value = defaults.datasetParams.noise;
  seedInput.value = defaults.datasetParams.seed;

  const store = createState({ algorithmsById: ALGORITHMS_BY_ID, defaults });

  const leftViz = createViz3d(leftVizHost, {});
  const rightViz = createViz3d(rightVizHost, {});
  const leftViz2d = createViz2d(leftVizHost, {});
  const rightViz2d = createViz2d(rightVizHost, {});
  const leftThumb = createViz3d(leftVizHost, { width: 140, height: 110, isThumbnail: true });
  const rightThumb = createViz3d(rightVizHost, { width: 140, height: 110, isThumbnail: true });
  hideEl(leftVizHost, '.viz2d'); hideEl(rightVizHost, '.viz2d');
  hideEl(leftVizHost, '.viz3d-thumb'); hideEl(rightVizHost, '.viz3d-thumb');

  const stepIndicator = createStepIndicator(stepHost, {
    onJump: (target) => {
      const left = store.state.cache.left;
      const right = store.state.cache.right;
      if (!left || !right) return;
      const all = unionSubSteps(left.presentSubSteps, right.presentSubSteps);
      if (target === 'prev' || target === 'next') {
        const idx = all.indexOf(store.state.currentSubStep);
        const next = target === 'prev' ? Math.max(0, idx - 1) : Math.min(all.length - 1, idx + 1);
        store.setStep(all[next]);
      } else if (all.includes(target)) {
        store.setStep(target);
      }
    },
  });

  const leftIfw = createIFW(leftIfwHost, 'a');
  const rightIfw = createIFW(rightIfwHost, 'b');
  const leftPseudo = createPseudocode(leftPseudoHost, 'a');
  const rightPseudo = createPseudocode(rightPseudoHost, 'b');

  function renderParamHost(host, algo, current, onChange) {
    host.innerHTML = '';
    for (const p of algo.params) {
      const wrap = document.createElement('label');
      wrap.className = 'mf-param';
      wrap.textContent = `${p.name} = `;
      const input = document.createElement('input');
      input.type = p.type === 'int' || p.type === 'float' ? 'number' : 'text';
      if (p.min !== undefined) input.min = p.min;
      if (p.max !== undefined) input.max = p.max;
      input.step = p.type === 'int' ? 1 : 'any';
      input.value = current[p.name] !== undefined ? current[p.name] : p.default;
      input.addEventListener('change', () => {
        const v = p.type === 'int' ? parseInt(input.value, 10) : parseFloat(input.value);
        onChange({ ...current, [p.name]: v });
      });
      wrap.appendChild(input);
      host.appendChild(wrap);
    }
    if (algo.params.length === 0) host.innerHTML = '<span class="mf-noparams">No parameters</span>';
  }

  function rebindParamHosts() {
    renderParamHost(leftParamsHost, ALGORITHMS_BY_ID[store.state.leftAlgoId], store.state.leftAlgoParams,
      (next) => store.set({ leftAlgoParams: next }));
    renderParamHost(rightParamsHost, ALGORITHMS_BY_ID[store.state.rightAlgoId], store.state.rightAlgoParams,
      (next) => store.set({ rightAlgoParams: next }));
  }
  rebindParamHosts();

  function updateSyntheticVisibility() {
    const isCsv = store.state.datasetId === 'csv';
    samplesControl.style.display = isCsv ? 'none' : '';
    noiseControl.style.display = isCsv ? 'none' : '';
    seedControl.style.display = isCsv ? 'none' : '';
    csvLabel.textContent = isCsv ? (store.state.csvFileName ? `Loaded: ${store.state.csvFileName} (${store.state.csvRows ? store.state.csvRows.length : 0} rows)` : '') : '';
  }

  datasetSelect.addEventListener('change', () => {
    const id = datasetSelect.value;
    if (id === 'csv') {
      csvInput.value = '';
      csvInput.click();
      return;
    }
    store.set({ datasetId: id, csvRows: null, csvFileName: '' });
    updateSyntheticVisibility();
  });
  csvInput.addEventListener('change', () => {
    const file = csvInput.files && csvInput.files[0];
    if (!file) {
      datasetSelect.value = store.state.datasetId === 'csv' ? 'csv' : store.state.datasetId;
      return;
    }
    file.text().then(text => {
      const rows = parseCSV(text);
      if (rows.length === 0) {
        csvLabel.textContent = `Could not parse "${file.name}". Need at least 2 numeric columns.`;
        datasetSelect.value = store.state.datasetId;
        return;
      }
      store.set({ datasetId: 'csv', csvRows: rows, csvFileName: file.name });
      datasetSelect.value = 'csv';
      updateSyntheticVisibility();
    });
  });
  samplesInput.addEventListener('change', () => {
    const v = Math.max(20, Math.min(1000, parseInt(samplesInput.value, 10) || 300));
    samplesInput.value = v;
    store.set({ datasetParams: { ...store.state.datasetParams, samples: v } });
  });
  noiseInput.addEventListener('change', () => {
    const v = Math.max(0, parseFloat(noiseInput.value) || 0);
    noiseInput.value = v;
    store.set({ datasetParams: { ...store.state.datasetParams, noise: v } });
  });
  seedInput.addEventListener('change', () => {
    const v = parseInt(seedInput.value, 10) || 0;
    store.set({ datasetParams: { ...store.state.datasetParams, seed: v } });
  });
  reseedBtn.addEventListener('click', () => {
    const v = Math.floor(Math.random() * 100000);
    seedInput.value = v;
    store.set({ datasetParams: { ...store.state.datasetParams, seed: v } });
  });
  leftSelect.addEventListener('change', () => {
    const id = leftSelect.value;
    const algo = ALGORITHMS_BY_ID[id];
    const params = {};
    for (const p of algo.params) params[p.name] = p.default;
    store.set({ leftAlgoId: id, leftAlgoParams: params });
    rebindParamHosts();
  });
  rightSelect.addEventListener('change', () => {
    const id = rightSelect.value;
    const algo = ALGORITHMS_BY_ID[id];
    const params = {};
    for (const p of algo.params) params[p.name] = p.default;
    store.set({ rightAlgoId: id, rightAlgoParams: params });
    rebindParamHosts();
  });

  store.subscribe((s) => {
    leftTitle.textContent = ALGORITHMS_BY_ID[s.leftAlgoId].label;
    rightTitle.textContent = ALGORITHMS_BY_ID[s.rightAlgoId].label;
    const left = s.cache.left, right = s.cache.right;
    if (!left || !right) {
      stepIndicator.render({
        leftLabel: ALGORITHMS_BY_ID[s.leftAlgoId].label,
        rightLabel: ALGORITHMS_BY_ID[s.rightAlgoId].label,
        leftSubSteps: ['0'], rightSubSteps: ['0'], currentSubStep: '0',
      });
      return;
    }
    const present = unionSubSteps(left.presentSubSteps, right.presentSubSteps);
    if (!present.includes(s.currentSubStep)) s.currentSubStep = present[0];
    const leftSub = nearestSub(s.currentSubStep, left.presentSubSteps);
    const rightSub = nearestSub(s.currentSubStep, right.presentSubSteps);
    const leftState = left.steps.get(leftSub);
    const rightState = right.steps.get(rightSub);

    const isFinal = s.currentSubStep === '6';
    setStep6Mode(leftVizHost, isFinal && leftState && leftState.embed2d);
    setStep6Mode(rightVizHost, isFinal && rightState && rightState.embed2d);

    if (isFinal && leftState && leftState.embed2d) {
      leftViz2d.setState({ embed2d: leftState.embed2d, colors: leftState.colors, t: leftState.t });
      leftThumb.setState({ points: leftState.points, t: leftState.t, edges: null, colors: null });
    } else if (leftState) {
      leftViz.setState({ points: leftState.points, t: leftState.t, edges: leftState.edges, colors: leftState.colors });
    }
    if (isFinal && rightState && rightState.embed2d) {
      rightViz2d.setState({ embed2d: rightState.embed2d, colors: rightState.colors, t: rightState.t });
      rightThumb.setState({ points: rightState.points, t: rightState.t, edges: null, colors: null });
    } else if (rightState) {
      rightViz.setState({ points: rightState.points, t: rightState.t, edges: rightState.edges, colors: rightState.colors });
    }

    stepIndicator.render({
      leftLabel: ALGORITHMS_BY_ID[s.leftAlgoId].label,
      rightLabel: ALGORITHMS_BY_ID[s.rightAlgoId].label,
      leftSubSteps: left.presentSubSteps,
      rightSubSteps: right.presentSubSteps,
      currentSubStep: s.currentSubStep,
    });
    leftIfw.setStep(leftState && leftSub === s.currentSubStep ? leftState.ifw : null);
    rightIfw.setStep(rightState && rightSub === s.currentSubStep ? rightState.ifw : null);
    leftPseudo.render({ algoLabel: ALGORITHMS_BY_ID[s.leftAlgoId].label, sections: ALGORITHMS_BY_ID[s.leftAlgoId].pseudocode, currentSubStep: leftSub });
    rightPseudo.render({ algoLabel: ALGORITHMS_BY_ID[s.rightAlgoId].label, sections: ALGORITHMS_BY_ID[s.rightAlgoId].pseudocode, currentSubStep: rightSub });
  });

  updateSyntheticVisibility();
  store.set({});
}

function nearestSub(target, present) {
  if (present.includes(target)) return target;
  const sorted = [...present].sort(compareSubSteps);
  let best = sorted[0];
  for (const id of sorted) if (compareSubSteps(id, target) <= 0) best = id;
  return best;
}

function setStep6Mode(host, isFinal) {
  const v3d = host.querySelector('.viz3d');
  const v2d = host.querySelector('.viz2d');
  const thumb = host.querySelector('.viz3d-thumb');
  if (!v3d || !v2d) return;
  if (isFinal) {
    v3d.style.display = 'none';
    v2d.style.display = '';
    if (thumb) thumb.style.display = '';
  } else {
    v3d.style.display = '';
    v2d.style.display = 'none';
    if (thumb) thumb.style.display = 'none';
  }
}

function hideEl(host, sel) {
  const el = host.querySelector(sel);
  if (el) el.style.display = 'none';
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
