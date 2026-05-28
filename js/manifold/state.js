import { DATASETS_BY_ID } from './datasets.js';

export function createState({ algorithmsById, defaults }) {
  const listeners = new Set();
  const state = {
    datasetId: defaults.datasetId,
    datasetParams: { ...defaults.datasetParams },
    csvRows: null,
    csvFileName: '',
    leftAlgoId: defaults.leftAlgoId,
    rightAlgoId: defaults.rightAlgoId,
    leftAlgoParams: { ...defaults.algoParams[defaults.leftAlgoId] },
    rightAlgoParams: { ...defaults.algoParams[defaults.rightAlgoId] },
    currentSubStep: '0',
    cache: { dataset: null, left: null, right: null, key: null },
  };

  function key() {
    return JSON.stringify({
      d: state.datasetId, dp: state.datasetParams,
      csv: state.csvRows ? state.csvRows.length : 0,
      la: state.leftAlgoId, lp: state.leftAlgoParams,
      ra: state.rightAlgoId, rp: state.rightAlgoParams,
    });
  }

  function recompute() {
    const k = key();
    if (state.cache.key === k) return;
    if (state.cache.left && state.cache.left.cancel) state.cache.left.cancel();
    if (state.cache.right && state.cache.right.cancel) state.cache.right.cancel();
    const ds = DATASETS_BY_ID[state.datasetId];
    const data = ds.generate({ ...state.datasetParams, csvRows: state.csvRows });
    if (data.empty) {
      state.cache = { dataset: data, left: null, right: null, key: k };
      return;
    }
    const left = algorithmsById[state.leftAlgoId].run(data, state.leftAlgoParams);
    const right = algorithmsById[state.rightAlgoId].run(data, state.rightAlgoParams);
    state.cache = { dataset: data, left, right, key: k };
    const onLeftProgress = () => { if (state.cache.left === left) emit(); };
    const onRightProgress = () => { if (state.cache.right === right) emit(); };
    if (left.start) left.start(onLeftProgress);
    if (right.start) right.start(onRightProgress);
  }

  function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }
  function emit() { for (const fn of listeners) fn(state); }
  function set(updates) { Object.assign(state, updates); recompute(); emit(); }
  function setStep(sub) {
    if (state.currentSubStep === sub) return;
    state.currentSubStep = sub;
    emit();
  }

  recompute();
  return { state, subscribe, set, setStep, recompute };
}
