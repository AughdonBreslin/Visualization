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

  let worker = null;
  let nextRunId = 1;
  const runHandlers = new Map();

  function ensureWorker() {
    if (worker) return worker;
    try {
      worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
      worker.addEventListener('message', (event) => {
        const msg = event.data;
        const handler = runHandlers.get(msg.runId);
        if (!handler) return;
        if (msg.type === 'step') handler.onStep(msg.stepId, msg.state);
        else if (msg.type === 'error') console.error('Worker error:', msg.error);
      });
      worker.addEventListener('error', (e) => {
        console.error('Worker runtime error:', e.message || e);
      });
    } catch (e) {
      console.warn('Web Worker unavailable, falling back to main thread:', e);
      worker = null;
    }
    return worker;
  }

  function startWorkerRun(algoId, data, params, onStep) {
    const w = ensureWorker();
    const runId = nextRunId++;
    if (!w) {
      const result = algorithmsById[algoId].run(data, params);
      const step0 = result.steps.get('0');
      if (step0) Promise.resolve().then(() => onStep('0', step0));
      if (result.start) {
        result.start((stepId) => {
          const state = result.steps.get(stepId);
          onStep(stepId, state);
        });
      }
      return { runId, cancel: result.cancel || (() => {}) };
    }
    runHandlers.set(runId, { onStep });
    w.postMessage({
      type: 'run', runId, algoId,
      X: data.X.buffer.slice(0),
      t: data.t.buffer.slice(0),
      params,
    });
    return {
      runId,
      cancel() {
        runHandlers.delete(runId);
        w.postMessage({ type: 'cancel', runId });
      },
    };
  }

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

    const leftAlgo = algorithmsById[state.leftAlgoId];
    const rightAlgo = algorithmsById[state.rightAlgoId];

    const left = {
      steps: new Map(),
      presentSubSteps: leftAlgo.presentSubSteps,
      pending: new Set(leftAlgo.presentSubSteps),
      cancel: null,
    };
    const right = {
      steps: new Map(),
      presentSubSteps: rightAlgo.presentSubSteps,
      pending: new Set(rightAlgo.presentSubSteps),
      cancel: null,
    };
    state.cache = { dataset: data, left, right, key: k };

    const leftRun = startWorkerRun(state.leftAlgoId, data, state.leftAlgoParams, (stepId, stepState) => {
      if (state.cache.left !== left) return;
      left.steps.set(stepId, stepState);
      left.pending.delete(stepId);
      emit();
    });
    left.cancel = leftRun.cancel;

    const rightRun = startWorkerRun(state.rightAlgoId, data, state.rightAlgoParams, (stepId, stepState) => {
      if (state.cache.right !== right) return;
      right.steps.set(stepId, stepState);
      right.pending.delete(stepId);
      emit();
    });
    right.cancel = rightRun.cancel;
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
