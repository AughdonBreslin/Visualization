import { PCA } from './algorithms/pca.js';
import { ISOMAP } from './algorithms/isomap.js';
import { MDS } from './algorithms/mds.js';
import { LLE } from './algorithms/lle.js';
import { LAPLACIAN } from './algorithms/laplacian.js';
import { KPCA } from './algorithms/kpca.js';

const ALGORITHMS = { pca: PCA, isomap: ISOMAP, mds: MDS, lle: LLE, laplacian: LAPLACIAN, kpca: KPCA };
const runs = new Map();

self.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === 'run') {
    const algo = ALGORITHMS[msg.algoId];
    if (!algo) {
      self.postMessage({ type: 'error', runId: msg.runId, error: `Unknown algorithm ${msg.algoId}` });
      return;
    }
    const dataset = {
      X: new Float64Array(msg.X),
      t: new Float64Array(msg.t),
    };
    const result = algo.run(dataset, msg.params);
    runs.set(msg.runId, result);

    const step0 = result.steps.get('0');
    if (step0) self.postMessage({ type: 'step', runId: msg.runId, stepId: '0', state: step0 });

    if (result.start) {
      result.start((stepId) => {
        const state = result.steps.get(stepId);
        self.postMessage({ type: 'step', runId: msg.runId, stepId, state });
      });
    }
  } else if (msg.type === 'cancel') {
    const r = runs.get(msg.runId);
    if (r && r.cancel) r.cancel();
    runs.delete(msg.runId);
  }
};
