import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PCA } from '../../../js/manifold/algorithms/pca.js';
import { ISOMAP } from '../../../js/manifold/algorithms/isomap.js';
import { MDS } from '../../../js/manifold/algorithms/mds.js';
import { LLE } from '../../../js/manifold/algorithms/lle.js';
import { LAPLACIAN } from '../../../js/manifold/algorithms/laplacian.js';
import { KPCA } from '../../../js/manifold/algorithms/kpca.js';

const ALGOS = [PCA, ISOMAP, MDS, LLE, LAPLACIAN, KPCA];

test('every algorithm parameter has a non-empty label and description', () => {
  for (const a of ALGOS) {
    for (const p of a.params) {
      assert.equal(typeof p.label, 'string', `${a.id}.${p.name} label type`);
      assert.ok(p.label.length > 0, `${a.id}.${p.name} label non-empty`);
      assert.equal(typeof p.desc, 'string', `${a.id}.${p.name} desc type`);
      assert.ok(p.desc.length > 10, `${a.id}.${p.name} desc non-trivial`);
    }
  }
});

test('LLE default k is 12', () => {
  const k = LLE.params.find(p => p.name === 'k');
  assert.equal(k.default, 12);
});
