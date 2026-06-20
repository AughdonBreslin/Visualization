import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DATASETS } from '../../../js/manifold/datasets/index.js';

test('every dataset parameter has a non-empty label and description', () => {
  for (const d of DATASETS) {
    for (const p of (d.params || [])) {
      assert.equal(typeof p.label, 'string', `${d.id}.${p.name} label type`);
      assert.ok(p.label.length > 0, `${d.id}.${p.name} label non-empty`);
      assert.equal(typeof p.desc, 'string', `${d.id}.${p.name} desc type`);
      assert.ok(p.desc.length > 10, `${d.id}.${p.name} desc non-trivial`);
    }
  }
});
