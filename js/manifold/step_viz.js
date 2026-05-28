import { createViz3d } from './viz3d.js';
import { createViz2d } from './viz2d.js';
import { mountCentering } from './viz/viz_centering.js';
import { mountKnn } from './viz/viz_knn.js';
import { mountMatrixStrip } from './viz/viz_matrix_strip.js';
import { mountSpectral } from './viz/viz_spectral.js';

export function createStepViz(host) {
  let activeKind = null;
  let active = null;
  const host3dThumb = host;
  let viz3d = null;
  let viz2d = null;
  let thumb = null;

  function ensure3d() {
    if (!viz3d) viz3d = createViz3d(host, {});
    return viz3d;
  }
  function ensure2d() {
    if (!viz2d) viz2d = createViz2d(host, {});
    return viz2d;
  }
  function ensureThumb() {
    if (!thumb) thumb = createViz3d(host3dThumb, { width: 140, height: 110, isThumbnail: true });
    return thumb;
  }

  function setVisible(sel, visible) {
    const el = host.querySelector(sel);
    if (el) el.style.display = visible ? '' : 'none';
  }

  function update(state) {
    if (!state) return;
    const kind = state.vizKind || 'point_cloud';

    if (active && activeKind !== kind && active.unmount) {
      active.unmount();
      active = null;
    }
    activeKind = kind;

    if (kind === 'point_cloud') {
      ensure3d();
      setVisible('.viz3d', true);
      setVisible('.viz2d', false);
      setVisible('.viz3d-thumb', false);
      viz3d.setState({ points: state.points, t: state.t, edges: null, colors: state.colors });
    } else if (kind === 'centering') {
      setVisible('.viz3d', false);
      setVisible('.viz2d', false);
      setVisible('.viz3d-thumb', false);
      active = mountCentering(host, state);
    } else if (kind === 'knn_graph') {
      setVisible('.viz3d', false);
      setVisible('.viz2d', false);
      setVisible('.viz3d-thumb', false);
      active = mountKnn(host, state);
    } else if (kind === 'matrix_strip') {
      setVisible('.viz3d', false);
      setVisible('.viz2d', false);
      setVisible('.viz3d-thumb', false);
      active = mountMatrixStrip(host, state);
    } else if (kind === 'spectral') {
      setVisible('.viz3d', false);
      setVisible('.viz2d', false);
      setVisible('.viz3d-thumb', false);
      active = mountSpectral(host, state);
    } else if (kind === 'embedding') {
      ensure2d();
      ensureThumb();
      setVisible('.viz3d', false);
      setVisible('.viz2d', true);
      setVisible('.viz3d-thumb', true);
      viz2d.setState({ embed2d: state.embed2d, colors: state.colors, t: state.t });
      thumb.setState({ points: state.points, t: state.t, edges: null, colors: null });
    }
  }

  return { update };
}
