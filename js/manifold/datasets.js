// Re-export shim. The dataset implementations now live in js/manifold/datasets/.
export { DATASETS, DATASETS_BY_ID, parseCSV } from './datasets/index.js';
export { SWISS_ROLL, S_CURVE, HELIX, TREFOIL_KNOT, TOROIDAL_HELIX, SPIRAL_DISK } from './datasets/synthetic_curves.js';
export { TWIN_PEAKS, SADDLE, CYLINDER, SEVERED_SPHERE, HILBERT, FULL_SPHERE } from './datasets/synthetic_surfaces.js';
export { CLUSTERS_3D } from './datasets/synthetic_clusters.js';
export { CSV_UPLOAD } from './datasets/csv_upload.js';
