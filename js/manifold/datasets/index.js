import { SWISS_ROLL, S_CURVE, HELIX, TREFOIL_KNOT, TOROIDAL_HELIX, SPIRAL_DISK } from './synthetic_curves.js';
import { TWIN_PEAKS, SADDLE, CYLINDER, SEVERED_SPHERE, PUNCTURED_SPHERE, FULL_SPHERE } from './synthetic_surfaces.js';
import { CLUSTERS_3D } from './synthetic_clusters.js';
import { CSV_UPLOAD, parseCSV } from './csv_upload.js';

export const DATASETS = [
  SWISS_ROLL, S_CURVE, HELIX, TREFOIL_KNOT, TOROIDAL_HELIX, SPIRAL_DISK,
  TWIN_PEAKS, SADDLE, CYLINDER, SEVERED_SPHERE, PUNCTURED_SPHERE, FULL_SPHERE,
  CLUSTERS_3D, CSV_UPLOAD,
];
export const DATASETS_BY_ID = Object.fromEntries(DATASETS.map(d => [d.id, d]));
export { parseCSV };
