// manifold_isomap.js - Isomap explainer player (ES module).
// Plays one continuous video and treats the steps as chapter markers, so playback
// is seamless across step boundaries and the active step always matches the
// pseudocode timeline in the top-left of the video. `start` is the step's start
// time in seconds (the section boundaries of the rendered walkthrough).
// Explanations mirror manimexp/isomap/walkthrough_explained.md.
import { typesetMath } from './manifold/mathjax.js';

const STEPS_ISOMAP = [
  {
    start: 0,
    title: '1. Raw data',
    caption: 'A 2D sheet rolled up in 3D; the goal is to recover the flat sheet.',
    explain: 'The data is a Swiss roll: a flat two-dimensional sheet rolled up inside ' +
      'three-dimensional space, so it has only two genuine degrees of freedom. Two points ' +
      'can sit close together in 3D yet be far apart along the sheet, the way two layers of ' +
      'a rolled poster nearly touch but are far apart along the paper. Recovering the flat ' +
      'sheet means finding a 2D coordinate for every point that respects distance along the ' +
      'surface. Color runs along a full rainbow so each region is easy to follow.',
    formula: '',
  },
  {
    start: 13.5,
    title: '2. kNN graph',
    caption: 'Link each point to its k = 8 nearest neighbors, weighted by distance.',
    explain: 'Every point is connected to its $k = 8$ closest points, and each link is ' +
      'weighted by the straight-line distance between its endpoints. Over a short hop ' +
      'between near neighbors that distance stays on the surface, so the weights are good ' +
      'local estimates of true surface distance. Done for every point, the links form a ' +
      'mesh that follows the sheet and serves as the scaffold for measuring distance along it.',
    formula: 'w_{ij} = \\lVert x_i - x_j \\rVert',
  },
  {
    start: 32.9,
    title: '3. Geodesic distances',
    caption: 'Distance measured along the graph, not straight through space.',
    explain: 'A geodesic distance is the length of the shortest route that stays on the ' +
      'surface: on the neighbor graph, the shortest path between two nodes, found by adding ' +
      'up edge weights (Dijkstra). A straight chord cuts directly through the empty space ' +
      'between the rolls; it is shorter in 3D but meaningless for the sheet. The cloud is ' +
      'colored by geodesic distance from one source across a full rainbow, so two points on ' +
      'different turns of the roll receive very different colors.',
    formula: 'D_{ij} = \\text{shortest path } i \\to j',
  },
  {
    start: 55.27,
    title: '4. Double-centering',
    caption: 'Turn the geodesic distances into the Gram matrix B of inner products.',
    explain: 'A distance table cannot pin down coordinates, because rotating or shifting all ' +
      'the points together leaves every pairwise distance unchanged. Inner products can: ' +
      '$G_{ij} = x_i \\cdot x_j$ records the two points’ lengths and the angle between ' +
      'them from a shared origin, and stacking them gives $G = X X^\\top$, one matrix product ' +
      'away from the coordinates. $G$ captures the relative geometry, a bridge between ' +
      'distances and coordinates. We do not have the coordinates to form $G$ directly, so we ' +
      'build it from distances: square every entry, then double-center, ' +
      '$B = -\\tfrac{1}{2} J D^2 J$. The result $B$ is the Gram matrix of the centered points.',
    formula: 'B = -\\tfrac{1}{2} J D^2 J',
  },
  {
    start: 120.07,
    title: '5. Eigendecomposition',
    caption: 'The top eigenvectors of B carry the recovered shape.',
    explain: 'Factor $B = V \\Lambda V^{\\top}$. Power iteration finds the dominant eigenvector: ' +
      'start from a random unit vector and repeatedly apply $v \\leftarrow Bv / \\lVert Bv \\rVert$. ' +
      'The Rayleigh quotient $v^\\top B v$, shown as an actual matrix product each step, climbs to ' +
      'the largest eigenvalue $\\lambda_1$. The eigenvectors with the largest eigenvalues are the ' +
      'directions in which the centered points vary the most, the genuine low-dimensional ' +
      'structure; the top two are kept.',
    formula: 'B = V \\Lambda V^{\\top}',
  },
  {
    start: 155.27,
    title: '6. Embedding',
    caption: 'The sheet unrolls into 2D, geodesic distances preserved.',
    explain: 'The recovered coordinates are $Y = [\\sqrt{\\lambda_1}\\, v_1,\\ ' +
      '\\sqrt{\\lambda_2}\\, v_2]$: each kept eigenvector scaled by the square root of its ' +
      'eigenvalue. The cloud settles into a flat band, the sheet unrolled, and the rainbow ' +
      'geodesic coloring now varies smoothly across the flat layout, the visual confirmation ' +
      'that along-the-sheet distances were preserved while the extra dimension was removed.',
    formula: 'Y = [\\sqrt{\\lambda_1} v_1,\\ \\sqrt{\\lambda_2} v_2]',
  },
];

// PCA steps (chapter markers are the rendered section boundaries of the PCA clip).
const STEPS_PCA = [
  {
    start: 0,
    title: '1. Raw data',
    caption: 'A curved cloud in 3D; PCA looks for the directions of greatest variance.',
    explain: 'PCA is a linear method: it keeps the directions along which the data varies most. ' +
      'It cannot bend, so it will project the cloud onto a plane rather than unroll it.',
    formula: '',
  },
  {
    start: 6.3,
    title: '2. Center',
    caption: 'Subtract the mean so the cloud sits at the origin.',
    explain: 'Variance is measured about a fixed origin, so the first step moves the centroid to ' +
      'the origin by subtracting the mean from every point.',
    formula: 'x_i \\leftarrow x_i - \\bar{x}',
  },
  {
    start: 12.3,
    title: '3. Covariance',
    caption: 'Form the 3 by 3 covariance matrix of the centered coordinates.',
    explain: 'The covariance matrix $C = \\tfrac{1}{N-1} X_c^{\\top} X_c$ summarizes how the three ' +
      'coordinates vary together. Its eigenvectors are the directions of variance.',
    formula: 'C = \\tfrac{1}{N-1}\\, X_c^{\\top} X_c',
  },
  {
    start: 20.9,
    title: '4. Principal axes',
    caption: 'Eigenvectors of C are the principal axes, ordered by variance.',
    explain: 'Eigendecomposing $C = V \\Lambda V^{\\top}$ gives orthogonal axes; the eigenvalue ' +
      '$\\lambda_k$ is the variance along axis $v_k$. The largest is the first principal component.',
    formula: 'C = V \\Lambda V^{\\top}',
  },
  {
    start: 30.7,
    title: '5. Project to 2D',
    caption: 'Project onto the top two axes, dropping the third direction.',
    explain: 'Keeping the two highest-variance axes and discarding the third gives the 2D ' +
      'coordinates $Y = X_c[v_1\\ v_2]$. This is an orthogonal projection onto a plane.',
    formula: 'Y = X_c\\, [\\, v_1 \\;\\; v_2 \\,]',
  },
  {
    start: 37.8,
    title: '6. Embedding',
    caption: 'PCA flattens by projection, so the rolled layers overlap; a linear method cannot unroll the swiss roll.',
    explain: 'Because the projection is linear, points that are far apart along the rolled sheet ' +
      'but close in 3D land on top of each other, so the color ordering is scrambled. This is ' +
      'exactly where a nonlinear method like Isomap, which measures distance along the surface, ' +
      'succeeds and PCA does not.',
    formula: 'Y = X_c\\, [\\, v_1 \\;\\; v_2 \\,]',
  },
];

// MDS steps (chapter markers from the rendered mds/walkthrough.mp4 section boundaries).
// The algorithm preserves straight-line Euclidean distances, so it cannot unroll the
// swiss roll; the embedding step makes that contrast explicit.
const STEPS_MDS = [
  {
    start: 0,
    title: '1. Raw data',
    caption: 'A 2D sheet rolled up in 3D. MDS looks for a 2D layout that preserves pairwise distances.',
    explain: 'The data is a Swiss roll: a flat two-dimensional sheet coiled inside three-dimensional ' +
      'space. MDS (classical multidimensional scaling) starts from an N by N table of ' +
      'pairwise distances and recovers a low-dimensional layout that best preserves those ' +
      'distances in the least-squares sense. Because MDS uses straight-line distances ' +
      'through 3D space rather than distances along the sheet surface, the rolled layers ' +
      'will remain overlapping in the output. Isomap improves on this by replacing ' +
      'straight-line distances with geodesic distances.',
    formula: '',
  },
  {
    start: 7.80,
    title: '2. Pairwise distances',
    caption: 'Collect all N squared pairwise Euclidean distances into the distance matrix D.',
    explain: 'Every pair of points $(i, j)$ contributes one entry $D_{ij} = \\|x_i - x_j\\|$ ' +
      'to the N by N distance matrix. The matrix is symmetric ($D_{ij} = D_{ji}$) and has ' +
      'zeros on the diagonal. For the Swiss roll the straight-line distance between two ' +
      'points on different turns of the roll can be much shorter than the path along the ' +
      'surface, so $D$ does not capture the true sheet geometry. The heatmap visualises ' +
      'the full matrix: bright cells are large distances, dark cells are small.',
    formula: 'D_{ij} = \\lVert x_i - x_j \\rVert',
  },
  {
    start: 20.43,
    title: '3. Double-centering',
    caption: 'Turn the squared distance matrix into the Gram matrix B of inner products.',
    explain: 'A distance table cannot pin down coordinates on its own: translating or rotating ' +
      'all points leaves every pairwise distance unchanged. Inner products can, because ' +
      '$G_{ij} = x_i \\cdot x_j$ fixes both lengths and angles relative to a shared origin. ' +
      'The double-centering step recovers inner products from distances: square every entry ' +
      'to get $D^2$, then apply $B = -\\tfrac{1}{2} H D^2 H$ where $H = I - \\tfrac{1}{N} ' +
      '\\mathbf{1}\\mathbf{1}^{\\top}$ is the centering matrix. The result $B$ equals the ' +
      'Gram matrix of the centered point cloud, a symmetric positive-semidefinite matrix ' +
      'whose entries are inner products relative to the centroid.',
    formula: 'B = -\\tfrac{1}{2}\\, H D^2 H,\\quad H = I - \\tfrac{1}{N}\\mathbf{1}\\mathbf{1}^{\\top}',
  },
  {
    start: 36.23,
    title: '4. Eigendecomposition',
    caption: 'Factor B into eigenvectors and eigenvalues to find the directions of largest spread.',
    explain: 'Decompose $B = V \\Lambda V^{\\top}$. Each eigenvector $v_k$ is a direction in ' +
      'the N-dimensional inner-product space, and the corresponding eigenvalue $\\lambda_k$ ' +
      'measures the variance of the cloud along that direction. The largest eigenvalue ' +
      'corresponds to the dominant axis of spread; the second largest corresponds to the ' +
      'next. Both eigenvalues are positive because $B$ is positive semidefinite when the ' +
      'distances come from a Euclidean space. Only the top two eigenpairs are needed to ' +
      'construct the 2D embedding.',
    formula: 'B = V \\Lambda V^{\\top}',
  },
  {
    start: 52.83,
    title: '5. Embedding',
    caption: 'Scale the top eigenvectors by the square roots of their eigenvalues to get 2D coordinates.',
    explain: 'The 2D embedding is $Y = [\\sqrt{\\lambda_1}\\, v_1,\\ \\sqrt{\\lambda_2}\\, v_2]$: ' +
      'each eigenvector is scaled by the square root of its eigenvalue so that the variance ' +
      'along each axis matches the original geometry. Because $D$ was built from straight-line ' +
      '3D distances, the Swiss roll\'s layers still overlap in the output: points that are ' +
      'close in 3D but far along the sheet land near each other, so the color ordering is ' +
      'scrambled. Isomap repairs this by substituting geodesic distances for Euclidean ones ' +
      'before the double-centering step.',
    formula: 'Y = [\\sqrt{\\lambda_1} v_1,\\ \\sqrt{\\lambda_2} v_2]',
  },
];

// LLE steps (chapter markers from the rendered lle/walkthrough.mp4 section boundaries).
const STEPS_LLE = [
  {
    start: 0,
    title: '1. Raw data',
    caption: 'A 2D sheet rolled up in 3D. LLE recovers the flat sheet by preserving local linear structure.',
    explain: 'Locally Linear Embedding (LLE) exploits a different insight than distance-based ' +
      'methods: if the manifold is locally flat, every point can be written as a weighted ' +
      'sum of its nearby neighbors. The weights that describe those local linear ' +
      'combinations are geometry-invariant; they stay the same whether the sheet is rolled ' +
      'or flat. LLE finds a low-dimensional layout that satisfies the same reconstruction ' +
      'weights, effectively unrolling the sheet by demanding that local patches unfold ' +
      'consistently.',
    formula: '',
  },
  {
    start: 8.80,
    title: '2. kNN graph',
    caption: 'Link each point to its k nearest neighbors to define local patches.',
    explain: 'Every point is connected to its $k$ closest points by Euclidean distance, ' +
      'forming a neighborhood graph. Each such neighborhood defines a small locally flat ' +
      'patch: because $k$ is small, the patch stays close to the tangent plane of the ' +
      'manifold, even when the manifold curves sharply. The $k$ nearest neighbors of one ' +
      'central point are highlighted to show a single patch; the full graph covers the ' +
      'entire surface. The choice of $k$ controls how much local curvature the method ' +
      'tolerates: too small and isolated points appear; too large and distant folds get ' +
      'linked across layers.',
    formula: '\\mathcal{N}_i = \\{ j : \\lVert x_j - x_i \\rVert \\text{ among the } k \\text{ smallest}\\}',
  },
  {
    start: 36.90,
    title: '3. Reconstruction weights',
    caption: 'For each point, find weights so it is a weighted combination of its k neighbors with weights summing to 1.',
    explain: 'For point $i$, find coefficients $w_{ij}$ (nonzero only for neighbors $j \\in \\mathcal{N}_i$) ' +
      'that minimize the reconstruction error $\\|x_i - \\sum_j w_{ij}\\, x_j\\|^2$ subject to ' +
      '$\\sum_j w_{ij} = 1$. This is a small constrained least-squares problem for each point ' +
      'independently: subtract $x_i$ from each neighbor to center the system, solve the ' +
      'resulting $k \\times k$ Gram system, then normalize. The weight matrix $W$ collects ' +
      'all coefficients; it is sparse because each row has at most $k$ nonzero entries. ' +
      'These weights capture the local geometry of the patch and are invariant to rotations, ' +
      'translations, and rescalings of the patch.',
    formula: '\\min_{w_i}\\ \\bigl\\lVert x_i - \\textstyle\\sum_{j \\in \\mathcal{N}_i} w_{ij}\\, x_j \\bigr\\rVert^2,\\quad \\sum_j w_{ij} = 1',
  },
  {
    start: 61.90,
    title: '4. Eigenvectors of M',
    caption: 'Form M = (I - W) transpose times (I - W) and take its smallest non-trivial eigenvectors.',
    explain: 'The embedding cost is $\\Phi(Y) = \\sum_i \\|y_i - \\sum_j w_{ij} y_j\\|^2$, ' +
      'which measures how well the same weights reconstruct the low-dimensional coordinates. ' +
      'Collecting terms gives $\\Phi(Y) = \\mathrm{tr}(Y^{\\top} M Y)$ where ' +
      '$M = (I - W)^{\\top}(I - W)$. Minimizing over unit-variance, zero-mean embeddings ' +
      'means taking the eigenvectors of $M$ with the smallest nonzero eigenvalues. The ' +
      'trivial zero eigenvalue corresponds to the constant eigenvector, which assigns the ' +
      'same coordinate to every point and carries no information; it is skipped. The next ' +
      'two eigenvectors become the two embedding axes.',
    formula: 'M = (I - W)^{\\top}(I - W)',
  },
  {
    start: 86.60,
    title: '5. Embedding',
    caption: 'The two smallest non-trivial eigenvectors of M give the 2D coordinates directly.',
    explain: 'Unlike MDS and Kernel PCA, LLE does not scale the eigenvectors by eigenvalues: ' +
      'the embedding is simply $Y = [v_1\\ v_2]$, the N-by-2 matrix of the two eigenvectors. ' +
      'The absolute scale is fixed by the normalization constraint built into the cost ' +
      'function. Because the weights $w_{ij}$ preserve local linear reconstructions, nearby ' +
      'points on the manifold remain nearby in the embedding, and the sheet unrolls into a ' +
      'flat layout with the color gradient running smoothly across it.',
    formula: 'Y = [\\, v_1 \\;\\; v_2 \\,]',
  },
];

// Laplacian Eigenmaps steps (chapter markers from laplacian/walkthrough.mp4).
const STEPS_LAPLACIAN = [
  {
    start: 0,
    title: '1. Raw data',
    caption: 'A 2D sheet rolled up in 3D. Laplacian Eigenmaps recovers the flat sheet by preserving local connections.',
    explain: 'Laplacian Eigenmaps builds on the observation that a smooth function on a manifold ' +
      'changes slowly between nearby points. The method constructs a graph that encodes ' +
      'local proximity, weights each edge by how close the two endpoints are, and then looks ' +
      'for embedding coordinates that vary as smoothly as possible across the graph. Points ' +
      'that are strongly connected in the graph are pulled close together in the embedding.',
    formula: '',
  },
  {
    start: 8.80,
    title: '2. kNN graph',
    caption: 'Link each point to its k nearest neighbors to capture the local structure of the surface.',
    explain: 'Every point is connected to its $k$ nearest Euclidean neighbors, forming a sparse ' +
      'neighborhood graph. Short edges stay on the surface of the manifold, so the graph ' +
      'faithfully represents the local topology even where the sheet is tightly rolled. ' +
      'Long-range links across separate layers of the roll are absent because the ' +
      'neighborhood radius is small. This graph is the skeleton on which the heat-kernel ' +
      'weights are defined in the next step.',
    formula: '\\mathcal{N}_i = \\{ j : \\lVert x_j - x_i \\rVert \\text{ among the } k \\text{ smallest}\\}',
  },
  {
    start: 41.10,
    title: '3. Heat-kernel affinity',
    caption: 'Turn each edge distance into a weight using the heat kernel; nearby pairs get values close to 1.',
    explain: 'For each edge $(i, j)$ in the kNN graph the affinity is ' +
      '$W_{ij} = \\exp(-\\|x_i - x_j\\|^2 / 2\\sigma^2)$. The bandwidth $\\sigma$ controls ' +
      'how quickly similarity decays with distance: a large $\\sigma$ makes all neighbors ' +
      'nearly equally similar; a small $\\sigma$ sharpens the distinction between near and ' +
      'far neighbors. Non-neighbor pairs are exactly zero, keeping $W$ sparse. The heatmap ' +
      'shows that pairs on the same turn of the roll receive large weights while pairs ' +
      'across turns (which are not kNN neighbors) receive zero.',
    formula: 'W_{ij} = \\exp\\!\\left(-\\frac{\\lVert x_i - x_j \\rVert^2}{2\\sigma^2}\\right)',
  },
  {
    start: 54.70,
    title: '4. Graph Laplacian',
    caption: 'Form the degree matrix D and subtract W to get the graph Laplacian L = D - W.',
    explain: 'The degree matrix $D$ is diagonal: $D_{ii} = \\sum_j W_{ij}$ is the total ' +
      'affinity at node $i$, a measure of how densely connected it is. The graph ' +
      'Laplacian $L = D - W$ has positive diagonal entries equal to the degree and ' +
      'non-positive off-diagonal entries equal to $-W_{ij}$. Its smallest eigenvectors ' +
      'are the smoothest functions on the graph: the one with eigenvalue zero is the ' +
      'constant function (all points equal), and the next ones vary as gently as possible ' +
      'across heavily-weighted edges. These smooth functions become the embedding coordinates.',
    formula: 'L = D - W,\\quad D_{ii} = \\sum_j W_{ij}',
  },
  {
    start: 74.90,
    title: '5. Eigenvectors of L',
    caption: 'Take the two smallest non-trivial eigenvectors of L; they are the smoothest non-constant functions on the graph.',
    explain: 'Solving $L v_k = \\lambda_k v_k$ in order of increasing eigenvalue gives the ' +
      'sequence of graph-smooth functions. The trivial eigenvalue $\\lambda_0 = 0$ ' +
      'corresponds to the constant eigenvector, which assigns the same coordinate to ' +
      'every point and carries no discriminating information; it is discarded. The next ' +
      'two eigenvectors $v_1$ and $v_2$, with the smallest nonzero eigenvalues, minimize ' +
      'the total energy $\\sum_{ij} W_{ij} (v_k(i) - v_k(j))^2$, so connected points stay ' +
      'close in the embedding. This is why we take the smallest eigenvalues here, not the ' +
      'largest: Laplacian Eigenmaps and LLE minimize a locality cost, so the useful ' +
      'directions are the smoothest, slowest-varying functions on the graph, whereas PCA, ' +
      'MDS, and Kernel PCA maximize variance and keep the largest eigenvalues. Coloring the ' +
      'original cloud by $v_1$ shows it traces the dominant unrolled axis of the sheet.',
    formula: 'L\\, v_k = \\lambda_k\\, v_k\\quad (\\text{skip } \\lambda_0 = 0)',
  },
  {
    start: 106.10,
    title: '6. Embedding',
    caption: 'Use the two smallest non-trivial eigenvectors directly as 2D coordinates.',
    explain: 'The 2D embedding is $Y = [v_1\\ v_2]$: each column is one of the two ' +
      'smoothest non-constant eigenvectors. No eigenvalue scaling is applied. Because ' +
      'the eigenvectors minimize variation across strongly weighted edges, nearby points ' +
      'on the manifold receive similar coordinates in the embedding. The Swiss roll ' +
      'flattens into a smooth band, with the rainbow color gradient varying continuously ' +
      'across the 2D layout, confirming that local proximity was preserved.',
    formula: 'Y = [\\, v_1 \\;\\; v_2 \\,]',
  },
];

// Helper that builds the five KPCA steps given the kernel formula string and
// the per-kernel outro text. Steps 1, 3, 4, and 5 are identical across kernels;
// only step 2 (the kernel formula display) and step 5 (the outro) vary.
function makeKpcaSteps(kernelFormula, kernelExplain, outroText) {
  return [
    {
      start: 0,
      title: '1. Raw data',
      caption: 'A 2D sheet rolled up in 3D. Kernel PCA replaces the inner product with a kernel, then runs PCA in the implicit feature space.',
      explain: 'Ordinary PCA finds directions of maximum variance in the original coordinate space. ' +
        'Kernel PCA replaces the dot product $x_i \\cdot x_j$ with a kernel function $k(x_i, x_j)$ ' +
        'that implicitly measures similarity in a higher-dimensional feature space $\\phi(x)$. ' +
        'The rest of the pipeline (centering, eigendecomposition, projection) is identical to ' +
        'standard PCA, but now operates on the kernel matrix rather than the data matrix. ' +
        'The choice of kernel determines which kinds of nonlinear structure the method can resolve.',
      formula: '',
    },
    {
      start: 8.80,
      title: '2. Kernel matrix K',
      caption: 'Evaluate the kernel function for every pair of points to form the N by N matrix K.',
      explain: kernelExplain,
      formula: kernelFormula,
    },
    {
      start: 20.40,
      title: '3. Center K',
      caption: 'Center the kernel matrix so the implicit feature-space cloud has zero mean.',
      explain: 'Before running PCA in feature space, the data must be centered. Because the ' +
        'feature map $\\phi$ is implicit, the centering is applied directly to $K$: ' +
        '$K_c = K - \\mathbf{1}_N K - K \\mathbf{1}_N + \\mathbf{1}_N K \\mathbf{1}_N$, ' +
        'where $\\mathbf{1}_N$ is the $N \\times N$ matrix with all entries $1/N$. The ' +
        'subtraction of row means and column means removes the per-sample shift, and adding ' +
        'back the grand mean corrects the double subtraction. The result $K_c$ is the Gram ' +
        'matrix of the centered feature vectors $\\phi(x_i) - \\bar{\\phi}$.',
      formula: 'K_c = K - \\mathbf{1}_N K - K \\mathbf{1}_N + \\mathbf{1}_N K \\mathbf{1}_N',
    },
    {
      start: 30.00,
      title: '4. Eigendecomposition',
      caption: 'Find the top eigenvectors of K_c; these are the principal components in feature space.',
      explain: 'The centered kernel matrix satisfies $K_c = V \\Lambda V^{\\top}$. Each eigenvector ' +
        '$v_k$ encodes the coefficients that project a new point into the $k$-th principal ' +
        'component of the feature space. The top eigenvalues measure how much variance each ' +
        'component captures. Because $K_c$ is positive semidefinite, all eigenvalues are ' +
        'non-negative. Two eigenpairs are kept for the 2D embedding, and the eigenvalues ' +
        'set the scale of each axis in the output.',
      formula: 'K_c = V \\Lambda V^{\\top}',
    },
    {
      start: 45.40,
      title: '5. Embedding',
      caption: outroText,
      explain: 'The 2D embedding coordinates are $y_{i,k} = \\sqrt{\\lambda_k}\\, v_{k,i}$: each ' +
        'eigenvector entry is scaled by the square root of its eigenvalue, matching the ' +
        'convention from classical MDS and PCA. The result places each point in the ' +
        'feature space where similarity is measured by the chosen kernel. ' +
        'Whether the Swiss roll unrolls depends on whether the kernel can separate ' +
        'points that are far along the sheet but close in 3D.',
      formula: 'y_{i,k} = \\sqrt{\\lambda_k}\\, v_{k,i}',
    },
  ];
}

const STEPS_KPCA_RBF = makeKpcaSteps(
  'K_{ij} = \\exp(-\\gamma \\lVert x_i - x_j \\rVert^2)',
  'The RBF (radial basis function) kernel $K_{ij} = \\exp(-\\gamma \\|x_i - x_j\\|^2)$ measures ' +
    'similarity by Gaussian falloff: two points that are close in 3D receive a value near 1; ' +
    'two points that are far apart receive a value near 0. The parameter $\\gamma$ controls ' +
    'the width of the Gaussian. For the Swiss roll, the RBF kernel assigns near-zero ' +
    'similarity to points that are far apart in 3D, but kernel PCA still keeps the directions ' +
    'of greatest variance in feature space. Those directions do not line up with the roll ' +
    'angle or the height of the sheet, so the top two components do not unroll the manifold.',
  'RBF kernel PCA produces a nonlinear embedding, but it does not recover the flat sheet. The top two components do not align with the roll angle or the height, so the swiss roll stays rolled in this 2D view. Unrolling it needs a method built to preserve manifold structure, like Isomap, LLE, or Laplacian Eigenmaps.',
);

const STEPS_KPCA_POLY = makeKpcaSteps(
  'K_{ij} = (x_i \\cdot x_j + 1)^d',
  'The polynomial kernel $K_{ij} = (x_i \\cdot x_j + 1)^d$ maps each point into a feature ' +
    'space of monomials up to degree $d$. The addition of the constant 1 ensures that ' +
    'lower-degree terms are included. Higher degree $d$ bends the feature space more, ' +
    'capturing higher-order interactions among the three coordinates. For the Swiss roll, ' +
    'the polynomial kernel creates a curved feature space, but the specific structure of ' +
    'the kernel does not cleanly separate the rolled layers, so the embedding remains ' +
    'partially entangled.',
  'Polynomial kernel PCA bends the feature space but does not cleanly separate the rolled layers; the embedding remains partially tangled.',
);

const STEPS_KPCA_LINEAR = makeKpcaSteps(
  'K_{ij} = x_i \\cdot x_j',
  'The linear kernel $K_{ij} = x_i \\cdot x_j$ is the ordinary Euclidean dot product. ' +
    'Kernel PCA with a linear kernel is exactly equivalent to standard PCA: the kernel ' +
    'matrix $K$ equals $X X^{\\top}$ (up to centering), and eigendecomposing it yields ' +
    'the same principal components as eigendecomposing the covariance matrix. Because ' +
    'PCA is a linear projection, it cannot separate the Swiss roll\'s layers: two points ' +
    'on different turns of the roll that are close in 3D will land near each other in ' +
    'the 2D output.',
  'Linear kernel PCA collapses to ordinary PCA and projects by variance; the rolled layers overlap as they do in standard PCA.',
);

const STEPS_BY_ALGO = {
  isomap: STEPS_ISOMAP,
  pca: STEPS_PCA,
  mds: STEPS_MDS,
  lle: STEPS_LLE,
  laplacian: STEPS_LAPLACIAN,
  kpca_rbf: STEPS_KPCA_RBF,
  kpca_polynomial: STEPS_KPCA_POLY,
  kpca_linear: STEPS_KPCA_LINEAR,
};
let STEPS = STEPS_ISOMAP;

const ASSET_BASE = '../assets/manim/';

// Poster image for the video element. Updated when the algorithm changes so the
// correct frame is shown before the clip loads.
function posterSrc(algo) {
  if (algo === 'kpca_rbf')        return ASSET_BASE + 'kpca/poster-rbf.png';
  if (algo === 'kpca_polynomial') return ASSET_BASE + 'kpca/poster-polynomial.png';
  if (algo === 'kpca_linear')     return ASSET_BASE + 'kpca/poster-linear.png';
  // Isomap and PCA predate the poster.png convention and ship their own frames.
  if (algo === 'isomap') return ASSET_BASE + 'isomap/step-1.png';
  if (algo === 'pca')    return ASSET_BASE + 'pca/walkthrough.png';
  return ASSET_BASE + algo + '/poster.png';
}

// Friendly labels for the dataset dropdown (match the sandbox labels).
const DATASET_LABELS = {
  swiss_roll: 'Swiss roll', s_curve: 'S-curve', twin_peaks: 'Twin peaks', saddle: 'Saddle',
  cylinder: 'Cylinder', severed_sphere: 'Severed sphere', helix: 'Helix',
  trefoil_knot: 'Trefoil knot', toroidal_helix: 'Toroidal helix', spiral_disk: 'Spiral disk',
  full_sphere: 'Full sphere', hilbert: 'Hilbert curve', clusters_3d: '3D clusters',
};

// Available walkthroughs, keyed by algorithm then dataset. Each entry is the clip
// path under assets/manim/ plus the opening caption. All clips share one step
// timeline, so the chapter markers (STEPS[i].start) apply to every one. The
// algorithm list also seeds the Algorithm picker; an empty algorithm shows a
// "coming soon" note.
const ALGOS = [
  { id: 'isomap', label: 'Isomap' },
  { id: 'pca', label: 'PCA' },
  { id: 'mds', label: 'MDS' },
  { id: 'lle', label: 'LLE' },
  { id: 'laplacian', label: 'Laplacian Eigenmaps' },
  { id: 'kpca_rbf', label: 'Kernel PCA (RBF)' },
  { id: 'kpca_polynomial', label: 'Kernel PCA (polynomial)' },
  { id: 'kpca_linear', label: 'Kernel PCA (linear)' },
];
const WALKS = {
  isomap: {
    swiss_roll: { video: 'isomap/walkthrough.mp4', intro: 'A 2D sheet rolled up in 3D; the goal is to recover the flat sheet.' },
    s_curve: { video: 'isomap/drafts/s_curve.mp4', intro: 'A 2D sheet bent into an S; the goal is to recover the flat sheet.' },
    twin_peaks: { video: 'isomap/drafts/twin_peaks.mp4', intro: 'A bumpy height surface in 3D; the goal is to recover its flat layout.' },
    saddle: { video: 'isomap/drafts/saddle.mp4', intro: 'A curved saddle surface; the goal is to recover its flat layout.' },
    cylinder: { video: 'isomap/drafts/cylinder.mp4', intro: 'A sheet wrapped into a cylinder; a closed band lays out as a loop, not a flat sheet.' },
    severed_sphere: { video: 'isomap/drafts/severed_sphere.mp4', intro: 'A sphere with its cap removed, an open curved surface; the goal is to flatten it.' },
    helix: { video: 'isomap/drafts/helix.mp4', intro: 'A ribbon wound into a helix; the goal is to unroll it to a flat strip.' },
    trefoil_knot: { video: 'isomap/drafts/trefoil_knot.mp4', intro: 'A ribbon tied into a trefoil knot; a closed band lays out as a loop, not a flat strip.' },
    toroidal_helix: { video: 'isomap/drafts/toroidal_helix.mp4', intro: 'A ribbon coiled around a torus; a closed band lays out as a loop, not a flat strip.' },
    spiral_disk: { video: 'isomap/drafts/spiral_disk.mp4', intro: 'A ribbon wound into a spiral; the goal is to unroll it to a flat strip.' },
    full_sphere: { video: 'isomap/drafts/full_sphere.mp4', intro: 'A full sphere, a closed surface; a sphere has no flat layout, so Isomap struggles.' },
    hilbert: { video: 'isomap/drafts/hilbert.mp4', intro: 'A Hilbert curve: a 1D path folded to fill a cube, not a surface.' },
    clusters_3d: { video: 'isomap/drafts/clusters_3d.mp4', intro: 'Separate clusters with no surface connecting them.' },
  },
  pca: {
    swiss_roll: { video: 'pca/walkthrough.mp4', intro: 'A swiss roll in 3D. PCA looks for the directions of greatest variance.' },
  },
  mds: {
    swiss_roll: { video: 'mds/walkthrough.mp4', intro: 'A 2D sheet rolled up in 3D. MDS looks for a 2D layout that preserves pairwise distances.' },
  },
  lle: {
    swiss_roll: { video: 'lle/walkthrough.mp4', intro: 'A 2D sheet rolled up in 3D. LLE recovers the flat sheet by preserving local linear structure.' },
  },
  laplacian: {
    swiss_roll: { video: 'laplacian/walkthrough.mp4', intro: 'A 2D sheet rolled up in 3D. Laplacian Eigenmaps recovers the flat sheet by preserving local connections.' },
  },
  kpca_rbf: {
    swiss_roll: { video: 'kpca/walkthrough-rbf.mp4', intro: 'A 2D sheet rolled up in 3D. Kernel PCA with an RBF kernel.' },
  },
  kpca_polynomial: {
    swiss_roll: { video: 'kpca/walkthrough-polynomial.mp4', intro: 'A 2D sheet rolled up in 3D. Kernel PCA with a polynomial kernel.' },
  },
  kpca_linear: {
    swiss_roll: { video: 'kpca/walkthrough-linear.mp4', intro: 'A 2D sheet rolled up in 3D. Kernel PCA with a linear kernel.' },
  },
};

const video = document.getElementById('mfiVideo');
const stepsEl = document.getElementById('mfiSteps');
const transcript = document.getElementById('mfiTranscript');
const scrub = document.getElementById('mfiScrub');
const playBtn = document.getElementById('mfiPlay');
const speedSel = document.getElementById('mfiSpeed');
let current = -1;

function stepIndexAt(t) {
  let idx = 0;
  for (let i = 0; i < STEPS.length; i++) {
    if (t >= STEPS[i].start - 0.05) idx = i;
  }
  return idx;
}

function renderSteps() {
  stepsEl.innerHTML = '';
  STEPS.forEach((s, i) => {
    const li = document.createElement('li');
    li.textContent = s.title;
    if (i === current) li.classList.add('is-active');
    li.addEventListener('click', () => seekToStep(i));
    stepsEl.appendChild(li);
  });
}

function renderTranscript() {
  const s = STEPS[current] || STEPS[0];
  transcript.innerHTML = '';

  const cap = document.createElement('div');
  cap.className = 'mfi-caption';
  cap.textContent = s.caption;
  transcript.appendChild(cap);

  if (s.formula) {
    const f = document.createElement('div');
    f.className = 'mfi-formula';
    f.textContent = `\\[${s.formula}\\]`;
    transcript.appendChild(f);
  }

  if (s.explain) {
    const e = document.createElement('p');
    e.className = 'mfi-explain';
    e.innerHTML = s.explain;
    transcript.appendChild(e);
  }

  // Retrying typeset so the subsidiary-text math renders even when MathJax (which
  // is loaded with defer) is not ready yet on the first transcript render.
  typesetMath(transcript);
}

// Set the active step (highlight + transcript) without moving the video.
function setActive(i) {
  i = Math.max(0, Math.min(STEPS.length - 1, i));
  if (i === current) return;
  current = i;
  renderSteps();
  renderTranscript();
}

// Move the video to a step boundary (clicking a step or Prev/Next).
function seekToStep(i) {
  i = Math.max(0, Math.min(STEPS.length - 1, i));
  if (video.duration) video.currentTime = Math.min(STEPS[i].start, video.duration - 0.05);
  else video.addEventListener('loadedmetadata', () => { video.currentTime = STEPS[i].start; }, { once: true });
  setActive(i);
}

// --- wiring ---
video.poster = posterSrc('isomap');
video.preload = 'auto';

const algoSel = document.getElementById('mfiAlgoSel');
const datasetSel = document.getElementById('mfiDatasetSel');
const datasetNote = document.createElement('div');
datasetNote.className = 'mfi-datasetnote';
stepsEl.insertAdjacentElement('afterend', datasetNote);
let currentBlobUrl = null;

function currentAlgo() { return (algoSel && algoSel.value) || 'isomap'; }
function currentDataset() { return (datasetSel && datasetSel.value) || ''; }
function algoLabel(id) { return (ALGOS.find((a) => a.id === id) || {}).label || id; }

// Populate the Algorithm picker once.
if (algoSel) {
  algoSel.innerHTML = '';
  for (const a of ALGOS) {
    const o = document.createElement('option');
    o.value = a.id; o.textContent = a.label;
    algoSel.appendChild(o);
  }
}

// Fill the Dataset picker with the datasets that have a clip for this algorithm.
function fillDatasets(algo) {
  if (!datasetSel) return;
  const ids = Object.keys(WALKS[algo] || {});
  datasetSel.innerHTML = '';
  for (const id of ids) {
    const o = document.createElement('option');
    o.value = id; o.textContent = DATASET_LABELS[id] || id;
    datasetSel.appendChild(o);
  }
  datasetSel.disabled = ids.length === 0;
}

// Load the clip for the chosen (algorithm, dataset) as a Blob so it is fully
// seekable on any static server. Updates the opening caption and resets to the
// first step. Shows a note when no clip exists for the combination yet.
function loadVideo() {
  const algo = currentAlgo();
  STEPS = STEPS_BY_ALGO[algo] || STEPS_ISOMAP;   // step list/chapters per algorithm
  video.poster = posterSrc(algo);
  const info = (WALKS[algo] || {})[currentDataset()];
  if (!info) {
    if (currentBlobUrl) { URL.revokeObjectURL(currentBlobUrl); currentBlobUrl = null; }
    video.removeAttribute('src');
    video.load();
    datasetNote.textContent = `The ${algoLabel(algo)} walkthrough is not available yet.`;
    STEPS[0].caption = 'Pick an algorithm and dataset with an available walkthrough.';
    current = -1;
    setActive(0);
    return;
  }
  datasetNote.textContent = '';
  STEPS[0].caption = info.intro;
  const url = ASSET_BASE + info.video;
  fetch(url)
    .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(String(r.status)))))
    .then((blob) => {
      if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
      currentBlobUrl = URL.createObjectURL(blob);
      video.src = currentBlobUrl;
    })
    .catch(() => { video.src = url; });
  current = -1;
  setActive(0);
}

video.addEventListener('timeupdate', () => {
  if (video.duration) scrub.value = String(Math.round((video.currentTime / video.duration) * 1000));
  setActive(stepIndexAt(video.currentTime));
});
scrub.addEventListener('input', () => {
  if (video.duration) video.currentTime = (scrub.value / 1000) * video.duration;
});

playBtn.addEventListener('click', () => {
  if (video.paused) video.play().catch(() => {});
  else video.pause();
});
video.addEventListener('play', () => { playBtn.textContent = 'Pause'; });
video.addEventListener('pause', () => { playBtn.textContent = 'Play'; });
video.addEventListener('ended', () => { playBtn.textContent = 'Play'; });

document.getElementById('mfiPrev').addEventListener('click', () => seekToStep(current - 1));
document.getElementById('mfiNext').addEventListener('click', () => seekToStep(current + 1));

if (speedSel) {
  speedSel.addEventListener('change', () => { video.playbackRate = parseFloat(speedSel.value) || 1; });
  video.playbackRate = parseFloat(speedSel.value) || 1;
}

if (algoSel) algoSel.addEventListener('change', () => { fillDatasets(currentAlgo()); loadVideo(); });
if (datasetSel) datasetSel.addEventListener('change', loadVideo);

fillDatasets(currentAlgo());
renderSteps();
loadVideo();
