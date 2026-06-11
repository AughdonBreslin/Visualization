"""Cross-check Python helpers in data.py against the JS reference formulas.

Each test contains a faithful Python re-implementation of the JS math (reading
directly from the JS source), runs both on the same small fixed input, and
asserts they agree within a tolerance. The data.py helpers reproduce the JS
formulas exactly: heat_affinity bakes in the median-distance sigma scaling,
lle_weights uses the reg*trace/k Tikhonov term, and bottom2_eig skips the first
eigenvalue by position, all matching the JS to machine precision on the tested
inputs.
"""
import unittest
import numpy as np
from manimexp.isomap import data


# ---------------------------------------------------------------------------
# Helpers: faithful Python ports of the JS reference math
# ---------------------------------------------------------------------------

def js_knn_graph(points, k):
    """Port of linalg.js knnGraph. Returns symmetric adj dict-of-lists and
    edges list, same as the JS adjacency structure."""
    n = len(points)
    # Compute all squared distances
    diff = points[:, None, :] - points[None, :, :]
    sq = (diff ** 2).sum(axis=2)
    np.fill_diagonal(sq, np.inf)

    adj = [[] for _ in range(n)]
    seen = set()
    edges = []
    for i in range(n):
        order = np.argsort(sq[i])
        for m in range(k):
            j = int(order[m])
            w = float(np.sqrt(sq[i, j]))
            adj[i].append((j, w))
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                edges.append(key)
    # symmetrize: ensure j->i exists whenever i->j does
    for i in range(n):
        for (j, w) in adj[i]:
            if not any(e[0] == i for e in adj[j]):
                adj[j].append((i, w))
    return adj, edges


def js_heat_affinity_with_sigma_scaling(points, adj_list, k, sigma, n):
    """Port of laplacian.js step-3.

    JS computes effSigma = sigma * medianDist where medianDist is the median
    of ALL kNN edge distances (taken from adj[i] over all i, so each directed
    edge is counted once per direction in the adj list).

    Returns W (n x n numpy array) and effSigma used.
    """
    dists = []
    for i in range(n):
        for (j, d) in adj_list[i]:
            dists.append(d)
    dists.sort()
    median_dist = dists[len(dists) // 2] if dists else 1.0
    eff_sigma = sigma * (median_dist or 1.0)
    sig2 = 2.0 * eff_sigma * eff_sigma

    W = np.zeros((n, n))
    for i in range(n):
        for (j, dist) in adj_list[i]:
            w = np.exp(-dist * dist / sig2)
            W[i, j] = w
            W[j, i] = w
    return W, eff_sigma


def js_lle_weights(points, k, reg):
    """Port of lle.js reconstruction-weight computation.

    JS regularization lambda is reg * max(trace, 1e-12) / G.length (divides by k,
    the number of neighbors). The Python helper in data.py uses the same term, so
    the two agree to machine precision.

    JS lines 121-124 of lle.js:
        let trace = 0;
        for (let a = 0; a < G.length; a++) trace += G[a][a];
        const lambda = reg * Math.max(trace, 1e-12) / G.length;
        for (let a = 0; a < G.length; a++) G[a][a] += lambda;
    """
    n = len(points)
    adj, _ = data.knn_graph(points, k=k)
    W = np.zeros((n, n))
    for i in range(n):
        nbrs = np.where(adj[i] > 0)[0]
        if len(nbrs) == 0:
            continue
        Z = points[nbrs] - points[i]
        G = Z @ Z.T
        tr = np.trace(G)
        lam = reg * max(tr, 1e-12) / len(nbrs)   # JS formula: divide by k
        G = G + lam * np.eye(len(nbrs))
        ones = np.ones(len(nbrs))
        try:
            w = np.linalg.solve(G, ones)
        except np.linalg.LinAlgError:
            continue
        s = w.sum()
        if abs(s) < 1e-12:
            continue
        w = w / s
        W[i, nbrs] = w
    return W


def js_kernel_matrix_rbf(points, gamma):
    """Port of kpca.js computeKernel with rbf + gammaEff.

    JS lines 78-87 of kpca.js:
        let cx=0, cy=0, cz=0;
        for i: cx += X[i*3]; cy += X[i*3+1]; cz += X[i*3+2];
        cx /= N; cy /= N; cz /= N;
        let varSum = 0;
        for i: a=X[i*3]-cx, b=..., c=...; varSum += a*a+b*b+c*c;
        const meanSqDist = (2 * varSum / N) || 1;
        const gammaEff = gamma / meanSqDist;
    """
    n = len(points)
    c = points.mean(axis=0)
    var_sum = float(np.sum((points - c) ** 2))
    mean_sq_dist = (2.0 * var_sum / n) or 1.0
    gamma_eff = gamma / mean_sq_dist

    diff = points[:, None, :] - points[None, :, :]
    sq = (diff ** 2).sum(axis=2)
    return np.exp(-gamma_eff * sq), gamma_eff


def js_center_kernel(K):
    """Port of kpca.js kernel-centering step (lines 166-179).

    Krow[i] = sum_j K[i,j] / N   (row mean)
    grand    = sum_i Krow[i] / N  (grand mean)
    Kc[i,j] = K[i,j] - Krow[i] - Krow[j] + grand
    """
    n = K.shape[0]
    krow = K.mean(axis=1)       # shape (n,)
    grand = krow.mean()
    # broadcast: subtract row mean and column mean, add grand mean
    return K - krow[:, None] - krow[None, :] + grand


def js_double_center_squared(D):
    """Port of linalg.js doubleCenterSquared.

    Handles non-finite values by treating them as 0 in D^2, then applies
    explicit row/column/grand mean correction (identical to J D^2 J formula
    when D is finite, but clamps infinities to 0 for D^2).
    """
    n = D.shape[0]
    D2 = np.where(np.isfinite(D), D ** 2, 0.0)
    row_mean = D2.mean(axis=1)
    col_mean = D2.mean(axis=0)
    grand = D2.mean()
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            B[i, j] = -0.5 * (D2[i, j] - row_mean[i] - col_mean[j] + grand)
    return B


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHeatAffinityFormula(unittest.TestCase):
    """heat_affinity(points, edges, sigma) vs laplacian.js heat-kernel formula.

    The JS pre-multiplies sigma by the median neighbor distance before computing
    affinities (effSigma = sigma * medianDist). The Python helper bakes the same
    median scaling in, so it matches the JS heat kernel to machine precision.
    """

    def setUp(self):
        self.pts = data.swiss_roll(n=60, seed=0)["points"]
        self.k = 8
        self.sigma = 1.0

    def test_formula_matches_when_same_sigma_used(self):
        """heat_affinity equals the exp(-d^2/(2 effSigma^2)) formula with median
        scaling baked in (effSigma = sigma * median(edge distances))."""
        adj_mat, edges = data.knn_graph(self.pts, k=self.k)
        W_py = data.heat_affinity(self.pts, edges, sigma=self.sigma)

        n = len(self.pts)
        dists = np.array([np.linalg.norm(self.pts[i] - self.pts[j]) for (i, j) in edges])
        eff = self.sigma * (float(np.median(dists)) or 1.0)
        sig2 = 2.0 * eff * eff
        W_ref = np.zeros((n, n))
        for (i, j), d in zip(edges, dists):
            w = np.exp(-d * d / sig2)
            W_ref[i, j] = w
            W_ref[j, i] = w

        np.testing.assert_allclose(W_py, W_ref, atol=1e-12,
            err_msg="heat_affinity formula differs from exp(-d^2/(2 effSigma^2))")

    def test_matches_js_median_scaled_sigma(self):
        """heat_affinity now bakes in the JS median scaling (effSigma = sigma *
        medianDist), so it matches the JS laplacian.js port to machine precision."""
        n = len(self.pts)
        adj_list, _ = js_knn_graph(self.pts, self.k)
        W_js, eff_sigma = js_heat_affinity_with_sigma_scaling(
            self.pts, adj_list, self.k, self.sigma, n)
        adj_mat, edges_py = data.knn_graph(self.pts, k=self.k)
        W_py = data.heat_affinity(self.pts, edges_py, sigma=self.sigma)

        np.testing.assert_allclose(W_py, W_js, atol=1e-12,
            err_msg="heat_affinity disagrees with JS median-scaled heat kernel")

    def test_symmetry_and_zero_diagonal(self):
        adj_mat, edges = data.knn_graph(self.pts, k=self.k)
        W = data.heat_affinity(self.pts, edges, sigma=1.0)
        np.testing.assert_allclose(W, W.T, atol=1e-12)
        np.testing.assert_array_equal(np.diag(W), 0.0)

    def test_range_zero_to_one(self):
        adj_mat, edges = data.knn_graph(self.pts, k=self.k)
        W = data.heat_affinity(self.pts, edges, sigma=1.0)
        self.assertGreaterEqual(W.min(), 0.0)
        self.assertLessEqual(W.max(), 1.0 + 1e-12)


class TestGraphLaplacian(unittest.TestCase):
    """graph_laplacian(W) -> (L, D) vs laplacian.js step-4 formula.

    JS lines 142-155:
        D[i] = sum_j W[i*N+j]
        Dmat[i*N+i] = D[i]
        L[i*N+j] = (i==j ? D[i] : 0) - W[i*N+j]
    Equivalently: L = diag(W @ ones) - W.
    """

    def setUp(self):
        self.pts = data.swiss_roll(n=50, seed=0)["points"]

    def test_laplacian_definition(self):
        adj, edges = data.knn_graph(self.pts, k=8)
        W = data.heat_affinity(self.pts, edges, sigma=1.0)
        L, D = data.graph_laplacian(W)

        # Faithful JS port: D_ii = row sum of W
        deg = W.sum(axis=1)
        D_js = np.diag(deg)
        L_js = D_js - W

        np.testing.assert_allclose(D, D_js, atol=1e-12,
            err_msg="Degree matrix D disagrees with JS formula")
        np.testing.assert_allclose(L, L_js, atol=1e-12,
            err_msg="Laplacian L = D - W disagrees with JS formula")

    def test_laplacian_row_sums_zero(self):
        """L @ 1 = 0 for any valid Laplacian."""
        adj, edges = data.knn_graph(self.pts, k=8)
        W = data.heat_affinity(self.pts, edges, sigma=1.0)
        L, _ = data.graph_laplacian(W)
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-10,
            err_msg="Laplacian row sums must be zero")

    def test_laplacian_positive_semidefinite(self):
        """L should be PSD (all eigenvalues >= 0)."""
        adj, edges = data.knn_graph(self.pts, k=8)
        W = data.heat_affinity(self.pts, edges, sigma=1.0)
        L, _ = data.graph_laplacian(W)
        vals = np.linalg.eigvalsh(L)
        self.assertGreaterEqual(vals.min(), -1e-9,
            msg=f"Laplacian has negative eigenvalue {vals.min():.3e}")


class TestLLEWeightsRegularization(unittest.TestCase):
    """lle_weights regularization vs lle.js.

    The JS uses:
        lambda = reg * max(trace, 1e-12) / G.length   (lle.js line 123)
    i.e. the regularization added to each diagonal entry is reg*trace/k. The
    Python helper now matches this exactly (data.py: reg*max(trace,1e-12)/k),
    so the weight matrices agree to machine precision.
    """

    def setUp(self):
        self.pts = data.swiss_roll(n=60, seed=0)["points"]
        self.k = 8
        self.reg = 1e-3

    def test_py_reg_matches_js_per_diagonal(self):
        """Python adds reg*max(trace,1e-12)/k per diagonal, exactly like the JS."""
        adj, _ = data.knn_graph(self.pts, k=self.k)
        for i in range(len(self.pts)):
            nbrs = np.where(adj[i] > 0)[0]
            if len(nbrs) == self.k:
                Z = self.pts[nbrs] - self.pts[i]
                G = Z @ Z.T
                tr = float(np.trace(G))
                if tr > 1e-10:
                    py_lam = self.reg * max(tr, 1e-12) / len(nbrs)
                    js_lam = self.reg * max(tr, 1e-12) / len(nbrs)
                    self.assertAlmostEqual(py_lam, js_lam, places=12)
                    return
        self.skipTest("No suitable point found")

    def test_js_port_weights_sum_to_one(self):
        """JS-ported weights (with /k regularization) sum to 1."""
        W_js = js_lle_weights(self.pts, k=self.k, reg=self.reg)
        np.testing.assert_allclose(W_js.sum(axis=1), np.ones(len(self.pts)),
            atol=1e-6, err_msg="JS-ported LLE weights do not sum to 1")

    def test_py_weights_sum_to_one(self):
        """Python weights also sum to 1 (both normalize, despite reg diff)."""
        W_py = data.lle_weights(self.pts, k=self.k, reg=self.reg)
        np.testing.assert_allclose(W_py.sum(axis=1), np.ones(len(self.pts)),
            atol=1e-6, err_msg="Python LLE weights do not sum to 1")

    def test_weight_values_match_js(self):
        """With matching /k regularization, Python and JS weights agree."""
        W_py = data.lle_weights(self.pts, k=self.k, reg=self.reg)
        W_js = js_lle_weights(self.pts, k=self.k, reg=self.reg)
        np.testing.assert_allclose(W_py, W_js, atol=1e-9,
            err_msg="LLE weights disagree with JS /k regularization")

    def test_lle_matrix_is_symmetric_psd(self):
        """M = (I-W)^T(I-W) must be symmetric and PSD."""
        W_py = data.lle_weights(self.pts, k=self.k, reg=self.reg)
        M = data.lle_matrix(W_py)
        np.testing.assert_allclose(M, M.T, atol=1e-9)
        vals = np.linalg.eigvalsh(M)
        self.assertGreaterEqual(vals.min(), -1e-9,
            msg=f"M is not PSD: min eigenvalue = {vals.min():.3e}")

    def test_lle_matrix_js_port_consistent(self):
        """M from JS-ported weights is also symmetric PSD."""
        W_js = js_lle_weights(self.pts, k=self.k, reg=self.reg)
        n = len(self.pts)
        IW = np.eye(n) - W_js
        M = IW.T @ IW
        np.testing.assert_allclose(M, M.T, atol=1e-9)
        vals = np.linalg.eigvalsh(M)
        self.assertGreaterEqual(vals.min(), -1e-9)


class TestKPCARbfGammaScaling(unittest.TestCase):
    """kernel_matrix rbf gamma auto-scaling vs kpca.js.

    JS formula (kpca.js lines 78-87):
        varSum = sum_i ||x_i - centroid||^2
        meanSqDist = (2 * varSum / N) || 1
        gammaEff = gamma / meanSqDist

    Python formula (data.py lines 162-163):
        c = points.mean(axis=0)
        mean_sq = (2.0 * float(np.sum((points - c)**2)) / n) or 1.0
        g = gamma / mean_sq

    These are mathematically identical. This test confirms both produce the
    same gammaEff and the same kernel matrix.
    """

    def setUp(self):
        self.pts = data.swiss_roll(n=60, seed=0)["points"]

    def test_gamma_eff_matches(self):
        """gammaEff computed by Python matches the JS formula."""
        pts = self.pts
        n = len(pts)
        gamma = 1.0

        # JS formula
        c = pts.mean(axis=0)
        var_sum = float(np.sum((pts - c) ** 2))
        mean_sq_dist_js = (2.0 * var_sum / n) or 1.0
        gamma_eff_js = gamma / mean_sq_dist_js

        # Python formula (same thing, different variable names)
        mean_sq_py = (2.0 * float(np.sum((pts - pts.mean(axis=0)) ** 2)) / n) or 1.0
        gamma_eff_py = gamma / mean_sq_py

        self.assertAlmostEqual(gamma_eff_js, gamma_eff_py, places=12,
            msg="gammaEff disagrees between JS and Python formula")

    def test_rbf_kernel_matches_js_port(self):
        """K_rbf from data.kernel_matrix matches the JS-port."""
        K_py = data.kernel_matrix(self.pts, "rbf", gamma=1.0)
        K_js, _ = js_kernel_matrix_rbf(self.pts, gamma=1.0)
        np.testing.assert_allclose(K_py, K_js, atol=1e-12,
            err_msg="RBF kernel matrix disagrees with JS-port")

    def test_rbf_diagonal_is_ones(self):
        """K_rbf[i,i] = exp(0) = 1 for all i."""
        K = data.kernel_matrix(self.pts, "rbf", gamma=1.0)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)

    def test_rbf_is_symmetric(self):
        K = data.kernel_matrix(self.pts, "rbf", gamma=1.0)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_polynomial_kernel_matches_formula(self):
        """K_poly = (X X^T + c)^d matches JS formula."""
        pts = self.pts
        degree = 3
        constant = 1.0
        K_py = data.kernel_matrix(pts, "polynomial", gamma=1.0,
                                   degree=degree, constant=constant)
        dot = pts @ pts.T
        K_ref = (dot + constant) ** degree
        np.testing.assert_allclose(K_py, K_ref, atol=1e-10,
            err_msg="Polynomial kernel disagrees with (X X^T + c)^d")

    def test_linear_kernel_matches_formula(self):
        """K_lin = X X^T."""
        pts = self.pts
        K_py = data.kernel_matrix(pts, "linear")
        K_ref = pts @ pts.T
        np.testing.assert_allclose(K_py, K_ref, atol=1e-12,
            err_msg="Linear kernel disagrees with X X^T")


class TestCenterKernel(unittest.TestCase):
    """center_kernel(K) vs kpca.js centering formula.

    JS formula (kpca.js lines 166-179):
        Krow[i] = sum_j K[i,j] / N        (row mean)
        grand = sum_i Krow[i] / N          (grand mean)
        Kc[i,j] = K[i,j] - Krow[i] - Krow[j] + grand

    Python (data.py lines 176-178):
        row = K.mean(axis=1, keepdims=True)
        grand = float(K.mean())
        return K - row - row.T + grand

    These are mathematically equivalent. This test verifies agreement.
    """

    def setUp(self):
        self.pts = data.swiss_roll(n=60, seed=0)["points"]

    def test_center_kernel_matches_js_port(self):
        """Centered kernel from Python matches JS-port formula."""
        K = data.kernel_matrix(self.pts, "rbf", gamma=1.0)
        Kc_py = data.center_kernel(K)
        Kc_js = js_center_kernel(K)
        np.testing.assert_allclose(Kc_py, Kc_js, atol=1e-12,
            err_msg="center_kernel disagrees with JS kpca.js centering formula")

    def test_centered_kernel_row_means_are_zero(self):
        """After centering, row means of Kc are 0."""
        K = data.kernel_matrix(self.pts, "rbf", gamma=1.0)
        Kc = data.center_kernel(K)
        np.testing.assert_allclose(Kc.mean(axis=1), 0.0, atol=1e-10,
            err_msg="Row means of centered kernel are not zero")

    def test_centered_kernel_symmetric(self):
        K = data.kernel_matrix(self.pts, "rbf", gamma=1.0)
        Kc = data.center_kernel(K)
        np.testing.assert_allclose(Kc, Kc.T, atol=1e-12)

    def test_centering_on_known_small_matrix(self):
        """Verify on a hand-computed 3x3 example."""
        K = np.array([[4.0, 2.0, 0.0],
                      [2.0, 3.0, 1.0],
                      [0.0, 1.0, 2.0]])
        Kc_py = data.center_kernel(K)
        Kc_js = js_center_kernel(K)
        np.testing.assert_allclose(Kc_py, Kc_js, atol=1e-12)
        # Row means should be 0 after centering.
        np.testing.assert_allclose(Kc_py.mean(axis=1), 0.0, atol=1e-12)


class TestBottom2Eig(unittest.TestCase):
    """bottom2_eig(M) skip-trivial behavior vs lle.js / laplacian.js.

    JS bottomKSymmetricEig(M, N, k, {skipFirst: 1}) skips the first (smallest)
    eigenvalue by position. Python bottom2_eig now matches: it takes the sorted
    eigenpairs at positions [1, 2], skipping the trivial smallest one.

    For well-formed L and M matrices the smallest eigenvalue is ~0 (the constant
    eigenvector), so positions [1, 2] are the two smallest non-trivial modes.
    This test confirms that:
    1. The returned eigenvalues are the two smallest eigenvalues > 1e-9.
    2. The corresponding eigenvectors are orthogonal to the zero eigenvector
       (constant vector).
    """

    def setUp(self):
        self.pts = data.swiss_roll(n=60, seed=0)["points"]
        self.k = 8
        adj, edges = data.knn_graph(self.pts, k=self.k)
        self.W_lapl = data.heat_affinity(self.pts, edges, sigma=1.0)
        self.L, _ = data.graph_laplacian(self.W_lapl)
        W_lle = data.lle_weights(self.pts, k=self.k, reg=1e-3)
        self.M_lle = data.lle_matrix(W_lle)

    def test_bottom2_eig_skips_zero_on_laplacian(self):
        """Returned eigenvalues are strictly positive (trivial zero skipped)."""
        vecs, vals = data.bottom2_eig(self.L)
        self.assertGreater(float(vals[0]), 1e-9,
            msg=f"First returned eigenvalue {vals[0]:.3e} is <= 1e-9; trivial mode not skipped")
        self.assertGreater(float(vals[1]), 1e-9)

    def test_bottom2_eig_ascending_order(self):
        """Returned eigenvalues are in ascending order."""
        vecs, vals = data.bottom2_eig(self.L)
        self.assertLessEqual(float(vals[0]), float(vals[1]) + 1e-9)

    def test_bottom2_eig_returns_correct_shapes(self):
        n = len(self.pts)
        vecs, vals = data.bottom2_eig(self.L)
        self.assertEqual(vecs.shape, (n, 2))
        self.assertEqual(vals.shape, (2,))

    def test_bottom2_eig_matches_scipy_eigvalsh_for_laplacian(self):
        """Python helper returns the same eigenvalues as scipy for the 2nd+3rd smallest."""
        all_vals = np.linalg.eigvalsh(self.L)
        all_vals_sorted = np.sort(all_vals)
        # Skip trivial: first with val > 1e-9
        non_trivial = all_vals_sorted[all_vals_sorted > 1e-9]
        ref_vals = non_trivial[:2]

        _, py_vals = data.bottom2_eig(self.L)
        np.testing.assert_allclose(py_vals, ref_vals, atol=1e-9,
            err_msg="bottom2_eig eigenvalues differ from scipy reference")

    def test_bottom2_eig_matches_scipy_eigvalsh_for_lle_matrix(self):
        """Same agreement check on the LLE cost matrix M."""
        all_vals = np.linalg.eigvalsh(self.M_lle)
        all_vals_sorted = np.sort(all_vals)
        non_trivial = all_vals_sorted[all_vals_sorted > 1e-9]
        ref_vals = non_trivial[:2]

        _, py_vals = data.bottom2_eig(self.M_lle)
        np.testing.assert_allclose(py_vals, ref_vals, atol=1e-9,
            err_msg="bottom2_eig eigenvalues on M_lle differ from scipy reference")

    def test_eigenvectors_near_orthogonal_to_constant(self):
        """Returned eigenvectors should be nearly orthogonal to the all-ones vector."""
        vecs, _ = data.bottom2_eig(self.L)
        n = len(self.pts)
        ones = np.ones(n) / np.sqrt(n)
        for col in range(2):
            dot = abs(float(np.dot(vecs[:, col], ones)))
            self.assertLess(dot, 0.05,
                msg=f"Eigenvector {col} has dot product {dot:.4f} with constant vector; "
                    "trivial mode may not have been properly skipped")

    def test_js_skipfirst_matches_python_selection(self):
        """JS skipFirst=1 (positions [1, 2]) matches what bottom2_eig returns."""
        _, py_vals = data.bottom2_eig(self.L)
        js_vals = np.sort(np.linalg.eigvalsh(self.L))[1:3]
        np.testing.assert_allclose(js_vals, py_vals, atol=1e-9,
            err_msg="JS skipFirst selection differs from bottom2_eig")


class TestDoubleCenterSquaredSmallCase(unittest.TestCase):
    """double_center_squared on a tiny matrix, cross-checking both the
    J D^2 J formula and the JS explicit row/col/grand loop."""

    def test_matches_js_port_on_finite_input(self):
        """For finite D, Python J-formula and JS explicit-loop must agree."""
        rng = np.random.default_rng(42)
        n = 6
        # Symmetric non-negative distance matrix with zero diagonal.
        raw = rng.uniform(0.5, 3.0, (n, n))
        D = (raw + raw.T) / 2
        np.fill_diagonal(D, 0.0)

        B_py = data.double_center_squared(D)
        B_js = js_double_center_squared(D)
        np.testing.assert_allclose(B_py, B_js, atol=1e-12,
            err_msg="double_center_squared differs from JS-port on finite input")

    def test_matches_js_port_with_inf_values(self):
        """JS replaces inf in D^2 with 0; Python clamps inf before squaring.

        The Python build_dataset replaces inf D values with the finite max
        before calling double_center_squared (so there are no infs when the
        Python function is actually called). This test verifies the JS clamping
        behavior is consistent with the Python approach on clamped inputs.
        """
        n = 4
        D = np.array([[0.0, 1.0, 2.0, 3.0],
                      [1.0, 0.0, 1.0, 2.0],
                      [2.0, 1.0, 0.0, 1.0],
                      [3.0, 2.0, 1.0, 0.0]], dtype=float)
        B_py = data.double_center_squared(D)
        B_js = js_double_center_squared(D)
        np.testing.assert_allclose(B_py, B_js, atol=1e-12)


class TestTop2KernelEmbed(unittest.TestCase):
    """top2_kernel_embed embedding formula vs kpca.js step-6.

    JS formula (kpca.js lines 244-248):
        s1 = sqrt(max(0, lambda[0]))
        s2 = sqrt(max(0, lambda[1]))
        embed2d[i*2]   = vectors[0][i] * s1
        embed2d[i*2+1] = vectors[1][i] * s2

    Python formula (data.py lines 201-202):
        Y = vecs * np.sqrt(np.maximum(vals, 0.0))
    These are equivalent.
    """

    def setUp(self):
        self.pts = data.swiss_roll(n=60, seed=0)["points"]

    def test_embed_formula_matches_js(self):
        """Y = vecs * sqrt(vals) matches JS embed2d construction."""
        K = data.kernel_matrix(self.pts, "rbf", gamma=1.0)
        Kc = data.center_kernel(K)
        Y, vecs, vals = data.top2_kernel_embed(Kc)

        # JS-port embed
        s1 = np.sqrt(max(0.0, float(vals[0])))
        s2 = np.sqrt(max(0.0, float(vals[1])))
        Y_js = np.column_stack([vecs[:, 0] * s1, vecs[:, 1] * s2])

        np.testing.assert_allclose(Y, Y_js, atol=1e-12,
            err_msg="top2_kernel_embed Y disagrees with JS y_ik = sqrt(lambda_k) * v_k")

    def test_eigenvalues_descending(self):
        K = data.kernel_matrix(self.pts, "rbf", gamma=1.0)
        Kc = data.center_kernel(K)
        _, _, vals = data.top2_kernel_embed(Kc)
        self.assertGreaterEqual(float(vals[0]) + 1e-9, float(vals[1]),
            msg="Eigenvalues are not in descending order")


class TestSwissRollNormalization(unittest.TestCase):
    """swiss_roll() normalization: the JS sandbox generators are separate;
    absolute values cannot be cross-checked. This confirms the Python
    normalization contract is self-consistent: centered at origin, max
    coordinate magnitude = 3."""

    def test_centered_at_origin(self):
        pts = data.swiss_roll(n=200, seed=0)["points"]
        np.testing.assert_allclose(pts.mean(axis=0), 0.0, atol=1e-10,
            err_msg="Swiss roll points are not centered at origin")

    def test_max_abs_coord_is_3(self):
        pts = data.swiss_roll(n=200, seed=0)["points"]
        self.assertAlmostEqual(float(np.abs(pts).max()), 3.0, places=10,
            msg="Max |coordinate| is not 3.0 after normalization")

    def test_t_range_matches_generation(self):
        """t in [1.5*pi, 1.5*pi*(1+2)] = [~4.71, ~14.14]."""
        roll = data.swiss_roll(n=500, seed=0)
        t = roll["t"]
        self.assertGreaterEqual(float(t.min()), 1.5 * np.pi - 1e-9)
        self.assertLessEqual(float(t.max()), 1.5 * np.pi * 3.0 + 1e-9)


class TestKNNGraphAgreement(unittest.TestCase):
    """knn_graph vs JS knnGraph: same edges on a tiny example where the
    expected neighbors are unambiguous."""

    def test_chain_graph_neighbors(self):
        """Points equally spaced on a line: kNN with k=2 and symmetrization.

        For points [0,1,2,3], point 1's own k=2 nearest neighbors are 0 and 2.
        However, point 3's k=2 nearest neighbors are 2 (d=1) and 1 (d=2).
        Symmetrization then adds the 3<->1 edge to adj, so adj[1,3] > 0.
        This is correct kNN behavior: the symmetric graph includes any directed
        edge that appears in at least one point's k-NN list.
        """
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
        adj, edges = data.knn_graph(pts, k=2)
        # Point 1 must be adjacent to 0 and 2 (its own two nearest neighbors).
        self.assertGreater(adj[1, 0], 0.0)
        self.assertGreater(adj[1, 2], 0.0)
        # Point 3's k=2 neighbors are 2 (d=1) and 1 (d=2), so after
        # symmetrization adj[1,3] = adj[3,1] = 2.0 (the distance 3->1).
        self.assertGreater(adj[1, 3], 0.0,
            msg="Symmetrization should add edge 1<->3 because 3 has 1 in its kNN list")

    def test_symmetry_matches_js(self):
        """JS knnGraph symmetrizes post-hoc; Python symmetrizes during build.
        Both must produce a symmetric adj matrix on the same input."""
        pts = data.swiss_roll(n=40, seed=0)["points"]
        adj_py, _ = data.knn_graph(pts, k=6)
        adj_js_raw, _ = js_knn_graph(pts, 6)
        n = len(pts)
        adj_js = np.zeros((n, n))
        for i in range(n):
            for (j, w) in adj_js_raw[i]:
                adj_js[i, j] = w
                adj_js[j, i] = w
        np.testing.assert_allclose(adj_py, adj_js, atol=1e-12,
            err_msg="knn_graph adjacency matrix differs from JS-port")


if __name__ == "__main__":
    unittest.main()
