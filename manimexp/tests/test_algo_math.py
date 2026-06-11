import unittest
import numpy as np
from manimexp.isomap import data


class TestAlgoMath(unittest.TestCase):
    def setUp(self):
        self.pts = data.swiss_roll(n=60, seed=0)["points"]

    def test_heat_affinity_and_laplacian(self):
        adj, edges = data.knn_graph(self.pts, k=8)
        W = data.heat_affinity(self.pts, edges, sigma=1.0)
        np.testing.assert_allclose(W, W.T)
        self.assertTrue(np.all(np.diag(W) == 0))
        L, Ddeg = data.graph_laplacian(W)
        np.testing.assert_allclose(np.diag(Ddeg), W.sum(axis=1))
        np.testing.assert_allclose(L, Ddeg - W)

    def test_lle_weights_reconstruct(self):
        Wlle = data.lle_weights(self.pts, k=8, reg=1e-3)
        # rows sum to 1, weights only on neighbors, reconstruction is close.
        np.testing.assert_allclose(Wlle.sum(axis=1), np.ones(len(self.pts)), atol=1e-6)
        recon = Wlle @ self.pts
        self.assertLess(np.linalg.norm(recon - self.pts) / len(self.pts), 0.5)
        M = data.lle_matrix(Wlle)
        np.testing.assert_allclose(M, M.T, atol=1e-9)

    def test_kernel_matrix_variants(self):
        Krbf = data.kernel_matrix(self.pts, "rbf", gamma=1.0)
        np.testing.assert_allclose(np.diag(Krbf), np.ones(len(self.pts)))
        Klin = data.kernel_matrix(self.pts, "linear")
        np.testing.assert_allclose(Klin, self.pts @ self.pts.T)
        Kc = data.center_kernel(Krbf)
        np.testing.assert_allclose(Kc, Kc.T, atol=1e-9)

    def test_bottom_eigvecs_skips_trivial(self):
        adj, edges = data.knn_graph(self.pts, k=8)
        W = data.heat_affinity(self.pts, edges, sigma=1.0)
        L, _ = data.graph_laplacian(W)
        vecs, vals = data.bottom2_eig(L)
        self.assertEqual(vecs.shape, (len(self.pts), 2))
        # eigenvalues are the two smallest non-trivial (>~0), ascending.
        self.assertLessEqual(vals[0], vals[1] + 1e-9)
        self.assertGreater(vals[0], 1e-9)

    def test_top2_kernel_embed_shapes(self):
        Krbf = data.kernel_matrix(self.pts, "rbf", gamma=1.0)
        Kc = data.center_kernel(Krbf)
        Y, vecs, vals = data.top2_kernel_embed(Kc)
        self.assertEqual(Y.shape, (len(self.pts), 2))
        self.assertEqual(vecs.shape, (len(self.pts), 2))
        # descending non-negative eigenvalues, embedding = sqrt(lambda) * v
        self.assertGreaterEqual(vals[0] + 1e-9, vals[1])
        np.testing.assert_allclose(Y, vecs * np.sqrt(np.maximum(vals, 0.0)), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
