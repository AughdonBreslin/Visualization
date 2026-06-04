import unittest
import numpy as np
from manimexp.isomap import data


class TestData(unittest.TestCase):
    def test_swiss_roll_deterministic_and_shaped(self):
        a = data.swiss_roll(n=200, seed=7)
        b = data.swiss_roll(n=200, seed=7)
        np.testing.assert_allclose(a["points"], b["points"])
        self.assertEqual(a["points"].shape, (200, 3))
        self.assertEqual(a["t"].shape, (200,))

    def test_knn_is_symmetric_with_k_neighbors(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], float)
        adj, edges = data.knn_graph(pts, k=2)
        self.assertEqual(adj.shape, (5, 5))
        np.testing.assert_allclose(adj, adj.T)
        self.assertTrue(np.all(np.diag(adj) == 0))
        for (i, j) in edges:
            self.assertLess(i, j)

    def test_double_center_matches_formula(self):
        D = np.array([[0.0, 1.0, 2.0],
                      [1.0, 0.0, 1.0],
                      [2.0, 1.0, 0.0]])
        B = data.double_center_squared(D)
        n = 3
        J = np.eye(n) - np.ones((n, n)) / n
        expected = -0.5 * J @ (D ** 2) @ J
        np.testing.assert_allclose(B, expected, atol=1e-12)
        np.testing.assert_allclose(B, B.T, atol=1e-12)

    def test_geodesic_at_least_euclidean(self):
        d = data.build_dataset(n=120, k=8, seed=1)
        i, j = d["src"], d["tgt"]
        eucl = np.linalg.norm(d["points"][i] - d["points"][j])
        self.assertGreaterEqual(d["D"][i, j] + 1e-9, eucl)
        self.assertGreater(len(d["path"]), 1)

    def test_embedding_shape_and_excerpts(self):
        d = data.build_dataset(n=120, k=8, seed=1)
        self.assertEqual(d["embedding"].shape, (120, 2))
        self.assertEqual(np.array(d["excerpt_D"]).shape, (4, 4))
        self.assertEqual(np.array(d["excerpt_B"]).shape, (4, 4))
        self.assertEqual(len(d["eigvals"]), 2)


if __name__ == "__main__":
    unittest.main()
