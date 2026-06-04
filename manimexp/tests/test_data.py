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

    # --- neighbor_edges ---

    def test_neighbor_edges_count_and_sorted(self):
        # Hand-built 4-node adj: only node 0 has edges to 1, 2, 3 with distinct weights.
        adj = np.zeros((4, 4))
        adj[0, 1] = adj[1, 0] = 3.0
        adj[0, 2] = adj[2, 0] = 1.0
        adj[0, 3] = adj[3, 0] = 2.0
        edges = data.neighbor_edges(None, adj, center=0)
        # Should have exactly 3 neighbors.
        self.assertEqual(len(edges), 3)
        # Weights should be sorted ascending.
        weights = [w for _, w in edges]
        self.assertEqual(weights, sorted(weights))
        # Verify indices and values.
        self.assertEqual(edges[0], (2, 1.0))
        self.assertEqual(edges[1], (3, 2.0))
        self.assertEqual(edges[2], (1, 3.0))

    def test_neighbor_edges_no_self_loop(self):
        adj = np.zeros((3, 3))
        adj[0, 1] = adj[1, 0] = 5.0
        adj[0, 0] = 99.0  # diagonal entry should be ignored
        edges = data.neighbor_edges(None, adj, center=0)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0], (1, 5.0))

    # --- dijkstra_order ---

    def test_dijkstra_order_starts_at_src(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
        adj, _ = data.knn_graph(pts, k=2)
        src = 0
        order, dist = data.dijkstra_order(adj, src)
        self.assertEqual(order[0], src)

    def test_dijkstra_order_length_equals_n(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
        adj, _ = data.knn_graph(pts, k=2)
        order, dist = data.dijkstra_order(adj, src=0)
        self.assertEqual(len(order), 4)

    def test_dijkstra_order_distances_nondecreasing(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], float)
        adj, _ = data.knn_graph(pts, k=2)
        order, dist = data.dijkstra_order(adj, src=0)
        settled_dists = [dist[i] for i in order]
        for a, b in zip(settled_dists, settled_dists[1:]):
            self.assertLessEqual(a, b + 1e-12)

    def test_dijkstra_dist_matches_scipy(self):
        roll = data.swiss_roll(n=50, seed=3)
        adj, _ = data.knn_graph(roll["points"], k=4)
        order, dist = data.dijkstra_order(adj, src=0)
        D, _ = data.geodesic_distances(adj)
        np.testing.assert_allclose(dist, D[0], atol=1e-9)

    # --- power_iteration_trace ---

    def test_power_iteration_trace_aligns_with_top_eigvec(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((8, 8))
        M = A @ A.T  # symmetric PSD
        vals, vecs = np.linalg.eigh(M)
        top_eig = vecs[:, -1]  # largest eigenvalue last from eigh
        vectors, rayleigh = data.power_iteration_trace(M, iters=60, seed=0)
        final_vec = vectors[-1]
        cosine = abs(float(np.dot(final_vec, top_eig)))
        self.assertGreater(cosine, 0.999)

    def test_power_iteration_trace_rayleigh_converges(self):
        rng = np.random.default_rng(7)
        A = rng.standard_normal((6, 6))
        M = A @ A.T
        vals, _ = np.linalg.eigh(M)
        top_val = vals[-1]
        vectors, rayleigh = data.power_iteration_trace(M, iters=60, seed=0)
        self.assertAlmostEqual(rayleigh[-1], top_val, places=3)

    def test_power_iteration_trace_lengths(self):
        M = np.eye(5)
        vectors, rayleigh = data.power_iteration_trace(M, iters=10, seed=0)
        self.assertEqual(len(vectors), 11)
        self.assertEqual(len(rayleigh), 11)
        # Each vector should be unit length.
        for v in vectors:
            self.assertAlmostEqual(float(np.dot(v, v)), 1.0, places=10)

    # --- sample_along_path ---

    def test_sample_along_path_includes_endpoints(self):
        path = list(range(20))
        sampled = data.sample_along_path(path, m=4)
        self.assertEqual(sampled[0], path[0])
        self.assertEqual(sampled[-1], path[-1])

    def test_sample_along_path_returns_m_indices(self):
        path = list(range(20))
        sampled = data.sample_along_path(path, m=4)
        self.assertEqual(len(sampled), 4)

    def test_sample_along_path_all_in_path(self):
        path = [10, 5, 3, 99, 42, 7]
        sampled = data.sample_along_path(path, m=4)
        for idx in sampled:
            self.assertIn(idx, path)

    def test_sample_along_path_short_path(self):
        # Path shorter than m: should return all elements (or m, clamped).
        path = [0, 1, 2]
        sampled = data.sample_along_path(path, m=4)
        # Must include first and last.
        self.assertEqual(sampled[0], 0)
        self.assertEqual(sampled[-1], 2)

    # --- build_dataset new keys ---

    def test_build_dataset_new_keys_exist(self):
        d = data.build_dataset(n=120, k=8, seed=1)
        for key in ("center", "center_edges", "dijkstra_order",
                    "power_vectors", "power_rayleigh",
                    "sample_idx", "D_sample", "B_sample"):
            self.assertIn(key, d, msg=f"Missing key: {key}")

    def test_build_dataset_center_is_valid_node(self):
        d = data.build_dataset(n=120, k=8, seed=1)
        self.assertIsInstance(d["center"], int)
        self.assertGreaterEqual(d["center"], 0)
        self.assertLess(d["center"], 120)

    def test_build_dataset_center_edges_nonempty(self):
        d = data.build_dataset(n=120, k=8, seed=1)
        self.assertGreater(len(d["center_edges"]), 0)

    def test_build_dataset_dijkstra_order_length(self):
        d = data.build_dataset(n=120, k=8, seed=1)
        self.assertEqual(len(d["dijkstra_order"]), 120)

    def test_build_dataset_power_vectors_and_rayleigh_length(self):
        d = data.build_dataset(n=120, k=8, seed=1)
        self.assertEqual(len(d["power_rayleigh"]), 11)
        self.assertEqual(len(d["power_vectors"]), 11)

    def test_build_dataset_D_sample_B_sample_shape(self):
        d = data.build_dataset(n=120, k=8, seed=1)
        self.assertEqual(np.array(d["D_sample"]).shape, (4, 4))
        self.assertEqual(np.array(d["B_sample"]).shape, (4, 4))

    def test_build_dataset_sample_idx_length(self):
        d = data.build_dataset(n=120, k=8, seed=1)
        self.assertEqual(len(d["sample_idx"]), 4)


if __name__ == "__main__":
    unittest.main()
