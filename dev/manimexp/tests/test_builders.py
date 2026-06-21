import unittest
import numpy as np
from manimexp.isomap import builders as B


class TestHeatmap(unittest.TestCase):
    def test_heatmap_downsamples_to_cell_budget(self):
        M = np.random.RandomState(0).rand(100, 100)
        hm = B.heatmap(M, 100, max_cells=32)
        # One square per displayed cell; downsampled grid is at most 32x32.
        rows = hm.meta["rows"]
        cols = hm.meta["cols"]
        self.assertLessEqual(rows, 32)
        self.assertLessEqual(cols, 32)
        self.assertEqual(len(hm.submobjects), rows * cols)

    def test_heatmap_small_matrix_is_exact(self):
        M = np.array([[0.0, 1.0], [1.0, 0.0]])
        hm = B.heatmap(M, 2, max_cells=32)
        self.assertEqual(hm.meta["rows"], 2)
        self.assertEqual(hm.meta["cols"], 2)
        self.assertEqual(len(hm.submobjects), 4)


if __name__ == "__main__":
    unittest.main()
