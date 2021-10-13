import unittest
import numpy as np
from matching_bandit.utils.p_dist import threshold2dist, origin_equal_distance, centered_equal_distance_with_zero, ends_fixed, equal_distance
from matching_bandit.utils.p_dist import linear_spaced, uncentered_equal_distance


precision = 0.00000001

class TestFunctions(unittest.TestCase):

    def test_threshold2dist(self):
        v = [0.1, 0.3, 0.6]
        self.assertEqual(threshold2dist(v).tolist(), [0.1, 0.1, 0.3, 0.3, 0.6, 0.6])
    
    def test_origin_equal_distance(self):
        n_pairs = 4
        Delta = 0.1
        expected = np.array([0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3])
        error = np.linalg.norm(origin_equal_distance(n_pairs, Delta) - expected, 2)
        self.assertLess(error, precision)

    def test_centered_equal_distance_with_zero(self):
        n_pairs = 5
        Delta = 0.1
        expected = np.array([0.35, 0.35, 0.45, 0.45, 0.55, 0.55, 0.65, 0.65, 0.0, 0.0])
        error = np.linalg.norm(centered_equal_distance_with_zero(n_pairs, Delta) - expected, 2)
        self.assertLess(error, precision)

        n_pairs = 4
        Delta = 0.2
        expected = np.array([0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.0, 0.0])
        error = np.linalg.norm(centered_equal_distance_with_zero(n_pairs, Delta) - expected, 2)
        self.assertLess(error, precision)

    def test_ends_fixed(self):
        n_pairs = 2
        Delta = 0.1
        expected = np.array([0.0, 0.0, 1.0, 1.0])
        error = np.linalg.norm(ends_fixed(n_pairs, Delta) - expected,2)
        self.assertLess(error, precision)

        n_pairs = 3
        Delta = 0.1
        expected = np.array([0.0, 0.0, 0.1, 0.1, 1.0, 1.0])
        error = np.linalg.norm(ends_fixed(n_pairs, Delta) - expected,2)
        self.assertLess(error, precision)

        n_pairs = 4
        Delta = 0.1
        expected = np.array([0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 1.0, 1.0])
        error = np.linalg.norm(ends_fixed(n_pairs, Delta) - expected,2)
        self.assertLess(error, precision)

        n_pairs = 11
        Delta = 0.1
        expected = np.array([0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0])
        error = np.linalg.norm(ends_fixed(n_pairs, Delta) - expected,2)
        self.assertLess(error, precision)

    def test_equal_distance(self):
        n_pairs = 2
        start = 0.5
        Delta = 0.1
        expected = np.array([0.5, 0.5, 0.6, 0.6])
        error = np.linalg.norm(equal_distance(n_pairs, start, Delta) - expected,2)
        self.assertLess(error, precision)

        n_pairs = 3
        start = 0.0
        Delta = 0.5
        expected = np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
        error = np.linalg.norm(equal_distance(n_pairs, start, Delta) - expected,2)
        self.assertLess(error, precision)

    def test_linear_spaced(self):
        n_pairs = 5
        expected = np.array([0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1])
        error = np.linalg.norm(linear_spaced(n_pairs) - expected,2)
        self.assertLess(error, precision)

        n_pairs = 2
        expected = np.array([0, 0, 1, 1])
        error = np.linalg.norm(linear_spaced(n_pairs) - expected,2)
        self.assertLess(error, precision)

        n_pairs = 3
        expected = np.array([0, 0, 0.5, 0.5, 1, 1])
        error = np.linalg.norm(linear_spaced(n_pairs) - expected,2)
        self.assertLess(error, precision)

        n_pairs = 4
        expected = np.array([0, 0, 1/3, 1/3, 2/3, 2/3, 1, 1])
        error = np.linalg.norm(linear_spaced(n_pairs) - expected,2)
        self.assertLess(error, precision)

    def test_uncentered_equal_distance(self):
        n_pairs = 4
        mu = 0.5
        Delta = 0.1
        expected = np.array([0.35, 0.35, 0.45, 0.45, 0.55, 0.55, 0.65, 0.65])
        error = np.linalg.norm(uncentered_equal_distance(n_pairs, mu, Delta) - expected, 2)
        self.assertLess(error, precision)

        n_pairs = 4
        mu = 0.6
        Delta = 0.1
        expected = np.array([0.45, 0.45, 0.55, 0.55, 0.65, 0.65, 0.75, 0.75])
        error = np.linalg.norm(uncentered_equal_distance(n_pairs, mu, Delta) - expected, 2)
        self.assertLess(error, precision)

        n_pairs = 4
        mu = 0.7
        Delta = 0.1
        expected = np.array([0.55, 0.55, 0.65, 0.65, 0.75, 0.75, 0.85, 0.85])
        error = np.linalg.norm(uncentered_equal_distance(n_pairs, mu, Delta) - expected, 2)
        self.assertLess(error, precision)

        n_pairs = 3
        mu = 0.8
        Delta = 0.2
        expected = np.array([0.6, 0.6, 0.8, 0.8, 1.0, 1.0])
        error = np.linalg.norm(uncentered_equal_distance(n_pairs, mu, Delta) - expected, 2)
        self.assertLess(error, precision)

if __name__ == '__main__':
    unittest.main()