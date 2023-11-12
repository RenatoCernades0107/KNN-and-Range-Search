import unittest
import numpy as np

from range_search import range_search


# Your test class
class TestRangeSearch(unittest.TestCase):
    def setUp(self):
        # Create a set of 10 5-dimensional vectors for training
        self.data = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
            [26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35],
            [36, 37, 38, 39, 40],
            [41, 42, 43, 44, 45],
            [46, 47, 48, 49, 50]
        ])

    # Se espera que devuelva el índice de los elementos dentro del radio
    def test_range_search_1(self):
        test_example = np.array([26, 28, 29, 31, 30])
        target_index = range_search(self.data, test_example, 5)
        self.assertEqual(target_index, [5])

    def test_range_search_2(self):
        test_example = np.array([26, 28, 29, 31, 30])
        target_index = range_search(self.data, test_example, 10)
        self.assertEqual(target_index, [5, 6])

    def test_range_search_3(self):
        test_example = np.array([26, 28, 29, 31, 30])
        target_index = range_search(self.data, test_example, 15)
        self.assertEqual(target_index, [4, 5, 6])


if __name__ == '__main__':
    unittest.main()