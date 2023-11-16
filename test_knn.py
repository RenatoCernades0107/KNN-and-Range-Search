import unittest
import numpy as np
from knn import KNN
from FixedMaxMinHeap import FixedMaxMinHeap


# Your test class
class TestKNN(unittest.TestCase):
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

    # Se espera que el fixed max min heap, después de hacer push a los elementos,
    # retorne los elementos elementos mas pequeños en orden descendente
    def test_fixed_max_min_heap(self):
        test_example = np.array([52, 16, 25, 72, 13, 1, 74, 35, 21, 62])
        expected_result = np.array([16, 13, 1])
        heap = FixedMaxMinHeap(3, comp_fun=lambda x: x[0])

        for i, elem in enumerate(test_example):
            heap.push((elem, i))

        for i in range(3):
            self.assertEqual(heap.pop()[0], expected_result[i])

    # Se espera que devuelva el índice del elemento más cercano de todos los datos
    def test_knn_1(self):
        test_example = np.array([26, 28, 29, 31, 30])
        target_index = KNN(self.data, test_example, k=1)
        self.assertEqual(target_index, [5])

    def test_knn_2(self):
        test_example = np.array([26, 28, 29, 31, 30])
        target_index = KNN(self.data, test_example, k=2)
        self.assertEqual(target_index, [5, 6])

    def test_knn_3(self):
        test_example = np.array([26, 28, 29, 31, 30])
        target_index = KNN(self.data, test_example, k=3, distance="inf")
        self.assertEqual(target_index, [5, 6, 4])


if __name__ == '__main__':
    unittest.main()