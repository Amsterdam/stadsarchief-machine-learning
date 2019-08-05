from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from .load_data import filter_empty_label


class TestFilter_empty_label(TestCase):

    def test_should_filter(self):
        X = np.arange(12).reshape(4, 3)
        yaml = [
            {'type': 'foo'},
            {'type': 'bar'},
            {'type': ''},
            {'type': 'foo'},
        ]
        ids = ['id1', 'id2', 'id3', 'id4']
        results = filter_empty_label(X, yaml, ids)

        self.assertSequenceEqual(results[2], ['id1', 'id2', 'id4'])
        self.assertSequenceEqual(results[1], [
            {'type': 'foo'},
            {'type': 'bar'},
            {'type': 'foo'},
        ])
        assert_array_equal(results[0], np.array([
            [0, 1, 2],
            [3, 4, 5],
            [9, 10, 11],
        ]))
