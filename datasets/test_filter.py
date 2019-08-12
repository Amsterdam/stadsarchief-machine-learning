from unittest import TestCase

from .filter import filter_unlabeled, filter_unchecked, filter_unknown


class TestFilter(TestCase):

    def test_filter_unlabeled(self):
        yaml = [
            {'type': 'foo'},
            {'type': 'bar'},
            {'type': ''},
            {'type': 'foo'},
        ]
        ids = ['id1', 'id2', 'id3', 'id4']
        yaml_filtered, ids_filtered = filter_unlabeled(yaml, ids)

        self.assertSequenceEqual(ids_filtered, ['id1', 'id2', 'id4'])
        self.assertSequenceEqual(yaml_filtered, [
            {'type': 'foo'},
            {'type': 'bar'},
            {'type': 'foo'},
        ])

    def test_filter_unchecked(self):
        yaml = [
            {'type': 'bar', 'checked': False},
            {'type': 'def'},
            {'type': 'xyz', 'checked': 'True'},
            {'type': 'foo', 'checked': True},
        ]
        ids = ['id1', 'id2', 'id3', 'id4']
        yaml_filtered, ids_filtered = filter_unchecked(yaml, ids)

        self.assertSequenceEqual(ids_filtered, ['id4', ])
        self.assertSequenceEqual(yaml_filtered, [
            {'type': 'foo', 'checked': True},
        ])

    def test_filter_unknown(self):
        yaml = [
            {'type': 'bar', 'otherprop': 'a'},
            {'type': 'unknown', 'otherprop': 'b'},
            {'type': 'xyz', 'otherprop': 'c'},
            {'type': 'foo', 'otherprop': 'd'},
        ]
        ids = ['id1', 'id2', 'id3', 'id4']
        yaml_filtered, ids_filtered = filter_unknown(yaml, ids)

        self.assertSequenceEqual(ids_filtered, ['id1', 'id3', 'id4' ])
        self.assertSequenceEqual(yaml_filtered, [
            {'type': 'bar', 'otherprop': 'a'},
            {'type': 'xyz', 'otherprop': 'c'},
            {'type': 'foo', 'otherprop': 'd'},
        ])
