from unittest import TestCase

from .load_y import filter_Y


class TestFilter_Y(TestCase):

    def test_only_aanvraag_remains(self):
        Y = [
            'aanvraag',
            'aanvraag-besluit',
            'other',
            'besluit',
            'foo/bar',
        ]

        filtered = filter_Y(Y)

        self.assertSequenceEqual(filtered, [
            'aanvraag',
            'other',
            'other',
            'other',
            'other',
        ])
