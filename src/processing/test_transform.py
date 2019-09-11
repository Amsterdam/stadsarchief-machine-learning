from unittest import TestCase

from .transform import transform_aanvraag_labels


class TestTransform(TestCase):

    def test_only_aanvraag_remains(self):
        Y = [
            'aanvraag',
            'aanvraag-besluit',
            'other',
            'besluit',
            'foo/bar',
        ]

        filtered = transform_aanvraag_labels(Y)

        self.assertSequenceEqual(filtered, [
            'aanvraag',
            'other',
            'other',
            'other',
            'other',
        ])
