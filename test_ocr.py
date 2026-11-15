import unittest
from unittest.mock import patch

import numpy as np

import ocr.extractor as extractor


class OCRRegionValidationTests(unittest.TestCase):
    def test_keeps_raw_text_confidence_and_validates_box(self):
        bbox = [[5, 5], [45, 5], [45, 20], [5, 20]]

        class FakeEngine:
            def __call__(self, _image):
                return [
                    [bbox, "DoB", 0.91],
                    [bbox, "DoB", 0.80],  # overlapping duplicate
                    [[[80, 80], [120, 80], [120, 95], [80, 95]], "outside", 0.99],
                    [bbox, "bad confidence", float("nan")],
                ], [0.1, 0.2, None]

        with patch.object(extractor, "get_reader", return_value=FakeEngine()):
            items = extractor.extract_text_from_image(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"], "DoB")
        self.assertEqual(items[0]["confidence"], 0.91)
        self.assertEqual(items[0]["bbox"], [[5.0, 5.0], [45.0, 5.0], [45.0, 20.0], [5.0, 20.0]])

    def test_preserves_distinct_regions_even_when_text_repeats(self):
        first = [[5, 5], [45, 5], [45, 20], [5, 20]]
        second = [[55, 5], [95, 5], [95, 20], [55, 20]]

        class FakeEngine:
            def __call__(self, _image):
                return [[first, "ID", 0.88], [second, "ID", 0.89]], 0.1

        with patch.object(extractor, "get_reader", return_value=FakeEngine()):
            items = extractor.extract_text_from_image(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual([item["text"] for item in items], ["ID", "ID"])

    def test_rejects_malformed_boxes(self):
        class FakeEngine:
            def __call__(self, _image):
                return [[[[1, 1], [20, 20]], "text", 0.9]], 0.1

        with patch.object(extractor, "get_reader", return_value=FakeEngine()):
            self.assertEqual(
                extractor.extract_text_from_image(np.zeros((50, 50, 3), dtype=np.uint8)),
                [],
            )


if __name__ == "__main__":
    unittest.main()
