import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from preprocessing.image_processing import (
    OCR_VARIANTS,
    make_ocr_variant,
    preprocess_image,
)


class OCRPreprocessingTests(unittest.TestCase):
    def setUp(self):
        # Synthetic green board with thin yellow and white strokes, a table, and
        # punctuation-like marks. Tests use temporary files only.
        self.image = np.full((240, 320, 3), (65, 105, 65), dtype=np.uint8)
        cv2.line(self.image, (30, 70), (275, 70), (40, 220, 235), 1)
        cv2.line(self.image, (40, 100), (285, 101), (230, 235, 230), 1)
        cv2.putText(self.image, "ID <= 12", (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 1)

    def test_all_evaluation_variants_are_bounded_and_do_not_mutate_source(self):
        original = self.image.copy()
        for variant_name in OCR_VARIANTS:
            variant = make_ocr_variant(self.image, variant_name)
            self.assertGreater(variant.size, 0, variant_name)
            self.assertLessEqual(max(variant.shape[:2]), 1600, variant_name)
        np.testing.assert_array_equal(self.image, original)

    def test_production_preprocessor_still_writes_for_classifier(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.png"
            output = Path(temp_dir) / "processed.jpg"
            self.assertTrue(cv2.imwrite(str(source), self.image))
            self.assertEqual(preprocess_image(str(source), str(output)), str(output))
            self.assertTrue(output.is_file())
            decoded = cv2.imread(str(output))
            self.assertEqual(decoded.shape[:2], self.image.shape[:2])

    def test_unknown_variant_is_an_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown OCR image variant"):
            make_ocr_variant(self.image, "magic")


if __name__ == "__main__":
    unittest.main()
