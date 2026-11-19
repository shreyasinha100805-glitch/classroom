import os
import tempfile
import unittest
import weakref
from pathlib import Path

import cv2
import numpy as np
from unittest.mock import patch

from preprocessing.image_processing import (
    OCR_VARIANTS,
    make_ocr_variant,
    preprocess_image,
    safe_imread,
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

    def test_large_ocr_inputs_are_resized_to_the_1600_pixel_cap(self):
        image = np.zeros((1700, 2100, 3), dtype=np.uint8)
        original = image.copy()

        for variant_name in ("original_rgb", "upscaled", "hsv_chalk"):
            variant = make_ocr_variant(image, variant_name)
            self.assertLessEqual(max(variant.shape[:2]), 1600, variant_name)

        np.testing.assert_array_equal(image, original)

    def test_small_text_upscale_preserves_aspect_ratio_and_uses_two_x_cubic_size(self):
        variant = make_ocr_variant(self.image, "upscaled")

        self.assertEqual(variant.shape[:2], (480, 640))
        self.assertAlmostEqual(
            variant.shape[1] / variant.shape[0],
            self.image.shape[1] / self.image.shape[0],
        )

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

    def test_failed_header_probe_rejects_before_opencv_decode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "malformed.jpg"
            source.write_bytes(b"not an image")
            with patch(
                "preprocessing.image_processing.PilImage.open",
                side_effect=OSError("invalid image header"),
            ), patch("preprocessing.image_processing.cv2.imdecode") as decode:
                self.assertIsNone(safe_imread(str(source)))
            decode.assert_not_called()

    def test_decode_retry_runs_after_encoded_array_is_released(self):
        class ImageMetadata:
            size = (20, 20)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "placeholder.jpg"
            source.write_bytes(b"small encoded placeholder")
            encoded_array_refs = []

            def fail_decode(encoded, _flags):
                encoded_array_refs.append(weakref.ref(encoded))
                raise ValueError("synthetic decode failure")

            def fallback_read(_path):
                self.assertIsNone(encoded_array_refs[0]())
                return self.image.copy()

            with patch("preprocessing.image_processing.PilImage.open", return_value=ImageMetadata()), \
                 patch("preprocessing.image_processing.cv2.imdecode", new=fail_decode), \
                 patch("preprocessing.image_processing.cv2.imread", side_effect=fallback_read):
                decoded = safe_imread(str(source))

        self.assertEqual(decoded.shape, self.image.shape)

    def test_memory_error_does_not_retry_a_second_full_decode(self):
        class ImageMetadata:
            size = (20, 20)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "placeholder.jpg"
            source.write_bytes(b"small encoded placeholder")

            def fail_decode(*_args):
                raise MemoryError("synthetic decode allocation failure")

            with patch("preprocessing.image_processing.PilImage.open", return_value=ImageMetadata()), \
                 patch("preprocessing.image_processing.cv2.imdecode", new=fail_decode), \
                 patch("preprocessing.image_processing.cv2.imread") as fallback:
                self.assertIsNone(safe_imread(str(source)))
            fallback.assert_not_called()

    def test_oversized_image_is_rejected_before_full_decode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "over_limit.jpg"
            source.write_bytes(b"metadata-only image fixture")

            class ImageMetadata:
                size = (2830, 2830)
                format = "JPEG"

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def verify(self):
                    return None

            with patch("preprocessing.image_processing.MAX_UPLOAD_PIXELS", 8_000_000), patch(
                "preprocessing.image_processing.PilImage.open",
                return_value=ImageMetadata(),
            ), patch("preprocessing.image_processing.cv2.imdecode") as decode:
                self.assertIsNone(safe_imread(str(source)))
            decode.assert_not_called()

            from pipeline import validate_image

            with patch.dict(os.environ, {"MAX_UPLOAD_PIXELS": "8000000"}), patch(
                "pipeline.Image.open", return_value=ImageMetadata()
            ):
                validation = validate_image(str(source))
            self.assertFalse(validation["valid"])
            self.assertIn("too large", validation["error"])


if __name__ == "__main__":
    unittest.main()
