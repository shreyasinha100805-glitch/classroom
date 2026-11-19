import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import ocr.extractor as extractor


class FakeDetectorPreprocess:
    def __init__(self, limit_side_len=736, limit_type="min"):
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type


class FakeTextDetector:
    def __init__(self, limit_side_len=736, limit_type="min"):
        self.preprocess = FakeDetectorPreprocess(limit_side_len, limit_type)

    def get_preprocess(self, _max_wh):
        return self.preprocess


class FakeRapidOCREngine:
    max_side_len = extractor.RAPIDOCR_MAX_SIDE_LEN
    min_side_len = 30
    min_height = 30
    width_height_ratio = 8.0

    def __init__(
        self,
        limit_side_len=736,
        limit_type="min",
        min_height=30,
        width_height_ratio=8.0,
    ):
        self.text_det = FakeTextDetector(limit_side_len, limit_type)
        self.min_height = min_height
        self.width_height_ratio = width_height_ratio


class OCRGeometryPolicyTests(unittest.TestCase):
    def test_normal_classroom_aspect_ratios_fit_detector_memory_limits(self):
        cases = {
            "4:3 near 8 MP": (3264, 2448),
            "16:9": (3264, 1836),
            "portrait near 8 MP": (2448, 3264),
            "square near 8 MP": (2828, 2828),
        }

        for label, (width, height) in cases.items():
            with self.subTest(label=label):
                estimate = extractor.estimate_ocr_detector_geometry(
                    width, height, engine=FakeRapidOCREngine()
                )
                self.assertLessEqual(
                    estimate["detector_tensor_bytes"],
                    extractor.OCR_MAX_DETECTOR_TENSOR_BYTES,
                )
                self.assertLessEqual(
                    estimate["detector_working_bytes"],
                    extractor.OCR_MAX_DETECTOR_WORKING_BYTES,
                )
                self.assertLessEqual(
                    estimate["detector_width"],
                    extractor.OCR_MAX_DETECTOR_WIDTH,
                )
                self.assertLessEqual(
                    estimate["detector_height"],
                    extractor.OCR_MAX_DETECTOR_HEIGHT,
                )

    def test_extremely_wide_and_tall_geometry_is_rejected_without_reader(self):
        for width, height in ((29_900, 267), (267, 29_900)):
            with self.subTest(size=(width, height)):
                with self.assertRaises(extractor.OCRGeometryRejected):
                    extractor.estimate_ocr_detector_geometry(width, height)

    def test_aspect_ratio_and_each_dimension_are_explicitly_bounded(self):
        with self.assertRaises(extractor.OCRGeometryRejected):
            extractor.estimate_ocr_detector_geometry(4_000, 1_000)
        with self.assertRaises(extractor.OCRGeometryRejected):
            extractor.estimate_ocr_detector_geometry(4_001, 1_000)

    def test_live_detector_config_is_used_in_tensor_estimate(self):
        with self.assertRaises(extractor.OCRGeometryRejected):
            extractor.estimate_ocr_detector_geometry(
                3264,
                2448,
                engine=FakeRapidOCREngine(limit_side_len=3_000, limit_type="min"),
            )

        max_mode = extractor.estimate_ocr_detector_geometry(
            3264,
            2448,
            engine=FakeRapidOCREngine(limit_side_len=736, limit_type="max"),
        )
        self.assertLessEqual(
            max_mode["detector_tensor_bytes"],
            extractor.OCR_MAX_DETECTOR_TENSOR_BYTES,
        )

    def test_live_letterbox_configuration_must_match_the_geometry_model(self):
        for engine in (
            FakeRapidOCREngine(width_height_ratio=2.0),
            FakeRapidOCREngine(min_height=400),
        ):
            with self.subTest(engine=engine.__dict__):
                with self.assertRaises(extractor.OCRGeometryRejected):
                    extractor.estimate_ocr_detector_geometry(
                        1280, 960, engine=engine
                    )

    def test_supported_classroom_aspect_ratios_run_the_ocr_engine(self):
        engine_calls = []

        class OCRTestEngine(FakeRapidOCREngine):
            def __call__(self, image_path):
                engine_calls.append(image_path)
                return [], 0.01

        class ImageMetadata:
            size = (3264, 2448)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
            metadata = ImageMetadata()
            with patch.object(extractor.Image, "open", return_value=metadata), patch.object(
                extractor, "get_reader", return_value=OCRTestEngine()
            ):
                for width, height in (
                    (3264, 2448),  # 4:3, near the 8 MP upload ceiling
                    (3264, 1836),  # 16:9
                    (2448, 3264),  # portrait
                ):
                    with self.subTest(size=(width, height)):
                        metadata.size = (width, height)
                        self.assertEqual(extractor.extract_text(image_file.name), [])

        self.assertEqual(engine_calls, [image_file.name] * 3)

    def test_geometry_rejection_happens_from_metadata_before_model_or_decode(self):
        class ImageMetadata:
            size = (29_900, 267)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        for dimensions in ((29_900, 267), (267, 29_900)):
            with self.subTest(dimensions=dimensions):
                ImageMetadata.size = dimensions
                with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
                    with patch.object(extractor.Image, "open", return_value=ImageMetadata()), patch.object(
                        extractor, "get_reader", side_effect=AssertionError("models must not load")
                    ):
                        with self.assertRaises(extractor.OCRGeometryRejected):
                            extractor.extract_text(image_file.name)

    def test_image_just_over_the_upload_pixel_limit_is_rejected(self):
        with self.assertRaises(extractor.OCRGeometryRejected):
            extractor.estimate_ocr_detector_geometry(3266, 2450)


if __name__ == "__main__":
    unittest.main()
