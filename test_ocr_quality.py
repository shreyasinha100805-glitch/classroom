import unittest

from nlp.quality import assess_ocr_quality, low_confidence_notes


class OCRQualityTests(unittest.TestCase):
    def test_supported_terms_numbers_and_relations_can_form_notes_evidence(self):
        items = [
            {"text": "Database table", "confidence": 0.93},
            {"text": "Primary key", "confidence": 0.91},
            {"text": "Student ID = 12", "confidence": 0.94},
        ]
        quality = assess_ocr_quality(items)
        self.assertTrue(quality["reliable"])
        self.assertIn("Database table", quality["analysis_text"])
        self.assertIn("ID = 12", quality["analysis_text"])
        self.assertEqual(quality["supported_token_count"], 8)

    def test_uncertain_gibberish_is_retained_raw_but_never_becomes_notes(self):
        raw = [{"text": "fhrwqzz 8biece", "confidence": 0.61}]
        quality = assess_ocr_quality(raw)
        notes = low_confidence_notes(quality["warning"])
        self.assertEqual(raw[0]["text"], "fhrwqzz 8biece")
        self.assertFalse(quality["reliable"])
        self.assertEqual(quality["analysis_text"], "")
        self.assertEqual(notes["key_points"], [])
        self.assertEqual(notes["full_text"], "")
        self.assertIn("retake the image", notes["markdown"])

    def test_low_confidence_noise_does_not_lower_verified_region_confidence(self):
        items = [
            {"text": "Neural network model algorithm", "confidence": 0.88},
            {"text": "Training prediction system", "confidence": 0.86},
            {"text": "xqz", "confidence": 0.30},
        ]

        quality = assess_ocr_quality(items)

        self.assertTrue(quality["reliable"])
        self.assertEqual(quality["analysis_region_count"], 2)
        self.assertEqual(quality["analysis_average_confidence"], 0.87)
        self.assertEqual(quality["average_confidence"], 0.68)
        self.assertNotIn("xqz", quality["analysis_text"])

    def test_nonfinite_or_out_of_range_confidence_is_ignored(self):
        items = [
            {"text": "Database", "confidence": float("nan")},
            {"text": "Primary", "confidence": 1.2},
            {"text": "Student", "confidence": 0.9},
        ]

        quality = assess_ocr_quality(items)

        self.assertEqual(quality["valid_region_count"], 1)
        self.assertEqual(quality["average_confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
