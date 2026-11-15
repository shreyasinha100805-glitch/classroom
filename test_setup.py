"""Check the production CPU/ONNX stack without pulling in legacy DL packages."""

import importlib.metadata
import unittest


class ProductionSetupTests(unittest.TestCase):
    def test_cpu_onnx_ocr_dependencies_are_available(self):
        packages = (
            "Flask",
            "numpy",
            "onnxruntime",
            "opencv-python",
            "Pillow",
            "rapidocr-onnxruntime",
        )
        versions = {name: importlib.metadata.version(name) for name in packages}
        self.assertTrue(all(versions.values()))
        import onnxruntime
        self.assertIn("CPUExecutionProvider", onnxruntime.get_available_providers())
        print("Production CPU/ONNX dependencies:", versions)


if __name__ == "__main__":
    unittest.main()
