"""Measure process RSS at each model and pipeline stage.

Example:
  python scripts/measure_pipeline_rss.py samples/sample_board.jpg

For the supported-input ceiling, pass a normal 4:3 JPEG at approximately
8 MP. The report separates cold app import, model-session initialization,
warm idle, preprocessing, CNN inference, OCR inference, the six-stage pipeline,
and serialized concurrent pipeline callers.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import runpy
import sys
import tempfile
import threading
import time
import ast

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _mb(byte_count):
    return round(byte_count / (1024 * 1024), 1)


def _measure_phase(process, label, operation, interval_seconds=0.005):
    rss_before = process.memory_info().rss
    peak_rss = [rss_before]
    done = threading.Event()

    def sample_rss():
        while not done.is_set():
            try:
                peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            except psutil.Error:
                pass
            done.wait(interval_seconds)

    monitor = threading.Thread(target=sample_rss, name=f"rss-{label}", daemon=True)
    monitor.start()
    started = time.perf_counter()
    result = None
    error = None
    try:
        result = operation()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        done.set()
        monitor.join(timeout=1)
        try:
            rss_after = process.memory_info().rss
            peak_rss[0] = max(peak_rss[0], rss_after)
        except psutil.Error:
            rss_after = peak_rss[0]

    measurement = {
        "phase": label,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "rss_before_mb": _mb(rss_before),
        "peak_rss_mb": _mb(peak_rss[0]),
        "rss_after_mb": _mb(rss_after),
        "peak_growth_mb": _mb(max(0, peak_rss[0] - rss_before)),
        "error": error,
    }
    return result, measurement


def _installed_versions():
    names = (
        "Flask",
        "gunicorn",
        "numpy",
        "opencv-python",
        "Pillow",
        "onnxruntime",
        "rapidocr-onnxruntime",
        "psutil",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT INSTALLED"
    return versions


def _gunicorn_config_snapshot():
    config_path = ROOT / "gunicorn.conf.py"
    tree = ast.parse(config_path.read_text(encoding="utf-8"))
    config = {}
    wanted = {"workers", "threads", "worker_class", "timeout", "preload_app"}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    try:
                        config[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a supported classroom image")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the full pipeline this many times in the same process",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Number of simultaneous pipeline callers in the serialization check",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        parser.error(f"image does not exist: {image_path}")

    process = psutil.Process()
    initial_rss = process.memory_info().rss
    measurements = []

    with tempfile.TemporaryDirectory(prefix="classmind-pipeline-") as output_dir:
        os.environ["PIPELINE_OUTPUT_FOLDER"] = output_dir

        def cold_application_startup():
            # Apply the same CPU thread limits and application preload ordering
            # as Gunicorn before importing the Flask app.
            runpy.run_path(str(ROOT / "gunicorn.conf.py"))
            return __import__("app")

        app, cold_start = _measure_phase(
            process, "cold_application_startup", cold_application_startup
        )
        measurements.append(cold_start)

        if app is None:
            report = {
                "success": False,
                "error": cold_start["error"],
                "measurements": measurements,
            }
        else:
            def initialize_sessions():
                from cnn.predict import get_model
                from ocr.extractor import get_reader

                get_model()
                get_reader()

            _, model_initialization = _measure_phase(
                process, "cnn_and_ocr_session_initialization", initialize_sessions
            )
            measurements.append(model_initialization)
            gc.collect()
            warm_idle_rss = process.memory_info().rss

            from ocr import extractor
            from PIL import Image

            with Image.open(image_path) as input_image:
                input_width, input_height = input_image.size
            ocr_policy = {
                "source_max_width": extractor.OCR_MAX_IMAGE_WIDTH,
                "source_max_height": extractor.OCR_MAX_IMAGE_HEIGHT,
                "source_max_pixels": extractor.OCR_MAX_IMAGE_PIXELS,
                "source_max_aspect_ratio": extractor.OCR_MAX_ASPECT_RATIO,
                "max_side_len": extractor.get_reader().max_side_len,
                "min_side_len": extractor.get_reader().min_side_len,
                "detector_limit_side_len": extractor.get_reader().text_det.limit_side_len,
                "detector_limit_type": extractor.get_reader().text_det.limit_type,
                "detector_tensor_limit_bytes": extractor.OCR_MAX_DETECTOR_TENSOR_BYTES,
                "detector_working_limit_bytes": extractor.OCR_MAX_DETECTOR_WORKING_BYTES,
                "input_geometry_estimate": extractor.estimate_ocr_detector_geometry(
                    input_width, input_height, engine=extractor.get_reader()
                ),
            }

            processed_path = Path(output_dir) / f"proc_{image_path.name}"

            def preprocess():
                from preprocessing.image_processing import preprocess_image

                return preprocess_image(str(image_path), str(processed_path))

            _, preprocessing = _measure_phase(process, "preprocessing", preprocess)
            measurements.append(preprocessing)

            def cnn_inference():
                from cnn.predict import predict_image

                return predict_image(str(processed_path))

            cnn_result, cnn_measurement = _measure_phase(
                process, "cnn_inference", cnn_inference
            )
            measurements.append(cnn_measurement)

            def ocr_inference():
                from ocr.extractor import extract_text

                return extract_text(str(image_path))

            ocr_result, ocr_measurement = _measure_phase(
                process, "ocr_inference", ocr_inference
            )
            measurements.append(ocr_measurement)

            def full_pipeline():
                runs = []
                for run_number in range(1, args.repeat + 1):
                    result = app._execute_pipeline(
                        str(image_path), semaphore_timeout=None
                    )
                    runs.append({
                        "run": run_number,
                        "success": bool(result.get("success")),
                        "stages": [
                            {
                                "step": stage.get("step"),
                                "title": stage.get("title"),
                                "status": stage.get("status"),
                            }
                            for stage in result.get("stages", [])
                        ],
                        "ocr_items": len(result.get("ocr_items", [])),
                        "extracted_text_chars": len(result.get("extracted_text", "")),
                        "notes_title": result.get("structured_notes", {}).get("title"),
                        "ocr_variant": result.get("ocr_quality", {}).get("variant"),
                        "ocr_warning": result.get("ocr_warning"),
                    })
                return runs

            full_runs, pipeline_measurement = _measure_phase(
                process, "complete_six_stage_pipeline", full_pipeline
            )
            measurements.append(pipeline_measurement)

            concurrent_runs = []
            concurrency_measurement = None
            if not any(item["error"] for item in measurements):
                def concurrent_pipeline_call(_index):
                    result = app._execute_pipeline(
                        str(image_path), semaphore_timeout=None
                    )
                    return bool(result.get("success"))

                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    concurrent_runs, concurrency_measurement = _measure_phase(
                        process,
                        "concurrent_pipeline_callers_serialized_by_app_semaphore",
                        lambda: list(pool.map(concurrent_pipeline_call, range(args.concurrency))),
                    )
                measurements.append(concurrency_measurement)

            failed_measurements = [item for item in measurements if item["error"]]
            latest_result = full_runs[-1] if full_runs else {}
            report = {
                "success": (
                    not failed_measurements
                    and bool(cnn_result)
                    and ocr_result is not None
                    and all(item.get("success") for item in (full_runs or []))
                    and all(concurrent_runs or [True])
                ),
                "image": str(image_path),
                "image_bytes": image_path.stat().st_size,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "runtime_dependency_versions": _installed_versions(),
                "gunicorn_config": _gunicorn_config_snapshot(),
                "ocr_geometry_policy": ocr_policy,
                "measurement_execution": (
                    "Gunicorn config is executed before Flask import to apply production CPU limits; "
                    "pipeline stages run directly in one process with the configured thread count "
                    "reported below. This harness does not launch a Gunicorn worker."
                ),
                "rss_sampling_interval_ms": 5,
                "warm_idle_after_cnn_and_ocr_initialization_mb": _mb(warm_idle_rss),
                "initial_script_rss_mb": _mb(initial_rss),
                "measured_peak_rss_mb": max(
                    [measurement["peak_rss_mb"] for measurement in measurements]
                    or [_mb(initial_rss)]
                ),
                "target_peak_rss_mb": 400,
                "target_met": max(
                    [measurement["peak_rss_mb"] for measurement in measurements]
                    or [_mb(initial_rss)]
                ) < 400,
                "cnn_result": cnn_result,
                "ocr_region_count": len(ocr_result or []),
                "full_pipeline_runs": full_runs or [],
                "concurrent_pipeline_results": concurrent_runs,
                "latest_ocr_variant": latest_result.get("ocr_variant"),
                "measurements": measurements,
            }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
