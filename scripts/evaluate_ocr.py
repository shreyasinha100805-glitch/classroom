"""Benchmark OCR preprocessing variants on classroom images.

Example:
  python scripts/evaluate_ocr.py samples/sample_board.jpg samples/sample_text.jpg \
      samples/sample_diagram.jpg uploads/blackboard.jpeg

Pass --ground-truth for one image to compute CER/WER. With no transcript file,
both metrics are explicitly reported as NOT MEASURED.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import unicodedata

# Match the production app's bounded CPU settings before importing ONNX Runtime.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
cv2.setNumThreads(1)

from ocr.extractor import extract_text_from_image
from preprocessing.image_processing import OCR_VARIANTS, make_ocr_variant, safe_imread


def _edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for row, a in enumerate(left, 1):
        current = [row]
        for column, b in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (a != b),
            ))
        previous = current
    return previous[-1]


def _normalize(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+|[^\w\s]", normalized, flags=re.UNICODE))


def _error_rates(reference, hypothesis):
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    chars_ref, chars_hyp = ref, hyp
    words_ref, words_hyp = ref.split(), hyp.split()
    return {
        "cer": round(_edit_distance(chars_ref, chars_hyp) / max(1, len(chars_ref)), 4),
        "wer": round(_edit_distance(words_ref, words_hyp) / max(1, len(words_ref)), 4),
    }


def _case_label(path):
    name = Path(path).name.lower()
    if "board" in name:
        return "handwritten blackboard; white/yellow chalk"
    if "diagram" in name:
        return "diagram labels / database-table layout"
    if "text" in name:
        return "printed classroom text"
    return "classroom image; label supplied by caller"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images", nargs="*",
        default=["samples/sample_board.jpg", "samples/sample_text.jpg", "samples/sample_diagram.jpg"],
        help="Images to evaluate (defaults to all three project samples)",
    )
    parser.add_argument("--ground-truth", help="UTF-8 transcript for a single input image")
    parser.add_argument("--case-label", help="Override the test type label")
    parser.add_argument("--output", help="Write the UTF-8 JSON report to this path")
    parser.add_argument(
        "--variants", default=",".join(OCR_VARIANTS),
        help="Comma-separated variant names from preprocessing.image_processing.OCR_VARIANTS",
    )
    args = parser.parse_args()
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(OCR_VARIANTS))
    if unknown:
        parser.error(f"Unknown variant(s): {', '.join(unknown)}")
    if args.ground_truth and len(args.images) != 1:
        parser.error("--ground-truth can be used with one image at a time")

    truth = Path(args.ground_truth).read_text(encoding="utf-8") if args.ground_truth else None
    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    done = threading.Event()

    def sample_rss():
        while not done.is_set():
            try:
                peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            except psutil.Error:
                pass
            done.wait(0.05)

    monitor = threading.Thread(target=sample_rss, name="ocr-eval-rss", daemon=True)
    monitor.start()
    started = time.perf_counter()
    results = []
    try:
        for image_path in args.images:
            image = safe_imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image or pixel limit exceeded: {image_path}")
            height, width = image.shape[:2]
            for variant_name in variants:
                variant = make_ocr_variant(image, variant_name)
                inference_started = time.perf_counter()
                items = extract_text_from_image(variant, label=f"{image_path}:{variant_name}")
                latency = time.perf_counter() - inference_started
                recognized = " ".join(item["text"] for item in items)
                result = {
                    "image": image_path,
                    "test_type": args.case_label or _case_label(image_path),
                    "input_size": [width, height],
                    "variant": variant_name,
                    "latency_seconds": round(latency, 3),
                    "valid_text_regions": len(items),
                    "mean_confidence": round(
                        sum(item["confidence"] for item in items) / len(items), 4
                    ) if items else None,
                    "cer_wer": _error_rates(truth, recognized) if truth is not None else "NOT MEASURED",
                    "recognized_text": recognized,
                    "regions": items,
                }
                results.append(result)
                del variant
            del image
    finally:
        done.set()
        monitor.join(timeout=1)

    report = {
        "engine": "RapidOCR 1.4.4 / ONNX Runtime CPU",
        "alternative_model": "NOT TESTED; no handwritten ONNX weights or OCR transcription labels are installed in this project",
        "transcription_ground_truth": "available" if truth is not None else "unavailable; CER/WER NOT MEASURED",
        "variant_count_per_image": len(variants),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "peak_process_rss_mb": round(peak_rss[0] / 1024 / 1024, 1),
        "results": results,
    }
    rendered_report = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        report_path = Path(args.output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered_report + "\n", encoding="utf-8")
        print(f"Wrote OCR evaluation report: {report_path}")
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(rendered_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
