"""Run the real analysis pipeline and sample its peak process RSS."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a valid classroom-board image")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the full pipeline this many times in the same process (default: 1)",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    output_dir = tempfile.mkdtemp(prefix="classmind-pipeline-")
    os.environ["PIPELINE_OUTPUT_FOLDER"] = output_dir

    import app

    image_path = os.path.abspath(args.image)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    peak_rss = [rss_before]
    sampling_done = threading.Event()

    def sample_rss():
        while not sampling_done.is_set():
            try:
                peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            except psutil.Error:
                pass
            sampling_done.wait(0.1)

    monitor = threading.Thread(target=sample_rss, name="rss-sampler", daemon=True)
    monitor.start()
    started_at = time.perf_counter()
    results = []
    errors = []
    try:
        for run_number in range(1, args.repeat + 1):
            run_started = time.perf_counter()
            try:
                result = app._execute_pipeline(image_path, semaphore_timeout=None)
                error = None
            except Exception as exc:
                result = {}
                error = f"{type(exc).__name__}: {exc}"
            results.append({
                "run": run_number,
                "success": bool(result.get("success")),
                "duration_seconds": round(time.perf_counter() - run_started, 2),
                "rss_after_mb": round(process.memory_info().rss / 1024 / 1024, 1),
                "error": error or result.get("error"),
                "failed_step": result.get("failed_step"),
                "stages": [
                    {"step": stage.get("step"), "title": stage.get("title"), "status": stage.get("status")}
                    for stage in result.get("stages", [])
                ],
                "ocr_items": len(result.get("ocr_items", [])),
                "extracted_text_chars": len(result.get("extracted_text", "")),
                "keyword_count": len(result.get("keywords", [])),
                "topic_count": len(result.get("topics", [])),
                "notes_title": result.get("structured_notes", {}).get("title"),
            })
            if error:
                errors.append(error)
                break
    finally:
        sampling_done.set()
        monitor.join(timeout=1)

    rss_after = process.memory_info().rss
    result = results[-1] if results else {}
    report = {
        "success": len(results) == args.repeat and all(run["success"] for run in results),
        "repeat_count_requested": args.repeat,
        "repeat_count_completed": len(results),
        "total_duration_seconds": round(time.perf_counter() - started_at, 2),
        "rss_before_mb": round(rss_before / 1024 / 1024, 1),
        "peak_rss_mb": round(peak_rss[0] / 1024 / 1024, 1),
        "rss_after_mb": round(rss_after / 1024 / 1024, 1),
        "rss_growth_mb": round((rss_after - rss_before) / 1024 / 1024, 1),
        "pipeline_output_dir": output_dir,
        "error": errors[0] if errors else result.get("error"),
        "runs": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
