"""Exercise upload, async job creation/polling, and real pipeline inference locally."""

import json
import argparse
import os
from pathlib import Path
import secrets
import sys
import tempfile
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=str(ROOT / "samples" / "sample_board.jpg"),
        help="Classroom image to upload and analyze (defaults to the repository board sample)",
    )
    parser.add_argument(
        "--origin",
        help="Optional exact browser Origin to validate through upload, gallery, async, and export routes",
    )
    args = parser.parse_args()
    sample_image = Path(args.image).expanduser().resolve()
    if not sample_image.is_file():
        raise FileNotFoundError(f"Sample classroom image is missing: {sample_image}")

    with tempfile.TemporaryDirectory(prefix="classmind-async-api-") as temp_dir:
        temp_root = Path(temp_dir)
        os.environ["UPLOAD_FOLDER"] = str(temp_root / "uploads")
        os.environ["PIPELINE_OUTPUT_FOLDER"] = str(temp_root / "pipeline_output")
        os.environ["DATABASE_PATH"] = str(temp_root / "database.db")
        os.environ["SECRET_KEY"] = secrets.token_hex(32)
        if args.origin:
            # Configure only the supplied exact origin; do not log environment values.
            os.environ["TRUSTED_ORIGINS"] = args.origin

        import app as app_module

        app_module.app.config.update(TESTING=True)
        client = app_module.app.test_client()
        origin_headers = {"Origin": args.origin} if args.origin else {}

        def check_cors(response, request_name):
            if args.origin and response.headers.get("Access-Control-Allow-Origin") != args.origin:
                raise AssertionError(f"{request_name} did not return CORS headers for the configured origin")

        preflight_response = None
        if args.origin:
            preflight_response = client.options(
                "/api/analyze/jobs",
                headers={
                    **origin_headers,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            if preflight_response.status_code != 200:
                raise AssertionError(f"Preview CORS preflight failed: {preflight_response.status_code}")
            check_cors(preflight_response, "Analysis preflight")

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

        monitor = threading.Thread(target=sample_rss, name="async-api-rss-sampler", daemon=True)
        monitor.start()
        started_at = time.perf_counter()
        try:
            with sample_image.open("rb") as image_file:
                upload_response = client.post(
                    "/api/upload",
                    data={"images": (image_file, sample_image.name)},
                    content_type="multipart/form-data",
                    headers=origin_headers,
                )
            if upload_response.status_code != 200:
                raise AssertionError(f"Upload failed: {upload_response.status_code} {upload_response.get_data(as_text=True)}")
            check_cors(upload_response, "Image upload")
            uploaded = upload_response.get_json()["files"][0]
            filename = uploaded["saved_name"]

            gallery_response = client.get("/api/files", headers=origin_headers)
            if gallery_response.status_code != 200:
                raise AssertionError(f"Gallery request failed: {gallery_response.status_code}")
            check_cors(gallery_response, "Gallery request")
            if not any(
                item.get("saved_name") == filename
                for item in gallery_response.get_json().get("files", [])
            ):
                raise AssertionError("The uploaded image was missing from the gallery response")

            post_started = time.perf_counter()
            accepted_response = client.post(
                "/api/analyze/jobs",
                json={"filename": filename},
                headers=origin_headers,
            )
            create_latency = time.perf_counter() - post_started
            if accepted_response.status_code != 202:
                raise AssertionError(
                    f"Job creation failed: {accepted_response.status_code} "
                    f"{accepted_response.get_data(as_text=True)}"
                )
            check_cors(accepted_response, "Analysis job creation")
            accepted = accepted_response.get_json()
            if accepted.get("status") != "queued" or create_latency >= 1.0:
                raise AssertionError(f"Expected fast queued HTTP 202, got {accepted} in {create_latency:.3f}s")
            job_id = accepted["job_id"]

            duplicate_response = client.post(
                "/api/analyze/jobs",
                json={"filename": filename},
                headers=origin_headers,
            )
            duplicate = duplicate_response.get_json()
            if (
                duplicate_response.status_code != 202
                or duplicate.get("job_id") != job_id
                or not duplicate.get("duplicate")
            ):
                raise AssertionError(f"Duplicate submission did not reuse the active job: {duplicate}")

            statuses = [accepted["status"]]
            stages_seen = []
            poll_latencies = []
            deadline = time.time() + 180
            final_job = None
            while time.time() < deadline:
                poll_started = time.perf_counter()
                status_response = client.get(
                    f"/api/analyze/jobs/{job_id}",
                    headers=origin_headers,
                )
                poll_latencies.append(time.perf_counter() - poll_started)
                if status_response.status_code != 200:
                    raise AssertionError(
                        f"Status poll failed: {status_response.status_code} "
                        f"{status_response.get_data(as_text=True)}"
                    )
                check_cors(status_response, "Analysis status poll")
                job = status_response.get_json()
                if not statuses or statuses[-1] != job["status"]:
                    statuses.append(job["status"])
                if job.get("current_stage") and job["current_stage"] not in stages_seen:
                    stages_seen.append(job["current_stage"])
                if job["status"] in ("completed", "failed"):
                    final_job = job
                    break
                time.sleep(0.1)

            if final_job is None:
                raise TimeoutError(f"Job {job_id} did not finish within 180 seconds")
            if final_job["status"] != "completed":
                raise AssertionError(f"Real pipeline job failed: {final_job}")

            result = final_job["result"]
            required_keys = {
                "success", "stages", "pipeline_spec", "classification", "extracted_text",
                "ocr_items", "keywords", "topics", "structured_notes", "processed_filename",
            }
            missing_keys = sorted(required_keys - result.keys())
            if missing_keys:
                raise AssertionError(f"Completed result is missing legacy pipeline fields: {missing_keys}")
            if len(result["stages"]) != 6:
                raise AssertionError(f"Expected six pipeline stages, got {len(result['stages'])}")
            if any(stage.get("status") not in ("completed", "warning") for stage in result["stages"]):
                raise AssertionError(f"Unexpected stage result: {result['stages']}")
            if result.get("ocr_warning") and not any(
                stage.get("step") == "4.0" and stage.get("status") == "warning"
                for stage in result["stages"]
            ):
                raise AssertionError("Unreliable OCR did not produce a visible stage warning")
            if not result["structured_notes"].get("title"):
                raise AssertionError("Structured notes were not returned")

            notes_available = (
                isinstance(result.get("ocr_quality"), dict)
                and result["ocr_quality"].get("reliable") is True
                and not result.get("ocr_warning")
                and bool(result["structured_notes"].get("markdown"))
                and bool(result["structured_notes"].get("plain_text"))
            )

            markdown_export = client.get(
                f"/api/export-notes/{filename}?format=md",
                headers=origin_headers,
            )
            text_export = client.get(
                f"/api/export-notes/{filename}?format=txt",
                headers=origin_headers,
            )
            expected_export_status = 200 if notes_available else 409
            for export_response in (markdown_export, text_export):
                if export_response.status_code != expected_export_status:
                    raise AssertionError(
                        f"Notes export returned {export_response.status_code}; "
                        f"expected {expected_export_status} based on OCR verification: "
                        f"{export_response.get_data(as_text=True)}"
                    )
                if notes_available:
                    if not export_response.headers.get("Content-Disposition", "").startswith("attachment;"):
                        raise AssertionError("Notes export did not set an attachment disposition")
                    if not export_response.get_data():
                        raise AssertionError("Notes export returned an empty file")
                elif "OCR transcript was not verified" not in export_response.get_data(as_text=True):
                    raise AssertionError("Unverified notes export did not explain why it was blocked")
                check_cors(export_response, "Notes export")

            unknown_response = client.get(
                "/api/analyze/jobs/unknown-local-test-job",
                headers=origin_headers,
            )
            if unknown_response.status_code != 404:
                raise AssertionError(f"Unknown job ID should return HTTP 404, got {unknown_response.status_code}")

            report = {
                "upload_status": upload_response.status_code,
                "gallery_status": gallery_response.status_code,
                "cors_origin_tested": args.origin,
                "cors_preflight_status": preflight_response.status_code if preflight_response else None,
                "upload_cors_origin": upload_response.headers.get("Access-Control-Allow-Origin"),
                "job_create_cors_origin": accepted_response.headers.get("Access-Control-Allow-Origin"),
                "poll_cors_origin": status_response.headers.get("Access-Control-Allow-Origin"),
                "create_status": accepted_response.status_code,
                "create_latency_seconds": round(create_latency, 4),
                "duplicate_reused_job": duplicate["job_id"] == job_id,
                "job_status_transitions": statuses,
                "progress_stages_observed": stages_seen,
                "status_poll_count": len(poll_latencies),
                "max_poll_latency_seconds": round(max(poll_latencies), 4),
                "unknown_job_status": unknown_response.status_code,
                "markdown_export_status": markdown_export.status_code,
                "text_export_status": text_export.status_code,
                "notes_available_for_export": notes_available,
                "final_job_status": final_job["status"],
                "pipeline_stages": result["stages"],
                "ocr_items": len(result["ocr_items"]),
                "keyword_count": len(result["keywords"]),
                "topic_count": len(result["topics"]),
                "notes_title": result["structured_notes"]["title"],
                "ocr_quality": result.get("ocr_quality"),
                "ocr_warning": result.get("ocr_warning"),
                "ocr_text_present": bool(result.get("extracted_text") or result.get("ocr_items")),
                "ocr_text_chars": len(result.get("extracted_text", "")),
                "duration_seconds": round(time.perf_counter() - started_at, 2),
                "rss_before_mb": round(rss_before / 1024 / 1024, 1),
                "peak_rss_mb": round(peak_rss[0] / 1024 / 1024, 1),
                "rss_after_mb": round(process.memory_info().rss / 1024 / 1024, 1),
            }
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        finally:
            sampling_done.set()
            monitor.join(timeout=1)


if __name__ == "__main__":
    raise SystemExit(main())
