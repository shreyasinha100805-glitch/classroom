"""Exercise upload, async job creation/polling, and real pipeline inference locally."""

import json
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
    sample_image = ROOT / "samples" / "sample_board.jpg"
    if not sample_image.is_file():
        raise FileNotFoundError(f"Sample classroom image is missing: {sample_image}")

    with tempfile.TemporaryDirectory(prefix="classmind-async-api-") as temp_dir:
        temp_root = Path(temp_dir)
        os.environ["UPLOAD_FOLDER"] = str(temp_root / "uploads")
        os.environ["PIPELINE_OUTPUT_FOLDER"] = str(temp_root / "pipeline_output")
        os.environ["DATABASE_PATH"] = str(temp_root / "database.db")
        os.environ["SECRET_KEY"] = secrets.token_hex(32)

        import app as app_module

        app_module.app.config.update(TESTING=True)
        client = app_module.app.test_client()
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
                )
            if upload_response.status_code != 200:
                raise AssertionError(f"Upload failed: {upload_response.status_code} {upload_response.get_data(as_text=True)}")
            uploaded = upload_response.get_json()["files"][0]
            filename = uploaded["saved_name"]

            post_started = time.perf_counter()
            accepted_response = client.post("/api/analyze/jobs", json={"filename": filename})
            create_latency = time.perf_counter() - post_started
            if accepted_response.status_code != 202:
                raise AssertionError(
                    f"Job creation failed: {accepted_response.status_code} "
                    f"{accepted_response.get_data(as_text=True)}"
                )
            accepted = accepted_response.get_json()
            if accepted.get("status") != "queued" or create_latency >= 1.0:
                raise AssertionError(f"Expected fast queued HTTP 202, got {accepted} in {create_latency:.3f}s")
            job_id = accepted["job_id"]

            duplicate_response = client.post("/api/analyze/jobs", json={"filename": filename})
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
                status_response = client.get(f"/api/analyze/jobs/{job_id}")
                poll_latencies.append(time.perf_counter() - poll_started)
                if status_response.status_code != 200:
                    raise AssertionError(
                        f"Status poll failed: {status_response.status_code} "
                        f"{status_response.get_data(as_text=True)}"
                    )
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
            if not result["structured_notes"].get("title"):
                raise AssertionError("Structured notes were not returned")

            unknown_response = client.get("/api/analyze/jobs/unknown-local-test-job")
            if unknown_response.status_code != 404:
                raise AssertionError(f"Unknown job ID should return HTTP 404, got {unknown_response.status_code}")

            report = {
                "upload_status": upload_response.status_code,
                "create_status": accepted_response.status_code,
                "create_latency_seconds": round(create_latency, 4),
                "duplicate_reused_job": duplicate["job_id"] == job_id,
                "job_status_transitions": statuses,
                "progress_stages_observed": stages_seen,
                "status_poll_count": len(poll_latencies),
                "max_poll_latency_seconds": round(max(poll_latencies), 4),
                "unknown_job_status": unknown_response.status_code,
                "final_job_status": final_job["status"],
                "pipeline_stages": result["stages"],
                "ocr_items": len(result["ocr_items"]),
                "keyword_count": len(result["keywords"]),
                "topic_count": len(result["topics"]),
                "notes_title": result["structured_notes"]["title"],
                "duration_seconds": round(time.perf_counter() - started_at, 2),
                "rss_before_mb": round(rss_before / 1024 / 1024, 1),
                "peak_rss_mb": round(peak_rss[0] / 1024 / 1024, 1),
                "rss_after_mb": round(process.memory_info().rss / 1024 / 1024, 1),
            }
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        finally:
            sampling_done.set()
            monitor.join(timeout=1)


if __name__ == "__main__":
    raise SystemExit(main())
