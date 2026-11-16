"""Tests for the bounded asynchronous analysis job workflow."""

import os
import runpy
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import analysis_jobs
from analysis_jobs import AnalysisJobError, AnalysisJobManager, JobQueueFullError
import pipeline
from PIL import Image


def wait_for_status(manager, job_id, expected, timeout=3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(job_id)
        if job and job["status"] == expected:
            return job
        time.sleep(0.01)
    raise AssertionError(f"Job {job_id} did not reach status {expected!r}")


class AnalysisJobManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = AnalysisJobManager(max_queued_jobs=2, max_retained_jobs=10)

    def tearDown(self):
        self.manager.shutdown()

    def test_worker_start_is_idempotent_in_current_process(self):
        first_pid = self.manager.start_worker()
        first_thread = self.manager._worker
        second_pid = self.manager.start_worker()
        self.assertEqual(first_pid, os.getpid())
        self.assertEqual(second_pid, first_pid)
        self.assertIs(self.manager._worker, first_thread)
        self.assertTrue(first_thread.is_alive())

    def test_gunicorn_post_fork_hook_starts_the_manager(self):
        config_path = Path(__file__).parent / "gunicorn.conf.py"
        config = runpy.run_path(str(config_path))

        class FakeLogger:
            def info(self, *_args, **_kwargs):
                pass

        class FakeServer:
            log = FakeLogger()

        class FakeWorker:
            pid = os.getpid()

        previous_manager = analysis_jobs.job_manager
        analysis_jobs.job_manager = self.manager
        try:
            config["post_fork"](FakeServer(), FakeWorker())
            self.assertEqual(self.manager._worker_pid, os.getpid())
            self.assertTrue(self.manager._worker.is_alive())
            self.assertTrue(config["worker_exit"](FakeServer(), FakeWorker()))
            self.assertFalse(self.manager._worker.is_alive())
        finally:
            analysis_jobs.job_manager = previous_manager

    def test_shutdown_fails_queued_jobs_and_finishes_active_worker(self):
        manager = AnalysisJobManager(max_queued_jobs=2, max_retained_jobs=10)
        active_started = threading.Event()
        release_active = threading.Event()

        def processor(filename, _report_stage):
            if filename == "active.jpg":
                active_started.set()
                release_active.wait(2)
            return {"success": True}

        try:
            active, _ = manager.submit("active.jpg", processor)
            self.assertTrue(active_started.wait(2))
            queued, _ = manager.submit("queued.jpg", processor)
            self.assertFalse(manager.shutdown(timeout=0.05))
            queued_state = manager.get(queued["job_id"])
            self.assertEqual(queued_state["status"], "failed")
            self.assertIn("server is restarting", queued_state["error"].lower())
            release_active.set()
            manager._worker.join(2)
            self.assertFalse(manager._worker.is_alive())
            self.assertEqual(manager.get(active["job_id"])["status"], "completed")
        finally:
            release_active.set()
            if manager._worker and manager._worker.is_alive():
                manager._worker.join(2)

    def test_progress_completion_and_duplicate_submission(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def processor(filename, report_stage):
            calls.append(filename)
            report_stage("OCR Text Extraction")
            started.set()
            self.assertTrue(release.wait(2))
            return {"success": True, "stages": [{"step": "1.0"}], "structured_notes": {"title": "Notes"}}

        first, duplicate = self.manager.submit("board.jpg", processor)
        self.assertFalse(duplicate)
        self.assertEqual(first["status"], "queued")
        self.assertTrue(started.wait(2))

        running = self.manager.get(first["job_id"])
        self.assertEqual(running["status"], "running")
        self.assertEqual(running["current_stage"], "OCR Text Extraction")

        repeated, duplicate = self.manager.submit("board.jpg", processor)
        self.assertTrue(duplicate)
        self.assertEqual(repeated["job_id"], first["job_id"])
        self.assertEqual(len(calls), 1)

        release.set()
        completed = wait_for_status(self.manager, first["job_id"], "completed")
        self.assertEqual(completed["result"]["structured_notes"]["title"], "Notes")

    def test_failed_job_exposes_safe_message_only(self):
        def processor(_filename, _report_stage):
            raise AnalysisJobError("Text extraction is temporarily unavailable.") from RuntimeError("private backend detail")

        job, _ = self.manager.submit("board.jpg", processor)
        failed = wait_for_status(self.manager, job["job_id"], "failed")
        self.assertEqual(failed["error"], "Text extraction is temporarily unavailable.")
        self.assertNotIn("private backend detail", str(failed))

    def test_queue_is_bounded_and_unknown_ids_are_safe(self):
        manager = AnalysisJobManager(max_queued_jobs=1, max_retained_jobs=10)
        started = threading.Event()
        release = threading.Event()

        def blocking_processor(_filename, _report_stage):
            started.set()
            release.wait(2)
            return {"success": True}

        try:
            first, _ = manager.submit("one.jpg", blocking_processor)
            self.assertTrue(started.wait(2))
            second, _ = manager.submit("two.jpg", blocking_processor)
            with self.assertRaises(JobQueueFullError):
                manager.submit("three.jpg", blocking_processor)
            self.assertIsNone(manager.get("unknown-job-id"))
            release.set()
            wait_for_status(manager, first["job_id"], "completed")
            wait_for_status(manager, second["job_id"], "completed")
        finally:
            release.set()
            manager.shutdown()

    def test_terminal_job_metadata_expires(self):
        manager = AnalysisJobManager(retention_seconds=10)
        try:
            job, _ = manager.submit("board.jpg", lambda _name, _progress: {"success": True})
            wait_for_status(manager, job["job_id"], "completed")
            with manager._lock:
                manager._jobs[job["job_id"]]["finished_at"] = time.time() - 11
            self.assertIsNone(manager.get(job["job_id"]))
        finally:
            manager.shutdown()

    def test_new_manager_does_not_retain_jobs_from_a_previous_process(self):
        old_manager = AnalysisJobManager()
        new_manager = AnalysisJobManager()
        try:
            job, _ = old_manager.submit("board.jpg", lambda _name, _progress: {"success": True})
            wait_for_status(old_manager, job["job_id"], "completed")
            self.assertIsNone(new_manager.get(job["job_id"]))
        finally:
            old_manager.shutdown()
            new_manager.shutdown()


class AnalysisJobRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as flask_app_module
        cls.flask_app_module = flask_app_module

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_upload_folder = self.flask_app_module.app.config["UPLOAD_FOLDER"]
        self.flask_app_module.app.config["UPLOAD_FOLDER"] = self.temp_dir.name
        with open(os.path.join(self.temp_dir.name, "board.jpg"), "wb") as image_file:
            image_file.write(b"test image bytes")

        self.previous_manager = analysis_jobs.job_manager
        self.manager = AnalysisJobManager(max_queued_jobs=2, max_retained_jobs=10)
        analysis_jobs.job_manager = self.manager
        self.client = self.flask_app_module.app.test_client()

    def tearDown(self):
        self.manager.shutdown()
        analysis_jobs.job_manager = self.previous_manager
        self.flask_app_module.app.config["UPLOAD_FOLDER"] = self.previous_upload_folder
        self.temp_dir.cleanup()

    def test_create_is_fast_and_status_returns_result(self):
        release = threading.Event()
        started = threading.Event()

        def processor(_filename, report_stage):
            report_stage("CNN Classification")
            started.set()
            release.wait(2)
            return {
                "success": True,
                "stages": [
                    {"step": f"{index}.0", "status": "completed"}
                    for index in range(1, 7)
                ],
                "structured_notes": {"title": "Test notes"},
            }

        with patch.object(self.flask_app_module, "_process_analysis_job", processor):
            start = time.perf_counter()
            response = self.client.post(
                "/api/analyze/jobs",
                json={"filename": "board.jpg"},
                headers={"Origin": "https://classroom-tan-iota.vercel.app"},
            )
            elapsed = time.perf_counter() - start
            self.assertEqual(response.status_code, 202)
            self.assertLess(elapsed, 0.5)
            self.assertEqual(
                response.headers["Access-Control-Allow-Origin"],
                "https://classroom-tan-iota.vercel.app",
            )
            self.assertIn("Origin", response.headers["Vary"])
            self.assertEqual(response.get_json()["status"], "queued")
            self.assertTrue(response.get_json()["success"])
            self.assertIsNone(response.get_json()["current_stage"])
            self.assertEqual(
                response.get_json()["progress"],
                {"completed_stages": 0, "total_stages": 6},
            )
            job_id = response.get_json()["job_id"]
            self.assertTrue(started.wait(2))

            running = self.client.get(f"/api/analyze/jobs/{job_id}")
            self.assertEqual(running.status_code, 200)
            self.assertEqual(running.get_json()["current_stage"], "CNN Classification")

            release.set()
            wait_for_status(self.manager, job_id, "completed")
            completed = self.client.get(f"/api/analyze/jobs/{job_id}")
            result = completed.get_json()["result"]
            self.assertEqual(len(result["stages"]), 6)
            self.assertTrue(all(stage["status"] == "completed" for stage in result["stages"]))
            self.assertEqual(result["structured_notes"]["title"], "Test notes")

    def test_cors_preflight_allows_only_required_headers_and_methods(self):
        origin = "https://classroom-tan-iota.vercel.app"
        with patch.object(self.flask_app_module, "TRUSTED_ORIGINS", [origin]):
            response = self.client.options(
                "/api/analyze/jobs",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type, authorization",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], origin)
        self.assertEqual(response.headers["Access-Control-Allow-Credentials"], "true")
        self.assertEqual(
            response.headers["Access-Control-Allow-Methods"],
            "GET, POST, OPTIONS, HEAD",
        )
        allowed_headers = {
            header.strip().lower()
            for header in response.headers["Access-Control-Allow-Headers"].split(",")
        }
        self.assertEqual(allowed_headers, {"content-type", "authorization", "accept"})
        self.assertNotIn("PUT", response.headers["Access-Control-Allow-Methods"])
        self.assertNotIn("DELETE", response.headers["Access-Control-Allow-Methods"])
        self.assertNotIn("origin", allowed_headers)
        self.assertIn("Origin", response.headers["Vary"])

    def test_cors_allows_configured_preview_origin_for_post_and_polling(self):
        preview_origin = "https://classroom-gdd95ascw-shreyasinha100805-glitchs-projects.vercel.app"
        preview_backend = "https://classroom-backend-preview.onrender.com"
        with patch.object(
            self.flask_app_module,
            "TRUSTED_ORIGINS",
            [preview_origin, preview_backend],
        ):
            create = self.client.post(
                "/api/analyze/jobs",
                json={},
                headers={"Origin": preview_origin},
            )
            poll = self.client.get(
                "/api/analyze/jobs/unknown-job",
                headers={"Origin": preview_origin},
            )

        self.assertEqual(create.status_code, 400)
        self.assertEqual(poll.status_code, 404)
        self.assertEqual(create.headers["Access-Control-Allow-Origin"], preview_origin)
        self.assertEqual(poll.headers["Access-Control-Allow-Origin"], preview_origin)

    def test_cors_headers_are_attached_to_api_error_responses(self):
        origin = "https://classroom-tan-iota.vercel.app"
        headers = {"Origin": origin}
        with patch.object(self.flask_app_module, "TRUSTED_ORIGINS", [origin]):
            responses = [
                self.client.post("/api/analyze/jobs", json={}, headers=headers),
                self.client.get("/api/analyze/jobs/unknown-job", headers=headers),
                self.client.get("/api/analyze/jobs", headers=headers),
                self.client.post(
                    "/api/auth/login",
                    json={"login_identifier": "unknown-user", "password": "invalid"},
                    headers=headers,
                ),
            ]
            with patch.object(
                self.manager,
                "submit",
                side_effect=JobQueueFullError(),
            ):
                responses.append(self.client.post(
                    "/api/analyze/jobs",
                    json={"filename": "board.jpg"},
                    headers=headers,
                ))
            with patch.object(
                self.flask_app_module,
                "run_pipeline_data",
                side_effect=RuntimeError("private pipeline details"),
            ):
                responses.append(self.client.post("/api/analyze/board.jpg", headers=headers))

        self.assertEqual([response.status_code for response in responses], [400, 404, 405, 401, 503, 500])
        for response in responses:
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], origin)
            self.assertEqual(response.headers["Access-Control-Allow-Credentials"], "true")
            self.assertIn("Origin", response.headers["Vary"])
            self.assertNotEqual(response.headers.get("Access-Control-Allow-Origin"), "*")

    def test_cors_rejects_an_untrusted_origin(self):
        origin = "https://untrusted.example"
        with patch.object(
            self.flask_app_module,
            "TRUSTED_ORIGINS",
            ["https://classroom-tan-iota.vercel.app"],
        ):
            response = self.client.get("/api/health", headers={"Origin": origin})

        self.assertEqual(response.status_code, 403)
        self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))
        self.assertIn("Origin", response.headers["Vary"])

    def test_cors_never_emits_wildcard_or_allows_opaque_origins(self):
        with patch.object(self.flask_app_module, "TRUSTED_ORIGINS", ["*"]):
            response = self.client.get(
                "/api/health",
                headers={"Origin": "https://classroom-tan-iota.vercel.app"},
            )
        self.assertEqual(response.status_code, 403)
        self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))

        with patch.object(self.flask_app_module, "TRUSTED_ORIGINS", ["null"]):
            response = self.client.get("/api/health", headers={"Origin": "null"})
        self.assertEqual(response.status_code, 403)
        self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))

    def test_duplicate_post_reuses_active_job_and_unknown_is_404(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def processor(filename, _report_stage):
            calls.append(filename)
            started.set()
            release.wait(2)
            return {"success": True}

        with patch.object(self.flask_app_module, "_process_analysis_job", processor):
            first = self.client.post("/api/analyze/jobs", json={"filename": "board.jpg"})
            self.assertTrue(started.wait(2))
            second = self.client.post("/api/analyze/jobs", json={"filename": "board.jpg"})
            self.assertEqual(first.status_code, 202)
            self.assertEqual(second.status_code, 202)
            self.assertTrue(second.get_json().get("duplicate"))
            self.assertEqual(first.get_json()["job_id"], second.get_json()["job_id"])
            self.assertEqual(
                self.client.get("/api/analyze/jobs/not-a-job").status_code,
                404,
            )
            self.client.get(f"/api/analyze/jobs/{first.get_json()['job_id']}")
            self.client.get(f"/api/analyze/jobs/{first.get_json()['job_id']}")
            self.assertEqual(calls, ["board.jpg"])
            release.set()

    def test_missing_filename_and_file_are_rejected(self):
        self.assertEqual(self.client.post("/api/analyze/jobs", json={}).status_code, 400)
        self.assertEqual(
            self.client.post("/api/analyze/jobs", json={"filename": "missing.jpg"}).status_code,
            404,
        )

    def test_get_jobs_is_reserved_and_never_treated_as_a_filename(self):
        response = self.client.get("/api/analyze/jobs")
        self.assertEqual(response.status_code, 405)
        self.assertEqual(response.get_json()["error_code"], "async_method_required")
        self.assertEqual(response.headers["Allow"], "POST, OPTIONS")
        self.assertIn("Use POST /api/analyze/jobs", response.get_json()["error"])
        self.assertNotIn("Image file 'jobs'", response.get_data(as_text=True))

    def test_failed_status_endpoint_returns_safe_error(self):
        def processor(_filename, _report_stage):
            raise AnalysisJobError("Text extraction is temporarily unavailable.") from RuntimeError("secret detail")

        with patch.object(self.flask_app_module, "_process_analysis_job", processor):
            accepted = self.client.post("/api/analyze/jobs", json={"filename": "board.jpg"})
            job_id = accepted.get_json()["job_id"]
            wait_for_status(self.manager, job_id, "failed")
            response = self.client.get(f"/api/analyze/jobs/{job_id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "failed")
        self.assertEqual(response.get_json()["error"], "Text extraction is temporarily unavailable.")
        self.assertIn("no-store", response.headers["Cache-Control"])

    def test_legacy_synchronous_endpoint_remains_available(self):
        result = {
            "success": True,
            "stages": [{"step": "1.0", "status": "completed"}],
            "classification": {"class": "board", "confidence": 90},
            "structured_notes": {"title": "Legacy notes"},
        }
        with patch.object(self.flask_app_module, "run_pipeline_data", return_value=result):
            response = self.client.get("/api/analyze/board.jpg")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), result)

    def test_legacy_synchronous_post_remains_available(self):
        result = {
            "success": True,
            "stages": [{"step": "1.0", "status": "completed"}],
            "structured_notes": {"title": "Legacy POST notes"},
        }
        with patch.object(self.flask_app_module, "run_pipeline_data", return_value=result):
            response = self.client.post("/api/analyze/board.jpg")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), result)

    def test_export_notes_uses_completed_result_without_running_inference(self):
        result = {
            "success": True,
            "structured_notes": {
                "markdown": "# Cached Board Notes",
                "plain_text": "Cached Board Notes",
            },
        }
        job, _ = self.manager.submit("board.jpg", lambda _name, _progress: result)
        wait_for_status(self.manager, job["job_id"], "completed")

        with patch.object(
            self.flask_app_module,
            "_execute_pipeline",
            side_effect=AssertionError("export must not rerun inference"),
        ):
            markdown = self.client.get("/api/export-notes/board.jpg?format=md")
            plain_text = self.client.get("/api/export-notes/board.jpg?format=txt")

        self.assertEqual(markdown.status_code, 200)
        self.assertEqual(markdown.get_data(as_text=True), "# Cached Board Notes")
        self.assertEqual(plain_text.status_code, 200)
        self.assertEqual(plain_text.get_data(as_text=True), "Cached Board Notes")

    def test_export_notes_rejects_withheld_unverified_notes(self):
        result = {
            "success": True,
            "ocr_warning": "OCR confidence is low; retake the image.",
            "ocr_quality": {"reliable": False},
            "structured_notes": {
                "title": "Text Capture Needs Review",
                "markdown": "# Text Capture Needs Review",
                "plain_text": "Text Capture Needs Review",
            },
        }
        with patch.object(
            analysis_jobs.job_manager,
            "completed_result_for_filename",
            return_value=result,
        ):
            markdown = self.client.get("/api/export-notes/board.jpg?format=md")
            plain_text = self.client.get("/api/export-notes/board.jpg?format=txt")

        for response in (markdown, plain_text):
            self.assertEqual(response.status_code, 409)
            self.assertIn("OCR transcript was not verified", response.get_json()["error"])

    def test_export_notes_requires_a_completed_job(self):
        response = self.client.get("/api/export-notes/board.jpg?format=md")
        self.assertEqual(response.status_code, 409)
        self.assertIn("wait for it to complete", response.get_json()["error"])

    def test_invalid_image_is_reported_without_running_ai_models(self):
        accepted = self.client.post("/api/analyze/jobs", json={"filename": "board.jpg"})
        self.assertEqual(accepted.status_code, 202)
        job_id = accepted.get_json()["job_id"]
        failed = wait_for_status(self.manager, job_id, "failed")
        self.assertIn("valid JPEG or PNG", failed["error"])

    def test_ocr_engine_failure_is_not_converted_to_empty_success(self):
        from ocr.extractor import OCREngineUnavailable

        with patch.object(
            self.flask_app_module,
            "_execute_pipeline",
            side_effect=OCREngineUnavailable("private OCR details"),
        ):
            with self.assertRaises(AnalysisJobError) as raised:
                self.flask_app_module._process_analysis_job("board.jpg", lambda _stage: None)

        self.assertEqual(
            raised.exception.user_message,
            "Text extraction is temporarily unavailable. Please retry shortly.",
        )


class UploadCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as flask_app_module
        cls.flask_app_module = flask_app_module

    def test_cleanup_preserves_active_upload_and_processed_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            uploads = Path(temp_dir) / "uploads"
            outputs = Path(temp_dir) / "pipeline_output"
            uploads.mkdir()
            outputs.mkdir()
            stale_time = time.time() - 25 * 60 * 60
            upload_names = ["active.jpg", "stale.jpg"]
            output_names = ["proc_active.jpg", "proc_stale.jpg"]
            for name in upload_names:
                path = uploads / name
                path.write_bytes(b"old upload")
                os.utime(path, (stale_time, stale_time))
            for name in output_names:
                path = outputs / name
                path.write_bytes(b"old processed image")
                os.utime(path, (stale_time, stale_time))

            with patch.object(self.flask_app_module, "UPLOAD_FOLDER", str(uploads)), \
                 patch.object(self.flask_app_module, "PIPELINE_OUTPUT_FOLDER", str(outputs)), \
                 patch.object(self.flask_app_module, "_last_cleanup_timestamp", 0), \
                 patch.object(
                     self.flask_app_module.analysis_jobs.job_manager,
                     "active_filenames",
                     return_value={"active.jpg"},
                 ):
                self.flask_app_module.cleanup_old_files()

            self.assertTrue((uploads / "active.jpg").exists())
            self.assertTrue((outputs / "proc_active.jpg").exists())
            self.assertFalse((uploads / "stale.jpg").exists())
            self.assertFalse((outputs / "proc_stale.jpg").exists())


class RapidOCRExtractorTests(unittest.TestCase):
    def test_list_and_tuple_elapsed_values_are_logged_safely(self):
        import ocr.extractor as extractor

        bbox = [[0, 0], [10, 0], [10, 10], [0, 10]]
        rapid_result = [[bbox, "board text", 0.95]]

        class FakeEngine:
            def __init__(self, elapsed):
                self.elapsed = elapsed

            def __call__(self, _image_path):
                return rapid_result, self.elapsed

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.jpg"
            Image.new("RGB", (20, 20), "white").save(image_path)
            for elapsed in (
                [0.1, None, "invalid", float("nan"), float("inf"), 0.2],
                (0.1, 0.2, None),
            ):
                with patch.object(extractor, "get_reader", return_value=FakeEngine(elapsed)):
                    result = extractor.extract_text(str(image_path))
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]["text"], "board text")
                self.assertEqual(result[0]["confidence"], 0.95)
                self.assertEqual(result[0]["bbox"], [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])

    def test_missing_ocr_input_is_not_reported_as_empty_success(self):
        import ocr.extractor as extractor

        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.jpg"
            with self.assertRaises(FileNotFoundError):
                extractor.extract_text(str(missing))


class CNNInferenceTests(unittest.TestCase):
    def test_real_onnx_prediction_is_repeatable_and_preserves_classes(self):
        import cnn.predict as cnn_predict

        image_path = Path(__file__).parent / "samples" / "sample_board.jpg"
        first = cnn_predict.predict_image(str(image_path))
        second = cnn_predict.predict_image(str(image_path))

        self.assertIn(first["class"], cnn_predict.CLASS_NAMES)
        self.assertEqual(set(first["probabilities"]), set(cnn_predict.CLASS_NAMES))
        self.assertAlmostEqual(sum(first["probabilities"].values()), 1.0, places=4)
        self.assertEqual(first["confidence"], first["probabilities"][first["class"]])
        self.assertEqual(first, second)

    def test_missing_model_fails_without_a_fallback(self):
        import cnn.predict as cnn_predict

        original_session = cnn_predict._session
        try:
            cnn_predict._session = None
            with tempfile.TemporaryDirectory() as temp_dir:
                missing_model = Path(temp_dir) / "missing-model.onnx"
                with patch.dict(os.environ, {"CNN_MODEL_PATH": str(missing_model)}):
                    with self.assertRaises(FileNotFoundError):
                        cnn_predict.get_model()
        finally:
            cnn_predict._session = original_session

    def test_missing_and_corrupt_images_fail_safely(self):
        import cnn.predict as cnn_predict

        missing = Path(tempfile.gettempdir()) / "classmind-missing-image.jpg"
        with self.assertRaises(FileNotFoundError):
            cnn_predict.predict_image(str(missing))

        with tempfile.NamedTemporaryFile(suffix=".jpg") as corrupt:
            corrupt.write(b"not an image")
            corrupt.flush()
            with self.assertRaises(OSError):
                cnn_predict.predict_image(corrupt.name)

    def test_empty_rapidocr_result_is_a_valid_empty_result(self):
        import ocr.extractor as extractor

        class EmptyEngine:
            def __call__(self, _image_path):
                return None, None

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.jpg"
            Image.new("RGB", (20, 20), "white").save(image_path)
            with patch.object(extractor, "get_reader", return_value=EmptyEngine()):
                self.assertEqual(extractor.extract_text(str(image_path)), [])


class PipelineProgressTests(unittest.TestCase):
    def test_cli_notes_receive_only_quality_filtered_ocr_regions(self):
        raw_items = [
            {"text": "Database table", "confidence": 0.99},
            {"text": "uncertain scribble", "confidence": 0.61},
        ]
        verified_items = [{"text": "Database table", "confidence": 0.99}]
        captured = {}

        def capture_notes(**kwargs):
            captured.update(kwargs)
            return {"title": "Verified Notes", "plain_text": "Verified Notes"}

        def make_module(name, **attributes):
            module = types.ModuleType(name)
            for key, value in attributes.items():
                setattr(module, key, value)
            return module

        fake_modules = {
            "preprocessing.image_processing": make_module(
                "preprocessing.image_processing",
                preprocess_image=lambda *_args: None,
            ),
            "cnn.predict": make_module(
                "cnn.predict",
                predict_image=lambda _path: {"class": "board", "confidence": 0.91},
            ),
            "ocr.extractor": make_module(
                "ocr.extractor", extract_text=lambda _path: raw_items,
            ),
            "nlp.quality": make_module(
                "nlp.quality",
                assess_ocr_quality=lambda _items: {
                    "analysis_items": verified_items,
                    "analysis_text": "Database table",
                    "reliable": True,
                    "warning": None,
                },
                low_confidence_notes=lambda _warning: {},
            ),
            "nlp.keywords": make_module(
                "nlp.keywords", extract_keywords=lambda *_args, **_kwargs: [],
            ),
            "nlp.topics": make_module(
                "nlp.topics", detect_topics=lambda _text: [],
            ),
            "nlp.notes": make_module(
                "nlp.notes", generate_structured_notes=capture_notes,
            ),
        }
        with patch.object(pipeline, "validate_image", return_value={
            "valid": True, "filename": "board.jpg", "width": 24, "height": 24,
            "format": "PNG", "size_bytes": 100,
        }), patch("pipeline.os.makedirs"), patch.dict(sys.modules, fake_modules):
            pipeline.run_pipeline("board.jpg")

        self.assertEqual(captured["ocr_items"], verified_items)

    def test_stage_callback_preserves_six_stage_result_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "board.jpg")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir)
            Image.new("RGB", (24, 24), color="white").save(image_path)

            preprocess_module = types.ModuleType("preprocessing.image_processing")

            def fake_preprocess(_input_path, output_path):
                Path(output_path).write_bytes(b"processed")

            preprocess_module.preprocess_image = fake_preprocess
            cnn_module = types.ModuleType("cnn.predict")
            cnn_module.predict_image = lambda _path: {
                "class": "board",
                "confidence": 0.91,
                "probabilities": {"board": 0.91, "text": 0.06, "background": 0.03},
            }
            ocr_module = types.ModuleType("ocr.extractor")
            ocr_module.OCREngineUnavailable = type("OCREngineUnavailable", (RuntimeError,), {})
            ocr_module.extract_text = lambda _path: [
                {"text": "Database table", "confidence": 0.99},
                {"text": "Primary key", "confidence": 0.98},
                {"text": "Student ID = 12", "confidence": 0.97},
            ]
            keywords_module = types.ModuleType("nlp.keywords")
            keywords_module.extract_keywords = lambda _text, max_keywords=10: ["biology"]
            topics_module = types.ModuleType("nlp.topics")
            topics_module.TOPIC_CATEGORIES = {}
            topics_module.detect_topics = lambda _text: [{"topic": "Biology", "keywords": ["biology"]}]
            notes_module = types.ModuleType("nlp.notes")
            notes_module.generate_structured_notes = lambda **_kwargs: {
                "title": "Biology Notes",
                "markdown": "# Biology Notes",
                "plain_text": "Biology Notes",
                "key_points": ["biology"],
                "topics": [],
                "keywords": ["biology"],
            }

            fake_modules = {
                "preprocessing.image_processing": preprocess_module,
                "cnn.predict": cnn_module,
                "ocr.extractor": ocr_module,
                "nlp.keywords": keywords_module,
                "nlp.topics": topics_module,
                "nlp.notes": notes_module,
            }
            reported_stages = []
            with patch.dict(sys.modules, fake_modules), patch.object(
                pipeline, "PIPELINE_OUTPUT_FOLDER", output_dir
            ):
                result = pipeline.run_pipeline_data(image_path, progress_callback=reported_stages.append)

            self.assertTrue(result["success"])
            self.assertEqual(len(result["stages"]), 6)
            self.assertTrue(all(stage["status"] == "completed" for stage in result["stages"]))
            self.assertEqual(reported_stages, [
                "Image Upload & Validation",
                "Image Preprocessing",
                "CNN Classification",
                "OCR Text Extraction",
                "NLP Analysis",
                "Structured Notes Generation",
            ])
            self.assertEqual(result["structured_notes"]["title"], "Biology Notes")

    def test_empty_ocr_text_generates_no_fabricated_board_points(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "blank.jpg")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir)
            Image.new("RGB", (24, 24), color="white").save(image_path)

            preprocess_module = types.ModuleType("preprocessing.image_processing")
            preprocess_module.preprocess_image = lambda source, output: Path(output).write_bytes(Path(source).read_bytes())
            cnn_module = types.ModuleType("cnn.predict")
            cnn_module.predict_image = lambda _path: {
                "class": "board",
                "confidence": 0.8,
                "probabilities": {"board": 0.8, "text": 0.1, "background": 0.1},
            }
            ocr_module = types.ModuleType("ocr.extractor")
            ocr_module.OCREngineUnavailable = type("OCREngineUnavailable", (RuntimeError,), {})
            ocr_module.extract_text = lambda _path: []

            with patch.dict(sys.modules, {
                "preprocessing.image_processing": preprocess_module,
                "cnn.predict": cnn_module,
                "ocr.extractor": ocr_module,
            }), patch.object(pipeline, "PIPELINE_OUTPUT_FOLDER", output_dir):
                result = pipeline.run_pipeline_data(image_path)

        self.assertTrue(result["success"])
        self.assertEqual(result["extracted_text"], "")
        self.assertEqual(result["ocr_items"], [])
        self.assertEqual(result["keywords"], [])
        self.assertEqual(result["topics"], [])
        self.assertEqual(result["structured_notes"]["key_points"], [])
        self.assertIn("No study notes were generated from uncertain text.", result["structured_notes"]["markdown"])
        self.assertEqual(result["stages"][3]["status"], "warning")

    def test_frontend_copies_use_async_jobs_and_are_identical(self):
        root = Path(__file__).parent
        source = (root / "js" / "script.js").read_text(encoding="utf-8")
        published = (root / "public" / "js" / "script.js").read_text(encoding="utf-8")
        flask_static = (root / "static" / "js" / "script.js").read_text(encoding="utf-8")
        css_source = (root / "css" / "style.css").read_text(encoding="utf-8")
        css_published = (root / "public" / "css" / "style.css").read_text(encoding="utf-8")
        css_flask_static = (root / "static" / "css" / "style.css").read_text(encoding="utf-8")
        self.assertEqual(source, published)
        self.assertEqual(source, flask_static)
        self.assertEqual(css_source, css_published)
        self.assertEqual(css_source, css_flask_static)
        self.assertIn("/api/analyze/jobs", source)
        self.assertIn("method: 'POST'", source)
        self.assertNotIn("JSON.stringify({ filename: 'jobs' })", source)
        self.assertIn("/api/analyze/jobs/${encodeURIComponent(jobId)}", source)
        self.assertIn("The connected backend does not support asynchronous analysis yet.", source)
        self.assertIn("asyncUnsupported", source)
        self.assertIn("setTimeout(resolve, 2500)", source)
        self.assertIn("Retry Pipeline Analysis", source)
        self.assertIn("runAnalysis(savedName, originalName || savedName)", source)
        self.assertIn("classmind_active_analysis:", source)
        self.assertIn("sessionStorage.setItem", source)
        self.assertIn("sessionStorage.getItem", source)
        self.assertIn("drawer.dataset.recoveredAnalysis", source)
        self.assertIn("setRunButtonBusy(activeAnalysisButton, true)", source)
        self.assertIn("button.disabled = isBusy", source)
        self.assertIn("let jobId = savedJobId", source)
        self.assertIn("clearActiveJobId(savedName, jobId)", source)
        self.assertNotIn("fetch(`/api/analyze/${encodeURIComponent(savedName)}`", source)


class InferenceSemaphoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as flask_app_module
        cls.flask_app_module = flask_app_module

    def test_pipeline_calls_are_serialized(self):
        lock = threading.Lock()
        active = 0
        peak_active = 0

        def fake_pipeline(_path, progress_callback=None):
            nonlocal active, peak_active
            with lock:
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return {"success": True}

        with patch.object(self.flask_app_module, "run_pipeline_data", fake_pipeline):
            workers = [
                threading.Thread(
                    target=self.flask_app_module._execute_pipeline,
                    args=(f"image-{index}.jpg",),
                    kwargs={"semaphore_timeout": None},
                )
                for index in range(2)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(2)

        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(peak_active, 1)

    def test_concurrent_reader_access_initializes_one_ocr_engine(self):
        import ocr.extractor as extractor

        previous_engine = extractor._ocr_engine
        previous_module = sys.modules.get("rapidocr_onnxruntime")
        constructor_calls = []
        constructor_lock = threading.Lock()

        class FakeRapidOCR:
            def __init__(self, **kwargs):
                with constructor_lock:
                    constructor_calls.append((self, kwargs))
                time.sleep(0.03)

        fake_module = types.ModuleType("rapidocr_onnxruntime")
        fake_module.RapidOCR = FakeRapidOCR
        extractor._ocr_engine = None
        sys.modules["rapidocr_onnxruntime"] = fake_module
        results = []
        threads = [threading.Thread(target=lambda: results.append(extractor.get_reader())) for _ in range(8)]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(2)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(len(constructor_calls), 1)
            engine, constructor_kwargs = constructor_calls[0]
            self.assertEqual(constructor_kwargs, {"intra_op_num_threads": 1, "inter_op_num_threads": 1})
            self.assertTrue(all(result is engine for result in results))
        finally:
            extractor._ocr_engine = previous_engine
            if previous_module is None:
                sys.modules.pop("rapidocr_onnxruntime", None)
            else:
                sys.modules["rapidocr_onnxruntime"] = previous_module


if __name__ == "__main__":
    unittest.main()
