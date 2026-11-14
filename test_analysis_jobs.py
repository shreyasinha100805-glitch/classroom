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
        finally:
            analysis_jobs.job_manager = previous_manager

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
            response = self.client.post("/api/analyze/jobs", json={"filename": "board.jpg"})
            elapsed = time.perf_counter() - start
            self.assertEqual(response.status_code, 202)
            self.assertLess(elapsed, 0.5)
            self.assertEqual(response.get_json()["status"], "queued")
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


class PipelineProgressTests(unittest.TestCase):
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
            ocr_module.extract_text = lambda _path: [{"text": "biology", "confidence": 0.99}]
            keywords_module = types.ModuleType("nlp.keywords")
            keywords_module.extract_keywords = lambda _text, max_keywords=10: ["biology"]
            topics_module = types.ModuleType("nlp.topics")
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

    def test_frontend_copies_use_async_jobs_and_are_identical(self):
        root = Path(__file__).parent
        source = (root / "js" / "script.js").read_text(encoding="utf-8")
        published = (root / "public" / "js" / "script.js").read_text(encoding="utf-8")
        self.assertEqual(source, published)
        self.assertIn("/api/analyze/jobs", source)
        self.assertIn("/api/analyze/jobs/${encodeURIComponent(jobId)}", source)
        self.assertIn("setTimeout(resolve, 2500)", source)
        self.assertIn("Retry Pipeline Analysis", source)
        self.assertIn("runAnalysis(savedName, originalName || savedName)", source)
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
            def __init__(self):
                with constructor_lock:
                    constructor_calls.append(self)
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
            self.assertTrue(all(engine is constructor_calls[0] for engine in results))
        finally:
            extractor._ocr_engine = previous_engine
            if previous_module is None:
                sys.modules.pop("rapidocr_onnxruntime", None)
            else:
                sys.modules["rapidocr_onnxruntime"] = previous_module


if __name__ == "__main__":
    unittest.main()
