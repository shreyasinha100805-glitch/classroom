"""Single-process, bounded background analysis jobs for the Render demo."""

from collections import OrderedDict
from copy import deepcopy
import logging
import os
import queue
import threading
import time
from uuid import uuid4

logger = logging.getLogger(__name__)

JOB_RETENTION_SECONDS = 6 * 60 * 60
MAX_RETAINED_JOBS = 50
MAX_QUEUED_JOBS = 10


class AnalysisJobError(Exception):
    """An analysis failure with a safe message suitable for the status API."""

    def __init__(self, user_message):
        super().__init__(user_message)
        self.user_message = user_message


class JobQueueFullError(Exception):
    """Raised when the bounded in-memory queue cannot accept another job."""


class AnalysisJobManager:
    """Thread-safe job state and one lazily started worker per process."""

    def __init__(
        self,
        max_queued_jobs=MAX_QUEUED_JOBS,
        max_retained_jobs=MAX_RETAINED_JOBS,
        retention_seconds=JOB_RETENTION_SECONDS,
    ):
        self._queue = queue.Queue(maxsize=max_queued_jobs)
        self._max_retained_jobs = max_retained_jobs
        self._retention_seconds = retention_seconds
        self._jobs = OrderedDict()
        self._active_by_filename = {}
        self._lock = threading.RLock()
        self._worker = None
        self._worker_pid = None
        self._stopping = False

    def submit(self, filename, processor):
        """Queue a filename once; duplicate active submissions reuse its job."""
        now = time.time()
        with self._lock:
            self._cleanup_locked(now)
            active_id = self._active_by_filename.get(filename)
            active_job = self._jobs.get(active_id) if active_id else None
            if active_job and active_job["status"] in ("queued", "running"):
                return self._snapshot_locked(active_job), True

            self._ensure_worker_locked()
            job_id = uuid4().hex
            try:
                self._queue.put_nowait((job_id, filename, processor))
            except queue.Full as exc:
                raise JobQueueFullError from exc

            job = {
                "job_id": job_id,
                "filename": filename,
                "status": "queued",
                "current_stage": None,
                "result": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
                "finished_at": None,
            }
            self._jobs[job_id] = job
            self._active_by_filename[filename] = job_id
            self._trim_locked()
            return self._snapshot_locked(job), False

    def get(self, job_id):
        with self._lock:
            self._cleanup_locked(time.time())
            job = self._jobs.get(job_id)
            return self._snapshot_locked(job) if job else None

    def active_filenames(self):
        """Return files queued or running, for safe upload cleanup."""
        with self._lock:
            return set(self._active_by_filename)

    def start_worker(self):
        """Start the single worker, used by Gunicorn's post_fork hook."""
        with self._lock:
            self._ensure_worker_locked()
            return self._worker_pid

    def shutdown(self, timeout=2):
        """Stop this manager's worker; primarily useful to clean up tests."""
        with self._lock:
            worker = self._worker
            if not worker or not worker.is_alive() or self._stopping:
                return
            self._stopping = True
            self._queue.put(None, timeout=timeout)
        worker.join(timeout=timeout)

    def _ensure_worker_locked(self):
        if self._stopping:
            raise RuntimeError("Analysis job manager is stopping.")
        current_pid = os.getpid()
        if (
            self._worker_pid == current_pid
            and self._worker is not None
            and self._worker.is_alive()
        ):
            return

        # The worker starts on the first POST, after Gunicorn has forked its
        # preloaded app, so the thread is never started in the master process.
        self._worker_pid = current_pid
        logger.info(
            "[AnalysisJobs] Starting the single job worker in pid=%s (parent pid=%s)",
            current_pid,
            os.getppid(),
        )
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="classmind-analysis-worker",
            daemon=True,
        )
        self._worker.start()

    def _worker_loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return

            job_id, filename, processor = item
            try:
                with self._lock:
                    job = self._jobs.get(job_id)
                    if not job:
                        continue
                    job["status"] = "running"
                    job["updated_at"] = time.time()

                def report_stage(stage):
                    with self._lock:
                        current = self._jobs.get(job_id)
                        if current and current["status"] == "running":
                            current["current_stage"] = str(stage)[:120]
                            current["updated_at"] = time.time()

                try:
                    result = processor(filename, report_stage)
                    if not isinstance(result, dict) or not result.get("success", False):
                        failed_step = str((result or {}).get("failed_step", ""))
                        if failed_step.startswith("1.0"):
                            message = (
                                "The uploaded image is invalid or exceeds the 8 MP limit. "
                                "Use a valid JPEG or PNG image."
                            )
                        else:
                            message = "The image could not be processed. Please try again."
                        self._finish(job_id, "failed", error=message)
                    else:
                        self._finish(job_id, "completed", result=result)
                except Exception as exc:
                    logger.exception("[AnalysisJobs] Job %s failed for %s", job_id, filename)
                    message = getattr(exc, "user_message", None)
                    if not isinstance(message, str) or not message:
                        message = "The image could not be processed. Please try again."
                    self._finish(job_id, "failed", error=message)
            finally:
                self._queue.task_done()

    def _finish(self, job_id, status, result=None, error=None):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.update(
                status=status,
                current_stage=None,
                result=result,
                error=error,
                updated_at=time.time(),
                finished_at=time.time(),
            )
            if self._active_by_filename.get(job["filename"]) == job_id:
                self._active_by_filename.pop(job["filename"], None)
            self._cleanup_locked(time.time())
            self._trim_locked()

    def _cleanup_locked(self, now):
        cutoff = now - self._retention_seconds
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job["status"] in ("completed", "failed")
            and job["finished_at"] is not None
            and job["finished_at"] < cutoff
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)

    def _trim_locked(self):
        terminal_ids = [
            job_id
            for job_id, job in self._jobs.items()
            if job["status"] in ("completed", "failed")
        ]
        while len(self._jobs) > self._max_retained_jobs and terminal_ids:
            self._jobs.pop(terminal_ids.pop(0), None)

    @staticmethod
    def _snapshot_locked(job):
        snapshot = {"job_id": job["job_id"], "status": job["status"]}
        if job["status"] == "running" and job["current_stage"]:
            snapshot["current_stage"] = job["current_stage"]
        elif job["status"] == "completed":
            snapshot["result"] = deepcopy(job["result"])
        elif job["status"] == "failed":
            snapshot["error"] = job["error"]
        return snapshot


job_manager = AnalysisJobManager()
