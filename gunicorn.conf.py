# gunicorn.conf.py — Render production configuration
# Tuned for Render free-tier (512 MB RAM) running RapidOCR + ONNX Runtime.
# No PyTorch or EasyOCR required in this stack.
import os

# --- Worker configuration ---
# 1 worker: ONNX models are loaded once per process.
# Multiple workers would multiply RSS by N (each loads its own CNN + OCR models).
# Keep one process so the in-memory job queue and status polls share one manager.
workers = 1

# 2 threads:
# - Allows lightweight concurrent requests (health, upload, gallery) while
#   one thread runs inference.
# - The /api/analyze endpoint uses threading.Semaphore(1) to serialise inference,
#   so more threads provide no extra throughput for analysis.
threads = 2

# Two request threads keep job polling available while the background analysis
# thread runs. Explicit gthread avoids relying on Gunicorn's sync-to-gthread
# auto-switch when threads > 1.
worker_class = "gthread"

# --- Timeout ---
# RapidOCR on a cold start (first request, model download ~8 MB) can take >30s.
# Subsequent requests on a warm worker typically complete in 2-5s.
timeout = 120

# --- Binding ---
# Render injects PORT as an environment variable. gunicorn.conf.py is a plain
# Python file — shell variable expansion ($PORT) does NOT work here.
# Read PORT at config-load time instead.
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# --- Logging ---
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(D)sus'

# --- Worker lifecycle and memory ---
# Do not recycle the worker by request count: an async analysis runs in a
# process-local thread while the browser makes many short polling requests.
# A count-based recycle could kill that job mid-inference. Local repeated-run
# RSS testing stays below the 512 MB target; continue monitoring Render RSS.
# Platform restarts can still interrupt in-memory jobs.
max_requests = 0
max_requests_jitter = 0

# Preload app/model imports in the master so read-only model pages are shared by
# fork. The job manager does not start a thread during import; it starts lazily
# on the first POST inside the forked worker process.
preload_app = True


def post_fork(server, worker):
    """Start the job thread in the child after Gunicorn has forked it."""
    from analysis_jobs import job_manager

    job_pid = job_manager.start_worker()
    server.log.info(
        "Analysis job manager pid=%s initialized for Gunicorn worker pid=%s",
        job_pid,
        worker.pid,
    )


def worker_exit(server, worker):
    """Stop the process-local job thread before its Gunicorn worker exits."""
    from analysis_jobs import job_manager

    stopped = job_manager.shutdown(timeout=2)
    if stopped:
        server.log.info("Analysis job worker stopped during worker exit pid=%s", worker.pid)
    else:
        server.log.warning(
            "Analysis job worker did not stop before Gunicorn worker exit pid=%s",
            worker.pid,
        )
    return stopped

# --- CPU thread limits (saves ~5-10 MB RSS on single-core Render instances) ---
# Applied before workers fork so each worker inherits the limits.
# OpenCV: limits CPU thread pool used for image decode/resize.
try:
    import cv2
    cv2.setNumThreads(1)
except Exception:
    pass

# ONNX Runtime: thread limits are set per-session in cnn/predict.py
# (inter_op_num_threads=1, intra_op_num_threads=1).
# RapidOCR uses ONNX Runtime internally; it respects OMP_NUM_THREADS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
