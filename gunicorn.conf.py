# gunicorn.conf.py — Render production configuration
# Tuned for Render free-tier (512 MB RAM) running RapidOCR + ONNX Runtime.
# No PyTorch or EasyOCR required in this stack.
import os

# --- Worker configuration ---
# 1 worker: ONNX models are loaded once per process.
# Multiple workers would multiply RSS by N (each loads its own CNN + OCR models).
workers = 1

# 2 threads:
# - Allows lightweight concurrent requests (health, upload, gallery) while
#   one thread runs inference.
# - The /api/analyze endpoint uses threading.Semaphore(1) to serialise inference,
#   so more threads provide no extra throughput for analysis.
threads = 2

# Worker class — sync is correct for CPU-bound inference.
worker_class = "sync"

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

# --- Memory management ---
# Restart a worker after 100 requests to prevent slow RSS growth
# from ONNX Runtime internal caches accumulating over many requests.
# Higher than the previous 50 (EasyOCR stack) since RapidOCR's
# per-request memory growth is minimal.
max_requests = 100
max_requests_jitter = 10

# Preload the app in the master process so model weights are loaded once
# before the worker starts serving requests (no cold-start penalty on first request).
preload_app = True

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
