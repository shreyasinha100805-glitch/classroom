# gunicorn.conf.py — Render production configuration
# Tuned for Render free-tier (512 MB RAM) running TF + PyTorch + EasyOCR
import os

# --- Worker configuration ---
# 1 worker: TensorFlow, Keras, and EasyOCR models are loaded once per process.
# Multiple workers would multiply RSS by N (each loads its own TF + EasyOCR copy).
workers = 1

# 2 threads (reduced from 4):
# - Allows concurrent lightweight requests (health, file list, upload) while
#   one thread runs OCR inference.
# - Reduced from 4 because TF model.predict() is NOT thread-safe without
#   explicit locking. The analyze endpoint uses a semaphore (see app.py)
#   to serialise inference, so more threads provide no extra throughput for
#   analysis and would only risk concurrent TF calls if the semaphore were removed.
threads = 2

# Worker class — sync is correct for CPU-bound inference.
worker_class = "sync"

# --- Timeout ---
# EasyOCR on a cold start (first request, model download) can take >60s.
# Subsequent requests on a warm worker typically complete in 10-40s.
timeout = 180

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
# Restart a worker after 50 requests to prevent slow RSS growth
# from TF/PyTorch internal caches accumulating over many requests.
max_requests = 50
max_requests_jitter = 5

# Preload the app in the master process so model weights are loaded once
# and copy-on-write shared with any additional workers if ever scaled up.
# NOTE: With workers=1, this provides no memory saving today but ensures
# models are initialised before the worker starts serving requests, so the
# first analysis request does not pay the cold-start penalty.
preload_app = True
