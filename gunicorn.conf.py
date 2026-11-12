# gunicorn.conf.py — Render production configuration
# Tuned for Render free-tier (512 MB RAM) running TF + PyTorch + EasyOCR

# --- Worker configuration ---
# 1 worker: TensorFlow, Keras, and EasyOCR models are loaded once per process.
# Multiple workers would multiply RSS by N (each loads its own TF + EasyOCR copy).
workers = 1

# 4 threads allow concurrent lightweight requests (health checks, file lists)
# while one thread is blocked on OCR inference.
threads = 4

# Worker class — sync is correct for CPU-bound inference; gevent/eventlet
# would not help here because GIL is held during TF/PyTorch compute.
worker_class = "sync"

# --- Timeout ---
# EasyOCR on a cold start (first request, model download) can take >60s.
# Subsequent requests on a warm worker typically complete in 10–40s.
timeout = 180

# --- Binding ---
bind = "0.0.0.0:$PORT"  # PORT is set by Render at runtime

# --- Logging ---
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(D)sµs'

# --- Memory management ---
# Restart a worker after 100 requests to prevent slow RSS growth
# from TF/PyTorch internal caches accumulating over many requests.
max_requests = 100
max_requests_jitter = 10

# Preload the app in the master process so model weights are loaded once
# and copy-on-write shared with any additional workers if ever scaled up.
preload_app = True
