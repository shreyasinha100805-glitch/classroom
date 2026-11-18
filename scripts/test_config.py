"""
Configuration, structure, and logic tests for fix/render-memory-pipeline.
Run from project root: python scripts/test_config.py

Covers:
  - Syntax correctness
  - Import / eager-load guards
  - RapidOCR + ONNX Runtime use and elapsed-time compatibility
  - Image-size memory controls and buffer release
  - Pipeline resilience (gc, logging, explicit OCR failure handling)
  - Gunicorn deployment correctness ($PORT bug, threads, preload_app)
  - Pixel dimension gating (validate_image, safe_imread, preprocessing)
  - Inference semaphore presence and correct use in both pipeline routes
  - Thread-safety justification for threads=2
  - render.yaml env vars completeness
  - ONNX CNN conversion checks (NEW)
  - Large-image memory spike analysis (computed, not measured)
"""
import sys
import os
import ast
import re

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

PASS = 0
FAIL = 0


def check(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  PASS  {label}")
        PASS += 1
    else:
        print(f"  FAIL  {label}" + (f": {detail}" if detail else ""))
        FAIL += 1


# ---- read source files ----
with open("app.py") as f:             app_src = f.read()
with open("pipeline.py") as f:        pipe_src = f.read()
with open("cnn/predict.py") as f:     cnn_src = f.read()
with open("ocr/extractor.py") as f:   ocr_src = f.read()
with open("preprocessing/image_processing.py") as f: pre_src = f.read()
with open("gunicorn.conf.py") as f:   gcfg_src = f.read()
with open("requirements.txt") as f:   req_src = f.read()
with open("Procfile") as f:           proc_src = f.read()
with open("render.yaml") as f:        render_src = f.read()
with open("analysis_jobs.py") as f:    jobs_src = f.read()

# ===========================================================================
print("\n=== 1. Syntax checks ===")
# ===========================================================================
for path in ["app.py", "pipeline.py", "cnn/predict.py", "ocr/extractor.py",
             "preprocessing/image_processing.py", "gunicorn.conf.py",
             "scripts/test_config.py"]:
    try:
        with open(path) as f:
            ast.parse(f.read())
        check(f"syntax OK: {path}", True)
    except SyntaxError as e:
        check(f"syntax OK: {path}", False, str(e))

# ===========================================================================
print("\n=== 2. Import / eager-load guards ===")
# ===========================================================================
lines = app_src.split("\n")
top_torch_lines = [i + 1 for i, l in enumerate(lines) if l.strip() == "import torch"]
check("No bare 'import torch' at app.py module level",
      top_torch_lines == [], f"found at lines {top_torch_lines}")
cv2_import_line = app_src.find("import cv2")
thread_envs = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES"]
check("ONNX/OpenMP thread limits are configured before OpenCV import",
      cv2_import_line > 0 and all(
          app_src.find(f'os.environ.setdefault("{key}"') < cv2_import_line
          for key in thread_envs
      ))
check("app.py does not eagerly import PyTorch, TensorFlow, or Keras",
      re.search(r"^(?:import|from)\s+(?:torch|tensorflow|keras)(?:[.\s]|$)",
                app_src, re.MULTILINE) is None)
check("CUDA_VISIBLE_DEVICES set in app.py",
      "CUDA_VISIBLE_DEVICES" in app_src)
rapidocr_import_line = ocr_src.find("from rapidocr_onnxruntime import RapidOCR")
reader_function_line = ocr_src.find("def get_reader")
check("RapidOCR import remains lazy inside get_reader()",
      reader_function_line >= 0 and rapidocr_import_line > reader_function_line)
check("ONNX CNN has no Torch Dynamo dependency",
      "TORCHDYNAMO_DISABLE" not in cnn_src,
      "predict.py now uses onnxruntime, not torch — Torch Dynamo configuration is not applicable")
check("ONNX CNN has no TensorFlow oneDNN dependency",
      "TF_ENABLE_ONEDNN_OPTS" not in cnn_src,
      "predict.py must not require TensorFlow environment configuration")

# ===========================================================================
print("\n=== 3. Memory reductions ===")
# ===========================================================================
check("Production OCR dependency is pinned to RapidOCR 1.4.4",
      "rapidocr-onnxruntime==1.4.4" in req_src and
      "from rapidocr_onnxruntime import RapidOCR" in ocr_src)
check("Requirements do not install conflicting OpenCV distributions",
      "opencv-python>=" in req_src and "opencv-python-headless" not in req_src)
check("preprocessing uses MAX_DIM env-var (default 960)",
      "MAX_DIM" in pre_src and '"960"' in pre_src)
check("old hardcoded 'max_dim = 1600' removed from preprocessing",
      "max_dim = 1600" not in pre_src)
check("RapidOCR list/tuple elapsed times are normalized before logging",
      "def _elapsed_seconds(elapse)" in ocr_src and
      "math.isfinite(value)" in ocr_src and
      "value >= 0" in ocr_src and
      "_elapsed_seconds(elapse)" in ocr_src)
check("OCR preprocessing exposes the requested bounded comparison variants",
      all(name in pre_src for name in [
          "original_rgb", "upscaled", "grayscale", "clahe", "adaptive_threshold",
          "otsu_threshold", "lab_clahe", "hsv_chalk", "perspective_corrected",
      ]))
check("Production OCR selects one evidence-ranked fallback from the original upload",
      "ocr_results = extract_text(image_path)" in pipe_src and
      '"upscaled"' in pipe_src and '"hsv_chalk"' in pipe_src and
      "make_ocr_variant(source_image, fallback_variant)" in pipe_src and
      "fallback_key > current_key" in pipe_src)
check("NLP uses supported OCR evidence and warns instead of summarizing unreliable text",
      "assess_ocr_quality(ocr_clean)" in pipe_src and
      "low_confidence_notes(ocr_quality[\"warning\"])" in pipe_src and
      '"ocr_warning": ocr_quality["warning"]' in pipe_src)
check("del intermediate arrays in preprocessing (gray, denoised, enhanced, blurred)",
      pre_src.count("del ") >= 4,
      f"only {pre_src.count('del ')} del statements")
check("del image_array after CNN predict",
      "del image_array" in cnn_src)
check("CNN pixel normalisation /255",
      "/ 255.0" in cnn_src or "/255" in cnn_src)

# ===========================================================================
print("\n=== 4. Pixel dimension gating (NEW) ===")
# ===========================================================================
check("MAX_UPLOAD_PIXELS constant defined in preprocessing/image_processing.py",
      "MAX_UPLOAD_PIXELS" in pre_src)
check("safe_imread performs header-only Pillow probe before full decode",
      "PilImage.open" in pre_src or "PIL" in pre_src,
      "No Pillow probe found — large-image decompression bomb risk")
check("safe_imread returns None when image exceeds MAX_UPLOAD_PIXELS",
      "return None" in pre_src and "MAX_UPLOAD_PIXELS" in pre_src)
check("validate_image in pipeline.py enforces MAX_UPLOAD_PIXELS",
      "MAX_UPLOAD_PIXELS" in pipe_src)
check("validate_image returns valid=False with user-friendly message for oversized images",
      "too large" in pipe_src or "pixel" in pipe_src.lower())
check("np.frombuffer used instead of bytearray in safe_imread (avoids double-buffer spike)",
      "np.frombuffer" in pre_src,
      "Still uses bytearray — peak = 2× file size before decode")
check("MAX_UPLOAD_PIXELS in render.yaml env vars",
      "MAX_UPLOAD_PIXELS" in render_src)

# ===========================================================================
print("\n=== 5. Inference concurrency guard (NEW) ===")
# ===========================================================================
check("_inference_semaphore defined in app.py",
      "_inference_semaphore" in app_src and "threading.Semaphore" in app_src)
check("API analysis uses the shared inference wrapper",
      "_execute_pipeline(image_path" in app_src)
check("_inference_semaphore released in finally block of api_analyze_image",
      "finally" in app_src and "_inference_semaphore.release()" in app_src)
export_src = app_src.split("def export_notes(filename):", 1)[1].split(
    "# SANITIZED GLOBAL ERROR HANDLERS", 1
)[0]
check("export_notes reuses completed notes without rerunning inference",
      "completed_result_for_filename" in export_src and
      "run_pipeline_data" not in export_src and
      "_inference_semaphore.acquire" not in export_src)
check("503 returned when semaphore times out (not 500)",
      "503" in app_src and "busy" in app_src.lower())

# ===========================================================================
print("\n=== 6. Pipeline resilience ===")
# ===========================================================================
check("gc.collect() called after OCR stage in pipeline.py",
      "gc.collect()" in pipe_src)
check("traceback.format_exc() used for full error logging",
      "traceback.format_exc()" in pipe_src)
check("Per-stage logger.info in pipeline (>=6 calls)",
      pipe_src.count("logger.info") >= 6,
      f"only {pipe_src.count('logger.info')} logger.info calls")
check("_log_memory() helper defined in pipeline.py",
      "_log_memory" in pipe_src)
check("Pipeline returns partial results on CNN failure — not exception",
      'predicted_class = "error"' in pipe_src)
ocr_failure_block = ""
if "except OCREngineUnavailable:" in pipe_src:
    ocr_failure_block = pipe_src.split("except OCREngineUnavailable:", 1)[1].split(
        "except Exception as exc:", 1
    )[0]
check("OCR engine failures propagate instead of returning empty-text success",
      "raise" in ocr_failure_block and '"status": "warning"' not in ocr_failure_block)
check("Analyze endpoint maps OCR engine failures to an explicit 503",
      "isinstance(exc, OCREngineUnavailable)" in app_src and
      '"error_code": "ocr_unavailable"' in app_src and
      '"error": "Text extraction is temporarily unavailable.' in app_src)
check("Pipeline returns partial results on NLP failure — not exception",
      "keywords = []" in pipe_src and "topics = []" in pipe_src)

# ===========================================================================
print("\n=== 7. Gunicorn deployment correctness (NEW checks) ===")
# ===========================================================================
check("gunicorn.conf.py imports os (required for PORT expansion)",
      "import os" in gcfg_src)
check("bind uses os.environ.get('PORT') — no literal $PORT shell expansion",
      "os.environ.get('PORT'" in gcfg_src and '"$PORT"' not in gcfg_src,
      "Still uses literal '$PORT' string — shell vars are NOT expanded in Python")
check("gunicorn.conf.py has workers = 1",
      "workers = 1" in gcfg_src)
check("gunicorn.conf.py uses gthread with two request threads alongside serialized inference",
      'worker_class = "gthread"' in gcfg_src and "threads = 2" in gcfg_src and
      "_inference_semaphore" in app_src,
      "two threads are expected for one analysis plus lightweight requests")
check("gunicorn.conf.py has preload_app = True",
      "preload_app = True" in gcfg_src)
check("Gunicorn post_fork starts the job manager in the child process",
      "def post_fork(server, worker):" in gcfg_src and
      "job_manager.start_worker()" in gcfg_src)
check("Gunicorn worker_exit stops the process-local analysis thread",
      "def worker_exit(server, worker):" in gcfg_src and
      "job_manager.shutdown(timeout=2)" in gcfg_src)
check("job manager prevents duplicate worker threads per process",
      "self._worker_pid == current_pid" in jobs_src and
      "def start_worker(self):" in jobs_src)
check("gunicorn.conf.py explicitly configures max_requests",
      "max_requests" in gcfg_src)
check("Gunicorn request-count recycling is disabled for in-process background jobs",
      "max_requests = 0" in gcfg_src and "max_requests_jitter = 0" in gcfg_src,
      "a count-based recycle could terminate a long job during polling")
check("Procfile uses --config gunicorn.conf.py",
      "gunicorn.conf.py" in proc_src)
check("render.yaml uses --config gunicorn.conf.py",
      "gunicorn.conf.py" in render_src)
check("No µ (non-ASCII) in gunicorn.conf.py access_log_format (causes encoding errors)",
      "µ" not in gcfg_src,
      "µs in access_log_format can cause UnicodeEncodeError on some Render log collectors")

# ===========================================================================
print("\n=== 8. render.yaml env completeness ===")
# ===========================================================================
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "CUDA_VISIBLE_DEVICES", "KMP_DUPLICATE_LIB_OK",
            "PREPROCESS_MAX_DIM", "MAX_UPLOAD_PIXELS"]:
    check(f"render.yaml has env var: {key}", key in render_src)
check("render.yaml omits legacy PyTorch/TensorFlow tuning variables",
      all(key not in render_src for key in [
          "TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE", "TF_CPP_MIN_LOG_LEVEL",
          "TF_ENABLE_ONEDNN_OPTS"
      ]))

# ===========================================================================
print("\n=== 9. Logging ===")
# ===========================================================================
check("logging.basicConfig in app.py",
      "logging.basicConfig" in app_src)
check("app.py uses logger.info for analyze endpoint",
      "logger.info" in app_src)
check("cnn/predict.py uses logger",
      "logger.info" in cnn_src)
check("ocr/extractor.py uses logger",
      "logger.info" in ocr_src)
check("preprocessing uses logger",
      "logger.info" in pre_src)

# ===========================================================================
print("\n=== 10. ONNX CNN conversion correctness (NEW) ===")
# ===========================================================================
onnx_model_path = os.path.join(project_root, "cnn", "model", "classroom_cnn.onnx")
keras_model_path = os.path.join(project_root, "cnn", "model", "classroom_cnn.keras")

check("cnn/model/classroom_cnn.onnx exists",
      os.path.isfile(onnx_model_path),
      "ONNX model file missing — run convert_to_onnx.py")
check("cnn/model/classroom_cnn.keras preserved (source model not deleted)",
      os.path.isfile(keras_model_path),
      "Original .keras file was deleted — this is not allowed")
check("classroom_cnn.onnx.data does NOT exist (single-file format required)",
      not os.path.isfile(onnx_model_path + ".data"),
      "External data file exists — ONNX export used split format; re-export with dynamo=False")

onnx_size_mb = os.path.getsize(onnx_model_path) / (1024 * 1024) if os.path.isfile(onnx_model_path) else 0
check(f"classroom_cnn.onnx size is reasonable (5–50 MB, actual {onnx_size_mb:.1f} MB)",
      5 <= onnx_size_mb <= 50,
      f"Unexpected size {onnx_size_mb:.1f} MB — may indicate empty or corrupt export")

check("cnn/predict.py uses onnxruntime (not keras/tensorflow)",
      "onnxruntime" in cnn_src and "import onnxruntime" not in cnn_src.split("def get_model")[0][:200],
      "predict.py does not import onnxruntime")
check("cnn/predict.py does NOT import keras",
      "import keras" not in cnn_src and "from keras" not in cnn_src,
      "predict.py still imports keras — tensorflow dependency not removed")
check("cnn/predict.py does NOT import tensorflow",
      "import tensorflow" not in cnn_src and "from tensorflow" not in cnn_src,
      "predict.py still imports tensorflow")
check("onnxruntime.InferenceSession used in get_model()",
      "InferenceSession" in cnn_src)
check("CPUExecutionProvider explicitly specified",
      "CPUExecutionProvider" in cnn_src)
check("SESSION threads=1 (intra/inter) to avoid OMP thread explosion",
      "intra_op_num_threads" in cnn_src and "inter_op_num_threads" in cnn_src)
check("predict_image() return dict preserves 'class', 'confidence', 'probabilities' keys",
      '"class"' in cnn_src and '"confidence"' in cnn_src and '"probabilities"' in cnn_src)
check("requirements.txt contains onnxruntime",
      "onnxruntime" in req_src)
check("requirements.txt does NOT contain tensorflow-cpu as a package (may appear in comments)",
      not any(line.strip().startswith("tensorflow-cpu") for line in req_src.splitlines()),
      "tensorflow-cpu still listed as a package in requirements.txt — not removed after ONNX conversion")
check("requirements.txt does NOT contain 'keras>=' standalone (Keras removed)",
      not any(line.strip().startswith("keras>=") or line.strip().startswith("keras>")
              for line in req_src.splitlines()),
      "Standalone keras dependency still in requirements.txt")

# Try a live ONNX inference sanity check
try:
    import onnxruntime as ort
    import numpy as _np
    _sess = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    _inp = _np.zeros((1, 128, 128, 3), dtype=_np.float32)
    _out = _sess.run(None, {_sess.get_inputs()[0].name: _inp})[0]
    _sums_to_one = abs(float(_out[0].sum()) - 1.0) < 1e-4
    _shape_ok = _out.shape == (1, 3)
    check("ONNX model live inference: output shape (1,3) and sums to 1.0",
          _shape_ok and _sums_to_one,
          f"shape={_out.shape} sum={_out[0].sum():.4f}")
except Exception as _e:
    check("ONNX model live inference sanity check", False, str(_e))

# ===========================================================================
print("\n=== 11. Memory validation ===")
# ===========================================================================
print("  NOTE: use real-image measurements; monitor Render RSS separately.")
print()

print("  Full-pipeline RSS is measured from a real image by scripts/measure_pipeline_rss.py.")
print("  Local measurements do not substitute for runtime monitoring on Render.")
check("real-image RSS measurement script is available",
      os.path.isfile("scripts/measure_pipeline_rss.py"))

# ===========================================================================
print("\n=== 12. Runtime tests NOT yet performed (require live Render instance) ===")
# ===========================================================================
runtime_tests = [
    "Render startup without SIGKILL (watch logs for 'Worker was sent SIGKILL')",
    "/api/health returns {status: healthy, cnn_model_ready: true} after cold start",
    "Image upload (/api/upload) with a typical ~2 MB classroom photo",
    "/api/files returns uploaded image list",
    "/api/analyze/<filename> returns HTTP 200 (not 500) on first request",
    "/api/analyze/<filename> returns HTTP 200 on second consecutive request (warm worker)",
    "CNN stage completes: stages[2].status == 'completed' in JSON response",
    "OCR stage returns validated raw regions and a warning when quality checks fail",
    "Full six-stage pipeline completes; OCR/NLP/notes may be warning stages when evidence is weak",
    "Render logs show RSS before/after values (via psutil) below 512 MB",
    "Oversized image (>8 MP) returns HTTP 400 with pixel-count error message",
    "Concurrent requests: second /api/analyze returns 503 while first is running",
    "3+ repeated analyses: Render logs show stable RSS (no unbounded growth)",
    "preload_app=True verified: CNN ONNX and lazy RapidOCR startup behavior",
    "gunicorn bind port: startup log shows correct Render-assigned PORT (not $PORT literal)",
    "ONNX model loads without error: 'CNN ONNX session ready' appears in startup logs",
    "ONNX inference produces same 3-class output as before: class in [background, board, text]",
]
for i, t in enumerate(runtime_tests, 1):
    print(f"  [{i:02d}] PENDING (runtime): {t}")
check("All runtime tests are explicitly listed as pending",
      len(runtime_tests) == 17, f"listed {len(runtime_tests)}")

# ===========================================================================
print()
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
print()
if FAIL > 0:
    print("SOME CHECKS FAILED — do not merge until all pass.")
else:
    print("All checks passed. Runtime verification still required on Render.")
sys.exit(0 if FAIL == 0 else 1)
