"""
Configuration & structure tests for the memory/pipeline fix.
Run from project root: python scripts/test_config.py
"""
import sys, os, ast

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

PASS = 0
FAIL = 0

def check(label, cond, msg=""):
    global PASS, FAIL
    if cond:
        print(f"  PASS  {label}")
        PASS += 1
    else:
        print(f"  FAIL  {label}: {msg}")
        FAIL += 1

# ---- read source files ----
with open("app.py") as f: app_src = f.read()
with open("pipeline.py") as f: pipe_src = f.read()
with open("cnn/predict.py") as f: cnn_src = f.read()
with open("ocr/extractor.py") as f: ocr_src = f.read()
with open("preprocessing/image_processing.py") as f: pre_src = f.read()
with open("gunicorn.conf.py") as f: gcfg_src = f.read()
with open("requirements.txt") as f: req_src = f.read()
with open("Procfile") as f: proc_src = f.read()
with open("render.yaml") as f: render_src = f.read()

print("\n=== Syntax checks ===")
for path in ["app.py","pipeline.py","cnn/predict.py","ocr/extractor.py",
             "preprocessing/image_processing.py"]:
    try:
        with open(path) as f: ast.parse(f.read())
        check(f"syntax: {path}", True)
    except SyntaxError as e:
        check(f"syntax: {path}", False, str(e))

print("\n=== Import / eager-load ===")
lines = app_src.split("\n")
dynamo_line = next((i for i,l in enumerate(lines) if "TORCHDYNAMO_DISABLE" in l and "setdefault" in l), None)
top_torch_lines = [i+1 for i,l in enumerate(lines) if l.strip() == "import torch"]
check("No bare 'import torch' at app.py module level",
      top_torch_lines == [], f"found at lines {top_torch_lines}")
check("TORCHDYNAMO_DISABLE set in app.py before any torch usage",
      dynamo_line is not None, "not found")
check("TORCHDYNAMO_DISABLE set in ocr/extractor.py",
      "TORCHDYNAMO_DISABLE" in ocr_src)
check("TORCHDYNAMO_DISABLE set in cnn/predict.py",
      "TORCHDYNAMO_DISABLE" in cnn_src)

print("\n=== Memory reductions ===")
check("OCR mag_ratio=1.0 (was 1.5)",
      "mag_ratio=1.0" in ocr_src and "mag_ratio=1.5" not in ocr_src,
      "mag_ratio=1.5 still present" if "mag_ratio=1.5" in ocr_src else "mag_ratio=1.0 not found")
check("preprocessing uses MAX_DIM (not hardcoded 1600 as max_dim)",
      "MAX_DIM" in pre_src,
      "MAX_DIM constant not found")
check("old 'max_dim = 1600' removed from preprocess",
      "max_dim = 1600" not in pre_src,
      "old hardcoded 1600 still present")
check("torch.inference_mode() used in OCR extractor",
      "torch.inference_mode" in ocr_src)
check("del intermediate arrays in preprocessing",
      pre_src.count("del ") >= 3, f"only {pre_src.count('del ')} del statements found")
check("del image_array after CNN predict",
      "del image_array" in cnn_src)

print("\n=== Pipeline resilience ===")
check("gc.collect() called in pipeline.py after OCR",
      "gc.collect()" in pipe_src)
check("traceback.format_exc() used for full error logging",
      "traceback.format_exc()" in pipe_src)
check("Stage-level logging with logger.info in pipeline",
      pipe_src.count("logger.info") >= 6)
check("Memory logging _log_memory() defined in pipeline",
      "_log_memory" in pipe_src)
check("Pipeline returns partial results on CNN failure (not exception)",
      "predicted_class = \"error\"" in pipe_src)
check("Pipeline returns partial results on OCR failure (not exception)",
      "ocr_clean = []" in pipe_src)

print("\n=== Gunicorn / deployment ===")
check("gunicorn.conf.py has workers = 1",
      "workers = 1" in gcfg_src)
check("gunicorn.conf.py has preload_app = True",
      "preload_app = True" in gcfg_src)
check("gunicorn.conf.py has max_requests (memory leak guard)",
      "max_requests" in gcfg_src)
check("Procfile uses --config gunicorn.conf.py",
      "gunicorn.conf.py" in proc_src)
check("render.yaml uses --config gunicorn.conf.py",
      "gunicorn.conf.py" in render_src)
check("psutil in requirements.txt",
      "psutil" in req_src)

print("\n=== Logging ===")
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

print()
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL} checks")
sys.exit(0 if FAIL == 0 else 1)
