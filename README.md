# AI-Based Classroom Board Content Extractor (ClassMind AI)
### Subject: Neural Network & Deep Learning (PCCCS582)

[![Branch](https://img.shields.io/badge/branch-main-blue.svg)](https://github.com/shreyasinha100805-glitch/classroom/tree/main)
[![Python](https://img.shields.io/badge/Python-3.11-brightgreen.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Inference-ONNX%20Runtime%201.17+-orange.svg)](https://onnxruntime.ai/)
[![Frontend](https://img.shields.io/badge/Frontend-Vercel%20Live-black.svg)](https://classroom-tan-iota.vercel.app/)
[![Backend](https://img.shields.io/badge/Backend-Render%20Live-46E3B7.svg)](https://classroom-backend-cp6o.onrender.com/)

An end-to-end intelligent multimodal deep learning system engineered to capture classroom-board imagery (blackboards, whiteboards, digital smartboards), preprocess visual artifacts, classify board content via Deep Convolutional Neural Networks (CNN), extract printed and handwritten academic text with deep OCR, perform Natural Language Processing (NLP) semantic analysis, and automatically synthesize structured, formatted study notes.

---

## 🌐 Official Deployment Endpoints

| Component | Platform | Live URL | Status |
|---|---|---|---|
| **Production Frontend** | Vercel CDN | [https://classroom-tan-iota.vercel.app/](https://classroom-tan-iota.vercel.app/) | **Active / Production** |
| **Preview Frontend** | Vercel CDN | [https://classroom-git-fix-ocr-014b03-shreyasinha100805-glitchs-projects.vercel.app/](https://classroom-git-fix-ocr-014b03-shreyasinha100805-glitchs-projects.vercel.app/)] | **Active / Preview** |
| **AI Inference Backend** | Render Container | [https://classroom-backend-cp6o.onrender.com/](https://classroom-backend-cp6o.onrender.com/) | **Active / Production** |

> [!NOTE]
> All frontend routes transparently proxy `/api/*` and image requests to the dedicated Render backend service, avoiding cross-origin issues and browser security blocks.

---

## 🏗️ 6-Stage Multimodal AI Pipeline Architecture

The system executes 6 sequential processes to transform raw classroom imagery into clean, academic study notes:

```
                  [ Classroom Board Image Upload ]
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│ 1.0 Image Upload & Validation                                      │
│ • Header-only Pillow resolution probe (≤ 8 Megapixels limit)      │
│ • Format check (JPEG, PNG, WebP) and payload size security check   │
│ • Safe buffer decoding via NumPy (halves memory spikes)           │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│ 2.0 Computer Vision Preprocessing (OpenCV)                        │
│ • Dynamic dimension clamping (MAX_DIM = 960 px)                   │
│ • Grayscale conversion & Fast Non-Local Means / Bilateral Filter  │
│ • Contrast-Limited Adaptive Histogram Equalization (CLAHE)        │
│ • Unsharp masking & Laplacian edge sharpening                     │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│ 3.0 Deep CNN Content Classification                               │
│ • Rescaling(1/255) → 3× [Conv2D + ReLU + MaxPool2D] blocks        │
│ • Dense(128, ReLU) + Dropout(0.5) → Dense(3, Softmax)             │
│ • Identifies image content type: [background, board, text]        │
│ • CPU ONNX Runtime inference; production path does not import Keras │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│ 4.0 Deep OCR Text Extraction                                      │
│ • RapidOCR DBNet + PP-OCRv4 recognition (ONNX Runtime) │
│ • Bounding box generation with character confidence scores        │
│ • One lazy CPU ONNX OCR engine; collect intermediates after OCR │
│ • Immediate Python garbage collection (gc.collect())              │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│ 5.0 NLP Semantic Analysis & Topic Categorization                  │
│ • Tokenization, punctuation removal & academic stopword filtering │
│ • Term Frequency keyword ranking & extraction                     │
│ • Academic subject topic mapping (CS, Math, Physics, Chemistry)   │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│ 6.0 Automated Structured Notes Synthesis                          │
│ • Merges CNN classification, detected topics, keywords, and OCR   │
│ • Formats into structured headings, bullet points, and key terms  │
│ • Exports to Markdown (.md), Plain Text (.txt), and Clipboard     │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
                    [ Structured Academic Notes ]
```

---

## 🧠 Deep Learning CNN Model Specification

### 1. Model Architecture
The custom CNN classifier evaluates classroom board imagery to determine content category before downstream processing:
- **Input Tensor Contract**: `(batch_size, 128, 128, 3)` RGB float32
- **Layer Breakdown**:
  1. `Rescaling(1./255)` — Internal normalization layer
  2. `Conv2D(32, kernel_size=(3, 3), activation='relu')`
  3. `MaxPooling2D(pool_size=(2, 2))`
  4. `Conv2D(64, kernel_size=(3, 3), activation='relu')`
  5. `MaxPooling2D(pool_size=(2, 2))`
  6. `Conv2D(128, kernel_size=(3, 3), activation='relu')`
  7. `MaxPooling2D(pool_size=(2, 2))`
  8. `Flatten()` — Channels-last `(H, W, C)` feature tensor flattening (25,088 features)
  9. `Dense(128, activation='relu')`
  10. `Dropout(0.5)`
  11. `Dense(3, activation='softmax')`
- **Output Classes**: `0: background`, `1: board`, `2: text`

### 2. Keras Reference vs. Production ONNX Verification
The trained Keras reference model was exported to ONNX Runtime. The included validation script compares both models on nine selected images; this checks export consistency, not classification accuracy.

| Metric | Keras Reference Model | Production ONNX Model | Verification Result |
|---|---|---|---|
| **File Path** | `cnn/model/classroom_cnn.keras` | `cnn/model/classroom_cnn.onnx` | Both preserved |
| **File Format** | Keras v3 Native Zip/HDF5 | Single self-contained ONNX binary | Single file, no external data |
| **Model Size** | 39,710,284 bytes (37.87 MB) | 13,223,638 bytes (12.61 MB) | **66.7% size reduction** |
| **SHA-256 Checksum** | `32928cdf3ea1...bacbd2` | `65f791b21044...916bc86` | Cryptographically verified |
| **Real Test Agreement** | 9/9 classes (100%) | 9/9 classes (100%) | **100% agreement** |
| **Max Absolute Softmax Difference** | Reference | **8.94e-08** | Within the script's `1e-4` tolerance |

### 3. Selected Image Export Comparison
The following are current outputs from `scripts/validate_final_cnn.py`. Model agreement is not a measure of classification accuracy against ground-truth labels.

| Test Image File | Keras Class | ONNX Class | Class Match | Keras Confidence | ONNX Confidence | Max Absolute Diff |
|---|---|---|:---:|---|---|---|
| `sample_board.jpg` | `background` | `background` | **YES** | 75.66% | 75.66% | $4.47 \times 10^{-8}$ |
| `sample_diagram.jpg` | `text` | `text` | **YES** | 44.80% | 44.80% | $5.96 \times 10^{-8}$ |
| `sample_text.jpg` | `text` | `text` | **YES** | 91.43% | 91.43% | $2.24 \times 10^{-8}$ |
| `board_1.jpg` | `text` | `text` | **YES** | 80.37% | 80.37% | $5.96 \times 10^{-8}$ |
| `text_1.jpg` | `text` | `text` | **YES** | 91.43% | 91.43% | $2.24 \times 10^{-8}$ |
| `text_2.jpg` | `background` | `background` | **YES** | 62.48% | 62.48% | $8.94 \times 10^{-8}$ |
| `text_3.jpg` | `background` | `background` | **YES** | 51.05% | 51.05% | $5.96 \times 10^{-8}$ |
| `background_1.jpg` | `background` | `background` | **YES** | 56.55% | 56.55% | $8.94 \times 10^{-8}$ |
| `background_2.jpg` | `text` | `text` | **YES** | 71.22% | 71.22% | $2.98 \times 10^{-8}$ |

---

## 🛡️ Production Hardening & Cloud Stability

The production inference path keeps peak memory bounded and avoids loading TensorFlow or Keras:

1. **Lightweight ONNX Runtime Execution Provider**:
   - Replaced production TensorFlow inference imports with `onnxruntime` CPU inference; the Keras model remains as an untouched reference artifact.
2. **Inference Concurrency Guard (`_inference_semaphore`)**:
   - A strict `Semaphore(1)` serializes CNN/OCR inference and avoids concurrent model copies; competing requests receive a retryable HTTP 503 response.
3. **Pillow Header-Only Resolution Probe (8 MP Gate)**:
   - Evaluates JPEG/PNG dimensions from file headers (`Image.open().size`) before full OpenCV rasterization. Images exceeding 8 Megapixels are safely rejected with HTTP 400.
4. **Zero-Copy NumPy Decoding**:
   - Uses `np.frombuffer(raw_bytes, np.uint8)` instead of `bytearray()`, eliminating memory buffer duplication.
5. **Memory-Tuned Gunicorn WSGI Config**:
   - Configured with `workers = 1`, `threads = 2`, and `preload_app = True`. Gunicorn starts the single job thread in `post_fork`; request-count recycling is disabled so polling cannot recycle a worker during an active job.
6. **RapidOCR ONNX Memory Use**:
   - RapidOCR is initialized lazily as one per-process ONNX Runtime engine; the pipeline releases OCR intermediates and calls `gc.collect()` after extraction.

---

## 💻 Local Development & Setup

### Prerequisites
- Python 3.11+
- Git

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/shreyasinha100805-glitch/classroom.git
cd classroom

# Use the production branch
git checkout main

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate    # Linux / macOS

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Local Flask Server
```powershell
# Windows: run Flask with the project's installed dependencies
.\venv\Scripts\python.exe -m flask --app app run --host 127.0.0.1 --port 5000
```

```bash
# Linux / macOS
venv/bin/python -m flask --app app run --host 127.0.0.1 --port 5000
```

Use the same virtual environment where `requirements.txt` was installed. Running Flask with a different system Python can make upload succeed but fail at stage 2 when OpenCV is unavailable. The health response includes `preprocessing_configured` to help identify that setup issue.

Open **http://127.0.0.1:5000/** in your browser.

### 3. CLI End-to-End Pipeline Execution
```bash
python pipeline.py
```
Enter a file path (e.g. `public/samples/sample_text.jpg`) to execute all 6 stages sequentially in the terminal.

### 4. Run Academic Validation Suites
```bash
# Run CNN numerical equivalence & robustness suite
python scripts/validate_final_cnn.py

# Run configuration, environment, and memory audit suite
python scripts/test_config.py

# Run backend unit & integration tests
python test_phase1.py
```

---

## 📡 REST API Reference

| Endpoint | Method | Payload / Params | Description |
|---|---|---|---|
| `/api/health` | `GET` | None | Returns system health, CNN model readiness status, and timestamp |
| `/api/upload` | `POST` | `multipart/form-data` (`images`) | Ingests classroom-board images with dimension and format validation |
| `/api/files` | `GET` | None | Retrieves list of all uploaded images available for processing |
| `/api/analyze/jobs` | `POST` | JSON: `{"filename":"<saved_name>"}` | Queues the six-stage pipeline and returns HTTP 202 with a job ID |
| `/api/analyze/jobs/<job_id>` | `GET` | Job ID in URL | Returns queued/running progress, the completed pipeline result, or a safe failure message |
| `/api/analyze/<filename>` | `GET` / `POST` | `<filename>` in URL | Backward-compatible synchronous six-stage analysis endpoint |
| `/api/export-notes/<filename>` | `GET` | `?format=md` or `?format=txt` | Downloads notes from the latest completed analysis for that image; returns HTTP 409 until analysis completes |
| `/uploads/<filename>` | `GET` | `<filename>` in URL | Serves the original raw uploaded image |
| `/pipeline_output/<filename>`| `GET` | `<filename>` in URL | Serves the OpenCV preprocessed and enhanced board image |

The browser uses the asynchronous job endpoints so long OCR runs do not hold a single proxied request open. Jobs run through one in-process worker and share the inference semaphore with the legacy synchronous endpoint. Under Gunicorn, the worker thread starts in the child process through `post_fork`; app preloading remains enabled and Gunicorn is configured for one worker so job creation and polling use the same manager. Job metadata and results are held in memory; the SQLite database is used for accounts, and no persistent job datastore is configured. A Render restart, deploy, or abrupt process exit can interrupt active jobs and clears their status/results. Because uploads and job state are process-local, this setup requires a single Render instance; multiple replicas need shared storage and queue state. The status endpoint returns HTTP 404 for unknown, expired, or lost job IDs. Upload cleanup protects queued and running job images.

The browser stores the active job ID per image in `sessionStorage`. After refreshing the same tab, the result page restores polling for the matching uploaded image instead of submitting another job. Closing the tab clears that recovery hint. A Render restart can still lose the in-memory job and its result; in that case the old ID returns HTTP 404 and the user can retry while the upload remains available.

### Vercel preview configuration

`vercel.mjs` is Vercel's programmatic configuration file. It keeps browser requests on relative `/api/...` URLs and rewrites them to Render. Production uses the fixed origin `https://classroom-backend-cp6o.onrender.com`. Preview and Development require `CLASSMIND_PREVIEW_BACKEND_URL`; missing or invalid values stop configuration instead of silently using production. Use the actual separate Preview Render service origin, with HTTPS and no credentials, port, path, query string, or fragment.

To configure Preview without changing Production:

1. In Render, choose **New → Web Service**, select this repository and a non-production branch, and create a separate Preview service. Use Python 3.11, the build command `pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt`, the start command `gunicorn app:app --config gunicorn.conf.py`, and health check path `/api/health`. Keep one instance. Set `TRUSTED_ORIGINS` to the exact Vercel Preview frontend origin(s); generate a Preview-only `SECRET_KEY` and copy the non-secret CPU and image-size environment values from `render.yaml`.
2. After Render provides the Preview service URL, open the Vercel **classroom** project → **Settings → Environment Variables**. Add `CLASSMIND_PREVIEW_BACKEND_URL` with that exact HTTPS origin and select **Preview** and **Development** only. Leave the Production value and environment untouched.
3. Save the variable and redeploy the Preview deployment. Do not use the production Render URL for Preview.

Install dependencies with `python -m pip install -r requirements.txt`. Verify static assets with `node build.js` followed by `npm test`; validate rewrites with `node scripts/test_vercel_config.mjs`. Run the job tests with `python -m unittest test_analysis_jobs -v`, Linux Gunicorn lifecycle smoke test with `python scripts/test_gunicorn_lifecycle.py`, and configuration checks with `python scripts/test_config.py`. Run the real local async upload-to-result check with `python scripts/test_async_api.py`. Measure standalone pipeline RSS with `python scripts/measure_pipeline_rss.py samples/sample_board.jpg`; the script writes its processed preview to a unique temporary directory. To test the UI, start Flask with `python -m flask --app app run --host 127.0.0.1 --port 5000`, upload an image, and confirm the browser sends `POST /api/analyze/jobs` followed by short `GET /api/analyze/jobs/<job_id>` polls.

---

## 👥 Academic Submission Details

- **Course**: Neural Network & Deep Learning (PCCCS582)
- **Repository**: [https://github.com/shreyasinha100805-glitch/classroom](https://github.com/shreyasinha100805-glitch/classroom)
- **Primary Branch**: `main`
- **Target Deployed System**: Vercel (Frontend) + Render (AI Inference Backend)
