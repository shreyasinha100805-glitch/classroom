# AI-Based Classroom Board Content Extractor (ClassMind AI)
### Subject: Neural Network & Deep Learning (PCCCS582)

[![Branch](https://img.shields.io/badge/branch-fix%2Frender--memory--pipeline-blue.svg)](https://github.com/shreyasinha100805-glitch/classroom/tree/fix/render-memory-pipeline)
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
| **Alternative Frontend** | Vercel CDN | [https://classroom-theta-silk.vercel.app/](https://classroom-theta-silk.vercel.app/) | **Active / Mirror** |
| **AI Inference Backend** | Render Container | [https://classroom-backend-cp6o.onrender.com/](https://classroom-backend-cp6o.onrender.com/) | **Active / Production** |
| *Legacy Preview Alias* | Vercel | `classroom-swart-one.vercel.app` | *Deprecated / 404 (Use active link above)* |

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
│ • Ultra-low memory ONNX Runtime engine (71 MB RSS, 8.5ms latency) │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│ 4.0 Deep OCR Text Extraction                                      │
│ • PyTorch CRAFT text detection + CRNN text recognition (EasyOCR)  │
│ • Bounding box generation with character confidence scores        │
│ • Memory-gated inference (mag_ratio=1.0, torch.inference_mode())  │
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
To deploy within Render's free tier (512 MB memory limit), the trained Keras model was exported to ONNX Runtime while preserving exact numerical equivalence.

| Metric | Keras Reference Model | Production ONNX Model | Verification Result |
|---|---|---|---|
| **File Path** | `cnn/model/classroom_cnn.keras` | `cnn/model/classroom_cnn.onnx` | Both preserved |
| **File Format** | Keras v3 Native Zip/HDF5 | Single self-contained ONNX binary | Single file, no external data |
| **Model Size** | 39,710,284 bytes (37.87 MB) | 13,223,638 bytes (12.61 MB) | **66.7% size reduction** |
| **SHA-256 Checksum** | `32928cdf3ea1...bacbd2` | `65f791b21044...916bc86` | Cryptographically verified |
| **Process RSS Memory** | **407.73 MB** (TensorFlow) | **71.00 MB** (ONNX Runtime) | **336.73 MB RAM saved** |
| **Inference Latency** | ~1700 ms (CPU cold) | **8.54 ms** (CPU) | **~200× faster** |
| **Real Test Agreement** | 9/9 classes (100%) | 9/9 classes (100%) | **100% agreement** |
| **Max Absolute Softmax Diff** | Reference | **8.94e-08** | **Mathematically identical** |

### 3. Real Test Image Benchmark Results
Tested across 9 representative classroom-board test images from `public/samples/` and `dataset/test/`:

| Test Image File | Keras Class | ONNX Class | Class Match | Keras Confidence | ONNX Confidence | Max Absolute Diff |
|---|---|---|:---:|---|---|---|
| `sample_board.jpg` | `background` | `background` | **YES** | 75.66% | 75.66% | $4.47 \times 10^{-8}$ |
| `sample_diagram.jpg` | `text` | `text` | **YES** | 44.80% | 44.80% | $5.96 \times 10^{-8}$ |
| `sample_text.jpg` | `text` | `text` | **YES** | 91.43% | 91.43% | $5.96 \times 10^{-8}$ |
| `board_1.jpg` | `text` | `text` | **YES** | 80.37% | 80.37% | $5.96 \times 10^{-8}$ |
| `text_1.jpg` | `text` | `text` | **YES** | 91.43% | 91.43% | $5.96 \times 10^{-8}$ |
| `text_2.jpg` | `background` | `background` | **YES** | 62.48% | 62.48% | $5.96 \times 10^{-8}$ |
| `text_3.jpg` | `background` | `background` | **YES** | 51.05% | 51.05% | $5.96 \times 10^{-8}$ |
| `background_1.jpg` | `background` | `background` | **YES** | 56.55% | 56.55% | $8.94 \times 10^{-8}$ |
| `background_2.jpg` | `text` | `text` | **YES** | 71.22% | 71.22% | $2.98 \times 10^{-8}$ |

---

## 🛡️ Production Hardening & Cloud Stability (512 MB Budget)

To prevent Linux Out-Of-Memory (OOM) `SIGKILL` termination on Render's 512 MB free tier:

1. **Lightweight ONNX Runtime Execution Provider**:
   - Replaced heavy TensorFlow runtime imports with `onnxruntime` CPU engine, freeing 336+ MB RSS.
2. **Inference Concurrency Guard (`_inference_semaphore`)**:
   - PyTorch CRAFT OCR is not thread-safe under concurrent Gunicorn threads. A strict `Semaphore(1)` ensures requests are serialized; concurrent spikes return clean HTTP 503 retry responses instead of crashing the process.
3. **Pillow Header-Only Resolution Probe (8 MP Gate)**:
   - Evaluates JPEG/PNG dimensions from file headers (`Image.open().size`) before full OpenCV rasterization. Images exceeding 8 Megapixels are safely rejected with HTTP 400.
4. **Zero-Copy NumPy Decoding**:
   - Uses `np.frombuffer(raw_bytes, np.uint8)` instead of `bytearray()`, eliminating memory buffer duplication.
5. **Memory-Tuned Gunicorn WSGI Config**:
   - Configured with `workers = 1`, `threads = 2`, `preload_app = True`, and `max_requests = 50` to recycle workers before fragmentation accumulates.
6. **EasyOCR Memory Reduction**:
   - Configured with `mag_ratio = 1.0` (saving ~173 MB vs. default 1.5) and wrapped in `torch.inference_mode()` with immediate `gc.collect()`.

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

# Switch to the stabilized branch
git checkout fix/render-memory-pipeline

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate    # Linux / macOS

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Local Flask Server
```bash
python app.py
```
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
| `/api/upload` | `POST` | `multipart/form-data` (`image`) | Ingests classroom-board images with dimension and format validation |
| `/api/files` | `GET` | None | Retrieves list of all uploaded images available for processing |
| `/api/analyze/<filename>` | `GET` / `POST` | `<filename>` in URL | Executes full 6-stage AI pipeline on uploaded image |
| `/api/export-notes/<filename>` | `GET` | `?format=md` or `?format=txt` | Generates and downloads structured academic study notes |
| `/uploads/<filename>` | `GET` | `<filename>` in URL | Serves the original raw uploaded image |
| `/pipeline_output/<filename>`| `GET` | `<filename>` in URL | Serves the OpenCV preprocessed and enhanced board image |

---

## 👥 Academic Submission Details

- **Course**: Neural Network & Deep Learning (PCCCS582)
- **Repository**: [https://github.com/shreyasinha100805-glitch/classroom](https://github.com/shreyasinha100805-glitch/classroom)
- **Primary Branch**: `fix/render-memory-pipeline`
- **Target Deployed System**: Vercel (Frontend) + Render (AI Inference Backend)
