# AI Classroom Board Content Extractor

An end-to-end intelligent multimodal pipeline designed to ingest classroom-board images (chalkboards, whiteboards, digital boards), preprocess them, identify content via Deep CNN, extract handwritten/printed text with deep OCR, perform NLP semantic analysis, and generate organized, structured study notes.

---

## 🏗️ 6-Stage System Architecture & Processing Pipeline

The project implements the following 6 core sequential processes:

```
[ User Image ]
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ 1.0 Image Upload & Validation                          │
│ User uploads the classroom-board image.                │
│ Format validation, resolution check, file integrity.   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2.0 Image Preprocessing                                │
│ The image is resized, denoised, enhanced & sharpened.  │
│ Bilateral denoising, CLAHE contrast, Gaussian unsharp. │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3.0 CNN Classification                                 │
│ The CNN identifies the type of image/content such as   │
│ board, text or background.                             │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4.0 OCR Text Extraction                                │
│ OCR reads and extracts text from the classroom image.  │
│ Deep learning character recognition & bounding boxes.  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 5.0 NLP Analysis                                       │
│ The extracted text is analyzed to identify important   │
│ keywords and topics.                                   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 6.0 Structured Notes Generation                        │
│ The system combines the extracted information and      │
│ generates organized notes for the user.                │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
             [ Structured Study Notes ]
             (.MD, .TXT, Copy Clipboard)
```

---

## 📋 Process Details

### 1. 1.0 Image Upload & Validation
- **Action**: User uploads the classroom-board image.
- **Implementation**: Multi-batch upload dropzone supporting PNG, JPG, and JPEG. Files undergo PIL/OpenCV image header verification, resolution measurement, and integrity checks before pipeline entry.

### 2. 2.0 Image Preprocessing
- **Action**: The image is resized, denoised, enhanced and sharpened.
- **Implementation**:
  - Image scaling to optimal recognition width.
  - Grayscale conversion.
  - Non-local means denoising (`fastNlMeansDenoising`) to remove chalk/marker glare and dust noise.
  - Contrast Limited Adaptive Histogram Equalization (`CLAHE`) to heighten text contrast.
  - Gaussian blur unsharp masking for crisp text edges.

### 3. 3.0 CNN Classification
- **Action**: The CNN identifies the type of image/content such as board, text or background.
- **Implementation**: Deep Convolutional Neural Network trained on classroom captures (`classroom_cnn.keras`), providing categorical classification (`board`, `text`, `background`) and softmax confidence scores.

### 4. 4.0 OCR Text Extraction
- **Action**: OCR reads and extracts text from the classroom image.
- **Implementation**: Deep OCR engine detecting small and irregular handwriting as well as printed characters, filtering by confidence thresholds and consolidating text snippets into cohesive transcripts.

### 5. 5.0 NLP Analysis
- **Action**: The extracted text is analyzed to identify important keywords and topics.
- **Implementation**:
  - Tokenization, stopword removal, and frequency weighting to discover key academic terminology.
  - Topic matching engine mapping content to subject categories (Artificial Intelligence, Mathematics, Physics, Computer Networks, Electronics, Database Systems, etc.).

### 6. 6.0 Structured Notes Generation
- **Action**: The system combines the extracted information and generates organized notes for the user.
- **Implementation**: Combines classification, academic topic taxonomy, key vocabulary, and segmented lecture points into clean, student-friendly structured notes exportable in Markdown (.md), Plaintext (.txt), or directly copyable to clipboard.

---

## 🚀 Running the Project

### 1. Web Application (Flask Studio)
```bash
# Activate Virtual Environment
.\venv\Scripts\activate

# Start Flask server
python app.py
```
Open your browser at `http://127.0.0.1:5000/`.

### 2. Command Line Pipeline Execution
```bash
python pipeline.py
```
Prompt will ask for an image path and execute stages 1.0 through 6.0 with formatted terminal outputs.

---

## 📁 Project Directory Structure
```
NNLDL PROJECT/
├── app.py                      # Flask Application & Web Endpoints (Upload, API, Notes Export)
├── pipeline.py                 # Core 6-Stage Execution Pipeline
├── preprocessing/
│   └── image_processing.py    # Step 2.0: Denoise, CLAHE, Sharpen
├── cnn/
│   ├── predict.py             # Step 3.0: CNN Classification
│   └── model/                 # Trained Keras CNN Model
├── ocr/
│   └── extractor.py           # Step 4.0: Deep OCR Text Extraction
├── nlp/
│   ├── keywords.py            # Step 5.0: Key Vocabulary Extraction
│   ├── topics.py              # Step 5.0: Academic Topic Detection
│   └── notes.py               # Step 6.0: Structured Notes Synthesis
├── templates/
│   ├── index.html             # Landing Page with 6-Stage Architecture Showcase
│   └── result.html            # Results Gallery & 6-Stage Analysis Drawer
├── static/
│   ├── css/style.css          # Futuristic Neural Glassmorphism Stylesheet
│   └── js/script.js           # Client-Side Pipeline Orchestrator & Live Drawer
└── uploads/                   # Staged Classroom Board Images
```
