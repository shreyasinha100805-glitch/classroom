# AI Classroom Board Content Extractor

An end-to-end intelligent multimodal pipeline designed to ingest classroom-board images (chalkboards, whiteboards, digital boards), preprocess them, identify content via Deep CNN, extract handwritten/printed text with deep OCR, perform NLP semantic analysis, and generate organized, structured study notes.

---

## 🏗️ 6-Stage System Architecture & Processing Pipeline

The project implements the following 6 core sequential processes:

```
[ User Image / Phone Upload ]
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
│ Fast non-local means, adaptive CLAHE, unsharp masking. │
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
│ EasyOCR reads and extracts text from the board image.  │
│ Deep learning character recognition & bounding boxes.  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 5.0 NLP Analysis                                       │
│ The extracted text is analyzed to identify important   │
│ keywords and academic subject topics.                  │
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

## 🌐 Production Deployment Architecture

Because deep learning frameworks (TensorFlow + PyTorch + EasyOCR models) exceed serverless size limits (~250MB), the application utilizes a high-performance decoupled/hybrid architecture:

- **AI Backend (Render / Railway / Cloud VM)**: Full Python container with TensorFlow-CPU, PyTorch, EasyOCR, OpenCV, Gunicorn WSGI, and SQLite database.
- **Frontend (Vercel / Direct)**: Fast global CDN serving the responsive glassmorphism web client with dynamic theme switching and multi-device support.

---

### Deploying the Backend to Render (Free / Starter)

1. Create a free account at [render.com](https://render.com/).
2. Click **New +** > **Web Service**.
3. Connect your GitHub repository (`classroom`).
4. Configure the service settings:
   - **Name**: `classroom-board-ai`
   - **Environment**: `Python 3`
   - **Branch**: `main`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT`
   - **Health Check Path**: `/api/health`
5. Click **Create Web Service**.
6. Once deployed, Render will provide a live URL (e.g. `https://classroom-board-ai.onrender.com`).
   - You can access the complete web application directly from this URL on any phone or laptop!

---

### Connecting Your Vercel Frontend to Render

1. In your [Vercel Project Dashboard](https://vercel.com/):
   - Go to **Settings** > **Environment Variables**.
   - Add:
     - **Name**: `BACKEND_URL`
     - **Value**: `https://classroom-board-ai.onrender.com` (your Render backend URL)
2. Redeploy Vercel.
3. Now when visiting `https://classroom-swart-one.vercel.app/` from any phone or computer, it will automatically connect to your Render AI backend for uploads, OCR, CNN, and note generation!

---

## 💻 Local Development & Testing

```bash
# 1. Activate Virtual Environment
.\venv\Scripts\activate

# 2. Start the local production-grade server
python app.py
```
Open **http://127.0.0.1:5000/** in your browser.

### Command Line Pipeline Execution
```bash
python pipeline.py
```
Enter an image path when prompted to run all 6 stages directly in the terminal.

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | System health, model readiness, and timestamp |
| `/api/upload` | POST | Ingest one or more board images (multipart/form-data) |
| `/api/analyze/<filename>` | GET / POST | Execute full 6-stage AI pipeline on an uploaded image |
| `/api/export-notes/<filename>?format=md` | GET | Download generated study notes in Markdown or TXT |
| `/uploads/<filename>` | GET | Access raw uploaded image |
| `/pipeline_output/<filename>` | GET | Access enhanced preprocessed image |
