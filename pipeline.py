import os
import gc
import logging
import tempfile
import traceback
from PIL import Image

logger = logging.getLogger(__name__)

PIPELINE_OUTPUT_FOLDER = os.environ.get(
    "PIPELINE_OUTPUT_FOLDER",
    os.path.join(tempfile.gettempdir() if os.environ.get("VERCEL") else ".", "pipeline_output")
)


# ==============================================================================
# PIPELINE STAGE SPECIFICATION (6 CORE SYSTEM PROCESSES)
# ==============================================================================
PIPELINE_PROCESSES = [
    {
        "step": "1.0",
        "title": "Image Upload & Validation",
        "description": "User uploads the classroom-board image.",
        "icon": "upload"
    },
    {
        "step": "2.0",
        "title": "Image Preprocessing",
        "description": "The image is resized, denoised, enhanced and sharpened.",
        "icon": "filter"
    },
    {
        "step": "3.0",
        "title": "CNN Classification",
        "description": "The CNN identifies the type of image/content such as board, text or background.",
        "icon": "cpu"
    },
    {
        "step": "4.0",
        "title": "OCR Text Extraction",
        "description": "OCR reads and extracts text from the classroom image.",
        "icon": "type"
    },
    {
        "step": "5.0",
        "title": "NLP Analysis",
        "description": "The extracted text is analyzed to identify important keywords and topics.",
        "icon": "brain"
    },
    {
        "step": "6.0",
        "title": "Structured Notes Generation",
        "description": "The system combines the extracted information and generates organized notes for the user.",
        "icon": "file-text"
    }
]


def _log_memory(label: str):
    """Log current process RSS memory if psutil is available."""
    try:
        import psutil, os as _os
        proc = psutil.Process(_os.getpid())
        rss_mb = proc.memory_info().rss / 1024 / 1024
        logger.info("[MEM] %s: RSS=%.1f MB", label, rss_mb)
    except ImportError:
        pass  # psutil not installed — skip silently


# ==============================================================================
# 1.0 IMAGE UPLOAD & VALIDATION
# ==============================================================================
def validate_image(image_path: str) -> dict:
    """
    1.0 Image Upload & Validation:
    User uploads the classroom-board image. Validates existence, file readability,
    dimensions, and color channels. Rejects images exceeding MAX_UPLOAD_PIXELS
    to prevent decompression-bomb memory spikes during preprocessing.
    """
    if not os.path.exists(image_path):
        return {"valid": False, "error": f"Image file not found: {image_path}"}

    file_size_bytes = os.path.getsize(image_path)
    if file_size_bytes == 0:
        return {"valid": False, "error": "Image file is empty (0 bytes)."}

    try:
        with Image.open(image_path) as img:
            img.verify()

        # Reopen to read dimensions safely
        with Image.open(image_path) as img:
            width, height = img.size
            format_name = img.format or "UNKNOWN"

        # Reject images that would cause excessive memory during preprocessing.
        # 8 MP (e.g. 3264×2448) is plenty for classroom board OCR.
        max_pixels = int(os.environ.get("MAX_UPLOAD_PIXELS", str(8_000_000)))
        pixel_count = width * height
        if pixel_count > max_pixels:
            return {
                "valid": False,
                "error": (
                    f"Image is too large ({width}x{height} = {pixel_count:,} pixels). "
                    f"Maximum allowed is {max_pixels:,} pixels (~8 MP). "
                    "Please resize your image before uploading."
                )
            }

        return {
            "valid": True,
            "filename": os.path.basename(image_path),
            "size_bytes": file_size_bytes,
            "width": width,
            "height": height,
            "format": format_name
        }
    except Exception as exc:
        return {"valid": False, "error": f"Invalid or corrupted image format: {str(exc)}"}


# ==============================================================================
# FULL CLI EXECUTION PIPELINE
# ==============================================================================
def run_pipeline(image_path: str):
    from preprocessing.image_processing import preprocess_image
    from cnn.predict import predict_image
    from ocr.extractor import extract_text
    from nlp.keywords import extract_keywords
    from nlp.topics import detect_topics
    from nlp.notes import generate_structured_notes

    print()
    print("==================================================================")
    print("       AI CLASSROOM BOARD CONTENT EXTRACTOR PIPELINE")
    print("==================================================================")
    print()

    # -------------------------------------------------------------------------
    # 1.0 Image Upload & Validation
    # -------------------------------------------------------------------------
    print("1.0 Image Upload & Validation")
    print("User uploads the classroom-board image.")
    print("------------------------------------------------------------------")
    val_result = validate_image(image_path)
    if not val_result["valid"]:
        print(f"ERROR: {val_result['error']}")
        return

    print(f"[OK] Validated: {val_result['filename']} ({val_result['width']}x{val_result['height']}, {val_result['format']}, {val_result['size_bytes']} bytes)")
    print()

    # -------------------------------------------------------------------------
    # 2.0 Image Preprocessing
    # -------------------------------------------------------------------------
    print("2.0 Image Preprocessing")
    print("The image is resized, denoised, enhanced and sharpened.")
    print("------------------------------------------------------------------")
    os.makedirs("pipeline_output", exist_ok=True)
    processed_path = os.path.join("pipeline_output", "processed.jpg")

    try:
        preprocess_image(image_path, processed_path)
        print("[OK] Preprocessed output saved to:", processed_path)
    except Exception as error:
        print("ERROR in Preprocessing:", error)
        return
    print()

    # -------------------------------------------------------------------------
    # 3.0 CNN Classification
    # -------------------------------------------------------------------------
    print("3.0 CNN Classification")
    print("The CNN identifies the type of image/content such as board, text or background.")
    print("------------------------------------------------------------------")
    cnn_result = predict_image(processed_path)
    predicted_class = cnn_result["class"]
    confidence = cnn_result["confidence"]

    print(f"[OK] Predicted Content Type: {predicted_class.upper()}")
    print(f"[OK] Confidence: {confidence * 100:.2f}%")
    print()

    # -------------------------------------------------------------------------
    # 4.0 OCR Text Extraction
    # -------------------------------------------------------------------------
    print("4.0 OCR Text Extraction")
    print("OCR reads and extracts text from the classroom image.")
    print("------------------------------------------------------------------")
    ocr_results = extract_text(processed_path)
    extracted_text = []

    if not ocr_results:
        print("Notice: No readable text detected with sufficient confidence.")
    else:
        for item in ocr_results:
            text = item["text"]
            score = item["confidence"]
            print(f"  - {text} (conf: {score:.2f})")
            if score >= 0.40:
                extracted_text.append(text)

    combined_text = " ".join(extracted_text)
    print()

    # -------------------------------------------------------------------------
    # 5.0 NLP Analysis
    # -------------------------------------------------------------------------
    print("5.0 NLP Analysis")
    print("The extracted text is analyzed to identify important keywords and topics.")
    print("------------------------------------------------------------------")
    keywords = extract_keywords(combined_text, max_keywords=10) if combined_text else []
    topics = detect_topics(combined_text) if combined_text else []

    print("Keywords:")
    if keywords:
        for kw in keywords:
            print(f"  - {kw}")
    else:
        print("  None detected.")

    print("Topics:")
    if topics:
        for item in topics:
            print(f"  - {item['topic']} (matches: {item['matches']}) -> {', '.join(item['keywords'])}")
    else:
        print("  None detected.")
    print()

    # -------------------------------------------------------------------------
    # 6.0 Structured Notes Generation
    # -------------------------------------------------------------------------
    print("6.0 Structured Notes Generation")
    print("The system combines the extracted information and generates organized notes for the user.")
    print("------------------------------------------------------------------")
    notes = generate_structured_notes(
        classification={"class": predicted_class, "confidence": round(confidence * 100, 2)},
        extracted_text=combined_text,
        keywords=keywords,
        topics=topics,
        ocr_items=ocr_results
    )

    print()
    print("==================================================================")
    print(notes["title"].upper())
    print("==================================================================")
    print(notes["plain_text"])
    print("==================================================================")
    print()


# ==============================================================================
# STRUCTURED DATA RETURN FOR FLASK API & CLIENT DASHBOARDS
# ==============================================================================
def run_pipeline_data(image_path: str) -> dict:
    """
    Executes the complete 6-stage pipeline and returns structured data
    for web clients, API consumers, and UI visualisation.

    Each stage is wrapped individually so a failure in one stage (CNN, OCR,
    NLP) degrades gracefully — the pipeline continues and returns partial
    results rather than raising an unhandled exception that would cause a 500.

    Memory is explicitly released between heavy stages (preprocessing →
    CNN → OCR → NLP) and gc.collect() is called once after OCR to return
    PyTorch tensor memory to the OS before NLP runs.
    """
    stages_log = []

    # -------------------------------------------------------------------------
    # 1.0 Image Upload & Validation
    # -------------------------------------------------------------------------
    logger.info("[Pipeline 1.0] Validating image: %s", image_path)
    _log_memory("before-validation")
    val = validate_image(image_path)
    if not val["valid"]:
        logger.error("[Pipeline 1.0] Validation failed: %s", val.get("error"))
        return {
            "success": False,
            "error": val.get("error", "Validation failed"),
            "failed_step": "1.0 Image Upload & Validation"
        }

    logger.info("[Pipeline 1.0] OK — %s (%dx%d, %s)",
                val["filename"], val["width"], val["height"], val["format"])
    stages_log.append({
        "step": "1.0",
        "title": "Image Upload & Validation",
        "description": "User uploads the classroom-board image.",
        "status": "completed",
        "details": f"Validated {val['filename']} ({val['width']}x{val['height']}, {val['format']})"
    })

    # -------------------------------------------------------------------------
    # 2.0 Image Preprocessing
    # -------------------------------------------------------------------------
    os.makedirs(PIPELINE_OUTPUT_FOLDER, exist_ok=True)
    base_name = os.path.basename(image_path)
    processed_path = os.path.join(PIPELINE_OUTPUT_FOLDER, f"proc_{base_name}")

    logger.info("[Pipeline 2.0] Preprocessing image → %s", processed_path)
    _log_memory("before-preprocess")
    try:
        from preprocessing.image_processing import preprocess_image
        preprocess_image(image_path, processed_path)
        logger.info("[Pipeline 2.0] OK — saved %s", os.path.basename(processed_path))
        stages_log.append({
            "step": "2.0",
            "title": "Image Preprocessing",
            "description": "The image is resized, denoised, enhanced and sharpened.",
            "status": "completed",
            "details": f"Saved enhanced image to {os.path.basename(processed_path)}"
        })
    except Exception as exc:
        logger.error("[Pipeline 2.0] FAILED:\n%s", traceback.format_exc())
        return {
            "success": False,
            "error": f"Preprocessing failed: {str(exc)}",
            "failed_step": "2.0 Image Preprocessing"
        }

    # -------------------------------------------------------------------------
    # 3.0 CNN Classification
    # -------------------------------------------------------------------------
    cnn_probabilities = {}
    logger.info("[Pipeline 3.0] Running CNN classification")
    _log_memory("before-cnn")
    try:
        from cnn.predict import predict_image
        cnn_result = predict_image(processed_path)
        predicted_class = cnn_result.get("class", "unknown")
        confidence = float(cnn_result.get("confidence", 0.0))
        raw_probs = cnn_result.get("probabilities", {})
        cnn_probabilities = {k: round(v * 100, 2) for k, v in raw_probs.items()}
        logger.info("[Pipeline 3.0] OK — class=%s confidence=%.1f%%",
                    predicted_class, confidence * 100)
        stages_log.append({
            "step": "3.0",
            "title": "CNN Classification",
            "description": "The CNN identifies the type of image/content such as board, text or background.",
            "status": "completed",
            "details": f"Identified as {predicted_class.upper()} with {round(confidence * 100, 1)}% confidence"
        })
    except Exception as exc:
        logger.error("[Pipeline 3.0] FAILED:\n%s", traceback.format_exc())
        predicted_class = "error"
        confidence = 0.0
        cnn_probabilities = {"board": 0.0, "text": 0.0, "background": 0.0}
        stages_log.append({
            "step": "3.0",
            "title": "CNN Classification",
            "description": "The CNN identifies the type of image/content such as board, text or background.",
            "status": "warning",
            "details": f"CNN classification skipped/fallback: {str(exc)}"
        })
    _log_memory("after-cnn")

    # -------------------------------------------------------------------------
    # 4.0 OCR Text Extraction
    # -------------------------------------------------------------------------
    logger.info("[Pipeline 4.0] Running OCR extraction")
    _log_memory("before-ocr")
    try:
        from ocr.extractor import extract_text
        ocr_results = extract_text(processed_path)
        extracted_text = []
        ocr_clean = []
        if ocr_results:
            for item in ocr_results:
                txt = item.get("text", "")
                conf = float(item.get("confidence", 0.0))
                ocr_clean.append({"text": txt, "confidence": round(conf, 2)})
                if conf >= 0.40:
                    extracted_text.append(txt)
        combined_text = " ".join(extracted_text)
        logger.info("[Pipeline 4.0] OK — %d segments, %d chars",
                    len(ocr_clean), len(combined_text))
        stages_log.append({
            "step": "4.0",
            "title": "OCR Text Extraction",
            "description": "OCR reads and extracts text from the classroom image.",
            "status": "completed",
            "details": f"Extracted {len(ocr_clean)} text segments ({len(combined_text)} characters)"
        })
    except Exception as exc:
        logger.error("[Pipeline 4.0] FAILED:\n%s", traceback.format_exc())
        ocr_clean = []
        combined_text = ""
        stages_log.append({
            "step": "4.0",
            "title": "OCR Text Extraction",
            "description": "OCR reads and extracts text from the classroom image.",
            "status": "warning",
            "details": f"OCR extraction issue: {str(exc)}"
        })

    # Explicit GC after OCR — releases PyTorch intermediate tensors back to OS
    _log_memory("before-gc")
    gc.collect()
    _log_memory("after-gc")

    # -------------------------------------------------------------------------
    # 5.0 NLP Analysis
    # -------------------------------------------------------------------------
    logger.info("[Pipeline 5.0] Running NLP analysis")
    try:
        from nlp.keywords import extract_keywords
        from nlp.topics import detect_topics
        keywords = extract_keywords(combined_text, max_keywords=10) if combined_text else []
        topics = detect_topics(combined_text) if combined_text else []
        logger.info("[Pipeline 5.0] OK — %d keywords, %d topics",
                    len(keywords), len(topics))
        stages_log.append({
            "step": "5.0",
            "title": "NLP Analysis",
            "description": "The extracted text is analyzed to identify important keywords and topics.",
            "status": "completed",
            "details": f"Identified {len(keywords)} keywords and {len(topics)} academic topics"
        })
    except Exception as exc:
        logger.error("[Pipeline 5.0] FAILED:\n%s", traceback.format_exc())
        keywords = []
        topics = []
        stages_log.append({
            "step": "5.0",
            "title": "NLP Analysis",
            "description": "The extracted text is analyzed to identify important keywords and topics.",
            "status": "warning",
            "details": f"NLP analysis fallback: {str(exc)}"
        })

    # -------------------------------------------------------------------------
    # 6.0 Structured Notes Generation
    # -------------------------------------------------------------------------
    logger.info("[Pipeline 6.0] Generating structured notes")
    try:
        from nlp.notes import generate_structured_notes
        structured_notes = generate_structured_notes(
            classification={"class": predicted_class, "confidence": round(confidence * 100, 2)},
            extracted_text=combined_text,
            keywords=keywords,
            topics=topics,
            ocr_items=ocr_clean
        )
        logger.info("[Pipeline 6.0] OK — '%s'", structured_notes.get("title", ""))
        stages_log.append({
            "step": "6.0",
            "title": "Structured Notes Generation",
            "description": "The system combines the extracted information and generates organized notes for the user.",
            "status": "completed",
            "details": f"Generated organized notes: '{structured_notes['title']}'"
        })
    except Exception as exc:
        logger.error("[Pipeline 6.0] FAILED:\n%s", traceback.format_exc())
        structured_notes = {
            "title": "Classroom Board Notes",
            "markdown": combined_text,
            "plain_text": combined_text,
            "key_points": [combined_text] if combined_text else [],
            "topics": topics,
            "keywords": keywords
        }
        stages_log.append({
            "step": "6.0",
            "title": "Structured Notes Generation",
            "description": "The system combines the extracted information and generates organized notes for the user.",
            "status": "warning",
            "details": f"Structured notes generated with fallback: {str(exc)}"
        })

    _log_memory("pipeline-complete")
    return {
        "success": True,
        "processed_image": processed_path,
        "processed_filename": os.path.basename(processed_path),
        "stages": stages_log,
        "pipeline_spec": PIPELINE_PROCESSES,
        "classification": {
            "class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "probabilities": cnn_probabilities
        },
        "extracted_text": combined_text,
        "ocr_items": ocr_clean,
        "keywords": keywords,
        "topics": topics,
        "structured_notes": structured_notes
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    image_path = sys.argv[1].strip() if len(sys.argv) > 1 else input("Enter image path: ").strip()
    run_pipeline(image_path)
