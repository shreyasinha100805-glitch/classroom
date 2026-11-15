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
        "description": "OCR checks the original image and at most one gentle color-enhanced variant.",
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
    print("OCR reads the original image before using any optional color enhancement.")
    print("------------------------------------------------------------------")
    ocr_results = extract_text(image_path)

    if not ocr_results:
        print("Notice: No readable text detected with sufficient confidence.")
    else:
        for item in ocr_results:
            text = item["text"]
            score = item["confidence"]
            print(f"  - {text} (conf: {score:.2f})")
    from nlp.quality import assess_ocr_quality, low_confidence_notes
    ocr_quality = assess_ocr_quality(ocr_results or [])
    if ocr_quality["warning"]:
        print(f"Warning: {ocr_quality['warning']}")
    combined_text = " ".join(
        item["text"] for item in (ocr_results or []) if item["confidence"] >= 0.40
    )
    analysis_text = ocr_quality["analysis_text"] if ocr_quality["reliable"] else ""
    print()

    # -------------------------------------------------------------------------
    # 5.0 NLP Analysis
    # -------------------------------------------------------------------------
    print("5.0 NLP Analysis")
    print("The extracted text is analyzed to identify important keywords and topics.")
    print("------------------------------------------------------------------")
    keywords = extract_keywords(analysis_text, max_keywords=10) if analysis_text else []
    topics = detect_topics(analysis_text) if analysis_text else []

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
    notes = (
        generate_structured_notes(
            classification={"class": predicted_class, "confidence": round(confidence * 100, 2)},
            extracted_text=analysis_text,
            keywords=keywords,
            topics=topics,
            ocr_items=ocr_quality["analysis_items"],
        )
        if ocr_quality["reliable"]
        else low_confidence_notes(ocr_quality["warning"])
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
def _report_stage(progress_callback, stage_title: str):
    """Report UI progress without allowing telemetry failures to break inference."""
    if progress_callback is None:
        return
    try:
        progress_callback(stage_title)
    except Exception:
        logger.exception("[Pipeline] Stage progress callback failed")


def run_pipeline_data(image_path: str, progress_callback=None) -> dict:
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
    _report_stage(progress_callback, "Image Upload & Validation")
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
    _log_memory("after-validation")
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
    _report_stage(progress_callback, "Image Preprocessing")
    os.makedirs(PIPELINE_OUTPUT_FOLDER, exist_ok=True)
    base_name = os.path.basename(image_path)
    processed_path = os.path.join(PIPELINE_OUTPUT_FOLDER, f"proc_{base_name}")

    logger.info("[Pipeline 2.0] Preprocessing image → %s", processed_path)
    _log_memory("before-preprocess")
    try:
        from preprocessing.image_processing import preprocess_image
        preprocess_image(image_path, processed_path)
        logger.info("[Pipeline 2.0] OK — saved %s", os.path.basename(processed_path))
        _log_memory("after-preprocess")
        stages_log.append({
            "step": "2.0",
            "title": "Image Preprocessing",
            "description": "A bounded image copy is prepared for classification; OCR also evaluates the original image.",
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
    _report_stage(progress_callback, "CNN Classification")
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
            "details": "CNN classification is unavailable; the remaining stages continued."
        })
    _log_memory("after-cnn")

    # -------------------------------------------------------------------------
    # 4.0 OCR Text Extraction
    # -------------------------------------------------------------------------
    _report_stage(progress_callback, "OCR Text Extraction")
    logger.info("[Pipeline 4.0] Running OCR extraction")
    _log_memory("before-ocr")
    try:
        import time
        from ocr.extractor import OCREngineUnavailable, extract_text
        from nlp.quality import assess_ocr_quality
        ocr_started = time.perf_counter()
        ocr_results = extract_text(image_path)
        ocr_clean = []
        if ocr_results:
            for item in ocr_results:
                if not isinstance(item, dict):
                    logger.warning("[Pipeline 4.0] Ignoring malformed OCR item: %r", item)
                    continue
                txt = item.get("text", "")
                try:
                    conf = float(item.get("confidence", 0.0))
                except (TypeError, ValueError):
                    logger.warning("[Pipeline 4.0] Ignoring OCR item with invalid confidence: %r", item)
                    continue
                clean_item = {"text": str(txt), "confidence": round(conf, 2)}
                bbox = item.get("bbox")
                if isinstance(bbox, (list, tuple)):
                    clean_item["bbox"] = bbox
                ocr_clean.append(clean_item)

        # OCR reads the original upload instead of the legacy CNN preprocessing
        # output, which applied denoising, threshold-like contrast and sharpening.
        ocr_variant = "original_rgb"
        ocr_quality = assess_ocr_quality(ocr_clean)
        if not ocr_quality["reliable"]:
            # One color-aware fallback is the production limit. No thresholded
            # variants are run on every request; the evaluator compares those
            # offline where latency and memory are measured explicitly.
            try:
                from preprocessing.image_processing import make_ocr_variant, safe_imread
                from ocr.extractor import extract_text_from_image
                source_image = safe_imread(image_path)
                if source_image is not None:
                    source_h, source_w = source_image.shape[:2]
                    color_variant = make_ocr_variant(source_image, "hsv_chalk")
                    variant_h, variant_w = color_variant.shape[:2]
                    fallback_items = extract_text_from_image(color_variant, label="HSV chalk enhancement")
                    if variant_w != source_w or variant_h != source_h:
                        scale_x, scale_y = source_w / variant_w, source_h / variant_h
                        for fallback_item in fallback_items:
                            fallback_item["bbox"] = [
                                [round(point[0] * scale_x, 3), round(point[1] * scale_y, 3)]
                                for point in fallback_item["bbox"]
                            ]
                    fallback_quality = assess_ocr_quality(fallback_items)
                    current_key = (
                        ocr_quality["meaningful_term_count"],
                        ocr_quality["average_confidence"],
                        ocr_quality["supported_token_count"],
                        ocr_quality["analysis_region_count"],
                    )
                    fallback_key = (
                        fallback_quality["meaningful_term_count"],
                        fallback_quality["average_confidence"],
                        fallback_quality["supported_token_count"],
                        fallback_quality["analysis_region_count"],
                    )
                    if fallback_key > current_key:
                        ocr_clean = fallback_items
                        ocr_quality = fallback_quality
                        ocr_variant = "hsv_chalk"
                    del color_variant, source_image
            except Exception:
                logger.warning("[Pipeline 4.0] Optional chalk enhancement failed; keeping original OCR output", exc_info=True)

        # Maintain the original API's raw combined recognized text. NLP and
        # note generation use the separate, conservative analysis transcript.
        combined_text = " ".join(
            item["text"] for item in ocr_clean if item["confidence"] >= 0.40
        )
        analysis_note_items = ocr_quality.pop("analysis_items", [])
        ocr_quality.update({
            "variant": ocr_variant,
            "latency_seconds": round(time.perf_counter() - ocr_started, 3),
        })
        logger.info(
            "[Pipeline 4.0] %s — %d raw regions, %d supported tokens, warning=%s",
            ocr_variant, len(ocr_clean), ocr_quality["supported_token_count"],
            not ocr_quality["reliable"],
        )
        stage_status = "completed" if ocr_quality["reliable"] else "warning"
        stage_details = (
            f"Kept {len(ocr_clean)} raw regions; {ocr_quality['analysis_region_count']} regions "
            f"supported the analysis transcript."
        )
        if ocr_quality["warning"]:
            stage_details = ocr_quality["warning"]
        stages_log.append({
            "step": "4.0",
            "title": "OCR Text Extraction",
            "description": "OCR checks the original image and at most one gentle color-enhanced variant.",
            "status": stage_status,
            "details": stage_details,
        })
    except OCREngineUnavailable:
        logger.error("[Pipeline 4.0] OCR service unavailable", exc_info=True)
        raise
    except Exception as exc:
        logger.error("[Pipeline 4.0] FAILED:\n%s", traceback.format_exc())
        raise

    _log_memory("after-ocr")

    # Explicit GC after OCR — releases PyTorch intermediate tensors back to OS
    _log_memory("before-gc")
    gc.collect()
    _log_memory("after-gc")

    # -------------------------------------------------------------------------
    # 5.0 NLP Analysis
    # -------------------------------------------------------------------------
    _report_stage(progress_callback, "NLP Analysis")
    logger.info("[Pipeline 5.0] Running NLP analysis")
    _log_memory("before-nlp")
    try:
        from nlp.keywords import extract_keywords
        from nlp.topics import detect_topics
        analysis_text = ocr_quality["analysis_text"] if ocr_quality["reliable"] else ""
        keywords = extract_keywords(analysis_text, max_keywords=10) if analysis_text else []
        topics = detect_topics(analysis_text) if analysis_text else []
        logger.info("[Pipeline 5.0] OK — %d keywords, %d topics",
                    len(keywords), len(topics))
        stages_log.append({
            "step": "5.0",
            "title": "NLP Analysis",
            "description": "The extracted text is analyzed to identify important keywords and topics.",
            "status": "completed" if ocr_quality["reliable"] else "warning",
            "details": (
                f"Identified {len(keywords)} keywords and {len(topics)} academic topics"
                if ocr_quality["reliable"]
                else "Skipped unreliable OCR text; retake the image for analysis."
            )
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
            "details": "Text analysis is unavailable; no keywords or topics were added."
        })
    _log_memory("after-nlp")

    # -------------------------------------------------------------------------
    # 6.0 Structured Notes Generation
    # -------------------------------------------------------------------------
    _report_stage(progress_callback, "Structured Notes Generation")
    logger.info("[Pipeline 6.0] Generating structured notes")
    _log_memory("before-notes")
    try:
        from nlp.notes import generate_structured_notes
        if ocr_quality["reliable"]:
            structured_notes = generate_structured_notes(
                classification={"class": predicted_class, "confidence": round(confidence * 100, 2)},
                extracted_text=ocr_quality["analysis_text"],
                keywords=keywords,
                topics=topics,
                ocr_items=analysis_note_items,
            )
        else:
            from nlp.quality import low_confidence_notes
            structured_notes = low_confidence_notes(ocr_quality["warning"])
        logger.info("[Pipeline 6.0] OK — '%s'", structured_notes.get("title", ""))
        stages_log.append({
            "step": "6.0",
            "title": "Structured Notes Generation",
            "description": "The system combines the extracted information and generates organized notes for the user.",
            "status": "completed" if ocr_quality["reliable"] else "warning",
            "details": (
                f"Generated organized notes: '{structured_notes['title']}'"
                if ocr_quality["reliable"]
                else "Notes were withheld because the OCR transcript is unreliable."
            ),
        })
    except Exception as exc:
        logger.error("[Pipeline 6.0] FAILED:\n%s", traceback.format_exc())
        if ocr_quality["reliable"]:
            structured_notes = {
                "title": "Classroom Board Notes",
                "markdown": ocr_quality["analysis_text"],
                "plain_text": ocr_quality["analysis_text"],
                "key_points": [ocr_quality["analysis_text"]] if ocr_quality["analysis_text"] else [],
                "topics": topics,
                "keywords": keywords,
            }
        else:
            from nlp.quality import low_confidence_notes
            structured_notes = low_confidence_notes(ocr_quality["warning"])
        stages_log.append({
            "step": "6.0",
            "title": "Structured Notes Generation",
            "description": "The system combines the extracted information and generates organized notes for the user.",
            "status": "warning",
            "details": "Fallback notes were generated because structured note formatting failed."
        })
    _log_memory("after-notes")

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
        "analysis_text": ocr_quality["analysis_text"] if ocr_quality["reliable"] else "",
        "ocr_quality": ocr_quality,
        "ocr_warning": ocr_quality["warning"],
        "keywords": keywords,
        "topics": topics,
        "structured_notes": structured_notes
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    image_path = sys.argv[1].strip() if len(sys.argv) > 1 else input("Enter image path: ").strip()
    run_pipeline(image_path)
