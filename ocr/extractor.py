import os
import easyocr
from typing import Any


# ============================================
# EASY OCR (LAZY SINGLETON INITIALIZATION)
# ============================================

_reader = None

def get_reader():
    global _reader
    if _reader is None:
        try:
            _reader = easyocr.Reader(
                ['en'],
                gpu=False
            )
        except Exception as exc:
            print(f"Notice: Initializing EasyOCR reader with fallback: {exc}")
            try:
                import tempfile
                tmp_dir = os.path.join(tempfile.gettempdir(), ".EasyOCR", "model")
                os.makedirs(tmp_dir, exist_ok=True)
                _reader = easyocr.Reader(
                    ['en'],
                    gpu=False,
                    model_storage_directory=tmp_dir,
                    user_network_directory=tmp_dir
                )
            except Exception as exc2:
                print(f"Warning: EasyOCR reader could not be initialized: {exc2}")
                _reader = None
    return _reader


# ============================================
# OCR EXTRACTION
# ============================================

def extract_text(
    image_path: str
) -> list[dict[str, Any]]:

    reader = get_reader()
    if reader is None:
        print("Warning: EasyOCR reader unavailable.")
        return []

    try:
        results = reader.readtext(
            image_path,

            # Detect individual text regions
            detail=1,

            # Do not merge everything into paragraphs
            paragraph=False,

            # Detect relatively small handwriting
            min_size=10,

            # OCR detection settings
            text_threshold=0.55,
            low_text=0.25,
            link_threshold=0.30,

            # Improve small text
            mag_ratio=1.5
        )
    except Exception as exc:
        print(f"Warning: EasyOCR extraction error on {image_path}: {exc}")
        return []

    extracted_text = []

    for result in results:

        if not isinstance(
            result,
            (list, tuple)
        ):
            continue

        if len(result) < 3:
            continue

        text = str(
            result[1]
        ).strip()

        try:
            confidence = float(
                result[2]
            )
        except (
            ValueError,
            TypeError
        ):
            continue

        if not text:
            continue

        # Remove extremely uncertain results
        if confidence < 0.35:
            continue

        extracted_text.append({
            "text": text,
            "confidence": confidence
        })

    return extracted_text
