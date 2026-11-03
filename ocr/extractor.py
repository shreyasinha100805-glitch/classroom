import easyocr
from typing import Any


# ============================================
# EASY OCR
# ============================================

reader = easyocr.Reader(
    ['en'],
    gpu=False
)


# ============================================
# OCR EXTRACTION
# ============================================

def extract_text(
    image_path: str
) -> list[dict[str, Any]]:

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