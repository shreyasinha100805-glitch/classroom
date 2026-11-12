import os
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Max dimension for OCR input — large enough for good text recognition,
# small enough to keep EasyOCR tensor sizes under ~25 MB (instead of 200 MB).
# At 960px with mag_ratio=1.0 → 960×720 × 3 × 4 bytes ≈ 7.9 MB float32 input.
MAX_DIM = int(os.environ.get("PREPROCESS_MAX_DIM", "960"))


def safe_imread(path: str):
    """Safely read image supporting Unicode file paths."""
    try:
        with open(path, "rb") as f:
            bytes_data = bytearray(f.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
            del bytes_data, numpy_array
            return img
    except Exception:
        return cv2.imread(path)


def safe_imwrite(path: str, img):
    """Safely write image supporting Unicode file paths."""
    output_folder = os.path.dirname(path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".jpg"
    try:
        is_success, buffer = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if is_success:
            with open(path, "wb") as f:
                f.write(buffer)
            del buffer
            return True
    except Exception:
        pass
    return cv2.imwrite(path, img)


def preprocess_image(input_path: str, output_path: str) -> str:
    """
    Preprocess classroom board image for OCR and CNN.

    Steps:
    1. Read image safely (supports mobile and Unicode filenames)
    2. Resize to MAX_DIM (default 960px) — reduced from 1600px to cut
       EasyOCR tensor memory from ~200 MB to ~25 MB at mag_ratio=1.0.
    3. Convert to grayscale (halves memory vs RGB for denoising/CLAHE)
    4. Reduce high-frequency noise with fast NlMeans
    5. Enhance localized contrast via adaptive CLAHE
    6. Sharpen handwriting strokes
    7. Save processed image; release intermediate arrays immediately
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    image = safe_imread(input_path)
    if image is None:
        raise ValueError(f"Could not decode image at: {input_path}")

    # 1. Resize to memory-safe OCR resolution while maintaining aspect ratio
    height, width = image.shape[:2]
    logger.info("Preprocessing: original size %dx%d, max_dim=%d", width, height, MAX_DIM)

    if max(width, height) > MAX_DIM:
        scale = MAX_DIM / float(max(width, height))
        new_width = max(int(width * scale), 64)
        new_height = max(int(height * scale), 64)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.info("Resized to %dx%d", new_width, new_height)

    # 2. Convert to grayscale — halves memory for downstream processing
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        del image  # release color image immediately
    else:
        gray = image

    # 3. Fast Denoising optimized for blackboard / chalkboard chalk strokes
    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=6,
        templateWindowSize=7,
        searchWindowSize=15
    )
    del gray  # release immediately

    # 4. Local Contrast Improvement via Adaptive CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    del denoised  # release immediately

    # 5. Gaussian unsharp masking for stroke edge sharpness
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)
    del enhanced, blurred  # release immediately

    # 6. Save preprocessed image
    success = safe_imwrite(output_path, sharpened)
    del sharpened
    if not success:
        raise ValueError(f"Could not save processed image to: {output_path}")

    logger.info("Preprocessing complete: saved to %s", output_path)
    return output_path
