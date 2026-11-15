import os
import logging
import cv2
import numpy as np
from PIL import Image as PilImage

logger = logging.getLogger(__name__)

# Max dimension for the CNN preprocessing/preview image. OCR uses the original
# image and its own bounded variants so this 960 px ceiling does not erase text.
MAX_DIM = int(os.environ.get("PREPROCESS_MAX_DIM", "960"))

# Hard pixel-count ceiling enforced BEFORE any numpy/cv2 decode to prevent
# decompression bombs (e.g. a 32 MB JPEG that decompresses to 150+ MP).
# 8 MP (e.g. 3264×2448) = 8_000_000 pixels — plenty for board OCR.
MAX_UPLOAD_PIXELS = int(os.environ.get("MAX_UPLOAD_PIXELS", str(8_000_000)))

# OCR gets the original color image first.  These alternatives are intentionally
# conservative: the thresholded variants are useful for evaluation, but are not
# used automatically in production because they can erase chalk strokes.
OCR_VARIANTS = (
    "original_rgb",
    "grayscale",
    "clahe",
    "adaptive_threshold",
    "otsu_threshold",
    "lab_clahe",
    "hsv_chalk",
    "perspective_corrected",
)


def _resize_for_ocr(image, max_dim=1600):
    """Keep fine strokes while bounding OCR input dimensions and memory."""
    height, width = image.shape[:2]
    if max(width, height) <= max_dim:
        return image.copy()
    scale = max_dim / float(max(width, height))
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _order_quad(points):
    """Return four perspective corners in top-left, top-right, bottom-right, bottom-left order."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    return np.array([
        points[np.argmin(sums)],
        points[np.argmin(diffs)],
        points[np.argmax(sums)],
        points[np.argmax(diffs)],
    ], dtype=np.float32)


def _correct_board_perspective(image):
    """Warp a clearly segmented green board surface; leave other images untouched."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    board_mask = cv2.inRange(hsv, np.array([30, 28, 20]), np.array([105, 255, 205]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image.copy(), False

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < width * height * 0.08:
        return image.copy(), False
    perimeter = cv2.arcLength(contour, True)
    quad = None
    for epsilon in (0.015, 0.02, 0.03, 0.04):
        candidate = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(candidate) == 4 and cv2.isContourConvex(candidate):
            quad = candidate.reshape(4, 2)
            break
    if quad is None:
        return image.copy(), False

    source = _order_quad(quad)
    top_left, top_right, bottom_right, bottom_left = source
    out_width = int(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
    out_height = int(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
    if out_width < 64 or out_height < 64:
        return image.copy(), False
    # Avoid turning an unreliable contour into a very large intermediate image.
    scale = min(1.0, 1600 / max(out_width, out_height))
    out_width, out_height = max(1, int(out_width * scale)), max(1, int(out_height * scale))
    target = np.array([
        [0, 0], [out_width - 1, 0],
        [out_width - 1, out_height - 1], [0, out_height - 1],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    corrected = cv2.warpPerspective(image, matrix, (out_width, out_height), flags=cv2.INTER_CUBIC)
    return corrected, True


def make_ocr_variant(image, variant):
    """Create one bounded OCR image variant from an OpenCV BGR image.

    The caller owns the returned array. No denoising or sharpening is applied;
    those operations can remove thin handwriting and punctuation.
    """
    if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
        raise ValueError("OCR variant input must be a decoded image array")
    # Let RapidOCR's own bounded resize operate on the source dimensions. A
    # separate 960 px resize here previously discarded thin strokes.
    base = image.copy()
    gray = None
    try:
        if variant == "original_rgb":
            return base
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        if variant == "grayscale":
            return gray
        if variant == "clahe":
            return cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(gray)
        if variant == "adaptive_threshold":
            # Keep the foreground polarity RapidOCR can use; this is evaluation-only.
            block = min(31, min(gray.shape[:2]) // 2 * 2 - 1)
            if block < 3:
                return gray.copy()
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block, 7,
            )
        if variant == "otsu_threshold":
            return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if variant == "lab_clahe":
            lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
            light, channel_a, channel_b = cv2.split(lab)
            enhanced = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(light)
            return cv2.cvtColor(cv2.merge((enhanced, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
        if variant == "hsv_chalk":
            hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
            hue, saturation, value = cv2.split(hsv)
            yellow = (hue >= 12) & (hue <= 42) & (saturation >= 35) & (value >= 70)
            white = (saturation <= 90) & (value >= 145)
            chalk = yellow | white
            enhanced_value = value.copy()
            enhanced_value[chalk] = np.minimum(
                255, enhanced_value[chalk].astype(np.float32) * 1.06 + 10
            ).astype(np.uint8)
            hsv[:, :, 2] = enhanced_value
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if variant == "perspective_corrected":
            corrected, found = _correct_board_perspective(base)
            return corrected if found else base
        raise ValueError(f"Unknown OCR image variant: {variant}")
    finally:
        # The returned image is independent of base except for original_rgb.
        if variant != "original_rgb":
            del base


def safe_imread(path: str):
    """
    Safely read image supporting Unicode file paths.

    Uses Pillow to check pixel dimensions FIRST (cheap, header-only read)
    before decoding the full image into a numpy array. This prevents
    decompression bombs from large JPEGs consuming >64 MB in the bytearray
    buffer before the resize step can shrink them.

    Returns None if the image exceeds MAX_UPLOAD_PIXELS.
    """
    # Cheap header-only size check via Pillow (does not load pixel data)
    try:
        with PilImage.open(path) as probe:
            w, h = probe.size
            pixels = w * h
            if pixels > MAX_UPLOAD_PIXELS:
                logger.warning(
                    "Image %s rejected: %dx%d = %d pixels exceeds MAX_UPLOAD_PIXELS=%d",
                    os.path.basename(path), w, h, pixels, MAX_UPLOAD_PIXELS
                )
                return None
    except Exception as probe_exc:
        logger.warning("Could not probe image dimensions for %s: %s", path, probe_exc)
        # Fall through and attempt full decode — cv2 will handle corrupt files

    try:
        with open(path, "rb") as f:
            bytes_data = bytearray(f.read())
        numpy_array = np.frombuffer(bytes_data, dtype=np.uint8)
        del bytes_data  # release as early as possible
        img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        del numpy_array
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
    Build the existing CNN classification/preview image.

    This path retains the historical denoising/contrast/sharpening steps for
    classifier compatibility. Do not use this output as OCR input: the 960 px
    resize and filters can erase narrow handwriting and punctuation.

    Steps:
    1. Read image safely — rejects images > MAX_UPLOAD_PIXELS (default 8 MP)
       before full decode to prevent decompression-bomb memory spikes.
    2. Resize longest side to MAX_DIM (default 960px).
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
        raise ValueError(
            f"Could not decode image at: {input_path}. "
            f"The image may be corrupt or exceed the {MAX_UPLOAD_PIXELS:,} pixel limit."
        )

    height, width = image.shape[:2]
    logger.info("Preprocessing: original size %dx%d, max_dim=%d", width, height, MAX_DIM)

    if max(width, height) > MAX_DIM:
        scale = MAX_DIM / float(max(width, height))
        new_width = max(int(width * scale), 64)
        new_height = max(int(height * scale), 64)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.info("Resized to %dx%d", new_width, new_height)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        del image  # release color image immediately
    else:
        gray = image.copy()
        del image

    denoised = cv2.fastNlMeansDenoising(gray, None, h=6, templateWindowSize=7, searchWindowSize=15)
    del gray  # release immediately

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    del denoised  # release immediately

    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)
    del enhanced, blurred  # release immediately

    success = safe_imwrite(output_path, sharpened)
    del sharpened
    if not success:
        raise ValueError(f"Could not save processed image to: {output_path}")

    logger.info("Preprocessing complete: saved to %s", output_path)
    return output_path
