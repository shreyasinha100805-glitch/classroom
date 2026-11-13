import os
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ==========================================
# CNN PREDICTION MODULE (ONNX RUNTIME)
# ==========================================
#
# Replaces TensorFlow/Keras with onnxruntime (CPU-only).
# Saves ~120 MB RSS on 512 MB Render instances.
#
# Public API is unchanged:
#   get_model()         -> onnxruntime.InferenceSession (singleton)
#   predict_image(path) -> {"class": str, "confidence": float, "probabilities": dict}
#   CLASS_NAMES         -> ["background", "board", "text"]
#   IMAGE_SIZE          -> (128, 128)
#
# Input contract (preserved from Keras version):
#   predict_image() divides pixel values by 255 before passing to the session.
#   The ONNX model itself also applies a /255 Rescaling layer baked in at export.
#   Net result: pixels are divided by 255 twice (same behaviour as Keras version).
#   This is intentional — changing it would shift all predictions.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model", "classroom_cnn.onnx")
MODEL_PATH = os.environ.get("CNN_MODEL_PATH", DEFAULT_MODEL_PATH)

IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "background",
    "board",
    "text"
]

_session = None
_session_lock = None


def _get_lock():
    global _session_lock
    if _session_lock is None:
        import threading
        _session_lock = threading.Lock()
    return _session_lock


def get_model():
    """
    Lazy singleton loader for the ONNX CNN inference session.
    Thread-safe. Loads once per process lifetime.

    Uses onnxruntime CPU execution provider.
    Avoids importing TensorFlow or Keras, saving ~120 MB RSS
    on 512 MB Render instances.
    """
    global _session
    if _session is not None:
        return _session

    with _get_lock():
        if _session is not None:
            return _session

        # Locate ONNX model file, with fallback to .keras path for error clarity
        target_path = MODEL_PATH
        if not os.path.exists(target_path):
            alt_path = os.path.join(os.getcwd(), "cnn", "model", "classroom_cnn.onnx")
            if os.path.exists(alt_path):
                target_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Classroom CNN ONNX model not found at '{target_path}' or '{alt_path}'. "
                    f"Ensure classroom_cnn.onnx is present in cnn/model/."
                )

        logger.info("Loading CNN ONNX model from: %s", target_path)
        try:
            import onnxruntime as ort
            # CPU-only execution; disable any GPU/CUDA providers
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            _session = ort.InferenceSession(
                target_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            input_meta  = _session.get_inputs()[0]
            output_meta = _session.get_outputs()[0]
            logger.info(
                "[OK] CNN ONNX session ready. input=%s%s output=%s%s",
                input_meta.name, input_meta.shape,
                output_meta.name, output_meta.shape,
            )
            return _session
        except Exception as exc:
            logger.error(
                "Error loading CNN ONNX model from %s: %s", target_path, exc, exc_info=True
            )
            raise


def predict_image(image_path: str) -> dict:
    """
    Predict classroom board content type: 'board', 'text', or 'background'.

    Normalises pixel values to [0, 1] before inference (preserves the
    same input contract as the previous Keras version — the ONNX model
    also applies an internal /255 Rescaling layer).

    Input image is resized to 128×128; no extra copies are retained
    after prediction.

    Returns:
        {
            "class":       str,   # predicted class name
            "confidence":  float, # softmax probability of predicted class
            "probabilities": {    # softmax probabilities for all classes
                "background": float,
                "board":      float,
                "text":       float,
            }
        }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    logger.info("CNN predict: loading image %s", image_path)
    session = get_model()

    with Image.open(image_path) as pil_img:
        rgb = pil_img.convert("RGB").resize(IMAGE_SIZE)
        image_array = np.array(rgb, dtype=np.float32) / 255.0  # normalise to [0, 1]

    image_array = np.expand_dims(image_array, axis=0)   # (1, 128, 128, 3)
    logger.info("CNN predict: running ONNX inference")

    input_name  = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: image_array})[0][0]
    del image_array  # release immediately after inference

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    probabilities = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    logger.info(
        "CNN predict: class=%s confidence=%.3f", predicted_class, confidence
    )
    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("==============================================")
    print("       CNN IMAGE PREDICTION TEST (ONNX)")
    print("==============================================")
    sample = os.path.join(os.path.dirname(BASE_DIR), "public", "samples", "sample_board.jpg")
    if os.path.exists(sample):
        print(f"Testing on sample: {sample}")
        print(predict_image(sample))
    else:
        path = input("Enter image path: ").strip()
        print(predict_image(path))
