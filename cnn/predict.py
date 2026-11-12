import os
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ==========================================
# CNN PREDICTION MODULE (PRODUCTION READY)
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model", "classroom_cnn.keras")
MODEL_PATH = os.environ.get("CNN_MODEL_PATH", DEFAULT_MODEL_PATH)

IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "background",
    "board",
    "text"
]

_model = None
_model_lock = None


def _get_lock():
    global _model_lock
    if _model_lock is None:
        import threading
        _model_lock = threading.Lock()
    return _model_lock


def get_model():
    """
    Lazy singleton loader for the trained Keras CNN model.
    Thread-safe. Loads once per process lifetime.

    Sets TORCHDYNAMO_DISABLE and TF_CPP_MIN_LOG_LEVEL before model load
    to suppress verbose TensorFlow logs and prevent PyTorch's dynamo JIT
    from initialising (saves ~65 MB RSS on 512 MB Render instances).
    """
    global _model
    if _model is not None:
        return _model

    with _get_lock():
        if _model is not None:
            return _model

        # Suppress TF verbose startup logs
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        # Disable torch.compile / dynamo — not needed for CPU inference
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        # TensorFlow memory reduction — disable GPU scanning and oneDNN overhead
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

        # Locate model file with fallback paths
        target_path = MODEL_PATH
        if not os.path.exists(target_path):
            alt_path = os.path.join(os.getcwd(), "cnn", "model", "classroom_cnn.keras")
            if os.path.exists(alt_path):
                target_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Classroom CNN model not found at '{target_path}' or '{alt_path}'."
                )

        logger.info("Loading CNN model from: %s", target_path)
        try:
            try:
                import keras
                _model = keras.models.load_model(target_path)
            except Exception:
                import tensorflow as tf
                _model = tf.keras.models.load_model(target_path)
            logger.info("[OK] CNN model loaded successfully from: %s", target_path)
            return _model
        except Exception as exc:
            logger.error("Error loading CNN model from %s: %s", target_path, exc, exc_info=True)
            raise


def predict_image(image_path: str) -> dict:
    """
    Predict classroom board content type: 'board', 'text', or 'background'.

    Normalises pixel values to [0, 1] before inference (required by the
    trained model). Input image is resized to 128×128 and kept as a
    minimal float32 array; no extra copies are retained after prediction.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    logger.info("CNN predict: loading image %s", image_path)
    model = get_model()

    with Image.open(image_path) as pil_img:
        rgb = pil_img.convert("RGB").resize(IMAGE_SIZE)
        image_array = np.array(rgb, dtype=np.float32) / 255.0  # normalise to [0, 1]

    image_array = np.expand_dims(image_array, axis=0)   # (1, 128, 128, 3)
    logger.info("CNN predict: running inference")
    predictions = model.predict(image_array, verbose=0)[0]
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
    print("       CNN IMAGE PREDICTION TEST")
    print("==============================================")
    sample = os.path.join(os.path.dirname(BASE_DIR), "public", "samples", "sample_board.jpg")
    if os.path.exists(sample):
        print(f"Testing on sample: {sample}")
        print(predict_image(sample))
    else:
        path = input("Enter image path: ").strip()
        print(predict_image(path))
