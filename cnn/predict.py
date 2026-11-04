import os
import numpy as np
from PIL import Image

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


def get_model():
    """
    Lazy singleton loader for the trained Keras CNN model.
    Ensures safe loading across local, cloud, and production server environments.
    """
    global _model
    if _model is not None:
        return _model

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

    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(target_path)
        print(f"[OK] Classroom CNN model loaded successfully from: {target_path}")
        return _model
    except Exception as exc:
        print(f"Error loading CNN model from {target_path}: {exc}")
        raise


def predict_image(image_path: str) -> dict:
    """
    Predict classroom board content type: 'board', 'text', or 'background'.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    model = get_model()

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    probabilities = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }


if __name__ == "__main__":
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
