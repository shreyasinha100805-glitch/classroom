import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ==========================================
# CNN PREDICTION MODULE
# ==========================================

MODEL_PATH = "cnn/model/classroom_cnn.keras"

IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "background",
    "board",
    "text"
]


# ==========================================
# LOAD MODEL
# ==========================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ==========================================
# PREDICT IMAGE
# ==========================================

def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(predictions[i])
            for i in range(len(CLASS_NAMES))
        }
    }


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print()
    print("==============================================")
    print("       CNN IMAGE PREDICTION")
    print("==============================================")

    image_path = input(
        "Enter image path: "
    ).strip()

    if not os.path.exists(image_path):

        print()
        print("Image not found.")
        exit()

    result = predict_image(
        image_path
    )

    print()
    print("Prediction:")
    print(
        "Class:",
        result["class"]
    )

    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )

    print()
    print("Probabilities:")

    for class_name, probability in result[
        "probabilities"
    ].items():

        print(
            f"{class_name}: "
            f"{probability * 100:.2f}%"
        )

    print()
    print("==============================================")