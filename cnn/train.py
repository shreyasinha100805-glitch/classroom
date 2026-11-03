import os
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore[reportMissingModuleSource]

# ==========================================
# CNN CLASSROOM BOARD CLASSIFIER
# ==========================================

DATASET_DIR = "dataset"
MODEL_DIR = "cnn/model"

IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 20

os.makedirs(MODEL_DIR, exist_ok=True)

print("==============================================")
print("       CNN CLASSROOM CONTENT CLASSIFIER")
print("==============================================")
print()

# ==========================================
# LOAD TRAINING DATA
# ==========================================

train_dir = os.path.join(
    DATASET_DIR,
    "train"
)

validation_dir = os.path.join(
    DATASET_DIR,
    "validation"
)

test_dir = os.path.join(
    DATASET_DIR,
    "test"
)

print("Loading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print()
print("Loading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print()
print("Loading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================
# CLASS NAMES
# ==========================================

class_names = train_dataset.class_names

print()
print("Classes:")
for index, name in enumerate(class_names):
    print(index, "->", name)

print()

# ==========================================
# PERFORMANCE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# ==========================================
# CNN MODEL
# ==========================================

model = models.Sequential([

    layers.Input(
        shape=(128, 128, 3)
    ),

    # Normalize pixels
    layers.Rescaling(
        1.0 / 255
    ),

    # CNN Layer 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # CNN Layer 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # CNN Layer 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convert feature maps to vector
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.3
    ),

    # Output layer
    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("==============================================")
print("MODEL ARCHITECTURE")
print("==============================================")

model.summary()

# ==========================================
# TRAIN
# ==========================================

print()
print("==============================================")
print("STARTING CNN TRAINING")
print("==============================================")
print()

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# ==========================================
# TEST
# ==========================================

print()
print("==============================================")
print("TESTING CNN")
print("==============================================")

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print()
print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

# ==========================================
# SAVE MODEL
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "classroom_cnn.keras"
)

model.save(model_path)

print()
print("==============================================")
print("CNN TRAINING COMPLETE")
print("==============================================")
print()
print("Model saved at:")
print(model_path)
print()
print("Classes:")
print(class_names)
print("==============================================")