import os
import cv2
import math
import numpy as np


# ============================================
# SETTINGS
# ============================================

INPUT_FOLDER = "dataset_samples"
OUTPUT_FILE = "crop_preview.jpg"

CROP_WIDTH = 180
CROP_HEIGHT = 140

COLUMNS = 5


# ============================================
# CHECK INPUT FOLDER
# ============================================

if not os.path.exists(INPUT_FOLDER):

    print("================================")
    print("ERROR")
    print("================================")
    print("dataset_samples folder not found.")
    print("Run create_dataset.py first.")
    exit()


# ============================================
# FIND ALL CROPS
# ============================================

images = []


for folder in sorted(
    os.listdir(INPUT_FOLDER)
):

    folder_path = os.path.join(
        INPUT_FOLDER,
        folder
    )

    if not os.path.isdir(folder_path):
        continue


    for filename in sorted(
        os.listdir(folder_path)
    ):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue


        image_path = os.path.join(
            folder_path,
            filename
        )


        image = cv2.imread(
            image_path
        )


        if image is None:
            continue


        # Resize crop
        image = cv2.resize(
            image,
            (
                CROP_WIDTH,
                CROP_HEIGHT
            )
        )


        # Label
        label = f"{folder}/{filename}"


        # Black label background
        cv2.rectangle(
            image,
            (0, 0),
            (CROP_WIDTH, 25),
            (0, 0, 0),
            -1
        )


        # Write label
        cv2.putText(
            image,
            label,
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        images.append(image)


# ============================================
# CHECK CROPS
# ============================================

if not images:

    print("================================")
    print("ERROR")
    print("================================")

    print(
        "No crop images found."
    )

    print(
        "Run create_dataset.py first."
    )

    exit()


# ============================================
# CREATE PREVIEW SIZE
# ============================================

total_images = len(images)

rows = math.ceil(
    total_images / COLUMNS
)


# ============================================
# CREATE WHITE CANVAS
# ============================================

preview = np.ones(
    (
        rows * CROP_HEIGHT,
        COLUMNS * CROP_WIDTH,
        3
    ),
    dtype=np.uint8
) * 255


# ============================================
# PLACE CROPS ON CANVAS
# ============================================

for index, image in enumerate(images):

    row = index // COLUMNS
    column = index % COLUMNS


    y1 = row * CROP_HEIGHT
    y2 = y1 + CROP_HEIGHT


    x1 = column * CROP_WIDTH
    x2 = x1 + CROP_WIDTH


    preview[
        y1:y2,
        x1:x2
    ] = image


# ============================================
# SAVE PREVIEW
# ============================================

success = cv2.imwrite(
    OUTPUT_FILE,
    preview
)


print()
print("================================")
print("CROP PREVIEW")
print("================================")

print(
    "Total crops:",
    total_images
)


if success:

    print(
        "Preview saved as:",
        OUTPUT_FILE
    )

else:

    print(
        "Failed to create preview."
    )


print("================================")