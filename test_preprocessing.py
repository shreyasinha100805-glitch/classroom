import os
import cv2
from preprocessing.image_processing import preprocess_image


UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "preprocessed"


os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================
# DELETE OLD PROCESSED IMAGES
# ============================================

for file in os.listdir(OUTPUT_FOLDER):

    file_path = os.path.join(
        OUTPUT_FOLDER,
        file
    )

    if os.path.isfile(file_path):

        os.remove(file_path)


# ============================================
# FIND UPLOADED IMAGES
# ============================================

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png"
)


files = os.listdir(UPLOAD_FOLDER)

image_files = [
    file
    for file in files
    if file.lower().endswith(image_extensions)
]


print("================================")
print("CLASSROOM IMAGE PREPROCESSING")
print("================================")

print(
    "Images found:",
    len(image_files)
)


# ============================================
# PROCESS IMAGES
# ============================================

for index, filename in enumerate(
    image_files,
    start=1
):

    input_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    try:

        # ------------------------------------
        # Enhanced version
        # ------------------------------------

        enhanced_path = os.path.join(
            OUTPUT_FOLDER,
            f"processed_{index}_enhanced.jpg"
        )

        preprocess_image(
            input_path,
            enhanced_path
        )


        # ------------------------------------
        # Grayscale version
        # ------------------------------------

        image = cv2.imread(input_path)

        if image is None:

            raise ValueError(
                "Could not read image"
            )


        height, width = image.shape[:2]

        max_width = 2000

        if width > max_width:

            scale = max_width / width

            image = cv2.resize(
                image,
                (
                    int(width * scale),
                    int(height * scale)
                ),
                interpolation=cv2.INTER_CUBIC
            )


        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )


        gray_path = os.path.join(
            OUTPUT_FOLDER,
            f"processed_{index}_gray.jpg"
        )

        cv2.imwrite(
            gray_path,
            gray
        )


        print(
            f"[✓] {filename}"
        )

        print(
            "    → enhanced"
        )

        print(
            "    → grayscale"
        )


    except Exception as error:

        print(
            f"[✗] {filename}: {error}"
        )


print("================================")
print("PREPROCESSING COMPLETE")
print("================================")