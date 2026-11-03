import cv2
import os


# ============================================
# CNN DATASET CROP CREATOR
# ============================================

INPUT_FOLDER = "uploads"
OUTPUT_FOLDER = "dataset_samples"


# Create output folder
os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================
# FIND UPLOADED IMAGES
# ============================================

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png"
)

files = os.listdir(INPUT_FOLDER)

image_files = [
    file
    for file in files
    if file.lower().endswith(image_extensions)
]


print("================================")
print("CNN DATASET CROP CREATOR")
print("================================")

print(
    "Images found:",
    len(image_files)
)


# ============================================
# CHECK IMAGES
# ============================================

if len(image_files) == 0:

    print()
    print("No images found in uploads folder.")
    print("Please upload an image first.")
    exit()


# ============================================
# CREATE CROPS
# ============================================

for image_number, filename in enumerate(
    image_files,
    start=1
):

    image_path = os.path.join(
        INPUT_FOLDER,
        filename
    )


    print()
    print(
        "Processing:",
        filename
    )


    # Read image
    image = cv2.imread(image_path)


    if image is None:

        print(
            "ERROR: Could not read image."
        )

        continue


    # Get image size
    height, width = image.shape[:2]


    print(
        "Image size:",
        width,
        "x",
        height
    )


    # ========================================
    # CREATE 3 x 3 GRID
    # ========================================

    rows = 3
    columns = 3


    crop_height = height // rows
    crop_width = width // columns


    # Folder for this image
    image_output_folder = os.path.join(
        OUTPUT_FOLDER,
        f"image_{image_number}"
    )


    os.makedirs(
        image_output_folder,
        exist_ok=True
    )


    crop_number = 1


    # ========================================
    # CREATE EACH CROP
    # ========================================

    for row in range(rows):

        for column in range(columns):


            # Starting coordinates
            x1 = column * crop_width
            y1 = row * crop_height


            # Ending coordinates
            if column == columns - 1:

                x2 = width

            else:

                x2 = (column + 1) * crop_width


            if row == rows - 1:

                y2 = height

            else:

                y2 = (row + 1) * crop_height


            # Crop image
            crop = image[
                y1:y2,
                x1:x2
            ]


            # Output filename
            output_filename = (
                f"crop_{crop_number}.jpg"
            )


            output_path = os.path.join(
                image_output_folder,
                output_filename
            )


            # Save crop
            success = cv2.imwrite(
                output_path,
                crop
            )


            if success:

                print(
                    f"[✓] Created {output_filename}"
                )

            else:

                print(
                    f"[✗] Failed {output_filename}"
                )


            crop_number += 1


# ============================================
# COMPLETE
# ============================================

print()
print("================================")
print("CROP CREATION COMPLETE")
print("================================")

print()
print(
    "Crops saved inside:"
)

print(
    "dataset_samples/"
)

print()
print(
    "Open the generated crops and check them."
)