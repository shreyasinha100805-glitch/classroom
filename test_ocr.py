import os
from ocr.extractor import extract_text


IMAGE_FOLDER = "preprocessed"


# ============================================
# FIND IMAGES
# ============================================

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png"
)


files = os.listdir(
    IMAGE_FOLDER
)


image_files = [
    file
    for file in files
    if file.lower().endswith(
        image_extensions
    )
]


print("================================")
print("CLASSROOM BOARD OCR")
print("================================")

print(
    "Images found:",
    len(image_files)
)


if not image_files:

    print(
        "No processed images found."
    )

    print(
        "Run test_preprocessing.py first."
    )

    exit()


# ============================================
# OCR
# ============================================

all_text = []


for index, filename in enumerate(
    image_files,
    start=1
):

    image_path = os.path.join(
        IMAGE_FOLDER,
        filename
    )


    print()
    print(
        f"Processing Image {index}: {filename}"
    )

    print("--------------------------------")


    try:

        results = extract_text(
            image_path
        )


        if not results:

            print(
                "No readable text detected."
            )

            continue


        for item in results:

            text = item["text"]

            confidence = item["confidence"]


            print(
                f"Text: {text}"
            )

            print(
                f"Confidence: {confidence:.2f}"
            )


            if confidence >= 0.45:

                all_text.append(text)


    except Exception as error:

        print(
            f"OCR Error: {error}"
        )


# ============================================
# COMBINED TEXT
# ============================================

print()
print("================================")
print("COMBINED EXTRACTED TEXT")
print("================================")


if all_text:

    for text in all_text:

        print(text)

else:

    print(
        "No reliable text extracted."
    )


print()
print("================================")
print("OCR COMPLETE")
print("================================")