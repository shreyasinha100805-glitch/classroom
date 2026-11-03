import cv2
import os


def preprocess_image(input_path, output_path):
    """
    Preprocess classroom board image for OCR.

    Steps:
    1. Read image
    2. Resize if too large
    3. Convert to grayscale
    4. Reduce noise
    5. Improve contrast
    6. Sharpen text
    7. Save processed image
    """

    image = cv2.imread(input_path)

    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    # ----------------------------------------
    # 1. Resize
    # ----------------------------------------

    height, width = image.shape[:2]

    max_width = 2000

    if width > max_width:

        scale = max_width / width

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC
        )

    # ----------------------------------------
    # 2. Convert to grayscale
    # ----------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ----------------------------------------
    # 3. Noise reduction
    # ----------------------------------------

    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=7,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # ----------------------------------------
    # 4. Improve contrast
    # ----------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(denoised)

    # ----------------------------------------
    # 5. Sharpen
    # ----------------------------------------

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.0
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.5,
        blurred,
        -0.5,
        0
    )

    # ----------------------------------------
    # 6. Save
    # ----------------------------------------

    output_folder = os.path.dirname(output_path)

    if output_folder:
        os.makedirs(
            output_folder,
            exist_ok=True
        )

    success = cv2.imwrite(
        output_path,
        sharpened
    )

    if not success:
        raise ValueError(
            f"Could not save processed image: {output_path}"
        )

    return output_path