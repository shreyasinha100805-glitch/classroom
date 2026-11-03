import os
import shutil
import random
import csv

# ==========================================
# CNN DATASET SPLITTER
# READS LABELS DIRECTLY FROM labels.csv
# ==========================================

LABEL_FILE = "labels.csv"
SOURCE_DATASET = "dataset"
TEMP_DATASET = "dataset_temp"

CLASSES = [
    "board",
    "text",
    "background"
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Same result every time
random.seed(42)

print()
print("==============================================")
print("       CNN DATASET SPLITTER")
print("==============================================")
print()


# ==========================================
# CHECK LABEL FILE
# ==========================================

if not os.path.exists(LABEL_FILE):

    print("ERROR: labels.csv not found.")
    print()
    print("Make sure labels.csv is in:")
    print(os.getcwd())
    print()

    exit()


# ==========================================
# READ labels.csv
# ==========================================

all_images = {
    "board": [],
    "text": [],
    "background": []
}

used_paths = set()

print("Reading labels.csv...")
print()


with open(
    LABEL_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        path = row.get("path", "").strip()
        label = row.get("label", "").strip().lower()

        # Ignore invalid rows
        if not path:
            continue

        # Ignore SKIP
        if label == "skip":
            continue

        # Ignore unknown labels
        if label not in CLASSES:
            continue

        # Normalize Windows paths
        path = os.path.normpath(path)

        # Prevent duplicate entries
        if path in used_paths:
            continue

        # Check image exists
        if not os.path.isfile(path):

            print("WARNING: Image not found:")
            print(path)

            continue

        # Check extension
        if not path.lower().endswith(IMAGE_EXTENSIONS):
            continue

        all_images[label].append(path)
        used_paths.add(path)


# ==========================================
# SHOW DATA
# ==========================================

print("CURRENT LABELED DATA")
print("----------------------------------------------")

total_images = 0

for class_name in CLASSES:

    count = len(all_images[class_name])

    print(
        f"{class_name}: {count} images"
    )

    total_images += count

print("----------------------------------------------")
print(f"TOTAL: {total_images} images")
print()


# ==========================================
# CHECK DATA
# ==========================================

for class_name in CLASSES:

    count = len(all_images[class_name])

    if count < 3:

        print("ERROR")
        print("----------------------------------------------")
        print(
            f"'{class_name}' has only "
            f"{count} image(s)."
        )

        print()
        print(
            "Please label more crops for this class."
        )

        exit()


# ==========================================
# REMOVE OLD TEMP DATASET
# ==========================================

if os.path.exists(TEMP_DATASET):

    shutil.rmtree(TEMP_DATASET)


# ==========================================
# CREATE NEW FOLDERS
# ==========================================

for split in [
    "train",
    "validation",
    "test"
]:

    for class_name in CLASSES:

        folder = os.path.join(
            TEMP_DATASET,
            split,
            class_name
        )

        os.makedirs(
            folder,
            exist_ok=True
        )


# ==========================================
# SPLIT DATA
# ==========================================

for class_name in CLASSES:

    images = all_images[class_name].copy()

    random.shuffle(images)

    total = len(images)

    # --------------------------------------
    # 70% TRAIN
    # 20% VALIDATION
    # 10% TEST
    # --------------------------------------

    train_count = int(total * 0.70)
    validation_count = int(total * 0.20)

    test_count = (
        total
        - train_count
        - validation_count
    )

    # Make sure every split has at least
    # one image
    if train_count < 1:
        train_count = 1

    if validation_count < 1:
        validation_count = 1

    if test_count < 1:
        test_count = 1

    # Prevent total from exceeding dataset
    while (
        train_count
        + validation_count
        + test_count
        > total
    ):

        if train_count > 1:
            train_count -= 1

        elif validation_count > 1:
            validation_count -= 1

        else:
            test_count -= 1


    # --------------------------------------
    # CREATE GROUPS
    # --------------------------------------

    train_images = images[
        :train_count
    ]

    validation_images = images[
        train_count:
        train_count + validation_count
    ]

    test_images = images[
        train_count + validation_count:
    ]


    groups = {
        "train": train_images,
        "validation": validation_images,
        "test": test_images
    }


    # --------------------------------------
    # COPY IMAGES
    # --------------------------------------

    for split, files in groups.items():

        for number, source_path in enumerate(
            files,
            start=1
        ):

            extension = os.path.splitext(
                source_path
            )[1].lower()

            new_filename = (
                f"{class_name}_{number}"
                f"{extension}"
            )

            destination_path = os.path.join(
                TEMP_DATASET,
                split,
                class_name,
                new_filename
            )

            shutil.copy2(
                source_path,
                destination_path
            )


# ==========================================
# BACKUP OLD DATASET
# ==========================================

OLD_DATASET = "dataset_old"

if os.path.exists(OLD_DATASET):

    shutil.rmtree(OLD_DATASET)


# ==========================================
# REPLACE DATASET
# ==========================================

if os.path.exists(SOURCE_DATASET):

    os.rename(
        SOURCE_DATASET,
        OLD_DATASET
    )

os.rename(
    TEMP_DATASET,
    SOURCE_DATASET
)


# ==========================================
# FINAL COUNTS
# ==========================================

print()
print("==============================================")
print("       DATASET SPLIT COMPLETE")
print("==============================================")


grand_total = 0

for split in [
    "train",
    "validation",
    "test"
]:

    print()
    print(split.upper())
    print("----------------------------------------------")

    split_total = 0

    for class_name in CLASSES:

        folder = os.path.join(
            SOURCE_DATASET,
            split,
            class_name
        )

        files = [
            f
            for f in os.listdir(folder)
            if f.lower().endswith(
                IMAGE_EXTENSIONS
            )
        ]

        count = len(files)

        split_total += count
        grand_total += count

        print(
            f"{class_name}: {count}"
        )

    print(
        f"Total {split}: {split_total}"
    )


print()
print("----------------------------------------------")
print(f"TOTAL DATASET: {grand_total}")
print("----------------------------------------------")

print()
print("==============================================")
print("Dataset is ready for CNN training.")
print("==============================================")
print()