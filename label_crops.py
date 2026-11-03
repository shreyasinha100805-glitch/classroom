import os
import csv
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageOps

# ==========================================
# CNN DATASET CROP LABELER
# ==========================================

INPUT_FOLDER = "dataset_samples"
LABEL_FILE = "labels.csv"

CLASSES = {
    "b": "board",
    "t": "text",
    "g": "background"
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# ==========================================
# CHECK INPUT FOLDER
# ==========================================

if not os.path.exists(INPUT_FOLDER):

    print("ERROR: dataset_samples folder not found.")
    print("Run create_dataset.py first.")
    exit()

# ==========================================
# FIND CROPS
# ==========================================

crop_files = []

for folder in sorted(os.listdir(INPUT_FOLDER)):

    folder_path = os.path.join(
        INPUT_FOLDER,
        folder
    )

    if not os.path.isdir(folder_path):
        continue

    for filename in sorted(os.listdir(folder_path)):

        if filename.lower().endswith(IMAGE_EXTENSIONS):

            crop_files.append(
                os.path.normpath(
                    os.path.join(
                        folder_path,
                        filename
                    )
                )
            )


if not crop_files:

    print("ERROR: No crop images found.")
    print("Run create_dataset.py first.")
    exit()


# ==========================================
# LOAD EXISTING LABELS
# ==========================================

labels = {}

if os.path.exists(LABEL_FILE):

    with open(
        LABEL_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            path = os.path.normpath(
                row["path"]
            )

            label = row["label"].strip().lower()

            labels[path] = label

    print(
        f"Loaded {len(labels)} existing labels."
    )


# ==========================================
# SAVE LABELS
# ==========================================

def save_labels():

    with open(
        LABEL_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["path", "label"]
        )

        for path in sorted(labels):

            writer.writerow(
                [
                    path,
                    labels[path]
                ]
            )


# ==========================================
# FIND FIRST UNLABELED IMAGE
# ==========================================

current_index = 0

while (
    current_index < len(crop_files)
    and crop_files[current_index] in labels
):

    current_index += 1


# ==========================================
# SHOW STATUS
# ==========================================

print()
print("==============================================")
print("       CNN DATASET CROP LABELER")
print("==============================================")
print()

print(
    "Total crops:",
    len(crop_files)
)

print(
    "Already labeled:",
    len(labels)
)

print(
    "Remaining:",
    len(crop_files) - current_index
)

print()

print("B = Board")
print("T = Text")
print("G = Background")
print("S = Skip")
print("Q = Quit")

print()
print("Starting image labeling window...")
print()


# ==========================================
# TKINTER WINDOW
# ==========================================

root = tk.Tk()

root.title(
    "CNN Dataset Crop Labeler"
)

root.geometry(
    "1100x700"
)

root.resizable(True, True)

root.configure(
    bg="#eeeeee"
)


# ==========================================
# UI
# ==========================================

title_label = tk.Label(
    root,
    text="AI CLASSROOM BOARD - DATASET LABELER",
    font=("Arial", 20, "bold"),
    bg="#eeeeee"
)

title_label.pack(
    pady=10
)


info_label = tk.Label(
    root,
    text="",
    font=("Arial", 15, "bold"),
    bg="#eeeeee"
)

info_label.pack(
    pady=5
)


image_frame = tk.Frame(
    root,
    width=950,
    height=400,
    bg="white",
    relief="solid",
    borderwidth=2
)

image_frame.pack(
    padx=20,
    pady=10
)

image_frame.pack_propagate(False)


image_label = tk.Label(
    image_frame,
    text="Loading...",
    bg="white",
    font=("Arial", 16)
)

image_label.pack(
    expand=True
)


instruction_label = tk.Label(
    root,
    text="Choose the correct category",
    font=("Arial", 14),
    bg="#eeeeee"
)

instruction_label.pack(
    pady=5
)


button_frame = tk.Frame(
    root,
    bg="#eeeeee"
)

button_frame.pack(
    pady=10
)


# ==========================================
# SHOW CROP
# ==========================================

def show_crop():

    global current_index

    if current_index >= len(crop_files):

        info_label.config(
            text=(
                f"ALL CROPS COMPLETED\n"
                f"Total labels: {len(labels)}"
            )
        )

        image_label.config(
            image="",
            text="LABELING COMPLETE",
            font=("Arial", 24, "bold"),
            bg="white"
        )

        setattr(image_label, "image", None)

        messagebox.showinfo(
            "Finished",
            "All crops have been labeled.\n\n"
            "labels.csv has been created."
        )

        return


    crop_path = crop_files[current_index]

    filename = os.path.basename(
        crop_path
    )

    info_label.config(
        text=(
            f"Crop {current_index + 1} / "
            f"{len(crop_files)}\n"
            f"{filename}"
        )
    )


    try:

        image = Image.open(
            crop_path
        )

        image = ImageOps.exif_transpose(
            image
        )

        if image.mode != "RGB":

            image = image.convert(
                "RGB"
            )

        image.thumbnail(
            (900, 380),
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(
            image
        )

        image_label.config(
            image=photo,
            text=""
        )

        setattr(image_label, "image", photo)

    except Exception as error:

        print(
            "ERROR loading image:",
            error
        )

        current_index += 1

        show_crop()


# ==========================================
# LABEL CURRENT CROP
# ==========================================

def label_current(label):

    global current_index

    if current_index >= len(crop_files):

        return


    crop_path = crop_files[
        current_index
    ]


    # Save label
    labels[crop_path] = label

    # Immediately save CSV
    save_labels()


    print(
        f"[{label.upper()}] "
        f"{os.path.basename(crop_path)}"
    )


    current_index += 1

    show_crop()


# ==========================================
# QUIT
# ==========================================

def quit_program():

    # Save before quitting
    save_labels()

    root.destroy()


# ==========================================
# BUTTONS
# ==========================================

board_button = tk.Button(
    button_frame,
    text="B - BOARD",
    font=("Arial", 14, "bold"),
    width=15,
    height=2,
    command=lambda:
        label_current("board")
)

board_button.grid(
    row=0,
    column=0,
    padx=10
)


text_button = tk.Button(
    button_frame,
    text="T - TEXT",
    font=("Arial", 14, "bold"),
    width=15,
    height=2,
    command=lambda:
        label_current("text")
)

text_button.grid(
    row=0,
    column=1,
    padx=10
)


background_button = tk.Button(
    button_frame,
    text="G - BACKGROUND",
    font=("Arial", 14, "bold"),
    width=15,
    height=2,
    command=lambda:
        label_current("background")
)

background_button.grid(
    row=0,
    column=2,
    padx=10
)


skip_button = tk.Button(
    button_frame,
    text="S - SKIP",
    font=("Arial", 14, "bold"),
    width=15,
    height=2,
    command=lambda:
        label_current("skip")
)

skip_button.grid(
    row=0,
    column=3,
    padx=10
)


quit_button = tk.Button(
    button_frame,
    text="Q - QUIT",
    font=("Arial", 14, "bold"),
    width=15,
    height=2,
    command=quit_program
)

quit_button.grid(
    row=0,
    column=4,
    padx=10
)


# ==========================================
# KEYBOARD SHORTCUTS
# ==========================================

root.bind(
    "<KeyPress-b>",
    lambda event:
        label_current("board")
)

root.bind(
    "<KeyPress-t>",
    lambda event:
        label_current("text")
)

root.bind(
    "<KeyPress-g>",
    lambda event:
        label_current("background")
)

root.bind(
    "<KeyPress-s>",
    lambda event:
        label_current("skip")
)

root.bind(
    "<KeyPress-q>",
    lambda event:
        quit_program()
)


# ==========================================
# START
# ==========================================

show_crop()

root.mainloop()


print()
print("==============================================")
print("           LABELING FINISHED")
print("==============================================")
print()
print("Labels saved to:")
print(LABEL_FILE)
print()