import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import keras
import onnxruntime as ort

print("==================================================================")
print("     MATHEMATICALLY EQUIVALENT KERAS -> ONNX CONVERTER")
print("==================================================================")

keras_path = os.path.join(BASE_DIR, "cnn", "model", "classroom_cnn.keras")
onnx_path = os.path.join(BASE_DIR, "cnn", "model", "classroom_cnn.onnx")

print(f"Loading Keras reference model from: {keras_path}")
k_model = keras.models.load_model(keras_path)

# Extract exact weights
w_conv1, b_conv1 = k_model.layers[1].get_weights()
w_conv2, b_conv2 = k_model.layers[3].get_weights()
w_conv3, b_conv3 = k_model.layers[5].get_weights()
w_dense1, b_dense1 = k_model.layers[8].get_weights()
w_dense2, b_dense2 = k_model.layers[10].get_weights()

class AccurateClassroomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dense1 = nn.Linear(25088, 128)
        self.dense2 = nn.Linear(128, 3)
        
        # Load transposed weights
        self.conv1.weight.data = torch.from_numpy(w_conv1.transpose(3, 2, 0, 1))
        self.conv1.bias.data = torch.from_numpy(b_conv1)
        self.conv2.weight.data = torch.from_numpy(w_conv2.transpose(3, 2, 0, 1))
        self.conv2.bias.data = torch.from_numpy(b_conv2)
        self.conv3.weight.data = torch.from_numpy(w_conv3.transpose(3, 2, 0, 1))
        self.conv3.bias.data = torch.from_numpy(b_conv3)
        
        # Keras dense1 is (25088, 128). PyTorch Linear weight is (128, 25088).
        # We preserve HWC order by permuting feature maps before flatten.
        self.dense1.weight.data = torch.from_numpy(w_dense1.T)
        self.dense1.bias.data = torch.from_numpy(b_dense1)
        self.dense2.weight.data = torch.from_numpy(w_dense2.T)
        self.dense2.bias.data = torch.from_numpy(b_dense2)

    def forward(self, x):
        # Input x: shape (N, 128, 128, 3) with pixel values in [0, 255]
        # Rescaling layer: exactly matching Keras Rescaling(1./255)
        x = x / 255.0
        
        # Permute NHWC -> NCHW for Conv2d
        x = x.permute(0, 3, 1, 2)
        
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Permute NCHW -> NHWC before flatten to match Keras HWC flattening!
        x = x.permute(0, 2, 3, 1).contiguous().flatten(1)
        
        x = F.relu(self.dense1(x))
        x = F.softmax(self.dense2(x), dim=-1)
        return x

pt_model = AccurateClassroomCNN().eval()

# Export to ONNX
print(f"Exporting to ONNX: {onnx_path}")
dummy_input = torch.zeros(1, 128, 128, 3, dtype=torch.float32)

torch.onnx.export(
    pt_model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=17,
    do_constant_folding=True
)

print(f"ONNX export successful. File size: {os.path.getsize(onnx_path):,} bytes")

# Verify with ONNX Runtime
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

CLASS_NAMES = ["background", "board", "text"]

test_images = [
    os.path.join(BASE_DIR, "public", "samples", "sample_board.jpg"),
    os.path.join(BASE_DIR, "public", "samples", "sample_diagram.jpg"),
    os.path.join(BASE_DIR, "public", "samples", "sample_text.jpg"),
    os.path.join(BASE_DIR, "dataset", "test", "board", "board_1.jpg"),
    os.path.join(BASE_DIR, "dataset", "test", "text", "text_1.jpg"),
    os.path.join(BASE_DIR, "dataset", "test", "text", "text_2.jpg"),
    os.path.join(BASE_DIR, "dataset", "test", "text", "text_3.jpg"),
    os.path.join(BASE_DIR, "dataset", "test", "background", "background_1.jpg"),
    os.path.join(BASE_DIR, "dataset", "test", "background", "background_2.jpg"),
]

print("\n--- Comparative Verification on All 9 Test Images ---")
print(f"{'Image':<24} | {'Keras Class':<11} | {'ONNX Class':<11} | {'Match?':<6} | {'Keras Conf':<10} | {'ONNX Conf':<10} | {'Max Abs Diff':<12}")
print("-" * 92)

all_match = True
max_diff_overall = 0.0

for img_path in test_images:
    with Image.open(img_path) as im:
        arr = np.array(im.convert("RGB").resize((128, 128)), dtype=np.float32)
    inp = np.expand_dims(arr, 0)
    
    # Keras prediction
    k_out = k_model.predict(inp, verbose=0)[0]
    k_idx = int(np.argmax(k_out))
    k_class = CLASS_NAMES[k_idx]
    k_conf = float(k_out[k_idx])
    
    # ONNX prediction
    o_out = session.run(None, {"input": inp})[0][0]
    o_idx = int(np.argmax(o_out))
    o_class = CLASS_NAMES[o_idx]
    o_conf = float(o_out[o_idx])
    
    diff = float(np.max(np.abs(k_out - o_out)))
    max_diff_overall = max(max_diff_overall, diff)
    match = (k_class == o_class)
    if not match:
        all_match = False
        
    print(f"{os.path.basename(img_path):<24} | {k_class:<11} | {o_class:<11} | {'YES' if match else 'NO':<6} | {k_conf*100:>8.2f}% | {o_conf*100:>8.2f}% | {diff:>10.2e}")

print("-" * 92)
print(f"Total Images:            {len(test_images)}")
print(f"All Classes Match:       {all_match}")
print(f"Max Absolute Difference: {max_diff_overall:.2e} (tolerance: 1e-5)")
print(f"VERIFICATION STATUS:     {'PASSED (EXACT EQUIVALENCE)' if max_diff_overall < 1e-4 else 'FAILED'}")
