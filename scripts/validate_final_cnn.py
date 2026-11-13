import os
import sys
import hashlib
import time
import json
import tempfile
import numpy as np
from PIL import Image

# Ensure repository root is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import psutil
import onnxruntime as ort

CLASS_NAMES = ["background", "board", "text"]
IMAGE_SIZE = (128, 128)

def get_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def run_onnx_inference(session, image_path):
    with Image.open(image_path) as pil_img:
        rgb = pil_img.convert("RGB").resize(IMAGE_SIZE)
        image_array = np.array(rgb, dtype=np.float32)  # [0, 255] float32
    image_array = np.expand_dims(image_array, axis=0) # (1, 128, 128, 3)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: image_array})[0][0]
    pred_idx = int(np.argmax(output))
    return {
        "class": CLASS_NAMES[pred_idx],
        "confidence": float(output[pred_idx]),
        "probabilities": {CLASS_NAMES[i]: float(output[i]) for i in range(len(CLASS_NAMES))},
        "raw_probs": [float(p) for p in output]
    }

def run_keras_inference(model, image_path):
    with Image.open(image_path) as pil_img:
        rgb = pil_img.convert("RGB").resize(IMAGE_SIZE)
        image_array = np.array(rgb, dtype=np.float32)  # [0, 255] float32
    image_array = np.expand_dims(image_array, axis=0)
    output = model.predict(image_array, verbose=0)[0]
    pred_idx = int(np.argmax(output))
    return {
        "class": CLASS_NAMES[pred_idx],
        "confidence": float(output[pred_idx]),
        "probabilities": {CLASS_NAMES[i]: float(output[i]) for i in range(len(CLASS_NAMES))},
        "raw_probs": [float(p) for p in output]
    }

def main():
    print("==================================================================")
    print("   FINAL VALIDATION: ONNX vs KERAS CNN MODEL VERIFICATION")
    print("==================================================================")
    
    keras_path = os.path.join(BASE_DIR, "cnn", "model", "classroom_cnn.keras")
    onnx_path = os.path.join(BASE_DIR, "cnn", "model", "classroom_cnn.onnx")
    
    # 1. File verification & SHA-256
    print("\n--- 1. MODEL FILE INTEGRITY & CHECKSUMS ---")
    keras_exists = os.path.exists(keras_path)
    onnx_exists = os.path.exists(onnx_path)
    
    print(f"Keras model path:  {keras_path}")
    print(f"  Exists:          {keras_exists}")
    if keras_exists:
        keras_size = os.path.getsize(keras_path)
        keras_hash = get_file_sha256(keras_path)
        print(f"  Size (bytes):    {keras_size:,} ({keras_size / (1024*1024):.2f} MB)")
        print(f"  SHA-256:         {keras_hash}")
        
    print(f"ONNX model path:   {onnx_path}")
    print(f"  Exists:          {onnx_exists}")
    if onnx_exists:
        onnx_size = os.path.getsize(onnx_path)
        onnx_hash = get_file_sha256(onnx_path)
        print(f"  Size (bytes):    {onnx_size:,} ({onnx_size / (1024*1024):.2f} MB)")
        print(f"  SHA-256:         {onnx_hash}")

    # 2. Tensor contract & metadata
    print("\n--- 2. ONNX TENSOR CONTRACT & ARCHITECTURE ---")
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    onnx_session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    
    input_meta = onnx_session.get_inputs()[0]
    output_meta = onnx_session.get_outputs()[0]
    print(f"ONNX Input Name:   {input_meta.name}")
    print(f"ONNX Input Shape:  {input_meta.shape}")
    print(f"ONNX Input Type:   {input_meta.type}")
    print(f"ONNX Output Name:  {output_meta.name}")
    print(f"ONNX Output Shape: {output_meta.shape}")
    print(f"ONNX Output Type:  {output_meta.type}")

    # 3. Load Keras model
    print("\n--- 3. KERAS MODEL LOADING & INPUT CONTRACT ---")
    try:
        import keras
        keras_model = keras.models.load_model(keras_path)
        print(f"Keras Input Shape:  {keras_model.input_shape}")
        print(f"Keras Output Shape: {keras_model.output_shape}")
        keras_available = True
    except Exception as e:
        print(f"Keras loading failed: {e}")
        keras_available = False
        keras_model = None

    # 4. Comparative evaluation on representative real classroom test images
    print("\n--- 4. COMPARATIVE INFERENCE ON REAL TEST IMAGES ---")
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
    
    comparisons = []
    print(f"{'Image':<26} | {'Keras Pred':<11} | {'ONNX Pred':<11} | {'Match?':<6} | {'Keras Conf':<10} | {'ONNX Conf':<10} | {'Max Abs Diff':<12}")
    print("-" * 96)
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
        rel_path = os.path.relpath(img_path, BASE_DIR)
        
        onnx_res = run_onnx_inference(onnx_session, img_path)
        
        if keras_available:
            keras_res = run_keras_inference(keras_model, img_path)
            k_class = keras_res["class"]
            o_class = onnx_res["class"]
            k_conf = keras_res["confidence"]
            o_conf = onnx_res["confidence"]
            diffs = [abs(k - o) for k, o in zip(keras_res["raw_probs"], onnx_res["raw_probs"])]
            max_diff = max(diffs)
            match = "YES" if k_class == o_class else "NO"
            
            comparisons.append({
                "image": rel_path,
                "keras_class": k_class,
                "onnx_class": o_class,
                "match": match == "YES",
                "keras_conf": k_conf,
                "onnx_conf": o_conf,
                "max_diff": max_diff,
                "keras_probs": keras_res["probabilities"],
                "onnx_probs": onnx_res["probabilities"]
            })
            
            print(f"{os.path.basename(rel_path):<26} | {k_class:<11} | {o_class:<11} | {match:<6} | {k_conf*100:>8.2f}% | {o_conf*100:>8.2f}% | {max_diff:>10.2e}")
        else:
            print(f"{os.path.basename(rel_path):<26} | {'N/A':<11} | {onnx_res['class']:<11} | {'N/A':<6} | {'N/A':<10} | {onnx_res['confidence']*100:>8.2f}% | {'N/A':<12}")

    # Summary of comparison
    if comparisons:
        all_match = all(c["match"] for c in comparisons)
        max_overall_diff = max(c["max_diff"] for c in comparisons)
        print("-" * 96)
        print(f"Total Test Images Evaluated: {len(comparisons)}")
        print(f"Class Agreement:             {sum(1 for c in comparisons if c['match'])}/{len(comparisons)} ({100*sum(1 for c in comparisons if c['match'])/len(comparisons):.1f}%)")
        print(f"Max Absolute Softmax Diff:   {max_overall_diff:.2e} (tolerance: 1e-4)")
        print(f"Numerical Equivalence:       {'PERFECT (within floating point precision)' if max_overall_diff < 1e-4 else 'DISCREPANCY DETECTED'}")

    # 5. Robustness & Error Handling Tests
    print("\n--- 5. ERROR HANDLING & ROBUSTNESS TESTS ---")
    from cnn.predict import predict_image
    
    # 5a. Missing image file
    try:
        predict_image("non_existent_image_file_12345.jpg")
        print("FAIL: Missing image did not raise FileNotFoundError")
    except FileNotFoundError as e:
        print(f"PASS: Missing image raised FileNotFoundError: {e}")
    except Exception as e:
        print(f"FAIL: Missing image raised unexpected error: {type(e).__name__}: {e}")

    # 5b. Corrupted image file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tf.write(b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_BYTES_1234567890")
        corrupted_path = tf.name
    try:
        predict_image(corrupted_path)
        print("FAIL: Corrupted image did not raise exception")
    except Exception as e:
        print(f"PASS: Corrupted image safely rejected: {type(e).__name__}: {e}")
    finally:
        if os.path.exists(corrupted_path):
            os.remove(corrupted_path)

    # 5c. Missing model file handling
    print("Testing missing model environment variable behavior:")
    old_env = os.environ.get("CNN_MODEL_PATH")
    os.environ["CNN_MODEL_PATH"] = "non_existent_model_file.onnx"
    # Reset singleton to test loader
    import cnn.predict as predict_module
    saved_session = predict_module._session
    predict_module._session = None
    try:
        predict_module.get_model()
        print("FAIL: Missing model did not raise FileNotFoundError")
    except FileNotFoundError as e:
        print(f"PASS: Missing model raised FileNotFoundError as expected: {e}")
    finally:
        predict_module._session = saved_session
        if old_env:
            os.environ["CNN_MODEL_PATH"] = old_env
        else:
            os.environ.pop("CNN_MODEL_PATH", None)

    # 5d. Repeated inference stability (20 iterations)
    print("\n--- 6. REPEATED INFERENCE & STABILITY TEST (20 ITERATIONS) ---")
    sample_img = os.path.join(BASE_DIR, "public", "samples", "sample_board.jpg")
    latencies = []
    
    for i in range(20):
        t0 = time.perf_counter()
        res = predict_image(sample_img)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        
    avg_lat = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    print(f"20 Inferences completed successfully. Output class: '{res['class']}' (confidence: {res['confidence']*100:.2f}%)")
    print(f"Latency: Avg={avg_lat:.2f} ms | Min={min_lat:.2f} ms | Max={max_lat:.2f} ms")

    # 7. Local Memory Usage Measurement
    print("\n--- 7. ACTUAL MEASURED PROCESS MEMORY USAGE ---")
    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    vms_mb = mem_info.vms / (1024 * 1024)
    print(f"Current Process RSS: {rss_mb:.2f} MB ({mem_info.rss:,} bytes)")
    print(f"Current Process VMS: {vms_mb:.2f} MB ({mem_info.vms:,} bytes)")

    print("\n==================================================================")
    print("                  VALIDATION COMPLETE")
    print("==================================================================")

if __name__ == "__main__":
    main()
