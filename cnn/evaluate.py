import os
import sys
import time
import numpy as np
from PIL import Image

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from cnn.predict import predict_image, CLASS_NAMES

def evaluate_test_set():
    test_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "test")
    if not os.path.exists(test_dir):
        print(f"Error: test dataset directory not found at {test_dir}")
        return

    print("==================================================================")
    print("       CNN MODEL TEST SET EVALUATION (REAL METRICS)")
    print("==================================================================")

    y_true = []
    y_pred = []
    latencies = []
    file_records = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(test_dir, class_name)
        if not os.path.exists(class_folder):
            continue

        images = [f for f in os.listdir(class_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for img_name in images:
            img_path = os.path.join(class_folder, img_name)
            
            t0 = time.perf_counter()
            res = predict_image(img_path)
            t1 = time.perf_counter()
            
            pred_class = res["class"]
            pred_idx = CLASS_NAMES.index(pred_class) if pred_class in CLASS_NAMES else -1
            
            latencies.append((t1 - t0) * 1000) # ms
            y_true.append(class_idx)
            y_pred.append(pred_idx)
            file_records.append((img_name, class_name, pred_class, res["confidence"]))

    n_samples = len(y_true)
    if n_samples == 0:
        print("No test samples found.")
        return

    # Accuracy
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / n_samples

    # Confusion matrix
    num_classes = len(CLASS_NAMES)
    conf_matrix = [[0] * num_classes for _ in range(num_classes)]
    for yt, yp in zip(y_true, y_pred):
        if 0 <= yp < num_classes:
            conf_matrix[yt][yp] += 1

    # Precision, Recall, F1 per class
    metrics_per_class = {}
    for i, cname in enumerate(CLASS_NAMES):
        tp = conf_matrix[i][i]
        fp = sum(conf_matrix[r][i] for r in range(num_classes) if r != i)
        fn = sum(conf_matrix[i][c] for c in range(num_classes) if c != i)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        support = sum(conf_matrix[i])
        metrics_per_class[cname] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support
        }

    # Macro averages
    macro_prec = sum(m["precision"] for m in metrics_per_class.values()) / num_classes
    macro_rec = sum(m["recall"] for m in metrics_per_class.values()) / num_classes
    macro_f1 = sum(m["f1"] for m in metrics_per_class.values()) / num_classes
    avg_latency = sum(latencies) / len(latencies)

    print(f"\nTotal Test Samples Evaluated: {n_samples}")
    print(f"Overall Accuracy:            {accuracy * 100:.2f}% ({correct}/{n_samples})")
    print(f"Macro Average Precision:     {macro_prec * 100:.2f}%")
    print(f"Macro Average Recall:        {macro_rec * 100:.2f}%")
    print(f"Macro Average F1-Score:      {macro_f1 * 100:.2f}%")
    print(f"Average Inference Latency:   {avg_latency:.2f} ms/image\n")

    print("Class-wise Breakdown:")
    print(f"{'Class':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 62)
    for cname, m in metrics_per_class.items():
        print(f"{cname:<14} | {m['precision']*100:>8.2f}% | {m['recall']*100:>8.2f}% | {m['f1']*100:>8.2f}% | {m['support']:>8}")

    print("\nConfusion Matrix (Rows: Actual, Columns: Predicted):")
    header = "Actual \\ Pred  | " + " | ".join(f"{c:<10}" for c in CLASS_NAMES)
    print(header)
    print("-" * len(header))
    for i, cname in enumerate(CLASS_NAMES):
        row_str = " | ".join(f"{conf_matrix[i][j]:>10}" for j in range(num_classes))
        print(f"{cname:<14} | {row_str}")

    print("==================================================================")
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "avg_latency_ms": avg_latency,
        "confusion_matrix": conf_matrix,
        "metrics_per_class": metrics_per_class
    }

if __name__ == "__main__":
    evaluate_test_set()
