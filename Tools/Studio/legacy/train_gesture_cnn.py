#!/usr/bin/env python3
"""
EdgeSense 3D Gesture Neural Network Trainer
Trains an optimized Multi-Layer Perceptron / Temporal Network
on the Full 3D CNH Tensor dataset, evaluates accuracy, and exports:
  1. gesture_model.json (for live Python visualizer inference)
  2. gesture_model_cortex_m.h (standalone C inference engine for STM32 Cortex-M33)
"""

import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
AI_DIR = SCRIPT_DIR
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Ensure augment_dataset is accessible
sys.path.append(AI_DIR)
from augment_dataset import GESTURE_CLASSES, INV_GESTURE_CLASSES, build_augmented_dataset


def extract_features(X_raw):
    """
    Extracts rich spatial-temporal features from (N, 15, 384) CNH tensors:
    - Temporal Mean and Max photon intensity across the 15-frame window
    - Temporal motion delta (Frame 14 - Frame 0) to capture directional velocity
    - Spatial horizontal gradient (Left columns - Right columns) over time
    - Spatial Push/Approaching energy profile (center zones over time)
    Returns feature matrix of shape (N, feature_dim).
    """
    N, T, F = X_raw.shape  # F = 16 zones * 24 bins = 384

    # 1. Temporal Mean & Max
    f_mean = np.mean(X_raw, axis=1)  # (N, 384)
    f_max = np.max(X_raw, axis=1)    # (N, 384)

    # 2. Reshape to (N, T, 4, 4, 24)
    X_4d = X_raw.reshape(N, T, 4, 4, 24)

    # 3. Column temporal energy centroids (when did each column peak?)
    col_energy = np.sum(X_4d, axis=(2, 4)) # (N, 15, 4)
    weights = col_energy + 1e-5
    frame_indices = np.arange(T).reshape(1, T, 1)
    col_centroids = np.sum(frame_indices * weights, axis=1) / np.sum(weights, axis=1) # (N, 4)
    col_dt = (col_centroids[:, 3] - col_centroids[:, 0]).reshape(-1, 1) # (N, 1)
    col_dt_inner = (col_centroids[:, 2] - col_centroids[:, 1]).reshape(-1, 1) # (N, 1)

    # 4. Column max energy and Left/Right symmetry ratio
    col_max = np.max(col_energy, axis=1) # (N, 4)
    lr_ratio = (np.min(col_max, axis=1) / np.maximum(1e-4, np.max(col_max, axis=1))).reshape(-1, 1)
    tot_energy = np.max(np.sum(col_energy, axis=2), axis=1).reshape(-1, 1)

    # 5. Spatial Left vs Right energy over time
    left_side = np.sum(X_4d[:, :, :, :2, :], axis=(2, 3, 4))   # (N, 15)
    right_side = np.sum(X_4d[:, :, :, 2:, :], axis=(2, 3, 4))  # (N, 15)
    lr_delta = left_side - right_side  # (N, 15)

    # 6. Center Push energy profile
    center_energy = np.sum(X_4d[:, :, 1:3, 1:3, :], axis=(2, 3, 4))  # (N, 15)

    # Concatenate into robust feature vector
    features = np.hstack([f_mean, f_max, col_centroids, col_dt, col_dt_inner, lr_ratio, tot_energy, lr_delta, center_energy])
    return features.astype(np.float32)


def generate_c_header(model, feature_dim, num_classes, output_path=None):
    """
    Generates a pure, standalone C header containing the neural network weights, biases,
    and inference function for immediate inclusion into STM32 Cortex-M33 main.c!
    Zero external dependencies, pure standard C, fully FPU accelerated!
    """
    if output_path is None:
        output_path = os.path.join(AI_DIR, "gesture_model_cortex_m.h")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    coefs = model.coefs_
    intercepts = model.intercepts_

    with open(output_path, "w") as f:
        f.write("/*\n")
        f.write(" * EdgeSense STM32H533 Cortex-M33 3D Gesture Inference Engine\n")
        f.write(" * Automatically generated from trained neural network.\n")
        f.write(" * Architecture: Dense(%d) -> ReLU -> Dense(%d) -> ReLU -> Softmax(%d)\n" % (
            coefs[0].shape[1], coefs[1].shape[1], num_classes
        ))
        f.write(" */\n\n")
        f.write("#ifndef GESTURE_MODEL_CORTEX_M_H\n")
        f.write("#define GESTURE_MODEL_CORTEX_M_H\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <math.h>\n\n")
        f.write(f"#define GESTURE_FEATURE_DIM  {coefs[0].shape[0]}\n")
        f.write(f"#define GESTURE_HIDDEN1_DIM  {coefs[0].shape[1]}\n")
        f.write(f"#define GESTURE_HIDDEN2_DIM  {coefs[1].shape[1]}\n")
        f.write(f"#define GESTURE_NUM_CLASSES  {num_classes}\n\n")

        f.write("// Gesture Class Definitions\n")
        f.write("#define GESTURE_IDLE         0\n")
        f.write("#define GESTURE_SWIPE_LEFT   1\n")
        f.write("#define GESTURE_SWIPE_RIGHT  2\n")
        f.write("#define GESTURE_PUSH         3\n\n")

        # Layer 1 Weights & Biases
        f.write(f"static const float W1[{coefs[0].shape[0]}][{coefs[0].shape[1]}] = {{\n")
        for r in range(coefs[0].shape[0]):
            f.write("  { " + ", ".join(f"{v:.6f}f" for v in coefs[0][r]) + " },\n")
        f.write("};\n\n")

        f.write(f"static const float b1[{len(intercepts[0])}] = {{\n")
        f.write("  " + ", ".join(f"{v:.6f}f" for v in intercepts[0]) + "\n};\n\n")

        # Layer 2 Weights & Biases
        f.write(f"static const float W2[{coefs[1].shape[0]}][{coefs[1].shape[1]}] = {{\n")
        for r in range(coefs[1].shape[0]):
            f.write("  { " + ", ".join(f"{v:.6f}f" for v in coefs[1][r]) + " },\n")
        f.write("};\n\n")

        f.write(f"static const float b2[{len(intercepts[1])}] = {{\n")
        f.write("  " + ", ".join(f"{v:.6f}f" for v in intercepts[1]) + "\n};\n\n")

        # Layer 3 (Output) Weights & Biases
        f.write(f"static const float W3[{coefs[2].shape[0]}][{coefs[2].shape[1]}] = {{\n")
        for r in range(coefs[2].shape[0]):
            f.write("  { " + ", ".join(f"{v:.6f}f" for v in coefs[2][r]) + " },\n")
        f.write("};\n\n")

        f.write(f"static const float b3[{len(intercepts[2])}] = {{\n")
        f.write("  " + ", ".join(f"{v:.6f}f" for v in intercepts[2]) + "\n};\n\n")

        # Standalone C Inference Function
        f.write("""
// Pure C hardware FPU inference function (< 1 millisecond on Cortex-M33 @ 250 MHz)
static inline int EdgeSense_Predict_Gesture(const float *features, float *out_probs, float *out_confidence) {
    float h1[GESTURE_HIDDEN1_DIM];
    float h2[GESTURE_HIDDEN2_DIM];
    float out[GESTURE_NUM_CLASSES];

    // Layer 1: Dense + ReLU
    for (int j = 0; j < GESTURE_HIDDEN1_DIM; j++) {
        float sum = b1[j];
        for (int i = 0; i < GESTURE_FEATURE_DIM; i++) {
            sum += features[i] * W1[i][j];
        }
        h1[j] = (sum > 0.0f) ? sum : 0.0f; // ReLU
    }

    // Layer 2: Dense + ReLU
    for (int j = 0; j < GESTURE_HIDDEN2_DIM; j++) {
        float sum = b2[j];
        for (int i = 0; i < GESTURE_HIDDEN1_DIM; i++) {
            sum += h1[i] * W2[i][j];
        }
        h2[j] = (sum > 0.0f) ? sum : 0.0f; // ReLU
    }

    // Layer 3: Output Logits + Softmax
    float max_logit = -9999.0f;
    for (int j = 0; j < GESTURE_NUM_CLASSES; j++) {
        float sum = b3[j];
        for (int i = 0; i < GESTURE_HIDDEN2_DIM; i++) {
            sum += h2[i] * W3[i][j];
        }
        out[j] = sum;
        if (sum > max_logit) max_logit = sum;
    }

    float exp_sum = 0.0f;
    for (int j = 0; j < GESTURE_NUM_CLASSES; j++) {
        out[j] = expf(out[j] - max_logit);
        exp_sum += out[j];
    }

    int best_class = 0;
    float best_prob = 0.0f;
    for (int j = 0; j < GESTURE_NUM_CLASSES; j++) {
        out[j] /= exp_sum;
        if (out_probs) out_probs[j] = out[j];
        if (out[j] > best_prob) {
            best_prob = out[j];
            best_class = j;
        }
    }

    if (out_confidence) *out_confidence = best_prob;
    return best_class;
}

#endif // GESTURE_MODEL_CORTEX_M_H
""")

    print(f"Generated standalone C inference header: {output_path}")


def train_gesture_model(data_dir=None):
    """
    Loads dataset_train.npz and dataset_val.npz, trains the network, and exports models.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    train_path = os.path.join(data_dir, "dataset_train.npz")
    val_path = os.path.join(data_dir, "dataset_val.npz")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Compiled dataset not found! Running data augmentation first...")
        n_tr, n_val = build_augmented_dataset(output_dir=data_dir)
        if n_tr == 0:
            return False

    data_tr = np.load(train_path)
    data_val = np.load(val_path)

    X_tr_raw, y_train = data_tr["X"], data_tr["y"]
    X_val_raw, y_val = data_val["X"], data_val["y"]

    print(f"\nExtracting spatial-temporal features from {len(X_tr_raw)} train & {len(X_val_raw)} val samples...")
    X_train = extract_features(X_tr_raw)
    X_val = extract_features(X_val_raw)

    print(f"Feature vector shape: {X_train.shape[1]} dimensions per sample")

    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report, accuracy_score

    print("Training Edge Neural Network (Dense 128 -> Dense 64 -> Softmax 4)...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=300,
        alpha=1e-4,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=15
    )

    clf.fit(X_train, y_train)

    y_pred_tr = clf.predict(X_train)
    y_pred_val = clf.predict(X_val)

    acc_tr = accuracy_score(y_train, y_pred_tr)
    acc_val = accuracy_score(y_val, y_pred_val)

    print("\n=======================================================")
    print(f"  Training Accuracy:   {acc_tr * 100:.1f}%")
    print(f"  Validation Accuracy: {acc_val * 100:.1f}%")
    print("=======================================================\n")

    target_names = [INV_GESTURE_CLASSES[i] for i in sorted(INV_GESTURE_CLASSES.keys())]
    print("Validation Classification Report:")
    print(classification_report(y_val, y_pred_val, target_names=target_names, zero_division=0))

    # Save weights as JSON for Python visualizer runtime
    model_json = {
        "classes": target_names,
        "input_dim": int(X_train.shape[1]),
        "coefs": [c.tolist() for c in clf.coefs_],
        "intercepts": [b.tolist() for b in clf.intercepts_],
        "train_acc": float(acc_tr),
        "val_acc": float(acc_val)
    }

    # Save to AI_DIR and to visualizer directory
    destinations = [
        os.path.join(AI_DIR, "gesture_model.json"),
        os.path.join(PROJECT_ROOT, "TOOLS", "visualizer", "gesture_model.json")
    ]
    for p in destinations:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        with open(p, "w") as fp:
            json.dump(model_json, fp, indent=2)
        print(f"Saved model JSON: {p}")

    # Generate standalone C header for Cortex-M33
    c_header_path = os.path.join(AI_DIR, "gesture_model_cortex_m.h")
    generate_c_header(clf, X_train.shape[1], len(target_names), c_header_path)

    # Also copy C header to CODE/EdgeSense/Core/Inc/ if directory exists
    stm32_inc = os.path.join(PROJECT_ROOT, "CODE", "EdgeSense", "Core", "Inc", "gesture_model_cortex_m.h")
    if os.path.isdir(os.path.dirname(stm32_inc)):
        generate_c_header(clf, X_train.shape[1], len(target_names), stm32_inc)

    print("\n>>> SUCCESS: Model trained and ready for Live Visualizer & MCU! <<<")
    return True


if __name__ == "__main__":
    train_gesture_model()
