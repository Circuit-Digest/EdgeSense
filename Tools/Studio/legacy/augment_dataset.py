#!/usr/bin/env python3
"""
EdgeSense ST MZAI-Inspired Data Augmentation & Preprocessing Engine
Loads raw gesture recordings, applies horizontal spatial mirroring, temporal warping,
and distance normalization, then outputs compiled training & validation datasets.
"""

import os
import sys
import glob
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "gestures")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")

# Mapping for 4x4 spatial horizontal flip (col -> 3 - col)
# Zone r*4 + c -> r*4 + (3 - c)
FLIP_LR_MAP = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]

GESTURE_CLASSES = {
    "idle": 0,
    "swipe_left": 1,
    "swipe_right": 2,
    "push": 3
}

INV_GESTURE_CLASSES = {v: k for k, v in GESTURE_CLASSES.items()}


def flip_sample_lr(sample_cnh, sample_dist, label):
    """
    Horizontally flips a (15, 16, 24) CNH tensor and (15, 16) dist tensor.
    Swaps swipe_left <-> swipe_right labels.
    """
    flipped_cnh = sample_cnh[:, FLIP_LR_MAP, :]
    flipped_dist = sample_dist[:, FLIP_LR_MAP]

    new_label = label
    if label == 1:  # swipe_left -> swipe_right
        new_label = 2
    elif label == 2:  # swipe_right -> swipe_left
        new_label = 1

    return flipped_cnh, flipped_dist, new_label


def temporal_resample(sample_cnh, sample_dist, scale=1.0):
    """
    Resamples 15-frame sequence across time by scale factor (e.g. 0.9x or 1.1x).
    """
    T, Z, B = sample_cnh.shape
    orig_times = np.linspace(0, 1, T)
    new_times = np.linspace(0, 1, T) * scale
    new_times = np.clip(new_times, 0, 1)

    new_cnh = np.zeros_like(sample_cnh)
    new_dist = np.zeros_like(sample_dist)

    for z in range(Z):
        for b in range(B):
            new_cnh[:, z, b] = np.interp(orig_times, new_times, sample_cnh[:, z, b])
        new_dist[:, z] = np.interp(orig_times, new_times, sample_dist[:, z])

    return new_cnh, new_dist


def add_noise_jitter(sample_cnh, noise_std=1.5):
    """
    Adds small photon shot noise jitter to improve model generalization.
    """
    noise = np.random.normal(0, noise_std, sample_cnh.shape)
    jittered = np.clip(sample_cnh + noise, 0, None)
    return jittered


def load_raw_dataset(data_dir=None):
    """
    Loads all .npz and .csv gesture recordings from data/gestures/<class_name>/.
    Checks both project root data/gestures and visualizer/data/gestures.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    candidate_dirs = [
        data_dir,
        os.path.join(PROJECT_ROOT, "TOOLS", "visualizer", "data", "gestures")
    ]

    raw_samples = []
    seen_files = set()

    for ddir in candidate_dirs:
        if not os.path.isdir(ddir):
            continue

        for class_name, label in GESTURE_CLASSES.items():
            class_dir = os.path.join(ddir, class_name)
            if not os.path.isdir(class_dir):
                continue

            npz_files = glob.glob(os.path.join(class_dir, "*.npz"))
            for f in npz_files:
                fname = os.path.basename(f)
                if fname in seen_files:
                    continue
                seen_files.add(fname)
                try:
                    data = np.load(f)
                    cnh = data["cnh"]
                    dist = data["dist"]
                    if cnh.shape == (15, 384):
                        cnh = cnh.reshape(15, 16, 24)
                    raw_samples.append((cnh, dist, label))
                except Exception as e:
                    print(f"Warning: Could not load {f}: {e}")

            csv_files = glob.glob(os.path.join(class_dir, "*.csv"))
            for f in csv_files:
                fname = os.path.basename(f)
                if fname in seen_files:
                    continue
                seen_files.add(fname)
                try:
                    rows = []
                    with open(f, "r") as fp:
                        for line in fp:
                            parts = [float(p) for p in line.strip().split(",") if p]
                            if len(parts) >= 384:
                                rows.append(parts[:384])
                    if len(rows) >= 15:
                        cnh = np.array(rows[:15]).reshape(15, 16, 24)
                        dist = np.full((15, 16), 500.0)
                        raw_samples.append((cnh, dist, label))
                except Exception as e:
                    print(f"Warning: Could not load {f}: {e}")

    return raw_samples


def build_augmented_dataset(data_dir=None, output_dir=None, val_split=0.2):
    """
    Applies spatial and temporal augmentations and saves dataset_train.npz and dataset_val.npz.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    raw_samples = load_raw_dataset(data_dir)
    if not raw_samples:
        print(f"No samples found in {data_dir}! Please record gestures in Gesture Studio first.")
        return 0, 0

    print(f"Found {len(raw_samples)} raw gesture recordings.")
    class_counts = {k: 0 for k in GESTURE_CLASSES.keys()}
    for _, _, l in raw_samples:
        class_counts[INV_GESTURE_CLASSES[l]] += 1

    for cname, cnt in class_counts.items():
        print(f"  - {cname:12s}: {cnt:3d} recordings")

    augmented_X = []
    augmented_y = []

    for cnh, dist, label in raw_samples:
        # 1. Original sample
        augmented_X.append(cnh.reshape(15, 384))
        augmented_y.append(label)

        # 2. Horizontal Flip (mirrors swipe_left <-> swipe_right)
        f_cnh, f_dist, f_label = flip_sample_lr(cnh, dist, label)
        augmented_X.append(f_cnh.reshape(15, 384))
        augmented_y.append(f_label)

        # 3. Fast speed variation (0.9x) + slight noise
        fast_cnh, _ = temporal_resample(cnh, dist, 0.9)
        augmented_X.append(add_noise_jitter(fast_cnh).reshape(15, 384))
        augmented_y.append(label)

        # 4. Slow speed variation (1.1x) + slight noise
        slow_cnh, _ = temporal_resample(cnh, dist, 1.1)
        augmented_X.append(add_noise_jitter(slow_cnh).reshape(15, 384))
        augmented_y.append(label)

        # 5. Flipped with speed variation
        f_fast_cnh, _ = temporal_resample(f_cnh, f_dist, 0.9)
        augmented_X.append(add_noise_jitter(f_fast_cnh).reshape(15, 384))
        augmented_y.append(f_label)

    X = np.array(augmented_X, dtype=np.float32)
    y = np.array(augmented_y, dtype=np.int64)

    # Normalize input features by max photon intensity scale (~250 kcps)
    X = X / 250.0

    # Shuffle and split into Train and Validation sets
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)

    val_count = max(4, int(len(X) * val_split))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "dataset_train.npz")
    val_path = os.path.join(output_dir, "dataset_val.npz")

    np.savez_compressed(train_path, X=X_train, y=y_train)
    np.savez_compressed(val_path, X=X_val, y=y_val)

    print("\n=== Dataset Compilation Complete ===")
    print(f"Augmented Total: {len(X)} samples (from {len(raw_samples)} raw)")
    print(f"Train Set: {X_train.shape} -> saved to {train_path}")
    print(f"Val Set:   {X_val.shape} -> saved to {val_path}")

    return len(X_train), len(X_val)


if __name__ == "__main__":
    build_augmented_dataset()
