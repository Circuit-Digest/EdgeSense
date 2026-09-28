#!/usr/bin/env python3
"""
edgesense_nn_studio.py
EdgeSense Neural Network Studio & Real-Time Sensor Interface

EdgeSense: STM32 Edge AI Texture and Gesture Classifier

Copyright (c) 2026 Dharagesh and Circuit Digest
https://github.com/Circuit-Digest/EdgeSense
Licensed under GNU General Public License v3.0

  ███████╗██████╗  ██████╗ ███████╗███████╗███████╗███╗   ██╗███████╗███████╗
  ██╔════╝██╔══██╗██╔════╝ ██╔════╝██╔════╝██╔════╝████╗  ██║██╔════╝██╔════╝
  █████╗  ██║  ██║██║  ███╗█████╗  ███████╗█████╗  ██╔██╗ ██║███████╗█████╗  
  ██╔══╝  ██║  ██║██║   ██║██╔══╝  ╚════██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝  
  ███████╗██████╔╝╚██████╔╝███████╗███████║███████╗██║ ╚████║███████║███████╗
  ╚══════╝╚═════╝  ╚═════╝ ╚══════╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝

Features:
  - 3D Gesture Recognition (8x8 Depth Matrix, Temporal Windowing)
  - Distance-Invariant Surface Classification (4x4 CNH Histogram, 1D-CNN)
  - Static Hand Posture Classification (8x8 Depth + Signal, 2D-CNN)
  - Optical Scattering Smoke & Particle Obscuration Detection
  - Real-time Visualizer, Dataset Recorder, and ST Edge AI Code Generator
"""


# PyTorch imported FIRST to prevent Windows C-Runtime DLL conflicts with Qt
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError, Exception):
    TORCH_AVAILABLE = False

import sys
import os
import time
import json
import subprocess
import numpy as np
import serial
import serial.tools.list_ports
from collections import deque
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QProgressBar,
    QGroupBox, QComboBox, QTextEdit, QFileDialog, QFrame, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy, QShortcut,
    QMessageBox, QStackedWidget, QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import (
    QColor, QFont, QPalette, QPainter, QPen, QBrush,
    QLinearGradient, QPolygonF, QKeySequence
)

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")
DATA_DIR = MODELS_DIR  # Production models and datasets
os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Constants & Classes
# ----------------------------------------------------------------------
# 1. Gesture Constants
GESTURE_CLASSES = ["IDLE", "SWIPE_LEFT", "SWIPE_RIGHT", "SWIPE_UP", "SWIPE_DOWN"]
GESTURE_COLORS = {
    "IDLE": "#475569",        # Slate Gray
    "SWIPE_LEFT": "#1d4ed8",   # Academic Blue
    "SWIPE_RIGHT": "#047857",  # Emerald Green
    "SWIPE_UP": "#d97706",     # Amber / Ochre
    "SWIPE_DOWN": "#b91c1c"    # Crimson Red
}
GESTURE_WINDOW_SIZE = 15  # 15 frames @ 15 FPS = 1.0 s window
GESTURE_DATA_FILE = os.path.join(MODELS_DIR, "gesture_dataset.npz")
GESTURE_ONNX_FILE = os.path.join(MODELS_DIR, "gesture_nn_model.onnx")

# 2. Surface Constants (ST-Aligned Physics)
SURFACE_CLASSES = ["HARD_FLOOR", "CARPET", "SPECULAR", "VOID"]
SURFACE_COLORS = {
    "HARD_FLOOR": "#0284c7",   # Ocean Blue (Tile / Wood / Laminate)
    "CARPET": "#d97706",       # Amber (Subsurface scattering, broadened pulse)
    "SPECULAR": "#7c3aed",     # Purple (Polished metal / Mirror)
    "VOID": "#64748b"          # Slate (Cliff / Drop-off / Open air)
}
SURFACE_NUM_BINS = 48
SURFACE_CANONICAL_BINS = 16  # Canonical peak-centered window
SURFACE_PRE_PEAK = 4         # Bins before peak (index 4 is ALWAYS the peak)
SURFACE_BIN_WIDTH_MM = 37.46 # Physical bin width from ST cnh_lib.py
SURFACE_DATA_FILE = os.path.join(MODELS_DIR, "surface_dataset.npz")
SURFACE_ONNX_FILE = os.path.join(MODELS_DIR, "surface_nn_model.onnx")

# 3. Hand Posture Constants (STSW-IMG050 Aligned)
POSTURE_CLASSES = ["NONE", "FLAT_HAND", "LIKE", "DISLIKE", "BREAK_TIME", "FIST"]
POSTURE_COLORS = {
    "NONE": "#64748b",        # Slate Gray
    "FLAT_HAND": "#0284c7",   # Sky Blue (Open Palm)
    "LIKE": "#059669",        # Emerald Green (Thumbs Up)
    "DISLIKE": "#dc2626",     # Crimson Red (Thumbs Down)
    "BREAK_TIME": "#d97706",  # Amber (Timeout T-sign)
    "FIST": "#7c3aed"         # Purple (Closed Fist)
}
POSTURE_ICONS = {
    "NONE": "🚫",
    "FLAT_HAND": "🖐️",
    "LIKE": "👍",
    "DISLIKE": "👎",
    "BREAK_TIME": "⏸️",
    "FIST": "✊"
}
POSTURE_DATA_FILE = os.path.join(MODELS_DIR, "posture_dataset.npz")
POSTURE_ONNX_FILE = os.path.join(MODELS_DIR, "posture_nn_model.onnx")

POSTURE_RANGING_CENTER = 295.0
POSTURE_RANGING_IQR = 196.0
POSTURE_SIGNAL_CENTER = 281.0
POSTURE_SIGNAL_IQR = 452.0
POSTURE_MIN_DIST = 100.0
POSTURE_MAX_DIST = 400.0
POSTURE_BG_REMOVAL = 120.0



def normalize_gesture_window(window_15x64):
    """
    Normalizes a 15-frame window of 64-zone ToF distances into a (15, 8, 8) planar tensor.
    - Background / invalid / negative / out-of-range (> 400 mm): strictly 0.0
    - Hand presence (50 <= d <= 400 mm): (400.0 - d) / 350.0 in (0.0, 1.0]
    """
    arr = np.array(window_15x64, dtype=np.float32)
    norm = np.zeros_like(arr)
    valid_mask = (arr >= 50.0) & (arr <= 400.0)
    norm[valid_mask] = (400.0 - arr[valid_mask]) / 350.0
    return norm.reshape(15, 8, 8)


# ----------------------------------------------------------------------
# Gesture Feature Extractor: Spatial-Temporal Depth Physics
# ----------------------------------------------------------------------
def extract_gesture_features(window_15x64):
    grid = normalize_gesture_window(window_15x64)
    norm = grid.reshape(15, 64)

    f_mean = np.mean(norm, axis=0)
    f_max = np.max(norm, axis=0)
    f_delta = norm[-1] - norm[0]

    grid = norm.reshape(15, 8, 8)

    col_energy = np.sum(grid, axis=1)
    weights_x = col_energy + 1e-4
    t_indices = np.arange(15).reshape(15, 1)
    col_centroids = np.sum(t_indices * weights_x, axis=0) / np.sum(weights_x, axis=0)
    col_dx = col_centroids[7] - col_centroids[0]

    row_energy = np.sum(grid, axis=2)
    weights_y = row_energy + 1e-4
    row_centroids = np.sum(t_indices * weights_y, axis=0) / np.sum(weights_y, axis=0)
    row_dy = row_centroids[7] - row_centroids[0]

    left_side = np.sum(col_energy[:, :4], axis=1)
    right_side = np.sum(col_energy[:, 4:], axis=1)
    lr_delta = left_side - right_side

    top_side = np.sum(row_energy[:, :4], axis=1)
    bot_side = np.sum(row_energy[:, 4:], axis=1)
    tb_delta = top_side - bot_side

    peak_val = float(np.max(np.sum(norm, axis=1)))

    features = np.concatenate([
        f_mean, f_max, f_delta,
        col_centroids, [float(col_dx)],
        row_centroids, [float(row_dy)],
        lr_delta, tb_delta,
        [peak_val]
    ])
    return features.astype(np.float32)


def augment_gesture_sample(sample_15x64):
    augmented = [sample_15x64]
    base = np.array(sample_15x64, dtype=np.float32)

    for _ in range(2):
        noise = np.random.normal(0.0, 6.0, base.shape)
        noisy = base.copy()
        valid = (base >= 50.0) & (base <= 400.0)
        noisy[valid] = np.clip(base[valid] + noise[valid], 50.0, 400.0)
        augmented.append(noisy)

    grid = base.reshape(15, 8, 8)
    s_left = np.full_like(grid, -1.0)
    s_left[:, :, :-1] = grid[:, :, 1:]
    augmented.append(s_left.reshape(15, 64))

    s_right = np.full_like(grid, -1.0)
    s_right[:, :, 1:] = grid[:, :, :-1]
    augmented.append(s_right.reshape(15, 64))

    s_up = np.full_like(grid, -1.0)
    s_up[:, :-1, :] = grid[:, 1:, :]
    augmented.append(s_up.reshape(15, 64))

    s_down = np.full_like(grid, -1.0)
    s_down[:, 1:, :] = grid[:, :-1, :]
    augmented.append(s_down.reshape(15, 64))

    idx_fast = np.clip(np.linspace(0, 14, 15, dtype=int), 0, 14)
    augmented.append(base[idx_fast])
    return augmented


# ----------------------------------------------------------------------
# Surface Optical Physics (ST cnh_lib.py Aligned)
# ----------------------------------------------------------------------
def smooth_cnh(pulse_48):
    """
    ST cnh_lib.py 3-tap binomial convolution smoothing filter.
    Kills single-bin photon shot noise while preserving true pulse physics.
    """
    pulse = np.array(pulse_48, dtype=np.float32)
    if len(pulse) < SURFACE_NUM_BINS:
        pulse = np.pad(pulse, (0, max(0, SURFACE_NUM_BINS - len(pulse))))
    pulse = pulse[:SURFACE_NUM_BINS]
    kern = np.array([0.25, 0.50, 0.25], dtype=np.float32)
    return np.convolve(pulse, kern, mode='same')


def extract_surface_pulse_metrics(cnh_bins_48):
    """
    Computes optical physical features with sub-bin linear interpolation:
      - 3-Tap Binomial Smoothed Curve
      - Floating-Point Sub-Bin FWHM in mm (np.interp at 50% half-max)
      - Peak Amplitude (kcps/SPAD)
      - Tail Decay Skewness
    """
    smoothed = smooth_cnh(cnh_bins_48)
    peak_val = float(np.max(smoothed))
    total_energy = float(np.sum(smoothed))

    if peak_val < 8.0 or total_energy < 40.0:
        return {
            "peak_val": peak_val,
            "peak_bin": 0,
            "dist_mm": 0.0,
            "fwhm_mm": 0.0,
            "tail_ratio": 1.0,
            "is_void": True,
            "smoothed": smoothed
        }

    peak_bin = int(np.argmax(smoothed))
    dist_mm = peak_bin * SURFACE_BIN_WIDTH_MM

    # Sub-bin linear interpolation from ST cnh_lib.py
    hist_norm = smoothed * (100.0 / max(1e-3, peak_val))
    rise_pos = float(peak_bin)
    for pos in range(peak_bin, -1, -1):
        if hist_norm[pos] < 50.0:
            rise_pos = float(np.interp(50.0, hist_norm[pos:pos+2], (pos, pos+1)))
            break

    fall_pos = float(peak_bin + 1)
    for pos in range(peak_bin, len(hist_norm)):
        if hist_norm[pos] < 50.0:
            fall_pos = float(np.interp(50.0, np.flip(hist_norm[pos-1:pos+1]), (pos, pos-1)))
            break

    fwhm_bins = max(0.5, fall_pos - rise_pos)
    fwhm_mm = fwhm_bins * SURFACE_BIN_WIDTH_MM
    denom = (fall_pos - rise_pos) if (fall_pos - rise_pos) > 1e-3 else 1.0
    tail_ratio = 100.0 * (peak_bin - rise_pos) / denom

    return {
        "peak_val": peak_val,
        "peak_bin": peak_bin,
        "dist_mm": dist_mm,
        "fwhm_mm": fwhm_mm,
        "tail_ratio": tail_ratio,
        "is_void": False,
        "smoothed": smoothed
    }


def extract_canonical_window(pulse_48, window_size=SURFACE_CANONICAL_BINS, pre_peak=SURFACE_PRE_PEAK):
    """
    Extracts a 16-bin peak-centered, distance-invariant canonical pulse.
    Peak is ALWAYS placed at index pre_peak (index 4), making the signal 100% immune
    to target distance variations from 10 cm to 50 cm!
    """
    metrics = extract_surface_pulse_metrics(pulse_48)
    if metrics["is_void"]:
        return np.zeros(window_size, dtype=np.float32), metrics

    smoothed = metrics["smoothed"]
    peak_bin = metrics["peak_bin"]

    canonical = np.zeros(window_size, dtype=np.float32)
    start_idx = peak_bin - pre_peak
    for i in range(window_size):
        src_idx = start_idx + i
        if 0 <= src_idx < len(smoothed):
            canonical[i] = smoothed[src_idx]

    p_max = np.max(canonical)
    if p_max > 1e-3:
        canonical = canonical / p_max

    return canonical, metrics


def augment_surface_canonical(canonical_16):
    """
    Augments the 16-bin canonical window with realistic optical variations:
      - Amplitude scaling (+/- 10%)
      - Photonic shot noise
      - Slight pulse width broadening / narrowing (incidence angle variations)
    """
    augmented = [np.array(canonical_16, dtype=np.float32)]
    base = np.array(canonical_16, dtype=np.float32)

    for scale in [0.92, 1.08]:
        augmented.append(np.clip(base * scale, 0.0, 1.0))

    for _ in range(2):
        noise = np.random.normal(0.0, 0.02 * (np.max(base) + 0.2), base.shape)
        augmented.append(np.clip(base + noise, 0.0, 1.0))

    # Broaden slightly
    b_wide = np.convolve(base, [0.10, 0.80, 0.10], mode='same')
    augmented.append(b_wide / (np.max(b_wide) + 1e-4))

    return augmented


# ----------------------------------------------------------------------
# Hand Posture Physics & RobustScaler Preprocessing (ST Aligned)
# ----------------------------------------------------------------------
def preprocess_posture_frame(dists_64, signals_64=None):
    """
    STSW-IMG050 aligned Hand Posture Preprocessor:
      - Validates minimum hand distance within [100, 400] mm.
      - Segments hand zones (dist <= min_dist + 120 mm).
      - Pads background zones with default values (dist=4000 mm, signal=0).
      - Normalizes using ST RobustScaler (median and IQR).
      - Returns (2, 8, 8) tensor [Ch0: Distance, Ch1: Peak Signal],
        along with telemetry metadata (min_dist, valid_frame, active_zones, bbox).
    """
    arr_d = np.array(dists_64, dtype=np.float32)
    if signals_64 is not None and len(signals_64) == 64:
        arr_s = np.array(signals_64, dtype=np.float32)
    else:
        arr_s = np.where(arr_d > 0, 50000.0 / np.clip(arr_d, 50.0, 1000.0), 0.0)

    valid_mask = (arr_d >= 50.0) & (arr_d <= 4000.0)
    valid_dists = arr_d[valid_mask]

    min_dist = float(np.min(valid_dists)) if len(valid_dists) > 0 else 4000.0
    valid_frame = (POSTURE_MIN_DIST <= min_dist <= POSTURE_MAX_DIST)

    bg_thresh = min_dist + POSTURE_BG_REMOVAL
    hand_mask = valid_mask & (arr_d <= bg_thresh)

    d_proc = np.where(hand_mask, arr_d, 4000.0)
    s_proc = np.where(hand_mask, arr_s, 0.0)

    ch0 = (d_proc - POSTURE_RANGING_CENTER) / POSTURE_RANGING_IQR
    ch1 = (s_proc - POSTURE_SIGNAL_CENTER) / POSTURE_SIGNAL_IQR

    tensor_2x8x8 = np.stack([ch0.reshape(8, 8), ch1.reshape(8, 8)], axis=0).astype(np.float32)

    active_indices = np.where(hand_mask)[0]
    active_count = int(len(active_indices))
    if active_count > 0:
        rows = active_indices // 8
        cols = active_indices % 8
        r_min, r_max = int(np.min(rows)), int(np.max(rows))
        c_min, c_max = int(np.min(cols)), int(np.max(cols))
        bbox = (r_min, r_max, c_min, c_max)
    else:
        bbox = (0, 0, 0, 0)

    return tensor_2x8x8, min_dist, valid_frame, active_count, bbox


def augment_posture_sample(tensor_2x8x8):
    """
    Data Augmentation for Hand Posture 2D-CNN:
      1. Horizontal flip (x <-> 7 - x): Left/Right hand symmetry.
      2. Spatial shift (+-1 zone) with default background padding.
      3. Gaussian noise on distance channel (+-8 mm) and signal gain (+-10%).
    """
    augmented = [tensor_2x8x8]
    ch0 = tensor_2x8x8[0].copy()
    ch1 = tensor_2x8x8[1].copy()

    # 1. Horizontal Flip (Left/Right hand symmetry)
    flip_ch0 = np.fliplr(ch0)
    flip_ch1 = np.fliplr(ch1)
    augmented.append(np.stack([flip_ch0, flip_ch1], axis=0))

    # 2. Spatial Shifts (+-1 row, +-1 col)
    bg_ch0 = (4000.0 - POSTURE_RANGING_CENTER) / POSTURE_RANGING_IQR
    bg_ch1 = (0.0 - POSTURE_SIGNAL_CENTER) / POSTURE_SIGNAL_IQR

    # Shift Left
    s_left0 = np.full_like(ch0, bg_ch0)
    s_left1 = np.full_like(ch1, bg_ch1)
    s_left0[:, :-1] = ch0[:, 1:]
    s_left1[:, :-1] = ch1[:, 1:]
    augmented.append(np.stack([s_left0, s_left1], axis=0))

    # Shift Right
    s_right0 = np.full_like(ch0, bg_ch0)
    s_right1 = np.full_like(ch1, bg_ch1)
    s_right0[:, 1:] = ch0[:, :-1]
    s_right1[:, 1:] = ch1[:, :-1]
    augmented.append(np.stack([s_right0, s_right1], axis=0))

    # Shift Up
    s_up0 = np.full_like(ch0, bg_ch0)
    s_up1 = np.full_like(ch1, bg_ch1)
    s_up0[:-1, :] = ch0[1:, :]
    s_up1[:-1, :] = ch1[1:, :]
    augmented.append(np.stack([s_up0, s_up1], axis=0))

    # Shift Down
    s_dn0 = np.full_like(ch0, bg_ch0)
    s_dn1 = np.full_like(ch1, bg_ch1)
    s_dn0[1:, :] = ch0[:-1, :]
    s_dn1[1:, :] = ch1[:-1, :]
    augmented.append(np.stack([s_dn0, s_dn1], axis=0))

    # 3. Depth & Signal Noise Jitter
    noise_d = np.random.normal(0.0, 8.0 / POSTURE_RANGING_IQR, ch0.shape)
    gain_s = np.random.uniform(0.9, 1.1)
    jitter_ch0 = ch0 + noise_d
    jitter_ch1 = ch1 * gain_s
    augmented.append(np.stack([jitter_ch0, jitter_ch1], axis=0).astype(np.float32))

    return augmented


# ----------------------------------------------------------------------
# Surface 1D-CNN PyTorch Architecture (16-Bin Canonical Input)
# ----------------------------------------------------------------------
if TORCH_AVAILABLE:
    class Surface1DCNN(nn.Module):
        """
        Ultra-efficient 1D-CNN designed for ST Edge AI & Cortex-M33:
          Input: (1, 1, 16) - Peak-Centered 16-Bin Canonical Optical Histogram
          Conv1D(8, k=3) -> MaxPool1D(2) -> Conv1D(16, k=3) -> MaxPool1D(2) -> Dense(16) -> Dense(4)
          Complexity: ~5,100 MACCs, 704 Bytes RAM, 6.1 KB Flash, < 20 us latency on Cortex-M33.
        """
        def __init__(self, num_classes=4):
            super(Surface1DCNN, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool1d(kernel_size=2) # 16 -> 8
            self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool1d(kernel_size=2) # 8 -> 4
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(16 * 4, 16)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(16, num_classes)

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)
            x = self.relu3(self.fc1(x))
            x = self.fc2(x)
            return x


    class Posture2DCNN(nn.Module):
        """
        ST-aligned 2D-CNN for 8x8 Time-of-Flight Hand Posture Recognition:
          Input: (B, 2, 8, 8) where Ch0 = RobustScaled Distance, Ch1 = RobustScaled Signal
          Conv2D(8, k=3, valid) -> ReLU -> MaxPool2D(2x2) -> Flatten(72) -> Dense(32) -> Dense(6)
          Complexity: 8,434 MACCs, 1,096 Bytes RAM, 10.5 KB Flash, < 40 us latency on Cortex-M33.
        """
        def __init__(self, num_classes=6):
            super(Posture2DCNN, self).__init__()
            self.conv = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=0)
            self.relu1 = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(8 * 3 * 3, 32)
            self.relu2 = nn.ReLU()
            self.fc2 = nn.Linear(32, num_classes)

        def forward(self, x):
            x = self.pool(self.relu1(self.conv(x)))
            x = x.flatten(1)
            x = self.relu2(self.fc1(x))
            x = self.fc2(x)
            return x


    class Gesture2DCNN(nn.Module):
        """
        Spatio-Temporal 2D-CNN for 8x8 15-Frame 3D Gesture Recognition:
          Input: (B, 15, 8, 8) - 15 temporal frames as channels, 8x8 spatial depth
          Conv2D(16, k=3, pad=1) -> ReLU -> MaxPool2D(2x2) -> (16, 4, 4)
          Conv2D(32, k=3, pad=1) -> ReLU -> AdaptiveAvgPool2D(2, 2) -> (32, 2, 2)
          Dense(128 -> 32) -> ReLU -> Dense(32 -> 5)
          Complexity: ~219k MACCs, 7.5 KB RAM, 43.4 KB Flash, < 1 ms latency on Cortex-M33.
        """
        def __init__(self, num_classes=5):
            super(Gesture2DCNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=15, out_channels=16, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(32 * 2 * 2, 32)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(32, num_classes)

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)
            x = self.relu3(self.fc1(x))
            x = self.fc2(x)
            return x


# ----------------------------------------------------------------------
# High-Speed Zero-Lag Serial Worker
# ----------------------------------------------------------------------
class SerialWorker(QThread):
    sig_frame = pyqtSignal(dict)
    sig_status = pyqtSignal(str, bool)

    def __init__(self, port="COM9", baudrate=921600):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = True
        self.ser = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.sig_status.emit(f"Connected to {self.port} (12 Mbps CDC)", True)
            time.sleep(0.05)
            try:
                self.ser.reset_input_buffer()
                self.ser.write(b'AT+STREAM=RAW\r\n')
            except Exception:
                pass
        except Exception as e:
            self.sig_status.emit(f"Connection Failed: {e}", False)
            return

        buffer = b""
        while self.running:
            try:
                waiting = self.ser.in_waiting
                if waiting > 0:
                    chunk = self.ser.read(waiting)
                    if chunk:
                        buffer += chunk
                        while b'\n' in buffer:
                            line_bytes, buffer = buffer.split(b'\n', 1)
                            line_str = line_bytes.decode('utf-8', errors='ignore').strip()
                            if line_str.startswith("{") and line_str.endswith("}"):
                                try:
                                    frame = json.loads(line_str)
                                    if "d" in frame:
                                        frame["distances"] = frame["d"]
                                    if "h" in frame:
                                        frame["histograms"] = frame["h"]
                                    if "a" in frame:
                                        frame["ambient"] = frame["a"]
                                    self.sig_frame.emit(frame)
                                except Exception:
                                    pass
                else:
                    time.sleep(0.005)
            except Exception:
                time.sleep(0.01)

        if self.ser and self.ser.is_open:
            self.ser.close()

    def send_command(self, cmd_str):
        if self.ser and self.ser.is_open:
            try:
                if not cmd_str.endswith('\n'):
                    cmd_str += '\r\n'
                self.ser.write(cmd_str.encode('ascii'))
                self.ser.flush()
            except Exception:
                pass

    def stop(self):
        self.running = False
        self.wait(1000)


class DeployWorker(QThread):
    sig_log = pyqtSignal(str)
    sig_done = pyqtSignal(bool, str)

    def __init__(self, mode="gesture"):
        super().__init__()
        self.mode = mode

    def run(self):
        try:
            if self.mode == "surface":
                bat_path = os.path.join(SCRIPT_DIR, "deploy_surface_model_to_mcu.bat")
            elif self.mode == "posture":
                bat_path = os.path.join(SCRIPT_DIR, "deploy_posture_model_to_mcu.bat")
            else:
                bat_path = os.path.join(SCRIPT_DIR, "deploy_model_to_mcu.bat")

            proc = subprocess.Popen(
                ["cmd.exe", "/c", bat_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=PROJECT_ROOT
            )
            for line in iter(proc.stdout.readline, ''):
                if line:
                    self.sig_log.emit(line.rstrip())
            proc.stdout.close()
            ret = proc.wait()
            if ret == 0:
                self.sig_done.emit(True, f"{self.mode.capitalize()} model deployed and firmware compiled successfully!")
            else:
                self.sig_done.emit(False, f"Deployment failed with exit code {ret}.")
        except Exception as e:
            self.sig_done.emit(False, str(e))


# ----------------------------------------------------------------------
# 8x8 Depth Heatmap Widget (Gesture Mode)
# ----------------------------------------------------------------------
class DepthHeatmap8x8(QWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(280, 240)
        self.labels = []
        grid = QGridLayout(self)
        grid.setSpacing(2)
        grid.setContentsMargins(4, 4, 4, 4)

        for r in range(8):
            row_labels = []
            for c in range(8):
                lbl = QLabel("---")
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setFont(QFont("Consolas", 8, QFont.Bold))
                lbl.setStyleSheet("background-color: #f8fafc; color: #94a3b8; border: 1px solid #e2e8f0; border-radius: 3px;")
                lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                grid.addWidget(lbl, r, c)
                row_labels.append(lbl)
            self.labels.append(row_labels)

    def update_distances(self, dists):
        if len(dists) < 64:
            return
        for idx in range(64):
            r = idx // 8
            c = idx % 8
            d = dists[idx]
            lbl = self.labels[r][c]

            if 50 <= d <= 450:
                ratio = np.clip((d - 80.0) / 300.0, 0.0, 1.0)
                prox = 1.0 - ratio
                if prox < 0.5:
                    t = prox / 0.5
                    red = int(59 * (1.0 - t) + 16 * t)
                    green = int(130 * (1.0 - t) + 185 * t)
                    blue = int(246 * (1.0 - t) + 129 * t)
                else:
                    t = (prox - 0.5) / 0.5
                    red = int(16 * (1.0 - t) + 245 * t)
                    green = int(185 * (1.0 - t) + 158 * t)
                    blue = int(129 * (1.0 - t) + 11 * t)

                lum = 0.299 * red + 0.587 * green + 0.114 * blue
                txt_col = "#0f172a" if lum > 130 else "#ffffff"
                lbl.setText(f"{d}")
                lbl.setStyleSheet(f"background-color: rgb({red}, {green}, {blue}); color: {txt_col}; font-weight: bold; border-radius: 3px; border: 1px solid rgba(0,0,0,0.08);")
            else:
                lbl.setText("---")
                lbl.setStyleSheet("background-color: #f8fafc; color: #94a3b8; border: 1px solid #e2e8f0; border-radius: 3px;")


# ----------------------------------------------------------------------
# 4x4 Depth Heatmap Widget with Zone Selection (Surface Mode)
# ----------------------------------------------------------------------
class DepthHeatmap4x4(QWidget):
    sig_zone_selected = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(280, 240)
        self.buttons = []
        self.selected_zone = 5
        grid = QGridLayout(self)
        grid.setSpacing(4)
        grid.setContentsMargins(4, 4, 4, 4)

        for r in range(4):
            row_btns = []
            for c in range(4):
                idx = r * 4 + c
                btn = QPushButton(f"Z{idx}\n---")
                btn.setFont(QFont("Consolas", 10, QFont.Bold))
                btn.setStyleSheet("background-color: #f8fafc; color: #64748b; border: 1px solid #cbd5e1; border-radius: 4px;")
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                btn.clicked.connect(lambda _, z=idx: self.select_zone(z))
                grid.addWidget(btn, r, c)
                row_btns.append(btn)
            self.buttons.append(row_btns)

        self.highlight_selected()

    def select_zone(self, zone_idx):
        self.selected_zone = zone_idx
        self.highlight_selected()
        self.sig_zone_selected.emit(zone_idx)

    def highlight_selected(self):
        for r in range(4):
            for c in range(4):
                idx = r * 4 + c
                btn = self.buttons[r][c]
                if idx == self.selected_zone:
                    btn.setStyleSheet("background-color: #eff6ff; color: #1d4ed8; border: 2px solid #2563eb; border-radius: 4px; font-weight: bold;")
                else:
                    btn.setStyleSheet("background-color: #f8fafc; color: #64748b; border: 1px solid #cbd5e1; border-radius: 4px;")

    def update_distances(self, dists):
        if len(dists) < 16:
            return
        for idx in range(16):
            r = idx // 4
            c = idx % 4
            d = dists[idx]
            btn = self.buttons[r][c]
            is_sel = (idx == self.selected_zone)
            border_style = "border: 2px solid #2563eb;" if is_sel else "border: 1px solid rgba(0,0,0,0.12);"

            if 50 <= d <= 1800:
                ratio = np.clip((d - 80.0) / 1200.0, 0.0, 1.0)
                prox = 1.0 - ratio
                red = int(2 * (1.0 - prox) + 217 * prox)
                green = int(132 * (1.0 - prox) + 119 * prox)
                blue = int(199 * (1.0 - prox) + 6 * prox)

                lum = 0.299 * red + 0.587 * green + 0.114 * blue
                txt_col = "#0f172a" if lum > 130 else "#ffffff"
                btn.setText(f"Z{idx}\n{d}mm")
                btn.setStyleSheet(f"background-color: rgb({red}, {green}, {blue}); color: {txt_col}; font-weight: bold; border-radius: 4px; {border_style}")
            else:
                btn.setText(f"Z{idx}\nVOID")
                if is_sel:
                    btn.setStyleSheet("background-color: #f1f5f9; color: #1d4ed8; border: 2px solid #2563eb; border-radius: 4px; font-weight: bold;")
                else:
                    btn.setStyleSheet("background-color: #f8fafc; color: #94a3b8; border: 1px solid #e2e8f0; border-radius: 4px;")


# ----------------------------------------------------------------------
# 2D Motion Trail / Trajectory Canvas (Gesture Mode)
# ----------------------------------------------------------------------
class MotionTrailWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(280, 140)
        self.trail = deque(maxlen=GESTURE_WINDOW_SIZE)
        self.latest_pos = None

    def update_trail(self, centroid_x, centroid_y):
        if centroid_x is not None and centroid_y is not None:
            self.trail.append((centroid_x, centroid_y))
            self.latest_pos = (centroid_x, centroid_y)
        else:
            if len(self.trail) > 0:
                self.trail.popleft()
            if len(self.trail) == 0:
                self.latest_pos = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        painter.fillRect(0, 0, w, h, QColor("#ffffff"))

        cx = w / 2.0
        cy = h / 2.0

        painter.setPen(QPen(QColor("#f1f5f9"), 1, Qt.SolidLine))
        for offset in (-80, -40, 40, 80):
            painter.drawLine(0, int(cy + offset), w, int(cy + offset))
            painter.drawLine(int(cx + offset), 0, int(cx + offset), h)

        painter.setPen(QPen(QColor("#cbd5e1"), 1, Qt.DashLine))
        painter.drawLine(0, int(cy), w, int(cy))
        painter.drawLine(int(cx), 0, int(cx), h)

        painter.setPen(QColor("#cbd5e1"))
        painter.drawRect(0, 0, w - 1, h - 1)

        painter.setPen(QColor("#475569"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(8, 16, "2D CENTROID MOTION TRAIL (XY)")

        if len(self.trail) < 2:
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(int(cx - 55), int(cy + 5), "Waiting for motion...")
            return

        scale_x = (w * 0.42) / 140.0
        scale_y = (h * 0.42) / 140.0

        points = [QPointF(cx + (x * scale_x), cy + (y * scale_y)) for (x, y) in self.trail]
        n = len(points)
        for i in range(n - 1):
            alpha = int(70 + (185.0 * (i / float(n))))
            painter.setPen(QPen(QColor(29, 78, 216, alpha), 2))
            painter.drawLine(points[i], points[i + 1])
            painter.setBrush(QBrush(QColor(4, 120, 87, alpha)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(points[i], 3, 3)

        painter.setBrush(QBrush(QColor("#ea580c")))
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.drawEllipse(points[-1], 6, 6)


# ----------------------------------------------------------------------
# Live 48-Bin Optical Pulse Waveform with Canonical Highlight Box
# ----------------------------------------------------------------------
class SurfacePulseWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(280, 150)
        self.pulse = [0.0] * SURFACE_NUM_BINS
        self.metrics = {
            "peak_val": 0.0,
            "peak_bin": 0,
            "dist_mm": 0.0,
            "fwhm_mm": 0.0,
            "tail_ratio": 1.0,
            "is_void": True
        }
        self.zone_idx = 5

    def update_pulse(self, cnh_bins, metrics, zone_idx=5):
        if len(cnh_bins) >= SURFACE_NUM_BINS:
            # Use smoothed curve if available
            smoothed = metrics.get("smoothed", None)
            if smoothed is not None:
                self.pulse = list(smoothed[:SURFACE_NUM_BINS])
            else:
                self.pulse = list(cnh_bins[:SURFACE_NUM_BINS])
            self.metrics = metrics
            self.zone_idx = zone_idx
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QColor("#ffffff"))

        painter.setPen(QPen(QColor("#f1f5f9"), 1, Qt.SolidLine))
        for y_rat in (0.25, 0.5, 0.75):
            painter.drawLine(0, int(h * y_rat), w, int(h * y_rat))

        painter.setPen(QColor("#cbd5e1"))
        painter.drawRect(0, 0, w - 1, h - 1)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor("#1e3a8a"))
        painter.drawText(8, 16, f"ZONE #{self.zone_idx} - 48-BIN PULSE & 16-BIN CANONICAL WINDOW")

        peak_v = self.metrics.get("peak_val", 0.0)
        fwhm_mm = self.metrics.get("fwhm_mm", 0.0)
        tail = self.metrics.get("tail_ratio", 1.0)
        dist_mm = self.metrics.get("dist_mm", 0.0)

        painter.setFont(QFont("Consolas", 8, QFont.Bold))
        if fwhm_mm > 70.0:
            painter.setPen(QColor("#d97706"))
            status_desc = f"FWHM: {fwhm_mm:.1f}mm (CARPET / BROAD) | D: {dist_mm:.0f}mm"
        elif peak_v > 500.0:
            painter.setPen(QColor("#7c3aed"))
            status_desc = f"FWHM: {fwhm_mm:.1f}mm (SPECULAR / METAL) | Peak: {peak_v:.0f}"
        elif fwhm_mm > 0.0:
            painter.setPen(QColor("#0284c7"))
            status_desc = f"FWHM: {fwhm_mm:.1f}mm (HARD FLOOR / SHARP) | D: {dist_mm:.0f}mm"
        else:
            painter.setPen(QColor("#64748b"))
            status_desc = "VOID / DROP-OFF HAZARD"

        painter.drawText(w - painter.fontMetrics().horizontalAdvance(status_desc) - 10, 16, status_desc)

        max_val = max(10.0, float(np.max(self.pulse)))
        margin_left = 20
        margin_right = 15
        margin_bottom = 22
        margin_top = 28

        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom

        points = []
        step_x = plot_w / float(SURFACE_NUM_BINS - 1)

        for b, val in enumerate(self.pulse):
            px = margin_left + b * step_x
            py = margin_top + plot_h - (min(1.0, val / max_val) * plot_h)
            points.append(QPointF(px, py))

        # Highlight Canonical 16-bin Window [peak_bin - 4 : peak_bin + 12]
        peak_idx = int(self.metrics.get("peak_bin", 0))
        if not self.metrics.get("is_void", True):
            win_start_bin = max(0, peak_idx - SURFACE_PRE_PEAK)
            win_end_bin = min(SURFACE_NUM_BINS - 1, peak_idx - SURFACE_PRE_PEAK + SURFACE_CANONICAL_BINS - 1)
            wx_start = margin_left + win_start_bin * step_x
            wx_end = margin_left + win_end_bin * step_x
            # Shaded canonical box
            painter.fillRect(int(wx_start), int(margin_top), int(wx_end - wx_start), int(plot_h), QColor(2, 132, 199, 18))
            painter.setPen(QPen(QColor(2, 132, 199, 80), 1, Qt.DashLine))
            painter.drawRect(int(wx_start), int(margin_top), int(wx_end - wx_start), int(plot_h))

        # Filled polygon under pulse
        poly = QPolygonF()
        poly.append(QPointF(margin_left, margin_top + plot_h))
        for pt in points:
            poly.append(pt)
        poly.append(QPointF(margin_left + plot_w, margin_top + plot_h))

        grad = QLinearGradient(0, margin_top, 0, margin_top + plot_h)
        if fwhm_mm > 70.0:
            grad.setColorAt(0.0, QColor(217, 119, 6, 90))
            grad.setColorAt(1.0, QColor(217, 119, 6, 10))
            pen_color = QColor("#d97706")
        elif peak_v > 500.0:
            grad.setColorAt(0.0, QColor(124, 58, 237, 90))
            grad.setColorAt(1.0, QColor(124, 58, 237, 10))
            pen_color = QColor("#7c3aed")
        else:
            grad.setColorAt(0.0, QColor(2, 132, 199, 90))
            grad.setColorAt(1.0, QColor(2, 132, 199, 10))
            pen_color = QColor("#0284c7")

        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(poly)

        painter.setPen(QPen(pen_color, 2))
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

        # Peak Marker
        if 0 <= peak_idx < len(points) and peak_v > 5.0:
            pk_pt = points[peak_idx]
            painter.setBrush(QBrush(QColor("#ef4444")))
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.drawEllipse(pk_pt, 5, 5)

            hm_y = margin_top + plot_h - (0.5 * (peak_v / max_val) * plot_h)
            painter.setPen(QPen(QColor("#94a3b8"), 1, Qt.DashLine))
            painter.drawLine(margin_left, int(hm_y), margin_left + plot_w, int(hm_y))

        painter.setPen(QColor("#94a3b8"))
        painter.setFont(QFont("Consolas", 7))
        painter.drawText(margin_left, h - 6, "Bin 0 (0mm)")
        painter.drawText(margin_left + plot_w // 2 - 25, h - 6, "Bin 24 (~900mm)")
        painter.drawText(margin_left + plot_w - 65, h - 6, "Bin 47 (~1.8m)")


# ----------------------------------------------------------------------
# Live 15-Frame Directional Energy Waveform (Gesture Mode)
# ----------------------------------------------------------------------
class PostureHUDWidget(QWidget):
    """
    Dedicated Hand Posture HUD Display:
      - Large animated posture badge (Icon, Class Name, Confidence Gauge)
      - Real-time 6-class softmax probability distribution bars
      - Spatial hand metrics: Min Distance, Active Zones, Hand BBox, Debounce Latch
    """
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # 1. Main Latch Badge
        self.badge_box = QFrame()
        self.badge_box.setFixedHeight(105)
        self.badge_box.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 2px solid #cbd5e1;
                border-radius: 8px;
            }
        """)
        badge_layout = QHBoxLayout(self.badge_box)
        badge_layout.setContentsMargins(14, 8, 14, 8)

        self.lbl_icon = QLabel("🚫")
        self.lbl_icon.setFont(QFont("Segoe UI Emoji", 42))
        self.lbl_icon.setAlignment(Qt.AlignCenter)
        self.lbl_icon.setFixedWidth(70)
        badge_layout.addWidget(self.lbl_icon)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        self.lbl_title = QLabel("NO HAND")
        self.lbl_title.setFont(QFont("Segoe UI", 17, QFont.Bold))
        self.lbl_title.setStyleSheet("color: #475569;")
        self.lbl_conf = QLabel("Confidence: 0.0% | Hand out of range")
        self.lbl_conf.setFont(QFont("Segoe UI", 11))
        self.lbl_conf.setStyleSheet("color: #64748b;")
        self.lbl_latch = QLabel("Debounce: Idle (0/3 frames)")
        self.lbl_latch.setFont(QFont("Consolas", 10, QFont.Bold))
        self.lbl_latch.setStyleSheet("color: #94a3b8;")

        text_layout.addWidget(self.lbl_title)
        text_layout.addWidget(self.lbl_conf)
        text_layout.addWidget(self.lbl_latch)
        badge_layout.addLayout(text_layout)
        badge_layout.addStretch()

        layout.addWidget(self.badge_box)

        # 2. Telemetry Cards
        tele_layout = QHBoxLayout()
        tele_layout.setSpacing(6)

        self.lbl_dist_card = QLabel("Dist: --- mm")
        self.lbl_dist_card.setAlignment(Qt.AlignCenter)
        self.lbl_dist_card.setStyleSheet("background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 5px; font-weight: bold; font-size: 11px; color: #1e293b;")
        tele_layout.addWidget(self.lbl_dist_card)

        self.lbl_zones_card = QLabel("Zones: 0 / 64")
        self.lbl_zones_card.setAlignment(Qt.AlignCenter)
        self.lbl_zones_card.setStyleSheet("background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 5px; font-weight: bold; font-size: 11px; color: #1e293b;")
        tele_layout.addWidget(self.lbl_zones_card)

        self.lbl_bbox_card = QLabel("BBox: 0x0")
        self.lbl_bbox_card.setAlignment(Qt.AlignCenter)
        self.lbl_bbox_card.setStyleSheet("background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 5px; font-weight: bold; font-size: 11px; color: #1e293b;")
        tele_layout.addWidget(self.lbl_bbox_card)

        layout.addLayout(tele_layout)

        # 3. Probability Bars Group
        p_group = QGroupBox("6-Class Softmax Probability Distribution (2D-CNN)")
        pg_layout = QVBoxLayout(p_group)
        pg_layout.setSpacing(4)
        pg_layout.setContentsMargins(8, 10, 8, 8)

        self.prob_bars = {}
        self.prob_labels = {}

        for cls in POSTURE_CLASSES:
            row = QHBoxLayout()
            icon_lbl = QLabel(f"{POSTURE_ICONS.get(cls, '')} {cls}")
            icon_lbl.setFixedWidth(120)
            icon_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
            icon_lbl.setStyleSheet(f"color: {POSTURE_COLORS.get(cls, '#1e293b')};")
            row.addWidget(icon_lbl)

            pbar = QProgressBar()
            pbar.setRange(0, 100)
            pbar.setValue(0)
            pbar.setFixedHeight(10)
            pbar.setTextVisible(False)
            pbar.setStyleSheet(f"""
                QProgressBar {{ background-color: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 5px; }}
                QProgressBar::chunk {{ background-color: {POSTURE_COLORS.get(cls, '#0284c7')}; border-radius: 4px; }}
            """)
            row.addWidget(pbar)
            self.prob_bars[cls] = pbar

            val_lbl = QLabel("0.0%")
            val_lbl.setFixedWidth(50)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setFont(QFont("Consolas", 10, QFont.Bold))
            val_lbl.setStyleSheet("color: #334155;")
            row.addWidget(val_lbl)
            self.prob_labels[cls] = val_lbl

            pg_layout.addLayout(row)

        layout.addWidget(p_group)
        layout.addStretch()

    def update_hud(self, latched_cls, probs, min_dist, valid_frame, active_zones, bbox, debounce_cnt, is_latched):
        icon = POSTURE_ICONS.get(latched_cls, "🖐️")
        col = POSTURE_COLORS.get(latched_cls, "#475569")
        conf = probs[POSTURE_CLASSES.index(latched_cls)] if (latched_cls in POSTURE_CLASSES and len(probs) == len(POSTURE_CLASSES)) else 0.0

        self.lbl_icon.setText(icon)
        self.lbl_title.setText(latched_cls)
        self.lbl_title.setStyleSheet(f"color: {col}; font-weight: bold;")

        if valid_frame:
            self.lbl_conf.setText(f"Confidence: {conf * 100.0:.1f}% | Hand In Range ({min_dist:.0f} mm)")
            self.lbl_conf.setStyleSheet("color: #047857; font-weight: 500;")
        else:
            self.lbl_conf.setText("Waiting for hand in [100, 400] mm zone...")
            self.lbl_conf.setStyleSheet("color: #94a3b8;")

        if is_latched:
            self.lbl_latch.setText("Debounce: [●●●] Confirmed & Latched (3/3)")
            self.lbl_latch.setStyleSheet("color: #047857; font-weight: bold;")
        else:
            self.lbl_latch.setText(f"Debounce: [{'●' * debounce_cnt}{'○' * (3 - debounce_cnt)}] Verifying ({debounce_cnt}/3)")
            self.lbl_latch.setStyleSheet("color: #d97706; font-weight: bold;")

        if valid_frame:
            self.lbl_dist_card.setText(f"Dist: {min_dist:.0f} mm")
            self.lbl_dist_card.setStyleSheet("background-color: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 4px; padding: 5px; font-weight: bold; font-size: 11px; color: #047857;")
        else:
            self.lbl_dist_card.setText("Dist: --- mm (Out)")
            self.lbl_dist_card.setStyleSheet("background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 5px; font-weight: bold; font-size: 11px; color: #94a3b8;")

        self.lbl_zones_card.setText(f"Zones: {active_zones} / 64")
        r_span = (bbox[1] - bbox[0] + 1) if active_zones > 0 else 0
        c_span = (bbox[3] - bbox[2] + 1) if active_zones > 0 else 0
        self.lbl_bbox_card.setText(f"BBox: {c_span}x{r_span}")

        for idx, cls in enumerate(POSTURE_CLASSES):
            p = float(probs[idx]) if idx < len(probs) else 0.0
            self.prob_bars[cls].setValue(int(p * 100))
            self.prob_labels[cls].setText(f"{p * 100:.1f}%")


class EnergyWaveformWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 130)
        self.lr_history = [0.0] * GESTURE_WINDOW_SIZE
        self.tb_history = [0.0] * GESTURE_WINDOW_SIZE

    def update_energy(self, window_15x64):
        if len(window_15x64) < GESTURE_WINDOW_SIZE:
            return
        grid = normalize_gesture_window(window_15x64)

        col_e = np.sum(grid, axis=1)
        self.lr_history = list(np.sum(col_e[:, :4], axis=1) - np.sum(col_e[:, 4:], axis=1))

        row_e = np.sum(grid, axis=2)
        self.tb_history = list(np.sum(row_e[:, :4], axis=1) - np.sum(row_e[:, 4:], axis=1))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        cy = h / 2.0

        painter.fillRect(0, 0, w, h, QColor("#ffffff"))
        painter.setPen(QPen(QColor("#f1f5f9"), 1, Qt.SolidLine))
        painter.drawLine(0, int(cy - h * 0.35), w, int(cy - h * 0.35))
        painter.drawLine(0, int(cy + h * 0.35), w, int(cy + h * 0.35))

        painter.setPen(QPen(QColor("#cbd5e1"), 1, Qt.DashLine))
        painter.drawLine(0, int(cy), w, int(cy))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawRect(0, 0, w - 1, h - 1)

        painter.setFont(QFont("Consolas", 8, QFont.Bold))
        painter.setPen(QColor("#1d4ed8"))
        painter.drawText(10, 16, "- X-Energy (Left/Right)")
        painter.setPen(QColor("#d97706"))
        painter.drawText(180, 16, "- Y-Energy (Up/Down)")

        step_x = (w - 30) / float(GESTURE_WINDOW_SIZE - 1)
        max_scale = 8.0

        pen_x = QPen(QColor("#1d4ed8"), 2)
        painter.setPen(pen_x)
        pts_x = [QPointF(15 + i * step_x, cy - (np.clip(val / max_scale, -1.0, 1.0) * (cy * 0.75))) for i, val in enumerate(self.lr_history)]
        for i in range(len(pts_x) - 1):
            painter.drawLine(pts_x[i], pts_x[i + 1])

        pen_y = QPen(QColor("#d97706"), 2)
        painter.setPen(pen_y)
        pts_y = [QPointF(15 + i * step_x, cy - (np.clip(val / max_scale, -1.0, 1.0) * (cy * 0.75))) for i, val in enumerate(self.tb_history)]
        for i in range(len(pts_y) - 1):
            painter.drawLine(pts_y[i], pts_y[i + 1])


# ----------------------------------------------------------------------
# Main Modular Neural Network Studio Window
# ----------------------------------------------------------------------
class NNStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EdgeSense - Multi-Capability Neural Network Studio (Gesture & Surface)")
        self.resize(1300, 840)
        self.setStyleSheet("""
            QMainWindow { background-color: #f8fafc; }
            QWidget { font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif; font-size: 13px; }
            QLabel { color: #1e293b; font-size: 13px; }
            QGroupBox { 
                background-color: #ffffff; 
                color: #0f172a; 
                font-weight: bold; 
                border: 1px solid #cbd5e1; 
                border-radius: 6px; 
                margin-top: 12px; 
                padding-top: 14px;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 6px; 
                background-color: #ffffff;
                color: #1e3a8a;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 0.3px;
            }
            QTabWidget::pane { 
                border: 1px solid #cbd5e1; 
                background: #ffffff; 
                border-radius: 6px; 
                top: -1px; 
            }
            QTabBar::tab { 
                background: #e2e8f0; 
                color: #475569; 
                padding: 9px 20px; 
                margin-right: 4px; 
                font-weight: bold; 
                font-size: 12px;
                border-top-left-radius: 5px; 
                border-top-right-radius: 5px; 
                border: 1px solid #cbd5e1;
                border-bottom: none;
            }
            QTabBar::tab:hover { background: #f1f5f9; color: #0f172a; }
            QTabBar::tab:selected { background: #ffffff; color: #1e3a8a; border-bottom: 2px solid #2563eb; }
            QPushButton { 
                background-color: #f8fafc; 
                color: #1e293b; 
                border: 1px solid #cbd5e1; 
                border-radius: 5px; 
                padding: 7px 14px; 
                font-weight: bold; 
                font-size: 12px; 
            }
            QPushButton:hover { background-color: #f1f5f9; border-color: #94a3b8; color: #0f172a; }
            QPushButton:pressed { background-color: #e2e8f0; }
            QPushButton:disabled { background-color: #f1f5f9; color: #94a3b8; border-color: #e2e8f0; }
            QTableWidget { 
                background-color: #ffffff; 
                color: #1e293b; 
                gridline-color: #e2e8f0; 
                border: 1px solid #cbd5e1; 
                border-radius: 5px; 
                selection-background-color: #eff6ff; 
                selection-color: #1e3a8a; 
            }
            QHeaderView::section { 
                background-color: #f8fafc; 
                color: #475569; 
                font-weight: bold; 
                border: 1px solid #e2e8f0; 
                padding: 6px; 
                font-size: 11px;
            }
            QComboBox {
                background-color: #ffffff; 
                color: #1e293b; 
                border: 1px solid #cbd5e1; 
                border-radius: 4px; 
                padding: 4px 10px; 
                font-size: 12px; 
                font-weight: 500;
            }
            QTextEdit { 
                background-color: #ffffff; 
                color: #0f172a; 
                border: 1px solid #cbd5e1; 
                border-radius: 5px; 
                font-family: 'Consolas', monospace; 
                font-size: 12px; 
            }
            QProgressBar { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; }
            QSplitter::handle { background-color: #cbd5e1; width: 1px; }
        """)

        self.current_mode = "GESTURE"

        # Gesture Data & States
        self.gesture_dataset = {cls: [] for cls in GESTURE_CLASSES}
        self.gesture_sliding_window = []
        self.gesture_trained_model = None
        self.is_gesture_recording = False
        self.gesture_target_class = ""
        self.gesture_record_buffer = []
        self.gesture_cooldown_until = 0.0

        # Surface Data & States (Distance-Invariant 16-Bin Canonical)
        self.surface_dataset = {cls: [] for cls in SURFACE_CLASSES}
        self.surface_trained_model = None
        self.is_surface_recording = False
        self.surface_target_class = ""
        self.surface_record_count = 0
        self.surface_record_target_total = 1
        self.surface_selected_zone = 5

        # Temporal Hysteresis & Debouncing State
        self.surface_smoothed_probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        self.surface_confirmed_class = "HARD_FLOOR"
        self.surface_candidate_class = "HARD_FLOOR"
        self.surface_candidate_count = 0

        # Hand Posture Data & States (STSW-IMG050 2D-CNN 8x8 Dual-Channel)
        self.posture_dataset = {cls: [] for cls in POSTURE_CLASSES}
        self.posture_trained_model = None
        self.is_posture_recording = False
        self.posture_target_class = ""
        self.posture_record_count = 0
        self.posture_record_target_total = 1
        self.posture_smoothed_probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.posture_latched_class = "NONE"
        self.posture_candidate_class = "NONE"
        self.posture_candidate_count = 0

        # Telemetry & FPS
        self.fps_frames = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        self.total_frames = 0

        self.init_ui()
        self.setup_shortcuts()
        self.load_default_datasets()
        self.start_serial_connection()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # 1. Top Header Bar
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(4, 2, 4, 6)
        top_bar.setAlignment(Qt.AlignVCenter)

        title_container = QVBoxLayout()
        title_container.setSpacing(1)
        title_lbl = QLabel("EdgeSense Neural Network Studio")
        title_lbl.setFont(QFont("Segoe UI", 15, QFont.Bold))
        title_lbl.setStyleSheet("color: #0f172a; letter-spacing: -0.2px;")
        sub_lbl = QLabel("Multi-Capability Edge AI: 3D Gesture Recognition & Distance-Invariant Surface 1D-CNN")
        sub_lbl.setFont(QFont("Segoe UI", 8))
        sub_lbl.setStyleSheet("color: #64748b; font-weight: 500;")
        title_container.addWidget(title_lbl)
        title_container.addWidget(sub_lbl)
        top_bar.addLayout(title_container)

        top_bar.addSpacing(20)

        # Segmented Capability Mode Switcher
        mode_box = QFrame()
        mode_box.setStyleSheet("background-color: #e2e8f0; border-radius: 6px; padding: 2px;")
        mode_layout = QHBoxLayout(mode_box)
        mode_layout.setContentsMargins(2, 2, 2, 2)
        mode_layout.setSpacing(2)

        self.btn_mode_gesture = QPushButton(" 3D Gesture (8x8)")
        self.btn_mode_gesture.setCheckable(True)
        self.btn_mode_gesture.setChecked(True)
        self.btn_mode_gesture.setFixedHeight(28)
        self.btn_mode_gesture.clicked.connect(lambda: self.switch_capability_mode("GESTURE"))
        mode_layout.addWidget(self.btn_mode_gesture)

        self.btn_mode_surface = QPushButton(" Surface 1D-CNN (4x4)")
        self.btn_mode_surface.setCheckable(True)
        self.btn_mode_surface.setFixedHeight(28)
        self.btn_mode_surface.clicked.connect(lambda: self.switch_capability_mode("SURFACE"))
        mode_layout.addWidget(self.btn_mode_surface)

        self.btn_mode_posture = QPushButton(" Hand Posture (8x8)")
        self.btn_mode_posture.setCheckable(True)
        self.btn_mode_posture.setFixedHeight(28)
        self.btn_mode_posture.clicked.connect(lambda: self.switch_capability_mode("POSTURE"))
        mode_layout.addWidget(self.btn_mode_posture)

        self.update_mode_buttons_style()
        top_bar.addWidget(mode_box)

        top_bar.addSpacing(15)
        self.lbl_model_badge = QLabel("Model: Untrained")
        self.lbl_model_badge.setFixedHeight(26)
        self.lbl_model_badge.setAlignment(Qt.AlignCenter)
        self.lbl_model_badge.setStyleSheet("background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; padding: 3px 10px; border-radius: 4px; font-weight: bold; font-size: 11px;")
        top_bar.addWidget(self.lbl_model_badge)
        top_bar.addStretch()

        lbl_port = QLabel("Port:")
        lbl_port.setStyleSheet("color: #475569; font-weight: 600; font-size: 12px;")
        top_bar.addWidget(lbl_port)

        self.port_combo = QComboBox()
        self.port_combo.setFixedHeight(30)
        self.refresh_ports()
        top_bar.addWidget(self.port_combo)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setFixedHeight(30)
        self.btn_connect.setStyleSheet("background-color: #1d4ed8; color: #ffffff; font-weight: bold; padding: 5px 14px; border: none; border-radius: 4px;")
        self.btn_connect.clicked.connect(self.toggle_connection)
        top_bar.addWidget(self.btn_connect)

        self.lbl_status = QLabel("Disconnected")
        self.lbl_status.setFixedHeight(26)
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("background-color: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; border-radius: 4px; padding: 3px 10px; font-weight: bold; font-size: 11px;")
        top_bar.addWidget(self.lbl_status)
        main_layout.addLayout(top_bar)

        # 2. Main Horizontal Splitter
        splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.heatmap_group = QGroupBox("Live Depth Heatmap (8x8 ToF)")
        hg_layout = QVBoxLayout(self.heatmap_group)
        self.heatmap_stack = QStackedWidget()
        self.heatmap_8x8 = DepthHeatmap8x8()
        self.heatmap_4x4 = DepthHeatmap4x4()
        self.heatmap_4x4.sig_zone_selected.connect(self.on_surface_zone_selected)
        self.heatmap_stack.addWidget(self.heatmap_8x8)
        self.heatmap_stack.addWidget(self.heatmap_4x4)
        hg_layout.addWidget(self.heatmap_stack)
        left_layout.addWidget(self.heatmap_group, stretch=5)

        self.visualizer_group = QGroupBox("2D Centroid Motion Trail (Gesture)")
        vg_layout = QVBoxLayout(self.visualizer_group)
        self.visualizer_stack = QStackedWidget()
        self.motion_trail = MotionTrailWidget()
        self.pulse_widget = SurfacePulseWidget()
        self.posture_hud = PostureHUDWidget()
        self.visualizer_stack.addWidget(self.motion_trail)
        self.visualizer_stack.addWidget(self.pulse_widget)
        self.visualizer_stack.addWidget(self.posture_hud)
        vg_layout.addWidget(self.visualizer_stack)
        left_layout.addWidget(self.visualizer_group, stretch=3)

        # Telemetry Stats Badge
        stats_box = QFrame()
        stats_box.setStyleSheet("background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px;")
        sb_layout = QHBoxLayout(stats_box)
        sb_layout.setContentsMargins(6, 4, 6, 4)

        self.lbl_fps_val = QLabel("15.0 FPS")
        self.lbl_fps_val.setStyleSheet("color: #047857; font-weight: bold; font-family: Consolas; font-size: 12px;")
        sb_layout.addWidget(self.lbl_fps_val)

        self.lbl_frame_cnt = QLabel("Frame: 0")
        self.lbl_frame_cnt.setStyleSheet("color: #475569; font-family: Consolas; font-size: 12px;")
        sb_layout.addWidget(self.lbl_frame_cnt)

        self.lbl_min_d = QLabel("Min: --- mm")
        self.lbl_min_d.setStyleSheet("color: #1d4ed8; font-weight: bold; font-family: Consolas; font-size: 12px;")
        sb_layout.addWidget(self.lbl_min_d)

        self.lbl_active_z = QLabel("Active: 0/64")
        self.lbl_active_z.setStyleSheet("color: #b45309; font-weight: bold; font-family: Consolas; font-size: 12px;")
        sb_layout.addWidget(self.lbl_active_z)

        left_layout.addWidget(stats_box, stretch=0)
        splitter.addWidget(left_widget)

        # Right Pane
        self.tabs_stack = QStackedWidget()

        # 1. Gesture Tabs
        self.tabs_gesture = QTabWidget()
        self.tab_gesture_collector = self.create_gesture_collector_tab()
        self.tab_gesture_trainer = self.create_gesture_trainer_tab()
        self.tab_gesture_inference = self.create_gesture_inference_tab()
        self.tabs_gesture.addTab(self.tab_gesture_collector, "  Capture Gestures  ")
        self.tabs_gesture.addTab(self.tab_gesture_trainer, "  Train MLP Model  ")
        self.tabs_gesture.addTab(self.tab_gesture_inference, "  Live Gesture Inference  ")
        self.tabs_stack.addWidget(self.tabs_gesture)

        # 2. Surface Tabs
        self.tabs_surface = QTabWidget()
        self.tab_surface_collector = self.create_surface_collector_tab()
        self.tab_surface_trainer = self.create_surface_trainer_tab()
        self.tab_surface_inference = self.create_surface_inference_tab()
        self.tabs_surface.addTab(self.tab_surface_collector, "  Capture CNH Pulses  ")
        self.tabs_surface.addTab(self.tab_surface_trainer, "  Train 1D-CNN Model  ")
        self.tabs_surface.addTab(self.tab_surface_inference, "  Live Surface Inference  ")
        self.tabs_stack.addWidget(self.tabs_surface)

        # 3. Posture Tabs
        self.tabs_posture = QTabWidget()
        self.tab_posture_collector = self.create_posture_collector_tab()
        self.tab_posture_trainer = self.create_posture_trainer_tab()
        self.tab_posture_inference = self.create_posture_inference_tab()
        self.tabs_posture.addTab(self.tab_posture_collector, "  Capture Postures  ")
        self.tabs_posture.addTab(self.tab_posture_trainer, "  Train 2D-CNN Model  ")
        self.tabs_posture.addTab(self.tab_posture_inference, "  Live Posture Inference  ")
        self.tabs_stack.addWidget(self.tabs_posture)

        splitter.addWidget(self.tabs_stack)
        splitter.setSizes([450, 850])
        main_layout.addWidget(splitter)

    def update_mode_buttons_style(self):
        btn_inactive = "background-color: transparent; color: #475569; font-weight: bold; border: none; border-radius: 4px; padding: 4px 12px;"
        if self.current_mode == "GESTURE":
            self.btn_mode_gesture.setStyleSheet("background-color: #1d4ed8; color: #ffffff; font-weight: bold; border: none; border-radius: 4px; padding: 4px 12px;")
            self.btn_mode_surface.setStyleSheet(btn_inactive)
            self.btn_mode_posture.setStyleSheet(btn_inactive)
        elif self.current_mode == "SURFACE":
            self.btn_mode_surface.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold; border: none; border-radius: 4px; padding: 4px 12px;")
            self.btn_mode_gesture.setStyleSheet(btn_inactive)
            self.btn_mode_posture.setStyleSheet(btn_inactive)
        else: # POSTURE
            self.btn_mode_posture.setStyleSheet("background-color: #7c3aed; color: #ffffff; font-weight: bold; border: none; border-radius: 4px; padding: 4px 12px;")
            self.btn_mode_gesture.setStyleSheet(btn_inactive)
            self.btn_mode_surface.setStyleSheet(btn_inactive)

    def switch_capability_mode(self, mode):
        self.current_mode = mode
        self.update_mode_buttons_style()

        if mode == "GESTURE":
            self.btn_mode_gesture.setChecked(True)
            self.btn_mode_surface.setChecked(False)
            self.heatmap_group.setTitle("Live Depth Heatmap (8x8 ToF)")
            self.visualizer_group.setTitle("2D Centroid Motion Trail (Gesture)")
            self.heatmap_stack.setCurrentIndex(0)
            self.visualizer_stack.setCurrentIndex(0)
            self.tabs_stack.setCurrentIndex(0)
            self.lbl_active_z.setText("Active: 0/64")

            if hasattr(self, 'worker') and self.worker:
                self.worker.send_command("AT+RES=8X8")
                time.sleep(0.05)
                self.worker.send_command("AT+MODE=GESTURE")

            if self.gesture_trained_model is not None:
                self.lbl_model_badge.setText("Gesture: Trained")
                self.lbl_model_badge.setStyleSheet("background-color: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; padding: 3px 10px; border-radius: 4px; font-weight: bold; font-size: 11px;")
            else:
                self.lbl_model_badge.setText("Gesture: Untrained")
                self.lbl_model_badge.setStyleSheet("background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; padding: 3px 10px; border-radius: 4px; font-weight: bold; font-size: 11px;")

        elif mode == "SURFACE":
            self.btn_mode_surface.setChecked(True)
            self.btn_mode_gesture.setChecked(False)
            self.btn_mode_posture.setChecked(False)
            self.heatmap_group.setTitle("Live 4x4 Grid Heatmap (Click Zone to Inspect CNH)")
            self.visualizer_group.setTitle("Live 48-Bin Pulse & 16-Bin Canonical Window (CNH)")
            self.heatmap_stack.setCurrentIndex(1)
            self.visualizer_stack.setCurrentIndex(1)
            self.tabs_stack.setCurrentIndex(1)
            self.lbl_active_z.setText("Active: 0/16")

            if hasattr(self, 'worker') and self.worker:
                self.worker.send_command("AT+RES=4X4")
                time.sleep(0.05)
                self.worker.send_command("AT+MODE=SURFACE")

            if self.surface_trained_model is not None:
                self.lbl_model_badge.setText("Surface 1D-CNN: Trained")
                self.lbl_model_badge.setStyleSheet("background-color: #ecfdf5; color: #0284c7; border: 1px solid #bae6fd; padding: 3px 10px; border-radius: 4px; font-weight: bold; font-size: 11px;")
            else:
                self.lbl_model_badge.setText("Surface 1D-CNN: Untrained")
                self.lbl_model_badge.setStyleSheet("background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; padding: 3px 10px; border-radius: 4px; font-weight: bold; font-size: 11px;")

        else: # POSTURE
            self.btn_mode_posture.setChecked(True)
            self.btn_mode_gesture.setChecked(False)
            self.btn_mode_surface.setChecked(False)
            self.heatmap_group.setTitle("Live Depth Heatmap (8x8 ToF - Hand Posture)")
            self.visualizer_group.setTitle("Hand Posture 2D-CNN Classifier & Debounce HUD")
            self.heatmap_stack.setCurrentIndex(0)
            self.visualizer_stack.setCurrentIndex(2)
            self.tabs_stack.setCurrentIndex(2)
            self.lbl_active_z.setText("Active: 0/64")

            if hasattr(self, 'worker') and self.worker:
                self.worker.send_command("AT+RES=8X8")
                time.sleep(0.05)
                self.worker.send_command("AT+MODE=POSTURE")

            if self.posture_trained_model is not None:
                self.lbl_model_badge.setText("Posture 2D-CNN: Trained")
                self.lbl_model_badge.setStyleSheet("background-color: #f5f3ff; color: #7c3aed; border: 1px solid #ddd6fe; padding: 3px 10px; border-radius: 4px; font-weight: bold; font-size: 11px;")
            else:
                self.lbl_model_badge.setText("Posture 2D-CNN: Baseline")
                self.lbl_model_badge.setStyleSheet("background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; padding: 3px 10px; border-radius: 4px; font-weight: bold; font-size: 11px;")

    def on_surface_zone_selected(self, zone_idx):
        self.surface_selected_zone = zone_idx

    # ------------------------------------------------------------------
    # Keyboard Shortcuts
    # ------------------------------------------------------------------
    def setup_shortcuts(self):
        # Gesture shortcuts
        QShortcut(QKeySequence("5"), self, lambda: self.handle_shortcut("IDLE", "GESTURE"))
        QShortcut(QKeySequence("4"), self, lambda: self.handle_shortcut("SWIPE_LEFT", "GESTURE"))
        QShortcut(QKeySequence("6"), self, lambda: self.handle_shortcut("SWIPE_RIGHT", "GESTURE"))
        QShortcut(QKeySequence("8"), self, lambda: self.handle_shortcut("SWIPE_UP", "GESTURE"))
        QShortcut(QKeySequence("2"), self, lambda: self.handle_shortcut("SWIPE_DOWN", "GESTURE"))

        # Surface shortcuts
        QShortcut(QKeySequence("1"), self, lambda: self.handle_shortcut("HARD_FLOOR", "SURFACE"))
        QShortcut(QKeySequence("3"), self, lambda: self.handle_shortcut("CARPET", "SURFACE"))
        QShortcut(QKeySequence("7"), self, lambda: self.handle_shortcut("SPECULAR", "SURFACE"))
        QShortcut(QKeySequence("9"), self, lambda: self.handle_shortcut("VOID", "SURFACE"))

        # Posture shortcuts
        QShortcut(QKeySequence("0"), self, lambda: self.handle_shortcut("NONE", "POSTURE"))
        QShortcut(QKeySequence("F"), self, lambda: self.handle_shortcut("FLAT_HAND", "POSTURE"))
        QShortcut(QKeySequence("L"), self, lambda: self.handle_shortcut("LIKE", "POSTURE"))
        QShortcut(QKeySequence("D"), self, lambda: self.handle_shortcut("DISLIKE", "POSTURE"))
        QShortcut(QKeySequence("B"), self, lambda: self.handle_shortcut("BREAK_TIME", "POSTURE"))
        QShortcut(QKeySequence("T"), self, lambda: self.handle_shortcut("FIST", "POSTURE"))

    def handle_shortcut(self, class_name, mode):
        if self.current_mode == mode:
            if mode == "GESTURE":
                self.start_gesture_capture(class_name)
            elif mode == "SURFACE":
                self.start_surface_capture(class_name, burst=False)
            elif mode == "POSTURE":
                self.start_posture_capture(class_name, burst=False)

    # ------------------------------------------------------------------
    # Tab: Gesture Data Collection
    # ------------------------------------------------------------------
    def create_gesture_collector_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        info_lbl = QLabel(
            "<b>Gesture Data Acquisition Protocol:</b><br>"
            "Trigger any capture button below (or keys <b>5, 4, 6, 8, 2</b>) to record a continuous 15-frame burst (1.0 s @ 15 Hz). "
            "Collect 10-15 physical repetitions per gesture class for robust model generalization."
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px; color: #334155; font-size: 12px; line-height: 1.4;")
        layout.addWidget(info_lbl)

        self.lbl_gesture_rec_status = QLabel("Ready for acquisition. Trigger via keys [5, 4, 6, 8, 2] or click capture button.")
        self.lbl_gesture_rec_status.setAlignment(Qt.AlignCenter)
        self.lbl_gesture_rec_status.setStyleSheet("background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 13px;")
        layout.addWidget(self.lbl_gesture_rec_status)

        btn_grid = QGridLayout()
        btn_grid.setSpacing(8)
        self.gesture_sample_labels = {}
        self.gesture_sample_bars = {}

        button_specs = [
            ("IDLE", "[5] Capture IDLE (Resting)", "#475569", 0, 0, 1, 2),
            ("SWIPE_LEFT", "[4] Capture SWIPE LEFT", "#1d4ed8", 1, 0, 1, 1),
            ("SWIPE_RIGHT", "[6] Capture SWIPE RIGHT", "#047857", 1, 1, 1, 1),
            ("SWIPE_UP", "[8] Capture SWIPE UP", "#d97706", 2, 0, 1, 1),
            ("SWIPE_DOWN", "[2] Capture SWIPE DOWN", "#b91c1c", 2, 1, 1, 1),
        ]

        for cls_name, btn_text, col, r, c, rspan, cspan in button_specs:
            box = QGroupBox(cls_name)
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(8, 8, 8, 8)

            btn = QPushButton(btn_text)
            btn.setStyleSheet(f"background-color: {col}; color: #ffffff; font-size: 12px; font-weight: bold; padding: 9px; border-radius: 4px; border: none;")
            btn.clicked.connect(lambda checked, cn=cls_name: self.start_gesture_capture(cn))
            box_layout.addWidget(btn)

            row_stat = QHBoxLayout()
            pbar = QProgressBar()
            pbar.setRange(0, 15)
            pbar.setValue(0)
            pbar.setFixedHeight(8)
            pbar.setTextVisible(False)
            pbar.setStyleSheet(f"QProgressBar {{ background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; }} QProgressBar::chunk {{ background-color: {col}; border-radius: 3px; }}")
            row_stat.addWidget(pbar)
            self.gesture_sample_bars[cls_name] = pbar

            cnt_lbl = QLabel("0 / 15")
            cnt_lbl.setStyleSheet("color: #334155; font-size: 11px; font-family: Consolas; font-weight: bold;")
            row_stat.addWidget(cnt_lbl)
            self.gesture_sample_labels[cls_name] = cnt_lbl

            box_layout.addLayout(row_stat)
            btn_grid.addWidget(box, r, c, rspan, cspan)

        layout.addLayout(btn_grid)

        ds_box = QGroupBox("Gesture Dataset Management")
        ds_layout = QHBoxLayout(ds_box)
        btn_save_ds = QPushButton("Save Dataset (.npz)")
        btn_save_ds.clicked.connect(self.save_gesture_dataset)
        ds_layout.addWidget(btn_save_ds)

        btn_load_ds = QPushButton("Load Dataset (.npz)")
        btn_load_ds.clicked.connect(self.load_gesture_dataset)
        ds_layout.addWidget(btn_load_ds)

        btn_clear = QPushButton("Reset All Recordings")
        btn_clear.clicked.connect(self.reset_gesture_recordings)
        ds_layout.addWidget(btn_clear)
        layout.addWidget(ds_box)

        layout.addStretch()
        return tab

    # ------------------------------------------------------------------
    # Tab: Gesture Model Trainer
    # ------------------------------------------------------------------
    def create_gesture_trainer_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        desc = QLabel(
            "<b>Gesture Model Architecture:</b> Multi-Layer Perceptron (MLP) with Spatial-Temporal Feature Engineering<br>"
            "Feature Vector: <b>241-Dimensional Orthogonal Depth Vector</b> (Mean/Max Depth, Velocity Deltas, Energy Balances)<br>"
            "Topology: <b>Dense(64, ReLU) &rarr; Dense(32, ReLU) &rarr; Softmax(5)</b> | Target: STM32 Cortex-M33"
        )
        desc.setStyleSheet("background-color: #f8fafc; border: 1px solid #cbd5e1; padding: 12px; border-radius: 6px; color: #1e293b; line-height: 1.4;")
        layout.addWidget(desc)

        btn_row = QHBoxLayout()
        self.btn_train_gesture = QPushButton("Train Gesture Neural Network")
        self.btn_train_gesture.setStyleSheet("background-color: #1d4ed8; color: #ffffff; font-size: 13px; font-weight: bold; padding: 10px 18px; border-radius: 5px; border: none;")
        self.btn_train_gesture.clicked.connect(self.train_gesture_model)
        btn_row.addWidget(self.btn_train_gesture)

        self.btn_deploy_gesture = QPushButton("Deploy Gesture Model to MCU")
        self.btn_deploy_gesture.setStyleSheet("background-color: #047857; color: #ffffff; font-size: 13px; font-weight: bold; padding: 10px 18px; border-radius: 5px; border: none;")
        self.btn_deploy_gesture.clicked.connect(self.deploy_gesture_model)
        btn_row.addWidget(self.btn_deploy_gesture)
        layout.addLayout(btn_row)

        self.txt_gesture_train_log = QTextEdit()
        self.txt_gesture_train_log.setReadOnly(True)
        self.txt_gesture_train_log.setFont(QFont("Consolas", 10))
        self.txt_gesture_train_log.setStyleSheet("background-color: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px;")
        self.txt_gesture_train_log.setPlaceholderText("Gesture model training log and validation metrics will appear here...")
        layout.addWidget(self.txt_gesture_train_log)
        return tab

    # ------------------------------------------------------------------
    # Tab: Gesture Live Inference
    # ------------------------------------------------------------------
    def create_gesture_inference_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self.gesture_banner_frame = QFrame()
        self.gesture_banner_frame.setStyleSheet("background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px;")
        bf_layout = QVBoxLayout(self.gesture_banner_frame)
        self.lbl_gesture_inference_result = QLabel("IDLE (Model Ready - Wave hand across 8x8 sensor)")
        self.lbl_gesture_inference_result.setAlignment(Qt.AlignCenter)
        self.lbl_gesture_inference_result.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.lbl_gesture_inference_result.setStyleSheet("color: #475569;")
        bf_layout.addWidget(self.lbl_gesture_inference_result)
        layout.addWidget(self.gesture_banner_frame)

        bars_group = QGroupBox("Live Gesture Probability Distribution (Softmax Confidence)")
        bg_layout = QVBoxLayout(bars_group)
        bg_layout.setSpacing(6)

        self.gesture_prob_bars = {}
        self.gesture_prob_labels = {}

        for cls_name in GESTURE_CLASSES:
            row = QHBoxLayout()
            name_lbl = QLabel(f"<b>{cls_name:12s}</b>")
            name_lbl.setFixedWidth(130)
            name_lbl.setStyleSheet(f"color: {GESTURE_COLORS[cls_name]}; font-family: Consolas; font-size: 13px; font-weight: bold;")
            row.addWidget(name_lbl)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setStyleSheet(f"""
                QProgressBar {{ background: #f1f5f9; border-radius: 4px; height: 18px; border: 1px solid #cbd5e1; }}
                QProgressBar::chunk {{ background-color: {GESTURE_COLORS[cls_name]}; border-radius: 3px; }}
            """)
            row.addWidget(bar)
            self.gesture_prob_bars[cls_name] = bar

            pct_lbl = QLabel("  0.0%")
            pct_lbl.setFixedWidth(70)
            pct_lbl.setFont(QFont("Consolas", 11, QFont.Bold))
            pct_lbl.setStyleSheet("color: #0f172a;")
            row.addWidget(pct_lbl)
            self.gesture_prob_labels[cls_name] = pct_lbl
            bg_layout.addLayout(row)

        layout.addWidget(bars_group)

        wave_group = QGroupBox("Live Directional Energy Waveform (Left/Right vs Up/Down Shifts)")
        wg_layout = QVBoxLayout(wave_group)
        self.waveform_widget = EnergyWaveformWidget()
        wg_layout.addWidget(self.waveform_widget)
        layout.addWidget(wave_group, stretch=1)

        history_group = QGroupBox("Recent Gesture Detections Log")
        hg_layout = QVBoxLayout(history_group)
        self.gesture_history_table = QTableWidget()
        self.gesture_history_table.setColumnCount(4)
        self.gesture_history_table.setHorizontalHeaderLabels(["Timestamp", "Gesture Name", "Confidence", "Status"])
        self.gesture_history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.gesture_history_table.verticalHeader().setVisible(False)
        self.gesture_history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.gesture_history_table.setFixedHeight(130)
        self.gesture_history_table.setAlternatingRowColors(True)
        self.gesture_history_table.setStyleSheet("QTableWidget { alternate-background-color: #f8fafc; background-color: #ffffff; }")
        hg_layout.addWidget(self.gesture_history_table)
        layout.addWidget(history_group, stretch=1)

        return tab

    # ------------------------------------------------------------------
    # Tab: Surface Data Collection (Distance-Invariant 16-Bin Canonical)
    # ------------------------------------------------------------------
    def create_surface_collector_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        info_lbl = QLabel(
            "<b>Distance-Invariant Surface Optical Acquisition (16-Bin Canonical Window @ 25 Hz):</b><br>"
            "Each recorded frame applies <b>ST 3-tap binomial smoothing</b> and <b>peak-relative centering</b>. "
            "The optical reflection peak is anchored at Index 4, making the model <b>100% immune to distance variations from 10 to 50 cm</b>!<br>"
            "Click a capture button (or hotkeys <b>1: Hard Floor, 3: Carpet, 7: Specular, 9: Void</b>) to record."
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 10px; color: #166534; font-size: 12px; line-height: 1.4;")
        layout.addWidget(info_lbl)

        self.lbl_surface_rec_status = QLabel("Ready for surface acquisition. Position sensor ~10-40 cm above material.")
        self.lbl_surface_rec_status.setAlignment(Qt.AlignCenter)
        self.lbl_surface_rec_status.setStyleSheet("background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 13px;")
        layout.addWidget(self.lbl_surface_rec_status)

        btn_grid = QGridLayout()
        btn_grid.setSpacing(8)
        self.surface_sample_labels = {}
        self.surface_sample_bars = {}

        surface_specs = [
            ("HARD_FLOOR", "[1] HARD FLOOR (Tile/Wood/Ceramic)", "#0284c7", 0, 0),
            ("CARPET", "[3] CARPET (Rug/Fabric/Broad Pulse)", "#d97706", 0, 1),
            ("SPECULAR", "[7] SPECULAR (Mirror/Stainless Steel)", "#7c3aed", 1, 0),
            ("VOID", "[9] VOID (Drop-Off/Cliff/Air)", "#64748b", 1, 1),
        ]

        for cls_name, btn_text, col, r, c in surface_specs:
            box = QGroupBox(cls_name)
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(8, 8, 8, 8)

            btn_row = QHBoxLayout()
            btn_snap = QPushButton(f"Snap 1 Frame")
            btn_snap.setStyleSheet(f"background-color: {col}; color: #ffffff; font-size: 11px; font-weight: bold; padding: 7px; border-radius: 4px; border: none;")
            btn_snap.clicked.connect(lambda _, cn=cls_name: self.start_surface_capture(cn, burst=False))
            btn_row.addWidget(btn_snap)

            btn_burst = QPushButton(f"Burst (10 Frames)")
            btn_burst.setStyleSheet(f"background-color: #f1f5f9; color: {col}; font-size: 11px; font-weight: bold; padding: 7px; border-radius: 4px; border: 1px solid {col};")
            btn_burst.clicked.connect(lambda _, cn=cls_name: self.start_surface_capture(cn, burst=True))
            btn_row.addWidget(btn_burst)
            box_layout.addLayout(btn_row)

            row_stat = QHBoxLayout()
            pbar = QProgressBar()
            pbar.setRange(0, 50)
            pbar.setValue(0)
            pbar.setFixedHeight(8)
            pbar.setTextVisible(False)
            pbar.setStyleSheet(f"QProgressBar {{ background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; }} QProgressBar::chunk {{ background-color: {col}; border-radius: 3px; }}")
            row_stat.addWidget(pbar)
            self.surface_sample_bars[cls_name] = pbar

            cnt_lbl = QLabel("0 / 50")
            cnt_lbl.setStyleSheet("color: #334155; font-size: 11px; font-family: Consolas; font-weight: bold;")
            row_stat.addWidget(cnt_lbl)
            self.surface_sample_labels[cls_name] = cnt_lbl

            box_layout.addLayout(row_stat)
            btn_grid.addWidget(box, r, c)

        layout.addLayout(btn_grid)

        ds_box = QGroupBox("Surface Dataset Management")
        ds_layout = QHBoxLayout(ds_box)
        btn_save_ds = QPushButton("Save Surface Dataset (.npz)")
        btn_save_ds.clicked.connect(self.save_surface_dataset)
        ds_layout.addWidget(btn_save_ds)

        btn_load_ds = QPushButton("Load Surface Dataset (.npz)")
        btn_load_ds.clicked.connect(self.load_surface_dataset)
        ds_layout.addWidget(btn_load_ds)

        btn_clear = QPushButton("Reset All Surface Recordings")
        btn_clear.clicked.connect(self.reset_surface_recordings)
        ds_layout.addWidget(btn_clear)
        layout.addWidget(ds_box)

        layout.addStretch()
        return tab

    # ------------------------------------------------------------------
    # Tab: Surface 1D-CNN Model Trainer
    # ------------------------------------------------------------------
    def create_surface_trainer_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        desc = QLabel(
            "<b>Distance-Invariant 1D-CNN Architecture:</b> 1D Convolution with Peak-Relative Canonical Window<br>"
            "Input Vector: <b>16-Bin Peak-Centered Normalized Canonical Pulse</b> (Index 4 is ALWAYS the peak!)<br>"
            "Topology: <b>Conv1D(8, k=3) &rarr; MaxPool(2) &rarr; Conv1D(16, k=3) &rarr; MaxPool(2) &rarr; Dense(16) &rarr; Dense(4)</b><br>"
            "Edge AI Metrics: <b>704 Bytes RAM, 6.1 KB Flash, &lt; 20 &mu;s execution on Cortex-M33</b>"
        )
        desc.setStyleSheet("background-color: #f8fafc; border: 1px solid #cbd5e1; padding: 12px; border-radius: 6px; color: #1e293b; line-height: 1.4;")
        layout.addWidget(desc)

        btn_row = QHBoxLayout()
        self.btn_train_surface = QPushButton("Train Distance-Invariant 1D-CNN Model")
        self.btn_train_surface.setStyleSheet("background-color: #0284c7; color: #ffffff; font-size: 13px; font-weight: bold; padding: 10px 18px; border-radius: 5px; border: none;")
        self.btn_train_surface.clicked.connect(self.train_surface_model)
        btn_row.addWidget(self.btn_train_surface)

        self.btn_deploy_surface = QPushButton("Deploy Surface Model to MCU")
        self.btn_deploy_surface.setStyleSheet("background-color: #047857; color: #ffffff; font-size: 13px; font-weight: bold; padding: 10px 18px; border-radius: 5px; border: none;")
        self.btn_deploy_surface.clicked.connect(self.deploy_surface_model)
        btn_row.addWidget(self.btn_deploy_surface)
        layout.addLayout(btn_row)

        self.txt_surface_train_log = QTextEdit()
        self.txt_surface_train_log.setReadOnly(True)
        self.txt_surface_train_log.setFont(QFont("Consolas", 10))
        self.txt_surface_train_log.setStyleSheet("background-color: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px;")
        self.txt_surface_train_log.setPlaceholderText("Surface 1D-CNN training logs, validation accuracy, and ONNX export details...")
        layout.addWidget(self.txt_surface_train_log)
        return tab

    # ------------------------------------------------------------------
    # Tab: Surface Live Inference (Multi-Zone Ensemble + Hysteresis)
    # ------------------------------------------------------------------
    def create_surface_inference_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self.surface_banner_frame = QFrame()
        self.surface_banner_frame.setStyleSheet("background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px;")
        bf_layout = QVBoxLayout(self.surface_banner_frame)
        self.lbl_surface_inference_result = QLabel("SURFACE: SCANNING (Place ToF above surface)")
        self.lbl_surface_inference_result.setAlignment(Qt.AlignCenter)
        self.lbl_surface_inference_result.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.lbl_surface_inference_result.setStyleSheet("color: #475569;")
        bf_layout.addWidget(self.lbl_surface_inference_result)
        layout.addWidget(self.surface_banner_frame)

        bars_group = QGroupBox("Live Surface Probability Distribution (16-Zone Ensemble & Debounced)")
        bg_layout = QVBoxLayout(bars_group)
        bg_layout.setSpacing(6)

        self.surface_prob_bars = {}
        self.surface_prob_labels = {}

        for cls_name in SURFACE_CLASSES:
            row = QHBoxLayout()
            name_lbl = QLabel(f"<b>{cls_name:12s}</b>")
            name_lbl.setFixedWidth(130)
            name_lbl.setStyleSheet(f"color: {SURFACE_COLORS[cls_name]}; font-family: Consolas; font-size: 13px; font-weight: bold;")
            row.addWidget(name_lbl)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setStyleSheet(f"""
                QProgressBar {{ background: #f1f5f9; border-radius: 4px; height: 18px; border: 1px solid #cbd5e1; }}
                QProgressBar::chunk {{ background-color: {SURFACE_COLORS[cls_name]}; border-radius: 3px; }}
            """)
            row.addWidget(bar)
            self.surface_prob_bars[cls_name] = bar

            pct_lbl = QLabel("  0.0%")
            pct_lbl.setFixedWidth(70)
            pct_lbl.setFont(QFont("Consolas", 11, QFont.Bold))
            pct_lbl.setStyleSheet("color: #0f172a;")
            row.addWidget(pct_lbl)
            self.surface_prob_labels[cls_name] = pct_lbl
            bg_layout.addLayout(row)

        layout.addWidget(bars_group)

        # Pulse Physical Telemetry Cards
        metric_group = QGroupBox("Optical Physics Telemetry (Sub-Bin Interpolation from ST cnh_lib.py)")
        mg_layout = QGridLayout(metric_group)
        mg_layout.setSpacing(8)

        self.lbl_tele_fwhm = QLabel("FWHM: --- mm")
        self.lbl_tele_fwhm.setStyleSheet("color: #1e293b; font-family: Consolas; font-weight: bold; font-size: 13px;")
        mg_layout.addWidget(self.lbl_tele_fwhm, 0, 0)

        self.lbl_tele_peak = QLabel("Peak: --- kcps/SPAD")
        self.lbl_tele_peak.setStyleSheet("color: #0284c7; font-family: Consolas; font-weight: bold; font-size: 13px;")
        mg_layout.addWidget(self.lbl_tele_peak, 0, 1)

        self.lbl_tele_dist = QLabel("Distance: --- mm")
        self.lbl_tele_dist.setStyleSheet("color: #047857; font-family: Consolas; font-weight: bold; font-size: 13px;")
        mg_layout.addWidget(self.lbl_tele_dist, 1, 0)

        self.lbl_tele_tail = QLabel("Decay Tail Ratio: ---")
        self.lbl_tele_tail.setStyleSheet("color: #d97706; font-family: Consolas; font-weight: bold; font-size: 13px;")
        mg_layout.addWidget(self.lbl_tele_tail, 1, 1)

        layout.addWidget(metric_group)

        # Event history table
        history_group = QGroupBox("Confirmed Surface Classification History Log")
        hg_layout = QVBoxLayout(history_group)
        self.surface_history_table = QTableWidget()
        self.surface_history_table.setColumnCount(4)
        self.surface_history_table.setHorizontalHeaderLabels(["Timestamp", "Confirmed Surface", "Confidence", "FWHM (mm)"])
        self.surface_history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.surface_history_table.verticalHeader().setVisible(False)
        self.surface_history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.surface_history_table.setFixedHeight(130)
        self.surface_history_table.setAlternatingRowColors(True)
        self.surface_history_table.setStyleSheet("QTableWidget { alternate-background-color: #f8fafc; background-color: #ffffff; }")
        hg_layout.addWidget(self.surface_history_table)
        layout.addWidget(history_group, stretch=1)

        return tab

    # ------------------------------------------------------------------
    # Tab: Hand Posture Data Collection (STSW-IMG050 2D-CNN 8x8 Dual-Channel)
    # ------------------------------------------------------------------
    def create_posture_collector_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        info_lbl = QLabel(
            "<b>Dual-Channel 2D-CNN Hand Posture Acquisition (8x8 ToF @ 15 Hz):</b><br>"
            "Records both <b>Distance (mm)</b> and <b>Peak Signal (kcps/SPAD)</b> channels. "
            "Applies <b>ST Background Removal</b> (cutoff = min_dist + 120 mm) and <b>RobustScaler normalization</b>. "
            "Valid operational range is <b>100 mm to 400 mm</b> above sensor.<br>"
            "Click a snap/burst button (or hotkeys <b>0: None, F: Flat Hand, L: Like, D: Dislike, B: Break Time, T: Fist</b>) to record."
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("background-color: #f5f3ff; border: 1px solid #ddd6fe; border-radius: 6px; padding: 10px; color: #5b21b6; font-size: 12px; line-height: 1.4;")
        layout.addWidget(info_lbl)

        self.lbl_posture_rec_status = QLabel("Ready for posture acquisition. Position hand 10-40 cm above sensor.")
        self.lbl_posture_rec_status.setAlignment(Qt.AlignCenter)
        self.lbl_posture_rec_status.setStyleSheet("background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 13px;")
        layout.addWidget(self.lbl_posture_rec_status)

        btn_grid = QGridLayout()
        btn_grid.setSpacing(8)
        self.posture_sample_labels = {}
        self.posture_sample_bars = {}

        posture_specs = [
            ("NONE", "[0] NONE (No Hand / Background)", "#64748b", 0, 0),
            ("FLAT_HAND", "[F] FLAT HAND (Open Palm)", "#0284c7", 0, 1),
            ("LIKE", "[L] LIKE (Thumbs Up)", "#059669", 1, 0),
            ("DISLIKE", "[D] DISLIKE (Thumbs Down)", "#dc2626", 1, 1),
            ("BREAK_TIME", "[B] BREAK TIME (Timeout T-sign)", "#d97706", 2, 0),
            ("FIST", "[T] FIST (Closed Fist)", "#7c3aed", 2, 1),
        ]

        for cls_name, btn_text, col, r, c in posture_specs:
            box = QGroupBox(cls_name)
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(8, 8, 8, 8)

            btn_row = QHBoxLayout()
            btn_snap = QPushButton(f"Snap 1 Frame")
            btn_snap.setStyleSheet(f"background-color: {col}; color: #ffffff; font-size: 11px; font-weight: bold; padding: 7px; border-radius: 4px; border: none;")
            btn_snap.clicked.connect(lambda _, cn=cls_name: self.start_posture_capture(cn, burst=False))
            btn_row.addWidget(btn_snap)

            btn_burst = QPushButton(f"Burst (15 Frames)")
            btn_burst.setStyleSheet(f"background-color: #f1f5f9; color: {col}; font-size: 11px; font-weight: bold; padding: 7px; border-radius: 4px; border: 1px solid {col};")
            btn_burst.clicked.connect(lambda _, cn=cls_name: self.start_posture_capture(cn, burst=True))
            btn_row.addWidget(btn_burst)
            box_layout.addLayout(btn_row)

            row_stat = QHBoxLayout()
            pbar = QProgressBar()
            pbar.setRange(0, 30)
            pbar.setValue(0)
            pbar.setFixedHeight(8)
            pbar.setTextVisible(False)
            pbar.setStyleSheet(f"QProgressBar {{ background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; }} QProgressBar::chunk {{ background-color: {col}; border-radius: 3px; }}")
            row_stat.addWidget(pbar)
            self.posture_sample_bars[cls_name] = pbar

            cnt_lbl = QLabel("0 / 30")
            cnt_lbl.setStyleSheet("color: #334155; font-size: 11px; font-family: Consolas; font-weight: bold;")
            row_stat.addWidget(cnt_lbl)
            self.posture_sample_labels[cls_name] = cnt_lbl

            box_layout.addLayout(row_stat)
            btn_grid.addWidget(box, r, c)

        layout.addLayout(btn_grid)

        ds_box = QGroupBox("Hand Posture Dataset Management")
        ds_layout = QHBoxLayout(ds_box)
        btn_save_ds = QPushButton("Save Posture Dataset (.npz)")
        btn_save_ds.clicked.connect(self.save_posture_dataset)
        ds_layout.addWidget(btn_save_ds)

        btn_load_ds = QPushButton("Load Posture Dataset (.npz)")
        btn_load_ds.clicked.connect(self.load_posture_dataset)
        ds_layout.addWidget(btn_load_ds)

        btn_clear = QPushButton("Reset All Posture Recordings")
        btn_clear.clicked.connect(self.reset_posture_recordings)
        ds_layout.addWidget(btn_clear)
        layout.addWidget(ds_box)

        layout.addStretch()
        return tab

    # ------------------------------------------------------------------
    # Tab: Hand Posture 2D-CNN Model Trainer
    # ------------------------------------------------------------------
    def create_posture_trainer_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        desc = QLabel(
            "<b>Dual-Channel 2D-CNN Architecture (STMicroelectronics STSW-IMG050 Aligned):</b><br>"
            "Input Tensor: <b>(2, 8, 8)</b> [Channel 0: RobustScaled Distance, Channel 1: RobustScaled Peak Signal]<br>"
            "Topology: <b>Conv2D(8, 3x3) &rarr; ReLU &rarr; MaxPool2D(2x2) &rarr; Dense(32) &rarr; Dense(6 Classes)</b><br>"
            "Edge AI Metrics: <b>1,096 Bytes RAM, 10.5 KB Flash, 8,434 MACCs, ~35 &mu;s execution on Cortex-M33</b>"
        )
        desc.setStyleSheet("background-color: #f8fafc; border: 1px solid #cbd5e1; padding: 12px; border-radius: 6px; color: #1e293b; line-height: 1.4;")
        layout.addWidget(desc)

        btn_row = QHBoxLayout()
        self.btn_train_posture = QPushButton("Train Hand Posture 2D-CNN Model")
        self.btn_train_posture.setStyleSheet("background-color: #7c3aed; color: #ffffff; font-size: 13px; font-weight: bold; padding: 10px 18px; border-radius: 5px; border: none;")
        self.btn_train_posture.clicked.connect(self.train_posture_model)
        btn_row.addWidget(self.btn_train_posture)

        self.btn_deploy_posture = QPushButton("Deploy Posture Model to MCU")
        self.btn_deploy_posture.setStyleSheet("background-color: #047857; color: #ffffff; font-size: 13px; font-weight: bold; padding: 10px 18px; border-radius: 5px; border: none;")
        self.btn_deploy_posture.clicked.connect(self.deploy_posture_model)
        btn_row.addWidget(self.btn_deploy_posture)
        layout.addLayout(btn_row)

        self.txt_posture_train_log = QTextEdit()
        self.txt_posture_train_log.setReadOnly(True)
        self.txt_posture_train_log.setFont(QFont("Consolas", 10))
        self.txt_posture_train_log.setStyleSheet("background-color: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px;")
        self.txt_posture_train_log.setPlaceholderText("Posture 2D-CNN training logs, validation accuracy, confusion matrix, and ONNX export details...")
        layout.addWidget(self.txt_posture_train_log)
        return tab

    # ------------------------------------------------------------------
    # Tab: Hand Posture Live Inference
    # ------------------------------------------------------------------
    def create_posture_inference_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self.posture_banner_frame = QFrame()
        self.posture_banner_frame.setStyleSheet("background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px;")
        bf_layout = QVBoxLayout(self.posture_banner_frame)
        self.lbl_posture_inference_result = QLabel("POSTURE: SCANNING (Place hand in 10-40 cm range)")
        self.lbl_posture_inference_result.setAlignment(Qt.AlignCenter)
        self.lbl_posture_inference_result.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.lbl_posture_inference_result.setStyleSheet("color: #475569;")
        bf_layout.addWidget(self.lbl_posture_inference_result)
        layout.addWidget(self.posture_banner_frame)

        bars_group = QGroupBox("Live Hand Posture Probability Distribution (2D-CNN & 3-Frame Debounce)")
        bg_layout = QVBoxLayout(bars_group)
        bg_layout.setSpacing(6)

        self.posture_infer_bars = {}
        self.posture_infer_labels = {}

        for cls_name in POSTURE_CLASSES:
            row = QHBoxLayout()
            name_lbl = QLabel(f"<b>{POSTURE_ICONS.get(cls_name, '')} {cls_name:12s}</b>")
            name_lbl.setFixedWidth(150)
            name_lbl.setStyleSheet(f"color: {POSTURE_COLORS[cls_name]}; font-family: Consolas; font-size: 13px; font-weight: bold;")
            row.addWidget(name_lbl)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setStyleSheet(f"""
                QProgressBar {{ background: #f1f5f9; border-radius: 4px; height: 18px; border: 1px solid #cbd5e1; }}
                QProgressBar::chunk {{ background-color: {POSTURE_COLORS[cls_name]}; border-radius: 3px; }}
            """)
            row.addWidget(bar)
            self.posture_infer_bars[cls_name] = bar

            pct_lbl = QLabel("  0.0%")
            pct_lbl.setFixedWidth(70)
            pct_lbl.setFont(QFont("Consolas", 11, QFont.Bold))
            pct_lbl.setStyleSheet("color: #0f172a;")
            row.addWidget(pct_lbl)
            self.posture_infer_labels[cls_name] = pct_lbl
            bg_layout.addLayout(row)

        layout.addWidget(bars_group)

        # Event history table
        history_group = QGroupBox("Confirmed Hand Posture Event Log (3-Frame Debounced)")
        hg_layout = QVBoxLayout(history_group)
        self.posture_history_table = QTableWidget()
        self.posture_history_table.setColumnCount(4)
        self.posture_history_table.setHorizontalHeaderLabels(["Timestamp", "Confirmed Posture", "Confidence", "Distance (mm)"])
        self.posture_history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.posture_history_table.verticalHeader().setVisible(False)
        self.posture_history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.posture_history_table.setFixedHeight(130)
        self.posture_history_table.setAlternatingRowColors(True)
        self.posture_history_table.setStyleSheet("QTableWidget { alternate-background-color: #f8fafc; background-color: #ffffff; }")
        hg_layout.addWidget(self.posture_history_table)
        layout.addWidget(history_group, stretch=1)

        return tab

        return tab

    # ------------------------------------------------------------------
    # Frame Reception & Dispatching
    # ------------------------------------------------------------------
    def on_frame_received(self, frame):
        self.total_frames += 1
        self.fps_frames += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_frames / (now - self.last_fps_time)
            self.fps_frames = 0
            self.last_fps_time = now
            self.lbl_fps_val.setText(f"{self.current_fps:.1f} FPS")

        self.lbl_frame_cnt.setText(f"Frame: #{self.total_frames}")

        dists = frame.get("distances", [])
        num_d = len(dists)

        if num_d == 64 and self.current_mode == "GESTURE":
            self.handle_gesture_frame(frame, dists)
        elif num_d == 16 and self.current_mode == "SURFACE":
            self.handle_surface_frame(frame, dists)
        elif num_d == 64 and self.current_mode == "POSTURE":
            self.handle_posture_frame(frame, dists)
        elif num_d == 16 and self.current_mode in ("GESTURE", "POSTURE"):
            self.switch_capability_mode("SURFACE")
            self.handle_surface_frame(frame, dists)
        elif num_d == 64 and self.current_mode == "SURFACE":
            self.switch_capability_mode("GESTURE")
            self.handle_gesture_frame(frame, dists)

    # ------------------------------------------------------------------
    # Gesture Mode Frame Handler
    # ------------------------------------------------------------------
    def handle_gesture_frame(self, frame, dists):
        valid = [(i, d) for i, d in enumerate(dists) if 50 <= d <= 400]
        self.lbl_active_z.setText(f"Active: {len(valid)}/64")
        if valid:
            min_d = min(d for _, d in valid)
            self.lbl_min_d.setText(f"Min: {min_d} mm")
            weights = [1.0 / max(1.0, d) for _, d in valid]
            w_sum = sum(weights)
            xs = [(((i % 8) - 3.5) * 35.0) * w for (i, d), w in zip(valid, weights)]
            ys = [(((i // 8) - 3.5) * 35.0) * w for (i, d), w in zip(valid, weights)]
            cx = sum(xs) / w_sum
            cy = sum(ys) / w_sum
            self.motion_trail.update_trail(cx, cy)
        else:
            self.lbl_min_d.setText("Min: --- mm")
            self.motion_trail.update_trail(None, None)

        self.heatmap_8x8.update_distances(dists)

        if self.is_gesture_recording:
            self.gesture_record_buffer.append(list(dists))
            if len(self.gesture_record_buffer) >= GESTURE_WINDOW_SIZE:
                self.is_gesture_recording = False
                self.gesture_dataset[self.gesture_target_class].append(np.array(self.gesture_record_buffer))
                count = len(self.gesture_dataset[self.gesture_target_class])
                self.gesture_sample_labels[self.gesture_target_class].setText(f"{count} / 15")
                self.gesture_sample_bars[self.gesture_target_class].setValue(min(15, count))
                self.lbl_gesture_rec_status.setText(f"Captured [{self.gesture_target_class}] Sample #{count} successfully.")
                self.lbl_gesture_rec_status.setStyleSheet("background-color: #ecfdf5; color: #047857; border: 1px solid #6ee7b7; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 13px;")

        self.gesture_sliding_window.append(dists)
        if len(self.gesture_sliding_window) > GESTURE_WINDOW_SIZE:
            self.gesture_sliding_window.pop(0)

        if len(self.gesture_sliding_window) == GESTURE_WINDOW_SIZE:
            self.waveform_widget.update_energy(self.gesture_sliding_window)
            if self.gesture_trained_model is not None:
                self.run_live_gesture_inference()

    # ------------------------------------------------------------------
    # Surface Mode Frame Handler (Multi-Zone Ensemble & Canonical Extraction)
    # ------------------------------------------------------------------
    def handle_surface_frame(self, frame, dists):
        valid = [(i, d) for i, d in enumerate(dists) if 50 <= d <= 1800]
        self.lbl_active_z.setText(f"Active: {len(valid)}/16")
        if valid:
            min_d = min(d for _, d in valid)
            self.lbl_min_d.setText(f"Min: {min_d} mm")
        else:
            self.lbl_min_d.setText("Min: VOID")

        self.heatmap_4x4.update_distances(dists)

        hist_matrix = frame.get("histograms", [])
        if not hist_matrix or len(hist_matrix) < 16:
            return

        # Selected zone pulse for visualizer
        sel_z = self.surface_selected_zone
        pulse_sel = hist_matrix[sel_z] if (0 <= sel_z < len(hist_matrix)) else hist_matrix[5]
        metrics_sel = extract_surface_pulse_metrics(pulse_sel)
        self.pulse_widget.update_pulse(pulse_sel, metrics_sel, zone_idx=sel_z)

        # Update telemetry labels
        self.lbl_tele_fwhm.setText(f"FWHM: {metrics_sel['fwhm_mm']:.1f} mm")
        self.lbl_tele_peak.setText(f"Peak: {metrics_sel['peak_val']:.1f} kcps/SPAD")
        self.lbl_tele_dist.setText(f"Distance: {metrics_sel['dist_mm']:.0f} mm")
        self.lbl_tele_tail.setText(f"Decay Tail Ratio: {metrics_sel['tail_ratio']:.1f}%")

        # Recording Mode: Extract 16-bin peak-centered canonical windows
        if self.is_surface_recording:
            # Capture all valid zones or central zones
            for z in range(16):
                z_pulse = hist_matrix[z]
                canonical_win, m = extract_canonical_window(z_pulse)
                if not m["is_void"] or self.surface_target_class == "VOID":
                    self.surface_dataset[self.surface_target_class].append(canonical_win)

            self.surface_record_count += 1
            if self.surface_record_count >= self.surface_record_target_total:
                self.is_surface_recording = False
                total_samples = len(self.surface_dataset[self.surface_target_class])
                self.surface_sample_labels[self.surface_target_class].setText(f"{total_samples} / 50")
                self.surface_sample_bars[self.surface_target_class].setValue(min(50, total_samples))
                self.lbl_surface_rec_status.setText(f"Captured [{self.surface_target_class}] Total Canonical Pulses: {total_samples}")
                self.lbl_surface_rec_status.setStyleSheet("background-color: #ecfdf5; color: #0284c7; border: 1px solid #7dd3fc; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 13px;")

        # Live Inference: Multi-Zone Ensemble + Hysteresis
        if self.surface_trained_model is not None:
            self.run_live_surface_inference(hist_matrix, dists, metrics_sel)

    # ------------------------------------------------------------------
    # Hand Posture Mode Frame Handler (8x8 Dual-Channel Preprocessing & Inference)
    # ------------------------------------------------------------------
    def handle_posture_frame(self, frame, dists):
        signals = frame.get("s", None)
        tensor_2x8x8, min_dist, valid_frame, active_count, bbox = preprocess_posture_frame(dists, signals)

        self.lbl_active_z.setText(f"Active: {active_count}/64")
        if valid_frame:
            self.lbl_min_d.setText(f"Min: {min_dist:.0f} mm")
        elif min_dist < POSTURE_MIN_DIST:
            self.lbl_min_d.setText(f"Min: {min_dist:.0f} mm (<100)")
        elif min_dist > POSTURE_MAX_DIST:
            self.lbl_min_d.setText(f"Min: {min_dist:.0f} mm (>400)")
        else:
            self.lbl_min_d.setText("Min: --- mm")

        self.heatmap_8x8.update_distances(dists)

        # Recording Mode
        if self.is_posture_recording:
            if valid_frame or self.posture_target_class == "NONE":
                self.posture_dataset[self.posture_target_class].append(tensor_2x8x8)
                self.posture_record_count += 1
                if self.posture_record_count >= self.posture_record_target_total:
                    self.is_posture_recording = False
                    total_samples = len(self.posture_dataset[self.posture_target_class])
                    if self.posture_target_class in self.posture_sample_labels:
                        self.posture_sample_labels[self.posture_target_class].setText(f"{total_samples} / 30")
                        self.posture_sample_bars[self.posture_target_class].setValue(min(30, total_samples))
                    self.lbl_posture_rec_status.setText(f"Captured [{self.posture_target_class}] Total Samples: {total_samples}")
                    self.lbl_posture_rec_status.setStyleSheet("background-color: #f5f3ff; color: #7c3aed; border: 1px solid #c4b5fd; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 13px;")
            else:
                self.lbl_posture_rec_status.setText(f"Hold hand steady in 100-400 mm range! (Current: {min_dist:.0f} mm)")
                self.lbl_posture_rec_status.setStyleSheet("background-color: #fef2f2; color: #dc2626; border: 1px solid #fca5a5; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 13px;")

        # Live Inference
        self.run_live_posture_inference(tensor_2x8x8, min_dist, valid_frame, active_count, bbox)

    def start_posture_capture(self, class_name, burst=False):
        self.posture_target_class = class_name
        self.posture_record_count = 0
        self.posture_record_target_total = 15 if burst else 1
        self.is_posture_recording = True
        desc = "15 Frames (Burst)" if burst else "1 Frame (Snap)"
        self.lbl_posture_rec_status.setText(f"RECORDING {desc} for [{class_name}]... Hold posture!")
        self.lbl_posture_rec_status.setStyleSheet("background-color: #f5f3ff; color: #6d28d9; border: 1px solid #c4b5fd; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 13px;")

    # ------------------------------------------------------------------
    # Data Recording Starters
    # ------------------------------------------------------------------
    def start_gesture_capture(self, class_name):
        self.gesture_target_class = class_name
        self.gesture_record_buffer = []
        self.is_gesture_recording = True
        self.lbl_gesture_rec_status.setText(f"RECORDING 15 FRAMES for [{class_name}]... Wave hand now!")
        self.lbl_gesture_rec_status.setStyleSheet("background-color: #fef2f2; color: #b91c1c; border: 1px solid #f87171; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 13px;")

    def start_surface_capture(self, class_name, burst=False):
        self.surface_target_class = class_name
        self.surface_record_count = 0
        self.surface_record_target_total = 10 if burst else 1
        self.is_surface_recording = True
        frames_desc = "10 Frames (Multi-Zone Canonical Pulses)" if burst else "1 Frame (16 Canonical Pulses)"
        self.lbl_surface_rec_status.setText(f"RECORDING {frames_desc} for [{class_name}]...")
        self.lbl_surface_rec_status.setStyleSheet("background-color: #eff6ff; color: #1d4ed8; border: 1px solid #93c5fd; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 13px;")

    # ------------------------------------------------------------------
    # Gesture Model Training (PyTorch Spatio-Temporal 2D-CNN)
    # ------------------------------------------------------------------
    def train_gesture_model(self):
        try:
            if not TORCH_AVAILABLE:
                QMessageBox.critical(self, "PyTorch Missing", "PyTorch is required for Gesture 2D-CNN training.")
                return

            insufficient = [cls for cls in GESTURE_CLASSES if len(self.gesture_dataset[cls]) < 2]
            if insufficient:
                msg = f"Every gesture class requires at least 2 recorded samples.\nInsufficient data for: {', '.join(insufficient)}"
                self.txt_gesture_train_log.append(f"Notice: {msg}")
                QMessageBox.warning(self, "Insufficient Samples", msg)
                return

            total_samples = sum(len(v) for v in self.gesture_dataset.values())
            min_samples = min(len(v) for v in self.gesture_dataset.values())
            self.txt_gesture_train_log.clear()
            self.txt_gesture_train_log.append("=== EdgeSense Gesture Spatio-Temporal 2D-CNN Training (PyTorch) ===")
            self.txt_gesture_train_log.append(f"Raw Samples Recorded: {total_samples}")
            self.txt_gesture_train_log.append(f"Balancing dataset to {min_samples} samples per class to prevent bias...")

            X_list = []
            y_list = []

            for class_idx, class_name in enumerate(GESTURE_CLASSES):
                samples = list(self.gesture_dataset[class_name])[:min_samples]
                self.txt_gesture_train_log.append(f"  - {class_name:12s}: {len(samples)} base samples")
                for s in samples:
                    augmented_variants = augment_gesture_sample(s)
                    for var in augmented_variants:
                        tensor_frame = normalize_gesture_window(var)  # (15, 8, 8)
                        X_list.append(tensor_frame)
                        y_list.append(class_idx)

                if class_name == "IDLE":
                    # Synthesize stationary hands across all spatial regions into IDLE
                    # This prevents the CNN from learning static hand positions as gestures
                    for _ in range(25):
                        # Static hand at top (rows 0..3)
                        w_top = np.full((15, 8, 8), -1.0, dtype=np.float32)
                        d = np.random.uniform(100.0, 350.0)
                        w_top[:, 0:np.random.randint(2, 5), np.random.randint(0, 3):np.random.randint(5, 9)] = d + np.random.normal(0, 4, (15, 1, 1))
                        X_list.append(normalize_gesture_window(w_top.reshape(15, 64)))
                        y_list.append(0)

                        # Static hand at bottom (rows 4..7)
                        w_bot = np.full((15, 8, 8), -1.0, dtype=np.float32)
                        d = np.random.uniform(100.0, 350.0)
                        w_bot[:, np.random.randint(4, 7):8, np.random.randint(0, 3):np.random.randint(5, 9)] = d + np.random.normal(0, 4, (15, 1, 1))
                        X_list.append(normalize_gesture_window(w_bot.reshape(15, 64)))
                        y_list.append(0)

                        # Static hand at center (rows 2..5, cols 2..5)
                        w_cen = np.full((15, 8, 8), -1.0, dtype=np.float32)
                        d = np.random.uniform(100.0, 350.0)
                        w_cen[:, 2:6, 2:6] = d + np.random.normal(0, 4, (15, 1, 1))
                        X_list.append(normalize_gesture_window(w_cen.reshape(15, 64)))
                        y_list.append(0)

            X = np.array(X_list, dtype=np.float32)  # (N, 15, 8, 8)
            y = np.array(y_list, dtype=np.int64)
            self.txt_gesture_train_log.append(f"\nTotal Augmented Dataset: {X.shape[0]} samples (Shape: {X.shape})")

            # Stratified train/val split (80/20)
            np.random.seed(42)
            indices = np.arange(len(y))
            np.random.shuffle(indices)
            split_idx = int(0.8 * len(indices))
            train_idx, val_idx = indices[:split_idx], indices[split_idx:]

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            self.txt_gesture_train_log.append("Training Spatio-Temporal 2D-CNN (Conv2D(15->16, k=3) -> MaxPool -> Conv2D(16->32, k=3) -> AdaptivePool -> Dense(128->32) -> Dense(32->5))...")
            QApplication.processEvents()

            torch.manual_seed(42)
            model = Gesture2DCNN(num_classes=len(GESTURE_CLASSES))
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)

            dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
            loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

            for epoch in range(35):
                model.train()
                epoch_loss = 0.0
                for bx, by in loader:
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                if (epoch + 1) % 5 == 0 or epoch == 34:
                    self.txt_gesture_train_log.append(f"  Epoch [{epoch+1:2d}/35] - Loss: {epoch_loss/len(loader):.4f}")
                    QApplication.processEvents()

            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(torch.tensor(X_val))
                val_preds = torch.argmax(val_out, dim=1).numpy()
                acc = np.mean(val_preds == y_val) * 100.0

            self.gesture_trained_model = model
            self.lbl_model_badge.setText(f"Gesture 2D-CNN: Trained ({acc:.1f}% Acc)")
            self.lbl_model_badge.setStyleSheet("background-color: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; padding: 3px 10px; border-radius: 4px; font-weight: bold; font-size: 11px;")

            self.txt_gesture_train_log.append(f"\nTraining Complete. Validation Accuracy: {acc:.1f}%\n")
            self.txt_gesture_train_log.append("Confusion Matrix:")
            cm = confusion_matrix(y_val, val_preds)
            self.txt_gesture_train_log.append(str(cm))

            # Export ONNX for ST Edge AI Core v10.2.1
            dummy = torch.randn(1, 15, 8, 8)
            torch.onnx.export(
                model,
                dummy,
                GESTURE_ONNX_FILE,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamo=False
            )
            self.txt_gesture_train_log.append(f"\nExported ST Edge AI ONNX: {GESTURE_ONNX_FILE}")
            self.tabs_gesture.setCurrentIndex(2)

        except Exception as e:
            err_msg = f"Gesture training error: {e}"
            self.txt_gesture_train_log.append(f"\n[ERROR] {err_msg}")
            QMessageBox.critical(self, "Training Error", err_msg)

    # ------------------------------------------------------------------
    # Distance-Invariant Surface Model Training (PyTorch 1D-CNN)
    # ------------------------------------------------------------------
    def train_surface_model(self):
        try:
            insufficient = [cls for cls in SURFACE_CLASSES if len(self.surface_dataset[cls]) < 2]
            if insufficient:
                msg = f"Every surface class requires at least 2 recorded samples.\nInsufficient data for: {', '.join(insufficient)}"
                self.txt_surface_train_log.append(f"Notice: {msg}")
                QMessageBox.warning(self, "Insufficient Samples", msg)
                return

            total_samples = sum(len(v) for v in self.surface_dataset.values())
            self.txt_surface_train_log.clear()
            self.txt_surface_train_log.append("=== EdgeSense Distance-Invariant Surface 1D-CNN Training ===")
            self.txt_surface_train_log.append(f"Raw Canonical Pulses Recorded: {total_samples}")

            # Balance dataset across classes to prevent dominance bias
            counts = [len(self.surface_dataset[cls]) for cls in SURFACE_CLASSES]
            min_samples = min(counts)
            self.txt_surface_train_log.append(f"Balancing dataset to {min_samples} samples per class to prevent bias...")

            X_list = []
            y_list = []

            for class_idx, class_name in enumerate(SURFACE_CLASSES):
                pulses = self.surface_dataset[class_name][:min_samples]
                self.txt_surface_train_log.append(f"  - {class_name:12s}: {len(pulses)} canonical pulses")
                for p in pulses:
                    augmented_variants = augment_surface_canonical(p)
                    for var in augmented_variants:
                        X_list.append(var)
                        y_list.append(class_idx)

            X = np.array(X_list, dtype=np.float32).reshape(-1, 1, SURFACE_CANONICAL_BINS)
            y = np.array(y_list, dtype=np.int64)

            self.txt_surface_train_log.append(f"\nTotal Augmented Dataset: {X.shape[0]} samples (Shape: {X.shape})")

            indices = np.arange(len(y))
            np.random.seed(42)
            np.random.shuffle(indices)
            split_idx = int(0.8 * len(y))
            train_idx, val_idx = indices[:split_idx], indices[split_idx:]

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            if TORCH_AVAILABLE:
                self.txt_surface_train_log.append("Training Distance-Invariant 1D-CNN (Conv1D 8 -> Conv1D 16 -> Dense 16 -> Softmax 4)...")
                QApplication.processEvents()

                model = Surface1DCNN(num_classes=4)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                criterion = nn.CrossEntropyLoss()

                dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
                loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

                model.train()
                for epoch in range(40):
                    epoch_loss = 0.0
                    for bx, by in loader:
                        optimizer.zero_grad()
                        out = model(bx)
                        loss = criterion(out, by)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    if (epoch + 1) % 10 == 0:
                        self.txt_surface_train_log.append(f"  Epoch [{epoch+1:2d}/40] - Loss: {epoch_loss/len(loader):.4f}")
                        QApplication.processEvents()

                model.eval()
                with torch.no_grad():
                    val_out = model(torch.tensor(X_val))
                    val_preds = torch.argmax(val_out, dim=1).numpy()
                    acc = np.mean(val_preds == y_val) * 100.0

                self.surface_trained_model = model
                self.lbl_model_badge.setText(f"Surface 1D-CNN: Trained ({acc:.1f}% Acc)")
                self.lbl_model_badge.setStyleSheet("background-color: #ecfdf5; color: #0284c7; border: 1px solid #bae6fd; padding: 3px 10px; border-radius: 4px; font-weight: bold; font-size: 11px;")

                self.txt_surface_train_log.append(f"\nTraining Complete. Validation Accuracy: {acc:.1f}%\n")
                self.txt_surface_train_log.append("Confusion Matrix:")
                cm = confusion_matrix(y_val, val_preds)
                self.txt_surface_train_log.append(str(cm))

                # Export 16-bin ONNX for ST Edge AI
                dummy = torch.randn(1, 1, SURFACE_CANONICAL_BINS)
                torch.onnx.export(
                    model,
                    dummy,
                    SURFACE_ONNX_FILE,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input_canonical'],
                    output_names=['output_logits'],
                    dynamo=False
                )
                self.txt_surface_train_log.append(f"\nExported ST Edge AI ONNX: {SURFACE_ONNX_FILE}")

            self.lbl_surface_inference_result.setText(f"Model Active (Validation Acc: {acc:.1f}%) - Distance-Invariant Scanning Active!")
            self.lbl_surface_inference_result.setStyleSheet("color: #0284c7; font-weight: bold;")
            self.tabs_surface.setCurrentIndex(2)

        except Exception as e:
            err_msg = f"Surface training error: {e}"
            self.txt_surface_train_log.append(f"\n[ERROR] {err_msg}")
            QMessageBox.critical(self, "Training Error", err_msg)

    # ------------------------------------------------------------------
    # Live Gesture Inference
    # ------------------------------------------------------------------
    def run_live_gesture_inference(self):
        try:
            if self.gesture_trained_model is None or len(self.gesture_sliding_window) < GESTURE_WINDOW_SIZE:
                return

            tensor_input = normalize_gesture_window(self.gesture_sliding_window)
            bx = torch.tensor(tensor_input, dtype=torch.float32).unsqueeze(0)

            self.gesture_trained_model.eval()
            with torch.no_grad():
                logits = self.gesture_trained_model(bx)
                probs = F.softmax(logits, dim=1).numpy()[0]

            if len(probs) != len(GESTURE_CLASSES):
                return

            # Temporal Motion Energy Guard:
            # Dynamic gestures require continuous motion across the 15-frame window.
            # When hand is stationary (e.g. at the end of a downward swipe),
            # motion energy drops below 16.0, immediately returning output to IDLE.
            diffs = np.abs(tensor_input[1:] - tensor_input[:-1])
            motion_energy = float(np.sum(diffs))

            if motion_energy < 16.0:
                probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            now = time.time()
            for idx, cls_name in enumerate(GESTURE_CLASSES):
                p = probs[idx] * 100.0
                self.gesture_prob_bars[cls_name].setValue(int(p))
                self.gesture_prob_labels[cls_name].setText(f"{p:5.1f}%")

            if now < self.gesture_cooldown_until:
                return

            best_idx = int(np.argmax(probs))
            best_p = probs[best_idx]

            if best_idx != 0 and best_p >= 0.80:
                gesture = GESTURE_CLASSES[best_idx]
                self.gesture_cooldown_until = now + 0.85
                self.lbl_gesture_inference_result.setText(f"GESTURE DETECTED: {gesture} (Confidence: {best_p*100:.1f}%)")
                self.lbl_gesture_inference_result.setStyleSheet(f"color: {GESTURE_COLORS[gesture]}; font-size: 16px; font-weight: bold;")
                self.gesture_banner_frame.setStyleSheet(f"background-color: #eff6ff; border: 2px solid {GESTURE_COLORS[gesture]}; border-radius: 8px; padding: 12px;")

                self.gesture_history_table.insertRow(0)
                time_str = datetime.now().strftime("%H:%M:%S.%f")[:11]
                self.gesture_history_table.setItem(0, 0, QTableWidgetItem(time_str))
                item_gest = QTableWidgetItem(f"{gesture}")
                item_gest.setForeground(QColor(GESTURE_COLORS[gesture]))
                self.gesture_history_table.setItem(0, 1, item_gest)
                self.gesture_history_table.setItem(0, 2, QTableWidgetItem(f"{best_p*100:.1f}%"))
                self.gesture_history_table.setItem(0, 3, QTableWidgetItem("CONFIRMED"))

                if self.gesture_history_table.rowCount() > 20:
                    self.gesture_history_table.removeRow(20)

            elif best_idx == 0:
                self.gesture_banner_frame.setStyleSheet("background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px;")
                self.lbl_gesture_inference_result.setText("IDLE (Monitoring 8x8 Spatial Depth)")
                self.lbl_gesture_inference_result.setStyleSheet("color: #475569;")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Live Surface Inference (Multi-Zone Ensemble + Hysteresis Debouncing)
    # ------------------------------------------------------------------
    def run_live_surface_inference(self, hist_matrix, dists, metrics_sel):
        try:
            if self.surface_trained_model is None or not TORCH_AVAILABLE:
                return

            # Multi-Zone Ensemble across all 16 zones
            zone_probs = []
            for z in range(16):
                z_pulse = hist_matrix[z]
                canon_win, m = extract_canonical_window(z_pulse)
                if m["is_void"] or dists[z] <= 0:
                    zone_probs.append([0.01, 0.01, 0.01, 0.97])
                else:
                    tensor_x = torch.tensor(canon_win.reshape(1, 1, SURFACE_CANONICAL_BINS), dtype=torch.float32)
                    with torch.no_grad():
                        logits = self.surface_trained_model(tensor_x)
                        p = F.softmax(logits, dim=1).numpy()[0]
                        zone_probs.append(p)

            # Spatial Consensus (Average probabilities across all zones)
            ensemble_probs = np.mean(zone_probs, axis=0)

            # Temporal Exponential Moving Average (EMA) to kill flicker
            alpha = 0.30
            self.surface_smoothed_probs = (1.0 - alpha) * self.surface_smoothed_probs + alpha * ensemble_probs

            for idx, cls_name in enumerate(SURFACE_CLASSES):
                p = self.surface_smoothed_probs[idx] * 100.0
                self.surface_prob_bars[cls_name].setValue(int(p))
                self.surface_prob_labels[cls_name].setText(f"{p:5.1f}%")

            best_idx = int(np.argmax(self.surface_smoothed_probs))
            best_p = self.surface_smoothed_probs[best_idx]
            candidate_cls = SURFACE_CLASSES[best_idx]

            # 3-Frame State Transition Debounce
            if candidate_cls != self.surface_confirmed_class:
                if candidate_cls == self.surface_candidate_class and best_p >= 0.65:
                    self.surface_candidate_count += 1
                    if self.surface_candidate_count >= 3:
                        self.surface_confirmed_class = candidate_cls
                        self.surface_candidate_count = 0
                else:
                    self.surface_candidate_class = candidate_cls
                    self.surface_candidate_count = 1
            else:
                self.surface_candidate_count = 0

            active_cls = self.surface_confirmed_class
            col = SURFACE_COLORS[active_cls]

            if active_cls == "CARPET":
                banner_text = f"SURFACE: CARPET (Cloth / Fabric / Leather) - Confidence: {best_p*100:.1f}%"
            elif active_cls == "HARD_FLOOR":
                banner_text = f"SURFACE: HARD FLOOR (Tile / Wood / Marble) - Confidence: {best_p*100:.1f}%"
            elif active_cls == "SPECULAR":
                banner_text = f"SURFACE: SPECULAR REFLECTION (Metal / Mirror) - Confidence: {best_p*100:.1f}%"
            else:
                banner_text = f"SURFACE: VOID / DROP-OFF HAZARD (Cliff Emergency Stop) - Confidence: {best_p*100:.1f}%"

            self.lbl_surface_inference_result.setText(banner_text)
            self.lbl_surface_inference_result.setStyleSheet(f"color: {col}; font-size: 15px; font-weight: bold;")
            self.surface_banner_frame.setStyleSheet(f"background-color: #ffffff; border: 2px solid {col}; border-radius: 8px; padding: 12px;")

            if self.total_frames % 25 == 0:
                self.surface_history_table.insertRow(0)
                time_str = datetime.now().strftime("%H:%M:%S.%f")[:11]
                self.surface_history_table.setItem(0, 0, QTableWidgetItem(time_str))
                item_cls = QTableWidgetItem(f"{active_cls}")
                item_cls.setForeground(QColor(col))
                self.surface_history_table.setItem(0, 1, item_cls)
                self.surface_history_table.setItem(0, 2, QTableWidgetItem(f"{best_p*100:.1f}%"))
                self.surface_history_table.setItem(0, 3, QTableWidgetItem(f"{metrics_sel['fwhm_mm']:.1f} mm"))
                if self.surface_history_table.rowCount() > 20:
                    self.surface_history_table.removeRow(20)

        except Exception:
            pass

    # ------------------------------------------------------------------
    # Live Hand Posture Inference & Debounce HUD
    # ------------------------------------------------------------------
    def run_live_posture_inference(self, tensor_2x8x8, min_dist, valid_frame, active_count, bbox):
        try:
            if not valid_frame:
                raw_probs = np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float32)
            elif self.posture_trained_model is not None and TORCH_AVAILABLE:
                with torch.no_grad():
                    bx = torch.tensor(tensor_2x8x8.reshape(1, 2, 8, 8), dtype=torch.float32)
                    logits = self.posture_trained_model(bx)
                    raw_probs = F.softmax(logits, dim=1).numpy()[0]
            elif os.path.exists(POSTURE_ONNX_FILE):
                if not hasattr(self, 'posture_onnx_session') or self.posture_onnx_session is None:
                    import onnxruntime as ort
                    self.posture_onnx_session = ort.InferenceSession(POSTURE_ONNX_FILE, providers=['CPUExecutionProvider'])
                input_name = self.posture_onnx_session.get_inputs()[0].name
                inp = tensor_2x8x8.reshape(1, 2, 8, 8).astype(np.float32)
                logits = self.posture_onnx_session.run(None, {input_name: inp})[0][0]
                exp_l = np.exp(logits - np.max(logits))
                raw_probs = exp_l / np.sum(exp_l)
            else:
                raw_probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            alpha = 0.35
            self.posture_smoothed_probs = (1.0 - alpha) * self.posture_smoothed_probs + alpha * raw_probs

            top_idx = int(np.argmax(self.posture_smoothed_probs))
            top_p = float(self.posture_smoothed_probs[top_idx])
            top_cls = POSTURE_CLASSES[top_idx]

            # 3-Frame Debounce Latch
            is_latched = False
            if valid_frame and top_cls != "NONE" and top_p >= 0.60:
                if top_cls == self.posture_candidate_class:
                    self.posture_candidate_count += 1
                    if self.posture_candidate_count >= 3:
                        self.posture_latched_class = top_cls
                        is_latched = True
                else:
                    self.posture_candidate_class = top_cls
                    self.posture_candidate_count = 1
            else:
                self.posture_candidate_count = 0
                if not valid_frame or top_cls == "NONE":
                    self.posture_latched_class = "NONE"

            # Update visualizer HUD
            self.posture_hud.update_hud(
                self.posture_latched_class,
                self.posture_smoothed_probs,
                min_dist,
                valid_frame,
                active_count,
                bbox,
                self.posture_candidate_count,
                is_latched or (self.posture_latched_class != "NONE")
            )

            # Update Inference Tab Bars
            for idx, cls_name in enumerate(POSTURE_CLASSES):
                p = self.posture_smoothed_probs[idx] * 100.0
                if cls_name in self.posture_infer_bars:
                    self.posture_infer_bars[cls_name].setValue(int(p))
                    self.posture_infer_labels[cls_name].setText(f"{p:5.1f}%")

            # Update Inference Tab Banner
            col = POSTURE_COLORS.get(self.posture_latched_class, "#475569")
            icon = POSTURE_ICONS.get(self.posture_latched_class, "🖐️")
            if self.posture_latched_class != "NONE":
                self.lbl_posture_inference_result.setText(f"POSTURE DETECTED: {icon} {self.posture_latched_class} (Confidence: {top_p*100:.1f}%)")
                self.lbl_posture_inference_result.setStyleSheet(f"color: {col}; font-size: 16px; font-weight: bold;")
                self.posture_banner_frame.setStyleSheet(f"background-color: #ffffff; border: 2px solid {col}; border-radius: 8px; padding: 12px;")
            else:
                self.lbl_posture_inference_result.setText("IDLE (Place hand in 10-40 cm range)")
                self.lbl_posture_inference_result.setStyleSheet("color: #64748b; font-size: 14px; font-weight: bold;")
                self.posture_banner_frame.setStyleSheet("background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px;")

            if is_latched and self.posture_candidate_count == 3:
                self.posture_history_table.insertRow(0)
                time_str = datetime.now().strftime("%H:%M:%S.%f")[:11]
                self.posture_history_table.setItem(0, 0, QTableWidgetItem(time_str))
                item_cls = QTableWidgetItem(f"{icon} {self.posture_latched_class}")
                item_cls.setForeground(QColor(col))
                self.posture_history_table.setItem(0, 1, item_cls)
                self.posture_history_table.setItem(0, 2, QTableWidgetItem(f"{top_p*100:.1f}%"))
                self.posture_history_table.setItem(0, 3, QTableWidgetItem(f"{min_dist:.0f} mm"))
                if self.posture_history_table.rowCount() > 20:
                    self.posture_history_table.removeRow(20)

        except Exception:
            pass

    # ------------------------------------------------------------------
    # Hand Posture Model Training (PyTorch 2D-CNN)
    # ------------------------------------------------------------------
    def train_posture_model(self):
        try:
            insufficient = [cls for cls in POSTURE_CLASSES if len(self.posture_dataset[cls]) < 2]
            if insufficient:
                msg = f"Every posture class requires at least 2 recorded samples.\nInsufficient data for: {', '.join(insufficient)}"
                self.txt_posture_train_log.append(f"Notice: {msg}")
                QMessageBox.warning(self, "Insufficient Samples", msg)
                return

            total_samples = sum(len(v) for v in self.posture_dataset.values())
            self.txt_posture_train_log.clear()
            self.txt_posture_train_log.append("=== EdgeSense Hand Posture 2D-CNN Training ===")
            self.txt_posture_train_log.append(f"Raw Samples Recorded: {total_samples}")

            counts = [len(self.posture_dataset[cls]) for cls in POSTURE_CLASSES]
            min_samples = min(counts)
            self.txt_posture_train_log.append(f"Balancing dataset to {min_samples} samples per class to prevent bias...")

            X_list = []
            y_list = []

            for class_idx, class_name in enumerate(POSTURE_CLASSES):
                samples = self.posture_dataset[class_name][:min_samples]
                self.txt_posture_train_log.append(f"  - {class_name:12s}: {len(samples)} base frames")
                for s in samples:
                    augmented_variants = augment_posture_sample(s)
                    for var in augmented_variants:
                        X_list.append(var)
                        y_list.append(class_idx)

            X = np.array(X_list, dtype=np.float32) # (N, 2, 8, 8)
            y = np.array(y_list, dtype=np.int64)

            self.txt_posture_train_log.append(f"\nTotal Augmented Dataset: {X.shape[0]} samples (Shape: {X.shape})")

            indices = np.arange(len(y))
            np.random.seed(42)
            np.random.shuffle(indices)
            split_idx = int(0.8 * len(y))
            train_idx, val_idx = indices[:split_idx], indices[split_idx:]

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            if TORCH_AVAILABLE:
                self.txt_posture_train_log.append("Training Posture 2D-CNN (Conv2D(8, 3x3) -> MaxPool(2x2) -> Dense(32) -> Dense(6))...")
                QApplication.processEvents()

                model = Posture2DCNN(num_classes=6)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
                criterion = nn.CrossEntropyLoss()

                dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
                loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

                model.train()
                for epoch in range(35):
                    epoch_loss = 0.0
                    for bx, by in loader:
                        optimizer.zero_grad()
                        out = model(bx)
                        loss = criterion(out, by)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    if (epoch + 1) % 5 == 0 or epoch == 0:
                        self.txt_posture_train_log.append(f"  Epoch [{epoch+1:2d}/35] - Loss: {epoch_loss/len(loader):.4f}")
                        QApplication.processEvents()

                model.eval()
                with torch.no_grad():
                    val_out = model(torch.tensor(X_val))
                    val_preds = torch.argmax(val_out, dim=1).numpy()
                    acc = np.mean(val_preds == y_val) * 100.0

                self.posture_trained_model = model
                self.lbl_model_badge.setText(f"Posture 2D-CNN: Trained ({acc:.1f}% Acc)")
                self.lbl_model_badge.setStyleSheet("background-color: #f5f3ff; color: #7c3aed; border: 1px solid #ddd6fe; padding: 3px 10px; border-radius: 4px; font-weight: bold; font-size: 11px;")

                self.txt_posture_train_log.append(f"\nTraining Complete. Validation Accuracy: {acc:.1f}%\n")
                self.txt_posture_train_log.append("Confusion Matrix:")
                cm = confusion_matrix(y_val, val_preds)
                self.txt_posture_train_log.append(str(cm))

                dummy = torch.randn(1, 2, 8, 8)
                torch.onnx.export(
                    model,
                    dummy,
                    POSTURE_ONNX_FILE,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input_posture'],
                    output_names=['output_logits'],
                    dynamo=False
                )
                self.txt_posture_train_log.append(f"\nExported ST Edge AI ONNX: {POSTURE_ONNX_FILE}")

                if hasattr(self, 'posture_onnx_session'):
                    self.posture_onnx_session = None

            self.lbl_posture_inference_result.setText(f"Model Active (Validation Acc: {acc:.1f}%) - Posture Recognition Active!")
            self.lbl_posture_inference_result.setStyleSheet("color: #7c3aed; font-weight: bold;")
            self.tabs_posture.setCurrentIndex(2)

        except Exception as e:
            err_msg = f"Posture training error: {e}"
            self.txt_posture_train_log.append(f"\n[ERROR] {err_msg}")
            QMessageBox.critical(self, "Training Error", err_msg)

    # ------------------------------------------------------------------
    # Model Deployment
    # ------------------------------------------------------------------
    def deploy_gesture_model(self):
        if not os.path.exists(GESTURE_ONNX_FILE):
            QMessageBox.warning(self, "No Model Found", "Please train a gesture neural network model first before deploying.")
            return

        self.btn_deploy_gesture.setEnabled(False)
        self.btn_deploy_gesture.setText("Deploying Gesture Model...")
        self.txt_gesture_train_log.append("\n" + "="*50)
        self.txt_gesture_train_log.append("Starting STM32Cube.AI Gesture Model Deployment Pipeline...")
        self.txt_gesture_train_log.append("="*50)

        self.deploy_worker = DeployWorker(mode="gesture")
        self.deploy_worker.sig_log.connect(lambda t: self.txt_gesture_train_log.append(t))
        self.deploy_worker.sig_done.connect(self.on_gesture_deploy_done)
        self.deploy_worker.start()

    def on_gesture_deploy_done(self, success, message):
        self.btn_deploy_gesture.setEnabled(True)
        self.btn_deploy_gesture.setText("Deploy Gesture Model to MCU")
        if success:
            self.txt_gesture_train_log.append(f"\n[SUCCESS] {message}")
            QMessageBox.information(self, "Deployment Successful", f"{message}\n\nBinary: Debug/EdgeSense.elf")
        else:
            self.txt_gesture_train_log.append(f"\n[ERROR] {message}")
            QMessageBox.critical(self, "Deployment Failed", message)

    def deploy_surface_model(self):
        if not os.path.exists(SURFACE_ONNX_FILE):
            QMessageBox.warning(self, "No Model Found", "Please train a surface 1D-CNN model first before deploying.")
            return

        self.btn_deploy_surface.setEnabled(False)
        self.btn_deploy_surface.setText("Deploying Surface 1D-CNN...")
        self.txt_surface_train_log.append("\n" + "="*50)
        self.txt_surface_train_log.append("Starting ST Edge AI Surface 1D-CNN Model Deployment Pipeline...")
        self.txt_surface_train_log.append("="*50)

        self.deploy_worker = DeployWorker(mode="surface")
        self.deploy_worker.sig_log.connect(lambda t: self.txt_surface_train_log.append(t))
        self.deploy_worker.sig_done.connect(self.on_surface_deploy_done)
        self.deploy_worker.start()

    def on_surface_deploy_done(self, success, message):
        self.btn_deploy_surface.setEnabled(True)
        self.btn_deploy_surface.setText("Deploy Surface Model to MCU")
        if success:
            self.txt_surface_train_log.append(f"\n[SUCCESS] {message}")
            QMessageBox.information(self, "Deployment Successful", f"{message}\n\nBinary: Debug/EdgeSense.elf")
        else:
            self.txt_surface_train_log.append(f"\n[ERROR] {message}")
            QMessageBox.critical(self, "Deployment Failed", message)

    def deploy_posture_model(self):
        if not os.path.exists(POSTURE_ONNX_FILE):
            QMessageBox.warning(self, "No Model Found", "Please train a posture 2D-CNN model first before deploying.")
            return

        self.btn_deploy_posture.setEnabled(False)
        self.btn_deploy_posture.setText("Deploying Posture 2D-CNN...")
        self.txt_posture_train_log.append("\n" + "="*50)
        self.txt_posture_train_log.append("Starting ST Edge AI Posture 2D-CNN Model Deployment Pipeline...")
        self.txt_posture_train_log.append("="*50)

        self.deploy_worker = DeployWorker(mode="posture")
        self.deploy_worker.sig_log.connect(lambda t: self.txt_posture_train_log.append(t))
        self.deploy_worker.sig_done.connect(self.on_posture_deploy_done)
        self.deploy_worker.start()

    def on_posture_deploy_done(self, success, message):
        self.btn_deploy_posture.setEnabled(True)
        self.btn_deploy_posture.setText("Deploy Posture Model to MCU")
        if success:
            self.txt_posture_train_log.append(f"\n[SUCCESS] {message}")
            QMessageBox.information(self, "Deployment Successful", f"{message}\n\nBinary: Debug/EdgeSense.elf")
        else:
            self.txt_posture_train_log.append(f"\n[ERROR] {message}")
            QMessageBox.critical(self, "Deployment Failed", message)

    # ------------------------------------------------------------------
    # Dataset Storage
    # ------------------------------------------------------------------
    def save_gesture_dataset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Gesture Dataset", GESTURE_DATA_FILE, "NPZ Files (*.npz)")
        if not path:
            return
        save_dict = {cls: np.array(self.gesture_dataset[cls]) for cls in GESTURE_CLASSES if len(self.gesture_dataset[cls]) > 0}
        np.savez_compressed(path, **save_dict)
        self.lbl_gesture_rec_status.setText(f"Saved gesture dataset to: {os.path.basename(path)}")

    def load_gesture_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Gesture Dataset", DATA_DIR, "NPZ Files (*.npz)")
        if not path or not os.path.exists(path):
            return
        data = np.load(path)
        for cls in GESTURE_CLASSES:
            if cls in data:
                self.gesture_dataset[cls] = list(data[cls])
                cnt = len(self.gesture_dataset[cls])
                self.gesture_sample_labels[cls].setText(f"{cnt} / 15")
                self.gesture_sample_bars[cls].setValue(min(15, cnt))
        self.lbl_gesture_rec_status.setText(f"Loaded dataset from: {os.path.basename(path)}")

    def reset_gesture_recordings(self):
        for cls in GESTURE_CLASSES:
            self.gesture_dataset[cls].clear()
            self.gesture_sample_labels[cls].setText("0 / 15")
            self.gesture_sample_bars[cls].setValue(0)
        self.lbl_gesture_rec_status.setText("All gesture recordings cleared.")

    def save_surface_dataset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Surface Dataset", SURFACE_DATA_FILE, "NPZ Files (*.npz)")
        if not path:
            return
        save_dict = {cls: np.array(self.surface_dataset[cls]) for cls in SURFACE_CLASSES if len(self.surface_dataset[cls]) > 0}
        np.savez_compressed(path, **save_dict)
        self.lbl_surface_rec_status.setText(f"Saved surface dataset to: {os.path.basename(path)}")

    def load_surface_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Surface Dataset", DATA_DIR, "NPZ Files (*.npz)")
        if not path or not os.path.exists(path):
            return
        data = np.load(path)
        for cls in SURFACE_CLASSES:
            if cls in data:
                self.surface_dataset[cls] = list(data[cls])
                cnt = len(self.surface_dataset[cls])
                self.surface_sample_labels[cls].setText(f"{cnt} / 50")
                self.surface_sample_bars[cls].setValue(min(50, cnt))
        self.lbl_surface_rec_status.setText(f"Loaded surface dataset from: {os.path.basename(path)}")

    def reset_surface_recordings(self):
        for cls in SURFACE_CLASSES:
            self.surface_dataset[cls].clear()
            self.surface_sample_labels[cls].setText("0 / 50")
            self.surface_sample_bars[cls].setValue(0)
        self.lbl_surface_rec_status.setText("All surface recordings cleared.")

    def save_posture_dataset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Posture Dataset", POSTURE_DATA_FILE, "NPZ Files (*.npz)")
        if not path:
            return
        save_dict = {cls: np.array(self.posture_dataset[cls]) for cls in POSTURE_CLASSES if len(self.posture_dataset[cls]) > 0}
        np.savez_compressed(path, **save_dict)
        self.lbl_posture_rec_status.setText(f"Saved posture dataset to: {os.path.basename(path)}")

    def load_posture_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Posture Dataset", DATA_DIR, "NPZ Files (*.npz)")
        if not path or not os.path.exists(path):
            return
        data = np.load(path)
        for cls in POSTURE_CLASSES:
            if cls in data:
                self.posture_dataset[cls] = list(data[cls])
                cnt = len(self.posture_dataset[cls])
                if cls in self.posture_sample_labels:
                    self.posture_sample_labels[cls].setText(f"{cnt} / 30")
                    self.posture_sample_bars[cls].setValue(min(30, cnt))
        self.lbl_posture_rec_status.setText(f"Loaded posture dataset from: {os.path.basename(path)}")

    def reset_posture_recordings(self):
        for cls in POSTURE_CLASSES:
            self.posture_dataset[cls].clear()
            if cls in self.posture_sample_labels:
                self.posture_sample_labels[cls].setText("0 / 30")
                self.posture_sample_bars[cls].setValue(0)
        self.lbl_posture_rec_status.setText("All posture recordings cleared.")

    def load_default_datasets(self):
        if os.path.exists(GESTURE_DATA_FILE):
            try:
                data = np.load(GESTURE_DATA_FILE)
                for cls in GESTURE_CLASSES:
                    if cls in data:
                        self.gesture_dataset[cls] = list(data[cls])
                        cnt = len(self.gesture_dataset[cls])
                        self.gesture_sample_labels[cls].setText(f"{cnt} / 15")
                        self.gesture_sample_bars[cls].setValue(min(15, cnt))
            except Exception:
                pass

        if os.path.exists(SURFACE_DATA_FILE):
            try:
                data = np.load(SURFACE_DATA_FILE)
                for cls in SURFACE_CLASSES:
                    if cls in data:
                        self.surface_dataset[cls] = list(data[cls])
                        cnt = len(self.surface_dataset[cls])
                        self.surface_sample_labels[cls].setText(f"{cnt} / 50")
                        self.surface_sample_bars[cls].setValue(min(50, cnt))
            except Exception:
                pass

        if os.path.exists(POSTURE_DATA_FILE):
            try:
                data = np.load(POSTURE_DATA_FILE)
                for cls in POSTURE_CLASSES:
                    if cls in data:
                        self.posture_dataset[cls] = list(data[cls])
                        cnt = len(self.posture_dataset[cls])
                        if cls in self.posture_sample_labels:
                            self.posture_sample_labels[cls].setText(f"{cnt} / 30")
                            self.posture_sample_bars[cls].setValue(min(30, cnt))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Serial Port Management
    # ------------------------------------------------------------------
    def refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        preferred_index = 0
        for i, p in enumerate(ports):
            self.port_combo.addItem(f"{p.device} - {p.description}", p.device)
            if "STLink" in p.description or "VCP" in p.description or "COM9" in p.device:
                preferred_index = i
        if ports:
            self.port_combo.setCurrentIndex(preferred_index)

    def start_serial_connection(self):
        port = self.port_combo.currentData()
        if not port:
            return
        self.worker = SerialWorker(port=port)
        self.worker.sig_frame.connect(self.on_frame_received)
        self.worker.sig_status.connect(self.on_serial_status)
        self.worker.start()

    def toggle_connection(self):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.stop()
            self.btn_connect.setText("Connect")
            self.btn_connect.setStyleSheet("background-color: #1d4ed8; color: #ffffff; font-weight: bold; padding: 5px 14px; border: none; border-radius: 4px;")
            self.lbl_status.setText("Disconnected")
            self.lbl_status.setStyleSheet("background-color: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; border-radius: 4px; padding: 3px 10px; font-weight: bold; font-size: 11px;")
        else:
            self.start_serial_connection()
            self.btn_connect.setText("Disconnect")
            self.btn_connect.setStyleSheet("background-color: #475569; color: #ffffff; font-weight: bold; padding: 5px 14px; border: none; border-radius: 4px;")

    def on_serial_status(self, msg, ok):
        if ok:
            self.lbl_status.setText("Connected")
            self.lbl_status.setStyleSheet("background-color: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; border-radius: 4px; padding: 3px 10px; font-weight: bold; font-size: 11px;")
        else:
            self.lbl_status.setText("Failed")
            self.lbl_status.setStyleSheet("background-color: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; border-radius: 4px; padding: 3px 10px; font-weight: bold; font-size: 11px;")

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker:
            self.worker.stop()
        event.accept()


# ----------------------------------------------------------------------
# Application Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = NNStudioWindow()
    win.show()
    sys.exit(app.exec_())
