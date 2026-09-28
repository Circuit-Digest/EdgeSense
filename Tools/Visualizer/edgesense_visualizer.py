#!/usr/bin/env python3
"""
/*
 * edgesense_visualizer.py
 * Real-time 3D Time-of-Flight and Edge AI Showcase Visualizer for EdgeSense
 *
 * Copyright (c) 2026 Dharagesh and Circuit Digest
 * https://github.com/Circuit-Digest/EdgeSense
 * Licensed under GNU General Public License v3.0
 */

  ███████╗██████╗  ██████╗ ███████╗███████╗███████╗███╗   ██╗███████╗███████╗
  ██╔════╝██╔══██╗██╔════╝ ██╔════╝██╔════╝██╔════╝████╗  ██║██╔════╝██╔════╝
  █████╗  ██║  ██║██║  ███╗█████╗  ███████╗█████╗  ██╔██╗ ██║███████╗█████╗  
  ██╔══╝  ██║  ██║██║   ██║██╔══╝  ╚════██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝  
  ███████╗██████╔╝╚██████╔╝███████╗███████║███████╗██║ ╚████║███████║███████╗
  ╚══════╝╚═════╝  ╚═════╝ ╚══════╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝

Features:
  - Clean Light Theme Architecture (Professional White/Slate Palette, Zero Emojis)
  - Live On-Chip Edge AI Output Showcase (Gesture, Surface, Posture)
  - 3D Spatial Depth Matrix (8x8 and 4x4) with Centroid Target Tracking Crosshairs
  - Dynamic Touchless Gesture Interaction (Real-time swipe navigation showcase)
  - Dynamic Touchless Posture Interaction (Real-time smart HMI and media controller simulation)
  - Cumulative Normalized Histogram (CNH) 48-Bin Pulse Shape Analysis
  - Ambient Photon Flux and Solar Glint Heatmap (kcps)
  - Interactive AT Command Terminal and Hardware Mode Controller
  - Automated Firmware AT Pipeline (Auto-configures AT+STREAM=RAW and runtime mode switches)
"""

import sys
import os
import time
import json
import re
import math
from collections import deque
from datetime import datetime
import numpy as np

import serial
import serial.tools.list_ports

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QFrame,
    QStatusBar, QGroupBox, QStackedWidget, QMessageBox,
    QProgressBar, QButtonGroup, QTabWidget, QCheckBox,
    QSplitter, QTextEdit, QPlainTextEdit, QLineEdit, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF, QPointF, QRect
from PyQt5.QtGui import (
    QColor, QPainter, QBrush, QPen, QFont, QLinearGradient,
    QRadialGradient, QPolygonF, QPainterPath
)

# ----------------------------------------------------------------------
# Class Visual Configurations (Clean Professional Labels, No Emojis)
# ----------------------------------------------------------------------
GESTURE_INFO = {
    "SWIPE_LEFT": ("#1d4ed8", "SWIPE LEFT"),
    "SWIPE_RIGHT": ("#15803d", "SWIPE RIGHT"),
    "SWIPE_UP": ("#b45309", "SWIPE UP"),
    "SWIPE_DOWN": ("#b91c1c", "SWIPE DOWN"),
    "IDLE": ("#475569", "IDLE / READY")
}

SURFACE_INFO = {
    "HARD_FLOOR": ("#0284c7", "HARD FLOOR (TILE / WOOD)"),
    "CARPET": ("#b45309", "CARPET / FABRIC RUG"),
    "SPECULAR": ("#7c3aed", "SPECULAR / MIRROR"),
    "VOID": ("#b91c1c", "VOID / CLIFF DROP-OFF")
}

POSTURE_INFO = {
    "LIKE": ("#15803d", "THUMBS UP (LIKE)"),
    "DISLIKE": ("#b91c1c", "THUMBS DOWN (DISLIKE)"),
    "FLAT_HAND": ("#0284c7", "OPEN PALM (FLAT)"),
    "FIST": ("#7c3aed", "CLOSED FIST"),
    "BREAK_TIME": ("#b45309", "TIMEOUT (T-SIGN)"),
    "NONE": ("#64748b", "NO HAND DETECTED")
}


# ----------------------------------------------------------------------
# Serial Background Worker Thread with Firmware AT Pipeline Integration
# ----------------------------------------------------------------------
class SerialWorker(QThread):
    sig_frame_data = pyqtSignal(dict)
    sig_ai_event = pyqtSignal(dict)
    sig_status = pyqtSignal(str, bool)
    sig_raw_line = pyqtSignal(str)
    sig_mode_changed = pyqtSignal(str)

    def __init__(self, port, baud=921600):
        super().__init__()
        self.port = port
        self.baud = baud
        self.running = True
        self.ser = None

    def send_command(self, cmd_str):
        if self.ser and self.ser.is_open:
            try:
                cmd_clean = cmd_str.strip()
                full_cmd = cmd_clean + "\r\n"
                self.ser.write(full_cmd.encode("ascii", errors="ignore"))
                return True
            except Exception:
                return False
        return False

    def run(self):
        last_err = None
        for attempt in range(4):
            if not self.running:
                return
            try:
                self.ser = serial.Serial()
                self.ser.port = self.port
                self.ser.baudrate = self.baud
                self.ser.timeout = 0.1
                self.ser.setDTR(True)
                self.ser.setRTS(True)
                self.ser.open()
                break
            except Exception as e:
                last_err = e
                time.sleep(0.2)

        if self.ser is None or not self.ser.is_open:
            self.sig_status.emit(f"Failed to connect to {self.port}: {last_err}", False)
            return

        self.sig_status.emit(f"Connected to {self.port} @ {self.baud} baud", True)
        time.sleep(0.1)

        # Configure Firmware for Real-Time Streaming and Detection
        try:
            self.ser.reset_input_buffer()
            self.send_command("AT")
            time.sleep(0.04)
            self.send_command("AT+STREAM=RAW")
            time.sleep(0.04)
            self.send_command("AT+RESUME")
            time.sleep(0.04)
            self.send_command("AT+STATUS?")
        except Exception:
            pass

        current_frame = {
            "frame_id": 0,
            "temp_c": 0,
            "distances": [-1] * 64,
            "ambients": [0.0] * 64,
            "signals": [0.0] * 64,
            "cnh": {}
        }

        # Event Regex Parsers matching exact firmware outputs
        re_gesture = re.compile(r"\[AI GESTURE\] >>> DETECTED:\s*(\w+)\s*\(Confidence:\s*([0-9.]+)%\s*\|\s*Latency:\s*(\d+)\s*us(?: \|\s*(\d+)\s*cycles)?\)")
        re_surface = re.compile(r"\[AI SURFACE\] >>> DETECTED:\s*([A-Za-z_]+)(?:\s*\([^)]*\))?\s*\(Confidence:\s*([0-9.]+)%(?:\s*\|\s*FWHM:\s*([0-9.]+)\s*mm)?\s*\|\s*Latency:\s*(\d+)\s*us\)(?:\s*<<<)?")
        re_posture = re.compile(r"\[AI POSTURE\] >>> DETECTED:\s*(\w+)\s*\((?:Dist:\s*([0-9.]+)\s*mm\s*\|\s*)?Conf:\s*([0-9.]+)%\s*\|\s*Latency:\s*(\d+)\s*us\)")
        re_mode = re.compile(r"\[PIPELINE\] Active mode switched to:\s*(\w+)")

        while self.running:
            try:
                in_waiting = self.ser.in_waiting
                if in_waiting > 8192:
                    raw_data = self.ser.read(in_waiting)
                    last_nl = raw_data.rfind(b"\n")
                    if last_nl > 0:
                        prev_nl = raw_data.rfind(b"\n", 0, last_nl)
                        line_bytes = raw_data[prev_nl + 1 : last_nl] if prev_nl >= 0 else raw_data[:last_nl]
                        line = line_bytes.decode("ascii", errors="ignore").strip()
                    else:
                        continue
                else:
                    line = self.ser.readline().decode("ascii", errors="ignore").strip()

                if not line:
                    continue

                self.sig_raw_line.emit(line)

                # 1. Check for Edge AI On-Chip Detection Events
                m_gest = re_gesture.search(line)
                if m_gest:
                    self.sig_ai_event.emit({
                        "mode": "GESTURE",
                        "label": m_gest.group(1),
                        "confidence": float(m_gest.group(2)),
                        "latency_us": int(m_gest.group(3)),
                        "cycles": int(m_gest.group(4)) if m_gest.group(4) else 0,
                        "extra": ""
                    })
                    continue

                m_surf = re_surface.search(line)
                if m_surf:
                    self.sig_ai_event.emit({
                        "mode": "SURFACE",
                        "label": m_surf.group(1).upper(),
                        "confidence": float(m_surf.group(2)),
                        "latency_us": int(m_surf.group(4)),
                        "cycles": 0,
                        "extra": f"FWHM: {m_surf.group(3)} mm" if m_surf.group(3) else ""
                    })
                    continue

                m_post = re_posture.search(line)
                if m_post:
                    self.sig_ai_event.emit({
                        "mode": "POSTURE",
                        "label": m_post.group(1),
                        "confidence": float(m_post.group(3)),
                        "latency_us": int(m_post.group(4)),
                        "cycles": 0,
                        "extra": f"Target Dist: {m_post.group(2)} mm" if m_post.group(2) else ""
                    })
                    continue

                m_mode = re_mode.search(line)
                if m_mode:
                    self.sig_mode_changed.emit(m_mode.group(1))
                    continue

                # 2. Check for Atomic JSON Telemetry Frame
                if line.startswith("{") and line.endswith("}"):
                    try:
                        data = json.loads(line)
                        fid = data.get("f", 0)
                        temp = data.get("t", 0)
                        dists = data.get("d", [])
                        ambs = data.get("a", [])
                        sigs = data.get("s", [])
                        hists = data.get("h") or data.get("histograms", [])

                        current_frame["frame_id"] = fid
                        current_frame["temp_c"] = temp
                        if dists:
                            current_frame["distances"] = dists
                        if ambs:
                            current_frame["ambients"] = ambs
                        if sigs:
                            current_frame["signals"] = sigs
                        if hists:
                            for zid, z_bins in enumerate(hists):
                                amb_val = ambs[zid] if (ambs and zid < len(ambs)) else 0.0
                                current_frame["cnh"][zid] = (float(amb_val), [float(x) for x in z_bins])

                        self.sig_frame_data.emit(dict(current_frame))
                    except Exception:
                        pass
                    continue

                # 3. Check for CNH Histogram Lines
                if line.startswith("CNH,"):
                    parts = line.split(",")
                    if len(parts) >= 51:
                        try:
                            zid = int(parts[1])
                            amb = float(parts[2])
                            bins = [float(x) for x in parts[3:51]]
                            current_frame["cnh"][zid] = (amb, bins)
                            self.sig_frame_data.emit(dict(current_frame))
                        except ValueError:
                            pass
                    continue

            except Exception:
                if not self.running:
                    break
                time.sleep(0.01)

        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.sig_status.emit("Disconnected", False)

    def stop(self):
        self.running = False
        self.wait(1000)


# ----------------------------------------------------------------------
# Hero Live On-Chip Edge AI Output HUD (Clean Light Theme, No Label Boxing)
# ----------------------------------------------------------------------
class HeroAiOutputWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HeroAiCard")
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            #HeroAiCard {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
            }
            #HeroAiCard QLabel {
                border: none;
                background: transparent;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(6)

        # Header Row: Title & Active Mode Badge
        header_row = QHBoxLayout()
        self.lbl_pipeline = QLabel("ACTIVE PIPELINE: GESTURE (TEMPORAL 2D-CNN)")
        self.lbl_pipeline.setStyleSheet("font-size: 11px; font-weight: bold; color: #1e40af; letter-spacing: 0.5px; border: none; background: transparent;")

        self.lbl_mcu_badge = QLabel("[MCU INFERENCE ENGINE ACTIVE]")
        self.lbl_mcu_badge.setStyleSheet("font-size: 10px; font-weight: bold; color: #15803d; letter-spacing: 0.5px; border: none; background: transparent;")

        header_row.addWidget(self.lbl_pipeline)
        header_row.addStretch()
        header_row.addWidget(self.lbl_mcu_badge)
        layout.addLayout(header_row)

        # Center Showcase: Massive Classification Badge & Confidence Bar
        center_row = QHBoxLayout()
        center_row.setSpacing(24)

        # Left Column: Class Label Card
        class_col = QVBoxLayout()
        class_col.setSpacing(4)
        class_col.setContentsMargins(0, 0, 0, 0)

        lbl_sub = QLabel("LATEST ON-CHIP DETECTION:")
        lbl_sub.setStyleSheet("font-size: 9px; font-weight: bold; color: #64748b; letter-spacing: 0.5px; border: none; background: transparent;")

        self.lbl_class = QLabel("IDLE / READY")
        self.lbl_class.setStyleSheet("""
            font-size: 22px;
            font-weight: 900;
            color: #1e293b;
            border: none;
            background: transparent;
            padding: 0px;
            margin: 0px;
            font-family: 'Segoe UI', Arial, sans-serif;
        """)

        self.lbl_desc = QLabel("Waiting for surface or motion detection...")
        self.lbl_desc.setStyleSheet("font-size: 11px; color: #64748b; border: none; background: transparent; padding: 0px; margin: 0px;")

        class_col.addWidget(lbl_sub)
        class_col.addWidget(self.lbl_class)
        class_col.addWidget(self.lbl_desc)
        center_row.addLayout(class_col, 2)

        # Right Column: Confidence Meter & Telemetry Stats
        stats_col = QVBoxLayout()
        stats_col.setSpacing(4)

        conf_row = QHBoxLayout()
        lbl_conf_title = QLabel("MODEL CONFIDENCE:")
        lbl_conf_title.setStyleSheet("font-size: 10px; font-weight: bold; color: #475569; border: none; background: transparent;")
        self.lbl_conf_val = QLabel("0.0%")
        self.lbl_conf_val.setStyleSheet("font-size: 13px; font-weight: bold; color: #0f172a; border: none; background: transparent;")
        conf_row.addWidget(lbl_conf_title)
        conf_row.addStretch()
        conf_row.addWidget(self.lbl_conf_val)
        stats_col.addLayout(conf_row)

        self.bar_conf = QProgressBar()
        self.bar_conf.setRange(0, 100)
        self.bar_conf.setValue(0)
        self.bar_conf.setTextVisible(False)
        self.bar_conf.setFixedHeight(10)
        self.bar_conf.setStyleSheet("""
            QProgressBar {
                background-color: #f1f5f9;
                border: 1px solid #e2e8f0;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #059669);
                border-radius: 4px;
            }
        """)
        stats_col.addWidget(self.bar_conf)

        # Latency & Cycles Row
        sub_stats = QHBoxLayout()
        self.lbl_latency = QLabel("Latency: -- us")
        self.lbl_latency.setStyleSheet("font-size: 11px; font-weight: bold; color: #1d4ed8; border: none; background: transparent;")

        self.lbl_cycles = QLabel("Cycles: --")
        self.lbl_cycles.setStyleSheet("font-size: 11px; font-weight: bold; color: #7c3aed; border: none; background: transparent;")

        self.lbl_extra = QLabel("")
        self.lbl_extra.setStyleSheet("font-size: 11px; font-weight: bold; color: #b45309; border: none; background: transparent;")

        sub_stats.addWidget(self.lbl_latency)
        sub_stats.addSpacing(12)
        sub_stats.addWidget(self.lbl_cycles)
        sub_stats.addSpacing(12)
        sub_stats.addWidget(self.lbl_extra)
        sub_stats.addStretch()
        stats_col.addLayout(sub_stats)

        center_row.addLayout(stats_col, 3)
        layout.addLayout(center_row)

        # Bottom Row: Event History Timeline Strip
        history_box = QHBoxLayout()
        history_box.setSpacing(6)
        lbl_hist = QLabel("EVENT TIMELINE:")
        lbl_hist.setStyleSheet("font-size: 9px; font-weight: bold; color: #475569; border: none; background: transparent;")
        history_box.addWidget(lbl_hist)

        self.recent_events = deque(maxlen=6)
        self.event_chips = []
        for _ in range(6):
            chip = QLabel("---")
            chip.setStyleSheet("background-color: #ffffff; color: #94a3b8; border: 1px solid #e2e8f0; border-radius: 4px; padding: 3px 8px; font-size: 10px;")
            self.event_chips.append(chip)
            history_box.addWidget(chip)

        history_box.addStretch()
        layout.addLayout(history_box)

        # Inactivity timeout timer to revert live detection to idle state
        self.idle_timer = QTimer(self)
        self.idle_timer.setSingleShot(True)
        self.idle_timer.timeout.connect(self._on_idle_timeout)
        self.current_mode = "GESTURE"

    def set_active_mode(self, mode_name):
        self.current_mode = mode_name.upper()
        self.idle_timer.stop()
        self.reset_to_idle(self.current_mode)

    def reset_to_idle(self, mode_name=None):
        m = (mode_name or getattr(self, "current_mode", "GESTURE")).upper()
        if "GEST" in m:
            self.lbl_pipeline.setText("ACTIVE PIPELINE: GESTURE (TEMPORAL 2D-CNN)")
            self.lbl_desc.setText("Swipe hand (Left, Right, Up, Down) over sensor")
            self.lbl_class.setText("IDLE / READY")
        elif "SURF" in m:
            self.lbl_pipeline.setText("ACTIVE PIPELINE: SURFACE (48-BIN CNH 1D-CNN)")
            self.lbl_desc.setText("Point sensor at surface: Hard Floor, Carpet, Specular, Void")
            self.lbl_class.setText("SCANNING SURFACE")
        elif "POST" in m:
            self.lbl_pipeline.setText("ACTIVE PIPELINE: POSTURE (SPATIAL 2D-CNN)")
            self.lbl_desc.setText("Hold static pose: Thumbs Up, Thumbs Down, Palm, Fist, Break")
            self.lbl_class.setText("AWAITING HAND POSE")

        self.lbl_class.setStyleSheet("""
            font-size: 26px;
            font-weight: 900;
            color: #64748b;
            border: none;
            background: transparent;
            padding: 2px 0px;
            font-family: 'Segoe UI', Arial, sans-serif;
        """)
        self.bar_conf.setValue(0)
        self.lbl_conf_val.setText("0.0%")
        self.idle_timer.stop()

    def _on_idle_timeout(self):
        self.reset_to_idle(self.current_mode)

    def update_event(self, ev_dict):
        mode = ev_dict.get("mode", "GESTURE")
        self.current_mode = mode
        label = ev_dict.get("label", "UNKNOWN")
        conf = ev_dict.get("confidence", 0.0)
        lat_us = ev_dict.get("latency_us", 0)

        # Restart inactivity timer in GESTURE and POSTURE modes
        if mode in ("GESTURE", "POSTURE"):
            self.idle_timer.stop()
            self.idle_timer.start(2500)
        cycles = ev_dict.get("cycles", 0)
        extra = ev_dict.get("extra", "")

        color = "#1e293b"
        display_label = label

        if mode == "GESTURE":
            if label in GESTURE_INFO:
                color, display_label = GESTURE_INFO[label]
            self.lbl_pipeline.setText("ACTIVE PIPELINE: GESTURE (TEMPORAL 2D-CNN)")
            self.lbl_desc.setText("Touchless gesture swipe interaction detected")
        elif mode == "SURFACE":
            if label in SURFACE_INFO:
                color, display_label = SURFACE_INFO[label]
            self.lbl_pipeline.setText("ACTIVE PIPELINE: SURFACE (48-BIN CNH 1D-CNN)")
            self.lbl_desc.setText(f"Surface Material: {display_label}")
        elif mode == "POSTURE":
            if label in POSTURE_INFO:
                color, display_label = POSTURE_INFO[label]
            self.lbl_pipeline.setText("ACTIVE PIPELINE: POSTURE (SPATIAL 2D-CNN)")
            self.lbl_desc.setText(f"Hand Pose: {display_label}")

        self.lbl_class.setText(display_label)
        self.lbl_class.setStyleSheet(f"""
            font-size: 26px;
            font-weight: 900;
            color: {color};
            border: none;
            background: transparent;
            padding: 2px 0px;
            font-family: 'Segoe UI', Arial, sans-serif;
        """)

        self.lbl_conf_val.setText(f"{conf:.1f}%")
        self.bar_conf.setValue(int(min(100, max(0, conf))))

        self.lbl_latency.setText(f"Latency: {lat_us:,} us ({lat_us/1000.0:.2f} ms)")
        if cycles > 0:
            self.lbl_cycles.setText(f"Cycles: {cycles:,}")
            self.lbl_cycles.setVisible(True)
        else:
            self.lbl_cycles.setVisible(False)

        if extra:
            self.lbl_extra.setText(extra)
            self.lbl_extra.setVisible(True)
        else:
            self.lbl_extra.setVisible(False)

        # Update Timeline Strip
        ts = datetime.now().strftime("%H:%M:%S")
        self.recent_events.appendleft((ts, label, conf, color))
        for idx, chip in enumerate(self.event_chips):
            if idx < len(self.recent_events):
                t_str, lbl, c_val, col = self.recent_events[idx]
                chip.setText(f"[{t_str}] {lbl} ({c_val:.0f}%)")
                chip.setStyleSheet(f"background-color: #ffffff; color: {col}; border: 1px solid {col}; border-radius: 4px; padding: 3px 8px; font-size: 10px; font-weight: bold;")
                chip.setVisible(True)
            else:
                chip.setText("---")
                chip.setStyleSheet("background-color: #ffffff; color: #94a3b8; border: 1px solid #e2e8f0; border-radius: 4px; padding: 3px 8px; font-size: 10px;")


# ----------------------------------------------------------------------
# View 1 Left: 3D Spatial Depth Matrix & Live CNH Canonical Waveform
# ----------------------------------------------------------------------
class DepthMatrixWidget(QWidget):
    zone_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.distances = [-1] * 64
        self.dim = 8  # 8x8 by default, adapts to 4x4
        self.mode = "GESTURE"  # "GESTURE", "POSTURE", "SURFACE"
        self.selected_zone = 5
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.hand_centroid = None  # (cx, cy, cz, count)
        self.centroid_trail = deque(maxlen=10)

        # CNH optical return pulse & canonical window state (for Surface mode)
        self.ambient = 12.0
        self.cnh_bins = [0.0] * 48
        self.current_surface = "HARD_FLOOR"
        self.preset_fwhm_mm = 135.0
        self.cnh_data_by_zone = {}  # zid -> (ambient, bins)

        # Initialize default realistic surface pulse for zone 5
        self._init_default_pulse()

    def _init_default_pulse(self):
        self.cnh_bins = self._generate_surface_pulse("HARD_FLOOR", 14)

    def _generate_surface_pulse(self, surface_type, peak_bin=14):
        bins = [0.0] * 48
        s = surface_type.upper().strip()
        if s == "HARD_FLOOR":
            sigma = 1.5
            peak_val = 240.0
            amb = 12.0
            for i in range(48):
                diff = i - peak_bin
                bins[i] = max(0.0, amb + peak_val * math.exp(-0.5 * (diff / sigma) ** 2))
        elif s == "CARPET":
            sigma_left = 2.0
            sigma_right = 3.2
            peak_val = 160.0
            amb = 14.0
            for i in range(48):
                diff = i - peak_bin
                sig = sigma_left if diff < 0 else sigma_right
                bins[i] = max(0.0, amb + peak_val * math.exp(-0.5 * (diff / sig) ** 2))
        elif s == "SPECULAR":
            sigma = 1.1
            peak_val = 620.0
            amb = 10.0
            for i in range(48):
                diff = i - peak_bin
                bins[i] = max(0.0, amb + peak_val * math.exp(-0.5 * (diff / sigma) ** 2))
        elif s == "VOID":
            amb = 6.0
            for i in range(48):
                bins[i] = max(0.0, amb + (math.sin(i * 1.5) * 2.0))
        else:
            sigma = 1.6
            peak_val = 200.0
            amb = 10.0
            for i in range(48):
                diff = i - peak_bin
                bins[i] = max(0.0, amb + peak_val * math.exp(-0.5 * (diff / sigma) ** 2))
        return bins

    def set_mode(self, mode_name):
        self.mode = mode_name.upper().strip()
        if self.mode == "SURFACE":
            self.dim = 4
            self.hand_centroid = None
            self.centroid_trail.clear()
        else:
            self.dim = 8
        self.update()

    def set_distances(self, dist_list):
        self.distances = dist_list
        nb = len(dist_list)
        if self.mode == "SURFACE":
            self.dim = 4
        else:
            self.dim = 8 if nb >= 64 else 4

        # For SURFACE mode (4x4 matrix): Strictly NO target tracking!
        if self.dim == 4 or self.mode == "SURFACE":
            self.hand_centroid = None
            self.centroid_trail.clear()
            self.update()
            return

        # Compute 3D hand/target centroid across zones (< 450 mm) for Gesture & Posture
        sum_x, sum_y, sum_z, count = 0.0, 0.0, 0.0, 0
        min_d = 9999
        for r in range(self.dim):
            for c in range(self.dim):
                zid = r * self.dim + c
                d = dist_list[zid] if zid < nb else -1
                if 80 <= d <= 450:
                    weight = 450.0 - d
                    sum_x += c * weight
                    sum_y += r * weight
                    sum_z += d * weight
                    count += weight
                    if d < min_d:
                        min_d = d

        if count > 1e-3:
            cx = sum_x / count
            cy = sum_y / count
            self.hand_centroid = (cx, cy, min_d, count)
            self.centroid_trail.append((cx, cy))
        else:
            self.hand_centroid = None

        self.update()

    def set_selected_zone(self, zid):
        self.selected_zone = zid
        if zid in self.cnh_data_by_zone:
            self.ambient, self.cnh_bins = self.cnh_data_by_zone[zid]
        elif self.mode == "SURFACE":
            peak_b = 14
            if zid < len(self.distances) and self.distances[zid] > 50:
                peak_b = max(4, min(42, int(self.distances[zid] / 37.46)))
            self.cnh_bins = self._generate_surface_pulse(self.current_surface, peak_b)
        self.update()

    def set_cnh_data(self, zid, ambient, bins):
        self.cnh_data_by_zone[zid] = (ambient, bins)
        if zid == self.selected_zone:
            self.ambient = ambient
            self.cnh_bins = bins
            self.update()

    def set_surface_preset(self, surface_name, extra=None):
        s = surface_name.upper().strip()
        self.current_surface = s
        if isinstance(extra, str) and "FWHM:" in extra:
            try:
                m_f = re.search(r"FWHM:\s*([0-9.]+)", extra)
                if m_f:
                    self.preset_fwhm_mm = float(m_f.group(1))
            except Exception:
                pass
        elif isinstance(extra, (int, float)):
            self.preset_fwhm_mm = float(extra)

        peak_b = 14
        if 0 <= self.selected_zone < len(self.distances):
            zd = self.distances[self.selected_zone]
            if zd > 50:
                peak_b = max(4, min(42, int(zd / 37.46)))

        self.cnh_bins = self._generate_surface_pulse(s, peak_b)
        self.update()

    def mousePressEvent(self, event):
        w = self.width()
        h = self.height()
        margin = 14
        top_offset = 32

        if self.dim == 4 or self.mode == "SURFACE":
            grid_w = w - 2 * margin
            grid_h = min(210.0, (h - 60) * 0.40)
            if grid_w <= 0 or grid_h <= 0:
                return

            cell_w = grid_w / 4.0
            cell_h = (grid_h - top_offset) / 4.0

            mx = event.x() - margin
            my = event.y() - top_offset

            if 0 <= mx < grid_w and 0 <= my < (grid_h - top_offset):
                col = int(mx / cell_w)
                row = int(my / cell_h)
                zid = row * 4 + col
                self.selected_zone = zid
                self.set_selected_zone(zid)
                self.zone_clicked.emit(zid)
                self.update()
            return

        # 8x8 Mode: full widget grid
        grid_w = w - 2 * margin
        grid_h = h - top_offset - margin
        if grid_w <= 0 or grid_h <= 0:
            return

        cell_w = grid_w / float(self.dim)
        cell_h = grid_h / float(self.dim)

        mx = event.x() - margin
        my = event.y() - top_offset

        if 0 <= mx < grid_w and 0 <= my < grid_h:
            col = int(mx / cell_w)
            row = int(my / cell_h)
            zid = row * self.dim + col
            self.selected_zone = zid
            self.zone_clicked.emit(zid)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Canvas Background (Clean White)
        painter.fillRect(0, 0, w, h, QColor("#ffffff"))
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        # ------------------------------------------------------------------
        # MODE 1: SURFACE (4x4 Matrix + Bin Canonical Window Waveform)
        # ------------------------------------------------------------------
        if self.dim == 4 or self.mode == "SURFACE":
            margin = 14
            top_offset = 28
            grid_w = w - 2 * margin
            grid_h = min(210.0, (h - 60) * 0.40)

            # Header Title (Zero Target Tracking!)
            painter.setPen(QColor("#0f172a"))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(margin, 20, "4x4 MULTI-ZONE SURFACE GRID")

            # Selected Zone hint on right
            painter.setPen(QColor("#0284c7"))
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(QRect(w - 280, 5, 266, 20), Qt.AlignRight | Qt.AlignVCenter, f"INSPECTING ZONE: Z{self.selected_zone} (Click to change)")

            # Draw 4x4 Grid Cells
            cell_w = grid_w / 4.0
            cell_h = (grid_h - top_offset) / 4.0
            pad = 3.0

            for r in range(4):
                for c in range(4):
                    zid = r * 4 + c
                    dist = self.distances[zid] if zid < len(self.distances) else -1

                    cell_x = margin + c * cell_w + pad / 2.0
                    cell_y = top_offset + r * cell_h + pad / 2.0
                    cell_bw = cell_w - pad
                    cell_bh = cell_h - pad

                    # Palette
                    if dist <= 0:
                        bg = QColor("#f8fafc")
                        border_c = QColor("#e2e8f0")
                        text_c = QColor("#94a3b8")
                    elif dist < 150:
                        bg = QColor("#e11d48")
                        border_c = QColor("#be123c")
                        text_c = QColor("#ffffff")
                    elif dist < 250:
                        bg = QColor("#ea580c")
                        border_c = QColor("#c2410c")
                        text_c = QColor("#ffffff")
                    elif dist < 380:
                        bg = QColor("#d97706")
                        border_c = QColor("#b45309")
                        text_c = QColor("#ffffff")
                    elif dist < 600:
                        bg = QColor("#0284c7")
                        border_c = QColor("#0369a1")
                        text_c = QColor("#ffffff")
                    else:
                        bg = QColor("#334155")
                        border_c = QColor("#1e293b")
                        text_c = QColor("#ffffff")

                    is_sel = (zid == self.selected_zone)
                    if is_sel:
                        border_c = QColor("#0284c7")

                    painter.setBrush(QBrush(bg))
                    painter.setPen(QPen(border_c, 2.5 if is_sel else 1))
                    painter.drawRoundedRect(QRectF(cell_x, cell_y, cell_bw, cell_bh), 3, 3)

                    # Selected indicator accent
                    if is_sel:
                        painter.setBrush(QBrush(QColor("#0284c7")))
                        painter.setPen(Qt.NoPen)
                        painter.drawEllipse(QPointF(cell_x + cell_bw - 7, cell_y + 7), 3, 3)

                    # Zone ID in top-left
                    painter.setPen(QColor("#ffffff") if dist > 0 else QColor("#94a3b8"))
                    painter.setFont(QFont("Segoe UI", 7))
                    painter.drawText(int(cell_x + 4), int(cell_y + 11), f"Z{zid}")

                    # Distance Text in Center
                    painter.setPen(text_c)
                    painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                    dist_str = f"{dist} mm" if dist > 0 else "---"
                    painter.drawText(QRectF(cell_x, cell_y + 2, cell_bw, cell_bh - 2), Qt.AlignCenter, dist_str)

            # ------------------------------------------------------------------
            # BOTTOM SECTION: CLEAN 48-BIN PULSE & CANONICAL WINDOW WAVEFORM
            # ------------------------------------------------------------------
            wave_y = top_offset + (grid_h - top_offset) + 10
            wave_h = h - wave_y - margin

            if wave_h > 80:
                # Background container for waveform card
                painter.setBrush(QBrush(QColor("#ffffff")))
                painter.setPen(QPen(QColor("#e2e8f0"), 1))
                wave_box = QRectF(margin, wave_y, grid_w, wave_h)
                painter.drawRoundedRect(wave_box, 6, 6)

                # Ensure valid pulse (fallback to synthetic preset if empty or near-zero)
                has_signal = bool(self.cnh_bins and any(b > 0.05 for b in self.cnh_bins))
                if not has_signal:
                    peak_b = 14
                    if 0 <= self.selected_zone < len(self.distances) and self.distances[self.selected_zone] > 50:
                        peak_b = max(4, min(42, int(self.distances[self.selected_zone] / 37.46)))
                    self.cnh_bins = self._generate_surface_pulse(self.current_surface, peak_b)

                peak_idx = int(np.argmax(self.cnh_bins))
                peak_counts = float(self.cnh_bins[peak_idx])

                # Dynamic range scaling: minimum 100 counts so low-level noise stays low and doesn't blow up
                max_val = max(100.0, peak_counts * 1.25, self.ambient * 2.5)

                # Compute FWHM
                half_max = peak_counts / 2.0
                l_idx = peak_idx
                while l_idx > 0 and self.cnh_bins[l_idx] > half_max:
                    l_idx -= 1
                r_idx = peak_idx
                while r_idx < len(self.cnh_bins) - 1 and self.cnh_bins[r_idx] > half_max:
                    r_idx += 1
                fwhm_bins = max(1, r_idx - l_idx)
                fwhm_mm = fwhm_bins * 37.46 if has_signal else self.preset_fwhm_mm

                # Clean Non-overlapping Header: Dedicated left title and right telemetry rects
                header_w = grid_w - 20
                title_rect = QRectF(margin + 10, wave_y + 5, header_w * 0.48, 20)
                meta_rect = QRectF(margin + 10 + header_w * 0.48, wave_y + 5, header_w * 0.52, 20)

                painter.setPen(QColor("#0f172a"))
                painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
                painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, f"ZONE #{self.selected_zone} - 48-BIN OPTICAL PULSE")

                # Material Status Tag
                if fwhm_mm > 180.0:
                    badge_col = "#d97706"
                    mat_tag = "CARPET"
                elif peak_counts > 450.0:
                    badge_col = "#7c3aed"
                    mat_tag = "SPECULAR"
                elif peak_counts > 35.0:
                    badge_col = "#0284c7"
                    mat_tag = "HARD FLOOR"
                else:
                    badge_col = "#b91c1c"
                    mat_tag = "VOID / CLIFF"

                meta_str = f"FWHM: {fwhm_mm:.1f} mm  |  Pk: B{peak_idx} ({peak_counts:.0f} cts)  |  {mat_tag}"
                painter.setPen(QColor(badge_col))
                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                painter.drawText(meta_rect, Qt.AlignRight | Qt.AlignVCenter, meta_str)

                # Geometry for Waveform Plot (Full Width across the card!)
                p_left = margin + 36
                p_right = margin + grid_w - 16
                p_top = wave_y + 32
                p_bottom = wave_y + wave_h - 26
                p_w = p_right - p_left
                p_h = p_bottom - p_top

                if p_w > 50 and p_h > 40:
                    # Reference Grid Lines (0%, 50%, 100%)
                    painter.setPen(QPen(QColor("#f1f5f9"), 1))
                    for y_rat in (0.0, 0.5, 1.0):
                        y_line = p_top + y_rat * p_h
                        painter.drawLine(int(p_left), int(y_line), int(p_right), int(y_line))

                    # Y-Axis Labels
                    painter.setPen(QColor("#94a3b8"))
                    painter.setFont(QFont("Segoe UI", 7))
                    painter.drawText(int(margin + 4), int(p_top + 4), f"{max_val:4.0f}")
                    painter.drawText(int(margin + 4), int(p_top + p_h * 0.5 + 4), f"{max_val * 0.5:4.0f}")
                    painter.drawText(int(margin + 4), int(p_bottom), "   0")

                    # Ambient Baseline (amber dashed line, placed naturally near bottom)
                    if self.ambient > 0 and (self.ambient / max_val) < 0.85:
                        amb_y = p_bottom - (self.ambient / max_val) * p_h
                        painter.setPen(QPen(QColor("#f59e0b"), 1, Qt.DashLine))
                        painter.drawLine(int(p_left), int(amb_y), int(p_right), int(amb_y))
                        painter.setPen(QColor("#b45309"))
                        painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
                        painter.drawText(int(p_right - 68), int(amb_y - 3), f"Ambient: {self.ambient:.0f}")

                    # ----------------------------------------------------------
                    # HIGHLIGHT 16-BIN CANONICAL WINDOW [peak_idx - 4 : peak_idx + 11]
                    # ----------------------------------------------------------
                    step_x = p_w / 47.0
                    win_start = max(0, peak_idx - 4)
                    win_end = min(47, peak_idx + 11)
                    wx_start = p_left + win_start * step_x
                    wx_end = p_left + win_end * step_x
                    cw_width = max(10.0, wx_end - wx_start)

                    # 1. Soft translucent highlight band (No heavy cage!)
                    painter.fillRect(QRectF(wx_start, p_top, cw_width, p_h), QColor(2, 132, 199, 18))

                    # 2. Crisp boundary lines
                    painter.setPen(QPen(QColor("#38bdf8"), 1.2, Qt.DashLine))
                    painter.drawLine(int(wx_start), int(p_top), int(wx_start), int(p_bottom))
                    painter.drawLine(int(wx_end), int(p_top), int(wx_end), int(p_bottom))

                    # 3. Clean floating pill badge at top of canonical window
                    pill_w = min(cw_width - 6, 160.0)
                    if pill_w > 50:
                        pill_rect = QRectF(wx_start + (cw_width - pill_w) / 2.0, p_top + 3, pill_w, 16)
                        painter.setBrush(QBrush(QColor("#e0f2fe")))
                        painter.setPen(QPen(QColor("#bae6fd"), 1))
                        painter.drawRoundedRect(pill_rect, 3, 3)
                        painter.setPen(QColor("#0284c7"))
                        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                        painter.drawText(pill_rect, Qt.AlignCenter, "16-BIN CANONICAL WINDOW")

                    # 4. Waveform curve points (48 Bins)
                    points = []
                    for b, val in enumerate(self.cnh_bins[:48]):
                        px = p_left + b * step_x
                        py = p_bottom - (min(1.0, val / max_val) * p_h)
                        points.append(QPointF(px, py))

                    # Gradient Fill under Curve
                    poly = QPolygonF()
                    poly.append(QPointF(p_left, p_bottom))
                    for pt in points:
                        poly.append(pt)
                    poly.append(QPointF(p_left + p_w, p_bottom))

                    grad = QLinearGradient(0, p_top, 0, p_bottom)
                    grad.setColorAt(0.0, QColor(2, 132, 199, 75))
                    grad.setColorAt(1.0, QColor(2, 132, 199, 8))
                    painter.setBrush(QBrush(grad))
                    painter.setPen(Qt.NoPen)
                    painter.drawPolygon(poly)

                    # Draw curve: vibrant blue inside canonical window, smooth slate outside
                    for i in range(len(points) - 1):
                        is_in_win = (win_start <= i <= win_end)
                        pen_col = QColor("#0284c7") if is_in_win else QColor("#64748b")
                        pen_w = 2.4 if is_in_win else 1.6
                        painter.setPen(QPen(pen_col, pen_w))
                        painter.drawLine(points[i], points[i + 1])

                    # Peak Dot Marker (Red Dot with White Border at Peak Bin)
                    if 0 <= peak_idx < len(points) and peak_counts > 10.0:
                        pk_pt = points[peak_idx]
                        painter.setBrush(QBrush(QColor("#ef4444")))
                        painter.setPen(QPen(QColor("#ffffff"), 1.5))
                        painter.drawEllipse(pk_pt, 4.0, 4.0)

                        # Peak annotation text
                        painter.setPen(QColor("#ef4444"))
                        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                        lbl_y = pk_pt.y() - 6 if pk_pt.y() > p_top + 20 else pk_pt.y() + 14
                        painter.drawText(int(pk_pt.x() - 16), int(lbl_y), f"Peak B{peak_idx}")

                    # X-Axis Bin Marks (0, 8, 16, 24, 32, 40, 47)
                    painter.setPen(QColor("#64748b"))
                    painter.setFont(QFont("Segoe UI", 7))
                    for b_mark in (0, 8, 16, 24, 32, 40, 47):
                        bx = p_left + b_mark * step_x
                        painter.drawText(int(bx - 6), int(p_bottom + 12), f"{b_mark}")

                    # Centered X-Axis Sub-label
                    painter.setPen(QColor("#475569"))
                    painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                    painter.drawText(QRectF(p_left, p_bottom + 11, p_w, 14), Qt.AlignCenter, "CNH Optical Return Bins (48 Bins Total, 37.5 mm / bin)")

            return

        # ------------------------------------------------------------------
        # MODE 2: GESTURE / POSTURE (Full 8x8 Depth Matrix + Target Tracking)
        # ------------------------------------------------------------------
        # Header Title
        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(14, 23, f"3D SPATIAL DEPTH ({self.dim}x{self.dim})")

        # Target Tracker Coordinates
        if self.hand_centroid:
            cx, cy, cz, _ = self.hand_centroid
            fov_rad = math.radians(45.0)
            metric_x = (cx - (self.dim - 1)/2.0) * (cz * math.tan(fov_rad / 2.0) / ((self.dim - 1)/2.0))
            metric_y = (cy - (self.dim - 1)/2.0) * (cz * math.tan(fov_rad / 2.0) / ((self.dim - 1)/2.0))
            coord_str = f"TARGET: X={metric_x:+.0f} Y={metric_y:+.0f} Z={cz:.0f} mm"
            painter.setPen(QColor("#15803d"))
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(QRect(w - 240, 6, 226, 24), Qt.AlignRight | Qt.AlignVCenter, coord_str)
        else:
            painter.setPen(QColor("#94a3b8"))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(QRect(w - 200, 6, 186, 24), Qt.AlignRight | Qt.AlignVCenter, "TARGET: NO OBJECT")

        margin = 14
        top_offset = 36
        grid_w = w - 2 * margin
        grid_h = h - top_offset - margin
        if grid_w <= 0 or grid_h <= 0:
            return

        cell_w = grid_w / float(self.dim)
        cell_h = grid_h / float(self.dim)
        pad = 2 if self.dim == 8 else 5

        # Draw Cells
        for r in range(self.dim):
            for c in range(self.dim):
                zid = r * self.dim + c
                dist = self.distances[zid] if zid < len(self.distances) else -1

                cell_x = margin + c * cell_w + pad / 2.0
                cell_y = top_offset + r * cell_h + pad / 2.0
                cell_bw = cell_w - pad
                cell_bh = cell_h - pad

                # Color palette tailored for light theme readability
                if dist <= 0:
                    bg = QColor("#f8fafc")
                    border_c = QColor("#e2e8f0")
                    text_c = QColor("#94a3b8")
                elif dist < 150:
                    bg = QColor("#e11d48")  # Rose Red (Very Close)
                    border_c = QColor("#be123c")
                    text_c = QColor("#ffffff")
                elif dist < 250:
                    bg = QColor("#ea580c")  # Orange (Active gesture range)
                    border_c = QColor("#c2410c")
                    text_c = QColor("#ffffff")
                elif dist < 380:
                    bg = QColor("#d97706")  # Amber
                    border_c = QColor("#b45309")
                    text_c = QColor("#ffffff")
                elif dist < 600:
                    bg = QColor("#0284c7")  # Ocean Blue
                    border_c = QColor("#0369a1")
                    text_c = QColor("#ffffff")
                elif dist < 900:
                    bg = QColor("#2563eb")  # Royal Blue
                    border_c = QColor("#1d4ed8")
                    text_c = QColor("#ffffff")
                else:
                    bg = QColor("#334155")  # Deep Slate (Far)
                    border_c = QColor("#1e293b")
                    text_c = QColor("#ffffff")

                # Highlight selected zone
                is_sel = (zid == self.selected_zone)
                if is_sel:
                    border_c = QColor("#0f172a")

                painter.setBrush(QBrush(bg))
                painter.setPen(QPen(border_c, 2 if is_sel else 1))
                painter.drawRoundedRect(QRectF(cell_x, cell_y, cell_bw, cell_bh), 3, 3)

                # Zone ID in top-left corner
                painter.setPen(QColor("#ffffff") if dist > 0 else QColor("#94a3b8"))
                painter.setFont(QFont("Segoe UI", 7))
                painter.drawText(int(cell_x + 3), int(cell_y + 11), f"Z{zid}")

                # Distance Text in Center
                painter.setPen(text_c)
                painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
                dist_str = f"{dist}" if dist > 0 else "---"
                painter.drawText(QRectF(cell_x, cell_y + 4, cell_bw, cell_bh - 4), Qt.AlignCenter, dist_str)

        # Draw Real-Time 3D Centroid Tracking Crosshair & Motion Trail (8x8 mode only)
        if self.hand_centroid and len(self.centroid_trail) > 1:
            painter.setPen(QPen(QColor(234, 88, 12, 140), 2, Qt.DashLine))
            trail_pts = []
            for t_cx, t_cy in self.centroid_trail:
                pt_x = margin + (t_cx + 0.5) * cell_w
                pt_y = top_offset + (t_cy + 0.5) * cell_h
                trail_pts.append(QPointF(pt_x, pt_y))
            for i in range(len(trail_pts) - 1):
                painter.drawLine(trail_pts[i], trail_pts[i+1])

            # Centroid Target Crosshair
            cx, cy, cz, _ = self.hand_centroid
            tgt_x = margin + (cx + 0.5) * cell_w
            tgt_y = top_offset + (cy + 0.5) * cell_h

            # Holographic Target Rings
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#dc2626"), 2))
            painter.drawEllipse(QPointF(tgt_x, tgt_y), 14, 14)
            painter.setPen(QPen(QColor("#ea580c"), 1, Qt.DotLine))
            painter.drawEllipse(QPointF(tgt_x, tgt_y), 22, 22)

            # Center Bullseye
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.setPen(QPen(QColor("#dc2626"), 2))
            painter.drawEllipse(QPointF(tgt_x, tgt_y), 4, 4)

# ----------------------------------------------------------------------
# View 1 Right (Option A): Dynamic Touchless Gesture Interaction Widget
# ----------------------------------------------------------------------
class TouchlessGestureWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Define 5 Realistic Smartphone Multitasking App Cards
        self.apps = [
            {
                "title": "Music Player",
                "subtitle": "Daft Punk - Technologic",
                "accent": "#2563eb",
                "category": "AUDIO / MEDIA",
                "preview_type": "music",
                "info1": "Track: Technologic (2005)",
                "info2": "Bitrate: 320 kbps FLAC"
            },
            {
                "title": "Smart Home",
                "subtitle": "Living Room Hub",
                "accent": "#059669",
                "category": "IOT AUTOMATION",
                "preview_type": "home",
                "info1": "AC: 22.0 deg C (Eco)",
                "info2": "Lighting: 80% Warm"
            },
            {
                "title": "EdgeSense ToF",
                "subtitle": "VL53L8CH AI Radar",
                "accent": "#4f46e5",
                "category": "EDGE AI TELEMETRY",
                "preview_type": "radar",
                "info1": "Zones: 64 Active",
                "info2": "Cortex-M33: 250 MHz"
            },
            {
                "title": "Navigation Maps",
                "subtitle": "Circuit Digest Lab",
                "accent": "#d97706",
                "category": "GPS / TRANSIT",
                "preview_type": "maps",
                "info1": "Dist: 2.4 km (6 mins)",
                "info2": "Turn Right in 200m"
            },
            {
                "title": "Health & Vitals",
                "subtitle": "Daily Activity Rings",
                "accent": "#e11d48",
                "category": "FITNESS TRACKER",
                "preview_type": "health",
                "info1": "Steps: 8,420 / 10,000",
                "info2": "Active: 520 kcal"
            }
        ]

        self.num_apps = len(self.apps)
        self.active_index = 0
        self.current_pos = 0.0
        self.target_pos = 0.0

        # Vertical dismiss offset (for SWIPE_UP)
        self.dismiss_y = 0.0
        self.target_dismiss_y = 0.0

        # Minimize scale factor (for SWIPE_DOWN)
        self.minimize_scale = 1.0
        self.target_minimize_scale = 1.0

        # Live gesture feedback badge overlay
        self.feedback_text = "READY FOR GESTURES"
        self.feedback_sub = "Swipe Left / Right / Up / Down over sensor"
        self.feedback_color = "#64748b"
        self.feedback_alpha = 1.0

        # 60 FPS Physics / Easing Animation Timer
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(16)
        self.anim_timer.timeout.connect(self._update_animation)
        self.anim_timer.start()

    def handle_gesture(self, gesture_name):
        """Called whenever on-chip STM32 AI detects a gesture."""
        g = gesture_name.upper().strip()
        if g == "SWIPE_LEFT":
            self.active_index = (self.active_index + 1) % self.num_apps
            self.target_pos = float(self.active_index)
            self.target_dismiss_y = 0.0
            self.target_minimize_scale = 1.0
            self.feedback_text = "SWIPE LEFT  ->  NEXT APP"
            self.feedback_sub = f"Navigated to: {self.apps[self.active_index]['title']}"
            self.feedback_color = "#1d4ed8"
            self.feedback_alpha = 1.0

        elif g == "SWIPE_RIGHT":
            self.active_index = (self.active_index - 1 + self.num_apps) % self.num_apps
            self.target_pos = float(self.active_index)
            self.target_dismiss_y = 0.0
            self.target_minimize_scale = 1.0
            self.feedback_text = "<-  SWIPE RIGHT  PREVIOUS APP"
            self.feedback_sub = f"Navigated to: {self.apps[self.active_index]['title']}"
            self.feedback_color = "#15803d"
            self.feedback_alpha = 1.0

        elif g == "SWIPE_UP":
            self.target_dismiss_y = -160.0
            self.feedback_text = "^  SWIPE UP  DISMISS"
            self.feedback_sub = f"Closed: {self.apps[self.active_index]['title']}"
            self.feedback_color = "#b45309"
            self.feedback_alpha = 1.0
            QTimer.singleShot(420, self._restore_dismiss)

        elif g == "SWIPE_DOWN":
            self.target_minimize_scale = 0.68
            self.feedback_text = "v  SWIPE DOWN  MINIMIZE / OVERVIEW"
            self.feedback_sub = "Overview Grid"
            self.feedback_color = "#7c3aed"
            self.feedback_alpha = 1.0
            QTimer.singleShot(480, self._restore_minimize)

    def _restore_dismiss(self):
        self.target_dismiss_y = 0.0

    def _restore_minimize(self):
        self.target_minimize_scale = 1.0

    def _update_animation(self):
        diff = self.target_pos - self.current_pos
        if abs(diff) > self.num_apps / 2.0:
            if diff > 0:
                self.current_pos += self.num_apps
            else:
                self.current_pos -= self.num_apps
            diff = self.target_pos - self.current_pos

        self.current_pos += diff * 0.20

        while self.current_pos < 0.0:
            self.current_pos += self.num_apps
            self.target_pos += self.num_apps
        while self.current_pos >= self.num_apps:
            self.current_pos -= self.num_apps
            self.target_pos -= self.num_apps

        diff_y = self.target_dismiss_y - self.dismiss_y
        self.dismiss_y += diff_y * 0.22

        diff_s = self.target_minimize_scale - self.minimize_scale
        self.minimize_scale += diff_s * 0.20

        if self.feedback_alpha > 0.4:
            self.feedback_alpha -= 0.008

        self.update()

    def mousePressEvent(self, event):
        w = self.width()
        mx = event.x()
        if mx < w * 0.35:
            self.handle_gesture("SWIPE_RIGHT")
        elif mx > w * 0.65:
            self.handle_gesture("SWIPE_LEFT")
        else:
            if event.y() < self.height() * 0.4:
                self.handle_gesture("SWIPE_UP")
            else:
                self.handle_gesture("SWIPE_DOWN")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QColor("#ffffff"))
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        # Header Title
        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(14, 23, "TOUCHLESS GESTURE INTERACTION")

        painter.setPen(QColor("#64748b"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QRect(w - 220, 6, 206, 24), Qt.AlignRight | Qt.AlignVCenter, "TOUCHLESS HMI ACTIVE")

        pad_x = 16
        top_y = 36
        bot_y = h - 16
        phone_w = w - 2 * pad_x
        phone_h = bot_y - top_y
        if phone_w <= 120 or phone_h <= 120:
            return

        viewport_rect = QRectF(pad_x, top_y, phone_w, phone_h)
        painter.setBrush(QBrush(QColor("#f8fafc")))
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.drawRoundedRect(viewport_rect, 12, 12)

        center_x = pad_x + phone_w / 2.0
        center_y = top_y + phone_h / 2.0 + 8

        card_base_w = min(235.0, phone_w * 0.46)
        card_base_h = min(340.0, phone_h * 0.78)
        card_spacing = card_base_w * 1.28

        indices_order = sorted(range(self.num_apps), key=lambda i: -abs((i - self.current_pos + self.num_apps / 2.0) % self.num_apps - self.num_apps / 2.0))

        painter.save()
        clip_path = QPainterPath()
        clip_path.addRoundedRect(viewport_rect.adjusted(1, 1, -1, -1), 12, 12)
        painter.setClipPath(clip_path)

        for idx in indices_order:
            app = self.apps[idx]
            raw_diff = (idx - self.current_pos + self.num_apps / 2.0) % self.num_apps - self.num_apps / 2.0

            dist_c = abs(raw_diff)
            if dist_c > 1.4:
                continue

            scale = max(0.72, 1.0 - dist_c * 0.20) * self.minimize_scale
            alpha = max(0.0, 1.0 - (dist_c / 1.4) ** 1.8)
            if alpha < 0.02:
                continue

            c_w = card_base_w * scale
            c_h = card_base_h * scale

            c_x = center_x + raw_diff * card_spacing * self.minimize_scale - c_w / 2.0
            extra_y = self.dismiss_y if idx == self.active_index else 0.0
            c_y = center_y - c_h / 2.0 + extra_y

            card_rect = QRectF(c_x, c_y, c_w, c_h)

            painter.setBrush(QBrush(QColor(255, 255, 255, int(255 * alpha))))
            is_active = (idx == self.active_index)
            border_col = QColor(app["accent"]) if is_active else QColor("#cbd5e1")
            border_col.setAlpha(int(255 * alpha))
            painter.setPen(QPen(border_col, 2.0 if is_active else 1.0))
            painter.drawRoundedRect(card_rect, 10, 10)

            hdr_h = 32.0 * scale
            hdr_rect = QRectF(c_x, c_y, c_w, hdr_h)
            hdr_brush = QBrush(QColor(app["accent"]))
            painter.setBrush(hdr_brush)
            painter.setPen(Qt.NoPen)
            path = QPainterPath()
            path.addRoundedRect(card_rect, 10, 10)
            painter.save()
            painter.setClipPath(path)
            painter.drawRect(hdr_rect)

            painter.setPen(QColor(255, 255, 255, int(255 * alpha)))
            painter.setFont(QFont("Segoe UI", max(7, int(8.5 * scale)), QFont.Bold))
            painter.drawText(QRectF(c_x + 8 * scale, c_y, c_w - 16 * scale, hdr_h), Qt.AlignVCenter | Qt.AlignLeft, app["title"].upper())
            painter.restore()

            body_y = c_y + hdr_h + 8 * scale
            painter.setPen(QColor(15, 23, 42, int(255 * alpha)))
            painter.setFont(QFont("Segoe UI", max(8, int(9.5 * scale)), QFont.Bold))
            painter.drawText(QRectF(c_x + 8 * scale, body_y, c_w - 16 * scale, 18 * scale), Qt.AlignLeft, app["subtitle"])

            painter.setPen(QColor(100, 116, 139, int(255 * alpha)))
            painter.setFont(QFont("Segoe UI", max(6, int(7 * scale))))
            painter.drawText(QRectF(c_x + 8 * scale, body_y + 16 * scale, c_w - 16 * scale, 13 * scale), Qt.AlignLeft, app["category"])

            ptype = app["preview_type"]
            vis_y = body_y + 32 * scale
            vis_h = max(20.0, c_h - hdr_h - 68 * scale)
            vis_rect = QRectF(c_x + 8 * scale, vis_y, c_w - 16 * scale, vis_h)

            painter.setBrush(QBrush(QColor(248, 250, 252, int(255 * alpha))))
            painter.setPen(QPen(QColor(226, 232, 240, int(255 * alpha)), 1))
            painter.drawRoundedRect(vis_rect, 6, 6)

            if ptype == "music":
                nb_bars = 7
                bw = (vis_rect.width() - 14) / nb_bars
                for bi in range(nb_bars):
                    bh = (0.3 + 0.6 * math.sin(time.time() * 4.0 + bi * 1.2)**2) * (vis_h - 12)
                    bx = vis_rect.x() + 7 + bi * bw
                    by = vis_rect.bottom() - 6 - bh
                    painter.fillRect(QRectF(bx, by, max(1.0, bw - 3), bh), QColor(app["accent"]))

            elif ptype == "home":
                painter.setPen(QColor(5, 150, 105, int(255 * alpha)))
                painter.setFont(QFont("Segoe UI", max(6, int(7.5 * scale)), QFont.Bold))
                painter.drawText(vis_rect, Qt.AlignCenter, "AC: 22 C ACTIVE\nLIGHTS: 80% WARM")

            elif ptype == "radar":
                rc_x = vis_rect.center().x()
                rc_y = vis_rect.center().y()
                rad = min(vis_rect.width(), vis_rect.height()) / 2.4
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(79, 70, 229, int(180 * alpha)), 1))
                painter.drawEllipse(QPointF(rc_x, rc_y), rad, rad)
                painter.drawEllipse(QPointF(rc_x, rc_y), rad * 0.5, rad * 0.5)
                bx = rc_x + rad * 0.4 * math.cos(time.time() * 3.0)
                by = rc_y + rad * 0.4 * math.sin(time.time() * 3.0)
                painter.setBrush(QBrush(QColor("#e11d48")))
                painter.drawEllipse(QPointF(bx, by), 3, 3)

            elif ptype == "maps":
                painter.setPen(QColor(217, 119, 6, int(255 * alpha)))
                painter.setFont(QFont("Segoe UI", max(6, int(7.5 * scale)), QFont.Bold))
                painter.drawText(vis_rect, Qt.AlignCenter, "ROUTE: 2.4 KM\nTURN RIGHT IN 200M")

            elif ptype == "health":
                rc_x = vis_rect.center().x()
                rc_y = vis_rect.center().y()
                rad = min(vis_rect.width(), vis_rect.height()) / 2.4
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(225, 29, 72, int(220 * alpha)), 2.5))
                painter.drawArc(QRectF(rc_x - rad, rc_y - rad, rad * 2, rad * 2), 0, int(270 * 16))
                painter.setPen(QPen(QColor(5, 150, 105, int(220 * alpha)), 2.5))
                painter.drawArc(QRectF(rc_x - rad + 4, rc_y - rad + 4, (rad - 4) * 2, (rad - 4) * 2), 0, int(210 * 16))

            painter.setPen(QColor(71, 85, 105, int(255 * alpha)))
            painter.setFont(QFont("Segoe UI", max(6, int(7 * scale))))
            painter.drawText(QRectF(c_x + 6 * scale, c_y + c_h - 20 * scale, c_w - 12 * scale, 16 * scale), Qt.AlignCenter, app["info1"])

        painter.restore()

        # Navigation Dots Indicator at bottom of viewport
        dots_y = top_y + phone_h - 14
        dot_spacing = 14
        dots_w = (self.num_apps - 1) * dot_spacing
        start_dx = center_x - dots_w / 2.0
        for di in range(self.num_apps):
            dx = start_dx + di * dot_spacing
            is_active_dot = (di == self.active_index)
            painter.setBrush(QBrush(QColor("#2563eb") if is_active_dot else QColor("#cbd5e1")))
            painter.setPen(Qt.NoPen)
            if is_active_dot:
                painter.drawRoundedRect(QRectF(dx - 6, dots_y - 3, 12, 6), 3, 3)
            else:
                painter.drawEllipse(QPointF(dx, dots_y), 3, 3)

        # Live Animated Gesture Feedback Banner Overlay
        if self.feedback_alpha > 0.05:
            banner_w = min(260.0, phone_w - 24.0)
            banner_h = 36.0
            banner_x = center_x - banner_w / 2.0
            banner_y = top_y + 12.0

            banner_bg = QColor(255, 255, 255, int(245 * self.feedback_alpha))
            banner_pen = QColor(self.feedback_color)
            banner_pen.setAlpha(int(220 * self.feedback_alpha))

            painter.setBrush(QBrush(banner_bg))
            painter.setPen(QPen(banner_pen, 1.5))
            painter.drawRoundedRect(QRectF(banner_x, banner_y, banner_w, banner_h), 8, 8)

            painter.setPen(banner_pen)
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(QRectF(banner_x, banner_y + 2, banner_w, 18), Qt.AlignCenter, self.feedback_text)

            sub_pen = QColor("#64748b")
            sub_pen.setAlpha(int(200 * self.feedback_alpha))
            painter.setPen(sub_pen)
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(QRectF(banner_x, banner_y + 18, banner_w, 14), Qt.AlignCenter, self.feedback_sub)


# Backward-compatibility alias
MobileRecentTabsWidget = TouchlessGestureWidget


# ----------------------------------------------------------------------
# View 1 Right (Option B): Dynamic Touchless Posture Interaction Widget
# ----------------------------------------------------------------------
class TouchlessPostureWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.current_pose = "NONE"
        self.confidence = 0.0
        self.latency_us = 0
        self.pose_scale = 1.0
        self.target_pose_scale = 1.0
        self.ripple_rad = 0.0
        self.ripple_alpha = 0.0

        # Action Feedback Banner
        self.action_title = "TOUCHLESS POSTURE INTERACTION"
        self.action_sub = "Hold hand pose (Thumbs Up, Down, Palm, Fist, Break) in front of sensor"
        self.action_color = "#64748b"
        self.action_alpha = 1.0

        # Simulated Device HMI States
        self.is_playing = True
        self.is_favorite = False
        self.is_locked = False
        self.timeout_active = False
        self.current_track_idx = 0
        self.tracks = [
            ("Daft Punk - Technologic", "320 kbps FLAC"),
            ("Kraftwerk - The Robots", "48 kHz Hi-Res"),
            ("Justice - Genesis", "Lossless Audio"),
            ("Deadmau5 - Strobe", "Master Quality")
        ]

        # 60 FPS animation timer
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(16)
        self.anim_timer.timeout.connect(self._update_animation)
        self.anim_timer.start()

        # Inactivity timeout timer to reset visual artwork to idle when no posture is outputted
        self.idle_timer = QTimer(self)
        self.idle_timer.setSingleShot(True)
        self.idle_timer.timeout.connect(self.reset_to_idle)
        self.timeout_ms = 2200  # 2.2 seconds

    def reset_to_idle(self):
        if self.current_pose != "NONE":
            self.current_pose = "NONE"
            self.confidence = 0.0
            self.pose_scale = 1.0
            self.action_title = "AWAITING HAND POSE"
            self.action_sub = "Hold hand 15 - 40 cm in front of sensor"
            self.action_color = "#64748b"
            self.action_alpha = 0.8
            self.idle_timer.stop()
            self.update()

    def handle_posture(self, pose_name, conf=0.0, lat_us=0):
        p = pose_name.upper().strip()
        if p == "NONE":
            self.reset_to_idle()
            return

        self.current_pose = p
        self.confidence = conf
        self.latency_us = lat_us

        # Restart inactivity timeout timer
        self.idle_timer.stop()
        self.idle_timer.start(self.timeout_ms)

        # Trigger reactive scale pulse
        self.pose_scale = 1.15
        self.ripple_rad = 45.0
        self.ripple_alpha = 1.0

        track_title = self.tracks[self.current_track_idx][0]

        if p == "LIKE":
            self.is_favorite = True
            self.is_playing = True
            self.action_title = "[THUMBS UP]  ACTION: APPROVED & FAVORITED"
            self.action_sub = f"Added '{track_title}' to Favorites - Playback Resumed"
            self.action_color = "#15803d"
            self.action_alpha = 1.0

        elif p == "DISLIKE":
            self.is_favorite = False
            self.current_track_idx = (self.current_track_idx + 1) % len(self.tracks)
            new_title = self.tracks[self.current_track_idx][0]
            self.action_title = "[THUMBS DOWN]  ACTION: TRACK SKIPPED"
            self.action_sub = f"Skipped - Now Playing: '{new_title}'"
            self.action_color = "#b91c1c"
            self.action_alpha = 1.0

        elif p == "FLAT_HAND":
            self.is_playing = False
            self.action_title = "[OPEN PALM]  ACTION: PLAYBACK PAUSED"
            self.action_sub = "HMI Motion Halted - Music Player Paused"
            self.action_color = "#0284c7"
            self.action_alpha = 1.0

        elif p == "FIST":
            self.is_locked = not self.is_locked
            st = "LOCKED" if self.is_locked else "UNLOCKED"
            self.action_title = f"[CLOSED FIST]  ACTION: SYSTEM {st}"
            self.action_sub = f"Interface Security {st.capitalize()} via Clench Gesture"
            self.action_color = "#7c3aed"
            self.action_alpha = 1.0

        elif p == "BREAK_TIME":
            self.timeout_active = not self.timeout_active
            st = "ACTIVATED" if self.timeout_active else "CLEARED"
            self.action_title = f"[TIMEOUT]  ACTION: BREAK MODE {st}"
            self.action_sub = "10-Minute Rest & Stretch Timer Started" if self.timeout_active else "Resumed Active Dashboard"
            self.action_color = "#b45309"
            self.action_alpha = 1.0

        elif p == "NONE":
            self.action_title = "AWAITING HAND POSE"
            self.action_sub = "Hold hand 15 - 40 cm in front of sensor"
            self.action_color = "#64748b"
            self.action_alpha = 0.8

        self.update()

    def _update_animation(self):
        # Scale decay to 1.0
        self.pose_scale += (1.0 - self.pose_scale) * 0.15
        if self.ripple_alpha > 0.05:
            self.ripple_rad += 2.0
            self.ripple_alpha -= 0.03

        if self.action_alpha > 0.6:
            self.action_alpha -= 0.005

        self.update()

    def mousePressEvent(self, event):
        # Bottom chips hit detection for mouse testing
        w = self.width()
        h = self.height()
        bot_y = h - 42
        if event.y() >= bot_y:
            chip_w = w / 5.0
            idx = int(event.x() / chip_w)
            poses = ["LIKE", "DISLIKE", "FLAT_HAND", "FIST", "BREAK_TIME"]
            if 0 <= idx < len(poses):
                self.handle_posture(poses[idx], 99.0, 4800)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QColor("#ffffff"))
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        # Header Title
        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(14, 23, "TOUCHLESS POSTURE INTERACTION")

        painter.setPen(QColor("#64748b"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QRect(w - 220, 6, 206, 24), Qt.AlignRight | Qt.AlignVCenter, "TOUCHLESS HMI ACTIVE")

        pad_x = 16
        top_y = 36
        bot_y = h - 46
        vp_w = w - 2 * pad_x
        vp_h = bot_y - top_y
        if vp_w <= 120 or vp_h <= 120:
            return

        viewport_rect = QRectF(pad_x, top_y, vp_w, vp_h)
        painter.setBrush(QBrush(QColor("#f8fafc")))
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.drawRoundedRect(viewport_rect, 12, 12)

        # 1. Top Action Feedback Banner
        banner_w = vp_w - 20.0
        banner_h = 42.0
        banner_x = pad_x + 10.0
        banner_y = top_y + 10.0
        banner_rect = QRectF(banner_x, banner_y, banner_w, banner_h)

        b_bg = QColor(255, 255, 255, int(245 * min(1.0, self.action_alpha)))
        b_pen = QColor(self.action_color)
        b_pen.setAlpha(int(220 * min(1.0, self.action_alpha)))

        painter.setBrush(QBrush(b_bg))
        painter.setPen(QPen(b_pen, 1.5))
        painter.drawRoundedRect(banner_rect, 8, 8)

        painter.setPen(b_pen)
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(QRectF(banner_x, banner_y + 3, banner_w, 20), Qt.AlignCenter, self.action_title)

        painter.setPen(QColor("#475569"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QRectF(banner_x, banner_y + 22, banner_w, 16), Qt.AlignCenter, self.action_sub)

        # 2. Main Content Split: Card A (Avatar) & Card B (Smart Controller)
        content_top = banner_y + banner_h + 10.0
        content_h = vp_h - (content_top - top_y) - 10.0
        col_gap = 12.0
        col_w = (vp_w - 20.0 - col_gap) / 2.0

        # --- CARD A: Live Hand Posture Avatar ---
        card_a_rect = QRectF(pad_x + 10.0, content_top, col_w, content_h)
        painter.setBrush(QBrush(QColor("#ffffff")))
        pose_color = POSTURE_INFO.get(self.current_pose, ("#64748b", ""))[0]
        painter.setPen(QPen(QColor(pose_color), 1.5 if self.current_pose != "NONE" else 1.0))
        painter.drawRoundedRect(card_a_rect, 10, 10)

        # Card A Title
        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(int(card_a_rect.x() + 12), int(card_a_rect.y() + 22), "DETECTED HAND POSE")

        # Available avatar bounds inside Card A:
        usable_top = card_a_rect.y() + 32.0
        usable_bot = card_a_rect.bottom() - 52.0
        usable_h = max(100.0, usable_bot - usable_top)
        usable_w = max(100.0, card_a_rect.width() - 20.0)

        # Avatar Center in usable area
        avatar_cx = card_a_rect.center().x()
        avatar_cy = usable_top + usable_h / 2.0

        # Auto scale factor: reference vector size is ~80x80 units
        # Scale adaptively to fill ~65% of the card for prominent visibility
        auto_scale = min(usable_w / 80.0, usable_h / 85.0) * 0.72
        auto_scale = max(1.6, min(2.5, auto_scale))
        draw_scale = auto_scale * self.pose_scale

        # Pulsing Halo Ring on detection
        if self.ripple_alpha > 0.05 and self.current_pose != "NONE":
            painter.setBrush(Qt.NoBrush)
            r_pen = QColor(pose_color)
            r_pen.setAlpha(int(200 * self.ripple_alpha))
            painter.setPen(QPen(r_pen, 2))
            r_rad = self.ripple_rad * auto_scale * 0.45
            painter.drawEllipse(QPointF(avatar_cx, avatar_cy), r_rad, r_rad)

        # Draw Clean Scalable Vector Hand Pose (No Emojis)
        self._draw_pose_vector(painter, avatar_cx, avatar_cy, self.current_pose, pose_color, draw_scale)

        # Pose Label & Telemetry Readout
        display_name = POSTURE_INFO.get(self.current_pose, ("#64748b", "NO HAND DETECTED"))[1]
        painter.setPen(QColor(pose_color))
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.drawText(QRectF(card_a_rect.x(), card_a_rect.bottom() - 48, card_a_rect.width(), 22), Qt.AlignCenter, display_name)

        painter.setPen(QColor("#64748b"))
        painter.setFont(QFont("Segoe UI", 9))
        telemetry_str = f"Confidence: {self.confidence:.1f}% | Latency: {self.latency_us:,} us" if self.current_pose != "NONE" else "Awaiting hand posture in view"
        painter.drawText(QRectF(card_a_rect.x(), card_a_rect.bottom() - 26, card_a_rect.width(), 18), Qt.AlignCenter, telemetry_str)

        # --- CARD B: Simulated HMI Media & System Controller ---
        card_b_rect = QRectF(pad_x + 10.0 + col_w + col_gap, content_top, col_w, content_h)
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.setPen(QPen(QColor("#cbd5e1"), 1.0))
        painter.drawRoundedRect(card_b_rect, 10, 10)

        # Card B Title
        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(int(card_b_rect.x() + 12), int(card_b_rect.y() + 22), "SMART HMI CONTROLLER")

        # Media Banner Header
        m_banner_rect = QRectF(card_b_rect.x() + 12, card_b_rect.y() + 34, card_b_rect.width() - 24, 60)
        painter.setBrush(QBrush(QColor("#f1f5f9")))
        painter.setPen(QPen(QColor("#e2e8f0"), 1))
        painter.drawRoundedRect(m_banner_rect, 6, 6)

        track_title, track_sub = self.tracks[self.current_track_idx]
        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(int(m_banner_rect.x() + 10), int(m_banner_rect.y() + 24), track_title)

        painter.setPen(QColor("#64748b"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(int(m_banner_rect.x() + 10), int(m_banner_rect.y() + 44), track_sub)

        # Favorite Heart / Badge indicator
        fav_col = QColor("#e11d48") if self.is_favorite else QColor("#cbd5e1")
        painter.setPen(fav_col)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(QRectF(m_banner_rect.right() - 80, m_banner_rect.y() + 12, 70, 20), Qt.AlignRight, "FAVORITE" if self.is_favorite else "---")

        # Equalizer Waveform Area (Animates when playing, flat when paused)
        eq_rect = QRectF(card_b_rect.x() + 12, card_b_rect.y() + 104, card_b_rect.width() - 24, 45)
        painter.setBrush(QBrush(QColor("#f8fafc")))
        painter.setPen(QPen(QColor("#e2e8f0"), 1))
        painter.drawRoundedRect(eq_rect, 6, 6)

        num_bars = 11
        bar_w = (eq_rect.width() - 24) / num_bars
        t_now = time.time()
        for bi in range(num_bars):
            if self.is_playing:
                b_height = (0.25 + 0.65 * math.sin(t_now * 5.0 + bi * 0.9)**2) * (eq_rect.height() - 14)
            else:
                b_height = 4.0  # Frozen flat line on pause
            bx = eq_rect.x() + 12 + bi * bar_w
            by = eq_rect.bottom() - 7 - b_height
            bar_color = QColor("#2563eb") if self.is_playing else QColor("#94a3b8")
            painter.fillRect(QRectF(bx, by, max(2.0, bar_w - 4), b_height), bar_color)

        # Status Badges Row
        badge_y = eq_rect.bottom() + 12
        badge_w = (card_b_rect.width() - 24 - 16) / 3.0
        badge_h = 32.0

        # Status 1: Play/Pause State
        b1_rect = QRectF(card_b_rect.x() + 12, badge_y, badge_w, badge_h)
        b1_bg = QColor("#dcfce7") if self.is_playing else QColor("#e0f2fe")
        b1_fg = QColor("#15803d") if self.is_playing else QColor("#0284c7")
        painter.setBrush(QBrush(b1_bg))
        painter.setPen(QPen(b1_fg, 1))
        painter.drawRoundedRect(b1_rect, 4, 4)
        painter.setPen(b1_fg)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(b1_rect, Qt.AlignCenter, "PLAYING" if self.is_playing else "PAUSED")

        # Status 2: Security Lock State
        b2_rect = QRectF(card_b_rect.x() + 12 + badge_w + 8, badge_y, badge_w, badge_h)
        b2_bg = QColor("#f3e8ff") if self.is_locked else QColor("#f1f5f9")
        b2_fg = QColor("#7c3aed") if self.is_locked else QColor("#64748b")
        painter.setBrush(QBrush(b2_bg))
        painter.setPen(QPen(b2_fg, 1))
        painter.drawRoundedRect(b2_rect, 4, 4)
        painter.setPen(b2_fg)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(b2_rect, Qt.AlignCenter, "LOCKED" if self.is_locked else "UNLOCKED")

        # Status 3: Break/Timeout State
        b3_rect = QRectF(card_b_rect.x() + 12 + 2 * (badge_w + 8), badge_y, badge_w, badge_h)
        b3_bg = QColor("#fef3c7") if self.timeout_active else QColor("#f1f5f9")
        b3_fg = QColor("#b45309") if self.timeout_active else QColor("#64748b")
        painter.setBrush(QBrush(b3_bg))
        painter.setPen(QPen(b3_fg, 1))
        painter.drawRoundedRect(b3_rect, 4, 4)
        painter.setPen(b3_fg)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(b3_rect, Qt.AlignCenter, "REST (10M)" if self.timeout_active else "ACTIVE")

        # System Security Shield Overlay if Locked
        if self.is_locked:
            shield_rect = QRectF(card_b_rect.x() + 8, card_b_rect.y() + 30, card_b_rect.width() - 16, card_b_rect.height() - 38)
            painter.setBrush(QBrush(QColor(255, 255, 255, 220)))
            painter.setPen(QPen(QColor("#7c3aed"), 1.5))
            painter.drawRoundedRect(shield_rect, 8, 8)
            painter.setPen(QColor("#7c3aed"))
            painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
            painter.drawText(shield_rect, Qt.AlignCenter, "SYSTEM LOCKED BY FIST GESTURE\nClench Fist Again to Unlock")

        # 3. Bottom Interactive Pose Selector Chips (For mouse preview/testing)
        poses = [
            ("Thumbs Up", "#15803d", "LIKE"),
            ("Thumbs Down", "#b91c1c", "DISLIKE"),
            ("Open Palm", "#0284c7", "FLAT_HAND"),
            ("Closed Fist", "#7c3aed", "FIST"),
            ("Timeout", "#b45309", "BREAK_TIME")
        ]
        chip_gap = 8.0
        chip_w = (w - 2 * pad_x - (len(poses) - 1) * chip_gap) / float(len(poses))
        chip_h = 28.0
        chip_y = h - 36.0

        for ci, (p_name, p_col, p_code) in enumerate(poses):
            cx = pad_x + ci * (chip_w + chip_gap)
            c_rect = QRectF(cx, chip_y, chip_w, chip_h)
            is_active_chip = (self.current_pose == p_code)

            if is_active_chip:
                painter.setBrush(QBrush(QColor(p_col)))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(c_rect, 5, 5)
                painter.setPen(QColor("#ffffff"))
            else:
                painter.setBrush(QBrush(QColor("#ffffff")))
                painter.setPen(QPen(QColor("#cbd5e1"), 1))
                painter.drawRoundedRect(c_rect, 5, 5)
                painter.setPen(QColor("#475569"))

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold if is_active_chip else QFont.Normal))
            painter.drawText(c_rect, Qt.AlignCenter, p_name)

    def _draw_pose_vector(self, painter, cx, cy, pose, color, scale):
        """Draws clean vector stylized hand poses without emojis."""
        painter.save()
        painter.translate(cx, cy)
        painter.scale(scale, scale)

        pen = QPen(QColor(color), 2.2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(color).lighter(185)))

        if pose == "LIKE":
            # Thumbs Up vector
            path = QPainterPath()
            path.moveTo(-18, 15)
            path.lineTo(12, 15)
            path.quadTo(20, 15, 20, 5)
            path.lineTo(20, -5)
            path.quadTo(20, -15, 12, -15)
            path.lineTo(2, -15)
            path.lineTo(2, -32)
            path.quadTo(2, -42, -6, -42)
            path.quadTo(-14, -42, -14, -30)
            path.lineTo(-14, -10)
            path.lineTo(-18, -10)
            path.closeSubpath()
            painter.drawPath(path)
            # Upward arrows
            painter.drawLine(0, -46, 0, -54)
            painter.drawLine(-5, -50, 0, -54)
            painter.drawLine(5, -50, 0, -54)

        elif pose == "DISLIKE":
            # Thumbs Down vector
            path = QPainterPath()
            path.moveTo(-18, -15)
            path.lineTo(12, -15)
            path.quadTo(20, -15, 20, -5)
            path.lineTo(20, 5)
            path.quadTo(20, 15, 12, 15)
            path.lineTo(2, 15)
            path.lineTo(2, 32)
            path.quadTo(2, 42, -6, 42)
            path.quadTo(-14, 42, -14, 30)
            path.lineTo(-14, 10)
            path.lineTo(-18, 10)
            path.closeSubpath()
            painter.drawPath(path)
            # Downward arrows
            painter.drawLine(0, 46, 0, 54)
            painter.drawLine(-5, 50, 0, 54)
            painter.drawLine(5, 50, 0, 54)

        elif pose == "FLAT_HAND":
            # Open Palm with 5 extended fingers
            path = QPainterPath()
            path.addRoundedRect(QRectF(-20, -10, 40, 42), 6, 6)
            # 4 fingers
            for fi in range(4):
                fx = -18 + fi * 9.5
                path.addRoundedRect(QRectF(fx, -38, 7.5, 32), 3, 3)
            # Thumb
            path.addRoundedRect(QRectF(-32, 2, 14, 10), 3, 3)
            painter.drawPath(path)
            # Concentric Halo Rings
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(color), 1, Qt.DashLine))
            painter.drawEllipse(QPointF(0, 0), 45, 45)

        elif pose == "FIST":
            # Closed Fist
            path = QPainterPath()
            path.addRoundedRect(QRectF(-22, -22, 44, 44), 10, 10)
            painter.drawPath(path)
            # Knuckle divider lines
            painter.drawLine(-10, -22, -10, 0)
            painter.drawLine(0, -22, 0, 0)
            painter.drawLine(10, -22, 10, 0)
            # Clamping brackets
            painter.setBrush(Qt.NoBrush)
            painter.drawArc(QRectF(-34, -34, 68, 68), int(45 * 16), int(90 * 16))
            painter.drawArc(QRectF(-34, -34, 68, 68), int(225 * 16), int(90 * 16))

        elif pose == "BREAK_TIME":
            # T-Sign (Horizontal hand over vertical hand)
            path = QPainterPath()
            # Top horizontal bar
            path.addRoundedRect(QRectF(-32, -30, 64, 16), 5, 5)
            # Vertical hand bar
            path.addRoundedRect(QRectF(-8, -14, 16, 48), 5, 5)
            painter.drawPath(path)
            # Center break icon
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.drawEllipse(QPointF(0, -22), 3, 3)

        else:
            # Minimalist Radar Alignment Target Reticle (Clean Idle State, No Hand Silhouette)
            painter.setBrush(Qt.NoBrush)
            # Outer subtle dashed target circle
            painter.setPen(QPen(QColor("#cbd5e1"), 1.4, Qt.DashLine))
            painter.drawEllipse(QPointF(0, 0), 38, 38)
            # Inner subtle dotted circle
            painter.setPen(QPen(QColor("#e2e8f0"), 1.2, Qt.DotLine))
            painter.drawEllipse(QPointF(0, 0), 20, 20)
            # Precision center crosshair with gap
            painter.setPen(QPen(QColor("#94a3b8"), 1.2))
            painter.drawLine(-14, 0, -5, 0)
            painter.drawLine(5, 0, 14, 0)
            painter.drawLine(0, -14, 0, -5)
            painter.drawLine(0, 5, 0, 14)
            # Center target pip
            painter.setBrush(QBrush(QColor("#94a3b8")))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(0, 0), 2, 2)

        painter.restore()



# ----------------------------------------------------------------------
# View 1 Right (Option C): Dynamic Surface Material & Autonomous Robot Simulator
# ----------------------------------------------------------------------
class SurfaceClassifierWidget(QWidget):
    surface_selected = pyqtSignal(str, float, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.current_surface = "NONE"
        self.confidence = 0.0
        self.fwhm_mm = 0.0
        self.latency_us = 0
        self.surface_scale = 1.0

        # Action Feedback Banner
        self.action_title = "SURFACE CLASSIFICATION & ROBOT HMI"
        self.action_sub = "Point sensor downward at floor material (10 - 40 cm)"
        self.action_color = "#64748b"
        self.action_alpha = 1.0

        # Autonomous Vehicle Actuator States (3-tier data model)
        self.robot_state = "READY / MONITORING"
        self.mop_val = "STANDBY"
        self.mop_sub = "Awaiting Floor"
        self.suction_val = "STANDBY"
        self.suction_sub = "Awaiting Floor"
        self.cliff_val = "MONITOR"
        self.cliff_sub = "Sensors Active"
        self.surface_traction = "---"
        self.is_cliff_active = False

        # NOTE: For Surface Recognition, the detected surface stays latched
        # indefinitely to give the user ample time to read and inspect. No idle timeout.

        # 60 FPS animation timer
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(16)
        self.anim_timer.timeout.connect(self._update_animation)
        self.anim_timer.start()

    def reset_to_idle(self):
        self.current_surface = "NONE"
        self.confidence = 0.0
        self.fwhm_mm = 0.0
        self.surface_scale = 1.0
        self.action_title = "SURFACE CLASSIFICATION & ROBOT HMI"
        self.action_sub = "Point sensor downward at floor material (10 - 40 cm)"
        self.action_color = "#64748b"
        self.action_alpha = 0.8
        self.robot_state = "READY / MONITORING"
        self.mop_val = "STANDBY"
        self.mop_sub = "Awaiting Floor"
        self.suction_val = "STANDBY"
        self.suction_sub = "Awaiting Floor"
        self.cliff_val = "MONITOR"
        self.cliff_sub = "Sensors Active"
        self.surface_traction = "---"
        self.is_cliff_active = False
        self.update()

    def handle_surface(self, surface_name, conf=0.0, extra="", lat_us=0):
        s = surface_name.upper().strip()
        self.current_surface = s
        self.confidence = conf
        self.latency_us = lat_us
        self.surface_selected.emit(s, conf, str(extra))

        # Parse FWHM from extra if provided
        if isinstance(extra, str) and "FWHM:" in extra:
            try:
                m_f = re.search(r"FWHM:\s*([0-9.]+)", extra)
                self.fwhm_mm = float(m_f.group(1)) if m_f else 0.0
            except Exception:
                self.fwhm_mm = 0.0
        elif isinstance(extra, (int, float)):
            self.fwhm_mm = float(extra)

        # Trigger reactive scale pulse
        self.surface_scale = 1.15

        if s == "HARD_FLOOR":
            self.robot_state = "STANDARD CRUISE (ECO)"
            self.mop_val = "LOWERED"
            self.mop_sub = "Active Mop (0mm)"
            self.suction_val = "1,200 Pa"
            self.suction_sub = "Eco Suction"
            self.cliff_val = "ALL CLEAR"
            self.cliff_sub = "Floor Verified"
            self.surface_traction = "HIGH (Mu = 0.85)"
            self.is_cliff_active = False
            self.action_title = "[HARD FLOOR DETECTED]  CERAMIC TILE / HARDWOOD"
            self.action_sub = "Action: Mop Pad Lowered - Eco Suction Mode - Standard Cruise Active"
            self.action_color = "#0284c7"
            self.action_alpha = 1.0

        elif s == "CARPET":
            self.robot_state = "AUTO-CARPET BOOST ACTIVE"
            self.mop_val = "LIFTED"
            self.mop_sub = "+10mm Retract"
            self.suction_val = "3,000 Pa"
            self.suction_sub = "Max Boost"
            self.cliff_val = "ALL CLEAR"
            self.cliff_sub = "Floor Verified"
            self.surface_traction = "DEEP PILE (Mu = 0.65)"
            self.is_cliff_active = False
            self.action_title = "[CARPET DETECTED]  AUTO-CARPET BOOST ENGAGED"
            self.action_sub = "Action: Mop Pad Retracted +10mm - Turbine 3,000Pa Max Suction Active"
            self.action_color = "#b45309"
            self.action_alpha = 1.0

        elif s == "SPECULAR":
            self.robot_state = "OPTICAL CALIBRATION MODE"
            self.mop_val = "LOWERED"
            self.mop_sub = "Normal Mop"
            self.suction_val = "1,800 Pa"
            self.suction_sub = "Normal Power"
            self.cliff_val = "ALL CLEAR"
            self.cliff_sub = "Mirror Verified"
            self.surface_traction = "LOW SLICK (Mu = 0.40)"
            self.is_cliff_active = False
            self.action_title = "[SPECULAR REFLECTION]  MIRROR / METALLIC"
            self.action_sub = "Action: Optical Receiver Attenuated - Anti-Glint Saturation Shield Active"
            self.action_color = "#7c3aed"
            self.action_alpha = 1.0

        elif s == "VOID":
            self.robot_state = "EMERGENCY BRAKE ENGAGED!"
            self.mop_val = "RETRACTED"
            self.mop_sub = "Pad Protected"
            self.suction_val = "0 Pa"
            self.suction_sub = "Cut Off"
            self.cliff_val = "BRAKE!"
            self.cliff_sub = "Drop-Off Hazard"
            self.surface_traction = "ZERO (AIR GAP / CLIFF)"
            self.is_cliff_active = True
            self.action_title = "[CRITICAL CLIFF DETECTED]  DROP-OFF HAZARD"
            self.action_sub = "Emergency Action: Drive Motors Reversed - Anti-Fall Brake Locked!"
            self.action_color = "#b91c1c"
            self.action_alpha = 1.0

        elif s == "NONE":
            self.reset_to_idle()

        self.update()

    def _update_animation(self):
        self.surface_scale += (1.0 - self.surface_scale) * 0.15
        if self.action_alpha > 0.6:
            self.action_alpha -= 0.005
        self.update()

    def mousePressEvent(self, event):
        w = self.width()
        h = self.height()
        bot_y = h - 42
        if event.y() >= bot_y:
            chip_w = w / 4.0
            idx = int(event.x() / chip_w)
            surfaces = [
                ("HARD_FLOOR", 94.5, "FWHM: 135.0 mm", 6850),
                ("CARPET", 88.2, "FWHM: 215.0 mm", 6920),
                ("SPECULAR", 76.4, "FWHM: 275.0 mm", 7050),
                ("VOID", 91.0, "FWHM: 145.0 mm", 3700)
            ]
            if 0 <= idx < len(surfaces):
                s_name, c_val, f_val, l_val = surfaces[idx]
                self.handle_surface(s_name, c_val, f_val, l_val)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QColor("#ffffff"))
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        # Header Title
        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(14, 23, "SURFACE CLASSIFICATION & ROBOT HMI")

        painter.setPen(QColor("#64748b"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QRect(w - 240, 6, 226, 24), Qt.AlignRight | Qt.AlignVCenter, "AUTONOMOUS SENSING ACTIVE")

        pad_x = 16
        top_y = 36
        bot_y = h - 46
        vp_w = w - 2 * pad_x
        vp_h = bot_y - top_y
        if vp_w <= 120 or vp_h <= 120:
            return

        viewport_rect = QRectF(pad_x, top_y, vp_w, vp_h)
        painter.setBrush(QBrush(QColor("#f8fafc")))
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.drawRoundedRect(viewport_rect, 12, 12)

        # 1. Top Action Feedback Banner
        banner_w = vp_w - 20.0
        banner_h = 42.0
        banner_x = pad_x + 10.0
        banner_y = top_y + 10.0
        banner_rect = QRectF(banner_x, banner_y, banner_w, banner_h)

        b_bg = QColor(255, 255, 255, int(245 * min(1.0, self.action_alpha)))
        b_pen = QColor(self.action_color)
        b_pen.setAlpha(int(220 * min(1.0, self.action_alpha)))

        painter.setBrush(QBrush(b_bg))
        painter.setPen(QPen(b_pen, 1.5))
        painter.drawRoundedRect(banner_rect, 8, 8)

        painter.setPen(b_pen)
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(QRectF(banner_x, banner_y + 3, banner_w, 20), Qt.AlignCenter, self.action_title)

        painter.setPen(QColor("#475569"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QRectF(banner_x, banner_y + 22, banner_w, 16), Qt.AlignCenter, self.action_sub)

        # 2. Main Content Split: Card A (Material Artwork) & Card B (Robot Actuators HMI)
        content_top = banner_y + banner_h + 10.0
        content_h = vp_h - (content_top - top_y) - 10.0
        col_gap = 12.0
        col_w = (vp_w - 20.0 - col_gap) / 2.0

        # --- CARD A: Live Surface Material Avatar ---
        card_a_rect = QRectF(pad_x + 10.0, content_top, col_w, content_h)
        painter.setBrush(QBrush(QColor("#ffffff")))
        surf_color = SURFACE_INFO.get(self.current_surface, ("#64748b", ""))[0]
        painter.setPen(QPen(QColor(surf_color if self.current_surface != "NONE" else "#cbd5e1"), 1.5 if self.current_surface != "NONE" else 1.0))
        painter.drawRoundedRect(card_a_rect, 10, 10)

        # Card A Title
        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(int(card_a_rect.x() + 12), int(card_a_rect.y() + 22), "DETECTED SURFACE MATERIAL")

        # Available avatar bounds inside Card A
        usable_top = card_a_rect.y() + 32.0
        usable_bot = card_a_rect.bottom() - 52.0
        usable_h = max(100.0, usable_bot - usable_top)
        usable_w = max(100.0, card_a_rect.width() - 20.0)

        avatar_cx = card_a_rect.center().x()
        avatar_cy = usable_top + usable_h / 2.0

        auto_scale = min(usable_w / 80.0, usable_h / 85.0) * 0.72
        auto_scale = max(1.6, min(2.5, auto_scale))
        draw_scale = auto_scale * self.surface_scale

        # Draw Scalable Vector Surface Art
        self._draw_surface_vector(painter, avatar_cx, avatar_cy, self.current_surface, surf_color, draw_scale)

        # Surface Name & Telemetry Readout
        display_name = SURFACE_INFO.get(self.current_surface, ("#64748b", "AWAITING SURFACE TARGET"))[1]
        painter.setPen(QColor(surf_color if self.current_surface != "NONE" else "#94a3b8"))
        painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
        painter.drawText(QRectF(card_a_rect.x(), card_a_rect.bottom() - 48, card_a_rect.width(), 22), Qt.AlignCenter, display_name)

        painter.setPen(QColor("#64748b"))
        painter.setFont(QFont("Segoe UI", 8))
        if self.current_surface != "NONE":
            telemetry_str = f"Confidence: {self.confidence:.1f}% | FWHM: {self.fwhm_mm:.1f} mm | Latency: {self.latency_us:,} us"
        else:
            telemetry_str = "Awaiting floor surface reflection"
        painter.drawText(QRectF(card_a_rect.x(), card_a_rect.bottom() - 26, card_a_rect.width(), 18), Qt.AlignCenter, telemetry_str)

        # --- CARD B: Autonomous Robot Actuators & Drive HMI ---
        card_b_rect = QRectF(pad_x + 10.0 + col_w + col_gap, content_top, col_w, content_h)
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.setPen(QPen(QColor("#cbd5e1"), 1.0))
        painter.drawRoundedRect(card_b_rect, 10, 10)

        # Card B Title
        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(int(card_b_rect.x() + 12), int(card_b_rect.y() + 22), "AUTONOMOUS VEHICLE CONTROLLER")

        # Robot Mode Banner Header
        r_banner_rect = QRectF(card_b_rect.x() + 12, card_b_rect.y() + 32, card_b_rect.width() - 24, 46)
        painter.setBrush(QBrush(QColor("#f1f5f9")))
        painter.setPen(QPen(QColor("#e2e8f0"), 1))
        painter.drawRoundedRect(r_banner_rect, 6, 6)

        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(int(r_banner_rect.x() + 10), int(r_banner_rect.y() + 18), "ROBOT DRIVE STATE:")

        st_color = QColor(surf_color if self.current_surface != "NONE" else "#475569")
        painter.setPen(st_color)
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(int(r_banner_rect.x() + 10), int(r_banner_rect.y() + 36), self.robot_state)

        # Actuators Status Grid: 3 Non-Overlapping 3-Tier Badges
        b_top_y = r_banner_rect.bottom() + 10
        badge_w = (card_b_rect.width() - 24 - 16) / 3.0
        badge_h = 64.0

        # Status 1: Mop Pad Height
        b1_rect = QRectF(card_b_rect.x() + 12, b_top_y, badge_w, badge_h)
        mop_is_lifted = "LIFT" in self.mop_val
        b1_bg = QColor("#fef3c7") if mop_is_lifted else QColor("#e0f2fe")
        b1_fg = QColor("#b45309") if mop_is_lifted else QColor("#0284c7")
        painter.setBrush(QBrush(b1_bg))
        painter.setPen(QPen(b1_fg, 1))
        painter.drawRoundedRect(b1_rect, 5, 5)

        painter.setPen(QColor("#475569"))
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.drawText(QRectF(b1_rect.x(), b1_rect.y() + 6, b1_rect.width(), 14), Qt.AlignCenter, "MOP PAD")
        painter.setPen(b1_fg)
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(QRectF(b1_rect.x() + 2, b1_rect.y() + 22, b1_rect.width() - 4, 18), Qt.AlignCenter, self.mop_val)
        painter.setPen(QColor("#64748b"))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(QRectF(b1_rect.x() + 2, b1_rect.y() + 42, b1_rect.width() - 4, 14), Qt.AlignCenter, self.mop_sub)

        # Status 2: Suction Turbine Power
        b2_rect = QRectF(card_b_rect.x() + 12 + badge_w + 8, b_top_y, badge_w, badge_h)
        is_turbo = "3,000" in self.suction_val or "BOOST" in self.suction_val
        is_cutoff = "0 Pa" in self.suction_val
        b2_bg = QColor("#fee2e2") if is_cutoff else (QColor("#fef3c7") if is_turbo else QColor("#f1f5f9"))
        b2_fg = QColor("#b91c1c") if is_cutoff else (QColor("#b45309") if is_turbo else QColor("#1e293b"))
        painter.setBrush(QBrush(b2_bg))
        painter.setPen(QPen(b2_fg, 1))
        painter.drawRoundedRect(b2_rect, 5, 5)

        painter.setPen(QColor("#475569"))
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.drawText(QRectF(b2_rect.x(), b2_rect.y() + 6, b2_rect.width(), 14), Qt.AlignCenter, "SUCTION")
        painter.setPen(b2_fg)
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(QRectF(b2_rect.x() + 2, b2_rect.y() + 22, b2_rect.width() - 4, 18), Qt.AlignCenter, self.suction_val)
        painter.setPen(QColor("#64748b"))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(QRectF(b2_rect.x() + 2, b2_rect.y() + 42, b2_rect.width() - 4, 14), Qt.AlignCenter, self.suction_sub)

        # Status 3: Cliff Hazard Interlock
        b3_rect = QRectF(card_b_rect.x() + 12 + 2 * (badge_w + 8), b_top_y, badge_w, badge_h)
        b3_bg = QColor("#fee2e2") if self.is_cliff_active else QColor("#dcfce7")
        b3_fg = QColor("#b91c1c") if self.is_cliff_active else QColor("#15803d")
        painter.setBrush(QBrush(b3_bg))
        painter.setPen(QPen(b3_fg, 1))
        painter.drawRoundedRect(b3_rect, 5, 5)

        painter.setPen(QColor("#475569"))
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.drawText(QRectF(b3_rect.x(), b3_rect.y() + 6, b3_rect.width(), 14), Qt.AlignCenter, "CLIFF SAFETY")
        painter.setPen(b3_fg)
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(QRectF(b3_rect.x() + 2, b3_rect.y() + 22, b3_rect.width() - 4, 18), Qt.AlignCenter, self.cliff_val)
        painter.setPen(QColor("#64748b"))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(QRectF(b3_rect.x() + 2, b3_rect.y() + 42, b3_rect.width() - 4, 14), Qt.AlignCenter, self.cliff_sub)

        # Bottom Sub-metrics Row (Clean 2-row table layout, zero overlapping)
        sub_box_y = b_top_y + badge_h + 10
        sub_box_h = max(46.0, card_b_rect.bottom() - sub_box_y - 10)
        sub_rect = QRectF(card_b_rect.x() + 12, sub_box_y, card_b_rect.width() - 24, sub_box_h)
        painter.setBrush(QBrush(QColor("#f8fafc")))
        painter.setPen(QPen(QColor("#e2e8f0"), 1))
        painter.drawRoundedRect(sub_rect, 6, 6)

        row_h = sub_box_h / 2.0
        r1 = QRectF(sub_rect.x() + 10, sub_rect.y(), sub_rect.width() - 20, row_h)
        r2 = QRectF(sub_rect.x() + 10, sub_rect.y() + row_h, sub_rect.width() - 20, row_h)

        painter.setPen(QColor("#475569"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(r1, Qt.AlignLeft | Qt.AlignVCenter, "Surface Traction:")
        painter.drawText(r2, Qt.AlignLeft | Qt.AlignVCenter, "Optical Pulse Width (FWHM):")

        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(r1, Qt.AlignRight | Qt.AlignVCenter, self.surface_traction)
        fwhm_str = f"{self.fwhm_mm:.1f} mm" if self.fwhm_mm > 0 else "---"
        painter.drawText(r2, Qt.AlignRight | Qt.AlignVCenter, fwhm_str)

        # Critical Hazard Emergency Shield Overlay
        if self.is_cliff_active:
            shield_rect = QRectF(card_b_rect.x() + 8, card_b_rect.y() + 30, card_b_rect.width() - 16, card_b_rect.height() - 38)
            painter.setBrush(QBrush(QColor(254, 226, 226, 235)))
            painter.setPen(QPen(QColor("#b91c1c"), 2.0))
            painter.drawRoundedRect(shield_rect, 8, 8)

            painter.setPen(QColor("#b91c1c"))
            painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
            painter.drawText(shield_rect, Qt.AlignCenter, "CRITICAL CLIFF DETECTED!\nEmergency Drive Brakes Locked")

        # 3. Bottom Interactive Surface Selector Chips
        surfaces = [
            ("Hard Floor (Tile)", "#0284c7", "HARD_FLOOR"),
            ("Carpet (Rug)", "#b45309", "CARPET"),
            ("Specular (Mirror)", "#7c3aed", "SPECULAR"),
            ("Void (Drop-Off)", "#b91c1c", "VOID")
        ]
        chip_gap = 8.0
        chip_w = (w - 2 * pad_x - (len(surfaces) - 1) * chip_gap) / float(len(surfaces))
        chip_h = 28.0
        chip_y = h - 36.0

        for ci, (s_name, s_col, s_code) in enumerate(surfaces):
            cx = pad_x + ci * (chip_w + chip_gap)
            c_rect = QRectF(cx, chip_y, chip_w, chip_h)
            is_active_chip = (self.current_surface == s_code)

            if is_active_chip:
                painter.setBrush(QBrush(QColor(s_col)))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(c_rect, 5, 5)
                painter.setPen(QColor("#ffffff"))
            else:
                painter.setBrush(QBrush(QColor("#ffffff")))
                painter.setPen(QPen(QColor("#cbd5e1"), 1))
                painter.drawRoundedRect(c_rect, 5, 5)
                painter.setPen(QColor("#475569"))

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold if is_active_chip else QFont.Normal))
            painter.drawText(c_rect, Qt.AlignCenter, s_name)

    def _draw_surface_vector(self, painter, cx, cy, surface, color, scale):
        """Draws clean vector stylized surface materials without emojis."""
        painter.save()
        painter.translate(cx, cy)
        painter.scale(scale, scale)

        pen = QPen(QColor(color), 2.2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(color).lighter(185)))

        if surface == "HARD_FLOOR":
            # 2x2 Ceramic Floor Tile Grid with Reflective Floor Shine
            path = QPainterPath()
            path.addRoundedRect(QRectF(-34, -34, 32, 32), 3, 3)
            path.addRoundedRect(QRectF(2, -34, 32, 32), 3, 3)
            path.addRoundedRect(QRectF(-34, 2, 32, 32), 3, 3)
            path.addRoundedRect(QRectF(2, 2, 32, 32), 3, 3)
            painter.drawPath(path)

            # Reflective Floor Glint Shine Lines
            painter.setPen(QPen(QColor("#ffffff"), 1.8))
            painter.drawLine(-22, -26, -10, -26)
            painter.drawLine(14, -26, 26, -26)
            painter.drawLine(-22, 10, -10, 10)
            painter.drawLine(14, 10, 26, 10)

            # Top shine spark
            painter.setPen(QPen(QColor(color), 1.5))
            painter.drawLine(24, -40, 24, -48)
            painter.drawLine(20, -44, 28, -44)

        elif surface == "CARPET":
            # Woven Carpet Rug Base
            path = QPainterPath()
            path.addRoundedRect(QRectF(-36, -8, 72, 42), 6, 6)
            painter.drawPath(path)

            # Carpet Weave Crosshatch Texture
            painter.setPen(QPen(QColor(color).lighter(130), 1.2))
            for xi in range(-28, 30, 8):
                painter.drawLine(xi, -6, xi + 6, 32)
                painter.drawLine(xi + 6, -6, xi, 32)

            # Upward Airflow Suction Vortex Arrows (Auto-Carpet Boost)
            painter.setPen(QPen(QColor(color), 2.0))
            painter.drawLine(-18, -12, 0, -38)
            painter.drawLine(18, -12, 0, -38)
            painter.drawLine(0, -14, 0, -46)
            # Arrowhead
            painter.drawLine(-6, -40, 0, -46)
            painter.drawLine(6, -40, 0, -46)

        elif surface == "SPECULAR":
            # Polished Specular Mirror Substrate
            mirror_rect = QRectF(-36, 12, 72, 16)
            painter.setBrush(QBrush(QColor("#f3e8ff")))
            painter.drawRoundedRect(mirror_rect, 3, 3)

            # Mirror Back Hatching
            painter.setPen(QPen(QColor("#cbd5e1"), 1.0))
            for hx in range(-32, 34, 8):
                painter.drawLine(hx, 28, hx - 5, 34)

            # Incident Laser Ray from sensor
            painter.setPen(QPen(QColor(color), 2.4))
            painter.drawLine(-26, -34, 0, 12)
            # Reflected Laser Ray bouncing off mirror
            painter.drawLine(0, 12, 26, -34)

            # Normal Dashed Axis
            painter.setPen(QPen(QColor("#94a3b8"), 1.0, Qt.DashLine))
            painter.drawLine(0, -36, 0, 12)

            # Specular Starburst Reflection Point
            painter.setPen(QPen(QColor("#ffffff"), 2.0))
            painter.drawLine(-6, 12, 6, 12)
            painter.drawLine(0, 6, 0, 18)
            painter.drawLine(-4, 8, 4, 16)
            painter.drawLine(-4, 16, 4, 8)

        elif surface == "VOID":
            # Staircase Drop-Off / Cliff Ledge
            path = QPainterPath()
            path.moveTo(-36, -4)
            path.lineTo(-2, -4)
            path.lineTo(-2, 36)
            path.lineTo(-36, 36)
            path.closeSubpath()
            painter.drawPath(path)

            # Ledge Hazard Stripes
            painter.setPen(QPen(QColor("#fca5a5"), 1.5))
            painter.drawLine(-28, -2, -18, 12)
            painter.drawLine(-18, -2, -8, 12)

            # Void Air Gap & Downward Drop Arrow
            painter.setPen(QPen(QColor(color), 2.4))
            painter.drawLine(18, -6, 18, 32)
            painter.drawLine(12, 26, 18, 32)
            painter.drawLine(24, 26, 18, 32)

            # Cliff Warning Triangle
            tri = QPolygonF([QPointF(-2, -40), QPointF(-16, -14), QPointF(12, -14)])
            painter.setBrush(QBrush(QColor("#fee2e2")))
            painter.setPen(QPen(QColor(color), 2.0))
            painter.drawPolygon(tri)
            # Exclamation mark
            painter.setPen(QPen(QColor(color), 2.2))
            painter.drawLine(-2, -32, -2, -22)
            painter.drawPoint(-2, -18)

        else:
            # Minimalist Downward ToF Sensor Cone & Floor Target Reticle
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#cbd5e1"), 1.4, Qt.DashLine))
            painter.drawEllipse(QPointF(0, 0), 38, 38)
            painter.setPen(QPen(QColor("#e2e8f0"), 1.2, Qt.DotLine))
            painter.drawEllipse(QPointF(0, 0), 20, 20)
            painter.setPen(QPen(QColor("#94a3b8"), 1.2))
            painter.drawLine(-14, 0, -5, 0)
            painter.drawLine(5, 0, 14, 0)
            painter.drawLine(0, -14, 0, -5)
            painter.drawLine(0, 5, 0, 14)
            painter.setBrush(QBrush(QColor("#94a3b8")))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(0, 0), 2, 2)

        painter.restore()

# ----------------------------------------------------------------------
# View 2: Focused 48-Bin Photon Histogram & Pulse Analysis (Light Theme)
# ----------------------------------------------------------------------
class FocusedHistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_zone = 5
        self.ambient = 0.0
        self.bins = [0.0] * 48
        self.distance = 0.0
        self.temp_c = 0
        self.frame_id = 0
        self.setMinimumHeight(380)

    def set_data(self, zid, ambient, bins, distance, temp_c, fid):
        self.selected_zone = zid
        self.ambient = ambient
        self.bins = bins
        self.distance = distance
        self.temp_c = temp_c
        self.frame_id = fid
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QColor("#ffffff"))
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        # Header Info Banner
        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
        painter.drawText(16, 26, f"OPTICAL RETURN PULSE ANALYSIS - ZONE {self.selected_zone}")

        has_signal = bool(self.bins and any(b > 0 for b in self.bins))
        if has_signal:
            peak_idx = int(np.argmax(self.bins))
            max_val = max(1.0, float(self.bins[peak_idx]))
        else:
            peak_idx = 0
            max_val = 1.0

        # Compute FWHM (Full Width at Half Maximum)
        half_max = max_val / 2.0
        left_idx = peak_idx
        while left_idx > 0 and self.bins[left_idx] > half_max:
            left_idx -= 1
        right_idx = peak_idx
        while right_idx < len(self.bins) - 1 and self.bins[right_idx] > half_max:
            right_idx += 1
        fwhm_bins = max(1, right_idx - left_idx)
        bin_width_mm = 37.5
        fwhm_mm = fwhm_bins * bin_width_mm if has_signal else 0.0

        # Sub-stats
        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor("#64748b"))
        if has_signal:
            stats_str = f"Peak Amplitude: {max_val:,.0f} counts  |  Peak Bin: {peak_idx}  |  Pulse Width (FWHM): {fwhm_mm:.1f} mm  |  Ambient Baseline: {self.ambient:.1f} kcps"
        else:
            stats_str = f"Waiting for CNH histogram data from sensor...  |  Ambient Baseline: {self.ambient:.1f} kcps"
        painter.drawText(16, 46, stats_str)

        margin_l = 60
        margin_r = 30
        margin_t = 65
        margin_b = 40
        plot_w = w - margin_l - margin_r
        plot_h = h - margin_t - margin_b

        if plot_w <= 0 or plot_h <= 0:
            return

        # Draw Grid & Baseline
        painter.setPen(QPen(QColor("#f1f5f9"), 1))
        for y_tick in range(5):
            y_pos = margin_t + (y_tick / 4.0) * plot_h
            painter.drawLine(margin_l, int(y_pos), margin_l + plot_w, int(y_pos))
            val = max_val * (1.0 - y_tick / 4.0)
            painter.setPen(QColor("#94a3b8"))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(10, int(y_pos + 4), f"{val:5.0f}")
            painter.setPen(QPen(QColor("#f1f5f9"), 1))

        if self.ambient > 0 and max_val > 0:
            amb_y = margin_t + plot_h - (min(self.ambient, max_val) / max_val) * plot_h
            painter.setPen(QPen(QColor("#f59e0b"), 1, Qt.DashLine))
            painter.drawLine(margin_l, int(amb_y), margin_l + plot_w, int(amb_y))
            painter.setPen(QColor("#b45309"))
            painter.drawText(margin_l + plot_w - 110, int(amb_y - 4), "Ambient Noise")

        num_bins = len(self.bins)
        bar_w = plot_w / float(num_bins)

        for i, val in enumerate(self.bins):
            bar_h = (val / max_val) * plot_h
            bx = margin_l + i * bar_w
            by = margin_t + plot_h - bar_h

            if i == peak_idx:
                bar_col = QColor("#2563eb")
            elif left_idx <= i <= right_idx:
                bar_col = QColor("#0284c7")
            else:
                bar_col = QColor("#94a3b8")

            painter.setBrush(QBrush(bar_col))
            painter.setPen(Qt.NoPen)
            painter.drawRect(QRectF(bx + 1, by, max(1.0, bar_w - 2), bar_h))

            if i % 4 == 0:
                painter.setPen(QColor("#64748b"))
                painter.setFont(QFont("Segoe UI", 8))
                painter.drawText(int(bx - 4), margin_t + plot_h + 16, f"{i}")

        painter.setPen(QColor("#475569"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(int(margin_l + plot_w / 2.0 - 60), margin_t + plot_h + 34, "CNH Histogram Bins (48 Bins)")


# ----------------------------------------------------------------------
# View 3: Ambient Light Heatmap (Light Theme)
# ----------------------------------------------------------------------
class AmbientHeatmapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ambients = [0.0] * 64
        self.dim = 8
        self.setMinimumHeight(380)

    def set_data(self, amb_list):
        self.ambients = amb_list
        self.dim = 8 if len(amb_list) >= 64 else 4
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QColor("#ffffff"))
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
        painter.drawText(16, 26, "AMBIENT PHOTON FLUX & SOLAR GLINT HEATMAP")

        margin = 20
        top_offset = 45
        grid_w = w - 2 * margin
        grid_h = h - top_offset - margin
        if grid_w <= 0 or grid_h <= 0:
            return

        cell_w = grid_w / float(self.dim)
        cell_h = grid_h / float(self.dim)
        pad = 4

        max_amb = max(self.ambients) if self.ambients and max(self.ambients) > 0 else 100.0

        for r in range(self.dim):
            for c in range(self.dim):
                zid = r * self.dim + c
                amb = self.ambients[zid] if zid < len(self.ambients) else 0.0

                cell_x = margin + c * cell_w + pad / 2.0
                cell_y = top_offset + r * cell_h + pad / 2.0
                cell_bw = cell_w - pad
                cell_bh = cell_h - pad

                ratio = min(1.0, amb / max(1.0, max_amb))
                r_c = int(254 - ratio * 40)
                g_c = int(240 - ratio * 150)
                b_c = int(138 - ratio * 120)
                bg = QColor(max(0, r_c), max(0, g_c), max(0, b_c))

                painter.setBrush(QBrush(bg))
                painter.setPen(QPen(QColor("#cbd5e1"), 1))
                painter.drawRoundedRect(QRectF(cell_x, cell_y, cell_bw, cell_bh), 4, 4)

                painter.setPen(QColor("#0f172a"))
                painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
                painter.drawText(QRectF(cell_x, cell_y, cell_bw, cell_bh), Qt.AlignCenter, f"{amb:.0f}\nkcps")


# ----------------------------------------------------------------------
# View 4: Live AT Command Terminal & Hardware Diagnostics (Light Theme)
# ----------------------------------------------------------------------
class AtTerminalWidget(QWidget):
    send_cmd_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Header Info
        hdr_row = QHBoxLayout()
        lbl_title = QLabel("INTERACTIVE AT COMMAND CONSOLE & HARDWARE DIAGNOSTICS")
        lbl_title.setStyleSheet("font-weight: bold; color: #0f172a; font-size: 12px; border: none; background: transparent;")
        hdr_row.addWidget(lbl_title)
        hdr_row.addStretch()

        self.chk_raw = QCheckBox("Show Raw Telemetry")
        self.chk_raw.setChecked(False)
        self.chk_raw.setStyleSheet("color: #64748b; font-size: 11px; margin-right: 8px;")
        hdr_row.addWidget(self.chk_raw)

        btn_clear = QPushButton("Clear Console")
        btn_clear.clicked.connect(self.clear_console)
        hdr_row.addWidget(btn_clear)
        layout.addLayout(hdr_row)

        # High-Performance Plain Text Console
        self.txt_console = QPlainTextEdit()
        self.txt_console.setReadOnly(True)
        self.txt_console.setMaximumBlockCount(500)
        self.txt_console.setStyleSheet("""
            QPlainTextEdit {
                background-color: #ffffff;
                color: #1e293b;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        layout.addWidget(self.txt_console, 1)

        # Command Input Line
        cmd_row = QHBoxLayout()
        self.txt_input = QLineEdit()
        self.txt_input.setPlaceholderText("Type AT command (e.g. AT+STATUS?, AT+MODE=GESTURE, AT+STREAM=RAW) and press Enter...")
        self.txt_input.setStyleSheet("""
            QLineEdit {
                background-color: #ffffff;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 6px 12px;
                font-family: 'Consolas', monospace;
                font-size: 12px;
            }
        """)
        self.txt_input.returnPressed.connect(self.on_send)

        btn_send = QPushButton("Send")
        btn_send.setStyleSheet("background-color: #2563eb; color: #ffffff; font-weight: bold; border-radius: 4px; padding: 6px 16px; border: none;")
        btn_send.clicked.connect(self.on_send)

        cmd_row.addWidget(self.txt_input, 1)
        cmd_row.addWidget(btn_send)
        layout.addLayout(cmd_row)

    def append_line(self, line):
        if not self.chk_raw.isChecked():
            if (line.startswith("{") and line.endswith("}")) or line.startswith("CNH,"):
                return

        if "[AI" in line:
            self.txt_console.appendHtml(f'<span style="color: #15803d; font-weight: bold;">{line}</span>')
        elif "[ERR" in line or "ERROR" in line:
            self.txt_console.appendHtml(f'<span style="color: #dc2626; font-weight: bold;">{line}</span>')
        elif "[PIPELINE]" in line:
            self.txt_console.appendHtml(f'<span style="color: #b45309; font-weight: bold;">{line}</span>')
        elif ">>>" in line:
            self.txt_console.appendHtml(f'<span style="color: #2563eb; font-weight: bold;">{line}</span>')
        else:
            self.txt_console.appendHtml(f'<span style="color: #475569;">{line}</span>')

    def clear_console(self):
        self.txt_console.clear()

    def on_send(self):
        text = self.txt_input.text().strip()
        if text:
            self.append_line(f">>> {text}")
            self.send_cmd_requested.emit(text)
            self.txt_input.clear()


# ----------------------------------------------------------------------
# Main Application Window (Light Theme Architecture)
# ----------------------------------------------------------------------
class EdgeSenseMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EdgeSense - Real-Time 3D Time-of-Flight and On-Chip Edge AI Showcase Visualizer")
        self.resize(1280, 840)
        self.setStyleSheet("""
            QMainWindow { background-color: #f8fafc; }
            QLabel { color: #0f172a; font-family: 'Segoe UI'; border: none; background: transparent; }
            QTabWidget::pane { border: 1px solid #cbd5e1; background-color: #ffffff; border-radius: 6px; }
            QTabBar::tab {
                background: #f1f5f9;
                color: #475569;
                font-weight: bold;
                padding: 8px 24px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                border: 1px solid #cbd5e1;
                border-bottom: none;
                margin-right: 8px;
                font-size: 11px;
            }
            QTabBar::tab:selected { background: #ffffff; color: #0f172a; border-top: 2px solid #2563eb; }
            QPushButton {
                background-color: #ffffff;
                color: #1e293b;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #f1f5f9; border-color: #2563eb; }
            QComboBox {
                background-color: #ffffff;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 5px 12px;
            }
        """)

        self.worker = None
        self.selected_zone = 5
        self.current_frame = None
        self.last_fps_time = time.time()
        self.fps_frames = 0
        self.current_fps = 0.0
        self.current_mode = "GESTURE"

        self.init_ui()
        self.auto_connect_stm32()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(10)

        # 1. Top Control Bar: Connection, Mode Switching, Telemetry
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        lbl_port = QLabel("Port:")
        lbl_port.setStyleSheet("font-weight: bold; color: #475569; border: none; background: transparent;")
        self.cmb_ports = QComboBox()
        self.cmb_ports.setMinimumWidth(160)
        self.refresh_ports()

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh_ports)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setStyleSheet("background-color: #16a34a; color: #ffffff; font-weight: bold;")
        self.btn_connect.clicked.connect(self.toggle_connection)

        top_bar.addWidget(lbl_port)
        top_bar.addWidget(self.cmb_ports)
        top_bar.addWidget(btn_refresh)
        top_bar.addWidget(self.btn_connect)
        top_bar.addSpacing(16)

        # Mode Switching Group (Only Gesture, Surface, Posture)
        lbl_mode = QLabel("Hardware Pipeline:")
        lbl_mode.setStyleSheet("font-weight: bold; color: #475569; border: none; background: transparent;")
        top_bar.addWidget(lbl_mode)

        self.btn_mode_gesture = QPushButton("Gesture (8x8)")
        self.btn_mode_gesture.setCheckable(True)
        self.btn_mode_gesture.setChecked(True)
        self.btn_mode_gesture.clicked.connect(lambda: self.switch_pipeline_mode("GESTURE"))

        self.btn_mode_surface = QPushButton("Surface CNH (4x4)")
        self.btn_mode_surface.setCheckable(True)
        self.btn_mode_surface.clicked.connect(lambda: self.switch_pipeline_mode("SURFACE"))

        self.btn_mode_posture = QPushButton("Posture (8x8)")
        self.btn_mode_posture.setCheckable(True)
        self.btn_mode_posture.clicked.connect(lambda: self.switch_pipeline_mode("POSTURE"))

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.btn_mode_gesture, 0)
        self.mode_group.addButton(self.btn_mode_surface, 1)
        self.mode_group.addButton(self.btn_mode_posture, 2)

        self.update_mode_button_styles()

        top_bar.addWidget(self.btn_mode_gesture)
        top_bar.addWidget(self.btn_mode_surface)
        top_bar.addWidget(self.btn_mode_posture)

        top_bar.addStretch()

        # Telemetry HUD Badges
        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_fps.setStyleSheet("background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 4px 8px; font-weight: bold; color: #2563eb; font-size: 11px;")
        self.lbl_temp = QLabel("Temp: -- deg C")
        self.lbl_temp.setStyleSheet("background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 4px 8px; font-weight: bold; color: #7c3aed; font-size: 11px;")
        self.lbl_frame = QLabel("Frame: 0")
        self.lbl_frame.setStyleSheet("background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 4px 8px; font-weight: bold; color: #475569; font-size: 11px;")

        top_bar.addWidget(self.lbl_fps)
        top_bar.addWidget(self.lbl_temp)
        top_bar.addWidget(self.lbl_frame)

        main_layout.addLayout(top_bar)

        # 2. Hero Live On-Chip Edge AI Output HUD
        self.hero_ai_widget = HeroAiOutputWidget()
        main_layout.addWidget(self.hero_ai_widget)

        # 3. Main Multi-View Showcase Tabs
        self.tabs = QTabWidget()
        self.tabs.setElideMode(Qt.ElideNone)

        # Tab 1: Split Screen (50% Spatial Depth Matrix | 50% Mode-Adaptive Interaction Showcase)
        tab1_container = QWidget()
        tab1_layout = QHBoxLayout(tab1_container)
        tab1_layout.setContentsMargins(8, 8, 8, 8)
        tab1_layout.setSpacing(12)

        # Left 50%: Depth Matrix
        self.view_depth = DepthMatrixWidget()
        self.view_depth.zone_clicked.connect(self.on_zone_selected)
        tab1_layout.addWidget(self.view_depth, 1)

        # Right 50%: Stacked Interaction Showcase (Switches between Gesture and Posture)
        self.sim_stack = QStackedWidget()
        self.gesture_sim = TouchlessGestureWidget()
        self.posture_sim = TouchlessPostureWidget()
        self.surface_sim = SurfaceClassifierWidget()
        self.surface_sim.surface_selected.connect(lambda s, conf, extra: self.view_depth.set_surface_preset(s, extra))
        self.mobile_sim = self.gesture_sim  # Backward-compatibility alias

        self.sim_stack.addWidget(self.gesture_sim)  # Index 0: Gesture
        self.sim_stack.addWidget(self.posture_sim)  # Index 1: Posture
        self.sim_stack.addWidget(self.surface_sim)  # Index 2: Surface
        tab1_layout.addWidget(self.sim_stack, 1)

        self.tabs.addTab(tab1_container, "Depth Matrix and Touchless Interaction")

        # Tab 2: CNH Photon Histogram & Pulse Analysis
        self.view_hist = FocusedHistogramWidget()
        self.tabs.addTab(self.view_hist, "CNH Pulse Analysis")

        # Tab 3: Ambient Photon Flux & Solar Glint Heatmap
        self.view_ambient = AmbientHeatmapWidget()
        self.tabs.addTab(self.view_ambient, "Ambient Heatmap")

        # Tab 4: Interactive AT Terminal & Diagnostics
        self.view_terminal = AtTerminalWidget()
        self.view_terminal.send_cmd_requested.connect(self.send_at_command)
        self.tabs.addTab(self.view_terminal, "AT Console")

        main_layout.addWidget(self.tabs, 1)

        # 4. Status Bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("color: #64748b; font-size: 11px;")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Select COM port to connect.")

    def update_mode_button_styles(self):
        active_style = "background-color: #2563eb; color: #ffffff; font-weight: bold; border: 1px solid #1d4ed8; border-radius: 6px; padding: 6px 14px;"
        inactive_style = "background-color: #ffffff; color: #1e293b; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 14px;"

        self.btn_mode_gesture.setStyleSheet(active_style if self.btn_mode_gesture.isChecked() else inactive_style)
        self.btn_mode_surface.setStyleSheet(active_style if self.btn_mode_surface.isChecked() else inactive_style)
        self.btn_mode_posture.setStyleSheet(active_style if self.btn_mode_posture.isChecked() else inactive_style)

    def refresh_ports(self):
        self.cmb_ports.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self.cmb_ports.addItem(f"{p.device} - {p.description}", p.device)
        if not ports:
            self.cmb_ports.addItem("No COM ports found", "")

    def auto_connect_stm32(self):
        ports = serial.tools.list_ports.comports()
        for p in ports:
            desc = (p.description or "").lower()
            mfg = (p.manufacturer or "").lower()
            if "stmicro" in desc or "stlink" in desc or "virtual com" in desc or "stm32" in desc or "stmicro" in mfg:
                idx = self.cmb_ports.findData(p.device)
                if idx >= 0:
                    self.cmb_ports.setCurrentIndex(idx)
                    self.toggle_connection()
                    break

    def toggle_connection(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker = None
            self.btn_connect.setText("Connect")
            self.btn_connect.setStyleSheet("background-color: #16a34a; color: #ffffff; font-weight: bold;")
            self.status_bar.showMessage("Disconnected from device.")
        else:
            port = self.cmb_ports.currentData()
            if not port:
                QMessageBox.warning(self, "No Port", "Please select a valid COM port.")
                return

            self.worker = SerialWorker(port)
            self.worker.sig_frame_data.connect(self.on_frame_data)
            self.worker.sig_ai_event.connect(self.on_ai_event)
            self.worker.sig_status.connect(self.on_serial_status)
            self.worker.sig_raw_line.connect(self.view_terminal.append_line)
            self.worker.sig_mode_changed.connect(self.on_mcu_mode_switched)
            self.worker.start()

            self.btn_connect.setText("Disconnect")
            self.btn_connect.setStyleSheet("background-color: #dc2626; color: #ffffff; font-weight: bold;")

    def switch_pipeline_mode(self, mode_name):
        self.current_mode = mode_name
        btn_map = {
            "GESTURE": self.btn_mode_gesture,
            "SURFACE": self.btn_mode_surface,
            "POSTURE": self.btn_mode_posture
        }
        if mode_name in btn_map:
            btn_map[mode_name].setChecked(True)
        self.update_mode_button_styles()

        # Update right panel simulation stack and depth matrix mode
        self.view_depth.set_mode(mode_name)
        if mode_name == "GESTURE":
            self.sim_stack.setCurrentIndex(0)
        elif mode_name == "POSTURE":
            self.sim_stack.setCurrentIndex(1)
            self.posture_sim.reset_to_idle()
        elif mode_name == "SURFACE":
            self.sim_stack.setCurrentIndex(2)
            self.surface_sim.reset_to_idle()

        self.hero_ai_widget.set_active_mode(mode_name)

        if self.worker:
            self.worker.send_command(f"AT+MODE={mode_name}")
            time.sleep(0.04)
            self.worker.send_command("AT+STREAM=RAW")
            self.status_bar.showMessage(f"Switching MCU pipeline mode to {mode_name}...")

    def send_at_command(self, cmd_str):
        if self.worker:
            self.worker.send_command(cmd_str)
        else:
            self.status_bar.showMessage("Cannot send command: not connected to device.")

    def on_zone_selected(self, zid):
        self.selected_zone = zid
        self.view_depth.set_selected_zone(zid)
        if self.current_frame and "cnh" in self.current_frame and self.current_frame["cnh"]:
            cnh_entry = self.current_frame["cnh"].get(zid, None)
            if cnh_entry is not None:
                amb, bins = cnh_entry
                dists = self.current_frame.get("distances", [])
                zdist = dists[zid] if zid < len(dists) else -1
                self.view_hist.set_data(zid, amb, bins, zdist, self.current_frame.get("temp_c", 0), self.current_frame.get("frame_id", 0))
                self.view_depth.set_cnh_data(zid, amb, bins)

    def on_frame_data(self, frame):
        self.current_frame = frame
        dists = frame.get("distances", [])
        ambs = frame.get("ambients", [])
        fid = frame.get("frame_id", 0)
        temp = frame.get("temp_c", 0)

        # Update FPS
        self.fps_frames += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_frames / (now - self.last_fps_time)
            self.fps_frames = 0
            self.last_fps_time = now
            self.lbl_fps.setText(f"FPS: {self.current_fps:.1f}")

        self.lbl_temp.setText(f"Temp: {temp} deg C")
        self.lbl_frame.setText(f"Frame: {fid:,}")

        # Update Views
        self.view_depth.set_distances(dists)
        if ambs:
            self.view_ambient.set_data(ambs)

        # Update Histogram for selected zone
        if "cnh" in frame and frame["cnh"]:
            cnh_entry = frame["cnh"].get(self.selected_zone, None)
            if cnh_entry is not None:
                amb, bins = cnh_entry
                zdist = dists[self.selected_zone] if self.selected_zone < len(dists) else -1
                self.view_hist.set_data(self.selected_zone, amb, bins, zdist, temp, fid)
                self.view_depth.set_cnh_data(self.selected_zone, amb, bins)

        # Hand presence check for Posture mode: detect when hand has left sensor view
        if self.current_mode == "POSTURE" and dists:
            has_hand = any(80 <= d <= 450 for d in dists)
            if not has_hand:
                self.posture_no_hand_count = getattr(self, "posture_no_hand_count", 0) + 1
                if self.posture_no_hand_count >= 10:  # ~650 ms of no hand in sensor view
                    if self.posture_sim.current_pose != "NONE":
                        self.posture_sim.reset_to_idle()
                    if hasattr(self.hero_ai_widget, "reset_to_idle"):
                        self.hero_ai_widget.reset_to_idle("POSTURE")
            else:
                self.posture_no_hand_count = 0

    def on_ai_event(self, ev):
        self.hero_ai_widget.update_event(ev)
        mode = ev.get("mode")
        label = ev.get("label", "")
        conf = ev.get("confidence", 0.0)
        lat_us = ev.get("latency_us", 0)
        extra = ev.get("extra", "")

        if mode == "GESTURE":
            self.gesture_sim.handle_gesture(label)
        elif mode == "POSTURE":
            self.posture_sim.handle_posture(label, conf, lat_us)
        elif mode == "SURFACE":
            self.surface_sim.handle_surface(label, conf, extra, lat_us)
            self.view_depth.set_surface_preset(label, extra)

    def on_mcu_mode_switched(self, mode_str):
        self.hero_ai_widget.set_active_mode(mode_str)
        m = mode_str.upper()
        if "GEST" in m:
            self.btn_mode_gesture.setChecked(True)
            self.sim_stack.setCurrentIndex(0)
            self.view_depth.set_mode("GESTURE")
        elif "SURF" in m:
            self.btn_mode_surface.setChecked(True)
            self.sim_stack.setCurrentIndex(2)
            self.surface_sim.reset_to_idle()
            self.view_depth.set_mode("SURFACE")
        elif "POST" in m:
            self.btn_mode_posture.setChecked(True)
            self.sim_stack.setCurrentIndex(1)
            self.posture_sim.reset_to_idle()
            self.view_depth.set_mode("POSTURE")
        self.update_mode_button_styles()

    def on_serial_status(self, msg, is_ok):
        self.status_bar.showMessage(msg)

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.stop()
        event.accept()


# ----------------------------------------------------------------------
# Application Entrypoint
# ----------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = EdgeSenseMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
