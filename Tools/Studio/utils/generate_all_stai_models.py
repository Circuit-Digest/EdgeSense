#!/usr/bin/env python3
"""
generate_all_stai_models.py
Generates native ST-AI Core v2 C code for Gesture, Posture, and Surface models.
Uses dynamic tool auto-discovery with zero hardcoded paths.
"""

import glob
import os
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")
OUT_DIR = os.path.join(PROJECT_ROOT, "X-CUBE-AI", "App")

def find_stedgeai():
    found = shutil.which("stedgeai") or shutil.which("stedgeai.exe")
    if found:
        return found
    patterns = [
        os.path.expanduser(r"~\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\*\Utilities\windows\stedgeai.exe"),
        r"C:\ST\STEdgeAI\*\stedgeai.exe",
        r"C:\Program Files\STMicroelectronics\STEdgeAI\*\stedgeai.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\STMicroelectronics\STEdgeAI\*\stedgeai.exe"),
        os.path.expandvars(r"%ProgramFiles%\STMicroelectronics\STEdgeAI\*\stedgeai.exe"),
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return sorted(matches)[-1]
    return None

MODELS = [
    ("gesture_nn", os.path.join(MODELS_DIR, "gesture_nn_model.onnx")),
    ("posture_nn", os.path.join(MODELS_DIR, "posture_nn_model.onnx")),
    ("surface_nn", os.path.join(MODELS_DIR, "surface_nn_model.onnx")),
]

def run_generation():
    stedgeai = find_stedgeai()
    if not stedgeai:
        print("[ERROR] ST Edge AI Core (stedgeai.exe) not found!")
        print("Please ensure STM32Cube.AI Studio or X-CUBE-AI pack is installed, or add stedgeai to your PATH.")
        sys.exit(1)

    print(f"[FOUND] Using ST Edge AI Core: {stedgeai}")

    for name, model_path in MODELS:
        print(f"\n==================================================")
        print(f"Generating ST-AI C Code for: {name}")
        print(f"Model: {model_path}")
        print(f"==================================================")
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found: {model_path}")
            sys.exit(1)

        cmd = [
            stedgeai,
            "generate",
            "--target", "stm32h5",
            "--name", name,
            "-m", model_path,
            "--c-api", "st-ai",
            "--compression", "none",
            "--output", OUT_DIR,
            "--no-workspace"
        ]
        start_t = time.time()
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"[ERROR] Generation failed for {name} with code {ret.returncode}")
            sys.exit(ret.returncode)
        print(f"[SUCCESS] {name} generated in {time.time() - start_t:.1f}s")

        # Post-process generated .c file to scope internal forward_lite functions as static
        c_file = os.path.join(OUT_DIR, f"{name}.c")
        if os.path.exists(c_file):
            with open(c_file, "r", encoding="utf-8") as f:
                c_content = f.read()
            c_content = c_content.replace("\nvoid forward_lite_", "\nstatic void forward_lite_")
            with open(c_file, "w", encoding="utf-8") as f:
                f.write(c_content)

    print("\n[ALL DONE] All 3 ST-AI models generated successfully!")

if __name__ == "__main__":
    run_generation()
