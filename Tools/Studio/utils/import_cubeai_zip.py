#!/usr/bin/env python3
"""
import_cubeai_zip.py
Automated Importer for STM32Cube.AI Studio (STEDGEAI-CUBEAI) Code Exports

STM32Cube.AI Studio packages generated C model files into a .zip archive
(<project_name>-...-ApplicationTemplate-code.zip).

This tool automatically:
  1. Finds the latest downloaded zip in your Downloads folder (or accepts a path).
  2. Extracts the generated C model files (network.c, network_data.c, etc.).
  3. Maps them cleanly into CODE/EdgeSense/X-CUBE-AI/App/ (e.g. gesture_nn.c).
  4. Updates the Middlewares/ST/AI/ runtime library if included.
"""

import os
import sys
import glob
import shutil
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
APP_DIR = os.path.join(PROJECT_ROOT, "X-CUBE-AI", "App")
LIB_DIR = os.path.join(PROJECT_ROOT, "Middlewares", "ST", "AI", "Lib")

DEFAULT_DOWNLOADS = os.path.expanduser("~/Downloads")

def find_latest_cubeai_zip():
    candidates = glob.glob(os.path.join(DEFAULT_DOWNLOADS, "*code.zip"))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]

def detect_model_name_from_zip(z):
    """Inspects files in the zip to identify if it is gesture, posture, or surface."""
    for member in z.namelist():
        lower = member.lower()
        if "gesture" in lower:
            return "gesture_nn"
        elif "posture" in lower:
            return "posture_nn"
        elif "surface" in lower:
            return "surface_nn"
    return "gesture_nn"

def import_zip(zip_path, target_model_name=None):
    print(f"[CUBE.AI IMPORT] Opening archive: {zip_path}")
    if not os.path.exists(zip_path):
        print(f"[ERROR] Zip archive not found: {zip_path}")
        return False

    os.makedirs(APP_DIR, exist_ok=True)
    os.makedirs(LIB_DIR, exist_ok=True)

    extracted_count = 0
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()

        # Auto-detect model if not explicitly forced
        if not target_model_name or target_model_name == "auto":
            target_model_name = detect_model_name_from_zip(z)
            print(f"[AUTO-DETECT] Identified target model: '{target_model_name}'")

        # 1. Extract model C/H files from ai_generated_network/
        for member in file_list:
            if member.startswith("ai_generated_network/"):
                basename = os.path.basename(member)
                if not basename or basename.endswith("/"):
                    continue

                # Read file bytes from zip
                data = z.read(member)

                # Handle naming: if named network.c/h, remap to target_model_name (e.g. gesture_nn.c/h)
                dest_name = basename
                # If model is already named properly in Studio (e.g. gesture_nn.c), preserve as is
                if not basename.startswith("network"):
                    dest_name = basename
                elif target_model_name and target_model_name != "network":
                    # Remap network.* to target_model_name.*
                    if basename.startswith("network."):
                        dest_name = basename.replace("network.", f"{target_model_name}.")
                    elif basename.startswith("network_data."):
                        dest_name = basename.replace("network_data.", f"{target_model_name}_data.")
                    elif basename.startswith("network_data_params."):
                        dest_name = basename.replace("network_data_params.", f"{target_model_name}_data_params.")
                    elif basename.startswith("network_details."):
                        dest_name = basename.replace("network_details.", f"{target_model_name}_details.")
                    elif basename.startswith("network_generate_report."):
                        dest_name = basename.replace("network_generate_report.", f"{target_model_name}_generate_report.")
                    elif "c_info_generate.json" in basename:
                        dest_name = f"{target_model_name}_c_info.json"

                    # If text file (.c, .h, .json, .txt), patch header references
                    if any(basename.endswith(ext) for ext in [".c", ".h", ".json", ".txt"]):
                        try:
                            text = data.decode("utf-8")
                            text = text.replace('"network.h"', f'"{target_model_name}.h"')
                            text = text.replace('"network_data.h"', f'"{target_model_name}_data.h"')
                            text = text.replace('"network_details.h"', f'"{target_model_name}_details.h"')
                            text = text.replace('STAI_NETWORK_', f'STAI_{target_model_name.upper()}_')
                            text = text.replace('AI_NETWORK_', f'AI_{target_model_name.upper()}_')
                            text = text.replace('_stai_network_context', f'_stai_{target_model_name}_context')
                            text = text.replace('g_network_weights_array', f'g_{target_model_name}_weights_array')
                            text = text.replace('g_network_details', f'g_{target_model_name}_details')

                            # Replace ST-AI function names specifically (without renaming global C types)
                            stai_fns = [
                                'init', 'deinit', 'run', 'get_context_size', 'get_info',
                                'get_activations', 'get_weights', 'get_inputs', 'get_outputs',
                                'get_states', 'get_error', 'set_activations', 'set_weights',
                                'set_inputs', 'set_outputs', 'set_states', 'set_callback'
                            ]
                            for fn in stai_fns:
                                text = text.replace(f'stai_network_{fn}', f'stai_{target_model_name}_{fn}')

                            # Legacy X-CUBE-AI fallback
                            text = text.replace('ai_network', f'ai_{target_model_name}')

                            # Guarantee that global STAI types are preserved
                            text = text.replace(f'stai_{target_model_name}*', 'stai_network*')
                            text = text.replace(f'stai_{target_model_name} *', 'stai_network *')
                            text = text.replace(f'stai_{target_model_name}_info', 'stai_network_info')
                            text = text.replace(f'stai_{target_model_name}_details', 'stai_network_details')
                            data = text.encode("utf-8")
                        except Exception:
                            pass

                dest_path = os.path.join(APP_DIR, dest_name)
                # Scope forward_lite functions as static to prevent multiple-definition errors in multi-model firmware
                if dest_name.endswith(".c"):
                    try:
                        text = data.decode("utf-8")
                        text = text.replace("\nvoid forward_lite_", "\nstatic void forward_lite_")
                        data = text.encode("utf-8")
                    except Exception:
                        pass

                with open(dest_path, "wb") as f:
                    f.write(data)
                print(f"  -> Extracted: {dest_name} to X-CUBE-AI/App/")
                extracted_count += 1

            # 2. Extract runtime library (.a) if present
            if member.endswith(".a") and "Middlewares/ST/AI/Lib" in member:
                lib_name = os.path.basename(member)
                dest_lib = os.path.join(LIB_DIR, lib_name)
                with open(dest_lib, "wb") as f:
                    f.write(z.read(member))
                print(f"  -> Updated Runtime Library: {lib_name} in Middlewares/ST/AI/Lib/")

    print(f"\n[SUCCESS] Successfully imported {extracted_count} model files into EdgeSense!")
    print(f"Location: {APP_DIR}")
    return True

if __name__ == "__main__":
    zip_arg = sys.argv[1] if len(sys.argv) > 1 else None
    model_arg = sys.argv[2] if len(sys.argv) > 2 else "auto"

    if not zip_arg:
        zip_arg = find_latest_cubeai_zip()
        if not zip_arg:
            print("[ERROR] No Cube.AI zip files found in Downloads folder.")
            print("Usage: python import_cubeai_zip.py <path_to_zip> [model_name]")
            sys.exit(1)

    success = import_zip(zip_arg, model_arg)
    sys.exit(0 if success else 1)
