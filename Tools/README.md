# EdgeSense Tools Guide

This folder contains the complete desktop toolchain for EdgeSense: AI dataset recording, model training, ST Edge AI code generation, firmware compilation, flashing, and real-time visualization.

---

## Tool Overview

| Tool | Location | Description |
|---|---|---|
| **EdgeSense AI Studio** | `Studio/run_nn_studio.bat` | 4-in-1 GUI for recording ToF datasets, training PyTorch neural networks, and deploying models to the project. |
| **Model Deployment Pipeline** | `Studio/deploy_to_project.bat` | Converts trained ONNX models using ST Edge AI Core (`stedgeai`) and compiles `EdgeSense.bin` / `EdgeSense.elf`. |
| **USB-C DFU Flasher** | `Flashing/flash_firmware_usb.bat` | One-click flasher via native STM32H5 USB-C ROM DFU (no external programmer required). |
| **ST-LINK Flasher** | `Flashing/flash_firmware.bat` | Flasher via hardware ST-LINK SWD programmer (J3 header). |
| **Real-Time Depth Visualizer** | `Visualizer/run_visualizer.bat` | Live 8x8 depth heatmap, target tracking, and raw 48-bin CNH optical waveform viewer. |

---

## 1. Quick Start: EdgeSense AI Studio

Double-click `Studio/run_nn_studio.bat` or run:
```cmd
cd Studio
run_nn_studio.bat
```

Inside the Studio:
1. Connect to the board's USB CDC COM port.
2. Record samples for Gestures (8x8 depth), Postures (8x8 depth), or Surfaces (raw CNH histograms).
3. Click **Train Model** to generate the optimized ONNX model.
4. Click **Deploy [Model] to Project** to automatically convert the model to optimized C code and compile the firmware binary.

---

## 2. Deploying Models to Project (`deploy_to_project.bat`)

Double-click `Studio/deploy_to_project.bat` from File Explorer, or run from the command prompt:

```cmd
:: Deploy all trained models found in Models/ and build firmware:
Studio\deploy_to_project.bat

:: Or deploy a specific model:
Studio\deploy_to_project.bat gesture
Studio\deploy_to_project.bat surface
Studio\deploy_to_project.bat posture
```

This script:
1. Dynamically detects your ST Edge AI Core (`stedgeai`) and STM32 GNU compiler toolchain.
2. Converts the ONNX models into optimized ST-AI C structures (`X-CUBE-AI/App/`).
3. Compiles the firmware into `Debug/EdgeSense.bin` and `Debug/EdgeSense.elf`.

---

## 3. Flashing Firmware

Choose your preferred method:

### Option A: USB-C DFU (Recommended - No Programmer Needed)
1. Connect EdgeSense to your PC using a USB-C cable.
2. Enter DFU mode: Press and hold **SW2** (BOOT0), press and release **SW1** (RESET), then release **SW2**.
3. Double-click `Flashing/flash_firmware_usb.bat`.

### Option B: ST-LINK Programmer
1. Connect your ST-LINK to the **J3** SWD header (`3V3`, `GND`, `NRST`, `SWDIO`, `SWCLK`).
2. Double-click `Flashing/flash_firmware.bat`.

---

## 4. Real-Time Depth Visualizer

Double-click `Visualizer/run_visualizer.bat` or run:
```cmd
cd Visualizer
run_visualizer.bat
```
View live 64-zone depth arrays, distance histograms, and CNH optical waveform pulses in real time.
