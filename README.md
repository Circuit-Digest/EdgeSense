# EdgeSense: STM32 Edge AI Texture and Gesture Classifier


```
  ███████╗██████╗  ██████╗ ███████╗███████╗███████╗███╗   ██╗███████╗███████╗
  ██╔════╝██╔══██╗██╔════╝ ██╔════╝██╔════╝██╔════╝████╗  ██║██╔════╝██╔════╝
  █████╗  ██║  ██║██║  ███╗█████╗  ███████╗█████╗  ██╔██╗ ██║███████╗█████╗  
  ██╔══╝  ██║  ██║██║   ██║██╔══╝  ╚════██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝  
  ███████╗██████╔╝╚██████╔╝███████╗███████║███████╗██║ ╚████║███████║███████╗
  ╚══════╝╚═════╝  ╚═════╝ ╚══════╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝
```

[![Platform](https://img.shields.io/badge/Platform-STM32H533CEUx-03234B.svg)](https://www.st.com/en/microcontrollers-microprocessors/stm32h5-series.html)
[![Core](https://img.shields.io/badge/Core-ARM%20Cortex--M33%20%40%20250MHz-blue.svg)](https://arm.com)
[![Sensor](https://img.shields.io/badge/Sensor-VL53L8CH%208x8%20dToF-orange.svg)](https://www.st.com/en/imaging-and-photonics-solutions/vl53l8ch.html)
[![Edge AI](https://img.shields.io/badge/X--CUBE--AI-v10.2.1-green.svg)](https://www.st.com/en/embedded-software/x-cube-ai.html)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**EdgeSense** is a compact, high-performance edge artificial intelligence platform that performs on-device 3D gesture recognition, surface/texture classification, static hand posture detection, and optical obscuration sensing. 

Powered by the **STM32H533CEU6** (ARM Cortex-M33 running at 250 MHz) and the **ST VL53L8CH** 8x8 direct Time-of-Flight (dToF) sensor, EdgeSense executes quantized neural networks entirely on-chip in real time with sub-7 ms inference latency.

---

## Key Capabilities

EdgeSense features a runtime multi-model execution engine (`pipeline_manager.c`) allowing dynamic model switching over USB CDC:

1. **3D Gesture Recognition (8x8 ToF, 15 FPS)**:
   - Recognizes dynamic directional swipes: `SWIPE_LEFT`, `SWIPE_RIGHT`, `SWIPE_UP`, `SWIPE_DOWN`, and `IDLE`.
   - Continuous 15-frame sliding window (1.0 s temporal context) with zero gesture-lockout.
   - Physical temporal motion energy guard to eliminate stationary phantom triggers.
   - Inference latency: **~6.8 ms** per inference.

2. **Distance-Invariant Surface Classifier (4x4 CNH, 48 Bins)**:
   - Classifies textures and materials: `HARD_FLOOR` (tile/wood), `CARPET` (subsurface scattering), `SPECULAR` (mirrors/polished metal), and `VOID` (drop-offs/cliffs).
   - Utilizes ST-aligned Cumulative Normalized Histogram (CNH) peak-centered canonical alignment.
   - Inference latency: **~3.2 ms** per inference.

3. **Static Hand Posture Classifier (8x8 Depth + Signal Intensity)**:
   - Identifies static human hand postures: `FLAT_HAND`, `LIKE`, `DISLIKE`, `BREAK_TIME`, `FIST`, and `NONE`.
   - Dual-channel 2D-CNN processing distance geometry and photon reflection intensity simultaneously.
   - Robust to ambient lighting variations and skin tone differences.
   - Inference latency: **~4.1 ms** per inference.

4. **Optical Obscuration & Smoke Detector**:
   - Detects microscopic particles, airborne obscuration, and smoke.
   - Computes multi-zone signal attenuation and ambient light scattering indices in real time.

---

## Hardware Specifications

| Component | Specification |
|---|---|
| **Microcontroller** | STMicroelectronics STM32H533CEUx (ARM Cortex-M33, 250 MHz, TrustZone, FPU, DSP) |
| **On-Chip Memory** | 512 KB Flash ROM, 256 KB SRAM |
| **ToF Sensor** | STMicroelectronics VL53L8CH / VL53LMZ Direct Time-of-Flight |
| **Sensor FoV** | 65° Diagonal Field of View |
| **Spatial Resolution** | 8x8 (64 zones) or 4x4 (16 zones with 48-bin CNH histograms) |
| **Ranging Distance** | 20 mm to 4000 mm |
| **Host Connectivity** | High-Speed USB 2.0 CDC (Virtual COM Port @ 921600 baud) |
| **Communication Bus** | Dedicated Fast-mode Plus (Fm+) 1 MHz I2C bus |
| **Power Supply** | 5V via USB-C or 3.3V external rail |

---

## Repository Structure

```
EdgeSense/
├── Core/                      # STM32H533 HAL drivers, ISRs, and main loop
├── Drivers/                   # CMSIS and STM32H5xx HAL libraries
├── Middlewares/               # ST X-CUBE-AI runtime v10.2.1 & USB Device stack
├── USB_Device/                # USB CDC Virtual COM Port descriptors & handlers
├── VL53LMZ_ULD/               # Ultra-Lite Driver & CNH Histogram Plugin
├── X-CUBE-AI/                 # Edge AI Pipeline Manager & Active Networks
│   └── App/                   # pipeline_manager, model_gesture, model_posture, model_surface, model_smoke
│
├── Models/                    # Pre-trained production ONNX models & datasets
│   ├── gesture_nn_model.onnx  # 5-class 2D-CNN Gesture Classifier
│   ├── posture_nn_model.onnx  # 6-class 2D-CNN Posture Classifier
│   ├── surface_nn_model.onnx  # 4-class 1D-CNN Surface Classifier
│   ├── gesture_dataset.npz    # Default Gesture training dataset
│   ├── posture_dataset.npz    # Default Posture training dataset
│   ├── surface_dataset.npz    # Default Surface training dataset
│   └── Archive/               # Intermediate training iterations
│
├── Tools/                     # Python Studio, GUIs, and deploy automation
│   ├── Studio/                # 4-in-1 EdgeSense Neural Network Studio
│   │   ├── edgesense_nn_studio.py
│   │   ├── run_nn_studio.bat
│   │   ├── deploy_model_to_mcu.bat
│   │   ├── deploy_posture_model_to_mcu.bat
│   │   ├── deploy_surface_model_to_mcu.bat
│   │   ├── requirements.txt
│   │   └── screenshots/
│   ├── Visualizer/            # Real-time multi-zone depth viewer
│   ├── Flashing/              # STM32CubeProgrammer CLI flasher
│   └── Tests/                 # Automated AT pipeline verification tests
│
├── Hardware/                  # KiCad EDA design files, schematics & BOM
│   ├── Schematics/            # KiCad project, schematics, PCB layout
│   └── Production/            # Gerber ZIP, Pick-and-Place, BOM
│
├── Docs/                      # Technical references & protocol manuals
│   ├── Architecture.md        # System architecture and memory specs
│   └── AT_Command_Manual.md   # Complete AT command serial reference
│
├── EdgeSense.ioc              # STM32CubeMX hardware pinout and clock config
├── STM32H533CEUX_FLASH.ld     # Flash linker script
├── LICENSE                    # GNU General Public License v3.0
└── README.md                  # This documentation
```

---

## Getting Started

### 1. Prerequisites
- **STM32CubeIDE** (v2.2.0 or newer) with GNU Tools for STM32 (ARM GCC 14.3+)
- **ST Edge AI Core (X-CUBE-AI)** v10.2.1+
- **Python** 3.10+ (64-bit)
- **ST-LINK V2 / V3** Programmer

### 2. Python Environment Setup
Install the required dependencies for the EdgeSense AI Studio:
```bash
cd Tools/Studio
pip install -r requirements.txt
```

### 3. Launching EdgeSense Studio
Double-click `Tools/Studio/run_nn_studio.bat` or run:
```bash
python Tools/Studio/edgesense_nn_studio.py
```
From the Studio, you can:
- View live 8x8 depth heatmaps and 4x4 CNH pulse shapes.
- Record new training datasets for Gestures, Postures, or Surfaces.
- Train PyTorch neural networks with 1 click.
- Deploy trained models to the STM32H533 via `deploy_*_model_to_mcu.bat` automatically.

### 4. Building Firmware via Command Line
```bash
make -C Debug all
```
The compiled binary will be located at `Debug/EdgeSense.elf` and `Debug/EdgeSense.bin`.

### 5. Flashing Firmware
Connect your ST-LINK to the SWD header and run:
```bash
Tools/Flashing/flash_firmware.bat
```

---

## Serial Communication & AT Commands

Connect to the USB CDC COM port at **921600 baud, 8-N-1**.

| Command | Action |
|---|---|
| `AT` | Verify connection (returns `OK`) |
| `AT+MODE=?` | List available modes: `(0:GESTURE, 1:SURFACE, 2:POSTURE, 3:SMOKE)` |
| `AT+MODE?` | Query currently active mode |
| `AT+MODE=0` | Switch to 3D Gesture Recognition mode |
| `AT+MODE=1` | Switch to Surface / Texture Classifier mode |
| `AT+MODE=2` | Switch to Hand Posture Classifier mode |
| `AT+MODE=3` | Switch to Optical Smoke / Obscuration mode |
| `AT+STREAM=1` | Enable real-time 8x8 JSON depth streaming |
| `AT+STREAM=0` | Disable streaming |
| `AT+STATUS` | View sensor health, frame counts, and FPS |
| `AT+RESET` | Software system reboot |

---

## License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.