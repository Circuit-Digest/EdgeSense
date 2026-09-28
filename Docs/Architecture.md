# EdgeSense System Architecture & Hardware Specification

/*
 * Architecture.md
 * System pipeline architecture and technical documentation for EdgeSense
 *
 * Copyright (c) 2026 Dharagesh and Circuit Digest
 * https://github.com/Circuit-Digest/EdgeSense
 * Licensed under GNU General Public License v3.0
 */

`
  ███████╗██████╗  ██████╗ ███████╗███████╗███████╗███╗   ██╗███████╗███████╗
  ██╔════╝██╔══██╗██╔════╝ ██╔════╝██╔════╝██╔════╝████╗  ██║██╔════╝██╔════╝
  █████╗  ██║  ██║██║  ███╗█████╗  ███████╗█████╗  ██╔██╗ ██║███████╗█████╗  
  ██╔══╝  ██║  ██║██║   ██║██╔══╝  ╚════██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝  
  ███████╗██████╔╝╚██████╔╝███████╗███████║███████╗██║ ╚████║███████║███████╗
  ╚══════╝╚═════╝  ╚═════╝ ╚══════╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝
`

---

## 1. Hardware Architecture

EdgeSense is an ultra-compact edge AI sensing node combining:
- **MCU**: STMicroelectronics **STM32H533CEUx**
  - ARM Cortex-M33 with TrustZone, FPU (single-precision), DSP instructions
  - 250 MHz core clock, 512 KB Flash, 256 KB SRAM
  - High-Speed USB 2.0 (FS Device PHY)
  - Dedicated I3C / Fast-mode Plus (Fm+) 1 MHz I2C bus
- **Sensor**: STMicroelectronics **VL53L8CH / VL53LMZ** Direct Time-of-Flight (dToF)
  - 8x8 multizone ranging (64 independent distance zones)
  - 4x4 Cumulative Normalized Histogram (CNH) binning mode (48 physical bins per zone)
  - 940 nm VCSEL emitter with integrated metasurface lens
  - Wide 65° diagonal Field of View (FoV)

---

## 2. Firmware Execution Pipeline

EdgeSense runs an interrupt-driven, bare-metal cooperative architecture designed for low latency and zero heap fragmentation:

`mermaid
graph TD
    A[VL53LMZ Hardware INT Pin] -->|EXTI Fallback / Poll| B[Ranging Data Ready]
    B --> C[vl53lmz_get_ranging_data]
    C --> D{Active Pipeline Mode}
    D -->|MODE_GESTURE| E[model_gesture_process]
    D -->|MODE_SURFACE| F[model_surface_process]
    D -->|MODE_POSTURE| G[model_posture_process]
    D -->|MODE_SMOKE| H[model_smoke_process]
    E --> I[pipeline_manager_report_event]
    F --> I
    G --> I
    H --> I
    I -->|JSON / ASCII| J[USB CDC Virtual COM Port]
`

### Memory Footprint on STM32H533CEUx
- **Flash ROM**: ~230 KB (45% of 512 KB)
- **SRAM**: ~87 KB (34% of 256 KB)
- **Network Weights & Activations**:
  - gesture_nn: 18.5 KB weights, 15.0 KB activations buffer (reusable pool)
  - posture_nn: 12.2 KB weights, 8.4 KB activations buffer
  - surface_nn: 7.8 KB weights, 4.2 KB activations buffer

---

## 3. Dynamic Mode Switching Engine

The runtime switching engine (pipeline_manager.c) controls sensor configuration and active neural network instances without resetting the MCU:

1. Sensor ranging is safely stopped via l53lmz_stop_ranging().
2. Sensor resolution is re-indexed:
   - **Gesture / Posture / Smoke**: VL53LMZ_RESOLUTION_8X8 (64 zones, 15 Hz)
   - **Surface Classification**: VL53LMZ_RESOLUTION_4X4 with CNH histogram readout
3. The AI network pointer is dynamically swapped in the pipeline table.
4. Feature buffers are zeroed to prevent cross-mode bleed.
5. Ranging is restarted via l53lmz_start_ranging().
