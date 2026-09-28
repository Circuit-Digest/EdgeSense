# EdgeSense AT Command Interface Manual

/*
 * AT_Command_Manual.md
 * Serial Command Protocol and Pipeline Control Specification
 *
 * Copyright (c) 2026 Dharagesh and Circuit Digest
 * https://github.com/Circuit-Digest/EdgeSense
 * Licensed under GNU General Public License v3.0
 */

All commands are sent over the USB CDC Virtual COM port (921600 baud, 8-N-1) terminated with \r\n (CRLF).

---

## Command Reference

| Command | Parameter | Response | Description |
|---|---|---|---|
| `AT` | *None* | `OK` | Link verification / Ping |
| `AT+MODE=?` | *None* | `+MODE: (0:GESTURE, 1:SURFACE, 2:POSTURE, 3:SMOKE)` | List supported operating modes |
| `AT+MODE?` | *None* | `+MODE: <id>,<NAME>` | Query current operating mode |
| `AT+MODE=<id>` | `0` to `3` | `OK` or `ERROR: ...` | Switch active pipeline mode at runtime |
| `AT+STREAM=<en>` | `0` or `1` | `OK` | Enable (`1`) or disable (`0`) real-time JSON depth matrix streaming |
| `AT+STATUS` | *None* | Multi-line diagnostic dump | Get sensor health, frame count, temperature, and FPS |
| `AT+INFO` | *None* | Firmware metadata | Print version, MCU target, sensor UID, and build date |
| `AT+RESET` | *None* | `OK` | Perform software system reset via NVIC |

---

## Operating Modes

### Mode 0: 3D Gesture Recognition (`GESTURE`)
- **Resolution**: 8x8 zones (64 zones) @ 15 Hz
- **Architecture**: 2D-CNN with 15-frame sliding window (1.0 s temporal history)
- **Output Classes**: `IDLE`, `SWIPE_LEFT`, `SWIPE_RIGHT`, `SWIPE_UP`, `SWIPE_DOWN`
- **Output Event**:
  `[AI GESTURE] >>> DETECTED: SWIPE_LEFT (Confidence:  96.7% | Latency: 6805 us | 1633263 cycles) <<<`

### Mode 1: Surface / Texture Classifier (`SURFACE`)
- **Resolution**: 4x4 zones (16 zones) with 48-bin CNH histograms @ 15 Hz
- **Architecture**: 1D-CNN Canonical Peak-Centered Histogram Network
- **Output Classes**: `HARD_FLOOR`, `CARPET`, `SPECULAR`, `VOID`
- **Output Event**:
  `[AI SURFACE] >>> DETECTED: CARPET (Confidence:  98.2% | Latency: 3210 us) <<<`

### Mode 2: Static Hand Posture Classifier (`POSTURE`)
- **Resolution**: 8x8 zones (64 zones) @ 15 Hz
- **Architecture**: Dual-Channel 2D-CNN (Depth + Signal Intensity)
- **Output Classes**: `NONE`, `FLAT_HAND`, `LIKE`, `DISLIKE`, `BREAK_TIME`, `FIST`
- **Output Event**:
  `[AI POSTURE] >>> DETECTED: LIKE (Confidence:  97.5% | Latency: 4120 us) <<<`

### Mode 3: Optical Obscuration / Smoke Detector (`SMOKE`)
- **Resolution**: 8x8 zones (64 zones) @ 15 Hz
- **Methodology**: Multi-zone signal attenuation and ambient light scattering ratio
- **Output Event**:
  `[AI SMOKE] >>> ALERT: OBSCURATION_DETECTED (Level: 42% | Attenuation: -3.8 dB) <<<`