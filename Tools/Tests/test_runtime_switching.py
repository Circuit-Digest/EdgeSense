import serial
import time
import json
import sys

def main():
    port = "COM9"
    print(f"Connecting to EdgeSense on {port}...")
    try:
        ser = serial.Serial(port, baudrate=921600, timeout=1.0)
    except Exception as e:
        print(f"Failed to open {port}: {e}")
        return 1

    time.sleep(0.5)

    def send_cmd(cmd, wait=0.4):
        ser.reset_input_buffer()
        ser.write((cmd + "\r\n").encode("ascii"))
        time.sleep(wait)
        lines = []
        while ser.in_waiting:
            raw = ser.readline().decode("ascii", errors="ignore").strip()
            if raw and not raw.startswith("{"):
                lines.append(raw)
        return lines

    print("\n--- 1. Ping Device ---")
    res = send_cmd("AT")
    print("Response:", res)
    assert any("OK" in l for l in res), "AT ping failed!"

    print("\n--- 2. Initial Resolution Query (AT+RES?) ---")
    res = send_cmd("AT+RES?")
    print("Response:", res)
    assert any("+RES:" in l for l in res), "AT+RES? failed!"

    print("\n--- 3. Reconfigure to 4x4 with 48 Bins (AT+RES=4X4) ---")
    res = send_cmd("AT+RES=4X4", wait=1.0)
    print("Response:", res)
    assert any("OK" in l for l in res), "AT+RES=4X4 failed!"

    print("\n--- 4. Verify Active Profile (AT+RES?) ---")
    res = send_cmd("AT+RES?")
    print("Response:", res)
    assert any("RESOLUTION=4X4" in l for l in res), "Resolution did not switch to 4X4!"
    assert any("BINS=48" in l for l in res), "Bins did not switch to 48!"

    print("\n--- 5. Query Status (AT+STATUS?) in 4x4 Mode ---")
    res = send_cmd("AT+STATUS?")
    print("Response:", res)
    assert any("RES=4X4" in l for l in res), "Status does not reflect 4X4!"

    print("\n--- 6. Capture 4x4 Raw Frame Stream ---")
    send_cmd("AT+STREAM=RAW", wait=0.2)
    time.sleep(0.2)
    ser.reset_input_buffer()
    t0 = time.time()
    frame_4x4 = None
    while time.time() - t0 < 3.0:
        if ser.in_waiting:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    f = json.loads(line)
                    if "res" in f and f["res"] == 16:
                        frame_4x4 = f
                        break
                except Exception:
                    pass
        time.sleep(0.01)

    if frame_4x4:
        print("Successfully captured 4x4 JSON Frame:")
        print(f"  Frame No : {frame_4x4.get('f')}")
        print(f"  Temp     : {frame_4x4.get('t')} C")
        print(f"  Res Tag  : {frame_4x4.get('res')} zones")
        print(f"  Bins Tag : {frame_4x4.get('bins')} bins")
        print(f"  Distances: {len(frame_4x4['d'])} elements -> {frame_4x4['d']}")
        print(f"  Ambient  : {len(frame_4x4['a'])} elements")
        print(f"  Hist Matrix: {len(frame_4x4['h'])} zones x {len(frame_4x4['h'][0])} bins")
        assert len(frame_4x4['d']) == 16, f"Expected 16 distances, got {len(frame_4x4['d'])}"
        assert len(frame_4x4['h']) == 16, f"Expected 16 histogram zones, got {len(frame_4x4['h'])}"
        assert len(frame_4x4['h'][0]) == 48, f"Expected 48 histogram bins, got {len(frame_4x4['h'][0])}"
        print(">>> 4x4 Frame Verification PASSED! <<<")
    else:
        print(">>> ERROR: Did not capture 4x4 frame! <<<")
        return 1

    print("\n--- 7. Reconfigure Back to 8x8 with 16 Bins (AT+RES=8X8) ---")
    res = send_cmd("AT+RES=8X8", wait=1.0)
    print("Response:", res)
    assert any("OK" in l for l in res), "AT+RES=8X8 failed!"

    print("\n--- 8. Verify Active Profile (AT+RES?) ---")
    res = send_cmd("AT+RES?")
    print("Response:", res)
    assert any("RESOLUTION=8X8" in l for l in res), "Resolution did not switch back to 8X8!"
    assert any("BINS=16" in l for l in res), "Bins did not switch back to 16!"

    print("\n--- 9. Capture 8x8 Raw Frame Stream ---")
    ser.reset_input_buffer()
    t0 = time.time()
    frame_8x8 = None
    while time.time() - t0 < 3.0:
        if ser.in_waiting:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    f = json.loads(line)
                    if "res" in f and f["res"] == 64:
                        frame_8x8 = f
                        break
                except Exception:
                    pass
        time.sleep(0.01)

    if frame_8x8:
        print("Successfully captured 8x8 JSON Frame:")
        print(f"  Frame No : {frame_8x8.get('f')}")
        print(f"  Res Tag  : {frame_8x8.get('res')} zones")
        print(f"  Bins Tag : {frame_8x8.get('bins')} bins")
        print(f"  Distances: {len(frame_8x8['d'])} elements")
        print(f"  Hist Matrix: {len(frame_8x8['h'])} zones x {len(frame_8x8['h'][0])} bins")
        assert len(frame_8x8['d']) == 64, f"Expected 64 distances, got {len(frame_8x8['d'])}"
        assert len(frame_8x8['h']) == 64, f"Expected 64 histogram zones, got {len(frame_8x8['h'])}"
        assert len(frame_8x8['h'][0]) == 16, f"Expected 16 histogram bins, got {len(frame_8x8['h'][0])}"
        print(">>> 8x8 Frame Verification PASSED! <<<")
    else:
        print(">>> ERROR: Did not capture 8x8 frame! <<<")
        return 1

    print("\n--- 10. Switch to Surface Mode (AT+MODE=SURFACE) & Query Status ---")
    res = send_cmd("AT+MODE=SURFACE")
    print("Mode response:", res)
    res = send_cmd("AT+STATUS?")
    print("Status response:", res)
    assert any("MODE=SURFACE" in l for l in res), "Mode did not switch to SURFACE!"

    send_cmd("AT+STREAM=RESULTS")
    send_cmd("AT+MODE=GESTURE")
    ser.close()
    print("\n==================================================================")
    print(">>> ALL 10 TESTS PASSED! RUNTIME SWITCHING OPERATES FLAWLESSLY! <<<")
    print("==================================================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
