"""
Запис екрану вікна Uncrashed у відеофайл.
ПРОБІЛ — старт/стоп запису, Q — вихід.
"""

import time
import cv2
import dxcam
import keyboard
import win32gui

WINDOW_TITLE = "Uncrashed"
OUTPUT_FILE = "screen_recording.mp4"
TARGET_FPS = 30


def find_window_rect(title_substring: str):
    result = []
    def _enum_cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            if title_substring.lower() in win32gui.GetWindowText(hwnd).lower():
                result.append(win32gui.GetWindowRect(hwnd))
    win32gui.EnumWindows(_enum_cb, None)
    return result[0] if result else None


def main():
    rect = find_window_rect(WINDOW_TITLE)
    if rect is None:
        print(f"[ERROR] Вікно '{WINDOW_TITLE}' не знайдено.")
        return

    left, top, right, bottom = rect
    left, top = max(0, left), max(0, top)
    width = right - left
    height = bottom - top
    region = (left, top, right, bottom)

    print(f"[INFO] Вікно: {width}x{height}")
    print("[INFO] ПРОБІЛ — старт/стоп запису")
    print("[INFO] Q — вихід")

    camera = dxcam.create(output_color="BGR")
    camera.start(region=region, target_fps=TARGET_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    recording = False
    frames = 0
    t_start = 0.0

    def toggle_recording(_event=None):
        nonlocal recording, writer, frames, t_start
        if not recording:
            writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, TARGET_FPS, (width, height))
            frames = 0
            t_start = time.perf_counter()
            recording = True
            print(f"\n[REC] Запис розпочато → {OUTPUT_FILE}")
        else:
            recording = False
            elapsed = time.perf_counter() - t_start
            if writer:
                writer.release()
                writer = None
            print(f"\n[STOP] {frames} кадрів за {elapsed:.1f}с ({frames/elapsed:.1f} fps) → {OUTPUT_FILE}")

    keyboard.on_press_key("space", toggle_recording)

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                continue

            if recording:
                writer.write(frame)
                frames += 1

            # Маленький превью (1/4 розміру)
            small = cv2.resize(frame, (width // 4, height // 4))
            if recording:
                elapsed = time.perf_counter() - t_start
                cv2.putText(small, f"REC {elapsed:.0f}s  {frames}f", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(small, "PAUSED (SPACE to rec)", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Screen Recorder", small)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if writer:
            writer.release()
        camera.stop()
        keyboard.unhook_all()
        cv2.destroyAllWindows()

    print("[INFO] Завершено.")


if __name__ == "__main__":
    main()
