"""
Система візуального автонаведення дрона на ціль (автомобіль) в симуляторі Uncrashed.

Архітектура passthrough:
    Скрипт ЗАВЖДИ виводить через віртуальний Xbox-геймпад (ViGEmBus).
    В MANUAL — читає фізичний контролер (pygame) і прокидає його значення.
    В AUTO   — PID-контролер наводить дрон на ціль.

    Фізичний контролер → скрипт → віртуальний геймпад → гра

Налаштування:
    1. Запустити скрипт (він створить віртуальний Xbox контролер)
    2. Запустити Uncrashed
    3. В налаштуваннях Uncrashed прив'язати керування до "Xbox 360 Controller"
    4. Тепер MANUAL передає фізичний контролер, AUTO — автонаведення

ПРОБІЛ  — перемикання MANUAL ↔ AUTO
Q       — вихід (у вікні OpenCV)

Залежності:
    pip install dxcam ultralytics opencv-python numpy pywin32 vgamepad keyboard pygame
    + Драйвер ViGEmBus: https://github.com/nefarius/ViGEmBus/releases

Запуск від адміністратора (для keyboard).
"""

import time
import struct
import threading
import cv2
import numpy as np
import ctypes
import ctypes.wintypes
import dxcam
import win32gui
import win32api
import win32con
import keyboard
import vgamepad as vg
import pygame
from ultralytics import YOLO
from simple_pid import PID

# ── Налаштування ────────────────────────────────────────────────────────────

WINDOW_TITLE = "Uncrashed"
MODEL_PATH = "runs/detect/runs/detect/uncrashed_cars9/weights/best.engine"  # шлях до TensorRT моделі (експорт з train_uncrashed.py)
# MODEL_PATH = "yolov8n.engine"  # шлях до TensorRT моделі (експорт з train_uncrashed.py)
CAR_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.20
IMGSZ = 640  # розмір інференсу (менше = швидше, 640→384 ≈ +60% FPS)

# Режим візуалізації: "overlay" = поверх гри (borderless/windowed), "window" = окреме вікно
DISPLAY_MODE = "overlay"  # "overlay" | "window"

# Візуалізація
CROSSHAIR_COLOR = (0, 255, 0)
CROSSHAIR_SIZE = 20
CROSSHAIR_THICKNESS = 2
BBOX_COLOR = (0, 0, 255)
BBOX_COLOR_LOCKED = (0, 165, 255)
BBOX_THICKNESS = 2
LINE_COLOR = (255, 255, 0)
LINE_THICKNESS = 2

# ── Маппінг фізичного контролера (pygame axes) ─────────────────────────────
# Підстав індекси осей свого контролера. Дізнатися можна запустивши:
#   python -c "import pygame; pygame.init(); j=pygame.joystick.Joystick(0); j.init(); [print(f'Axis {i}: {j.get_axis(i):.3f}') for i in range(j.get_numaxes())]"
# Типовий маппінг для радіоапаратури (Mode 2):
PHYS_AXIS_ROLL     = 0   # Правий стік X
PHYS_AXIS_PITCH    = 1   # Правий стік Y
PHYS_AXIS_THROTTLE = 2   # Лівий стік Y
PHYS_AXIS_YAW      = 3   # Лівий стік X

# Інвертування осей (True якщо вісь працює навпаки)
INVERT_ROLL     = False
INVERT_PITCH    = False
INVERT_THROTTLE = False
INVERT_YAW      = False

# ── PID параметри ───────────────────────────────────────────────────────────

PID_ROLL_KP = 0.3
PID_ROLL_KI = 0.02
PID_ROLL_KD = 0.1

PID_PITCH_KP = 0.25
PID_PITCH_KI = 0.015
PID_PITCH_KD = 0.15

# Yaw (lівий стік X) — обертання на ціль по горизонталі
PID_YAW_KP = 0.2
PID_YAW_KI = 0.01
PID_YAW_KD = 0.05

# Throttle PID (автоматичний газ залежно від відстані до цілі)
# error_size = розмір bbox відносно екрану (0..1). Більший bbox = ближче = менше газу.
THROTTLE_BASE = 0.28       # базовий газ (тримати висоту)
THROTTLE_APPROACH = 0.12   # додатковий газ при наближенні (скалюється за відстанню)
THROTTLE_MAX = 0.50        # макс газ в авторежимі
TARGET_SIZE_CLOSE = 0.25   # bbox/екран > це значення = ціль близько, менше газу
THROTTLE_ERROR_CUT = 0.5   # при вертикальній помилці > цього — різко зменшити газ

# Кількість кадрів grace period: якщо ціль зникла, чекаємо стільки кадрів перед відміною.
GRACE_FRAMES = 45  # ~0.8с при 55 inferFPS

# Макс. відхилення стіка (0..1). 0.3 = 30% — дрон не зможе перевернутися.
MAX_STICK = 0.3

# Експоненціальне згладжування: чим ближче ціль до центру, тим повільніше реакція.
# error підноситься до степеня EXPO (>1 = повільніше поблизу центру)
EXPO = 1.5

# Rate limiter: макс зміна стіка за кадр (плавність руху)
RATE_LIMIT = 0.08


# ── Прозорий overlay поверх гри (Win32) ────────────────────────────────────

class GameOverlay:
    """Прозоре, click-through, always-on-top вікно для малювання поверх гри.
    Використовує UpdateLayeredWindow з per-pixel alpha — без блимання."""

    OVERLAY_CLASS = "DroneOverlay"

    # ctypes структури для UpdateLayeredWindow
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    class SIZE(ctypes.Structure):
        _fields_ = [("cx", ctypes.c_long), ("cy", ctypes.c_long)]

    class BLENDFUNCTION(ctypes.Structure):
        _fields_ = [
            ("BlendOp", ctypes.c_byte),
            ("BlendFlags", ctypes.c_byte),
            ("SourceConstantAlpha", ctypes.c_byte),
            ("AlphaFormat", ctypes.c_byte),
        ]

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x, self.y, self.w, self.h = x, y, w, h
        self._hdc_mem = None
        self._hbm = None
        self._ppvBits = ctypes.c_void_p()
        self._bm_w, self._bm_h = 0, 0
        self._dib_array = None
        self._busy = threading.Lock()
        self._create_window()

    def _create_window(self):
        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = {}
        wc.lpszClassName = self.OVERLAY_CLASS
        wc.hbrBackground = win32gui.GetStockObject(win32con.BLACK_BRUSH)
        wc.hCursor = win32api.LoadCursor(0, win32con.IDC_ARROW)
        try:
            win32gui.RegisterClass(wc)
        except Exception:
            pass  # вже зареєстровано

        ex_style = (win32con.WS_EX_LAYERED |
                    win32con.WS_EX_TRANSPARENT |
                    win32con.WS_EX_TOPMOST |
                    win32con.WS_EX_TOOLWINDOW)  # не показує в taskbar
        style = win32con.WS_POPUP

        self.hwnd = win32gui.CreateWindowEx(
            ex_style, self.OVERLAY_CLASS, "Overlay",
            style, self.x, self.y, self.w, self.h,
            0, 0, 0, None)

        # НЕ викликаємо SetLayeredWindowAttributes — UpdateLayeredWindow керує прозорістю
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)

        # ---------- argtypes для GDI/User32 (64-bit safe) ----------
        gdi = ctypes.windll.gdi32
        usr = ctypes.windll.user32
        gdi.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
        gdi.CreateCompatibleDC.restype = ctypes.c_void_p
        gdi.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        gdi.SelectObject.restype = ctypes.c_void_p
        gdi.DeleteObject.argtypes = [ctypes.c_void_p]
        gdi.DeleteObject.restype = ctypes.c_int
        usr.GetDC.argtypes = [ctypes.c_void_p]
        usr.GetDC.restype = ctypes.c_void_p
        usr.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        usr.ReleaseDC.restype = ctypes.c_int
        usr.UpdateLayeredWindow.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint
        ]
        usr.UpdateLayeredWindow.restype = ctypes.c_int

        # Кешований memory DC для double-buffered рендеру
        hdc_screen = usr.GetDC(0)
        self._hdc_mem = gdi.CreateCompatibleDC(hdc_screen)
        usr.ReleaseDC(0, hdc_screen)

    def _ensure_dib(self, w: int, h: int):
        """Створює або оновлює DIB section якщо розмір змінився."""
        if self._bm_w == w and self._bm_h == h and self._hbm:
            return
        if self._hbm:
            ctypes.windll.gdi32.DeleteObject(self._hbm)
        bmi = struct.pack('<IiiHHIIiiII', 40, w, h, 1, 32, 0, 0, 0, 0, 0, 0)

        _CreateDIBSection = ctypes.windll.gdi32.CreateDIBSection
        _CreateDIBSection.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint
        ]
        _CreateDIBSection.restype = ctypes.c_void_p

        self._ppvBits = ctypes.c_void_p()
        self._hbm = _CreateDIBSection(
            self._hdc_mem, bmi, 0, ctypes.byref(self._ppvBits), None, 0)
        ctypes.windll.gdi32.SelectObject(self._hdc_mem, self._hbm)
        self._bm_w, self._bm_h = w, h
        # Zero-copy numpy view на пам'ять DIB section
        buf = (ctypes.c_uint8 * (w * h * 4)).from_address(self._ppvBits.value)
        self._dib_array = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))

    def update(self, img_bgr: np.ndarray):
        """Атомарне оновлення overlay через UpdateLayeredWindow (без блимання).
        Чорні пікселі (0,0,0) → прозорі (alpha=0). Пише напряму в DIB memory."""
        h, w = img_bgr.shape[:2]
        self._ensure_dib(w, h)

        # Пишемо напряму в пам'ять DIB (zero-copy, без cvtColor, flip, memmove)
        src = img_bgr[::-1]                    # bottom-up view (без копіювання)
        self._dib_array[:, :, :3] = src        # BGR канали → DIB
        # Alpha: будь-який ненульовий піксель → 255, чорний → 0 (одна операція)
        self._dib_array[:, :, 3] = (np.max(src, axis=2) > 0) * np.uint8(255)

        # Атомарне оновлення вікна
        pt_pos = self.POINT(self.x, self.y)
        sz = self.SIZE(w, h)
        pt_src = self.POINT(0, 0)
        blend = self.BLENDFUNCTION(0, 0, 255, 1)

        ctypes.windll.user32.UpdateLayeredWindow(
            self.hwnd, 0,
            ctypes.byref(pt_pos), ctypes.byref(sz),
            self._hdc_mem, ctypes.byref(pt_src),
            0, ctypes.byref(blend), 2)  # ULW_ALPHA = 2

    def update_async(self, img_bgr: np.ndarray):
        """Non-blocking: рендер в фоновому потоці. Якщо попередній ще не закінчив — пропуск."""
        if not self._busy.acquire(blocking=False):
            return  # попередній кадр ще рендериться
        data = img_bgr.copy()  # snapshot для фонового потоку
        def _work():
            try:
                self.update(data)
            finally:
                self._busy.release()
        threading.Thread(target=_work, daemon=True).start()

    def wait_done(self):
        """Чекає завершення фонового рендеру (для cleanup)."""
        self._busy.acquire()
        self._busy.release()

    def reposition(self, x: int, y: int, w: int, h: int):
        self.x, self.y, self.w, self.h = x, y, w, h
        win32gui.MoveWindow(self.hwnd, x, y, w, h, True)

    def destroy(self):
        try:
            if self._hbm:
                ctypes.windll.gdi32.DeleteObject(self._hbm)
            if self._hdc_mem:
                ctypes.windll.gdi32.DeleteDC(self._hdc_mem)
            win32gui.DestroyWindow(self.hwnd)
        except Exception:
            pass


# ── PID контролер ───────────────────────────────────────────────────────────

def expo_curve(error: float, exp: float = EXPO) -> float:
    """Експо-крива: зберігає знак, але зменшує реакцію поблизу нуля."""
    sign = 1.0 if error >= 0 else -1.0
    return sign * (abs(error) ** exp)


class RateLimiter:
    """Обмежує швидкість зміни значення для плавного керування."""
    def __init__(self, max_rate: float):
        self.max_rate = max_rate
        self.prev = 0.0

    def __call__(self, value: float) -> float:
        delta = max(-self.max_rate, min(self.max_rate, value - self.prev))
        self.prev += delta
        return self.prev

    def reset(self):
        self.prev = 0.0


# ── Фізичний контролер (pygame) ─────────────────────────────────────────────

def init_physical_joystick() -> pygame.joystick.JoystickType | None:
    """Ініціалізує перший знайдений фізичний джойстик."""
    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if count == 0:
        print("[WARN] Фізичний контролер не знайдено! MANUAL режим не працюватиме.")
        return None
    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"[INFO] Фізичний контролер: {joy.get_name()} ({joy.get_numaxes()} осей)")
    return joy


def read_physical(joy: pygame.joystick.JoystickType | None) -> tuple[float, float, float, float]:
    """Читає осі фізичного контролера. Повертає (roll, pitch, throttle, yaw) в -1..+1."""
    if joy is None:
        return (0.0, 0.0, -1.0, 0.0)  # нейтраль

    pygame.event.pump()  # оновити стан

    def read_axis(idx: int, invert: bool) -> float:
        if idx < joy.get_numaxes():
            val = joy.get_axis(idx)
            return -val if invert else val
        return 0.0

    roll     = read_axis(PHYS_AXIS_ROLL, INVERT_ROLL)
    pitch    = read_axis(PHYS_AXIS_PITCH, INVERT_PITCH)
    throttle = read_axis(PHYS_AXIS_THROTTLE, INVERT_THROTTLE)
    yaw      = read_axis(PHYS_AXIS_YAW, INVERT_YAW)

    return (roll, pitch, throttle, yaw)


# ── Допоміжні функції ────────────────────────────────────────────────────────

def find_window_rect(title_substring: str) -> tuple | None:
    result = []
    def _enum_cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            text = win32gui.GetWindowText(hwnd)
            if title_substring.lower() in text.lower():
                result.append(win32gui.GetWindowRect(hwnd))
    win32gui.EnumWindows(_enum_cb, None)
    return result[0] if result else None


def draw_crosshair(frame: np.ndarray, cx: int, cy: int) -> None:
    cv2.line(frame, (cx - CROSSHAIR_SIZE, cy), (cx + CROSSHAIR_SIZE, cy),
             CROSSHAIR_COLOR, CROSSHAIR_THICKNESS)
    cv2.line(frame, (cx, cy - CROSSHAIR_SIZE), (cx, cy + CROSSHAIR_SIZE),
             CROSSHAIR_COLOR, CROSSHAIR_THICKNESS)


def get_all_cars(results) -> list:
    """Повертає список всіх знайдених машин: [(x1,y1,x2,y2, conf, track_id), ...]"""
    cars = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == CAR_CLASS_ID and conf >= CONFIDENCE_THRESHOLD:
            track_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cars.append((int(x1), int(y1), int(x2), int(y2), conf, track_id))
    return cars


def pick_closest_to_center(cars: list, cx: int, cy: int) -> tuple | None:
    """З списку машин обирає найближчу до центру екрану."""
    best = None
    best_dist = float('inf')
    for car in cars:
        x1, y1, x2, y2 = car[:4]
        car_cx = (x1 + x2) // 2
        car_cy = (y1 + y2) // 2
        dist = (car_cx - cx) ** 2 + (car_cy - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best = car
    return best


def find_car_by_track_id(results, target_id: int) -> tuple | None:
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == CAR_CLASS_ID and conf >= CONFIDENCE_THRESHOLD:
            track_id = int(box.id[0]) if box.id is not None else -1
            if track_id == target_id:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                return (int(x1), int(y1), int(x2), int(y2), conf)
    return None


# ── Головний цикл ───────────────────────────────────────────────────────────

def main() -> None:
    model = YOLO(MODEL_PATH)

    # Шукаємо вікно Uncrashed
    rect = find_window_rect(WINDOW_TITLE)
    if rect is None:
        print(f"[ERROR] Вікно '{WINDOW_TITLE}' не знайдено. Запусти симулятор.")
        return

    left, top, right, bottom = rect
    SCREEN_W, SCREEN_H = 2560, 1440
    left = max(0, left)
    top = max(0, top)
    right = min(SCREEN_W, right)
    bottom = min(SCREEN_H, bottom)
    width = right - left
    height = bottom - top
    region = (left, top, right, bottom)
    cx_screen = width // 2
    cy_screen = height // 2

    camera = dxcam.create(output_color="BGR")
    camera.start(region=region, target_fps=75)  # streaming mode — не блокує

    # ── Overlay вікно (поверх гри) ──
    game_overlay = None
    if DISPLAY_MODE == "overlay":
        game_overlay = GameOverlay(left, top, width, height)
        print(f"[INFO] Overlay поверх гри ({width}x{height})")

    # ── Фізичний контролер ──
    physical_joy = init_physical_joystick()

    # ── Віртуальний геймпад ──
    print("[INFO] Створюю віртуальний Xbox 360 контролер...")
    gamepad = vg.VX360Gamepad()
    gamepad.left_joystick_float(x_value_float=0.0, y_value_float=-1.0)
    gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
    gamepad.update()
    time.sleep(0.5)

    # ── Стан ──
    auto_mode = False
    locked_track_id = None
    closest_track_id = None     # найближча машина в MANUAL (для локу при перемиканні)
    frames_without_target = 0  # лічильник grace period
    last_target = None          # остання відома позиція цілі

    pid_roll = PID(PID_ROLL_KP, PID_ROLL_KI, PID_ROLL_KD,
                   setpoint=0, output_limits=(-MAX_STICK, MAX_STICK),
                   sample_time=None, differential_on_measurement=True)
    pid_pitch = PID(PID_PITCH_KP, PID_PITCH_KI, PID_PITCH_KD,
                    setpoint=0, output_limits=(-MAX_STICK, MAX_STICK),
                    sample_time=None, differential_on_measurement=True)
    pid_yaw = PID(PID_YAW_KP, PID_YAW_KI, PID_YAW_KD,
                  setpoint=0, output_limits=(-MAX_STICK, MAX_STICK),
                  sample_time=None, differential_on_measurement=True)

    rl_roll = RateLimiter(RATE_LIMIT)
    rl_pitch = RateLimiter(RATE_LIMIT)
    rl_yaw = RateLimiter(RATE_LIMIT)

    def switch_to_manual():
        nonlocal auto_mode, locked_track_id, frames_without_target, last_target
        auto_mode = False
        locked_track_id = None
        frames_without_target = 0
        last_target = None
        pid_roll.reset()
        pid_pitch.reset()
        pid_yaw.reset()
        rl_roll.reset()
        rl_pitch.reset()
        rl_yaw.reset()
        print("\n[MODE] MANUAL — passthrough фізичного контролера")

    def toggle_mode(_event=None):
        nonlocal auto_mode, locked_track_id
        if auto_mode:
            switch_to_manual()
        else:
            auto_mode = True
            # Лок на найближчу машину з MANUAL режиму
            if closest_track_id is not None:
                locked_track_id = closest_track_id
                print(f"\n[MODE] AUTO — захоплено ID={locked_track_id}")
            else:
                print("\n[MODE] AUTO — чекаю ціль для захоплення...")

    keyboard.on_press_key("space", toggle_mode)

    # Q — вихід (працює і в overlay, і в window режимі)
    quit_flag = False
    def on_quit(_event=None):
        nonlocal quit_flag
        quit_flag = True
    keyboard.on_press_key("q", on_quit)

    print(f"[INFO] Вікно '{WINDOW_TITLE}': {width}x{height}")
    print("[INFO] ПРОБІЛ — перемкнути MANUAL / AUTO")
    print("[INFO] 'q' у вікні OpenCV — вихід")
    print("[INFO] В Uncrashed прив'яжи керування до Xbox 360 Controller!\n")
    print("[MODE] MANUAL — passthrough фізичного контролера\n")

    # ── Async inference: окремий потік для model.track() ──
    _infer_lock = threading.Lock()
    _latest_results = None      # останні результати детекції
    _infer_fps = 0.0
    _infer_seq = 0              # лічильник нових inference кадрів
    _infer_running = True

    def _inference_loop():
        nonlocal _latest_results, _infer_fps, _infer_seq
        while _infer_running:
            f = camera.get_latest_frame()
            if f is None:
                time.sleep(0.001)
                continue
            # --- Попередній ресайз для максимального FPS ---
            if f.shape[0] != IMGSZ or f.shape[1] != IMGSZ:
                f = cv2.resize(f, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
            t = time.perf_counter()
            res = model.track(f, verbose=False, conf=CONFIDENCE_THRESHOLD,
                              classes=[CAR_CLASS_ID], device="cuda", persist=True,
                              imgsz=IMGSZ, half=True)
            dt = time.perf_counter() - t
            _infer_fps = 1.0 / (dt + 1e-9)
            with _infer_lock:
                _latest_results = res
                _infer_seq += 1

    infer_thread = threading.Thread(target=_inference_loop, daemon=True)
    infer_thread.start()

    frame_count = 0
    # Профілювання (середні часи кожного етапу)
    _prof_capture = 0.0
    _prof_infer = 0.0
    _prof_logic = 0.0
    _prof_draw = 0.0
    _prof_overlay = 0.0
    _prof_total = 0.0
    _prof_n = 0
    prev_infer_seq = 0
    try:
        while True:
            t0 = time.perf_counter()
            frame_count += 1
            roll = pitch = yaw = 0.0
            auto_throttle = THROTTLE_BASE

            if frame_count % 600 == 0:
                updated = find_window_rect(WINDOW_TITLE)
                if updated is not None:
                    left, top, right, bottom = updated
                    left, top = max(0, left), max(0, top)
                    right, bottom = min(SCREEN_W, right), min(SCREEN_H, bottom)
                    width, height = right - left, bottom - top
                    new_region = (left, top, right, bottom)
                    if new_region != region:
                        region = new_region
                        camera.stop()
                        camera.start(region=region, target_fps=75)
                        if game_overlay:
                            game_overlay.reposition(left, top, width, height)
                    cx_screen, cy_screen = width // 2, height // 2

            # Захоплення: main loop НЕ чекає dxcam — inference у фоновому потоці
            t_cap = time.perf_counter()
            with _infer_lock:
                results = _latest_results
                infer_seq = _infer_seq
            t_cap = time.perf_counter() - t_cap

            if results is None:
                time.sleep(0.001)
                continue  # ще немає першого результату

            new_infer = (infer_seq != prev_infer_seq)
            prev_infer_seq = infer_seq

            # ── Вибір цілі ──
            t_logic = time.perf_counter()
            all_cars = get_all_cars(results)
            target = None

            if auto_mode:
                if locked_track_id is not None:
                    # Шукаємо ТІЛЬКИ залочену ціль
                    target = find_car_by_track_id(results, locked_track_id)
                    if target is not None:
                        frames_without_target = 0
                        last_target = target
                    else:
                        if new_infer:
                            frames_without_target += 1
                        if frames_without_target <= GRACE_FRAMES:
                            target = last_target
                        else:
                            print(f"\n[ВТРАТА] Ціль ID={locked_track_id} зникла на {GRACE_FRAMES} кадрів → MANUAL")
                            switch_to_manual()
                else:
                    # Ще не залочили — беремо найближчу до центру
                    closest = pick_closest_to_center(all_cars, cx_screen, cy_screen)
                    if closest is not None:
                        locked_track_id = closest[5]
                        target = closest[:5]
                        last_target = target
                        frames_without_target = 0
                        print(f"\n[ЗАХОПЛЕНО] Ціль ID={locked_track_id}")

            if not auto_mode:
                # MANUAL: відстежуємо найближчу до центру для стрілки і майбутнього локу
                closest = pick_closest_to_center(all_cars, cx_screen, cy_screen)
                if closest is not None:
                    closest_track_id = closest[5]
                    target = closest[:5]  # для стрілки
                else:
                    closest_track_id = None

            # ── Керування геймпадом ──
            if auto_mode:
                if target is not None:
                    x1, y1, x2, y2, conf = target
                    # --- автонаведення: error_x/y у координатах IMGSZ (детекції) ---
                    cx_target = (x1 + x2) // 2
                    cy_target = (y1 + y2) // 2
                    error_x = (cx_target - (IMGSZ // 2)) / (IMGSZ // 2)
                    error_y = (cy_target - (IMGSZ // 2)) / (IMGSZ // 2)

                    # simple-pid: error = setpoint(0) - input, тому передаємо -expo
                    roll = rl_roll(pid_roll(-expo_curve(error_x)))
                    pitch = rl_pitch(pid_pitch(-expo_curve(error_y)))
                    yaw = rl_yaw(pid_yaw(-expo_curve(error_x)))



                    # Авто-газ: базовий + додатковий залежно від відстані
                    bbox_h = (y2 - y1) / IMGSZ  # відн. розмір bbox у просторі моделі
                    # Якщо bbox маленький (далеко) → більше газу, якщо великий (близько) → менше
                    distance_factor = max(0.0, 1.0 - bbox_h / TARGET_SIZE_CLOSE)
                    auto_throttle = min(THROTTLE_MAX, THROTTLE_BASE + THROTTLE_APPROACH * distance_factor)

                    # Зменшуємо газ при великій вертикальній помилці (щоб не пролітати над ціллю)
                    vert_error_abs = abs(error_y)
                    if vert_error_abs > THROTTLE_ERROR_CUT:
                        auto_throttle *= max(0.3, 1.0 - (vert_error_abs - THROTTLE_ERROR_CUT))
                else:
                    roll = rl_roll(0.0)
                    pitch = rl_pitch(0.0)
                    yaw = rl_yaw(0.0)
                    auto_throttle = THROTTLE_BASE  # тримаємо висоту

                throttle_stick = -1.0 + auto_throttle * 2.0
                gamepad.right_joystick_float(x_value_float=roll, y_value_float=pitch)
                gamepad.left_joystick_float(x_value_float=yaw, y_value_float=throttle_stick)
            else:
                # MANUAL — passthrough фізичного контролера
                p_roll, p_pitch, p_throttle, p_yaw = read_physical(physical_joy)
                gamepad.right_joystick_float(x_value_float=p_roll, y_value_float=p_pitch)
                gamepad.left_joystick_float(x_value_float=p_yaw, y_value_float=p_throttle)

            gamepad.update()
            t_logic = time.perf_counter() - t_logic

            # ── Малюємо ──
            t_draw = time.perf_counter()
            overlay = np.zeros((height, width, 3), dtype=np.uint8)
            draw_crosshair(overlay, cx_screen, cy_screen)
            # Коефіцієнти масштабування bbox (з IMGSZ → overlay)
            scale_x = width / IMGSZ
            scale_y = height / IMGSZ

            mode_text = "AUTO" if auto_mode else "MANUAL"
            mode_color = (0, 0, 255) if auto_mode else (0, 255, 0)
            cv2.putText(overlay, mode_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, mode_color, 3)

            if not auto_mode:
                # MANUAL: малюємо ВСІ машини (без ліній і відстаней)
                for car in all_cars:
                    cx1, cy1, cx2, cy2, cconf, ctid = car
                    col = BBOX_COLOR
                    sx1, sy1, sx2, sy2 = int(cx1 * scale_x), int(cy1 * scale_y), int(cx2 * scale_x), int(cy2 * scale_y)
                    cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), col, BBOX_THICKNESS)
                    cv2.putText(overlay, f"car {cconf:.0%} ID:{ctid}", (sx1, sy1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

                if target is not None:
                    x1, y1, x2, y2, conf = target
                    sx1, sy1, sx2, sy2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                    cx_target = (sx1 + sx2) // 2
                    cy_target = (sy1 + sy2) // 2
                    # cv2.line(overlay, (cx_screen, cy_screen), (cx_target, cy_target),
                    #          LINE_COLOR, LINE_THICKNESS)
                    delta_x = cx_target - cx_screen
                    delta_y = cy_target - cy_screen
                    fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
                    print(f"[MANUAL] →ID:{closest_track_id} dx={delta_x:+5d} dy={delta_y:+5d} | "
                          f"cars:{len(all_cars)} | FPS: {fps:.1f}   ", end="\r")
                else:
                    fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
                    print(f"[MANUAL] NO TARGET | FPS: {fps:.1f}   ", end="\r")

            else:
                # AUTO: малюємо ВСІ машини, лінію тільки до залоченої
                for car in all_cars:
                    cx1, cy1, cx2, cy2, cconf, ctid = car
                    is_locked = (ctid == locked_track_id)
                    col = BBOX_COLOR_LOCKED if is_locked else BBOX_COLOR
                    sx1, sy1, sx2, sy2 = int(cx1 * scale_x), int(cy1 * scale_y), int(cx2 * scale_x), int(cy2 * scale_y)
                    cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), col, BBOX_THICKNESS)
                    cv2.putText(overlay, f"car {cconf:.0%} ID:{ctid}", (sx1, sy1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

                if target is not None:
                    x1, y1, x2, y2, conf = target
                    sx1, sy1, sx2, sy2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                    cx_target = (sx1 + sx2) // 2
                    cy_target = (sy1 + sy2) // 2
                    delta_x = cx_target - cx_screen
                    delta_y = cy_target - cy_screen

                    cv2.line(overlay, (cx_screen, cy_screen), (cx_target, cy_target),
                             LINE_COLOR, LINE_THICKNESS)

                    grace_text = f" grace:{frames_without_target}" if frames_without_target > 0 else ""
                    cv2.putText(overlay, f"R:{roll:+.2f} P:{pitch:+.2f} Y:{yaw:+.2f} T:{auto_throttle:.0%}{grace_text}",
                                (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(overlay, f"LOCKED ID:{locked_track_id}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                    fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
                    print(f"[AUTO] dx={delta_x:+5d} dy={delta_y:+5d} | "
                          f"R={roll:+.2f} P={pitch:+.2f} Y={yaw:+.2f} T={auto_throttle:.0%} | "
                          f"FPS: {fps:.1f}   ", end="\r")
                else:
                    fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
                    print(f"[AUTO] NO TARGET | FPS: {fps:.1f}   ", end="\r")

            t_draw = time.perf_counter() - t_draw

            t_ovr = time.perf_counter()
            if DISPLAY_MODE == "overlay":
                game_overlay.update_async(overlay)  # фоновий потік — не блокує
                win32gui.PumpWaitingMessages()
                if quit_flag:
                    break
            else:
                cv2.imshow("Drone Visual Aim", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or quit_flag:
                    break
            t_ovr = time.perf_counter() - t_ovr

            # Профілювання (ковзне середнє)
            _prof_n += 1
            a = 0.05  # коефіцієнт згладжування
            _prof_capture = _prof_capture * (1 - a) + t_cap * a
            _prof_logic = _prof_logic * (1 - a) + t_logic * a
            _prof_draw = _prof_draw * (1 - a) + t_draw * a
            _prof_overlay = _prof_overlay * (1 - a) + t_ovr * a
            _prof_total = _prof_total * (1 - a) + (time.perf_counter() - t0) * a

            if _prof_n % 60 == 0:
                fps = 1.0 / (_prof_total + 1e-9)
                print(f"\n[PROF] cap={_prof_capture*1000:.1f}ms  logic={_prof_logic*1000:.1f}ms  "
                      f"draw={_prof_draw*1000:.1f}ms  overlay={_prof_overlay*1000:.1f}ms  "
                      f"TOTAL={_prof_total*1000:.1f}ms  loopFPS={fps:.0f}  inferFPS={_infer_fps:.0f}")
    finally:
        _infer_running = False
        infer_thread.join(timeout=2)
        if game_overlay:
            game_overlay.wait_done()  # дочекатися фонового рендеру перед cleanup
        gamepad.left_joystick_float(x_value_float=0.0, y_value_float=-1.0)
        gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
        gamepad.update()
        keyboard.unhook_all()
        pygame.quit()
        if game_overlay:
            game_overlay.destroy()
        camera.stop()
        del camera

    cv2.destroyAllWindows()
    print("\n[INFO] Завершено.")


if __name__ == "__main__":
    main()
