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

# ── Налаштування ────────────────────────────────────────────────────────────

WINDOW_TITLE = "Uncrashed"
# MODEL_PATH = "runs/detect/runs/detect/uncrashed_cars9/weights/best.engine"  # шлях до TensorRT моделі (експорт з train_uncrashed.py)
MODEL_PATH = "yolov8s.pt"  # шлях до TensorRT моделі (експорт з train_uncrashed.py)
CAR_CLASS_ID = 2
CONFIDENCE_THRESHOLD = 0.20
IMGSZ = 960  # розмір інференсу (менше = швидше, 640→384 ≈ +60% FPS)

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

# ── Автонаведення + таран (PID + фази атаки) ───────────────────────────────
# Дрон працює в режимі ACRO: стіки задають ШВИДКІСТЬ ОБЕРТАННЯ, не кут.
# Коли стік = 0, дрон тримає поточний кут (не вирівнюється).
# Yaw   → швидкість повороту вліво/вправо
# Pitch → швидкість нахилу вперед/назад (НЕ тримати — дрон перекрутиться!)
# Throttle → тяга

# PID для YAW (горизонтальне наведення — yaw rate)
PID_YAW_KP = 0.6
PID_YAW_KI = 0.02
PID_YAW_KD = 0.12

# PID для THROTTLE (вертикальне наведення)
PID_THR_KP = 0.5
PID_THR_KI = 0.02
PID_THR_KD = 0.10

# PID для PITCH (керує pitch rate щоб тримати ціль в центрі по Y + додає нахил вперед)
# В Acro: ціль нижче центру → нахилити вперед (pitch rate +), вище → назад (-)
PID_PITCH_KP = 0.25
PID_PITCH_KI = 0.01
PID_PITCH_KD = 0.08

# Базовий газ
BASE_THROTTLE_NORM = 0.40

# EMA-згладжування виходу (0..1, менше = плавніше)
SMOOTH_ALPHA = 0.3

# Базовий pitch rate вперед (постійно підкручує дрон вперед, щоб летіти до цілі)
# В Acro це RATE, не кут! Маленьке значення = повільно нахиляється
BASE_PITCH_RATE        = 0.06    # постійна добавка pitch вперед
PHASE_ATTACK_RATIO     = 0.12    # bbox > 12% екрану → атака
PHASE_ATTACK_PITCH_ADD = 0.10    # додатковий pitch rate в атаці
PHASE_TERMINAL_RATIO   = 0.28    # bbox > 28% екрану → термінальна
PHASE_TERMINAL_PITCH_ADD = 0.20  # агресивний pitch rate для тарану

# Максимальні значення PID-виходу
PID_OUTPUT_MAX = 0.5
PID_PITCH_MAX  = 0.3   # макс pitch rate від PID (щоб не перекрутити дрон)


class PIDController:
    """Дискретний PID-регулятор з anti-windup."""

    def __init__(self, kp: float, ki: float, kd: float, output_max: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_max = output_max
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def update(self, error: float) -> float:
        now = time.perf_counter()
        if self._prev_time is None:
            self._prev_time = now
            self._prev_error = error
            return self.kp * error  # перший кадр — тільки P

        dt = now - self._prev_time
        if dt < 1e-6:
            return 0.0
        self._prev_time = now

        # P
        p = self.kp * error
        # I з anti-windup
        self._integral += error * dt
        i_limit = self.output_max / max(self.ki, 1e-9)
        self._integral = max(-i_limit, min(i_limit, self._integral))
        i = self.ki * self._integral
        # D
        d = self.kd * (error - self._prev_error) / dt
        self._prev_error = error

        out = p + i + d
        return max(-self.output_max, min(self.output_max, out))


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

    # ── PID контролери для автонаведення ──
    pid_yaw = PIDController(PID_YAW_KP, PID_YAW_KI, PID_YAW_KD, PID_OUTPUT_MAX)
    pid_thr = PIDController(PID_THR_KP, PID_THR_KI, PID_THR_KD, PID_OUTPUT_MAX)
    pid_pitch = PIDController(PID_PITCH_KP, PID_PITCH_KI, PID_PITCH_KD, PID_PITCH_MAX)
    # EMA-згладжені значення стіків
    _smooth_yaw = 0.0
    _smooth_thr = -1.0  # стартуємо з 0 газу
    _smooth_pitch = 0.0

    # ── Стан ──
    # Режими: MANUAL → LOCK → AUTO → MANUAL
    # MANUAL: passthrough джойстіка, підсвічує найближчу машину
    # LOCK:   ціль залоковано, трекінг працює, але керування з джойстіка
    # AUTO:   автонаведення + таран
    flight_mode = "MANUAL"      # MANUAL | LOCK | AUTO
    locked_track_id = None
    closest_track_id = None
    lost_since = None
    LOST_TIMEOUT = 1.0
    attack_phase = "SEARCH"

    def switch_to_manual():
        nonlocal flight_mode, locked_track_id, lost_since, attack_phase
        flight_mode = "MANUAL"
        locked_track_id = None
        lost_since = None
        attack_phase = "SEARCH"
        pid_yaw.reset()
        pid_thr.reset()
        pid_pitch.reset()
        print("\n[MODE] MANUAL — passthrough фізичного контролера")

    def toggle_mode(_event=None):
        nonlocal flight_mode, locked_track_id, attack_phase
        if flight_mode == "MANUAL":
            # MANUAL → LOCK: локуємо найближчу машину
            if closest_track_id is not None:
                flight_mode = "LOCK"
                locked_track_id = closest_track_id
                print(f"\n[MODE] LOCK — ціль ID={locked_track_id} залоковано, керування з джойстіка")
            else:
                print("\n[НЕМАЄ ЦІЛІ] Немає машини для локу")
        elif flight_mode == "LOCK":
            # LOCK → AUTO: атакуємо залоковану ціль
            flight_mode = "AUTO"
            attack_phase = "APPROACH"
            pid_yaw.reset()
            pid_thr.reset()
            pid_pitch.reset()
            print(f"\n[MODE] AUTO — атака на ID={locked_track_id}!")
        elif flight_mode == "AUTO":
            # AUTO → MANUAL: скидаємо все
            switch_to_manual()

    keyboard.on_press_key("space", toggle_mode)

    # Q — вихід (працює і в overlay, і в window режимі)
    quit_flag = False
    def on_quit(_event=None):
        nonlocal quit_flag
        quit_flag = True
    keyboard.on_press_key("q", on_quit)

    print(f"[INFO] Вікно '{WINDOW_TITLE}': {width}x{height}")
    print("[INFO] ПРОБІЛ — MANUAL → LOCK → AUTO → MANUAL")
    print("[INFO] 'q' — вихід")
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
            # if f.shape[0] != IMGSZ or f.shape[1] != IMGSZ:
            #     f = cv2.resize(f, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
            t = time.perf_counter()
            res = model.track(f, verbose=False, conf=CONFIDENCE_THRESHOLD,
                              classes=[CAR_CLASS_ID], device="cuda", persist=True,
                              imgsz=IMGSZ, half=True)
            # res = model.predict(f, verbose=False, conf=CONFIDENCE_THRESHOLD,
            #                   classes=[CAR_CLASS_ID], device="cuda",
            #                   imgsz=IMGSZ, half=True)
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
    try:
        while True:
            t0 = time.perf_counter()
            frame_count += 1

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

            # ── Вибір цілі ──
            t_logic = time.perf_counter()
            all_cars = get_all_cars(results)
            target = None
            auto_mode = (flight_mode == "AUTO")

            if auto_mode:
                if locked_track_id is not None:
                    target = find_car_by_track_id(results, locked_track_id)
                    if target is not None:
                        lost_since = None
                    else:
                        if lost_since is None:
                            lost_since = time.perf_counter()
                        if (time.perf_counter() - lost_since) >= LOST_TIMEOUT:
                            print(f"\n[ВТРАТА] Ціль ID={locked_track_id} зникла на {LOST_TIMEOUT}с → MANUAL")
                            switch_to_manual()

            elif flight_mode == "LOCK":
                # LOCK: трекінг цілі, але керування з джойстіка
                if locked_track_id is not None:
                    target = find_car_by_track_id(results, locked_track_id)
                    if target is not None:
                        lost_since = None
                    else:
                        if lost_since is None:
                            lost_since = time.perf_counter()
                        if (time.perf_counter() - lost_since) >= LOST_TIMEOUT:
                            print(f"\n[ВТРАТА] Ціль ID={locked_track_id} зникла → MANUAL")
                            switch_to_manual()

            else:
                # MANUAL: підсвічуємо найближчу
                closest = pick_closest_to_center(all_cars, cx_screen, cy_screen)
                if closest is not None:
                    closest_track_id = closest[5]
                    target = closest[:5]
                else:
                    closest_track_id = None

            # ── Керування геймпадом ──
            if flight_mode != "AUTO":
                p_roll, p_pitch, p_throttle, p_yaw = read_physical(physical_joy)

            if auto_mode and target is not None:
                # ── АВТОНАВЕДЕННЯ + ТАРАН ──
                x1, y1, x2, y2, conf = target
                cx_target = (x1 + x2) // 2
                cy_target = (y1 + y2) // 2
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                bbox_ratio = bbox_w / max(width, 1)  # розмір цілі відносно екрану

                # Нормалізована похибка: -1..+1 (де 0 = центр екрану)
                err_x = (cx_target - cx_screen) / (width / 2)    # >0 = ціль справа
                err_y = (cy_target - cy_screen) / (height / 2)   # >0 = ціль знизу

                # --- Визначення фази атаки ---
                if bbox_ratio >= PHASE_TERMINAL_RATIO:
                    attack_phase = "TERMINAL"
                elif bbox_ratio >= PHASE_ATTACK_RATIO:
                    attack_phase = "ATTACK"
                else:
                    attack_phase = "APPROACH"

                # --- PID: yaw для горизонтального наведення ---
                yaw_cmd = pid_yaw.update(err_x)

                # --- PID: throttle для вертикального наведення ---
                # err_y > 0 → ціль нижче центру → дрон занадто високо → менше газу
                # err_y < 0 → ціль вище центру → дрон занадто низько → більше газу
                thr_correction = pid_thr.update(err_y)
                base_thr_stick = -1.0 + BASE_THROTTLE_NORM * 2.0
                throttle_raw = base_thr_stick - thr_correction
                throttle_raw = max(-1.0, min(1.0, throttle_raw))

                # --- PID: pitch rate (ACRO) ---
                # Базовий pitch rate вперед + PID корекція на основі err_y
                # Ціль нижче центру (err_y>0) → нахилити вперед (+pitch rate)
                # Ціль вище центру (err_y<0) → нахилити назад (-pitch rate)
                pitch_pid = pid_pitch.update(err_y)

                if attack_phase == "TERMINAL":
                    base_pitch = BASE_PITCH_RATE + PHASE_TERMINAL_PITCH_ADD
                elif attack_phase == "ATTACK":
                    base_pitch = BASE_PITCH_RATE + PHASE_ATTACK_PITCH_ADD
                else:  # APPROACH
                    base_pitch = BASE_PITCH_RATE

                pitch_raw = base_pitch + pitch_pid
                pitch_raw = max(-PID_PITCH_MAX * 2, min(1.0, pitch_raw))

                # --- EMA-згладжування ---
                _smooth_yaw = _smooth_yaw * (1 - SMOOTH_ALPHA) + yaw_cmd * SMOOTH_ALPHA
                _smooth_thr = _smooth_thr * (1 - SMOOTH_ALPHA) + throttle_raw * SMOOTH_ALPHA
                _smooth_pitch = _smooth_pitch * (1 - SMOOTH_ALPHA) + pitch_raw * SMOOTH_ALPHA

                roll_cmd = 0.0

                # --- Відправка на віртуальний геймпад ---
                gamepad.left_joystick_float(
                    x_value_float=max(-1.0, min(1.0, _smooth_yaw)),
                    y_value_float=max(-1.0, min(1.0, _smooth_thr))
                )
                gamepad.right_joystick_float(
                    x_value_float=roll_cmd,
                    y_value_float=max(-1.0, min(1.0, _smooth_pitch))
                )
            else:
                if auto_mode:
                    # ── AUTO без цілі → тримати останній курс + pitch=0 (в acro не докручувати) ──
                    gamepad.left_joystick_float(
                        x_value_float=max(-1.0, min(1.0, _smooth_yaw)),
                        y_value_float=max(-1.0, min(1.0, _smooth_thr))
                    )
                    gamepad.right_joystick_float(
                        x_value_float=0.0,
                        y_value_float=0.0  # pitch=0 — тримати поточний кут в acro
                    )
                else:
                    # ── MANUAL → passthrough ──
                    gamepad.right_joystick_float(x_value_float=p_roll, y_value_float=p_pitch)
                    gamepad.left_joystick_float(x_value_float=p_yaw, y_value_float=p_throttle)

            gamepad.update()
            t_logic = time.perf_counter() - t_logic

            # ── Малюємо ──
            t_draw = time.perf_counter()
            overlay = np.zeros((height, width, 3), dtype=np.uint8)
            draw_crosshair(overlay, cx_screen, cy_screen)
            scale_x = 1
            scale_y = 1

            mode_text = flight_mode
            mode_colors = {"MANUAL": (0, 255, 0), "LOCK": (0, 255, 255), "AUTO": (0, 0, 255)}
            mode_color = mode_colors.get(flight_mode, (255, 255, 255))
            cv2.putText(overlay, mode_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, mode_color, 3)

            # Малюємо ВСІ машини
            for car in all_cars:
                cx1, cy1, cx2, cy2, cconf, ctid = car
                is_locked = (flight_mode in ("LOCK", "AUTO")) and (ctid == locked_track_id)
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
                bbox_w_draw = sx2 - sx1
                bbox_ratio_draw = bbox_w_draw / max(width, 1)

                if flight_mode == "LOCK":
                    # LOCK: лінія до цілі + інфо
                    cv2.line(overlay, (cx_screen, cy_screen), (cx_target, cy_target),
                             (0, 255, 255), LINE_THICKNESS)
                    cv2.putText(overlay, f"LOCKED ID:{locked_track_id} [ПРОБІЛ = АТАКА]",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.circle(overlay, (cx_target, cy_target), 5, (0, 255, 255), -1)

                elif auto_mode:
                    # AUTO: лінія + фаза + прогрес
                    cv2.line(overlay, (cx_screen, cy_screen), (cx_target, cy_target),
                             LINE_COLOR, LINE_THICKNESS)

                    # Колір фази
                    phase_colors = {
                        "SEARCH": (128, 128, 128),
                        "APPROACH": (0, 255, 255),   # жовтий
                        "ATTACK": (0, 165, 255),     # оранжевий
                        "TERMINAL": (0, 0, 255),     # червоний
                    }
                    ph_color = phase_colors.get(attack_phase, (255, 255, 255))

                    cv2.putText(overlay, f"LOCKED ID:{locked_track_id}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    cv2.putText(overlay, f"PHASE: {attack_phase}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ph_color, 2)
                    cv2.putText(overlay, f"BBOX: {bbox_ratio_draw:.0%}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ph_color, 2)

                    # Прогрес-бар наближення
                    bar_y = 140
                    bar_w = 200
                    bar_h = 16
                    fill = min(1.0, bbox_ratio_draw / PHASE_TERMINAL_RATIO)
                    cv2.rectangle(overlay, (10, bar_y), (10 + bar_w, bar_y + bar_h), (80, 80, 80), -1)
                    cv2.rectangle(overlay, (10, bar_y), (10 + int(bar_w * fill), bar_y + bar_h), ph_color, -1)
                    cv2.putText(overlay, "IMPACT", (10 + bar_w + 5, bar_y + 13),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ph_color, 1)

                    # Маленька точка на цілі
                    cv2.circle(overlay, (cx_target, cy_target), 5, ph_color, -1)

                fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
                print(f"[{mode_text}|{attack_phase:8s}] dx={delta_x:+5d} dy={delta_y:+5d} "
                      f"bbox={bbox_ratio_draw:.0%} | cars:{len(all_cars)} | FPS: {fps:.1f}   ", end="\r")
            else:
                fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
                phase_str = attack_phase if auto_mode else ""
                print(f"[{mode_text}] NO TARGET {phase_str} | FPS: {fps:.1f}   ", end="\r")

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
