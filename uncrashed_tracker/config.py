"""Конфігурація системи візуального автонаведення."""

# ── Режим вводу ─────────────────────────────────────────────────────────────
# "simulator" — захоплення екрану Uncrashed (DXcam) + джойстік + автопілот
# "camera"    — USB/вбудована камера + лише детекція у вікні OpenCV
# "drone"     — USB-камера + джойстік + автопілот → реальний дрон через USB-VCP
INPUT_MODE = "drone"  # "simulator" | "camera" | "drone"

# ── Камера (для INPUT_MODE = "camera" / "drone") ────────────────────────────
CAMERA_INDEX = 0         # індекс камери (0 = перша камера)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ── Дрон (для INPUT_MODE = "drone") ─────────────────────────────────────────
# COM-порт USB-VCP пульта (Windows: COM5, macOS: /dev/cu.usbmodemXXXX)
DRONE_PORT = "COM5"

# ── Загальні ────────────────────────────────────────────────────────────────

WINDOW_TITLE = "Uncrashed"
# MODEL_PATH = "runs/detect/runs/detect/uncrashed_cars9/weights/best.engine"
MODEL_PATH = "yolo11n.pt"

CAR_CLASS_ID = 2
CONFIDENCE_THRESHOLD = 0.20
IMGSZ = 960

SCREEN_W, SCREEN_H = 2560, 1440

# Режим візуалізації: "overlay" = поверх гри (borderless/windowed), "window" = окреме вікно
DISPLAY_MODE = "overlay"  # "overlay" | "window"

# ── Візуалізація ────────────────────────────────────────────────────────────

CROSSHAIR_COLOR = (0, 255, 0)
CROSSHAIR_SIZE = 20
CROSSHAIR_THICKNESS = 2
BBOX_COLOR = (0, 0, 255)
BBOX_COLOR_LOCKED = (0, 165, 255)
BBOX_THICKNESS = 2
LINE_COLOR = (255, 255, 0)
LINE_THICKNESS = 2

# ── PID для YAW (горизонтальне наведення — yaw rate) ───────────────────────

PID_YAW_KP = 0.6
PID_YAW_KI = 0.02
PID_YAW_KD = 0.12

# ── PID для THROTTLE (вертикальне наведення) ────────────────────────────────

PID_THR_KP = 0.5
PID_THR_KI = 0.02
PID_THR_KD = 0.10

# ── PID для PITCH (pitch rate в ACRO) ──────────────────────────────────────

PID_PITCH_KP = 0.25
PID_PITCH_KI = 0.01
PID_PITCH_KD = 0.08

# ── Автонаведення + таран ───────────────────────────────────────────────────

BASE_THROTTLE_NORM = 0.40
SMOOTH_ALPHA = 0.3
BASE_PITCH_RATE = 0.06
PHASE_ATTACK_RATIO = 0.12
PHASE_ATTACK_PITCH_ADD = 0.10
PHASE_TERMINAL_RATIO = 0.28
PHASE_TERMINAL_PITCH_ADD = 0.20
PID_OUTPUT_MAX = 0.5
PID_PITCH_MAX = 0.3

# ── Маппінг фізичного контролера (pygame axes) ─────────────────────────────

PHYS_AXIS_ROLL = 0
PHYS_AXIS_PITCH = 1
PHYS_AXIS_THROTTLE = 2
PHYS_AXIS_YAW = 3

INVERT_ROLL = False
INVERT_PITCH = False
INVERT_THROTTLE = False
INVERT_YAW = False

# ── Таймаути ────────────────────────────────────────────────────────────────

LOST_TIMEOUT = 1.0

# ── Brake (optical flow гальмування, клавіша B) ────────────────────────────

BRAKE_ROLL_KP = 0.0          # X-axis: вліво/вправо → Roll
BRAKE_ROLL_KI = 0.01
BRAKE_ROLL_KD = 0.08

BRAKE_THR_KP = 0.05           # Y-axis: вверх/вниз → Throttle
BRAKE_THR_KI = 0.03
BRAKE_THR_KD = 0.000

BRAKE_PITCH_KP = 0.0         # Z-axis: вперед/назад (дивергенція) → Pitch
BRAKE_PITCH_KI = 0.0
BRAKE_PITCH_KD = 0.20

BRAKE_OUTPUT_MAX = 0.5
BRAKE_ROLL_SMOOTH_ALPHA = 0.9  # EMA/сек для roll (0.9 = ~90% затухає за 1с, імпульсний)
BRAKE_THR_SMOOTH_ALPHA = 0.0   # EMA/сек для throttle (0.9 = ~90% затухає до base за 1с, імпульсний)
BRAKE_PITCH_SMOOTH_ALPHA = 0.9 # EMA/сек для pitch (0.9 = ~90% затухає за 1с, швидкий імпульс)
BRAKE_FLOW_SCALE = 0.25     # масштаб кадру для optical flow (0.25 = 25%)
BRAKE_FLOW_NORM = 0.08      # нормалізація flow → ~[-1,+1]
BRAKE_BASE_THROTTLE = 0.44  # базовий газ для зависання (0.0=мін, 1.0=макс). Менше = нижче hover
BRAKE_INTERVAL = 0.01         # інтервал обрахунку optical flow (секунди)
