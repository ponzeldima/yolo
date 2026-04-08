"""
Autonomous FPV Drone Hover (5 sec) + Landing — Closed-Loop Control.

Пайплайн:
  1. Захоплення відео (HDMI capture card → cv2.VideoCapture) — окремий потік
  2. Гіроскоп-телеметрія (Pitch/Roll з Serial/COM) — окремий потік
  3. Optical Flow (Lucas-Kanade) з компенсацією обертання камери
  4. PID-контролери (Throttle, Roll, Pitch)
  5. State Machine: HOVER → LAND → DONE
  6. Mock вивід команд у консоль

Апарат: Meteor 75 Pro (без GPS/баро), камера вперед, зображення через HDMI capture.
Телеметрія гіроскопа: пульт → USB Serial → ПК.

Залежності:
    pip install opencv-python numpy pyserial
"""

import math
import threading
import time

import cv2
import numpy as np

# pyserial — опціональний, для реального залізу
try:
    import serial
except ImportError:
    serial = None
    print("[WARN] pyserial не встановлено. GyroReader працюватиме з нулями.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Конфігурація
# ═══════════════════════════════════════════════════════════════════════════════

# -- Відео --
CAMERA_INDEX = 0          # Індекс пристрою HDMI capture card
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# -- Serial (телеметрія гіроскопа) --
SERIAL_PORT = "/dev/cu.usbserial-0001"   # Підстав свій порт
SERIAL_BAUD = 115200

# -- Камера (внутрішні параметри) --
# Наближена фокусна відстань у пікселях.
# Для FPV камери з ~120° HFOV на 640 px:  f ≈ (640/2) / tan(60°) ≈ 185.
# Значення 300 — «безпечний» дефолт; після калібрування замінити.
FOCAL_LENGTH_PX = 300.0

# -- Optical Flow (Lucas-Kanade) --
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
MAX_CORNERS = 200
CORNER_QUALITY = 0.01
CORNER_MIN_DIST = 10
REDETECT_THRESHOLD = 50   # Переобчислити фічі, коли залишилось менше

# -- PID коефіцієнти (TUNE під свій дрон!) --
PID_THROTTLE = {"kp": 0.5, "ki": 0.05, "kd": 0.1}   # Висота (vertical flow)
PID_ROLL     = {"kp": 0.3, "ki": 0.02, "kd": 0.08}   # Бічний дрейф (horizontal flow)
PID_PITCH    = {"kp": 0.3, "ki": 0.02, "kd": 0.08}   # Вперед/назад (divergence)

# -- RC канали (мікросекунди) --
RC_MID = 1500
RC_THROTTLE_HOVER = 1450  # Базовий газ для висіння (tune!)
RC_MIN = 1000
RC_MAX = 2000

# -- Місія --
HOVER_DURATION_SEC = 5.0
LAND_THROTTLE_STEP = 5    # µs зменшення кожну ітерацію при посадці
CONTROL_LOOP_HZ = 30


# ═══════════════════════════════════════════════════════════════════════════════
#  PID Controller
# ═══════════════════════════════════════════════════════════════════════════════

class PIDController:
    """Дискретний PID-регулятор з anti-windup clamping."""

    def __init__(self, kp: float, ki: float, kd: float,
                 output_min: float = -500.0, output_max: float = 500.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        """Обчислити корекцію за поточною похибкою та кроком часу dt (секунди)."""
        if dt <= 0:
            return 0.0

        # P
        p_term = self.kp * error

        # I з anti-windup
        self._integral += error * dt
        integral_limit = self.output_max / max(self.ki, 1e-9)
        self._integral = max(-integral_limit, min(integral_limit, self._integral))
        i_term = self.ki * self._integral

        # D
        d_term = self.kd * (error - self._prev_error) / dt
        self._prev_error = error

        output = p_term + i_term + d_term
        return max(self.output_min, min(self.output_max, output))


# ═══════════════════════════════════════════════════════════════════════════════
#  Gyro Reader  (фоновий потік Serial)
# ═══════════════════════════════════════════════════════════════════════════════

class GyroReader:
    """
    Зчитує кути Pitch / Roll з Serial-порту у фоновому потоці.

    Підтримувані формати на вході (один на рядок):
      • "P:12.5,R:-3.2\\n"            — простий key:value
      • '{"pitch": 12.5, "roll": -3.2}' — JSON

    Доступ до останніх значень — через thread-safe .pitch / .roll.
    """

    def __init__(self, port: str = SERIAL_PORT, baudrate: int = SERIAL_BAUD):
        self._port = port
        self._baudrate = baudrate
        self._lock = threading.Lock()
        self._pitch = 0.0   # градуси
        self._roll = 0.0    # градуси
        self._running = False
        self._thread: threading.Thread | None = None
        self._serial = None

    # -- Thread-safe getters --------------------------------------------------

    @property
    def pitch(self) -> float:
        with self._lock:
            return self._pitch

    @property
    def roll(self) -> float:
        with self._lock:
            return self._roll

    # -- Lifecycle ------------------------------------------------------------

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()

    # -- Internal -------------------------------------------------------------

    def _read_loop(self):
        if serial is None:
            print("[GyroReader] pyserial відсутній — pitch/roll = 0.")
            return

        try:
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                timeout=0.05,
            )
            print(f"[GyroReader] Підключено до {self._port}")
        except Exception as e:
            print(f"[GyroReader] Не вдалось відкрити {self._port}: {e}. Використовуємо 0.")
            return

        while self._running:
            try:
                raw = self._serial.readline()
                line = raw.decode("utf-8", errors="ignore").strip()
                if line:
                    self._parse_line(line)
            except Exception:
                continue

    def _parse_line(self, line: str):
        """Парсить рядок телеметрії.  Підтримує 'P:…,R:…' та JSON."""
        # Спроба 1: простий формат "P:<pitch>,R:<roll>"
        try:
            if line.startswith("P:"):
                parts = line.split(",")
                pitch_val = float(parts[0].split(":")[1])
                roll_val = float(parts[1].split(":")[1])
                with self._lock:
                    self._pitch = pitch_val
                    self._roll = roll_val
                return
        except (IndexError, ValueError):
            pass

        # Спроба 2: JSON
        try:
            import json
            data = json.loads(line)
            with self._lock:
                self._pitch = float(data.get("pitch", self._pitch))
                self._roll = float(data.get("roll", self._roll))
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Threaded Video Capture
# ═══════════════════════════════════════════════════════════════════════════════

class ThreadedVideoCapture:
    """
    Захоплює кадри з cv2.VideoCapture у фоновому потоці.
    Зберігає лише останній кадр — control loop ніколи не блокується.
    """

    def __init__(self, source=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def frame(self) -> np.ndarray | None:
        """Повертає копію останнього кадру або None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._cap.release()

    def _grab_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame


# ═══════════════════════════════════════════════════════════════════════════════
#  Optical Flow Processor  (Lucas-Kanade + компенсація обертання)
# ═══════════════════════════════════════════════════════════════════════════════

class OpticalFlowProcessor:
    """
    Обчислює розріджений оптичний потік (Lucas-Kanade) і компенсує
    обертальну складову за допомогою даних гіроскопа.

    ╔══════════════════════════════════════════════════════════════════════╗
    ║  МАТЕМАТИКА КОМПЕНСАЦІЇ ОБЕРТАННЯ                                  ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║  Модель камери: pinhole з фокусною відстанню f (пікселі)           ║
    ║  та головною точкою (cx, cy).                                      ║
    ║                                                                    ║
    ║  Оптичний потік у пікселі (u, v) = TRANSLATION + ROTATION.         ║
    ║                                                                    ║
    ║  Нехай x = u − cx,  y = v − cy,  а кутові швидкості камери:       ║
    ║    ω_x — обертання навколо горизонтальної осі (= pitch дрона)      ║
    ║    ω_y — обертання навколо вертикальної осі  (= yaw, невідомий)    ║
    ║    ω_z — обертання навколо оптичної осі       (= roll дрона)       ║
    ║                                                                    ║
    ║  Обертальна складова потоку (класичні рівняння, div. за Trucco):    ║
    ║                                                                    ║
    ║    flow_u_rot = (x·y / f)·ω_x − (f + x²/f)·ω_y + y·ω_z          ║
    ║    flow_v_rot = (f + y²/f)·ω_x − (x·y / f)·ω_y − x·ω_z          ║
    ║                                                                    ║
    ║  Ми маємо лише pitch і roll з телеметрії → ω_y = 0:               ║
    ║                                                                    ║
    ║    flow_u_rot = (x·y / f)·ω_x  +  y·ω_z                          ║
    ║    flow_v_rot = (f + y²/f)·ω_x  −  x·ω_z                         ║
    ║                                                                    ║
    ║  TRANSLATIONAL flow = виміряний потік − обертальна складова.        ║
    ║  Це — «чистий» зсув дрона у просторі.                             ║
    ║                                                                    ║
    ║  Інтерпретація (камера вперед):                                    ║
    ║    avg(flow_v_trans) > 0 → дрон опускається (втрата висоти)        ║
    ║    avg(flow_u_trans) > 0 → дрон дрейфує вправо                    ║
    ║    divergence > 0        → дрон летить вперед (розширення потоку)  ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, focal_length: float = FOCAL_LENGTH_PX):
        self.f = focal_length
        self._prev_gray: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None
        self._prev_pitch = 0.0
        self._prev_roll = 0.0

    # ------------------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
        pitch_deg: float,
        roll_deg: float,
    ) -> tuple[float, float, float, np.ndarray]:
        """
        Обробити один кадр.

        Returns:
            flow_x  — середній горизонтальний трансляційний потік (px/frame).
                       Позитивний = дрон дрейфує вправо.
            flow_y  — середній вертикальний трансляційний потік (px/frame).
                       Позитивний = дрон опускається.
            divergence — дивергенція трансляційного потоку (скаляр).
                       Позитивна = рух вперед (наближення до сцени).
            debug   — кадр з візуалізацією.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cx, cy = w / 2.0, h / 2.0

        # ── Перший кадр — ініціалізація ──
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            self._prev_pitch = pitch_deg
            self._prev_roll = roll_deg
            return 0.0, 0.0, 0.0, frame

        # ── Дельта кутів (радіани) між кадрами ──
        # Гіроскоп дає кут → різниця = обертання за один кадр.
        omega_x = math.radians(pitch_deg - self._prev_pitch)   # pitch → cam X
        omega_z = math.radians(roll_deg - self._prev_roll)     # roll  → cam Z

        # ── Переобчислити фічі, якщо мало ──
        if self._prev_points is None or len(self._prev_points) < REDETECT_THRESHOLD:
            self._prev_points = self._detect_features(self._prev_gray)

        if self._prev_points is None or len(self._prev_points) == 0:
            self._update_state(gray, pitch_deg, roll_deg)
            return 0.0, 0.0, 0.0, frame

        # ── Lucas-Kanade ──
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None, **LK_PARAMS,
        )

        if new_points is None:
            self._update_state(gray, pitch_deg, roll_deg, redetect=True)
            return 0.0, 0.0, 0.0, frame

        # Фільтрація «хороших» збігів
        good = status.flatten() == 1
        old_pts = self._prev_points[good].reshape(-1, 2)
        new_pts = new_points[good].reshape(-1, 2)

        if len(old_pts) < 5:
            self._update_state(gray, pitch_deg, roll_deg, redetect=True)
            return 0.0, 0.0, 0.0, frame

        # Виміряний потік
        flow = new_pts - old_pts   # shape (N, 2)

        # ── Компенсація обертання ──
        x = old_pts[:, 0] - cx
        y = old_pts[:, 1] - cy
        f = self.f

        flow_u_rot = (x * y / f) * omega_x + y * omega_z
        flow_v_rot = (f + y ** 2 / f) * omega_x - x * omega_z

        flow_u_trans = flow[:, 0] - flow_u_rot
        flow_v_trans = flow[:, 1] - flow_v_rot

        # ── Робастне середнє (медіана, щоб відсіяти викиди) ──
        avg_u = float(np.median(flow_u_trans))
        avg_v = float(np.median(flow_v_trans))

        # ── Дивергенція (для оцінки руху вперед/назад) ──
        # Радіальна складова потоку = dot(flow_trans, r_hat), де r = (x, y).
        # Позитивна → розширення → наближення до сцени (рух вперед).
        r_len = np.sqrt(x ** 2 + y ** 2)
        r_len = np.where(r_len < 1.0, 1.0, r_len)   # уникнути div/0
        radial = (flow_u_trans * x + flow_v_trans * y) / r_len
        divergence = float(np.median(radial))

        # ── Debug візуалізація ──
        debug = frame.copy()
        for i in range(len(old_pts)):
            a = tuple(old_pts[i].astype(int))
            b = tuple(new_pts[i].astype(int))
            cv2.arrowedLine(debug, a, b, (0, 255, 0), 1, tipLength=0.3)

        arrow_scale = 10.0
        center = (int(cx), int(cy))
        tip = (int(cx + avg_u * arrow_scale), int(cy + avg_v * arrow_scale))
        cv2.arrowedLine(debug, center, tip, (0, 0, 255), 3, tipLength=0.2)

        cv2.putText(debug, f"Trans Flow: u={avg_u:+.2f}  v={avg_v:+.2f}  div={divergence:+.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(debug, f"Gyro: P={pitch_deg:+.1f}  R={roll_deg:+.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ── Оновити стан ──
        self._prev_gray = gray
        self._prev_points = new_pts.reshape(-1, 1, 2)
        self._prev_pitch = pitch_deg
        self._prev_roll = roll_deg

        return avg_u, avg_v, divergence, debug

    # ------------------------------------------------------------------

    def _update_state(self, gray, pitch, roll, redetect=False):
        self._prev_gray = gray
        self._prev_pitch = pitch
        self._prev_roll = roll
        if redetect:
            self._prev_points = self._detect_features(gray)

    @staticmethod
    def _detect_features(gray: np.ndarray) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MAX_CORNERS,
            qualityLevel=CORNER_QUALITY,
            minDistance=CORNER_MIN_DIST,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Command Output  (Mock)
# ═══════════════════════════════════════════════════════════════════════════════

def send_command(throttle: int, pitch: int, roll: int, yaw: int = RC_MID):
    """
    Відправка RC-каналів на пульт (заглушка).
    Для реального залізу замінити на CRSFSender.send_channels() з rc_crsf_serial.py.
    """
    throttle = max(RC_MIN, min(RC_MAX, throttle))
    pitch = max(RC_MIN, min(RC_MAX, pitch))
    roll = max(RC_MIN, min(RC_MAX, roll))
    yaw = max(RC_MIN, min(RC_MAX, yaw))
    print(f"COMMAND -> Throttle: {throttle}, Pitch: {pitch}, Roll: {roll}, Yaw: {yaw}")


# ═══════════════════════════════════════════════════════════════════════════════
#  State Machine
# ═══════════════════════════════════════════════════════════════════════════════

class MissionState:
    HOVER = "HOVER"
    LAND = "LAND"
    DONE = "DONE"


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Control Loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Ініціалізація підсистем ──
    gyro = GyroReader()
    video = ThreadedVideoCapture()
    flow_proc = OpticalFlowProcessor()

    pid_throttle = PIDController(**PID_THROTTLE)
    pid_roll = PIDController(**PID_ROLL)
    pid_pitch = PIDController(**PID_PITCH)

    gyro.start()
    video.start()

    print("[MAIN] Чекаємо перший кадр...")
    deadline = time.monotonic() + 10.0
    while video.frame is None:
        if time.monotonic() > deadline:
            print("[MAIN] Таймаут — камера не відповідає.")
            video.stop()
            gyro.stop()
            return
        time.sleep(0.05)
    print("[MAIN] Відеопотік активний. Старт місії.")

    # ── Параметри стейт-машини ──
    state = MissionState.HOVER
    hover_start = time.monotonic()
    landing_throttle = float(RC_THROTTLE_HOVER)
    loop_period = 1.0 / CONTROL_LOOP_HZ

    try:
        while state != MissionState.DONE:
            loop_start = time.monotonic()
            dt = loop_period  # фіксований крок для стабільності PID

            # ── 1. Зчитування сенсорів ──
            frame = video.frame
            if frame is None:
                time.sleep(0.01)
                continue

            pitch_deg = gyro.pitch
            roll_deg = gyro.roll

            # ── 2. Optical Flow + компенсація обертання ──
            flow_x, flow_y, divergence, debug_frame = flow_proc.process(
                frame, pitch_deg, roll_deg,
            )

            # ═══════════════════════════════════════════════════════════
            #  3. PID + State Machine
            # ═══════════════════════════════════════════════════════════

            if state == MissionState.HOVER:
                elapsed = time.monotonic() - hover_start

                # Помилки позиції (target = нуль руху):
                #   flow_y > 0 → дрон падає → throttle потрібно більше (error < 0 → інвертуємо)
                #   flow_x > 0 → дрейф вправо → roll потрібно вліво (інвертуємо)
                #   divergence > 0 → летить вперед → pitch потрібно назад (інвертуємо)

                throttle_corr = pid_throttle.update(-flow_y, dt)
                roll_corr = pid_roll.update(-flow_x, dt)
                pitch_corr = pid_pitch.update(-divergence, dt)

                throttle_cmd = int(RC_THROTTLE_HOVER + throttle_corr)
                roll_cmd = int(RC_MID + roll_corr)
                pitch_cmd = int(RC_MID + pitch_corr)

                send_command(throttle_cmd, pitch_cmd, roll_cmd)

                if debug_frame is not None:
                    cv2.putText(
                        debug_frame,
                        f"STATE: HOVER  {elapsed:.1f} / {HOVER_DURATION_SEC:.0f} s",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )

                if elapsed >= HOVER_DURATION_SEC:
                    print("[MAIN] Hover завершено. Перехід до LAND.")
                    state = MissionState.LAND
                    landing_throttle = float(throttle_cmd)
                    pid_throttle.reset()

            elif state == MissionState.LAND:
                # Плавне зменшення газу; roll-корекція залишається.
                landing_throttle -= LAND_THROTTLE_STEP
                landing_throttle = max(float(RC_MIN), landing_throttle)

                roll_corr = pid_roll.update(-flow_x, dt)
                roll_cmd = int(RC_MID + roll_corr)

                send_command(int(landing_throttle), RC_MID, roll_cmd)

                if debug_frame is not None:
                    cv2.putText(
                        debug_frame,
                        f"STATE: LAND  thr={int(landing_throttle)}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                    )

                if landing_throttle <= RC_MIN:
                    print("[MAIN] Посадку завершено. Мотори вимкнено.")
                    send_command(RC_MIN, RC_MID, RC_MID)
                    state = MissionState.DONE

            # ── 4. Debug-вікно ──
            if debug_frame is not None:
                cv2.imshow("Drone Hover — Optical Flow", debug_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[MAIN] Ручна зупинка (q).")
                send_command(RC_MIN, RC_MID, RC_MID)
                break

            # ── Rate limiting ──
            elapsed_loop = time.monotonic() - loop_start
            sleep_time = loop_period - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[MAIN] KeyboardInterrupt — аварійна зупинка.")
        send_command(RC_MIN, RC_MID, RC_MID)

    finally:
        video.stop()
        gyro.stop()
        cv2.destroyAllWindows()
        print("[MAIN] Завершення.")


if __name__ == "__main__":
    main()
