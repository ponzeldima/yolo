"""HUNT — автопошук і таран нерухомого манекена.

Стан-машина всередині BRAKE:
    SCAN_ROTATE  — повертаємо yaw на ~15° (за часом)
    SCAN_SETTLE  — зупиняємо yaw, даємо кадру заспокоїтися, запускаємо YOLOv8s
    ALIGN        — PID по yaw, щоб ціль була по центру кадру (по X)
    ATTACK       — ALIGN + forward_setpoint > 0 (рух вперед на таран)

Детектор: YOLOv8s, лише клас person (id=0). Для трекінгу використовується
вбудований `model.track(persist=True)` + ByteTrack.
"""

import threading
import time
import numpy as np
from ultralytics import YOLO

from .models import Detection


class HuntController:

    PHASE_SCAN_ROTATE = "SCAN_ROTATE"
    PHASE_SCAN_SETTLE = "SCAN_SETTLE"
    PHASE_ALIGN_PULSE = "ALIGN_PULSE"
    PHASE_ALIGN_SETTLE = "ALIGN_SETTLE"
    PHASE_ATTACK = "ATTACK"

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        imgsz: int = 640,
        conf: float = 0.20,
        device: str = "cuda",
        half: bool = True,
    ):
        self._model = YOLO(model_path)
        self._imgsz = imgsz
        self._conf = conf
        self._device = device
        self._half = half

        # ── Параметри (можна тюнити) ──
        # Час утримання yaw-стіка для повороту приблизно на 15° (калібрується емпірично).
        self.YAW_STICK_CMD = 0.4
        self.YAW_STEP_DURATION = 0.03     # сек: тримання стіка для ~15° повороту
        self.SETTLE_DURATION = 4.0       # сек: пауза після повороту (кадр і детектор)
        # ── ALIGN (ривками, за зразком SCAN) ──
        # Амплітуда yaw-стіка під час ALIGN-пульсу (зазвичай менша за SCAN).
        self.ALIGN_YAW_STICK = 0.4
        # Мінімальна і максимальна тривалість пульсу. Фактична тривалість =
        # clip(|err_x| * ALIGN_PULSE_K, MIN, MAX). Чим більша помилка — довший пульс.
        self.ALIGN_PULSE_MIN = 0.05        # сек (для мікродоворотів)
        self.ALIGN_PULSE_MAX = 0.18        # сек
        self.ALIGN_PULSE_K = 0.03          # сек на одиницю |err_x|
        self.ALIGN_SETTLE_DURATION = 2.0   # сек: пауза між пульсами
        self.ALIGN_TOL = 0.05              # |err_x| < TOL вважається "по центру"
        # Скільки послідовних перевірок у межах TOL треба перед ATTACK (захист від випадку).
        self.ALIGN_OK_COUNT = 2
        # Forward-швидкість на таран (у нормалізованих одиницях flow_div).
        self.ATTACK_FORWARD_SETPOINT = 0.40
        # Втрата цілі: якщо >LOST_TIMEOUT не бачимо bbox — назад у SCAN.
        self.LOST_TIMEOUT = 0.8

        # ── Стан ──
        self.phase: str = self.PHASE_SCAN_ROTATE
        self._phase_start = 0.0
        self.target: Detection | None = None
        self.locked_track_id: int | None = None
        self._last_seen = 0.0
        self._align_ok_count = 0
        self._align_pulse_duration = 0.0
        self._align_pulse_dir = 0  # -1 / 0 / +1
        self._yaw_direction = +1  # напрямок сканування
        # Трекінг активний — трекаємо ціль на кожному кадрі, а не лише в settle.
        # вмикається після першого вдалого SCAN-детекту; вимикається при LOST_TIMEOUT.
        self._tracking_active = False
        # Seq останнього вже обробленого результату (щоб не перечитувати той самий).
        self._last_processed_seq = 0

        # ── Асинхронний детектор (фоновий потік) ──
        # YOLO крутиться у власному потоці, щоб не блокувати головний цикл
        # пайплайна (інакше cv2.imshow/waitKey «замерзає» на час інференсу).
        self._det_lock = threading.Lock()
        self._det_submit_frame: np.ndarray | None = None
        self._det_submit_evt = threading.Event()
        self._det_result_dets: list[Detection] = []
        self._det_result_seq = 0
        self._det_result_frame_w = 0
        self._det_result_frame_h = 0
        # Снімок seq на момент входу у фазу-settle: результати вважаються
        # «свіжими» лише якщо seq > цього снапшоту.
        self._det_settle_seq_baseline = 0
        self._det_running = True
        self._det_thread = threading.Thread(
            target=self._detect_worker, name="HuntDetect", daemon=True
        )
        self._det_thread.start()

    # ── API ──────────────────────────────────────────────────────────────────

    def reset(self):
        self.phase = self.PHASE_SCAN_ROTATE
        self._phase_start = time.perf_counter()
        self.target = None
        self.locked_track_id = None
        self._last_seen = 0.0
        self._align_ok_count = 0
        self._align_pulse_duration = 0.0
        self._align_pulse_dir = 0
        self._tracking_active = False
        with self._det_lock:
            self._det_settle_seq_baseline = self._det_result_seq
            self._last_processed_seq = self._det_result_seq

    def shutdown(self):
        """Зупинити фоновий детектор (викликати при завершенні пайплайна)."""
        self._det_running = False
        self._det_submit_evt.set()

    def update(self, frame: np.ndarray | None) -> tuple[float, float, bool]:
        """Повертає (yaw_cmd, forward_setpoint, yaw_is_scanning).

        yaw_cmd            — команда на yaw-стік [-1, +1]
        forward_setpoint   — цільова дивергенція для BrakeController
        yaw_is_scanning    — True під час SCAN_ROTATE/SETTLE (для заморозки roll PID)
        """
        now = time.perf_counter()
        if frame is None:
            return (0.0, 0.0, True)

        # Годуємо фоновий YOLO останнім кадром (не блокуючись).
        self._submit_frame(frame)

        h, w = frame.shape[:2]
        cx_screen = w / 2.0

        # ── Безперервне трекання після першої знахідки ──
        # YOLO працює на кожному кадрі — якщо трекінг активний, читаємо свіжий
        # результат і оновлюємо self.target / _last_seen ЩО РАЗУ, щоб ByteTrack
        # не втрачав track_id через розриви в часі між інференсами.
        if self._tracking_active:
            seen_now = self._refresh_tracked_target(w, h, now)
            if not seen_now and (now - self._last_seen >= self.LOST_TIMEOUT):
                print(f"\n[HUNT] Ціль втрачена (track_id={self.locked_track_id}) → SCAN")
                self._tracking_active = False
                self.target = None
                self.locked_track_id = None
                self._align_ok_count = 0
                self._set_phase(self.PHASE_SCAN_ROTATE)
                return (0.0, 0.0, True)

        if self.phase == self.PHASE_SCAN_ROTATE:
            # Поворот стіка на фіксовану тривалість
            yaw_cmd = self.YAW_STICK_CMD * self._yaw_direction
            if now - self._phase_start >= self.YAW_STEP_DURATION:
                self._set_phase(self.PHASE_SCAN_SETTLE)
            return (yaw_cmd, 0.0, True)

        if self.phase == self.PHASE_SCAN_SETTLE:
            if now - self._phase_start < self.SETTLE_DURATION:
                return (0.0, 0.0, True)
            # Пробуємо детектувати людину (неблокуюче, з фонового воркера)
            ready, target = self._take_detection(w, h, pick_closest_to_center=True)
            if not ready:
                # Ще немає свіжого інференсу після початку settle — чекаємо
                return (0.0, 0.0, True)
            if target is not None:
                self.target = target
                self.locked_track_id = target.track_id if target.track_id > 0 else None
                self._last_seen = now
                self._align_ok_count = 0
                # Вмикаємо безперервне трекання — з цього моменту YOLO-результат
                # оновлюватиметься КОЖЕН кадр (не тільки у settle-фазах).
                self._tracking_active = True
                # Переходимо у ALIGN_SETTLE (пауза перед першим пульсом — щоб
                # detect встиг оновитись, а pipeline побачив стабільний кадр).
                self._set_phase(self.PHASE_ALIGN_SETTLE)
                print(f"\n[HUNT] Ціль знайдена (id={self.locked_track_id}, conf={target.confidence:.2f}) → ALIGN (continuous tracking)")
                return (0.0, 0.0, True)
            # Немає цілі — ще один крок сканування
            self._set_phase(self.PHASE_SCAN_ROTATE)
            return (0.0, 0.0, True)

        if self.phase == self.PHASE_ALIGN_PULSE:
            # Видаємо yaw-стік на заздалегідь розраховану тривалість.
            # yaw_is_scanning=True → pipeline заморозить roll + throttle PID
            # (так само як у SCAN_ROTATE), щоб дрон не втрачав висоту.
            yaw_cmd = self.ALIGN_YAW_STICK * self._align_pulse_dir
            if now - self._phase_start >= self._align_pulse_duration:
                self._set_phase(self.PHASE_ALIGN_SETTLE)
            return (yaw_cmd, 0.0, True)

        if self.phase == self.PHASE_ALIGN_SETTLE:
            if now - self._phase_start < self.ALIGN_SETTLE_DURATION:
                return (0.0, 0.0, True)
            # Пауза закінчилась. self.target уже оновлено зверху (continuous
            # tracking). Якщо ціль тимчасово не видно — чекаємо ще settle,
            # LOST-timeout відпрацює глобальна перевірка зверху.
            target = self.target
            if target is None:
                self._phase_start = now
                return (0.0, 0.0, True)

            cx_t = (target.x1 + target.x2) / 2.0
            err_x = (cx_t - cx_screen) / (w / 2.0)

            if abs(err_x) < self.ALIGN_TOL:
                self._align_ok_count += 1
                if self._align_ok_count >= self.ALIGN_OK_COUNT:
                    print(f"\n[HUNT] Ціль по центру (err_x={err_x:+.3f}) → ATTACK")
                    self._set_phase(self.PHASE_ATTACK)
                    return (0.0, self.ATTACK_FORWARD_SETPOINT, False)
                # Ще одна перевірка — ще раз дамо settle
                self._phase_start = now
                return (0.0, 0.0, True)

            # Помилка поза TOL → плануємо наступний пульс
            self._align_ok_count = 0
            self._align_pulse_dir = 1 if err_x > 0 else -1
            pulse = abs(err_x) * self.ALIGN_PULSE_K
            pulse = max(self.ALIGN_PULSE_MIN, min(self.ALIGN_PULSE_MAX, pulse))
            self._align_pulse_duration = pulse
            self._set_phase(self.PHASE_ALIGN_PULSE)
            return (0.0, 0.0, True)

        if self.phase == self.PHASE_ATTACK:
            # self.target оновлюється зверху (continuous tracking). LOST_TIMEOUT
            # теж відпрацьовує там. Тут просто читаємо поточну ціль.
            target = self.target
            if target is None:
                # Ціль тимчасово не видно — тримаємо forward без корекції yaw
                return (0.0, self.ATTACK_FORWARD_SETPOINT, False)

            # В ATTACK залишаємо м'який P-контролер yaw (не ривками),
            # щоб під час руху плавно доводити центр. Якщо помилка велика —
            # повертаємось у ALIGN-ривки.
            cx_t = (target.x1 + target.x2) / 2.0
            err_x = (cx_t - cx_screen) / (w / 2.0)

            if abs(err_x) > 100 * self.ALIGN_TOL:
                print(f"\n[HUNT] ATTACK: помилка {err_x:+.2f} завелика → назад у ALIGN")
                self._align_ok_count = 0
                self._set_phase(self.PHASE_ALIGN_SETTLE)
                return (0.0, 0.0, True)

            # Невелика пропорційна корекція yaw під час атаки
            yaw_cmd = 0.5 * err_x
            yaw_cmd = max(-self.ALIGN_YAW_STICK, min(self.ALIGN_YAW_STICK, yaw_cmd))
            yaw_is_scanning = abs(yaw_cmd) > 0.15
            return (0.0, self.ATTACK_FORWARD_SETPOINT, yaw_is_scanning)

        return (0.0, 0.0, False)

    # ── Внутрішні ────────────────────────────────────────────────────────────

    def _set_phase(self, phase: str):
        self.phase = phase
        self._phase_start = time.perf_counter()
        # При вході у settle-фазу фіксуємо «нульовий» seq, щоб вимагати свіжий
        # інференс ПІСЛЯ початку settle (а не використовувати старий результат).
        if phase in (self.PHASE_SCAN_SETTLE, self.PHASE_ALIGN_SETTLE):
            with self._det_lock:
                self._det_settle_seq_baseline = self._det_result_seq

    def _submit_frame(self, frame: np.ndarray) -> None:
        """Передати кадр фоновому YOLO-воркеру (перезаписує попередній, якщо той ще не взяв)."""
        with self._det_lock:
            # copy() щоб воркер міг працювати з власною версією без гонок
            self._det_submit_frame = frame.copy()
        self._det_submit_evt.set()

    def _take_detection(
        self,
        frame_w: int,
        frame_h: int,
        pick_closest_to_center: bool = False,
        pick_locked_id: int | None = None,
    ) -> tuple[bool, Detection | None]:
        """Повертає (ready, target) з кешу фонового воркера.

        ready=False означає, що свіжого інференсу (новішого за момент входу
        у поточну settle-фазу) ще немає — caller має зачекати наступний update().
        """
        with self._det_lock:
            if self._det_result_seq <= self._det_settle_seq_baseline:
                return (False, None)
            dets = list(self._det_result_dets)
            self._last_processed_seq = self._det_result_seq

        if not dets:
            return (True, None)

        if pick_locked_id is not None:
            for d in dets:
                if d.track_id == pick_locked_id:
                    return (True, d)
            # Локований ID зник — fallback: найближчий до центру кадру
            pick_closest_to_center = True

        if pick_closest_to_center:
            cx, cy = frame_w / 2.0, frame_h / 2.0
            best = min(
                dets,
                key=lambda d: ((d.x1 + d.x2) / 2 - cx) ** 2 + ((d.y1 + d.y2) / 2 - cy) ** 2,
            )
            return (True, best)

        return (True, dets[0])

    def _refresh_tracked_target(self, frame_w: int, frame_h: int, now: float) -> bool:
        """Оновити self.target з останнього YOLO-результату (якщо він новий).

        Викликається кожен кадр при активному трекінгу. Повертає True, якщо
        ми щойно бачили ціль (locked_track_id АБО найближчу до попередньої
        позиції детекцію — на випадок, якщо ByteTrack переприсвоїв ID).
        """
        with self._det_lock:
            seq_now = self._det_result_seq
            if seq_now <= self._last_processed_seq:
                return False
            self._last_processed_seq = seq_now
            dets = list(self._det_result_dets)

        if not dets:
            return False

        # 1) Спершу шукаємо locked_track_id серед свіжих детекцій.
        if self.locked_track_id is not None:
            for d in dets:
                if d.track_id == self.locked_track_id:
                    self.target = d
                    self._last_seen = now
                    return True

        # 2) ID не знайдено (ByteTrack міг переприв'язати) — шукаємо найближчу
        # детекцію до ОСТАННЬОЇ ВІДОМОЇ позиції цілі і переприв'язуємось до неї.
        # Це ключовий момент: рамка має оновлюватись навіть коли track_id
        # перескакує — інакше дрон "втрачає" ціль, хоча вона в кадрі.
        ref_cx: float
        ref_cy: float
        max_dist: float
        if self.target is not None:
            ref_cx = (self.target.x1 + self.target.x2) / 2.0
            ref_cy = (self.target.y1 + self.target.y2) / 2.0
            # Дозволений стрибок — ~30% меншої сторони кадру за один YOLO-тік.
            max_dist = 0.30 * min(frame_w, frame_h)
        else:
            ref_cx, ref_cy = frame_w / 2.0, frame_h / 2.0
            max_dist = float("inf")

        best: Detection | None = None
        best_d2 = float("inf")
        for d in dets:
            dcx = (d.x1 + d.x2) / 2.0
            dcy = (d.y1 + d.y2) / 2.0
            d2 = (dcx - ref_cx) ** 2 + (dcy - ref_cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = d

        if best is None:
            return False
        if best_d2 > max_dist * max_dist:
            # Найближча детекція занадто далеко — це інша людина. Не крадемо.
            return False

        # Переприв'язуємось до найближчого кандидата.
        self.target = best
        self._last_seen = now
        if best.track_id > 0:
            if best.track_id != self.locked_track_id:
                print(f"[HUNT] re-lock track_id {self.locked_track_id} → {best.track_id}")
            self.locked_track_id = best.track_id
        return True

    def _detect_worker(self) -> None:
        """Фоновий потік: крутить YOLO на останньому сабмітнутому кадрі."""
        while self._det_running:
            if not self._det_submit_evt.wait(timeout=0.1):
                continue
            with self._det_lock:
                frame = self._det_submit_frame
                self._det_submit_frame = None
                self._det_submit_evt.clear()
            if frame is None or not self._det_running:
                continue
            try:
                results = self._model.track(
                    frame, verbose=False, conf=self._conf,
                    classes=[0],  # person
                    device=self._device, persist=True,
                    imgsz=self._imgsz, half=self._half,
                    tracker="bytetrack.yaml",
                )
            except Exception as e:
                print(f"[HUNT] YOLO error: {e}")
                continue

            dets: list[Detection] = []
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf < self._conf:
                    continue
                tid = int(box.id[0]) if box.id is not None else -1
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append(Detection(int(x1), int(y1), int(x2), int(y2), conf, tid))

            with self._det_lock:
                self._det_result_dets = dets
                self._det_result_frame_w = frame.shape[1]
                self._det_result_frame_h = frame.shape[0]
                self._det_result_seq += 1
