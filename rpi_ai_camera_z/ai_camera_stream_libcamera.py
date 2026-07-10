#!/usr/bin/env python3
"""
AI Camera Stream з підтримкою libcamera (для Raspberry Pi 4/5).

На новіших Raspberry Pi (4, 5) камери по замовчуванню використовують libcamera,
а не V4L2. Цей скрипт використовує picamera2 (libcamera wrapper).

Установка на Pi:
    sudo apt-get update
    sudo apt-get install -y build-essential libcap-dev pkg-config
    sudo apt-get install -y python3-picamera2 python3-libcamera

Якщо використовуєш .venv:
    source .venv/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install picamera2

Запуск:
    python3 ai_camera_stream_libcamera.py
"""
from __future__ import annotations

import argparse
import json
import os
import select
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

try:
    from picamera2 import Picamera2
    from libcamera import controls
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False
    print("⚠️ picamera2 не встановлений. Встанови:")
    print("   sudo apt-get install -y build-essential libcap-dev pkg-config")
    print("   sudo apt-get install -y python3-picamera2 python3-libcamera")

try:
    from ultralytics import YOLO
except ImportError:
    print("ПОМИЛКА: ultralytics не встановлений!")
    print("Встанови: pip install ultralytics opencv-python numpy")
    sys.exit(1)

import cv2


# ═══════════════════════════════════════════════════════════════════════════
# MJPEG Server
# ═══════════════════════════════════════════════════════════════════════════

class FrameBus:
    """Шина для передачі кадрів і контролю між потоками."""
    
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._stopped = False
        self._controls: dict[str, bool] = {}
        self._status: str = "IDLE"
        self.stop_requested = False

    def publish(self, jpeg: bytes) -> None:
        """Опублікувати JPEG кадр."""
        with self._cond:
            self._jpeg = jpeg
            self._cond.notify_all()

    def wait_next(self, timeout: float = 1.0) -> bytes | None:
        """Чекати наступний кадр."""
        with self._cond:
            self._cond.wait(timeout=timeout)
            return self._jpeg

    def stop(self) -> None:
        """Зупинити шину."""
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    @property
    def stopped(self) -> bool:
        return self._stopped

    def signal(self, name: str) -> None:
        """Відправити сигнал."""
        with self._cond:
            self._controls[name] = True

    def consume(self, name: str) -> bool:
        """Спожити сигнал один раз."""
        with self._cond:
            return self._controls.pop(name, False)

    def set_status(self, status: str) -> None:
        """Встановити статус."""
        with self._cond:
            self._status = status

    def get_status(self) -> str:
        """Отримати поточний статус."""
        with self._cond:
            return self._status


def _load_html() -> bytes:
    """Загрузити HTML з файлу index.html"""
    script_dir = Path(__file__).parent
    html_path = script_dir / "index.html"
    
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read().encode("utf-8")
    
    # Fallback
    return b"""<!doctype html>
<html><head><meta charset='utf-8'><title>AI Camera Stream</title></head>
<body>
<h1>AI Camera Stream (libcamera)</h1>
<img src='/stream.mjpg' style='max-width:100%;'>
<p><button onclick="fetch('/signal/quit')">Quit</button></p>
</body></html>"""


class _HTTPHandler(BaseHTTPRequestHandler):
    """HTTP запити для MJPEG стріму і контролю."""
    bus: FrameBus = None
    
    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_load_html())
        
        elif path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", 
                           "multipart/x-mixed-replace; boundary=--boundary")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            
            while not self.bus.stopped:
                jpeg = self.bus.wait_next(timeout=1.0)
                if jpeg:
                    try:
                        self.wfile.write(b"--boundary\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n"
                                       .encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    except BrokenPipeError:
                        break
        
        elif path == "/status":
            status = self.bus.get_status()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": status}).encode())
        
        elif path.startswith("/signal/"):
            signal_name = path.split("/signal/", 1)[1]
            self.bus.signal(signal_name)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format: str, *args) -> None:
        """Тихі логи."""
        pass


def start_mjpeg_server(bus: FrameBus, port: int = 8080) -> None:
    """Запустити MJPEG HTTP-сервер."""
    _HTTPHandler.bus = bus
    server = ThreadingHTTPServer(("0.0.0.0", port), _HTTPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"✓ MJPEG сервер запущений на http://localhost:{port}")
    return server


# ═══════════════════════════════════════════════════════════════════════════
# Picamera2 Streamer
# ═══════════════════════════════════════════════════════════════════════════

class AICameraStreamerLibcamera:
    """Читає з AI камери (libcamera), детектує обєкти, стрімить у браузер."""
    
    def __init__(self, model_path: str = "yolov8n.pt",
                 conf: float = 0.5, enable_detection: bool = True,
                 width: int = 1280, height: int = 720,
                 verbose: bool = False) -> None:
        if not HAS_PICAMERA2:
            raise ImportError("picamera2 не встановлений!")
        
        self.conf = conf
        self.enable_detection = enable_detection
        self.width = width
        self.height = height
        self.verbose = verbose
        self.bus = FrameBus()
        self.running = False
        
        # YOLO модель
        self.model = None
        if enable_detection:
            print(f"📦 Завантажую модель {model_path}...")
            self.model = YOLO(model_path)
            print(f"✓ Модель завантажена")
        
        # Статистика
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.detect_enabled = True
        self.track_enabled = False
    
    def _open_camera(self) -> Picamera2 | None:
        """Відкрити камеру через libcamera."""
        print(f"🎥 Відкриваю камеру через libcamera...")
        
        try:
            picam2 = Picamera2()
            
            # Конфігурація
            config = picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (self.width, self.height)},
                controls={"FrameRate": 30}
            )
            picam2.configure(config)
            
            # Запусти камеру
            picam2.start()
            
            print(f"   ⏳ Ініціалізація камери (2 сек)...")
            time.sleep(2)
            
            # Прочитай кілька кадрів для прогріву
            print("   🔄 Прогрів камери...")
            for i in range(3):
                try:
                    frame = picam2.capture_array()
                    if frame is not None:
                        if self.verbose:
                            print(f"   ✓ Кадр {i+1}/3 OK ({frame.shape})")
                    time.sleep(0.2)
                except Exception as e:
                    if self.verbose:
                        print(f"   ⚠️ Кадр {i+1}/3: {e}")

            props = picam2.camera_properties
            if isinstance(props, dict) and "PixelArraySize" in props:
                h, w = props["PixelArraySize"]
            else:
                h, w = self.height, self.width
            print(f"✓ Камера (libcamera) відкрита: {w}x{h}")
            return picam2
        
        except Exception as e:
            print(f"❌ Помилка відкриття камери: {e}")
            return None
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Додати детекцію, трекінг, текст на кадр."""
        # Конвертуй RGB в BGR для OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Малюємо бокси вручну, щоб контролювати кольори (позбутися від синього)
        if self.model and self.detect_enabled:
            results = self.model.predict(frame_bgr, conf=self.conf, verbose=False)
            r = results[0]
            try:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            except Exception:
                boxes = []

            # Колір бокса (B, G, R) — за замовчуванням зелений
            default_color = (0, 200, 0)
            # Можна задати мапу кольорів для певних класів
            class_colors = {}

            for i, box in enumerate(boxes if boxes is not None else []):
                x1, y1, x2, y2 = map(int, box)
                conf = float(confs[i]) if len(confs) > i else 0.0
                cls = int(cls_ids[i]) if len(cls_ids) > i else -1
                name = self.model.names[cls] if (hasattr(self.model, 'names') and cls >= 0) else str(cls)

                color = class_colors.get(name, default_color)
                # rectangle and label
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                label = f"{name} {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # background box for label
                cv2.rectangle(frame_bgr, (x1, y1 - lh - 8), (x1 + lw + 8, y1), color, -1)
                cv2.putText(frame_bgr, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # Додати FPS і статус (текст світліший для темної гами)
        status_text = (
            f"FPS: {self.fps:.1f} | "
            f"Detection: {'ON' if self.detect_enabled else 'OFF'} | "
            f"Tracking: {'ON' if self.track_enabled else 'OFF'}"
        )
        cv2.putText(frame_bgr, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 180), 2)
        
        return frame_bgr
    
    def _encode_jpeg(self, frame: np.ndarray, quality: int = 80) -> bytes:
        """Закодувати кадр у JPEG."""
        _, buffer = cv2.imencode(".jpg", frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, quality])
        return bytes(buffer)
    
    def run(self) -> None:
        """Основний цикл: читання, обробка, стрім."""
        picam2 = self._open_camera()
        if not picam2:
            print("❌ Не можу запустити камеру!")
            return
        
        self.running = True
        print("🔴 Стрімування активне. Натисніть Ctrl-C для вихіддю...")
        
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        try:
            while self.running:
                # Обробити сигнали з браузера
                if self.bus.consume("quit"):
                    print("📤 Сигнал quit отримано")
                    break
                
                if self.bus.consume("detect_toggle"):
                    self.detect_enabled = not self.detect_enabled
                    print(f"🎯 Детекція: {'ON' if self.detect_enabled else 'OFF'}")
                
                if self.bus.consume("track_toggle"):
                    self.track_enabled = not self.track_enabled
                    print(f"📍 Трекінг: {'ON' if self.track_enabled else 'OFF'}")
                
                if self.bus.consume("fps_reset"):
                    self.frame_count = 0
                    self.last_fps_time = time.time()
                    print("📊 FPS лічильник скинутий")
                
                # Читати кадр з камери
                try:
                    frame = picam2.capture_array()
                    if frame is None:
                        raise ValueError("None frame")
                    
                    consecutive_errors = 0
                    
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors % 5 == 0:
                        print(f"⚠️ Помилка читання ({consecutive_errors}x): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"❌ Занадто много помилок. Перезапускаю...")
                        picam2.stop()
                        time.sleep(1)
                        picam2 = self._open_camera()
                        if not picam2:
                            break
                        consecutive_errors = 0
                    
                    time.sleep(0.1)
                    continue
                
                # Обробити кадр (детекція, текст)
                frame = self._process_frame(frame)
                
                # Закодувати у JPEG і опублікувати
                jpeg = self._encode_jpeg(frame)
                self.bus.publish(jpeg)
                
                # Оновити FPS
                self.frame_count += 1
                now = time.time()
                elapsed = now - self.last_fps_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    status = (
                        f"🟢 LIVE | FPS: {self.fps:.1f} | "
                        f"Det: {'ON' if self.detect_enabled else 'OFF'} | "
                        f"Track: {'ON' if self.track_enabled else 'OFF'}"
                    )
                    self.bus.set_status(status)
                    self.frame_count = 0
                    self.last_fps_time = now
        
        except KeyboardInterrupt:
            print("\n⏹️ Перервано користувачем")
        
        finally:
            self.running = False
            try:
                picam2.stop()
            except:
                pass
            self.bus.stop()
            print("✓ Камера закрита")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AI Camera (libcamera) MJPEG стрімер з детекцією"
    )
    parser.add_argument("--model", default="yolov8n.pt",
                       help="Шлях до YOLO моделі")
    parser.add_argument("--port", type=int, default=8080,
                       help="HTTP порт")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Поріг конфіденції детекції")
    parser.add_argument("--no-detection", action="store_true",
                       help="Вимкнути детекцію")
    parser.add_argument("--width", type=int, default=1280,
                       help="Ширина кадру")
    parser.add_argument("--height", type=int, default=720,
                       help="Висота кадру")
    parser.add_argument("--verbose", action="store_true",
                       help="Деталізоване логування")
    
    args = parser.parse_args()
    
    if not HAS_PICAMERA2:
        print("❌ picamera2 не встановлений!")
        print("\nУстанови на Raspberry Pi:")
        print("  sudo apt-get update")
        print("  sudo apt-get install -y build-essential libcap-dev pkg-config")
        print("  sudo apt-get install -y python3-picamera2 python3-libcamera")
        sys.exit(1)
    
    print("╔════════════════════════════════════════════╗")
    print("║ AI Camera Stream (libcamera) with YOLOv8  ║")
    print("╚════════════════════════════════════════════╝")
    print(f"Модель: {args.model}")
    print(f"Роздільність: {args.width}x{args.height}")
    print(f"Поріг конфіденції: {args.conf}")
    print(f"Детекція: {'OFF' if args.no_detection else 'ON'}")
    print(f"Вербоза: {args.verbose}")
    print()
    
    # Запустити стрімер
    streamer = AICameraStreamerLibcamera(
        model_path=args.model,
        conf=args.conf,
        enable_detection=not args.no_detection,
        width=args.width,
        height=args.height,
        verbose=args.verbose
    )
    
    # Запустити MJPEG сервер
    server = start_mjpeg_server(streamer.bus, port=args.port)
    
    # Основний цикл
    try:
        streamer.run()
    except KeyboardInterrupt:
        print("\n⏹️ Завершення...")
    finally:
        streamer.bus.stop()
        server.shutdown()
    
    print("✓ Готово")


if __name__ == "__main__":
    main()
