#!/usr/bin/env python3
"""
AI Camera (aitrois) стрімер з об'єктною детекцією у браузер.

Підключення камери: CnDC Standard-mini до cam/disp1
Стрім: MJPEG via HTTP на localhost:8080
Детекція: YOLOv8 (обєкти + bounding boxes у браузері)

Використання:
    python3 ai_camera_stream.py [--model yolov8n.pt] [--device 0] [--port 8080]

Опції:
    --model MODEL_PATH      Шлях до YOLO моделі (default: yolov8n.pt)
    --device DEVICE         ID камери (default: 0)
    --port PORT             HTTP порт (default: 8080)
    --conf CONF             Поріг конфіденції (default: 0.5)
    --no-detection          Вимкнути детекцію (лише стрім відео)

Кнопки у браузері:
    - START/STOP детекції
    - Toggle tracking on/off
    - Quit скрипт
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

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ПОМИЛКА: ultralytics не встановлений!")
    print("Встанови: pip install ultralytics opencv-python")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# MJPEG Server (копія з rpi_thermal_tracking_z)
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


# HTML завантажується з файлу index.html
_DEFAULT_INDEX_HTML = None  # Буде загружено пізніше


def _load_html() -> bytes:
    """Загрузити HTML з файлу index.html"""
    global _DEFAULT_INDEX_HTML
    if _DEFAULT_INDEX_HTML is not None:
        return _DEFAULT_INDEX_HTML
    
    # Спробуй знайти index.html в тій же папці що скрипт
    script_dir = Path(__file__).parent
    html_path = script_dir / "index.html"
    
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            _DEFAULT_INDEX_HTML = f.read().encode("utf-8")
    else:
        # Fallback: мінімальний HTML якщо файл не знайдено
        _DEFAULT_INDEX_HTML = b"""<!doctype html>
<html><head><meta charset='utf-8'><title>AI Camera Stream</title></head>
<body>
<h1>AI Camera Stream</h1>
<img src='/stream.mjpg' style='max-width:100%;'>
<p><button onclick="fetch('/signal/quit')">Quit</button></p>
</body></html>"""
        print(f"WARNING: index.html not found at {html_path}")
    
    return _DEFAULT_INDEX_HTML


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
# AI Camera Streamer з YOLOv8 детекцією
# ═══════════════════════════════════════════════════════════════════════════

class AICameraStreamer:
    """Читає з AI камери, детектує обєкти, стрімить у браузер."""
    
    def __init__(self, device: int = 0, model_path: str = "yolov8n.pt",
                 conf: float = 0.5, enable_detection: bool = True,
                 width: int = 1280, height: int = 720,
                 verbose: bool = False) -> None:
        self.device = device
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
    
    def _open_camera(self) -> cv2.VideoCapture | None:
        """Відкрити AI камеру."""
        print(f"🎥 Відкриваю камеру (device={self.device})...")
        cap = cv2.VideoCapture(self.device)
        
        if not cap.isOpened():
            print(f"❌ Не можу відкрити камеру {self.device}")
            print(f"   Спробуй: python3 ai_camera_stream.py --list-cameras")
            return None
        
        # Дати камері час на ініціалізацію
        print("   ⏳ Ініціалізація камери (3 сек)...")
        time.sleep(3)
        
        # Встановити параметри
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # мінімальний буфер
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Дати час на застосування параметрів
        time.sleep(1)
        
        # Прочитай кілька кадрів для прогріву
        print("   🔄 Прогрів камери...")
        for i in range(5):
            ret, _ = cap.read()
            if ret:
                if self.verbose:
                    print(f"   ✓ Кадр {i+1}/5 OK")
            time.sleep(0.2)
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✓ Камера відкрита: {w}x{h} @ {fps:.1f} FPS")
        if self.verbose:
            print(f"   Буфер: {int(cap.get(cv2.CAP_PROP_BUFFERSIZE))}")
        return cap
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Додати детекцію, трекінг, текст на кадр."""
        if self.model and self.detect_enabled:
            results = self.model.predict(frame, conf=self.conf, verbose=False)
            frame = results[0].plot()
        
        # Додати FPS і статус
        status_text = (
            f"FPS: {self.fps:.1f} | "
            f"Detection: {'ON' if self.detect_enabled else 'OFF'} | "
            f"Tracking: {'ON' if self.track_enabled else 'OFF'}"
        )
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def _encode_jpeg(self, frame: np.ndarray, quality: int = 80) -> bytes:
        """Закодувати кадр у JPEG."""
        _, buffer = cv2.imencode(".jpg", frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, quality])
        return bytes(buffer)
    
    def run(self) -> None:
        """Основний цикл: читання, обробка, стрім."""
        cap = self._open_camera()
        if not cap:
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
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors % 5 == 0:
                        print(f"⚠️ Помилка читання ({consecutive_errors}x). "
                              f"Спробую перепідключити...")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"❌ Занадто много помилок ({consecutive_errors}x). "
                              f"Перепідключаю камеру...")
                        cap.release()
                        time.sleep(1)
                        cap = self._open_camera()
                        if not cap:
                            print("❌ Не можу перепідключити камеру!")
                            break
                        consecutive_errors = 0
                    
                    time.sleep(0.1)
                    continue
                
                consecutive_errors = 0  # Скинути лічильник при успіху
                
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
            cap.release()
            self.bus.stop()
            print("✓ Камера закрита")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AI Camera (aitrois) MJPEG стрімер з детекцією"
    )
    parser.add_argument("--model", default="yolov8n.pt",
                       help="Шлях до YOLO моделі")
    parser.add_argument("--device", type=int, default=0,
                       help="ID пристрою камери")
    parser.add_argument("--port", type=int, default=8080,
                       help="HTTP порт")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Поріг конфіденції детекції")
    parser.add_argument("--no-detection", action="store_true",
                       help="Вимкнути детекцію")
    parser.add_argument("--list-cameras", action="store_true",
                       help="Показати доступні камери і вихід")
    parser.add_argument("--width", type=int, default=1280,
                       help="Ширина кадру")
    parser.add_argument("--height", type=int, default=720,
                       help="Висота кадру")
    parser.add_argument("--verbose", action="store_true",
                       help="Деталізоване логування")
    
    args = parser.parse_args()
    
    # Якщо потрібно список камер
    if args.list_cameras:
        print("🎥 Сканування доступних камер...")
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"   ✓ /dev/video{i}: {w}x{h} @ {fps:.0f} FPS")
                cap.release()
        print("\nДля запуску: python3 ai_camera_stream.py --device <ID>")
        return
    
    print("╔════════════════════════════════════════════╗")
    print("║   AI Camera Stream with YOLOv8 Detection  ║")
    print("╚════════════════════════════════════════════╝")
    print(f"Модель: {args.model}")
    print(f"Пристрій: /dev/video{args.device}")
    print(f"Роздільність: {args.width}x{args.height}")
    print(f"Поріг конфіденції: {args.conf}")
    print(f"Детекція: {'OFF' if args.no_detection else 'ON'}")
    print(f"Вербоза: {args.verbose}")
    print()
    
    # Запустити стрімер
    streamer = AICameraStreamer(
        device=args.device,
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
