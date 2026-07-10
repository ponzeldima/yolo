"""Спільний HTTP MJPEG-стрім + керування через браузер.

FrameBus тримає:
  * останній JPEG-кадр (стрім),
  * довільні булеві «сигнали» (controls), які виставляються з HTML-сторінки
    через /signal/<name> і споживаються головним циклом скрипта-виробника
    через bus.consume("<name>").
  * текстовий статус (status), який скрипт публікує через bus.set_status(...),
    а HTML-сторінка опитує через /status (показує, наприклад, REC ● / IDLE).

Назад-сумісність:
  * /stop досі виставляє bus.stop_requested + сигнал "stop"
    (стара поведінка для record_session.py першої версії).
"""
from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


class FrameBus:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._stopped = False
        # generic signals: name -> True, consumed-once by producer
        self._controls: dict[str, bool] = {}
        # human-readable status string (e.g. "REC ●  frames=120")
        self._status: str = "IDLE"
        # back-compat прапор
        self.stop_requested = False

    # ── стрім ─────────────────────────────────────────────
    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._cond.notify_all()

    def wait_next(self, timeout: float = 1.0) -> bytes | None:
        with self._cond:
            self._cond.wait(timeout=timeout)
            return self._jpeg

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    @property
    def stopped(self) -> bool:
        return self._stopped

    # ── сигнали (start/stop/quit/...) ────────────────────
    def signal(self, name: str) -> None:
        with self._cond:
            self._controls[name] = True

    def consume(self, name: str) -> bool:
        """Повертає True один раз, якщо сигнал прийшов; одразу скидає його."""
        with self._cond:
            return self._controls.pop(name, False)

    # ── статус ────────────────────────────────────────────
    def set_status(self, status: str) -> None:
        with self._cond:
            self._status = status

    def get_status(self) -> str:
        with self._cond:
            return self._status

    # ── back-compat ──────────────────────────────────────
    def request_stop(self) -> None:
        self.stop_requested = True
        self.signal("stop")


# Дефолтна сторінка з трьома кнопками для record_session.py.
_DEFAULT_INDEX_HTML = b"""<!doctype html>
<html><head><meta charset='utf-8'><title>MLX90642 stream</title></head>
<body style='margin:0;background:#111;font-family:sans-serif;color:#eee'>
<div style='padding:8px;text-align:center;background:#222;
            display:flex;align-items:center;justify-content:center;gap:12px'>
  <span id='status' style='font-weight:bold'>...</span>
  <button onclick="signal('start')"
          style='padding:6px 14px;font-weight:bold;background:#2a2;
                 color:#fff;border:0;border-radius:4px;cursor:pointer'>
    &#9679; START REC
  </button>
  <button onclick="signal('stop')"
          style='padding:6px 14px;font-weight:bold;background:#c22;
                 color:#fff;border:0;border-radius:4px;cursor:pointer'>
    &#9632; STOP REC
  </button>
  <button onclick="if(confirm('Shutdown script?')) signal('quit')"
          style='padding:6px 14px;background:#444;color:#fff;border:0;
                 border-radius:4px;cursor:pointer'>
    Quit
  </button>
</div>
<img src='/stream' style='display:block;margin:auto'/>
<script>
async function signal(name) {
  try { await fetch('/signal/' + name, {method:'POST'}); } catch(e){}
  refresh();
}
async function refresh() {
  try {
    const r = await fetch('/status'); const j = await r.json();
    const el = document.getElementById('status');
    el.textContent = j.status;
    el.style.color = j.recording ? '#f44' : '#7c7';
  } catch(e){}
}
refresh(); setInterval(refresh, 1000);
</script>
</body></html>
"""


def make_handler(bus: FrameBus, index_html: bytes):
    boundary = b"--frame"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send_html(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle_signal(self, name: str) -> None:
            bus.signal(name)
            if name in ("stop", "quit"):
                # назад-сумісність зі старим /stop
                bus.stop_requested = True
            self._send_json({"ok": True, "signal": name})

        def do_GET(self):                            # noqa: N802
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._send_html(index_html)
                return
            if path == "/status":
                status = bus.get_status()
                self._send_json({
                    "status": status,
                    "recording": status.startswith("REC"),
                })
                return
            if path == "/stop":
                # back-compat: повна зупинка скрипта
                self._handle_signal("stop")
                return
            if path.startswith("/signal/"):
                self._handle_signal(path[len("/signal/"):])
                return
            if path != "/stream":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not bus.stopped:
                    jpeg = bus.wait_next(timeout=1.0)
                    if jpeg is None:
                        continue
                    self.wfile.write(boundary + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):                           # noqa: N802
            path = urlsplit(self.path).path
            if path.startswith("/signal/"):
                self._handle_signal(path[len("/signal/"):])
                return
            self.send_error(404)

    return Handler


def start_mjpeg_server(host: str, port: int,
                       html: bytes | None = None
                       ) -> tuple[FrameBus, ThreadingHTTPServer]:
    bus = FrameBus()
    index_html = html if html is not None else _DEFAULT_INDEX_HTML
    httpd = ThreadingHTTPServer((host, port), make_handler(bus, index_html))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return bus, httpd


def lan_ip_hint() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 53))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "<pi-ip>"
