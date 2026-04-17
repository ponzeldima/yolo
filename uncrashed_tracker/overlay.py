"""Прозорий click-through overlay поверх гри (Win32 UpdateLayeredWindow)."""

import struct
import threading
import ctypes
import ctypes.wintypes

import numpy as np
import win32gui
import win32api
import win32con


class GameOverlay:
    """Прозоре, click-through, always-on-top вікно для малювання поверх гри.
    Використовує UpdateLayeredWindow з per-pixel alpha — без блимання."""

    OVERLAY_CLASS = "DroneOverlay"

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
                    win32con.WS_EX_TOOLWINDOW)
        style = win32con.WS_POPUP

        self.hwnd = win32gui.CreateWindowEx(
            ex_style, self.OVERLAY_CLASS, "Overlay",
            style, self.x, self.y, self.w, self.h,
            0, 0, 0, None)

        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)

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

        hdc_screen = usr.GetDC(0)
        self._hdc_mem = gdi.CreateCompatibleDC(hdc_screen)
        usr.ReleaseDC(0, hdc_screen)

    def _ensure_dib(self, w: int, h: int):
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
        buf = (ctypes.c_uint8 * (w * h * 4)).from_address(self._ppvBits.value)
        self._dib_array = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))

    def update(self, img_bgr: np.ndarray):
        h, w = img_bgr.shape[:2]
        self._ensure_dib(w, h)

        src = img_bgr[::-1]
        self._dib_array[:, :, :3] = src
        self._dib_array[:, :, 3] = (np.max(src, axis=2) > 0) * np.uint8(255)

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
        if not self._busy.acquire(blocking=False):
            return
        data = img_bgr.copy()
        def _work():
            try:
                self.update(data)
            finally:
                self._busy.release()
        threading.Thread(target=_work, daemon=True).start()

    def wait_done(self):
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
