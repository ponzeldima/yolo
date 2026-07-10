"""Track a small fast-moving drone in aerial video shot from a carrier drone.

Pipeline (no ML, per supervisor's note: TV-L1 / Lucas-Kanade family,
"static scene + one fast target"):

  1. Sparse features (Shi-Tomasi) + pyramidal Lucas-Kanade between
     consecutive frames -> point correspondences.
  2. RANSAC homography -> global carrier-drone motion.
  3. Warp previous frame onto current; absdiff -> residual motion.
  4. Threshold + morphology -> motion mask.
  5. Contour blobs -> Kalman tracker (constant velocity, gated NN).

Two run modes:

  * Batch (default, no --show)        -> writes annotated MP4.
  * Interactive (--show / --tune)     -> step navigation + live trackbars +
                                         extra diagnostic windows.

Interactive keys (focus must be on an OpenCV window):

  space / n     next frame
  p             previous frame (from in-memory buffer)
  r             re-render current pair with current params
  H             print homography matrix and stats to console
  t             reset Kalman tracker
  o             toggle motion-mask overlay on main view
  1..5          toggle aux windows: 1=mask 2=raw_diff 3=stab_heat 4=flow 5=params
  s             save screenshots of all open windows
  q / Esc       quit

Works with stock opencv-python.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import numpy as np


# --------------------------- params ---------------------------------------

@dataclass
class Params:
    # --- motion-mask (cheap to change) ---
    diff_thresh: int = 18       # 0..80
    morph_open: int = 3         # 0..9
    morph_dilate: int = 5       # 0..15
    border: int = 12            # 0..80
    min_area: int = 4           # 1..500
    max_area: int = 2000        # 100..10000
    # --- tracker ---
    gate_px: int = 60           # 5..200
    max_lost: int = 15          # 0..60
    # --- LK / homography (expensive: invalidates LK cache) ---
    blur: int = 3                    # 0..15  (odd; 0/1 disables)
    max_features: int = 500          # 50..2000
    min_distance: int = 8            # 1..30
    quality_x1000: int = 10          # 1..100  -> /1000
    lk_win: int = 21                 # 5..51  (odd)
    pyr_levels: int = 3              # 0..5
    ransac_thresh_x10: int = 30      # 1..100 -> /10

    @property
    def quality_level(self) -> float:
        return max(0.001, self.quality_x1000 / 1000.0)

    @property
    def ransac_thresh(self) -> float:
        return max(0.1, self.ransac_thresh_x10 / 10.0)

    def snapshot(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# Param names whose change forces LK / homography recomputation.
EXPENSIVE = {
    "blur", "max_features", "min_distance", "quality_x1000",
    "lk_win", "pyr_levels", "ransac_thresh_x10",
}

# Trackbar definitions for the params panel (label, attr, max).
TRACKBARS = [
    ("diff_thresh",     "diff_thresh",       80),
    ("morph_open",      "morph_open",         9),
    ("morph_dilate",    "morph_dilate",      15),
    ("border",          "border",            80),
    ("min_area",        "min_area",         500),
    ("max_area/10",     "max_area",        1000),  # scaled x10
    ("gate_px",         "gate_px",          200),
    ("max_lost",        "max_lost",          60),
    ("blur",            "blur",              15),
    ("max_features",    "max_features",    2000),
    ("min_distance",    "min_distance",      30),
    ("quality/1000",    "quality_x1000",    100),
    ("lk_win",          "lk_win",            51),
    ("pyr_levels",      "pyr_levels",         5),
    ("ransac*10",       "ransac_thresh_x10",100),
]
# Trackbars whose displayed value is 10x smaller than stored.
SCALED_X10 = {"max_area/10"}


# --------------------------- detection ------------------------------------

@dataclass
class Detection:
    cx: float
    cy: float
    area: float
    bbox: tuple[int, int, int, int]


# --------------------------- tracker --------------------------------------

class DroneTracker:
    def __init__(self, gate_px: float, max_lost: int, trail_len: int = 200):
        self.gate_px = gate_px
        self.max_lost = max_lost
        self.trail_len = trail_len
        self._kf = self._make_kalman()
        self._initialized = False
        self._lost = 0
        self.trajectory: list[tuple[int, int]] = []

    @staticmethod
    def _make_kalman() -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2, 0)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32)
        kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        return kf

    def reset(self) -> None:
        self._kf = self._make_kalman()
        self._initialized = False
        self._lost = 0
        self.trajectory.clear()

    def _predict(self) -> tuple[float, float] | None:
        if not self._initialized:
            return None
        pred = self._kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def _correct(self, x: float, y: float) -> None:
        if not self._initialized:
            self._kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self._initialized = True
            return
        self._kf.correct(np.array([[x], [y]], dtype=np.float32))

    def update(self, detections: list[Detection]) -> Detection | None:
        pred = self._predict()
        chosen: Detection | None = None
        if pred is not None:
            px, py = pred
            gate_sq = self.gate_px ** 2
            best = float("inf")
            for det in detections:
                d = (det.cx - px) ** 2 + (det.cy - py) ** 2
                if d < best and d <= gate_sq:
                    best = d
                    chosen = det
        if chosen is None and not self._initialized and detections:
            chosen = max(detections, key=lambda d: d.area)
        if chosen is not None:
            self._correct(chosen.cx, chosen.cy)
            self._lost = 0
            self.trajectory.append((int(chosen.cx), int(chosen.cy)))
            if len(self.trajectory) > self.trail_len:
                self.trajectory.pop(0)
        else:
            self._lost += 1
            if self._lost > self.max_lost:
                self._initialized = False
                self.trajectory.clear()
        return chosen

    @property
    def predicted_state(self) -> tuple[float, float] | None:
        if not self._initialized:
            return None
        s = self._kf.statePost
        return float(s[0, 0]), float(s[1, 0])


# --------------------------- compute helpers -------------------------------

def preprocess(gray_raw: np.ndarray, blur: int) -> np.ndarray:
    if blur and blur > 1:
        k = blur | 1
        return cv2.GaussianBlur(gray_raw, (k, k), 0)
    return gray_raw


def compute_lk(prev_gray: np.ndarray, curr_gray: np.ndarray, p: Params) -> dict | None:
    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=max(8, p.max_features),
        qualityLevel=p.quality_level,
        minDistance=max(1, p.min_distance),
        blockSize=7,
    )
    if p0 is None or len(p0) < 8:
        return None
    win = max(5, p.lk_win | 1)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, p0, None,
        winSize=(win, win), maxLevel=max(0, p.pyr_levels),
    )
    if p1 is None:
        return None
    st = st.reshape(-1).astype(bool)
    if st.sum() < 8:
        return None
    src = p0[st]
    dst = p1[st]

    M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or mask is None or int(mask.sum()) < 8:
        return None
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, p.ransac_thresh)
    if H is None or mask is None or int(mask.sum()) < 8:
        return None
    return {
        "H": H,
        "M": M,
        "src": src.reshape(-1, 2),
        "dst": dst.reshape(-1, 2),
        "inliers": mask.reshape(-1).astype(bool),
    }


def compute_diff_and_mask(
    prev_gray: np.ndarray, curr_gray: np.ndarray,
    lk: dict | None, p: Params,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if lk is not None:
        # warped = cv2.warpPerspective(
        #     prev_gray, lk["H"],
        #     (curr_gray.shape[1], curr_gray.shape[0]),
        #     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        # )
        warped = cv2.warpAffine(
            prev_gray, lk["M"],
            (curr_gray.shape[1], curr_gray.shape[0]), 
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        stabilised = True
    else:
        warped = prev_gray
        stabilised = False
    raw_diff = cv2.absdiff(curr_gray, warped)
    _, mask = cv2.threshold(raw_diff, p.diff_thresh, 255, cv2.THRESH_BINARY)
    b = max(0, p.border)
    if b > 0:
        mask[:b, :] = 0
        mask[-b:, :] = 0
        mask[:, :b] = 0
        mask[:, -b:] = 0
    if p.morph_open >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (p.morph_open, p.morph_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if p.morph_dilate >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (p.morph_dilate, p.morph_dilate))
        mask = cv2.dilate(mask, k)
    return raw_diff, mask, stabilised


def detect_blobs(motion_mask: np.ndarray, p: Params) -> list[Detection]:
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out: list[Detection] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < p.min_area or area > p.max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        m = cv2.moments(c)
        if m["m00"] > 0:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
        else:
            cx = x + w / 2.0
            cy = y + h / 2.0
        out.append(Detection(cx=cx, cy=cy, area=area, bbox=(x, y, w, h)))
    return out


# --------------------------- viz ------------------------------------------

def draw_main(curr_color: np.ndarray, detections: list[Detection],
              chosen: Detection | None, tracker: DroneTracker,
              frame_idx: int, mask: np.ndarray | None, overlay_mask: bool,
              stabilised: bool) -> np.ndarray:
    vis = curr_color.copy()
    for det in detections:
        x, y, w, h = det.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (80, 80, 80), 1)
    if chosen is not None:
        x, y, w, h = chosen.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, f"drone a={int(chosen.area)}",
                    (x, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        pred = tracker.predicted_state
        if pred is not None:
            cv2.circle(vis, (int(pred[0]), int(pred[1])), 8, (0, 165, 255), 2)
    traj = tracker.trajectory
    for i in range(1, len(traj)):
        cv2.line(vis, traj[i - 1], traj[i], (0, 200, 255), 1, cv2.LINE_AA)
    if overlay_mask and mask is not None:
        overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        vis = cv2.addWeighted(vis, 0.75, overlay, 0.5, 0)
    stab_txt = "STAB" if stabilised else "RAW"
    cv2.putText(vis, f"frame {frame_idx}  [{stab_txt}]", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_flow(curr_color: np.ndarray, lk: dict | None) -> np.ndarray:
    out = curr_color.copy()
    if lk is None:
        cv2.putText(out, "LK / homography FAILED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out
    src = lk["src"]
    dst = lk["dst"]
    inl = lk["inliers"]
    for s, d, ok in zip(src, dst, inl):
        col = (0, 200, 0) if ok else (0, 0, 255)
        cv2.line(out, (int(s[0]), int(s[1])), (int(d[0]), int(d[1])), col, 1)
        cv2.circle(out, (int(d[0]), int(d[1])), 2, col, -1)
    n_in = int(inl.sum())
    n_out = int((~inl).sum())
    cv2.putText(out, f"inliers {n_in}  outliers {n_out}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def diff_heatmap(diff: np.ndarray) -> np.ndarray:
    mx = max(1, int(diff.max()))
    norm = np.clip(diff.astype(np.float32) * (255.0 / mx), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(diff*4, cv2.COLORMAP_INFERNO)


# --------------------------- frame buffer ---------------------------------

class FrameBuffer:
    """Holds last N raw frames (color + gray) to allow step-back."""

    def __init__(self, cap: cv2.VideoCapture, capacity: int = 128):
        self.cap = cap
        self.capacity = capacity
        self.buf: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.cursor: int = -1
        self.next_idx: int = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    def _read_next(self) -> bool:
        ok, frame = self.cap.read()
        if not ok:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.buf.append((self.next_idx, frame, gray))
        self.next_idx += 1
        if len(self.buf) > self.capacity:
            self.buf.pop(0)
            self.cursor -= 1
        return True

    def advance(self) -> bool:
        if self.cursor + 1 < len(self.buf):
            self.cursor += 1
            return True
        if self._read_next():
            self.cursor = len(self.buf) - 1
            return True
        return False

    def back(self) -> bool:
        if self.cursor > 0:
            self.cursor -= 1
            return True
        return False

    def current(self):
        if self.cursor < 0:
            return None
        return self.buf[self.cursor]

    def previous(self):
        if self.cursor < 1:
            return None
        return self.buf[self.cursor - 1]


# --------------------------- interactive mode -----------------------------

WIN_MAIN = "tracker"
WIN_PARAMS = "params"
WIN_MASK = "motion_mask"
WIN_RAW = "raw_diff"
WIN_STAB = "stabilised_diff"
WIN_FLOW = "flow_inliers"

AUX_TOGGLES = {
    "1": WIN_MASK,
    "2": WIN_RAW,
    "3": WIN_STAB,
    "4": WIN_FLOW,
    "5": WIN_PARAMS,
}


def make_params_window(params: Params, on_change) -> None:
    cv2.namedWindow(WIN_PARAMS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_PARAMS, 420, 720)
    for label, attr, maxv in TRACKBARS:
        init = getattr(params, attr)
        disp = init // 10 if label in SCALED_X10 else init
        cv2.createTrackbar(label, WIN_PARAMS, max(0, disp), maxv,
                           _make_cb(params, attr, label, on_change))


def _make_cb(params: Params, attr: str, label: str, on_change):
    def cb(val: int):
        if label in SCALED_X10:
            val = val * 10
        setattr(params, attr, val)
        on_change(attr)
    return cb


def print_homography(frame_idx: int, prev_idx: int, lk: dict | None,
                     raw_diff: np.ndarray, mask: np.ndarray,
                     detections: list[Detection], chosen: Detection | None,
                     p: Params) -> None:
    np.set_printoptions(precision=4, suppress=True)
    print(f"\n=== pair  prev={prev_idx}  curr={frame_idx} ===")
    if lk is None:
        print("  homography: FAILED")
    else:
        n_in = int(lk["inliers"].sum())
        n_out = int((~lk["inliers"]).sum())
        print(f"  inliers={n_in}  outliers={n_out}  "
              f"ransac_px={p.ransac_thresh:.2f}")
        print("  H =")
        for row in lk["H"]:
            print("   ", row)
    print(f"  diff: mean={raw_diff.mean():.2f}  max={int(raw_diff.max())}  "
          f"thresh={p.diff_thresh}")
    print(f"  mask: nonzero={int((mask > 0).sum())}  "
          f"candidates={len(detections)}  "
          f"chosen={'yes' if chosen else 'no'}")
    if chosen is not None:
        print(f"  bbox={chosen.bbox}  area={chosen.area:.1f}  "
              f"center=({chosen.cx:.1f},{chosen.cy:.1f})")


def interactive_loop(cap: cv2.VideoCapture, params: Params,
                     start: int, screenshot_dir: Path) -> None:
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    fb = FrameBuffer(cap, capacity=128)
    tracker = DroneTracker(gate_px=params.gate_px, max_lost=params.max_lost)

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
    open_windows: set[str] = {WIN_MAIN, WIN_PARAMS, WIN_MASK}
    overlay_mask = False

    state = {"dirty_lk": False, "dirty_render": False,
             "announce_pair": True, "last_imgs": {}}

    def on_param_change(attr: str) -> None:
        if attr in EXPENSIVE:
            state["dirty_lk"] = True
        state["dirty_render"] = True

    make_params_window(params, on_param_change)

    cache: dict = {"prev_idx": None, "curr_idx": None, "lk": None}

    if not fb.advance():
        print("Empty video.")
        return

    def render(advance_tracker: bool) -> None:
        cur = fb.current()
        prv = fb.previous()
        if cur is None:
            return
        curr_idx, curr_color, curr_gray_raw = cur

        if prv is None:
            cv2.imshow(WIN_MAIN, curr_color)
            if WIN_MASK in open_windows:
                cv2.imshow(WIN_MASK, np.zeros(curr_gray_raw.shape, np.uint8))
            state["last_imgs"] = {WIN_MAIN: curr_color}
            state["dirty_render"] = False
            return

        prev_idx, _, prev_gray_raw = prv
        prev_gray = preprocess(prev_gray_raw, params.blur)
        curr_gray = preprocess(curr_gray_raw, params.blur)

        if (cache["prev_idx"] != prev_idx or cache["curr_idx"] != curr_idx
                or state["dirty_lk"]):
            cache["lk"] = compute_lk(prev_gray, curr_gray, params)
            cache["prev_idx"] = prev_idx
            cache["curr_idx"] = curr_idx
            state["dirty_lk"] = False
        lk = cache["lk"]

        raw_diff, mask, stabilised = compute_diff_and_mask(
            prev_gray, curr_gray, lk, params)
        detections = detect_blobs(mask, params)

        tracker.gate_px = params.gate_px
        tracker.max_lost = params.max_lost

        chosen: Detection | None = None
        if advance_tracker:
            chosen = tracker.update(detections)
        else:
            pred = tracker.predicted_state
            if pred is not None and detections:
                px, py = pred
                gate_sq = tracker.gate_px ** 2
                best = float("inf")
                for det in detections:
                    d = (det.cx - px) ** 2 + (det.cy - py) ** 2
                    if d < best and d <= gate_sq:
                        best = d
                        chosen = det
            elif detections:
                chosen = max(detections, key=lambda d: d.area)

        main_vis = draw_main(curr_color, detections, chosen, tracker,
                             curr_idx, mask, overlay_mask, stabilised)
        cv2.imshow(WIN_MAIN, main_vis)
        if WIN_MASK in open_windows:
            cv2.imshow(WIN_MASK, mask)
        if WIN_RAW in open_windows:
            cv2.imshow(WIN_RAW, raw_diff)
        heat = None
        if WIN_STAB in open_windows:
            heat = diff_heatmap(raw_diff)
            cv2.imshow(WIN_STAB, heat)
        flow = None
        if WIN_FLOW in open_windows:
            flow = draw_flow(curr_color, lk)
            cv2.imshow(WIN_FLOW, flow)

        if state["announce_pair"]:
            print_homography(curr_idx, prev_idx, lk, raw_diff, mask,
                             detections, chosen, params)
            state["announce_pair"] = False

        state["dirty_render"] = False
        state["last_imgs"] = {
            WIN_MAIN: main_vis,
            WIN_MASK: cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            WIN_RAW: cv2.cvtColor(raw_diff, cv2.COLOR_GRAY2BGR),
            WIN_STAB: heat if heat is not None else diff_heatmap(raw_diff),
            WIN_FLOW: flow if flow is not None else draw_flow(curr_color, lk),
        }

    state["announce_pair"] = True
    render(advance_tracker=True)

    print("\nInteractive mode. Keys: space/n=next, p=prev, r=rerender, "
          "H=print homography, t=reset tracker, o=overlay mask, "
          "1..5=toggle aux windows, s=save screenshots, q=quit.\n")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            if state["dirty_render"]:
                render(advance_tracker=False)
            continue

        ch = chr(key) if 32 <= key < 127 else ""
        if key in (ord("q"), 27):
            break
        elif key in (ord("n"), ord(" ")):
            state["announce_pair"] = True
            if not fb.advance():
                print("end of stream")
                state["announce_pair"] = False
            render(advance_tracker=True)
        elif ch == "p":
            state["announce_pair"] = True
            if not fb.back():
                print("buffer start (no earlier frames)")
                state["announce_pair"] = False
            render(advance_tracker=False)
        elif ch == "r":
            state["announce_pair"] = True
            render(advance_tracker=False)
        elif ch == "H":
            state["announce_pair"] = True
            render(advance_tracker=False)
        elif ch == "t":
            tracker.reset()
            print("tracker reset")
            render(advance_tracker=False)
        elif ch == "o":
            overlay_mask = not overlay_mask
            print(f"overlay_mask = {overlay_mask}")
            render(advance_tracker=False)
        elif ch in AUX_TOGGLES:
            win = AUX_TOGGLES[ch]
            if win in open_windows:
                open_windows.discard(win)
                try:
                    cv2.destroyWindow(win)
                except cv2.error:
                    pass
                print(f"closed {win}")
            else:
                open_windows.add(win)
                if win == WIN_PARAMS:
                    make_params_window(params, on_param_change)
                else:
                    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                print(f"opened {win}")
                render(advance_tracker=False)
        elif ch == "s":
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time())
            for name, img in state["last_imgs"].items():
                if name in open_windows or name == WIN_MAIN:
                    out = screenshot_dir / f"{stamp}_{name}.png"
                    cv2.imwrite(str(out), img)
                    print(f"saved {out}")

    cv2.destroyAllWindows()


# --------------------------- batch mode -----------------------------------

def batch_loop(cap: cv2.VideoCapture, params: Params, out_path: Path,
               start: int, end: int | None, debug_mask: bool) -> None:
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open writer: {out_path}")
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    tracker = DroneTracker(gate_px=params.gate_px, max_lost=params.max_lost)
    prev_gray: np.ndarray | None = None
    frame_idx = start
    processed = 0
    stab_ok = 0
    t0 = time.time()

    while True:
        if end is not None and frame_idx >= end:
            break
        ok, frame = cap.read()
        if not ok:
            break
        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = preprocess(gray_raw, params.blur)

        if prev_gray is not None:
            lk = compute_lk(prev_gray, gray, params)
            raw_diff, mask, stab = compute_diff_and_mask(prev_gray, gray, lk, params)
            if stab:
                stab_ok += 1
            detections = detect_blobs(mask, params)
            chosen = tracker.update(detections)
            vis = draw_main(frame, detections, chosen, tracker, frame_idx,
                            mask, debug_mask, stab)
        else:
            vis = frame.copy()
            cv2.putText(vis, f"frame {frame_idx}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)

        writer.write(vis)
        prev_gray = gray
        frame_idx += 1
        processed += 1
        if processed % 50 == 0:
            elapsed = time.time() - t0
            eff = processed / elapsed if elapsed > 0 else 0.0
            print(f"  ... frame {frame_idx} ({eff:.1f} fps)")

    writer.release()
    elapsed = time.time() - t0
    eff = processed / elapsed if elapsed > 0 else 0.0
    stab_pct = (100.0 * stab_ok / max(1, processed - 1)) if processed > 1 else 0.0
    print(f"Done. {processed} frames in {elapsed:.1f}s ({eff:.1f} fps); "
          f"camera-motion estimated on {stab_pct:.1f}% of pairs.")
    print(f"Saved: {out_path}")


# --------------------------- cli ------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", "-i", required=True, type=Path)
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Annotated MP4 (batch mode). Default: <input>_tracked.mp4")
    ap.add_argument("--show", "--tune", action="store_true",
                    help="Interactive step/tune mode (no MP4 written).")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--debug-mask", action="store_true",
                    help="Batch mode: overlay motion mask on output.")
    ap.add_argument("--screenshot-dir", type=Path,
                    default=Path("optical_flow/screenshots"))

    ap.add_argument("--diff-thresh", type=int, default=18)
    ap.add_argument("--blur", type=int, default=3)
    ap.add_argument("--min-area", type=int, default=4)
    ap.add_argument("--max-area", type=int, default=2000)
    ap.add_argument("--gate-px", type=int, default=60)
    ap.add_argument("--max-lost", type=int, default=15)
    ap.add_argument("--max-features", type=int, default=500)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    params = Params(
        diff_thresh=args.diff_thresh,
        blur=args.blur,
        min_area=args.min_area,
        max_area=args.max_area,
        gate_px=args.gate_px,
        max_lost=args.max_lost,
        max_features=args.max_features,
    )

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input: {args.input} ({width}x{height} @ {fps:.2f} fps, {total} frames)")

    try:
        if args.show:
            interactive_loop(cap, params, args.start, args.screenshot_dir)
        else:
            out_path = args.output or args.input.with_name(
                args.input.stem + "_tracked.mp4")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Output: {out_path}")
            batch_loop(cap, params, out_path, args.start, args.end,
                       args.debug_mask)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
