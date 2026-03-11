import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

DEFAULT_MODEL_PATH = Path("datasets/fomo/dataset_test_244-fomo/fomo_model.keras")
DEFAULT_META_PATH = Path("datasets/fomo/dataset_test_244-fomo/fomo_dataset.json")
DEFAULT_VIDEO_PATH = Path("shahed_test_2.MP4")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize FOMO predictions on video")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH, help="Input video path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to .keras model")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META_PATH, help="Path to fomo_dataset.json")
    parser.add_argument("--threshold", type=float, default=0.45, help="Per-cell confidence threshold")
    parser.add_argument(
        "--draw",
        choices=["both", "box", "center"],
        default="both",
        help="Draw bounding boxes, centers, or both",
    )
    parser.add_argument("--no-window", action="store_true", help="Disable realtime preview window")
    parser.add_argument("--save", type=Path, default=None, help="Optional output video path")
    return parser.parse_args()


def load_meta(meta_path: Path):
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    image_size = int(meta["image_size"])
    grid_size = int(meta["grid_size"])
    class_names = meta["class_names"]

    if not class_names or class_names[0] != "background":
        raise ValueError("Expected class_names with background at index 0")

    return image_size, grid_size, class_names


def connected_components(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            pixels = []

            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))

                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            components.append(pixels)

    return components


def decode_detections(pred_map: np.ndarray, threshold: float):
    """
    pred_map shape: [grid, grid, classes_total], where class 0 is background.
    Returns list of detections with cell-level cluster boxes/centers.
    """
    grid_size = pred_map.shape[0]
    classes_total = pred_map.shape[-1]
    detections = []

    for cls_idx in range(1, classes_total):
        cls_scores = pred_map[:, :, cls_idx]
        mask = cls_scores >= threshold
        clusters = connected_components(mask)

        for cluster in clusters:
            ys = np.array([p[0] for p in cluster], dtype=np.int32)
            xs = np.array([p[1] for p in cluster], dtype=np.int32)
            weights = np.array([cls_scores[y, x] for y, x in cluster], dtype=np.float32)

            weight_sum = float(np.sum(weights))
            if weight_sum <= 1e-8:
                continue

            cx_cell = float(np.sum((xs + 0.5) * weights) / weight_sum)
            cy_cell = float(np.sum((ys + 0.5) * weights) / weight_sum)
            score = float(np.max(weights))

            detections.append(
                {
                    "class_id": cls_idx,
                    "score": score,
                    "x1_cell": int(xs.min()),
                    "y1_cell": int(ys.min()),
                    "x2_cell": int(xs.max()),
                    "y2_cell": int(ys.max()),
                    "cx_cell": cx_cell,
                    "cy_cell": cy_cell,
                    "grid_size": grid_size,
                }
            )

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


def draw_detections(frame: np.ndarray, detections, class_names, draw_mode: str):
    h, w = frame.shape[:2]

    for det in detections:
        grid = det["grid_size"]
        x_scale = w / grid
        y_scale = h / grid

        x1 = int(det["x1_cell"] * x_scale)
        y1 = int(det["y1_cell"] * y_scale)
        x2 = int((det["x2_cell"] + 1) * x_scale)
        y2 = int((det["y2_cell"] + 1) * y_scale)

        cx = int(det["cx_cell"] * x_scale)
        cy = int(det["cy_cell"] * y_scale)

        class_name = class_names[det["class_id"]]
        score_text = f"{class_name} {det['score']:.2f}"

        color = (0, 255, 0)
        if draw_mode in ("both", "box"):
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if draw_mode in ("both", "center"):
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        cv2.putText(
            frame,
            score_text,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def preprocess_frame(frame: np.ndarray, image_size: int):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)


def main():
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    image_size, grid_size, class_names = load_meta(args.meta)
    model = tf.keras.models.load_model(args.model)

    output_shape = model.output_shape
    if output_shape[1] != grid_size or output_shape[2] != grid_size:
        raise ValueError(
            f"Model output grid {output_shape[1]}x{output_shape[2]} does not match meta grid {grid_size}x{grid_size}"
        )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save), fourcc, fps, (frame_w, frame_h))

    print("Press 'q' to stop preview")
    print(f"Video: {args.video}")
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        x = preprocess_frame(frame, image_size)
        pred = model.predict(x, verbose=0)[0]
        detections = decode_detections(pred, args.threshold)

        draw_detections(frame, detections, class_names, args.draw)

        if writer is not None:
            writer.write(frame)

        if not args.no_window:
            cv2.imshow("FOMO video preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_window:
        cv2.destroyAllWindows()

    print("Done")


if __name__ == "__main__":
    main()
