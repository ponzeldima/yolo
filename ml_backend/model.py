import os
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs, unquote
from urllib.request import Request, urlopen

from flask import Flask, request, jsonify
from ultralytics import YOLO
import json

app = Flask(__name__)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
MODEL_NAME = os.environ.get("YOLO_MODEL", "best.pt")
MODEL_PATH = MODEL_DIR / MODEL_NAME

# Shared Label Studio data
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/app/data")).resolve()
LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL", "").rstrip("/")
LABEL_STUDIO_API_KEY = os.environ.get("LABEL_STUDIO_API_KEY")
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", 25 * 1024 * 1024))

print(f"Loading YOLO model: {MODEL_PATH}", flush=True)
print(f"Data root: {DATA_ROOT}", flush=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully", flush=True)
print(f"Classes: {model.names}", flush=True)


def resolve_image_path(image_url: str, download_dir: Path) -> Path:
    """
    Convert Label Studio local-files URL:

        /data/local-files/?d=images/foo/bar.jpg

    into actual filesystem path:

        /app/data/images/foo/bar.jpg
    """

    print(f"Resolving image URL: {image_url}", flush=True)

    parsed = urlparse(image_url)

    # Label Studio local files endpoint
    if parsed.path == "/data/local-files/":
        query = parse_qs(parsed.query)

        if "d" not in query or not query["d"]:
            raise FileNotFoundError(
                f"Missing 'd' parameter in Label Studio URL: {image_url}"
            )

        relative_path = unquote(query["d"][0]).lstrip("/")

        file_path = (DATA_ROOT / relative_path).resolve()

        # Security: prevent path traversal outside DATA_ROOT
        try:
            file_path.relative_to(DATA_ROOT)
        except ValueError:
            raise FileNotFoundError(
                f"Invalid path outside DATA_ROOT: {relative_path}"
            )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {file_path}"
            )

        return file_path

    if parsed.scheme in ("http", "https") or (image_url.startswith("/") and LABEL_STUDIO_URL):
        url = image_url if parsed.scheme else urljoin(f"{LABEL_STUDIO_URL}/", image_url)
        request_headers = {}
        target_url = urlparse(url)
        label_studio_url = urlparse(LABEL_STUDIO_URL)
        if (
            LABEL_STUDIO_API_KEY
            and target_url.scheme == label_studio_url.scheme
            and target_url.netloc == label_studio_url.netloc
        ):
            request_headers["Authorization"] = f"Token {LABEL_STUDIO_API_KEY}"

        try:
            with urlopen(Request(url, headers=request_headers), timeout=30) as response:
                content_type = response.headers.get_content_type()
                if not content_type.startswith("image/"):
                    raise ValueError(f"Expected an image response, received {content_type}")

                image_path = download_dir / Path(urlparse(url).path).name
                if not image_path.name:
                    image_path = download_dir / "label-studio-image"

                bytes_written = 0
                with image_path.open("wb") as image_file:
                    while chunk := response.read(1024 * 1024):
                        bytes_written += len(chunk)
                        if bytes_written > MAX_IMAGE_BYTES:
                            raise ValueError(
                                f"Image exceeds MAX_IMAGE_BYTES ({MAX_IMAGE_BYTES} bytes)"
                            )
                        image_file.write(chunk)
        except OSError as error:
            raise FileNotFoundError(f"Unable to download image: {url}") from error

        return image_path

    # In case Label Studio sends a direct filesystem path
    path = Path(image_url)

    if path.exists():
        return path

    raise FileNotFoundError(
        f"Unsupported or missing image path: {image_url}"
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "UP",
        "model": str(MODEL_PATH),
        "classes": model.names,
        "data_root": str(DATA_ROOT)
    })


@app.route("/setup", methods=["POST"])
def setup():
    return jsonify({
        "model_version": MODEL_NAME
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    tasks = data.get("tasks", [])

    predictions = []

    for task in tasks:

        image_url = task.get("data", {}).get("image")

        if not image_url:
            predictions.append({
                "result": [],
                "score": 0
            })
            continue

        try:
            with tempfile.TemporaryDirectory(prefix="label-studio-") as temporary_directory:
                image_path = resolve_image_path(image_url, Path(temporary_directory))

                print(
                    f"Predicting image: {image_path}",
                    flush=True
                )

                results = model.predict(
                    source=str(image_path),
                    device="cpu",
                    verbose=False
                )

            result = results[0]
            print(
                f"YOLO detected {len(result.boxes) if result.boxes is not None else 0} objects",
                flush=True
            )
            height, width = result.orig_shape

            task_results = []

            if result.boxes is not None:
                for box in result.boxes:

                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    task_results.append({
                        "id": f"yolo_{cls_id}_{len(task_results)}",
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",

                        "original_width": width,
                        "original_height": height,
                        "image_rotation": 0,

                        "value": {
                            "x": x1 / width * 100,
                            "y": y1 / height * 100,
                            "width": (x2 - x1) / width * 100,
                            "height": (y2 - y1) / height * 100,
                            "rotation": 0,
                            "rectanglelabels": [
                                model.names[cls_id]
                            ]
                        },

                        "score": confidence
                    })

            predictions.append({
                "model_version": MODEL_NAME,
                "result": task_results,
                "score": max(
                    [x["score"] for x in task_results],
                    default=0
                )
            })

        except Exception as e:

            print(
                f"Prediction failed: {e}",
                flush=True
            )

            predictions.append({
                "result": [],
                "score": 0
            })


    print(
        "PREDICTION RESPONSE:",
        json.dumps(predictions, indent=2),
        flush=True
    )
    return jsonify({
        "results": predictions
    })

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=9090,
        threaded=True
    )