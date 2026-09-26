# yolo

Install ultralitics for GPU

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install ultralytics


## Export YOLO to Hailo HEF

The Hailo Dataflow Compiler must run in WSL. Export from the Linux filesystem,
not directly from `/mnt/c`, because calibration image access from the Windows
drive is much slower.

The export configuration is in `fpv_detector/utils/export_models.py`:

- `FOLDER_NAME` selects the trained run to export.
- `imgsz` must match the model input size.
- `CALIBRATION_FRACTION = 0.10` uses 10% of the validation set for INT8
	calibration. Increase it for potentially better accuracy; it also increases
	export time.

Open Ubuntu and stage the model, export script, and validation split locally:

```bash
source /home/ponzel/ml/.venv/bin/activate

PROJECT_WIN=/mnt/c/Users/ponze/Desktop/ML/yolo/fpv_detector
EXPORT_ROOT=/home/ponzel/hailo_export/project
RUN_NAME=v8n_640_fpv_public_datasts_310826
DATASET_NAME=fpv_public_datasts_310826

rm -rf "$EXPORT_ROOT"
mkdir -p "$EXPORT_ROOT/runs/detect/$RUN_NAME/weights"
mkdir -p "$EXPORT_ROOT/datasets/$DATASET_NAME"
mkdir -p "$EXPORT_ROOT/utils"

cp "$PROJECT_WIN/runs/detect/$RUN_NAME/weights/best.pt" \
	"$EXPORT_ROOT/runs/detect/$RUN_NAME/weights/best.pt"
cp "$PROJECT_WIN/datasets/$DATASET_NAME/data.yaml" \
	"$EXPORT_ROOT/datasets/$DATASET_NAME/data.yaml"
cp -a "$PROJECT_WIN/datasets/$DATASET_NAME/valid" \
	"$EXPORT_ROOT/datasets/$DATASET_NAME/"
cp "$PROJECT_WIN/utils/export_models.py" "$EXPORT_ROOT/utils/export_models.py"
```

Run the export from the staged project:

```bash
cd "$EXPORT_ROOT"
python utils/export_models.py
```

Copy the resulting files back to the Windows project:

```bash
cp -a "$EXPORT_ROOT/runs/detect/$RUN_NAME/weights/best_hailo_model" \
	"$PROJECT_WIN/runs/detect/$RUN_NAME/weights/"
```

The output directory contains `best.hef`, `metadata.yaml`, and
`nms_config.json`:

```text
fpv_detector/runs/detect/<run-name>/weights/best_hailo_model/
```

If WSL reports `Wsl/Service/E_UNEXPECTED`, start an Administrator PowerShell
and run `wsl --shutdown`, then `Restart-Service WslService` before reopening
Ubuntu.

