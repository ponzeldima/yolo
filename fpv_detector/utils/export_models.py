from pathlib import Path

from ultralytics import YOLO
import torch
print(torch.__version__)


FOLDER_NAME = "v8n_640_fpv_public_datasts_310826"
PROJECT_DIR = Path(__file__).resolve().parents[1]
CALIBRATION_FRACTION = 0.10

def main():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    dataset_yaml = PROJECT_DIR / "datasets" / "fpv_public_datasts_310826" / "data.yaml"
    
    best_weights_path = PROJECT_DIR / "runs" / "detect" / FOLDER_NAME / "weights" / "best.pt"
    model = YOLO(best_weights_path) 
    # model.train(
    #     data=dataset_yaml,
    #     quantize=8,
    #     epochs=5,
    #     batch=8,
    #     device="cuda",  # Використовуємо GPU (якщо доступний)
    #     optimizer="AdamW",
    #     lr0=0.00001,
    #     lrf=0.1,
    #     warmup_epochs=0.5,
    #     cos_lr=True,
    #     mosaic=0.0,
    # )
    # model.export(format="onnx", quantize=8)
    model.export(
        format="hailo",
        data=dataset_yaml,
        name="hailo8l",
        imgsz=640,
        fraction=CALIBRATION_FRACTION,
    )

    print("Експорт завершено!")

if __name__ == "__main__":
    main()