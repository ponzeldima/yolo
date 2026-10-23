import os

from pathlib import Path

from ultralytics import YOLO
import torch
print(torch.__version__)


FOLDER_NAME = "v8m_1080_dataset_3"
PROJECT_DIR = Path(__file__).resolve().parents[1]
CALIBRATION_FRACTION = 1.0

def main():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    dataset_yaml = PROJECT_DIR / "datasets" / "dataset_3" / "data.yaml"
    
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
        name="hailo8",
        imgsz=800,
        fraction=CALIBRATION_FRACTION
    )

    print("Експорт завершено!")

if __name__ == "__main__":
    main()