import cv2
from ultralytics import YOLO

model = YOLO('best.pt')
# persist=True ОБОВ'ЯЗКОВО для роботи трекера
results = model.track(source='shahed_test_8.mp4', show=True, tracker="bytetrack.yaml", persist=True, stream=True)

for r in results:
    # Отримання ID об'єктів (вони є, тільки якщо об'єкт трекується)
    if r.boxes.id is not None:
        ids = r.boxes.id.int().tolist()
        boxes = r.boxes.xyxy # Координати [x1, y1, x2, y2]
        
        for id, box in zip(ids, boxes):
            x1, y1, x2, y2 = box.tolist()
            # Вираховуємо центр рамки для PhD задач
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            print(f"Frame ID processed. Target ID: {id} at Center: ({cx:.1f}, {cy:.1f})")