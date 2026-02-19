import requests
from roboflow import Roboflow
import time

# --- КОНФІГУРАЦІЯ ---
API_KEY = "WyrwiZBN0HWacmWZiZke"
WORKSPACE_ID = "test-tnsd2"
PROJECT_ID = "yolo-swaqy"
TAG_TO_KEEP = "new"  # Тег, який треба залишити
# --------------------

def clean_dataset_via_api():
    # 1. Отримуємо список зображень через SDK (це зручно для пошуку)
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)

    print(f"🔄 Завантажуємо список усіх зображень...")
    
    # Отримуємо інформацію про групу зображень
    # limit=10000 гарантує, що ми побачимо весь датасет
    images_info = project.search(limit=10000)
    
    if not images_info:
        print("Зображень не знайдено або помилка доступу.")
        return

    print(f"🔍 Всього зображень у проекті: {len(images_info)}")
    
    deleted_count = 0
    kept_count = 0

    # 2. Перебір і видалення через прямий API запит
    for img in images_info:
        image_id = img['id']
        image_name = img['name']
        
        # Отримуємо теги (вони можуть бути списком об'єктів або рядків)
        tags_raw = img.get('tags', [])
        current_tags = []
        
        # Нормалізація тегів у список рядків
        for t in tags_raw:
            if isinstance(t, dict):
                current_tags.append(t.get('name', '')) # або t.get('tag')
            else:
                current_tags.append(str(t))

        # 3. Логіка видалення
        if TAG_TO_KEEP not in current_tags:
            # Формуємо URL для видалення конкретного зображення
            # Документація API: DELETE /dataset/:datasetId/image/:imageId
            delete_url = f"https://api.roboflow.com/dataset/{PROJECT_ID}/image/{image_id}?api_key={API_KEY}"
            
            try:
                response = requests.delete(delete_url)
                
                if response.status_code == 200:
                    print(response)
                    print(f"❌ Видалено: {image_name}")
                    deleted_count += 1
                else:
                    print(f"⚠️ Помилка видалення {image_name}: {response.text}")
                
                # Пауза, щоб сервер не заблокував за частоту запитів
                time.sleep(0.1) 
                
            except Exception as e:
                print(f"⚠️ Критична помилка запиту: {e}")
        else:
            print(f"✅ Залишено (має тег '{TAG_TO_KEEP}'): {image_name}")
            kept_count += 1

    print("-" * 30)
    print(f"🏁 Готово!")
    print(f"🗑️ Видалено зображень: {deleted_count}")
    print(f"💾 Залишилося зображень: {kept_count}")

if __name__ == "__main__":
    print(f"УВАГА! Ви працюєте з проектом: {PROJECT_ID}")
    confirm = input(f"Видалити ВСІ зображення, які НЕ мають тегу '{TAG_TO_KEEP}'? (yes/no): ")
    if confirm.lower() == "yes":
        clean_dataset_via_api()
    else:
        print("Операцію скасовано.")