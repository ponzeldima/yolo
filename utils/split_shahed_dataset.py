import os
import shutil
import random
from collections import defaultdict
from pathlib import Path

# --- НАЛАШТУВАННЯ ---
# Папка, де зараз лежать всі ваші фото (на кучі)
source_images_dir = "dataset_shahed/all/images"
# Папка, де зараз лежать всі ваші txt (на кучі)
source_labels_dir = "dataset_shahed/all/labels"

# Куди зберегти готовий датасет
output_dir = "dataset_shahed"

# Частка для тренування (0.8 = 80% train, 20% val)
train_ratio = 0.8
seed = 42  # Для відтворюваності (щоб кожного разу розбивало однаково)

# --- ЛОГІКА ---

def get_image_file(filename_no_ext, img_dir):
    """Шукає картинку з різними розширеннями."""
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        full_path = os.path.join(img_dir, filename_no_ext + ext)
        if os.path.exists(full_path):
            return full_path, ext
    return None, None

def stratified_split():
    random.seed(seed)
    
    # 1. Створюємо структуру папок
    for split in ['train', 'val']:
        for kind in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, kind), exist_ok=True)

    # 2. Зчитуємо всі лейбли і групуємо їх за класами
    # Структура: { 'class_0': [file1, file2], 'class_1': [file3], 'empty': [file4] }
    files_by_class = defaultdict(list)
    
    # Отримуємо список всіх txt файлів
    label_files = [f for f in os.listdir(source_labels_dir) if f.endswith('.txt')]
    
    print(f"Знайдено {len(label_files)} файлів розмітки. Аналізуємо вміст...")

    for label_file in label_files:
        file_path = os.path.join(source_labels_dir, label_file)
        
        # Читаємо класи з файлу
        classes_in_file = set()
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    classes_in_file.add(parts[0]) # Перше число - це id класу
        
        # Логіка групування:
        # Якщо файл порожній -> категорія 'background'
        # Якщо є класи -> беремо їх як унікальний ключ (наприклад, '0' або '0_1' якщо два класи)
        if not classes_in_file:
            category = 'background'
        else:
            # Сортуємо, щоб '0, 1' і '1, 0' були однією групою
            category = "_".join(sorted(classes_in_file))
            
        files_by_class[category].append(label_file)

    # 3. Розбиваємо кожну групу окремо
    train_files = []
    val_files = []

    print("\n--- Результати розподілу ---")
    for category, files in files_by_class.items():
        random.shuffle(files)
        
        split_idx = int(len(files) * train_ratio)
        
        # Якщо файлів дуже мало (наприклад, 1), кидаємо його в train
        if split_idx == 0 and len(files) > 0:
            split_idx = 1
            
        t_files = files[:split_idx]
        v_files = files[split_idx:]
        
        train_files.extend(t_files)
        val_files.extend(v_files)
        
        print(f"Клас/Група '{category}': Всього {len(files)} -> Train: {len(t_files)} | Val: {len(v_files)}")

    # 4. Функція копіювання
    def copy_batch(file_list, split_name):
        for label_file in file_list:
            name_no_ext = os.path.splitext(label_file)[0]
            
            # Знаходимо відповідну картинку
            src_img, ext = get_image_file(name_no_ext, source_images_dir)
            if not src_img:
                print(f"УВАГА: Не знайдено картинку для {label_file}")
                continue
                
            src_lbl = os.path.join(source_labels_dir, label_file)
            
            # Цільові шляхи
            dst_img = os.path.join(output_dir, split_name, 'images', name_no_ext + ext)
            dst_lbl = os.path.join(output_dir, split_name, 'labels', label_file)
            
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

    print("\nКопіювання файлів...")
    copy_batch(train_files, 'train')
    copy_batch(val_files, 'val')
    
    print(f"\nГотово! Ваші дані в папці: {output_dir}")
    print(f"Всього Train: {len(train_files)}")
    print(f"Всього Val: {len(val_files)}")

if __name__ == "__main__":
    stratified_split()