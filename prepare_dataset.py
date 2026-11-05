import os
import shutil
import random

SOURCE_DIR = './dataset_raw'
DEST_DIR = './dataset_binary'
CLASSES = ['Acne', 'Normal']
SPLIT_RATIO = (0.7, 0.15, 0.15)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def split_and_copy(class_name):
    src_class_dir = os.path.join(SOURCE_DIR, class_name)
    if not os.path.exists(src_class_dir):
        print(f"❌ Không tìm thấy: {src_class_dir}")
        return

    files = [f for f in os.listdir(src_class_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)

    n = len(files)
    n_train = int(n * SPLIT_RATIO[0])
    n_valid = int(n * SPLIT_RATIO[1])

    splits = {
        'train': files[:n_train],
        'valid': files[n_train:n_train + n_valid],
        'test':  files[n_train + n_valid:]
    }

    for split, file_list in splits.items():
        dest = os.path.join(DEST_DIR, split, class_name)
        os.makedirs(dest, exist_ok=True)
        for f in file_list:
            shutil.copy2(os.path.join(src_class_dir, f), os.path.join(dest, f))
        print(f"✅ {split}/{class_name}: {len(file_list)} ảnh")

if __name__ == '__main__':
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    for cls in CLASSES:
        split_and_copy(cls)
    print(f"\n🎉 Xong! Dữ liệu tại: {DEST_DIR}")