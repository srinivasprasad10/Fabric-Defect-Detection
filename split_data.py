import os
import shutil
from sklearn.model_selection import train_test_split

DATASET_DIR = "dataset"
OUTPUT_DIR = "split_data"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")
SPLIT_RATIO = 0.2  

for cls in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, cls)
    if os.path.isdir(class_path):
        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)

        images = os.listdir(class_path)
        train_imgs, test_imgs = train_test_split(images, test_size=SPLIT_RATIO, random_state=42)

        for img in train_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(TRAIN_DIR, cls, img))
        for img in test_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(TEST_DIR, cls, img))

print("Dataset split completed.")
