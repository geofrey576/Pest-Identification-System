"""
Run this ONCE before running main.py.
"""

import os
import pandas as pd

DATASET_PATH = r"C:\Users\HP\Documents\AI\AI project"

NON_PEST_FOLDERS = {'test', 'pest', 'PlantVillage', '__pycache__', '.git'}

DUPLICATE_MAP = {
    'mite'        : 'mites',
    'catterpillar': 'caterpillar',  
    'tea-aphid'   : 'aphids',
    'borer'       : 'stem_borer',
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

data = []

for label in sorted(os.listdir(DATASET_PATH)):

    if label in NON_PEST_FOLDERS or label.startswith('.'):
        print(f"  [Skip] {label}")
        continue

    class_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_path):
        continue

    clean_label = DUPLICATE_MAP.get(label, label)

    count = 0
    for img in os.listdir(class_path):
        ext = os.path.splitext(img)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            img_path = os.path.join(class_path, img)
            data.append([img_path, clean_label])
            count += 1

    if count == 0:
        print(f"  [Warning] No images found in folder: {label}")


df = pd.DataFrame(data, columns=["image_path", "pest_type"])
df.to_csv("pests_labels.csv", index=False)

print(f"\n pests_labels.csv created: {len(df)} images across "
      f"{df['pest_type'].nunique()} classes\n")
print(df['pest_type'].value_counts().to_string())