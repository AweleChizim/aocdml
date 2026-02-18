import os
import shutil
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Paths
dataset_base = 'datasets'
output_base = 'cross_val_folds'
n_splits = 5 

os.makedirs(output_base, exist_ok=True)

# Gather images and labels
images = []
labels = []

for class_name in os.listdir(dataset_base):
    class_folder = os.path.join(dataset_base, class_name)
    if not os.path.isdir(class_folder):
        continue
    for img_name in os.listdir(class_folder):
        images.append(os.path.join(class_folder, img_name))
        labels.append(class_name)

images = np.array(images)
labels = np.array(labels)

# Stratified K-Fold split
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(images, labels), 1):
    print(f"Creating fold {fold}...")
    fold_train_dir = os.path.join(output_base, f'fold_{fold}', 'train')
    fold_test_dir = os.path.join(output_base, f'fold_{fold}', 'test')
    
    # Create class subfolders in train/test
    for cls in np.unique(labels):
        os.makedirs(os.path.join(fold_train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(fold_test_dir, cls), exist_ok=True)
    
    # Move/copy images to train
    for idx in train_idx:
        src = images[idx]
        cls = labels[idx]
        dst = os.path.join(fold_train_dir, cls, os.path.basename(src))
        shutil.copy2(src, dst)
    
    # Move/copy images to test
    for idx in test_idx:
        src = images[idx]
        cls = labels[idx]
        dst = os.path.join(fold_test_dir, cls, os.path.basename(src))
        shutil.copy2(src, dst)

print("5-fold cross-validation directories created successfully!")
