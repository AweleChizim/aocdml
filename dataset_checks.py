import os
from PIL import Image
from collections import Counter


file_path = r"datasets/OTU_2d/train_cls.txt"
# image_dir = r"datasets/uterine_fibroid_ultrasound_images/UF"
image_dir = r"datasets/OTU_2d/images"

# MMOTU Class Labelling
label_counts = Counter()

with open(file_path, "r") as f:
    for line in f:
        parts = line.strip().split()  # splits on whitespace
        if len(parts) == 2:
            _, label = parts
            label_counts[int(label)] += 1

for i in range(8):
    print(f"Label {i}: {label_counts[i]} images")


# Dataset Corrupted File Check
bad_images = []

for root, _, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()  # checks corruption
            except Exception as e:
                bad_images.append(path)

print(f"Corrupted images found: {len(bad_images)}")


# Dataset Image Dimension Check
sizes = []

for root, _, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(root, file)
            img = Image.open(path)
            sizes.append(img.size)  # (width, height)

size_counts = Counter(sizes)
print(size_counts.most_common(5))


# Dataset RGB/Grayscale Check
modes = []

for root, _, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(root, file))
            modes.append(img.mode)

print(Counter(modes))