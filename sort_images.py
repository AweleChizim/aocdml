import os
import shutil

# Paths
images_folder = r"datasets/OTU_2d/images"
labels_file = r"datasets/OTU_2d/val_cls.txt"
output_base = 'sorted_images'

# Make output folder if it doesn't exist
os.makedirs(output_base, exist_ok=True)

# Read the label file
with open(labels_file, 'r') as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()  # Split by whitespace
    if len(parts) != 2:
        print(f"Skipping invalid line: {line}")
        continue
    image_name, class_label = parts
    # Make folder for this class if it doesn't exist
    class_folder = os.path.join(output_base, class_label)
    os.makedirs(class_folder, exist_ok=True)
    
    # Move the image
    src_path = os.path.join(images_folder, image_name)
    dst_path = os.path.join(class_folder, image_name)
    
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
    else:
        print(f"Image not found: {src_path}")