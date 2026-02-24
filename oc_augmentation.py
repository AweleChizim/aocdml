import os
from PIL import Image

input_folder = "datasets/OC" 
output_folder = "datasets copy/OC"
angles = list(range(-20, 21, 5))

os.makedirs(output_folder, exist_ok=True)
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Augmentation by rotation
for filename in os.listdir(input_folder):
    if filename.lower().endswith(valid_extensions):
        
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        name, ext = os.path.splitext(filename)

        for angle in angles:
            rotated = img.rotate(angle, expand=True)

            new_filename = f"{name}_rot{angle}{ext}"
            save_path = os.path.join(output_folder, new_filename)

            rotated.save(save_path)

print("OC class augmentation completed successfully!")