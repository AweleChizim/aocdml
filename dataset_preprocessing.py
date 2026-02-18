from PIL import Image
import os
# from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
            


dataset_bot = r"datasets/BOT"
dataset_uf = r"datasets/UF"
dataset_no = r"datasets/NO"
dataset_oc = r"datasets/OC"

def convert_rgb_to_grayscale(path):
    img = Image.open(path).convert("L")  # L = grayscale (converts to grayscale)
    img.save(path)  # overwrites the file

def resize_images(path):
    img = Image.open(path)
    img_resized = img.resize((224, 224), Image.BILINEAR)
    img_resized.save(path) 

def normalize_and_filter_images(path):
    img = Image.open(path)
    img_normalized = np.array(img).astype("float32") / 255.0 # Min-Max Normalization
    img_filtered = cv2.medianBlur(img_normalized, ksize=3) # Median Filtering
    img_filtered = Image.fromarray((img_filtered * 255).astype("uint8")) # Convert back to image format
    img_filtered.save(path)

def preprocess_images(dataset_path):
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                img = convert_rgb_to_grayscale(path)
                img = resize_images(path)
                img = normalize_and_filter_images(path)
                print(f"Preprocessed: {path}")

preprocess_images(dataset_bot)
preprocess_images(dataset_no)
preprocess_images(dataset_oc)
preprocess_images(dataset_uf)