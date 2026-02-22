import cv2
import numpy as np
import os
import glob

def crop_and_overwrite(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading: {image_path}")
        return False
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No contours found in: {image_path}")
        return False
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(largest_contour)
    final_cropped_img = masked_img[y:y+h, x:x+w]

    cv2.imwrite(image_path, final_cropped_img)
    return True



dataset_folder = "cross_val_folds/fold_1/test/UF"
image_paths = glob.glob(os.path.join(dataset_folder, "**", "*.jpg"), recursive=True)

processed_count = 0
for path in image_paths:
    if crop_and_overwrite(path):
        processed_count += 1

print(f"Finished processing! Cropped {processed_count} out of {len(image_paths)} images.")