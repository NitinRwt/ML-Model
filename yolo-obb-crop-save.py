import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

def crop_and_save_detections(model, image_path, output_root):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Unable to read {image_path}")
        return

    # Run inference
    results = model(img)

    # If no detections, skip
    if not any(r.obb is not None for r in results):
        return

    # For each detection result
    for r in results:
        if r.obb is None:
            continue

        boxes = r.obb.xyxyxyxy.cpu().numpy()
        class_ids = r.obb.cls.cpu().numpy().astype(int)

        # Crop each detected oriented bounding box
        for box, cls_id in zip(boxes, class_ids):
            pts = box.reshape(4, 2).astype(np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            crop = img[y:y+h, x:x+w]

            # Prepare class-specific output directory
            class_name = r.names[cls_id]
            class_dir = os.path.join(output_root, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Save crop with unique filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            crop_filename = f"{base_name}_{class_name}_{x}_{y}.jpg"
            crop_path = os.path.join(class_dir, crop_filename)
            cv2.imwrite(crop_path, crop)
            print(f"Saved crop: {crop_path}")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python detect_and_crop.py <model_path> <input_folder> <output_folder>")
        sys.exit(1)

    model_path = sys.argv[1]
    input_folder = sys.argv[2]
    output_folder = sys.argv[3]

    # Load YOLO OBB model
    model = YOLO(model_path)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in input folder
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_path = os.path.join(input_folder, fname)
        crop_and_save_detections(model, image_path, output_folder)